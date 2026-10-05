#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// Check CUDA API errors
#define CUDA_CHECK(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess)                           \
        {                                                 \
            std::cerr << "CUDA error: "                   \
                      << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE);                      \
        }                                                 \
    } while (0)

// Attention dimensions
constexpr int SEQ_LEN = 512;
constexpr int HEADS = 12;
constexpr int HEAD_DIM = 64;

// Benchmark settings
constexpr int WARMUP = 20;
constexpr int ITERATIONS = 100;

// Attention scaling factor: 1 / sqrt(64)
constexpr float SCALE = 1.0f / 8.0f;

//
// QK^T
// Compute attention scores from Query and Key
//

__global__ void qkKernel(
    const __half *Q,
    const __half *K,
    __half *scores)
{
    __shared__ __half q_shared[16][64];
    __shared__ __half k_shared[16][64];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load Q tile
    for (int idx = tid; idx < 16 * HEAD_DIM; idx += 256)
    {
        int row = idx / HEAD_DIM;
        int k = idx % HEAD_DIM;
        int global_i = blockIdx.y * 16 + row;

        if (global_i < SEQ_LEN)
        {
            int q_base = (h * SEQ_LEN + global_i) * HEAD_DIM;
            q_shared[row][k] = Q[q_base + k];
        }
    }

    // Load K tile
    for (int idx = tid; idx < 16 * HEAD_DIM; idx += 256)
    {
        int row = idx / HEAD_DIM;
        int k = idx % HEAD_DIM;
        int global_j = blockIdx.x * 16 + row;

        if (global_j < SEQ_LEN)
        {
            int k_base = (h * SEQ_LEN + global_j) * HEAD_DIM;
            k_shared[row][k] = K[k_base + k];
        }
    }

    __syncthreads();

    if (i >= SEQ_LEN || j >= SEQ_LEN)
        return;

    // Locate output score
    int out_idx = (h * SEQ_LEN + i) * SEQ_LEN + j;

    float sum = 0.0f;

    for (int k = 0; k < HEAD_DIM; k++)
    {
        float q = __half2float(q_shared[threadIdx.y][k]);
        float key = __half2float(k_shared[threadIdx.x][k]);

        sum += q * key;
    }

    scores[out_idx] = __float2half(sum * SCALE);
}

//
// Softmax
// Convert attention scores into probabilities
//

__global__ void softmaxKernel(
    const __half *scores,
    __half *probs)
{
    // One block handles one attention row
    int row = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= SEQ_LEN || h >= HEADS)
        return;

    // Shared memory for parallel reductions
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    // Start of this attention row
    int base = (h * SEQ_LEN + row) * SEQ_LEN;

    // --------------------------------------------------------
    // Step 1: Find maximum score
    // --------------------------------------------------------

    float local_max = -INFINITY;

    for (int j = tid; j < SEQ_LEN; j += blockDim.x)
    {
        float x = __half2float(scores[base + j]);

        local_max = fmaxf(local_max, x);
    }

    // Store each thread's local maximum
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce local maxima to one row maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {

        if (tid < stride)
        {
            shared_max[tid] =
                fmaxf(
                    shared_max[tid],
                    shared_max[tid + stride]);
        }

        __syncthreads();
    }

    // Final maximum score
    float row_max = shared_max[0];

    // --------------------------------------------------------
    // Step 2: Compute exp(score - max) and local sum
    // --------------------------------------------------------

    float local_sum = 0.0f;

    for (int j = tid; j < SEQ_LEN; j += blockDim.x)
    {

        float x = __half2float(scores[base + j]);

        // Numerically stable softmax
        float e = expf(x - row_max);

        // Temporarily store exponential value
        probs[base + j] = __float2half(e);

        local_sum += e;
    }

    // Store each thread's local sum
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce local sums to one total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {

        if (tid < stride)
        {
            shared_sum[tid] +=
                shared_sum[tid + stride];
        }

        __syncthreads();
    }

    // Final sum of exponentials
    float sum = shared_sum[0];

    // --------------------------------------------------------
    // Step 3: Normalize into probabilities
    // --------------------------------------------------------

    for (int j = tid; j < SEQ_LEN; j += blockDim.x)
    {

        float p = __half2float(probs[base + j]);

        // Final softmax probability
        probs[base + j] =
            __float2half(p / sum);
    }
}

//
// P × V  (V2: shared memory tiling)
// Apply attention probabilities to Value vectors
//
// Output tile = 16 (query rows) x 16 (head dims).
// The reduction over j (SEQ_LEN) is walked in chunks of 16:
// each chunk of P and V is loaded once into shared memory
// and reused by all threads in the block.
//

constexpr int PV_TILE = 16;

__global__ void pvKernel(
    const __half *probs,
    const __half *V,
    __half *output)
{
    __shared__ __half p_shared[PV_TILE][PV_TILE]; // P[i][j]
    __shared__ __half v_shared[PV_TILE][PV_TILE]; // V[j][d]

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int d = blockIdx.x * PV_TILE + tx;
    int i = blockIdx.y * PV_TILE + ty;
    int h = blockIdx.z;

    float sum = 0.0f;

    for (int t = 0; t < SEQ_LEN; t += PV_TILE)
    {
        // Load P tile: row i, columns t..t+15
        int j_p = t + tx;
        p_shared[ty][tx] =
            (i < SEQ_LEN && j_p < SEQ_LEN)
                ? probs[(h * SEQ_LEN + i) * SEQ_LEN + j_p]
                : __float2half(0.0f);

        // Load V tile: rows t..t+15, column d
        int j_v = t + ty;
        v_shared[ty][tx] =
            (j_v < SEQ_LEN && d < HEAD_DIM)
                ? V[(h * SEQ_LEN + j_v) * HEAD_DIM + d]
                : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < PV_TILE; k++)
        {
            sum += __half2float(p_shared[ty][k]) *
                   __half2float(v_shared[k][tx]);
        }

        __syncthreads();
    }

    if (i < SEQ_LEN && d < HEAD_DIM)
    {
        output[(h * SEQ_LEN + i) * HEAD_DIM + d] =
            __float2half(sum);
    }
}

//
// CPU Reference
// Used to validate GPU attention output
//

void cpuAttention(
    const std::vector<__half> &Q,
    const std::vector<__half> &K,
    const std::vector<__half> &V,
    std::vector<float> &output)
{
    // Store QK^T scores
    std::vector<float> scores(
        HEADS * SEQ_LEN * SEQ_LEN);

    for (int h = 0; h < HEADS; h++)
    {

        for (int i = 0; i < SEQ_LEN; i++)
        {

            // Find maximum score for stable softmax
            float max_score = -INFINITY;

            // ------------------------------------------------
            // QK^T
            // ------------------------------------------------

            for (int j = 0; j < SEQ_LEN; j++)
            {

                float sum = 0.0f;

                for (int k = 0; k < HEAD_DIM; k++)
                {

                    int q_idx =
                        (h * SEQ_LEN + i) * HEAD_DIM + k;

                    int k_idx =
                        (h * SEQ_LEN + j) * HEAD_DIM + k;

                    sum +=
                        __half2float(Q[q_idx]) *
                        __half2float(K[k_idx]);
                }

                float score = sum * SCALE;

                int idx =
                    (h * SEQ_LEN + i) * SEQ_LEN + j;

                scores[idx] = score;

                max_score =
                    fmaxf(max_score, score);
            }

            // ------------------------------------------------
            // Softmax
            // ------------------------------------------------

            float denom = 0.0f;

            for (int j = 0; j < SEQ_LEN; j++)
            {

                int idx =
                    (h * SEQ_LEN + i) * SEQ_LEN + j;

                scores[idx] =
                    expf(scores[idx] - max_score);

                denom += scores[idx];
            }

            // ------------------------------------------------
            // P × V
            // ------------------------------------------------

            for (int d = 0; d < HEAD_DIM; d++)
            {

                float sum = 0.0f;

                for (int j = 0; j < SEQ_LEN; j++)
                {

                    int p_idx =
                        (h * SEQ_LEN + i) * SEQ_LEN + j;

                    int v_idx =
                        (h * SEQ_LEN + j) * HEAD_DIM + d;

                    float p =
                        scores[p_idx] / denom;

                    float v =
                        __half2float(V[v_idx]);

                    sum += p * v;
                }

                int out_idx =
                    (h * SEQ_LEN + i) * HEAD_DIM + d;

                output[out_idx] = sum;
            }
        }
    }
}

//
// Main
//

int main()
{
    std::cout << "=== GPU Attention V2 (QK + PV shared memory) ===\n";
    std::cout << "Sequence Length : " << SEQ_LEN << "\n";
    std::cout << "Heads           : " << HEADS << "\n";
    std::cout << "Head Dimension  : " << HEAD_DIM << "\n";
    std::cout << "Precision       : FP16\n";
    std::cout << "Warmup          : " << WARMUP << "\n";
    std::cout << "Iterations      : " << ITERATIONS << "\n\n";

    // --------------------------------------------------------
    // Allocate host-side sizes
    // --------------------------------------------------------

    size_t qkv_elements =
        static_cast<size_t>(HEADS) *
        SEQ_LEN *
        HEAD_DIM;

    size_t attention_elements =
        static_cast<size_t>(HEADS) *
        SEQ_LEN *
        SEQ_LEN;

    size_t qkv_bytes =
        qkv_elements * sizeof(__half);

    size_t attention_bytes =
        attention_elements * sizeof(__half);

    // --------------------------------------------------------
    // Create host Q, K, V
    // --------------------------------------------------------

    std::vector<__half> h_Q(qkv_elements);
    std::vector<__half> h_K(qkv_elements);
    std::vector<__half> h_V(qkv_elements);

    // Generate deterministic random input
    std::mt19937 rng(42);

    std::uniform_real_distribution<float>
        dist(-0.1f, 0.1f);

    for (size_t i = 0; i < qkv_elements; i++)
    {

        h_Q[i] = __float2half(dist(rng));
        h_K[i] = __float2half(dist(rng));
        h_V[i] = __float2half(dist(rng));
    }

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    __half *d_Q;
    __half *d_K;
    __half *d_V;

    __half *d_scores;
    __half *d_probs;

    __half *d_output;

    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_scores,
        attention_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_probs,
        attention_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        qkv_bytes));

    // --------------------------------------------------------
    // Copy Q, K, V: CPU → GPU
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        d_Q,
        h_Q.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_K,
        h_K.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_V,
        h_V.data(),
        qkv_bytes,
        cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Configure QK kernel
    // --------------------------------------------------------

    dim3 block_qk(16, 16);

    dim3 grid_qk(
        (SEQ_LEN + 15) / 16,
        (SEQ_LEN + 15) / 16,
        HEADS);

    // --------------------------------------------------------
    // Configure Softmax kernel
    // --------------------------------------------------------

    dim3 block_softmax(256);

    dim3 grid_softmax(
        SEQ_LEN,
        HEADS);

    // --------------------------------------------------------
    // Configure P × V kernel
    // --------------------------------------------------------

    dim3 block_pv(16, 16);

    dim3 grid_pv(
        (HEAD_DIM + 15) / 16,
        (SEQ_LEN + 15) / 16,
        HEADS);

    // ========================================================
    // Warmup
    // ========================================================

    for (int i = 0; i < WARMUP; i++)
    {

        // QK^T → scores
        qkKernel<<<grid_qk, block_qk>>>(
            d_Q,
            d_K,
            d_scores);

        // scores → probabilities
        softmaxKernel<<<grid_softmax, block_softmax>>>(
            d_scores,
            d_probs);

        // probabilities × V → output
        pvKernel<<<grid_pv, block_pv>>>(
            d_probs,
            d_V,
            d_output);
    }

    // Check kernel launch and execution
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Create CUDA timing events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEvent_t qk_start;
    cudaEvent_t qk_stop;

    cudaEvent_t softmax_start;
    cudaEvent_t softmax_stop;

    cudaEvent_t pv_start;
    cudaEvent_t pv_stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventCreate(&qk_start));
    CUDA_CHECK(cudaEventCreate(&qk_stop));

    CUDA_CHECK(cudaEventCreate(&softmax_start));
    CUDA_CHECK(cudaEventCreate(&softmax_stop));

    CUDA_CHECK(cudaEventCreate(&pv_start));
    CUDA_CHECK(cudaEventCreate(&pv_stop));

    // ========================================================
    // Benchmark: full attention
    // ========================================================

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        qkKernel<<<grid_qk, block_qk>>>(
            d_Q,
            d_K,
            d_scores);

        softmaxKernel<<<grid_softmax, block_softmax>>>(
            d_scores,
            d_probs);

        pvKernel<<<grid_pv, block_pv>>>(
            d_probs,
            d_V,
            d_output);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    float avg_ms =
        total_ms / ITERATIONS;

    // ========================================================
    // Benchmark: QK kernel
    // ========================================================

    CUDA_CHECK(cudaEventRecord(qk_start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        qkKernel<<<grid_qk, block_qk>>>(
            d_Q,
            d_K,
            d_scores);
    }

    CUDA_CHECK(cudaEventRecord(qk_stop));
    CUDA_CHECK(cudaEventSynchronize(qk_stop));

    float qk_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &qk_total_ms,
        qk_start,
        qk_stop));

    float qk_avg_ms =
        qk_total_ms / ITERATIONS;

    // ========================================================
    // Benchmark: Softmax kernel
    // ========================================================

    CUDA_CHECK(cudaEventRecord(softmax_start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        softmaxKernel<<<grid_softmax, block_softmax>>>(
            d_scores,
            d_probs);
    }

    CUDA_CHECK(cudaEventRecord(softmax_stop));
    CUDA_CHECK(cudaEventSynchronize(softmax_stop));

    float softmax_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &softmax_total_ms,
        softmax_start,
        softmax_stop));

    float softmax_avg_ms =
        softmax_total_ms / ITERATIONS;

    // ========================================================
    // Benchmark: P × V kernel
    // ========================================================

    CUDA_CHECK(cudaEventRecord(pv_start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        pvKernel<<<grid_pv, block_pv>>>(
            d_probs,
            d_V,
            d_output);
    }

    CUDA_CHECK(cudaEventRecord(pv_stop));
    CUDA_CHECK(cudaEventSynchronize(pv_stop));

    float pv_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &pv_total_ms,
        pv_start,
        pv_stop));

    float pv_avg_ms =
        pv_total_ms / ITERATIONS;

    std::cout << "\n=== Kernel Timing ===\n";

    std::cout << "QK Time      : "
              << qk_avg_ms << " ms\n";

    std::cout << "Softmax Time : "
              << softmax_avg_ms << " ms\n";

    std::cout << "PV Time      : "
              << pv_avg_ms << " ms\n";

    std::cout << "\n=== Benchmark ===\n";

    std::cout << "Total Time : "
              << total_ms << " ms\n";

    std::cout << "Avg Time   : "
              << avg_ms << " ms\n";

    // ========================================================
    // Correctness validation
    // ========================================================

    std::vector<__half>
        h_output(qkv_elements);

    std::vector<float>
        h_reference(qkv_elements);

    // Copy GPU output back to CPU
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        qkv_bytes,
        cudaMemcpyDeviceToHost));

    // Compute CPU reference
    cpuAttention(
        h_Q,
        h_K,
        h_V,
        h_reference);

    // Compare GPU and CPU results
    float max_error = 0.0f;

    for (size_t i = 0;
         i < qkv_elements;
         i++)
    {
        float gpu =
            __half2float(h_output[i]);

        float cpu =
            h_reference[i];

        max_error =
            fmaxf(
                max_error,
                fabsf(gpu - cpu));
    }

    // --------------------------------------------------------
    // Print validation result
    // --------------------------------------------------------

    std::cout << "\n=== Validation ===\n";

    std::cout << "Max Error  : "
              << max_error << "\n";

    if (max_error < 1e-2f)
        std::cout << "Validation : PASSED\n";
    else
        std::cout << "Validation : FAILED\n";

    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaEventDestroy(qk_start));
    CUDA_CHECK(cudaEventDestroy(qk_stop));

    CUDA_CHECK(cudaEventDestroy(softmax_start));
    CUDA_CHECK(cudaEventDestroy(softmax_stop));

    CUDA_CHECK(cudaEventDestroy(pv_start));
    CUDA_CHECK(cudaEventDestroy(pv_stop));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_probs));

    CUDA_CHECK(cudaFree(d_output));

    return 0;
}