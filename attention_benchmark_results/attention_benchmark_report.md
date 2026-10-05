# GPU Attention Performance — V0 / V1 / V2

## Configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Sequence Length | 512 |
| Heads | 12 |
| Head Dimension | 64 |
| Precision | FP16 |
| Warmup | 20 |
| Iterations | 100 |

## Benchmark

| Version | QK (ms) | Softmax (ms) | PV (ms) | Average (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| V0 | 3.55002 | 0.139611 | 0.684278 | 4.72884 | 1.000× |
| V1 | 1.73496 | 0.139923 | 0.680815 | 3.02066 | 1.565× |
| V2 | 1.75110 | 0.140719 | 0.429425 | 2.81819 | 1.678× |

## Validation

All versions passed validation with Max Error = 5.37187e-06.

## Key Results

- V1 QK reduction: 51.13%
- V2 PV reduction: 37.24%
- V2 end-to-end latency reduction: 40.40%
- V2 end-to-end speedup: 1.678×

## Experiments

### V0 — Baseline
Reference implementation with baseline QK, Softmax and PV kernels.

### V1 — QK Shared Memory
QK uses shared-memory tiling. QK latency improved from 3.55002 ms to 1.73496 ms.

### V2 — QK + PV Shared Memory
PV also uses shared-memory tiling. PV latency improved to 0.429425 ms, reducing end-to-end latency to 2.81819 ms.

## Conclusion

V2 improves the baseline from 4.72884 ms to 2.81819 ms per iteration,
achieving approximately 1.68× end-to-end speedup while maintaining numerical correctness.
