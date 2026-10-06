=== GPU Attention Baseline V0 ===
Sequence Length : 512
Heads           : 12
Head Dimension  : 64
Precision       : FP16
Warmup          : 20
Iterations      : 100


=== Kernel Timing ===
QK Time      : 3.57684 ms
Softmax Time : 0.140607 ms
PV Time      : 0.727831 ms

=== Benchmark ===
Total Time : 465.545 ms
Avg Time   : 4.65545 ms

=== Validation ===
Max Error  : 5.37187e-06
Validation : PASSED


---------------------------------------------

=== GPU Attention Baseline V1 (QK shared memory) ===
Sequence Length : 512
Heads           : 12
Head Dimension  : 64
Precision       : FP16
Warmup          : 20
Iterations      : 100


=== Kernel Timing ===
QK Time      : 1.74121 ms
Softmax Time : 0.14073 ms
PV Time      : 0.720315 ms

=== Benchmark ===
Total Time : 344.829 ms
Avg Time   : 3.44829 ms

=== Validation ===
Max Error  : 5.37187e-06
Validation : PASSED

----------------------------------------------

=== GPU Attention V2 (QK + PV shared memory) ===
Sequence Length : 512
Heads           : 12
Head Dimension  : 64
Precision       : FP16
Warmup          : 20
Iterations      : 100


=== Kernel Timing ===
QK Time      : 1.74888 ms
Softmax Time : 0.140632 ms
PV Time      : 0.417794 ms

=== Benchmark ===
Total Time : 256.644 ms
Avg Time   : 2.56644 ms

=== Validation ===
Max Error  : 5.37187e-06
Validation : PASSED



-------------------------------------------------------



## Performance Comparison

| Version | Optimization | Avg Time (ms) | Speedup vs Previous | Time Reduction vs Previous | Speedup vs V0 | Time Reduction vs V0 |
|---|---|---:|---:|---:|---:|---:|
| V0 | Baseline | 4.65545 | 1.00× | — | 1.00× | — |
| V1 | QK shared memory | 3.44829 | 1.35× | 25.93% | 1.35× | 25.93% |
| V2 | QK + PV shared memory | 2.56644 | 1.34× | 25.58% | 1.81× | 44.87% |

### Kernel-Level Improvements

| Comparison | Kernel | Before (ms) | After (ms) | Time Reduction |
|---|---|---:|---:|---:|
| V0 → V1 | QK | 3.57684 | 1.74121 | 51.31% |
| V1 → V2 | PV | 0.720315 | 0.417794 | 41.99% |
| V0 → V2 | Total | 4.65545 | 2.56644 | 44.87% |

### Final Result

- **Overall speedup:** 1.81×
- **Overall latency reduction:** 44.87%
- **QK improvement (V0 → V1):** 51.31%
- **PV improvement (V1 → V2):** 41.99%
- **Validation:** PASSED
- **Maximum error:** 5.37187e-06
