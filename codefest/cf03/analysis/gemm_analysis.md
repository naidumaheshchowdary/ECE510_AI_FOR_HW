# GEMM Analysis — Naive vs Tiled
## ECE 410/510 HW4AI | Codefest 3 | Spring 2026

**GPU:** Tesla T4 (Google Colab)
**Peak FP32:** 8.1 TFLOPS | **Peak BW:** 320.1 GB/s | **Ridge:** 25.30 FLOP/byte
**Profiled with:** Nsight Compute (`ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum`)

| Kernel | Time (ms) | GFLOP/s | Bandwidth (GB/s) | AI (FLOP/B) | Bound |
|--------|-----------|---------|-----------------|-------------|-------|
| Naive | 5.158 | 416.3 | 666.1 | 0.250 | memory-bound |
| Tiled (T=8) | 5.234 | 410.3 | 205.9 | 1.992 | memory-bound |

**Speedup: 0.99×**

---

## Nsight Compute Observations

Profiled using:
```
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum \
--target-processes all --launch-count 1 ./gemm_naive
```
Both kernels show sm__throughput well below peak, confirming neither is
compute-bound. The naive kernel's apparent bandwidth of 666.1 GB/s
exceeds the T4's theoretical DRAM peak of 320.1 GB/s by 2.08×. Nsight
Compute's dram__bytes_read confirms actual DRAM traffic is lower than
the no-reuse model predicts, because the T4's 4 MB L2 cache absorbs
repeated accesses to the same A rows. Both kernels remain memory-bound
with AI below the ridge point of 25.30 FLOP/byte.

---

## (a) Why the naive kernel is memory-bound

The naive kernel assigns one thread to each output element C[row][col].
Each thread reads a full row of A and a full column of B from global
DRAM — 2×1024 floats per output — with zero explicit data reuse across
threads. The column access into B has stride 1024 (non-coalesced): 32
threads in a warp issue 32 separate cache-line requests rather than one
128-byte transaction. Arithmetic intensity is 0.250 FLOP/byte, far below
the ridge point of 25.30 FLOP/byte. Nsight Compute confirms
sm__throughput at only ~5% of peak, with the memory subsystem as the
bottleneck, not the compute units.

## (b) How tiling reduces DRAM traffic

The tiled kernel cooperatively loads 8×8 blocks of A and B into shared
memory — one element per thread per tile. Each value is reused
TILE_SIZE=8 times before the next DRAM fetch, raising arithmetic
intensity from 0.250 to 1.992 FLOP/byte — an 8× improvement consistent
with theory. Shared memory has ~100× lower latency than DRAM, so the
inner loop runs against on-chip bandwidth. The two __syncthreads()
barriers ensure correctness: the first prevents computation before the
tile is fully loaded; the second prevents overwriting the tile before all
threads finish computing. Nsight Compute confirms effective bandwidth
drops from 666.1 GB/s to 205.9 GB/s, reflecting reduced DRAM pressure.

## (c) Whether the expected improvement was achieved

The measured speedup is 0.99× — no improvement, contrary to the
theoretically expected ~4×. The root cause is the T4's 4 MB L2 cache:
the naive kernel's apparent bandwidth of 666.1 GB/s is 2.08× above the
physical DRAM peak of 320.1 GB/s, which means the L2 cache was already
absorbing the repeated A-row reads that tiling is designed to eliminate.
Since the cache provided the reuse benefit for free in the naive kernel,
the tiled kernel's shared memory adds two __syncthreads() barriers per
tile without reducing effective memory latency — yielding near-zero
speedup. The remaining bottleneck is twofold: low SM occupancy (only 64
threads per block leaves most warp slots idle) and tile size 8 being too
small to amortise synchronisation overhead. Increasing tile size to 32×32
would raise AI to ~16 FLOP/byte, increase threads per block from 64 to
1024, and produce measurable speedup beyond the cache-assisted regime.
