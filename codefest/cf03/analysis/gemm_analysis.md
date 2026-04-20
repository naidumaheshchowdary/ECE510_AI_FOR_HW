# GEMM Analysis — Naive vs Tiled
## ECE 410/510 HW4AI | Codefest 3 | Spring 2026

**GPU:** Tesla T4
**Peak FP32:** 8.1 TFLOPS | **Peak BW:** 320.1 GB/s | **Ridge:** 25.30 FLOP/byte

| Kernel | Time (ms) | GFLOP/s | Bandwidth (GB/s) | AI (FLOP/B) | Bound |
|--------|-----------|---------|-----------------|-------------|-------|
| Naive | 5.158 | 416.3 | 1666.1 | 0.250 | memory-bound |
| Tiled (T=8) | 5.234 | 410.3 | 205.9 | 1.992 | memory-bound |

**Speedup: 0.99×**

---

## (a) Why the naive kernel is memory-bound

The naive kernel assigns one thread to each output element C[row][col].
Each thread reads a full row of A and a full column of B from global
DRAM — 2×1024 floats per output — with zero data reuse across threads.
The column access into B has stride 1024 (non-coalesced): 32 threads in
a warp issue 32 separate cache-line requests rather than one 128-byte
transaction. The arithmetic intensity is 0.2499 FLOP/byte,
far below the ridge point of 25.30 FLOP/byte. The GPU spends most
of its time waiting for DRAM data, not computing MACs.

## (b) How tiling reduces DRAM traffic

The tiled kernel cooperatively loads 8×8 blocks of A and B into
shared memory — one element per thread per tile. Each value in the
shared tile is reused 8 times by the 8 threads that need it before
the next DRAM load. This reduces total DRAM reads by a factor of
TILE_SIZE=8, raising arithmetic intensity from 0.250 to
1.992 FLOP/byte. Shared memory has ~100× lower latency
than DRAM, so the inner loop executes against on-chip bandwidth.
The two __syncthreads() barriers prevent race conditions when
loading and using each tile.

## (c) Whether the expected improvement was achieved

The measured speedup is 0.99×. This is consistent with the
8× reduction in DRAM loads being partially offset by synchronisation
overhead and low SM occupancy — tile size 8 means only 64 threads per
block, leaving most SMs underutilised. Both kernels remain memory-bound
(AI=1.992 < ridge=25.30 FLOP/byte), indicating that
DRAM bandwidth is still the primary bottleneck. The remaining
improvement requires a larger tile: tile size 32 would raise AI to
~16 FLOP/byte, improve warp occupancy from 64 to 1024 threads per
block, and move the kernel significantly closer to the compute-bound
region of the roofline.
