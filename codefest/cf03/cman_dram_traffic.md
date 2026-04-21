# CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply
**ECE 410/510 HW4AI — Spring 2026 — Codefest 3**
**CMAN completed without AI assistance**

N = 32, T = 8, FP32 = 4 bytes/element
Peak Compute = 10 TFLOPS = 10,000 GFLOP/s | DRAM BW = 320 GB/s | Ridge = 31.25 FLOP/byte

---

## (a) Naive DRAM Traffic

For one output element C[i][j] = Σ A[i][k] × B[k][j]:
- Each element of **B is accessed N = 32 times** (once per row i of the output)

Total element accesses across full N×N output:
```
A accesses = N × N × N = N³ = 32³ = 32,768
B accesses = N × N × N = N³ = 32³ = 32,768
C writes   = N²             = 32²  =  1,024
```

Total DRAM traffic:
```
Traffic_naive = (N³ + N³ + N²) × 4
              = (32,768 + 32,768 + 1,024) × 4
              = 66,560 × 4
              = 266,240 bytes
```

---

## (b) Tiled DRAM Traffic (T = 8)

```
Tiles per dimension = N/T = 32/8 = 4
Output tiles        = (N/T)² = 16
K-phases per tile   = N/T = 4
```

Each output tile requires N/T K-phases; each phase loads one A tile and one B tile (T×T each) from DRAM once:
```
A tile loads = (N/T)³ = 4³ = 64 tiles  →  64 × T² = 64 × 64 = 4,096 elements
B tile loads = (N/T)³ = 4³ = 64 tiles  →  64 × T² = 64 × 64 = 4,096 elements
C writes     = N²                                              = 1,024 elements
```

Total DRAM traffic:
```
Traffic_tiled = (4,096 + 4,096 + 1,024) × 4
              = 9,216 × 4
              = 36,864 bytes = 36KB
```

---

## (c) Traffic Ratio

```
Ratio = Traffic_naive / Traffic_tiled = 266,240 / 36,864 ≈ 7.22
```

For A+B traffic only (excluding C writes, same in both):
```
Ratio (A+B) = (2 × N³) / (2 × N³/T) = T = 8
```

**One-sentence explanation:** The ratio equals T (the tile size) because tiling loads each T×T tile of A and B from DRAM exactly once into shared memory and reuses each element T times within the tile, replacing T redundant DRAM fetches with a single load per element.

---

## (d) Execution Times and Bound Classification

FLOPs = 2 × N³ = 2 × 32³ = **65,536 FLOPs**

### Naive

```
AI_naive = 65,536 / 266,240 = 0.246 FLOP/byte

Bottleneck: MEMORY-BOUND  (0.246 < 31.25 ridge point)

Execution time = Traffic / BW = 266,240 / (320 × 10⁹) = 832 ns
```

### Tiled

```
AI_tiled = 65,536 / 36,864 = 1.778 FLOP/byte

Bottleneck: MEMORY-BOUND  (1.778 < 31.25 ridge point, but 7.2× less traffic)

Execution time = Traffic / BW = 36,864 / (320 × 10⁹) = 115 ns
```

### Summary

| | Naive | Tiled (T=8) |
|---|---|---|
| DRAM Traffic | 266,240 bytes | 36,864 bytes |
| AI (FLOP/byte) | 0.246 | 1.778 |
| Bound | **Memory-bound** | **Memory-bound** |
| Execution Time | **832 ns** | **115 ns** |
