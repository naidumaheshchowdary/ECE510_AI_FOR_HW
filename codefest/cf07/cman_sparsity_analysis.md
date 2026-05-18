# CMAN - Sparsity Breakeven Analysis
**ECE 410/510 - Codefest 7**

---
Given
```
N = 512, NxN weight Matrix, N² Mac and N² FP32 loads
CSR Storage: value array[int32, 4bytes], Column array[int32,4bytes], Row Pointer array[int32per pointer,4bytes, length(N+1)]
Dense Storage 4bytes
```

## Task 1: Four Expressions (N = 512, sparsity = s)

Let **nnz = N²(1 - s)** = number of non zero weights.

| | Expression | Value at N=512 |
|---|---|---|
| **(a) Dense compute (FLOPs)** | `2 × N²` | 524,288 FLOPs |
| **(b) Dense memory (bytes)** | `4 × N²` | 1,048,576 B (1 MB) |
| **(c) Sparse compute (FLOPs)** | `2 × N²(1 - s)` | depends on s |
| **(d) Sparse memory (bytes)** | `8 × N²(1-s) + 4(N-1)` | depends on s |

**Derivation of (d):**
CSR format uses three arrays:
- `values[]`: nnz × 4 bytes (FP32 weights)
- `col_idx[]`: nnz × 4 bytes (INT32 column indices)
- `row_ptr[]`: (N+1) × 4 bytes (INT32 row pointers)

Total = 4·nnz + 4·nnz + 4(N+1) = **8·N²(1-s) + 4(N+1)**

---

## Task 2: FLOPs Speedup and 2× Sparsity

**Speedup formula:**

```
Speedup(s) = FLOPs_dense / FLOPs_sparse
           = 2N² / [2N²(1-s)] (2N² is common)
           = 1 / (1-s)
```

The 2N² cancels because speedup depends only on sparsity, not matrix size.

**Solving for 2× speedup:**

```
1 / (1-s) = 2
1-s = 0.5 = (1/2)
s = 1-0.5 = 0.5
```

> **At s = 0.5 (50% sparsity), the FLOPs speedup equals 2×.**

---

## Task 3: Memory Breakeven Sparsity (Derivation)

Set sparse bytes = dense bytes and solve for s:

```
8·N²(1-s) + 4(N+1) = 4·N²

8·N²(1-s) = 4N² - 4N - 4

(1-s) = [4N² - 4N - 4] / (8N²)
(1-s) = 1/2 - 1/(2N) - 1/(2N²)

s = 1/2 + 1/(2N) + 1/(2N²)
```

**Plugging in N = 512:**

```
s = 0.5 + 1/1024 + 1/524288
s = 0.5 + 0.000977 + 0.0000019
s ≈ 0.501
```

> **Memory breakeven: s ≈ 0.5010 (≈ 50.1%)**
>
> Above this sparsity, CSR sparse format uses less memory than dense.
> The row pointer overhead (4 × 513 = 2,052 bytes) is negligible vs. N² savings, so the breakeven is essentially 50.1%.

---

## Task 4: End-to-End Execution Time Speedup at s = 0.9

**System:** Memory bandwidth limited, BW = 320 GB/s, N = 512, s = 0.9

**Step 1: Compute bytes loaded:**

```
Dense:
  Bytes_dense = 4 × N² = 4 × 262,144 = 1,048,576 bytes

Sparse (CSR):
  nnz = N²(1-s) = 262,144 × 0.1 = 26,214.4 ≈ 26,214
  Bytes_sparse = 8·N²(1-s) + 4(N+1)
               = 8 × 262,144 × 0.1 + 4 × 513
               = 209,715 + 2,052
               = 211,767 bytes
```

**Step 2: Compute execution time (T = Bytes / BW):**

```
T_dense  = 1,048,576 / (320 × 10⁹) = 3.277 µs
T_sparse =   211,767 / (320 × 10⁹) = 0.662 µs
```

**Step 3: Speedup:**

```
Speedup = T_dense / T_sparse = 1,048,576 / 211,767 ≈ 4.95×
```

> **End-to-end speedup at s = 0.9: ≈ 4.95×**
>
> This is significantly less than the 10× FLOPs speedup because CSR index overhead is non-trivial:
> the `col_idx[]` array doubles the per-nonzero storage cost, so sparse memory is ~20.2% of dense
> (not 10%). The row pointer array adds another 2 KB. In a memory-bandwidth-limited system,
> actual speedup is bounded by the memory ratio, not the compute ratio.

---
