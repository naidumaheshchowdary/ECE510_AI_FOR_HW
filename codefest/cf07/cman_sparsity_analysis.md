# CMAN - Sparsity Breakeven Analysis
ECE 410/510 - Codefest 7

---

Given

N = 512, NxN weight Matrix, N^2 MACs and N^2 FP32 loads
CSR Storage: values array (FP32, 4 bytes each), col_idx array (INT32, 4 bytes each), row_ptr array (INT32, 4 bytes each, length N+1)
Dense Storage: 4 bytes per element

---

## Task 1 - Four Expressions (N = 512, sparsity = s)

Let nnz = N^2 * (1-s) = number of non-zero weights

(a) Dense compute (FLOPs)

Each element needs 1 multiply and 1 add, so 2 FLOPs per element.

FLOPs_dense = 2 * N^2 = 2 * 512^2 = 524,288 FLOPs

(b) Dense memory (bytes)

Every element stored as FP32 including zeros.

Bytes_dense = 4 * N^2 = 4 * 262,144 = 1,048,576 bytes (1 MB)

(c) Sparse compute (FLOPs) as a function of s

Zero weights contribute nothing so we skip them. Only nnz elements need computation.

FLOPs_sparse = 2 * N^2 * (1-s)

(d) Sparse memory (bytes) as a function of s

CSR uses three arrays:
- values[]: nnz x 4 bytes (FP32 weights)
- col_idx[]: nnz x 4 bytes (INT32 column indices)
- row_ptr[]: (N+1) x 4 bytes (INT32 row pointers)

Total = 4*nnz + 4*nnz + 4*(N+1)
      = 8 * N^2 * (1-s) + 4*(N+1)

---

## Task 2 - FLOPs Speedup and 2x Sparsity

Speedup formula:

Speedup(s) = FLOPs_dense / FLOPs_sparse
           = 2*N^2 / (2*N^2*(1-s))
           = 1 / (1-s)

The 2*N^2 cancels so speedup depends only on sparsity, not matrix size.

Solving for 2x speedup:

1 / (1-s) = 2
1 - s = 1/2
s = 0.5

At s = 0.5 (50% sparsity), the FLOPs speedup equals 2x.

---

## Task 3 - Memory Breakeven Sparsity

Set sparse bytes = dense bytes and solve for s:

8*N^2*(1-s) + 4*(N+1) = 4*N^2

8*N^2*(1-s) = 4*N^2 - 4*N - 4

(1-s) = (4*N^2 - 4*N - 4) / (8*N^2)

(1-s) = 1/2 - 1/(2N) - 1/(2*N^2)

s = 1/2 + 1/(2N) + 1/(2*N^2)

Plugging in N = 512:

s = 0.5 + 1/1024 + 1/524288
s = 0.5 + 0.000977 + 0.0000019
s = 0.501

Memory breakeven: s = 0.501 (about 50.1%)

Above this sparsity level, CSR sparse format uses less memory than dense storage.
The row pointer overhead (4 x 513 = 2,052 bytes) is negligible compared to the N^2 savings,
so the breakeven point is essentially 50%.

---

## Task 4 - End-to-End Execution Time Speedup at s = 0.9

System: memory-bandwidth-limited, BW = 320 GB/s, N = 512, s = 0.9

Step 1 - Compute bytes loaded:

Dense:
  Bytes_dense = 4 * N^2 = 4 * 262,144 = 1,048,576 bytes

Sparse (CSR):
  nnz = N^2 * (1-s) = 262,144 * 0.1 = 26,214
  Bytes_sparse = 8 * N^2 * (1-s) + 4*(N+1)
               = 8 * 262,144 * 0.1 + 4 * 513
               = 209,715 + 2,052
               = 211,767 bytes

Step 2 - Compute execution time (T = Bytes / BW):

T_dense  = 1,048,576 / (320 x 10^9) = 3.277 us
T_sparse =   211,767 / (320 x 10^9) = 0.662 us

Step 3 - Speedup:

Speedup = T_dense / T_sparse = 1,048,576 / 211,767 = 4.95x

End-to-end speedup at s = 0.9 is approximately 4.95x.

This is less than the 10x FLOPs speedup because the col_idx[] array adds 4 bytes per
non-zero on top of the 4 bytes for the weight value itself. So each non-zero costs 8 bytes
in CSR vs 4 bytes in dense. At s = 0.9, sparse memory ends up being about 20.2% of dense
(not 10%), which is why the memory-bound speedup is roughly half the FLOPs speedup.

---
