
/*
 * gemm_naive.cu — Naive CUDA GEMM: one thread per output element
 * ECE 410/510 HW4AI | Codefest 3 | Spring 2026
 *
 * Compile:  nvcc -O2 -o gemm_naive gemm_naive.cu
 * Run:      ./gemm_naive
 *
 * Each thread computes one C[row][col] by looping over K=1024.
 * All A and B accesses go directly to DRAM with no reuse.
 * B column access is non-coalesced (stride=N) — primary bottleneck.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N     1024
#define DTYPE float
#define RUNS  10
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__, \
  cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void gemm_naive_kernel(
    const DTYPE* __restrict__ A,
    const DTYPE* __restrict__ B,
          DTYPE* __restrict__ C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    DTYPE acc = 0.0f;
    for (int k = 0; k < n; ++k)
        acc += A[row*n+k] * B[k*n+col];
    C[row*n+col] = acc;
}

static void rand_init(DTYPE *M, int n) {
    for (int i = 0; i < n*n; ++i)
        M[i] = (DTYPE)(rand()/(float)RAND_MAX - 0.5f);
}

int main(void) {
    size_t bytes = (size_t)N*N*sizeof(DTYPE);

    /* GPU info */
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    double th_bw = 2.0 * p.memoryClockRate * 1e3 *
                   (p.memoryBusWidth / 8) / 1e9;
    printf("GPU: %s\n", p.name);
    printf("Theoretical BW: %.1f GB/s\n", th_bw);
    printf("Matrix: %dx%d FP32\n\n", N, N);

    /* Host alloc + init */
    DTYPE *hA = (DTYPE*)malloc(bytes);
    DTYPE *hB = (DTYPE*)malloc(bytes);
    DTYPE *hC = (DTYPE*)malloc(bytes);
    srand(42);
    rand_init(hA, N); rand_init(hB, N);

    /* Device alloc + copy */
    DTYPE *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N+31)/32, (N+31)/32);

    /* Warmup */
    for (int i = 0; i < 3; ++i)
        gemm_naive_kernel<<<grid,block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < RUNS; ++i)
        gemm_naive_kernel<<<grid,block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_total = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
    float ms = ms_total / RUNS;

    /* Metrics */
    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms * 1e-3) / 1e9;
    double bw_bytes = 2.0*N*N*N*sizeof(DTYPE) + (double)N*N*sizeof(DTYPE);
    double bw_gbs   = bw_bytes / (ms * 1e-3) / 1e9;
    double ai       = flops / bw_bytes;

    printf("=== gemm_naive results ===\n");
    printf("Avg time  : %.4f ms\n", ms);
    printf("GFLOP/s   : %.2f\n",   gflops);
    printf("BW        : %.2f GB/s\n", bw_gbs);
    printf("AI        : %.4f FLOP/byte\n", ai);

    /* Verify */
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    double ref = 0;
    for (int k = 0; k < N; ++k) ref += hA[k] * (double)hB[k*N];
    printf("C[0][0] GPU=%.5f  CPU=%.5f  err=%.2e\n",
           hC[0], (float)ref, fabs(hC[0]-(float)ref));

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
