
/*
 * gemm_tiled.cu — Shared-memory tiled CUDA GEMM, tile size 8
 * ECE 410/510 HW4AI | Codefest 3 | Spring 2026
 *
 * Compile:  nvcc -O2 -o gemm_tiled gemm_tiled.cu
 * Run:      ./gemm_tiled
 *
 * 8x8 tile loaded into __shared__ memory per block.
 * Each DRAM value reused TILE_SIZE=8 times -> 8x less DRAM traffic.
 * AI_tiled = TILE_SIZE / 2 = 4.0 FLOP/byte (vs 0.25 naive)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N         1024
#define TILE_SIZE 8
#define DTYPE     float
#define RUNS      10
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__, \
  cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void gemm_tiled_kernel(
    const DTYPE* __restrict__ A,
    const DTYPE* __restrict__ B,
          DTYPE* __restrict__ C, int n)
{
    __shared__ DTYPE sA[TILE_SIZE][TILE_SIZE];
    __shared__ DTYPE sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    DTYPE acc = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int ac = t * TILE_SIZE + threadIdx.x;
        int br = t * TILE_SIZE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row<n && ac<n) ? A[row*n+ac] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (br<n  && col<n)? B[br*n+col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row*n+col] = acc;
}

static void rand_init(DTYPE *M, int n) {
    for (int i = 0; i < n*n; ++i)
        M[i] = (DTYPE)(rand()/(float)RAND_MAX - 0.5f);
}

int main(void) {
    size_t bytes = (size_t)N*N*sizeof(DTYPE);

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    double th_bw = 2.0 * p.memoryClockRate * 1e3 *
                   (p.memoryBusWidth / 8) / 1e9;
    printf("GPU: %s\n", p.name);
    printf("Theoretical BW: %.1f GB/s\n", th_bw);
    printf("Matrix: %dx%d FP32  Tile: %d\n\n", N, N, TILE_SIZE);

    DTYPE *hA = (DTYPE*)malloc(bytes);
    DTYPE *hB = (DTYPE*)malloc(bytes);
    DTYPE *hC = (DTYPE*)malloc(bytes);
    srand(42);
    rand_init(hA, N); rand_init(hB, N);

    DTYPE *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    for (int i = 0; i < 3; ++i)
        gemm_tiled_kernel<<<grid,block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < RUNS; ++i)
        gemm_tiled_kernel<<<grid,block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_total = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
    float ms = ms_total / RUNS;

    double flops = 2.0 * N * N * N;
    double gflops = flops / (ms * 1e-3) / 1e9;
    double bw_bytes = 2.0*(double)N*N*(N/TILE_SIZE)*sizeof(DTYPE)
                    + (double)N*N*sizeof(DTYPE);
    double bw_gbs   = bw_bytes / (ms * 1e-3) / 1e9;
    double ai       = flops / bw_bytes;

    printf("=== gemm_tiled results (TILE=%d) ===\n", TILE_SIZE);
    printf("Avg time  : %.4f ms\n", ms);
    printf("GFLOP/s   : %.2f\n",   gflops);
    printf("BW        : %.2f GB/s\n", bw_gbs);
    printf("AI        : %.4f FLOP/byte  (%.1fx naive)\n",
           ai, (double)TILE_SIZE / 2.0);

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
