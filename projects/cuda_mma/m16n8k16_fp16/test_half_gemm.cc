#include <iostream>

#include "half_gemm_m16n8k16.h"
#include "macros.h"

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

int main() {
  // malloc host
  half* A = new half[MMA_M * MMA_K];
  half* B = new half[MMA_K * MMA_M];
  half* C = new half[MMA_M * MMA_N];

  // init host
  for (int i = 0; i < MMA_M * MMA_K; ++i) A[i] = 1.0f;
  for (int i = 0; i < MMA_N * MMA_K; ++i) B[i] = 0.5f;
  for (int i = 0; i < MMA_M * MMA_N; ++i) C[i] = 0.0f;

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  half* d_A = nullptr;
  half* d_B = nullptr;
  half* d_C = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(half) * MMA_M * MMA_K));
  CUDA_CHECK(cudaMalloc((void**)&d_B, sizeof(half) * MMA_N * MMA_K));
  CUDA_CHECK(cudaMalloc((void**)&d_C, sizeof(half) * MMA_M * MMA_N));

  // copy host to device
  CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(half) * MMA_M * MMA_K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(half) * MMA_N * MMA_K, cudaMemcpyHostToDevice));

  call_half_gemm_m16n8k16(d_C, d_A, d_B, stream);

  CUDA_CHECK(cudaMemcpy(C, d_C, sizeof(half) * MMA_M * MMA_N, cudaMemcpyDeviceToHost));

  // print c
  for (int i = 0; i < MMA_M; ++i) {
    for (int j = 0; j < MMA_N; ++j) {
      std::cout << (float)C[i * MMA_N + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // free device
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // free host
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
