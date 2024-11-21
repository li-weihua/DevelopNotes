#include "half_gemm_m16n8k16.h"

#include <cuda_fp16.h>
#include <cstdint>

#include "macros.h"

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// C = A * B
__global__ void half_gemm_m16n8k16(half* C, half* A, half* B) {
  __shared__ half tileA[MMA_M][MMA_K];  // row-major
  __shared__ half tileB[MMA_N][MMA_K];  // column-major
  __shared__ half tileC[MMA_M][MMA_N];  // row-major

  const int lane_id = threadIdx.x % 32;

  // 1. load global memory to shared memory
  for (int i = 0; i < MMA_M; ++i) {
    for (int j = 0; j < MMA_K; ++j) {
      tileA[i][j] = A[i * MMA_K + j];
    }
  }

  for (int i = 0; i < MMA_K; ++i) {
    for (int j = 0; j < MMA_N; ++j) {
      tileB[j][i] = B[i * MMA_N + j];
    }
  }

  __syncthreads();

  uint32_t regA[4];
  uint32_t regB[2];
  uint32_t regC[2] = {0, 0};

  uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&tileA[lane_id % 16][(lane_id / 16) * 8]);
  uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&tileB[lane_id % 8][((lane_id / 8) % 2) * 8]);

  // 2. load A and B to shared memory
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(regA[0]), "=r"(regA[1]), "=r"(regA[2]), "=r"(regA[3])
               : "r"(A_smem_lane_addr));

  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(regB[0]), "=r"(regB[1])
               : "r"(B_smem_lane_addr));

  // 3. tensor core mma
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      : "=r"(regC[0]), "=r"(regC[1])
      : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]), "r"(regB[0]), "r"(regB[1]), "r"(regC[0]), "r"(regC[1]));

  // 4. store to shared memory
  int groupID = lane_id >> 2;
  int threadID_in_group = lane_id % 4;

  *((uint32_t*)(&tileC[groupID][0]) + threadID_in_group) = regC[0];
  *((uint32_t*)(&tileC[groupID + 8][0]) + threadID_in_group) = regC[1];

  __syncthreads();

  // store to global memory
  for (int i = 0; i < MMA_M; ++i) {
    for (int j = 0; j < MMA_N; ++j) {
      C[i * MMA_N + j] = tileC[i][j];
    }
  }
}

void call_half_gemm_m16n8k16(half* C, half* A, half* B, cudaStream_t stream) {
  int grid = 1;
  int block = 32;

  half_gemm_m16n8k16<<<grid, block, 0, stream>>>(C, A, B);

  CUDA_CHECK(cudaPeekAtLastError());
}
