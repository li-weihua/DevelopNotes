#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void ldmatrix_m8n8_kernel(uint32_t* R, void* smem_ptr) {
  // convert generic to shared memory
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%1}, [%2];\n" : "=r"(R[0]) : "r"(smem_addr));
}

constexpr int kTileSize = 8;

// load 8x8 matrix
__global__ void ldmatrix_m8n8(half* output, half* input, int lda) {
  __shared__ half shared_matrix[kTileSize * kTileSize];

  // load input to shared memory
  int lane_id = threadIdx.x % 32;

  shared_matrix[lane_id] = input[lane_id];
  shared_matrix[lane_id + 32] = input[lane_id + 32];

  __syncthreads();

  // load shared memory to register
  uint32_t R[1];
  ldmatrix_m8n8_kernel(&R, &shared_matrix[(lane_id % 8) * 8]);

  // store register to global memory for test
  // output[lane_id] = R[0];
}

// input A shape: [8,8], bfloat16 or half
torch::Tensor LoadMatrixM8N8(const torch::Tensor& A) {}
