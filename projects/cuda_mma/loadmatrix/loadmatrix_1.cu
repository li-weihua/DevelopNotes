#include <cuda_runtime.h>

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void ldmatrix_m8n8(uint32_t* R, void* smem_ptr) {
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
}

// input A shape: [8,8], bfloat16 or half
torch::Tensor LoadMatrixM8N8(const torch::Tensor& A) {}
