#include "half_gemm_m16n8k16.h"

#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor DoHalfGemmShapeM16N8K16(const torch::Tensor& A, const torch::Tensor& B) {
  auto stream = at::cuda::getCurrentCUDAStream();

  // check A, B on cuda
  TORCH_CHECK(A.device().type() == c10::kCUDA);
  TORCH_CHECK(B.device().type() == c10::kCUDA);

  // check A, B type
  TORCH_CHECK(A.dtype() == torch::kHalf);
  TORCH_CHECK(B.dtype() == torch::kHalf);

  // check A, B shape
  TORCH_CHECK(A.dim() == 2);
  TORCH_CHECK(B.dim() == 2);

  TORCH_CHECK(A.size(0) == 16);
  TORCH_CHECK(A.size(1) == 16);
  TORCH_CHECK(B.size(0) == 16);
  TORCH_CHECK(B.size(1) == 8);

  auto C = torch::empty({16, 8}, A.options());

  half* ptr_C = reinterpret_cast<half*>(C.data_ptr());
  half* ptr_A = reinterpret_cast<half*>(A.data_ptr());
  half* ptr_B = reinterpret_cast<half*>(B.data_ptr());

  call_half_gemm_m16n8k16(ptr_C, ptr_A, ptr_B, stream);

  return C;
}

// Define the operator
TORCH_LIBRARY(mma, m) { m.def("half_gemm_m16n8k16", &DoHalfGemmShapeM16N8K16); }
