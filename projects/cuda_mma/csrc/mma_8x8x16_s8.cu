#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

__global__ void tensor_core_example_8x8x16(int32_t *D, uint32_t const *A, uint32_t const *B,
                                           int32_t const *C) {
  int outer = threadIdx.x / 4;
  int inner = threadIdx.x % 4;

  int c_row = threadIdx.x / 4;
  int c_col = (threadIdx.x % 4) * 2;

  int ab_idx = outer * 4 + inner;
  int cd_idx = c_row * 8 + c_col;

  int32_t regc[2] = {0};

  if (C != nullptr) {
    regc[0] = C[cd_idx];
    regc[1] = C[cd_idx + 1];
  }

  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(D[cd_idx]), "=r"(D[cd_idx + 1])
      : "r"(A[ab_idx]), "r"(B[ab_idx]), "r"(regc[0]), "r"(regc[1]));
}

torch::Tensor Matmul_8x8x16_s8(const torch::Tensor &A, const torch::Tensor &B,
                               std::optional<const torch::Tensor> &C) {
  constexpr int M = 8;
  constexpr int N = 8;
  constexpr int K = 16;

  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major >= 8;
  TORCH_CHECK(is_sm8x, "m8n8k16 only supports Ampere GPUs or newer.");

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());

  auto D = torch::empty({M, N}, options);

  TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
  TORCH_CHECK(B.is_cuda(), "B must be on CUDA");

  CHECK_SHAPE(A, M, K);
  CHECK_SHAPE(B, K, N);

  bool has_c = C.has_value();

  at::Tensor ctensor;
  if (has_c) {
    ctensor = C.value();
    TORCH_CHECK(ctensor.is_cuda(), "C must be on CUDA");
  }

  // at::cuda::CUDAGuard device_guard{(char)A.get_device()};

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int32_t *ptr_D = reinterpret_cast<int32_t *>(D.data_ptr());
  uint32_t *ptr_A = reinterpret_cast<uint32_t *>(A.data_ptr());
  uint32_t *ptr_B = reinterpret_cast<uint32_t *>(B.data_ptr());
  int32_t *ptr_C = nullptr;
  if (has_c) ptr_C = reinterpret_cast<int32_t *>(ctensor.data_ptr());

  tensor_core_example_8x8x16<<<1, 32>>>(ptr_D, ptr_A, ptr_B, ptr_C);

  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CudaMMA";
  m.def("matmul_8x8x16_s8", &Matmul_8x8x16_s8, "matmul for 8x8x16 s8");
}
