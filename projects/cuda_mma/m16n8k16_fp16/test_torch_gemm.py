import torch

torch.ops.load_library('./build/libhalf_gemm_m16n8k16_torch.so')

device = torch.device("cuda")
dtype = torch.float16

M, N, K = 16, 8, 16

torch.manual_seed(1)

A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)
C0 = torch.matmul(A, B)

C1 = torch.ops.mma.half_gemm_m16n8k16(A, B)
