#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>  // half

void call_half_gemm_m16n8k16(half* C, half* A, half* B, cudaStream_t stream);
