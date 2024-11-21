#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

// cuda check macro
#define CUDA_CHECK(statments)                                                                            \
  do {                                                                                                   \
    cudaError_t status = (statments);                                                                    \
    if (cudaSuccess != status) {                                                                         \
      std::fprintf(stderr, "[%s:%d] cuda error: %s, %s\n", __FILE__, __LINE__, cudaGetErrorName(status), \
                   cudaGetErrorString(status));                                                          \
      std::abort();                                                                                      \
    }                                                                                                    \
  } while (false)
