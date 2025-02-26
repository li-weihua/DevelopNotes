cmake_minimum_required(VERSION 3.22)

project(CUDA_HALF_GEMM LANGUAGES CXX CUDA)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real)
find_package(CUDAToolkit REQUIRED)

set(TORCH_CUDA_ARCH_LIST "8.0 8.6 8.9 9.0")
include(cmake/set_torch.cmake)

add_library(half_gemm_m16n8k16 STATIC half_gemm_m16n8k16.cu)
target_link_libraries(half_gemm_m16n8k16 PUBLIC CUDA::cudart)

add_executable(test_half_gemm test_half_gemm.cc)
target_link_libraries(test_half_gemm PRIVATE half_gemm_m16n8k16)

add_library(half_gemm_m16n8k16_torch SHARED half_gemm_m16n8k16_torch.cc)
target_link_libraries(half_gemm_m16n8k16_torch PRIVATE half_gemm_m16n8k16 ${TORCH_LIBRARIES})
