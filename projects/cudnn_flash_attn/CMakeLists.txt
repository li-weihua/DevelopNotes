cmake_minimum_required(VERSION 3.28)

project(cudnn_flash_attn CXX)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

include(cmake/torch.cmake)
include(cmake/cudnn.cmake)
include(FetchContent)

set(CUDNN_FRONTEND_SKIP_JSON_LIB ON)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_TESTS OFF)
FetchContent_Declare(
  cudnn_frontend
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
  GIT_TAG        936021bfed8c91dc416af1588b2c4eca631a9e45 # v1.8.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cudnn_frontend)

add_library(fusedattn STATIC cpp/fusedattn.cc)
target_link_libraries(fusedattn PUBLIC cudnn_frontend CUDNN::cudnn_all CUDA::cudart)

add_library(fusedattn_torch SHARED cpp/fusedattn_torch.cc)
target_link_libraries(fusedattn_torch PUBLIC fusedattn ${TORCH_LIBRARIES})

add_executable(test_fusedattn cpp/test_fusedattn.cc)
target_link_libraries(test_fusedattn PUBLIC fusedattn)
