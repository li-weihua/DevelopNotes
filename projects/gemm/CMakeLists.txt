cmake_minimum_required(VERSION 3.22)

project(Gemm CXX ASM)

include(cmake/cpu_arch.cmake)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mlas SHARED
  lib/platform.cpp
  lib/sgemm.cpp
)
target_compile_definitions(mlas PRIVATE BUILD_MLAS_NO_ONNXRUNTIME)
target_include_directories(mlas PUBLIC inc)

if (ARCH_X64)
  set(mlas_platform_srcs_sse2
    lib/x86_64/SgemmKernelSse2.S
    lib/x86_64/SgemmTransposePackB16x4Sse2.S
  )
  set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

  set(mlas_platform_srcs_avx
    lib/x86_64/SgemmKernelAvx.S
    lib/x86_64/SgemmKernelM1Avx.S
    lib/x86_64/SgemmKernelM1TransposeBAvx.S
    lib/x86_64/SgemmTransposePackB16x4Avx.S
  )
  set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

  set(mlas_platform_srcs_avx2
    lib/x86_64/SgemmKernelFma3.S
  )

  set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

  set(mlas_platform_srcs_avx512f
    lib/x86_64/SgemmKernelAvx512F.S
  )
  set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

  set(mlas_platform_srcs
    ${mlas_platform_srcs_sse2}
    ${mlas_platform_srcs_avx}
    ${mlas_platform_srcs_avx2}
    ${mlas_platform_srcs_avx512f}
  )

elseif(ARCH_ARM64)
  set(mlas_platform_srcs
    ${MLAS_SRC_DIR}/aarch64/SgemmKernelNeon.S
    ${MLAS_SRC_DIR}/aarch64/SgemvKernelNeon.S
  )
else()
  message(FATAL_ERROR "not support platform!")
endif()

target_sources(mlas PRIVATE ${mlas_platform_srcs})


add_executable(test_gemm
  test/gemm.cc
)
target_link_libraries(test_gemm PRIVATE mlas)
