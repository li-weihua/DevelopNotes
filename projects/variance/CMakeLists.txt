cmake_minimum_required(VERSION 3.22.1)

project(Variance CXX)

include(cmake/cpu_arch.cmake)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(var
  src/main.cc
  src/var_scalar.cc
)

if (ARCH_X86 OR ARCH_X64)
  target_sources(var PRIVATE
    src/var_sse.cc
  )

  if (MSVC)
    set_source_files_properties(var_sse.cc PROPERTIES COMPILE_FLAGS "/arch:SSE2")
  else()
    set_source_files_properties(var_sse.cc PROPERTIES COMPILE_FLAGS "-msse2")
  endif()

elseif(ARCH_ARM OR ARCH_ARM64)
  target_sources(var PRIVATE
    src/var_neon.cc
  )
endif()
