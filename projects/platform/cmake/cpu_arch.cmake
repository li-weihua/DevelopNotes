# Adapted from onnxruntime's cmake
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# We define flowing cpu architectures:
#   ARCH_X86
#   ARCH_X64
#   ARCH_ARM
#   ARCH_ARM64
#   ARCH_RISCV64
#   ARCH_POWER
#   ARCH_LOONGARCH64

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message("Build Emscripten!")

elseif (MSVC)
  if (CMAKE_VS_PLATFORM_NAME)
    # Multi-platform generator
    set(target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  message(STATUS "target_platform: ${target_platform}")

  if (target_platform STREQUAL "ARM64")
    set(target_platform "ARM64")
    set(ARCH_ARM64 TRUE)
    enable_language(ASM_MARMASM)
  elseif (target_platform STREQUAL "ARM64EC")
    enable_language(ASM_MARMASM)
  elseif (target_platform STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(target_platform "ARM")
    set(ARCH_ARM TRUE)
    enable_language(ASM_MARMASM)
  elseif (target_platform STREQUAL "x64" OR target_platform STREQUAL "x86_64" OR target_platform STREQUAL "AMD64" OR CMAKE_GENERATOR MATCHES "Win64")
    set(target_platform "x64")
    set(ARCH_X64 TRUE)
    enable_language(ASM_MASM)
  elseif (target_platform STREQUAL "win32" OR target_platform STREQUAL "Win32" OR target_platform STREQUAL "x86" OR target_platform STREQUAL "i386" OR target_platform STREQUAL "i686")
    set(target_platform "x86")
    set(ARCH_X86 TRUE)
    enable_language(ASM_MASM)
    message("Enabling SAFESEH for x86 build")
    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")
  else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

elseif (APPLE)
  enable_language(ASM)

  if (CMAKE_OSX_ARCHITECTURES)
    set(target_platform ${CMAKE_OSX_ARCHITECTURES})
  endif()

  if (NOT target_platform)
    set(target_platform ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()

  foreach(OSX_ARCH ${target_platform})
    if (OSX_ARCH STREQUAL "arm64")
      set(ARCH_ARM64 TRUE)
    elseif (OSX_ARCH STREQUAL "arm64e")
      set(ARCH_ARM64 TRUE)
    elseif (OSX_ARCH STREQUAL "arm")
      set(ARCH_ARM TRUE)
    elseif (OSX_ARCH STREQUAL "x86_64")
      set(ARCH_X64 TRUE)
    elseif (OSX_ARCH STREQUAL "i386")
      set(ARCH_X86 TRUE)
    endif()
  endforeach()

else ()
  set(target_platform ${CMAKE_SYSTEM_PROCESSOR})
  enable_language(ASM)

  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(ARCH_ARM TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
      set(ARCH_ARM64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
      set(ARCH_X64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
      set(ARCH_X86 TRUE)
    endif()

  else()
    #Android: armv7-a aarch64 i686 x86_64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
      set(ARCH_ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
      set(ARCH_ARM TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
      set(ARCH_ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(ARCH_X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
      set(ARCH_X64 TRUE)
    endif()
  endif()
endif()

message(STATUS "target_platform: ${target_platform}")
