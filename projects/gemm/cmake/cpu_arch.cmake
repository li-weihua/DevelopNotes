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
  elseif (target_platform STREQUAL "Win32" OR target_platform STREQUAL "x86" OR target_platform STREQUAL "i386" OR target_platform STREQUAL "i686")
    set(target_platform "x86")
    set(ARCH_X86 TRUE)
    enable_language(ASM_MASM)
    message("Enabling SAFESEH for x86 build")
    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")
  else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

else()

  set(target_platform ${CMAKE_SYSTEM_PROCESSOR})

  if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR is not set. Please set it in cmake toolchain file.")
    set(target_platform ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()

  enable_language(ASM)

  if (APPLE)
    list(LENGTH CMAKE_OSX_ARCHITECTURES CMAKE_OSX_ARCHITECTURES_LEN)

    # NOTE: We do not support build universal binary on macos!
    if (CMAKE_OSX_ARCHITECTURES_LEN GREATER 1)
      message(FATAL_ERROR "We do not support build universal binary for macos currently!")
    elseif (CMAKE_OSX_ARCHITECTURES_LEN EQUAL 1)
      set(target_platform ${CMAKE_OSX_ARCHITECTURES})
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

  elseif (ANDROID)
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
    #Linux/FreeBSD/PowerPC/...
    #The value of CMAKE_SYSTEM_PROCESSOR should be from `uname -m`
    #Example values:
    #arm64v8/ubuntu -> aarch64
    #arm32v6/alpine -> armv7l
    #arm32v7/centos -> armv7l
    #ppc64le/debian -> ppc64le
    #s390x/ubuntu -> s390x
    #ppc64le/busybox -> ppc64le
    #arm64v8/ubuntu -> aarch64
    #Android: armv7-a aarch64 i686 x86_64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
      set(ARCH_ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
      set(ARCH_ARM TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
      set(ARCH_ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
      set(ARCH_POWER TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
      set(ARCH_RISCV64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(ARCH_X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
      set(ARCH_X64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^loongarch64.*")
      set(ARCH_LOONGARCH64 TRUE)
    endif()
  endif()

  # TODO: support APPLE multi-arch build (universal)

endif()

message(STATUS "target_platform: ${target_platform}")

