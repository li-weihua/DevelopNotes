#pragma once

//
// Define the target architecture.
//
#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#define ARCH_AMD64
#endif

#if defined(_M_IX86) || defined(__i386__)
#define ARCH_IX86
#endif

#if defined(ARCH_AMD64) || defined(ARCH_IX86)
#define ARCH_ADM64_IX86
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
#define ARCH_ARM64
#endif

#if defined(_M_ARM64EC)
#define ARCH_ARM64EC
#endif

#if defined(_M_ARM) || defined(__arm__)
#define ARCH_ARM
#endif

#if defined(ARCH_ARM64) || defined(ARCH_ARM64EC) || defined(ARCH_ARM)
#define ARCH_ARM_ANY
#endif

#if defined(__wasm_simd128__)
#define ARCH_WASM_SIMD
#else
#define ARCH_WASM_SCALAR
#endif
