#pragma once

// X86
#if defined(_M_IX86) || defined(__i386__)
#define ARCH_IX86
#endif

// AMD64
#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#define ARCH_AMD64
#endif

// X86 or X64: x86 general
#if defined(ARCH_IX86) || defined(ARCH_AMD64)
#define ARCH_IX86_AMD64
#endif

// arm 32
#if defined(_M_ARM) || defined(__arm__)
#define ARCH_ARM32
#endif

// arm 64
#if defined(_M_ARM64) || defined(__aarch64__)
#define ARCH_ARM64
#endif

// arm 64 ec
#if defined(_M_ARM64EC)
#define ARCH_ARM64EC
#endif

// ARM general
#if defined(ARCH_ARM32) || defined(ARCH_ARM64) || defined(ARCH_ARM64EC)
#define ARCH_ARM
#endif

// Use scalar version when no simd or not implemented
#if !defined(ARCH_IX86_AMD64) && !defined(ARCH_ARM)
#define ARCH_SCALAR
#endif

// wasm
#if defined(__wasm_simd128__)
#define ARCH_WASM_SIMD
#else
#define ARCH_WASM_SCALAR
#endif

// loong arch
#if defined(__loongarch64)
#define ARCH_LARCH64
#endif
