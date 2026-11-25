#pragma once
// Kept apart from volta_compat.cuh (which everything includes: changing it recompiles the 19 MMQ variants).
// A Blackwell instruction (FP4, FP8, block-scaled mma, TMA...) in a kernel that a Q4 pack never launches: below compute
// capability 10.0 the kernel compiles, and stops at once if it is launched. With CUDA 12.6 these targets are compiled for
// sm_70 (CMake: basalt_arch_120a); with CUDA 12.8 and later, for sm_120a as upstream (the original asm).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#define BASALT_SM100_ASM(...) __trap()
#else
#define BASALT_SM100_ASM(...) asm volatile(__VA_ARGS__)
#endif
