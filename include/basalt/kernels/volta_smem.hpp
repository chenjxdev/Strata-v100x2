#pragma once
// Kept out of volta_compat.cuh, which everything includes (changing that header recompiles the 19 MMQ variants).

// ---------------------------------------------------------------- shared memory: a ceiling clamped to what the card offers
// Some kernels reserve 99 KB per block as a CEILING (the launch asks for less); a V100 offers 96 KB at most. The ceiling
// becomes the smaller of the two; the "fits in 99 KB" tests become "fits in what the card offers".
#include <cuda_runtime.h>
namespace basalt::volta {
inline int smem_ceiling(int want) {
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return want;
    }
    return want < optin ? want : optin;
}
}  // namespace basalt::volta
