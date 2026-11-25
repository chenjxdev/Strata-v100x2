// SPDX-License-Identifier: MIT
// The MMQ dp4a path (Q8_0, Q5_1, Q4_K, Q5_K): its constants and tile sizes. Extracted from ggml-cuda (mmq.cuh,
// mmq-load-tiles.cuh, mmq-vec-dot.cuh, vecdotq.cuh) of llama.cpp at commit 3cf03257f219afbe7334045ff7c6a06ac68c627d
// (https://github.com/ggml-org/llama.cpp), branches resolved for sm_70 (V100). Copyright (c) 2023-2026 The ggml authors.
// MIT license: see THIRD_PARTY_LICENSES.
// Mechanically extracted; keep in sync with upstream when editing.
#pragma once

#define VDR_Q8_0_Q8_1_MMQ 8
#define VDR_Q5_1_Q8_1_MMQ  4
#define VDR_Q4_K_Q8_1_MMQ  8
#define VDR_Q5_K_Q8_1_MMQ  8

#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_0 + I/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_1 + I/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_K,                 I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI5_K   + I/QI5_K,     I*MMQ_TILE_NE_K/8 + I/8}
// (Lapilli, V100) NVFP4 on the dp4a path (mmq_dp4a_funcs.cuh): Q8_0's int8 tile, one float per 16-value sub-block (16 a
// tile row, plus one of padding). -DBASALT_MMQ_DP4A_NVFP4=0 keeps NVFP4 on the emulated mma path.
#ifndef BASALT_MMQ_DP4A_NVFP4
#define BASALT_MMQ_DP4A_NVFP4 1
#endif
#define VDR_NVFP4_Q8_1_MMQ 8
#define MMQ_DP4A_TXS_NVFP4   tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*(MMQ_TILE_NE_K*2/4 + 1),          0}

// the formats that take the dp4a path (BASALT_MMQ_DP4A, defined in mmq.cuh); the others keep the mma path
static constexpr __host__ __device__ bool ggml_cuda_mmq_dp4a_type(ggml_type type) {
    return BASALT_MMQ_DP4A && (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
                               (BASALT_MMQ_DP4A_NVFP4 && type == GGML_TYPE_NVFP4));
}

// the dp4a path's x tile sizes (those of ggml's mmq.cuh, for these four formats only)
static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int I) {
    switch (type) {
        case GGML_TYPE_Q5_1: return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q8_0: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q4_K: return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K: return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_NVFP4: return MMQ_DP4A_TXS_NVFP4;
        default:             return tile_x_sizes{0, 0, 0};
    }
}
