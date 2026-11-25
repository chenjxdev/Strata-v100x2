// src/prefill/mmq/mmq_hmma_funcs.cuh - (Lapilli, V100) the NVFP4 x q8_1 prompt products on Volta's tensor
// cores, with the SAME BITS as the dp4a path (mmq_dp4a_funcs.cuh's ggml_cuda_mmq_vec_dot_nvfp4_q8_1_dp4a).
//
// The dp4a path sums, for each (row i, column j) and 32-value block, two exact int32 dot products s0 and s1 (the block's
// two NVFP4 sub-blocks: int8 kvalues times the q8_1 codes), then sum = fma(fma(s1, dx1, s0 dx0), d8, sum) in fp32 with
// flush-to-zero (its machine code, read in its SASS: FMUL.FTZ, FFMA.FTZ, FFMA.FTZ; the mmq library is built with
// -use_fast_math), the blocks in order.  Here the same tiles (the same loader, the same shared memory) feed
// mma.sync.m8n8k4 (mma884.cuh) with the int8 values as fp16 INTEGERS (exact: |kvalue| <= 12, |q| <= 127), so that the
// tensor cores' fp32 accumulation of the 16 products of a sub-block is the exact integer (|s| <= 16 x 12 x 127 < 2^24,
// every partial sum an integer): s0 and s1 are the dp4a's values, and the same three float operations follow in the same
// order.  Only the threads change: the block's 32 quad pairs own 8x8 sub-tiles of the 128 x J tile (NRB row blocks x
// NCB column blocks each), and write_back writes the same (i, j) values.  J a multiple of 16 (the others keep dp4a).
#pragma once

#include "basalt/kernels/dense_decode.cuh"
#include "basalt/kernels/mma884.cuh"

template <int J> struct mmq_hmma_shape {
    static_assert(J % 16 == 0 && J >= 16 && J <= 128, "the tile's columns: a multiple of 16");
    static constexpr int NRB = (J % 32 == 0) ? 2 : 1;   // row blocks of 8 a quad pair (A shared by its column blocks)
    static constexpr int NCB = J / 16 / NRB;            // column blocks of 8 a quad pair
    static constexpr int NE  = NRB * NCB * 8;           // accumulators a lane: J * 128 / 256
};

// the quad pair's first row block and first column block of the tile
template <int J> static __device__ __forceinline__ void mmq_hmma_place(int & rb0, int & cb0) {
    using S = mmq_hmma_shape<J>;
    const int gq = threadIdx.y * 4 + ((threadIdx.x >> 2) & 3);   // the block's quad pair, 0..31
    if constexpr (S::NRB == 2) {
        rb0 = 2 * (gq % 8);
        cb0 = (gq / 8) * S::NCB;
    } else {
        rb0 = gq % 16;
        cb0 = (gq / 16) * S::NCB;
    }
}

// an int of 4 int8 to the two half2 of an A or B fragment, exactly
static __device__ __forceinline__ void mmq_hmma_i8x4(int w, unsigned & h0, unsigned & h1) {
    __half2 a, b;
    basalt::kernels::i8x4_to_half((unsigned) w, a, b);
    h0 = basalt::kernels::h2_bits(a);
    h1 = basalt::kernels::h2_bits(b);
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    using S = mmq_hmma_shape<J>;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    static_assert(I == 128, "the tile's 16 row blocks");
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const int lane = threadIdx.x, ab = basalt::kernels::mma884_ab_index(lane);
    int rb0, cb0;
    mmq_hmma_place<J>(rb0, cb0);

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_NVFP4_Q8_1_MMQ) {   // the 32-value blocks, in order
        const int k0 = k00 + k01;
        unsigned a[S::NRB][8][2];   // the lane's row (ab) of each row block: the block's 8 ints (4 values each)
#pragma unroll
        for (int r = 0; r < S::NRB; ++r) {
            const int * xq = &x_qs[(8 * (rb0 + r) + ab) * (2 * MMQ_TILE_NE_K + 1) + k0];
#pragma unroll
            for (int s = 0; s < 8; ++s) mmq_hmma_i8x4(xq[s], a[r][s][0], a[r][s][1]);
        }
        // the scales of the lane's accumulator rows (r0 and r0 + 2 of each row block) and columns, once a block
        const int r0 = (lane & 1) + 4 * (lane >> 4), c0l = lane & 2;
        float dx[S::NRB][2][2];
#pragma unroll
        for (int r = 0; r < S::NRB; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float * p = &x_df[(8 * (rb0 + r) + r0 + 2 * h) * (2 * MMQ_TILE_NE_K / 4 + 1) + k0 / 4];
                dx[r][h][0] = p[0];
                dx[r][h][1] = p[1];
            }
#pragma unroll
        for (int cb = 0; cb < S::NCB; ++cb) {
            unsigned b[8][2];   // the lane's column (ab) of the column block
            const int * yq = &y_qs[(8 * (cb0 + cb) + ab) * MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K];
#pragma unroll
            for (int s = 0; s < 8; ++s) mmq_hmma_i8x4(yq[s], b[s][0], b[s][1]);
            float d8[4];   // columns c0l + {0, 1, 4, 5} of the block: accumulator elements (e & 4) + (e & 1)
#pragma unroll
            for (int c = 0; c < 4; ++c)
                d8[c] = y_df[(8 * (cb0 + cb) + c0l + (c & 1) + 4 * (c >> 1)) * MMQ_TILE_Y_K + (k0 / QI8_1) % (MMQ_TILE_NE_K / QI8_1)];
#pragma unroll
            for (int r = 0; r < S::NRB; ++r) {
                float c0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, c1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int s = 0; s < 4; ++s) basalt::kernels::mma884(c0, a[r][s][0], a[r][s][1], b[s][0], b[s][1]);
#pragma unroll
                for (int s = 4; s < 8; ++s) basalt::kernels::mma884(c1, a[r][s][0], a[r][s][1], b[s][0], b[s][1]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {   // element e: row r0 + (e & 2), column c0l + (e & 4) + (e & 1)
                    const float * d = dx[r][(e >> 1) & 1];
                    float t = c0[e] * d[0];   // the dp4a path's FMUL.FTZ, FFMA.FTZ, FFMA.FTZ
                    t = fmaf(c1[e], d[1], t);
                    float & acc = sum[(r * S::NCB + cb) * 8 + e];
                    acc = fmaf(t, d8[(e & 1) + ((e >> 2) << 1)], acc);
                }
            }
        }
    }
}

// the y tile's q8 codes as fp16 integers, once a tile for the whole block (column j at yh + MMQ_HMMA_YHL j: its 128 values;
// the stride keeps a quad pair's 8 columns on distinct banks); the dot below reads its B fragments there
constexpr int MMQ_HMMA_YHL = MMQ_TILE_NE_K * 4 + 4;
template <int J> static __device__ __forceinline__ void mmq_hmma_y_to_half(const int * __restrict__ y, __half * __restrict__ yh) {
    const int * y_qs = (const int *) y + 4;
    for (int idx = threadIdx.y * 32 + threadIdx.x; idx < J * MMQ_TILE_NE_K; idx += 256) {
        const int j = idx / MMQ_TILE_NE_K, k = idx % MMQ_TILE_NE_K;
        unsigned h0, h1;
        mmq_hmma_i8x4(y_qs[j * MMQ_TILE_Y_K + k], h0, h1);
        *reinterpret_cast<uint2 *>(yh + j * MMQ_HMMA_YHL + 4 * k) = make_uint2(h0, h1);
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma_yh(
        const int * __restrict__ x, const int * __restrict__ y, const __half * __restrict__ yh, float * __restrict__ sum,
        const int k00) {
    using S = mmq_hmma_shape<J>;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    static_assert(I == 128, "the tile's 16 row blocks");
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const float * y_df = (const float *) y;
    const int lane = threadIdx.x, ab = basalt::kernels::mma884_ab_index(lane);
    int rb0, cb0;
    mmq_hmma_place<J>(rb0, cb0);

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_NVFP4_Q8_1_MMQ) {   // the 32-value blocks, in order
        const int k0 = k00 + k01;
        unsigned a[S::NRB][8][2];   // the lane's row (ab) of each row block: the block's 8 ints (4 values each)
#pragma unroll
        for (int r = 0; r < S::NRB; ++r) {
            const int * xq = &x_qs[(8 * (rb0 + r) + ab) * (2 * MMQ_TILE_NE_K + 1) + k0];
#pragma unroll
            for (int s = 0; s < 8; ++s) mmq_hmma_i8x4(xq[s], a[r][s][0], a[r][s][1]);
        }
        // the scales of the lane's accumulator rows (r0 and r0 + 2 of each row block) and columns, once a block
        const int r0 = (lane & 1) + 4 * (lane >> 4), c0l = lane & 2;
        float dx[S::NRB][2][2];
#pragma unroll
        for (int r = 0; r < S::NRB; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float * p = &x_df[(8 * (rb0 + r) + r0 + 2 * h) * (2 * MMQ_TILE_NE_K / 4 + 1) + k0 / 4];
                dx[r][h][0] = p[0];
                dx[r][h][1] = p[1];
            }
#pragma unroll
        for (int cb = 0; cb < S::NCB; ++cb) {
            unsigned b[8][2];   // the lane's column (ab) of the column block, from the fp16 copy
            const __half * yc = &yh[(8 * (cb0 + cb) + ab) * MMQ_HMMA_YHL + 4 * (k0 % MMQ_TILE_NE_K)];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2 *>(yc + 4 * s);
                b[s][0] = w.x;
                b[s][1] = w.y;
            }
            float d8[4];   // columns c0l + {0, 1, 4, 5} of the block: accumulator elements (e & 4) + (e & 1)
#pragma unroll
            for (int c = 0; c < 4; ++c)
                d8[c] = y_df[(8 * (cb0 + cb) + c0l + (c & 1) + 4 * (c >> 1)) * MMQ_TILE_Y_K + (k0 / QI8_1) % (MMQ_TILE_NE_K / QI8_1)];
#pragma unroll
            for (int r = 0; r < S::NRB; ++r) {
                float c0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, c1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int s = 0; s < 4; ++s) basalt::kernels::mma884(c0, a[r][s][0], a[r][s][1], b[s][0], b[s][1]);
#pragma unroll
                for (int s = 4; s < 8; ++s) basalt::kernels::mma884(c1, a[r][s][0], a[r][s][1], b[s][0], b[s][1]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {   // element e: row r0 + (e & 2), column c0l + (e & 4) + (e & 1)
                    const float * d = dx[r][(e >> 1) & 1];
                    float t = c0[e] * d[0];   // the dp4a path's FMUL.FTZ, FFMA.FTZ, FFMA.FTZ
                    t = fmaf(c1[e], d[1], t);
                    float & acc = sum[(r * S::NCB + cb) * 8 + e];
                    acc = fmaf(t, d8[(e & 1) + ((e >> 2) << 1)], acc);
                }
            }
        }
    }
}

// The shared memory the fp16 y tile adds after the x tile (mmq.cuh's mmq_get_nbytes_shared, variant 1).
static constexpr __host__ __device__ int mmq_hmma_nbytes_yh(const int J) { return J * MMQ_HMMA_YHL * 2; }

// mul_mat_q_process_tile's vec_dot on the tensor cores (mmq.cuh, variant 1): the y tile to fp16 integers after the x tile,
// a barrier, the dot (process_tile's own barrier follows); the fp16 tile is found from the kernel's shared memory as
// process_tile lays it out, not from x (x is a const restrict pointer).
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma_tile(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = ggml_cuda_mmq_get_nthreads(type, J, fallback);
    static_assert(nthreads == 256 && warp_size == 32, "mmq_hmma_y_to_half: 8 warps");
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, ggml_cuda_mmq_get_I(type, J, fallback));
    extern __shared__ int data_mul_mat_q[];
    int * tile_x = data_mul_mat_q + J + GGML_PAD(J * MMQ_TILE_Y_K, nthreads);
    __half * yh = reinterpret_cast<__half *>(tile_x + txs.qs + txs.dm + txs.sc);
    mmq_hmma_y_to_half<J>(y, yh);
    __syncthreads();
    ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma_yh<type, J, fallback>(x, y, yh, sum, k00);
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_write_back_hmma(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    using S = mmq_hmma_shape<J>;
    const int lane = threadIdx.x;
    int rb0, cb0;
    mmq_hmma_place<J>(rb0, cb0);
#pragma unroll
    for (int r = 0; r < S::NRB; ++r)
#pragma unroll
        for (int cb = 0; cb < S::NCB; ++cb)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = 8 * (rb0 + r) + basalt::kernels::mma884_c_row(lane, e);
                const int j = 8 * (cb0 + cb) + basalt::kernels::mma884_c_col(lane, e);
                if (j > j_max || (fallback && i > i_max)) continue;
                dst[ids_dst[j] * stride + i] = sum[(r * S::NCB + cb) * 8 + e];
            }
}
