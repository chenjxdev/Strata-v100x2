// tests/kernels/dense_mmvq_v0.cu - dense_mmvq as of commit dd695013, namespace basalt::kernels::v0: the
// bench's reference for the kernel's later versions (the same arithmetic; only the speed is compared).
//
// The two dot products are iq_kernels.cu's Split<40> and Split<8> (llama.cpp's vec_dot_nvfp4_q8_1 and vec_dot_q8_0_q8_1,
// ggml/src/ggml-cuda/vecdotq.cuh at llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d; MIT, Copyright (c)
// 2023-2026 The ggml authors, see THIRD_PARTY_LICENSES), split into the weight part, the activation part and the apply,
// the activation part read from this file's layout.  The quantizer is iq_kernels.cu's q8_1_store without the sum.
#include "basalt/kernels/dense_mmvq.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels::v0 {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kQ8_0 = 8, kNvfp4 = 40;

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh), as iq_kernels.cu has them
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
// NVFP4's UE4M3 sub-block scale as ggml decodes it (iq_kernels.cu's nvfp4_scale_dev)
__device__ __forceinline__ float nvfp4_scale_dev(uint32_t x) {
    const uint32_t v = x & 0x7F;
    if (v == 0 || v == 0x7F) return 0.0f;
    const uint32_t e = v >> 3, m = v & 7;
    if (e == 0) return (float) m * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119) << 23) | (m << 20));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The layout (dense_mmvq.hpp): chunk j of kDenseChunk values, row t of the window at (j T + t) kChunkBytes: its 512
// int8 then its 16 fp16 scales.
constexpr int kChunkBytes = kDenseChunk + kDenseChunk / 32 * 2;   // 544

// The calls of a weight row as iq_kernels.cu's Fmt<TY> numbers them (call k: block kbx = k / ipb, iqs = step (k % ipb));
// W what the weight gives a call, A what the activation gives it, apply the call's float.  at(): the iteration's
// activation bytes and scale of column 0 (column c at + c kChunkBytes); iteration i is the calls k = lane + 32 i.
template<int TY> struct Dense;
template<> struct Dense<kNvfp4> {   // Fmt<40>: one call per 16-value sub-block; an iteration = one chunk
    static constexpr int qk = 64, ipb = 4, step = 1, iters_per_chunk = 1;
    struct W { int2 v[2]; float d4; };
    struct A { int4 q; float d8; };
    __device__ static W load(const uint8_t* __restrict__ row, int kbx, int iqs) {
        const block_nvfp4* b = (const block_nvfp4*) row + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = get_int_from_table_16(get_int_b4(b->qs, 2 * iqs + i), kvalues_fp4);
        r.d4 = nvfp4_scale_dev(b->d[iqs]);
        return r;
    }
    // call k = lane + 32 i: values [16 k, +16) = chunk i, bytes 16 lane; its q8_1 block's scale is the chunk's lane / 2
    template<int NC>
    __device__ static void at(const uint8_t* __restrict__ x, int i, int lane, const uint8_t*& q, const uint8_t*& s) {
        const uint8_t* c = x + (size_t) i * (NC * kChunkBytes);
        q = c + 16 * lane;
        s = c + kDenseChunk + 2 * (lane >> 1);
    }
    __device__ static A act(const uint8_t* __restrict__ q, const uint8_t* __restrict__ s) {
        A a;
        a.q = *(const int4*) q;
        a.d8 = __half2float(*(const __half*) s);
        return a;
    }
    // Split<40>'s words a0 + i and a0 + i + 2: the low nibbles of weight word i meet activation word i, the high ones i + 2
    __device__ static float apply(const W& r, const A& a) {
        int sumi = 0;
        sumi = DP4A(r.v[0].x, a.q.x, sumi);
        sumi = DP4A(r.v[0].y, a.q.z, sumi);
        sumi = DP4A(r.v[1].x, a.q.y, sumi);
        sumi = DP4A(r.v[1].y, a.q.w, sumi);
        return a.d8 * r.d4 * (float) sumi;
    }
};
template<> struct Dense<kQ8_0> {   // Fmt<8>: VDR 2, one call per 8 values; two iterations = one chunk
    static constexpr int qk = 32, ipb = 4, step = 2, iters_per_chunk = 2;
    struct W { int q[2]; float d; };
    struct A { int2 q; float d8; };
    __device__ static W load(const uint8_t* __restrict__ row, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) row + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.q[i] = get_int_b2(b->qs, iqs + i);
        r.d = __half2float(b->d);
        return r;
    }
    // call k = lane + 32 i: values [8 k, +8) = chunk i / 2, bytes 256 (i & 1) + 8 lane; scale (256 (i & 1) + 8 lane) / 32
    template<int NC>
    __device__ static void at(const uint8_t* __restrict__ x, int i, int lane, const uint8_t*& q, const uint8_t*& s) {
        const uint8_t* c = x + (size_t) (i >> 1) * (NC * kChunkBytes);
        const int h = i & 1;
        q = c + 256 * h + 8 * lane;
        s = c + kDenseChunk + 16 * h + 2 * (lane >> 2);
    }
    __device__ static A act(const uint8_t* __restrict__ q, const uint8_t* __restrict__ s) {
        A a;
        a.q = *(const int2*) q;
        a.d8 = __half2float(*(const __half*) s);
        return a;
    }
    // Split<8>'s words iqs, iqs + 1 of q8_1 block kbx
    __device__ static float apply(const W& r, const A& a) {
        int sumi = 0;
        sumi = DP4A(r.q[0], a.q.x, sumi);
        sumi = DP4A(r.q[1], a.q.y, sumi);
        return r.d * a.d8 * (float) sumi;
    }
};

// R weight rows a warp, NC columns.  Per (row, column): mmvq_multi_kernel's calls k = lane + 32 i in order, the same
// float per call added in the same order, then the same warp_sum; the weight part of a call is decoded once for the
// NC columns and the activation part once for the R rows, G columns' activations loaded before their products.  The
// next call's weight words are loaded before this call's products (the first ones before pdl_wait, weights being
// constant).  After warp_sum every lane holds every sum: lane (r NC + c) % 32 stores (row r, column c).
template<int TY, int NC, int R>
__global__ void __launch_bounds__(128) dense_mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                         const uint8_t* __restrict__ x, float* __restrict__ y, int n_in,
                                                         int n_out) {
    using D = Dense<TY>;
    constexpr int G = NC < (R >= 4 ? 4 : 8) ? NC : (R >= 4 ? 4 : 8);   // columns a group
    const int row0 = (int) (blockIdx.x * 4 + threadIdx.y) * R;
    if (row0 >= n_out) return;
    const int lane = threadIdx.x;
    const int iters = n_in / kDenseChunk * D::iters_per_chunk;   // every lane makes the same calls
    const uint8_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = w + (size_t) min(row0 + r, n_out - 1) * row_bytes;
    float s[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = 0.0f;
    typename D::W cur[R];
#pragma unroll
    for (int r = 0; r < R; ++r) cur[r] = D::load(wr[r], lane / D::ipb, D::step * (lane % D::ipb));
    pdl_wait();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        const int kn = lane + 32 * (i + 1 < iters ? i + 1 : i);   // the last iteration loads its own words again (an L1 hit)
        typename D::W nxt[R];
#pragma unroll
        for (int r = 0; r < R; ++r) nxt[r] = D::load(wr[r], kn / D::ipb, D::step * (kn % D::ipb));
        const uint8_t *q, *sc;
        D::template at<NC>(x, i, lane, q, sc);
#pragma unroll
        for (int c0 = 0; c0 < NC; c0 += G) {
            typename D::A a[G];
#pragma unroll
            for (int g = 0; g < G; ++g)
                if (c0 + g < NC) a[g] = D::act(q + (c0 + g) * kChunkBytes, sc + (c0 + g) * kChunkBytes);
#pragma unroll
            for (int g = 0; g < G; ++g)
                if (c0 + g < NC)
#pragma unroll
                    for (int r = 0; r < R; ++r) s[r][c0 + g] += D::apply(cur[r], a[g]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = nxt[r];
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = warp_sum(s[r][c]);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == ((r * NC + c) & 31) && row0 + r < n_out) y[(size_t) c * n_out + row0 + r] = s[r][c];
}

// value i of the rows (the warp holds 32 values of one q8_1 block: n_cols % 32 == 0): q8_1_store's d and int8, at their
// place in the chunked layout
__global__ void dense_quantize_q8_kernel(const float* __restrict__ x, uint8_t* __restrict__ y, long long n, int n_cols,
                                         int n_rows) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long r = i / n_cols, p = i - r * n_cols;
    uint8_t* c = y + ((size_t) (p / kDenseChunk) * n_rows + (size_t) r) * kChunkBytes;
    const int o = (int) (p % kDenseChunk);
    ((int8_t*) c)[o] = q;
    if ((o & 31) == 0) ((__half*) (c + kDenseChunk))[o >> 5] = __float2half(q8_1_finite(d));
}

int g_rows_per_warp = 0;
// Measured on a V100 (dense_mmvq_bench, the 27B's shapes: per width, us per call of a layer's matrices of the
// type, gate + up + down for NVFP4, GDN qkv + out for Q8_0, each width's best of 1, 2 or 4 rows a warp): the rows a warp
// for each width; negative where MMQ stays faster (NVFP4 13 and 15 by 2% and 13%, Q8_0 13 by 8%; the rows if forced).
// The best widths do not follow a rule (ptxas schedules each instance its own way: the same width is up to 2.3x apart from
// one row count to the next), hence a table.  Against iq_mmvq: 0.92 (1 row) to 0.34 (16) for NVFP4, 0.96 to 0.22 for Q8_0;
// against MMQ: 0.43 to 0.99 and 0.59 to 0.85.
constexpr signed char kRows[2][kDenseMaxRows + 1] = {
    {0, 4, 1, 1, 4, 4, 1, 1, 4, 2, 4, 4, 2, -2, 4, -1, 4},   // NVFP4
    {0, 2, 4, 1, 4, 2, 2, 1, 2, 2, 4, 2, 2, -2, 4, 4, 4}};   // Q8_0
int table_rows(int t, int ncols) {
    const int v = ncols >= 1 && ncols <= kDenseMaxRows ? kRows[t == kNvfp4 ? 0 : 1][ncols] : 1;
    return v < 0 ? -v : v;
}
// weight rows a warp for a call of `ncols` columns (the same bits either way)
int rows_per_warp(int t, int ncols) {
    if (g_rows_per_warp == 1 || g_rows_per_warp == 2 || g_rows_per_warp == 4) return g_rows_per_warp;
    return table_rows(t, ncols);
}

template<int TY, int NC, int R>
void go(const uint8_t* w, size_t rb, const uint8_t* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const unsigned rows = 4 * R;
    launch_pdl(dense_mmvq_kernel<TY, NC, R>, dim3((unsigned) ((n_out + rows - 1) / rows)), dim3(32, 4), 0, s, w, rb, x, y,
               n_in, n_out);
}
template<int TY, int R>
void by_width(int nc, const uint8_t* w, size_t rb, const uint8_t* x, float* y, int n_in, int n_out, cudaStream_t s) {
    switch (nc) {
#define DENSE_NC(N) case N: go<TY, N, R>(w, rb, x, y, n_in, n_out, s); break;
        DENSE_NC(1) DENSE_NC(2) DENSE_NC(3) DENSE_NC(4) DENSE_NC(5) DENSE_NC(6) DENSE_NC(7) DENSE_NC(8)
        DENSE_NC(9) DENSE_NC(10) DENSE_NC(11) DENSE_NC(12) DENSE_NC(13) DENSE_NC(14) DENSE_NC(15) DENSE_NC(16)
#undef DENSE_NC
        default: break;
    }
}
template<int TY>
void by_rows(int nc, int r, const uint8_t* w, size_t rb, const uint8_t* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (r == 4) by_width<TY, 4>(nc, w, rb, x, y, n_in, n_out, s);
    else if (r == 2) by_width<TY, 2>(nc, w, rb, x, y, n_in, n_out, s);
    else by_width<TY, 1>(nc, w, rb, x, y, n_in, n_out, s);
}

}  // namespace

bool dense_mmvq_supported(int t) noexcept { return t == kQ8_0 || t == kNvfp4; }
bool dense_mmvq_faster(int t, int ncols) noexcept {
    return dense_mmvq_supported(t) && ncols >= 1 && ncols <= kDenseMaxRows && kRows[t == kNvfp4 ? 0 : 1][ncols] > 0;
}
size_t dense_q8_bytes(int64_t n_cols, int64_t n_rows) noexcept {
    return (size_t) n_rows * (size_t) ((n_cols + kDenseChunk - 1) / kDenseChunk) * kChunkBytes;
}
void dense_mmvq_set_rows_per_warp(int r) { g_rows_per_warp = r; }

void dense_quantize_q8(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    if (n_cols % kDenseChunk != 0) {
        basalt::log::error("basalt", "dense_quantize_q8: %lld columns", (long long) n_cols);
        std::exit(1);
    }
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    dense_quantize_q8_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (uint8_t*) y, n, (int) n_cols,
                                                                                            (int) n_rows);
    check("dense_quantize_q8");
}

void dense_mmvq(int t, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!dense_mmvq_supported(t) || n_in % kDenseChunk != 0 || n_out <= 0 || ncols < 1 || ncols > kDenseMaxRows) {
        basalt::log::error("basalt", "dense_mmvq: type %d, %d x %d, %d columns is not supported", t, n_out, n_in, ncols);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const size_t rb = t == kNvfp4 ? (size_t) (n_in / 64) * sizeof(block_nvfp4) : (size_t) (n_in / 32) * sizeof(block_q8_0);
    const int r = rows_per_warp(t, ncols);
    if (t == kNvfp4) by_rows<kNvfp4>(ncols, r, (const uint8_t*) w, rb, (const uint8_t*) x, y, n_in, n_out, s);
    else by_rows<kQ8_0>(ncols, r, (const uint8_t*) w, rb, (const uint8_t*) x, y, n_in, n_out, s);
    check("dense_mmvq");
}

}  // namespace basalt::kernels::v0
