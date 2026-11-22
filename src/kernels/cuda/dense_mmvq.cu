// src/kernels/cuda/dense_mmvq.cu - see include/basalt/kernels/dense_mmvq.hpp.
//
// The two dot products are iq_kernels.cu's Split<40> and Split<8> (llama.cpp's vec_dot_nvfp4_q8_1 and vec_dot_q8_0_q8_1,
// ggml/src/ggml-cuda/vecdotq.cuh at llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d; MIT, Copyright (c)
// 2023-2026 The ggml authors, see THIRD_PARTY_LICENSES), split into the weight part, the activation part and the apply,
// the activation part read from this file's layout.  The quantizer is iq_kernels.cu's q8_1_store without the sum.
#include "basalt/kernels/dense_mmvq.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kQ8_0 = 8, kNvfp4 = 40;

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh), as iq_kernels.cu has them
// get_int_b2 / get_int_b4 (vecdotq.cuh) on the weights, through the read-only path (ld.global.nc), whatever the pointer's
// provenance: read from a launch's table of matrices, ptxas no longer proved them constant and used plain loads (measured:
// NVFP4 15-46% slower).  The same bytes.
__device__ __forceinline__ int ldg_int_b4(const void* x, int i32) { return __ldg((const int*) x + i32); }
__device__ __forceinline__ int ldg_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*) x;
    int x32 = (int) __ldg(x16 + 2 * i32 + 0) << 0;
    x32 |= (int) __ldg(x16 + 2 * i32 + 1) << 16;
    return x32;
}
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
// NVFP4's UE4M3 sub-block scale as ggml decodes it (iq_kernels.cu's nvfp4_scale_dev: the same float for each of the 256
// bytes), without its branches: both values computed, then selected.  Its two branches were a reconvergence region
// (BSSY/BSYNC) per weight row and call in the loop, which ptxas does not schedule across.
__device__ __forceinline__ float nvfp4_scale_dev(uint32_t x) {
    const uint32_t v = x & 0x7F;
    const uint32_t e = v >> 3, m = v & 7;
    const float sub = (float) m * (1.0f / 1024.0f);
    const float nor = __uint_as_float(((e + 119) << 23) | (m << 20));
    const float r = e == 0 ? sub : nor;
    return v == 0 || v == 0x7F ? 0.0f : r;
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
        for (int i = 0; i < 2; ++i) r.v[i] = get_int_from_table_16(ldg_int_b4(b->qs, 2 * iqs + i), kvalues_fp4);
        r.d4 = nvfp4_scale_dev(__ldg(&b->d[iqs]));
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
        for (int i = 0; i < 2; ++i) r.q[i] = ldg_int_b2(b->qs, iqs + i);
        r.d = __half2float(__ushort_as_half(__ldg((const unsigned short*) &b->d)));
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

// The matrices of one launch (one format, the same x and n_in): blocks [first[i], first[i + 1]) are matrix i's, each with
// its rows, output and tensor scale.  EPI_STORE: y[c][r] = scale * sum (one FMUL, scale_values' after the store; no scale:
// the sum); EPI_ADD_BF16 (one matrix): the BF16 residual [c][r] += scale * sum, add_scaled's FMUL then FADD (its SASS:
// not contracted) and the same rounding.
constexpr int EPI_STORE = 0, EPI_ADD_BF16 = 1;
struct Segs {
    const uint8_t* w[kDenseMaxSegs];
    float* y[kDenseMaxSegs];
    const float* scale[kDenseMaxSegs];
    int rows[kDenseMaxSegs];
    int first[kDenseMaxSegs + 1];
    int n;
    __nv_bfloat16* resid;
};

// the warp's sums, every lane holding every one after warp_sum: lane (r NC + c) % 32 stores (row r, column c) of the rows
// below n_out, through the launch's epilogue
template<int R, int NC, int EPI>
__device__ __forceinline__ void store_sums(const Segs& sg, int seg, int n_out, int row0, int lane, const float (&s)[R][NC]) {
    const float* sc_ = sg.scale[seg];
    float* y = sg.y[seg];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == ((r * NC + c) & 31) && row0 + r < n_out) {
                const size_t o = (size_t) c * n_out + row0 + r;
                const float v = sc_ ? __fmul_rn(s[r][c], *sc_) : s[r][c];
                if constexpr (EPI == EPI_STORE) y[o] = v;
                else sg.resid[o] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(sg.resid[o]), v));
            }
}

// R weight rows a warp, NC columns.  Per (row, column): mmvq_multi_kernel's calls k = lane + 32 i in order, the same
// float per call added in the same order, then the same warp_sum; the weight part of a call is decoded once for the
// NC columns and the activation part once for the R rows, G columns' activations loaded before their products.  The
// next call's weight words are loaded before this call's products (the first ones before pdl_wait, weights being
// constant).  After warp_sum every lane holds every sum: lane (r NC + c) % 32 stores (row r, column c).
template<int TY, int NC, int R, int EPI>
__global__ void __launch_bounds__(128) dense_mmvq_kernel(const Segs sg, size_t row_bytes, const uint8_t* __restrict__ x,
                                                         int n_in) {
    using D = Dense<TY>;
    constexpr int G = NC < (R >= 4 ? 4 : 8) ? NC : (R >= 4 ? 4 : 8);   // columns a group
    int seg = 0;
#pragma unroll
    for (int i = 1; i < kDenseMaxSegs; ++i)
        if (i < sg.n && (int) blockIdx.x >= sg.first[i]) seg = i;
    const int n_out = sg.rows[seg];
    const int row0 = (int) ((blockIdx.x - sg.first[seg]) * 4 + threadIdx.y) * R;
    if (row0 >= n_out) return;
    const uint8_t* __restrict__ w = sg.w[seg];
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
    store_sums<R, NC, EPI>(sg, seg, n_out, row0, lane, s);
}

// The same products with the window's activations staged in shared memory one chunk ahead: a chunk of 512
// values of the NC rows is NC kChunkBytes contiguous in x (the layout), read by the block's 128 threads into registers
// while the chunk before is multiplied, then stored into the other buffer, one __syncthreads a chunk; each warp reads its
// columns there at immediate offsets, of a short fixed latency, the block's 4 warps sharing one read of x.  (The direct
// kernel's speed shifted up to 2x with ptxas's schedule of its activation loads from L1 / L2.)  The same values, calls and
// sums: the same bits.  No warp returns early (every warp stages and syncs): rows past n_out compute on the last row and
// store nothing.
template<int TY, int NC, int R, int EPI>
__global__ void __launch_bounds__(128) dense_mmvq_staged_kernel(const Segs sg, size_t row_bytes,
                                                                const uint8_t* __restrict__ x, int n_in) {
    using D = Dense<TY>;
    constexpr int G = NC < (R >= 4 ? 4 : 8) ? NC : (R >= 4 ? 4 : 8);   // columns a group
    constexpr int CB = NC * kChunkBytes, V = CB / 16, PER = (V + 127) / 128;
    __shared__ alignas(16) uint8_t sm[2][CB];
    int seg = 0;
#pragma unroll
    for (int i = 1; i < kDenseMaxSegs; ++i)
        if (i < sg.n && (int) blockIdx.x >= sg.first[i]) seg = i;
    const int n_out = sg.rows[seg];
    const int row0 = (int) ((blockIdx.x - sg.first[seg]) * 4 + threadIdx.y) * R;
    const uint8_t* __restrict__ w = sg.w[seg];
    const int lane = threadIdx.x, tid = threadIdx.y * 32 + threadIdx.x;
    const int chunks = n_in / kDenseChunk, iters = chunks * D::iters_per_chunk;
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
    int4 st[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p)
        if (tid + 128 * p < V) st[p] = __ldg((const int4*) x + tid + 128 * p);
#pragma unroll
    for (int p = 0; p < PER; ++p)
        if (tid + 128 * p < V) ((int4*) sm[0])[tid + 128 * p] = st[p];
    __syncthreads();
#pragma unroll 1
    for (int j = 0; j < chunks; ++j) {
        const int b = j & 1;
        if (j + 1 < chunks) {
            const int4* nx = (const int4*) (x + (size_t) (j + 1) * CB);
#pragma unroll
            for (int p = 0; p < PER; ++p)
                if (tid + 128 * p < V) st[p] = __ldg(nx + tid + 128 * p);
        }
#pragma unroll
        for (int h = 0; h < D::iters_per_chunk; ++h) {
            const int i = j * D::iters_per_chunk + h;
            const int kn = lane + 32 * (i + 1 < iters ? i + 1 : i);
            typename D::W nxt[R];
#pragma unroll
            for (int r = 0; r < R; ++r) nxt[r] = D::load(wr[r], kn / D::ipb, D::step * (kn % D::ipb));
            const uint8_t *q, *sc;
            D::template at<NC>(sm[b], h, lane, q, sc);   // the chunk's own bytes: as chunk 0 of the layout
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
        if (j + 1 < chunks) {
#pragma unroll
            for (int p = 0; p < PER; ++p)
                if (tid + 128 * p < V) ((int4*) sm[b ^ 1])[tid + 128 * p] = st[p];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = warp_sum(s[r][c]);
    store_sums<R, NC, EPI>(sg, seg, n_out, row0, lane, s);
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

// q8_1_store's value for v (the warp holds the 32 values of its q8_1 block), written at column p of row r of the layout
__device__ __forceinline__ void dense_store_q8(float v, uint8_t* __restrict__ y, int n_rows, long long r, long long p) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(v, d, amax);
    uint8_t* c = y + ((size_t) (p / kDenseChunk) * n_rows + (size_t) r) * kChunkBytes;
    const int o = (int) (p % kDenseChunk);
    ((int8_t*) c)[o] = q;
    if ((o & 31) == 0) ((__half*) (c + kDenseChunk))[o >> 5] = __float2half(q8_1_finite(d));
}

// dense_ops.cu's block_sum (blockDim.x a multiple of 32): the same sums in the same order
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

// dense_ops.cu's rms_norm_rows_bf16_kernel (512 threads a row, the same loops and sums: the same xn), each value then
// quantized as dense_quantize_q8_kernel does from xn (n_cols a multiple of 512: every warp holds whole q8_1 blocks)
__global__ void __launch_bounds__(512) dense_norm_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const float* __restrict__ w, float* __restrict__ out,
                                                                  int cols, float eps, uint8_t* __restrict__ y, int n_rows) {
    __shared__ float red[32];
    const __nv_bfloat16* xr = x + (int64_t) blockIdx.x * cols;
    float* yr = out + (int64_t) blockIdx.x * cols;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]);
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]) * inv * w[c];
        yr[c] = v;
        dense_store_q8(v, y, n_rows, blockIdx.x, c);
    }
}

// dense_ops.cu's swiglu_scaled_kernel (the same expression: h = silu(sg g) (su u)), each h quantized as
// dense_quantize_q8_kernel does from h (width a multiple of 32); h itself is not written
__global__ void dense_swiglu_quantize_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                             const float* __restrict__ sg, const float* __restrict__ su, long long n,
                                             int width, int n_rows, uint8_t* __restrict__ y) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long r = i / width, c = i % width;
    const float gv = g[r * width + c] * (sg ? *sg : 1.0f), uv = u[r * width + c] * (su ? *su : 1.0f);
    const float h = gv / (1.0f + expf(-gv)) * uv;
    dense_store_q8(h, y, n_rows, r, c);
}

int g_rows_per_warp = 0;
// Measured on a V100 (dense_mmvq_bench, the 27B's shapes: per width, us per call of a layer's
// matrices of the type, gate + up + down for NVFP4, qkv + out for Q8_0, each width's best of the six variants, a staged
// one where it is within 2% of the best: the staged kernel is the steadier from one build to the next): the variant for
// each width, 1/2/4 rows a warp on the direct kernel, 11/12/14 on the staged one.  Every width beats MMQ now (NVFP4: 0.37
// of it at 1 row to 0.86 at 16; Q8_0: 0.58 to 0.88); against the direct kernel of the morning (v0), the sum over the 16
// widths is 11.7% lower (NVFP4 12 to 15 rows: -14 to -22%).  A negative entry would send that width to MMQ.
// (Then dense_forward_bench, a whole forward's products as forward.cpp launches them, picked NVFP4 5 rows and the Q8_0 head
// at 9, 12 and 15 rows: 5-7% each; the other widths agree within 1%.)
constexpr signed char kRows[2][kDenseMaxRows + 1] = {
    {0, 11, 12, 12, 12, 14, 12, 14, 14, 14, 14, 14, 12, 14, 14, 14, 14},   // NVFP4
    {0, 11, 12, 12, 12, 12, 12, 14, 14, 4, 14, 14, 14, 4, 4, 14, 4}};       // Q8_0
int table_rows(int t, int ncols) {
    const int v = ncols >= 1 && ncols <= kDenseMaxRows ? kRows[t == kNvfp4 ? 0 : 1][ncols] : 1;
    return v < 0 ? -v : v;
}
// the variant for a call of `ncols` columns (the same bits whichever): 1, 2, 4 weight rows a warp on the direct kernel,
// 11, 12, 14 on the staged one
int rows_per_warp(int t, int ncols) {
    const int v = g_rows_per_warp;
    if (v == 1 || v == 2 || v == 4 || v == 11 || v == 12 || v == 14) return v;
    return table_rows(t, ncols);
}

template<int TY, int NC, int R, int EPI, bool STAGED>
void go(Segs sg, size_t rb, const uint8_t* x, int n_in, cudaStream_t s) {
    for (int i = 0; i < sg.n; ++i) sg.first[i + 1] = sg.first[i] + (sg.rows[i] + 4 * R - 1) / (4 * R);
    if constexpr (STAGED)
        launch_pdl(dense_mmvq_staged_kernel<TY, NC, R, EPI>, dim3((unsigned) sg.first[sg.n]), dim3(32, 4), 0, s, sg, rb, x, n_in);
    else
        launch_pdl(dense_mmvq_kernel<TY, NC, R, EPI>, dim3((unsigned) sg.first[sg.n]), dim3(32, 4), 0, s, sg, rb, x, n_in);
}
template<int TY, int R, int EPI, bool STAGED>
void by_width(int nc, const Segs& sg, size_t rb, const uint8_t* x, int n_in, cudaStream_t s) {
    switch (nc) {
#define DENSE_NC(N) case N: go<TY, N, R, EPI, STAGED>(sg, rb, x, n_in, s); break;
        DENSE_NC(1) DENSE_NC(2) DENSE_NC(3) DENSE_NC(4) DENSE_NC(5) DENSE_NC(6) DENSE_NC(7) DENSE_NC(8)
        DENSE_NC(9) DENSE_NC(10) DENSE_NC(11) DENSE_NC(12) DENSE_NC(13) DENSE_NC(14) DENSE_NC(15) DENSE_NC(16)
#undef DENSE_NC
        default: break;
    }
}
template<int TY, int EPI>
void by_rows(int nc, int r, const Segs& sg, size_t rb, const uint8_t* x, int n_in, cudaStream_t s) {
    switch (r) {
        case 14: by_width<TY, 4, EPI, true>(nc, sg, rb, x, n_in, s); break;
        case 12: by_width<TY, 2, EPI, true>(nc, sg, rb, x, n_in, s); break;
        case 11: by_width<TY, 1, EPI, true>(nc, sg, rb, x, n_in, s); break;
        case 4: by_width<TY, 4, EPI, false>(nc, sg, rb, x, n_in, s); break;
        case 2: by_width<TY, 2, EPI, false>(nc, sg, rb, x, n_in, s); break;
        default: by_width<TY, 1, EPI, false>(nc, sg, rb, x, n_in, s); break;
    }
}
template<int EPI>
void launch_segs(int t, const Segs& sg, const void* x, int n_in, int ncols, cudaStream_t s) {
    const size_t rb = t == kNvfp4 ? (size_t) (n_in / 64) * sizeof(block_nvfp4) : (size_t) (n_in / 32) * sizeof(block_q8_0);
    const int r = rows_per_warp(t, ncols);
    if (t == kNvfp4) by_rows<kNvfp4, EPI>(ncols, r, sg, rb, (const uint8_t*) x, n_in, s);
    else by_rows<kQ8_0, EPI>(ncols, r, sg, rb, (const uint8_t*) x, n_in, s);
}
bool dense_args_ok(const char* what, int t, int n_in, int ncols) {
    if (dense_mmvq_supported(t) && n_in % kDenseChunk == 0 && ncols >= 1 && ncols <= kDenseMaxRows) return true;
    basalt::log::error("basalt", "%s: type %d, %d inputs, %d columns is not supported", what, t, n_in, ncols);
    std::exit(1);
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
    const DenseSeg one{w, n_out, nullptr, y};
    dense_mmvq_segs(t, &one, 1, x, n_in, ncols, stream);
}

void dense_mmvq_segs(int t, const DenseSeg* segs, int n_segs, const void* x, int n_in, int ncols, void* stream) {
    dense_args_ok("dense_mmvq", t, n_in, ncols);
    if (n_segs < 1 || n_segs > kDenseMaxSegs) {
        basalt::log::error("basalt", "dense_mmvq: %d matrices in one launch", n_segs);
        std::exit(1);
    }
    Segs sg{};
    sg.n = n_segs;
    for (int i = 0; i < n_segs; ++i) {
        sg.w[i] = (const uint8_t*) segs[i].w;
        sg.y[i] = segs[i].y;
        sg.scale[i] = segs[i].scale;
        sg.rows[i] = segs[i].rows;
    }
    launch_segs<EPI_STORE>(t, sg, x, n_in, ncols, (cudaStream_t) stream);
    check("dense_mmvq");
}

void dense_mmvq_add_bf16(int t, const void* w, const float* scale, const void* x, uint16_t* resid, int n_in, int n_out,
                         int ncols, void* stream) {
    dense_args_ok("dense_mmvq_add_bf16", t, n_in, ncols);
    Segs sg{};
    sg.n = 1;
    sg.w[0] = (const uint8_t*) w;
    sg.scale[0] = scale;
    sg.rows[0] = n_out;
    sg.resid = reinterpret_cast<__nv_bfloat16*>(resid);
    launch_segs<EPI_ADD_BF16>(t, sg, x, n_in, ncols, (cudaStream_t) stream);
    check("dense_mmvq_add_bf16");
}

void dense_norm_quantize(const uint16_t* x, const float* w, float* xn, float eps, int64_t n_rows, int64_t n_cols, void* y,
                         void* stream) {
    if (n_cols % kDenseChunk != 0) {
        basalt::log::error("basalt", "dense_norm_quantize: %lld columns", (long long) n_cols);
        std::exit(1);
    }
    if (n_rows <= 0) return;
    dense_norm_quantize_kernel<<<(unsigned) n_rows, 512, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), w, xn, (int) n_cols, eps, (uint8_t*) y, (int) n_rows);
    check("dense_norm_quantize");
}

void dense_swiglu_quantize(const float* g, const float* u, const float* sg, const float* su, int64_t n_rows, int64_t width,
                           void* y, void* stream) {
    if (width % kDenseChunk != 0) {
        basalt::log::error("basalt", "dense_swiglu_quantize: width %lld", (long long) width);
        std::exit(1);
    }
    const long long n = (long long) n_rows * width;
    if (n <= 0) return;
    dense_swiglu_quantize_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(g, u, sg, su, n, (int) width,
                                                                                                (int) n_rows, (uint8_t*) y);
    check("dense_swiglu_quantize");
}

}  // namespace basalt::kernels
