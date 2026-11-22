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
#include "basalt/kernels/mma884.cuh"
#include "basalt/kernels/dense_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <cstdint>
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
// nvfp4_scale_dev: dense_decode.cuh
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
int table_dp4a(int t, int ncols) {
    const int v = ncols >= 1 && ncols <= kDenseMaxRows ? kRows[t == kNvfp4 ? 0 : 1][ncols] : 1;
    return v < 0 ? -v : v;
}
// (dense_forward_bench: a whole forward's NVFP4 products as forward.cpp launches them, ms) the widths
// where variant 21, the tensor cores, beats the table above: NVFP4 from 5 rows (5: 23.34 against 24.08; 8: 23.88 against
// 29.04; 12: 30.26 against 39.97; 16: 31.35 against 50.57), Q8_0 at 15 and 16 (the head: 2.73 against 2.73, 2.74 against
// 2.87; dense_mmvq_bench's Q8_0 shapes at 16: 117.8 against 128.6 and 76.3 against 77.9 us).  Not the same bits as the
// table's dp4a variants (fp16: ~2e-4 of the largest value).  OFF by default (one arithmetic at
// every width): with the dp4a variants alone a row's result is the same at every width (the drafter changes
// the speed, never the answer: the same text with and without it), the arithmetic is the one checked bit for bit against
// iq_mmvq, and an int8 activation cannot overflow (fp16 tops at 65504: no guard yet).  Mixing the two made the greedy text
// change from one run to the next (the drafter picks its widths from measured times).  dense_mmvq_set_tensor_cores(true):
// the widths below go to the tensor cores.
constexpr bool kMma[2][kDenseMaxRows + 1] = {
    {false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true},       // NVFP4
    {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true}};   // Q8_0
bool g_mma_on = false;
// (then variant 22, NVFP4's warp-wide weight reads: 5 rows 21.84, 8 rows 22.47, 12 rows 26.45, 16 rows 27.38 ms; 4 rows
// 21.49 against the table's 21.62, left to dp4a): NVFP4 goes to 22, Q8_0 to 21.
int table_rows(int t, int ncols) {
    if (g_mma_on && ncols >= 1 && ncols <= kDenseMaxRows && kMma[t == kNvfp4 ? 0 : 1][ncols]) return t == kNvfp4 ? 22 : 21;
    return table_dp4a(t, ncols);
}
// the variant for a call of `ncols` columns (the same bits whichever): 1, 2, 4 weight rows a warp on the direct kernel,
// 11, 12, 14 on the staged one
int rows_per_warp(int t, int ncols) {
    const int v = g_rows_per_warp;
    if (v == 1 || v == 2 || v == 4 || v == 11 || v == 12 || v == 14 || v == 21 || v == 22) return v;
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
// ---------------------------------------------------------------- wide windows on Volta's tensor cores
// dense_mmvq's products cost a dp4a chain per (row, column, call): at 16 columns a layer's NVFP4 products took ~2.5x
// their bytes' time.  Variant 21 runs them on HMMA (mma884.cuh: m8n8k4, fp16 in, fp32 accumulate).  A lane holds ONE
// weight row (A .row: 4 consecutive values a k-step), decoded in registers: NVFP4 code x sub-block scale, EXACT in fp16
// for each of the 4096 code x scale-byte pairs (the code's fp16 high byte from a 16-byte table, PRMT), Q8_0 q x d rounded
// to fp16; the window's columns (8 a tile, one or two tiles) come from shared memory, where the block stages each chunk
// of 512 activations as fp16 (dense_quantize_q8's int8 x its scale, rounded), one chunk ahead.  A quad pair = 8 rows x 8
// columns; QS quad pairs share their rows and split each chunk's values (QS = 1: 32 rows a warp, 2: 16, 4: 8, chosen
// from the launch's rows so that a small matrix still fills the card), summed by shuffles at the end, in a fixed order.
// Another rounding than dense_mmvq (fp16 activations, the tensor cores' sums; ~2e-4 of the largest value at the 27B's
// shapes): dense_mmvq_bench reports its error instead of a bitwise check, and the quality gate runs through it.  The
// first version (WMMA 16x16x16, weights through shared memory, commit 3d68a178) was 2.5 to 4 times slower than
// dense_mmvq: each warp waited for its loads, then its decode, then its products.
// (the decoding: dense_decode.cuh)
constexpr int kMmaWarps = 8;
template<int QS, int NT> struct MmaShape {
    static constexpr int RW = 32 / QS;              // weight rows a warp
    static constexpr int ROWS = kMmaWarps * RW;     // a block
    static constexpr int KQC = 128 / QS;            // k-quads (4 values) of a chunk per class
    static constexpr int NG = 4 / QS;               // 128-value groups of a chunk per class
    // a chunk's activations: [k-quad][tile][8 columns][4 halves].  An 8-byte shared load or store runs as two half warps
    // (nvprof: 4 transactions a request before these pads): one tile, 64 bytes more every 4 k-quads (a staging half warp
    // writes rows kq and kq + 4: on the two halves of the banks); QS > 1, 128 / QS more per class (a half warp's QS
    // classes then read QS distinct quarters or halves of the banks)
    static constexpr int PADQ = NT == 1 ? 64 : 0, PADC = QS > 1 ? 128 / QS : 0;
    static constexpr int BUF = 128 * NT * 64 + 32 * PADQ + (QS - 1) * PADC;
    __host__ __device__ static constexpr int off(int kq, int t, int col) {
        return kq * (NT * 64) + (kq >> 2) * PADQ + (kq / KQC) * PADC + t * 64 + col * 8;
    }
};
// a lane's 128 weight values: GW 8-byte loads (GB bytes) at offset (group) GB of its class in each chunk of CB bytes
template<int TY> struct MmaW;
template<> struct MmaW<kNvfp4> { static constexpr int GW = 9, GB = 72, CB = 288; };    // 2 blocks of 36 bytes
template<> struct MmaW<kQ8_0> { static constexpr int GW = 17, GB = 136, CB = 544; };   // 4 blocks of 34 bytes
template<int GW>
__device__ __forceinline__ void mma_load(const uint8_t* p, uint2 (&g)[GW]) {
#pragma unroll
    for (int i = 0; i < GW; ++i) g[i] = __ldg(reinterpret_cast<const uint2*>(p) + i);
}
template<int GW>
__device__ __forceinline__ uint32_t mma_word(const uint2 (&g)[GW], int k) { return (k & 1) ? g[k >> 1].y : g[k >> 1].x; }

// one k-quad (4 values) against the window's tiles: the lane's A fragment, its B column from the staged chunk
template<int NT>
__device__ __forceinline__ void mma_kq(const uint8_t* bq, unsigned a0, unsigned a1, float (&acc)[NT][8]) {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const uint2 b = *reinterpret_cast<const uint2*>(bq + t * 64);
        mma884(acc[t], a0, a1, b.x, b.y);
    }
}
// a lane's 128 values (32 k-quads in order), bg: its B column at the group's first k-quad
template<int TY, int QS, int NT>
__device__ __forceinline__ void mma_group(const uint2 (&g)[MmaW<TY>::GW], const uint8_t* bg, float (&acc)[NT][8]) {
    using Sh = MmaShape<QS, NT>;
    auto at = [&](int dk) { return bg + dk * (NT * 64) + (dk >> 2) * Sh::PADQ; };   // the group starts at a multiple of 4
    if constexpr (TY == kNvfp4) {   // 2 blocks: word 9 b = the 4 sub-block scales, words 9 b + 1 + 2 s + w the codes
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const uint32_t sw = mma_word(g, 9 * b);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const unsigned s2 = nvfp4_scale_h2(sw >> (8 * s));
#pragma unroll
                for (int w = 0; w < 2; ++w) {   // values 4 w .. 4 w + 3 (low nibbles) and 8 + 4 w .. (high nibbles)
                    unsigned lo[2], hi[2];
                    fp4_frags(mma_word(g, 9 * b + 1 + 2 * s + w), s2, lo, hi);
                    const int kq = 16 * b + 4 * s + w;
                    mma_kq<NT>(at(kq), lo[0], lo[1], acc);
                    mma_kq<NT>(at(kq + 2), hi[0], hi[1], acc);
                }
            }
        }
    } else {   // 4 blocks of 34 bytes: d, then 32 int8; block i's words start 2 bytes off a word when i is even
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b0 = 34 * i;
            const uint32_t dw = mma_word(g, b0 / 4);
            const __half d = __ushort_as_half((unsigned short) (b0 % 4 == 0 ? dw & 0xFFFF : dw >> 16));
            const __half2 d2 = __halves2half2(d, d);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const int s0 = b0 + 2 + 4 * m;
                const uint32_t q = s0 % 4 == 0 ? mma_word(g, s0 / 4)
                                                : __byte_perm(mma_word(g, s0 / 4), mma_word(g, s0 / 4 + 1), 0x5432);
                __half2 h0, h1;
                i8x4_to_half(q, h0, h1);
                mma_kq<NT>(at(8 * i + m), h2_bits(__hmul2(h0, d2)), h2_bits(__hmul2(h1, d2)), acc);
            }
        }
    }
}

template<int TY, int QS, int NT, int EPI>
__global__ void __launch_bounds__(kMmaWarps * 32, 2) dense_mma_kernel(const Segs sg, size_t row_bytes,
                                                                    const uint8_t* __restrict__ x, int n_in, int ncols) {
    using Sh = MmaShape<QS, NT>;
    using Wf = MmaW<TY>;
    __shared__ __align__(16) uint8_t bsm[2][Sh::BUF];
    int seg = 0;
#pragma unroll
    for (int i = 1; i < kDenseMaxSegs; ++i)
        if (i < sg.n && (int) blockIdx.x >= sg.first[i]) seg = i;
    const int n_out = sg.rows[seg];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qp = (lane >> 2) & 3, rs = qp / QS, kc = qp % QS, ab = mma884_ab_index(lane);
    const int row0 = ((int) blockIdx.x - sg.first[seg]) * Sh::ROWS + warp * Sh::RW + rs * 8;   // the quad pair's 8 rows
    const uint8_t* __restrict__ wrow = sg.w[seg] + (size_t) min(row0 + ab, n_out - 1) * row_bytes + kc * Sh::NG * Wf::GB;
    const int chunks = n_in / kDenseChunk, groups = chunks * Sh::NG;
    auto gaddr = [&](int n) { return wrow + (size_t) (n / Sh::NG) * Wf::CB + (n % Sh::NG) * Wf::GB; };
    // the staging of a chunk: thread tid takes column sc and values [VPT sp, +VPT) (16 or 32: within one q8 block)
    constexpr int CPB = NT * 8, VPT = 4096 * NT / (kMmaWarps * 32), XW = VPT / 16;
    const int sc = tid % CPB, sp = tid / CPB;
    const bool live = sc < ncols;   // a column past the window reads the last one and stores zeros (its scale 0): no branch
    const uint8_t* xcol = x + (size_t) min(sc, ncols - 1) * kChunkBytes + sp * VPT;
    int4 xq[XW];
    unsigned short xd = 0;
    auto fetch = [&](int j) {
        const uint8_t* xc = xcol + (size_t) j * ncols * kChunkBytes;
#pragma unroll
        for (int i = 0; i < XW; ++i) xq[i] = __ldg(reinterpret_cast<const int4*>(xc) + i);
        xd = __ldg(reinterpret_cast<const unsigned short*>(xc - sp * VPT + kDenseChunk) + sp * VPT / 32);
    };
    auto commit = [&](uint8_t* buf) {
        const __half d = __ushort_as_half(live ? xd : (unsigned short) 0);
        const __half2 d2 = __halves2half2(d, d);
#pragma unroll
        for (int i = 0; i < VPT / 4; ++i) {
            const int4 q = xq[i / 4];
            __half2 h0, h1;
            i8x4_to_half((unsigned) ((i & 3) == 0 ? q.x : (i & 3) == 1 ? q.y : (i & 3) == 2 ? q.z : q.w), h0, h1);
            *reinterpret_cast<uint2*>(buf + Sh::off(sp * (VPT / 4) + i, sc / 8, sc % 8)) =
                make_uint2(h2_bits(__hmul2(h0, d2)), h2_bits(__hmul2(h1, d2)));
        }
    };
    float acc[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
    uint2 ga[Wf::GW], gb[Wf::GW];
    mma_load(gaddr(0), ga);   // weights before the wait: they do not depend on the previous launch
    pdl_wait();
    fetch(0);
    commit(bsm[0]);
    const int bcol = Sh::off(kc * Sh::KQC, 0, ab);   // the lane's B column at its class's first k-quad
    // group n of the lane (chunk n / NG): the chunk's B was staged one chunk ahead; cur holds n, n + 1 loads into nxt.
    // No branch: the last group and chunk load themselves again, the last commit fills the buffer no one reads any more.
    auto step = [&](int n, uint2 (&cur)[Wf::GW], uint2 (&nxt)[Wf::GW]) {
        const int j = n / Sh::NG, g = n % Sh::NG;
        if (g == 0) {
            __syncthreads();   // chunk j staged; every warp is done with the other buffer
            fetch(min(j + 1, chunks - 1));
        }
        mma_load(gaddr(min(n + 1, groups - 1)), nxt);
        mma_group<TY, QS, NT>(cur, bsm[j & 1] + bcol + Sh::off(32 * g, 0, 0), acc);
        if (g == Sh::NG - 1) commit(bsm[(j + 1) & 1]);
    };
#pragma unroll 1
    for (int n = 0; n < groups; n += 2) {
        step(n, ga, gb);
        if (n + 1 < groups) step(n + 1, gb, ga);
    }
    // the classes of a quad-pair group hold partial sums of the same cells: xor 4 (QS 2), then 8 (QS 4)
    if constexpr (QS > 1) {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[t][i] += __shfl_xor_sync(0xffffffffu, acc[t][i], 4);
                if constexpr (QS == 4) acc[t][i] += __shfl_xor_sync(0xffffffffu, acc[t][i], 8);
            }
    }
    if (kc != 0) return;
    const float* sc_ = sg.scale[seg];
    float* y = sg.y[seg];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + mma884_c_row(lane, i), col = 8 * t + mma884_c_col(lane, i);
            if (row >= n_out || col >= ncols) continue;
            const size_t o = (size_t) col * n_out + row;
            const float v = sc_ ? __fmul_rn(acc[t][i], *sc_) : acc[t][i];
            if constexpr (EPI == EPI_STORE) y[o] = v;
            else sg.resid[o] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(sg.resid[o]), v));
        }
}
int g_mma_qs = 0;   // 0: chosen from the launch's rows
template<int TY, int QS, int NT, int EPI>
void go_mma(Segs sg, size_t rb, const uint8_t* x, int n_in, int ncols, cudaStream_t s) {
    constexpr int rows = MmaShape<QS, NT>::ROWS;
    for (int i = 0; i < sg.n; ++i) sg.first[i + 1] = sg.first[i] + (sg.rows[i] + rows - 1) / rows;
    launch_pdl(dense_mma_kernel<TY, QS, NT, EPI>, dim3((unsigned) sg.first[sg.n]), dim3(kMmaWarps * 32), 0, s, sg, rb, x, n_in,
               ncols);
}
template<int TY, int NT, int EPI>
void mma_by_split(int qs, const Segs& sg, size_t rb, const uint8_t* x, int n_in, int ncols, cudaStream_t s) {
    if (qs == 1) go_mma<TY, 1, NT, EPI>(sg, rb, x, n_in, ncols, s);
    else if (qs == 2) go_mma<TY, 2, NT, EPI>(sg, rb, x, n_in, ncols, s);
    else go_mma<TY, 4, NT, EPI>(sg, rb, x, n_in, ncols, s);
}
// whether variant 21 can take the launch: 8-byte aligned matrices (its loads), then the split for its rows: at least
// ~1000 warps when it can (the card holds 16 a SM)
bool mma_ok(const Segs& sg) {
    for (int i = 0; i < sg.n; ++i)
        if (reinterpret_cast<uintptr_t>(sg.w[i]) % 8 != 0) return false;
    return true;
}
template<int TY, int EPI>
void launch_mma(const Segs& sg, size_t rb, const uint8_t* x, int n_in, int ncols, cudaStream_t s) {
    int rows = 0;
    for (int i = 0; i < sg.n; ++i) rows += sg.rows[i];
    // QS 4 by default: the weight reads alone (dense_mmvq_bench's probe) reach 700-770 GB/s with 8 rows a warp, 240-570 with
    // 32 (each load then spans 32 rows), and the split sweep at the 27B's shapes agrees
    (void) rows;
    const int qs = g_mma_qs == 1 || g_mma_qs == 2 || g_mma_qs == 4 ? g_mma_qs : 4;
    if (ncols <= 8) mma_by_split<TY, 1, EPI>(qs, sg, rb, x, n_in, ncols, s);
    else mma_by_split<TY, 2, EPI>(qs, sg, rb, x, n_in, ncols, s);
}

// ---------------------------------------------------------------- variant 22: NVFP4, a warp's weights read whole
// nvprof on variant 21 (gate/up, 8 rows): each 8-byte weight load touched 28 to 30 sectors (8 rows x 4 pieces 72 bytes
// apart), 31 % of the stalls waited for data and 13 % on the memory pipe.  Here a warp loads its 8 rows' chunk (8 x 288
// bytes) as consecutive 8-byte pieces (a load: 256 bytes of at most two rows) and hands them over through shared memory to
// the lanes that decode them (each lane its row's 72 bytes, as in variant 21; neither the stores nor the loads conflict).
// And a sub-block's values pair as get_int_from_table_16 leaves them before its last two byte permutes: (0, 8), (1, 9) ...
// (fp4_frags_paired); the staged activations follow that order (act_pairs).  QS 4 (8 rows a warp, the 4 quad pairs one
// class of the chunk each).  One tile: the activations double-buffered (one barrier a chunk); two: single (two barriers).
template<int NT> struct Mma3Shape {
    static constexpr int ROWS = kMmaWarps * 8, KQC = 32;
    static constexpr int PADQ = NT == 1 ? 64 : 0, PADC = 32;
    static constexpr int BUF = 128 * NT * 64 + 32 * PADQ + 3 * PADC;
    static constexpr bool DB = NT == 1;
    static constexpr int WB = 8 * 288;   // a warp's 8 rows of a chunk
    __host__ __device__ static constexpr int off(int kq, int t, int col) {
        return kq * (NT * 64) + (kq >> 2) * PADQ + (kq / KQC) * PADC + t * 64 + col * 8;
    }
};
template<int NT>
__device__ __forceinline__ void mma3_group(const uint2 (&g)[9], const uint8_t* bg, float (&acc)[NT][8]) {
    using Sh = Mma3Shape<NT>;
    auto at = [&](int dk) { return bg + dk * (NT * 64) + (dk >> 2) * Sh::PADQ; };
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const uint32_t sw = mma_word(g, 9 * b);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned s2 = nvfp4_scale_h2(sw >> (8 * s));
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                unsigned q0[2], q1[2];
                fp4_frags_paired(mma_word(g, 9 * b + 1 + 2 * s + w), s2, q0, q1);
                const int kq = 16 * b + 4 * s + 2 * w;
                mma_kq<NT>(at(kq), q0[0], q0[1], acc);
                mma_kq<NT>(at(kq + 1), q1[0], q1[1], acc);
            }
        }
    }
}
template<int NT, int EPI>
__global__ void __launch_bounds__(kMmaWarps * 32, 2) dense_mma3_kernel(const Segs sg, size_t row_bytes,
                                                                     const uint8_t* __restrict__ x, int n_in, int ncols) {
    using Sh = Mma3Shape<NT>;
    __shared__ __align__(16) uint8_t bsm[Sh::DB ? 2 : 1][Sh::BUF];
    __shared__ __align__(16) uint8_t wsm[kMmaWarps][Sh::WB];
    int seg = 0;
#pragma unroll
    for (int i = 1; i < kDenseMaxSegs; ++i)
        if (i < sg.n && (int) blockIdx.x >= sg.first[i]) seg = i;
    const int n_out = sg.rows[seg];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kc = (lane >> 2) & 3, ab = mma884_ab_index(lane);
    const int row0 = ((int) blockIdx.x - sg.first[seg]) * Sh::ROWS + warp * 8;
    // the warp's pieces: piece p = 32 i + lane of its rows' chunk, row p / 36, bytes 8 (p % 36); in wsm at 8 p
    const uint8_t* __restrict__ wbase = sg.w[seg];
    uint32_t po[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int p = 32 * i + lane, r = p / 36;
        po[i] = (uint32_t) min(row0 + r, n_out - 1) * (uint32_t) row_bytes + 8u * (uint32_t) (p - 36 * r);
    }
    uint8_t* const wmine = wsm[warp];
    const uint8_t* const wlane = wmine + ab * 288 + 72 * kc;   // the lane's 72 bytes (2 blocks of its row)
    const int chunks = n_in / kDenseChunk;
    auto wload = [&](int j, uint2 (&raw)[9]) {
        const uint8_t* c = wbase + (size_t) j * 288;
#pragma unroll
        for (int i = 0; i < 9; ++i) raw[i] = __ldg(reinterpret_cast<const uint2*>(c + po[i]));
    };
    auto wpass = [&](const uint2 (&raw)[9], uint2 (&cur)[9]) {   // the warp's pieces to its lanes (after __syncwarp)
#pragma unroll
        for (int i = 0; i < 9; ++i) *reinterpret_cast<uint2*>(wmine + 256 * i + 8 * lane) = raw[i];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 9; ++i) cur[i] = *reinterpret_cast<const uint2*>(wlane + 8 * i);
        __syncwarp();   // every lane has its pieces before the next chunk's are stored
    };
    // the activations' staging: thread tid takes column sc and values [VPT sp, +VPT) (one or two sub-blocks)
    constexpr int CPB = NT * 8, VPT = 4096 * NT / (kMmaWarps * 32), XW = VPT / 16;
    const int sc = tid % CPB, sp = tid / CPB;
    const bool live = sc < ncols;
    const uint8_t* xcol = x + (size_t) min(sc, ncols - 1) * kChunkBytes;
    int4 xq[XW];
    unsigned short xd = 0;
    auto fetch = [&](int j) {
        const uint8_t* xc = xcol + (size_t) j * ncols * kChunkBytes;
#pragma unroll
        for (int i = 0; i < XW; ++i) xq[i] = __ldg(reinterpret_cast<const int4*>(xc + sp * VPT) + i);
        xd = __ldg(reinterpret_cast<const unsigned short*>(xc + kDenseChunk) + sp * VPT / 32);
    };
    auto commit = [&](uint8_t* buf) {
        const __half d = __ushort_as_half(live ? xd : (unsigned short) 0);
        const __half2 d2 = __halves2half2(d, d);
#pragma unroll
        for (int u = 0; u < XW; ++u)       // sub-block u of the thread's values
#pragma unroll
            for (int w = 0; w < 2; ++w) {   // its words w and w + 2: k-quads 2 w and 2 w + 1
                const int4 q = xq[u];
                uint32_t k0, k1;
                act_pairs((uint32_t) (w ? q.y : q.x), (uint32_t) (w ? q.w : q.z), k0, k1);
                __half2 h0, h1, h2, h3;
                i8x4_to_half(k0, h0, h1);
                i8x4_to_half(k1, h2, h3);
                const int kq = sp * (VPT / 4) + 4 * u + 2 * w;
                *reinterpret_cast<uint2*>(buf + Sh::off(kq, sc / 8, sc % 8)) =
                    make_uint2(h2_bits(__hmul2(h0, d2)), h2_bits(__hmul2(h1, d2)));
                *reinterpret_cast<uint2*>(buf + Sh::off(kq + 1, sc / 8, sc % 8)) =
                    make_uint2(h2_bits(__hmul2(h2, d2)), h2_bits(__hmul2(h3, d2)));
            }
    };
    float acc[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
    uint2 raw[9], cur[9];
    wload(0, raw);   // weights before the wait: they do not depend on the previous launch
    pdl_wait();
    fetch(0);
    commit(bsm[0]);
    wpass(raw, cur);
    __syncthreads();
    const int bcol = Sh::off(kc * Sh::KQC, 0, ab);
#pragma unroll 1
    for (int j = 0; j < chunks; ++j) {
        const int jn = min(j + 1, chunks - 1);   // the last chunk loads itself again: no branch
        wload(jn, raw);
        fetch(jn);
        mma3_group<NT>(cur, bsm[Sh::DB ? (j & 1) : 0] + bcol, acc);
        if constexpr (Sh::DB) {
            commit(bsm[(j + 1) & 1]);
        } else {
            __syncthreads();   // every warp is done with chunk j's activations
            commit(bsm[0]);
        }
        wpass(raw, cur);
        __syncthreads();       // chunk j + 1's activations staged
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] += __shfl_xor_sync(0xffffffffu, acc[t][i], 4);
            acc[t][i] += __shfl_xor_sync(0xffffffffu, acc[t][i], 8);
        }
    if (kc != 0) return;
    const float* sc_ = sg.scale[seg];
    float* y = sg.y[seg];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + mma884_c_row(lane, i), col = 8 * t + mma884_c_col(lane, i);
            if (row >= n_out || col >= ncols) continue;
            const size_t o = (size_t) col * n_out + row;
            const float v = sc_ ? __fmul_rn(acc[t][i], *sc_) : acc[t][i];
            if constexpr (EPI == EPI_STORE) y[o] = v;
            else sg.resid[o] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(sg.resid[o]), v));
        }
}
template<int NT, int EPI>
void go_mma3(Segs sg, size_t rb, const uint8_t* x, int n_in, int ncols, cudaStream_t s) {
    constexpr int rows = Mma3Shape<NT>::ROWS;
    for (int i = 0; i < sg.n; ++i) sg.first[i + 1] = sg.first[i] + (sg.rows[i] + rows - 1) / rows;
    launch_pdl(dense_mma3_kernel<NT, EPI>, dim3((unsigned) sg.first[sg.n]), dim3(kMmaWarps * 32), 0, s, sg, rb, x, n_in, ncols);
}
// variant 22's offsets are 32-bit: a matrix of NVFP4 rows within 4 GB
bool mma3_ok(const Segs& sg, size_t rb) {
    for (int i = 0; i < sg.n; ++i)
        if ((size_t) sg.rows[i] * rb >= (size_t) 0xFFFFFFFFu) return false;
    return mma_ok(sg);
}

template<int EPI>
void launch_segs(int t, const Segs& sg, const void* x, int n_in, int ncols, cudaStream_t s) {
    const size_t rb = t == kNvfp4 ? (size_t) (n_in / 64) * sizeof(block_nvfp4) : (size_t) (n_in / 32) * sizeof(block_q8_0);
    int r = rows_per_warp(t, ncols);
    if (r == 22 && (t != kNvfp4 || !mma3_ok(sg, rb))) r = 21;   // variant 22 is NVFP4's; Q8_0 keeps 21
    if (r == 22) {
        if (ncols <= 8) go_mma3<1, EPI>(sg, rb, (const uint8_t*) x, n_in, ncols, s);
        else go_mma3<2, EPI>(sg, rb, (const uint8_t*) x, n_in, ncols, s);
        return;
    }
    if (r == 21 && !mma_ok(sg)) {
        static bool told = false;   // said once: the engine's matrices are expected aligned
        if (!told) basalt::log::warn("basalt", "dense_mmvq: a matrix not 8-byte aligned, the dp4a variants instead of the tensor cores");
        told = true;
        r = table_dp4a(t, ncols);
    }
    if (r == 21) {   // the tensor cores
        if (t == kNvfp4) launch_mma<kNvfp4, EPI>(sg, rb, (const uint8_t*) x, n_in, ncols, s);
        else launch_mma<kQ8_0, EPI>(sg, rb, (const uint8_t*) x, n_in, ncols, s);
        return;
    }
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
void dense_mmvq_set_mma_split(int qs) { g_mma_qs = qs; }
void dense_mmvq_set_tensor_cores(bool on) { g_mma_on = on; }

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
