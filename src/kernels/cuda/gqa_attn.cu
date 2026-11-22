// src/kernels/cuda/gqa_attn.cu - see include/basalt/kernels/gqa_attn.hpp.
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/dense_decode.cuh"
#include "basalt/kernels/mma884.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "basalt/log.hpp"
#include "basalt/kernels/volta_compat.cuh"

namespace basalt::kernels {
namespace {

constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = 32;             // cells per decode chunk
constexpr int kDecSplits = 64;        // a decode row's blocks per KV head (its chunks dealt round robin)
constexpr int HD = 256;              // the prompt kernel's head width
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;   // 1/sqrt(256) * log2(e)
constexpr float kNeg = -1e30f;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- decode
// Row t of a window sees positions [lo_t, hi_t): hi_t = pos0 + t + 1 (causal) or `hi`, lo_t = pos0 + t - window + 1 (a
// sliding window, else 0); position p sits in cache cell p % cap.  Cells are split in CHUNK-position chunks from chunk
// `base`; block b of a row takes chunks b, b + gx, ... (gx from the capacity alone) and folds them in order, so a row's
// sums depend on its own range only, never on the window's rows or where the first cell is read from.
struct Range {
    int64_t pos0 = 0, hi = 0, window = 0, cap = 0;
    int causal = 1, base = 0;
    const int32_t* pos_dev = nullptr;   // pos0 read from the device (a captured window)
};
__device__ __forceinline__ int64_t row_pos0(const Range& r) { return r.pos_dev ? (int64_t) *r.pos_dev : r.pos0; }
// A split cache (KvFar): true when cell c lives in the far buffers, `c` then its row there
__device__ __forceinline__ bool kv_far(const KvFar& f, int64_t& c) {
    if (c < f.at) return false;
    c -= f.at;
    return true;
}

// The windows one launch takes: segment k is rows [row0[k], row0[k + 1]) of q, the partials and the output, its 16-row
// tiles of (row, head) pairs [tile0[k], tile0[k + 1]) of the grid, with its own cache and range.  Each block and each
// merged row computes exactly what a launch of its segment alone would.
constexpr int kMaxSegs = kGqaMaxWindow;
struct DecSegs {
    int n = 1;
    int row0[kMaxSegs + 1] = {};
    int tile0[kMaxSegs + 1] = {};
    int grp0[kMaxSegs + 1] = {};   // decode_wmma2_kernel: segment k's groups of NTB tiles, blocks [grp0[k], grp0[k + 1])
    const __half* kc[kMaxSegs] = {};
    const __half* vc[kMaxSegs] = {};
    KvInt8 q8[kMaxSegs] = {};
    KvFar far_kv[kMaxSegs] = {};
    Range r[kMaxSegs];
};
__device__ __forceinline__ int seg_of(const int* first, int n, int i) {
    int k = 0;
    while (k + 1 < n && i >= first[k + 1]) ++k;
    return k;
}
__device__ __forceinline__ int64_t row_hi(const Range& r, int t) { return r.causal ? row_pos0(r) + t + 1 : r.hi; }
__device__ __forceinline__ int64_t row_lo(const Range& r, int t) {
    return r.window > 0 ? max((int64_t) 0, row_pos0(r) + t - r.window + 1) : 0;
}

// Block (query head, row t), one thread per dim: the row's splits merged in order.  Splits past the row's last chunk
// saw no cell of it ({max kNeg, sum 0, zeros}: every term they add is an exact zero), so the merge stops at the last
// split that holds one of its cells; the sums and their order are those of the full loop.
template <int HD>
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* part, int n_head, int gx, float* out, const float* gate_q,
                                                          uint16_t* gated16, const DecSegs segs) {
    pdl_launch_dependents();
    pdl_wait();
    const int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;   // row: of the launch, t: of its segment
    const int k = seg_of(segs.row0, segs.n, row), t = row - segs.row0[k];
    const Range& r = segs.r[k];
    const float* p = part + ((int64_t) row * n_head + h) * gx * (HD + 2);
    const int64_t chunks = (row_hi(r, t) + CHUNK - 1) / CHUNK - r.base;   // the row's chunks from the first split's
    const int used = (int) max((int64_t) 1, min((int64_t) gx, chunks));
    float m = kNeg;
    for (int b = 0; b < used; ++b) m = fmaxf(m, p[(int64_t) b * (HD + 2)]);
    float l = 0.0f, a = 0.0f;
    for (int b = 0; b < used; ++b) {
        const float* pb = p + (int64_t) b * (HD + 2);
        const float w = exp2f(pb[0] - m);
        l = fmaf(pb[1], w, l);
        a = fmaf(pb[2 + d], w, a);
    }
    const float o = a / l;
    const int64_t i = ((int64_t) row * n_head + h) * HD + d;
    if (out) out[i] = o;
    if (gated16) {
        const float g = gate_q[((int64_t) row * n_head + h) * 2 * HD + HD + d];
        gated16[i] = __bfloat16_as_ushort(__float2bfloat16_rn(o / (1.0f + expf(-g))));
    }
}

// the splits a row's chunks go to: from the capacity's chunks alone
int decode_splits(int64_t n_chunks) { return (int) std::max<int64_t>(1, std::min<int64_t>(kDecSplits, n_chunks)); }

// ---------------------------------------------------------------- prompt: shared helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    basalt::volta::ldmatrix_x2<false>(r0, r1, smem_u32(p));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    basalt::volta::ldmatrix_x4<false>(r0, r1, r2, r3, smem_u32(p));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    basalt::volta::ldmatrix_x4<true>(r0, r1, r2, r3, smem_u32(p));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    basalt::volta::mma_16816_f32_f16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
}
__device__ __forceinline__ void mma16816_h(unsigned (&c)[2], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                           unsigned b1) {
    basalt::volta::mma_16816_f16_f16(c[0], c[1], a0, a1, a2, a3, b0, b1, c[0], c[1]);
}
__device__ __forceinline__ void mma16832_s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1) {
    basalt::volta::mma_16832_s32_s8<false>(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
}

__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    basalt::volta::ldmatrix_x2<true>(r0, r1, smem_u32(p));
}

// ---------------------------------------------------------------- decode on the tensor cores
// Block (split b, KV head, 16 query rows): the rows are the window's (row, head) pairs of the KV group, m = t G + h.  Its
// chunks b, b + gx, ... of CHUNK cells: K and V decoded to fp16 in shared memory, S = Q K^T on mma m16n8k16 (fp16 in,
// fp32 accumulate), an online softmax per row over its cells, O = O alpha + P V (P in fp16) on mma; the rows' partials
// {max, sum, O} go to the merge.  Every row's arithmetic is its own (mma columns and rows never meet), the chunks follow
// positions and the splits the capacity, so a row's bits are the same at any window width.
template <int HD>
// (launched through launch_pdl: q and the cache come from the kernels before, read after pdl_wait, so not __restrict__)
__global__ void __launch_bounds__(128) decode_mma_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs, int n_chunks,
                                                         float scale, float* part_all) {
    constexpr int LD = HD + 8;          // padded rows (halves): ldmatrix without bank conflicts
    constexpr int NT = HD / 32;         // P V: a warp's 8-dim tiles
    __shared__ __align__(16) __half qs[16][LD];
    __shared__ __align__(16) __half ks[CHUNK][LD];
    __shared__ __align__(16) __half vs[CHUNK][LD];
    __shared__ float ss[16][CHUNK + 1];
    __shared__ __align__(16) __half ps[16][CHUNK + 8];
    __shared__ float rm[16], rl[16], ra[16];
    __shared__ int64_t rlo[16], rhi[16];
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.tile0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G, m0 = ((int) blockIdx.z - segs.tile0[sk]) * 16;
    // the chunk's cells [from, to) of K and V as fp16 (int8: code x scale), zero past the last row's cells
    const auto load_kv = [&](int64_t c0, int64_t hi_max, int from, int to) {
        const int64_t cell0 = c0 % r.cap;
        for (int idx = tid; idx < CHUNK * HD / 8; idx += 128) {
            const int cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            if (cell < from || cell >= to) continue;
            uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
            if (c0 + cell < hi_max) {
                int64_t ce = cell0 + cell;
                if (ce >= r.cap) ce -= r.cap;
                const bool far_kv = kv_far(fr, ce);
                const int64_t row = ce * n_kv + kvh;
                if (q8.k) {
                    const KvInt8& c8 = far_kv ? fr.q8 : q8;
                    const float sk = __half2float(__ushort_as_half(c8.ks[row * (HD / 64) + d8 / 64]));
                    const float sv = __half2float(__ushort_as_half(c8.vs[row * (HD / 64) + d8 / 64]));
                    int8_t kb[8], vb[8];
                    *reinterpret_cast<int2*>(kb) = *reinterpret_cast<const int2*>(c8.k + row * HD + d8);
                    *reinterpret_cast<int2*>(vb) = *reinterpret_cast<const int2*>(c8.v + row * HD + d8);
                    __half2 kh[4], vh[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        kh[e] = __floats2half2_rn((float) kb[2 * e] * sk, (float) kb[2 * e + 1] * sk);
                        vh[e] = __floats2half2_rn((float) vb[2 * e] * sv, (float) vb[2 * e + 1] * sv);
                    }
                    kw = *reinterpret_cast<uint4*>(kh);
                    vw = *reinterpret_cast<uint4*>(vh);
                } else {
                    kw = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.k : kc) + row * HD + d8);
                    vw = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.v : vc) + row * HD + d8);
                }
            }
            *reinterpret_cast<uint4*>(&ks[cell][d8]) = kw;
            *reinterpret_cast<uint4*>(&vs[cell][d8]) = vw;
        }
    };
    // before the wait: the rows' ranges (positions come from the host) and, of the block's first chunk, the cells below
    // the window's first position, which earlier windows wrote; the window's own cells and the queries come after it
    if (tid < 16) {
        const int m = m0 + tid;
        rlo[tid] = m < M ? row_lo(r, m / G) : 0;
        rhi[tid] = m < M ? row_hi(r, m / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[i]);
    const int64_t first_c0 = (int64_t) (r.base + blockIdx.x) * CHUNK;
    const int64_t own = row_pos0(r);   // cells from the window's first position on are this window's (written just before)
    const int pre = (int) max((int64_t) 0, min((int64_t) CHUNK, own - first_c0));
    if ((int) blockIdx.x < n_chunks && first_c0 < hi_max && pre > 0) load_kv(first_c0, hi_max, 0, pre);
    pdl_wait();
    for (int idx = tid; idx < 16 * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        const float v = m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi_max) break;   // past every row's cells
        load_kv(c0, hi_max, chunk == (int) blockIdx.x ? pre : 0, CHUNK);   // the first chunk's earlier cells are in
        __syncthreads();
        // S = Q K^T: warp w the cells 8w .. 8w + 7
        for (int nc = warp; nc < CHUNK / 8; nc += 4) {
            float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < HD; k += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, &qs[lane & 15][k + (lane >> 4) * 8]);
                ldsm_x2(b0, b1, &ks[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s4, a0, a1, a2, a3, b0, b1);
            }
            ss[g][nc * 8 + 2 * c] = s4[0];
            ss[g][nc * 8 + 2 * c + 1] = s4[1];
            ss[g + 8][nc * 8 + 2 * c] = s4[2];
            ss[g + 8][nc * 8 + 2 * c + 1] = s4[3];
        }
        __syncthreads();
        // each row: the chunk's max over its cells, p = exp2(s - max') there (0 elsewhere), the running {max, sum}
        for (int i = warp; i < 16; i += 4) {
            float pv[CHUNK / 32], mx = kNeg;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const int cc = lane + 32 * u;
                const int64_t pos = c0 + cc;
                pv[u] = pos >= rlo[i] && pos < rhi[i] ? ss[i][cc] : -INFINITY;
                mx = fmaxf(mx, pv[u]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mold = rm[i], mnew = fmaxf(mold, mx);
            float l = 0.0f;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const float p = pv[u] == -INFINITY ? 0.0f : exp2f(pv[u] - mnew);
                ps[i][lane + 32 * u] = __float2half_rn(p);
                l += p;
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float alpha = exp2f(mold - mnew);
                ra[i] = alpha;
                rl[i] = fmaf(rl[i], alpha, l);
                rm[i] = mnew;
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w the dims w HD / 4 ..
        {
            const float ag = ra[g], ag8 = ra[g + 8];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                acc[n][0] *= ag;
                acc[n][1] *= ag;
                acc[n][2] *= ag8;
                acc[n][3] *= ag8;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &ps[lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, &vs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
                    mma16816(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();   // the shared tiles are free for the next chunk
    }
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int i = g + 8 * hr, m = m0 + i;
        if (m >= M) continue;
        float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
        if (warp == 0 && c == 0) {
            out[0] = rm[i];
            out[1] = rl[i];
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int d = warp * (HD / 4) + n * 8 + 2 * c;
            out[2 + d] = acc[n][2 * hr];
            out[3 + d] = acc[n][2 * hr + 1];
        }
    }
}

// ---------------------------------------------------------------- decode on Volta's own tensor cores (Lapilli, V100)
// decode_mma_kernel's mma m16n8k16 and ldmatrix are Ampere's: on a V100 volta_compat.cuh emulates them with shuffles and
// scalar FMAs (gqa_attn_parity: 16 rows x 12K cells in 1.47 ms, ~104 GB/s of K/V).  This kernel computes the same partials
// {max, sum, O} per (row, split) on WMMA 16x16x16 (HMMA: fp16 in, fp32 accumulate), in two passes over the block's chunks:
// the rows' max over their cells first, then p = 2^(s - max) (fp16 for P V, fp32 for the sum) and O = P V accumulated in
// fragments that need no rescaling (WMMA does not expose their layout).  Same grid, chunks, splits and partials: the merge is
// decode_merge_kernel's.  Another rounding than decode_mma_kernel (the tensor cores' sums, S in two halves of the head, one
// max instead of a running one); every row still its own, so a row's bits do not depend on the window's width.
template <int HD>
__global__ void __launch_bounds__(128) decode_wmma_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                          int n_chunks, float scale, float* part_all) {
    using namespace nvcuda;
    constexpr int LD = HD + 8;          // halves (a multiple of 8)
    constexpr int SLD = CHUNK + 4;      // floats (a multiple of 4)
    constexpr int PLD = CHUNK + 8;      // halves
    constexpr int OLD = HD + 4;         // floats: the output tile, over the K/V tiles once the chunks are done
    constexpr int NTW = HD / 16 / 4;    // a warp's 16-dim output tiles
    static_assert(CHUNK == 32 && HD % 64 == 0, "two 16-cell tiles a chunk, the head in 4 warps' tiles");
    __shared__ __align__(32) __half qs[16][LD];
    __shared__ __align__(32) __half kvs[2 * CHUNK][LD];   // K cells 0..31, V cells 32..63; then the output tile
    __shared__ __align__(32) float ss[2][16][SLD];       // S of the head's two halves
    __shared__ __align__(32) __half ps[16][PLD];
    __shared__ float rm[16], rl[16];
    __shared__ int64_t rlo[16], rhi[16];
    static_assert(16 * OLD * 4 <= 2 * CHUNK * LD * 2, "the output tile fits over K and V");
    float (*os)[OLD] = reinterpret_cast<float (*)[OLD]>(&kvs[0][0]);
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.tile0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G, m0 = ((int) blockIdx.z - segs.tile0[sk]) * 16;
    // A chunk's cells of K (and V), fetched into registers (every load of the chunk in flight at once: one at a time, each
    // waiting for its store, they made the kernel latency-bound), then decoded to fp16 into the tiles (int8: code x scale,
    // as decode_mma_kernel); zero past the last row's cells.  The next chunk is fetched while this one is multiplied.
    constexpr int LPT = CHUNK * HD / 8 / 128;   // 8-value pieces a thread: 8 (HD 256), 4 (HD 128)
    struct Fetch {
        uint4 k[LPT], v[LPT];   // fp16 cells; int8: k[].x/.y the codes, .z the scale's bits
    };
    const bool i8 = q8.k != nullptr;
    const auto fetch = [&](int64_t c0, int64_t hi_max, bool with_v, Fetch& f) {
        const int64_t cell0 = c0 % r.cap;
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + 128 * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            f.k[u] = f.v[u] = make_uint4(0, 0, 0, 0);
            if (c0 + cell < hi_max) {
                int64_t ce = cell0 + cell;
                if (ce >= r.cap) ce -= r.cap;
                const bool far_kv = kv_far(fr, ce);
                const int64_t row = ce * n_kv + kvh;
                if (i8) {
                    const KvInt8& c8 = far_kv ? fr.q8 : q8;
                    const int2 kb = *reinterpret_cast<const int2*>(c8.k + row * HD + d8);
                    f.k[u] = make_uint4((unsigned) kb.x, (unsigned) kb.y, c8.ks[row * (HD / 64) + d8 / 64], 0);
                    if (with_v) {
                        const int2 vb = *reinterpret_cast<const int2*>(c8.v + row * HD + d8);
                        f.v[u] = make_uint4((unsigned) vb.x, (unsigned) vb.y, c8.vs[row * (HD / 64) + d8 / 64], 0);
                    }
                } else {
                    f.k[u] = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.k : kc) + row * HD + d8);
                    if (with_v) f.v[u] = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.v : vc) + row * HD + d8);
                }
            }
        }
    };
    // 8 int8 codes x their fp16 scale, as decode_mma_kernel rounds them: the code exact in fp16 (dense_decode.cuh's
    // i8x4_to_half, no I2F: a quarter of the rate on a V100), then one fp16 product, the exact product rounded once as
    // the fp32 product (exact: 8 x 11 bits) rounded to fp16 was (measured: the same values)
    const auto dec8 = [](const uint4& w) {
        const __half sch = __ushort_as_half((unsigned short) w.z);
        const __half2 sc2 = __halves2half2(sch, sch);
        __half2 h[4];
        i8x4_to_half(w.x, h[0], h[1]);
        i8x4_to_half(w.y, h[2], h[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[e] = __hmul2(h[e], sc2);
        return *reinterpret_cast<uint4*>(h);
    };
    const auto commit = [&](const Fetch& f, bool with_v) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + 128 * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            *reinterpret_cast<uint4*>(&kvs[cell][d8]) = i8 ? dec8(f.k[u]) : f.k[u];
            if (with_v) *reinterpret_cast<uint4*>(&kvs[CHUNK + cell][d8]) = i8 ? dec8(f.v[u]) : f.v[u];
        }
    };
    // S = Q K^T of the chunk into ss: warp w the 16 cells 16 (w & 1) .. over the head's half w >> 1
    const auto scores = [&]() {
        const int tile = warp & 1, half = warp >> 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
        wmma::fill_fragment(fs, 0.0f);
#pragma unroll
        for (int k = half * (HD / 2); k < (half + 1) * (HD / 2); k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, &qs[0][k], LD);
            wmma::load_matrix_sync(fb, &kvs[tile * 16][k], LD);
            wmma::mma_sync(fs, fa, fb, fs);
        }
        wmma::store_matrix_sync(&ss[half][0][tile * 16], fs, SLD, wmma::mem_row_major);
    };
    if (tid < 16) {
        const int m = m0 + tid;
        rlo[tid] = m < M ? row_lo(r, m / G) : 0;
        rhi[tid] = m < M ? row_hi(r, m / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    pdl_wait();
    for (int idx = tid; idx < 16 * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        const float v = m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[i]);
    // the block's chunks: b, b + gx, ... below hi_max
    const auto chunk_c0 = [&](int chunk) { return (int64_t) (r.base + chunk) * CHUNK; };
    const auto live = [&](int chunk) { return chunk < n_chunks && chunk_c0(chunk) < hi_max; };
    Fetch f;
    // pass 1: every row's max over its cells of the block's chunks
    if (live(blockIdx.x)) fetch(chunk_c0(blockIdx.x), hi_max, false, f);
    for (int chunk = blockIdx.x; live(chunk); chunk += gx) {
        const int64_t c0 = chunk_c0(chunk);
        commit(f, false);
        __syncthreads();
        if (live(chunk + gx)) fetch(chunk_c0(chunk + gx), hi_max, false, f);   // in flight while this chunk is multiplied
        scores();
        __syncthreads();
        for (int i = warp; i < 16; i += 4) {
            const int64_t pos = c0 + lane;
            const float sc = ss[0][i][lane] + ss[1][i][lane];
            float mx = pos >= rlo[i] && pos < rhi[i] ? sc : -INFINITY;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            if (lane == 0) rm[i] = fmaxf(rm[i], mx);
        }
        __syncthreads();
    }
    // pass 2: p = 2^(s - max) and O = P V
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) wmma::fill_fragment(fo[j], 0.0f);
    if (live(blockIdx.x)) fetch(chunk_c0(blockIdx.x), hi_max, true, f);
    for (int chunk = blockIdx.x; live(chunk); chunk += gx) {
        const int64_t c0 = chunk_c0(chunk);
        commit(f, true);
        __syncthreads();
        if (live(chunk + gx)) fetch(chunk_c0(chunk + gx), hi_max, true, f);
        scores();
        __syncthreads();
        for (int i = warp; i < 16; i += 4) {
            const int64_t pos = c0 + lane;
            const float sc = ss[0][i][lane] + ss[1][i][lane];
            const float pr = pos >= rlo[i] && pos < rhi[i] ? exp2f(sc - rm[i]) : 0.0f;
            ps[i][lane] = __float2half_rn(pr);
            const float l = warp_sum(pr);
            if (lane == 0) rl[i] += l;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < CHUNK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, &ps[0][kk], PLD);
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, &kvs[CHUNK + kk][(warp * NTW + j) * 16], LD);
                wmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
        __syncthreads();   // the tiles are free for the next chunk
    }
#pragma unroll
    for (int j = 0; j < NTW; ++j) wmma::store_matrix_sync(&os[0][(warp * NTW + j) * 16], fo[j], OLD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 16 * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        if (m >= M) continue;
        float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
        if (d == 0) {
            out[0] = rm[i];
            out[1] = rl[i];
        }
        out[2 + d] = os[i][d];
    }
}

// ---------------------------------------------------------------- the same arithmetic, laid out for the card
// decode_wmma2_kernel: decode_wmma_kernel row for row (the same loads, the same fp16 values, the same mma sequences, the
// same reductions in the same order: the same bits, gqa_attn_parity compares the two), with the work laid out otherwise:
// a chunk's K and V share one tile of shared memory (V staged after the scores: 3 blocks a SM instead of 2), the next
// chunk's K in flight from the moment this one's is staged; and NTB tiles of 16 rows a block (4 warps each), which share
// every chunk: a window of T rows has T G rows a KV head (16 rows, G = 6: 6 tiles), each chunk was read once a tile.
template <int HD, int NTB>
__global__ void __launch_bounds__(128 * NTB) decode_wmma2_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                                int n_chunks, float scale, float* part_all) {
    using namespace nvcuda;
    constexpr int LD = HD + 8, SLD = CHUNK + 4, PLD = CHUNK + 8, OLD = HD + 4, NTW = HD / 16 / 4, NTH = 128 * NTB;
    static_assert(CHUNK == 32 && HD % 64 == 0, "two 16-cell tiles a chunk, the head in 4 warps' tiles");
    __shared__ __align__(32) __half qs[NTB][16][LD];
    __shared__ __align__(32) __half kv[CHUNK][LD];   // the chunk's K, then (pass 2) its V; a tile's output at the end
    __shared__ __align__(32) float ss[NTB][2][16][SLD];
    __shared__ __align__(32) __half ps[NTB][16][PLD];
    __shared__ float rm[NTB][16], rl[NTB][16];
    __shared__ int64_t rlo[NTB][16], rhi[NTB][16];
    static_assert(16 * OLD * 4 <= CHUNK * LD * 2, "a tile's output fits over the K/V tile");
    float (*os)[OLD] = reinterpret_cast<float (*)[OLD]>(&kv[0][0]);
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2, w4 = warp & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.grp0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const int tiles = segs.tile0[sk + 1] - segs.tile0[sk], tile1 = ((int) blockIdx.z - segs.grp0[sk]) * NTB;
    const bool mine = tile1 + wg < tiles;   // this warp group has a tile
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G;
    constexpr int LPT = CHUNK * HD / 8 / NTH;   // 8-value pieces a thread
    const bool i8 = q8.k != nullptr;
    uint4 fk[LPT], fv[LPT];                     // the next chunk's K and V (int8: .x/.y the codes, .z the scale's bits)
    const auto fetch = [&](int64_t c0, int64_t hi_max, bool v_too) {
        const int64_t cell0 = c0 % r.cap;
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + NTH * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            if (v_too) fv[u] = make_uint4(0, 0, 0, 0);
            else fk[u] = make_uint4(0, 0, 0, 0);
            if (c0 + cell < hi_max) {
                int64_t ce = cell0 + cell;
                if (ce >= r.cap) ce -= r.cap;
                const bool far_kv = kv_far(fr, ce);
                const int64_t row = ce * n_kv + kvh;
                if (i8) {
                    const KvInt8& c8 = far_kv ? fr.q8 : q8;
                    const int8_t* src = v_too ? c8.v : c8.k;
                    const uint16_t* sc = v_too ? c8.vs : c8.ks;
                    const int2 b = *reinterpret_cast<const int2*>(src + row * HD + d8);
                    const uint4 w = make_uint4((unsigned) b.x, (unsigned) b.y, sc[row * (HD / 64) + d8 / 64], 0);
                    if (v_too) fv[u] = w;
                    else fk[u] = w;
                } else {
                    const __half* src = v_too ? (far_kv ? (const __half*) fr.v : vc) : (far_kv ? (const __half*) fr.k : kc);
                    const uint4 w = *reinterpret_cast<const uint4*>(src + row * HD + d8);
                    if (v_too) fv[u] = w;
                    else fk[u] = w;
                }
            }
        }
    };
    const auto dec8 = [](const uint4& w) {   // decode_wmma_kernel's: the code exact in fp16, one fp16 product
        const __half sch = __ushort_as_half((unsigned short) w.z);
        const __half2 sc2 = __halves2half2(sch, sch);
        __half2 h[4];
        i8x4_to_half(w.x, h[0], h[1]);
        i8x4_to_half(w.y, h[2], h[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[e] = __hmul2(h[e], sc2);
        return *reinterpret_cast<uint4*>(h);
    };
    const auto commit = [&](bool v_too) {   // the fetched K (or V) into the tile
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + NTH * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            const uint4 w = v_too ? fv[u] : fk[u];
            *reinterpret_cast<uint4*>(&kv[cell][d8]) = i8 ? dec8(w) : w;
        }
    };
    // S = Q K^T of the chunk for this warp group's tile: warp w4 the 16 cells 16 (w4 & 1) .. over the head's half w4 >> 1
    const auto scores = [&]() {
        const int ct = w4 & 1, half = w4 >> 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
        wmma::fill_fragment(fs, 0.0f);
#pragma unroll
        for (int k = half * (HD / 2); k < (half + 1) * (HD / 2); k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, &qs[wg][0][k], LD);
            wmma::load_matrix_sync(fb, &kv[ct * 16][k], LD);
            wmma::mma_sync(fs, fa, fb, fs);
        }
        wmma::store_matrix_sync(&ss[wg][half][0][ct * 16], fs, SLD, wmma::mem_row_major);
    };
    if (tid < NTB * 16) {
        const int g = tid >> 4, i = tid & 15, m = (tile1 + g) * 16 + i;
        const bool ok = tile1 + g < tiles && m < M;
        rlo[g][i] = ok ? row_lo(r, m / G) : 0;
        rhi[g][i] = ok ? row_hi(r, m / G) : 0;
        rm[g][i] = kNeg;
        rl[g][i] = 0.0f;
    }
    pdl_wait();
    for (int idx = tid; idx < NTB * 16 * HD; idx += NTH) {
        const int g = idx / (16 * HD), i = (idx / HD) & 15, d = idx % HD, m = (tile1 + g) * 16 + i;
        const float v = tile1 + g < tiles && m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[g][i][d] = __float2half_rn(v);
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int g = 0; g < NTB; ++g)
#pragma unroll
        for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[g][i]);
    const auto chunk_c0 = [&](int chunk) { return (int64_t) (r.base + chunk) * CHUNK; };
    const auto live = [&](int chunk) { return chunk < n_chunks && chunk_c0(chunk) < hi_max; };
    // pass 1: every row's max over its cells of the block's chunks
    if (live(blockIdx.x)) fetch(chunk_c0(blockIdx.x), hi_max, false);
    for (int chunk = blockIdx.x; live(chunk); chunk += gx) {
        const int64_t c0 = chunk_c0(chunk);
        commit(false);
        __syncthreads();   // K staged; the previous chunk's maxima are done with ss
        if (live(chunk + gx)) fetch(chunk_c0(chunk + gx), hi_max, false);
        if (mine) scores();
        __syncthreads();   // scores done: the tile is free for the next K
        if (mine)
            for (int i = w4; i < 16; i += 4) {
                const int64_t pos = c0 + lane;
                const float sc = ss[wg][0][i][lane] + ss[wg][1][i][lane];
                float mx = pos >= rlo[wg][i] && pos < rhi[wg][i] ? sc : -INFINITY;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                if (lane == 0) rm[wg][i] = fmaxf(rm[wg][i], mx);
            }
    }
    __syncthreads();
    // pass 2: p = 2^(s - max) and O = P V
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) wmma::fill_fragment(fo[j], 0.0f);
    if (live(blockIdx.x)) {
        fetch(chunk_c0(blockIdx.x), hi_max, false);
        fetch(chunk_c0(blockIdx.x), hi_max, true);
    }
    for (int chunk = blockIdx.x; live(chunk); chunk += gx) {
        const int64_t c0 = chunk_c0(chunk);
        const bool more = live(chunk + gx);
        commit(false);
        __syncthreads();   // K staged (every warp done with the previous V)
        if (more) fetch(chunk_c0(chunk + gx), hi_max, false);
        if (mine) scores();
        __syncthreads();   // scores done: V can take the tile
        commit(true);
        if (mine)
            for (int i = w4; i < 16; i += 4) {
                const int64_t pos = c0 + lane;
                const float sc = ss[wg][0][i][lane] + ss[wg][1][i][lane];
                const float pr = pos >= rlo[wg][i] && pos < rhi[wg][i] ? exp2f(sc - rm[wg][i]) : 0.0f;
                ps[wg][i][lane] = __float2half_rn(pr);
                const float l = warp_sum(pr);
                if (lane == 0) rl[wg][i] += l;
            }
        __syncthreads();   // V and P staged
        if (more) fetch(chunk_c0(chunk + gx), hi_max, true);
        if (mine) {
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                wmma::load_matrix_sync(fp, &ps[wg][0][kk], PLD);
#pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv8;
                    wmma::load_matrix_sync(fv8, &kv[kk][(w4 * NTW + j) * 16], LD);
                    wmma::mma_sync(fo[j], fp, fv8, fo[j]);
                }
            }
        }
        __syncthreads();   // P V done: the tile is free for the next K
    }
    // the tiles' partials, one tile at a time through the K/V tile
#pragma unroll
    for (int g = 0; g < NTB; ++g) {
        if (wg == g && mine)
#pragma unroll
            for (int j = 0; j < NTW; ++j) wmma::store_matrix_sync(&os[0][(w4 * NTW + j) * 16], fo[j], OLD, wmma::mem_row_major);
        __syncthreads();
        if (tile1 + g < tiles)
            for (int idx = tid; idx < 16 * HD; idx += NTH) {
                const int i = idx / HD, d = idx - i * HD, m = (tile1 + g) * 16 + i;
                if (m >= M) continue;
                float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
                if (d == 0) {
                    out[0] = rm[g][i];
                    out[1] = rl[g][i];
                }
                out[2 + d] = os[i][d];
            }
        __syncthreads();
    }
}

// decode_wmma3_kernel: decode_wmma2_kernel in ONE pass over the chunks (the K of each chunk read once
// instead of twice, two barriers a chunk fewer): an online softmax per row, as decode_mma_kernel does it: the running max,
// the factor alpha = 2^(m_old - m_new) on what came before (the sum, and O's accumulator elements of the row: their rows
// are Volta's WMMA layout, measured), p = 2^(s - m_new).  Another rounding than the two-pass kernels; every row still its
// own (the same steps in the same order at any width, NTB 1 or 2).
template <int HD, int NTB>
__global__ void __launch_bounds__(128 * NTB) decode_wmma3_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                                int n_chunks, float scale, float* part_all) {
    using namespace nvcuda;
    constexpr int LD = HD + 8, SLD = CHUNK + 4, PLD = CHUNK + 8, OLD = HD + 4, NTW = HD / 16 / 4, NTH = 128 * NTB;
    static_assert(CHUNK == 32 && HD % 64 == 0, "two 16-cell tiles a chunk, the head in 4 warps' tiles");
    __shared__ __align__(32) __half qs[NTB][16][LD];
    __shared__ __align__(32) __half kv[CHUNK][LD];   // the chunk's K, then (pass 2) its V; a tile's output at the end
    __shared__ __align__(32) float ss[NTB][2][16][SLD];
    __shared__ __align__(32) __half ps[NTB][16][PLD];
    __shared__ float rm[NTB][16], rl[NTB][16], ra[NTB][16];
    __shared__ int64_t rlo[NTB][16], rhi[NTB][16];
    static_assert(16 * OLD * 4 <= CHUNK * LD * 2, "a tile's output fits over the K/V tile");
    float (*os)[OLD] = reinterpret_cast<float (*)[OLD]>(&kv[0][0]);
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2, w4 = warp & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.grp0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const int tiles = segs.tile0[sk + 1] - segs.tile0[sk], tile1 = ((int) blockIdx.z - segs.grp0[sk]) * NTB;
    const bool mine = tile1 + wg < tiles;   // this warp group has a tile
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G;
    constexpr int LPT = CHUNK * HD / 8 / NTH;   // 8-value pieces a thread
    const bool i8 = q8.k != nullptr;
    uint4 fk[LPT], fv[LPT];                     // the next chunk's K and V (int8: .x/.y the codes, .z the scale's bits)
    const auto fetch = [&](int64_t c0, int64_t hi_max, bool v_too) {
        const int64_t cell0 = c0 % r.cap;
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + NTH * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            if (v_too) fv[u] = make_uint4(0, 0, 0, 0);
            else fk[u] = make_uint4(0, 0, 0, 0);
            if (c0 + cell < hi_max) {
                int64_t ce = cell0 + cell;
                if (ce >= r.cap) ce -= r.cap;
                const bool far_kv = kv_far(fr, ce);
                const int64_t row = ce * n_kv + kvh;
                if (i8) {
                    const KvInt8& c8 = far_kv ? fr.q8 : q8;
                    const int8_t* src = v_too ? c8.v : c8.k;
                    const uint16_t* sc = v_too ? c8.vs : c8.ks;
                    const int2 b = *reinterpret_cast<const int2*>(src + row * HD + d8);
                    const uint4 w = make_uint4((unsigned) b.x, (unsigned) b.y, sc[row * (HD / 64) + d8 / 64], 0);
                    if (v_too) fv[u] = w;
                    else fk[u] = w;
                } else {
                    const __half* src = v_too ? (far_kv ? (const __half*) fr.v : vc) : (far_kv ? (const __half*) fr.k : kc);
                    const uint4 w = *reinterpret_cast<const uint4*>(src + row * HD + d8);
                    if (v_too) fv[u] = w;
                    else fk[u] = w;
                }
            }
        }
    };
    const auto dec8 = [](const uint4& w) {   // decode_wmma_kernel's: the code exact in fp16, one fp16 product
        const __half sch = __ushort_as_half((unsigned short) w.z);
        const __half2 sc2 = __halves2half2(sch, sch);
        __half2 h[4];
        i8x4_to_half(w.x, h[0], h[1]);
        i8x4_to_half(w.y, h[2], h[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[e] = __hmul2(h[e], sc2);
        return *reinterpret_cast<uint4*>(h);
    };
    const auto commit = [&](bool v_too) {   // the fetched K (or V) into the tile
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int idx = tid + NTH * u, cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            const uint4 w = v_too ? fv[u] : fk[u];
            *reinterpret_cast<uint4*>(&kv[cell][d8]) = i8 ? dec8(w) : w;
        }
    };
    // S = Q K^T of the chunk for this warp group's tile: warp w4 the 16 cells 16 (w4 & 1) .. over the head's half w4 >> 1
    const auto scores = [&]() {
        const int ct = w4 & 1, half = w4 >> 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
        wmma::fill_fragment(fs, 0.0f);
#pragma unroll
        for (int k = half * (HD / 2); k < (half + 1) * (HD / 2); k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, &qs[wg][0][k], LD);
            wmma::load_matrix_sync(fb, &kv[ct * 16][k], LD);
            wmma::mma_sync(fs, fa, fb, fs);
        }
        wmma::store_matrix_sync(&ss[wg][half][0][ct * 16], fs, SLD, wmma::mem_row_major);
    };
    if (tid < NTB * 16) {
        const int g = tid >> 4, i = tid & 15, m = (tile1 + g) * 16 + i;
        const bool ok = tile1 + g < tiles && m < M;
        rlo[g][i] = ok ? row_lo(r, m / G) : 0;
        rhi[g][i] = ok ? row_hi(r, m / G) : 0;
        rm[g][i] = kNeg;
        rl[g][i] = 0.0f;
    }
    pdl_wait();
    for (int idx = tid; idx < NTB * 16 * HD; idx += NTH) {
        const int g = idx / (16 * HD), i = (idx / HD) & 15, d = idx % HD, m = (tile1 + g) * 16 + i;
        const float v = tile1 + g < tiles && m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[g][i][d] = __float2half_rn(v);
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int g = 0; g < NTB; ++g)
#pragma unroll
        for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[g][i]);
    const auto chunk_c0 = [&](int chunk) { return (int64_t) (r.base + chunk) * CHUNK; };
    const auto live = [&](int chunk) { return chunk < n_chunks && chunk_c0(chunk) < hi_max; };
    // pass 2: p = 2^(s - max) and O = P V
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) wmma::fill_fragment(fo[j], 0.0f);
    if (live(blockIdx.x)) {
        fetch(chunk_c0(blockIdx.x), hi_max, false);
        fetch(chunk_c0(blockIdx.x), hi_max, true);
    }
    for (int chunk = blockIdx.x; live(chunk); chunk += gx) {
        const int64_t c0 = chunk_c0(chunk);
        const bool more = live(chunk + gx);
        commit(false);
        __syncthreads();   // K staged (every warp done with the previous V)
        if (more) fetch(chunk_c0(chunk + gx), hi_max, false);
        if (mine) scores();
        __syncthreads();   // scores done: V can take the tile
        commit(true);
        if (mine)
            for (int i = w4; i < 16; i += 4) {   // the running max, the factor of what came before, P, the sum
                const int64_t pos = c0 + lane;
                const bool in = pos >= rlo[wg][i] && pos < rhi[wg][i];
                const float sc = ss[wg][0][i][lane] + ss[wg][1][i][lane];
                float mx = in ? sc : -INFINITY;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                const float m_old = rm[wg][i], m_new = fmaxf(m_old, mx), alpha = exp2f(m_old - m_new);
                const float pr = in ? exp2f(sc - m_new) : 0.0f;
                ps[wg][i][lane] = __float2half_rn(pr);
                const float l = warp_sum(pr);
                __syncwarp();   // every lane has read rm before lane 0 moves it
                if (lane == 0) {
                    rm[wg][i] = m_new;
                    rl[wg][i] = fmaf(rl[wg][i], alpha, l);
                    ra[wg][i] = alpha;
                }
            }
        __syncthreads();   // V, P and the factors staged
        if (more) fetch(chunk_c0(chunk + gx), hi_max, true);
        if (mine) {
            // O = O alpha + P V: a lane's accumulator elements are in rows r0 (elements 0, 1, 4, 5) and r0 + 2 (2, 3, 6, 7)
            // (Volta's WMMA 16x16x16 f32 layout, measured: dense_mmvq_bench --wmma-layout; checked by gqa_wmma_layout_ok)
            const int r0 = (lane & 1) + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1);
            const float a0 = ra[wg][r0], a2 = ra[wg][r0 + 2];
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) fo[j].x[e] *= (e & 2) ? a2 : a0;
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
                wmma::load_matrix_sync(fp, &ps[wg][0][kk], PLD);
#pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv8;
                    wmma::load_matrix_sync(fv8, &kv[kk][(w4 * NTW + j) * 16], LD);
                    wmma::mma_sync(fo[j], fp, fv8, fo[j]);
                }
            }
        }
        __syncthreads();   // P V done: the tile is free for the next K
    }
    // the tiles' partials, one tile at a time through the K/V tile
#pragma unroll
    for (int g = 0; g < NTB; ++g) {
        if (wg == g && mine)
#pragma unroll
            for (int j = 0; j < NTW; ++j) wmma::store_matrix_sync(&os[0][(w4 * NTW + j) * 16], fo[j], OLD, wmma::mem_row_major);
        __syncthreads();
        if (tile1 + g < tiles)
            for (int idx = tid; idx < 16 * HD; idx += NTH) {
                const int i = idx / HD, d = idx - i * HD, m = (tile1 + g) * 16 + i;
                if (m >= M) continue;
                float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
                if (d == 0) {
                    out[0] = rm[g][i];
                    out[1] = rl[g][i];
                }
                out[2 + d] = os[i][d];
            }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- decode on Volta's own instruction
// decode_884_kernel: mma.sync.m8n8k4 (mma884.cuh), a warp on its own: no block barrier in the sweep.  A row group = 8 rows
// of the KV group (a quad pair's rows); a warp = one row group and one quarter (sub-split) of the split's chunks, chunk
// b + gx (4 k + sub).  Per chunk of 32 cells: S = Q K^T (quad pair q the cells 8 q .. 8 q + 7; a lane's B column its own
// cell, K and V decoded as the WMMA kernels decode them: the same fp16 values), an online softmax per row (the max and the
// sum over the 32 cells by shuffles xor 2, 4, 8, in that order), P through a warp tile of shared memory into A's layout,
// O = O alpha + P V (quad pair q the dims 64 q .., B row-major: a lane four consecutive dims of one cell).  The block's 4
// sub-splits of a row group then merge in a fixed tree (1 into 0, 3 into 2, then 2 into 0) into the split's partial
// {m, l, O}: decode_merge_kernel's input.  Every row its own: the same steps at any width (RG 1, 2 or 4 row groups a
// block), another rounding than the WMMA kernels.  MEASURED (gqa_attn_parity, 12K cells): right (max abs
// 0.000394 against double, the same bits at every width and in segments) but faster than decode_wmma3_kernel at one row
// only (122 against 141 us) and slower above (5 rows 378 against 251, 16 rows 862 against 581: each row group's warps
// read K and V again, and S is one chain of HD / 4 dependent mma).  One arithmetic at every width
// forbids it at one row and the WMMA kernel above: kept OFF (BASALT_ATTN_V=4), a base for a kernel that shares K and V.
template <int HD, int RG>
__global__ void __launch_bounds__(128 * RG) decode_884_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                             int n_chunks, float scale, float* part_all) {
    constexpr int QLD = HD + 8, PLD = CHUNK + 8, NDT = HD / 32, NE = NDT * 8 + 4;
    static_assert(CHUNK == 32 && HD % 64 == 0, "a chunk = 4 quad pairs x 8 cells; dims by groups of 64");
    __shared__ __align__(16) __half qs[RG][8][QLD];
    __shared__ __align__(16) __half pt[4 * RG][8][PLD];
    __shared__ __align__(16) float slot[2][NE * 32];   // the merge: a warp's O elements, then m, l of its two rows, by lane
    __shared__ int64_t rlo_s[RG][8], rhi_s[RG][8];
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, sub = warp & 3, rgl = warp >> 2;
    const int qp = (lane >> 2) & 3, ab = mma884_ab_index(lane), hh = lane >> 4, r0 = (lane & 1) + 4 * hh;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.grp0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const int M = T * G, groups = (M + 7) / 8, rgi = ((int) blockIdx.z - segs.grp0[sk]) * RG + rgl;
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvFar fr = segs.far_kv[sk];
    const bool i8 = q8.k != nullptr;
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    if (tid < RG * 8) {
        const int g = tid >> 3, i = tid & 7, m = (((int) blockIdx.z - segs.grp0[sk]) * RG + g) * 8 + i;
        rlo_s[g][i] = m < M ? row_lo(r, m / G) : 0;
        rhi_s[g][i] = m < M ? row_hi(r, m / G) : 0;
    }
    pdl_wait();
    for (int idx = tid; idx < RG * 8 * HD; idx += 128 * RG) {
        const int g = idx / (8 * HD), i = (idx / HD) & 7, d = idx % HD;
        const int m = (((int) blockIdx.z - segs.grp0[sk]) * RG + g) * 8 + i;
        const float v = m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[g][i][d] = __float2half_rn(v);
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int g = 0; g < RG; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i) hi_max = max(hi_max, rhi_s[g][i]);
    const auto chunk_c0 = [&](int chunk) { return (int64_t) (r.base + chunk) * CHUNK; };
    const auto live = [&](int chunk) { return chunk < n_chunks && chunk_c0(chunk) < hi_max; };
    // NV consecutive values of a cell from dim d (one 64-dim scale group), as NV / 2 half2: the WMMA kernels' fp16 (int8:
    // code x scale rounded once); zero past the block's last cell
    struct Cell {
        int64_t row;
        bool far_kv, ok;
    };
    const auto cell_of = [&](int64_t pos) {
        Cell c;
        c.ok = pos < hi_max;
        int64_t ce = pos % r.cap;
        c.far_kv = kv_far(fr, ce);
        c.row = ce * n_kv + kvh;
        return c;
    };
    const auto values = [&](const Cell& c, bool v_side, int d, auto& out) {   // out: NV / 2 unsigned
        constexpr int NV = (int) (sizeof(out) / sizeof(out[0])) * 2;
        if (!c.ok) {
#pragma unroll
            for (int u = 0; u < NV / 2; ++u) out[u] = 0u;
            return;
        }
        if (i8) {
            const KvInt8& c8 = c.far_kv ? fr.q8 : q8;
            const int8_t* src = (v_side ? c8.v : c8.k) + c.row * HD + d;
            const __half sch = __ushort_as_half((v_side ? c8.vs : c8.ks)[c.row * (HD / 64) + d / 64]);
            const __half2 sc2 = __halves2half2(sch, sch);
#pragma unroll
            for (int u = 0; u < NV / 16; ++u) {
                const uint4 w = *reinterpret_cast<const uint4*>(src + 16 * u);
                const unsigned ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    __half2 h0, h1;
                    i8x4_to_half(ws[x], h0, h1);
                    out[8 * u + 2 * x] = h2_bits(__hmul2(h0, sc2));
                    out[8 * u + 2 * x + 1] = h2_bits(__hmul2(h1, sc2));
                }
            }
        } else {
            const __half* src = (v_side ? (c.far_kv ? (const __half*) fr.v : vc) : (c.far_kv ? (const __half*) fr.k : kc)) +
                                c.row * HD + d;
#pragma unroll
            for (int u = 0; u < NV / 8; ++u) {
                const uint4 w = *reinterpret_cast<const uint4*>(src + 8 * u);
                out[4 * u] = w.x;
                out[4 * u + 1] = w.y;
                out[4 * u + 2] = w.z;
                out[4 * u + 3] = w.w;
            }
        }
    };
    float o[NDT][8];
#pragma unroll
    for (int t = 0; t < NDT; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) o[t][e] = 0.0f;
    float m0 = kNeg, m2 = kNeg, l0 = 0.0f, l2 = 0.0f;
    const bool rg_ok = rgi < groups;
    const int rowa = r0, rowb = r0 + 2;   // the lane's two rows of the group (accumulator elements 0,1,4,5 and 2,3,6,7)
    const int64_t lo_a = rlo_s[rgl][rowa], hi_a = rhi_s[rgl][rowa], lo_b = rlo_s[rgl][rowb], hi_b = rhi_s[rgl][rowb];
    __half (*pw)[PLD] = pt[warp];
    for (int k = 0; rg_ok; ++k) {
        const int chunk = (int) blockIdx.x + gx * (4 * k + sub);
        if (!live(chunk)) break;
        const int64_t c0 = chunk_c0(chunk);
        // S: the lane's cell 8 qp + ab, the head's dims by groups of 64 (4 at a time into the mma)
        float sacc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            const Cell ck = cell_of(c0 + 8 * qp + ab);
#pragma unroll 1
            for (int g = 0; g < HD / 64; ++g) {
                unsigned kb[32];
                values(ck, false, 64 * g, kb);
#pragma unroll
                for (int s4 = 0; s4 < 16; ++s4) {
                    const uint2 a = *reinterpret_cast<const uint2*>(&qs[rgl][ab][64 * g + 4 * s4]);
                    mma884(sacc, a.x, a.y, kb[2 * s4], kb[2 * s4 + 1]);
                }
            }
        }
        // the online softmax of the lane's two rows over the chunk's 32 cells
        float mxa = -INFINITY, mxb = -INFINITY;
        bool in[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int64_t pos = c0 + 8 * qp + (e & 4) + (lane & 2) + (e & 1);
            in[e] = (e & 2) ? (pos >= lo_b && pos < hi_b) : (pos >= lo_a && pos < hi_a);
            if (in[e]) {
                if (e & 2) mxb = fmaxf(mxb, sacc[e]);
                else mxa = fmaxf(mxa, sacc[e]);
            }
        }
#pragma unroll
        for (int x = 2; x <= 8; x <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, x));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, x));
        }
        const float na = fmaxf(m0, mxa), nb = fmaxf(m2, mxb), aa = exp2f(m0 - na), ab2 = exp2f(m2 - nb);
        float sa = 0.0f, sb = 0.0f, pv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            pv[e] = in[e] ? exp2f(sacc[e] - ((e & 2) ? nb : na)) : 0.0f;
            if (e & 2) sb += pv[e];
            else sa += pv[e];
        }
#pragma unroll
        for (int x = 2; x <= 8; x <<= 1) {
            sa += __shfl_xor_sync(0xffffffffu, sa, x);
            sb += __shfl_xor_sync(0xffffffffu, sb, x);
        }
        m0 = na;
        m2 = nb;
        l0 = fmaf(l0, aa, sa);
        l2 = fmaf(l2, ab2, sb);
#pragma unroll
        for (int t = 0; t < NDT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) o[t][e] *= (e & 2) ? ab2 : aa;
        // P into A's layout: the warp's tile [row][cell], then a lane its row ab, 4 cells a k-step
#pragma unroll
        for (int e = 0; e < 8; ++e)
            pw[r0 + (e & 2)][8 * qp + (e & 4) + (lane & 2) + (e & 1)] = __float2half_rn(pv[e]);
        __syncwarp();
        // O += P V: k-step j = cells 4 j .. 4 j + 3; a lane's B: cell 4 j + (lane & 3), dims 64 qp + 32 hh + 4 t ..
#pragma unroll 2
        for (int j = 0; j < CHUNK / 4; ++j) {
            const uint2 a = *reinterpret_cast<const uint2*>(&pw[ab][4 * j]);
            unsigned vb[2 * NDT];   // dims (HD / 4) qp + (HD / 8) hh .. + HD / 8 of the cell (one scale group)
            values(cell_of(c0 + 4 * j + (lane & 3)), true, (HD / 4) * qp + (HD / 8) * hh, vb);
#pragma unroll
            for (int t = 0; t < NDT; ++t) mma884_rr(o[t], a.x, a.y, vb[2 * t], vb[2 * t + 1]);
        }
        __syncwarp();   // the P tile is free for the next chunk
    }
    // the 4 sub-splits of each row group, merged in a fixed tree, one row group at a time through the slots
    const auto put = [&](float* sl) {
#pragma unroll
        for (int t = 0; t < NDT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) sl[(t * 8 + e) * 32 + lane] = o[t][e];
        sl[(NE - 4) * 32 + lane] = m0;
        sl[(NE - 3) * 32 + lane] = m2;
        sl[(NE - 2) * 32 + lane] = l0;
        sl[(NE - 1) * 32 + lane] = l2;
    };
    const auto take = [&](const float* sl) {   // (m, l, O) <- ours and the slot's, each times 2^(its max - the new max)
        const float M0 = sl[(NE - 4) * 32 + lane], M2 = sl[(NE - 3) * 32 + lane];
        const float L0 = sl[(NE - 2) * 32 + lane], L2 = sl[(NE - 1) * 32 + lane];
        const float na = fmaxf(m0, M0), nb = fmaxf(m2, M2);
        const float xa = exp2f(m0 - na), ya = exp2f(M0 - na), xb = exp2f(m2 - nb), yb = exp2f(M2 - nb);
#pragma unroll
        for (int t = 0; t < NDT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float other = sl[(t * 8 + e) * 32 + lane];
                o[t][e] = (e & 2) ? fmaf(o[t][e], xb, other * yb) : fmaf(o[t][e], xa, other * ya);
            }
        l0 = fmaf(l0, xa, L0 * ya);
        l2 = fmaf(l2, xb, L2 * yb);
        m0 = na;
        m2 = nb;
    };
#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const bool here = rgl == g;
        __syncthreads();
        if (here && (sub & 1)) put(slot[sub >> 1]);
        __syncthreads();
        if (here && !(sub & 1)) take(slot[sub >> 1]);
        __syncthreads();
        if (here && sub == 2) put(slot[0]);
        __syncthreads();
        if (here && sub == 0 && rg_ok) {
            take(slot[0]);
            // the split's partial of the lane's two rows: m and l from the lanes of quad pair 0 with (lane & 2) == 0
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int mrow = rgi * 8 + r0 + 2 * half;
                if (mrow >= M) continue;
                float* out = part + (((int64_t) (mrow / G) * n_head + (int64_t) kvh * G + mrow % G) * gx + blockIdx.x) * (HD + 2);
                if (qp == 0 && (lane & 2) == 0) {
                    out[0] = half ? m2 : m0;
                    out[1] = half ? l2 : l0;
                }
#pragma unroll
                for (int t = 0; t < NDT; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        if (((e & 2) != 0) != (half != 0)) continue;
                        const int n = (e & 4) + (lane & 2) + (e & 1);
                        out[2 + (HD / 4) * qp + (HD / 8) * (n >> 2) + 4 * t + (n & 3)] = o[t][e];
                    }
            }
        }
    }
}

// decode_wmma_kernel where the card has no mma m16n8k16 (compute capability below 8.0), per device
bool volta_decode() {
    static int known[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (!known[dev]) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) cudaGetLastError();
        const char* e = std::getenv("BASALT_ATTN_WMMA");   // 0: the emulated kernel (an A/B)
        known[dev] = major > 0 && major < 8 && !(e && *e == '0') ? 1 : 2;
    }
    return known[dev] == 1;
}

// a 16x16 matrix of r 16 + c through a WMMA accumulator fragment: each lane's elements, for gqa_wmma_layout_ok
__global__ void wmma_acc_layout_kernel(float* out) {
    using namespace nvcuda;
    __shared__ float m[256];
    for (int i = threadIdx.x; i < 256; i += 32) m[i] = (float) i;
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(acc, m, 16, wmma::mem_row_major);
    for (int e = 0; e < 8 && e < acc.num_elements; ++e) out[threadIdx.x * 8 + e] = acc.x[e];
}

// the prompt kernel on Volta: prompt_wmma_kernel, the default (the 27B, 100 blocks of the quality gate: the
// same KLD, +0.000008 +/- 0.000091 row by row; reading +18 % at 16K, +33 % at 32K); BASALT_ATTN_PROMPT=i8 (read once) or
// gqa_set_prompt_wmma(2): the int8 kernel (or the fp16 one); gqa_set_prompt_wmma(1): on, 0: as the variable says
int g_prompt_wmma = 0;
bool prompt_wmma_on() {
    if (!volta_decode()) return false;
    if (g_prompt_wmma == 1 || g_prompt_wmma == 2) return g_prompt_wmma == 1;
    static const bool on = [] {
        const char* e = std::getenv("BASALT_ATTN_PROMPT");
        return !(e && std::strcmp(e, "i8") == 0);
    }();
    return on;
}

// the decode kernel on Volta: 3 = decode_wmma3_kernel (one pass, the default), 2 = decode_wmma2_kernel, 1 =
// decode_wmma_kernel (2 and 1: the same bits, two passes); BASALT_ATTN_V (read once), or gqa_set_decode_variant (the test)
int g_decode_variant = 0;
int decode_variant() {
    if (g_decode_variant >= 1 && g_decode_variant <= 4) return g_decode_variant;
    static const int v = [] {
        const char* e = std::getenv("BASALT_ATTN_V");
        return e && (*e == '1' || *e == '2' || *e == '4') ? *e - '0' : 3;
    }();
    return v;
}

// segs.n and segs.row0 set: the tiles from the rows, then both kernels over every segment
template <int HD>
void launch_segs(const float* q, int n_head, int n_kv, DecSegs& segs, float* scratch, int n_chunks, float* out, cudaStream_t s,
                 const float* gate_q = nullptr, uint16_t* gated16 = nullptr) {
    const float scale = 1.4426950408889634f / sqrtf((float) HD);
    const int gx = decode_splits(n_chunks);
    for (int k = 0; k < segs.n; ++k)
        segs.tile0[k + 1] = segs.tile0[k] + ((segs.row0[k + 1] - segs.row0[k]) * (n_head / n_kv) + 15) / 16;
    // decode_wmma2_kernel where it pays (gqa_attn_parity, 12K cells: 5 rows 333 against 356 us, 16 rows 762
    // against 825; but 1 row 180 against 173 and 8 rows 501 against 452: a tile alone, or a group of two tiles whose second
    // is missing, waits on its extra barriers): every segment with an even number of tiles.  The same bits either way.
    bool pairs = true;
    for (int k = 0; k < segs.n; ++k) pairs = pairs && (segs.tile0[k + 1] - segs.tile0[k]) % 2 == 0;
    if (volta_decode() && decode_variant() == 4) {   // Volta's own instruction: row groups of 8, 1 to 4 a block
        int most = 1;
        for (int k = 0; k < segs.n; ++k) most = std::max(most, ((segs.row0[k + 1] - segs.row0[k]) * (n_head / n_kv) + 7) / 8);
        const int rg = most >= 4 ? 4 : most >= 2 ? 2 : 1;
        segs.grp0[0] = 0;
        for (int k = 0; k < segs.n; ++k)
            segs.grp0[k + 1] = segs.grp0[k] + (((segs.row0[k + 1] - segs.row0[k]) * (n_head / n_kv) + 7) / 8 + rg - 1) / rg;
        const dim3 grid((unsigned) gx, (unsigned) n_kv, (unsigned) segs.grp0[segs.n]);
        if (rg == 4) launch_pdl(decode_884_kernel<HD, 4>, grid, dim3(512), 0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
        else if (rg == 2) launch_pdl(decode_884_kernel<HD, 2>, grid, dim3(256), 0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
        else launch_pdl(decode_884_kernel<HD, 1>, grid, dim3(128), 0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
    } else if (volta_decode() && decode_variant() == 3) {   // one pass: two tiles a block where every segment pairs its tiles
        const int ntb = pairs ? 2 : 1;
        segs.grp0[0] = 0;
        for (int k = 0; k < segs.n; ++k) segs.grp0[k + 1] = segs.grp0[k] + (segs.tile0[k + 1] - segs.tile0[k] + ntb - 1) / ntb;
        const dim3 grid((unsigned) gx, (unsigned) n_kv, (unsigned) segs.grp0[segs.n]);
        if (ntb == 2)
            launch_pdl(decode_wmma3_kernel<HD, 2>, grid, dim3(256), 0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
        else
            launch_pdl(decode_wmma3_kernel<HD, 1>, grid, dim3(128), 0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
    } else if (volta_decode() && decode_variant() == 2 && pairs) {
        segs.grp0[0] = 0;
        for (int k = 0; k < segs.n; ++k) segs.grp0[k + 1] = segs.grp0[k] + (segs.tile0[k + 1] - segs.tile0[k]) / 2;
        launch_pdl(decode_wmma2_kernel<HD, 2>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.grp0[segs.n]), dim3(256), 0, s,
                   q, n_head, n_kv, segs, n_chunks, scale, scratch);
    } else if (volta_decode())
        launch_pdl(decode_wmma_kernel<HD>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.tile0[segs.n]), dim3(128), 0, s,
                   q, n_head, n_kv, segs, n_chunks, scale, scratch);
    else
        launch_pdl(decode_mma_kernel<HD>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.tile0[segs.n]), dim3(128), 0, s,
                   q, n_head, n_kv, segs, n_chunks, scale, scratch);
    check("gqa decode attention");
    launch_pdl(decode_merge_kernel<HD>, dim3((unsigned) n_head, (unsigned) segs.row0[segs.n]), dim3(HD), 0, s,
               (const float*) scratch, n_head, gx, out, gate_q, gated16, segs);
    check("gqa decode attention/merge");
}

template <int HD>
void launch_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_kv, const Range& r, int T,
                   float* scratch, int n_chunks, float* out, cudaStream_t s, KvInt8 q8 = {}, const float* gate_q = nullptr,
                   uint16_t* gated16 = nullptr, KvFar far_kv = {}) {
    DecSegs segs;
    segs.row0[1] = T;
    segs.kc[0] = (const __half*) kc;
    segs.vc[0] = (const __half*) vc;
    segs.q8[0] = q8;
    segs.far_kv[0] = far_kv;
    segs.r[0] = r;
    launch_segs<HD>(q, n_head, n_kv, segs, scratch, n_chunks, out, s, gate_q, gated16);
}

__device__ __forceinline__ void mma16832_u8s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1) {
    basalt::volta::mma_16832_s32_s8<true>(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    basalt::volta::cp_async_cg16(smem_u32(dst), src, ok ? 16 : 0);
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    basalt::volta::cp_async_ca<8>(smem_u32(dst), src, ok ? 8 : 0);
}
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ float ex2(float x) {   // 2^x, -inf -> 0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---------------------------------------------------------------- prompt, fp16 cache
// Block (query tile, KV head): its 96 rows are the tile's positions times the KV group's query heads (head fastest), so
// a K/V tile is read once for every head of the group.  Warp w owns rows 16 w .. 16 w + 15.  Cells stream in tiles of
// 32.  QK^T on fp16 tensor cores with fp32 accumulation; P V on fp16 tensor cores accumulating each 32-cell tile in
// fp16, added to the fp32 output.  Online softmax in fp32.
constexpr int PNW = 6, PROWS = 16 * PNW, PBN = 32;
constexpr int LDH = HD + 8;    // padded fp16 row (halves)

struct PromptSmem16 {
    __half q[PROWS][LDH];
    __half k[PBN][LDH];
    __half v[PBN][LDH];
};

__global__ void __launch_bounds__(PNW * 32) prompt_f16_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                              const __half* __restrict__ vc, int n_head, int n_kv,
                                                              int64_t pos0, int64_t T, float* __restrict__ out, KvFar far_kv) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PromptSmem16& sm = *reinterpret_cast<PromptSmem16*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = n_head / n_kv, PB = PROWS / G;   // query heads per KV head, positions per tile
    const int kvh = blockIdx.y;
    const int64_t q0 = (int64_t) (gridDim.x - 1 - blockIdx.x) * PB;   // the longest tiles first
    const int64_t last_pos = min(T, q0 + PB) - 1;
    const int64_t cells = pos0 + last_pos + 1;
    const int g = lane >> 2, tq = lane & 3;
    const int ra = warp * 16 + g, rb = ra + 8;                         // this thread's two rows
    const int64_t pos_a = q0 + ra / G, pos_b = q0 + rb / G;
    const int head_a = kvh * G + ra % G, head_b = kvh * G + rb % G;

    for (int i = tid; i < PROWS * HD / 4; i += PNW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        const int64_t pos = q0 + r / G;
        const int head = kvh * G + r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pos < T) v = *reinterpret_cast<const float4*>(q + (pos * n_head + head) * HD + c);
        *reinterpret_cast<__half2*>(&sm.q[r][c]) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(&sm.q[r][c + 2]) = __floats2half2_rn(v.z, v.w);
    }

    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int n_tiles = (int) ((cells + PBN - 1) / PBN);

    for (int ti = 0; ti < n_tiles; ++ti) {
        const int64_t c0 = (int64_t) ti * PBN;
        __syncthreads();   // the previous tile is consumed (and the query tile is written)
        for (int i = tid; i < PBN * HD / 8; i += PNW * 32) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = c0 + r < cells;
            int64_t cell = ok ? c0 + r : 0;
            const bool fa = kv_far(far_kv, cell);
            const int64_t row = cell * n_kv + kvh;
            cp16(&sm.k[r][c], (fa ? (const __half*) far_kv.k : kc) + row * HD + c, ok);
            cp16(&sm.v[r][c], (fa ? (const __half*) far_kv.v : vc) + row * HD + c, ok);
        }
        (basalt::volta::cp_async_commit(), basalt::volta::cp_async_wait<0>());
        __syncthreads();

        // S = Q K^T for the warp's 16 rows and the tile's 32 cells (four 8-cell tiles)
        float s[PBN / 8][4];
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < HD; k += 16) {
            unsigned a0, a1, a2, a3;
            ldsm_x4(a0, a1, a2, a3, &sm.q[warp * 16 + (lane & 15)][k + (lane >> 4) * 8]);
#pragma unroll
            for (int n = 0; n < PBN / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sm.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], a0, a1, a2, a3, b0, b1);
                mma16816(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        // mask and online softmax: thread holds rows ra (s[n][0..1]) and rb (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t cell = c0 + n * 8 + tq * 2 + (i & 1);
                const int64_t pos = i < 2 ? pos_a : pos_b;
                const bool vis = cell <= pos0 + pos && cell < cells;
                s[n][i] = vis ? s[n][i] * kScaleLog2 : kNeg;
                if (i < 2) mx_a = fmaxf(mx_a, s[n][i]);
                else mx_b = fmaxf(mx_b, s[n][i]);
            }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float sc_a = exp2f(m_a - mx_a), sc_b = exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) {
            s[n][0] = s[n][0] <= kNeg ? 0.0f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] <= kNeg ? 0.0f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] <= kNeg ? 0.0f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] <= kNeg ? 0.0f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * sc_a + ps_a;
        l_b = l_b * sc_b + ps_b;
        // O = O * scale + P V, the tile's P V summed in fp16 on the tensor cores
        unsigned pa[PBN / 16][4];
#pragma unroll
        for (int j = 0; j < PBN / 16; ++j) {
            pa[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pa[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pa[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pa[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < HD / 8; n += 2) {
            unsigned d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < PBN / 16; ++j) {
                unsigned b0, b1, b2, b3;
                // V rows are cells (k), columns dims (n): transposed loads give the col-major B of dims n*8.., n*8+8..
                ldsm_x4_t(b0, b1, b2, b3, &sm.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816_h(d0, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b0, b1);
                mma16816_h(d1, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b2, b3);
            }
            const float2 e0 = __half22float2(*reinterpret_cast<const __half2*>(&d0[0]));
            const float2 e1 = __half22float2(*reinterpret_cast<const __half2*>(&d0[1]));
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d1[0]));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1[1]));
            o[n][0] = fmaf(o[n][0], sc_a, e0.x); o[n][1] = fmaf(o[n][1], sc_a, e0.y);
            o[n][2] = fmaf(o[n][2], sc_b, e1.x); o[n][3] = fmaf(o[n][3], sc_b, e1.y);
            o[n + 1][0] = fmaf(o[n + 1][0], sc_a, f0.x); o[n + 1][1] = fmaf(o[n + 1][1], sc_a, f0.y);
            o[n + 1][2] = fmaf(o[n + 1][2], sc_b, f1.x); o[n + 1][3] = fmaf(o[n + 1][3], sc_b, f1.y);
        }
    }
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int col = n * 8 + tq * 2;
        if (pos_a < T) *reinterpret_cast<float2*>(out + (pos_a * n_head + head_a) * HD + col) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (pos_b < T) *reinterpret_cast<float2*>(out + (pos_b * n_head + head_b) * HD + col) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

// ---------------------------------------------------------------- prompt on Volta's tensor cores
// prompt_wmma_kernel: prompt_f16_kernel's tiling (block = 16 positions x the KV group's 6 heads, 96 rows; warp w rows
// 16 w ..; cells by tiles of 32, read once for the block) on nvcuda::wmma 16x16x16 (fp16 in, fp32 accumulate: the
// V100's own tensor cores, where the m16n8 mma and ldmatrix of both prompt kernels are emulated), for the fp16 AND the
// int8 cache (K and V decoded to fp16 as the decode kernels decode them: code x scale rounded once).  Q x 1/sqrt(HD)
// log2(e) in fp16 (the decode kernels' queries), S kept in two accumulator fragments, the online softmax of a row over
// its 4 lanes (Volta's accumulator layout, measured: a lane holds rows r0 and r0 + 2; the 4 lanes of a row differ in
// bits 1 and 3), P through a warp tile into A's layout, O = O alpha + P V accumulated in fp32 (the fp16 kernel sums a
// tile's P V in fp16).  Another rounding than the int8 prompt kernel (which rounds Q and P to 8 bits): the decode path's
// kind of arithmetic.  The default on Volta (prompt_wmma_on: BASALT_ATTN_PROMPT=i8 for the int8 kernel).
constexpr int WNW = 6, WROWS = 16 * WNW, WBN = 32;
struct PromptSmemW {
    __half q[WROWS][LDH];
    __half k[WBN][LDH];
    __half v[WBN][LDH];
    __half p[WNW][16][WBN + 8];
};

__global__ void __launch_bounds__(WNW * 32, 1) prompt_wmma_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                                 const __half* __restrict__ vc, KvInt8 q8, int n_head, int n_kv,
                                                                 int64_t pos0, int64_t T, float* __restrict__ out, KvFar far_kv) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    PromptSmemW& sm = *reinterpret_cast<PromptSmemW*>(smem_raw);
    constexpr int NDF = HD / 16;   // the warp's 16-dim output fragments
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = n_head / n_kv, PB = WROWS / G;
    const int kvh = blockIdx.y;
    const int64_t q0 = (int64_t) (gridDim.x - 1 - blockIdx.x) * PB;   // the longest tiles first
    const int64_t last_pos = min(T, q0 + PB) - 1;
    const int64_t cells = pos0 + last_pos + 1;
    const bool i8 = q8.k != nullptr;
    const float scale = kScaleLog2;
    for (int i = tid; i < WROWS * HD / 4; i += WNW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        const int64_t pos = q0 + r / G;
        const int head = kvh * G + r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pos < T) v = *reinterpret_cast<const float4*>(q + (pos * n_head + head) * HD + c);
        *reinterpret_cast<__half2*>(&sm.q[r][c]) = __floats2half2_rn(v.x * scale, v.y * scale);
        *reinterpret_cast<__half2*>(&sm.q[r][c + 2]) = __floats2half2_rn(v.z * scale, v.w * scale);
    }
    // the lane's two rows of the warp's 16 (accumulator elements 0, 1, 4, 5 and 2, 3, 6, 7) and their positions
    const int r0 = (lane & 1) + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1);
    const int ra = warp * 16 + r0, rb = ra + 2;
    const int64_t pos_a = q0 + ra / G, pos_b = q0 + rb / G;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[NDF];
#pragma unroll
    for (int j = 0; j < NDF; ++j) wmma::fill_fragment(fo[j], 0.0f);
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int n_tiles = (int) ((cells + WBN - 1) / WBN);
    for (int ti = 0; ti < n_tiles; ++ti) {
        const int64_t c0 = (int64_t) ti * WBN;
        __syncthreads();   // the previous tile is consumed (and the queries are written)
        for (int i = tid; i < WBN * HD / 8; i += WNW * 32) {   // 8 values a step, fp16 in shared memory
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = c0 + r < cells;
            uint4 kw = make_uint4(0, 0, 0, 0), vw = kw;
            if (ok) {
                int64_t cell = c0 + r;
                const bool fa = kv_far(far_kv, cell);
                const int64_t row = cell * n_kv + kvh;
                if (i8) {
                    const KvInt8& c8 = fa ? far_kv.q8 : q8;
                    const int2 kb = *reinterpret_cast<const int2*>(c8.k + row * HD + c);
                    const int2 vb = *reinterpret_cast<const int2*>(c8.v + row * HD + c);
                    const __half ksh = __ushort_as_half(c8.ks[row * (HD / 64) + c / 64]);
                    const __half vsh = __ushort_as_half(c8.vs[row * (HD / 64) + c / 64]);
                    const __half2 ks2 = __halves2half2(ksh, ksh), vs2 = __halves2half2(vsh, vsh);
                    __half2 h[4];
                    i8x4_to_half((unsigned) kb.x, h[0], h[1]);
                    i8x4_to_half((unsigned) kb.y, h[2], h[3]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) h[e] = __hmul2(h[e], ks2);
                    kw = *reinterpret_cast<uint4*>(h);
                    i8x4_to_half((unsigned) vb.x, h[0], h[1]);
                    i8x4_to_half((unsigned) vb.y, h[2], h[3]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) h[e] = __hmul2(h[e], vs2);
                    vw = *reinterpret_cast<uint4*>(h);
                } else {
                    kw = *reinterpret_cast<const uint4*>((fa ? (const __half*) far_kv.k : kc) + row * HD + c);
                    vw = *reinterpret_cast<const uint4*>((fa ? (const __half*) far_kv.v : vc) + row * HD + c);
                }
            }
            *reinterpret_cast<uint4*>(&sm.k[r][c]) = kw;
            *reinterpret_cast<uint4*>(&sm.v[r][c]) = vw;
        }
        __syncthreads();
        if ((int64_t) warp * 16 / G + q0 >= T) continue;   // the warp's rows are past the chunk (it keeps the barriers)
        // S = Q K^T: the warp's 16 rows, cells 0..15 and 16..31
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[2];
        wmma::fill_fragment(fs[0], 0.0f);
        wmma::fill_fragment(fs[1], 0.0f);
#pragma unroll
        for (int k = 0; k < HD; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &sm.q[warp * 16][k], LDH);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &sm.k[ct * 16][k], LDH);
                wmma::mma_sync(fs[ct], fa, fb, fs[ct]);
            }
        }
        // the mask, then the online softmax: element e of fragment ct is row r0 + (e & 2), cell
        // 16 ct + 8 ((lane >> 3) & 1) + 4 (e >> 2) + (lane & 2) + (e & 1)
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int64_t cell = c0 + 16 * ct + 8 * ((lane >> 3) & 1) + 4 * (e >> 2) + (lane & 2) + (e & 1);
                const int64_t pos = (e & 2) ? pos_b : pos_a;
                const bool vis = cell <= pos0 + pos && cell < cells;
                fs[ct].x[e] = vis ? fs[ct].x[e] : -INFINITY;
                if (e & 2) mx_b = fmaxf(mx_b, fs[ct].x[e]);
                else mx_a = fmaxf(mx_a, fs[ct].x[e]);
            }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 8));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 8));
        const float na = fmaxf(m_a, mx_a), nb = fmaxf(m_b, mx_b);
        const float al_a = exp2f(m_a - na), al_b = exp2f(m_b - nb);
        m_a = na;
        m_b = nb;
        float ps_a = 0.0f, ps_b = 0.0f;
        __half (*pw)[WBN + 8] = sm.p[warp];
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float pr = fs[ct].x[e] == -INFINITY ? 0.0f : exp2f(fs[ct].x[e] - ((e & 2) ? m_b : m_a));
                if (e & 2) ps_b += pr;
                else ps_a += pr;
                pw[r0 + (e & 2)][16 * ct + 8 * ((lane >> 3) & 1) + 4 * (e >> 2) + (lane & 2) + (e & 1)] = __float2half_rn(pr);
            }
        ps_a += __shfl_xor_sync(0xffffffffu, ps_a, 2);
        ps_a += __shfl_xor_sync(0xffffffffu, ps_a, 8);
        ps_b += __shfl_xor_sync(0xffffffffu, ps_b, 2);
        ps_b += __shfl_xor_sync(0xffffffffu, ps_b, 8);
        l_a = fmaf(l_a, al_a, ps_a);
        l_b = fmaf(l_b, al_b, ps_b);
        __syncwarp();
        // O = O alpha + P V
#pragma unroll
        for (int j = 0; j < NDF; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) fo[j].x[e] *= (e & 2) ? al_b : al_a;
#pragma unroll
        for (int kk = 0; kk < WBN; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, &pw[0][kk], WBN + 8);
#pragma unroll
            for (int j = 0; j < NDF; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, &sm.v[kk][16 * j], LDH);
                wmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
        __syncwarp();   // the P tile is free for the next tile
    }
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
    const int head_a = kvh * G + ra % G, head_b = kvh * G + rb % G;
#pragma unroll
    for (int j = 0; j < NDF; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {   // element e: row r0 + (e & 2), dim 16 j + 8 ((lane >> 3) & 1) + 4 (e >> 2) + (lane & 2) + (e & 1)
            const int d = 16 * j + 8 * ((lane >> 3) & 1) + 4 * (e >> 2) + (lane & 2) + (e & 1);
            if (e & 2) {
                if (pos_b < T) out[(pos_b * n_head + head_b) * HD + d] = fo[j].x[e] * ib;
            } else {
                if (pos_a < T) out[(pos_a * n_head + head_a) * HD + d] = fo[j].x[e] * ia;
            }
        }
}

// ---------------------------------------------------------------- prompt, int8 cache
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.  The structure follows
// Infernix's fast int8 prompt kernel (src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh); Basalt's
// cache is unrotated and unpaged, so the queries are rounded to int8 per 64 dims directly and a key tile is gathered
// cell by cell.
//
// Block (query tile, query head), longest tiles first: warp w owns the tile's positions 16 w .. 16 w + 15 of one head
// for the whole sweep, so scores, probabilities and the 256-wide output stay in registers and the only block barrier
// publishes a key tile.  Keys stream in tiles of 64 cells, K and V double-buffered as raw int8 codes with their scales
// (16-byte pieces XOR-swizzled by the row).  QK^T: int8 tensor cores, each 64-dim group's integer sum times the query's
// and the cell's scales in fp32.  P V in 8 bits (Infernix's prompt-pv8): per 64-dim V group, each probability times
// its cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum); u8 x s8
// tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the fp32 output.
// Online softmax in fp32.
//
// A split launch (gridDim.z > 1) gives each block a contiguous run of its key tiles; the block writes its rows
// normalized by its own sum with their (max, sum) and prompt_i8_merge_kernel combines the splits.
constexpr int I8BC = 64;                                   // cells per key tile
constexpr int I8_TILE = I8BC * HD;                         // bytes of one tile's K (or V) codes
constexpr int I8_STAGE = 2 * I8_TILE + 2 * I8BC * 4 * 2;   // K, V codes and their fp16 scales

template <int W> constexpr int i8_smem() { return W * 16 * HD + 2 * I8_STAGE; }

// IO: the queries and the output rows in FP32 or BF16 (the prompt chunk's)
__device__ __forceinline__ float io_ld(const float* p) { return *p; }
__device__ __forceinline__ float io_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void io_st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void io_st4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
}
__device__ __forceinline__ void io_st1(float* p, float v) { *p = v; }
__device__ __forceinline__ void io_st1(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <int W, bool Split, class IO = float>
__global__ void __launch_bounds__(W * 32, 1) prompt_i8_kernel(const IO* __restrict__ q, int n_head, int n_kv, int pos0,
                                                              int T, IO* __restrict__ out, KvInt8 q8,
                                                              float* __restrict__ part_rows, float2* __restrict__ part_stats,
                                                              KvFar far_kv) {
    constexpr int Threads = W * 32, Br = W * 16;
    constexpr int G4 = HD / 64;           // scale groups per row
    constexpr int NT = I8BC / 8;          // score n8 tiles per key tile
    constexpr int KS = I8BC / 16;         // P V k-steps per key tile
    constexpr int DB = HD / 16;           // 16-dim output blocks
    static_assert(I8_TILE / 16 % Threads == 0 && 2 * I8BC <= Threads, "int8 prompt attention: tile copy split");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* q_s = reinterpret_cast<int8_t*>(smem_raw);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;   // the first n_head blocks take every head's last tile
    const int head = linear % n_head;
    const int q0 = ((int) gridDim.x - 1 - linear / n_head) * Br;
    const int kvh = head / (n_head / n_kv);
    const int rows = min(Br, T - q0);
    const int max_qabs = pos0 + q0 + rows - 1;
    const int key_tiles = max_qabs / I8BC + 1;
    const int split = Split ? (int) blockIdx.z : 0;
    const int per_split = Split ? (key_tiles + (int) gridDim.z - 1) / (int) gridDim.z : 0;
    const int kb_begin = Split ? min(key_tiles, split * per_split) : 0;
    const int kb_end = Split ? min(key_tiles, kb_begin + per_split) : key_tiles;

    // each warp rounds its own 16 rows to int8 per 64 dims (amax / 127); a thread keeps the scales of its two C rows
    float qsr[2][G4];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = warp * 16 + r;
        const bool valid = row < rows;
        const IO* src = q + ((int64_t) (q0 + row) * n_head + head) * HD;
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = valid ? io_ld(src + lane + 32 * k) : 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            const float amax = warp_max(fmaxf(fabsf(x[2 * grp]), fabsf(x[2 * grp + 1])));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = grp * 64 + h * 32 + lane;
                const int c = max(-127, min(127, __float2int_rn(x[2 * grp + h] * inv)));
                q_s[row * HD + (((d >> 4) ^ (row & 7)) << 4) + (d & 15)] = (int8_t) c;
            }
            if (g == (r & 7)) qsr[r >> 3][grp] = amax / 127.0f;
        }
    }

    // a key tile: 256-byte K and V rows per cell (the cache interleaves the KV heads), cells past the block's last
    // visible one zero-filled so masked columns stay finite
    const auto issue_tile = [&](int kb) {
        unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        int8_t* k_s = reinterpret_cast<int8_t*>(base);
        int8_t* v_s = k_s + I8_TILE;
        uint16_t* ks_s = reinterpret_cast<uint16_t*>(v_s + I8_TILE);
        uint16_t* vs_s = ks_s + I8BC * G4;
        const int valid = min(I8BC, max_qabs + 1 - kb * I8BC);
#pragma unroll
        for (int i = 0; i < I8_TILE / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key = chunk >> 4, c = chunk & 15;
            const bool ok = key < valid;
            int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const KvInt8& c8 = kv_far(far_kv, cell) ? far_kv.q8 : q8;
            const int64_t src = (cell * n_kv + kvh) * HD + c * 16;
            const int dst = key * HD + ((c ^ (key & 7)) << 4);
            cp16(k_s + dst, c8.k + src, ok);
            cp16(v_s + dst, c8.v + src, ok);
        }
        if (tid < 2 * I8BC) {
            const int key = tid & (I8BC - 1);
            const bool ok = key < valid;
            int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const KvInt8& c8 = kv_far(far_kv, cell) ? far_kv.q8 : q8;
            const int64_t src = (cell * n_kv + kvh) * G4;
            if (tid < I8BC) cp8(ks_s + key * G4, c8.ks + src, ok);
            else cp8(vs_s + key * G4, c8.vs + src, ok);
        }
        basalt::volta::cp_async_commit();
    };

    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = warp * 16 + a_rin + ((a_mat & 1) << 3);
    const int a_chunk = a_mat >> 1;        // 16-byte piece within a 32-byte k step
    const int b_chunk = (lane >> 3) & 1;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool warp_active = warp * 16 < rows;
    const int warp_min_qabs = pos0 + q0 + warp * 16;
    const int warp_max_qabs = pos0 + q0 + min(warp * 16 + 15, rows - 1);
    const int qabs0 = row0 < rows ? pos0 + q0 + row0 : -1;
    const int qabs1 = row1 < rows ? pos0 + q0 + row1 : -1;

    float acc[DB][2][4];
#pragma unroll
    for (int b = 0; b < DB; ++b)
#pragma unroll
        for (int p = 0; p < 2; ++p) acc[b][p][0] = acc[b][p][1] = acc[b][p][2] = acc[b][p][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (kb_begin < kb_end) issue_tile(kb_begin);
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        basalt::volta::cp_async_wait<0>();
        __syncthreads();   // tile kb is in, and every warp is done with the other stage
        if (kb + 1 < kb_end) issue_tile(kb + 1);
        const int k0 = kb * I8BC;
        if (!warp_active || k0 > warp_max_qabs) continue;
        const unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        const int8_t* k_s = reinterpret_cast<const int8_t*>(base);
        const int8_t* v_s = k_s + I8_TILE;
        const uint16_t* ks_s = reinterpret_cast<const uint16_t*>(v_s + I8_TILE);
        const uint16_t* vs_s = ks_s + I8BC * G4;

        // S = Q K^T: 16 rows x 64 cells
        float s[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            unsigned af[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int ch = (grp * 2 + kk) * 2 + a_chunk;
                ldsm_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3], q_s + a_row * HD + ((ch ^ (a_row & 7)) << 4));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int c[4] = {0, 0, 0, 0};
                const int key = n * 8 + a_rin;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int ch = (grp * 2 + kk) * 2 + b_chunk;
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, k_s + key * HD + ((ch ^ (key & 7)) << 4));
                    mma16832_s8(c, af[kk][0], af[kk][1], af[kk][2], af[kk][3], b0, b1);
                }
                const int ka = n * 8 + 2 * tq;
                const float ks0 = h2f(ks_s[ka * G4 + grp]), ks1 = h2f(ks_s[(ka + 1) * G4 + grp]);
                s[n][0] = fmaf(qsr[0][grp] * ks0, (float) c[0], s[n][0]);
                s[n][1] = fmaf(qsr[0][grp] * ks1, (float) c[1], s[n][1]);
                s[n][2] = fmaf(qsr[1][grp] * ks0, (float) c[2], s[n][2]);
                s[n][3] = fmaf(qsr[1][grp] * ks1, (float) c[3], s[n][3]);
            }
        }

        // mask (only tiles crossing the warp's diagonal) and online softmax, scores kept unscaled
        const bool full = k0 + I8BC - 1 <= warp_min_qabs;
        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (!full) {
                const int c0 = k0 + n * 8 + 2 * tq, c1 = c0 + 1;
                s[n][0] = c0 <= qabs0 ? s[n][0] : -INFINITY;
                s[n][1] = c1 <= qabs0 ? s[n][1] : -INFINITY;
                s[n][2] = c0 <= qabs1 ? s[n][2] : -INFINITY;
                s[n][3] = c1 <= qabs1 ? s[n][3] : -INFINITY;
            }
            bm0 = fmaxf(bm0, fmaxf(s[n][0], s[n][1]));
            bm1 = fmaxf(bm1, fmaxf(s[n][2], s[n][3]));
        }
        const float nm0 = fmaxf(m0, quad_max(bm0)), nm1 = fmaxf(m1, quad_max(bm1));
        const float ns0 = nm0 == -INFINITY ? 0.0f : nm0 * kScaleLog2;
        const float ns1 = nm1 == -INFINITY ? 0.0f : nm1 * kScaleLog2;
        const float al0 = m0 == -INFINITY ? 0.0f : ex2(m0 * kScaleLog2 - ns0);
        const float al1 = m1 == -INFINITY ? 0.0f : ex2(m1 * kScaleLog2 - ns1);
        m0 = nm0;
        m1 = nm1;
        // the probabilities become the P V A fragments: k step j covers cells 16 j .. 16 j + 15 (score tiles 2 j, 2 j + 1)
        unsigned pa[KS][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float p00 = ex2(fmaf(s[n][0], kScaleLog2, -ns0)), p01 = ex2(fmaf(s[n][1], kScaleLog2, -ns0));
            const float p10 = ex2(fmaf(s[n][2], kScaleLog2, -ns1)), p11 = ex2(fmaf(s[n][3], kScaleLog2, -ns1));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[n >> 1][(n & 1) * 2] = pack_h2(p00, p01);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_h2(p10, p11);
        }
        l0 = fmaf(l0, al0, bl0);   // lane-partial; the quad is summed after the sweep
        l1 = fmaf(l1, al1, bl1);

        // 8-bit P V (Infernix's prompt-pv8, fast_tiled_mma.cuh): per 64-dim V group, each probability times its
        // cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum);
        // u8 x s8 tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the
        // fp32 output.  A k32 step takes the lane's own cells {2t, 2t+1, 8+2t, 9+2t} (+16), as pa holds them.
        const __half2 origin = __float2half2_rn(1024.0f);
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            __half2 prod[KS][4];
            __half2 mx0 = __float2half2_rn(0.0f), mx1 = mx0;
#pragma unroll
            for (int j = 0; j < KS; ++j) {
                const int key = j * 16 + 2 * tq;
                const __half2 lo = __halves2half2(__ushort_as_half(vs_s[key * G4 + grp]), __ushort_as_half(vs_s[(key + 1) * G4 + grp]));
                const __half2 hi = __halves2half2(__ushort_as_half(vs_s[(key + 8) * G4 + grp]),
                                                  __ushort_as_half(vs_s[(key + 9) * G4 + grp]));
                prod[j][0] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][0]), lo);
                prod[j][1] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][1]), lo);
                prod[j][2] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][2]), hi);
                prod[j][3] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][3]), hi);
                mx0 = __hmax2(mx0, __hmax2(prod[j][0], prod[j][2]));
                mx1 = __hmax2(mx1, __hmax2(prod[j][1], prod[j][3]));
            }
            const float rm0 = quad_max(fmaxf(__low2float(mx0), __high2float(mx0)));
            const float rm1 = quad_max(fmaxf(__low2float(mx1), __high2float(mx1)));
            // codes rint(product * inverse), the fp16 inverse rounded down so none passes 255; the step is its reciprocal
            const __half inv0 = __float2half_rd(rm0 > 0.0f ? 255.0f / rm0 : 0.0f);
            const __half inv1 = __float2half_rd(rm1 > 0.0f ? 255.0f / rm1 : 0.0f);
            const float step0 = __hgt(inv0, __float2half(0.0f)) ? 1.0f / __half2float(inv0) : 0.0f;
            const float step1 = __hgt(inv1, __float2half(0.0f)) ? 1.0f / __half2float(inv1) : 0.0f;
            const __half2 i02 = __half2half2(inv0), i12 = __half2half2(inv1);
            unsigned a8[2][4];   // fp16 1024 + n holds n in its low byte
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                unsigned qv[2][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * st + h;
                    const __half2 c0 = __hfma2(prod[j][0], i02, origin), c1 = __hfma2(prod[j][1], i12, origin);
                    const __half2 c2 = __hfma2(prod[j][2], i02, origin), c3 = __hfma2(prod[j][3], i12, origin);
                    qv[h][0] = *reinterpret_cast<const unsigned*>(&c0);
                    qv[h][1] = *reinterpret_cast<const unsigned*>(&c1);
                    qv[h][2] = *reinterpret_cast<const unsigned*>(&c2);
                    qv[h][3] = *reinterpret_cast<const unsigned*>(&c3);
                }
                a8[st][0] = __byte_perm(qv[0][0], qv[0][2], 0x6420);
                a8[st][1] = __byte_perm(qv[0][1], qv[0][3], 0x6420);
                a8[st][2] = __byte_perm(qv[1][0], qv[1][2], 0x6420);
                a8[st][3] = __byte_perm(qv[1][1], qv[1][3], 0x6420);
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int db = grp * 4 + b;
                int ev[4] = {0, 0, 0, 0}, od[4] = {0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    // matrices: cells 32 st + 0..7, 8..15, 16..23, 24..31 of d block db; lane (g, t) gets cells
                    // {2t, 2t+1} at dims {2g, 2g+1}
                    const int key = st * 32 + (a_mat << 3) + a_rin;
                    unsigned r[4];
                    ldsm_x4_t(r[0], r[1], r[2], r[3], v_s + key * HD + ((db ^ (key & 7)) << 4));
                    mma16832_u8s8(ev, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x6420),
                                  __byte_perm(r[2], r[3], 0x6420));
                    mma16832_u8s8(od, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x7531),
                                  __byte_perm(r[2], r[3], 0x7531));
                }
                float (&e)[4] = acc[db][0];
                float (&o)[4] = acc[db][1];
                e[0] = fmaf(e[0], al0, step0 * (float) ev[0]);
                e[1] = fmaf(e[1], al0, step0 * (float) ev[1]);
                e[2] = fmaf(e[2], al1, step1 * (float) ev[2]);
                e[3] = fmaf(e[3], al1, step1 * (float) ev[3]);
                o[0] = fmaf(o[0], al0, step0 * (float) od[0]);
                o[1] = fmaf(o[1], al0, step0 * (float) od[1]);
                o[2] = fmaf(o[2], al1, step1 * (float) od[2]);
                o[3] = fmaf(o[3], al1, step1 * (float) od[3]);
            }
        }
    }

    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    // even n8 tiles hold dims 16 b + 2 n, odd ones 16 b + 2 n + 1: lane (g, tq) owns dims 16 b + 4 tq .. + 3 of its rows
    const auto store = [&](int row, int e, float m, float l, float inv) {
        if (row >= rows) return;
        const int64_t idx = Split ? ((int64_t) split * T + q0 + row) * n_head + head : (int64_t) (q0 + row) * n_head + head;
        if (Split && tq == 0) part_stats[idx] = make_float2(m, l);
#pragma unroll
        for (int b = 0; b < DB; ++b) {
            const float4 v = make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv);
            if (Split) io_st4(part_rows + idx * HD + b * 16 + 4 * tq, v);   // the splits' rows stay FP32
            else io_st4(out + idx * HD + b * 16 + 4 * tq, v);
        }
    };
    store(row0, 0, m0, l0, i0);
    store(row1, 2, m1, l1, i1);
}

// Block (row, query head), one thread per dim: the splits' rows weighted by sum_s 2^((m_s - M) scale).
template <class IO>
__global__ void __launch_bounds__(HD) prompt_i8_merge_kernel(const float* __restrict__ part_rows,
                                                             const float2* __restrict__ part_stats, int n_head, int T,
                                                             int splits, IO* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        const float2 st = part_stats[((int64_t) s * T + t) * n_head + head];
        if (st.y > 0.0f) mx = fmaxf(mx, st.x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int64_t idx = ((int64_t) s * T + t) * n_head + head;
        const float2 st = part_stats[idx];
        if (st.y > 0.0f) {
            const float w = st.y * exp2f((st.x - mx) * kScaleLog2);
            num = fmaf(w, part_rows[idx * HD + d], num);
            den += w;
        }
    }
    io_st1(out + ((int64_t) t * n_head + head) * HD + d, den > 0.0f ? num / den : 0.0f);
}

// the current device, for a per-device shared-memory opt-in
int opt_in_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        basalt::log::error("basalt", "gqa_prompt_attn: device %d out of range", dev);
        std::exit(1);
    }
    return dev;
}

template <int W, bool Split, class IO = float>
void launch_prompt_i8(const IO* q, int n_head, int n_kv, int pos0, int T, IO* out, KvInt8 q8, int splits,
                      float* scratch, cudaStream_t st, KvFar far_kv) {
    static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
    const int dev = opt_in_device();
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(prompt_i8_kernel<W, Split, IO>, cudaFuncAttributeMaxDynamicSharedMemorySize, i8_smem<W>()) !=
            cudaSuccess) {
            basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    float* rows = scratch;
    float2* stats = reinterpret_cast<float2*>(scratch + (size_t) splits * T * n_head * HD);
    const dim3 grid((unsigned) ((T + W * 16 - 1) / (W * 16)), (unsigned) n_head, (unsigned) splits);
    prompt_i8_kernel<W, Split, IO><<<grid, W * 32, i8_smem<W>(), st>>>(q, n_head, n_kv, pos0, T, out, q8, rows, stats, far_kv);
    if (Split) {
        check("gqa_prompt_attn");
        prompt_i8_merge_kernel<IO><<<dim3((unsigned) T, (unsigned) n_head), HD, 0, st>>>(rows, stats, n_head, T, splits, out);
    }
}

// The int8 launch's block shape and key splits, as Infernix plans them (int8_fast_prompt_plan): a launch takes about
// (waves) x (visible cells / splits) sweeps of one cell by an 8-warp block; a 4-warp block covers half the rows in 0.8 of
// that time, unsplit; a split launch also writes and merges every split's rows, about half a cell's sweep per row, split
// and head over 24 heads.  Infernix measured this within 0.14 % of the fastest choice on average, 3.8 % at worst.
struct I8Plan {
    int warps = 8, splits = 1;
};
std::atomic<int> plan_device{-1};   // gqa_prompt_plan_device
I8Plan plan_prompt_i8(int n_head, int64_t T, int64_t cells, size_t split_floats) {
    constexpr int kMaxSplits = 8;
    int dev = plan_device.load(std::memory_order_relaxed), sms = 0;
    if (dev < 0) cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    sms = std::max(sms, 1);
    const auto waves = [&](int64_t blocks) { return (double) ((blocks + sms - 1) / sms); };
    const double keys = (double) cells;
    I8Plan best{4, 1};
    // (Lapilli, V100) the 8-warp kernel's shared memory (100,352 bytes at head_dim 256) is above a V100's opt-in limit
    // (98,304): there the 4-warp kernel (83,968) takes every chunk, unsplit
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (optin < i8_smem<8>()) return best;
    double best_cost = waves((T + 63) / 64 * n_head) * 0.8 * keys;
    const int64_t blocks = (T + 127) / 128 * n_head;
    for (int s = 1; s <= kMaxSplits; ++s) {
        if (s > 1 && gqa_prompt_split_floats(T, n_head, s) > split_floats) break;
        double cost = waves(blocks * s) * keys / s;
        if (s > 1) cost += 0.5 / 24.0 * (double) T * s * n_head;
        if (cost < best_cost) {
            best_cost = cost;
            best = {8, s};
        }
    }
    return best;
}

}  // namespace

size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim) {
    const int64_t chunks = (max_cells + CHUNK - 1) / CHUNK + 1;
    return (size_t) T * n_head * decode_splits(chunks) * (head_dim + 2);
}

void gqa_set_decode_variant(int v) { g_decode_variant = v; }
void gqa_set_prompt_wmma(int v) { g_prompt_wmma = v; }

bool gqa_wmma_layout_ok() {
    float* d = nullptr;
    if (cudaMalloc(&d, 256 * 4) != cudaSuccess) return false;
    wmma_acc_layout_kernel<<<1, 32>>>(d);
    float h[256];
    const bool copied = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    if (!copied) return false;
    for (int lane = 0; lane < 32; ++lane)
        for (int e = 0; e < 8; ++e) {   // decode_wmma3_kernel's rows: r0 for elements 0, 1, 4, 5 and r0 + 2 for 2, 3, 6, 7
            const int r0 = (lane & 1) + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1);
            if ((int) h[lane * 8 + e] / 16 != r0 + ((e & 2) ? 2 : 0)) return false;
        }
    return true;
}

void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                     const int32_t* pos_dev, KvInt8 q8, const float* gate_q, uint16_t* gated16, KvFar far_kv) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || pos0 + T > max_cells) {
        basalt::log::error("basalt", "gqa_decode_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.cap = max_cells;
    r.pos_dev = pos_dev;
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, q8, gate_q, gated16,
                       far_kv);
}

void gqa_decode_attn_segs(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                          int64_t max_cells, float* out, void* stream, const float* gate_q, uint16_t* gated16) {
    if (n_seg < 1 || n_seg > kMaxSegs || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX) {
        basalt::log::error("basalt", "gqa_decode_attn_segs: invalid arguments");
        std::exit(1);
    }
    DecSegs d;
    d.n = n_seg;
    for (int k = 0; k < n_seg; ++k) {
        const GqaSeg& g = segs[k];
        if (g.T < 1 || g.pos0 + g.T > max_cells || d.row0[k] + g.T > kGqaMaxWindow) {
            basalt::log::error("basalt", "gqa_decode_attn_segs: invalid segment");
            std::exit(1);
        }
        d.row0[k + 1] = d.row0[k] + g.T;
        d.kc[k] = (const __half*) g.k_cache;
        d.vc[k] = (const __half*) g.v_cache;
        d.q8[k] = g.q8;
        d.far_kv[k] = g.far_kv;
        d.r[k].pos0 = g.pos0;
        d.r[k].cap = max_cells;
        d.r[k].pos_dev = g.pos_dev;
    }
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    launch_segs<256>(q, n_head, n_kv, d, scratch, n_chunks, out, (cudaStream_t) stream, gate_q, gated16);
}

void gqa_window_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv, int head_dim,
                     int64_t pos0, int T, int64_t hi, int64_t window, int64_t cap, float* scratch, int64_t scratch_cells,
                     float* out, void* stream) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || window <= 0 ||
        (head_dim != 128 && head_dim != 256) || hi < pos0 + T) {
        basalt::log::error("basalt", "gqa_window_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.hi = hi;
    r.window = window;
    r.cap = cap;
    r.causal = 0;
    const int64_t lo = std::max<int64_t>(0, pos0 - window + 1);
    r.base = (int) (lo / CHUNK);
    const int used = (int) ((hi + CHUNK - 1) / CHUNK) - r.base;
    const int n_chunks = (int) ((scratch_cells + CHUNK - 1) / CHUNK) + 1;
    if (used > n_chunks) {
        basalt::log::error("basalt", "gqa_window_attn: the window passes its scratch");
        std::exit(1);
    }
    if (head_dim == 128) launch_window<128>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream);
    else launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream);
}

void gqa_prompt_plan_device(int device) { plan_device.store(device, std::memory_order_relaxed); }

size_t gqa_prompt_split_floats(int64_t T, int n_head, int splits) {
    return (size_t) splits * T * n_head * (HD + 2);   // the splits' rows, then their (max, sum) pairs
}

void gqa_prompt_attn_bf16(const uint16_t* q, int n_head, int n_kv, int64_t pos0, int64_t T, uint16_t* out, void* stream,
                          KvInt8 q8, float* split_scratch, size_t split_floats, KvFar far_kv) {
    if (T <= 0) return;
    if (!q8.k || n_kv < 1 || n_head % n_kv || pos0 + T > INT32_MAX) {
        basalt::log::error("basalt", "gqa_prompt_attn_bf16: an int8 cache and a supported head layout");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
    if (p.warps == 4) launch_prompt_i8<4, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st, far_kv);
    else if (p.splits == 1) launch_prompt_i8<8, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st, far_kv);
    else launch_prompt_i8<8, true>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, p.splits, split_scratch, st, far_kv);
    check("gqa_prompt_attn_bf16");
}

void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8, float* split_scratch, size_t split_floats,
                     KvFar far_kv) {
    if (T <= 0) return;
    if (n_kv < 1 || n_head % n_kv || (!q8.k && PROWS % (n_head / n_kv)) || pos0 + T > INT32_MAX) {
        basalt::log::error("basalt", "gqa_prompt_attn: unsupported head layout or cell count");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (prompt_wmma_on() && WROWS % (n_head / n_kv) == 0) {   // Volta's tensor cores, both caches (until it is chosen: an option)
        static std::atomic<bool> raised_w[64];
        const int dev = opt_in_device();
        if (!raised_w[dev].load(std::memory_order_acquire)) {
            if (cudaFuncSetAttribute(prompt_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PromptSmemW)) !=
                cudaSuccess) {
                basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed (tensor cores)");
                std::exit(1);
            }
            raised_w[dev].store(true, std::memory_order_release);
        }
        const int pb = WROWS / (n_head / n_kv);
        const dim3 grid((unsigned) ((T + pb - 1) / pb), (unsigned) n_kv);
        prompt_wmma_kernel<<<grid, WNW * 32, sizeof(PromptSmemW), st>>>(q, (const __half*) k_cache, (const __half*) v_cache, q8, n_head,
                                                                         n_kv, pos0, T, out, far_kv);
        check("gqa_prompt_attn (tensor cores)");
        return;
    }
    if (q8.k) {
        const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
        if (p.warps == 4) launch_prompt_i8<4, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st, far_kv);
        else if (p.splits == 1) launch_prompt_i8<8, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st, far_kv);
        else launch_prompt_i8<8, true>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, p.splits, split_scratch, st, far_kv);
    } else {
        static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
        const int dev = opt_in_device();
        if (!raised[dev].load(std::memory_order_acquire)) {
            if (cudaFuncSetAttribute(prompt_f16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PromptSmem16)) !=
                cudaSuccess) {
                basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
                std::exit(1);
            }
            raised[dev].store(true, std::memory_order_release);
        }
        const int pb = PROWS / (n_head / n_kv);
        const dim3 grid((unsigned) ((T + pb - 1) / pb), (unsigned) n_kv);
        prompt_f16_kernel<<<grid, PNW * 32, sizeof(PromptSmem16), st>>>(q, (const __half*) k_cache, (const __half*) v_cache,
                                                                         n_head, n_kv, pos0, T, out, far_kv);
    }
    check("gqa_prompt_attn");
}

}  // namespace basalt::kernels
