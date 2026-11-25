// src/models/qwen4exp/kernels/hc8_gr.cu - the hyper-connection read on the matrices' 8-bit copies (int8, one f16 scale per 32
// values of a row; hc_fused.hpp) for every caller that does not run hc_fused_read: fused_gr_read, fused_gr_read_multi
// and gr_read hand such matrices here.  Three kernels:
//
//   norm : R' = R + bo_prev * 2 sigmoid(inj_prev / hc) into R_out (when apply),  rs[c] = rsqrt(mean(R'[c]^2) + eps)
//   down : lo[k] = silu((q_down[k] . xn) / hc),  inject[c] = q_inject[c] . xn,  xn = R' * w_norm * rs (on the fly)
//   up   : mixed[d] = mean_c xn[c,d] * sigmoid(q_up[c*n_embd + d] . lo)
//
// Every token's values come from the same operations in the same order whatever the token count.
#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int MT = kFusedGrMaxT;
constexpr int MAX_HC = 4;
constexpr int MAX_LR = 512;
constexpr int G8 = 32;
constexpr int DOWN_ROWS = 2;      // rows per down CTA
constexpr int DOWN_THREADS = 160;
constexpr int UP_THREADS = 256;

struct Hc8Gr {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    const float* w_norm;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    int T, n, hc, lr, ninj;
    bool apply;
    float eps;
};

__device__ __forceinline__ const float* src_of(const Hc8Gr& p, int t) { return p.apply ? p.R_out[t] : p.R[t]; }
__device__ __forceinline__ float s8(uint32_t w, int j) { return (float) (int) (int8_t) (w >> (8 * j)); }
__device__ __forceinline__ float f16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// one CTA per (stream, token)
__device__ void hc8_norm(const Hc8Gr& p, int c, int t) {
    __shared__ float red[32];
    const int n = p.n;
    const float* r = p.R[t] + (size_t) c * n;
    float w = 0.0f;
    if (p.apply) w = 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) p.hc));
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = r[i];
        if (p.apply) {
            v = fmaf(p.bo[t][i], w, v);
            p.R_out[t][(size_t) c * n + i] = v;
        }
        ss = fmaf(v, v, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) s += red[k];
        p.rs[t][c] = rsqrtf(s / (float) n + p.eps);
    }
}
__global__ void hc8_norm_kernel(Hc8Gr p) { hc8_norm(p, blockIdx.x, blockIdx.y); }

// DOWN_ROWS rows of [q_down; q_inject] per CTA; a thread takes whole scale groups
__device__ void hc8_down(const Hc8Gr& p, int blk) {
    __shared__ float red[DOWN_THREADS / 32][DOWN_ROWS][MT];
    const int D = p.n * p.hc, gpr = D / G8, nrows = p.lr + p.ninj;
    const int8_t* q[DOWN_ROWS];
    const uint16_t* s[DOWN_ROWS];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r) {
        const int k = min(blk * DOWN_ROWS + r, nrows - 1);
        if (k < p.lr) { q[r] = p.qd + (size_t) k * D; s[r] = p.sd + (size_t) k * p.sd_row; }
        else { q[r] = p.qi + (size_t) (k - p.lr) * D; s[r] = p.si + (size_t) (k - p.lr) * p.si_row; }
    }
    float acc[DOWN_ROWS][MT];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.0f;
    for (int g = threadIdx.x; g < gpr; g += blockDim.x) {
        const int col0 = g * G8, c = col0 / p.n;
        float wq[DOWN_ROWS][G8];
        float sc[DOWN_ROWS];
        {
            float wn[G8];
            const float4* w4 = reinterpret_cast<const float4*>(p.w_norm + col0);
#pragma unroll
            for (int j = 0; j < G8 / 4; ++j) {
                const float4 v = w4[j];
                wn[4 * j] = v.x; wn[4 * j + 1] = v.y; wn[4 * j + 2] = v.z; wn[4 * j + 3] = v.w;
            }
#pragma unroll
            for (int r = 0; r < DOWN_ROWS; ++r) {
                const uint4 a = reinterpret_cast<const uint4*>(q[r] + col0)[0];
                const uint4 b = reinterpret_cast<const uint4*>(q[r] + col0)[1];
                const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < G8; ++i) wq[r][i] = s8(w[i >> 2], i & 3) * wn[i];
                sc[r] = f16f(s[r][g]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                const float4* x4 = reinterpret_cast<const float4*>(src_of(p, t) + col0);
                float d[DOWN_ROWS];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) d[r] = 0.0f;
#pragma unroll
                for (int j = 0; j < G8 / 4; ++j) {
                    const float4 v = x4[j];
#pragma unroll
                    for (int r = 0; r < DOWN_ROWS; ++r) {
                        d[r] = fmaf(wq[r][4 * j], v.x, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 1], v.y, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 2], v.z, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 3], v.w, d[r]);
                    }
                }
                const float rs = p.rs[t][c];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) acc[r][t] = fmaf(sc[r] * rs, d[r], acc[r][t]);
            }
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                float v = acc[r][t];
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) red[warp][r][t] = v;
            }
        }
    __syncthreads();
    if (threadIdx.x < DOWN_ROWS * MT) {
        const int r = threadIdx.x / MT, t = threadIdx.x % MT, k = blk * DOWN_ROWS + r;
        if (t < p.T && k < nrows) {
            float v = 0.0f;
            for (int w = 0; w < (int) (blockDim.x >> 5); ++w) v += red[w][r][t];
            if (k < p.lr) {
                const float x = v / (float) p.hc;
                p.lo[t][k] = x / (1.0f + expf(-x));
            } else {
                p.inj_out[t][k - p.lr] = v;
            }
        }
    }
}
__global__ void __launch_bounds__(DOWN_THREADS) hc8_down_kernel(Hc8Gr p) { hc8_down(p, blockIdx.x); }

// one warp per output column d: the hc rows c*n + d, lo staged in shared memory; a lane takes 4-value chunks
__device__ void hc8_up(const Hc8Gr& p, int blk) {
    __shared__ __align__(16) float slo[MT * MAX_LR];
    const int lr = p.lr;
    for (int i = threadIdx.x; i < p.T * lr; i += blockDim.x) slo[i] = p.lo[i / lr][i % lr];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int d = blk * (UP_THREADS / 32) + warp;
    if (d >= p.n) return;
    const int nch = lr / 4;   // chunks of 4 per row
    float part[MAX_HC][MT];
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t) part[c][t] = 0.0f;
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c) {
        if (c < p.hc) {
            const size_t row = (size_t) c * p.n + d;
            const uint32_t* qr = reinterpret_cast<const uint32_t*>(p.qu + row * lr);
            const uint16_t* sr = p.su + row * p.su_row;
#pragma unroll
            for (int j = 0; j < MAX_LR / 128; ++j) {
                const int ch = lane + 32 * j;
                if (ch < nch) {
                    const uint32_t w = qr[ch];
                    const float sc = f16f(sr[ch / (G8 / 4)]);
                    const float q0 = s8(w, 0), q1 = s8(w, 1), q2 = s8(w, 2), q3 = s8(w, 3);
#pragma unroll
                    for (int t = 0; t < MT; ++t) {
                        if (t < p.T) {
                            const float4 l = *reinterpret_cast<const float4*>(slo + t * lr + 4 * ch);
                            float v = q0 * l.x;
                            v = fmaf(q1, l.y, v);
                            v = fmaf(q2, l.z, v);
                            v = fmaf(q3, l.w, v);
                            part[c][t] = fmaf(sc, v, part[c][t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t)
            if (c < p.hc && t < p.T)
                for (int o = 16; o > 0; o >>= 1) part[c][t] += __shfl_xor_sync(0xffffffffu, part[c][t], o);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t < p.T && lane == t) {
            const float* src = src_of(p, t);
            float m = 0.0f;
#pragma unroll
            for (int c = 0; c < MAX_HC; ++c) {
                if (c < p.hc) {
                    const size_t i = (size_t) c * p.n + d;
                    const float xn = src[i] * p.w_norm[i] * p.rs[t][c];
                    m = fmaf(xn, 1.0f / (1.0f + expf(-part[c][t])), m);
                }
            }
            p.mixed[t][d] = m / (float) p.hc;
        }
    }
}
__global__ void __launch_bounds__(UP_THREADS) hc8_up_kernel(Hc8Gr p) { hc8_up(p, blockIdx.x); }


// ---- V100: the down read in column slices
constexpr int SK_COLS = 128;                    // columns per block (a slice stays within one stream: n % 128 == 0)
constexpr int SK_THREADS = 512;
constexpr int SK_PADG = 20;                     // one group of 16 columns every 20 floats: 8 distinct banks
constexpr int SK_PADW = SK_COLS / 16 * SK_PADG;
constexpr int SK_WARPS = SK_THREADS / 32;
constexpr int SK_AHEAD = 6;                     // groups of 4 rows loaded ahead per warp
constexpr int SK_MAX_S = 128;                   // slices at most (D <= 16384)
constexpr int SK_ROWS = MAX_LR + MAX_HC;
__device__ float g_sk_part[SK_MAX_S * MT * SK_ROWS];

__device__ __forceinline__ void sk_row(const Hc8Gr& p, int r, int D, const int8_t*& q, const uint16_t*& s) {
    if (r < p.lr) { q = p.qd + (size_t) r * D; s = p.sd + (size_t) r * p.sd_row; }
    else { q = p.qi + (size_t) (r - p.lr) * D; s = p.si + (size_t) (r - p.lr) * p.si_row; }
}

// 8 threads per row (16 columns each, one uint4 of weights), 4 rows per warp at a time: a row's sum takes 3 shuffles.
// One kernel per token count NT (fully unrolled), the weights converted once per row, two accumulators per token (a
// kernel generic in the token count measured 54% of its stalls on execution dependencies, at 23% occupancy). The slice
// of activations sits in shared memory in groups of 16 columns spaced 20 floats apart (not 16), so the 8 threads of a
// row see no bank conflict (the conflicts measured 3.7 us per token); 512 threads (no register spill).
template <int NT>
__global__ void __launch_bounds__(SK_THREADS) hc8_down_sk_kernel(Hc8Gr p) {
    __shared__ __align__(16) float xs[NT][SK_PADW];
    const int sl = blockIdx.x, k0 = sl * SK_COLS, c = k0 / p.n, D = p.n * p.hc, nrows = p.lr + p.ninj;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, sub = lane & 7, rq = lane >> 3;
    const int nq = (nrows + 3) / 4, g = (k0 + 16 * sub) / G8;
    // ALL the thread's weights load before the slice of activations (one pass: nrows <= 4 * SK_WARPS * SK_AHEAD), so
    // that the two memory waits overlap
    uint4 w[SK_AHEAD];
    float sf[SK_AHEAD];
#pragma unroll
    for (int a = 0; a < SK_AHEAD; ++a) {
        const int r = 4 * (warp + a * SK_WARPS) + rq;
        w[a] = make_uint4(0u, 0u, 0u, 0u); sf[a] = 0.0f;
        if (r < nrows) {
            const int8_t* q; const uint16_t* s;
            sk_row(p, r, D, q, s);
            w[a] = *reinterpret_cast<const uint4*>(q + k0 + 16 * sub);
            sf[a] = f16f(s[g]);
        }
    }
    for (int i = threadIdx.x; i < NT * SK_COLS; i += blockDim.x) {
        const int t = i / SK_COLS, j = i % SK_COLS;
        xs[t][(j >> 4) * SK_PADG + (j & 15)] = src_of(p, t)[k0 + j] * p.w_norm[k0 + j] * p.rs[t][c];
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < SK_AHEAD; ++a) {
        const int qd = warp + a * SK_WARPS, r = 4 * qd + rq;
        if (qd >= nq) break;
        const uint32_t wv[4] = {w[a].x, w[a].y, w[a].z, w[a].w};
        float wf[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) wf[j] = s8(wv[j >> 2], j & 3);
        float v[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* x4 = reinterpret_cast<const float4*>(&xs[t][SK_PADG * sub]);
            float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const float4 xa = x4[j], xb = x4[j + 1];
                d0 = fmaf(wf[4 * j], xa.x, d0);
                d1 = fmaf(wf[4 * j + 4], xb.x, d1);
                d0 = fmaf(wf[4 * j + 1], xa.y, d0);
                d1 = fmaf(wf[4 * j + 5], xb.y, d1);
                d0 = fmaf(wf[4 * j + 2], xa.z, d0);
                d1 = fmaf(wf[4 * j + 6], xb.z, d1);
                d0 = fmaf(wf[4 * j + 3], xa.w, d0);
                d1 = fmaf(wf[4 * j + 7], xb.w, d1);
            }
            v[t] = sf[a] * (d0 + d1);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) v[t] += __shfl_xor_sync(0xffffffffu, v[t], o);
            if (sub == t && r < nrows) g_sk_part[((size_t) sl * MT + t) * SK_ROWS + r] = v[t];
        }
    }
}

// ---- up on V100: 16 outputs d per block; the weights of the 4 streams (rows c*n + d, lr = 320 bytes) copied at once into
// shared memory (a stride of 81 words: no bank conflict), the scales and lo too; warp c, lane (d, half of K): a whole
// output per thread with no shuffle, the two halves joined by one shuffle per token; then mixed as in hc8_up.
constexpr int UV_D = 16;
constexpr int UV_LR = 320;
constexpr int UV_THREADS = 32 * MAX_HC;
constexpr int UV_WS = UV_LR / 4 + 1;            // words per row in shared memory (odd)
constexpr int UV_SS = UV_LR / G8 + 1;
template <int NT>
__global__ void __launch_bounds__(UV_THREADS) hc8_up_v_kernel(Hc8Gr p) {
    __shared__ uint32_t sw[MAX_HC * UV_D][UV_WS];
    __shared__ float ssc[MAX_HC * UV_D][UV_SS];
    __shared__ __align__(16) float slo[NT][UV_LR];
    __shared__ float pr[MAX_HC][NT][UV_D];
    const int d0 = blockIdx.x * UV_D, n = p.n, nrow = p.hc * UV_D;
    // A thread issues all its loads first (into registers), then stores them to shared memory: copying one load then one
    // store at a time measured flat at about 30 us
    constexpr int U4 = UV_LR / 16;                                  // uint4 per weight row
    constexpr int NW = MAX_HC * UV_D * U4 / UV_THREADS;             // weight uint4 per thread
    constexpr int NS = (MAX_HC * UV_D * (UV_LR / G8) + UV_THREADS - 1) / UV_THREADS;
    constexpr int NL = (MT * UV_LR / 4 + UV_THREADS - 1) / UV_THREADS;
    uint4 wr[NW];
    uint16_t sr[NS];
    float4 lr4[NL];
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, rr = idx / U4, wj = idx % U4, c = rr / UV_D, i = rr % UV_D;
        wr[k] = rr < nrow ? reinterpret_cast<const uint4*>(p.qu + ((size_t) c * n + d0 + i) * UV_LR)[wj] : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int k = 0; k < NS; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, rr = idx / (UV_LR / G8), j = idx % (UV_LR / G8), c = rr / UV_D, i = rr % UV_D;
        sr[k] = rr < nrow ? p.su[((size_t) c * n + d0 + i) * p.su_row + j] : (uint16_t) 0;
    }
#pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, t = idx / (UV_LR / 4), j = idx % (UV_LR / 4);
        lr4[k] = t < NT ? reinterpret_cast<const float4*>(p.lo[t])[j] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, rr = idx / U4, wj = idx % U4;
        if (rr < nrow) {
            sw[rr][4 * wj] = wr[k].x; sw[rr][4 * wj + 1] = wr[k].y; sw[rr][4 * wj + 2] = wr[k].z; sw[rr][4 * wj + 3] = wr[k].w;
        }
    }
#pragma unroll
    for (int k = 0; k < NS; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, rr = idx / (UV_LR / G8), j = idx % (UV_LR / G8);
        if (rr < nrow) ssc[rr][j] = f16f(sr[k]);
    }
#pragma unroll
    for (int k = 0; k < NL; ++k) {
        const int idx = threadIdx.x + k * UV_THREADS, t = idx / (UV_LR / 4), j = idx % (UV_LR / 4);
        if (t < NT) *reinterpret_cast<float4*>(&slo[t][4 * j]) = lr4[k];
    }
    __syncthreads();
    const int c = threadIdx.x >> 5, lane = threadIdx.x & 31, i = lane & (UV_D - 1), h = lane >> 4;
    if (c < p.hc) {
        const int rr = c * UV_D + i, kb = h * (UV_LR / 2);
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        for (int k = kb; k < kb + UV_LR / 2; k += G8) {
            float gs[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) gs[t] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < G8; kk += 4) {
                const uint32_t w = sw[rr][(k + kk) / 4];
                const float q0 = s8(w, 0), q1 = s8(w, 1), q2 = s8(w, 2), q3 = s8(w, 3);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const float4 l = *reinterpret_cast<const float4*>(&slo[t][k + kk]);
                    float v = q0 * l.x;
                    v = fmaf(q1, l.y, v);
                    v = fmaf(q2, l.z, v);
                    v = fmaf(q3, l.w, v);
                    gs[t] += v;
                }
            }
            const float sc = ssc[rr][k / G8];
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[t] = fmaf(sc, gs[t], acc[t]);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 16);   // half 0 + half 1, the same order for both
            if (h == 0) pr[c][t][i] = acc[t];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < NT * UV_D; idx += blockDim.x) {
        const int t = idx / UV_D, ii = idx % UV_D, d = d0 + ii;
        if (d >= n) continue;
        const float* src = src_of(p, t);
        float m = 0.0f;
        for (int cc = 0; cc < p.hc; ++cc) {
            const size_t x = (size_t) cc * n + d;
            const float xn = src[x] * p.w_norm[x] * p.rs[t][cc];
            m = fmaf(xn, 1.0f / (1.0f + expf(-pr[cc][t][ii])), m);
        }
        p.mixed[t][d] = m / (float) p.hc;
    }
}
template <int NT> void uv_launch(const Hc8Gr& p, cudaStream_t st) {
    hc8_up_v_kernel<NT><<<(unsigned) ((p.n + UV_D - 1) / UV_D), UV_THREADS, 0, st>>>(p);
}
void uv_up(const Hc8Gr& p, cudaStream_t st) {
    switch (p.T) {
        case 1: uv_launch<1>(p, st); break;
        case 2: uv_launch<2>(p, st); break;
        case 3: uv_launch<3>(p, st); break;
        case 4: uv_launch<4>(p, st); break;
        case 5: uv_launch<5>(p, st); break;
        case 6: uv_launch<6>(p, st); break;
        case 7: uv_launch<7>(p, st); break;
        default: uv_launch<8>(p, st); break;
    }
}

template <int NT> void sk_launch(const Hc8Gr& p, int ns, cudaStream_t st) { hc8_down_sk_kernel<NT><<<(unsigned) ns, SK_THREADS, 0, st>>>(p); }
void sk_down(const Hc8Gr& p, int ns, cudaStream_t st) {
    switch (p.T) {
        case 1: sk_launch<1>(p, ns, st); break;
        case 2: sk_launch<2>(p, ns, st); break;
        case 3: sk_launch<3>(p, ns, st); break;
        case 4: sk_launch<4>(p, ns, st); break;
        case 5: sk_launch<5>(p, ns, st); break;
        case 6: sk_launch<6>(p, ns, st); break;
        case 7: sk_launch<7>(p, ns, st); break;
        default: sk_launch<8>(p, ns, st); break;
    }
}

__global__ void hc8_sk_reduce_kernel(Hc8Gr p, int nslices) {
    const int t = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= p.lr + p.ninj) return;
    float v = 0.0f;
    for (int sl = 0; sl < nslices; ++sl) v += g_sk_part[((size_t) sl * MT + t) * SK_ROWS + r];   // fixed order
    if (r < p.lr) {
        const float x = v / (float) p.hc;
        p.lo[t][r] = x / (1.0f + expf(-x));
    } else {
        p.inj_out[t][r - p.lr] = v;
    }
}

// the current card takes the sliced read: compute capability < 8, or BASALT_HC8_SK=1 (=0: never)
bool hc8_sk_on() {
    static int forced = -2;
    if (forced == -2) {
        const char* e = std::getenv("BASALT_HC8_SK");
        forced = (e && *e) ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }
    if (forced >= 0) return forced == 1;
    static int known[16] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) { (void) cudaGetLastError(); return false; }
    if (known[dev] == 0) {
        int major = 0;
        const bool q = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess;
        if (!q) (void) cudaGetLastError();
        known[dev] = q && major < 8 ? 1 : 2;
    }
    return known[dev] == 1;
}

// ---- V100: the read in three kernels. Measured on the sliced read above (nvprof, counters): four kernels at 6-24%
// occupancy that wait on memory, ~43 us for ~7 MB of weights.
//   hc8_down4b_kernel: the norm folded into the slice (writes R', the slice's sum of squares, products on R' * w_norm)
//   hc8_red4b_kernel:  rs per stream, then lo and the injection (the sum of the slices, each times its stream's rs)
//   hc8_up4b_kernel:   4 threads per q_up row, 16 outputs d per block (the 4 streams), then mixed as in hc8_up
// Every token's values come from the same operations in the same order whatever the token count.
constexpr int V4_COLS = 128;                      // columns per slice (a slice stays within one stream)
constexpr int V4_DT = 672;                        // down kernel threads: 21 warps (4 passes of 4 rows cover 336 rows)
constexpr int V4_DROWS = V4_DT / 2;               // rows of [q_down; q_inject] at most
constexpr int V4_RT = 256;                        // sum kernel threads: 8 warps x 32 rows
constexpr int V4_RW = V4_RT / 32;
constexpr int V4_SPW = (SK_MAX_S + V4_RW - 1) / V4_RW;   // slices per warp at most
constexpr int V4_UD = 16;                         // outputs d per block of the up kernel
constexpr int V4_UT = 4 * MAX_HC * V4_UD;         // 256 threads: 4 per row
constexpr int V4_LR = 320;
constexpr int V4_LRP = V4_LR + V4_LR / 32 * 4;    // lo in shared memory: a 4-float offset every 32
__device__ float g_v4_ssq[SK_MAX_S * MT];         // sums of squares [slice][token]

// 16 8-bit integers to float without I2F: the bits of 2^23 + (q ^ 0x80), minus 2^23 + 128 (exact)
__device__ __forceinline__ void v4_cvt16(const uint4 w, float* f) {
    const uint32_t v[4] = {w.x ^ 0x80808080u, w.y ^ 0x80808080u, w.z ^ 0x80808080u, w.w ^ 0x80808080u};
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            f[4 * k + j] = __int_as_float((int) __byte_perm(v[k], 0x4B000000u, 0x7440u | (unsigned) j)) - 8388736.0f;
}

// 16 products w . x in two FMA chains (x in shared memory, 16-byte aligned)
__device__ __forceinline__ float v4_dot16(const float* wf, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
        const float4 xa = x4[e], xb = x4[e + 1];
        d0 = fmaf(wf[4 * e], xa.x, d0);
        d1 = fmaf(wf[4 * e + 4], xb.x, d1);
        d0 = fmaf(wf[4 * e + 1], xa.y, d0);
        d1 = fmaf(wf[4 * e + 5], xb.y, d1);
        d0 = fmaf(wf[4 * e + 2], xa.z, d0);
        d1 = fmaf(wf[4 * e + 6], xb.z, d1);
        d0 = fmaf(wf[4 * e + 3], xa.w, d0);
        d1 = fmaf(wf[4 * e + 7], xb.w, d1);
    }
    return d0 + d1;
}

// One slice of 128 columns per block; 8 threads per row, one uint4 each (4 whole 128-byte lines per instruction), 4
// passes of 4 rows per warp (with 2 threads per row the kernel measured 31% of its stalls on the memory queue: 16 cache
// lines per instruction). What warp t < NT prepares (R, bo, inj, w_norm) is loaded BEFORE the weights (otherwise these
// small loads wait behind the 3 MB of weights).
constexpr int V4_PADG = 20;                       // one group of 16 columns every 20 floats: 8 distinct banks
constexpr int V4_PADW = V4_COLS / 16 * V4_PADG;
constexpr int V4_DW = V4_DT / 32;                 // 21 warps
constexpr int V4_DA = 4;                          // passes: 4 x 4 rows x 21 warps = 336 rows
template <int NT>
__global__ void __launch_bounds__(V4_DT, 1) hc8_down4b_kernel(Hc8Gr p) {
    __shared__ __align__(16) float xs[NT][V4_PADW];
    const int sl = blockIdx.x, k0 = sl * V4_COLS, n = p.n, c = k0 / n, D = n * p.hc, nrows = p.lr + p.ninj;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, sub = lane & 7, rq = lane >> 3;
    const bool stage = warp < NT;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = v, wn = v;
    float ij = 0.0f;
    if (stage) {
        const int t = warp, j = 4 * lane;
        v = *reinterpret_cast<const float4*>(p.R[t] + k0 + j);
        wn = *reinterpret_cast<const float4*>(p.w_norm + k0 + j);
        if (p.apply) {
            b = *reinterpret_cast<const float4*>(p.bo[t] + (k0 - c * n) + j);
            ij = p.inj_prev[t][c];
        }
    }
    // Branch-free loads (a thread without a row reads a valid one), converted after the barrier: with one load per branch
    // and its first use right behind it, the machine code issued the 4 loads one after the other
    uint4 w[V4_DA];
    uint16_t sh[V4_DA];
#pragma unroll
    for (int a = 0; a < V4_DA; ++a) {
        const int r = min(4 * (warp + V4_DW * a) + rq, nrows - 1);
        const bool inj = r >= p.lr;
        const int8_t* q = inj ? p.qi + (size_t) (r - p.lr) * D : p.qd + (size_t) r * D;
        const uint16_t* s = inj ? p.si + (size_t) (r - p.lr) * p.si_row : p.sd + (size_t) r * p.sd_row;
        w[a] = reinterpret_cast<const uint4*>(q + k0)[sub];
        sh[a] = s[k0 / G8 + (sub >> 1)];
    }
    if (stage) {   // R' = R + bo * 2 sigmoid(inj / hc) (as in hc8_norm), its sum of squares, R' * w_norm into shared memory
        const int t = warp, j = 4 * lane;
        if (p.apply) {
            const float wa = 2.0f / (1.0f + expf(-ij / (float) p.hc));
            v.x = fmaf(b.x, wa, v.x);
            v.y = fmaf(b.y, wa, v.y);
            v.z = fmaf(b.z, wa, v.z);
            v.w = fmaf(b.w, wa, v.w);
            *reinterpret_cast<float4*>(p.R_out[t] + k0 + j) = v;
        }
        float ss = 0.0f;
        ss = fmaf(v.x, v.x, ss);
        ss = fmaf(v.y, v.y, ss);
        ss = fmaf(v.z, v.z, ss);
        ss = fmaf(v.w, v.w, ss);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) g_v4_ssq[sl * MT + t] = ss;
        *reinterpret_cast<float4*>(&xs[t][(j >> 4) * V4_PADG + (j & 15)]) =
            make_float4(v.x * wn.x, v.y * wn.y, v.z * wn.z, v.w * wn.w);
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < V4_DA; ++a) {
        if (4 * (warp + V4_DW * a) >= nrows) break;   // the same decision for the whole warp
        const int r = 4 * (warp + V4_DW * a) + rq;
        float wf[16];
        v4_cvt16(w[a], wf);
        float vt[NT];
        const float sfa = f16f(sh[a]);
#pragma unroll
        for (int t = 0; t < NT; ++t) vt[t] = sfa * v4_dot16(wf, &xs[t][V4_PADG * sub]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) vt[t] += __shfl_xor_sync(0xffffffffu, vt[t], o);   // the 8 threads of the row
            if (sub == t && r < nrows) g_sk_part[((size_t) sl * MT + t) * SK_ROWS + r] = vt[t];
        }
    }
}

// block (32 rows, token t): warp w sums the slices [w * per, (w + 1) * per) of its row, loaded at once; warps c < hc
// compute the rs of stream c. The sums of squares are loaded first, and each slice's stream is tracked without a division
__global__ void __launch_bounds__(V4_RT) hc8_red4b_kernel(Hc8Gr p, int nslices) {
    __shared__ float rs_s[MAX_HC];
    __shared__ float pw[V4_RW][32];
    const int t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * 32 + lane;
    const int nrows = p.lr + p.ninj, spc = p.n / V4_COLS, per = (nslices + V4_RW - 1) / V4_RW;
    const int s0 = warp * per, s1 = min(s0 + per, nslices);
    float sq = 0.0f;
    const bool rsw = warp < p.hc;
    if (rsw)
        for (int k = lane; k < spc; k += 32) sq += g_v4_ssq[(warp * spc + k) * MT + t];
    float part[V4_SPW];
#pragma unroll
    for (int k = 0; k < V4_SPW; ++k)
        part[k] = (s0 + k < s1 && r < nrows) ? g_sk_part[((size_t) (s0 + k) * MT + t) * SK_ROWS + r] : 0.0f;
    if (rsw) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) {
            rs_s[warp] = rsqrtf(sq / (float) p.n + p.eps);
            if (blockIdx.x == 0) p.rs[t][warp] = rs_s[warp];
        }
    }
    __syncthreads();
    int cs = s0 / spc, next = (cs + 1) * spc;          // the stream of slice s0 + k, tracked without a division
    float v = 0.0f;
#pragma unroll
    for (int k = 0; k < V4_SPW; ++k) {
        if (s0 + k >= s1) break;
        if (s0 + k == next) { ++cs; next += spc; }
        v = fmaf(rs_s[cs], part[k], v);
    }
    pw[warp][lane] = v;
    __syncthreads();
    if (warp == 0 && r < nrows) {
        float tot = 0.0f;
#pragma unroll
        for (int k = 0; k < V4_RW; ++k) tot += pw[k][lane];
        if (r < p.lr) {
            const float x = tot / (float) p.hc;
            p.lo[t][r] = x / (1.0f + expf(-x));
        } else {
            p.inj_out[t][r - p.lr] = tot;
        }
    }
}

// 16 outputs d per block; thread (row c * 16 + i, quarter q): the row's uint4 number q + 4m (m = 0..4), scale group
// (q >> 1) + 2m; then mixed, thread t * 16 + i
template <int NT>
__global__ void __launch_bounds__(V4_UT) hc8_up4b_kernel(Hc8Gr p) {
    __shared__ __align__(16) float slo[NT][V4_LRP];
    __shared__ float gs[MAX_HC][NT][V4_UD];
    __shared__ float* s_mix[NT];
    const int d0 = blockIdx.x * V4_UD, n = p.n, tid = threadIdx.x;
    const int rr = tid >> 2, q = tid & 3, c = rr / V4_UD, i = rr % V4_UD;
    const bool on = c < p.hc && d0 + i < n;
    // what mixed reads (R', w_norm, rs; thread t * 16 + i: token t, output d0 + i), loaded BEFORE the weights
    const int mt = tid / V4_UD, mi = tid % V4_UD;
    const bool mon = tid < NT * V4_UD && d0 + mi < n;
    float xr[MAX_HC], xw[MAX_HC], xq[MAX_HC];
#pragma unroll
    for (int cc = 0; cc < MAX_HC; ++cc) { xr[cc] = 0.0f; xw[cc] = 0.0f; xq[cc] = 0.0f; }
    if (mon) {
        const float* src = p.apply ? p.R_out[mt] : p.R[mt];
        const float* rsp = p.rs[mt];
#pragma unroll
        for (int cc = 0; cc < MAX_HC; ++cc)
            if (cc < p.hc) {
                const size_t x = (size_t) cc * n + d0 + mi;
                xr[cc] = src[x];
                xw[cc] = p.w_norm[x];
                xq[cc] = rsp[cc];
            }
    }
    uint4 w[5];
    uint4 sa = make_uint4(0u, 0u, 0u, 0u), sb = sa;
#pragma unroll
    for (int m = 0; m < 5; ++m) w[m] = make_uint4(0u, 0u, 0u, 0u);
    if (on) {
        const size_t row = (size_t) c * n + d0 + i;
        const uint4* q4 = reinterpret_cast<const uint4*>(p.qu + row * V4_LR);
#pragma unroll
        for (int m = 0; m < 5; ++m) w[m] = q4[q + 4 * m];
        const uint4* s4 = reinterpret_cast<const uint4*>(p.su + row * p.su_row);
        sa = s4[0];
        sb = s4[1];
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
        if (tid == t) s_mix[t] = p.mixed[t];
    {
        float4 lv[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t)
            lv[t] = tid < V4_LR / 4 ? reinterpret_cast<const float4*>(p.lo[t])[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (tid < V4_LR / 4) {
            const int a = 4 * tid;
#pragma unroll
            for (int t = 0; t < NT; ++t) *reinterpret_cast<float4*>(&slo[t][a + 4 * (a >> 5)]) = lv[t];
        }
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const uint32_t scw[5] = {sa.x, sa.y, sa.z, sa.w, sb.x};   // word m: groups 2m and 2m + 1
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        float wf[16];
        v4_cvt16(w[m], wf);
        const float sc = f16f((uint16_t) (scw[m] >> (16 * (q >> 1))));
        const int a = 16 * (q + 4 * m), ap = a + 4 * (a >> 5);
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = fmaf(sc, v4_dot16(wf, &slo[t][ap]), acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {   // the four quarters: every thread ends with the same bits
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 1);
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 2);
    }
    if (q == 0 && on)
#pragma unroll
        for (int t = 0; t < NT; ++t) gs[c][t][i] = 1.0f / (1.0f + expf(-acc[t]));
    __syncthreads();
    if (mon) {
        float m = 0.0f;
#pragma unroll
        for (int cc = 0; cc < MAX_HC; ++cc) {
            if (cc < p.hc) {
                const float xn = xr[cc] * xw[cc] * xq[cc];
                m = fmaf(xn, gs[cc][mt][mi], m);
            }
        }
        s_mix[mt][d0 + mi] = m / (float) p.hc;
    }
}

template <int NT> void d4_launch(const Hc8Gr& p, int ns, cudaStream_t st) { hc8_down4b_kernel<NT><<<(unsigned) ns, V4_DT, 0, st>>>(p); }
template <int NT> void u4_launch(const Hc8Gr& p, cudaStream_t st) {
    hc8_up4b_kernel<NT><<<(unsigned) ((p.n + V4_UD - 1) / V4_UD), V4_UT, 0, st>>>(p);
}
void read4b(const Hc8Gr& p, cudaStream_t st) {
    const int ns = p.n * p.hc / V4_COLS;
    switch (p.T) {
        case 1: d4_launch<1>(p, ns, st); break;
        case 2: d4_launch<2>(p, ns, st); break;
        case 3: d4_launch<3>(p, ns, st); break;
        case 4: d4_launch<4>(p, ns, st); break;
        case 5: d4_launch<5>(p, ns, st); break;
        case 6: d4_launch<6>(p, ns, st); break;
        case 7: d4_launch<7>(p, ns, st); break;
        default: d4_launch<8>(p, ns, st); break;
    }
    hc8_red4b_kernel<<<dim3((unsigned) ((p.lr + p.ninj + 31) / 32), (unsigned) p.T), V4_RT, 0, st>>>(p, ns);
    switch (p.T) {
        case 1: u4_launch<1>(p, st); break;
        case 2: u4_launch<2>(p, st); break;
        case 3: u4_launch<3>(p, st); break;
        case 4: u4_launch<4>(p, st); break;
        case 5: u4_launch<5>(p, st); break;
        case 6: u4_launch<6>(p, st); break;
        case 7: u4_launch<7>(p, st); break;
        default: u4_launch<8>(p, st); break;
    }
}

// the three-kernel read: BASALT_HC8_4B=1 / =0, otherwise where the sliced read runs (hc8_sk_on: compute capability < 8)
bool hc8_4b_on() {
    static int forced = -2;
    if (forced == -2) {
        const char* e = std::getenv("BASALT_HC8_4B");
        forced = (e && *e) ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }
    if (forced >= 0) return forced == 1;
    return hc8_sk_on();
}
bool v4_a16(const void* x) { return (reinterpret_cast<uintptr_t>(x) & 15u) == 0; }
// the geometry and alignments the three kernels read (float4, uint4, uint2); otherwise the sliced read runs
bool hc8_4b_ok(const Hc8Gr& p) {
    if (!hc8_4b_on()) return false;
    const long long D = (long long) p.n * p.hc;
    if (p.T < 1 || p.T > MT || p.hc < 1 || p.hc > MAX_HC || p.n % V4_COLS != 0 || D / V4_COLS > SK_MAX_S ||
        p.lr != V4_LR || p.lr + p.ninj > V4_DROWS || p.lr + p.ninj > SK_ROWS || p.n % V4_UD != 0 || p.n / V4_COLS < 1)
        return false;
    if (!v4_a16(p.qd) || !v4_a16(p.sd) || !v4_a16(p.qu) || !v4_a16(p.su) || !v4_a16(p.w_norm) || p.sd_row % 8 != 0 ||
        p.su_row % 8 != 0 || p.su_row < 16)
        return false;
    if (p.ninj > 0 && (!v4_a16(p.qi) || !v4_a16(p.si) || p.si_row % 8 != 0)) return false;
    for (int t = 0; t < p.T; ++t)
        if (!v4_a16(p.R[t]) || !v4_a16(p.lo[t]) || (p.apply && (!v4_a16(p.R_out[t]) || !v4_a16(p.bo[t])))) return false;
    return true;
}

}  // namespace

bool hc8_gr_read_multi(const FusedGrArgs* a, int n_tok, int64_t n_embd, int64_t hc, void* stream) {
    if (!hc8_enabled() || n_tok < 1 || a == nullptr) return false;
    Hc8View vd, vu, vi;
    if (!hc8_view(a[0].w_down, vd)) return false;
    const bool inj = a[0].w_inject != nullptr;
    const int64_t D = n_embd * hc;
    const bool have = hc8_view(a[0].w_up, vu) && (!inj || hc8_view(a[0].w_inject, vi));
    if (!have) {
        if (vd.owned || vu.owned) {
            basalt::log::error("basalt", "hc8_gr_read: a read's matrices are not all 8-bit copies");
            std::exit(1);
        }
        return false;
    }
    if (n_tok > MT || hc < 1 || hc > MAX_HC || vd.cols != D || vu.rows != D || vu.cols != vd.rows || vd.rows > MAX_LR ||
        (inj && (vi.rows != hc || vi.cols != D)) || n_embd % G8 != 0) {
        basalt::log::error("basalt", "hc8_gr_read: unsupported geometry (%d tokens, n_embd %lld, hc %lld, lr %d)", n_tok,
                     (long long) n_embd, (long long) hc, vd.rows);
        std::exit(1);
    }
    Hc8Gr p{};
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.lo || !x.rs || !x.mixed || (inj && !x.inject_out) || x.apply != a[0].apply ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            basalt::log::error("basalt", "hc8_gr_read: invalid arguments for token %d", t);
            std::exit(1);
        }
        p.R[t] = x.R; p.R_out[t] = x.R_out; p.bo[t] = x.bo_prev; p.inj_prev[t] = x.inj_prev;
        p.lo[t] = x.lo; p.rs[t] = x.rs; p.inj_out[t] = x.inject_out; p.mixed[t] = x.mixed;
    }
    p.w_norm = a[0].w_norm;
    p.qd = vd.q; p.sd = vd.s; p.sd_row = vd.srow;
    p.qu = vu.q; p.su = vu.s; p.su_row = vu.srow;
    if (inj) { p.qi = vi.q; p.si = vi.s; p.si_row = vi.srow; }
    p.T = n_tok; p.n = (int) n_embd; p.hc = (int) hc; p.lr = vd.rows; p.ninj = inj ? (int) hc : 0;
    p.apply = a[0].apply; p.eps = a[0].eps;
    cudaStream_t st = (cudaStream_t) stream;
    if (hc8_4b_ok(p)) {   // on V100, the read in three kernels (the norm inside the slice)
        read4b(p, st);
        const cudaError_t e4 = cudaGetLastError();
        if (e4 != cudaSuccess) {
            basalt::log::error("basalt", "hc8_gr_read (three-kernel path): %s", cudaGetErrorString(e4));
            std::exit(1);
        }
        return true;
    }
    hc8_norm_kernel<<<dim3((unsigned) hc, (unsigned) n_tok), 256, 0, st>>>(p);
    // on V100, the down read in column slices, then the sum of the slices
    bool up_v = false;   // the up kernel for V100 (above)
    if (hc8_sk_on() && D % SK_COLS == 0 && n_embd % SK_COLS == 0 && D / SK_COLS <= SK_MAX_S &&
        p.lr + p.ninj <= SK_ROWS && p.lr + p.ninj <= 4 * SK_WARPS * SK_AHEAD) {
        const int ns = (int) (D / SK_COLS);
        sk_down(p, ns, st);
        hc8_sk_reduce_kernel<<<dim3((unsigned) ((p.lr + p.ninj + 127) / 128), (unsigned) n_tok), 128, 0, st>>>(p, ns);
        up_v = p.lr == UV_LR && n_embd % UV_D == 0;
    } else {
        hc8_down_kernel<<<(unsigned) ((p.lr + p.ninj + DOWN_ROWS - 1) / DOWN_ROWS), DOWN_THREADS, 0, st>>>(p);
    }
    if (up_v) uv_up(p, st);
    else hc8_up_kernel<<<(unsigned) ((n_embd + UP_THREADS / 32 - 1) / (UP_THREADS / 32)), UP_THREADS, 0, st>>>(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "hc8_gr_read: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace basalt::kernels
