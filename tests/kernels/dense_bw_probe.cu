// tests/kernels/dense_bw_probe.cu - (Lapilli, V100) what reading a matrix's bytes costs without the products:
// (1) a plain stream of 16-byte loads over the buffer (the card's practical bandwidth), (2) dense_mmvq's access to NVFP4
// weights at one column (per lane and call: two 32-bit code words and the scale byte of its sub-block, the next call's
// words loaded first), the products replaced by a XOR of the words.  For dense_mmvq_bench: if (2) reads as fast as (1),
// the one-column kernel's gap to the bandwidth is its arithmetic and latencies, not its access pattern.
#include "basalt/kernels/mma884.cuh"
#include "basalt/kernels/dense_decode.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace basalt::kernels::probe {
namespace {

__global__ void stream_kernel(const int4* __restrict__ w, size_t n, unsigned* __restrict__ sink) {
    unsigned a = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = __ldg(w + i);
        a ^= (unsigned) (v.x ^ v.y ^ v.z ^ v.w);
    }
    if (a == 0x9E3779B9u) sink[0] = a;   // never in practice: keeps the loads
}

// one warp per R rows of NVFP4 blocks (36 bytes: 4 scale bytes, 32 code bytes), n_in / 16 calls a row: lane L, iteration i
// is call k = L + 32 i, block k / 4, sub-block k % 4, as dense_mmvq_kernel reads them
template<int R>
__global__ void __launch_bounds__(128) nvfp4_pattern_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int n_in,
                                                            int n_out, unsigned* __restrict__ sink) {
    const int row0 = (int) (blockIdx.x * 4 + threadIdx.y) * R;
    if (row0 >= n_out) return;
    const int lane = threadIdx.x, iters = n_in / 512;
    const uint8_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = w + (size_t) min(row0 + r, n_out - 1) * row_bytes;
    auto load = [&](const uint8_t* row, int k, unsigned& q0, unsigned& q1, unsigned& d) {
        const uint8_t* b = row + (size_t) (k >> 2) * 36;
        const int iqs = k & 3;
        q0 = __ldg((const unsigned*) (b + 4) + 2 * iqs);
        q1 = __ldg((const unsigned*) (b + 4) + 2 * iqs + 1);
        d = __ldg(b + iqs);
    };
    unsigned c0[R], c1[R], cd[R], acc = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) load(wr[r], lane, c0[r], c1[r], cd[r]);
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        const int kn = lane + 32 * (i + 1 < iters ? i + 1 : i);
        unsigned n0[R], n1[R], nd[R];
#pragma unroll
        for (int r = 0; r < R; ++r) load(wr[r], kn, n0[r], n1[r], nd[r]);
#pragma unroll
        for (int r = 0; r < R; ++r) acc ^= c0[r] ^ (c1[r] << 1) ^ cd[r];
#pragma unroll
        for (int r = 0; r < R; ++r) { c0[r] = n0[r]; c1[r] = n1[r]; cd[r] = nd[r]; }
    }
    if (acc == 0x9E3779B9u) sink[row0] = acc;
}

}  // namespace

// us per pass of each probe over `bytes` of device memory (NVFP4 rows of n_in values for the pattern), 20 passes timed
double stream_us(const void* w, size_t bytes, void* stream) {
    unsigned* sink = nullptr;
    cudaMalloc(&sink, 64);
    cudaStream_t s = (cudaStream_t) stream;
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    const size_t n = bytes / 16;
    for (int i = 0; i < 3; ++i) stream_kernel<<<80 * 16, 256, 0, s>>>((const int4*) w, n, sink);
    cudaEventRecord(a, s);
    for (int i = 0; i < 20; ++i) stream_kernel<<<80 * 16, 256, 0, s>>>((const int4*) w, n, sink);
    cudaEventRecord(e, s);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, e);
    cudaFree(sink);
    return 1000.0 * ms / 20;
}
double nvfp4_pattern_us(const void* w, int n_in, int n_out, int rows_per_warp, void* stream) {
    unsigned* sink = nullptr;
    cudaMalloc(&sink, (size_t) n_out * 4);
    cudaStream_t s = (cudaStream_t) stream;
    const size_t rb = (size_t) (n_in / 64) * 36;
    auto go = [&] {
        if (rows_per_warp == 4)
            nvfp4_pattern_kernel<4><<<(n_out + 15) / 16, dim3(32, 4), 0, s>>>((const uint8_t*) w, rb, n_in, n_out, sink);
        else if (rows_per_warp == 2)
            nvfp4_pattern_kernel<2><<<(n_out + 7) / 8, dim3(32, 4), 0, s>>>((const uint8_t*) w, rb, n_in, n_out, sink);
        else
            nvfp4_pattern_kernel<1><<<(n_out + 3) / 4, dim3(32, 4), 0, s>>>((const uint8_t*) w, rb, n_in, n_out, sink);
    };
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    for (int i = 0; i < 3; ++i) go();
    cudaEventRecord(a, s);
    for (int i = 0; i < 20; ++i) go();
    cudaEventRecord(e, s);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, e);
    cudaFree(sink);
    return 1000.0 * ms / 20;
}

}  // namespace basalt::kernels::probe

// ---- device-side random weights and activations for dense_forward_bench (the 27B's 1.6 GB of matrices)
namespace basalt::kernels::probe {
namespace {
__device__ __forceinline__ uint32_t mix(uint64_t x) {   // splitmix64's finalizer, 32 bits
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t) (x ^ (x >> 31));
}
// random bytes, then sane block scales: NVFP4 (36-byte blocks) UE4M3 0x30..0x3F, Q8_0 (34-byte blocks) fp16 2^-7
__global__ void fill_kernel(uint8_t* p, size_t n, int type, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint8_t b = (uint8_t) mix(((uint64_t) seed << 40) ^ i);
        if (type == 40 && i % 36 < 4) b = (uint8_t) (0x30 + (b & 15));
        if (type == 8 && i % 34 < 2) b = i % 34 == 0 ? 0x00 : 0x20;
        p[i] = b;
    }
}
__global__ void fill_floats_kernel(float* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = ((float) (mix(((uint64_t) seed << 40) ^ i) & 0xFFFFFF) / 16777216.0f - 0.5f) * 4.0f;
}
}  // namespace
void fill_weights(void* p, size_t bytes, int type, uint32_t seed, void* stream) {
    fill_kernel<<<80 * 32, 256, 0, (cudaStream_t) stream>>>((uint8_t*) p, bytes, type, seed);
}
void fill_floats(float* p, size_t n, uint32_t seed, void* stream) {
    fill_floats_kernel<<<80 * 32, 256, 0, (cudaStream_t) stream>>>(p, n, seed);
}
}  // namespace basalt::kernels::probe

// ---- for the tensor-core products (dense_mmvq.cu's variant 21)
// (1) mma884's fragments (mma884.cuh): each lane's registers filled with random small integers, one mma, and the result
// against the host product of the matrices the layout says they form (every value exact in fp16 and fp32).
// (2) the weight reads of variant 21 without the products: a thread's own row, 128 values (72 bytes NVFP4, 136 Q8_0) at a
// time as 8-byte loads, the next group's loads issued before the XOR of the current one; QS lanes' classes per row as
// the kernel splits a chunk (warp rows 32 / QS).
namespace basalt::kernels::probe {
namespace {
__global__ void mma884_probe_kernel(const unsigned short* a, const unsigned short* b, const float* c, float* d, int rr) {
    const int lane = threadIdx.x;
    const unsigned a0 = a[4 * lane] | ((unsigned) a[4 * lane + 1] << 16), a1 = a[4 * lane + 2] | ((unsigned) a[4 * lane + 3] << 16);
    const unsigned b0 = b[4 * lane] | ((unsigned) b[4 * lane + 1] << 16), b1 = b[4 * lane + 2] | ((unsigned) b[4 * lane + 3] << 16);
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = c[8 * lane + i];
    if (rr) mma884_rr(acc, a0, a1, b0, b1);
    else mma884(acc, a0, a1, b0, b1);
    for (int i = 0; i < 8; ++i) d[8 * lane + i] = acc[i];
}
template<int QS, int GW>
__global__ void __launch_bounds__(256) rows_pattern_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int n_in, int n_out,
                                                           unsigned* __restrict__ sink) {
    constexpr int RW = 32 / QS, NG = 4 / QS, GB = GW * 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, qp = (lane >> 2) & 3, rs = qp / QS, kc = qp % QS;
    const int row = (int) blockIdx.x * 8 * RW + warp * RW + rs * 8 + (lane & 3) + 4 * (lane >> 4);
    const uint8_t* wr = w + (size_t) min(row, n_out - 1) * row_bytes + kc * NG * GB;
    const int groups = n_in / 512 * NG;
    auto addr = [&](int n) { return reinterpret_cast<const uint2*>(wr + (size_t) (n / NG) * 4 * GB + (n % NG) * GB); };
    uint2 cur[GW], nxt[GW];
    for (int i = 0; i < GW; ++i) cur[i] = __ldg(addr(0) + i);
    unsigned acc = 0;
    for (int n = 0; n < groups; ++n) {
        if (n + 1 < groups)
            for (int i = 0; i < GW; ++i) nxt[i] = __ldg(addr(n + 1) + i);
        for (int i = 0; i < GW; ++i) acc ^= cur[i].x ^ (cur[i].y << 1);
        for (int i = 0; i < GW; ++i) cur[i] = nxt[i];
    }
    if (acc == 0x9E3779B9u) sink[row & 1023] = acc;
}
}  // namespace

// every NVFP4 code x scale byte through the tensor-core variant's decoding: nvfp4_scale_h2 = __float2half2_rn of
// nvfp4_scale_dev, and each of fp4_frags' eight halves = the exact product kvalue x scale; every int8 through i8x4_to_half
__global__ void decode_check_kernel(unsigned* bad) {
    const int x = blockIdx.x, c = threadIdx.x;   // scale byte, code
    const float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const float sc = nvfp4_scale_dev((uint32_t) x);
    const unsigned s2 = nvfp4_scale_h2((uint32_t) x);
    if (s2 != h2_bits(__float2half2_rn(sc))) atomicAdd(bad, 1u);
    unsigned lo[2], hi[2];
    fp4_frags((uint32_t) c * 0x11111111u, s2, lo, hi);
    const float want = kv[c] * sc;   // exact in fp32
    const unsigned all[4] = {lo[0], lo[1], hi[0], hi[1]};
    for (int i = 0; i < 4; ++i) {
        const __half2 h = bits_h2(all[i]);
        if (__low2float(h) != want || __high2float(h) != want) atomicAdd(bad + 1, 1u);
    }
    // variant 22's order: nibble k of the word = code (k + c) % 16, so that every code meets every position
    {
        uint32_t word = 0;
        for (int k = 0; k < 8; ++k) word |= (uint32_t) ((k + c) & 15) << (4 * k);
        unsigned q0[2], q1[2];
        fp4_frags_paired(word, s2, q0, q1);
        const unsigned got[4] = {q0[0], q0[1], q1[0], q1[1]};
        for (int i = 0; i < 4; ++i) {   // register i holds nibbles 2i (low half) and 2i + 1 (high half)
            const __half2 h = bits_h2(got[i]);
            if (__low2float(h) != kv[(2 * i + c) & 15] * sc || __high2float(h) != kv[(2 * i + 1 + c) & 15] * sc)
                atomicAdd(bad + 1, 1u);
        }
        uint32_t k0, k1;   // the activations' pairing: bytes (a0, b0, a1, b1) and (a2, b2, a3, b3)
        act_pairs(0x03020100u, 0x0B0A0908u, k0, k1);
        if (k0 != 0x09010800u || k1 != 0x0B030A02u) atomicAdd(bad + 2, 1u);
    }
    if (x == 0) {   // int8 q = 4 c + j - 128 .. : the 16 threads x 16 cover the 256 values
        for (int j = 0; j < 16; ++j) {
            const int q = (c * 16 + j) - 128;
            __half2 a, b;
            i8x4_to_half((unsigned) (q & 0xFF) * 0x01010101u, a, b);
            if (__low2float(a) != (float) q || __high2float(a) != (float) q || __low2float(b) != (float) q ||
                __high2float(b) != (float) q)
                atomicAdd(bad + 2, 1u);
        }
    }
}
bool nvfp4_decode_check() {
    unsigned* d = nullptr;
    cudaMalloc(&d, 12);
    cudaMemset(d, 0, 12);
    decode_check_kernel<<<256, 16>>>(d);
    unsigned h[3] = {9, 9, 9};
    const cudaError_t err = cudaMemcpy(h, d, 12, cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (err != cudaSuccess) { std::printf("decode check: CUDA error %s\n", cudaGetErrorString(err)); return false; }
    std::printf("decode check (256 scale bytes x 16 codes, 256 int8): scale %u, products %u, int8 %u differ\n", h[0], h[1], h[2]);
    return h[0] == 0 && h[1] == 0 && h[2] == 0;
}

// the layout of a WMMA 16x16x16 f32 accumulator on this card (not documented): a 16x16 matrix of r 16 + c
// loaded into a fragment, each lane's x[i] printed as (r, c); for gqa_attn.cu's one-pass decode, which rescales rows
__global__ void wmma_acc_probe_kernel(float* out) {
    using namespace nvcuda;
    __shared__ float m[16 * 16];
    for (int i = threadIdx.x; i < 256; i += 32) m[i] = (float) i;
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(acc, m, 16, wmma::mem_row_major);
    for (int i = 0; i < acc.num_elements; ++i) out[threadIdx.x * 8 + i] = acc.x[i];
    if (threadIdx.x == 0) out[256] = (float) acc.num_elements;
}
void wmma_acc_layout_print() {
    float* d = nullptr;
    cudaMalloc(&d, 257 * 4);
    cudaMemset(d, 0xFF, 257 * 4);
    wmma_acc_probe_kernel<<<1, 32>>>(d);
    std::vector<float> h(257);
    cudaMemcpy(h.data(), d, 257 * 4, cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::printf("wmma accumulator: %d elements a lane\n", (int) h[256]);
    for (int lane = 0; lane < 32; ++lane) {
        std::printf("  lane %2d:", lane);
        for (int i = 0; i < 8; ++i) {
            const int v = (int) h[lane * 8 + i];
            std::printf(" (%2d,%2d)", v / 16, v % 16);
        }
        std::printf("\n");
    }
}

bool mma884_layout_check() {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> small(-4, 4), mid(-16, 16);
    std::vector<unsigned short> ha(128), hb(128);
    std::vector<float> hc(256), hd(256);
    std::vector<int> ia(128), ib(128);
    for (int i = 0; i < 128; ++i) {
        ia[i] = small(rng);
        ib[i] = small(rng);
        ha[i] = __half_as_ushort(__float2half((float) ia[i]));
        hb[i] = __half_as_ushort(__float2half((float) ib[i]));
    }
    for (int i = 0; i < 256; ++i) hc[i] = (float) mid(rng);
    unsigned short *da, *db;
    float *dc, *dd;
    cudaMalloc(&da, 256); cudaMalloc(&db, 256); cudaMalloc(&dc, 1024); cudaMalloc(&dd, 1024);
    cudaMemcpy(da, ha.data(), 256, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc.data(), 1024, cudaMemcpyHostToDevice);
    std::vector<float> hd_rr(256);
    mma884_probe_kernel<<<1, 32>>>(da, db, dc, dd, 1);
    cudaError_t err = cudaMemcpy(hd_rr.data(), dd, 1024, cudaMemcpyDeviceToHost);
    mma884_probe_kernel<<<1, 32>>>(da, db, dc, dd, 0);
    if (err == cudaSuccess) err = cudaMemcpy(hd.data(), dd, 1024, cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);
    if (err != cudaSuccess) { std::printf("mma884 layout: CUDA error %s\n", cudaGetErrorString(err)); return false; }
    // the matrices the layout says the lanes hold: product p, A[8][4], B[4][8], C[8][8]; each C cell exactly once
    float A[4][8][4] = {}, B[4][4][8] = {}, C[4][8][8] = {};
    int seen[4][8][8] = {};
    for (int lane = 0; lane < 32; ++lane) {
        const int p = (lane % 16) / 4, ab = (lane & 3) + 4 * (lane >> 4);
        for (int i = 0; i < 4; ++i) { A[p][ab][i] = (float) ia[4 * lane + i]; B[p][i][ab] = (float) ib[4 * lane + i]; }
        for (int i = 0; i < 8; ++i) {
            const int r = (lane & 1) + (i & 2) + 4 * (lane >> 4), c = (i & 4) + (lane & 2) + (i & 1);
            C[p][r][c] = hc[8 * lane + i];
            ++seen[p][r][c];
        }
    }
    int bad = 0;
    for (int p = 0; p < 4; ++p)
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 8; ++c)
                if (seen[p][r][c] != 1) ++bad;
    if (bad) { std::printf("mma884 layout: the C layout does not cover each cell once (%d cells)\n", bad); return false; }
    for (int lane = 0; lane < 32; ++lane) {
        const int p = (lane % 16) / 4;
        for (int i = 0; i < 8; ++i) {
            const int r = (lane & 1) + (i & 2) + 4 * (lane >> 4), c = (i & 4) + (lane & 2) + (i & 1);
            float v = C[p][r][c];
            for (int k = 0; k < 4; ++k) v += A[p][r][k] * B[p][k][c];
            if (v != hd[8 * lane + i]) {
                if (bad < 4) std::printf("mma884 layout: lane %d d%d = %g, the layout says %g\n", lane, i, hd[8 * lane + i], v);
                ++bad;
            }
        }
    }
    std::printf("mma884 layout: %s (%d of 256 values differ)\n", bad ? "DIFFERS" : "as mma884.cuh says", bad);
    // .row.row: lane's b_i = B[lane % 4][i + 4 (lane >= 16)] of its product; the same A and C
    float Brr[4][4][8] = {};
    for (int lane = 0; lane < 32; ++lane) {
        const int p = (lane % 16) / 4;
        for (int i = 0; i < 4; ++i) Brr[p][lane % 4][i + 4 * (lane >= 16)] = (float) ib[4 * lane + i];
    }
    int bad_rr = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const int p = (lane % 16) / 4;
        for (int i = 0; i < 8; ++i) {
            const int r = (lane & 1) + (i & 2) + 4 * (lane >> 4), c = (i & 4) + (lane & 2) + (i & 1);
            float v = C[p][r][c];
            for (int k = 0; k < 4; ++k) v += A[p][r][k] * Brr[p][k][c];
            if (v != hd_rr[8 * lane + i]) ++bad_rr;
        }
    }
    std::printf("mma884 layout, B row-major (.row.row): %s (%d of 256 values differ)\n", bad_rr ? "DIFFERS" : "as mma884.cuh says",
                bad_rr);
    return bad == 0 && bad_rr == 0;
}

double rows_pattern_us(const void* w, int type, int n_in, int n_out, int qs, void* stream) {
    unsigned* sink = nullptr;
    cudaMalloc(&sink, 4096);
    cudaStream_t s = (cudaStream_t) stream;
    const size_t rb = type == 40 ? (size_t) (n_in / 64) * 36 : (size_t) (n_in / 32) * 34;
    const int rows_per_block = 8 * 32 / qs;
    const dim3 grid((unsigned) ((n_out + rows_per_block - 1) / rows_per_block));
    auto go = [&] {
        const uint8_t* p = (const uint8_t*) w;
        if (type == 40) {
            if (qs == 1) rows_pattern_kernel<1, 9><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
            else if (qs == 2) rows_pattern_kernel<2, 9><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
            else rows_pattern_kernel<4, 9><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
        } else {
            if (qs == 1) rows_pattern_kernel<1, 17><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
            else if (qs == 2) rows_pattern_kernel<2, 17><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
            else rows_pattern_kernel<4, 17><<<grid, 256, 0, s>>>(p, rb, n_in, n_out, sink);
        }
    };
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    for (int i = 0; i < 3; ++i) go();
    cudaEventRecord(a, s);
    for (int i = 0; i < 20; ++i) go();
    cudaEventRecord(e, s);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, e);
    cudaFree(sink);
    return 1000.0 * ms / 20;
}
}  // namespace basalt::kernels::probe
