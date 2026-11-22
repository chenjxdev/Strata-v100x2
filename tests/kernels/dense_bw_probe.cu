// tests/kernels/dense_bw_probe.cu - (Lapilli, V100) what reading a matrix's bytes costs without the products:
// (1) a plain stream of 16-byte loads over the buffer (the card's practical bandwidth), (2) dense_mmvq's access to NVFP4
// weights at one column (per lane and call: two 32-bit code words and the scale byte of its sub-block, the next call's
// words loaded first), the products replaced by a XOR of the words.  For dense_mmvq_bench: if (2) reads as fast as (1),
// the one-column kernel's gap to the bandwidth is its arithmetic and latencies, not its access pattern.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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
