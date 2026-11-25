// route_mmvf_v2_parity - the router's fast kernels (route_v2 and the exact-pairs bf16 products) against the generic ones
// (BASALT_ROUTE_V1=1, BASALT_MMVF_V1=1), bit for bit, on router-shaped inputs (2560 -> 512, 10 experts): 1 to 8 tokens,
// random values, ties (repeated logits) and tiny gaps; also the shape of the shared expert's gate (2560 -> 1).
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/models/qwen4exp/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
uint16_t bf16(float f) { uint32_t i; std::memcpy(&i, &f, 4); return (uint16_t) ((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16); }
void mode(const char* name, const char* v) {
#ifdef _WIN32
    _putenv_s(name, v);
#else
    setenv(name, v, 1);
#endif
}
template <class T> std::vector<T> get(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "get");
    return h;
}
}  // namespace

int main() {
    const int N = 2560, E = 512, KK = 10;
    int bad = 0, cases = 0;
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    for (int n_out : {E, 1}) {
        std::vector<uint16_t> w((size_t) n_out * N);
        for (auto& v : w) v = bf16(nd(rng) * 0.05f);
        uint16_t* dw;
        ck(cudaMalloc(&dw, w.size() * 2), "w");
        ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "up w");
        for (int T = 1; T <= 8; ++T) {
            std::vector<float> x((size_t) T * N);
            for (auto& v : x) v = nd(rng);
            float *dx, *y1, *y2;
            ck(cudaMalloc(&dx, x.size() * 4), "x");
            ck(cudaMalloc(&y1, (size_t) T * n_out * 4), "y1");
            ck(cudaMalloc(&y2, (size_t) T * n_out * 4), "y2");
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up x");
            mode("BASALT_MMVF_V1", "1");
            if (T == 1) K::bf16_gemv_fp32_mmvf(dx, dw, y1, N, n_out, st);
            else K::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y1, n_out, N, n_out, T, st);
            mode("BASALT_MMVF_V1", "0");
            if (T == 1) K::bf16_gemv_fp32_mmvf(dx, dw, y2, N, n_out, st);
            else K::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y2, n_out, N, n_out, T, st);
            ck(cudaStreamSynchronize(st), "mmvf");
            const auto a = get(y1, (size_t) T * n_out), b = get(y2, (size_t) T * n_out);
            ++cases;
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) { std::printf("FAIL mmvf n_out %d T %d\n", n_out, T); ++bad; }
            cudaFree(dx); cudaFree(y1); cudaFree(y2);
        }
        cudaFree(dw);
    }
    // route: random logits, ties (repeated values), tiny gaps
    for (int set = 0; set < 3; ++set)
        for (int T = 1; T <= 8; ++T) {
            std::vector<float> l((size_t) T * E);
            for (size_t i = 0; i < l.size(); ++i) {
                const float v = nd(rng) * 2.0f;
                l[i] = set == 0 ? v : set == 1 ? (float) ((int) (v * 2.0f)) * 0.5f : 1.0f + (float) (i % 7) * 1e-7f;
            }
            float *dl, *w1, *w2;
            int32_t *i1, *i2;
            ck(cudaMalloc(&dl, l.size() * 4), "l");
            ck(cudaMalloc(&w1, (size_t) T * KK * 4), "w1"); ck(cudaMalloc(&w2, (size_t) T * KK * 4), "w2");
            ck(cudaMalloc(&i1, (size_t) T * KK * 4), "i1"); ck(cudaMalloc(&i2, (size_t) T * KK * 4), "i2");
            ck(cudaMemcpy(dl, l.data(), l.size() * 4, cudaMemcpyHostToDevice), "up l");
            mode("BASALT_ROUTE_V1", "1");
            if (T == 1) K::native_router_top10(dl, i1, w1, st); else K::native_router_top10_multi(dl, i1, w1, T, st);
            mode("BASALT_ROUTE_V1", "0");
            if (T == 1) K::native_router_top10(dl, i2, w2, st); else K::native_router_top10_multi(dl, i2, w2, T, st);
            ck(cudaStreamSynchronize(st), "route");
            const auto a = get(i1, (size_t) T * KK), b = get(i2, (size_t) T * KK);
            const auto c = get(w1, (size_t) T * KK), d = get(w2, (size_t) T * KK);
            ++cases;
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0 || std::memcmp(c.data(), d.data(), c.size() * 4) != 0) {
                std::printf("FAIL route set %d T %d\n", set, T);
                ++bad;
            }
            cudaFree(dl); cudaFree(w1); cudaFree(w2); cudaFree(i1); cudaFree(i2);
        }
    std::printf("route_mmvf_v2_parity: %d cases, %d different: %s\n", cases, bad, bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
