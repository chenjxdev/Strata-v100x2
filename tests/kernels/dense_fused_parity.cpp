// tests/kernels/dense_fused_parity.cpp - (Lapilli, V100) the fused steps of dense_mmvq against the steps they replace,
// BITWISE, for windows of 1 to 16 rows at the Qwen3.8-27B's shapes:
//   dense_norm_quantize      = rms_norm_rows (BF16 rows) then dense_quantize_q8 (xn and the activations);
//   dense_swiglu_quantize    = swiglu_scaled then dense_quantize_q8 (with and without the scales);
//   dense_mmvq_segs          = one dense_mmvq then scale_values per matrix (q, k, v; qkv, z; gate, up);
//   dense_mmvq_add_bf16      = dense_mmvq into a buffer then add_scaled into the BF16 residual (NVFP4 with its scale,
//                              Q8_0 without).
// Exit 1 on a difference, 77 without a GPU.
#include "basalt/kernels/dense_mmvq.hpp"
#include "basalt/kernels/dense_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

std::mt19937 rng(11);

size_t weight_bytes(int type, int rows, int cols) { return type == 40 ? (size_t) rows * (cols / 64) * 36 : (size_t) rows * (cols / 32) * 34; }

void* random_weights(int type, int rows, int cols) {
    const size_t wb = weight_bytes(type, rows, cols);
    std::vector<uint8_t> w(wb);
    for (auto& b : w) b = (uint8_t) rng();
    if (type == 8)
        for (size_t i = 0; i < wb; i += 34) { w[i] = 0x00; w[i + 1] = 0x20; }
    else
        for (size_t i = 0; i < wb; i += 36) for (int j = 0; j < 4; ++j) w[i + j] = (uint8_t) (0x30 + (rng() & 15));
    void* d = nullptr;
    cudaMalloc(&d, wb);
    cudaMemcpy(d, w.data(), wb, cudaMemcpyHostToDevice);
    return d;
}
float* device_floats(size_t n, float sigma) {
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (auto& v : h) v = nd(rng);
    float* d = nullptr;
    cudaMalloc(&d, n * 4);
    cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice);
    return d;
}
float* device_scalar(float v) {
    float* d = nullptr;
    cudaMalloc(&d, 4);
    cudaMemcpy(d, &v, 4, cudaMemcpyHostToDevice);
    return d;
}
uint16_t bf16_of(float f) {   // round to nearest even, as __float2bfloat16_rn for finite values
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
uint16_t* device_bf16(size_t n, float sigma) {
    std::vector<uint16_t> h(n);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (auto& v : h) v = bf16_of(nd(rng));
    uint16_t* d = nullptr;
    cudaMalloc(&d, n * 2);
    cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice);
    return d;
}
// the first difference of two device buffers, -1 if none
long first_diff(const void* a, const void* b, size_t bytes) {
    std::vector<uint8_t> ha(bytes), hb(bytes);
    cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < bytes; ++i)
        if (ha[i] != hb[i]) return (long) i;
    return -1;
}

int bad = 0;
void report(const char* what, int T, long d) {
    if (d < 0) return;
    ++bad;
    std::printf("  DIFFERS: %s, T=%d, first byte %ld\n", what, T, d);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    cudaStream_t st;
    cudaStreamCreate(&st);
    constexpr int H = 5120, FF = 17408, MAXT = k::kDenseMaxRows;
    const float eps = 1e-6f;
    // inputs shared by the products: x of the norms (BF16 residual rows) and its quantized rows for each T
    uint16_t* resid0 = device_bf16((size_t) MAXT * H, 2.0f);
    float* wnorm_w = device_floats(H, 0.3f);
    float* xn_ref = nullptr; float* xn_new = nullptr;
    cudaMalloc(&xn_ref, (size_t) MAXT * H * 4);
    cudaMalloc(&xn_new, (size_t) MAXT * H * 4);
    const size_t qb_h = k::dense_q8_bytes(H, MAXT), qb_ff = k::dense_q8_bytes(FF, MAXT);
    void *q_ref, *q_new;
    cudaMalloc(&q_ref, qb_ff);
    cudaMalloc(&q_new, qb_ff);
    // matrices: NVFP4 q (12288), k, v (1024), qkv (10240), z (6144), gate/up (17408) of H inputs; down (5120) of FF inputs;
    // Q8_0: a head-like 8192 rows of H inputs
    struct M { int type, rows, cols; void* w; float* scale; };
    M mq{40, 12288, H, nullptr, nullptr}, mk{40, 1024, H, nullptr, nullptr}, mv{40, 1024, H, nullptr, nullptr};
    M mqkv{40, 10240, H, nullptr, nullptr}, mz{40, 6144, H, nullptr, nullptr};
    M mg{40, FF, H, nullptr, nullptr}, mu{40, FF, H, nullptr, nullptr}, md{40, H, FF, nullptr, nullptr};
    M mh{8, 8192, H, nullptr, nullptr};
    for (M* m : {&mq, &mk, &mv, &mqkv, &mz, &mg, &mu, &md, &mh}) {
        m->w = random_weights(m->type, m->rows, m->cols);
        m->scale = m->type == 40 ? device_scalar(0.37f + 0.01f * (float) (rng() % 50)) : nullptr;
    }
    float *ya, *yb, *yc, *za, *zb, *zc;
    const size_t ymax = (size_t) MAXT * FF;
    for (float** p : {&ya, &yb, &yc, &za, &zb, &zc}) cudaMalloc(p, ymax * 4);
    float* g = device_floats((size_t) MAXT * FF, 3.0f);
    float* u = device_floats((size_t) MAXT * FF, 2.0f);
    float* h = nullptr;
    cudaMalloc(&h, (size_t) MAXT * FF * 4);
    uint16_t *r_ref, *r_new;
    cudaMalloc(&r_ref, (size_t) MAXT * H * 2);
    cudaMalloc(&r_new, (size_t) MAXT * H * 2);
    float* sg = device_scalar(0.81f);
    float* su = device_scalar(1.27f);

    for (int T = 1; T <= MAXT; ++T) {
        // 1. the norm and its quantizer
        cudaMemset(q_ref, 0xAB, qb_ff); cudaMemset(q_new, 0xAB, qb_ff);
        k::rms_norm_rows(resid0, wnorm_w, xn_ref, T, H, eps, st);
        k::dense_quantize_q8(xn_ref, T, H, q_ref, st);
        k::dense_norm_quantize(resid0, wnorm_w, xn_new, eps, T, H, q_new, st);
        cudaStreamSynchronize(st);
        report("norm: xn", T, first_diff(xn_ref, xn_new, (size_t) T * H * 4));
        report("norm: activations", T, first_diff(q_ref, q_new, k::dense_q8_bytes(H, T)));
        // the activations of H inputs for the products below (q_ref, T rows)
        void* xh = q_ref;
        // 2. one launch per group against one product + scale_values each
        struct G { const char* name; std::vector<M*> ms; };
        for (const G& grp : {G{"q k v", {&mq, &mk, &mv}}, G{"qkv z", {&mqkv, &mz}}, G{"gate up", {&mg, &mu}}, G{"head Q8_0", {&mh}}}) {
            float* yref[3] = {ya, yb, yc};
            float* ynew[3] = {za, zb, zc};
            k::DenseSeg segs[3];
            int ns = 0;
            for (M* m : grp.ms) {
                k::dense_mmvq(m->type, m->w, xh, yref[ns], H, m->rows, T, st);
                k::scale_values(yref[ns], m->scale, (int64_t) T * m->rows, st);
                cudaMemset(ynew[ns], 0xFF, (size_t) T * m->rows * 4);
                segs[ns] = {m->w, m->rows, m->scale, ynew[ns]};
                ++ns;
            }
            k::dense_mmvq_segs(grp.ms[0]->type, segs, ns, xh, H, T, st);
            cudaStreamSynchronize(st);
            for (int i = 0; i < ns; ++i) report(grp.name, T, first_diff(yref[i], ynew[i], (size_t) T * grp.ms[i]->rows * 4));
        }
        // 3. SwiGLU and down's quantizer (with the scales, then without)
        for (int with = 1; with >= 0; --with) {
            cudaMemset(q_ref, 0xAB, qb_ff); cudaMemset(q_new, 0xAB, qb_ff);
            k::swiglu_scaled(g, u, with ? sg : nullptr, with ? su : nullptr, h, T, FF, FF, st);
            k::dense_quantize_q8(h, T, FF, q_ref, st);
            k::dense_swiglu_quantize(g, u, with ? sg : nullptr, with ? su : nullptr, T, FF, q_new, st);
            cudaStreamSynchronize(st);
            report(with ? "swiglu (scaled)" : "swiglu", T, first_diff(q_ref, q_new, k::dense_q8_bytes(FF, T)));
        }
        // 4. down (NVFP4, its scale) and the Q8_0 head-like matrix as an output projection (no scale) added into the residual
        k::dense_quantize_q8(h, T, FF, q_ref, st);   // down's activations: the last h (no scales)
        cudaMemcpyAsync(r_ref, resid0, (size_t) T * H * 2, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(r_new, resid0, (size_t) T * H * 2, cudaMemcpyDeviceToDevice, st);
        k::dense_mmvq(md.type, md.w, q_ref, ya, FF, md.rows, T, st);
        k::add_scaled(r_ref, ya, md.scale, (int64_t) T * md.rows, st);
        k::dense_mmvq_add_bf16(md.type, md.w, md.scale, q_ref, r_new, FF, md.rows, T, st);
        cudaStreamSynchronize(st);
        report("down + residual", T, first_diff(r_ref, r_new, (size_t) T * H * 2));
        k::dense_quantize_q8(xn_ref, T, H, q_new, st);
        M mo{8, H, H, nullptr, nullptr};
        mo.w = mh.w;   // the first 5120 rows of the Q8_0 matrix, 5120 inputs: an output projection without a scale
        cudaMemcpyAsync(r_ref, resid0, (size_t) T * H * 2, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(r_new, resid0, (size_t) T * H * 2, cudaMemcpyDeviceToDevice, st);
        k::dense_mmvq(mo.type, mo.w, q_new, ya, H, mo.rows, T, st);
        k::add_scaled(r_ref, ya, nullptr, (int64_t) T * mo.rows, st);
        k::dense_mmvq_add_bf16(mo.type, mo.w, nullptr, q_new, r_new, H, mo.rows, T, st);
        cudaStreamSynchronize(st);
        report("Q8_0 output projection + residual", T, first_diff(r_ref, r_new, (size_t) T * H * 2));
        std::printf("T=%2d done\n", T);
    }
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    std::printf("%s\n", bad ? "DENSE-FUSED DIFFERS" : "DENSE-FUSED BITWISE OK");
    return bad ? 1 : 0;
}
