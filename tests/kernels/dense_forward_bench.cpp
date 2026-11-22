// tests/kernels/dense_forward_bench.cpp - (Lapilli, V100) dense_mmvq's variants over the products of a WHOLE
// forward of the Qwen3.8-27B, as src/dense/forward.cpp runs them on dense_mmvq's route: 48 GDN layers (qkv + z in one
// launch, out added into the residual, gate + up in one launch, down added), 16 attention layers (q + k + v, o added,
// the same MLP), then the Q8_0 head (248 320 rows).  Distinct buffers per matrix shape, read in a forward's order (the
// MLP's 50 MB flush the L2 between two uses of one).  For every width 1 to 16 and every variant (1, 2, 4 rows a warp,
// direct; 11, 12, 14, staged): ms per forward of the NVFP4 products and of the head.  The numbers dense_mmvq's table is
// chosen from.  Exit 77 without a GPU.
#include "basalt/kernels/dense_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace k = basalt::kernels;
namespace basalt::kernels::probe {
void fill_weights(void* p, size_t bytes, int type, uint32_t seed, void* stream);
void fill_floats(float* p, size_t n, uint32_t seed, void* stream);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    cudaStream_t st;
    cudaStreamCreate(&st);
    constexpr int H = 5120, FF = 17408, MAXT = k::kDenseMaxRows;
    struct M { int rows, cols, type; void* w; float* scale; };
    auto mat = [&](int rows, int cols, int type, uint32_t seed) {
        M m{rows, cols, type, nullptr, nullptr};
        const size_t b = type == 40 ? (size_t) rows * (cols / 64) * 36 : (size_t) rows * (cols / 32) * 34;
        cudaMalloc(&m.w, b);
        k::probe::fill_weights(m.w, b, type, seed, st);
        if (type == 40) {
            const float sc = 0.5f;
            cudaMalloc(&m.scale, 4);
            cudaMemcpy(m.scale, &sc, 4, cudaMemcpyHostToDevice);
        }
        return m;
    };
    M gate = mat(FF, H, 40, 1), up = mat(FF, H, 40, 2), down = mat(H, FF, 40, 3), qkv = mat(10240, H, 40, 4),
      z = mat(6144, H, 40, 5), out = mat(H, 6144, 40, 6), q = mat(12288, H, 40, 7), kk = mat(1024, H, 40, 8),
      v = mat(1024, H, 40, 9), o = mat(H, 6144, 40, 10), head = mat(248320, H, 8, 11);
    // activations (each input width's layout for MAXT rows), outputs, the residual
    float* xf = nullptr;
    cudaMalloc(&xf, (size_t) MAXT * FF * 4);
    k::probe::fill_floats(xf, (size_t) MAXT * FF, 12, st);
    void *xh, *xff, *x6;
    cudaMalloc(&xh, k::dense_q8_bytes(H, MAXT));
    cudaMalloc(&xff, k::dense_q8_bytes(FF, MAXT));
    cudaMalloc(&x6, k::dense_q8_bytes(6144, MAXT));
    float *y1, *y2, *y3;
    cudaMalloc(&y1, (size_t) MAXT * 248320 * 4);
    cudaMalloc(&y2, (size_t) MAXT * FF * 4);
    cudaMalloc(&y3, (size_t) MAXT * 1024 * 4);
    uint16_t* resid = nullptr;
    cudaMalloc(&resid, (size_t) MAXT * H * 2);
    cudaMemset(resid, 0, (size_t) MAXT * H * 2);
    cudaStreamSynchronize(st);
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    auto time = [&](auto f, int reps) {
        f();
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(e, st);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, e);
        return ms / reps;
    };
    const int variants[] = {1, 2, 4, 11, 12, 14};
    std::printf("ms per forward: the NVFP4 products (48 GDN + 16 attention layers) | the Q8_0 head\n");
    std::printf("  T |   R1    R2    R4    S1    S2    S4 | best || head R1  R2    R4    S1    S2    S4 | best\n");
    for (int T = 1; T <= MAXT; ++T) {
        k::dense_quantize_q8(xf, T, H, xh, st);
        k::dense_quantize_q8(xf, T, FF, xff, st);
        k::dense_quantize_q8(xf, T, 6144, x6, st);
        double tn[6], th[6];
        for (int iv = 0; iv < 6; ++iv) {
            k::dense_mmvq_set_rows_per_warp(variants[iv]);
            auto mlp = [&] {
                const k::DenseSeg gu[2] = {{gate.w, gate.rows, nullptr, y2}, {up.w, up.rows, nullptr, y1}};
                k::dense_mmvq_segs(40, gu, 2, xh, H, T, st);
                k::dense_mmvq_add_bf16(40, down.w, down.scale, xff, resid, FF, H, T, st);
            };
            auto forward = [&] {
                for (int l = 0; l < 64; ++l) {
                    if (l % 4 == 3) {   // attention
                        const k::DenseSeg s3[3] = {{q.w, q.rows, q.scale, y1}, {kk.w, kk.rows, kk.scale, y3}, {v.w, v.rows, v.scale, y2}};
                        k::dense_mmvq_segs(40, s3, 3, xh, H, T, st);
                        k::dense_mmvq_add_bf16(40, o.w, o.scale, x6, resid, 6144, H, T, st);
                    } else {   // GDN
                        const k::DenseSeg s2[2] = {{qkv.w, qkv.rows, qkv.scale, y1}, {z.w, z.rows, z.scale, y2}};
                        k::dense_mmvq_segs(40, s2, 2, xh, H, T, st);
                        k::dense_mmvq_add_bf16(40, out.w, out.scale, x6, resid, 6144, H, T, st);
                    }
                    mlp();
                }
            };
            tn[iv] = time(forward, 3);
            th[iv] = time([&] {
                const k::DenseSeg hs{head.w, head.rows, nullptr, y1};
                k::dense_mmvq_segs(8, &hs, 1, xh, H, T, st);
            }, 5);
        }
        k::dense_mmvq_set_rows_per_warp(0);
        const double tt = time([&] {
            for (int l = 0; l < 64; ++l) {
                if (l % 4 == 3) {
                    const k::DenseSeg s3[3] = {{q.w, q.rows, q.scale, y1}, {kk.w, kk.rows, kk.scale, y3}, {v.w, v.rows, v.scale, y2}};
                    k::dense_mmvq_segs(40, s3, 3, xh, H, T, st);
                    k::dense_mmvq_add_bf16(40, o.w, o.scale, x6, resid, 6144, H, T, st);
                } else {
                    const k::DenseSeg s2[2] = {{qkv.w, qkv.rows, qkv.scale, y1}, {z.w, z.rows, z.scale, y2}};
                    k::dense_mmvq_segs(40, s2, 2, xh, H, T, st);
                    k::dense_mmvq_add_bf16(40, out.w, out.scale, x6, resid, 6144, H, T, st);
                }
                const k::DenseSeg gu[2] = {{gate.w, gate.rows, nullptr, y2}, {up.w, up.rows, nullptr, y1}};
                k::dense_mmvq_segs(40, gu, 2, xh, H, T, st);
                k::dense_mmvq_add_bf16(40, down.w, down.scale, xff, resid, FF, H, T, st);
            }
        }, 3);
        int bn = 0, bh = 0;
        for (int iv = 1; iv < 6; ++iv) { if (tn[iv] < tn[bn]) bn = iv; if (th[iv] < th[bh]) bh = iv; }
        std::printf(" %2d | %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %-3d %5.2f (table %5.2f) || %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %-3d %5.2f\n", T,
                    tn[0], tn[1], tn[2], tn[3], tn[4], tn[5], variants[bn], tn[bn], tt, th[0], th[1], th[2], th[3], th[4], th[5],
                    variants[bh], th[bh]);
    }
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    std::printf("DENSE-FORWARD-BENCH OK\n");
    return 0;
}
