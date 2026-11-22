// tests/kernels/dense_mmvq_bench.cpp - (Lapilli, V100) dense_mmvq against iq_mmvq and MMQ at the Qwen3.8-27B's decode
// shapes, for windows of 1 to 16 rows: us per call (and GB/s of weights) for iq_mmvq on q8_1 blocks, dense_mmvq with 1,
// 2 and 4 weight rows a warp, and MMQ (its tile of 8 rows up to 8, else 16, as src/dense/forward.cpp runs it); and whether
// dense_mmvq's values are BITWISE iq_mmvq's (every row of every column).  Exit 1 on a difference, 77 without a GPU.
// `dense_mmvq_bench <shape 0-3> <T> <R> [calls]`: only dense_mmvq at that width, `calls` times (5 by default), for a
// profiler (nvprof replays each call per metric group).
#include "basalt/kernels/dense_mmvq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = basalt::prefill::mmq;
namespace k = basalt::kernels;
namespace basalt::kernels::v0 {   // tests/kernels/dense_mmvq_v0.cu: the kernel as of dd695013, the reference for the speed
void dense_mmvq(int ggml_type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream);
void dense_mmvq_set_rows_per_warp(int r);
}
namespace basalt::kernels::probe {   // tests/kernels/dense_bw_probe.cu: the bytes read without the products
double stream_us(const void* w, size_t bytes, void* stream);
double nvfp4_pattern_us(const void* w, int n_in, int n_out, int rows_per_warp, void* stream);
}

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device 0: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(3);
    struct Shape { int type; int rows, cols; const char* name; };
    const Shape shapes[] = {{40, 17408, 5120, "NVFP4 gate/up 17408x5120"}, {40, 5120, 17408, "NVFP4 down 5120x17408"},
                            {8, 10240, 5120, "Q8_0 GDN qkv 10240x5120"}, {8, 5120, 6144, "Q8_0 out 5120x6144"}};
    const int widths[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const int only = argc >= 4 ? std::atoi(argv[1]) : -1, only_t = argc >= 4 ? std::atoi(argv[2]) : 0,
              only_r = argc >= 4 ? std::atoi(argv[3]) : 0, only_n = argc >= 5 ? std::atoi(argv[4]) : 5;
    constexpr int kMax = 16;
    mmq::Context ctx;
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);
    auto time = [&](auto f) {
        for (int i = 0; i < 3; ++i) f();
        cudaEventRecord(a, st);
        for (int i = 0; i < 40; ++i) f();
        cudaEventRecord(e, st);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, e);
        return 1000.0 * ms / 40;
    };
    int bad = 0, shape_i = -1;
    for (const Shape& sh : shapes) {
        if (only >= 0 && ++shape_i != only) continue;
        const size_t wb = mmq::matrix_bytes(sh.type, sh.rows, sh.cols);
        std::vector<uint8_t> w(wb);
        for (auto& b : w) b = (uint8_t) rng();
        if (sh.type == 8)   // sane fp16 block scales
            for (size_t i = 0; i < wb; i += 34) { w[i] = 0x00; w[i + 1] = 0x20; }
        else   // UE4M3 scales without the sign bit or NaN
            for (size_t i = 0; i < wb; i += 36) for (int j = 0; j < 4; ++j) w[i + j] = (uint8_t) (0x30 + (rng() & 15));
        std::vector<float> x((size_t) kMax * sh.cols);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        void *d_w, *d_xq, *d_xs, *d_xm8, *d_xm16;
        float *d_x, *d_yo, *d_yn, *d_ym;
        int32_t *d_ids, *d_b;
        cudaMalloc(&d_w, wb);
        cudaMemcpy(d_w, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMalloc(&d_x, x.size() * 4);
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_xq, (size_t) kMax * (sh.cols / 32) * 36);
        cudaMalloc(&d_xs, k::dense_q8_bytes(sh.cols, kMax));
        cudaMalloc(&d_xm8, mmq::q8_bytes(8, sh.cols));
        cudaMalloc(&d_xm16, mmq::q8_bytes(kMax, sh.cols));
        const size_t ny = (size_t) kMax * sh.rows;
        cudaMalloc(&d_yo, ny * 4);
        cudaMalloc(&d_yn, ny * 4);
        cudaMalloc(&d_ym, ny * 4);
        cudaMalloc(&d_ids, 64 * 4);
        cudaMalloc(&d_b, 8);
        mmq::iota(d_ids, 64, st);
        k::quantize_q8_1_rows(d_x, kMax, sh.cols, d_xq, st);
        k::dense_quantize_q8(d_x, kMax, sh.cols, d_xs, st);
        mmq::quantize(d_x, nullptr, d_xm8, sh.type, sh.cols, sh.cols, 8, st);
        mmq::quantize(d_x, nullptr, d_xm16, sh.type, sh.cols, sh.cols, kMax, st);
        mmq::load(sh.type, sh.rows);
        if (only >= 0) {   // the profiler's mode: dense_mmvq alone
            k::dense_quantize_q8(d_x, only_t, sh.cols, d_xs, st);   // the layout is the window's (T rows a chunk)
            k::dense_mmvq_set_rows_per_warp(only_r);
            for (int i = 0; i < only_n; ++i) k::dense_mmvq(sh.type, d_w, d_xs, d_yn, sh.cols, sh.rows, only_t, st);
            cudaStreamSynchronize(st);
            std::printf("%s T=%d R=%d: %d calls\n", sh.name, only_t, only_r, only_n);
            cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_xs); cudaFree(d_xm8); cudaFree(d_xm16);
            cudaFree(d_yo); cudaFree(d_yn); cudaFree(d_ym); cudaFree(d_ids); cudaFree(d_b);
            continue;
        }
        auto run_mmq = [&](int T, float* y) {
            const int tile = T <= 8 ? 8 : kMax;
            const int32_t b[2] = {0, T};
            cudaMemcpyAsync(d_b, b, 8, cudaMemcpyHostToDevice, st);
            mmq::Product p;
            p.w = d_w; p.type = sh.type; p.w_rows = sh.rows; p.w_cols = sh.cols; p.expert_bytes = wb; p.n = 1;
            p.xq = tile == 8 ? d_xm8 : d_xm16; p.bounds = d_b; p.ids = d_ids; p.total_rows = tile; p.max_rows = tile;
            p.dst = y; p.ld_dst = sh.rows; p.opt_rows = tile;
            ctx.run(p, st);
        };
        // the quantizers, 16 rows
        const double tq_old = time([&] { k::quantize_q8_1_rows(d_x, kMax, sh.cols, d_xq, st); });
        const double tq_new = time([&] { k::dense_quantize_q8(d_x, kMax, sh.cols, d_xs, st); });
        std::printf("\n%s (%.1f MB of weights)  quantize 16 rows: q8_1 blocks %.1f us, dense layout %.1f us\n", sh.name,
                    wb / 1e6, tq_old, tq_new);
        {   // the bytes alone: a plain 16-byte stream, and (NVFP4) dense_mmvq's one-column access pattern
            const double ts = k::probe::stream_us(d_w, wb, st);
            if (sh.type == 40)
                std::printf("   read only: 16-byte stream %.0f GB/s; NVFP4 pattern R1 %.0f, R2 %.0f, R4 %.0f GB/s\n", wb / ts / 1e3,
                            wb / k::probe::nvfp4_pattern_us(d_w, sh.cols, sh.rows, 1, st) / 1e3,
                            wb / k::probe::nvfp4_pattern_us(d_w, sh.cols, sh.rows, 2, st) / 1e3,
                            wb / k::probe::nvfp4_pattern_us(d_w, sh.cols, sh.rows, 4, st) / 1e3);
            else
                std::printf("   read only: 16-byte stream %.0f GB/s\n", wb / ts / 1e3);
        }
        std::printf("   T   iq_mmvq |  v0 R1   v0 R2   v0 R4 |   R1      R2      R4   |  S R1   S R2   S R4  |   mmq   | best GB/s | bitwise\n");
        std::vector<float> yo(ny), yn(ny);
        for (int T : widths) {
            const double to = time([&] { k::iq_mmvq(sh.type, d_w, d_xq, d_yo, sh.cols, sh.rows, T, st); });
            double tn[15] = {};
            const char* same[15] = {};
            cudaMemcpy(yo.data(), d_yo, (size_t) T * sh.rows * 4, cudaMemcpyDeviceToHost);
            k::dense_quantize_q8(d_x, T, sh.cols, d_xs, st);   // the layout is the window's (T rows a chunk)
            for (int r : {1, 2, 4, 11, 12, 14}) {
                k::dense_mmvq_set_rows_per_warp(r);
                cudaMemset(d_yn, 0xFF, ny * 4);
                tn[r] = time([&] { k::dense_mmvq(sh.type, d_w, d_xs, d_yn, sh.cols, sh.rows, T, st); });
                cudaMemcpy(yn.data(), d_yn, (size_t) T * sh.rows * 4, cudaMemcpyDeviceToHost);
                long diff = 0, first = -1;
                for (size_t i = 0; i < (size_t) T * sh.rows; ++i)
                    if (std::memcmp(&yo[i], &yn[i], 4) != 0) { if (first < 0) first = (long) i; ++diff; }
                same[r] = diff == 0 ? "bitwise" : "DIFFERS";
                if (diff) {
                    ++bad;
                    std::printf("   T=%d variant %d: %ld values differ, first at column %ld row %ld: %.9g against %.9g\n", T, r,
                                diff, first / sh.rows, first % sh.rows, (double) yn[(size_t) first], (double) yo[(size_t) first]);
                }
            }
            k::dense_mmvq_set_rows_per_warp(0);
            double t0[5] = {0, 0, 0, 0, 0};
            for (int r = 1; r <= 4; r *= 2) {
                k::v0::dense_mmvq_set_rows_per_warp(r);
                t0[r] = time([&] { k::v0::dense_mmvq(sh.type, d_w, d_xs, d_yn, sh.cols, sh.rows, T, st); });
            }
            k::v0::dense_mmvq_set_rows_per_warp(0);
            const double tm = time([&] { run_mmq(T, d_ym); });
            const double best = std::min({tn[1], tn[2], tn[4], tn[11], tn[12], tn[14]});
            bool all = true;
            for (int r : {1, 2, 4, 11, 12, 14}) all = all && std::strcmp(same[r], "bitwise") == 0;
            std::printf("  %2d %8.1f | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f | %6.1f %6.1f %6.1f | %7.1f | %6.0f | %s\n", T, to,
                        t0[1], t0[2], t0[4], tn[1], tn[2], tn[4], tn[11], tn[12], tn[14], tm, wb / best / 1e3,
                        all ? "all bitwise" : "DIFFERS");
        }
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_xs); cudaFree(d_xm8); cudaFree(d_xm16);
        cudaFree(d_yo); cudaFree(d_yn); cudaFree(d_ym); cudaFree(d_ids); cudaFree(d_b);
    }
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    std::printf("\n%s\n", bad ? "DENSE-MMVQ DIFFERS" : "DENSE-MMVQ BITWISE OK");
    return bad ? 1 : 0;
}
