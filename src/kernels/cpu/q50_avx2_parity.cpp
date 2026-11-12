// src/kernels/cpu/q50_avx2_parity.cpp - q50_256_down_rows (the AVX-2 Q5_0 down rows, several tokens at once) against
// ggml-cpu's ggml_vec_dot_q5_0_q8_0, bit for bit, at 1..8 tokens on random blocks, and both timed on one thread.
//
//     build/q50_avx2_parity           (x86 with AVX2; no model)
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    if (!cpu::cpu_avx2_ok()) { std::printf("q50_avx2_parity: needs AVX2, skipped\n"); return 77; }
    ggml_cpu_init();
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(GGML_TYPE_Q5_0);
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    constexpr int n = 640, rows = 2560;
    const size_t rb = ggml_row_size(GGML_TYPE_Q5_0, n), bb = rb / (n / 32);
    uint32_t seed = 12345u;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    std::vector<uint8_t> w((size_t) rows * rb);
    for (auto& b : w) b = (uint8_t) rnd();
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < n / 32; ++k) {   // a finite fp16 scale, 2^-12 .. 2^-6, either sign
            const uint16_t h = (uint16_t) (((rnd() & 1) << 15) | ((3 + rnd() % 7) << 10) | (rnd() & 1023));
            std::memcpy(&w[(size_t) r * rb + (size_t) k * bb], &h, 2);
        }
    constexpr int NT = 8;
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
    const void* hp[NT];
    for (int t = 0; t < NT; ++t) {
        std::vector<float> x(n);
        for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.003f;
        t8->from_float(x.data(), act[(size_t) t].data(), n);
        hp[t] = act[(size_t) t].data();
    }
    std::vector<float> want((size_t) NT * rows), got((size_t) NT * rows);
    float* wp[NT];
    float* gp[NT];
    for (int t = 0; t < NT; ++t) { wp[t] = &want[(size_t) t * rows]; gp[t] = &got[(size_t) t * rows]; }
    int fails = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        std::fill(got.begin(), got.end(), -1.f);
        cpu::q50_256_down_rows(w.data(), rb, n, hp, nt, gp, 0, rows);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r) tr->vec_dot(n, &wp[t][r], 0, w.data() + (size_t) r * rb, 0, hp[t], 0, 1);
        const bool ok = std::memcmp(got.data(), want.data(), (size_t) nt * rows * sizeof(float)) == 0;
        if (!ok) ++fails;
        // timing: the whole matrix (1.1 MB, out of L2) a pass, best of 20
        auto best = [&](auto&& fn) {
            double b = 1e30;
            for (int it = 0; it < 20; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                b = std::min(b, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            return b;
        };
        const double tg = best([&] {
            for (int r = 0; r < rows; ++r)
                for (int t = 0; t < nt; ++t) tr->vec_dot(n, &wp[t][r], 0, w.data() + (size_t) r * rb, 0, hp[t], 0, 1);
        });
        const double to = best([&] { cpu::q50_256_down_rows(w.data(), rb, n, hp, nt, gp, 0, rows); });
        std::printf("Q5_0 down %d x %d, %d token(s): %s  ggml %.0f us, AVX-2 rows %.0f us (%.2fx)\n", rows, n, nt,
                    ok ? "bit-equal" : "DIFFER", tg, to, tg / to);
    }
    std::printf("q50_avx2_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
