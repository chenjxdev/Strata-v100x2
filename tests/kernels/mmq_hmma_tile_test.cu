// tests/kernels/mmq_hmma_tile_test.cu - (Lapilli, V100) mmq_hmma_funcs.cuh against the dp4a path of the NVFP4
// prompt products, BITWISE, on the same tiles: one block runs MMQ's tile sequence (the dp4a loader, the y tile copied
// in two halves, the dot) for both dots side by side, then both write-backs.  Synthetic NVFP4 rows (UE4M3 scales over the
// whole range, subnormal and zero included) and q8_1 columns (codes -128..127, scales over binades), tile widths 16 to
// 128, 1 and 20 iterations (256 and 5120 values).  Then the time of each dot alone over 640 row tiles (a 81920 x 5120
// matrix against 128 columns).  Then the REAL entry, mul_mat_q_case (the tile choice, stream-k and its fixup, fallback rows):
// the same products with mmq.cuh's variant 1 off and on, bitwise, and their times at the 27B's shapes.  Exit 1 on a
// difference, 77 without a GPU.
#include "../../src/prefill/mmq/mmq.cuh"
#include "../../src/prefill/mmq/mmq_hmma_funcs.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr ggml_type kT = GGML_TYPE_NVFP4;

// MODE 0: both dots, both outputs; 1: the dp4a dot alone; 2: the tensor-core dot alone; 3: the loads alone; 4: the tensor-core
// dot with the fp16 copy of the y tile (mmq_hmma_y_to_half), in MODE 0 compared too (out_b then holds it)
template <int J, int MODE>
__global__ void __launch_bounds__(256) tile_kernel(const char * x, const int * y, float * out_a, float * out_b, int stride_row_x,
                                                   int n_iter, int ncols_y) {
    constexpr bool fallback  = false;
    constexpr int  warp_size = 32, nwarps = 8;
    constexpr int  qk        = ggml_cuda_type_traits<kT>::qk;
    constexpr int  I         = ggml_cuda_mmq_get_I(kT, J, fallback);
    constexpr int  ITER_K    = ggml_cuda_mmq_get_K_vram(kT, J, fallback);
    constexpr int  bpi       = ITER_K / qk;
    constexpr int  sz        = sizeof(block_q8_1_mmq) / sizeof(int);
    extern __shared__ int smem[];
    int * ids    = smem;
    int * tile_y = smem + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, nwarps * warp_size);
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(kT, 128);
    __half * yh = reinterpret_cast<__half *>(tile_x + txs.qs + txs.dm + txs.sc);
    for (int j = threadIdx.y * warp_size + threadIdx.x; j < J; j += nwarps * warp_size) ids[j] = j;
    __syncthreads();
    const char * xb = x + (size_t) blockIdx.x * I * stride_row_x * sizeof(block_nvfp4);   // this block's 128 rows
    float sa[J * I / (nwarps * warp_size)] = {0.0f};
    float sb[J * I / (nwarps * warp_size)] = {0.0f};
    for (int it = 0; it < n_iter; ++it) {
        const int kb0 = it * bpi;
        ggml_cuda_mmq_load_tiles_nvfp4_dp4a<kT, J, fallback>(xb, tile_x, kb0, I - 1, stride_row_x);
        for (int half = 0; half < 2; ++half) {
            const int * by0 = y + ncols_y * ((kb0 * qk / QK8_1_MMQ) * sz + half * sz);
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
            __syncthreads();
            if (MODE == 0 || MODE == 1) ggml_cuda_mmq_vec_dot_nvfp4_q8_1_dp4a<kT, J, fallback>(tile_x, tile_y, sa, half * MMQ_TILE_NE_K);
            if (MODE == 2) ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma<kT, J, fallback>(tile_x, tile_y, sb, half * MMQ_TILE_NE_K);
            if (MODE == 0 || MODE == 4) {
                mmq_hmma_y_to_half<J>(tile_y, yh);
                __syncthreads();
                ggml_cuda_mmq_vec_dot_nvfp4_q8_1_hmma_yh<kT, J, fallback>(tile_x, tile_y, yh, sb, half * MMQ_TILE_NE_K);
            }
            if (MODE == 3 && tile_x[threadIdx.x] == 0x7F7F7F7F && tile_y[threadIdx.x] == 1) sa[0] += 1.0f;   // keeps the loads
            __syncthreads();
        }
    }
    float * oa = out_a + (size_t) blockIdx.x * I * J;
    float * ob = out_b + (size_t) blockIdx.x * I * J;
    if (MODE == 0 || MODE == 1 || MODE == 3) ggml_cuda_mmq_write_back_dp4a<kT, J, fallback>(sa, ids, oa, I, I - 1, J - 1);
    if (MODE == 0 || MODE == 2 || MODE == 4) ggml_cuda_mmq_write_back_hmma<kT, J, fallback>(sb, ids, ob, I, I - 1, J - 1);
}

int fails = 0;

template <int J>
size_t smem_bytes() {
    constexpr int I = 128;
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(kT, I);
    return (size_t) J * 4 + GGML_PAD(J * MMQ_TILE_Y_K, 256) * 4 + (size_t) (txs.qs + txs.dm + txs.sc) * 4 + (size_t) J * MMQ_HMMA_YHL * 2;
}

template <int J>
void run_case(std::mt19937 & rng, int n_iter, int n_tiles, bool timing) {
    const int n_in = n_iter * 256, blocks = n_in / 64, rows = 128 * n_tiles;
    std::vector<uint8_t> hx((size_t) rows * blocks * sizeof(block_nvfp4));
    for (auto & b : hx) b = (uint8_t) rng();
    for (size_t i = 0; i < hx.size(); i += sizeof(block_nvfp4))
        for (int s = 0; s < 4; ++s) {
            const unsigned r = rng() % 100;   // mostly normal scales, some subnormal, zero and 0x7F
            hx[i + s] = (uint8_t) (r < 80 ? 0x20 + rng() % 0x40 : r < 92 ? rng() % 8 : r < 96 ? 0 : 0x7F);
        }
    const int kblocks = n_in / 128;   // q8_1 blocks of 128 values a column
    std::vector<int> hy((size_t) (kblocks * J + 8) * 36, 0);
    for (int kb = 0; kb < kblocks; ++kb)
        for (int j = 0; j < J; ++j) {
            int * blk = &hy[((size_t) kb * J + j) * 36];
            for (int s = 0; s < 4; ++s) {
                const float d = std::ldexp(1.0f + (float) (rng() % 1000) / 1000.0f, -(int) (rng() % 12));
                std::memcpy(&blk[s], &d, 4);
            }
            for (int q = 0; q < 32; ++q) blk[4 + q] = (int) rng();
        }
    char * dx;
    int * dy;
    float *oa, *ob;
    cudaMalloc(&dx, hx.size());
    cudaMalloc(&dy, hy.size() * 4);
    cudaMalloc(&oa, (size_t) rows * J * 4);
    cudaMalloc(&ob, (size_t) rows * J * 4);
    cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), hy.size() * 4, cudaMemcpyHostToDevice);
    const size_t sm = smem_bytes<J>();
    cudaFuncSetAttribute(tile_kernel<J, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    cudaFuncSetAttribute(tile_kernel<J, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    cudaFuncSetAttribute(tile_kernel<J, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    cudaFuncSetAttribute(tile_kernel<J, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    cudaFuncSetAttribute(tile_kernel<J, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    if (!timing) {
        cudaMemset(oa, 0xFF, (size_t) rows * J * 4);
        cudaMemset(ob, 0xEE, (size_t) rows * J * 4);
        tile_kernel<J, 0><<<n_tiles, dim3(32, 8), sm>>>(dx, dy, oa, ob, blocks, n_iter, J);
        const cudaError_t err = cudaDeviceSynchronize();
        std::vector<float> ha((size_t) rows * J), hb(ha.size());
        cudaMemcpy(ha.data(), oa, ha.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), ob, hb.size() * 4, cudaMemcpyDeviceToHost);
        long diff = 0, first = -1;
        for (size_t i = 0; i < ha.size(); ++i)
            if (std::memcmp(&ha[i], &hb[i], 4) != 0) { if (first < 0) first = (long) i; ++diff; }
        std::printf("J=%3d, %4d values: %s", J, n_in, err != cudaSuccess ? cudaGetErrorString(err) : diff ? "DIFFERS" : "bitwise");
        if (diff) std::printf(" (%ld of %zu, first at %ld: %.9g against %.9g)", diff, ha.size(), first, (double) hb[(size_t) first],
                              (double) ha[(size_t) first]);
        std::printf("\n");
        if (diff || err != cudaSuccess) ++fails;
    } else {
        cudaEvent_t a, e;
        cudaEventCreate(&a);
        cudaEventCreate(&e);
        float ms[5] = {};
        for (int mode = 1; mode <= 4; ++mode) {
            auto go = [&] {
                if (mode == 1) tile_kernel<J, 1><<<n_tiles, dim3(32, 8), sm>>>(dx, dy, oa, ob, blocks, n_iter, J);
                else if (mode == 2) tile_kernel<J, 2><<<n_tiles, dim3(32, 8), sm>>>(dx, dy, oa, ob, blocks, n_iter, J);
                else if (mode == 3) tile_kernel<J, 3><<<n_tiles, dim3(32, 8), sm>>>(dx, dy, oa, ob, blocks, n_iter, J);
                else tile_kernel<J, 4><<<n_tiles, dim3(32, 8), sm>>>(dx, dy, oa, ob, blocks, n_iter, J);
            };
            go();
            cudaEventRecord(a);
            for (int r = 0; r < 5; ++r) go();
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms[mode], a, e);
        }
        const double macs = (double) rows * J * n_in;
        std::printf("time J=%d, %d x %d against %d columns: dp4a %.1f us (%.1f TMAC/s), tensor cores %.1f us, with the fp16 y tile %.1f us (%.1f TMAC/s), loads alone %.1f us\n",
                    J, rows, n_in, J, 1000.0 * ms[1] / 5, macs / (ms[1] / 5 * 1e-3) / 1e12, 1000.0 * ms[2] / 5, 1000.0 * ms[4] / 5,
                    macs / (ms[4] / 5 * 1e-3) / 1e12, 1000.0 * ms[3] / 5);
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(oa);
    cudaFree(ob);
}

// NVFP4 rows (the generator above) and q8_1 columns in MMQ's activation layout (block kb of column j at kb T + j, 36 ints,
// 128 blocks of slack as moe_mmq::q8_bytes leaves them: a column tile reads past the last column)
void fill_x(std::mt19937 & rng, std::vector<uint8_t> & hx) {
    for (auto & b : hx) b = (uint8_t) rng();
    for (size_t i = 0; i < hx.size(); i += sizeof(block_nvfp4))
        for (int s = 0; s < 4; ++s) {
            const unsigned r = rng() % 100;
            hx[i + s] = (uint8_t) (r < 80 ? 0x20 + rng() % 0x40 : r < 92 ? rng() % 8 : r < 96 ? 0 : 0x7F);
        }
}
void fill_y(std::mt19937 & rng, std::vector<int> & hy, int kblocks, int T) {
    for (int kb = 0; kb < kblocks; ++kb)
        for (int j = 0; j < T; ++j) {
            int * blk = &hy[((size_t) kb * T + j) * 36];
            for (int s = 0; s < 4; ++s) {
                const float d = std::ldexp(1.0f + (float) (rng() % 1000) / 1000.0f, -(int) (rng() % 12));
                std::memcpy(&blk[s], &d, 4);
            }
            for (int q = 0; q < 32; ++q) blk[4 + q] = (int) rng();
        }
}

void real_case(std::mt19937 & rng, int rows, int cols, int T, bool timing) {
    const int blocks = cols / 64, kblocks = cols / 128;
    std::vector<uint8_t> hx((size_t) rows * blocks * sizeof(block_nvfp4));
    fill_x(rng, hx);
    std::vector<int> hy(((size_t) kblocks * T + 128) * 36, 0);
    fill_y(rng, hy, kblocks, T);
    char * dx;
    int * dy;
    float * dd;
    cudaMalloc(&dx, hx.size());
    cudaMalloc(&dy, hy.size() * 4);
    cudaMalloc(&dd, (size_t) rows * T * 4);
    cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), hy.size() * 4, cudaMemcpyHostToDevice);
    int dev = 0;
    cudaGetDevice(&dev);
    ggml_backend_cuda_context ctx(dev);
    const mmq_args a = {dx, kT, dy, nullptr, nullptr, dd, cols, rows, T, blocks, T, rows, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, T, T};
    std::vector<float> out[2];
    float ms[2] = {};
    for (int v = 0; v < 2; ++v) {
        mmq_hmma_set(v == 0 ? 2 : 1);
        cudaMemset(dd, v == 0 ? 0xEE : 0xDD, (size_t) rows * T * 4);
        mul_mat_q_case<kT>(ctx, a, 0);
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("real %d x %d, %d columns: %s\n", rows, cols, T, cudaGetErrorString(err));
            ++fails;
            return;
        }
        out[v].resize((size_t) rows * T);
        cudaMemcpy(out[v].data(), dd, out[v].size() * 4, cudaMemcpyDeviceToHost);
        if (timing) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int r = 0; r < 10; ++r) mul_mat_q_case<kT>(ctx, a, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[v], e0, e1);
        }
    }
    mmq_hmma_set(0);
    long diff = 0, first = -1;
    for (size_t i = 0; i < out[0].size(); ++i)
        if (std::memcmp(&out[0][i], &out[1][i], 4) != 0) { if (first < 0) first = (long) i; ++diff; }
    std::printf("real %5d x %5d, %4d columns: %s", rows, cols, T, diff ? "DIFFERS" : "bitwise");
    if (diff) std::printf(" (%ld of %zu, first at %ld: %.9g against %.9g)", diff, out[0].size(), first, (double) out[1][(size_t) first],
                          (double) out[0][(size_t) first]);
    if (timing) std::printf("; dp4a %.1f us, tensor cores %.1f us (%+.1f %%)", 100.0 * ms[0], 100.0 * ms[1], 100.0 * (ms[1] / ms[0] - 1.0));
    std::printf("\n");
    if (diff) ++fails;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dd);
}

}  // namespace

int main(int argc, char ** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    std::mt19937 rng(17);
    for (int it : {1, 20}) {
        run_case<16>(rng, it, 2, false);
        run_case<32>(rng, it, 2, false);
        run_case<48>(rng, it, 2, false);
        run_case<64>(rng, it, 2, false);
        run_case<96>(rng, it, 2, false);
        run_case<112>(rng, it, 2, false);
        run_case<128>(rng, it, 2, false);
    }
    const bool bench = !(argc > 1 && std::strcmp(argv[1], "--no-bench") == 0);
    if (bench) {
        run_case<128>(rng, 20, 640, true);
        run_case<64>(rng, 20, 640, true);
    }
    // the real entry: tile widths 16 to 128 (and the dp4a ones between), stream-k with and without its fixup, rows not a
    // multiple of 128 (the fallback tiles), then the 27B's products on a 2048-row chunk
    for (const int T : {16, 48, 100, 384, 640, 1000, 2048}) real_case(rng, 5120, 5120, T, false);
    real_case(rng, 17408, 5120, 384, false);
    real_case(rng, 1000, 5120, 256, false);
    real_case(rng, 6000, 2560, 77, false);
    if (bench) {
        real_case(rng, 6144, 5120, 2048, true);
        real_case(rng, 17408, 5120, 2048, true);
        real_case(rng, 5120, 17408, 2048, true);
        real_case(rng, 5120, 5120, 512, true);
    }
    std::printf("%s\n", fails ? "MMQ-HMMA DIFFERS" : "MMQ-HMMA BITWISE OK");
    return fails ? 1 : 0;
}
