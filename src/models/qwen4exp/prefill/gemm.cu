// src/models/qwen4exp/prefill/gemm.cu - see include/basalt/models/qwen4exp/prefill/gemm.hpp.
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/kernels/f16_bits.hpp"   // f16_from_f32
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>


#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "basalt/log.hpp"


namespace basalt::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        basalt::log::error("basalt", "prefill gemm: %s: cuBLAS status %d", what, (int) s);
        std::exit(1);
    }
}

// A setup call whose failure the engine survives (the handle keeps its defaults), but says so.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) basalt::log::warn("basalt", "prefill gemm: %s: cuBLAS status %d (continuing)", what, (int) s);
}


// BF16 bits to FP16 through Basalt's own f16_from_f32 (f16_bits.hpp): a finite value saturated at +-65504 (exact for |x|
// in [2^-14, 65504]; smaller ones become subnormal), a NaN or an inf kept as is
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = __uint_as_float((uint32_t) in[i] << 16);
        out[i] = basalt::kernels::f16_from_f32(isfinite(v) ? fminf(fmaxf(v, -65504.0f), 65504.0f) : v);
    }
}
void bf16_to_f16(const uint16_t* in, void* out, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    bf16_to_f16_kernel<<<(unsigned) (b < 4096 ? b : 4096), 256, 0, (cudaStream_t) stream>>>(in, (uint16_t*) out, n);
}
// a Volta card (sm_70 to sm_74, no BF16 tensor cores) and BASALT_BF16_F16 is not 0; read once per device
bool bf16_as_f16_here() {
    static const bool off = [] {
        const char* e = std::getenv("BASALT_BF16_F16");
        return e != nullptr && e[0] == '0';
    }();
    if (off) return false;
    static std::atomic<int> cc[16];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) {
        cudaGetLastError();
        return false;
    }
    int c = cc[dev].load(std::memory_order_acquire);
    if (c == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        c = 10 * major + minor;
        cc[dev].store(c, std::memory_order_release);
    }
    return c >= 70 && c < 75;
}
constexpr int64_t kF16RowElems = (int64_t) 1 << 23;   // X's FP16 image: 8 Mi elements (16 MiB) per block of rows

// One handle made ahead of time per device, taken by the first Gemm that starts there.
cublasHandle_t g_prepared[16] = {};

}  // namespace

bool Gemm::prepare_handle(std::string& err) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16 || g_prepared[dev] != nullptr) return true;
    if (const cublasStatus_t s = cublasCreate(&g_prepared[dev]); s != CUBLAS_STATUS_SUCCESS) {
        g_prepared[dev] = nullptr;
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    return true;
}

Gemm::~Gemm() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (wh_) cudaFree(wh_);   // the FP16 images (Volta)
    if (xh_) cudaFree(xh_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 16 && g_prepared[dev] != nullptr) {
        h = g_prepared[dev];
        g_prepared[dev] = nullptr;
    } else if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

// the two FP16 images, made at their first use and grown when a product needs more; false: keep BF16
bool Gemm::f16_images(int64_t w_elems, int64_t x_elems) {
    if (f16_failed_) return false;
    if (w_elems > wh_elems_) {
        if (wh_) cudaFree(wh_);
        wh_ = nullptr;
        wh_elems_ = 0;
        if (cudaMalloc(&wh_, (size_t) w_elems * 2) != cudaSuccess) { cudaGetLastError(); wh_ = nullptr; f16_failed_ = true; }
        else wh_elems_ = w_elems;
    }
    if (!f16_failed_ && x_elems > xh_elems_) {
        if (xh_) cudaFree(xh_);
        xh_ = nullptr;
        xh_elems_ = 0;
        if (cudaMalloc(&xh_, (size_t) x_elems * 2) != cudaSuccess) { cudaGetLastError(); xh_ = nullptr; f16_failed_ = true; }
        else xh_elems_ = x_elems;
    }
    if (f16_failed_) basalt::log::warn("basalt", "prefill gemm: no room for the FP16 images (Volta): the BF16 products stay on SIMT");
    return !f16_failed_;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (K > 0 && bf16_as_f16_here()) {   // Volta: the tensor cores, through FP16 images of W and of X's row blocks
        const int64_t rb = std::max<int64_t>(1, std::min<int64_t>(T, kF16RowElems / K));
        if (f16_images(N * K, rb * K)) {
            bf16_to_f16(W, wh_, N * K, stream_);
            for (int64_t r0 = 0; r0 < T; r0 += rb) {
                const int64_t tb = std::min<int64_t>(rb, T - r0);
                bf16_to_f16(X + (size_t) r0 * (size_t) K, xh_, tb * K, stream_);
                f16((const uint16_t*) xh_, (const uint16_t*) wh_, Y + (size_t) r0 * (size_t) ldy, tb, N, K, ldy, beta);
            }
            return;
        }
    }
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::bf16_w(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    basalt::kernels::Hc8View v;
    if (basalt::kernels::hc8_view(W, v) && v.owned) {
        if (v.rows != N || v.cols != K) {
            basalt::log::error("basalt", "prefill gemm: an 8-bit copy is %d x %d, the product wants %lld x %lld", v.rows,
                         v.cols, (long long) N, (long long) K);
            std::exit(1);
        }
        if (ldy <= 0) ldy = N;
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0 || scratch_ == nullptr) {
            basalt::log::error("basalt", "prefill gemm: no scratch for an 8-bit matrix (K=%lld)", (long long) K);
            std::exit(1);
        }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min<int64_t>(rows, N - r0);
            basalt::kernels::hc8_dequant_bf16(v, r0, n, scratch_, stream_);
            bf16(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
    bf16(X, W, Y, T, N, K, ldy);
}

const uint16_t* Gemm::bf16_rows(const uint16_t* W, int64_t N, int64_t K) {
    basalt::kernels::Hc8View v;
    if (!basalt::kernels::hc8_view(W, v) || !v.owned) return W;
    if (v.rows != N || v.cols != K || scratch_ == nullptr || N * K > scratch_elems_) return nullptr;
    basalt::kernels::hc8_dequant_bf16(v, 0, N, scratch_, stream_);
    return scratch_;
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { basalt::log::error("basalt", "prefill gemm: scratch too small for K=%lld", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            basalt::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    basalt::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace basalt::prefill
