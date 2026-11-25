// include/basalt/kernels/dense_mmvq.hpp - (Lapilli, V100) a dense model's decode-window products of NVFP4 and Q8_0
// matrices on a card without FP4 tensor cores, for windows of 1 to 16 rows.
//
// iq_mmvq (iq_kernels.cu) reads q8_1 blocks: 36 bytes per 32 values, each lane's 16 activation bytes as four 32-bit
// loads that touch some five cache lines each (the blocks' 36-byte stride keeps them off 16-byte boundaries), so every
// column of a window adds about as much L1 traffic as the weights themselves: a verify window cost 29.9 ms for one row
// and 8 ms more for each further row on a V100 (27B).  Here the activations are the same q8_1 values in
// another layout: by chunks of 512 values, each chunk holding the window's rows one after the other, each row's 512 int8
// values (16-byte aligned) then their 16 fp16 scales (544 bytes).  A lane's 16 bytes (NVFP4) or 8 bytes (Q8_0) are one
// load, a warp's 512 bytes four cache lines, and a column's place in a chunk is a constant of the kernel: one base
// address an iteration, the columns at immediate offsets.  (With one address register a column, ptxas saved registers
// by loading the columns one after the other, each load waiting for the products of the one before: measured up to 2x
// slower from one width to the next, 70-87% of the time stalled on memory against 49-62%.)
//
// The arithmetic is iq_mmvq's (mmvq_multi_kernel): per (row, column) the same calls k = lane + 32 i, the same dp4a
// chain on the same words, the same float expression, the same sums in the same order and the same warp_sum, so each
// value is BITWISE the one iq_mmvq gives (dense_mmvq_bench checks it for every width and every variant).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// The widest window (rows of x) and the chunk of values the layout groups: n_cols a multiple of kDenseChunk.
constexpr int kDenseMaxRows = 16, kDenseChunk = 512;

/// The formats dense_mmvq takes: Q8_0 (8) and NVFP4 (40).
bool dense_mmvq_supported(int ggml_type) noexcept;
/// Whether dense_mmvq beats MMQ at that width for that format (measured on a V100, dense_mmvq.cu's table).
bool dense_mmvq_faster(int ggml_type, int ncols) noexcept;
/// Bytes of the layout above for `n_rows` rows of `n_cols` values (n_cols a multiple of kDenseChunk).
size_t dense_q8_bytes(int64_t n_cols, int64_t n_rows) noexcept;
/// The q8_1 values and scales of `n_rows` rows of `n_cols` floats (quantize_q8_1_rows' values: the same d and the same
/// int8 per value), in the layout above.
void dense_quantize_q8(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);
/// y[c][r] = W[r] . x[c] for the `ncols` rows (1 to kDenseMaxRows) that dense_quantize_q8 left in x with
/// n_rows = ncols (n_in values each), as iq_mmvq leaves them.
void dense_mmvq(int ggml_type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream);

/// The fused steps around the products (the same bits as the steps they replace, dense_fused_parity):
/// one matrix of a dense_mmvq_segs launch: its n_out rows, its output y[c][r] (ld = rows) and its tensor scale (null: none,
/// else scale_values' multiply applied to the sum).
constexpr int kDenseMaxSegs = 4;
struct DenseSeg {
    const void* w;
    int rows;
    const float* scale;
    float* y;
};
/// Up to kDenseMaxSegs matrices of one format reading the same x (n_in values a row) in one launch.
void dense_mmvq_segs(int ggml_type, const DenseSeg* segs, int n_segs, const void* x, int n_in, int ncols, void* stream);
/// resid[c][r] (BF16, ld = n_out) += scale * W[r] . x[c]: dense_mmvq then add_scaled, in one launch.
void dense_mmvq_add_bf16(int ggml_type, const void* w, const float* scale, const void* x, uint16_t* resid, int n_in, int n_out,
                         int ncols, void* stream);
/// rms_norm_rows of BF16 rows (into xn, as before) then dense_quantize_q8 of xn, in one launch (n_cols % 512 == 0).
void dense_norm_quantize(const uint16_t* x, const float* w, float* xn, float eps, int64_t n_rows, int64_t n_cols, void* y,
                         void* stream);
/// swiglu_scaled (h = silu(sg g) (su u), g and u [n_rows][width]) then dense_quantize_q8 of h, in one launch; h is not
/// written (width % 512 == 0).
void dense_swiglu_quantize(const float* g, const float* u, const float* sg, const float* su, int64_t n_rows, int64_t width,
                           void* y, void* stream);

/// The variant for every width: 1, 2 or 4 weight rows a warp on the direct kernel, 11, 12 or 14 on the staged one (the
/// activations through shared memory), 21 on the tensor cores (fp16: not the same bits); 0 (the default) = the measured
/// table.  For the bench and the engine's forced variant.
void dense_mmvq_set_rows_per_warp(int r);
/// Variant 21's split: 1, 2 or 4 quad pairs per 8 rows (32, 16 or 8 rows a warp); 0 (the default) = from the rows.
void dense_mmvq_set_mma_split(int qs);
/// Whether the table sends the widths where they win to the tensor cores (21, 22) or keeps the dp4a variants (false, the
/// default: the same bits at every width; see dense_mmvq.cu's kMma).
void dense_mmvq_set_tensor_cores(bool on);

}  // namespace basalt::kernels
