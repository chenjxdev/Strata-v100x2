// include/basalt/kernels/mma884.cuh - (Lapilli, V100) Volta's own tensor-core instruction:
// mma.sync.aligned.m8n8k4, fp16 inputs, fp32 accumulators (HMMA.884 on sm_70).  A warp computes FOUR independent
// 8x8x4 products, one per quad pair; the fragments, as tests/kernels/dense_bw_probe.cu's mma884_layout_check verifies
// them on the card (random small integers against a host product, exact in fp16 and in fp32), are:
//   product p belongs to lanes 4p..4p+3 (A rows 0-3, B columns 0-3, C rows 0-3) and 4p+16..4p+19 (rows and columns 4-7);
//   A (.row): a_i = A[r][i], r = lane % 4 + 4 (lane >= 16); a0 a1 in the first 32-bit register (a0 in its low half);
//   B (.col): b_i = B[i][c], c = lane % 4 + 4 (lane >= 16); b0 b1 in the first register;
//   C, D:     c_i = C[r][c], r = (lane & 1) + (i & 2) + 4 (lane >= 16), c = (i & 4) + (lane & 2) + (i & 1).
#pragma once

namespace basalt::kernels {

/// d += A B for the warp's four products (d: the accumulators c0..c7 above).
__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

/// The same product with B row-major (.row.row): b_i = B[r][c], r = lane % 4, c = i + 4 (lane >= 16): four consecutive
/// columns of one row of B a lane (dense_bw_probe.cu's mma884_layout_check verifies this one too).
__device__ __forceinline__ void mma884_rr(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

/// The row of A (and of C's quad-pair half) and the column of B a lane holds, as above.
__device__ __forceinline__ int mma884_ab_index(int lane) { return (lane & 3) + 4 * (lane >> 4); }
/// The row and the column of accumulator i of a lane.
__device__ __forceinline__ int mma884_c_row(int lane, int i) { return (lane & 1) + (i & 2) + 4 * (lane >> 4); }
__device__ __forceinline__ int mma884_c_col(int lane, int i) { return (i & 4) + (lane & 2) + (i & 1); }

}  // namespace basalt::kernels
