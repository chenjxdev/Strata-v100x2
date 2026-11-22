// include/basalt/kernels/dense_decode.cuh - (Lapilli, V100) the weight decoding of dense_mmvq.cu, shared with
// its tests (tests/kernels/dense_bw_probe.cu's nvfp4_decode_check runs every one of the 16 codes x 256 scale bytes).
//
// nvfp4_scale_dev: NVFP4's UE4M3 sub-block scale as ggml decodes it (iq_kernels.cu's nvfp4_scale_dev: the same float for
// each of the 256 bytes), without its branches: both values computed, then selected.  Its two branches were a
// reconvergence region (BSSY/BSYNC) per weight row and call in the loop, which ptxas does not schedule across.
// The rest is for the tensor cores (dense_mmvq.cu's variant 21), in fp16:
//   nvfp4_scale_h2  the same scale as a half2 (d, d), without conversions (I2F and F2F run at a quarter of the rate on a
//                   V100): a normal UE4M3 (e >= 1) is the half ((e + 7) << 10) | (m << 7), a subnormal one (m 2^-10) the
//                   high byte of an 8-entry table (PRMT), 0 and 0x7F are 0; every value is exact in fp16;
//   fp4_hi_bytes    get_int_from_table_16 (llama.cpp's vecdotq.cuh, MIT, see THIRD_PARTY_LICENSES) on the fp16 high bytes of
//                   kvalues_fp4 (each value's low byte is zero), the table's four words in registers;
//   fp4_frags       a code word of a sub-block (bytes 4w..4w+3: value 4w+j in byte j's low nibble, 8+4w+j in its high one) to
//                   the A fragments (two half2 each) of values 4w..4w+3 and 8+4w..8+4w+3, times the scale: code x scale is
//                   EXACT in fp16 for each of the 4096 pairs;
//   i8x4_to_half    four int8 to two half2, exactly (1024 + (q + 128) is the half 0x6400 | (q + 128), less 1152).
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace basalt::kernels {

__device__ __forceinline__ float nvfp4_scale_dev(uint32_t x) {
    const uint32_t v = x & 0x7F;
    const uint32_t e = v >> 3, m = v & 7;
    const float sub = (float) m * (1.0f / 1024.0f);
    const float nor = __uint_as_float(((e + 119) << 23) | (m << 20));
    const float r = e == 0 ? sub : nor;
    return v == 0 || v == 0x7F ? 0.0f : r;
}

__device__ __forceinline__ unsigned nvfp4_scale_h2(uint32_t x) {
    const uint32_t v = x & 0x7F, m = v & 7;
    const uint32_t sub = __byte_perm(0x1A181400u, 0x1F1E1D1Cu, m * 0x1010u);   // {0, 14, 18, 1A, 1C, 1D, 1E, 1F}[m] << 8, twice
    const uint32_t nor = v * 0x00800080u + 0x1C001C00u;                         // (v << 7) + 0x1C00, twice
    const uint32_t r = v < 8 ? sub : nor;
    return v == 0x7F ? 0u : r;
}

constexpr uint32_t kFp4T0 = 0x42403C00u, kFp4T1 = 0x4A484644u, kFp4T2 = 0xC2C0BC00u, kFp4T3 = 0xCAC8C6C4u;
__device__ __forceinline__ int2 fp4_hi_bytes(uint32_t q4) {
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(kFp4T0, kFp4T1, q4 >> sh), hi = __byte_perm(kFp4T2, kFp4T3, q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ unsigned h2_bits(__half2 v) { return *reinterpret_cast<const unsigned*>(&v); }
__device__ __forceinline__ __half2 bits_h2(unsigned v) { return *reinterpret_cast<const __half2*>(&v); }
__device__ __forceinline__ unsigned hmul2_bits(unsigned v, unsigned s2) { return h2_bits(__hmul2(bits_h2(v), bits_h2(s2))); }

/// lo: the A fragment (two half2) of the word's low nibbles, hi: of its high ones, each value times the half2 s2
__device__ __forceinline__ void fp4_frags(uint32_t word, unsigned s2, unsigned (&lo)[2], unsigned (&hi)[2]) {
    const int2 hb = fp4_hi_bytes(word);
    lo[0] = hmul2_bits(__byte_perm(0u, (unsigned) hb.x, 0x5040), s2);
    lo[1] = hmul2_bits(__byte_perm(0u, (unsigned) hb.x, 0x7060), s2);
    hi[0] = hmul2_bits(__byte_perm(0u, (unsigned) hb.y, 0x5040), s2);
    hi[1] = hmul2_bits(__byte_perm(0u, (unsigned) hb.y, 0x7060), s2);
}

/// get_int_from_table_16's two words before its last two byte permutes: the table bytes of the word's nibbles in order,
/// byte j's low nibble then its high one (values 4w+j and 8+4w+j of the sub-block): .x bytes 0-1, .y bytes 2-3
__device__ __forceinline__ uint2 fp4_hi_pairs(uint32_t q4) {
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(kFp4T0, kFp4T1, q4 >> sh), hi = __byte_perm(kFp4T2, kFp4T3, q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_uint2(tmp[0], tmp[1]);
}
/// variant 22's fragments of a code word (two byte permutes fewer than fp4_frags): q0 the k-quad of values
/// (4w, 8+4w, 4w+1, 9+4w), q1 (4w+2, 10+4w, 4w+3, 11+4w), times s2; act_pairs puts the activations in that order
__device__ __forceinline__ void fp4_frags_paired(uint32_t word, unsigned s2, unsigned (&q0)[2], unsigned (&q1)[2]) {
    const uint2 t = fp4_hi_pairs(word);
    q0[0] = hmul2_bits(__byte_perm(0u, t.x, 0x5040), s2);
    q0[1] = hmul2_bits(__byte_perm(0u, t.x, 0x7060), s2);
    q1[0] = hmul2_bits(__byte_perm(0u, t.y, 0x5040), s2);
    q1[1] = hmul2_bits(__byte_perm(0u, t.y, 0x7060), s2);
}
/// a: the int8 of values 4w..4w+3 of a sub-block, b: of 8+4w..11+4w; the words of fp4_frags_paired's two k-quads
__device__ __forceinline__ void act_pairs(uint32_t a, uint32_t b, uint32_t& k0, uint32_t& k1) {
    k0 = __byte_perm(a, b, 0x5140);
    k1 = __byte_perm(a, b, 0x7362);
}

__device__ __forceinline__ void i8x4_to_half(unsigned w, __half2& a, __half2& b) {
    const unsigned u = w ^ 0x80808080u;
    const __half2 k = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152
    const unsigned lo = __byte_perm(u, 0x64646464u, 0x4140), hi = __byte_perm(u, 0x64646464u, 0x4342);
    a = __hsub2(bits_h2(lo), k);
    b = __hsub2(bits_h2(hi), k);
}

}  // namespace basalt::kernels
