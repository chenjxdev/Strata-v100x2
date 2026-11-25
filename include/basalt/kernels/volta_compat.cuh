#pragma once
// The port of Basalt to the Tesla V100 (Volta, compute capability 7.0): CORRECT but slow equivalents of the PTX instructions
// that Volta lacks. Above Volta, each function keeps the original asm word for word; on Volta (BASALT_VOLTA_ASM), it runs
// the equivalent:
//   - cp.async (Ampere)           : a synchronous copy (a load, then st.shared); commit / wait become no-ops;
//   - ldmatrix m8n8 (Turing)      : each lane loads its own elements, taking the address of the row it needs from the lane
//                                   that holds it (__shfl_sync);
//   - mma.sync m16n8k16 / k32 / k8 (Ampere): the fragments gathered with __shfl_sync, then the product computed with FMAs
//     (dp4a for int8), following the fragment layouts of the PTX ISA documentation (g = lane / 4, t = lane % 4);
//   - cvt.rna.tf32.f32 and cvt.rn.bf16x2.f32: the same rounding in integer or intrinsic code; __reduce_add_sync: a shuffle
//     tree; griddepcontrol and cp.async.bulk.prefetch.L2: no-ops;
//   - thread block clusters and the other Hopper-only instructions: still compiled, they trap if reached (end of the file).
// The kernels call these functions where the upstream code has the inline asm.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define BASALT_VOLTA_ASM 1
#endif

namespace basalt::volta {

__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & 31u; }

// ---------------------------------------------------------------- shared memory by 32-bit address
__device__ __forceinline__ uint32_t ld_s32(unsigned a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint16_t ld_s16(unsigned a) { uint16_t v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ void st_s32(unsigned a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_s64(unsigned a, uint2 v) { asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(a), "r"(v.x), "r"(v.y) : "memory"); }
__device__ __forceinline__ void st_s128(unsigned a, uint4 v) {
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ---------------------------------------------------------------- cp.async, emulated
// N bytes copied from g to the shared address s; with src_n < N, the rest is zero-filled (the "zfill" form of cp.async)
template <int N>
__device__ __forceinline__ void cp_async(unsigned s, const void* g, int src_n = N) {
    static_assert(N == 4 || N == 8 || N == 16, "cp.async: 4, 8 or 16 bytes");
    if constexpr (N == 16) {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (src_n >= 16) v = __ldcg(reinterpret_cast<const uint4*>(g));
        else for (int i = 0; i < src_n; ++i) reinterpret_cast<uint8_t*>(&v)[i] = reinterpret_cast<const uint8_t*>(g)[i];
        st_s128(s, v);
    } else if constexpr (N == 8) {
        uint2 v = make_uint2(0u, 0u);
        if (src_n >= 8) v = *reinterpret_cast<const uint2*>(g);
        else for (int i = 0; i < src_n; ++i) reinterpret_cast<uint8_t*>(&v)[i] = reinterpret_cast<const uint8_t*>(g)[i];
        st_s64(s, v);
    } else {
        uint32_t v = 0u;
        if (src_n >= 4) v = *reinterpret_cast<const uint32_t*>(g);
        else for (int i = 0; i < src_n; ++i) reinterpret_cast<uint8_t*>(&v)[i] = reinterpret_cast<const uint8_t*>(g)[i];
        st_s32(s, v);
    }
}

// ---------------------------------------------------------------- ldmatrix.m8n8.xK[.trans].b16, emulated
// lanes 8 m .. 8 m + 7 give the addresses of the 8 rows of matrix m; register m of lane l receives:
//   without trans: row l / 4, columns 2 (l % 4) and 2 (l % 4) + 1;
//   trans        : column l / 4, rows 2 (l % 4) and 2 (l % 4) + 1.
template <int K, bool TRANS>
__device__ __forceinline__ void ldsm(uint32_t (&r)[K], unsigned addr) {
    const unsigned l = lane_id();
#pragma unroll
    for (int m = 0; m < K; ++m) {
        if constexpr (!TRANS) {
            const unsigned row = __shfl_sync(0xffffffffu, addr, m * 8 + (l >> 2));
            r[m] = ld_s32(row + (l & 3u) * 4u);
        } else {
            const unsigned r0 = __shfl_sync(0xffffffffu, addr, m * 8 + 2 * (l & 3u));
            const unsigned r1 = __shfl_sync(0xffffffffu, addr, m * 8 + 2 * (l & 3u) + 1);
            r[m] = (uint32_t) ld_s16(r0 + (l >> 2) * 2u) | ((uint32_t) ld_s16(r1 + (l >> 2) * 2u) << 16);
        }
    }
}
// the same (without trans) when the address is a generic pointer (ldmatrix without .shared, as in ggml's MMQ kernels)
template <int K>
__device__ __forceinline__ void ldsm_ptr(int* r, const void* p) {
    const unsigned l = lane_id();
    const unsigned long long a = reinterpret_cast<unsigned long long>(p);
#pragma unroll
    for (int m = 0; m < K; ++m) {
        const unsigned long long row = __shfl_sync(0xffffffffu, a, m * 8 + (l >> 2));
        r[m] = *reinterpret_cast<const int*>(row + (l & 3u) * 4u);
    }
}

// ---------------------------------------------------------------- mma.sync, emulated
__device__ __forceinline__ float h_lo(uint32_t v) { return __half2float(__ushort_as_half((unsigned short) (v & 0xffffu))); }
__device__ __forceinline__ float h_hi(uint32_t v) { return __half2float(__ushort_as_half((unsigned short) (v >> 16))); }
__device__ __forceinline__ float b_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float b_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ int s8_at(uint32_t v, int i) { return (int) (int8_t) ((v >> (8 * i)) & 0xffu); }
__device__ __forceinline__ int u8_at(uint32_t v, int i) { return (int) ((v >> (8 * i)) & 0xffu); }

// m16n8k16, A f16|bf16 (4 registers), B (2), C/D f32 (4). Lane (g, t) computes D[g][2t..2t+1] and D[g+8][2t..2t+1].
template <bool BF16>
__device__ __forceinline__ void mma_m16n8k16_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4]) {
    const unsigned l = lane_id(), g = l >> 2, t = l & 3u;
    float acc[4] = {c[0], c[1], c[2], c[3]};
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        // from A: rows g and g+8, columns k = 2 tt, 2 tt + 1 (a0/a1) and 2 tt + 8, 2 tt + 9 (a2/a3)
        const unsigned src = g * 4 + tt;
        const uint32_t a0 = __shfl_sync(0xffffffffu, a[0], src), a1 = __shfl_sync(0xffffffffu, a[1], src);
        const uint32_t a2 = __shfl_sync(0xffffffffu, a[2], src), a3 = __shfl_sync(0xffffffffu, a[3], src);
#pragma unroll
        for (int j = 0; j < 2; ++j) {   // column n = 2 t + j of B, held by lane group n: k = 2 tt.. (b0), 2 tt + 8.. (b1)
            const unsigned bs = (2 * t + j) * 4 + tt;
            const uint32_t b0 = __shfl_sync(0xffffffffu, b[0], bs), b1 = __shfl_sync(0xffffffffu, b[1], bs);
            const float B0 = BF16 ? b_lo(b0) : h_lo(b0), B1 = BF16 ? b_hi(b0) : h_hi(b0);
            const float B8 = BF16 ? b_lo(b1) : h_lo(b1), B9 = BF16 ? b_hi(b1) : h_hi(b1);
            acc[j] += (BF16 ? b_lo(a0) : h_lo(a0)) * B0 + (BF16 ? b_hi(a0) : h_hi(a0)) * B1
                    + (BF16 ? b_lo(a2) : h_lo(a2)) * B8 + (BF16 ? b_hi(a2) : h_hi(a2)) * B9;
            acc[2 + j] += (BF16 ? b_lo(a1) : h_lo(a1)) * B0 + (BF16 ? b_hi(a1) : h_hi(a1)) * B1
                        + (BF16 ? b_lo(a3) : h_lo(a3)) * B8 + (BF16 ? b_hi(a3) : h_hi(a3)) * B9;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = acc[i];
}

// m16n8k16 f16 -> f16 (C/D: 2 registers of two f16 each): computed in f32, rounded at the end
__device__ __forceinline__ void mma_m16n8k16_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2], const uint32_t (&c)[2]) {
    float cf[4] = {h_lo(c[0]), h_hi(c[0]), h_lo(c[1]), h_hi(c[1])}, df[4];
    mma_m16n8k16_f32<false>(df, a, b, cf);
    const __half2 lo = __floats2half2_rn(df[0], df[1]), hi = __floats2half2_rn(df[2], df[3]);
    d[0] = *reinterpret_cast<const uint32_t*>(&lo);
    d[1] = *reinterpret_cast<const uint32_t*>(&hi);
}

// dp4a (sm_61 and later, so the V100 too): 4 byte products added to c in one instruction, in exact integer arithmetic. The
// two int8 emulations below use it instead of 16 byte extractions and 4 multiplications per word (with those, the MMQ
// prefill spent 70% of its time here). The result is bit-identical: the int32 sum is exact.
__device__ __forceinline__ int dp4a_ss(uint32_t a, uint32_t b, int c) { return __dp4a((int) a, (int) b, c); }
__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// m16n8k32 s8|u8 x s8 -> s32: A 4 registers (k = 4 t.. and 4 t + 16..), B 2 registers
template <bool AU8>
__device__ __forceinline__ void mma_m16n8k32_s32(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const int (&c)[4]) {
    const unsigned l = lane_id(), g = l >> 2, t = l & 3u;
    int acc[4] = {c[0], c[1], c[2], c[3]};
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        const unsigned src = g * 4 + tt;
        const uint32_t a0 = __shfl_sync(0xffffffffu, a[0], src), a1 = __shfl_sync(0xffffffffu, a[1], src);
        const uint32_t a2 = __shfl_sync(0xffffffffu, a[2], src), a3 = __shfl_sync(0xffffffffu, a[3], src);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned bs = (2 * t + j) * 4 + tt;
            const uint32_t b0 = __shfl_sync(0xffffffffu, b[0], bs), b1 = __shfl_sync(0xffffffffu, b[1], bs);
            if constexpr (AU8) {
                acc[j] = dp4a_us(a2, b1, dp4a_us(a0, b0, acc[j]));
                acc[2 + j] = dp4a_us(a3, b1, dp4a_us(a1, b0, acc[2 + j]));
            } else {
                acc[j] = dp4a_ss(a2, b1, dp4a_ss(a0, b0, acc[j]));
                acc[2 + j] = dp4a_ss(a3, b1, dp4a_ss(a1, b0, acc[2 + j]));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = acc[i];
}

// m16n8k16 s8 x s8 -> s32: A 2 registers (k = 4 t..), B 1 register
__device__ __forceinline__ void mma_m16n8k16_s32(int (&d)[4], const uint32_t (&a)[2], const uint32_t (&b)[1], const int (&c)[4]) {
    const unsigned l = lane_id(), g = l >> 2, t = l & 3u;
    int acc[4] = {c[0], c[1], c[2], c[3]};
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        const unsigned src = g * 4 + tt;
        const uint32_t a0 = __shfl_sync(0xffffffffu, a[0], src), a1 = __shfl_sync(0xffffffffu, a[1], src);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t b0 = __shfl_sync(0xffffffffu, b[0], (2 * t + j) * 4 + tt);
            acc[j] = dp4a_ss(a0, b0, acc[j]);
            acc[2 + j] = dp4a_ss(a1, b0, acc[2 + j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = acc[i];
}

// m16n8k8 tf32 -> f32: A 4 registers (a0: row g, k = t; a1: g+8, t; a2: g, t+4; a3: g+8, t+4), B 2 (k = t, t+4)
__device__ __forceinline__ void mma_m16n8k8_tf32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4]) {
    const unsigned l = lane_id(), g = l >> 2, t = l & 3u;
    float acc[4] = {c[0], c[1], c[2], c[3]};
    auto tf = [](uint32_t v) { return __uint_as_float(v & 0xffffe000u); };   // tf32: 10 mantissa bits
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        const unsigned src = g * 4 + tt;
        const float a0 = tf(__shfl_sync(0xffffffffu, a[0], src)), a1 = tf(__shfl_sync(0xffffffffu, a[1], src));
        const float a2 = tf(__shfl_sync(0xffffffffu, a[2], src)), a3 = tf(__shfl_sync(0xffffffffu, a[3], src));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned bs = (2 * t + j) * 4 + tt;
            const float b0 = tf(__shfl_sync(0xffffffffu, b[0], bs)), b1 = tf(__shfl_sync(0xffffffffu, b[1], bs));
            acc[j] += a0 * b0 + a2 * b1;
            acc[2 + j] += a1 * b0 + a3 * b1;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = acc[i];
}

// __reduce_add_sync (Ampere) for a full mask
__device__ __forceinline__ unsigned reduce_add_sync(unsigned mask, unsigned v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

// ================================================================ the instructions, under one name for Volta and newer GPUs
__device__ __forceinline__ void cp_async_commit() {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}
// cp.async.cg (16 bytes) and .ca (4, 8, 16), with or without a source size (zfill)
__device__ __forceinline__ void cp_async_cg16(unsigned s, const void* g) {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
#else
    cp_async<16>(s, g);
#endif
}
__device__ __forceinline__ void cp_async_cg16(unsigned s, const void* g, int src_n) {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(g), "r"(src_n) : "memory");
#else
    cp_async<16>(s, g, src_n);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_ca(unsigned s, const void* g) {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(s), "l"(g), "n"(N) : "memory");
#else
    cp_async<N>(s, g);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_ca(unsigned s, const void* g, int src_n) {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(s), "l"(g), "n"(N), "r"(src_n) : "memory");
#else
    cp_async<N>(s, g, src_n);
#endif
}
// cp.async.bulk.prefetch.L2: only a cache hint, a no-op on Volta
__device__ __forceinline__ void bulk_prefetch_l2(const void* g, unsigned n) {
#ifndef BASALT_VOLTA_ASM
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(g), "r"(n) : "memory");
#else
    (void) g; (void) n;
#endif
}

// ldmatrix on a 32-bit shared address
template <int K, bool TRANS, class R>
__device__ __forceinline__ void ldmatrix_s(R* r, unsigned addr) {
#ifndef BASALT_VOLTA_ASM
    uint32_t* u = reinterpret_cast<uint32_t*>(r);
    if constexpr (K == 1 && !TRANS) asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(u[0]) : "r"(addr));
    else if constexpr (K == 1) asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(u[0]) : "r"(addr));
    else if constexpr (K == 2 && !TRANS)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(u[0]), "=r"(u[1]) : "r"(addr));
    else if constexpr (K == 2)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(u[0]), "=r"(u[1]) : "r"(addr));
    else if constexpr (!TRANS)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]) : "r"(addr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]) : "r"(addr));
#else
    uint32_t t[K];
    ldsm<K, TRANS>(t, addr);
#pragma unroll
    for (int i = 0; i < K; ++i) reinterpret_cast<uint32_t*>(r)[i] = t[i];
#endif
}
// the same, one register per argument (the asm sites name their outputs one by one)
template <bool TRANS, class R0>
__device__ __forceinline__ void ldmatrix_x1(R0& r0, unsigned a) { uint32_t t[1]; ldmatrix_s<1, TRANS>(t, a); r0 = (R0) t[0]; }
template <bool TRANS, class R0, class R1>
__device__ __forceinline__ void ldmatrix_x2(R0& r0, R1& r1, unsigned a) {
    uint32_t t[2]; ldmatrix_s<2, TRANS>(t, a); r0 = (R0) t[0]; r1 = (R1) t[1];
}
template <bool TRANS, class R0, class R1, class R2, class R3>
__device__ __forceinline__ void ldmatrix_x4(R0& r0, R1& r1, R2& r2, R3& r3, unsigned a) {
    uint32_t t[4]; ldmatrix_s<4, TRANS>(t, a); r0 = (R0) t[0]; r1 = (R1) t[1]; r2 = (R2) t[2]; r3 = (R3) t[3];
}

// mma.sync in the form of the asm sites: D by reference, A, B, C by value; above Volta a pure asm (not volatile)
template <class D>
__device__ __forceinline__ void mma_16816_f32_bf16(D& d0, D& d1, D& d2, D& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3) {
#ifndef BASALT_VOLTA_ASM
    float o0, o1, o2, o3;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    d0 = o0; d1 = o1; d2 = o2; d3 = o3;
#else
    float d[4]; const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1}; const float c[4] = {c0, c1, c2, c3};
    mma_m16n8k16_f32<true>(d, a, b, c); d0 = d[0]; d1 = d[1]; d2 = d[2]; d3 = d[3];
#endif
}
template <class D>
__device__ __forceinline__ void mma_16816_f32_f16(D& d0, D& d1, D& d2, D& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3) {
#ifndef BASALT_VOLTA_ASM
    float o0, o1, o2, o3;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    d0 = o0; d1 = o1; d2 = o2; d3 = o3;
#else
    float d[4]; const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1}; const float c[4] = {c0, c1, c2, c3};
    mma_m16n8k16_f32<false>(d, a, b, c); d0 = d[0]; d1 = d[1]; d2 = d[2]; d3 = d[3];
#endif
}
template <class D>
__device__ __forceinline__ void mma_16816_f16_f16(D& d0, D& d1, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                  uint32_t b1, uint32_t c0, uint32_t c1) {
#ifndef BASALT_VOLTA_ASM
    uint32_t o0, o1;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(o0), "=r"(o1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    d0 = (D) o0; d1 = (D) o1;
#else
    uint32_t d[2]; const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1}, c[2] = {c0, c1};
    mma_m16n8k16_f16(d, a, b, c); d0 = (D) d[0]; d1 = (D) d[1];
#endif
}
template <class D>
__device__ __forceinline__ void mma_1688_f32_tf32(D& d0, D& d1, D& d2, D& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3) {
#ifndef BASALT_VOLTA_ASM
    float o0, o1, o2, o3;
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    d0 = o0; d1 = o1; d2 = o2; d3 = o3;
#else
    float d[4]; const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1}; const float c[4] = {c0, c1, c2, c3};
    mma_m16n8k8_tf32(d, a, b, c); d0 = d[0]; d1 = d[1]; d2 = d[2]; d3 = d[3];
#endif
}
template <bool AU8, class D>
__device__ __forceinline__ void mma_16832_s32_s8(D& d0, D& d1, D& d2, D& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1, int c0, int c1, int c2, int c3) {
#ifndef BASALT_VOLTA_ASM
    int o0, o1, o2, o3;
    if constexpr (AU8)
        asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    else
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    d0 = (D) o0; d1 = (D) o1; d2 = (D) o2; d3 = (D) o3;
#else
    int d[4]; const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1}; const int c[4] = {c0, c1, c2, c3};
    mma_m16n8k32_s32<AU8>(d, a, b, c); d0 = (D) d[0]; d1 = (D) d[1]; d2 = (D) d[2]; d3 = (D) d[3];
#endif
}
template <class D>
__device__ __forceinline__ void mma_16816_s32_s8(D& d0, D& d1, D& d2, D& d3, uint32_t a0, uint32_t a1, uint32_t b0, int c0, int c1,
                                                 int c2, int c3) {
#ifndef BASALT_VOLTA_ASM
    int o0, o1, o2, o3;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    d0 = (D) o0; d1 = (D) o1; d2 = (D) o2; d3 = (D) o3;
#else
    int d[4]; const uint32_t a[2] = {a0, a1}, b[1] = {b0}; const int c[4] = {c0, c1, c2, c3};
    mma_m16n8k16_s32(d, a, b, c); d0 = (D) d[0]; d1 = (D) d[1]; d2 = (D) d[2]; d3 = (D) d[3];
#endif
}
// f32 -> tf32 rounded to nearest (cvt.rna.tf32.f32): Volta lacks the instruction, so the same rounding with integer ops
__device__ __forceinline__ uint32_t cvt_rna_tf32(float x) {
#ifndef BASALT_VOLTA_ASM
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return u;   // inf, nan: unchanged
    return (u + 0x1000u) & 0xffffe000u;              // half added, then the 13 low bits cut: to nearest, ties away from zero
#endif
}
// cvt.rn.bf16x2.f32 d, hi, lo
__device__ __forceinline__ uint32_t cvt_rn_bf16x2(float hi, float lo) {
#ifndef BASALT_VOLTA_ASM
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#else
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);   // .x = the low half
    return *reinterpret_cast<const uint32_t*>(&v);
#endif
}
// griddepcontrol (Hopper and later): a no-op on older GPUs (like pdl_wait in pdl.hpp)
__device__ __forceinline__ void griddep_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void griddep_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// ---------------------------------------------------------------- thread block clusters (Hopper and later): absent on Volta
// A cluster kernel still compiles, but its body stops at once (__trap): on a V100 the host must take a path without
// clusters (dec_gdn_clusters() selects one for dec_gdn_rec and dec_gdn_rec_seg; gdn_gating_proj, on the dense model path,
// has none yet).
struct no_cluster {
    template <class T> __device__ T* map_shared_rank(T* p, unsigned) const { __trap(); return p; }
    __device__ void sync() const { __trap(); }
};

}  // namespace basalt::volta

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define BASALT_NO_CLUSTER 1
#define BASALT_CLUSTER_DIMS(x, y, z)
#define BASALT_CLUSTER_T basalt::volta::no_cluster
#define BASALT_THIS_CLUSTER() basalt::volta::no_cluster{}
#define BASALT_CLUSTER_ASM(txt) __trap()
#else
#define BASALT_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#define BASALT_CLUSTER_T cooperative_groups::cluster_group
#define BASALT_THIS_CLUSTER() cooperative_groups::this_cluster()
#define BASALT_CLUSTER_ASM(txt) asm volatile(txt ::: "memory")
#endif

// A Hopper-and-later instruction in a kernel that the host launches only on such GPUs (a choice made at run time): on older
// GPUs the kernel compiles, and stops at once if it is launched anyway.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define BASALT_SM90_ASM(...) __trap()
#else
#define BASALT_SM90_ASM(...) asm volatile(__VA_ARGS__)
#endif
