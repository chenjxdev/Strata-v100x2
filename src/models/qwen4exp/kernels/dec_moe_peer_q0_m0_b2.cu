// src/models/qwen4exp/kernels/dec_moe_peer_q0_m0_b2.cu - the peer window kernel, Q2 = false, MIX = false,
// 2 blocks an SM at least (BASALT_PEER_MINB=2; see dec_moe_peer.hpp).
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.cuh"
#include "dec_moe_peer.hpp"

namespace basalt::kernels {
const void* dm_peer_fn_q0_m0_b2() { return (const void*) dm_peer_kernel<false, false, 2>; }
}  // namespace basalt::kernels
