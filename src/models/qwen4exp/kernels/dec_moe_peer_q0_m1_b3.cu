// src/models/qwen4exp/kernels/dec_moe_peer_q0_m1_b3.cu - the peer window kernel, Q2 = false, MIX = true,
// 3 blocks an SM at least (BASALT_PEER_MINB=3; see dec_moe_peer.hpp).
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.cuh"
#include "dec_moe_peer.hpp"

namespace basalt::kernels {
const void* dm_peer_fn_q0_m1_b3() { return (const void*) dm_peer_kernel<false, true, 3>; }
}  // namespace basalt::kernels
