// src/models/qwen4exp/kernels/dec_moe_peer_q0_m0_b3.cu - the peer window kernel, Q2 = false, MIX = false, at least 3 blocks
// per SM (BASALT_PEER_MINB=3; see dec_moe_peer.hpp).
#include "dec_moe_prelude.cuh"
#include "dec_moe_peer.cuh"
#include "dec_moe_peer.hpp"

namespace basalt::kernels {
const void* dm_peer_fn_q0_m0_b3() { return (const void*) dm_peer_kernel<false, false, 3>; }
}  // namespace basalt::kernels
