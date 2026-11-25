// src/models/qwen4exp/kernels/dec_moe_peer.hpp - the peer window kernel's four variants, one TU each (dec_moe_peer_*.cu): each returns its
// kernel's address for dec_moe.cu to launch.  The name carries the variant: q2 = Q2_0 gate/up rows, mix = the GDN mixer split's item.
#pragma once

namespace basalt::kernels {
const void* dm_peer_fn_q0_m0();
const void* dm_peer_fn_q0_m1();
const void* dm_peer_fn_q1_m0();
const void* dm_peer_fn_q1_m1();
// The variants without Q2_0 at 2 and 3 blocks per SM at least (BASALT_PEER_MINB)
const void* dm_peer_fn_q0_m0_b2();
const void* dm_peer_fn_q0_m1_b2();
const void* dm_peer_fn_q0_m0_b3();
const void* dm_peer_fn_q0_m1_b3();
const void* dm_peer_fn_q0_m0_b1();   // and at 1 block per SM at least
const void* dm_peer_fn_q0_m1_b1();
}  // namespace basalt::kernels
