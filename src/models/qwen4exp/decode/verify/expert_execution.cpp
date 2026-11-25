#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "basalt/log.hpp"
#include <atomic>
#include <string>

namespace basalt::core {
using namespace verify_detail;

// Peer-card diagnostic: BASALT_PEER_STAMPS=1 logs the phases of dm_peer_kernel (timestamps it writes to mapped host memory)
namespace {
struct PeerStamps {
    int on = -1;                         // -1: variable not read yet
    unsigned long long* h = nullptr;     // mapped, kPstRings * 2 * 8 words
    double sum[14] = {};                 // fetch+sync, group, experts, done, to the next doorbell, without entries; [6..9] inside the experts; [10..13] first gate/up item
    long long n = 0, n_empty = 0, n_next = 0, windows = 0, n_in = 0, n_it = 0, n_it4 = 0;   // n_in: inside the experts; n_it, n_it4: first gate/up item
    // Per block, the sums (time since the first start) of the start, the gate/up end, the experts end and the down
    // items; the last SM seen; the rings collected and their groups
    double pb_gs[160] = {}, pb_ge[160] = {}, pb_xe[160] = {}, pb_nd[160] = {};
    unsigned pb_sm[160] = {};
    int pb_nb = 0;
    long long pb_n = 0;
    double pb_ng = 0.0;
    double fb[4] = {};                   // the start of a ring: reread, barrier (with entries), reread, barrier (without)
    long long fb_n[2] = {};
};
constexpr int kPstRings = 128;
PeerStamps g_pst;
bool peer_stamps_on() {
    if (g_pst.on < 0) {
        const char* e = std::getenv("BASALT_PEER_STAMPS");
        g_pst.on = (e != nullptr && *e != 0 && std::atoi(e) != 0) ? 1 : 0;
        if (g_pst.on && cudaHostAlloc((void**) &g_pst.h, (size_t) kPstRings * 2 * 8 * sizeof(unsigned long long),
                                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            g_pst.on = 0;
            g_pst.h = nullptr;
        }
    }
    return g_pst.on == 1;
}
void peer_stamps_collect(int64_t rings, int nb) {
    const volatile unsigned long long* t = g_pst.h;
    const int nr = (int) std::min<int64_t>(rings, kPstRings);
    if (nb > 0 && nb <= 160 && 24 * rings + 4 * nb <= 2048) {   // the middle ring, block by block
        const volatile unsigned long long* z = t + (size_t) 24 * rings;
        unsigned long long tmin = ~0ull;
        bool all = true;
        for (int b = 0; b < nb && all; ++b) {
            const unsigned long long a0 = z[4 * b], a1 = z[4 * b + 1], a2 = z[4 * b + 2];
            all = a0 != 0 && a1 >= a0 && a2 >= a1;
            tmin = std::min(tmin, a0);
        }
        if (all) {
            if (g_pst.pb_nb != nb) {
                for (int b = 0; b < 160; ++b) g_pst.pb_gs[b] = g_pst.pb_ge[b] = g_pst.pb_xe[b] = g_pst.pb_nd[b] = 0.0;
                g_pst.pb_n = 0;
                g_pst.pb_ng = 0.0;
                g_pst.pb_nb = nb;
            }
            for (int b = 0; b < nb; ++b) {
                g_pst.pb_gs[b] += (double) (z[4 * b] - tmin);
                g_pst.pb_ge[b] += (double) (z[4 * b + 1] - tmin);
                g_pst.pb_xe[b] += (double) (z[4 * b + 2] - tmin);
                const unsigned long long w3 = z[4 * b + 3];
                g_pst.pb_nd[b] += (double) (w3 & 0xffffull);
                g_pst.pb_sm[b] = (unsigned) (w3 >> 24);
            }
            g_pst.pb_ng += (double) ((z[3] >> 16) & 0xffull);
            ++g_pst.pb_n;
        }
    }
    for (int s = 0; s < nr && 2 * nr + s < 2 * kPstRings; ++s) {   // the first gate/up item
        const volatile unsigned long long* u = t + (size_t) (2 * rings + s) * 8;
        const unsigned long long a0 = u[0], a1 = u[1], a2 = u[2], a3 = u[3], a4 = u[4];
        {   // the start of the ring: zone 1's [0] and [1] around the end of the reread, u[5]
            const unsigned long long d0 = t[s * 8 + 0], d1 = t[s * 8 + 1], fe = u[5];
            if (d0 != 0 && fe >= d0 && d1 >= fe) {
                const int k = t[s * 8 + 2] != 0 ? 0 : 1;   // [2] written: the ring had entries
                g_pst.fb[2 * k] += (double) (fe - d0);
                g_pst.fb[2 * k + 1] += (double) (d1 - fe);
                ++g_pst.fb_n[k];
            }
        }
        if (a0 != 0 && a1 >= a0 && a2 >= a1 && a3 >= a2) {
            g_pst.sum[10] += (double) (a1 - a0); g_pst.sum[11] += (double) (a2 - a1); g_pst.sum[12] += (double) (a3 - a2);
            ++g_pst.n_it;
            if (a4 != 0 && a4 >= a3) { g_pst.sum[13] += (double) (a4 - a3); ++g_pst.n_it4; }
        }
    }
    for (int s = 0; s < nr; ++s) {
        const unsigned long long a = t[s * 8 + 0], b = t[s * 8 + 1], c = t[s * 8 + 2], d = t[s * 8 + 3], f = t[s * 8 + 4];
        if (a == 0 || b < a) continue;
        if (c != 0 && d >= c && f >= d && c >= b) {
            g_pst.sum[0] += (double) (b - a); g_pst.sum[1] += (double) (c - b);
            g_pst.sum[2] += (double) (d - c); g_pst.sum[3] += (double) (f - d);
            ++g_pst.n;
            // inside block 0's experts
            const unsigned long long t5 = t[s * 8 + 5], t6 = t[s * 8 + 6], t7 = t[s * 8 + 7];
            if (t5 != 0 && t5 >= c && t6 >= t5 && t7 >= t6 && d >= t7) {
                g_pst.sum[6] += (double) (t5 - c); g_pst.sum[7] += (double) (t6 - t5);
                g_pst.sum[8] += (double) (t7 - t6); g_pst.sum[9] += (double) (d - t7);
                ++g_pst.n_in;
            }
            if (s + 1 < nr && t[(s + 1) * 8 + 0] >= f) { g_pst.sum[4] += (double) (t[(s + 1) * 8 + 0] - f); ++g_pst.n_next; }
        } else {
            g_pst.sum[5] += (double) (b - a);
            ++g_pst.n_empty;
        }
    }
    if (++g_pst.windows == 200) {
        const double n = g_pst.n > 0 ? (double) g_pst.n : 1.0, ne = g_pst.n_empty > 0 ? (double) g_pst.n_empty : 1.0;
        const double nx = g_pst.n_next > 0 ? (double) g_pst.n_next : 1.0;
        basalt::log::raw("basalt peer phases over %lld windows: %.2f rings/window with peer entries, us per ring: fetch+sync %.1f, "
                         "group %.1f, experts %.1f, done %.1f (then %.1f to the next doorbell seen) ; %.2f rings/window without "
                         "entries, fetch+sync %.1f\n",
                         g_pst.windows, (double) g_pst.n / (double) g_pst.windows, g_pst.sum[0] / n / 1e3, g_pst.sum[1] / n / 1e3,
                         g_pst.sum[2] / n / 1e3, g_pst.sum[3] / n / 1e3, g_pst.sum[4] / nx / 1e3,
                         (double) g_pst.n_empty / (double) g_pst.windows, g_pst.sum[5] / ne / 1e3);
        if (g_pst.n_in > 0) {   // inside the experts
            const double ni = (double) g_pst.n_in;
            basalt::log::raw("basalt peer phases, inside the experts (block 0, %lld rings): gate/up %.1f, wait for its first "
                             "group %.1f, swiglu %.1f, down + rows %.1f us\n",
                             g_pst.n_in, g_pst.sum[6] / ni / 1e3, g_pst.sum[7] / ni / 1e3, g_pst.sum[8] / ni / 1e3,
                             g_pst.sum[9] / ni / 1e3);
        }
        if (g_pst.pb_n > 0 && g_pst.pb_nb > 0) {   // per block
            const int nbk = g_pst.pb_nb;
            const double m = (double) g_pst.pb_n * 1e3;   // summed ns -> mean us
            int ord[160];
            for (int b = 0; b < nbk; ++b) ord[b] = b;
            std::sort(ord, ord + nbk, [](int x, int y) { return g_pst.pb_xe[x] > g_pst.pb_xe[y]; });
            auto med = [&](const double* v) {
                double c[160];
                for (int b = 0; b < nbk; ++b) c[b] = v[b];
                std::sort(c, c + nbk);
                return c[nbk / 2] / m;
            };
            auto lo = [&](const double* v) { double r = v[0]; for (int b = 1; b < nbk; ++b) r = std::min(r, v[b]); return r / m; };
            auto hi = [&](const double* v) { double r = v[0]; for (int b = 1; b < nbk; ++b) r = std::max(r, v[b]); return r / m; };
            basalt::log::raw("basalt peer blocks (the middle ring, %lld windows, %.1f groups, %d blocks; us from the first start, "
                             "min/median/max of the block means): start %.1f/%.1f/%.1f, gate/up end %.1f/%.1f/%.1f, experts end "
                             "%.1f/%.1f/%.1f\n",
                             g_pst.pb_n, g_pst.pb_ng / (double) g_pst.pb_n, nbk, lo(g_pst.pb_gs), med(g_pst.pb_gs), hi(g_pst.pb_gs),
                             lo(g_pst.pb_ge), med(g_pst.pb_ge), hi(g_pst.pb_ge), lo(g_pst.pb_xe), med(g_pst.pb_xe), hi(g_pst.pb_xe));
            std::string sl, fa;
            char buf[96];
            for (int i = 0; i < 8 && i < nbk; ++i) {
                const int b = ord[i];
                std::snprintf(buf, sizeof(buf), " b%d(sm%u) %.1f/%.1f/%.1f d%.2f", b, g_pst.pb_sm[b], g_pst.pb_gs[b] / m,
                              g_pst.pb_ge[b] / m, g_pst.pb_xe[b] / m, g_pst.pb_nd[b] / (double) g_pst.pb_n);
                sl += buf;
            }
            for (int i = nbk - 1; i >= 0 && i >= nbk - 4; --i) {
                const int b = ord[i];
                std::snprintf(buf, sizeof(buf), " b%d(sm%u) %.1f/%.1f/%.1f d%.2f", b, g_pst.pb_sm[b], g_pst.pb_gs[b] / m,
                              g_pst.pb_ge[b] / m, g_pst.pb_xe[b] / m, g_pst.pb_nd[b] / (double) g_pst.pb_n);
                fa += buf;
            }
            basalt::log::raw("basalt peer blocks, slowest (start/gate-up end/experts end, down items):%s\n", sl.c_str());
            basalt::log::raw("basalt peer blocks, fastest:%s\n", fa.c_str());
            g_pst.pb_nb = 0;
        }
        if (g_pst.fb_n[0] + g_pst.fb_n[1] > 0) {   // the start of a ring
            const double n0 = g_pst.fb_n[0] > 0 ? (double) g_pst.fb_n[0] : 1.0, n1 = g_pst.fb_n[1] > 0 ? (double) g_pst.fb_n[1] : 1.0;
            basalt::log::raw("basalt peer phases, the start of a ring (block 0): reread %.1f, grid barrier %.1f us (%lld rings with "
                             "entries) ; reread %.1f, grid barrier %.1f us (%lld without)\n",
                             g_pst.fb[0] / n0 / 1e3, g_pst.fb[1] / n0 / 1e3, g_pst.fb_n[0], g_pst.fb[2] / n1 / 1e3,
                             g_pst.fb[3] / n1 / 1e3, g_pst.fb_n[1]);
            for (double& v : g_pst.fb) v = 0.0;
            g_pst.fb_n[0] = g_pst.fb_n[1] = 0;
        }
        if (g_pst.n_it > 0) {   // the first gate/up item
            const double nt = (double) g_pst.n_it;
            const double n4 = g_pst.n_it4 > 0 ? (double) g_pst.n_it4 : 1.0;
            basalt::log::raw("basalt peer phases, first gate/up item (block 0, %lld rings): warp 0's row %.2f, threadfence %.2f, "
                             "barrier+atomic %.2f us ; the next item starts %.2f us later (%lld rings)\n",
                             g_pst.n_it, g_pst.sum[10] / nt / 1e3, g_pst.sum[11] / nt / 1e3, g_pst.sum[12] / nt / 1e3,
                             g_pst.sum[13] / n4 / 1e3, g_pst.n_it4);
        }
        for (double& v : g_pst.sum) v = 0.0;
        g_pst.n = g_pst.n_empty = g_pst.n_next = g_pst.windows = g_pst.n_in = g_pst.n_it = g_pst.n_it4 = 0;
    }
}
}  // namespace

// The device-planned tail's window start: residency tables, the flag words, and the peer card's persistent window.
bool Verifier::dm_begin(std::string& err, DecodeGaps* gm) {
    if (!dm_sync(err)) return false;
    decode_events().stamp(4);
    if (gm) gm->mark(13);
    *(volatile uint32_t*) (dm_->h_w + 0) = 0;
    *(volatile uint32_t*) (dm_->h_w + 16) = 0;
    *(volatile uint32_t*) (dm_->h_w + 32) = 0;
    *(volatile uint32_t*) (dm_->h_w + 48) = 0;
    *(volatile uint32_t*) (dm_->h_w + 52) = 0;
    *(volatile uint32_t*) (dm_->h_w + 60) = 0;
    if (dm_->tag && dm_->tag_dirty) {   // a released window left rows written: back to unwritten
        for (size_t i = 0; i < dm_->rows_words; ++i)
            ((volatile uint32_t*) dm_->h_rows)[i] = basalt::kernels::dec_moe_tag_word();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        dm_->tag_dirty = false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (dm_->pdev >= 0) {
        const OnDevice on(dm_->pdev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = dm_->h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = dm_->h_w + 0; pa.done_m = dm_->h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = dm_->ptab; pa.lay = dm_->lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = dm_->pscr; pa.arrive = dm_->parr; pa.diag_m = dm_->h_w + 32;
        pa.rq = dm_->rq; pa.rs = dm_->rs;
        if (gs_nk0_ > 0) {
            pa.mix = gs_mix_; pa.mseq_m = dm_->h_w + 48; pa.mdone_m = dm_->h_w + 52;
            pa.mxq_m = gs_h_xq_; pa.mxq = gs_pxq_;
            pa.mx_m = gs_h_x_; pa.mixl = gs_mixl_; pa.part_m = gs_h_part_;
        }
        if (peer_stamps_on()) {   // peer-card diagnostic (BASALT_PEER_STAMPS)
            std::memset(g_pst.h, 0, (size_t) kPstRings * 2 * 8 * sizeof(unsigned long long));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            pa.stamps_m = g_pst.h;
        }
        if (!basalt::kernels::dec_moe_zero_words(dm_->parr, 16, dm_->ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, dm_->pblocks, dm_->ps, dm_->pq2)) {
            err = "verify: the peer window did not launch";
            return false;
        }
        dm_->peer_live = true;
    }
    for (DecMoe::XPeer& x : dm_->xp) {   // further peers: the same window on their own ring
        for (int w : {0, 16, 32}) *(volatile uint32_t*) (x.h_w + w) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const OnDevice on(x.dev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = x.h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = x.h_w + 0; pa.done_m = x.h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = x.ptab; pa.lay = x.lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = x.pscr; pa.arrive = x.parr; pa.diag_m = x.h_w + 32;
        pa.rq = x.rq; pa.rs = x.rs;
        if (!basalt::kernels::dec_moe_zero_words(x.parr, 16, x.ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, x.pblocks, x.ps, dm_->pq2)) {
            err = "verify: a further peer's window did not launch";
            return false;
        }
        x.live = true;
    }
    if (gm) gm->mark(14);
    return true;
}

// The peer window ends with CUDA0's last ring.
bool Verifier::dm_end(std::string& err) {
    if (dm_ == nullptr) return true;
    for (DecMoe::XPeer& x : dm_->xp) {
        if (!x.live) continue;
        const OnDevice on(x.dev);
        x.live = false;
        const cudaError_t pe = cudaStreamSynchronize(x.ps);
        if (pe != cudaSuccess) { err = std::string("verify: a further peer's window: ") + cudaGetErrorString(pe); return false; }
        const std::string ab = basalt::kernels::dec_moe_peer_diag(x.h_w + 32);
        if (!ab.empty()) { err = "verify: a further peer's window " + ab; return false; }
    }
    if (!dm_->peer_live) return true;
    const OnDevice on(dm_->pdev);
    dm_->peer_live = false;
    const cudaError_t pe = cudaStreamSynchronize(dm_->ps);
    if (pe != cudaSuccess) { err = std::string("verify: the peer window: ") + cudaGetErrorString(pe); return false; }
    const std::string ab = basalt::kernels::dec_moe_peer_diag(dm_->h_w + 32);
    if (!ab.empty()) { err = "verify: the peer window " + ab; return false; }
    if (peer_stamps_on()) peer_stamps_collect(dm_->rings, dm_->pblocks);   // peer-card diagnostic; pblocks: the per-block stamps
    return true;
}

}  // namespace basalt::core
