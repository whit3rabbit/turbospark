// ============================================================================
// moe_inkling2_lamq - routed-expert decode pair for Inkling 2 LookAtMyQuant
// installs (docs/LAMQ.md). PORT-LOCAL. Contract: turbospark_compute::lamq
// (LamqLinear rows) and turbospark_compute::inkling2::situ, held by
// tests/moe_inkling2_lamq_parity.rs.
//
// COMPILED CONCATENATED AFTER moe.metal (RoutedBlobs) and lamq.metal
// (lamq_window). The rotations are separate lamq_rht dispatches:
//   x_rot   = rht_in(x, gu_signs)               once per token
//   acts    = situ(gate . x_rot, up . x_rot)    phase 1 (this file)
//   act_rot = rht_in(acts, dn_signs)            per slot
//   y       = sum_slot w * (down . act_rot)     phase 2 (this file)
//
// One expert blob (repack `inkling2_lamq`):
//   fmt   -> 4 x u32: gate-up code id, down code id, gate-up row bytes, down
//            row bytes (ids are compute::lamq::LamqCode::id, trellis only)
//   gu    -> 2F rows of D: gate is rows [0, F), up is [F, 2F) (NOT the HF
//            interleave); padded to the layer's widest format
//   gu_s  -> 2F f16 row scales
//   dn    -> D rows of F, padded likewise; dn_s -> D f16 row scales
// Formats differ per expert, so k, g and the hash come from the record at
// run time, and every trellis LUT is bound in one packed buffer.
//
// Like moe_inkling2: the sigmoid sees the raw gate, phase 2 adds no residual,
// and phase 2 reduces exactly top_k slots in slot (router ranking) order.
// ============================================================================

struct LamqExpertOffsets {
    uint fmt;
    uint gu;
    uint gu_s;
    uint dn;
    uint dn_s;
};

// Packed LUT offsets in halves: ht2 (512 x 4), ht3 (1024 x 2), rtr1..rtr4
// (4096 each). The host builds the buffer in this order (LAMQ_LUT_PACK).
struct LamqCodeParams {
    uint k;
    uint g;
    uint shift;
    uint lut_off;
};

static inline LamqCodeParams lamq_code_params(uint id) {
    switch (id) {
        case 1u: return {2u, 4u, 23u, 0u};
        case 2u: return {3u, 2u, 22u, 2048u};
        case 3u: return {1u, 1u, 0u, 4096u};
        case 4u: return {2u, 1u, 0u, 8192u};
        case 5u: return {3u, 1u, 0u, 12288u};
        // The host validates every record at open; an unknown id still reads
        // in bounds (rtr4's table) rather than past the pack.
        default: return {4u, 1u, 0u, 16384u};
    }
}

// One coded row against an F32 x. `lut` is the code's table, from device
// memory or staged in threadgroup memory: the values and the order are the
// same either way, so the two forms are bit-identical.
template <typename Lut>
static inline float lamq_expert_row(
    device const uchar* row, uint row_bytes, uint n, LamqCodeParams c,
    Lut lut, device const float* x, uint lane
) {
    device const uint* w = (device const uint*)row;
    const uint words = row_bytes / 4u;
    float acc = 0.0f;
    for (uint j = lane * c.g; j < n; j += 32u * c.g) {
        const uint s = lamq_window(w, words, (n - 1u - (j + c.g - 1u)) * c.k);
        const uint e = (c.shift != 0u ? (s * 0x9E3779B1u) >> c.shift : s) * c.g;
        for (uint t = 0; t < c.g; ++t) acc += float(lut[e + t]) * x[j + t];
    }
    return simd_sum(acc);
}

// Table length in halves for a code id (ht2 512x4, ht3 1024x2, rtr 4096).
static inline uint lamq_table_len(uint id) {
    return (id == 1u || id == 2u) ? 2048u : 4096u;
}

static inline float lamq_softcap(float x, float cap) {
    return cap * precise::tanh(x / cap);
}

// `tlut` holds this expert's gate-up table when `staged`, else the device
// pack is read.
static inline void lamq_phase1_body(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    threadgroup const half* tlut, bool staged,
    device const float* x, device float* acts, uint D, uint F, float2 caps,
    uint slot, uint f, uint lane
) {
    device const uint* fmt = (device const uint*)(base + o.fmt);
    const LamqCodeParams c = lamq_code_params(fmt[0]);
    const uint rb = fmt[2];
    device const half* sc = (device const half*)(base + o.gu_s);
    device const uchar* gr = base + o.gu + ulong(f) * rb;
    device const uchar* ur = base + o.gu + ulong(F + f) * rb;
    float g, u;
    if (staged) {
        g = lamq_expert_row(gr, rb, D, c, tlut, x, lane);
        u = lamq_expert_row(ur, rb, D, c, tlut, x, lane);
    } else {
        g = lamq_expert_row(gr, rb, D, c, luts + c.lut_off, x, lane);
        u = lamq_expert_row(ur, rb, D, c, luts + c.lut_off, x, lane);
    }
    g *= float(sc[f]);
    u *= float(sc[F + f]);
    if (lane == 0u) {
        const float sig = 1.0f / (1.0f + precise::exp(-g));
        acts[slot * F + f] = lamq_softcap(g, caps.x) * sig * lamq_softcap(u, caps.y);
    }
}

// Stages the gate-up table of `base`'s expert when every row of this
// threadgroup belongs to it; returns whether it did. Every thread calls it
// and meets the barrier.
static inline bool lamq_stage_gu_table(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    threadgroup half* tlut, bool same_slot, uint tid, uint nt
) {
    if (same_slot) {
        device const uint* fmt = (device const uint*)(base + o.fmt);
        const LamqCodeParams c = lamq_code_params(fmt[0]);
        const uint len = lamq_table_len(fmt[0]);
        for (uint i = tid; i < len; i += nt) tlut[i] = luts[c.lut_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return same_slot;
}

static inline float lamq_phase2_term(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    device const float* act, uint F, uint d, uint lane
) {
    device const uint* fmt = (device const uint*)(base + o.fmt);
    const LamqCodeParams c = lamq_code_params(fmt[1]);
    const uint rb = fmt[3];
    device const half* sc = (device const half*)(base + o.dn_s);
    return lamq_expert_row(base + o.dn + ulong(d) * rb, rb, F, c, luts + c.lut_off, act, lane) * float(sc[d]);
}

constant constexpr uint kLamqExpRowsPerTG = 32; // phase 1, = ROWS_PER_THREADGROUP
constant constexpr uint kLamqExpMaxSlots = 8;

// Streamed: slot s's blob is routed.blob[s].
[[kernel, max_total_threads_per_threadgroup(1024)]]
kernel void moe_inkling2_lamq_phase1(
    device const RoutedBlobs&    routed [[buffer(0)]],
    constant LamqExpertOffsets&  o      [[buffer(1)]],
    device const float*          x      [[buffer(2)]],
    device float*                acts   [[buffer(3)]],
    constant uint&               D      [[buffer(4)]],
    constant uint&               F      [[buffer(5)]],
    constant uint&               top_k  [[buffer(6)]],
    constant float2&             caps   [[buffer(7)]],
    device const half*           luts   [[buffer(11)]],
    uint                         tg     [[threadgroup_position_in_grid]],
    uint                         sg     [[simdgroup_index_in_threadgroup]],
    uint                         lane   [[thread_index_in_simdgroup]],
    uint                         tid    [[thread_position_in_threadgroup]],
    uint                         nt     [[threads_per_threadgroup]]
) {
    threadgroup half tlut[4096];
    const uint total = top_k * F, row0 = tg * kLamqExpRowsPerTG;
    const uint last = min(row0 + kLamqExpRowsPerTG, total) - 1u;
    const uint slot0 = min(row0, total - 1u) / F;
    const bool staged = lamq_stage_gu_table(
        routed.blob[slot0], o, luts, tlut, row0 < total && last / F == slot0, tid, nt);
    const uint rowg = row0 + sg;
    if (rowg >= total) return;
    const uint slot = rowg / F, f = rowg % F;
    lamq_phase1_body(routed.blob[slot], o, luts, tlut, staged, x, acts, D, F, caps, slot, f, lane);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_inkling2_lamq_phase2(
    device const RoutedBlobs&    routed    [[buffer(0)]],
    constant LamqExpertOffsets&  o         [[buffer(1)]],
    device const float*          act       [[buffer(2)]],
    device const half*           routing_w [[buffer(3)]],
    device half*                 y         [[buffer(4)]],
    constant uint&               D         [[buffer(5)]],
    constant uint&               F         [[buffer(6)]],
    constant uint&               top_k     [[buffer(7)]],
    device const half*           luts      [[buffer(11)]],
    uint                         d         [[threadgroup_position_in_grid]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kLamqExpMaxSlots];
    float w = 0.0f;
    if (sg < top_k && d < D) {
        w = float(routing_w[sg]) * lamq_phase2_term(routed.blob[sg], o, luts, act + sg * F, F, d, lane);
    }
    if (lane == 0u) partial[sg] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u && d < D) {
        float acc = 0.0f;
        for (uint i = 0; i < top_k; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}

// Resident: the layer in one buffer at a uniform stride, slot s's expert is
// sel[s] (router ranking order, clamped in bounds).
[[kernel, max_total_threads_per_threadgroup(1024)]]
kernel void moe_inkling2_lamq_phase1_idx(
    device const uchar*          layer      [[buffer(0)]],
    constant LamqExpertOffsets&  o          [[buffer(1)]],
    device const float*          x          [[buffer(2)]],
    device float*                acts       [[buffer(3)]],
    constant uint&               D          [[buffer(4)]],
    constant uint&               F          [[buffer(5)]],
    constant uint&               top_k      [[buffer(6)]],
    constant float2&             caps       [[buffer(7)]],
    device const uint*           sel        [[buffer(8)]],
    constant ulong&              stride     [[buffer(9)]],
    constant uint&               n_experts  [[buffer(10)]],
    device const half*           luts       [[buffer(11)]],
    uint                         tg         [[threadgroup_position_in_grid]],
    uint                         sg         [[simdgroup_index_in_threadgroup]],
    uint                         lane       [[thread_index_in_simdgroup]],
    uint                         tid        [[thread_position_in_threadgroup]],
    uint                         nt         [[threads_per_threadgroup]]
) {
    threadgroup half tlut[4096];
    const uint total = top_k * F, row0 = tg * kLamqExpRowsPerTG;
    const uint last = min(row0 + kLamqExpRowsPerTG, total) - 1u;
    const uint slot0 = min(row0, total - 1u) / F;
    const uint e0 = min(sel[slot0], n_experts - 1u);
    const bool staged = lamq_stage_gu_table(
        layer + ulong(e0) * stride, o, luts, tlut, row0 < total && last / F == slot0, tid, nt);
    const uint rowg = row0 + sg;
    if (rowg >= total) return;
    const uint slot = rowg / F, f = rowg % F;
    const uint e = min(sel[slot], n_experts - 1u);
    lamq_phase1_body(layer + ulong(e) * stride, o, luts, tlut, staged, x, acts, D, F, caps, slot, f, lane);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_inkling2_lamq_phase2_idx(
    device const uchar*          layer      [[buffer(0)]],
    constant LamqExpertOffsets&  o          [[buffer(1)]],
    device const float*          act        [[buffer(2)]],
    device const half*           routing_w  [[buffer(3)]],
    device half*                 y          [[buffer(4)]],
    constant uint&               D          [[buffer(5)]],
    constant uint&               F          [[buffer(6)]],
    constant uint&               top_k      [[buffer(7)]],
    device const uint*           sel        [[buffer(8)]],
    constant ulong&              stride     [[buffer(9)]],
    constant uint&               n_experts  [[buffer(10)]],
    device const half*           luts       [[buffer(11)]],
    uint                         d          [[threadgroup_position_in_grid]],
    uint                         sg         [[simdgroup_index_in_threadgroup]],
    uint                         lane       [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kLamqExpMaxSlots];
    float w = 0.0f;
    if (sg < top_k && d < D) {
        const uint e = min(sel[sg], n_experts - 1u);
        w = float(routing_w[sg]) * lamq_phase2_term(layer + ulong(e) * stride, o, luts, act + sg * F, F, d, lane);
    }
    if (lane == 0u) partial[sg] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u && d < D) {
        float acc = 0.0f;
        for (uint i = 0; i < top_k; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}
