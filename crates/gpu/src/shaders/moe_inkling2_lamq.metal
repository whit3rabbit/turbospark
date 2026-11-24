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

// Offset (halves) of the rtr1 three-weight table in the pack: entry `idx` (14
// bits) holds the 12-bit table values at idx >> 2, idx >> 1 and idx, so one
// lookup gives three consecutive columns (LAMQ_RTR1_MT_OFF in the host).
constant constexpr uint kLamqRtr1MtOff = 20480u;

// Code parameters by id (compute::lamq::LamqCode::id): bits per weight k,
// weights per lookup g, hash shift (0 = direct index), and the table offset
// in halves in the LUT pack, which the host builds in LAMQ_LUT_PACK order:
// ht2 (512 x 4), ht3 (1024 x 2), rtr1..rtr4 (4096 each). Compile-time, so
// each format's 16-column loop unrolls with constant window offsets; the
// run-time switch on the id is hoisted to the row (docs/LAMQ.md step 13).
constexpr uint lamq_ck(uint id) { return id == 1u ? 2u : (id == 2u ? 3u : id - 2u); }
constexpr uint lamq_cg(uint id) { return id == 1u ? 4u : (id == 2u ? 2u : 1u); }
constexpr uint lamq_cshift(uint id) { return id == 1u ? 23u : (id == 2u ? 22u : 0u); }
constexpr uint lamq_coff(uint id) { return id == 1u ? 0u : (id == 2u ? 2048u : 4096u * (id - 2u)); }

// Dot product of columns [16c, 16c + 16) of one coded row with `xv[16]`
// (the same columns of x). Every window is the 12 bits `lamq_window` reads
// for that column (words past the row end read as zero), so the weights are
// identical to the per-column loop; only the summation order differs. The
// chunk spans at most 16k + 12 <= 76 bits from a word-aligned base offset of
// 0 or 16, so three words always hold it. Needs n % 16 == 0. `lut` is this
// format's table, in device memory or staged in threadgroup memory (the
// same f16 values either way); rtr1 reads the three-weight table in `luts`.
template <uint ID, typename LP>
static inline float lamq_chunk16(
    device const uint* w, uint words, uint n, uint c,
    device const half* luts, LP lut, thread const float* xv
) {
    constexpr uint K = lamq_ck(ID), G = lamq_cg(ID), SHIFT = lamq_cshift(ID);
    const uint B = (n - 16u * (c + 1u)) * K, wb = B >> 5, o = B & 31u;
    const uint w0 = w[wb];
    const uint w1 = wb + 1u < words ? w[wb + 1u] : 0u;
    const uint w2 = wb + 2u < words ? w[wb + 2u] : 0u;
    const ulong lo = ulong(w0) | (ulong(w1) << 32);
    const ulong hi = ulong(w1) | (ulong(w2) << 32);
    float acc = 0.0f;
    if (ID == 3u) {
        // rtr1: five three-column lookups and one single (12 bits, slot 2).
        device const half4* mt = (device const half4*)(luts + kLamqRtr1MtOff);
        for (uint i = 0; i < 15u; i += 3u) {
            const half4 v = mt[lamq_bits96(lo, hi, o + (15u - (i + 2u)), 14u)];
            acc += float(v.x) * xv[i] + float(v.y) * xv[i + 1u] + float(v.z) * xv[i + 2u];
        }
        acc += float(mt[lamq_bits96(lo, hi, o, 12u)].z) * xv[15];
    } else if (SHIFT != 0u) {
        // ht2 (g = 4) and ht3 (g = 2): one hashed lookup per group.
        for (uint i = 0; i < 16u; i += G) {
            const uint st = lamq_bits96(lo, hi, o + (15u - (i + G - 1u)) * K, 12u);
            const uint e = ((st * 0x9E3779B1u) >> SHIFT) * G;
            for (uint t = 0; t < G; ++t) acc += float(lut[e + t]) * xv[i + t];
        }
    } else {
        // rtr2..rtr4: one direct lookup per column.
        for (uint j = 0; j < 16u; ++j) {
            acc += float(lut[lamq_bits96(lo, hi, o + (15u - j) * K, 12u)]) * xv[j];
        }
    }
    return acc;
}

static inline void lamq_load16(device const float* p, thread float* xv) {
    for (uint q = 0; q < 4u; ++q) {
        const float4 v = ((device const float4*)p)[q];
        xv[4u * q] = v.x; xv[4u * q + 1u] = v.y; xv[4u * q + 2u] = v.z; xv[4u * q + 3u] = v.w;
    }
}

// Phase 1 row-pair loop for one format: this half-warp's chunks of the gate
// and up rows, x loaded once per chunk.
template <uint ID, typename LP>
static inline float2 lamq_p1_rows(
    device const uint* gr, device const uint* ur, uint words, uint D,
    device const float* x, device const half* luts, LP lut, uint hl
) {
    float g = 0.0f, u = 0.0f;
    for (uint ch = hl; ch < D / 16u; ch += 16u) {
        float xv[16];
        lamq_load16(x + ch * 16u, xv);
        g += lamq_chunk16<ID>(gr, words, D, ch, luts, lut, xv);
        u += lamq_chunk16<ID>(ur, words, D, ch, luts, lut, xv);
    }
    return float2(g, u);
}

// Phase 2 row loop for one format, chunks across the simdgroup.
template <uint ID>
static inline float lamq_p2_row(
    device const uint* row, uint words, uint F, device const float* act,
    device const half* luts, uint lane
) {
    float acc = 0.0f;
    for (uint ch = lane; ch < F / 16u; ch += 32u) {
        float xv[16];
        lamq_load16(act + ch * 16u, xv);
        acc += lamq_chunk16<ID>(row, words, F, ch, luts, luts + lamq_coff(ID), xv);
    }
    return acc;
}

static inline float lamq_softcap(float x, float cap) {
    return cap * precise::tanh(x / cap);
}

// One gate/up row pair per 16-lane half of a simdgroup (`hl` its lane in the
// half): each step loads a 16-column chunk of x once and decodes that chunk
// of both rows, the authors' row layout (docs/LAMQ.md step 9). `tlut` holds
// this row's ht2/ht3 table when `staged` (the caller staged it for the whole
// threadgroup, every row of which shares the expert).
static inline void lamq_phase1_body(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    threadgroup const half* tlut, bool staged,
    device const float* x, device float* acts, uint D, uint F, float2 caps,
    uint slot, uint f, uint hl
) {
    device const uint* fmt = (device const uint*)(base + o.fmt);
    const uint id = fmt[0];
    const uint rb = fmt[2], words = rb / 4u;
    device const half* sc = (device const half*)(base + o.gu_s);
    device const uint* gr = (device const uint*)(base + o.gu + ulong(f) * rb);
    device const uint* ur = (device const uint*)(base + o.gu + ulong(F + f) * rb);
    float2 gu;
    switch (id) {
        case 1u:
            gu = staged ? lamq_p1_rows<1u>(gr, ur, words, D, x, luts, tlut, hl)
                        : lamq_p1_rows<1u>(gr, ur, words, D, x, luts, luts + lamq_coff(1u), hl);
            break;
        case 2u:
            gu = staged ? lamq_p1_rows<2u>(gr, ur, words, D, x, luts, tlut, hl)
                        : lamq_p1_rows<2u>(gr, ur, words, D, x, luts, luts + lamq_coff(2u), hl);
            break;
        case 3u: gu = lamq_p1_rows<3u>(gr, ur, words, D, x, luts, luts + lamq_coff(3u), hl); break;
        case 4u: gu = lamq_p1_rows<4u>(gr, ur, words, D, x, luts, luts + lamq_coff(4u), hl); break;
        case 5u: gu = lamq_p1_rows<5u>(gr, ur, words, D, x, luts, luts + lamq_coff(5u), hl); break;
        // The host validates every record at open; an unknown id still reads
        // in bounds (rtr4's table), as lamq_code_params does.
        default: gu = lamq_p1_rows<6u>(gr, ur, words, D, x, luts, luts + lamq_coff(6u), hl); break;
    }
    float g = gu.x, u = gu.y;
    for (uint s = 8u; s >= 1u; s >>= 1) {
        g += simd_shuffle_xor(g, s);
        u += simd_shuffle_xor(u, s);
    }
    g *= float(sc[f]);
    u *= float(sc[F + f]);
    if (hl == 0u) {
        const float sig = 1.0f / (1.0f + precise::exp(-g));
        acts[slot * F + f] = lamq_softcap(g, caps.x) * sig * lamq_softcap(u, caps.y);
    }
}

// Stages the ht2/ht3 gate-up table of the expert at `base` (2048 halves for
// either) into `tlut` for the whole threadgroup; returns whether it did. Only
// called when every row of the threadgroup shares that expert.
static inline bool lamq_stage_gu_table(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    threadgroup half* tlut, uint tid, uint nt
) {
    const uint id = ((device const uint*)(base + o.fmt))[0];
    if (id != 1u && id != 2u) return false;
    const uint off = id == 1u ? lamq_coff(1u) : lamq_coff(2u);
    for (uint i = tid; i < 2048u; i += nt) tlut[i] = luts[off + i];
    return true;
}

static inline float lamq_phase2_term(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    device const float* act, uint F, uint d, uint lane
) {
    device const uint* fmt = (device const uint*)(base + o.fmt);
    const uint id = fmt[1];
    const uint rb = fmt[3], words = rb / 4u;
    device const half* sc = (device const half*)(base + o.dn_s);
    device const uint* row = (device const uint*)(base + o.dn + ulong(d) * rb);
    // The same 16-column chunk decode as phase 1, across the simdgroup.
    float acc;
    switch (id) {
        case 1u: acc = lamq_p2_row<1u>(row, words, F, act, luts, lane); break;
        case 2u: acc = lamq_p2_row<2u>(row, words, F, act, luts, lane); break;
        case 3u: acc = lamq_p2_row<3u>(row, words, F, act, luts, lane); break;
        case 4u: acc = lamq_p2_row<4u>(row, words, F, act, luts, lane); break;
        case 5u: acc = lamq_p2_row<5u>(row, words, F, act, luts, lane); break;
        default: acc = lamq_p2_row<6u>(row, words, F, act, luts, lane); break;
    }
    return simd_sum(acc) * float(sc[d]);
}

constant constexpr uint kLamqExpRowsPerTG = 64; // phase 1: 2 per simdgroup, = ROWS_PER_THREADGROUP
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
    // When F is a multiple of the threadgroup's rows, all of them belong to
    // one slot, so its ht table is staged once before any thread returns.
    threadgroup half tlut[2048];
    bool staged = false;
    if (F % kLamqExpRowsPerTG == 0u) {
        const uint slot0 = (tg * kLamqExpRowsPerTG) / F;
        if (slot0 < top_k) staged = lamq_stage_gu_table(routed.blob[slot0], o, luts, tlut, tid, nt);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rowg = tg * kLamqExpRowsPerTG + sg * 2u + (lane >> 4);
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F, f = rowg % F;
    lamq_phase1_body(routed.blob[slot], o, luts, tlut, staged, x, acts, D, F, caps, slot, f, lane & 15u);
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
    threadgroup half tlut[2048];
    bool staged = false;
    if (F % kLamqExpRowsPerTG == 0u) {
        const uint slot0 = (tg * kLamqExpRowsPerTG) / F;
        if (slot0 < top_k) {
            const uint e0 = min(sel[slot0], n_experts - 1u);
            staged = lamq_stage_gu_table(layer + ulong(e0) * stride, o, luts, tlut, tid, nt);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rowg = tg * kLamqExpRowsPerTG + sg * 2u + (lane >> 4);
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F, f = rowg % F;
    const uint e = min(sel[slot], n_experts - 1u);
    lamq_phase1_body(layer + ulong(e) * stride, o, luts, tlut, staged, x, acts, D, F, caps, slot, f, lane & 15u);
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
