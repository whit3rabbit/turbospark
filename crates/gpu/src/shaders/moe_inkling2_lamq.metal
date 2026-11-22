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

// One coded row against an F32 x, one weight per lane per step (phase 2).
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

// Offset (halves) of the rtr1 three-weight table in the pack: entry `idx` (14
// bits) holds the 12-bit table values at idx >> 2, idx >> 1 and idx, so one
// lookup gives three consecutive columns (LAMQ_RTR1_MT_OFF in the host).
constant constexpr uint kLamqRtr1MtOff = 20480u;

// Bits [off, off + nb) of the 96 bits held in w0..w2 (off + nb <= 96).
static inline uint lamq_bits96(ulong lo, ulong hi, uint off, uint nb) {
    const uint m = (1u << nb) - 1u;
    return off + nb <= 64u ? uint(lo >> off) & m : uint(hi >> (off - 32u)) & m;
}

// Dot product of columns [16c, 16c + 16) of one coded row with `xv[16]`
// (the same columns of x). Every window is the 12 bits `lamq_window` reads
// for that column (words past the row end read as zero), so the weights are
// identical to the per-column loop; only the summation order differs. The
// chunk spans at most 16k + 12 <= 76 bits from a word-aligned base offset of
// 0 or 16, so three words always hold it. Needs n % 16 == 0.
static inline float lamq_chunk16(
    device const uint* w, uint words, uint n, uint c, uint id, LamqCodeParams cp,
    device const half* luts, thread const float* xv
) {
    const uint B = (n - 16u * (c + 1u)) * cp.k, wb = B >> 5, o = B & 31u;
    const uint w0 = w[wb];
    const uint w1 = wb + 1u < words ? w[wb + 1u] : 0u;
    const uint w2 = wb + 2u < words ? w[wb + 2u] : 0u;
    const ulong lo = ulong(w0) | (ulong(w1) << 32);
    const ulong hi = ulong(w1) | (ulong(w2) << 32);
    device const half* lut = luts + cp.lut_off;
    float acc = 0.0f;
    if (id == 3u) {
        // rtr1: five three-column lookups and one single (12 bits, slot 2).
        device const half4* mt = (device const half4*)(luts + kLamqRtr1MtOff);
        for (uint i = 0; i < 15u; i += 3u) {
            const half4 v = mt[lamq_bits96(lo, hi, o + (15u - (i + 2u)), 14u)];
            acc += float(v.x) * xv[i] + float(v.y) * xv[i + 1u] + float(v.z) * xv[i + 2u];
        }
        acc += float(mt[lamq_bits96(lo, hi, o, 12u)].z) * xv[15];
    } else if (cp.shift != 0u) {
        // ht2 (g = 4) and ht3 (g = 2): one hashed lookup per group.
        for (uint i = 0; i < 16u; i += cp.g) {
            const uint st = lamq_bits96(lo, hi, o + (15u - (i + cp.g - 1u)) * cp.k, 12u);
            const uint e = ((st * 0x9E3779B1u) >> cp.shift) * cp.g;
            for (uint t = 0; t < cp.g; ++t) acc += float(lut[e + t]) * xv[i + t];
        }
    } else {
        // rtr2..rtr4: one direct lookup per column.
        for (uint j = 0; j < 16u; ++j) {
            acc += float(lut[lamq_bits96(lo, hi, o + (15u - j) * cp.k, 12u)]) * xv[j];
        }
    }
    return acc;
}

static inline float lamq_softcap(float x, float cap) {
    return cap * precise::tanh(x / cap);
}

// One gate/up row pair per 16-lane half of a simdgroup (`hl` its lane in the
// half): each step loads a 16-column chunk of x once and decodes that chunk
// of both rows, the authors' row layout (docs/LAMQ.md step 9).
static inline void lamq_phase1_body(
    device const uchar* base, constant LamqExpertOffsets& o, device const half* luts,
    device const float* x, device float* acts, uint D, uint F, float2 caps,
    uint slot, uint f, uint hl
) {
    device const uint* fmt = (device const uint*)(base + o.fmt);
    const uint id = fmt[0];
    const LamqCodeParams c = lamq_code_params(id);
    const uint rb = fmt[2], words = rb / 4u;
    device const half* sc = (device const half*)(base + o.gu_s);
    device const uint* gr = (device const uint*)(base + o.gu + ulong(f) * rb);
    device const uint* ur = (device const uint*)(base + o.gu + ulong(F + f) * rb);
    float g = 0.0f, u = 0.0f;
    for (uint ch = hl; ch < D / 16u; ch += 16u) {
        float xv[16];
        for (uint q = 0; q < 4u; ++q) {
            const float4 v = ((device const float4*)(x + ch * 16u))[q];
            xv[4u * q] = v.x; xv[4u * q + 1u] = v.y; xv[4u * q + 2u] = v.z; xv[4u * q + 3u] = v.w;
        }
        g += lamq_chunk16(gr, words, D, ch, id, c, luts, xv);
        u += lamq_chunk16(ur, words, D, ch, id, c, luts, xv);
    }
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
    uint                         lane   [[thread_index_in_simdgroup]]
) {
    const uint rowg = tg * kLamqExpRowsPerTG + sg * 2u + (lane >> 4);
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F, f = rowg % F;
    lamq_phase1_body(routed.blob[slot], o, luts, x, acts, D, F, caps, slot, f, lane & 15u);
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
    uint                         lane       [[thread_index_in_simdgroup]]
) {
    const uint rowg = tg * kLamqExpRowsPerTG + sg * 2u + (lane >> 4);
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F, f = rowg % F;
    const uint e = min(sel[slot], n_experts - 1u);
    lamq_phase1_body(layer + ulong(e) * stride, o, luts, x, acts, D, F, caps, slot, f, lane & 15u);
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
