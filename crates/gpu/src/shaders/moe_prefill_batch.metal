// ============================================================================
// Batched routed-expert pair for chunked prefill (docs/BATCHED_PREFILL.md
// steps 2 and 3) -- PORT-LOCAL, not vendored.
//
// The vendored decode pair (`moe_phase1_gate_up_act_u16load` +
// `moe_phase2_down_reduce_k8`) is single-token by construction: one `x`
// vector, one routing-weight row, and a `RoutedBlobs` argument buffer
// holding top_k pointers. Swift's answer for prefill is vendored one
// section over (`dsv4_prefill_moe_*`, pair-major route lists), but it
// TILES the route list eight experts at a time because its argument
// buffer is also eight pointers wide, which is why its phase 2 is split
// into a per-route down kernel and a separate rank-ordered reduce.
//
// This port's slot cache holds every expert of a chunk's UNION resident
// at once (the caller guarantees `union <= slot_count`, capping the
// routed sub-batch near M=8 at 32 slots), so a WIDE argument buffer
// removes the tiling: one dense dispatch per kernel, and phase 2 stays
// fused exactly like the decode kernel it is derived from.
//
// Numerics are the decode pair's numerics on the same bytes -- the same
// `moe_int4_*` row helpers, the same activation, the same f16 routing
// weights -- so a chunk's routed output is bit-identical to M sequential
// decode passes. The reduce order is router-rank order per token
// (AGENTS.md Gotcha 27): rank r IS the router's ranking of that token's
// experts, so summing partial[0..top_k) in rank order is the same
// summation order the decode kernel takes over dispatch slots.
// ============================================================================

constant constexpr uint kMaxPrefillExpertBindings = 32;

struct RoutedBlobsWide {
    device const uint8_t* blob[kMaxPrefillExpertBindings];
};

struct MoePrefillRoute {
    uint token;   // row of `x` and of the output `y`
    uint rank;    // routing slot 0..top_k-1, in the router's ranking
    uint slot;    // cache slot the expert's blob lives in
    uint reserved;
};

// acts[(token * top_k + rank) * F + f] = act(gate_f(x_token)) * up_f(x_token),
// one SIMD group per (route, f) row -- the decode phase-1 body with the
// route list replacing the slot arithmetic.
kernel void moe_prefill_phase1_routes_int4(
    device const RoutedBlobsWide& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],           // [tokens, D]
    device half* acts [[buffer(3)]],              // [tokens * top_k, F]
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const MoePrefillRoute* routes [[buffer(7)]],
    constant uint& route_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint KK = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= route_count * FF) return;
    const uint route_index = rowg / FF;
    const uint f = rowg % FF;
    const MoePrefillRoute r = routes[route_index];

    device const uint8_t* base = routed.blob[r.slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        x + uint(r.token) * DD, f, DD, lane);
    if (lane == 0) {
        acts[(uint(r.token) * KK + uint(r.rank)) * FF + f] =
            half(moe_hidden_activation(gu.x) * gu.y);
    }
}

// y[token] = sum_rank routing_w[token, rank] * down(acts[token, rank]) --
// `moe_phase2_down_reduce_k8` with a token axis. One threadgroup per
// (token, d) output; SIMD group r owns rank r's down row. The residual
// seed is the literal the decode kernel reads out of the all-zero
// `zero_hidden` row, and the accumulation is rank-ordered and per token,
// so it cannot depend on the chunk boundary (AGENTS.md Gotcha 27).
kernel void moe_prefill_phase2_fused_int4(
    device const RoutedBlobsWide& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],         // [tokens * top_k, F]
    device const half* routing_w [[buffer(3)]],    // [tokens * top_k]
    device const MoePrefillRoute* routes [[buffer(4)]],
    device half* y [[buffer(5)]],                  // [tokens, D]
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    constant uint& top_k [[buffer(8)]],
    constant uint& tokens [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint KK = moe_fc_top_k(top_k);
    const uint t = tg.y;
    const uint d = tg.x;
    if (t >= tokens) return;

    if (sg_idx < KK) {
        const uint pair = t * KK + sg_idx;
        const MoePrefillRoute r = routes[pair];
        device const uint8_t* base = routed.blob[r.slot];
        const ExpertOffsets re = routed_offsets;
        const float value = moe_int4_gemv_row_simd_dev_vec(
            base + re.down_W_off,
            (device const bfloat*)(base + re.down_s_off),
            (device const bfloat*)(base + re.down_b_off),
            acts + pair * FF, d, FF, lane);
        if (lane == 0) partial[sg_idx] = float(routing_w[pair]) * value;
    } else if (lane == 0) {
        // Padding mirrors the decode kernel's zero-padded routing weights
        // past top_k: +0.0f added in the same rank positions.
        partial[sg_idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = 0.0f;
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[t * DD + d] = half(acc);
    }
}
