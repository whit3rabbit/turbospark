// ============================================================================
// Batched routed-expert pair for chunked prefill, MXFP4 blobs
// (docs/BATCHED_PREFILL.md step 5) -- PORT-LOCAL, not vendored.
//
// This is `moe_prefill_batch.metal`'s two kernels with `gpt-oss`'s block
// type and expert math substituted in, and it is deliberately a
// SUBSTITUTION rather than a generalization: the two pairs share no row
// addressing at all (an affine blob carries scale and bias PLANES the row
// helper strides through, an MXFP4 blob carries a 17-byte block with its
// own E8M0 exponent inline), so a function-constant variant selecting
// between them would be two kernels wearing one name -- and one whose
// specialization byte has to reach `constants_key` or the pipeline cache
// hands back whichever compiled first (`crates/gpu` Gotcha 1).
//
// It lives in its own file, rather than at the end of
// `moe_prefill_batch.metal`, because it needs a LONGER concatenation: the
// MXFP4 row helper and the clamped SwiGLU live in `moe_gguf.metal`, which
// itself sits behind three dequant files. Keeping the affine pair's source
// short keeps that pair's compile off this one's dependency chain.
//
// WHY THIS ARM AND NOT THE Q4_K/Q6_K ONE FIRST, measured 2026-08-27 before
// either was written (`docs/BATCHED_PREFILL.md`, "Step 5's two arms"):
// `gpt-oss`'s routed pair is 61.4% of its prefill GPU device time against
// Gemma's 38.2%, its un-batchable expert `pread` is 8.2% against 25-37%,
// and with 32 experts at top-4 its union stays under the slot count at
// every M -- so it is the ONE family here that reaches M=16, where both
// 128-expert families cap at M=8 on `union(M) <= slot_count`.
//
// Numerics are the MXFP4 DECODE pair's numerics on the same bytes: the
// same `dequant_mxfp4_row_simd`, the same `moe_activate_mxfp4`, the same
// bias placement, the same f16 routing weights. So a chunk's routed output
// is bit-identical to M sequential decode passes, which is what
// `crates/gpu/tests/moe_prefill_batch_gguf_parity.rs` asserts.
// ============================================================================

// acts[(token * top_k + rank) * F + f] = swiglu_oai(gate_f(x_token) + gb[f],
//                                                   up_f(x_token) + ub[f]),
// one SIMD group per (route, f) row -- the MXFP4 decode phase-1 body with
// the route list replacing the slot arithmetic.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_prefill_phase1_routes_mxfp4(
    device const RoutedBlobsWide&  routed          [[buffer(0)]],
    constant ExpertOffsets&        routed_offsets  [[buffer(1)]],
    device const half*             x               [[buffer(2)]],   // [tokens, D]
    device half*                   acts            [[buffer(3)]],   // [tokens * top_k, F]
    constant uint&                 D               [[buffer(4)]],
    constant uint&                 F               [[buffer(5)]],
    constant uint&                 top_k           [[buffer(6)]],
    device const MoePrefillRoute*  routes          [[buffer(7)]],
    constant uint&                 route_count     [[buffer(8)]],
    constant uint&                 has_bias        [[buffer(9)]],
    constant float&                alpha           [[buffer(10)]],
    constant float&                limit           [[buffer(11)]],
    uint                           tg_idx          [[threadgroup_position_in_grid]],
    uint                           sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                           lane            [[thread_index_in_simdgroup]]
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

    // AGENTS.md/CLAUDE.md S3: `r.slot` is a device-supplied index into a
    // fixed-size argument-buffer array; an out-of-range slot from a
    // corrupted or mis-encoded route reads past `routed.blob`'s end.
    if (r.slot >= kMaxPrefillExpertBindings) return;
    device const uint8_t* base = routed.blob[r.slot];
    const ExpertOffsets re = routed_offsets;
    const uint row_bytes = mxfp4_row_bytes(DD);
    device const half* x_row = x + uint(r.token) * DD;

    float gate = dequant_mxfp4_row_simd(
        base + re.gate_W_off + f * row_bytes, x_row, DD, lane);
    float up = dequant_mxfp4_row_simd(
        base + re.up_W_off + f * row_bytes, x_row, DD, lane);
    // BEFORE the activation, which is where the clamp reads it too --
    // `min(gate + b, limit)`, not `min(gate, limit) + b`. Decode's comment
    // and decode's order; getting it the other way round is a different
    // model that still reads fluently.
    if (has_bias != 0u) {
        device const float* gb = (device const float*)(base + re.gate_b_off);
        device const float* ub = (device const float*)(base + re.up_b_off);
        gate += gb[f];
        up += ub[f];
    }
    if (lane == 0) {
        acts[(uint(r.token) * KK + uint(r.rank)) * FF + f] =
            half(moe_activate_mxfp4(gate, up, alpha, limit));
    }
}

// y[token] = residual[token] + sum_rank routing_w[token, rank] *
//            (down(acts[token, rank]) + db) -- the MXFP4 phase 2 with a
// token axis. One threadgroup per (token, d) output; SIMD group r owns
// rank r's down row, so the reduce is in the ROUTER'S ranking per token,
// independently of the chunk boundary (AGENTS.md Gotcha 27 holds by
// construction here exactly as it does for the affine sibling).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_prefill_phase2_fused_mxfp4(
    device const RoutedBlobsWide&  routed          [[buffer(0)]],
    constant ExpertOffsets&        routed_offsets  [[buffer(1)]],
    device const half*             acts            [[buffer(2)]],   // [tokens * top_k, F]
    device const half*             routing_w       [[buffer(3)]],   // [tokens * top_k]
    device const MoePrefillRoute*  routes          [[buffer(4)]],
    device half*                   y               [[buffer(5)]],   // [tokens, D]
    constant uint&                 D               [[buffer(6)]],
    constant uint&                 F               [[buffer(7)]],
    constant uint&                 top_k           [[buffer(8)]],
    constant uint&                 tokens          [[buffer(9)]],
    // [tokens, D], the accumulator SEED, exactly as the decode kernel's
    // `residual` is. `gpt-oss` has no shared expert and binds zeros, which
    // reproduces the decode path's `0.0f + p` bit for bit.
    device const half*             residual        [[buffer(10)]],
    constant uint&                 has_bias        [[buffer(11)]],
    uint2                          tg              [[threadgroup_position_in_grid]],
    uint                           sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                           lane            [[thread_index_in_simdgroup]]
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
        // AGENTS.md/CLAUDE.md S3: `r.slot` is a device-supplied index into a
        // fixed-size argument-buffer array. `sg_idx` (hence `r.slot`) can
        // differ across simdgroups of this ONE threadgroup, so an early
        // `return` here would let some threads skip the unconditional
        // `threadgroup_barrier` below while others reach it -- Metal's
        // divergent-barrier-participation UB. Mask the compute instead,
        // falling back to the same zero contribution the padding arm below
        // already uses.
        if (r.slot < kMaxPrefillExpertBindings) {
            device const uint8_t* base = routed.blob[r.slot];
            const ExpertOffsets re = routed_offsets;
            float value = dequant_mxfp4_row_simd(
                base + re.down_W_off + d * mxfp4_row_bytes(FF),
                acts + pair * FF, FF, lane);
            // PER SLOT AND INSIDE THE ROUTING WEIGHT: it is that expert's own
            // bias on that expert's own output, so llama.cpp adds it to the
            // expert result BEFORE the weighted sum. Hoisting it out of the
            // reduce would apply one expert's bias to every token and scale
            // it by the wrong weight.
            if (has_bias != 0u && lane == 0) {
                device const float* db = (device const float*)(base + re.down_b_off);
                value += db[d];
            }
            if (lane == 0) partial[sg_idx] = float(routing_w[pair]) * value;
        } else if (lane == 0) {
            partial[sg_idx] = 0.0f;
        }
    } else if (lane == 0) {
        // Padding mirrors the decode kernel's zero-padded routing weights
        // past top_k: +0.0f added in the same rank positions. `gpt-oss` is
        // top-4, so ranks 4..8 take this arm on every dispatch.
        partial[sg_idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        // Seeded, then rank order -- `moe_phase2_down_reduce_k8_mxfp4`'s
        // exact sequence. FP addition is not associative, so seeding is not
        // the same operation as adding the residual to the finished sum.
        float acc = float(residual[t * DD + d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[t * DD + d] = half(acc);
    }
}
