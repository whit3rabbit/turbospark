// ============================================================================
// moe_gguf_ext - generated routed-expert decode kernels for every GGUF block
// type that had none, plus the missing HALF of the types that had only one
// phase. PORT-LOCAL. Two macros generate a phase each from any row dot
// `ROWFN(row, x, n, lane)`; every resident GEMV in this port already exposes
// one (`dequant_*_row_simd`), and the element-addressed types (legacy
// Q4_0/Q4_1/Q5_0/Q5_1 and Q1_0, TQ2_0, TQ1_0, NVFP4) get one built from their
// `*_elem` decoder by MOE_ELEM_ROW.
//
// COMPILED CONCATENATED AFTER `moe_gguf.metal`, which is after `moe.metal`
// (`RoutedBlobs`, `ExpertOffsets`, `moe_hidden_activation`, `moe_fc_*`) and
// after the dequantizer files (the row dots these kernels call). The decoders are the SAME functions the resident GEMVs
// use, so a routed row and a resident row are held to one implementation and
// one CPU reference (`turbospark_compute::dequantize_*`).
//
// Layout contract is the Q8_0 pair's: a GGUF blob has no scale planes, so only
// the three WEIGHT offsets of `ExpertOffsets` mean anything. Row `f` of a
// projection starts `f * row_bytes` into it. One SIMD group owns a row and each
// lane takes elements lane, lane + 32, ... of every block.
// ============================================================================

#define MOE_ELEM_ROW(NAME, BLOCK_ELEMS, BLOCK_BYTES, ELEM)                               \
static inline float moe_##NAME##_row_simd(                                               \
    device const uint8_t* row, device const half* x, uint n, uint lane) {                \
    const uint n_blocks = n / BLOCK_ELEMS;                                               \
    float acc = 0.0f;                                                                    \
    for (uint b = 0; b < n_blocks; ++b) {                                                \
        device const uint8_t* blk = row + b * BLOCK_BYTES;                               \
        for (uint e = lane; e < BLOCK_ELEMS; e += 32) {                                  \
            acc = fma(ELEM(blk, e), float(x[b * BLOCK_ELEMS + e]), acc);                 \
        }                                                                                \
    }                                                                                    \
    return simd_sum(acc);                                                                \
}

#define MOE_PHASE1(NAME, BLOCK_ELEMS, BLOCK_BYTES, ROWFN)                                \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_phase1_gate_up_act_##NAME(                                               \
    device const RoutedBlobs& routed          [[buffer(0)]],                             \
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],                             \
    device const half*        x               [[buffer(2)]],                             \
    device half*              acts            [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    uint                      tg_idx          [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    constexpr uint rows_per_tg = 8;                                                      \
    const uint DD = moe_fc_d(D);                                                         \
    const uint FF = moe_fc_f(F);                                                         \
    const uint rowg = tg_idx * rows_per_tg + sg_idx;                                     \
    if (rowg >= moe_fc_top_k(top_k) * FF) return;                                        \
    const uint slot = rowg / FF;                                                         \
    const uint f = rowg % FF;                                                            \
    const ulong row_bytes = ulong(DD / BLOCK_ELEMS) * BLOCK_BYTES;                       \
    device const uint8_t* base = routed.blob[slot];                                      \
    const ExpertOffsets re = routed_offsets;                                             \
    const float g = ROWFN(base + re.gate_W_off + f * row_bytes, x, DD, lane);            \
    const float u = ROWFN(base + re.up_W_off + f * row_bytes, x, DD, lane);              \
    if (lane == 0) acts[slot * FF + f] = half(moe_gguf_glu(g, u, base, re));             \
}

#define MOE_PHASE2(NAME, BLOCK_ELEMS, BLOCK_BYTES, ROWFN)                                \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_phase2_down_reduce_k8_##NAME(                                            \
    device const RoutedBlobs& routed          [[buffer(0)]],                             \
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],                             \
    device const half*        acts            [[buffer(2)]],                             \
    device const half*        routing_w       [[buffer(3)]],                             \
    device const half*        residual        [[buffer(4)]],                             \
    device half*              y               [[buffer(5)]],                             \
    constant uint&            D               [[buffer(6)]],                             \
    constant uint&            F               [[buffer(7)]],                             \
    uint                      d               [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    threadgroup float partial[8];                                                        \
    const uint DD = moe_fc_d(D);                                                         \
    const uint FF = moe_fc_f(F);                                                         \
    if (d >= DD) return;                                                                 \
    device const uint8_t* base = routed.blob[sg_idx];                                    \
    const ExpertOffsets re = routed_offsets;                                             \
    device const half* act_slot = acts + sg_idx * FF;                                    \
    const ulong row_bytes = ulong(FF / BLOCK_ELEMS) * BLOCK_BYTES;                       \
    const float value = ROWFN(base + re.down_W_off + d * row_bytes, act_slot, FF, lane); \
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * moe_gguf_down(value, base, re); \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
    if (sg_idx == 0 && lane == 0) {                                                      \
        float acc = moe_gguf_residual(residual, d);                                      \
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];      \
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];      \
        y[d] = half(acc);                                                                \
    }                                                                                    \
}

#define MOE_PAIR(NAME, BLOCK_ELEMS, BLOCK_BYTES, ROWFN)                                  \
    MOE_PHASE1(NAME, BLOCK_ELEMS, BLOCK_BYTES, ROWFN)                                    \
    MOE_PHASE2(NAME, BLOCK_ELEMS, BLOCK_BYTES, ROWFN)

// Element-addressed types: both phases.
MOE_ELEM_ROW(q4_0, 32, 18, q4_0_elem)
MOE_ELEM_ROW(q4_1, 32, 20, q4_1_elem)
MOE_ELEM_ROW(q5_0, 32, 22, q5_0_elem)
MOE_ELEM_ROW(q5_1, 32, 24, q5_1_elem)
MOE_ELEM_ROW(q1_0, 128, 18, q1_0_elem)
MOE_ELEM_ROW(tq2_0, 256, 66, tq2_0_elem)
MOE_ELEM_ROW(tq1_0, 256, 54, tq1_0_elem)
MOE_ELEM_ROW(nvfp4, 64, 36, nvfp4_elem)
MOE_PAIR(q4_0, 32, 18, moe_q4_0_row_simd)
MOE_PAIR(q4_1, 32, 20, moe_q4_1_row_simd)
MOE_PAIR(q5_0, 32, 22, moe_q5_0_row_simd)
MOE_PAIR(q5_1, 32, 24, moe_q5_1_row_simd)
MOE_PAIR(q1_0, 128, 18, moe_q1_0_row_simd)
MOE_PAIR(tq2_0, 256, 66, moe_tq2_0_row_simd)
MOE_PAIR(tq1_0, 256, 54, moe_tq1_0_row_simd)
MOE_PAIR(nvfp4, 64, 36, moe_nvfp4_row_simd)

// Q2_0 had a phase 1 and a top-10-only phase 2; this is the general k8 phase 2.
MOE_ELEM_ROW(q2_0, 64, 18, q2_0_elem)
MOE_PHASE2(q2_0, 64, 18, moe_q2_0_row_simd)

// K and IQ types that had no routed kernel at all: both phases.
MOE_PAIR(q5_k, 256, 176, dequant_q5_k_row_simd)
MOE_PAIR(q3_k, 256, 110, dequant_q3_k_row_simd)
MOE_PAIR(q2_k, 256, 84, dequant_q2_k_row_simd)
MOE_PAIR(iq2_xs, 256, 74, dequant_iq2_xs_row_simd)
MOE_PAIR(iq1_s, 256, 50, dequant_iq1_s_row_simd)
MOE_PAIR(iq3_s, 256, 110, dequant_iq3_s_row_simd)

// The missing HALF of types that had one phase (the hand-written kernels in
// moe_gguf.metal keep the other half).
MOE_PHASE2(iq3_xxs, 256, 98, dequant_iq3_xxs_row_simd)
MOE_PHASE2(iq4_xs, 256, 136, dequant_iq4_xs_row_simd)
MOE_PHASE2(iq2_s, 256, 82, dequant_iq2_s_row_simd)
MOE_PHASE2(iq2_xxs, 256, 66, dequant_iq2_xxs_row_simd)
MOE_PHASE2(iq1_m, 256, 56, dequant_iq1_m_row_simd)
MOE_PHASE1(iq4_nl, 32, 18, dequant_iq4_nl_row_simd)
MOE_PHASE1(q6_k, 256, 210, dequant_q6_k_row_simd)

// Unsloth fork IQ1 narrow types: both phases.
MOE_ELEM_ROW(iq1_xs, 256, 46, iq1_xs_elem)
MOE_ELEM_ROW(iq1_xxs, 256, 42, iq1_xxs_elem)
MOE_ELEM_ROW(iq1_xxxs, 256, 38, iq1_xxxs_elem)
MOE_PAIR(iq1_xs, 256, 46, moe_iq1_xs_row_simd)
MOE_PAIR(iq1_xxs, 256, 42, moe_iq1_xxs_row_simd)
MOE_PAIR(iq1_xxxs, 256, 38, moe_iq1_xxxs_row_simd)
