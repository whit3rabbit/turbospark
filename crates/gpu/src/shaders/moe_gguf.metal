// ============================================================================
// moe_gguf - the streamed routed-expert decode pair for GGUF block-quantized
// expert blobs (ROADMAP Phase G Stage 2). PORT-LOCAL, not vendored: the Swift
// engine has no GGUF intake.
//
// THIS FILE IS COMPILED CONCATENATED AFTER `moe.metal` AND
// `dequant_q4_k.metal`. The first is where `RoutedBlobs`, `ExpertOffsets`,
// `moe_hidden_activation` and the `moe_fc_*` function-constant accessors come
// from; the second is where the Q4_K pair below gets `dequant_q4_k_row_simd`
// and `q4_k_row_bytes`. It never appears on its own:
// `crates/gpu/src/moe_gguf.rs` passes `concat!` of the three as ONE
// `&'static str`, which is what the address-keyed pipeline cache needs.
// Same arrangement as `gdn.metal` over `dequant_int4.metal`.
//
// The vendored kernels next door read an INT4-AFFINE blob: three planar
// regions per projection (packed nibbles, BF16 scales, BF16 biases) at group
// 64, located by `ExpertOffsets`' nine fields. A GGUF blob has no planes at
// all. Its scale sits inside each block, ahead of the weights it scales, so
// only the three WEIGHT offsets in `ExpertOffsets` mean anything here and the
// six scale/bias offsets are ignored (the host writes zeros).
//
// That is also why these are separate kernels rather than a function-constant
// variant of the vendored ones: the two layouts share no addressing, only a
// signature.
// ============================================================================

constant constexpr uint kGgufQ8_0BlockElems = 32;
constant constexpr uint kGgufQ8_0BlockBytes = 34;

// Gate and up rows of one expert, both Q8_0, dotted against `x` in one pass.
// 32 lanes over a 32-element block, one weight per lane, which is the same
// shape `dequant_q8_0_gemv_simd` uses; the two are held to one CPU reference.
static inline float2 moe_q8_0_gate_up_rows_simd(
    device const uint8_t* gW,
    device const uint8_t* uW,
    device const half* x,
    uint f,
    uint D,
    uint lane
) {
    const uint n_blocks = D / kGgufQ8_0BlockElems;
    const uint row_bytes = n_blocks * kGgufQ8_0BlockBytes;
    device const uint8_t* g_row = gW + f * row_bytes;
    device const uint8_t* u_row = uW + f * row_bytes;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* gb = g_row + b * kGgufQ8_0BlockBytes;
        device const uint8_t* ub = u_row + b * kGgufQ8_0BlockBytes;
        // Named ushorts: `ushort | ushort` promotes to int in MSL, and
        // `as_type<half>` rejects an int operand outright rather than
        // truncating it.
        const ushort g_raw = ushort(gb[0]) | (ushort(gb[1]) << 8);
        const ushort u_raw = ushort(ub[0]) | (ushort(ub[1]) << 8);
        const float gd = float(as_type<half>(g_raw));
        const float ud = float(as_type<half>(u_raw));
        // SIGNED quants. Reading them as uchar mirrors half the weights and
        // costs no crash and no NaN.
        const float gq = float(int(as_type<int8_t>(gb[2 + lane])));
        const float uq = float(int(as_type<int8_t>(ub[2 + lane])));
        const float xv = float(x[b * kGgufQ8_0BlockElems + lane]);
        g_acc = fma(gd * gq, xv, g_acc);
        u_acc = fma(ud * uq, xv, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// One Q8_0 output row of the down projection against the slot's activations.
static inline float moe_q8_0_gemv_row_simd(
    device const uint8_t* W,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kGgufQ8_0BlockElems;
    const uint row_bytes = n_blocks * kGgufQ8_0BlockBytes;
    device const uint8_t* W_row = W + row * row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kGgufQ8_0BlockBytes;
        const ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const float d = float(as_type<half>(raw));
        const float q = float(int(as_type<int8_t>(blk[2 + lane])));
        const float xv = float(x[b * kGgufQ8_0BlockElems + lane]);
        acc = fma(d * q, xv, acc);
    }
    return simd_sum(acc);
}

// Phase 1, Q8_0: for each of `top_k` slots,
// acts[slot * F + f] = activation(gate_f(x)) * up_f(x).
// Dispatch matches the vendored sibling: one SIMD group per (slot, f) row,
// eight rows per threadgroup, 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_q8_0(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        x               [[buffer(2)]],
    device half*              acts            [[buffer(3)]],
    constant uint&            D               [[buffer(4)]],
    constant uint&            F               [[buffer(5)]],
    constant uint&            top_k           [[buffer(6)]],
    uint                      tg_idx          [[threadgroup_position_in_grid]],
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                      lane            [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    const uint slot = rowg / FF;
    const uint f = rowg % FF;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_q8_0_gate_up_rows_simd(
        base + re.gate_W_off, base + re.up_W_off, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

// Phase 2, Q8_0: y[d] = residual[d] + sum_slot routing_w[slot] *
// down_d(acts[slot]). Reduces ALL EIGHT slots unconditionally, exactly like
// the vendored sibling, so an unused slot needs a zero routing weight, a
// valid blob pointer, and a finite acts row (AGENTS.md Gotcha 8).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase2_down_reduce_k8_q8_0(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        acts            [[buffer(2)]],
    device const half*        routing_w       [[buffer(3)]],
    device const half*        residual        [[buffer(4)]],
    device half*              y               [[buffer(5)]],
    constant uint&            D               [[buffer(6)]],
    constant uint&            F               [[buffer(7)]],
    uint                      d               [[threadgroup_position_in_grid]],
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                      lane            [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_q8_0_gemv_row_simd(base + re.down_W_off, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// The same pair over Q4_K expert blobs, which is what a Qwen 3.6 Q4_K_M
// install streams. The row unpack is NOT written again here: it is
// `dequant_q4_k_row_simd` out of `dequant_q4_k.metal`, which this file is
// compiled after. Q4_K's 6-bit sub-scale split is the single most
// error-prone piece of arithmetic in the GGUF intake, and a second
// hand-written copy of it is exactly how two call sites come to disagree.
//
// The only Q4_K-specific constraint on the caller is the block size: rows
// must be a whole number of 256-element superblocks, where Q8_0 needs 32.
// Qwen's hidden 2048 and moe_intermediate 512 both qualify.
// ---------------------------------------------------------------------------

// Phase 1, Q4_K. Same dispatch as the Q8_0 sibling: one SIMD group per
// (slot, f) row, eight rows per threadgroup, 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_q4_k(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        x               [[buffer(2)]],
    device half*              acts            [[buffer(3)]],
    constant uint&            D               [[buffer(4)]],
    constant uint&            F               [[buffer(5)]],
    constant uint&            top_k           [[buffer(6)]],
    uint                      tg_idx          [[threadgroup_position_in_grid]],
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                      lane            [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    const uint slot = rowg / FF;
    const uint f = rowg % FF;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const uint row_bytes = q4_k_row_bytes(DD);
    const float gate = dequant_q4_k_row_simd(
        base + re.gate_W_off + f * row_bytes, x, DD, lane);
    const float up = dequant_q4_k_row_simd(
        base + re.up_W_off + f * row_bytes, x, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gate) * up);
}

// Phase 2, Q4_K. Reduces ALL EIGHT slots unconditionally, exactly like the
// vendored pair and the Q8_0 one, so an unused slot needs a zero routing
// weight, a valid blob pointer, and a finite acts row (AGENTS.md Gotcha 8).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase2_down_reduce_k8_q4_k(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        acts            [[buffer(2)]],
    device const half*        routing_w       [[buffer(3)]],
    device const half*        residual        [[buffer(4)]],
    device half*              y               [[buffer(5)]],
    constant uint&            D               [[buffer(6)]],
    constant uint&            F               [[buffer(7)]],
    uint                      d               [[threadgroup_position_in_grid]],
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                      lane            [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const half* act_slot = acts + sg_idx * FF;

    const float value = dequant_q4_k_row_simd(
        base + re.down_W_off + d * q4_k_row_bytes(FF), act_slot, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}
