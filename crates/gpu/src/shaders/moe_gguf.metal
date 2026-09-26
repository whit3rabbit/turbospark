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

// ---------------------------------------------------------------------------
// The IQ pairs (ROADMAP Phase S). `dequant_iq.metal` joins the concatenation
// ahead of this file, so the row unpacks below are ITS helpers rather than a
// second copy: an IQ3_XXS codebook lookup written twice is exactly how two
// call sites come to disagree, and unlike an arithmetic reconstruction there
// is no formula a reader could check the second copy against.
//
// ONLY THREE KERNELS, not six, and the asymmetry is the real file rather than
// an omission. The Phase S candidate
// (unsloth/gemma-4-26B-A4B-it-UD-Q3_K_M) puts its types on FIXED SIDES of a
// routed expert:
//
//   ffn_gate_up_exps  IQ3_XXS x29, IQ4_XS on layer 29   -> phase 1
//   ffn_down_exps     IQ4_NL  x29, Q8_0   on layer 29   -> phase 2
//
// so phase 1 needs IQ3_XXS and IQ4_XS, phase 2 needs IQ4_NL, and Q8_0 already
// has both. An IQ4_NL phase 1 or an IQ3_XXS phase 2 would be a kernel no real
// file dispatches, which is the same call the Q6_K GEMV made when it shipped
// without an embedding or MoE sibling. A checkpoint that needs one fails at
// the dispatch site by name.
//
// The block-size preconditions differ and the host asserts them: an IQ4_NL
// row must be a whole number of 32 elements, the other two of 256.
// ---------------------------------------------------------------------------

// Phase 1, IQ3_XXS. Same dispatch as every sibling: one SIMD group per
// (slot, f) row, eight rows per threadgroup, 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_iq3_xxs(
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
    const uint row_bytes = iq3_xxs_row_bytes(DD);
    const float gate = dequant_iq3_xxs_row_simd(
        base + re.gate_W_off + f * row_bytes, x, DD, lane);
    const float up = dequant_iq3_xxs_row_simd(
        base + re.up_W_off + f * row_bytes, x, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gate) * up);
}

// Phase 1, IQ4_XS. One layer of the candidate uses this and twenty-nine use
// the sibling above; see the header for why that is not an accident.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_iq4_xs(
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
    const uint row_bytes = iq4_xs_row_bytes(DD);
    const float gate = dequant_iq4_xs_row_simd(
        base + re.gate_W_off + f * row_bytes, x, DD, lane);
    const float up = dequant_iq4_xs_row_simd(
        base + re.up_W_off + f * row_bytes, x, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gate) * up);
}

// Phase 2, IQ4_NL. Reduces ALL EIGHT slots unconditionally, exactly like every
// sibling, so an unused slot needs a zero routing weight, a valid blob
// pointer, and a finite acts row (AGENTS.md Gotcha 8).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase2_down_reduce_k8_iq4_nl(
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

    const float value = dequant_iq4_nl_row_simd(
        base + re.down_W_off + d * iq4_nl_row_bytes(FF), act_slot, FF, lane);
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
// The Q6_K phase 2 (ROADMAP Phase M2). `dequant_q6_k.metal` joins the
// concatenation ahead of this file, so the row unpack below is ITS helper.
//
// PHASE 2 ONLY, and like the IQ trio above that asymmetry is the real file
// rather than an omission. Mixtral 8x7B's Q4_K_M puts Q6_K on the
// `ffn_down_exps` of 16 of its 32 layers and Q4_K on the other 16, with
// `ffn_gate_exps` and `ffn_up_exps` Q4_K throughout:
//
//   ffn_gate_exps / ffn_up_exps  Q4_K x32                 -> phase 1
//   ffn_down_exps                Q4_K x16, Q6_K x16       -> phase 2
//
// so phase 2 needs Q6_K and phase 1 does not. A Q6_K phase 1 would be a kernel
// no real file dispatches; a checkpoint that needs one fails at the dispatch
// site by name. Note this is the FIRST file to mix two block types across
// LAYERS on the same routed sub-tensor, which is exactly what Phase S made
// `RoutedBlobLayout` per-layer and per-phase for.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase2_down_reduce_k8_q6_k(
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

    const float value = dequant_q6_k_row_simd(
        base + re.down_W_off + d * q6_k_row_bytes_msl(FF), act_slot, FF, lane);
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
// The MXFP4 pair (ROADMAP M5, the `gpt-oss` family). PORT-LOCAL like every
// kernel in this file.
//
// TWO KERNELS, NOT THREE, and for once the asymmetry is not a judgement call
// about what a real file asks for: `gpt-oss-20b-MXFP4.gguf` puts MXFP4 in
// `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` and NOWHERE ELSE. Its
// attention, `token_embd` and `output` are all Q8_0, so there is no resident
// GEMV and no embedding lookup to write, and a file that wanted one would
// fail at the dispatch site by name like Q6_K's and the IQ trio's do.
//
// The row unpack is written HERE rather than borrowed, because unlike Q4_K,
// Q6_K and the IQ types there is no `dequant_mxfp4.metal` to borrow it from --
// nothing else in this port decodes MXFP4. The contract it is held to is
// `turbospark_compute::dequantize_mxfp4`, by
// `crates/gpu/tests/moe_gguf_parity.rs`.
//
// FOUR THINGS ARE SILENTLY WRONG IF CARRIED OVER FROM A NEIGHBOUR BY HABIT,
// and they are the same four `quant_gguf_mxfp4.rs` lists for the CPU side:
//
//  1. The nibbles of a byte are 16 elements apart, not adjacent. Byte j
//     serves elements j and j + 16 (IQ4_NL's split-half layout, the opposite
//     of the K-quant habit). Reading them adjacent gives a correctly-scaled
//     PERMUTATION, which correlates well and is wrong.
//  2. The codebook is not affine. Index 8 is a SECOND ZERO, not `-0.5`, so a
//     `q - 8` read gets every magnitude past index 4 wrong.
//  3. The scale is `2^(e - 128)`, not `2^(e - 127)`. ggml applies the
//     E8M0-to-fp32 "half" variant because its codebook is the FP4 grid scaled
//     by two; the plain bias doubles every weight in the model.
//  4. `e = 0` and `e = 1` land in the subnormal range of that encoding and
//     ggml builds them by shifting a fixed pattern. ~1e-39 and unable to
//     matter, and exactly the two an expression written from the common case
//     gets wrong.
//
// ONE THING IS NEW TO THIS FILE RATHER THAN TO MXFP4: A 17-BYTE BLOCK MAKES
// ROWS ODD-LENGTHED IN GENERAL, so a row pointer is not guaranteed to be even.
// Every read below is `uint8_t` and there is no f16 scale to `as_type`, which
// is what makes that free -- a `ushort` read copied in from the Q8_0 helper
// next door would be misaligned on half the rows.
// ---------------------------------------------------------------------------

constant constexpr uint kMxfp4BlockElems = 32;
constant constexpr uint kMxfp4BlockBytes = 17;

// The FP4 codebook an MXFP4 index expands to. Mirrors
// `turbospark_compute::MXFP4_VALUES`, which was RECOVERED FROM GGML rather
// than transcribed; the parity test is what holds the two equal. Both zeros
// are POSITIVE: ggml's `kvalues_mxfp4` is an int8 array and has no signed
// zero, whatever the FP4 E2M1 encoding the format is named for would say.
constant float kMxfp4Values[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

// The scale one E8M0 shared-exponent byte stands for. See trap 3 and 4 above.
static inline float mxfp4_scale_msl(uint8_t e) {
    const uint bits = (uint(e) < 2u) ? (0x00200000u << uint(e)) : ((uint(e) - 1u) << 23);
    return as_type<float>(bits);
}

static inline uint mxfp4_row_bytes(uint n) {
    return (n / kMxfp4BlockElems) * kMxfp4BlockBytes;
}

// One MXFP4 row dotted against `x`.
//
// ONE ELEMENT PER LANE, as for the IQ types and Q8_0 and unlike Q4_K (where a
// lane owns a BYTE and therefore two elements 32 apart). Lane L owns element
// L of each block, which is the LOW nibble of byte L when L < 16 and the HIGH
// nibble of byte L - 16 otherwise. Two lanes therefore read the same byte and
// take different halves of it, which costs nothing and keeps the element
// index a plain `lane`.
static inline float dequant_mxfp4_row_simd(
    device const uint8_t* row,
    device const half* x,
    uint n,
    uint lane
) {
    const uint n_blocks = n / kMxfp4BlockElems;
    const uint half_elems = kMxfp4BlockElems / 2;
    const uint byte_idx = (lane < half_elems) ? lane : (lane - half_elems);
    const bool high = lane >= half_elems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = row + b * kMxfp4BlockBytes;
        const float d = mxfp4_scale_msl(blk[0]);
        const uint8_t packed = blk[1 + byte_idx];
        const uint q = high ? uint(packed >> 4) : uint(packed & 0x0Fu);
        const float xv = float(x[b * kMxfp4BlockElems + lane]);
        acc = fma(kMxfp4Values[q] * d, xv, acc);
    }
    return simd_sum(acc);
}

// gpt-oss's expert activation (ROADMAP M5), which is NOT the one every other
// flow here uses. Verbatim from ggml's `ggml_compute_forward_swiglu_oai_f32`:
//
//     x = min(gate, limit)
//     y = clamp(up, -limit, limit)
//     out = (x / (1 + exp(-alpha * x))) * (y + 1)
//
// The CLAMP half is `moe.metal`'s `moe_swiglu_clamp` byte for byte. The
// ACTIVATION half is new twice over: a swish with `alpha` where every
// existing flow uses silu (which is alpha 1) or gelu-tanh, and a `(y + 1)`
// where every existing flow multiplies by `up` directly. `alpha = 1.702` and
// `limit = 7.0` are hardcoded constants in llama.cpp's graph builder, not
// metadata, so they arrive here from the family baseline.
//
// PASSED AS UNIFORMS RATHER THAN FUNCTION CONSTANTS, deliberately.
// `moe_function_constants` already carries a `FC_MOE_SWIGLU_LIMIT` slot and
// `constants_key` is ONE BYTE wide (the silu flag); adding two specialization
// axes to a key that narrow is how a pipeline cache silently hands back the
// wrong kernel, which is AGENTS.md Gotcha 18's ring-capacity trap in a new
// place. A uniform costs a branch on a code path that just read a whole
// expert row from memory.
//
// `alpha <= 0` means "not gpt-oss": the plain `activation(gate) * up` every
// other block type's pair does, so the block-type parity cases and the
// family cases run through one kernel.
static inline float moe_activate_mxfp4(float gate, float up, float alpha, float limit) {
    if (alpha <= 0.0f) return moe_hidden_activation(gate) * up;
    const float x = min(gate, limit);
    const float y = clamp(up, -limit, limit);
    return (x / (1.0f + exp(-alpha * x))) * (y + 1.0f);
}

// Phase 1, MXFP4. Same dispatch as every sibling: one SIMD group per
// (slot, f) row, eight rows per threadgroup, 256 threads.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_mxfp4(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        x               [[buffer(2)]],
    device half*              acts            [[buffer(3)]],
    constant uint&            D               [[buffer(4)]],
    constant uint&            F               [[buffer(5)]],
    constant uint&            top_k           [[buffer(6)]],
    constant uint&            has_bias        [[buffer(7)]],
    constant float&           alpha           [[buffer(8)]],
    constant float&           limit           [[buffer(9)]],
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
    const uint row_bytes = mxfp4_row_bytes(DD);
    float gate = dequant_mxfp4_row_simd(
        base + re.gate_W_off + f * row_bytes, x, DD, lane);
    float up = dequant_mxfp4_row_simd(
        base + re.up_W_off + f * row_bytes, x, DD, lane);
    // THE BIAS IS ADDED BEFORE THE ACTIVATION, which is where the clamp
    // reads it too -- `min(gate + b, limit)`, not `min(gate, limit) + b`.
    // The blob carries the biases as F32, verbatim from the GGUF, because
    // routed bytes are never transcoded.
    if (has_bias != 0u) {
        device const float* gb = (device const float*)(base + re.gate_b_off);
        device const float* ub = (device const float*)(base + re.up_b_off);
        gate += gb[f];
        up += ub[f];
    }
    if (lane == 0) acts[slot * FF + f] = half(moe_activate_mxfp4(gate, up, alpha, limit));
}

// Phase 2, MXFP4. Unlike every sibling `moe_phase2_down_reduce_k8*` kernel
// (which still reduce all EIGHT slots unconditionally, AGENTS.md Gotcha 8),
// this one is specialized down to `KK`, resolved from `FC_MOE_TOP_K`
// (ROADMAP's Phase-2 top_k specialization, crates/gpu/CLAUDE.md Gotcha 3):
// `gpt-oss` routes top-4 of 32 experts, so slots 4..7 of this kernel's fixed
// 8 always carried a zero routing weight and a wasted dequant-and-dot-product
// over F elements. Masking the COMPUTE by `sg_idx < KK` and never RETURNING
// early is what keeps this safe: every one of the 256 threads (8 simdgroups)
// still reaches `threadgroup_barrier` below, uniformly, whether or not `KK`
// is baked at compile time -- an early `return` for `sg_idx >= KK` would let
// some simdgroups skip the barrier while others still wait on it, which is
// undefined behaviour on Metal. Masked-out slots contribute exactly `0.0f`
// to `partial[sg_idx]` the same way an unmasked slot with a zero
// `routing_w` always did (`0 * value == 0` for any finite `value`), so this
// is provably bit-identical to the unspecialized `KK == 8` path and to the
// pre-existing unconditional one -- unused slots still need a zero routing
// weight, but no longer need a valid blob pointer or a finite acts row,
// since they are never read.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase2_down_reduce_k8_mxfp4(
    device const RoutedBlobs& routed          [[buffer(0)]],
    constant ExpertOffsets&   routed_offsets  [[buffer(1)]],
    device const half*        acts            [[buffer(2)]],
    device const half*        routing_w       [[buffer(3)]],
    device const half*        residual        [[buffer(4)]],
    device half*              y               [[buffer(5)]],
    constant uint&            D               [[buffer(6)]],
    constant uint&            F               [[buffer(7)]],
    constant uint&            has_bias        [[buffer(8)]],
    constant uint&            top_k           [[buffer(9)]],
    uint                      d               [[threadgroup_position_in_grid]],
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                      lane            [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    // NOT `moe_fc_top_k`: that helper is gated on `FC_MOE_USE_FC`, which
    // `moe_fc_d`/`moe_fc_f` above share -- turning it on to bake `KK` would
    // also flip `DD`/`FF` to their baked-but-never-set value of zero. The
    // host always sets `FC_MOE_TOP_K` for this kernel (see
    // `phase2_function_constants`), so this resolves independently of that
    // shared gate.
    const uint KK = is_function_constant_defined(FC_MOE_TOP_K) ? FC_MOE_TOP_K : top_k;
    if (d >= DD) return;

    float value = 0.0f;
    if (sg_idx < KK) {
        device const uint8_t* base = routed.blob[sg_idx];
        const ExpertOffsets re = routed_offsets;
        device const half* act_slot = acts + sg_idx * FF;

        value = dequant_mxfp4_row_simd(
            base + re.down_W_off + d * mxfp4_row_bytes(FF), act_slot, FF, lane);
        // PER SLOT AND INSIDE THE ROUTING WEIGHT, because it is that expert's
        // own bias on that expert's own output -- llama.cpp adds it to the
        // expert result before the weighted sum. Adding it once outside the
        // reduce would apply one expert's bias to every token and scale it
        // wrongly.
        if (has_bias != 0u && lane == 0) {
            device const float* db = (device const float*)(base + re.down_b_off);
            value += db[d];
        }
    }
    if (lane == 0) partial[sg_idx] = (sg_idx < KK) ? float(routing_w[sg_idx]) * value : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

// The Swift candidate uses these IQ codebooks on routed gate/up rows. Reuse
// the row decoders from `dequant_iq.metal`, which is concatenated before
// this file, to keep the dense and MoE readers on one format implementation.
static inline float2 moe_iq2_s_gate_up_rows_simd(
    device const uint8_t* gW, device const uint8_t* uW,
    device const half* x, uint f, uint D, uint lane
) {
    const uint row_bytes = (D / 256) * 82;
    return float2(
        dequant_iq2_s_row_simd(gW + f * row_bytes, x, D, lane),
        dequant_iq2_s_row_simd(uW + f * row_bytes, x, D, lane));
}

static inline float2 moe_iq2_xxs_gate_up_rows_simd(
    device const uint8_t* gW, device const uint8_t* uW,
    device const half* x, uint f, uint D, uint lane
) {
    const uint row_bytes = (D / 256) * 66;
    return float2(
        dequant_iq2_xxs_row_simd(gW + f * row_bytes, x, D, lane),
        dequant_iq2_xxs_row_simd(uW + f * row_bytes, x, D, lane));
}

static inline float2 moe_iq1_m_gate_up_rows_simd(
    device const uint8_t* gW, device const uint8_t* uW,
    device const half* x, uint f, uint D, uint lane
) {
    const uint row_bytes = (D / 256) * 56;
    return float2(
        dequant_iq1_m_row_simd(gW + f * row_bytes, x, D, lane),
        dequant_iq1_m_row_simd(uW + f * row_bytes, x, D, lane));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_iq2_s(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    const uint slot = rowg / FF, f = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_iq2_s_gate_up_rows_simd(
        base + routed_offsets.gate_W_off, base + routed_offsets.up_W_off, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_iq2_xxs(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    const uint slot = rowg / FF, f = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_iq2_xxs_gate_up_rows_simd(
        base + routed_offsets.gate_W_off, base + routed_offsets.up_W_off, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_iq1_m(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    const uint slot = rowg / FF, f = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_iq1_m_gate_up_rows_simd(
        base + routed_offsets.gate_W_off, base + routed_offsets.up_W_off, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

// Qwen4Exp's routed down projection uses Q2_0 and top-10. Keep this as a
// dedicated ten-slot reducer so existing GGUF kernels retain their eight
// slot reduction order and dispatch width.
static inline float moe_q2_0_gemv_row_simd(
    device const uint8_t* W,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_blocks = N / 64;
    const uint row_bytes = n_blocks * 18;
    device const uint8_t* W_row = W + row * row_bytes;
    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * 18;
        const ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const float d = float(as_type<half>(raw));
        for (uint half_block = 0; half_block < 2; ++half_block) {
            const uint e = lane + half_block * 32;
            const uint packed = blk[2 + e / 4];
            const uint q = (packed >> (2 * (e % 4))) & 3u;
            const float w = (float(q) - 1.0f) * d;
            acc = fma(w, float(x[b * 64 + e]), acc);
        }
    }
    return simd_sum(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void moe_phase1_gate_up_act_q2_0(
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
    const float gate = moe_q2_0_gemv_row_simd(
        base + re.gate_W_off, x, f, DD, lane);
    const float up = moe_q2_0_gemv_row_simd(
        base + re.up_W_off, x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(moe_hidden_activation(gate) * up);
}

[[kernel, max_total_threads_per_threadgroup(320)]]
kernel void moe_phase2_down_reduce_k10_q2_0(
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
    threadgroup float partial[10];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const half* act_slot = acts + sg_idx * FF;
    const float value = moe_q2_0_gemv_row_simd(
        base + re.down_W_off, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        acc += partial[8]; acc += partial[9];
        y[d] = half(acc);
    }
}
