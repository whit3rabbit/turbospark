// ============================================================================
// moe_inkling2 - routed-expert decode pair for Inkling 2 (`inkling2_loop`).
// PORT-LOCAL. Contract: `turbospark_compute::inkling2` (`dequant_nvfp4_row`,
// `dequant_mxfp8_row`, `situ`), held by `tests/moe_inkling2_parity.rs`.
//
// COMPILED CONCATENATED AFTER `moe.metal` (`RoutedBlobs`, `ExpertOffsets`) and
// `dequant_ext.metal` (`nvfp4_elem`, so routed NVFP4 rows decode through the
// same function as the GGUF NVFP4 GEMV and one CPU reference).
//
// One expert blob, as the repack writes it:
//   gate_s_off, up_s_off, down_s_off -> one little-endian f32 each, the
//     per-expert NVFP4 outer scale (amax / outer_scale_denominator; w13's for
//     gate and up, w2's for down). MXFP8 experts store 1.0. 4-byte aligned.
//   gate_W_off, up_W_off -> F rows of D elements; down_W_off -> D rows of F.
//   NVFP4 rows are ggml `block_nvfp4` (36 B per 64, `nvfp4_row_to_blocks`).
//   MXFP8 rows are 33 B per 32: byte 0 the E8M0 exponent, then 32 E4M3 bytes
//     (`mxfp8_row_to_blocks`). The repack refuses NaN codes, so none reach here.
//
// Differences from the GGUF pair, each from the Inkling layer flow:
//   * phase 1 applies the softcapped SwiGLU `softcap(g, cg) * sigmoid(g) *
//     softcap(u, cu)`; the sigmoid sees the RAW gate. The caps are arguments.
//   * phase 2 adds NO residual: `sconv_mlp` runs on the expert sum before the
//     residual add, so folding the residual in here would be wrong.
//   * phase 2 reduces exactly `top_k` (<= 8) slots, in slot order, which is
//     router ranking order. Unused simdgroups contribute nothing, rather than
//     reading a stale blob pointer times a zero weight.
// ============================================================================

constant constexpr uint kInkRowsPerTG = 8;
constant constexpr uint kInkMaxSlots = 8;

static inline float ink_outer(device const uint8_t* base, uint off) {
    return *(device const float*)(base + off);
}

static inline float ink_softcap(float x, float cap) {
    return cap * precise::tanh(x / cap);
}

#define INK_PAIR(NAME, BLOCK_ELEMS, BLOCK_BYTES)                                         \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase1_##NAME(                                                  \
    device const RoutedBlobs& routed          [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        x               [[buffer(2)]],                             \
    device half*              acts            [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    constant float2&          caps            [[buffer(7)]],                             \
    uint                      tg_idx          [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint rowg = tg_idx * kInkRowsPerTG + sg_idx;                                   \
    if (rowg >= top_k * F) return;                                                       \
    const uint slot = rowg / F;                                                          \
    const uint f = rowg % F;                                                             \
    const ulong row_bytes = ulong(D / BLOCK_ELEMS) * BLOCK_BYTES;                        \
    device const uint8_t* base = routed.blob[slot];                                      \
    const float g = ink_##NAME##_row(base + re.gate_W_off + f * row_bytes, x, D, lane)   \
        * ink_outer(base, re.gate_s_off);                                                \
    const float u = ink_##NAME##_row(base + re.up_W_off + f * row_bytes, x, D, lane)     \
        * ink_outer(base, re.up_s_off);                                                  \
    if (lane == 0) {                                                                     \
        const float sig = 1.0f / (1.0f + precise::exp(-g));                              \
        acts[slot * F + f] = half(ink_softcap(g, caps.x) * sig * ink_softcap(u, caps.y)); \
    }                                                                                    \
}                                                                                        \
                                                                                         \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase2_##NAME(                                                  \
    device const RoutedBlobs& routed          [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        acts            [[buffer(2)]],                             \
    device const half*        routing_w       [[buffer(3)]],                             \
    device half*              y               [[buffer(4)]],                             \
    constant uint&            D               [[buffer(5)]],                             \
    constant uint&            F               [[buffer(6)]],                             \
    constant uint&            top_k           [[buffer(7)]],                             \
    uint                      d               [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    threadgroup float partial[kInkMaxSlots];                                             \
    if (d >= D) return;                                                                  \
    float w = 0.0f;                                                                      \
    if (sg_idx < top_k) {                                                                \
        device const uint8_t* base = routed.blob[sg_idx];                                \
        const ulong row_bytes = ulong(F / BLOCK_ELEMS) * BLOCK_BYTES;                    \
        const float v = ink_##NAME##_row(base + re.down_W_off + d * row_bytes,           \
                                         acts + sg_idx * F, F, lane)                     \
            * ink_outer(base, re.down_s_off);                                            \
        w = float(routing_w[sg_idx]) * v;                                                \
    }                                                                                    \
    if (lane == 0) partial[sg_idx] = w;                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
    if (sg_idx == 0 && lane == 0) {                                                      \
        float acc = 0.0f;                                                                \
        for (uint i = 0; i < top_k; ++i) acc += partial[i];                              \
        y[d] = half(acc);                                                                \
    }                                                                                    \
}

// E2M1 by code; index 8 is -0, which adds nothing either way.
constant constexpr float kInkE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// NVFP4 row dot, one 16-element sub-block per lane per step. The earlier
// per-element form (a lane per element, `nvfp4_elem`) re-decoded the
// sub-block scale and did a byte load per element, and ran at ~50 GB/s on
// the Mini's experts (dispatch profile, 2026-11-21). Here a
// lane loads the sub-block's 8 code bytes as two words, decodes its scale
// once, and multiplies the scale into the 16-term sum. Values are the same as
// `nvfp4_elem`'s; only the summation order differs.
//
// Sub-block j of the row is block j / 4, sub-block j % 4: scale byte
// blk[s], codes blk[4 + 8 s .. 12 + 8 s], element r in the low nibble of
// byte r (r < 8) or the high nibble of byte r - 8. Its x elements are
// x[16 j .. 16 j + 16]. REQUIRES a 4-byte-aligned row (36-byte blocks keep
// it once the row start is aligned; the runtime refuses unaligned offsets).
static inline float ink_nvfp4_row(
    device const uint8_t* row, device const half* x, uint n, uint lane) {
    const uint n_sub = n / 16u;
    float acc = 0.0f;
    for (uint j = lane; j < n_sub; j += 32u) {
        device const uint8_t* blk = row + (j >> 2) * 36u;
        const uint s = j & 3u;
        device const uint* q = (device const uint*)(blk + 4u + s * 8u);
        const uint q0 = q[0];
        const uint q1 = q[1];
        device const half4* xv = (device const half4*)(x + j * 16u);
        const float4 x0 = float4(xv[0]);
        const float4 x1 = float4(xv[1]);
        const float4 x2 = float4(xv[2]);
        const float4 x3 = float4(xv[3]);
        const float4 w0 = float4(kInkE2M1[q0 & 15u], kInkE2M1[(q0 >> 8) & 15u],
                                 kInkE2M1[(q0 >> 16) & 15u], kInkE2M1[(q0 >> 24) & 15u]);
        const float4 w1 = float4(kInkE2M1[q1 & 15u], kInkE2M1[(q1 >> 8) & 15u],
                                 kInkE2M1[(q1 >> 16) & 15u], kInkE2M1[(q1 >> 24) & 15u]);
        const float4 w2 = float4(kInkE2M1[(q0 >> 4) & 15u], kInkE2M1[(q0 >> 12) & 15u],
                                 kInkE2M1[(q0 >> 20) & 15u], kInkE2M1[q0 >> 28]);
        const float4 w3 = float4(kInkE2M1[(q1 >> 4) & 15u], kInkE2M1[(q1 >> 12) & 15u],
                                 kInkE2M1[(q1 >> 20) & 15u], kInkE2M1[q1 >> 28]);
        const float sum = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
        acc = fma(nvfp4_scale(blk[s]), sum, acc);
    }
    return simd_sum(acc);
}

// E4M3FN (bias 7, no infinities) without a branch: a normal code builds
// its fp32 bits directly (exponent e - 7 + 127, mantissa m << 20), a
// subnormal is m * 2^-9. NaN codes never reach here (the repack refuses them).
static inline float ink_e4m3_fast(uint b) {
    const uint e = (b >> 3) & 15u;
    const uint m = b & 7u;
    const float norm = as_type<float>(((e + 120u) << 23) | (m << 20));
    const float mag = (e == 0u) ? float(m) * 0.001953125f : norm;
    return as_type<float>(as_type<uint>(mag) | ((b & 0x80u) << 24));
}

// MXFP8 row dot, one 33-byte block (32 elements) per lane per step: the 32
// codes as eight unaligned 4-byte loads, the block exponent applied once to
// the block sum. `ldexp` on the sum, not a separately formed 2^(e-127):
// small exponents would make that factor subnormal, and fast math may flush
// it to zero.
static inline float ink_mxfp8_row(
    device const uint8_t* row, device const half* x, uint n, uint lane) {
    const uint n_blocks = n / 32u;
    float acc = 0.0f;
    for (uint b = lane; b < n_blocks; b += 32u) {
        device const uint8_t* blk = row + b * 33u;
        device const packed_uchar4* q = (device const packed_uchar4*)(blk + 1u);
        device const half4* xv = (device const half4*)(x + b * 32u);
        float sum = 0.0f;
        for (uint i = 0; i < 8u; ++i) {
            const uchar4 c = uchar4(q[i]);
            const float4 w = float4(ink_e4m3_fast(c.x), ink_e4m3_fast(c.y),
                                    ink_e4m3_fast(c.z), ink_e4m3_fast(c.w));
            sum += dot(w, float4(xv[i]));
        }
        acc += ldexp(sum, int(blk[0]) - 127);
    }
    return simd_sum(acc);
}
INK_PAIR(nvfp4, 64, 36)
INK_PAIR(mxfp8, 32, 33)

// ----------------------------------------------------------------------------
// Indexed pair (`TURBOSPARK_INKLING2_RESIDENT_EXPERTS`): every expert of the
// layer is resident in ONE buffer at a uniform `stride` (the packed layer
// file as written), and the slots name experts through `sel` (the GPU
// router's ranked ids) instead of host-bound blob pointers. With routing on
// the GPU too, nothing in a layer application waits for the host. Same math
// and slot order as the blob pair above, so the same experts give the same
// output.
// ----------------------------------------------------------------------------
#define INK_PAIR_IDX(NAME)                                                               \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase1_idx_##NAME(                                              \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        x               [[buffer(2)]],                             \
    device half*              acts            [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    constant float2&          caps            [[buffer(7)]],                             \
    device const uint*        sel             [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    constant uint&            n_experts       [[buffer(10)]],                            \
    uint                      tg_idx          [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint rowg = tg_idx * kInkRowsPerTG + sg_idx;                                   \
    if (rowg >= top_k * F) return;                                                       \
    const uint slot = rowg / F;                                                          \
    const uint f = rowg % F;                                                             \
    const uint e = min(sel[slot], n_experts - 1u);                                       \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(D / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    const float g = ink_##NAME##_row(base + re.gate_W_off + f * row_bytes, x, D, lane)   \
        * ink_outer(base, re.gate_s_off);                                                \
    const float u = ink_##NAME##_row(base + re.up_W_off + f * row_bytes, x, D, lane)     \
        * ink_outer(base, re.up_s_off);                                                  \
    if (lane == 0) {                                                                     \
        const float sig = 1.0f / (1.0f + precise::exp(-g));                              \
        acts[slot * F + f] = half(ink_softcap(g, caps.x) * sig * ink_softcap(u, caps.y)); \
    }                                                                                    \
}                                                                                        \
                                                                                         \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase2_idx_##NAME(                                              \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        acts            [[buffer(2)]],                             \
    device const half*        routing_w       [[buffer(3)]],                             \
    device half*              y               [[buffer(4)]],                             \
    constant uint&            D               [[buffer(5)]],                             \
    constant uint&            F               [[buffer(6)]],                             \
    constant uint&            top_k           [[buffer(7)]],                             \
    device const uint*        sel             [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    constant uint&            n_experts       [[buffer(10)]],                            \
    uint                      d               [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    threadgroup float partial[kInkMaxSlots];                                             \
    if (d >= D) return;                                                                  \
    float w = 0.0f;                                                                      \
    if (sg_idx < top_k) {                                                                \
        const uint e = min(sel[sg_idx], n_experts - 1u);                                 \
        device const uint8_t* base = layer + ulong(e) * stride;                          \
        const ulong row_bytes = ulong(F / ink_block_elems_##NAME) * ink_block_bytes_##NAME; \
        const float v = ink_##NAME##_row(base + re.down_W_off + d * row_bytes,           \
                                         acts + sg_idx * F, F, lane)                     \
            * ink_outer(base, re.down_s_off);                                            \
        w = float(routing_w[sg_idx]) * v;                                                \
    }                                                                                    \
    if (lane == 0) partial[sg_idx] = w;                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
    if (sg_idx == 0 && lane == 0) {                                                      \
        float acc = 0.0f;                                                                \
        for (uint i = 0; i < top_k; ++i) acc += partial[i];                              \
        y[d] = half(acc);                                                                \
    }                                                                                    \
}

constant constexpr uint ink_block_elems_nvfp4 = 64;
constant constexpr uint ink_block_bytes_nvfp4 = 36;
constant constexpr uint ink_block_elems_mxfp8 = 32;
constant constexpr uint ink_block_bytes_mxfp8 = 33;
INK_PAIR_IDX(nvfp4)
INK_PAIR_IDX(mxfp8)

// ----------------------------------------------------------------------------
// Multi-row indexed pair (speculative verify): the indexed pair above for row
// r = threadgroup grid y, reading `x + r * x_stride` and the row's own
// selection and weights (`sel + r * kInkMaxSlots`, `routing_w + r *
// kInkMaxSlots`), writing `acts + r * top_k * F` and `y + r * D`. Row r's math
// is the single-row kernel's, so it writes the same bits.
// ----------------------------------------------------------------------------
#define INK_PAIR_IDX_ROWS(NAME)                                                          \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase1_idx_rows_##NAME(                                         \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        x_all           [[buffer(2)]],                             \
    device half*              acts_all        [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    constant float2&          caps            [[buffer(7)]],                             \
    device const uint*        sel_all         [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    constant uint&            n_experts       [[buffer(10)]],                            \
    constant uint&            x_stride        [[buffer(11)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint rowg = tg.x * kInkRowsPerTG + sg_idx;                                     \
    if (rowg >= top_k * F) return;                                                       \
    device const half* x = x_all + ulong(tg.y) * x_stride;                               \
    device half* acts = acts_all + ulong(tg.y) * top_k * F;                              \
    device const uint* sel = sel_all + ulong(tg.y) * kInkMaxSlots;                       \
    const uint slot = rowg / F;                                                          \
    const uint f = rowg % F;                                                             \
    const uint e = min(sel[slot], n_experts - 1u);                                       \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(D / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    const float g = ink_##NAME##_row(base + re.gate_W_off + f * row_bytes, x, D, lane)   \
        * ink_outer(base, re.gate_s_off);                                                \
    const float u = ink_##NAME##_row(base + re.up_W_off + f * row_bytes, x, D, lane)     \
        * ink_outer(base, re.up_s_off);                                                  \
    if (lane == 0) {                                                                     \
        const float sig = 1.0f / (1.0f + precise::exp(-g));                              \
        acts[slot * F + f] = half(ink_softcap(g, caps.x) * sig * ink_softcap(u, caps.y)); \
    }                                                                                    \
}                                                                                        \
                                                                                         \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase2_idx_rows_##NAME(                                         \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        acts_all        [[buffer(2)]],                             \
    device const half*        rw_all          [[buffer(3)]],                             \
    device half*              y_all           [[buffer(4)]],                             \
    constant uint&            D               [[buffer(5)]],                             \
    constant uint&            F               [[buffer(6)]],                             \
    constant uint&            top_k           [[buffer(7)]],                             \
    device const uint*        sel_all         [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    constant uint&            n_experts       [[buffer(10)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    threadgroup float partial[kInkMaxSlots];                                             \
    const uint d = tg.x;                                                                 \
    if (d >= D) return;                                                                  \
    device const half* acts = acts_all + ulong(tg.y) * top_k * F;                        \
    device const half* routing_w = rw_all + ulong(tg.y) * kInkMaxSlots;                  \
    device const uint* sel = sel_all + ulong(tg.y) * kInkMaxSlots;                       \
    float w = 0.0f;                                                                      \
    if (sg_idx < top_k) {                                                                \
        const uint e = min(sel[sg_idx], n_experts - 1u);                                 \
        device const uint8_t* base = layer + ulong(e) * stride;                          \
        const ulong row_bytes = ulong(F / ink_block_elems_##NAME) * ink_block_bytes_##NAME; \
        const float v = ink_##NAME##_row(base + re.down_W_off + d * row_bytes,           \
                                         acts + sg_idx * F, F, lane)                     \
            * ink_outer(base, re.down_s_off);                                            \
        w = float(routing_w[sg_idx]) * v;                                                \
    }                                                                                    \
    if (lane == 0) partial[sg_idx] = w;                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
    if (sg_idx == 0 && lane == 0) {                                                      \
        float acc = 0.0f;                                                                \
        for (uint i = 0; i < top_k; ++i) acc += partial[i];                              \
        y_all[ulong(tg.y) * D + d] = half(acc);                                          \
    }                                                                                    \
}

INK_PAIR_IDX_ROWS(nvfp4)
INK_PAIR_IDX_ROWS(mxfp8)

// ----------------------------------------------------------------------------
// Expert-grouped multi-row pair (chunked prefill). The `_idx_rows` pair reads
// every (row, slot) pair's expert rows independently; here each distinct
// expert's weight row is decoded ONCE per lane chunk and dotted with every
// pair routed to it (up to 8 at a time). Per pair the arithmetic is the
// `_idx_rows` kernel's term for term, so acts and the per-slot down terms
// carry the same bits; phase 2 writes each pair's weighted F32 term and
// `moe_inkling2_reduce_slots` sums the slots in rank order, the single-row
// kernel's reduction. After oMLX's `_gather_sort` and mtplx's distinct-expert
// verify (both Apache-2.0); this port's own code.
//
// `moe_inkling2_group` (one threadgroup) lists the pairs by expert: `offs[e]`
// .. `offs[e + 1]` index `pairs`, each `row * kInkMaxSlots + slot`, in no
// fixed order within an expert.
// ----------------------------------------------------------------------------
constant constexpr uint kInkGroupBatch = 8;
constant constexpr uint kInkGroupMaxExperts = 256;

kernel void moe_inkling2_group(
    device const uint*   sel_all   [[buffer(0)]],
    device uint*         offs      [[buffer(1)]],
    device uint*         pairs     [[buffer(2)]],
    constant uint&       n_rows    [[buffer(3)]],
    constant uint&       top_k     [[buffer(4)]],
    constant uint&       n_experts [[buffer(5)]],
    uint                 tid       [[thread_index_in_threadgroup]],
    uint                 nthreads  [[threads_per_threadgroup]]
) {
    // One threadgroup: count pairs per expert, prefix-sum, scatter. The order
    // of pairs WITHIN an expert's list is not fixed (atomic cursors), which
    // is safe: every consumer computes each pair on its own (outputs indexed
    // by pair id, slots reduced in rank order), so no output bit depends on
    // list order (`tests/moe_inkling2_grouped_parity.rs` and `_mma.rs`).
    threadgroup atomic_uint count[kInkGroupMaxExperts];
    threadgroup atomic_uint cursor[kInkGroupMaxExperts];
    const uint total = n_rows * top_k;
    for (uint e = tid; e < n_experts; e += nthreads) {
        atomic_store_explicit(&count[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < total; p += nthreads) {
        const uint r = p / top_k, s = p % top_k;
        const uint e = min(sel_all[r * kInkMaxSlots + s], n_experts - 1u);
        atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint acc = 0u;
        for (uint e = 0; e < n_experts; ++e) {
            offs[e] = acc;
            atomic_store_explicit(&cursor[e], acc, memory_order_relaxed);
            acc += atomic_load_explicit(&count[e], memory_order_relaxed);
        }
        offs[n_experts] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < total; p += nthreads) {
        const uint r = p / top_k, s = p % top_k;
        const uint e = min(sel_all[r * kInkMaxSlots + s], n_experts - 1u);
        const uint at = atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
        pairs[at] = r * kInkMaxSlots + s;
    }
}

// `ink_nvfp4_row` for up to 8 inputs sharing one weight row: out[p] is the
// bits `ink_nvfp4_row(row, xs[p], n, lane)` returns.
static inline void ink_nvfp4_rows8(
    device const uint8_t* row, device const half* const thread* xs, uint np, uint n,
    uint lane, thread float* out) {
    #pragma clang fp reassociate(off)
    const uint n_sub = n / 16u;
    float acc[kInkGroupBatch];
    for (uint p = 0; p < kInkGroupBatch; ++p) acc[p] = 0.0f;
    for (uint j = lane; j < n_sub; j += 32u) {
        device const uint8_t* blk = row + (j >> 2) * 36u;
        const uint s = j & 3u;
        device const uint* q = (device const uint*)(blk + 4u + s * 8u);
        const uint q0 = q[0];
        const uint q1 = q[1];
        const float4 w0 = float4(kInkE2M1[q0 & 15u], kInkE2M1[(q0 >> 8) & 15u],
                                 kInkE2M1[(q0 >> 16) & 15u], kInkE2M1[(q0 >> 24) & 15u]);
        const float4 w1 = float4(kInkE2M1[q1 & 15u], kInkE2M1[(q1 >> 8) & 15u],
                                 kInkE2M1[(q1 >> 16) & 15u], kInkE2M1[(q1 >> 24) & 15u]);
        const float4 w2 = float4(kInkE2M1[(q0 >> 4) & 15u], kInkE2M1[(q0 >> 12) & 15u],
                                 kInkE2M1[(q0 >> 20) & 15u], kInkE2M1[q0 >> 28]);
        const float4 w3 = float4(kInkE2M1[(q1 >> 4) & 15u], kInkE2M1[(q1 >> 12) & 15u],
                                 kInkE2M1[(q1 >> 20) & 15u], kInkE2M1[q1 >> 28]);
        const float sc = nvfp4_scale(blk[s]);
        for (uint p = 0; p < kInkGroupBatch; ++p) {
            if (p < np) {
                device const half4* xv = (device const half4*)(xs[p] + j * 16u);
                const float4 x0 = float4(xv[0]);
                const float4 x1 = float4(xv[1]);
                const float4 x2 = float4(xv[2]);
                const float4 x3 = float4(xv[3]);
                const float sum = dot(w0, x0) + dot(w1, x1) + dot(w2, x2) + dot(w3, x3);
                acc[p] = fma(sc, sum, acc[p]);
            }
        }
    }
    for (uint p = 0; p < kInkGroupBatch; ++p) {
        if (p < np) out[p] = simd_sum(acc[p]);
    }
}

// `ink_mxfp8_row` for up to 8 inputs sharing one weight row.
static inline void ink_mxfp8_rows8(
    device const uint8_t* row, device const half* const thread* xs, uint np, uint n,
    uint lane, thread float* out) {
    #pragma clang fp reassociate(off)
    const uint n_blocks = n / 32u;
    float acc[kInkGroupBatch];
    for (uint p = 0; p < kInkGroupBatch; ++p) acc[p] = 0.0f;
    for (uint b = lane; b < n_blocks; b += 32u) {
        device const uint8_t* blk = row + b * 33u;
        device const packed_uchar4* q = (device const packed_uchar4*)(blk + 1u);
        float sum[kInkGroupBatch];
        for (uint p = 0; p < kInkGroupBatch; ++p) sum[p] = 0.0f;
        for (uint i = 0; i < 8u; ++i) {
            const uchar4 c = uchar4(q[i]);
            const float4 w = float4(ink_e4m3_fast(c.x), ink_e4m3_fast(c.y),
                                    ink_e4m3_fast(c.z), ink_e4m3_fast(c.w));
            for (uint p = 0; p < kInkGroupBatch; ++p) {
                if (p < np) {
                    device const half4* xv = (device const half4*)(xs[p] + b * 32u);
                    sum[p] += dot(w, float4(xv[i]));
                }
            }
        }
        const int ex = int(blk[0]) - 127;
        for (uint p = 0; p < kInkGroupBatch; ++p) {
            if (p < np) acc[p] += ldexp(sum[p], ex);
        }
    }
    for (uint p = 0; p < kInkGroupBatch; ++p) {
        if (p < np) out[p] = simd_sum(acc[p]);
    }
}

#define INK_PAIR_GROUPED(NAME)                                                           \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase1_grouped_##NAME(                                          \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        x_all           [[buffer(2)]],                             \
    device half*              acts_all        [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    constant float2&          caps            [[buffer(7)]],                             \
    device const uint*        offs            [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    device const uint*        pairs           [[buffer(10)]],                            \
    constant uint&            x_stride        [[buffer(11)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint f = tg.x * kInkRowsPerTG + sg_idx;                                        \
    const uint e = tg.y;                                                                 \
    if (f >= F) return;                                                                  \
    const uint p0 = offs[e], p1 = offs[e + 1u];                                          \
    if (p0 == p1) return;                                                                \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(D / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    const float gs = ink_outer(base, re.gate_s_off);                                     \
    const float us = ink_outer(base, re.up_s_off);                                       \
    for (uint b = p0; b < p1; b += kInkGroupBatch) {                                     \
        const uint np = min(kInkGroupBatch, p1 - b);                                     \
        device const half* xs[kInkGroupBatch];                                           \
        for (uint p = 0; p < kInkGroupBatch; ++p) {                                      \
            const uint pr = pairs[b + min(p, np - 1u)];                                  \
            xs[p] = x_all + ulong(pr / kInkMaxSlots) * x_stride;                         \
        }                                                                                \
        float gv[kInkGroupBatch], uv[kInkGroupBatch];                                    \
        ink_##NAME##_rows8(base + re.gate_W_off + f * row_bytes, xs, np, D, lane, gv);   \
        ink_##NAME##_rows8(base + re.up_W_off + f * row_bytes, xs, np, D, lane, uv);     \
        if (lane == 0) {                                                                 \
            for (uint p = 0; p < np; ++p) {                                              \
                const uint pr = pairs[b + p];                                            \
                const uint r = pr / kInkMaxSlots, slot = pr % kInkMaxSlots;              \
                const float g = gv[p] * gs;                                              \
                const float u = uv[p] * us;                                              \
                const float sig = 1.0f / (1.0f + precise::exp(-g));                      \
                acts_all[ulong(r) * top_k * F + slot * F + f] =                          \
                    half(ink_softcap(g, caps.x) * sig * ink_softcap(u, caps.y));         \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
}                                                                                        \
                                                                                         \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void moe_inkling2_phase2_grouped_##NAME(                                          \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        acts_all        [[buffer(2)]],                             \
    device const half*        rw_all          [[buffer(3)]],                             \
    device float*             terms           [[buffer(4)]],                             \
    constant uint&            D               [[buffer(5)]],                             \
    constant uint&            F               [[buffer(6)]],                             \
    constant uint&            top_k           [[buffer(7)]],                             \
    device const uint*        offs            [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    device const uint*        pairs           [[buffer(10)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    _Pragma("clang fp reassociate(off)")                                                 \
    const uint d = tg.x * kInkRowsPerTG + sg_idx;                                        \
    const uint e = tg.y;                                                                 \
    if (d >= D) return;                                                                  \
    const uint p0 = offs[e], p1 = offs[e + 1u];                                          \
    if (p0 == p1) return;                                                                \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(F / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    const float ds = ink_outer(base, re.down_s_off);                                     \
    for (uint b = p0; b < p1; b += kInkGroupBatch) {                                     \
        const uint np = min(kInkGroupBatch, p1 - b);                                     \
        device const half* xs[kInkGroupBatch];                                           \
        for (uint p = 0; p < kInkGroupBatch; ++p) {                                      \
            const uint pr = pairs[b + min(p, np - 1u)];                                  \
            const uint r = pr / kInkMaxSlots, slot = pr % kInkMaxSlots;                  \
            xs[p] = acts_all + ulong(r) * top_k * F + slot * F;                          \
        }                                                                                \
        float vv[kInkGroupBatch];                                                        \
        ink_##NAME##_rows8(base + re.down_W_off + d * row_bytes, xs, np, F, lane, vv);   \
        if (lane == 0) {                                                                 \
            for (uint p = 0; p < np; ++p) {                                              \
                const uint pr = pairs[b + p];                                            \
                const float v = vv[p] * ds;                                              \
                terms[ulong(pr) * D + d] = float(rw_all[pr]) * v;                        \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
}

INK_PAIR_GROUPED(nvfp4)
INK_PAIR_GROUPED(mxfp8)

// y[r, d] = sum over slots in rank order of the phase-2 terms: the
// `_idx_rows` phase 2's `acc += partial[i]`, then FP16.
kernel void moe_inkling2_reduce_slots(
    device const float*  terms  [[buffer(0)]],
    device half*         y_all  [[buffer(1)]],
    constant uint&       D      [[buffer(2)]],
    constant uint&       top_k  [[buffer(3)]],
    uint2                gid    [[thread_position_in_grid]]
) {
    const uint d = gid.x, r = gid.y;
    if (d >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < top_k; ++i) acc += terms[(ulong(r) * kInkMaxSlots + i) * D + d];
    y_all[ulong(r) * D + d] = half(acc);
}

// ----------------------------------------------------------------------------
// Matrix-unit form of the expert-grouped pair (chunked prefill, KL-bounded,
// `TURBOSPARK_INKLING2_PREFILL_MMA`). Same pair lists (`moe_inkling2_group`),
// same per-slot terms and the same slot reduction (`moe_inkling2_reduce_slots`)
// as the scalar grouped pair, but each expert's rows are multiplied with up
// to 32 of its pairs at a time on FP32 simdgroup_matrix with register
// operands (the dense prefill GEMM's lane layout and k order). Weights decode
// exactly: E2M1 x the E4M3 sub-block scale, or E4M3 x 2^(E8M0 - 127), are
// exact in FP32; the outer scale is applied after the dot as in the scalar
// pair. A pair's result depends only on its own input row (C[i][j] reads
// weight row i and pair column j), not on which pairs share its tile, so the
// prefill stays independent of chunking (`tests/moe_inkling2_grouped_mma.rs`).
//
// K runs in blocks of 64 with k-slot kk of step s at physical column
// 8 kk + s. A lane's weight k-slots are fn and fn + 1, i.e. block columns
// 8 fn .. 8 fn + 15 (fn even): one NVFP4 sub-block (16 elements, element i in
// byte i % 8, low nibble for i < 8), or 16 bytes of one MXFP8 block. Its input
// k-slot fm is columns 8 fm .. 8 fm + 7 of each pair's row.
// One simdgroup = 8 weight rows; 4 simdgroups per threadgroup; up to 4 pair
// tiles of 8 per sweep, more pairs in further sweeps.
// ----------------------------------------------------------------------------
constant constexpr uint kInkGmSG = 4;
constant constexpr uint kInkGmRowsPerTG = 8 * kInkGmSG;
constant constexpr uint kInkGmTiles = 4;

inline uint2 ink_gm_coord(uint lane) {
    const uint q = lane >> 2;
    return uint2((q & 4u) | ((lane >> 1) & 3u), ((q & 2u) << 1) | ((lane & 1u) << 1));
}

// A lane's 16 weights of 64-block `kb` of a row: w[st] is k-slot fn at step
// st, w[8 + st] k-slot fn + 1.
static inline void ink_gm_weights_nvfp4(device const uint8_t* row, uint kb, uint fn,
                                        thread float* w) {
    device const uint8_t* blk = row + (kb / 64u) * 36u;
    const uint s = fn >> 1;
    const float sc = nvfp4_scale(blk[s]);
    device const uint8_t* q = blk + 4u + s * 8u;
    for (uint st = 0; st < 8u; ++st) {
        const uint b = q[st];
        w[st] = kInkE2M1[b & 15u] * sc;
        w[8u + st] = kInkE2M1[b >> 4] * sc;
    }
}

static inline void ink_gm_weights_mxfp8(device const uint8_t* row, uint kb, uint fn,
                                        thread float* w) {
    const uint e0 = kb + 16u * (fn >> 1);           // first of the lane's 16
    device const uint8_t* blk = row + (e0 / 32u) * 33u;
    const int ex = int(blk[0]) - 127;
    device const uint8_t* q = blk + 1u + (e0 % 32u);
    for (uint st = 0; st < 8u; ++st) {
        w[st] = ldexp(ink_e4m3_fast(q[st]), ex);
        w[8u + st] = ldexp(ink_e4m3_fast(q[8u + st]), ex);
    }
}

static inline void ink_gm_mac(thread float2& c, float a0, float a1, float b0, float b1) {
    simdgroup_float8x8 A, B, C;
    A.thread_elements()[0] = a0;
    A.thread_elements()[1] = a1;
    B.thread_elements()[0] = b0;
    B.thread_elements()[1] = b1;
    C.thread_elements()[0] = c.x;
    C.thread_elements()[1] = c.y;
    simdgroup_multiply_accumulate(C, A, B, C);
    c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}

// Half operands, FP32 accumulate: NVFP4 weights (E2M1 x an E4M3 scale, at
// most 5 significant bits, within half range) and the FP16 inputs are exact
// in half, so the operands are the same values as the FP32 form's.
static inline void ink_gm_mac_nvfp4(thread float2& c, float a0, float a1, float b0, float b1) {
    simdgroup_half8x8 A, B;
    simdgroup_float8x8 C;
    A.thread_elements()[0] = half(a0);
    A.thread_elements()[1] = half(a1);
    B.thread_elements()[0] = half(b0);
    B.thread_elements()[1] = half(b1);
    C.thread_elements()[0] = c.x;
    C.thread_elements()[1] = c.y;
    simdgroup_multiply_accumulate(C, A, B, C);
    c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}

// FP32 operands: MXFP8 block exponents do not bound a weight to half range.
static inline void ink_gm_mac_mxfp8(thread float2& c, float a0, float a1, float b0, float b1) {
    ink_gm_mac(c, a0, a1, b0, b1);
}


// 8 contiguous halfs of a pair's input at block column `col`, or zeros.
static inline void ink_gm_x8(device const half* x, bool live, thread float* out) {
    if (!live) {
        for (uint i = 0; i < 8u; ++i) out[i] = 0.0f;
        return;
    }
    const half4 a = *(device const half4*)x;
    const half4 b = *(device const half4*)(x + 4);
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}

#define INK_PAIR_GROUPED_MMA(NAME)                                                       \
[[kernel, max_total_threads_per_threadgroup(128)]]                                       \
kernel void moe_inkling2_phase1_grouped_mma_##NAME(                                      \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        x_all           [[buffer(2)]],                             \
    device half*              acts_all        [[buffer(3)]],                             \
    constant uint&            D               [[buffer(4)]],                             \
    constant uint&            F               [[buffer(5)]],                             \
    constant uint&            top_k           [[buffer(6)]],                             \
    constant float2&          caps            [[buffer(7)]],                             \
    device const uint*        offs            [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    device const uint*        pairs           [[buffer(10)]],                            \
    constant uint&            x_stride        [[buffer(11)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint e = tg.y;                                                                 \
    const uint p0 = offs[e], p1 = offs[e + 1u];                                          \
    if (p0 == p1) return;                                                                \
    const uint f0 = tg.x * kInkGmRowsPerTG + sg_idx * 8u;                                \
    if (f0 >= F) return;                                                                 \
    const uint2 c = ink_gm_coord(lane);                                                  \
    const uint fm = c.x, fn = c.y;                                                       \
    const uint f = f0 + fm;                                                              \
    const bool live_f = f < F;                                                           \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(D / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    device const uint8_t* grow = base + re.gate_W_off + ulong(live_f ? f : f0) * row_bytes; \
    device const uint8_t* urow = base + re.up_W_off + ulong(live_f ? f : f0) * row_bytes; \
    const float gs = ink_outer(base, re.gate_s_off);                                     \
    const float us = ink_outer(base, re.up_s_off);                                       \
    for (uint b0 = p0; b0 < p1; b0 += 8u * kInkGmTiles) {                                \
        const uint ntile = min(kInkGmTiles, (p1 - b0 + 7u) / 8u);                        \
        float2 ag[kInkGmTiles], au[kInkGmTiles];                                         \
        for (uint tt = 0; tt < kInkGmTiles; ++tt) { ag[tt] = float2(0.0f); au[tt] = float2(0.0f); } \
        for (uint kb = 0; kb < D; kb += 64u) {                                           \
            float wg[16], wu[16];                                                        \
            ink_gm_weights_##NAME(grow, kb, fn, wg);                                     \
            ink_gm_weights_##NAME(urow, kb, fn, wu);                                     \
            for (uint tt = 0; tt < ntile; ++tt) {                                        \
                const uint pa = b0 + tt * 8u + fn, pb = pa + 1u;                         \
                float xa[8], xb[8];                                                      \
                ink_gm_x8(x_all + ulong((pa < p1 ? pairs[pa] : 0u) / kInkMaxSlots) * x_stride + kb + 8u * fm, pa < p1, xa); \
                ink_gm_x8(x_all + ulong((pb < p1 ? pairs[pb] : 0u) / kInkMaxSlots) * x_stride + kb + 8u * fm, pb < p1, xb); \
                for (uint st = 0; st < 8u; ++st) {                                       \
                    ink_gm_mac_##NAME(ag[tt], wg[st], wg[8u + st], xa[st], xb[st]);             \
                    ink_gm_mac_##NAME(au[tt], wu[st], wu[8u + st], xa[st], xb[st]);             \
                }                                                                        \
            }                                                                            \
        }                                                                                \
        if (!live_f) continue;                                                           \
        for (uint tt = 0; tt < ntile; ++tt) {                                            \
            for (uint h = 0; h < 2u; ++h) {                                              \
                const uint pi = b0 + tt * 8u + fn + h;                                   \
                if (pi >= p1) continue;                                                  \
                const uint pr = pairs[pi];                                               \
                const uint r = pr / kInkMaxSlots, slot = pr % kInkMaxSlots;              \
                const float g = (h == 0u ? ag[tt].x : ag[tt].y) * gs;                    \
                const float u = (h == 0u ? au[tt].x : au[tt].y) * us;                    \
                const float sig = 1.0f / (1.0f + precise::exp(-g));                      \
                acts_all[ulong(r) * top_k * F + slot * F + f] =                          \
                    half(ink_softcap(g, caps.x) * sig * ink_softcap(u, caps.y));         \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
}                                                                                        \
                                                                                         \
[[kernel, max_total_threads_per_threadgroup(128)]]                                       \
kernel void moe_inkling2_phase2_grouped_mma_##NAME(                                      \
    device const uint8_t*     layer           [[buffer(0)]],                             \
    constant ExpertOffsets&   re              [[buffer(1)]],                             \
    device const half*        acts_all        [[buffer(2)]],                             \
    device const half*        rw_all          [[buffer(3)]],                             \
    device float*             terms           [[buffer(4)]],                             \
    constant uint&            D               [[buffer(5)]],                             \
    constant uint&            F               [[buffer(6)]],                             \
    constant uint&            top_k           [[buffer(7)]],                             \
    device const uint*        offs            [[buffer(8)]],                             \
    constant ulong&           stride          [[buffer(9)]],                             \
    device const uint*        pairs           [[buffer(10)]],                            \
    uint2                     tg              [[threadgroup_position_in_grid]],          \
    uint                      sg_idx          [[simdgroup_index_in_threadgroup]],        \
    uint                      lane            [[thread_index_in_simdgroup]]              \
) {                                                                                      \
    const uint e = tg.y;                                                                 \
    const uint p0 = offs[e], p1 = offs[e + 1u];                                          \
    if (p0 == p1) return;                                                                \
    const uint d0 = tg.x * kInkGmRowsPerTG + sg_idx * 8u;                                \
    if (d0 >= D) return;                                                                 \
    const uint2 c = ink_gm_coord(lane);                                                  \
    const uint fm = c.x, fn = c.y;                                                       \
    const uint d = d0 + fm;                                                              \
    const bool live_d = d < D;                                                           \
    device const uint8_t* base = layer + ulong(e) * stride;                              \
    const ulong row_bytes = ulong(F / ink_block_elems_##NAME) * ink_block_bytes_##NAME;  \
    device const uint8_t* drow = base + re.down_W_off + ulong(live_d ? d : d0) * row_bytes; \
    const float ds = ink_outer(base, re.down_s_off);                                     \
    for (uint b0 = p0; b0 < p1; b0 += 8u * kInkGmTiles) {                                \
        const uint ntile = min(kInkGmTiles, (p1 - b0 + 7u) / 8u);                        \
        float2 ac[kInkGmTiles];                                                          \
        for (uint tt = 0; tt < kInkGmTiles; ++tt) ac[tt] = float2(0.0f);                 \
        for (uint kb = 0; kb < F; kb += 64u) {                                           \
            float w[16];                                                                 \
            ink_gm_weights_##NAME(drow, kb, fn, w);                                      \
            for (uint tt = 0; tt < ntile; ++tt) {                                        \
                const uint pa = b0 + tt * 8u + fn, pb = pa + 1u;                         \
                float xa[8], xb[8];                                                      \
                const uint qa = pa < p1 ? pairs[pa] : 0u, qb = pb < p1 ? pairs[pb] : 0u; \
                ink_gm_x8(acts_all + ulong(qa / kInkMaxSlots) * top_k * F + (qa % kInkMaxSlots) * F + kb + 8u * fm, pa < p1, xa); \
                ink_gm_x8(acts_all + ulong(qb / kInkMaxSlots) * top_k * F + (qb % kInkMaxSlots) * F + kb + 8u * fm, pb < p1, xb); \
                for (uint st = 0; st < 8u; ++st) {                                       \
                    ink_gm_mac_##NAME(ac[tt], w[st], w[8u + st], xa[st], xb[st]);               \
                }                                                                        \
            }                                                                            \
        }                                                                                \
        if (!live_d) continue;                                                           \
        for (uint tt = 0; tt < ntile; ++tt) {                                            \
            for (uint h = 0; h < 2u; ++h) {                                              \
                const uint pi = b0 + tt * 8u + fn + h;                                   \
                if (pi >= p1) continue;                                                  \
                const uint pr = pairs[pi];                                               \
                const float v = (h == 0u ? ac[tt].x : ac[tt].y) * ds;                    \
                terms[ulong(pr) * D + d] = float(rw_all[pr]) * v;                        \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
}

INK_PAIR_GROUPED_MMA(nvfp4)
INK_PAIR_GROUPED_MMA(mxfp8)
