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
