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

// E4M3FN, bias 7, no infinities. 0x7F / 0xFF (NaN) never reach this function.
static inline float ink_e4m3(uint b) {
    const uint e = (b >> 3) & 15u;
    const uint m = b & 7u;
    const float mag = (e == 0u) ? float(m) * 0.001953125f   // m * 2^-9
                                : ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return (b & 0x80u) ? -mag : mag;
}

// ldexp on the element, not a separately formed 2^(e-127): small exponents
// would make that factor subnormal, and fast math may flush it to zero.
static inline float mxfp8_elem(device const uint8_t* blk, uint e) {
    return ldexp(ink_e4m3(blk[1u + e]), int(blk[0]) - 127);
}

#define INK_ROW(NAME, BLOCK_ELEMS, BLOCK_BYTES, ELEM)                                    \
static inline float ink_##NAME##_row(                                                    \
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

INK_ROW(nvfp4, 64, 36, nvfp4_elem)
INK_ROW(mxfp8, 32, 33, mxfp8_elem)
INK_PAIR(nvfp4, 64, 36)
INK_PAIR(mxfp8, 32, 33)
