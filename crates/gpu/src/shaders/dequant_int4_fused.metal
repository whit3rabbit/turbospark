#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4_fused -- bit-exact fusions of the dense-FFN / shared-expert
// dispatch chain (docs/KAPITSA_KERNELS.md items 4 and 5).
//
// The unfused chain per FFN is five dispatches: two `dequant_int4_gemv_simd`
// calls (gate, up), one activation elementwise kernel, one down GEMV, one
// residual add. Each GEMV rounds its FP32 accumulator to half at the store,
// and every elementwise kernel reads those halves back -- so a fusion is
// bit-exact if and only if it reproduces the per-output accumulation order
// of `dequant_int4_gemv_simd_body` (dequant_int4.metal) and rounds each
// accumulator to half BEFORE the activation or epilogue reads it. Both
// kernels here are that body with a second accumulator (or an epilogue)
// bolted on at exactly those boundaries; the element order, the affine
// factoring (acc = fma(s, dot, acc); acc = fma(b, sum, acc) per group), the
// lane split, and the simd_sum are unchanged.
//
// The activation and epilogue expressions are vendored verbatim from
// utility.metal (`silu_mul_fp16`, `gelu_mul_fp16`'s `gelu_pytorch_tanh`,
// `sigmoid_scalar_mul_fp16`'s division form, `residual_add_fp16`) -- a
// "equivalent-looking" rewrite is exactly the drift these fusions exist to
// avoid.
//
// Constants 20/21/22 bake (M, N) exactly as `dequant_int4.metal` does since
// 2026-08-16 (+3-5.2% cold weight-read rate at the decode shapes; a shared
// key would reuse the first-compiled shape's pipeline and read the wrong
// fraction of every row). 30 is the gate+up kernel's activation selector
// (false = silu, true = gelu-pytorch-tanh); 31/32 are the down kernel's
// epilogue selectors (residual add, sigmoid scalar gate). Every one is
// ALWAYS set by the dispatcher and ALWAYS in the constants key (crate
// Gotcha 1).
// ============================================================================

constant uint FC_FUSED_M [[function_constant(20)]];
constant uint FC_FUSED_N [[function_constant(21)]];
constant bool FC_FUSED_USE_FC [[function_constant(22)]];

static inline uint fused_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_FUSED_USE_FC) &&
            FC_FUSED_USE_FC) ? FC_FUSED_M : M;
}

static inline uint fused_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_FUSED_USE_FC) &&
            FC_FUSED_USE_FC) ? FC_FUSED_N : N;
}

constant constexpr float kFusedGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kFusedGeluCubicCoeff = 0.044715f;

// Byte-for-byte moe.metal's `gelu_pytorch_tanh` (the same vendored copy
// utility.metal carries), including the NaN clamp.
static inline float fused_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kFusedGeluSqrt2OverPi * (x + kFusedGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

constant bool FC_FUSED_ACT_GELU [[function_constant(30)]];
constant bool FC_FUSED_EPI_RESIDUAL [[function_constant(31)]];
constant bool FC_FUSED_EPI_SIGMOID_GATE [[function_constant(32)]];

// One simdgroup computes BOTH the gate row and the up row of `row` over the
// same x, rounding each accumulator to half before the activation reads it:
// exactly `dequant_int4_gemv_simd(gate)` + `dequant_int4_gemv_simd(up)` +
// `silu_mul_fp16`/`gelu_mul_fp16` as three dispatches would. The x loads are
// shared; nothing else is.
kernel void dequant_int4_gate_up_act_gemv_simd(
    device const uint8_t* gW      [[buffer(0)]],
    device const bfloat*  gScales [[buffer(1)]],
    device const bfloat*  gBiases [[buffer(2)]],
    device const uint8_t* uW      [[buffer(3)]],
    device const bfloat*  uScales [[buffer(4)]],
    device const bfloat*  uBiases [[buffer(5)]],
    device const half*    x       [[buffer(6)]],
    device half*          y       [[buffer(7)]],   // [M] activated
    constant uint&        M       [[buffer(8)]],
    constant uint&        N       [[buffer(9)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    constexpr uint kGroupSize = 64;
    const uint MM = fused_fc_m(M);
    const uint NN = fused_fc_n(N);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups  = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    device const uint8_t* g_row = gW      + uint(row) * row_bytes;
    device const bfloat*  gs_row = gScales + uint(row) * n_groups;
    device const bfloat*  gb_row = gBiases + uint(row) * n_groups;
    device const uint8_t* u_row = uW      + uint(row) * row_bytes;
    device const bfloat*  us_row = uScales + uint(row) * n_groups;
    device const bfloat*  ub_row = uBiases + uint(row) * n_groups;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Same ushort-pair load as the scalar kernel: the resident weights are
        // 2-byte aligned, not 4.
        device const ushort* gwp = (device const ushort*)(g_row + byte_base);
        const uint gw4 = uint(gwp[0]) | (uint(gwp[1]) << 16);
        device const ushort* uwp = (device const ushort*)(u_row + byte_base);
        const uint uw4 = uint(uwp[0]) | (uint(uwp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float gs = float(gs_row[g]);
        const float gb = float(gb_row[g]);
        const float us = float(us_row[g]);
        const float ub = float(ub_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint gb0 =  gw4        & 0xFFu;
        const uint gb1 = (gw4 >> 8)  & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        const uint ub0 =  uw4        & 0xFFu;
        const uint ub1 = (uw4 >> 8)  & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot_g = 0.0f;
        dot_g = fma(float(gb0 & 0x0Fu), e0, dot_g); dot_g = fma(float(gb0 >> 4), e1, dot_g);
        dot_g = fma(float(gb1 & 0x0Fu), e2, dot_g); dot_g = fma(float(gb1 >> 4), e3, dot_g);
        dot_g = fma(float(gb2 & 0x0Fu), e4, dot_g); dot_g = fma(float(gb2 >> 4), e5, dot_g);
        dot_g = fma(float(gb3 & 0x0Fu), e6, dot_g); dot_g = fma(float(gb3 >> 4), e7, dot_g);
        float dot_u = 0.0f;
        dot_u = fma(float(ub0 & 0x0Fu), e0, dot_u); dot_u = fma(float(ub0 >> 4), e1, dot_u);
        dot_u = fma(float(ub1 & 0x0Fu), e2, dot_u); dot_u = fma(float(ub1 >> 4), e3, dot_u);
        dot_u = fma(float(ub2 & 0x0Fu), e4, dot_u); dot_u = fma(float(ub2 >> 4), e5, dot_u);
        dot_u = fma(float(ub3 & 0x0Fu), e6, dot_u); dot_u = fma(float(ub3 >> 4), e7, dot_u);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc_g = fma(gs, dot_g, acc_g);
        acc_g = fma(gb, sum, acc_g);
        acc_u = fma(us, dot_u, acc_u);
        acc_u = fma(ub, sum, acc_u);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gs_row[g]);
        const float gb = float(gb_row[g]);
        const float us = float(us_row[g]);
        const float ub = float(ub_row[g]);
        const uint8_t gbyte = g_row[g * (kGroupSize / 2) + lane];
        const uint8_t ubyte = u_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot_g = fma(float(uint(gbyte & 0x0Fu)), x0, 0.0f);
        dot_g = fma(float(uint(gbyte >> 4)), x1, dot_g);
        float dot_u = fma(float(uint(ubyte & 0x0Fu)), x0, 0.0f);
        dot_u = fma(float(uint(ubyte >> 4)), x1, dot_u);
        const float sum = x0 + x1;
        acc_g = fma(gs, dot_g, acc_g);
        acc_g = fma(gb, sum, acc_g);
        acc_u = fma(us, dot_u, acc_u);
        acc_u = fma(ub, sum, acc_u);
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (lane == 0) {
        // The halves the two GEMV stores would have produced, read back by
        // the activation kernel; then that kernel's exact expression. The
        // volatile locals force the f32->f16->f32 round-trip to actually
        // happen: under Metal's default fast-math the compiler elides a
        // convert pair it can see through, which silently re-rounds every
        // exact f16 tie differently from the unfused chain (found by the
        // parity test at a tie, 1 ulp).
        volatile half g_h = half(acc_g);
        volatile half u_h = half(acc_u);
        const float g = float(g_h);
        const float u = float(u_h);
        if (is_function_constant_defined(FC_FUSED_ACT_GELU) && FC_FUSED_ACT_GELU) {
            y[row] = half(fused_gelu_pytorch_tanh(g) * u);
        } else {
            y[row] = half((g / (1.0f + exp(-g))) * u);
        }
    }
}

// The down projection with the FFN tail's LAST elementwise dispatch folded
// into the store epilogue. EPILOGUE 0 is the plain `half(acc)` store (bit
// identical to `dequant_int4_gemv_simd`); 1 adds the residual in place
// (`residual_add_fp16`'s double round: the FFN output is rounded to half
// BEFORE it meets the stream); 2 applies the shared-expert scalar gate
// (`sigmoid_scalar_mul_fp16`'s DIVISION form on the rounded half).
kernel void dequant_int4_down_epi_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],   // [M]
    device const half*    aux    [[buffer(5)]],   // residual (epi 1) or the [1] logit (epi 2)
    constant uint&        M      [[buffer(6)]],
    constant uint&        N      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    constexpr uint kGroupSize = 64;
    const uint MM = fused_fc_m(M);
    const uint NN = fused_fc_n(N);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups  = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        // Volatile for the same reason as the gate+up kernel's comment: the
        // epilogue must see the ROUNDED half, not the accumulator fast-math
        // would happily keep.
        volatile half out_v = half(acc);
        const half out = out_v;
        if (is_function_constant_defined(FC_FUSED_EPI_RESIDUAL) && FC_FUSED_EPI_RESIDUAL) {
            y[row] = half(float(y[row]) + float(out));
        } else if (is_function_constant_defined(FC_FUSED_EPI_SIGMOID_GATE)
                   && FC_FUSED_EPI_SIGMOID_GATE) {
            const float gate = float(aux[0]);
            y[row] = half(float(out) / (1.0f + exp(-gate)));
        } else {
            y[row] = out;
        }
    }
}
