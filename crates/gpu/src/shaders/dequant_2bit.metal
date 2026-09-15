#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_2bit - MLX `affine` dequant at TWO bits (the
// `prism-ml/Ternary-Bonsai-27B-mlx-2bit` layout). PORT-LOCAL, not vendored:
// the Swift engine reads no such checkpoint, so there is no upstream kernel to
// mirror. Its contract is `turbospark_compute::quant_2bit` and nothing else,
// which is why the kernels below carry that module's names rather than this
// file's.
//
// Layout (per row of length N, N a multiple of the group size G):
//   W       : N/4 bytes. Element i is the two-bit field at bit 2 * (i % 4) of
//             byte i / 4, LEAST significant field first.
//   scales  : N/G FP16, one per group.
//   biases  : N/G FP16, one per group.
//   value   : w[i] = float(q[i]) * scale[i/G] + bias[i/G], q in {0,1,2,3}.
//
// Three ways this differs from the INT4 affine sibling, each of which is a
// silently wrong answer rather than a crash if carried over:
//   1. The companions are `half`, not `bfloat`. The two are the same width,
//      so binding one as the other passes every length check and reads this
//      checkpoint's 0.0137 scales as 7e-18.
//   2. The group size is 128, not 64, and it is a RUNTIME UNIFORM here rather
//      than the sibling's `constant constexpr`. It is a property of the
//      checkpoint, not of the container, and a function constant that missed
//      the pipeline cache's constants key would silently reuse the wrong
//      pipeline (crates/gpu Gotcha 1). A uniform cannot.
//   3. A byte holds 4 elements, not 2, and the field order within it is
//      load-bearing. It was measured against `mx.dequantize` rather than read
//      off a doc; a wrong order permutes elements within a run of four and
//      leaves every group scale and the whole level histogram untouched, so
//      nothing but an oracle can see it.
//
// AND ONE WAY IT DIFFERS FROM THE 1-BIT SIBLING that is worth stating because
// the file next door is the obvious template: there is NO `+/-1`-style fast
// path here. The checkpoint is ternary (`bias == -scale`, level 3 unused), so
// the analogue would exist, and it would reassociate the sum exactly as the
// 1-bit one does (AGENTS.md Gotcha 27). That kernel is already reachable from
// no decode flow; a second one would be too.
//
// Affine factoring, as in both siblings: over any run inside one group,
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias cost one FMA each per BYTE rather than per element. The
// run here is one byte, which is inside one group because G is a multiple
// of 4.
//
// `x` is read element by element rather than through `half4`, as in the 1-bit
// sibling and for the same reason: an offset-bound `x` need not be aligned,
// and nothing dispatches this on a hot path yet. Vectorize it when something
// does, not before.
// ============================================================================

constant constexpr uint kRowsPerTG2Bit = 8;
constant constexpr uint kLanesPerRow2Bit = 32;
constant constexpr uint kElemsPerByte2Bit = 4;

// y[m] = sum_n W[m, n] * x[n], the GENERAL affine form: every group is decoded
// through its own `(scale, bias)` pair and no relationship between the two is
// assumed, so a checkpoint that used all four levels or an off-centre bias
// decodes correctly here. One SIMD group per output row, each lane walking the
// row's bytes at a stride of 32, so a row shorter than 32 bytes simply leaves
// the high lanes contributing zero. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_int2_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    scales [[buffer(1)]],
    device const half*    biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        G      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTG2Bit + sg_idx;
    if (row >= M) return;

    const uint row_bytes       = N / kElemsPerByte2Bit;
    const uint n_groups        = N / G;
    const uint bytes_per_group = G / kElemsPerByte2Bit;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const half*    s_row = scales + uint(row) * n_groups;
    device const half*    b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint j = lane; j < row_bytes; j += kLanesPerRow2Bit) {
        // Resolved per byte, not hoisted: a row spans many groups and each
        // carries its own pair.
        const uint  g = j / bytes_per_group;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint  byte = uint(W_row[j]);
        const uint  elem = j * kElemsPerByte2Bit;
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint k = 0; k < kElemsPerByte2Bit; ++k) {
            const float xv = float(x[elem + k]);
            // LSB-first: element `elem + k` is the two-bit field at bit 2k.
            dot = fma(float((byte >> (2u * k)) & 3u), xv, dot);
            sum += xv;
        }
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// One row of a 2-bit affine embedding table, dequantized into `out` and
// scaled. The sibling of `embed_lookup_int1` in `dequant_1bit.metal`, and it
// exists because the real checkpoint quantizes its EMBEDDING TABLE at two bits
// like everything else -- read off the safetensors header
// (`language_model.model.embed_tokens.weight` is `U32 [248320, 320]` with F16
// companions), not assumed. So 2-bit is a type with a GEMV and a lookup and no
// routed-expert pair, which is the per-type footing AGENTS.md Gotcha 29
// describes, the model being dense.
//
// One thread per element; `out_scale` is sqrt(hidden) for a scaled-embedding
// family, 1.0 otherwise. The row stride is `D / 4`, not `D`: getting that
// factor wrong lands inside a neighbouring token's weights, which decodes
// perfectly well and is the wrong token.
kernel void embed_lookup_int2(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/4] packed fields
    device const half*    scales    [[buffer(1)]],   // [V, D/G] FP16
    device const half*    biases    [[buffer(2)]],   // [V, D/G] FP16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant uint&        G         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / G;
    device const uint8_t* row_q = table  + uint(token_id) * (D / kElemsPerByte2Bit);
    device const half*    row_s = scales + uint(token_id) * groups_per_row;
    device const half*    row_b = biases + uint(token_id) * groups_per_row;
    const uint  q = (uint(row_q[gid / kElemsPerByte2Bit]) >> (2u * (gid % kElemsPerByte2Bit))) & 3u;
    const float s = float(row_s[gid / G]);
    const float b = float(row_b[gid / G]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// ============================================================================
// dequant_int2_gemm_simd - the GEMV above with B right-hand sides, the 1-bit
// batch kernel's structure at two bits. PORT-LOCAL. Read that kernel's header
// for the amortization argument and this one's GEMV for the layout; the one
// thing worth restating here is the parity story, because it is the whole
// reason this kernel is gated by a bit-exactness test instead of a quality
// gate: the row mapping, the lane-strided byte walk, the per-byte group
// resolution, the affine factoring order and the `simd_sum` partition are the
// GEMV's own, so B rows of this kernel ARE B calls of
// `dequant_int2_gemv_simd`, element for element. The ternary checkpoint's
// level-3-never-occurs property (AGENTS.md Gotcha 48) is a fact about the
// DATA and says nothing about the kernel; the general affine form is
// implemented, as in the GEMV.
//
// B IS BAKED (function constant), for the INT4 batch kernel's recorded
// reason, and the pipeline-cache key MUST carry it (crate Gotcha 1).
//
// NO R AXIS: unmeasured on this kernel, so unbaked. One row per SIMD group
// is the GEMV's mapping and the parity test's baseline.
// ============================================================================

constant constexpr uint kMaxBatchRows2BitGemm = 16;

constant uint FC_GEMM2_M      [[function_constant(100)]];
constant uint FC_GEMM2_N      [[function_constant(101)]];
constant uint FC_GEMM2_B      [[function_constant(102)]];
constant bool FC_GEMM2_USE_FC [[function_constant(103)]];

kernel void dequant_int2_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    scales [[buffer(1)]],
    device const half*    biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        G      [[buffer(7)]],
    constant uint&        B      [[buffer(8)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint m_dim = (is_function_constant_defined(FC_GEMM2_USE_FC) && FC_GEMM2_USE_FC &&
                        is_function_constant_defined(FC_GEMM2_M)) ? FC_GEMM2_M : M;
    const uint n_dim = (is_function_constant_defined(FC_GEMM2_USE_FC) && FC_GEMM2_USE_FC &&
                        is_function_constant_defined(FC_GEMM2_N)) ? FC_GEMM2_N : N;
    const uint b_dim = (is_function_constant_defined(FC_GEMM2_USE_FC) && FC_GEMM2_USE_FC &&
                        is_function_constant_defined(FC_GEMM2_B)) ? FC_GEMM2_B : B;

    const uint row = tg_idx * kRowsPerTG2Bit + sg_idx;
    if (row >= m_dim) return;

    const uint row_bytes       = n_dim / kElemsPerByte2Bit;
    const uint n_groups        = n_dim / G;
    const uint bytes_per_group = G / kElemsPerByte2Bit;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const half*    s_row = scales + uint(row) * n_groups;
    device const half*    b_row = biases + uint(row) * n_groups;

    float acc[kMaxBatchRows2BitGemm];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint j = lane; j < row_bytes; j += kLanesPerRow2Bit) {
        const uint  g = j / bytes_per_group;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint  byte = uint(W_row[j]);
        const uint  elem = j * kElemsPerByte2Bit;
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            float dot = 0.0f;
            float sum = 0.0f;
            for (uint k = 0; k < kElemsPerByte2Bit; ++k) {
                const float xv = float(x_b[elem + k]);
                // LSB-first fields in k order, exactly the GEMV's walk: the
                // field order is load-bearing (measured against
                // `mx.dequantize`, see this file's header) and the
                // accumulation order is load-bearing (bit-exact parity).
                dot = fma(float((byte >> (2u * k)) & 3u), xv, dot);
                sum += xv;
            }
            acc[bi] = fma(s, dot, acc[bi]);
            acc[bi] = fma(b, sum, acc[bi]);
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float total = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * m_dim + row] = half(total);
        }
    }
}
