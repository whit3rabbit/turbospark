#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_1bit - MLX `affine` dequant at ONE bit (the
// `prism-ml/Bonsai-27B-mlx-1bit` layout). PORT-LOCAL, not vendored: the Swift
// engine reads no 1-bit checkpoint, so there is no upstream kernel to mirror.
// Its contract is `turbospark_compute::quant_1bit` and nothing else, which is
// why the kernels below carry that module's names rather than this file's.
//
// Layout (per row of length N, N a multiple of the group size G):
//   W       : N/8 bytes. Element i is bit (i % 8) of byte i / 8, LEAST
//             significant bit first.
//   scales  : N/G FP16, one per group.
//   biases  : N/G FP16, one per group.
//   value   : w[i] = float(bit[i]) * scale[i/G] + bias[i/G], bit in {0, 1}.
//
// Three ways this differs from the INT4 affine sibling next door, each of
// which is a silently wrong answer rather than a crash if carried over:
//   1. The companions are `half`, not `bfloat`. The two are the same width,
//      so binding one as the other passes every length check and reads this
//      checkpoint's 0.0271 scales as 1.7e-16.
//   2. The group size is 128, not 64, and it is a RUNTIME UNIFORM here
//      rather than the sibling's `constant constexpr`. It is a property of
//      the checkpoint, not of the container, and a function constant that
//      missed the pipeline cache's constants key would silently reuse the
//      wrong pipeline (crates/gpu Gotcha 1). A uniform cannot.
//   3. A byte holds 8 elements, not 2, so the bit ORDER within the byte is
//      load-bearing. It was measured against `mx.dequantize` rather than
//      read off a doc; at one bit the MSB-first alternative produces weights
//      of exactly the right magnitude and the wrong sign.
//
// Affine factoring, as in the INT4 sibling: over any run inside one group,
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias cost one FMA each per BYTE rather than per element. The
// run here is one byte, which is inside one group because G is a multiple
// of 8.
//
// `x` is read element by element rather than through `half4`, unlike the
// INT4 sibling. A byte's eight elements are 16-byte aligned relative to the
// buffer's base, but an offset-bound `x` need not be, and nothing dispatches
// this on a hot path yet. Vectorize it when something does, not before.
// ============================================================================

constant constexpr uint kRowsPerTG1Bit = 8;
constant constexpr uint kLanesPerRow1Bit = 32;

// y[m] = sum_n W[m, n] * x[n], the GENERAL affine form: every group is
// decoded through its own `(scale, bias)` pair and no relationship between
// the two is assumed. One SIMD group per output row, each lane walking the
// row's bytes at a stride of 32, so a row shorter than 32 bytes simply
// leaves the high lanes contributing zero. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_int1_gemv_simd(
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
    const uint row = tg_idx * kRowsPerTG1Bit + sg_idx;
    if (row >= M) return;

    const uint row_bytes       = N / 8u;
    const uint n_groups        = N / G;
    const uint bytes_per_group = G / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const half*    s_row = scales + uint(row) * n_groups;
    device const half*    b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint j = lane; j < row_bytes; j += kLanesPerRow1Bit) {
        // Resolved per byte, not hoisted: a row spans many groups and each
        // carries its own pair.
        const uint  g = j / bytes_per_group;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint  byte = uint(W_row[j]);
        const uint  elem = j * 8u;
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint k = 0; k < 8u; ++k) {
            const float xv = float(x[elem + k]);
            // LSB-first: element `elem + k` is bit k of this byte.
            dot = fma(float((byte >> k) & 1u), xv, dot);
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

// One row of a 1-bit affine embedding table, dequantized into `out` and
// scaled. The sibling of `embed_lookup_int4` in `dequant_int4.metal`, and it
// exists because the real checkpoint quantizes its EMBEDDING TABLE at one bit
// like everything else -- read off the safetensors header
// (`language_model.model.embed_tokens` is `U32 [248320, 160]` with F16
// companions), not assumed. So 1-bit is a type with a GEMV and a lookup and
// no routed-expert pair, which is the per-type footing AGENTS.md Gotcha 29
// describes.
//
// One thread per element; `out_scale` is sqrt(hidden) for a scaled-embedding
// family, 1.0 otherwise. The row stride is `D / 8`, not `D`: getting that
// factor wrong lands inside a neighbouring token's weights, which decodes
// perfectly well and is the wrong token.
kernel void embed_lookup_int1(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/8] packed bits
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
    device const uint8_t* row_q = table  + uint(token_id) * (D / 8u);
    device const half*    row_s = scales + uint(token_id) * groups_per_row;
    device const half*    row_b = biases + uint(token_id) * groups_per_row;
    const uint  q = (uint(row_q[gid / 8u]) >> (gid % 8u)) & 1u;
    const float s = float(row_s[gid / G]);
    const float b = float(row_b[gid / G]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// The `+/-1` form, for rows whose every group satisfies `bias == -scale/2`
// (`turbospark_compute::is_symmetric`). Its two representable values are
// then `+/- scale/2`, so a byte contributes `(scale/2) * sum(+/-x)` and the
// weights never materialize.
//
// A SEPARATE KERNEL rather than a branch inside the one above, for two
// reasons that both matter. It is not bit-identical: factoring the scale out
// of the group reassociates the sum, which is the territory AGENTS.md Gotcha
// 27 is about. And it BINDS NO BIAS PLANE AT ALL, so a caller that has not
// established the symmetry cannot reach it by accident - the check lives in
// the repack walk, and the argument it would need does not exist here.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_int1_gemv_symmetric_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    scales [[buffer(1)]],
    device const half*    x      [[buffer(2)]],
    device half*          y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    constant uint&        G      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTG1Bit + sg_idx;
    if (row >= M) return;

    const uint row_bytes       = N / 8u;
    const uint n_groups        = N / G;
    const uint bytes_per_group = G / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const half*    s_row = scales + uint(row) * n_groups;

    float acc = 0.0f;
    for (uint j = lane; j < row_bytes; j += kLanesPerRow1Bit) {
        const uint  g = j / bytes_per_group;
        const float half_scale = float(s_row[g]) * 0.5f;
        const uint  byte = uint(W_row[j]);
        const uint  elem = j * 8u;
        float signed_sum = 0.0f;
        for (uint k = 0; k < 8u; ++k) {
            const float xv = float(x[elem + k]);
            // Bit set means `+scale/2`, clear means `-scale/2`.
            signed_sum += ((byte >> k) & 1u) ? xv : -xv;
        }
        acc = fma(half_scale, signed_sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// ============================================================================
// dequant_int1_gemm_simd - the GEMV above with B right-hand sides instead of
// one, so the speculative verify pass can run B drafted tokens through one
// 1-bit matrix in ONE dispatch. PORT-LOCAL, like everything in this file:
// there is no upstream kernel to mirror.
//
// The idea is `dequant_int4_batch.metal`'s, and it pays MORE here: at one bit
// the weight bytes are eight times smaller than the FP16 activations they
// feed, so B separate GEMV dispatches re-read the whole matrix B times for
// the sake of B token activations that would fit in registers together.
// Each packed byte is read once and multiplied into B accumulators.
//
// IT IS BIT-IDENTICAL TO B CALLS OF `dequant_int1_gemv_simd`, and the
// structure is what makes that true rather than an assertion about it: the
// row mapping (one SIMD group per output row, `row = tg * 8 + sg`), the
// lane-strided byte walk (`j = lane; j < row_bytes; j += 32`), the per-byte
// group resolution, the affine factoring order (`fma(s, dot)` then
// `fma(b, sum)` into one FP32 accumulator per byte) and the 32-lane
// `simd_sum` partition are the GEMV's own, unchanged. Only the inner loop
// is new, and it replicates the GEMV's per-byte arithmetic once per right-
// hand side in the same element order. `dequant_1bit_gemm_parity.rs` holds
// every batch width 1..16 to that contract against the actual GEMV.
//
// B IS BAKED as a function constant, for the reason the INT4 batch kernel's
// header records: baked B lets the `for (bi < B)` loop unroll and bounds the
// live accumulator set to the batch in flight, on a kernel whose register
// file is the binding constraint. The key MUST carry the baked values
// (crate Gotcha 1): an axis missing from `pipeline`'s key is served whichever
// shape compiled first, and a wrong baked B here produces finite, plausible,
// quietly wrong rows.
//
// NO R AXIS. The INT4 kernel's row-block constant exists because a measured
// sweep found R=2/4 wins on its shapes; nothing has measured R for this one,
// and a shape nobody measured is not a shape to bake. One row per SIMD
// group is the GEMV's mapping and the parity test's baseline. Measure first.
// ============================================================================

constant constexpr uint kMaxBatchRows1BitGemm = 16;

constant uint FC_GEMM1_M      [[function_constant(100)]];
constant uint FC_GEMM1_N      [[function_constant(101)]];
constant uint FC_GEMM1_B      [[function_constant(102)]];
constant bool FC_GEMM1_USE_FC [[function_constant(103)]];

kernel void dequant_int1_gemm_simd(
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
    const uint m_dim = (is_function_constant_defined(FC_GEMM1_USE_FC) && FC_GEMM1_USE_FC &&
                        is_function_constant_defined(FC_GEMM1_M)) ? FC_GEMM1_M : M;
    const uint n_dim = (is_function_constant_defined(FC_GEMM1_USE_FC) && FC_GEMM1_USE_FC &&
                        is_function_constant_defined(FC_GEMM1_N)) ? FC_GEMM1_N : N;
    const uint b_dim = (is_function_constant_defined(FC_GEMM1_USE_FC) && FC_GEMM1_USE_FC &&
                        is_function_constant_defined(FC_GEMM1_B)) ? FC_GEMM1_B : B;

    const uint row = tg_idx * kRowsPerTG1Bit + sg_idx;
    if (row >= m_dim) return;

    const uint row_bytes       = n_dim / 8u;
    const uint n_groups        = n_dim / G;
    const uint bytes_per_group = G / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const half*    s_row = scales + uint(row) * n_groups;
    device const half*    b_row = biases + uint(row) * n_groups;

    // Declared at the cap because MSL needs a compile-time bound; the baked
    // `b_dim` stops every loop at the batch in flight so the optimizer drops
    // the rest. THAT COLLAPSE IS UNMEASURED HERE -- the INT4 kernel's header
    // records that no static instrument on this device can see register
    // pressure, and the same caveat applies until a c(M) sweep says otherwise.
    float acc[kMaxBatchRows1BitGemm];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint j = lane; j < row_bytes; j += kLanesPerRow1Bit) {
        const uint  g = j / bytes_per_group;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint  byte = uint(W_row[j]);
        const uint  elem = j * 8u;
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            float dot = 0.0f;
            float sum = 0.0f;
            for (uint k = 0; k < 8u; ++k) {
                const float xv = float(x_b[elem + k]);
                // LSB-first, exactly the GEMV's order: the bit order within
                // the byte is load-bearing (see this file's header), and so
                // is the accumulation order, because the parity contract is
                // bit-exactness with the GEMV rather than equality up to
                // rounding.
                dot = fma(float((byte >> k) & 1u), xv, dot);
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
