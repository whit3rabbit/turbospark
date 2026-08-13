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
