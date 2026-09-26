#include <metal_stdlib>
using namespace metal;

// ============================================================================
// rmsnorm — RMS-norm over hidden dim.
//
//   inv     = rsqrt(mean(x[i]^2) + eps)
//   y[i]    = x[i] * inv * weight[i]
//
// FP32 accumulator (numerical stability: D=2816 with FP16 inputs can overflow
// FP16 sum-of-squares once activations grow past ~1.2 in magnitude).
// FP16 storage in and out. Learned weights are BF16 where present.
//
// Dispatch: one threadgroup per row, 256 threads per group. Two-stage block
// reduce — SIMD-group simd_sum, then a single SIMD-group merges the partials.
// ============================================================================

// Threadgroup memory carries at most simdgroups_per_threadgroup = 256/32 = 8
// partial sums. Slot 0 is reused after the merge to broadcast the final inv.
constant constexpr uint kRmsMaxSimdGroups = 8;
constant uint FC_RMS_D [[function_constant(30)]];
constant bool FC_RMS_USE_FC [[function_constant(31)]];

static inline uint rms_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_RMS_USE_FC) &&
            FC_RMS_USE_FC &&
            is_function_constant_defined(FC_RMS_D)) ? FC_RMS_D : D;
}

// Common block reduction. Returns `inv = rsqrt(mean(x^2) + eps)` broadcast to
// every thread via threadgroup memory slot 0.
static inline float rms_block_inv(
    device const half* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// Gemma 4 RMS norms ship as BF16 weight vectors (all 30 layers' input /
// post-attn / pre-FFN / post-FFN norms, plus q/k_norm). Math is identical to
// the no-scale form below, with a learned weight applied after normalization.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w(
    device const half*   x          [[buffer(0)]],   // [D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [D] BF16
    device       half*   out        [[buffer(2)]],   // [D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        float wv = float(weight[i]);
        out[i] = half(xv * inv * wv);
    }
}

// PORT-LOCAL (not in the Swift rmsnorm.metal): the CENTERED form, whose
// learned weight is an OFFSET FROM UNITY rather than the scale itself.
//
//   y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + weight[i])
//
// Swift's engine reads no architecture with this convention, so there is no
// upstream kernel to diff this against; its only contract is
// `turbospark_compute::rms_norm_centered`.
//
// Added for ROADMAP's `muse_glimmer` entry, whose four per-layer norms are
// `CenteredRMSNorm` in the reference. Note its FINAL norm is a plain
// `nn.RMSNorm` and therefore uses `rmsnorm_bf16w` above: one model, both
// conventions, so this is a SEPARATE KERNEL rather than a function constant
// on that one. A specialization axis whose byte missed the pipeline cache's
// `constants_key` would silently reuse whichever pipeline was compiled first
// (crate Gotcha 1), and here that would mean the two norms of one layer
// quietly becoming the same function -- a wrong model that still decodes.
// A separate kernel name is a separate pipeline by construction.
//
// The `1.0f +` is applied on the FP32 accumulator and never baked into the
// stored weight: BF16's resolution near 1.0 is 2^-8, so a centred weight of
// 0.01 would lose ~39% of its magnitude to a repack-time bake.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_centered(
    device const half*   x          [[buffer(0)]],   // [D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [D] BF16, centered at 0
    device       half*   out        [[buffer(2)]],   // [D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        float wv = float(weight[i]);
        out[i] = half(xv * inv * (1.0f + wv));
    }
}

// Gemma 4 applies q_norm/k_norm
// (BF16 weight, shared across heads) and v_norm (no-scale) to each attention
// head independently. These kernels process all heads in one dispatch, with
// one threadgroup per head, avoiding a chain of tiny serialized encoders.
// Math is identical to the single-row kernels applied per head.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, shared per head
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        oh[i] = half(xv * inv * wv);
    }
}

// PORT-LOCAL: the CENTERED form of the per-head norm above, whose learned
// weight is an OFFSET FROM UNITY rather than the scale itself.
//
//   y[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + weight[i])
//
// `rmsnorm_bf16w_centered` is this same `1 +` over a whole vector; this one
// exists because the `qwen3_5` MTP head's per-head `q_norm`/`k_norm` carry the
// convention too, while the TRUNK's per-head norms of the same names do not.
// So the choice is per TENSOR and not per family, and for the same reason the
// whole-vector pair is two kernels this is a separate kernel rather than a
// function constant on its plain sibling: a specialization axis whose byte
// missed `MetalContext::pipeline`'s `constants_key` would silently reuse
// whichever pipeline compiled first, making the two conventions one function
// (crate Gotcha 1, and AGENTS.md Gotcha 50).
//
// The `1.0f +` is applied on the FP32 accumulator and never baked into the
// stored weight. These weights read ~0.78, so a bake is less destructive here
// than for the near-zero whole-vector norms, but it still halves the effective
// resolution: BF16's quantum near 0.78 is 2^-8 and near 1.78 it is 2^-7.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead_centered(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, centered at 0
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        oh[i] = half(xv * inv * (1.0f + wv));
    }
}

// PORT-LOCAL: the GROUPED CENTERED form (`qwen4_exp`'s `hc_norm` and PLE's
// `norm_key`/`norm_query`/`norm_conv`, `docs/QWEN4_PHASE0.md` item 9's norm
// taxonomy, row 1). Structurally close to `rmsnorm_bf16w_perhead_centered`
// above -- an independent reduction per `GD`-wide group, one threadgroup per
// group -- but the weight is NOT shared across groups the way a per-head
// weight is shared across heads: it is a single `groups * GD`-wide vector,
// read at each element's own GLOBAL index.
//
//   for group g:
//     inv_g = rsqrt(mean(x[g*GD .. (g+1)*GD]^2) + eps)
//     y[g*GD+i] = x[g*GD+i] * inv_g * (1 + weight[g*GD+i])
//
// NOT a wider `rmsnorm_bf16w_centered`: that kernel takes ONE statistic over
// its whole input, and `qwen4_exp`'s `pre_fc_norm_hidden` (taxonomy row 3) is
// exactly that at the SAME 10240-element width this kernel runs at. Both
// bind a full-width BF16 weight, so no buffer-size mismatch would catch a
// call site that passed the wrong `groups` -- at `groups=1` this kernel IS
// `rmsnorm_bf16w_centered` mathematically (one threadgroup, one statistic
// over the whole vector), which is exactly why a caller cannot be trusted to
// get `groups` right by construction: `hc_norm` (`groups=4`) and
// `pre_fc_norm_hidden` (a DIFFERENT tensor, plain) are two call sites on two
// tensors, not two settings of one dial, and the host wrapper for the plain
// tensor dispatches `rmsnorm_bf16w_centered` by name rather than this kernel
// at `groups=1`.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_grouped_centered(
    device const half*   x          [[buffer(0)]],   // [groups * GD] FP16
    device const bfloat* weight     [[buffer(1)]],   // [groups * GD] BF16, centered at 0
    device       half*   out        [[buffer(2)]],   // [groups * GD] FP16
    constant     uint&   groupDim   [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  group            [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint GD = rms_fc_d(groupDim);
    const uint base = group * GD;
    device const half*   xg = x      + base;
    device const bfloat* wg = weight + base;
    device       half*   og = out    + base;
    const float inv = rms_block_inv(xg, GD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < GD; i += lsize) {
        float xv = float(xg[i]);
        float wv = float(wg[i]);
        og[i] = half(xv * inv * (1.0f + wv));
    }
}

// The plain-scale grouped form is used when the checkpoint stores gamma
// directly (for example, llama.cpp-converted Qwen4 GGUF tensors). The
// reduction and global weight indexing match the centered sibling; only the
// scale convention differs.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_grouped(
    device const half*   x          [[buffer(0)]],   // [groups * GD] FP16
    device const bfloat* weight     [[buffer(1)]],   // [groups * GD] BF16, direct scale
    device       half*   out        [[buffer(2)]],   // [groups * GD] FP16
    constant     uint&   groupDim   [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  group            [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint GD = rms_fc_d(groupDim);
    const uint base = group * GD;
    device const half*   xg = x      + base;
    device const bfloat* wg = weight + base;
    device       half*   og = out    + base;
    const float inv = rms_block_inv(xg, GD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < GD; i += lsize) {
        og[i] = half(float(xg[i]) * inv * float(wg[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale_perhead(
    device const half*  x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device       half*  out        [[buffer(1)]],   // [numHeads * headDim] FP16
    constant     uint&  headDim    [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        oh[i] = half(float(xh[i]) * inv);
    }
}

// Gemma 4 v_norm and the MoE router's internal norm are no-scale RMSNorm:
// y[i] = x[i] * rsqrt(mean(x^2) + eps). There is no resident weight tensor.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale(
    device const half*  x          [[buffer(0)]],   // [D] FP16
    device       half*  out        [[buffer(1)]],   // [D] FP16
    constant     uint&  D          [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        out[i] = half(xv * inv);
    }
}
