#include <metal_stdlib>
using namespace metal;

// ============================================================================
// PORT-LOCAL (ROADMAP M-V2): the qwen3_5 vision tower.
//
// The Swift engine has no vision tower, so nothing here is vendored and there
// is no upstream kernel to diff against. `turbospark_compute::vision` is the
// only definition of what these compute, and `tests/vision_parity.rs` is what
// holds them to it.
//
// EVERY WEIGHT HERE IS `half`, NOT `bfloat`, and that is a deliberate break
// with the rest of this workspace. Every other learned weight in these
// shaders is BF16, because every other family's resident tensors are stored
// that way. The vision tower is signed off at FP16 end to end
// (`docs/VISION_PHASE0.md`): its own checkpoints ship F16, the extreme-page
// probe puts peak activations at 13.8% of FP16's ceiling with a factor of 7.3
// in hand, and INT4 was measured and REJECTED on OCR quality. Binding a
// `bfloat*` at one of these buffers reads the same bytes as a different
// number and produces finite, wrong output.
//
// Accumulators are FP32 throughout, which is free here and is what makes the
// FP16 storage decision safe rather than merely measured.
// ============================================================================

constant constexpr uint kVisionMaxSimdGroups = 8;   // 256 threads / 32
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

// ----------------------------------------------------------------------------
// LayerNorm. NOT RMSNorm: the mean is subtracted and a bias is added, which is
// what `nn.LayerNorm` does and what every other norm in this workspace does
// not. `eps` is a runtime argument and sits INSIDE the square root.
//
// Two reduction passes over the row (mean, then variance about that mean)
// rather than the one-pass sum/sum-of-squares identity. The identity is
// `var = E[x^2] - E[x]^2`, and it CANCELS catastrophically when the mean is
// large relative to the spread -- which is this tower's normal condition, not
// an edge case: block 26's activations reach absmax 9,024 with rms 145.
// A row there has E[x^2] and E[x]^2 agreeing to several digits, and the
// subtraction keeps only the digits that disagree.
//
// Dispatch: one threadgroup per row, 256 threads.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_layer_norm_fp16(
    device const half*  x       [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    device half*        out     [[buffer(3)]],
    constant uint&      D       [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint  row           [[threadgroup_position_in_grid]],
    uint  lid           [[thread_position_in_threadgroup]],
    uint  lsize         [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kVisionMaxSimdGroups];
    device const half* xr = x + uint64_t(row) * D;
    device half*       yr = out + uint64_t(row) * D;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        acc += float(xr[i]);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) { partial[0] = v / float(D); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float vacc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float d = float(xr[i]) - mean;
        vacc = fma(d, d, vacc);
    }
    vacc = simd_sum(vacc);
    if (simd_lane_id == 0) { partial[simd_group_id] = vacc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) { partial[0] = rsqrt(v / float(D) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = partial[0];

    for (uint i = lid; i < D; i += lsize) {
        const float hat = (float(xr[i]) - mean) * inv_std;
        yr[i] = half(fma(hat, float(weight[i]), float(bias[i])));
    }
}

// ----------------------------------------------------------------------------
// The two GELUs, elementwise and in place.
//
// SEPARATE KERNELS RATHER THAN ONE WITH A MODE FLAG. This tower uses both --
// tanh in each block's MLP, exact erf in the merger -- and they agree to about
// 3e-4, so selecting between them wrongly is invisible in every coarse check.
// A distinct kernel name is a distinct pipeline by construction, where a mode
// byte that missed `MetalContext::pipeline`'s constants key would silently
// reuse whichever compiled first (`crates/gpu` Gotcha 1).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_gelu_tanh_fp16(
    device half*    y     [[buffer(0)]],
    constant uint&  count [[buffer(1)]],
    uint            tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float x = float(y[tid]);
    const float x3 = x * x * x;
    // Clamped for the reason utility.metal's copy is: Metal's tanh returns
    // NaN at large magnitudes, and at FP32 precision the clamped result is
    // the saturated one.
    const float inner = clamp(kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3), -20.0f, 20.0f);
    y[tid] = half(0.5f * x * (1.0f + tanh(inner)));
}

// METAL SHIPS NO `erf`, which is worth stating because it reads like it
// should: `metal_math` has `exp`, `tanh`, `rsqrt` and every other libm name a
// reader expects, and the first version of this kernel called `erf` and failed
// to COMPILE rather than silently doing something else. So the series is
// written out, and it is the same Abramowitz-Stegun 7.1.26 the CPU reference
// evaluates -- deliberately the same approximation rather than a different
// one, so the parity bound measures the FP32-vs-FP64 evaluation and the FP16
// storage, and not a gap between two rival expansions of erf.
static inline float vision_erf(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + p * ax);
    const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign * y;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_gelu_erf_fp16(
    device half*    y     [[buffer(0)]],
    constant uint&  count [[buffer(1)]],
    uint            tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float x = float(y[tid]);
    y[tid] = half(0.5f * x * (1.0f + vision_erf(x * M_SQRT1_2_F)));
}

// ----------------------------------------------------------------------------
// The tower's 2-D rotary embedding.
//
// One thread per (token, head, pair). Element `i` rotates with `i + head/2`
// (the NeoX half-split), and the angle comes from a per-token FREQUENCY ROW
// rather than from a scalar position and a theta -- a patch has a
// two-dimensional position and no single scalar encodes it. `freqs` is
// `[seq, head_dim/2]`, shared by every head of a token.
//
// `qkv` is laid out `[seq, heads, head_dim]`, which is what the qkv
// projection writes and what the attention kernel below reads, so no
// transpose happens between them.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_rope_2d_fp16(
    device half*        qkv       [[buffer(0)]],
    device const half*  freqs     [[buffer(1)]],
    constant uint&      seq       [[buffer(2)]],
    constant uint&      heads     [[buffer(3)]],
    constant uint&      head_dim  [[buffer(4)]],
    uint3               gid       [[thread_position_in_grid]]
) {
    const uint pair = gid.x;          // 0 .. head_dim/2
    const uint head = gid.y;
    const uint token = gid.z;
    const uint half_dim = head_dim / 2;
    if (pair >= half_dim || head >= heads || token >= seq) return;

    const float angle = float(freqs[uint64_t(token) * half_dim + pair]);
    const float c = cos(angle);
    const float s = sin(angle);

    device half* row = qkv + (uint64_t(token) * heads + head) * head_dim;
    const float a = float(row[pair]);
    const float b = float(row[pair + half_dim]);
    row[pair]            = half(a * c - b * s);
    row[pair + half_dim] = half(b * c + a * s);
}

// ----------------------------------------------------------------------------
// Bidirectional multi-head attention over one image's patches.
//
// NO MASK AND NO KV CACHE, which is the whole difference from
// `attention.metal`. Every patch attends to every patch; nothing persists
// between images. A causal mask here leaves the top-left patch blind to
// everything below and right of it, which degrades an image rather than
// corrupting it -- the failure mode no smoke test catches.
//
// ONE PASS OVER THE KEYS, online softmax. One threadgroup per (query token,
// head); its 8 SIMD groups split the keys, and within a SIMD group the 32
// lanes split the head dimension. Each lane keeps `head_dim / 32` output
// accumulators in registers alongside a running max and a running denominator,
// so the `seq x seq` score matrix is never materialized.
//
// THE ALTERNATIVE SHAPE IS A TRAP WORTH NAMING, because it is the one a
// straightforward port writes: three passes (max, denominator, weighted sum)
// with the threads split over the head dimension in the last one. That is
// correct, and it recomputes every query-key dot product once per OUTPUT
// ELEMENT -- `head_dim` times too much work, which at 72 is a factor of 72 on
// the tower's dominant cost. The first draft of this kernel did exactly that.
//
// The running max is not an optimization either. Block 26's activations reach
// absmax 9,024 (`docs/VISION_PHASE0.md` item 3), so an unshifted exp overflows
// on a real page rather than on a contrived one.
// ----------------------------------------------------------------------------

// Head dimension ceiling, from the register tile: a lane owns
// `head_dim / 32` accumulators. The tower runs at 72 (hidden 1152 / 16 heads),
// so 4 slots is 128 and comfortable. `VisionAttentionShape::validate` on the
// host refuses anything larger rather than letting it silently truncate.
constant constexpr uint kVisionAttnMaxSlots = 4;

[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_attention_bidir_fp16(
    device const half*  q         [[buffer(0)]],
    device const half*  k         [[buffer(1)]],
    device const half*  v         [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    constant uint&      seq       [[buffer(4)]],
    constant uint&      heads     [[buffer(5)]],
    constant uint&      head_dim  [[buffer(6)]],
    constant float&     scale     [[buffer(7)]],
    uint2 tg            [[threadgroup_position_in_grid]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]]
) {
    const uint token = tg.x;
    const uint head = tg.y;
    if (token >= seq || head >= heads) return;

    threadgroup float tg_max[kVisionMaxSimdGroups];
    threadgroup float tg_sum[kVisionMaxSimdGroups];
    threadgroup float tg_acc[kVisionMaxSimdGroups][kVisionAttnMaxSlots * 32];

    device const half* qrow = q + (uint64_t(token) * heads + head) * head_dim;

    // The lane's slice of the head dimension: d = lane, lane+32, lane+64, ...
    const uint slots = (head_dim + 31u) / 32u;

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    float acc[kVisionAttnMaxSlots] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint j = simd_group_id; j < seq; j += simdgroups) {
        device const half* krow = k + (uint64_t(j) * heads + head) * head_dim;
        // Cooperative dot product: each lane sums its own slice, then
        // `simd_sum` broadcasts the full score to every lane of the group.
        float dot = 0.0f;
        for (uint t = 0; t < slots; ++t) {
            const uint d = t * 32u + simd_lane_id;
            if (d < head_dim) {
                dot = fma(float(qrow[d]), float(krow[d]), dot);
            }
        }
        dot = simd_sum(dot);
        const float score = dot * scale;

        // Online softmax: rescale what is already accumulated, then add this
        // key's contribution at the new reference point.
        const float new_max = max(run_max, score);
        const float correction = exp(run_max - new_max);
        const float weight = exp(score - new_max);
        run_sum = fma(run_sum, correction, weight);
        device const half* vrow = v + (uint64_t(j) * heads + head) * head_dim;
        for (uint t = 0; t < slots; ++t) {
            const uint d = t * 32u + simd_lane_id;
            acc[t] *= correction;
            if (d < head_dim) {
                acc[t] = fma(weight, float(vrow[d]), acc[t]);
            }
        }
        run_max = new_max;
    }

    // Merge the SIMD groups' partial softmaxes, each at its own reference
    // point. Same rescale-to-a-common-max rule as the loop above.
    if (simd_lane_id == 0) {
        tg_max[simd_group_id] = run_max;
        tg_sum[simd_group_id] = run_sum;
    }
    for (uint t = 0; t < slots; ++t) {
        tg_acc[simd_group_id][t * 32u + simd_lane_id] = acc[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id != 0) return;

    float global_max = -INFINITY;
    for (uint g = 0; g < simdgroups; ++g) {
        global_max = max(global_max, tg_max[g]);
    }
    // A SIMD group processes no keys at all when `seq < simdgroups`, leaving
    // its max at -INFINITY and its sum and accumulators at zero. NO GUARD IS
    // NEEDED for that: `global_max` is finite whenever any key exists, so the
    // term is `0 * exp(-inf) == 0 * 0 == 0` and contributes nothing. The one
    // case that WOULD produce `exp(-inf - -inf) == NaN` is every group empty,
    // i.e. `seq == 0`, which dispatches no threadgroups at all.
    //
    // This carried an `if (tg_sum[g] > 0)` guard until a mutation test found
    // that deleting it changed no result -- it was dead, and its comment
    // asserted a NaN that IEEE arithmetic does not produce here.
    float denom = 0.0f;
    for (uint g = 0; g < simdgroups; ++g) {
        denom += tg_sum[g] * exp(tg_max[g] - global_max);
    }
    const float inv_denom = 1.0f / denom;

    device half* orow = out + (uint64_t(token) * heads + head) * head_dim;
    for (uint t = 0; t < slots; ++t) {
        const uint d = t * 32u + simd_lane_id;
        if (d >= head_dim) continue;
        float total = 0.0f;
        for (uint g = 0; g < simdgroups; ++g) {
            total = fma(tg_acc[g][d], exp(tg_max[g] - global_max), total);
        }
        orow[d] = half(total * inv_denom);
    }
}

// ----------------------------------------------------------------------------
// The tower's GEMM: `out[m][n] = sum_k a[m][k] * b[n][k] + bias[n]`.
//
// `b` is ROW-MAJOR BY OUTPUT (`[n_out, k]`), which is how an `nn.Linear`
// weight ships, so the inner loop walks one output's whole row contiguously
// and no transpose happens anywhere. Reading it `[k, n_out]` instead
// transposes every projection while keeping every buffer-size check happy.
//
// One threadgroup per (row, output), 256 threads reducing over `k`. Serves
// every projection in the tower: qkv, proj, fc1, fc2, patch_embed and both
// merger layers.
//
// `has_bias` is a runtime uint rather than a function constant, for
// `crates/gpu` Gotcha 1's reason -- `MetalContext::pipeline`'s constants key
// would have to carry it, and a specialization axis missing from that key
// silently reuses the wrong pipeline.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_matmul_fp16(
    device const half*  a         [[buffer(0)]],
    device const half*  b         [[buffer(1)]],
    device const half*  bias      [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    constant uint&      k_dim     [[buffer(4)]],
    constant uint&      n_dim     [[buffer(5)]],
    constant uint&      has_bias  [[buffer(6)]],
    // ALL THREE POSITION ATTRIBUTES ARE `uint2`, and that is a Metal rule
    // rather than a style: a kernel's position inputs must be all scalar or
    // all vectors of the SAME width, so pairing a `uint2` threadgroup
    // position with a scalar `thread_position_in_threadgroup` does not
    // compile. The simdgroup attributes are a different family and stay
    // scalar.
    uint2 tg            [[threadgroup_position_in_grid]],
    uint2 lid2          [[thread_position_in_threadgroup]],
    uint2 lsize2        [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups    [[simdgroups_per_threadgroup]]
) {
    const uint lid = lid2.x;
    const uint lsize = lsize2.x;
    const uint row = tg.x;
    const uint col = tg.y;
    if (col >= n_dim) return;

    threadgroup float partial[kVisionMaxSimdGroups];
    device const half* arow = a + uint64_t(row) * k_dim;
    device const half* brow = b + uint64_t(col) * k_dim;

    float acc = 0.0f;
    for (uint i = lid; i < k_dim; i += lsize) {
        acc = fma(float(arow[i]), float(brow[i]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            out[uint64_t(row) * n_dim + col] =
                half(v + (has_bias != 0 ? float(bias[col]) : 0.0f));
        }
    }
}

// ----------------------------------------------------------------------------
// Residual add, `y += x`, in FP16 storage with an FP32 intermediate.
//
// Its own kernel rather than a fused tail on the matmul: the block adds the
// attention output to the pre-norm residual and the MLP output to the
// post-attention one, and neither addend is the matmul that produced it.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_residual_add_fp16(
    device half*        y     [[buffer(0)]],
    device const half*  x     [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint                tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    y[tid] = half(float(y[tid]) + float(x[tid]));
}
