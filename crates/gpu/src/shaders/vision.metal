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
// `freqs` is F32, not FP16 (AGENTS.md/CLAUDE.md B7): the angle at pair 0
// equals the raw patch coordinate, so on a wide grid it can reach the tens,
// where FP16's step (~0.0625 near 100) rounds the angle by up to half a
// step and moves that pair's cos/sin by a comparable amount -- a real,
// avoidable error, since this table is a host-computed buffer with no
// checkpoint tensor behind it (unlike `qkv`, whose FP16 storage is real
// production width and is not narrowed further here).
//
// `qkv` is laid out `[seq, heads, head_dim]`, which is what the qkv
// projection writes and what the attention kernel below reads, so no
// transpose happens between them.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_rope_2d_fp16(
    device half*        qkv       [[buffer(0)]],
    device const float* freqs     [[buffer(1)]],
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

    const float angle = freqs[uint64_t(token) * half_dim + pair];
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
// The same GEMM on the matrix units (`simdgroup_matrix`), the tower's default.
// `vision_matmul_fp16` above stays as the `TURBOSPARK_VISION_MMA=0` arm.
//
// One threadgroup of 4 simdgroups (128 threads) owns a 64 x 64 output tile;
// each simdgroup owns a 32 x 32 quarter as 4 x 4 FP32 8x8 accumulators. Each
// 32-wide K step stages a [64 rows][32 k] slice of `a` and of `b` in
// threadgroup memory, both row-major by their own leading dimension, so `b`
// is read `[n, k]` exactly as above and the multiply loads it TRANSPOSED.
//
// Out-of-range rows, outputs and K tail are zero-filled in the staging tiles,
// so any (m, k, n) is legal. The fused tower shapes leave K tails
// (fc2's k = 4304 is 134.5 steps of 32), which is what `FC_VISION_GEMM_K_ALIGNED`
// specializes away: false keeps the per-element K check in every load and
// skips the 8-wide sub-steps past K. It is in the pipeline cache key
// (`vision.rs`).
//
// EXACTNESS CLASS: not bit-identical to `vision_matmul_fp16` (the hardware
// reduces each 8-wide K block, then the blocks run in K order), but an output's
// bits depend only on its own row of `a`, its own row of `b` and K, never on M,
// N or where the tile sits. `vision_parity.rs` pins that.
// ----------------------------------------------------------------------------
constant bool FC_VISION_GEMM_K_ALIGNED [[function_constant(110)]];

constant constexpr uint kGemmBM = 64;
constant constexpr uint kGemmBN = 64;
constant constexpr uint kGemmBK = 32;
// Padded row stride of a staging tile, in halves. 8 extra halves keep the
// transposed 8x8 loads of `b` off one bank, and 40 * 2 bytes keeps every row
// 8-byte aligned for the half4 stores below.
constant constexpr uint kGemmLd = kGemmBK + 8;
constant constexpr uint kGemmThreads = 128;

// Lane -> (row, first column) of the two elements a lane holds in an 8x8
// simdgroup_matrix (MLX steel's layout, also `mma8_frag_coord`).
inline uint2 vision_gemm_frag_coord(uint lane) {
    const uint q = lane >> 2;
    return uint2((q & 4u) | ((lane >> 1) & 3u), ((q & 2u) << 1) | ((lane & 1u) << 1));
}

// Stage rows [row0, row0 + 64) x k [k0, k0 + 32) of a row-major [rows, K]
// matrix into `dst` ([64][kGemmLd]). Eight threads per row, four halves each.
// `full` is the caller's proof that the whole slice is in range.
inline void vision_gemm_stage(
    threadgroup half* dst,
    device const half* src,
    uint row0,
    uint rows,
    uint k0,
    uint K,
    uint tid,
    bool full
) {
    const uint c = (tid & 7u) * 4u;
    for (uint r = tid >> 3; r < kGemmBM; r += kGemmThreads / 8u) {
        threadgroup half* d = dst + r * kGemmLd + c;
        const uint gr = row0 + r;
        const uint gk = k0 + c;
        if (full) {
            // packed_half4 has half's 2-byte alignment, so this is legal at
            // any caller offset and row stride.
            const packed_half4 v =
                *(device const packed_half4*)(src + uint64_t(gr) * K + gk);
            *(threadgroup half4*)d = half4(v);
        } else {
            device const half* s = src + uint64_t(gr) * K;
            half4 v = half4(0.0h);
            if (gr < rows) {
                for (uint e = 0; e < 4u; ++e) {
                    if (gk + e < K) { v[e] = s[gk + e]; }
                }
            }
            *(threadgroup half4*)d = v;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void vision_matmul_fp16_mma(
    device const half*  a         [[buffer(0)]],
    device const half*  b         [[buffer(1)]],
    device const half*  bias      [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    constant uint&      k_dim     [[buffer(4)]],
    constant uint&      n_dim     [[buffer(5)]],
    constant uint&      has_bias  [[buffer(6)]],
    constant uint&      m_dim     [[buffer(7)]],
    uint2 tg            [[threadgroup_position_in_grid]],
    uint2 lid2          [[thread_position_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint tid = lid2.x;
    const uint K = k_dim;
    const uint N = n_dim;
    const uint M = m_dim;
    const uint n0 = tg.x * kGemmBN;
    const uint m0 = tg.y * kGemmBM;
    const uint sg_r = simd_group_id >> 1;
    const uint sg_c = simd_group_id & 1u;

    threadgroup half as[kGemmBM * kGemmLd];
    threadgroup half bs[kGemmBN * kGemmLd];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const bool rows_full = m0 + kGemmBM <= M;
    const bool cols_full = n0 + kGemmBN <= N;
    threadgroup const half* a_sg = as + (sg_r * 32u) * kGemmLd;
    threadgroup const half* b_sg = bs + (sg_c * 32u) * kGemmLd;

    for (uint k0 = 0; k0 < K; k0 += kGemmBK) {
        const bool k_full = FC_VISION_GEMM_K_ALIGNED || k0 + kGemmBK <= K;
        vision_gemm_stage(as, a, m0, M, k0, K, tid, rows_full && k_full);
        vision_gemm_stage(bs, b, n0, N, k0, K, tid, cols_full && k_full);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < kGemmBK; kk += 8u) {
            // A sub-step wholly past K would add exact zeros; skipping it
            // changes no bits and saves the K tail's work.
            if (!FC_VISION_GEMM_K_ALIGNED && k0 + kk >= K) { break; }
            simdgroup_half8x8 af[4];
            simdgroup_half8x8 bf[4];
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(af[i], a_sg + (i * 8u) * kGemmLd + kk, kGemmLd);
            }
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_load(bf[j], b_sg + (j * 8u) * kGemmLd + kk, kGemmLd,
                               ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4u; ++i) {
                for (uint j = 0; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Straight from the accumulators to `out`: each lane owns two adjacent
    // outputs of every 8x8 tile, bounds-checked, bias added in FP32 and
    // narrowed once, as `vision_matmul_fp16` does.
    const uint2 fc = vision_gemm_frag_coord(simd_lane_id);
    for (uint i = 0; i < 4u; ++i) {
        const uint r = m0 + sg_r * 32u + i * 8u + fc.x;
        if (r >= M) { continue; }
        device half* orow = out + uint64_t(r) * N;
        for (uint j = 0; j < 4u; ++j) {
            const uint c = n0 + sg_c * 32u + j * 8u + fc.y;
            for (uint e = 0; e < 2u; ++e) {
                if (c + e < N) {
                    const float v = acc[i][j].thread_elements()[e];
                    orow[c + e] =
                        half(v + (has_bias != 0 ? float(bias[c + e]) : 0.0f));
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// The same attention on the matrix units, flash style: the tower's default.
// `vision_attention_bidir_fp16` above stays as the
// `TURBOSPARK_VISION_ATTN_MMA=0` arm and the parity cross-check.
//
// One threadgroup of 4 simdgroups (128 threads) owns 32 query rows of one
// head; each simdgroup owns 8 of them. Q lives in registers as 8x8 FP16
// fragments for the whole kernel. Each step stages a block of 16 keys' K and
// V rows in threadgroup memory (shared by the 4 simdgroups), computes the
// 8 x 16 score tile with `simdgroup_multiply_accumulate` (K loaded
// TRANSPOSED), runs the online softmax in registers, and accumulates P V into
// FP32 8x8 output fragments. Still no mask: the only -inf is the key tail
// past `seq`, which is padding, not masking.
//
// THE SOFTMAX NEVER LEAVES REGISTERS. A lane of an 8x8 fragment holds two
// adjacent columns of ONE row (`vision_gemm_frag_coord`), and the four lanes
// holding the same row differ only in lane bits 0 and 3, so a row max or sum
// is two `simd_shuffle_xor`s (1, then 8). Every fragment of a simdgroup's
// 8 rows, score or output, uses that same row mapping, so each lane's running
// max, denominator and rescale apply to exactly the outputs it holds.
//
// P IS NARROWED TO FP16 for the P V multiply, as flash attention does on half
// inputs, and the denominator sums the narrowed weights so the normalization
// matches the weights actually applied. Scores, max, denominator and the
// output accumulators stay FP32; the running max is kept for the same reason
// as above (block 26's absmax 9,024).
//
// `FC_VISION_ATTN_HEAD_DIM` (function constant 111) is the head dimension:
// Metal will not size an array from a function constant, so the register and
// staging arrays are sized for the 128 ceiling and every loop runs to the
// specialized tile count, which the compiler folds. It is in the pipeline
// cache key (`vision.rs`). The host requires `head_dim % 4 == 0` (the half4
// staging), which the tower's rope already forces.
// ----------------------------------------------------------------------------
constant uint FC_VISION_ATTN_HEAD_DIM [[function_constant(111)]];

constant constexpr uint kAttnBQ = 32;
constant constexpr uint kAttnBK = 16;
constant constexpr uint kAttnMaxTiles = 16;   // MAX_ATTENTION_HEAD_DIM / 8
// Padded staging row stride in halves: 8 extra keep the transposed K loads
// off one bank and every row 8-byte aligned for the half4 stores.
constant constexpr uint kAttnLd = kAttnMaxTiles * 8 + 8;
constant constexpr uint kAttnThreads = 128;

[[kernel, max_total_threads_per_threadgroup(128)]]
void vision_attention_bidir_fp16_mma(
    device const half*  q         [[buffer(0)]],
    device const half*  k         [[buffer(1)]],
    device const half*  v         [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    constant uint&      seq       [[buffer(4)]],
    constant uint&      heads     [[buffer(5)]],
    constant float&     scale     [[buffer(7)]],
    uint2 tg            [[threadgroup_position_in_grid]],
    uint2 lid2          [[thread_position_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint D = FC_VISION_ATTN_HEAD_DIM;
    const uint tiles = (D + 7u) / 8u;
    const uint chunks = tiles * 2u;           // half4 chunks per staged row
    const uint tid = lid2.x;
    const uint head = tg.y;
    const uint stride = heads * D;
    const uint2 fc = vision_gemm_frag_coord(simd_lane_id);
    const uint qrow = tg.x * kAttnBQ + simd_group_id * 8u + fc.x;
    const bool live = qrow < seq;
    // exp(x) == exp2(x * log2(e)); folded into the score scale once.
    const float scale2 = scale * M_LOG2E_F;

    threadgroup half ks[kAttnBK * kAttnLd];
    threadgroup half vs[kAttnBK * kAttnLd];

    // Q straight from device: each lane reads only its own two elements, so
    // the row tail past `seq` and the column tail past `D` are zeros without
    // any staging.
    simdgroup_half8x8 qf[kAttnMaxTiles];
    simdgroup_float8x8 o[kAttnMaxTiles];
    device const half* qp = q + uint64_t(qrow) * stride + head * D;
    for (uint t = 0; t < tiles; ++t) {
        const uint c = t * 8u + fc.y;
        const bool in = live && c < D;
        qf[t].thread_elements()[0] = in ? qp[c] : 0.0h;
        qf[t].thread_elements()[1] = in ? qp[c + 1] : 0.0h;
        o[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float run_max = -INFINITY;
    float run_sum = 0.0f;

    for (uint kb = 0; kb < seq; kb += kAttnBK) {
        // Stage K and V rows [kb, kb + 16), zero past `seq` and past `D`.
        for (uint i = tid; i < kAttnBK * chunks; i += kAttnThreads) {
            const uint r = i / chunks;
            const uint c = (i % chunks) * 4u;
            const uint key = kb + r;
            half4 kv = half4(0.0h);
            half4 vv = half4(0.0h);
            if (key < seq && c < D) {
                const uint64_t off = uint64_t(key) * stride + head * D + c;
                kv = half4(*(device const packed_half4*)(k + off));
                vv = half4(*(device const packed_half4*)(v + off));
            }
            *(threadgroup half4*)(ks + r * kAttnLd + c) = kv;
            *(threadgroup half4*)(vs + r * kAttnLd + c) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s[kAttnBK / 8];
        for (uint j = 0; j < kAttnBK / 8u; ++j) {
            s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint t = 0; t < tiles; ++t) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, ks + (j * 8u) * kAttnLd + t * 8u, kAttnLd,
                               ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qf[t], kt, s[j]);
            }
        }

        // Online softmax over this block, per row, in registers.
        float block_max = -INFINITY;
        for (uint j = 0; j < kAttnBK / 8u; ++j) {
            for (uint e = 0; e < 2u; ++e) {
                const uint key = kb + j * 8u + fc.y + e;
                const float x = key < seq ? s[j].thread_elements()[e] * scale2 : -INFINITY;
                s[j].thread_elements()[e] = x;
                block_max = max(block_max, x);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        // Every block holds at least one key below `seq`, so `new_max` is
        // finite from the first block on, and the first correction is
        // exp2(-inf) == 0 against the zero accumulators.
        const float new_max = max(run_max, block_max);
        const float correction = exp2(run_max - new_max);

        simdgroup_half8x8 p[kAttnBK / 8];
        float block_sum = 0.0f;
        for (uint j = 0; j < kAttnBK / 8u; ++j) {
            for (uint e = 0; e < 2u; ++e) {
                const half w = half(exp2(s[j].thread_elements()[e] - new_max));
                p[j].thread_elements()[e] = w;
                block_sum += float(w);
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = fma(run_sum, correction, block_sum);
        run_max = new_max;

        for (uint t = 0; t < tiles; ++t) {
            o[t].thread_elements()[0] *= correction;
            o[t].thread_elements()[1] *= correction;
        }
        for (uint j = 0; j < kAttnBK / 8u; ++j) {
            for (uint t = 0; t < tiles; ++t) {
                simdgroup_half8x8 vt;
                simdgroup_load(vt, vs + (j * 8u) * kAttnLd + t * 8u, kAttnLd);
                simdgroup_multiply_accumulate(o[t], p[j], vt, o[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!live) { return; }
    const float inv = 1.0f / run_sum;
    device half* op = out + uint64_t(qrow) * stride + head * D;
    for (uint t = 0; t < tiles; ++t) {
        const uint c = t * 8u + fc.y;
        if (c < D) {
            op[c] = half(o[t].thread_elements()[0] * inv);
            op[c + 1] = half(o[t].thread_elements()[1] * inv);
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

// ============================================================================
// The Gemma 4 tower (`docs/GEMMA4_VISION_PHASE0.md`). Same FP16 storage and
// FP32 accumulators as above. Matmul, attention, the tanh GELU and the
// residual add are shared; these three are what that tower adds.
// ============================================================================

// ----------------------------------------------------------------------------
// RMSNorm, `x * rsqrt(mean(x^2) + eps) * w`, with `w` applied DIRECTLY (no
// `1 + w` offset, Gemma 4's own convention) or not at all when `has_weight`
// is 0 (the v norm and the projector's pre-norm carry no weight).
//
// One threadgroup per row, so a per-head norm is the same kernel at
// `rows = seq * heads, D = head_dim`. Safe IN PLACE (`out == x`): the
// reduction finishes behind a barrier before any write, and each thread
// writes only the elements it alone reads in the final loop.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_rms_norm_fp16(
    device const half*  x          [[buffer(0)]],
    device const half*  weight     [[buffer(1)]],
    device half*        out        [[buffer(2)]],
    constant uint&      D          [[buffer(3)]],
    constant float&     eps        [[buffer(4)]],
    constant uint&      has_weight [[buffer(5)]],
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
        const float v = float(xr[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) { partial[0] = rsqrt(v / float(D) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = partial[0];

    for (uint i = lid; i < D; i += lsize) {
        const float w = (has_weight != 0) ? float(weight[i]) : 1.0f;
        yr[i] = half(float(xr[i]) * inv_rms * w);
    }
}

// ----------------------------------------------------------------------------
// The axial 2-D rope. NOT `vision_rope_2d_fp16`: that one pairs element `i`
// with `i + head_dim/2` across the whole head. Here the head is two parts of
// `head_dim/2`, part 0 turned by the patch's x and part 1 by its y, and each
// part rotates WITHIN itself: element `i` pairs with `i + head_dim/4`. Both
// kernels compute plausible output on the other's input.
//
// `freqs` is `[seq, head_dim/2]` F32 angles, the x part's `head_dim/4` angles
// then the y part's, for the reason the kernel above keeps them F32.
// One thread per (pair, head, token).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_rope_axial_fp16(
    device half*        qk        [[buffer(0)]],
    device const float* freqs     [[buffer(1)]],
    constant uint&      seq       [[buffer(2)]],
    constant uint&      heads     [[buffer(3)]],
    constant uint&      head_dim  [[buffer(4)]],
    uint3               gid       [[thread_position_in_grid]]
) {
    const uint pair = gid.x;          // 0 .. head_dim/2
    const uint head = gid.y;
    const uint token = gid.z;
    const uint pairs = head_dim / 2;
    const uint quarter = head_dim / 4;
    if (pair >= pairs || head >= heads || token >= seq) return;

    const uint part = pair / quarter;
    const uint i = pair % quarter;
    const uint a_idx = part * pairs + i;
    const uint b_idx = a_idx + quarter;

    const float angle = freqs[uint64_t(token) * pairs + pair];
    const float c = cos(angle);
    const float s = sin(angle);

    device half* row = qk + (uint64_t(token) * heads + head) * head_dim;
    const float a = float(row[a_idx]);
    const float b = float(row[b_idx]);
    row[a_idx] = half(a * c - b * s);
    row[b_idx] = half(b * c + a * s);
}

// ----------------------------------------------------------------------------
// `y *= x` elementwise, FP32 intermediate: the gated MLP's
// `gelu(gate) * up`, after the GELU has run in place on `gate`.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void vision_mul_fp16(
    device half*        y     [[buffer(0)]],
    device const half*  x     [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint                tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    y[tid] = half(float(y[tid]) * float(x[tid]));
}
