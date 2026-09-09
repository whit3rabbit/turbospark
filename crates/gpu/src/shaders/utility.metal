#include <metal_stdlib>
using namespace metal;

// Vendored verbatim from Metal/Primitives/utility.metal, with one
// addition: the Swift build concatenates all shader modules into a single
// library, so utility.metal freely calls gelu_pytorch_tanh defined in
// moe.metal. This port compiles each file as its own library, so that
// helper (and its two constants) is vendored here too, byte-for-byte from
// moe.metal's own definition.

constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(gelu_pytorch_tanh(g) * u);
}

// SwiGLU counterpart for architectures with silu hidden activation (Qwen 3.6).
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half((g / (1.0f + exp(-g))) * u);
}

// Port-local addition: the EXACT-erf GELU as a gated pair,
// out = gelu_erf(gate) * up. Spark-X2.5's MLP gates with the erf form where
// Gemma's blocks above use the tanh form, and the two agree to about 5e-4
// absolute -- close enough that a shared kernel with a mode byte would be
// invisible in every coarse check, and a specialization axis whose byte
// missed the pipeline-cache constants key would silently reuse whichever
// compiled first (crate Gotcha 1). A distinct kernel name is a distinct
// pipeline by construction, the same reasoning vision.metal's two GELUs
// record. Metal ships no `erf` (vision.metal's `vision_erf` comment records
// the trap: the first version that called `erf` failed to COMPILE, it did
// not silently do something else), so the series is written out -- the same
// Abramowitz-Stegun 7.1.26 the CPU reference evaluates, deliberately the
// same approximation rather than a rival one, so the parity bound measures
// the FP32-vs-FP64 evaluation and the FP16 storage and not a gap between two
// expansions of erf. Contract: `turbospark_compute::gating::gelu_erf_mul`.
static inline float utility_erf(float x) {
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
void gelu_erf_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(0.5f * g * (1.0f + utility_erf(g * M_SQRT1_2_F)) * u);
}

// out[i] *= sigmoid(gate[i]) — Qwen 3.6 full-attention output gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_gate_mul_fp16(
    device half*       out  [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    out[tid] = half(float(out[tid]) / (1.0f + exp(-g)));
}

// y[i] *= sigmoid(gate[0]) — Qwen 3.6 shared-expert scalar gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_scalar_mul_fp16(
    device half*       y    [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[0]);
    y[tid] = half(float(y[tid]) / (1.0f + exp(-g)));
}

// Port-local addition: a per-HEAD output gate, out[i] *= sigmoid(gate[i /
// head_dim]). Spark-X2.5 carries ONE gate logit per attention head, scaling
// that head's whole slice of the projection, where sigmoid_gate_mul_fp16
// above gates per ELEMENT (its gate buffer is as long as the row). Neither
// kernel subsumes the other: feeding the elementwise kernel a head-broadcast
// gate would cost a gate buffer head_dim times larger plus a repack of the
// projection output, so the broadcast is folded into the load. The head
// index is the same division every per-head kernel here performs
// (split_q_gate_fp16's `tid / dim`), and the compute/store convention is
// sigmoid_gate_mul_fp16's own (float sigmoid, division form, half store).
// Contract: `turbospark_compute::gating::sigmoid_head_gate_mul`.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_head_gate_mul_fp16(
    device half*       out      [[buffer(0)]],
    device const half* gate     [[buffer(1)]],
    constant uint&     head_dim [[buffer(2)]],
    constant uint&     count    [[buffer(3)]],
    uint               tid      [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid / head_dim]);
    out[tid] = half(float(out[tid]) / (1.0f + exp(-g)));
}

// Qwen 3.6 q_proj emits per-head [query(D) ; gate(D)] pairs. Split them into
// contiguous q [H, D] and gate [H, D] so the per-head norm, RoPE, and
// attention kernels see their usual layout.
[[kernel, max_total_threads_per_threadgroup(256)]]
void split_q_gate_fp16(
    device const half* packed [[buffer(0)]],   // [H, 2*D]
    device half*       q      [[buffer(1)]],   // [H, D]
    device half*       gate   [[buffer(2)]],   // [H, D]
    constant uint&     heads  [[buffer(3)]],
    constant uint&     dim    [[buffer(4)]],
    uint               tid   [[thread_position_in_grid]]
) {
    const uint total = heads * dim;
    if (tid >= total) return;
    const uint h = tid / dim;
    const uint d = tid % dim;
    q[tid] = packed[h * 2u * dim + d];
    gate[tid] = packed[h * 2u * dim + dim + d];
}

// Port-local addition: split a FUSED qkv projection output, [q (q_elems) |
// k (kv_elems) | v (kv_elems)] contiguous, into the three destination
// buffers the attention kernels consume. Spark's `q_k_v_proj` is one weight,
// so what three separate GEMVs used to produce arrives as one row and the
// split moves to the activation side. One thread per output element,
// index-selecting its source range -- three range copies, not a permute, so
// the tests can assert exact bits. Contract:
// `turbospark_compute::gating::split_qkv`.
[[kernel, max_total_threads_per_threadgroup(256)]]
void split_qkv_fp16(
    device const half* src      [[buffer(0)]],  // [q_elems + 2*kv_elems]
    device half*       q        [[buffer(1)]],  // [q_elems]
    device half*       k        [[buffer(2)]],  // [kv_elems]
    device half*       v        [[buffer(3)]],  // [kv_elems]
    constant uint&     q_elems  [[buffer(4)]],
    constant uint&     kv_elems [[buffer(5)]],
    uint               tid      [[thread_position_in_grid]]
) {
    const uint total = q_elems + 2u * kv_elems;
    if (tid >= total) return;
    if (tid < q_elems) {
        q[tid] = src[tid];
    } else if (tid < q_elems + kv_elems) {
        k[tid - q_elems] = src[tid];
    } else {
        v[tid - q_elems - kv_elems] = src[tid];
    }
}

// hidden[i] += delta[i] — plain pre-norm residual add for architectures
// without Gemma's fused sandwich tail.
[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_fp16(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + float(delta[tid]));
}

// Port-local addition (not in the Swift utility.metal): x[i] *= scalar in
// half precision. The Swift original folds this multiply into
// fused_layer_tail's final loop (`hidden4[i] * hScale`, fused.metal),
// which this port does not vendor; the standalone kernel reproduces that
// exact half-precision multiply for the layer_scalar step.
[[kernel, max_total_threads_per_threadgroup(256)]]
void scalar_mul_fp16(
    device half*    x      [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    constant uint&  count  [[buffer(2)]],
    uint            tid    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    x[tid] = x[tid] * half(scalar);
}

// Plain UNARY silu, in place: `x[i] = x[i] * sigmoid(x[i])`. Port-local
// (`qwen4_exp`'s hyper-connection mix, `docs/QWEN4_PHASE0.md` section 3):
// `silu(down_proj(normed) / hc_count)` is a genuine unary activation, not a
// gated pair, so `silu_mul_fp16` (which multiplies TWO buffers) does not
// fit -- every other silu call site in this port is a SwiGLU gate*up pair.
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_fp16(
    device half*   x     [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint           tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]);
    x[tid] = half(v / (1.0f + exp(-v)));
}

// Plain UNARY sigmoid, in place. Port-local, same reason as `silu_fp16`
// above: `sigmoid(input_mix_weight_up(w))` and
// `2 * sigmoid(block_inject_weight(normed) / hc_count)` are both unary,
// where every other sigmoid call site here gates a SECOND buffer.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_fp16(
    device half*   x     [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint           tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]);
    x[tid] = half(1.0f / (1.0f + exp(-v)));
}

// Port-local addition (not in the Swift utility.metal): the softcap half of
// logit.metal's `logit_softcap_softmax`, in place, without the softmax.
//
// The Swift original samples on the GPU from softmaxed probs, so its head
// pairs the cap with a softmax in one kernel. This port samples on the host
// through `selection::select`, which takes LOGITS and runs its own softmax
// -- so the head must hand it the softcapped logits (what HF's
// `Gemma*ForCausalLM.forward` returns) and stop there. Capping with a
// separate softmax pass would double-softmax and flatten the distribution.
//
// FP32 tanh, matching the fused kernel's inline `softcap * tanh(z / softcap)`
// and `mrefrust_compute::logit_softcap_softmax`'s first step.
[[kernel, max_total_threads_per_threadgroup(256)]]
void logit_softcap_fp16(
    device half*    logits  [[buffer(0)]],
    constant float& softcap [[buffer(1)]],
    constant uint&  count   [[buffer(2)]],
    uint            tid     [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    logits[tid] = half(softcap * tanh(float(logits[tid]) / softcap));
}

// Port-local addition (ROADMAP M5): y[i] += bf16(bias[i]).
//
// `gpt-oss` is the first family here whose projections carry BIASES, and
// every one of them arrives as F32 in the GGUF and is narrowed to BF16 by
// `transcode_f32`'s default -- the same width the norms take, read the same
// way, through MSL's native `bfloat`.
//
// A SEPARATE PASS RATHER THAN A BINDING ON EVERY GEMV, deliberately. There
// are seven quant GEMV kernels here plus their resident variants, and a bias
// argument on each would touch every one of them and every existing family's
// dispatch sites, to serve one family. An elementwise add over `[D]` after a
// GEMV that already read `D x N` weights is not a cost worth that: it is one
// more dispatch on a stream the residual add already walks twice.
[[kernel, max_total_threads_per_threadgroup(256)]]
void bias_add_bf16_fp16(
    device half*         y     [[buffer(0)]],
    device const bfloat* bias  [[buffer(1)]],
    constant uint&       count [[buffer(2)]],
    uint                 tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    y[tid] = half(float(y[tid]) + float(bias[tid]));
}

// ============================================================================
// PORT-LOCAL (ROADMAP item 9, in its RUNTIME form): directional steering of a
// residual stream row. Its contract is `turbospark_compute::steering`, which
// is the only definition of what this computes -- the Swift engine has no
// steering surface, so there is no upstream kernel to diff against.
//
//   c     = d . x                       (the raw dot product)
//   c_hat = c * inv_norm                (the coefficient along the UNIT d)
//
//   ablate: x -= alpha * c_hat * d_hat  == x - alpha * c * inv_norm^2 * d
//   add:    x += alpha * d
//   clamp:  x += (target - c_hat) * d_hat
//
// `ablate` at alpha = 1 is exactly `x - d_hat d_hat^T x`, which is the
// operation the weight edit `W - d_hat d_hat^T W` precomputes. Doing it here
// rather than at repack time gives up nothing numerically and gains the two
// things this engine cares about: it writes no weight byte (so a quantized
// install is untouched, and requantizing an edited weight is exactly the
// damage `quality_sensitivity.rs` measures at +10.5% perplexity for 0.0122%
// of expert bytes), and it can be turned off between two generations in one
// process, which is what makes a steered-vs-unsteered A/B possible at all.
//
// THE MODE IS A UNIFORM AND NOT A FUNCTION CONSTANT, deliberately.
// `MetalContext::pipeline`'s function and pipeline caches key on the shader
// source ADDRESS plus an explicit `constants_key`, so a specialization axis
// whose byte does not reach that key silently reuses whichever pipeline
// compiled first (crate Gotcha 1). Here that would make `ablate` and `add`
// the same function on the second dispatch -- a different edit than the one
// asked for, applied silently, producing fluent output either way. A branch
// on a uniform cannot do that. The same reasoning made
// `rmsnorm_bf16w_centered` a separate kernel rather than a flag, and made
// MXFP4's activation a uniform rather than a constant.
//
// ONE DISPATCH, NOT TWO. The dot product and the write could be separate
// kernels, and separating them would double an already dispatch-bound cost:
// this runs once or twice per layer per token, so on a 64-layer model it adds
// 64-128 dispatches against a decode step's ~810. The bytes are negligible
// (one row of `hidden` against a projection's `hidden * 4 * hidden` weights);
// the encode is not.
//
// The block reduce is `rmsnorm.metal`'s `rms_block_inv` shape, not a new one:
// per-SIMD `simd_sum`, partials through threadgroup memory, then one SIMD
// group merges them. That shape reads `simdgroups` at runtime instead of
// hardcoding a loop bound, which is what keeps it correct where `gdn.metal`'s
// two norms are correct at EXACTLY 128 threads and silently wrong otherwise
// (crate Gotcha 5). `partial_c` and `partial_xx` still size for the
// 256-thread maximum the attribute above pins, so the Rust dispatch must not
// widen the threadgroup.
//
// FP32 accumulator, FP16 store, matching `residual_add_fp16` next door. Note
// `ablate` can only ever REDUCE |x| and so cannot overflow, and `renorm`
// restores a magnitude the row already carried so it cannot either, while
// `add` and `clamp` can push an FP16 stream past 65,504 at a large enough alpha; that
// overflow arrives as inf and then as NaN, and NaN reads as a perfect score
// on any rank instrument (AGENTS.md Gotchas 59 and 60). The finiteness check
// belongs to the caller, at the point a measurement is taken.
// ============================================================================

// Threadgroup memory carries at most 256/32 = 8 partial sums per reduction,
// as in rmsnorm.metal. Slot 0 of each is reused after the merge to broadcast
// the total. There are TWO reductions here rather than one, and the second
// costs no extra memory traffic: `x[i]` is already in register for the dot
// product, so `||x||^2` is one more fma per element. Only `renorm` reads it.
constant constexpr uint kSteerMaxSimdGroups = 8;

// These must equal `foundation::SteeringMode::as_u32`. Pinned from the Rust
// side by `steering_mode_codes_are_pinned`, because a reordering here
// swaps two edits that both decode fluently.
constant constexpr uint kSteerModeAblate = 0;
constant constexpr uint kSteerModeAdd    = 1;
constant constexpr uint kSteerModeClamp  = 2;
constant constexpr uint kSteerModeRenorm = 3;

[[kernel, max_total_threads_per_threadgroup(256)]]
void steer_direction_fp16(
    device       half*  x              [[buffer(0)]],  // [rows, row_stride] FP16, in place
    device const half*  d              [[buffer(1)]],  // [D] FP16
    device       float* coeff          [[buffer(2)]],  // [rows] FP32, ALWAYS bound
    constant     uint&  D              [[buffer(3)]],
    constant     uint&  row_stride     [[buffer(4)]],  // ELEMENTS, never bytes
    constant     uint&  mode           [[buffer(5)]],
    constant     float& alpha          [[buffer(6)]],
    constant     float& inv_norm       [[buffer(7)]],  // 1 / ||d||, precomputed
    constant     float& target         [[buffer(8)]],
    constant     float& gate_threshold [[buffer(9)]],
    uint  row              [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_c[kSteerMaxSimdGroups];
    threadgroup float partial_xx[kSteerMaxSimdGroups];
    device half* xr = x + row * row_stride;

    // Both reductions in ONE pass over the row. `xv` is loaded once and used
    // twice, so `||x||^2` adds arithmetic and no memory traffic.
    float acc_c  = 0.0f;
    float acc_xx = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float xv = float(xr[i]);
        acc_c  = fma(xv, float(d[i]), acc_c);
        acc_xx = fma(xv, xv, acc_xx);
    }
    acc_c  = simd_sum(acc_c);
    acc_xx = simd_sum(acc_xx);
    if (simd_lane_id == 0) {
        partial_c[simd_group_id]  = acc_c;
        partial_xx[simd_group_id] = acc_xx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool live = (simd_lane_id < simdgroups);
        float vc  = live ? partial_c[simd_lane_id]  : 0.0f;
        float vxx = live ? partial_xx[simd_lane_id] : 0.0f;
        vc  = simd_sum(vc);
        vxx = simd_sum(vxx);
        if (simd_lane_id == 0) {
            partial_c[0]  = vc;
            partial_xx[0] = vxx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = partial_c[0];
    const float xx = partial_xx[0];
    const float c_hat = c * inv_norm;

    // Reported before the gate and before the edit. After the edit it would
    // measure the parameters rather than the model: ablation drives it to
    // (1 - alpha) * c_hat and clamp drives it to `target`, by construction.
    if (lid == 0) {
        coeff[row] = c_hat;
    }

    // The gate is evaluated HERE and never by a host reading `coeff` back: a
    // host-side gate would cost a command-buffer synchronization per layer
    // per token. Every thread in the threadgroup sees the same uniform
    // operands, so this returns for all of them or none, and no barrier
    // follows it.
    if (gate_threshold > 0.0f && fabs(c_hat) < gate_threshold) {
        return;
    }

    float scale;
    if (mode == kSteerModeAblate || mode == kSteerModeRenorm) {
        scale = -alpha * c * inv_norm * inv_norm;
    } else if (mode == kSteerModeAdd) {
        scale = alpha;
    } else {  // kSteerModeClamp
        scale = (target - c_hat) * inv_norm;
    }

    // `renorm` restores the norm ablation removed. The post-edit norm is
    // ANALYTIC rather than a second reduction over the written row --
    // `||x'||^2 = ||x||^2 - alpha*(2 - alpha)*c_hat^2`, exactly, because the
    // projection is orthogonal. `turbospark_compute::steering::renorm_gamma`
    // is the contract and carries the derivation.
    //
    // NOTE `alpha * (2 - alpha)` equals `alpha` at exactly 1.0, so any test
    // of this at full strength alone cannot see the difference between them.
    //
    // `denom <= 0` means the row lay entirely along `d` and there is no norm
    // left to restore: the identity, never an infinity, which would reach the
    // stream as NaN and read as a PERFECT score on any rank instrument.
    float gamma = 1.0f;
    if (mode == kSteerModeRenorm) {
        const float denom = xx - alpha * (2.0f - alpha) * c_hat * c_hat;
        if (denom > 0.0f) {
            const float g = sqrt(xx / denom);
            gamma = isfinite(g) ? g : 1.0f;
        }
    }

    // ONE write loop for all four modes. `gamma` is exactly 1.0f in the other
    // three and `1.0f * v == v` exactly in IEEE-754, so this leaves them
    // unchanged to the last bit rather than to within a tolerance -- which is
    // what lets a fourth mode land without a second pipeline, without a
    // branch here, and without moving the real-model null control.
    for (uint i = lid; i < D; i += lsize) {
        xr[i] = half(gamma * fma(scale, float(d[i]), float(xr[i])));
    }
}
