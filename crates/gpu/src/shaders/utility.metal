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
