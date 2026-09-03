#include <metal_stdlib>
using namespace metal;

// ============================================================================
// hyper — `qwen4_exp`'s hyper-connection kernels (`docs/QWEN4_PHASE0.md`
// section 3). PORT-LOCAL: the Swift engine has no architecture with this
// mechanism.
//
// **NOT `HyperConnectionConfig`'s Sinkhorn-normalised mHC.** That is
// DeepSeek-V4-Flash's mechanism -- a different algorithm, with its own
// config fields (`mult`/`sinkhorn_iters`/`eps`) and no kernel anywhere in
// this port. `qwen4_exp`'s mechanism is a low-rank silu/sigmoid mix; the two
// share a name prefix in the config struct and nothing else.
// ============================================================================

// mixed[h] = mean over c in [0, C) of w[c*H+h] * normed[c*H+h]
//
// `w` is the low-rank gate
// (`sigmoid(input_mix_weight_up(silu(input_mix_weight_down(normed) / C)))`,
// two ordinary GEMVs with no kernel of their own) and `normed` is
// `hc_norm`'s output (`rmsnorm_bf16w_grouped_centered`). Both are `[C * H]`,
// viewed as `C` streams of `H` each.
//
// ONE THREAD PER OUTPUT ELEMENT, NO THREADGROUP REDUCTION: `C` is small (4
// for `qwen4_exp`) and both inputs are already materialized, so this is nothing
// like the within-row block reduction `rmsnorm.metal`'s kernels need for a
// `D`-wide (thousands of elements) row. A flat elementwise dispatch, one
// thread owning one `h` and looping over its own `C` strided reads, is the
// whole kernel.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_mix_fp16(
    device const half* w      [[buffer(0)]],   // [C * H] FP16
    device const half* normed [[buffer(1)]],   // [C * H] FP16
    device half*       mixed  [[buffer(2)]],   // [H] FP16
    constant uint&     C      [[buffer(3)]],
    constant uint&     H      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= H) return;
    float acc = 0.0f;
    for (uint c = 0; c < C; c++) {
        uint idx = c * H + tid;
        acc = fma(float(w[idx]), float(normed[idx]), acc);
    }
    mixed[tid] = half(acc / float(C));
}

// hidden[c*H+h] = raw[c*H+h] + out[h] * inject_w[c]
//
// The SCATTER half of a hyper-connection sublayer step: `out` is the
// sublayer's `H`-wide output (attention/GDN or the FFN), `inject_w` is a
// `C`-wide per-stream gate
// (`2 * sigmoid(block_inject_weight(normed) / C)`, another ordinary GEMV
// with no kernel of its own), and `hidden` already holds `raw` -- the
// UN-normalized `C * H`-wide hyper-connection residual, the value from
// BEFORE `hc_norm` ran, never `normed`. In place, like `residual_add_fp16`
// in `utility.metal`: no reduction, one thread per `(c, h)` output element.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_inject_add_fp16(
    device half*        hidden   [[buffer(0)]],   // [C * H] FP16, in/out: holds `raw` on entry
    device const half*  out      [[buffer(1)]],   // [H] FP16
    device const half*  inject_w [[buffer(2)]],   // [C] FP16
    constant uint&      C        [[buffer(3)]],
    constant uint&      H        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total = C * H;
    if (tid >= total) return;
    const uint c = tid / H;
    const uint h = tid % H;
    hidden[tid] = half(float(hidden[tid]) + float(out[h]) * float(inject_w[c]));
}
