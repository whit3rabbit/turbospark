#include <metal_stdlib>
using namespace metal;

// ============================================================================
// short_conv -- LFM3's gated short convolution (`Lfm3ShortConv`,
// `LiquidAI/LFM3-1.7B` `modeling_lfm3.py`).
//
// PORT-LOCAL: there is no Swift original to diff against. The contract is
// the CPU reference `turbospark_compute::short_conv_reference`
// (`crates/compute/src/short_conv.rs`), checked by
// `crates/gpu/tests/short_conv_parity.rs`.
//
//   BCx     = x @ in_proj                 [3H], split B | C | xx in that order
//   u       = B * xx
//   y[t,ch] = sum_{j<K} w[ch, j] * u[t-(K-1)+j, ch]   causal depthwise, no activation
//   o       = C * y                       then @ out_proj (an ordinary GEMV)
//
// Why not `gdn_conv_mix_*` (`shaders/gdn.metal`): those apply silu to the
// conv output unconditionally, and have no B/C gating, so the input product
// and the output gate would need two extra elementwise dispatches and two
// [T, H] scratch round trips per layer. LFM3's conv has no activation at all.
//
// The tail holds the last K-1 rows of `u` (the PRE-conv product), FP16,
// oldest first, `[K-1, H]`. `u` is rounded to half BEFORE it is convolved,
// in every kernel, so the current row's tap reads the same value a later
// step reads back from the tail. That is what makes prefill over T rows
// bit-identical to T decode steps.
//
// Summation order is shared by decode and prefill, term by term: the
// current row's tap first as a plain product, then the K-1 earlier taps
// oldest first through fma (crates/gpu/AGENTS.md: a decode/verify pair that
// associates differently diverges about once in ten thousand elements).
//
// Weights: BF16 `[H, K]` channel-major (w[ch * K + j], tap K-1 multiplies
// the current row), the checkpoint's own dtype. `taps` is a runtime
// argument, not a function constant, so no specialization enters the
// pipeline cache key. No kernel here keeps a per-tap register array, so no
// tap count is a hidden layout contract.
//
// Dispatch: one thread per (channel[, row]). FP32 accumulate, FP16 in/out.
// ============================================================================

// One row. `in_proj` is that row's [3H] output; `out` gets C * y [H]; the
// tail shifts in place (each thread owns its channel column exclusively).
kernel void short_conv_decode(
    device half*         tail    [[buffer(0)]],   // [K-1, H], shifted in place
    device const half*   in_proj [[buffer(1)]],   // [3H]: B | C | xx
    device const bfloat* conv_w  [[buffer(2)]],   // [H, K]
    device half*         out     [[buffer(3)]],   // [H]
    constant uint&       hidden  [[buffer(4)]],
    constant uint&       taps    [[buffer(5)]],
    uint ch [[thread_position_in_grid]]
) {
    const uint H = hidden;
    const uint K = taps;
    if (ch >= H) return;

    const half u = half(float(in_proj[ch]) * float(in_proj[2u * H + ch]));
    float acc = float(u) * float(conv_w[ch * K + (K - 1u)]);
    for (uint j = 0; j + 1u < K; ++j) {
        acc = fma(float(tail[j * H + ch]), float(conv_w[ch * K + j]), acc);
    }
    out[ch] = half(float(in_proj[H + ch]) * acc);

    for (uint j = 0; j + 2u < K; ++j) {
        tail[j * H + ch] = tail[(j + 1u) * H + ch];
    }
    if (K > 1u) {
        tail[(K - 2u) * H + ch] = u;
    }
}

// T rows at once. The incoming tail is READ-ONLY here;
// `short_conv_tail_update` refreshes it after the chunk. Earlier chunk rows'
// `u` is recomputed from `in_proj` with the same rounding, never read from
// another thread's output.
kernel void short_conv_prefill(
    device const half*   tail    [[buffer(0)]],   // [K-1, H] entering the chunk
    device const half*   in_proj [[buffer(1)]],   // [T, 3H]
    device const bfloat* conv_w  [[buffer(2)]],   // [H, K]
    device half*         out     [[buffer(3)]],   // [T, H]
    constant uint&       hidden  [[buffer(4)]],
    constant uint&       taps    [[buffer(5)]],
    constant uint&       rows    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint H = hidden;
    const uint K = taps;
    const uint ch = gid.x;
    const uint t = gid.y;
    if (ch >= H || t >= rows) return;

    const uint W = 3u * H;
    const uint history = K - 1u;
    const half u = half(float(in_proj[t * W + ch]) * float(in_proj[t * W + 2u * H + ch]));
    float acc = float(u) * float(conv_w[ch * K + history]);
    for (uint j = 0; j < history; ++j) {
        // Index in the virtual sequence [tail rows | chunk rows].
        const int src = int(t) + int(j) - int(history);
        half value;
        if (src >= 0) {
            const uint r = uint(src) * W;
            value = half(float(in_proj[r + ch]) * float(in_proj[r + 2u * H + ch]));
        } else {
            value = tail[uint(int(history) + src) * H + ch];
        }
        acc = fma(float(value), float(conv_w[ch * K + j]), acc);
    }
    out[t * H + ch] = half(float(in_proj[t * W + H + ch]) * acc);
}

// After a prefill chunk: tail := last K-1 rows of [old tail | u(chunk)].
// One thread per channel walks the K-1 rows ascending. New row j reads
// virtual row T + j - (K-1): a chunk row, or old tail row j + T, which is
// at or after j and so not yet overwritten. Keeping the whole column in one
// thread is what makes T < K-1 (an in-place downward shift) race-free.
kernel void short_conv_tail_update(
    device half*         tail    [[buffer(0)]],   // [K-1, H]
    device const half*   in_proj [[buffer(1)]],   // [T, 3H]
    constant uint&       hidden  [[buffer(2)]],
    constant uint&       taps    [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    const uint H = hidden;
    if (ch >= H) return;
    const uint W = 3u * H;
    const uint history = taps - 1u;
    for (uint j = 0; j < history; ++j) {
        const int src = int(rows) + int(j) - int(history);
        if (src >= 0) {
            const uint r = uint(src) * W;
            tail[j * H + ch] = half(float(in_proj[r + ch]) * float(in_proj[r + 2u * H + ch]));
        } else {
            tail[j * H + ch] = tail[uint(int(history) + src) * H + ch];
        }
    }
}
