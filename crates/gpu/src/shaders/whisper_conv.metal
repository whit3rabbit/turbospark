#include <metal_stdlib>
using namespace metal;

// ============================================================================
// whisper_conv — 1-D convolution, kernel 3, symmetric zero padding, GELU.
//
//   out[c, t'] = gelu(sum_{ic, k} in[ic, t'*stride + k - pad]
//                            * w[c, ic, k] + bias[c])
//
// This is the whisper front end's conv1 (stride 1, pad 1) and conv2
// (stride 2, pad 1): 3000 mel frames in, 1500 encoder positions out.
// Weights arrive [out, in, 3] flat -- the layout both the HF and the MLX
// checkpoints transpose to at load -- and the activation is exact-erf
// GELU, matching the CPU reference in compute::whisper.
//
// Dispatch: one thread per output element (channel, time), linear over
// out_ch * out_t with a bounds check; the shapes here are small (384 x
// 1500 worst case) so the simple mapping beats any tiling.
// ============================================================================

kernel void whisper_conv1d3_gelu(
    device const float* input    [[buffer(0)]],  // [in_ch, t_len]
    device const float* weight   [[buffer(1)]],  // [out_ch, in_ch, 3]
    device const float* bias     [[buffer(2)]],  // [out_ch]
    device float*       out      [[buffer(3)]],  // [out_ch, out_t]
    constant uint& in_ch         [[buffer(4)]],
    constant uint& t_len         [[buffer(5)]],
    constant uint& out_ch        [[buffer(6)]],
    constant uint& stride        [[buffer(7)]],
    constant uint& pad           [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint out_t = (t_len + 2u * pad - 3u) / stride + 1u;
    uint total = out_ch * out_t;
    if (gid >= total) {
        return;
    }
    uint c = gid / out_t;
    uint t = gid % out_t;

    float acc = 0.0;
    for (uint ic = 0; ic < in_ch; ++ic) {
        for (uint k = 0; k < 3u; ++k) {
            int src = int(t * stride + k) - int(pad);
            float v = 0.0;
            if (src >= 0 && src < int(t_len)) {
                v = input[ic * t_len + uint(src)];
            }
            acc += v * weight[(c * in_ch + ic) * 3u + k];
        }
    }
    acc += bias[c];

    // GELU with the SAME erf the CPU reference computes: Abramowitz and
    // Stegun 7.1.26 (stated bound 1.5e-7, odd symmetry). precise::erf
    // does not exist in this MSL version, so the reference's own
    // approximation is ported verbatim (f32 instead of f64; the parity
    // test bounds the difference).
    float x = acc;
    // GELU evaluates erf at x / sqrt(2); the A&S poly approximates erf
    // of its own argument, so the scaling happens BEFORE the polynomial.
    float arg = x * 0.70710678118654752440;
    float z = 1.0 / (1.0 + 0.3275911 * abs(arg));
    float poly = z * (0.254829592 + z * (-0.284496736 + z * (1.421413741
        + z * (-1.453152027 + z * 1.061405429))));
    float erf = 1.0 - exp(-arg * arg) * poly;
    erf = arg < 0.0 ? -erf : erf;
    out[gid] = 0.5 * x * (1.0 + erf);
}
