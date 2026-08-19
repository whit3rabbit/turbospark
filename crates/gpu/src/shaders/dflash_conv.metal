#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dflash_grouped_conv — DFlash2's grouped dynamic depthwise convolution
// (`docs/DFLASH2.md`), applied around every attention and MLP sublayer of
// the drafter's blocks.
//
//   out[i, c] = sum_t (base[side, t, c] + delta[i, t, g(c)]) * x[i - t, c]
//
// where `i` is the row's index within the draft block (row 0 is the bonus
// row), `t < taps` (2 in the published checkpoint), the sum's term is ZERO
// when `i < t` (a tap cannot read before the block starts), and `g(c) =
// c / group_size` (16) is the channel's group, of which hidden/16 = 320
// partition the channel axis. The coefficient is shared by every channel
// in a group: `base` is per-channel (broadcast across the group), `delta`
// is per-group.
//
// `delta` is a slice of the kernel_projection GEMM's output. That GEMM
// emits `[rows, 2 sides * taps * groups]`; `prepare` consumes side 0 and
// `finish` side 1, so the caller binds the WHOLE output and passes its row
// stride (`delta_row_stride`, 2 * taps * groups) plus the side's column
// base (`delta_col_base`, side * taps * groups) rather than copying the
// slice out.
//
// `out_scale` multiplies the result on the way out. It exists for the
// `finish` call sites, whose output is what a drafter layer ADDS to its
// residual stream, and it is how this port holds that stream in FP16 at
// all: this drafter's residual reaches ~1e5, over FP16's 65504 ceiling,
// where the reference's BF16 has range to spare. Scaling by a power of two
// is exact (it shifts the exponent and leaves the mantissa alone) and the
// scale then cancels, because `x` is read only by RMS norms and RMS norm is
// scale-invariant. See `DFLASH_RESIDUAL_SCALE`.
//
// All other parameters are runtime arguments, not function constants: the
// kernel is dispatched for a handful of rows x 5120 elements per round,
// which is elementwise-kernel territory, and there is no specialization axis
// whose pipeline-cache key could then be forgotten (crate Gotcha 1's trap).
//
// Dispatch: one thread per output element, 256 per threadgroup. FP32
// accumulate, FP16 in and out, BF16 base taps.
// ============================================================================

kernel void dflash_grouped_conv_fp16(
    device const half*   x     [[buffer(0)]],  // [rows * hidden], row-major
    device const half*   delta [[buffer(1)]],  // projection output, row stride delta_row_stride
    device const bfloat* base  [[buffer(2)]],  // [2 sides * taps * hidden]
    device half*         out   [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& hidden           [[buffer(5)]],
    constant uint& taps             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant uint& delta_row_stride [[buffer(8)]],
    constant uint& delta_col_base   [[buffer(9)]],
    constant uint& side             [[buffer(10)]],
    constant float& out_scale       [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * hidden) {
        return;
    }
    const uint i = gid / hidden;
    const uint c = gid % hidden;
    const uint groups = hidden / group_size;
    const uint g = c / group_size;
    float acc = 0.0f;
    for (uint t = 0; t < taps; ++t) {
        if (i < t) {
            break;  // later taps only reach further back; nothing to read
        }
        const float coeff =
            float(base[(side * taps + t) * hidden + c]) +
            float(delta[i * delta_row_stride + delta_col_base + t * groups + g]);
        acc += coeff * float(x[(i - t) * hidden + c]);
    }
    out[gid] = half(acc * out_scale);
}

// ============================================================================
// copy_strided_rows — a strided row gather, and the whole of the DFlash2
// aux-state CAPTURE (`docs/DFLASH2.md`). After trunk layers 5/19/33/47/61
// complete, the residual stream's rows are copied into the capture buffer
// at `[row][aux][hidden]`, whose layout is deliberately the fc projection's
// own input layout: five aux states, concatenated, exactly as `fc` reads
// them. One kernel per tapped layer per trunk pass; the per-token form is
// the same dispatch with `rows = 1`.
// ============================================================================

kernel void copy_strided_rows_fp16(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint& rows      [[buffer(2)]],
    constant uint& hidden    [[buffer(3)]],
    constant uint& src_stride [[buffer(4)]],
    constant uint& dst_stride [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows * hidden) {
        return;
    }
    const uint r = gid / hidden;
    const uint c = gid % hidden;
    dst[r * dst_stride + c] = src[r * src_stride + c];
}
