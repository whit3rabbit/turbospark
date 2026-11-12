#include <metal_stdlib>
using namespace metal;

// NeoX rotary over rows of `heads * head_dim` values (`ops::rope_neox` on
// the CPU). Rows are read with `x_stride` and written with `y_stride`, so
// Q and K can be rotated straight out of the fused `[rows, 3 * d_model]`
// projection. In each head the first `rotary` lanes rotate in two halves,
// `lo' = a * c - b * s` and `hi' = b * c + a * s`, with `a` the low lane,
// `b` its partner `rotary / 2` lanes up, and `c`/`s` from `[rows, rotary /
// 2]` tables. Lanes at or past `rotary` copy through. One thread per
// output element.
kernel void nemotron_rope_neox_f32(
    device const float* x      [[buffer(0)]],
    device const float* cos_t  [[buffer(1)]],
    device const float* sin_t  [[buffer(2)]],
    device float*       y      [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& heads       [[buffer(5)]],
    constant uint& head_dim    [[buffer(6)]],
    constant uint& rotary      [[buffer(7)]],
    constant uint& x_stride    [[buffer(8)]],
    constant uint& y_stride    [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint width = heads * head_dim;
    if (gid >= rows * width) {
        return;
    }
    uint row = gid / width;
    uint col = gid % width;
    uint head = col / head_dim;
    uint lane = col % head_dim;
    uint half_rot = rotary / 2;
    device const float* src = x + row * x_stride + head * head_dim;
    float value;
    if (lane >= rotary) {
        value = src[lane];
    } else if (lane < half_rot) {
        float a = src[lane];
        float b = src[lane + half_rot];
        float c = cos_t[row * half_rot + lane];
        float s = sin_t[row * half_rot + lane];
        value = a * c - b * s;
    } else {
        uint i = lane - half_rot;
        float a = src[i];
        float b = src[lane];
        float c = cos_t[row * half_rot + i];
        float s = sin_t[row * half_rot + i];
        value = b * c + a * s;
    }
    y[row * y_stride + col] = value;
}
