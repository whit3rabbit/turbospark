#include <metal_stdlib>
using namespace metal;

// Moonshine's valid, ungrouped channel-major convolution. The source
// checkpoint stores weights as [out_channel, in_channel, kernel].
kernel void moonshine_conv1d_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_len [[buffer(4)]],
    constant uint& output_len [[buffer(5)]],
    constant uint& input_channels [[buffer(6)]],
    constant uint& output_channels [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint index [[thread_position_in_grid]])
{
    uint count = output_channels * output_len;
    if (index >= count) {
        return;
    }
    uint channel = index / output_len;
    uint position = index % output_len;
    float acc = has_bias != 0u ? bias[channel] : 0.0f;
    uint input_base = position * stride;
    for (uint source = 0; source < input_channels; ++source) {
        uint weight_base = (channel * input_channels + source) * kernel_size;
        uint source_base = source * input_len + input_base;
        for (uint tap = 0; tap < kernel_size; ++tap) {
            acc += input[source_base + tap] * weight[weight_base + tap];
        }
    }
    output[index] = acc;
}

kernel void moonshine_tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[index] = tanh(input[index]);
    }
}

// GroupNorm with one group, matching Moonshine's channel-major affine.
// One threadgroup owns the complete normalization so the statistics are
// shared before any thread writes the result.
kernel void moonshine_groupnorm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& sequence [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    threadgroup float scratch[256];
    uint count = channels * sequence;
    float sum = 0.0f;
    for (uint index = lane; index < count; index += lanes) {
        sum += input[index];
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            scratch[lane] += scratch[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(count);
    float squared = 0.0f;
    for (uint index = lane; index < count; index += lanes) {
        float delta = input[index] - mean;
        squared += delta * delta;
    }
    scratch[lane] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = lanes / 2u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            scratch[lane] += scratch[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(count) + epsilon);
    for (uint index = lane; index < count; index += lanes) {
        uint channel = index / sequence;
        output[index] = (input[index] - mean) * inv * weight[channel] + bias[channel];
    }
}

// Moonshine uses interleaved rotary pairs in the leading rotary dimensions
// of each head. The decoder passes one row with a nonzero position offset.
kernel void moonshine_rope_f32(
    device float* values [[buffer(0)]],
    device const float* cos_values [[buffer(1)]],
    device const float* sin_values [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rotary [[buffer(6)]],
    constant uint& position_offset [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    uint half_rotary = rotary / 2u;
    uint count = rows * heads * half_rotary;
    if (index >= count) {
        return;
    }
    uint pair = index % half_rotary;
    uint head = (index / half_rotary) % heads;
    uint row = index / (half_rotary * heads);
    uint value_index = row * heads * head_dim + head * head_dim + pair * 2u;
    uint angle_index = (position_offset + row) * half_rotary + pair;
    float a = values[value_index];
    float b = values[value_index + 1u];
    float c = cos_values[angle_index];
    float s = sin_values[angle_index];
    values[value_index] = a * c - b * s;
    values[value_index + 1u] = b * c + a * s;
}

kernel void moonshine_embed_f32(
    device const float* embedding [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < hidden) {
        output[index] = embedding[token * hidden + index];
    }
}

kernel void moonshine_swiglu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& intermediate [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < intermediate) {
        float gate = input[intermediate + index];
        float silu = gate / (1.0f + exp(-gate));
        output[index] = input[index] * silu;
    }
}
