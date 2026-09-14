#include <metal_stdlib>
using namespace metal;

// Z-Image uses a different storage contract from the text runtime. Quantized
// rows are [packed low/high nibbles][BF16 scales][BF16 biases], one row at a
// time. The text runtime kernels expect three contiguous regions, so these
// kernels deliberately decode the image format in place from a mapped
// tensors.bin buffer.

struct LinearParams {
    uint rows;
    uint in_dim;
    uint out_dim;
    uint row_stride;
    uint storage;       // 1 = F32, 2 = BF16, 3 = interleaved INT4 affine
    uint bias_storage;  // 0 = no bias, otherwise 1 or 2
};

struct NormParams {
    uint rows;
    uint dim;
    uint weight_storage;
    float eps;
};

struct RopeParams {
    uint rows;
    uint heads;
    uint dim;
    uint weight_storage;
    float eps;
};

struct AttentionParams {
    uint q_rows;
    uint kv_rows;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint causal;
};

struct ConvParams {
    uint in_channels;
    uint out_channels;
    uint in_height;
    uint in_width;
    uint kernel_size;
    uint stride;
    uint padding;
    uint weight_storage;
    uint bias_storage;
};

struct GroupNormParams {
    uint channels;
    uint height;
    uint width;
    uint groups;
    uint weight_storage;
    uint bias_storage;
    float eps;
};

struct UpsampleParams {
    uint channels;
    uint in_height;
    uint in_width;
};

struct LookupParams {
    uint rows;
    uint dim;
    uint vocab;
    uint storage;
};

inline float bf16_value(device const uchar *ptr) {
    const ushort bits = *reinterpret_cast<device const ushort *>(ptr);
    return as_type<float>(uint(bits) << 16);
}

inline float stored_value(device const uchar *base, uint index, uint storage) {
    if (storage == 1) {
        return reinterpret_cast<device const float *>(base)[index];
    }
    if (storage == 2) {
        return bf16_value(base + uint64_t(index) * 2);
    }
    return 0.0f;
}

inline float affine_int4_value(device const uchar *row, uint col, uint cols) {
    const uchar packed = row[col >> 1];
    const uint nibble = ((col & 1) == 0) ? uint(packed & 0x0f) : uint(packed >> 4);
    const uint group = col >> 6;
    const uint packed_bytes = cols >> 1;
    device const uchar *scale_ptr = row + packed_bytes + uint64_t(group) * 2;
    device const uchar *bias_ptr = row + packed_bytes + uint64_t(cols >> 6) * 2 + uint64_t(group) * 2;
    const float scale = bf16_value(scale_ptr);
    const float bias = bf16_value(bias_ptr);
    return float(nibble) * scale + bias;
}

inline float matrix_value(device const uchar *row, uint col, constant LinearParams &p) {
    if (p.storage == 3) {
        return affine_int4_value(row, col, p.in_dim);
    }
    return stored_value(row, col, p.storage);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_linear(
    device const uchar *weight [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uchar *bias [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.rows * p.out_dim;
    if (gid >= total) return;
    const uint row_index = gid / p.out_dim;
    const uint col = gid - row_index * p.out_dim;
    device const uchar *row = weight + uint64_t(row_index) * p.row_stride;
    float sum = 0.0f;
    for (uint k = 0; k < p.in_dim; ++k) {
        sum = fma(input[uint64_t(row_index) * p.in_dim + k], matrix_value(row, k, p), sum);
    }
    if (p.bias_storage != 0) {
        sum += stored_value(bias, col, p.bias_storage);
    }
    output[gid] = sum;
}

// Input-tiled version of image_linear. One 8x8 output tile shares each 32-wide
// input tile across its 256 threads. The row-major packed contract is unchanged,
// including the per-row INT4 scales and biases.
[[kernel, max_total_threads_per_threadgroup(256)]]
void image_linear_tiled(
    device const uchar *weight [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uchar *bias [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float x_tile[8][32];
    const uint row = group.y * 8 + tid.y;
    const uint col = group.x * 8 + tid.x;
    float sum = 0.0f;
    for (uint k_base = 0; k_base < p.in_dim; k_base += 32) {
        const uint k = k_base + tid.x;
        x_tile[tid.y][tid.x] = (row < p.rows && k < p.in_dim)
            ? input[uint64_t(row) * p.in_dim + k]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < p.rows && col < p.out_dim) {
            device const uchar *weight_row = weight + uint64_t(col) * p.row_stride;
            for (uint offset = 0; offset < 32; ++offset) {
                sum = fma(x_tile[tid.y][offset], matrix_value(weight_row, k_base + offset, p), sum);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.rows && col < p.out_dim) {
        if (p.bias_storage != 0) {
            sum += stored_value(bias, col, p.bias_storage);
        }
        output[uint64_t(row) * p.out_dim + col] = sum;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_rms_norm(
    device const float *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint row = gid / p.dim;
    const uint col = gid - row * p.dim;
    if (row >= p.rows) return;
    device const float *x = input + uint64_t(row) * p.dim;
    float sum = 0.0f;
    for (uint i = 0; i < p.dim; ++i) sum = fma(x[i], x[i], sum);
    const float inv = rsqrt(sum / float(p.dim) + p.eps);
    const float scale = stored_value(weight, col, p.weight_storage);
    output[gid] = x[col] * inv * scale;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_lookup(
    device const uchar *weight [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LookupParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint row = gid / p.dim;
    const uint col = gid - row * p.dim;
    if (row >= p.rows) return;
    const uint token = ids[row];
    if (token >= p.vocab) {
        output[gid] = 0.0f;
        return;
    }
    const uint index = token * p.dim + col;
    output[gid] = stored_value(weight, index, p.storage);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_rope(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const uchar *weight [[buffer(2)]],
    device const float *freqs [[buffer(3)]],
    constant RopeParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.rows * p.heads * p.dim;
    if (gid >= total) return;
    const uint pair_dim = p.dim >> 1;
    const uint col = gid % p.dim;
    const uint token = gid / (p.heads * p.dim);
    const uint pair = col % pair_dim;
    const uint row_base = (gid / p.dim) * p.dim;
    const uint dim_start = row_base;
    float rms = 0.0f;
    for (uint i = 0; i < p.dim; ++i) {
        const float v = input[dim_start + i];
        rms = fma(v, v, rms);
    }
    const float v = input[gid] * rsqrt(rms / float(p.dim) + p.eps)
        * stored_value(weight, col, p.weight_storage);
    const float angle = freqs[uint64_t(token) * pair_dim + pair];
    const float c = cos(angle);
    const float s = sin(angle);
    const float partner = input[row_base + ((col < pair_dim) ? col + pair_dim : col - pair_dim)]
        * rsqrt(rms / float(p.dim) + p.eps)
        * stored_value(weight, (col < pair_dim) ? col + pair_dim : col - pair_dim, p.weight_storage);
    output[gid] = (col < pair_dim) ? v * c - partner * s : v * c + partner * s;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_rope_adjacent(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const uchar *weight [[buffer(2)]],
    device const float *freqs [[buffer(3)]],
    constant RopeParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.rows * p.heads * p.dim;
    if (gid >= total) return;
    const uint pair_dim = p.dim >> 1;
    const uint col = gid % p.dim;
    const uint token = gid / (p.heads * p.dim);
    const uint pair = col >> 1;
    const uint row_base = (gid / p.dim) * p.dim;
    float rms = 0.0f;
    for (uint i = 0; i < p.dim; ++i) {
        const float v = input[row_base + i];
        rms = fma(v, v, rms);
    }
    const float inv = rsqrt(rms / float(p.dim) + p.eps);
    const float value = input[gid] * inv * stored_value(weight, col, p.weight_storage);
    const float partner = input[row_base + ((col & 1) == 0 ? col + 1 : col - 1)] * inv
        * stored_value(weight, ((col & 1) == 0 ? col + 1 : col - 1), p.weight_storage);
    device const float *frequency = freqs + uint64_t(token) * pair_dim * 2 + uint64_t(pair) * 2;
    const float c = frequency[0];
    const float s = frequency[1];
    output[gid] = ((col & 1) == 0) ? value * c - partner * s : partner * c + value * s;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.q_rows * p.q_heads * p.head_dim;
    if (gid >= total) return;
    const uint d = gid % p.head_dim;
    const uint head = (gid / p.head_dim) % p.q_heads;
    const uint query = gid / (p.q_heads * p.head_dim);
    const uint group = p.q_heads / p.kv_heads;
    const uint kv_head = head / group;
    const float inv_scale = rsqrt(float(p.head_dim));
    float maximum = -INFINITY;
    for (uint key = 0; key < p.kv_rows; ++key) {
        if (p.causal != 0 && key > query) continue;
        float score = 0.0f;
        for (uint i = 0; i < p.head_dim; ++i) {
            const float qv = q[(uint64_t(query) * p.q_heads + head) * p.head_dim + i];
            const float kv = k[(uint64_t(key) * p.kv_heads + kv_head) * p.head_dim + i];
            score = fma(qv, kv, score);
        }
        maximum = max(maximum, score * inv_scale);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint key = 0; key < p.kv_rows; ++key) {
        if (p.causal != 0 && key > query) continue;
        float score = 0.0f;
        for (uint i = 0; i < p.head_dim; ++i) {
            const float qv = q[(uint64_t(query) * p.q_heads + head) * p.head_dim + i];
            const float kv = k[(uint64_t(key) * p.kv_heads + kv_head) * p.head_dim + i];
            score = fma(qv, kv, score);
        }
        const float probability = exp(score * inv_scale - maximum);
        denominator += probability;
        numerator += probability * v[(uint64_t(key) * p.kv_heads + kv_head) * p.head_dim + d];
    }
    output[gid] = numerator / denominator;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_add(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = left[gid] + right[gid];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_silu_mul(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        const float x = left[gid];
        output[gid] = (x / (1.0f + exp(-x))) * right[gid];
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_scale_shift(
    device const float *input [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = shape.x * shape.y;
    if (gid < total) output[gid] = input[gid] * (1.0f + scale[gid % shape.y]);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_scheduler_step(
    device const float *sample [[buffer(0)]],
    device const float *velocity [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant float &delta [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = sample[gid] + delta * velocity[gid];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_gate_add(
    device const float *residual [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint2 &shape [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = shape.x * shape.y;
    if (gid < total) output[gid] = residual[gid] + gate[gid % shape.y] * value[gid];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_layer_norm(
    device const float *input [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint row = gid / p.dim;
    const uint col = gid - row * p.dim;
    if (row >= p.rows) return;
    device const float *x = input + uint64_t(row) * p.dim;
    float mean = 0.0f;
    for (uint i = 0; i < p.dim; ++i) mean += x[i];
    mean /= float(p.dim);
    float variance = 0.0f;
    for (uint i = 0; i < p.dim; ++i) {
        const float d = x[i] - mean;
        variance = fma(d, d, variance);
    }
    const float normalized = (x[col] - mean) * rsqrt(variance / float(p.dim) + p.eps);
    output[gid] = normalized * scale[col];
}

// VAE attention keeps its tensors in CHW layout. It is a single-head,
// bidirectional attention and therefore cannot use the text attention
// wrapper's token-major assumptions.
struct VaeAttentionParams {
    uint channels;
    uint height;
    uint width;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_vae_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VaeAttentionParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint area = p.height * p.width;
    const uint total = p.channels * area;
    if (gid >= total) return;
    const uint channel = gid / area;
    const uint spatial = gid - channel * area;
    const float inv_scale = rsqrt(float(p.channels));
    float maximum = -INFINITY;
    for (uint key_spatial = 0; key_spatial < area; ++key_spatial) {
        float score = 0.0f;
        for (uint c = 0; c < p.channels; ++c) {
            score = fma(q[c * area + spatial], k[c * area + key_spatial], score);
        }
        maximum = max(maximum, score * inv_scale);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint key_spatial = 0; key_spatial < area; ++key_spatial) {
        float score = 0.0f;
        for (uint c = 0; c < p.channels; ++c) {
            score = fma(q[c * area + spatial], k[c * area + key_spatial], score);
        }
        const float probability = exp(score * inv_scale - maximum);
        denominator += probability;
        numerator += probability * v[channel * area + key_spatial];
    }
    output[gid] = numerator / denominator;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_conv2d(
    device const uchar *weight [[buffer(0)]],
    device const uchar *bias [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint out_area = p.in_height * p.in_width;
    const uint total = p.out_channels * out_area;
    if (gid >= total) return;
    const uint oc = gid / out_area;
    const uint flat = gid - oc * out_area;
    const uint oy = flat / p.in_width;
    const uint ox = flat - oy * p.in_width;
    float sum = (p.bias_storage == 0) ? 0.0f : stored_value(bias, oc, p.bias_storage);
    const int base_y = int(oy * p.stride) - int(p.padding);
    const int base_x = int(ox * p.stride) - int(p.padding);
    for (uint ic = 0; ic < p.in_channels; ++ic) {
        for (uint ky = 0; ky < p.kernel_size; ++ky) {
            for (uint kx = 0; kx < p.kernel_size; ++kx) {
                const int iy = base_y + int(ky);
                const int ix = base_x + int(kx);
                if (iy < 0 || ix < 0 || iy >= int(p.in_height) || ix >= int(p.in_width)) continue;
                const uint in_index = (ic * p.in_height + uint(iy)) * p.in_width + uint(ix);
                const uint w_index = ((oc * p.in_channels + ic) * p.kernel_size + ky) * p.kernel_size + kx;
                sum = fma(input[in_index], stored_value(weight, w_index, p.weight_storage), sum);
            }
        }
    }
    output[gid] = sum;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_group_norm(
    device const float *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const uchar *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GroupNormParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint area = p.height * p.width;
    const uint channel = gid / area;
    if (channel >= p.channels) return;
    const uint group_size = p.channels / p.groups;
    const uint group = channel / group_size;
    const uint first = group * group_size;
    float mean = 0.0f;
    uint count = group_size * area;
    for (uint c = first; c < first + group_size; ++c) {
        for (uint s = 0; s < area; ++s) mean += input[c * area + s];
    }
    mean /= float(count);
    float variance = 0.0f;
    for (uint c = first; c < first + group_size; ++c) {
        for (uint s = 0; s < area; ++s) {
            const float d = input[c * area + s] - mean;
            variance = fma(d, d, variance);
        }
    }
    const float normalized = (input[gid] - mean) * rsqrt(variance / float(count) + p.eps);
    const float scale = stored_value(weight, channel, p.weight_storage);
    const float shift = stored_value(bias, channel, p.bias_storage);
    output[gid] = normalized * scale + shift;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_upsample_nearest(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant UpsampleParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint out_height = p.in_height * 2;
    const uint out_width = p.in_width * 2;
    const uint area = out_height * out_width;
    const uint channel = gid / area;
    const uint flat = gid - channel * area;
    if (channel >= p.channels) return;
    const uint oy = flat / out_width;
    const uint ox = flat - oy * out_width;
    output[gid] = input[(channel * p.in_height + oy / 2) * p.in_width + ox / 2];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_silu(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        const float x = input[gid];
        output[gid] = x / (1.0f + exp(-x));
    }
}
