#include <metal_stdlib>
using namespace metal;

inline float image_round_bf16_value(float value) {
    uint bits = as_type<uint>(value);
    uint rounding = 0x7fff + ((bits >> 16) & 1);
    return as_type<float>((bits + rounding) & 0xffff0000);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void image_round_bf16(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = image_round_bf16_value(input[index]);
    }
}

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

// Input-tiled version of image_linear. One 32x8 output tile shares each 32-wide
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
    const uint col = group.x * 32 + tid.x;
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
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float *x = input + uint64_t(row) * p.dim;
    float sum = 0.0f;
    const uint chunk = (p.dim + 255) / 256;
    const uint start = tid * chunk;
    const uint end = min(start + chunk, p.dim);
    for (uint i = start; i < end; ++i) sum = fma(x[i], x[i], sum);
    threadgroup float partials[256];
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = partials[0];
    const float inv = rsqrt(sum / float(p.dim) + p.eps);
    for (uint col = tid; col < p.dim; col += 256) {
        const float scale = stored_value(weight, col, p.weight_storage);
        output[uint64_t(row) * p.dim + col] = x[col] * inv * scale;
    }
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
void image_rope_orthogonal(
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
    uint tid [[thread_position_in_threadgroup]],
    uint head_row [[threadgroup_position_in_grid]]) {
    const uint total_rows = p.rows * p.heads;
    if (head_row >= total_rows) return;
    const uint pair_dim = p.dim >> 1;
    const uint token = head_row / p.heads;
    const uint row_base = head_row * p.dim;
    device const float *x = input + row_base;
    float rms = 0.0f;
    const uint chunk = (p.dim + 255) / 256;
    const uint start = tid * chunk;
    const uint end = min(start + chunk, p.dim);
    for (uint i = start; i < end; ++i) rms = fma(x[i], x[i], rms);
    threadgroup float partials[256];
    partials[tid] = rms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    rms = partials[0];
    const float inv = rsqrt(rms / float(p.dim) + p.eps);
    for (uint col = tid; col < p.dim; col += 256) {
        const uint pair = col >> 1;
        const float value = x[col] * inv * stored_value(weight, col, p.weight_storage);
        const uint partner_col = (col & 1) == 0 ? col + 1 : col - 1;
        const float partner = x[partner_col] * inv
            * stored_value(weight, partner_col, p.weight_storage);
        device const float *frequency =
            freqs + uint64_t(token) * pair_dim * 2 + uint64_t(pair) * 2;
        const float c = frequency[0];
        const float s = frequency[1];
        output[uint64_t(row_base) + col] =
            ((col & 1) == 0) ? value * c - partner * s : partner * s + value * c;
    }
}

// Four queries for one head share each key vector. The old elementwise kernel
// recomputed every dot product once per output element, while the first
// grouped version still loaded each key vector once per query. Keeping four
// query lanes in one group cuts that dominant global-memory read by four.
[[kernel, max_total_threads_per_threadgroup(512)]]
void image_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup float key_shared[128];
    threadgroup float partials[4][4];
    threadgroup float scores[4];
    threadgroup float running_max[4];
    threadgroup float denominator[4];
    threadgroup float rescale[4];
    threadgroup float probability[4];

    const uint query_blocks = (p.q_rows + 3) / 4;
    const uint head = group_id / query_blocks;
    const uint query_block = group_id - head * query_blocks;
    const uint group_size = p.q_heads / p.kv_heads;
    const uint kv_head = head / group_size;
    const uint query_slot = lane / p.head_dim;
    const uint local_lane = lane - query_slot * p.head_dim;
    const uint local_simd_group = local_lane / 32;
    const uint query = query_block * 4 + query_slot;
    const bool active = query < p.q_rows;
    const uint64_t q_base = (uint64_t(query) * p.q_heads + head) * p.head_dim;
    const float q_value = active ? q[q_base + local_lane] : 0.0f;
    const float inv_scale = rsqrt(float(p.head_dim));

    if (local_lane == 0) {
        running_max[query_slot] = -INFINITY;
        denominator[query_slot] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float numerator = 0.0f;
    for (uint key = 0; key < p.kv_rows; ++key) {
        if (lane < p.head_dim) {
            const uint64_t k_base = (uint64_t(key) * p.kv_heads + kv_head) * p.head_dim;
            key_shared[lane] = k[k_base + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool allowed = active && (p.causal == 0 || key <= query);
        const float partial = allowed ? q_value * key_shared[local_lane] : 0.0f;
        const float reduced = simd_sum(partial);
        if (simd_lane == 0) partials[query_slot][local_simd_group] = reduced;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_simd_group == 0) {
            const uint groups_per_query = p.head_dim / 32;
            float merged = simd_lane < groups_per_query ? partials[query_slot][simd_lane] : 0.0f;
            merged = simd_sum(merged) * inv_scale;
            if (simd_lane == 0) scores[query_slot] = merged;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_lane == 0) {
            if (allowed) {
                const float next_max = max(running_max[query_slot], scores[query_slot]);
                rescale[query_slot] = exp(running_max[query_slot] - next_max);
                probability[query_slot] = exp(scores[query_slot] - next_max);
                running_max[query_slot] = next_max;
                denominator[query_slot] =
                    denominator[query_slot] * rescale[query_slot] + probability[query_slot];
            } else {
                rescale[query_slot] = 1.0f;
                probability[query_slot] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (allowed) {
            const uint64_t v_base = (uint64_t(key) * p.kv_heads + kv_head) * p.head_dim;
            numerator = numerator * rescale[query_slot] + probability[query_slot] * v[v_base + local_lane];
        } else {
            numerator *= rescale[query_slot];
        }
    }

    if (active) {
        output[(uint64_t(query) * p.q_heads + head) * p.head_dim + local_lane] =
            numerator / denominator[query_slot];
    }
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

// One threadgroup owns one query position. The previous elementwise launch
// recomputed every query/key dot product once per output channel, multiplying
// the production-shape work by another 512x. Scores are tiled through shared
// memory so each query/key dot is computed once and reused by all channels.
[[kernel, max_total_threads_per_threadgroup(256)]]
void image_vae_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VaeAttentionParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    const uint area = p.height * p.width;
    if (query >= area) return;

    threadgroup float scores[256];
    threadgroup float maximum;
    const float inv_scale = rsqrt(float(p.channels));

    // Match the reference's two-pass softmax order while sharing the score
    // calculation. The barrier at the end of each tile protects scores from
    // being overwritten while another lane is still consuming them.
    maximum = -INFINITY;
    for (uint tile = 0; tile < area; tile += 256) {
        const uint key = tile + tid;
        float score = -INFINITY;
        if (key < area) {
            score = 0.0f;
            for (uint c = 0; c < p.channels; ++c) {
                score = fma(q[c * area + query], k[c * area + key], score);
            }
            score *= inv_scale;
        }
        scores[tid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint tile_end = min(tile + 256, area);
            for (uint index = 0; index < tile_end - tile; ++index) {
                maximum = max(maximum, scores[index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float denominator = 0.0f;
    float numerator0 = 0.0f;
    float numerator1 = 0.0f;
    const uint channel0 = tid;
    const uint channel1 = tid + 256;
    for (uint tile = 0; tile < area; tile += 256) {
        const uint key = tile + tid;
        float score = -INFINITY;
        if (key < area) {
            score = 0.0f;
            for (uint c = 0; c < p.channels; ++c) {
                score = fma(q[c * area + query], k[c * area + key], score);
            }
            score *= inv_scale;
        }
        scores[tid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint tile_end = min(tile + 256, area);
        for (uint index = 0; index < tile_end - tile; ++index) {
            const uint key_index = tile + index;
            const float probability = exp(scores[index] - maximum);
            denominator += probability;
            if (channel0 < p.channels) {
                numerator0 = fma(probability, v[channel0 * area + key_index], numerator0);
            }
            if (channel1 < p.channels) {
                numerator1 = fma(probability, v[channel1 * area + key_index], numerator1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (channel0 < p.channels) {
        output[channel0 * area + query] = numerator0 / denominator;
    }
    if (channel1 < p.channels) {
        output[channel1 * area + query] = numerator1 / denominator;
    }
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
