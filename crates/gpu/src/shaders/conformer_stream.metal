#include <metal_stdlib>
using namespace metal;

// Kernels for a cache-aware streaming Conformer step (NeMo FastConformer
// with relative-position attention), batched over speakers. Activations
// are time-major rows of `channels` floats. The CPU reference is
// `crates/audio`'s `NemotronEncoder::stream_layers`.

// y = v * (1 / (1 + exp(-v))), the CPU `silu`.
kernel void conformer_silu_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint& count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float v = x[gid];
    y[gid] = v * (1.0f / (1.0f + exp(-v)));
}

// dst[i] = src[i] over `count` floats (cache staging between buffers).
kernel void conformer_copy_f32(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint& count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    dst[gid] = src[gid];
}

// GLU of the pointwise-in rows into each speaker's extended conv input:
// `pw` is `[speakers * rows, 2 * channels]`, `ext` is `[speakers, context
// + rows, channels]`, and row `t` of speaker `b` lands at `context + t`.
// Rows at or past `valid` are zero (NeMo's pad mask). `a / (1 + exp(-g))`
// is the CPU's form.
kernel void conformer_glu_extend_f32(
    device const float* pw   [[buffer(0)]],
    device float*       ext  [[buffer(1)]],
    constant uint& speakers  [[buffer(2)]],
    constant uint& rows      [[buffer(3)]],
    constant uint& valid     [[buffer(4)]],
    constant uint& channels  [[buffer(5)]],
    constant uint& context   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = speakers * rows * channels;
    if (gid >= total) {
        return;
    }
    uint c = gid % channels;
    uint t = (gid / channels) % rows;
    uint b = gid / (channels * rows);
    float value = 0.0f;
    if (t < valid) {
        uint src = (b * rows + t) * 2u * channels;
        float a = pw[src + c];
        float g = pw[src + channels + c];
        value = a / (1.0f + exp(-g));
    }
    ext[(b * (context + rows) + context + t) * channels + c] = value;
}

// Causal depthwise conv over each speaker's `[context + rows, channels]`
// extended input with `kernel = context + 1` taps: `out[b, t, c] =
// bias[c] + sum_k w[c * kernel + k] * ext[b, t + k, c]`, taps ascending,
// starting at the bias or `0` (the CPU conv1d order).
kernel void conformer_depthwise_causal_f32(
    device const float* ext  [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float*       out  [[buffer(3)]],
    constant uint& speakers  [[buffer(4)]],
    constant uint& rows      [[buffer(5)]],
    constant uint& channels  [[buffer(6)]],
    constant uint& kernel_w  [[buffer(7)]],
    constant uint& has_bias  [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = speakers * rows * channels;
    if (gid >= total) {
        return;
    }
    uint c = gid % channels;
    uint t = (gid / channels) % rows;
    uint b = gid / (channels * rows);
    uint width = kernel_w - 1u + rows;
    float acc = has_bias != 0u ? bias[c] : 0.0f;
    for (uint k = 0; k < kernel_w; ++k) {
        acc += ext[(b * width + t + k) * channels + c] * w[c * kernel_w + k];
    }
    out[(b * rows + t) * channels + c] = acc;
}

constant uint MAX_KEYS = 256;
constant uint MAX_HEAD_DIM = 256;

// Cached relative-position attention, one threadgroup per (speaker, head,
// query). `q` is `[speakers * rows, d_model]`; `k`/`v` are `[speakers *
// kv_rows, d_model]` with `kv_rows = attn_frames + rows` (`[cache ;
// chunk]` per speaker); `pos` is the layer's `[2 * window - 1, d_model]`
// projected table. Keys `firsts[b] .. attn_frames + valid` are valid; the
// score of key `j` for query `a` is `((q + bias_u) . k_j + (q + bias_v) .
// pos[window - 1 + j - attn_frames - a]) / sqrt_head_dim`, then a softmax
// and the value mix. Queries at or past `valid` give zero rows.
kernel void conformer_relpos_attn_f32(
    device const float* q      [[buffer(0)]],
    device const float* k      [[buffer(1)]],
    device const float* v      [[buffer(2)]],
    device const float* pos    [[buffer(3)]],
    device const float* bias_u [[buffer(4)]],
    device const float* bias_v [[buffer(5)]],
    device const uint*  firsts [[buffer(6)]],
    device float*       out    [[buffer(7)]],
    constant uint& rows        [[buffer(8)]],
    constant uint& valid       [[buffer(9)]],
    constant uint& heads       [[buffer(10)]],
    constant uint& head_dim    [[buffer(11)]],
    constant uint& attn_frames [[buffer(12)]],
    constant uint& window      [[buffer(13)]],
    constant float& sqrt_head_dim [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup float qu[MAX_HEAD_DIM];
    threadgroup float qv[MAX_HEAD_DIM];
    threadgroup float scores[MAX_KEYS];
    threadgroup float norm;

    uint a = group % rows;
    uint h = (group / rows) % heads;
    uint b = group / (rows * heads);
    uint d_model = heads * head_dim;
    uint kv_rows = attn_frames + rows;
    uint head = h * head_dim;
    device float* dst = out + (b * rows + a) * d_model + head;
    if (a >= valid) {
        for (uint d = tid; d < head_dim; d += threads) {
            dst[d] = 0.0f;
        }
        return;
    }
    device const float* q_row = q + (b * rows + a) * d_model + head;
    for (uint d = tid; d < head_dim; d += threads) {
        qu[d] = q_row[d] + bias_u[head + d];
        qv[d] = q_row[d] + bias_v[head + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint first = firsts[b];
    uint end = attn_frames + valid;
    uint count = end - first;
    for (uint j = tid; j < count; j += threads) {
        uint key = first + j;
        device const float* k_row = k + (b * kv_rows + key) * d_model + head;
        device const float* p_row = pos + (window - 1u + key - attn_frames - a) * d_model + head;
        float content = 0.0f;
        float position = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            content += qu[d] * k_row[d];
            position += qv[d] * p_row[d];
        }
        scores[j] = (content + position) / sqrt_head_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax as `ops::softmax_row`: max, exp(x - max), sum, divide.
    if (tid == 0) {
        float m = -INFINITY;
        for (uint j = 0; j < count; ++j) {
            m = max(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < count; ++j) {
            float e = exp(scores[j] - m);
            scores[j] = e;
            sum += e;
        }
        norm = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = norm;
    for (uint j = tid; j < count; j += threads) {
        if (sum > 0.0f) {
            scores[j] = scores[j] / sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (uint j = 0; j < count; ++j) {
            acc += scores[j] * v[(b * kv_rows + first + j) * d_model + head + d];
        }
        dst[d] = acc;
    }
}
