#include <metal_stdlib>
using namespace metal;

// ============================================================================
// whisper_encoder -- f32 kernels for the whisper encoder stack and the
// incremental decoder, the Metal counterpart of compute::whisper's CPU
// reference. Everything is f32 in and out with f32 accumulation, matching
// the reference's precision so parity stays a tolerance question rather
// than a numerics redesign.
//
// Kernels:
//   whisper_matmul_bias_f32   out[m,n] = sum_k a[m,k] * w[n,k] + bias[n]
//                             (w row-major by output, the nn.Linear layout
//                             both checkpoints ship; bias optional; row
//                             strides for head-slice views)
//   whisper_softmax_rows_f32  row softmax over n, max-subtracted
//   whisper_gelu_erf_f32      exact-erf GELU, elementwise
//   whisper_add_f32           out[i] = a[i] + b[i]
//   whisper_transpose_pos_f32 [d, seq] conv output -> [seq, d] plus the
//                             positional row broadcast per position
//   whisper_layer_norm_f32    LayerNorm per row (biased variance, eps)
//   whisper_attn_step_f32     single-query attention over a [len, d] K/V
//                             cache, optional K/V append at `pos` first
//
// Encoder attention reuses the GEMM: per head, scores = Q_head x K_head^T
// is a matmul call (m = seq, k = head_dim, n = seq over head-offset
// buffers with row strides), softmax rows, then out_head = scores x
// V_head. No dedicated encoder attention kernel.
//
// The one-thread-per-row kernels (softmax, layer_norm) require a
// power-of-two threadgroup size; the host dispatches 256.
// ============================================================================

// ---------------------------------------------------------------------------
// Tiled GEMM. Each threadgroup computes a 32 x 32 tile of outputs with one
// thread per output; A and B tiles stream through threadgroup memory in
// BK-wide chunks so the weight bytes are read once per tile row/column
// rather than once per output. Without the staging, seq 1500 x ffn 1536 x
// d 384 re-reads the weight 1500 times (~3.5 GB per projection) and the
// kernel is bandwidth-bound at ~6 ms; with it the whole encoder is
// compute-bound at well under a millisecond on M4-class parts.
//
// a_stride / w_stride / out_stride are the ROW STRIDES of the respective
// operand (elements between consecutive rows). They default to k / k / n
// at the call site and exist for head-slice views: a head's Q/K/V rows
// live `d_model` apart while the reduction walks only `head_dim` of them.
//
// w_transposed selects how the B tile is read. Clear (the nn.Linear
// layout): w[n, k], address n * w_stride + k. Set (the attention value
// mix): w[k, n], address k * w_stride + n -- the mix multiplies the
// score matrix by V, and V's rows are keys, not outputs.
// ---------------------------------------------------------------------------

constant uint WM = 32; // output tile rows (m direction)
constant uint WN = 32; // output tile cols (n direction)
constant uint BK = 16; // reduction chunk staged per iteration

kernel void whisper_matmul_bias_f32(
    device const float* a     [[buffer(0)]],  // [m, a_stride] (>= k used)
    device const float* w     [[buffer(1)]],  // [n, w_stride] or [k, w_stride]
    device const float* bias  [[buffer(2)]],  // [n] or bound-to-a when unused
    device float*       out   [[buffer(3)]],  // [m, out_stride] (>= n used)
    constant uint& m            [[buffer(4)]],
    constant uint& k            [[buffer(5)]],
    constant uint& n            [[buffer(6)]],
    constant uint& a_stride     [[buffer(7)]],
    constant uint& w_stride     [[buffer(8)]],
    constant uint& out_stride   [[buffer(9)]],
    constant uint& has_bias     [[buffer(10)]],
    constant uint& w_transposed [[buffer(11)]],
    constant float& out_scale   [[buffer(12)]],
    uint2 gid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float as[WM][BK];
    threadgroup float bs[BK][WN];

    uint row = group.y * WM + gid.y; // output row in [0, m)
    uint col = group.x * WN + gid.x; // output col in [0, n)

    float acc = 0.0;
    uint tid = gid.y * WN + gid.x; // 0..1023 linear lane
    for (uint k0 = 0; k0 < k; k0 += BK) {
        // Stage A[WM x BK] and B[BK x WN]; both tiles are 512 elements,
        // covered once by the first 512 of the 1024 threads.
        if (tid < WM * BK) {
            uint ar = tid / BK;
            uint ac = tid % BK;
            uint gr = group.y * WM + ar;
            uint gc = k0 + ac;
            as[ar][ac] = (gr < m && gc < k) ? a[gr * a_stride + gc] : 0.0;
        }
        if (tid < BK * WN) {
            uint br = tid / WN;
            uint bc = tid % WN;
            uint gn = group.x * WN + bc;
            uint gc = k0 + br;
            float v = w_transposed != 0u
                ? w[gc * w_stride + gn]
                : w[gn * w_stride + gc];
            bs[br][bc] = (gn < n && gc < k) ? v : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; ++kk) {
            acc += as[gid.y][kk] * bs[kk][gid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        float v = acc * out_scale + (has_bias != 0u ? bias[col] : 0.0);
        out[row * out_stride + col] = v;
    }
}

// ---------------------------------------------------------------------------
// Row softmax over [m, n]: one threadgroup per row, 256 threads, shared-
// memory tree reductions (the dispatch guarantees a power-of-two group).
// Max-subtracted like the reference: encoder scores on real audio reach
// magnitudes where an unshifted exp overflows.
// ---------------------------------------------------------------------------

kernel void whisper_softmax_rows_f32(
    device const float* x [[buffer(0)]], // [m, n]
    device float*       y [[buffer(1)]], // [m, n]
    constant uint& n [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    device const float* row = x + (uint)group * n;
    device float* out_row = y + (uint)group * n;

    float thread_max = -FLT_MAX;
    for (uint i = lane; i < n; i += lanes) {
        thread_max = max(thread_max, row[i]);
    }
    shared[lane] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = lanes / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            shared[lane] = max(shared[lane], shared[lane + off]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_max = shared[0];

    float thread_sum = 0.0;
    for (uint i = lane; i < n; i += lanes) {
        float e = exp(row[i] - m_max);
        out_row[i] = e;
        thread_sum += e;
    }
    shared[lane] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = lanes / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            shared[lane] += shared[lane + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / shared[0];
    for (uint i = lane; i < n; i += lanes) {
        out_row[i] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Exact-erf GELU, elementwise, over `count` f32 values. Same A&S 7.1.26
// erf the CPU reference and the conv kernel compute; evaluated at
// x / sqrt(2) with the scaling before the polynomial.
// ---------------------------------------------------------------------------

kernel void whisper_gelu_erf_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    float v = x[gid];
    float arg = v * 0.70710678118654752440;
    float z = 1.0 / (1.0 + 0.3275911 * abs(arg));
    float poly = z * (0.254829592 + z * (-0.284496736 + z * (1.421413741
        + z * (-1.453152027 + z * 1.061405429))));
    float erf = 1.0 - exp(-arg * arg) * poly;
    erf = arg < 0.0 ? -erf : erf;
    y[gid] = 0.5 * v * (1.0 + erf);
}

// ---------------------------------------------------------------------------
// out[i] = a[i] + b[i], elementwise. Serves residual adds; the positional
// add rides transpose_pos.
// ---------------------------------------------------------------------------

kernel void whisper_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) {
        return;
    }
    y[gid] = a[gid] + b[gid];
}

// ---------------------------------------------------------------------------
// The conv front end emits [d_model, seq] band-major; the encoder stack
// consumes [seq, d_model] rows with the sinusoidal position row added.
// One thread per output element: out[t*d + i] = in[i*seq + t] + pos[t*d+i].
// ---------------------------------------------------------------------------

kernel void whisper_transpose_pos_f32(
    device const float* x [[buffer(0)]],   // [d, seq]
    device const float* pos [[buffer(1)]], // [seq, d]
    device float*       y [[buffer(2)]],   // [seq, d]
    constant uint& seq [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = seq * d;
    if (gid >= total) {
        return;
    }
    uint t = gid / d;
    uint i = gid % d;
    y[gid] = x[i * seq + t] + pos[gid];
}

// ---------------------------------------------------------------------------
// LayerNorm per row of [rows, d]: biased variance, mean subtracted, weight
// and bias applied. One threadgroup per row, 256 threads, shared-memory
// tree reductions. Matches compute::vision::layer_norm's f32 arithmetic
// (biased divide by d, single mean and variance pass each).
// ---------------------------------------------------------------------------

kernel void whisper_layer_norm_f32(
    device const float* x [[buffer(0)]],      // [rows, d]
    device const float* weight [[buffer(1)]], // [d]
    device const float* bias [[buffer(2)]],   // [d]
    device float*       y [[buffer(3)]],      // [rows, d]
    constant uint& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    device const float* row = x + (uint)group * d;
    device float* out_row = y + (uint)group * d;

    float partial = 0.0;
    for (uint i = lane; i < d; i += lanes) {
        partial += row[i];
    }
    shared[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = lanes / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            shared[lane] += shared[lane + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / (float)d;

    float var_partial = 0.0;
    for (uint i = lane; i < d; i += lanes) {
        float diff = row[i] - mean;
        var_partial += diff * diff;
    }
    shared[lane] = var_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = lanes / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            shared[lane] += shared[lane + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = shared[0] / (float)d;
    float inv_std = 1.0 / sqrt(var + eps);

    for (uint i = lane; i < d; i += lanes) {
        out_row[i] = (row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------------------
// Single-query attention over a cached [len_max, d_model] K/V pair, one
// threadgroup per head. Scores live in threadgroup memory (attend_len is
// bounded by MAX_ATTN_STEP, asserted at dispatch), so the softmax needs
// no second pass over the keys and no global scratch -- heads cannot race
// because each head's threadgroup owns its own scratch.
//
// Self-attention step (`append` != 0): the layer's fresh k/v vectors
// [d_model] are written into the cache row `pos` first, then the query
// attends over rows 0..=pos. Cross-attention (`append` == 0): the caches
// hold the encoded window's projected keys/values and the query attends
// over rows 0..seq. `scale` is head_dim^-0.5.
// ---------------------------------------------------------------------------

constant uint MAX_ATTN_STEP = 1536;

kernel void whisper_attn_step_f32(
    device const float* q [[buffer(0)]],      // [d_model] current stream row
    device const float* k_new [[buffer(1)]],  // [d_model] (self) or q again
    device const float* v_new [[buffer(2)]],  // [d_model] (self) or q again
    device float* k_cache [[buffer(3)]],      // [len_max, d_model]
    device float* v_cache [[buffer(4)]],      // [len_max, d_model]
    device float* out [[buffer(5)]],          // [d_model]
    constant uint& pos [[buffer(6)]],         // append row (self) or 0
    constant uint& attend_len [[buffer(7)]],  // rows attended: pos+1 or seq
    constant uint& d_model [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& append [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant uint& v_transposed [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_ATTN_STEP];
    threadgroup float shared[256];

    uint heads = d_model / head_dim;
    uint h = group;
    if (h >= heads || head_dim * heads != d_model) {
        return;
    }
    uint off = h * head_dim;

    if (append != 0u && lane == 0u) {
        for (uint i = 0; i < head_dim; ++i) {
            k_cache[pos * d_model + off + i] = k_new[off + i];
            v_cache[pos * d_model + off + i] = v_new[off + i];
        }
    }
    // Device-memory fence: the append above wrote DEVICE storage and the
    // reads below consume it; mem_threadgroup would fence only threadgroup
    // address space and leave the cache write unordered.
    threadgroup_barrier(mem_flags::mem_device);

    device const float* q_head = q + off;
    float thread_max = -FLT_MAX;
    for (uint t = lane; t < attend_len; t += lanes) {
        device const float* k_row = k_cache + t * d_model + off;
        float dot = 0.0;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * k_row[i];
        }
        float s = dot * scale;
        scores[t] = s;
        thread_max = max(thread_max, s);
    }
    shared[lane] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = lanes / 2u; o > 0u; o >>= 1u) {
        if (lane < o) {
            shared[lane] = max(shared[lane], shared[lane + o]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float thread_sum = 0.0;
    for (uint t = lane; t < attend_len; t += lanes) {
        float e = exp(scores[t] - m_max);
        scores[t] = e;
        thread_sum += e;
    }
    shared[lane] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = lanes / 2u; o > 0u; o >>= 1u) {
        if (lane < o) {
            shared[lane] += shared[lane + o];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / shared[0];

    // Weighted value sum. With v_transposed the value lives at [dim, key]
    // so each lane's inner loop is contiguous; the natural [key, dim]
    // layout strides every read by d_model and measured 77% of a whole
    // decode step. head_dim is often narrower than the threadgroup (64 vs
    // 256), so lanes also split the KEY range into lanes/head_dim blocks
    // and reduce their partials in shared memory -- without the split,
    // three quarters of the lanes idle in the value phase.
    //
    // blocks == 1 when lanes <= head_dim (then every lane strides outputs
    // as before and no reduce is needed).
    uint blocks = lanes / head_dim;
    if (blocks == 0u) {
        blocks = 1u;
    }
    uint block_len = (attend_len + blocks - 1u) / blocks;
    if (blocks == 1u) {
        // One output dim per lane, whole key range.
        for (uint i = lane; i < head_dim; i += lanes) {
            float acc = 0.0;
            if (v_transposed != 0u) {
                device const float* v_row = v_cache + (off + i) * attend_len;
                for (uint t = 0; t < attend_len; ++t) {
                    acc += scores[t] * inv * v_row[t];
                }
            } else {
                for (uint t = 0; t < attend_len; ++t) {
                    acc += scores[t] * inv * v_cache[t * d_model + off + i];
                }
            }
            out[off + i] = acc;
        }
    } else {
        // i = lane % head_dim, block = lane / head_dim: every lane owns
        // one (dim, key-block) partial, reduced in shared memory below.
        for (uint i = lane; i < head_dim * blocks; i += lanes) {
            uint dim = i % head_dim;
            uint blk = i / head_dim;
            float acc = 0.0;
            uint t0 = blk * block_len;
            uint t1 = min(t0 + block_len, attend_len);
            if (v_transposed != 0u) {
                device const float* v_row = v_cache + (off + dim) * attend_len;
                for (uint t = t0; t < t1; ++t) {
                    acc += scores[t] * inv * v_row[t];
                }
            } else {
                for (uint t = t0; t < t1; ++t) {
                    acc += scores[t] * inv * v_cache[t * d_model + off + dim];
                }
            }
            shared[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Reduce the per-block partials: head_dim dims x blocks partials.
        for (uint i = lane; i < head_dim; i += lanes) {
            float acc = 0.0;
            for (uint blk = 0; blk < blocks; ++blk) {
                acc += shared[i + blk * head_dim];
            }
            out[off + i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Single-row GEMV, the decode-loop form of the tiled kernel: one
// simd-group (32 lanes) per output row, lanes striding the reduction.
// No staging and no threadgroup barriers -- the m = 1 case cannot reuse a
// staged tile anyway, and the tiled kernel's chunk barriers dominate its
// runtime here (measured 4.2 ms GPU-busy per decoder step, most of it in
// two dozen barrier-bound GEMVs).
//
// w_transposed: rows of w are outputs in [n, w_stride] layout, or when
// set, outputs index the SECOND axis of a [k, w_stride] layout (the
// attention value mix reading V's rows as keys). out_scale multiplies the
// reduction (head_dim^-0.5 for score rows); bias and residual are each
// optional via flags, with any valid buffer bound in their place.
// ---------------------------------------------------------------------------

kernel void whisper_gemv_f32(
    device const float* a [[buffer(0)]],        // [k] input row
    device const float* w [[buffer(1)]],        // [n, w_stride] or [k, w_stride]
    device const float* bias [[buffer(2)]],     // [n] or bound-to-a when unused
    device const float* residual [[buffer(3)]], // [n] or bound-to-a when unused
    device float*       out [[buffer(4)]],      // [n]
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& w_stride [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    constant uint& has_residual [[buffer(9)]],
    constant uint& w_transposed [[buffer(10)]],
    constant float& out_scale [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    uint row = group;
    if (row >= n) {
        return;
    }
    float acc = 0.0;
    for (uint i = lane; i < k; i += 32u) {
        float wv = w_transposed != 0u ? w[i * w_stride + row] : w[row * w_stride + i];
        acc += a[i] * wv;
    }
    acc *= out_scale;
    // Full-warp reduction: exactly 32 lanes per threadgroup.
    acc += simd_shuffle_xor(acc, 16u);
    acc += simd_shuffle_xor(acc, 8u);
    acc += simd_shuffle_xor(acc, 4u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 1u);
    if (lane == 0u) {
        float v = acc;
        if (has_bias != 0u) {
            v += bias[row];
        }
        if (has_residual != 0u) {
            v += residual[row];
        }
        out[row] = v;
    }
}

// ---------------------------------------------------------------------------
// Matrix transpose [rows, cols] -> [cols, rows], one thread per element.
// Serves the cross-attention value caches: the attention step's weighted
// value sum walks one output dimension across every key, which is
// contiguous in the transposed layout and strided by d_model in the
// natural one -- measured at 77% of the whole decode step before this.
// ---------------------------------------------------------------------------

kernel void whisper_transpose_f32(
    device const float* x [[buffer(0)]], // [rows, cols]
    device float*       y [[buffer(1)]], // [cols, rows]
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = rows * cols;
    if (gid >= total) {
        return;
    }
    uint r = gid / cols;
    uint c = gid % cols;
    y[c * rows + r] = x[gid];
}
