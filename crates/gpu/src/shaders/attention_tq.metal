#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention_tq -- single-token decode attention over TurboQuant-quantized
// K/V rows. Three kernels, all decode-only (M_q = 1): the dense split-KV
// pair (`attention_decode_partial_tq` / `attention_decode_combine_tq`,
// pass 1/2 of attention.metal's own Flash-Decoding scheme) and a THIRD
// pass-1 kernel over an explicit position list
// (`attention_decode_indexed_partial_tq`, `qwen4_exp`'s QSA sparse path --
// see `attention_indexed.metal`), which shares `attention_decode_combine_tq`
// as its pass 2 exactly the way the dense indexed kernel shares the plain
// combine.
//
// New kernel NAMES in a new source file rather than a mode constant on
// `attention.metal` or `attention_indexed.metal`: those FP16 kernels are
// untouched byte for byte, and a distinct name is a distinct pipeline by
// construction (this crate's Gotcha 1).
//
// Contract mirrors `turbospark_compute::kv_quant_attention::causal_attention_tq`
// exactly: the query is rotated with the KEY signs once, scores are
// computed in ROTATED space against each K row's codebook lookup
// (`norm_k * dot(q_rot, codebook[indices])`), values accumulate in ROTATED
// space (`weight * norm_v * codebook[indices]`), and the INVERSE rotation
// (with the VALUE signs) runs ONCE on the merged, un-normalized
// accumulator in `attention_decode_combine_tq` -- never per row, never per
// chunk. See `docs/TRUBOQUANT.md` for the codec itself.
//
// Packed row layout (written by `kv_quantize_tq`), `device const uint*`:
// word 0 is `as_type<uint>(norm)`, words `[1, 1 + packed_words)` are the
// LSB-first packed codebook indices, `packed_words = ceil(head_dim *
// bits / 32)`.
//
// Ring addressing is NOT wired here: TurboQuant only ever quantizes
// FULL-attention layers (`model_io::layer_is_quantized` requires
// `mask_value == 1`), so a quantized layer is never a ring and these
// kernels take no ring-capacity argument -- callers must not reach them for
// a sliding-window layer.
// ============================================================================

constant constexpr uint kTqAttnThreads       = 256;
constant constexpr uint kTqAttnMaxSimdGroups = 8;   // kTqAttnThreads / 32
constant constexpr uint kTqAttnMaxHeadDim    = 512; // matches attention.metal

static inline float tq_attn_exp(float x) {
    return fast::exp(x);
}

// Verbatim shape of attention.metal's `block_reduce_sum`.
inline float tqattn_block_reduce_sum(float v,
                                      uint simd_lane_id,
                                      uint simd_group_id,
                                      uint simdgroups,
                                      threadgroup float* scratch,
                                      threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}

// Cooperative Randomized Hadamard Transform over `dim` threadgroup floats.
// `forward` selects `hadamard(signs * x) / sqrt(dim)` (mlx-vlm's
// `_rht_forward`) or `signs * hadamard(x) / sqrt(dim)` (`_rht_inverse`).
// The butterfly is the natural (Sylvester) order, the SAME linear map as
// `turbospark_compute::wht::wht`'s recursive top-down halving -- the
// Walsh-Hadamard matrix is symmetric, so the two butterfly topologies
// compute identical output for any power-of-two `dim` (checked by hand at
// D=4 in this port's own notes; both reduce to the same four sign
// patterns). `dim` must be a power of two and `<= kTqAttnMaxHeadDim`.
static inline void tq_rht(threadgroup float* buf,
                           device const float* signs,
                           uint dim,
                           bool forward,
                           uint lid,
                           uint lsize) {
    if (forward) {
        for (uint i = lid; i < dim; i += lsize) { buf[i] *= signs[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint h = 1; h < dim; h <<= 1) {
        for (uint i = lid; i < dim / 2; i += lsize) {
            const uint block = i / h;
            const uint offset = i % h;
            const uint idx0 = block * 2u * h + offset;
            const uint idx1 = idx0 + h;
            const float a = buf[idx0];
            const float b = buf[idx1];
            buf[idx0] = a + b;
            buf[idx1] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(float(dim));
    for (uint i = lid; i < dim; i += lsize) { buf[i] *= scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!forward) {
        for (uint i = lid; i < dim; i += lsize) { buf[i] *= signs[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Extracts one `bits`-wide LSB-first packed index from a packed row.
// `words` points at word 0 of the PACKED INDEX RUN (i.e. one past the
// row's leading norm word). Mirrors
// `turbospark_compute::kv_quant::unpack_lsb_first`'s per-index formula.
static inline uint tq_unpack_index(device const uint* words, uint i, uint bits) {
    const uint bit_offset = i * bits;
    const uint word_idx = bit_offset / 32u;
    const uint offset = bit_offset % 32u;
    uint value = words[word_idx] >> offset;
    const int spill = int(offset) + int(bits) - 32;
    if (spill > 0) {
        value |= words[word_idx + 1u] << (bits - uint(spill));
    }
    return value & ((1u << bits) - 1u);
}

// ============================================================================
// Pass 1, dense: attention_decode_partial_tq. Grid = num_q_heads *
// num_chunks, same chunking contract as attention.metal's own partial.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kTqAttnThreads)]]
void attention_decode_partial_tq(
    device const half*  Q             [[buffer(0)]],
    device const uint*  K             [[buffer(1)]],   // [seq_len, num_kv_heads, 1 + k_packed_words]
    device const uint*  V             [[buffer(2)]],   // [seq_len, num_kv_heads, 1 + v_packed_words]
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],   // ROTATED (value-space) partials
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  k_bits        [[buffer(14)]],
    constant     uint&  v_bits        [[buffer(15)]],
    device const float* k_signs       [[buffer(16)]],  // [head_dim]
    device const float* k_codebook    [[buffer(17)]],  // [2^k_bits]
    device const float* v_codebook    [[buffer(18)]],  // [2^v_bits]
    constant     uint&  k_packed_words [[buffer(19)]],
    constant     uint&  v_packed_words [[buffer(20)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kTqAttnMaxHeadDim];
    threadgroup float reduce_scratch[kTqAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = head_dim;
    const uint NQ = num_q_heads;
    const uint NKV = num_kv_heads;
    const uint NC = num_chunks;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The query is rotated with the KEY signs ONCE, up front -- scores are
    // then computed in rotated space against each row's own codebook
    // lookup, never by dequantizing K first.
    tq_rht(q_smem, k_signs, HD, /*forward=*/true, lid, lsize);

    constexpr uint kPerThread = (kTqAttnMaxHeadDim + kTqAttnThreads - 1) / kTqAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    const uint k_row_words = 1u + k_packed_words;
    const uint v_row_words = 1u + v_packed_words;

    for (uint p = p_start; p < p_end; ++p) {
        device const uint* K_row = K + (p * NKV + kv_head) * k_row_words;
        device const uint* V_row = V + (p * NKV + kv_head) * v_row_words;
        const float norm_k = as_type<float>(K_row[0]);
        const float norm_v = as_type<float>(V_row[0]);
        device const uint* K_idx = K_row + 1u;
        device const uint* V_idx = V_row + 1u;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const uint idx = tq_unpack_index(K_idx, i, k_bits);
            partial = fma(q_smem[i], k_codebook[idx], partial);
        }
        float s = tqattn_block_reduce_sum(partial,
                                           simd_lane_id, simd_group_id, simdgroups,
                                           reduce_scratch, &bcast);
        s = s * norm_k * scale;

        const float m_new = max(m_run, s);
        const float alpha = tq_attn_exp(m_run - m_new);
        const float p_exp = tq_attn_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        const float coeff = p_exp * norm_v;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            const uint idx = tq_unpack_index(V_idx, i, v_bits);
            o_local[slot] = o_local[slot] * alpha + coeff * v_codebook[idx];
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// Pass 1, sparse (qwen4_exp QSA): attention_decode_indexed_partial_tq.
// Identical to attention_decode_partial_tq except the loop walks LIST
// INDICES and dereferences `positions` for the physical row, mirroring
// attention_indexed.metal's own relationship to attention.metal's partial.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kTqAttnThreads)]]
void attention_decode_indexed_partial_tq(
    device const half*  Q             [[buffer(0)]],
    device const uint*  K             [[buffer(1)]],
    device const uint*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    device const uint*  positions     [[buffer(9)]],   // [n_sel]
    constant     uint&  n_sel         [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  k_bits        [[buffer(14)]],
    constant     uint&  v_bits        [[buffer(15)]],
    device const float* k_signs       [[buffer(16)]],
    device const float* k_codebook    [[buffer(17)]],
    device const float* v_codebook    [[buffer(18)]],
    constant     uint&  k_packed_words [[buffer(19)]],
    constant     uint&  v_packed_words [[buffer(20)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kTqAttnMaxHeadDim];
    threadgroup float reduce_scratch[kTqAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = head_dim;
    const uint NQ = num_q_heads;
    const uint NKV = num_kv_heads;
    const uint NC = num_chunks;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint i_start = chunk * chunk_len;
    uint i_end = i_start + chunk_len;
    if (i_end > n_sel) { i_end = n_sel; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_rht(q_smem, k_signs, HD, /*forward=*/true, lid, lsize);

    constexpr uint kPerThread = (kTqAttnMaxHeadDim + kTqAttnThreads - 1) / kTqAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    const uint k_row_words = 1u + k_packed_words;
    const uint v_row_words = 1u + v_packed_words;

    for (uint li = i_start; li < i_end; ++li) {
        const uint p = positions[li];
        device const uint* K_row = K + (p * NKV + kv_head) * k_row_words;
        device const uint* V_row = V + (p * NKV + kv_head) * v_row_words;
        const float norm_k = as_type<float>(K_row[0]);
        const float norm_v = as_type<float>(V_row[0]);
        device const uint* K_idx = K_row + 1u;
        device const uint* V_idx = V_row + 1u;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const uint idx = tq_unpack_index(K_idx, i, k_bits);
            partial = fma(q_smem[i], k_codebook[idx], partial);
        }
        float s = tqattn_block_reduce_sum(partial,
                                           simd_lane_id, simd_group_id, simdgroups,
                                           reduce_scratch, &bcast);
        s = s * norm_k * scale;

        const float m_new = max(m_run, s);
        const float alpha = tq_attn_exp(m_run - m_new);
        const float p_exp = tq_attn_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        const float coeff = p_exp * norm_v;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            const uint idx = tq_unpack_index(V_idx, i, v_bits);
            o_local[slot] = o_local[slot] * alpha + coeff * v_codebook[idx];
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// Pass 2 (shared by both pass-1 kernels above): attention_decode_combine_tq.
// Merges chunk partials in ROTATED value space exactly as
// attention_decode_combine does (same max/denominator recurrence, same
// sinks handling), then applies the INVERSE value rotation ONCE on the
// merged row before dividing by D and writing FP16 -- never per chunk,
// never per position.
// ============================================================================

constant bool FC_TQATTN_HAS_SINKS [[function_constant(80)]];

[[kernel, max_total_threads_per_threadgroup(kTqAttnThreads)]]
void attention_decode_combine_tq(
    device const float* m_in         [[buffer(0)]],
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // ROTATED partials
    device       half*  out          [[buffer(3)]],
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    device const bfloat* sinks       [[buffer(6), function_constant(FC_TQATTN_HAS_SINKS)]],
    device const float* v_signs      [[buffer(7)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    threadgroup float o_smem[kTqAttnMaxHeadDim];
    const uint HD = head_dim;
    const uint NC = num_chunks;
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float sink = 0.0f;
    if (FC_TQATTN_HAS_SINKS) {
        sink = float(sinks[q_head]);
        m_glob = max(m_glob, sink);
    }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * tq_attn_exp(m_row[c] - m_glob); }
    if (FC_TQATTN_HAS_SINKS) { D += tq_attn_exp(sink - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * tq_attn_exp(m_row[c] - m_glob);
        }
        o_smem[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ONE inverse rotation on the merged row, with the VALUE signs.
    tq_rht(o_smem, v_signs, HD, /*forward=*/false, lid, lsize);

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        out_row[i] = half(o_smem[i] * inv_d);
    }
}
