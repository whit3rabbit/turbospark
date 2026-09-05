#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention_indexed -- single-token decode attention over an EXPLICIT LIST
// of key/value positions. PORT-LOCAL: the Swift engine has no architecture
// with this mechanism. Built for `qwen4_exp`'s QSA (query-sparse attention),
// whose indexer selects `block_topk` blocks of `compress_ratio` tokens plus
// the ragged tail once the context exceeds `indexer_budget`
// (`docs/QWEN4_PHASE0.md` section 5); the host turns that boolean mask into
// the sorted `positions` list this kernel walks.
//
// THIS IS `attention_decode_partial` (attention.metal) WITH ONE CHANGE, AND
// THE CHANGE IS DELIBERATELY THE ONLY ONE. The block reduction, the online
// softmax recurrence, the FP32 accumulators, the per-thread `o_local` slots
// and the `(m, d, o)` partial layout are copied verbatim, so that with the
// identity list (`positions[i] == i`) at the same chunk count this kernel is
// BIT-IDENTICAL to the dense one -- `tests/attention_indexed_parity.rs` pins
// that, which is what lets the dense kernel's real-model verification carry
// over to this one. The change: the loop walks list indices `i` in
// `[i_start, i_end)` and reads K/V row `positions[i]` instead of walking
// positions `p` in `[p_start, p_end)` directly. Pass 2 is the UNCHANGED
// `attention_decode_combine` from attention.metal, dispatched by the host
// over this kernel's partials.
//
// Why not gather the selected rows into a compact buffer and run the dense
// kernel over it: that copies ~2 MiB of K/V per QSA layer per token on the
// real shape (2,051 rows x 2 kv heads x 256 x 2 bytes, K and V) and needs a
// gather kernel anyway. Indexing costs one 8 KiB `uint` list per layer and
// no copy. Why not a from-scratch fused kernel mirroring mlx-vlm: the copy of
// mlx-vlm on this machine has none (it builds a boolean mask and calls dense
// SDPA with it), and new online-softmax code is exactly the correctness risk
// this port's history warns against taking without a reference.
//
// Layout (caller-side contract), matching attention.metal:
//   Q         : [num_q_heads, head_dim]                FP16
//   K, V      : [stored_tokens, num_kv_heads, head_dim] FP16, LINEAR layout
//               (no ring addressing: qwen4_exp has no sliding-window layers)
//   positions : [n_sel] uint, each < stored_tokens
//   m_out/d_out/o_out : the split-KV partials, [num_q_heads * num_chunks (* head_dim)]
//
// Chunking: list indices, not positions. Chunk c owns [c*chunk_len,
// min((c+1)*chunk_len, n_sel)); the host derives chunk_len and num_chunks
// from n_sel exactly as the dense host derives them from the position range.
// ============================================================================

constant constexpr uint kIdxAttnThreads       = 256;
constant constexpr uint kIdxAttnMaxSimdGroups = 8;   // kIdxAttnThreads / 32
constant constexpr uint kIdxAttnMaxHeadDim    = 512; // attention.metal's kAttnMaxHeadDim

static inline float idx_attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Verbatim `block_reduce_sum` from attention.metal.
inline float idx_block_reduce_sum(float v,
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

[[kernel, max_total_threads_per_threadgroup(kIdxAttnThreads)]]
void attention_decode_indexed_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    device const uint*  positions     [[buffer(9)]],   // [n_sel]
    constant     uint&  n_sel         [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kIdxAttnMaxHeadDim];
    threadgroup float reduce_scratch[kIdxAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD  = head_dim;
    const uint NQ  = num_q_heads;
    const uint NKV = num_kv_heads;
    const uint NC  = num_chunks;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    // The ONE difference from attention_decode_partial: the range is over
    // LIST INDICES, and each index dereferences `positions`.
    const uint i_start = chunk * chunk_len;
    uint i_end = i_start + chunk_len;
    if (i_end > n_sel) { i_end = n_sel; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kIdxAttnMaxHeadDim + kIdxAttnThreads - 1) / kIdxAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // i_start can land past the end when num_chunks > n_sel (empty tail
    // chunks); the loop does not execute and the partial is (-inf, 0, 0),
    // which the combine weights to zero via e^{-inf}.
    for (uint i = i_start; i < i_end; ++i) {
        const uint phys_p = positions[i];
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint e = lid; e < HD; e += lsize) {
            partial = fma(q_smem[e], float(K_row[e]), partial);
        }
        float s = idx_block_reduce_sum(partial,
                                       simd_lane_id, simd_group_id, simdgroups,
                                       reduce_scratch, &bcast);
        s *= scale;

        const float m_new = max(m_run, s);
        const float alpha = idx_attn_softmax_exp(m_run - m_new);
        const float p_exp = idx_attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint e = lid; e < HD; e += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[e]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint e = lid; e < HD; e += lsize) {
        o_row[e] = o_local[slot];
        slot += 1;
    }
}
