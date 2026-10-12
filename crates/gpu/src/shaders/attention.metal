#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];
// ROADMAP M5: whether this dispatch has ATTENTION SINKS, one learned logit
// per q head added to the softmax DENOMINATOR and to nothing else. A
// function constant rather than a runtime flag so the `sinks` argument does
// not exist at all for the four families that have none -- an unbound buffer
// is undefined behaviour, and a dummy buffer would mean touching every one
// of their dispatch sites. Its byte is in `attention_constants_key`, without
// which a sink dispatch would silently reuse the sink-free pipeline.
constant bool FC_ATTN_HAS_SINKS [[function_constant(70)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
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


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
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
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

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

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
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
// GQA-shared one-row partial pass (attention_decode_gqa_partial).
//
// attention_decode_partial above runs one threadgroup per (Q head, chunk) and
// pays two threadgroup barriers per KV position, so a threadgroup walks its
// chunk at memory latency and every K/V row is read once per Q head of its
// group. At Qwen3.8's shape (24 Q over 4 KV heads, head_dim 256) that is
// latency-bound from a few hundred positions up.
//
// Here one threadgroup serves EVERY Q head of one KV head over one chunk, so
// each K/V row is loaded once for the whole group (the layout open-jet's CUDA
// decoder uses). Simdgroups take interleaved positions and run independent
// online-softmax states, so the position loop has no threadgroup barrier;
// the per-simdgroup states merge once at the end. Lane l holds elements
// [l * PER, l * PER + PER) of each row, PER = head_dim / 32 (4 or 8).
//
// Grid = num_kv_heads * NC threadgroups of FC_ATTN_GQA_SIMDGROUPS * 32
// threads. Output is the same un-normalized (m, d, o) partial layout as
// attention_decode_partial, indexed by Q head and chunk, so
// attention_decode_combine (sinks included) is shared unchanged. Summation
// order differs from the barrier kernel (per-lane then simd_sum, and
// positions split across simdgroups), so results are close, not identical.
// ============================================================================

constant uint FC_ATTN_GQA_GROUP      [[function_constant(71)]];
constant uint FC_ATTN_GQA_SIMDGROUPS [[function_constant(72)]];
constant uint FC_ATTN_GQA_HEAD_DIM   [[function_constant(73)]];
// Positions each simdgroup loads per iteration, 1..=kGqaMaxUnroll.
constant uint FC_ATTN_GQA_UNROLL     [[function_constant(74)]];

constant constexpr uint kGqaMaxPerLane   = 8;   // head_dim <= 256
constant constexpr uint kGqaMaxGroup     = 8;
constant constexpr uint kGqaMaxSimdgroups = 16;
constant constexpr uint kGqaMaxUnroll = 4;

[[kernel, max_total_threads_per_threadgroup(kGqaMaxSimdgroups * 32)]]
void attention_decode_gqa_partial(
    device const half*  Q             [[buffer(0)]],   // [num_q_heads, head_dim]
    device const half*  K             [[buffer(1)]],   // [position, num_kv_heads, head_dim]
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * NC]
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * NC * head_dim]
    constant     uint&  num_kv_heads  [[buffer(6)]],
    constant     uint&  seq_len       [[buffer(7)]],
    constant     uint&  kv_start      [[buffer(8)]],
    constant     uint&  chunk_len     [[buffer(9)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float tg_m[kGqaMaxSimdgroups * kGqaMaxGroup];
    threadgroup float tg_d[kGqaMaxSimdgroups * kGqaMaxGroup];
    threadgroup float tg_o[kGqaMaxSimdgroups * kGqaMaxPerLane * 32];

    const uint HD  = FC_ATTN_GQA_HEAD_DIM;
    const uint PER = HD / 32;
    const uint G   = FC_ATTN_GQA_GROUP;
    const uint NSG = FC_ATTN_GQA_SIMDGROUPS;
    const uint U   = FC_ATTN_GQA_UNROLL;
    const uint NC  = FC_ATTN_NUM_CHUNKS;
    const uint NKV = num_kv_heads;
    const float scale = FC_ATTN_SCALE;

    const uint kv_head = tg_id / NC;
    const uint chunk   = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    const uint p_end   = min(p_start + chunk_len, seq_len);
    const uint e0      = simd_lane_id * PER;

    // Loops run to the compile-time maxima and test the specialized bound,
    // so they fully unroll and the arrays stay in registers.
    float q[kGqaMaxGroup][kGqaMaxPerLane];
    float o[kGqaMaxGroup][kGqaMaxPerLane];
    float m[kGqaMaxGroup];
    float d[kGqaMaxGroup];
    for (uint g = 0; g < kGqaMaxGroup; ++g) {
        if (g >= G) { break; }
        device const half4* q4 = (device const half4*)(Q + (kv_head * G + g) * HD + e0);
        for (uint j = 0; j < kGqaMaxPerLane / 4; ++j) {
            if (j * 4 >= PER) { break; }
            const float4 qv = float4(q4[j]);
            q[g][4 * j + 0] = qv.x; q[g][4 * j + 1] = qv.y;
            q[g][4 * j + 2] = qv.z; q[g][4 * j + 3] = qv.w;
        }
        for (uint i = 0; i < kGqaMaxPerLane; ++i) { o[g][i] = 0.0f; }
        m[g] = -INFINITY;
        d[g] = 0.0f;
    }

    // U positions per iteration (p, p + NSG, ...) so each
    // simdgroup has that many K/V row loads in flight; positions past the
    // chunk end are masked to zero weight (and read a live row instead).
    for (uint p0 = p_start + simd_group_id; p0 < p_end; p0 += U * NSG) {
        float k[kGqaMaxUnroll][kGqaMaxPerLane];
        float v[kGqaMaxUnroll][kGqaMaxPerLane];
        bool live[kGqaMaxUnroll];
        for (uint u = 0; u < kGqaMaxUnroll; ++u) {
            if (u >= U) { break; }
            const uint p = p0 + u * NSG;
            live[u] = p < p_end;
            const uint row = (attn_ring_slot(live[u] ? p : p0) * NKV + kv_head) * HD + e0;
            device const half4* k4 = (device const half4*)(K + row);
            device const half4* v4 = (device const half4*)(V + row);
            for (uint j = 0; j < kGqaMaxPerLane / 4; ++j) {
                if (j * 4 >= PER) { break; }
                const float4 kv = float4(k4[j]);
                const float4 vv = float4(v4[j]);
                k[u][4 * j + 0] = kv.x; k[u][4 * j + 1] = kv.y;
                k[u][4 * j + 2] = kv.z; k[u][4 * j + 3] = kv.w;
                v[u][4 * j + 0] = vv.x; v[u][4 * j + 1] = vv.y;
                v[u][4 * j + 2] = vv.z; v[u][4 * j + 3] = vv.w;
            }
        }
        for (uint g = 0; g < kGqaMaxGroup; ++g) {
            if (g >= G) { break; }
            float s[kGqaMaxUnroll];
            float m_new = m[g];
            for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                if (u >= U) { break; }
                float partial = 0.0f;
                for (uint i = 0; i < kGqaMaxPerLane; ++i) {
                    if (i >= PER) { break; }
                    partial = fma(q[g][i], k[u][i], partial);
                }
                s[u] = live[u] ? simd_sum(partial) * scale : -INFINITY;
                m_new = max(m_new, s[u]);
            }
            const float alpha = attn_softmax_exp(m[g] - m_new);
            float e[kGqaMaxUnroll];
            float e_sum = 0.0f;
            for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                if (u >= U) { break; }
                e[u] = live[u] ? attn_softmax_exp(s[u] - m_new) : 0.0f;
                e_sum += e[u];
            }
            d[g] = d[g] * alpha + e_sum;
            for (uint i = 0; i < kGqaMaxPerLane; ++i) {
                if (i >= PER) { break; }
                float acc = o[g][i] * alpha;
                for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                    if (u >= U) { break; }
                    acc = fma(e[u], v[u][i], acc);
                }
                o[g][i] = acc;
            }
            m[g] = m_new;
        }
    }

    // Merge the simdgroup states, one Q head at a time. A simdgroup that saw
    // no position has m = -inf and is weighted to exactly zero; a chunk no
    // simdgroup saw keeps (-inf, 0, 0), the barrier kernel's empty-chunk
    // output, which the combine weights to zero.
    if (simd_lane_id == 0) {
        for (uint g = 0; g < kGqaMaxGroup; ++g) {
            if (g >= G) { break; }
            tg_m[simd_group_id * G + g] = m[g];
            tg_d[simd_group_id * G + g] = d[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < kGqaMaxGroup; ++g) {
        if (g >= G) { break; }
        float m_glob = -INFINITY;
        for (uint s = 0; s < NSG; ++s) { m_glob = max(m_glob, tg_m[s * G + g]); }
        const float f = (m[g] == -INFINITY) ? 0.0f : attn_softmax_exp(m[g] - m_glob);
        for (uint i = 0; i < kGqaMaxPerLane; ++i) {
            if (i >= PER) { break; }
            tg_o[simd_group_id * HD + e0 + i] = o[g][i] * f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint base = (kv_head * G + g) * NC + chunk;
        if (lid == 0) {
            float d_sum = 0.0f;
            for (uint s = 0; s < NSG; ++s) {
                const float ms = tg_m[s * G + g];
                if (ms != -INFINITY) { d_sum += tg_d[s * G + g] * attn_softmax_exp(ms - m_glob); }
            }
            m_out[base] = m_glob;
            d_out[base] = d_sum;
        }
        device float* o_row = o_out + base * HD;
        for (uint i = lid; i < HD; i += NSG * 32) {
            float acc = 0.0f;
            for (uint s = 0; s < NSG; ++s) { acc += tg_o[s * HD + i]; }
            o_row[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// GQA-shared multi-row partial pass (attention_decode_gqa_rows_partial).
//
// A prefill micro-batch's query rows sit at consecutive positions: row r
// attends over [kv_start, first_end + r), so the rows share all but their
// last few positions. Calling attention_decode_gqa_partial once per row costs
// two serial dispatches per row, each too small to fill the GPU, and reads
// the shared range from device memory once per row. Here one threadgroup
// serves RG = NSG / (G / HPS) consecutive rows of one KV head over one
// chunk. Each simdgroup owns HPS of one row's G Q heads and holds their Q
// and O in registers (HPS = G is a one-row simdgroup's layout), and each
// tile of T K/V positions is staged in threadgroup memory once for all of
// them. Fewer heads per simdgroup trade K/V reuse for registers, and at
// Qwen3.8's shape one head per simdgroup and two rows per threadgroup won
// (gqa_rows_launch in attention_decode.rs).
//
// Grid = ceil(live_rows / RG) * num_kv_heads * NC threadgroups of NSG * 32
// threads. Chunk boundaries come from the LAST row's range and are shared by
// every row; each row masks positions at or past its own end. Output is one
// (m, d, o) partial per (row, Q head, chunk), row-major, neutral (-inf, 0, 0)
// where a row sees no position of the chunk, merged by
// attention_decode_batch_combine. Summation order differs from the one-row
// kernel, so results are close, not identical: prefill only, where the arm
// is held to a KL bound rather than to identity.
// ============================================================================

// Positions staged per tile, 1..=kRowsMaxTile.
constant uint FC_ATTN_ROWS_TILE [[function_constant(75)]];
// Q heads per simdgroup, a divisor of the group.
constant uint FC_ATTN_ROWS_HEADS [[function_constant(76)]];

constant constexpr uint kRowsMaxTile = 16;
// One head per simdgroup leaves room for more rows per threadgroup than the
// one-row kernel's 16 simdgroups; 32 is Metal's 1024-thread ceiling.
constant constexpr uint kRowsMaxSimdgroups = 32;

[[kernel, max_total_threads_per_threadgroup(kRowsMaxSimdgroups * 32)]]
void attention_decode_gqa_rows_partial(
    device const half*  Q            [[buffer(0)]],   // [live_rows, num_q_heads, head_dim]
    device const half*  K            [[buffer(1)]],   // [position, num_kv_heads, head_dim]
    device const half*  V            [[buffer(2)]],
    device       float* m_out        [[buffer(3)]],   // [live_rows * num_q_heads * NC]
    device       float* d_out        [[buffer(4)]],
    device       float* o_out        [[buffer(5)]],   // [live_rows * num_q_heads * NC * head_dim]
    constant     uint&  num_kv_heads [[buffer(6)]],
    constant     uint&  first_end    [[buffer(7)]],   // row 0 attends over [kv_start, first_end)
    constant     uint&  kv_start     [[buffer(8)]],
    constant     uint&  chunk_len    [[buffer(9)]],
    constant     uint&  live_rows    [[buffer(10)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half k_tile[kRowsMaxTile * kGqaMaxPerLane * 32];
    threadgroup half v_tile[kRowsMaxTile * kGqaMaxPerLane * 32];

    const uint HD  = FC_ATTN_GQA_HEAD_DIM;
    const uint PER = HD / 32;
    const uint G   = FC_ATTN_GQA_GROUP;
    const uint NSG = FC_ATTN_GQA_SIMDGROUPS;
    const uint U   = FC_ATTN_GQA_UNROLL;
    const uint T   = FC_ATTN_ROWS_TILE;
    const uint HPS = FC_ATTN_ROWS_HEADS;
    const uint SPR = G / HPS;             // simdgroups per row
    const uint RG  = NSG / SPR;           // rows per threadgroup
    const uint NC  = FC_ATTN_NUM_CHUNKS;
    const uint NKV = num_kv_heads;
    const uint NQ  = NKV * G;
    const float scale = FC_ATTN_SCALE;

    const uint chunk    = tg_id % NC;
    const uint kv_head  = (tg_id / NC) % NKV;
    const uint row_base = (tg_id / (NC * NKV)) * RG;
    const uint row      = row_base + simd_group_id / SPR;
    const uint head0    = kv_head * G + (simd_group_id % SPR) * HPS;
    const bool row_live = row < live_rows;
    // The grid holds ceil(live_rows / RG) row groups, so row_base is live.
    const uint last_row = min(row_base + RG, live_rows) - 1;
    const uint row_end  = first_end + row;
    const uint p_start  = kv_start + chunk * chunk_len;
    // The group's last row reaches furthest, so it bounds what is staged.
    const uint p_end    = min(p_start + chunk_len, first_end + last_row);
    const uint e0       = simd_lane_id * PER;

    float q[kGqaMaxGroup][kGqaMaxPerLane];
    float o[kGqaMaxGroup][kGqaMaxPerLane];
    float m[kGqaMaxGroup];
    float d[kGqaMaxGroup];
    for (uint g = 0; g < kGqaMaxGroup; ++g) {
        if (g >= HPS) { break; }
        // A dead row's simdgroup still stages tiles and meets every barrier;
        // it reads row_base's Q so no load leaves the buffer.
        device const half4* q4 = (device const half4*)(
            Q + ((row_live ? row : row_base) * NQ + head0 + g) * HD + e0);
        for (uint j = 0; j < kGqaMaxPerLane / 4; ++j) {
            if (j * 4 >= PER) { break; }
            const float4 qv = float4(q4[j]);
            q[g][4 * j + 0] = qv.x; q[g][4 * j + 1] = qv.y;
            q[g][4 * j + 2] = qv.z; q[g][4 * j + 3] = qv.w;
        }
        for (uint i = 0; i < kGqaMaxPerLane; ++i) { o[g][i] = 0.0f; }
        m[g] = -INFINITY;
        d[g] = 0.0f;
    }

    for (uint t0 = p_start; t0 < p_end; t0 += T) {
        const uint n = min(T, p_end - t0);
        // Stage n positions of this KV head, one half4 per thread per step.
        // HD is a multiple of 4, so a half4 never straddles two positions.
        for (uint i = lid * 4; i < n * HD; i += NSG * 32 * 4) {
            const uint t = i / HD;
            const uint src = (attn_ring_slot(t0 + t) * NKV + kv_head) * HD + (i - t * HD);
            *(threadgroup half4*)(k_tile + i) = *(device const half4*)(K + src);
            *(threadgroup half4*)(v_tile + i) = *(device const half4*)(V + src);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile positions [0, n_row) are inside this row's causal range.
        const uint n_row = row_live ? min(n, row_end > t0 ? row_end - t0 : 0u) : 0u;
        for (uint tb = 0; tb < n_row; tb += U) {
            float k[kGqaMaxUnroll][kGqaMaxPerLane];
            float v[kGqaMaxUnroll][kGqaMaxPerLane];
            bool live[kGqaMaxUnroll];
            for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                if (u >= U) { break; }
                const uint t = tb + u;
                live[u] = t < n_row;
                const uint off = (live[u] ? t : tb) * HD + e0;
                threadgroup const half4* k4 = (threadgroup const half4*)(k_tile + off);
                threadgroup const half4* v4 = (threadgroup const half4*)(v_tile + off);
                for (uint j = 0; j < kGqaMaxPerLane / 4; ++j) {
                    if (j * 4 >= PER) { break; }
                    const float4 kv = float4(k4[j]);
                    const float4 vv = float4(v4[j]);
                    k[u][4 * j + 0] = kv.x; k[u][4 * j + 1] = kv.y;
                    k[u][4 * j + 2] = kv.z; k[u][4 * j + 3] = kv.w;
                    v[u][4 * j + 0] = vv.x; v[u][4 * j + 1] = vv.y;
                    v[u][4 * j + 2] = vv.z; v[u][4 * j + 3] = vv.w;
                }
            }
            for (uint g = 0; g < kGqaMaxGroup; ++g) {
                if (g >= HPS) { break; }
                float s[kGqaMaxUnroll];
                float m_new = m[g];
                for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                    if (u >= U) { break; }
                    float partial = 0.0f;
                    for (uint i = 0; i < kGqaMaxPerLane; ++i) {
                        if (i >= PER) { break; }
                        partial = fma(q[g][i], k[u][i], partial);
                    }
                    s[u] = live[u] ? simd_sum(partial) * scale : -INFINITY;
                    m_new = max(m_new, s[u]);
                }
                const float alpha = attn_softmax_exp(m[g] - m_new);
                float e[kGqaMaxUnroll];
                float e_sum = 0.0f;
                for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                    if (u >= U) { break; }
                    e[u] = live[u] ? attn_softmax_exp(s[u] - m_new) : 0.0f;
                    e_sum += e[u];
                }
                d[g] = d[g] * alpha + e_sum;
                for (uint i = 0; i < kGqaMaxPerLane; ++i) {
                    if (i >= PER) { break; }
                    float acc = o[g][i] * alpha;
                    for (uint u = 0; u < kGqaMaxUnroll; ++u) {
                        if (u >= U) { break; }
                        acc = fma(e[u], v[u][i], acc);
                    }
                    o[g][i] = acc;
                }
                m[g] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After the last barrier, so a dead row's early exit cannot strand one.
    if (!row_live) { return; }
    for (uint g = 0; g < kGqaMaxGroup; ++g) {
        if (g >= HPS) { break; }
        const uint base = (row * NQ + head0 + g) * NC + chunk;
        if (simd_lane_id == 0) { m_out[base] = m[g]; d_out[base] = d[g]; }
        device float* o_row = o_out + base * HD + e0;
        for (uint i = 0; i < kGqaMaxPerLane; ++i) {
            if (i >= PER) { break; }
            o_row[i] = o[g][i];
        }
    }
}

// Batch-only split-KV partial pass. The grid is [ceil(live_rows / 4) * NQ * NC],
// with one four-query tile/head/chunk per threadgroup. Each row keeps an
// independent online-softmax state and reads its own interval from buffer 9;
// K/V positions in the tile's interval union are staged once and reused.
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_batch_partial(
    device const half*  Q             [[buffer(0)]],   // [capacity, num_q_heads, head_dim]
    device const half*  K             [[buffer(1)]],   // [position, num_kv_heads, head_dim]
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [capacity, num_q_heads, NC]
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],   // [capacity, num_q_heads, NC, head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    device const uint4* row_plans     [[buffer(9)]],   // {key_start, seq_len, chunk_len, num_chunks}
    constant     uint&  live_rows     [[buffer(10)]],
    constant     float& scale         [[buffer(11)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[4][kAttnMaxHeadDim];
    threadgroup half k_smem[kAttnMaxHeadDim];
    threadgroup half v_smem[kAttnMaxHeadDim];
    threadgroup uint4 plans_smem[4];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = FC_ATTN_NUM_CHUNKS;

    const uint chunk = tg_id % NC;
    const uint q_head = (tg_id / NC) % NQ;
    const uint tile_start = (tg_id / (NC * NQ)) * 4u;
    const uint kv_head = q_head / (NQ / NKV);

    // Only live rows' plan entries are read. Inactive tile slots receive a
    // threadgroup-local sentinel so they never participate in the interval
    // union or access Q, K, V, or output buffers.
    if (lid < 4u) {
        const uint row = tile_start + lid;
        plans_smem[lid] = row < live_rows ? row_plans[row] : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_start[4];
    uint row_end[4];
    bool row_has_interval[4];
    uint union_start = 0xffffffffu;
    uint union_end = 0u;
    for (uint r = 0; r < 4u; ++r) {
        const uint row = tile_start + r;
        const uint4 plan = plans_smem[r];
        row_start[r] = 0u;
        row_end[r] = 0u;
        row_has_interval[r] = false;
        if (row < live_rows && chunk < plan.w) {
            const uint start = plan.x + chunk * plan.z;
            const uint end = min(plan.y, start + plan.z);
            if (start < end) {
                row_start[r] = start;
                row_end[r] = end;
                row_has_interval[r] = true;
                union_start = min(union_start, start);
                union_end = max(union_end, end);
            }
        }
    }

    // Stage only query rows that own a nonempty interval for this chunk.
    for (uint r = 0; r < 4u; ++r) {
        const uint row = tile_start + r;
        if (row_has_interval[r]) {
            device const half* Q_row = Q + (row * NQ + q_head) * HD;
            for (uint i = lid; i < HD; i += lsize) {
                q_smem[r][i] = float(Q_row[i]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[4][kPerThread];
    float m_run[4];
    float d_run[4];
    for (uint r = 0; r < 4u; ++r) {
        m_run[r] = -INFINITY;
        d_run[r] = 0.0f;
        for (uint k = 0; k < kPerThread; ++k) { o_local[r][k] = 0.0f; }
    }

    // Walk the union in increasing order. The membership check skips gaps
    // between disjoint row intervals, so no K/V position outside that union
    // is loaded. Each in-union position is staged once for all four rows.
    for (uint p = union_start; p < union_end; ++p) {
        bool in_union = false;
        for (uint r = 0; r < 4u; ++r) {
            in_union = in_union ||
                (row_has_interval[r] && p >= row_start[r] && p < row_end[r]);
        }
        if (!in_union) { continue; }

        device const half* K_row = K + (p * NKV + kv_head) * HD;
        device const half* V_row = V + (p * NKV + kv_head) * HD;
        for (uint i = lid; i < HD; i += lsize) {
            k_smem[i] = K_row[i];
            v_smem[i] = V_row[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < 4u; ++r) {
            if (row_has_interval[r] && p >= row_start[r] && p < row_end[r]) {
                float partial = 0.0f;
                for (uint i = lid; i < HD; i += lsize) {
                    partial = fma(q_smem[r][i], float(k_smem[i]), partial);
                }
                float score = block_reduce_sum(partial,
                                               simd_lane_id, simd_group_id, simdgroups,
                                               reduce_scratch, &bcast);
                score *= attn_fc_scale(scale);

                const float m_new = max(m_run[r], score);
                const float alpha = attn_softmax_exp(m_run[r] - m_new);
                const float p_exp = attn_softmax_exp(score - m_new);
                d_run[r] = d_run[r] * alpha + p_exp;

                uint slot = 0;
                for (uint i = lid; i < HD; i += lsize) {
                    o_local[r][slot] = o_local[r][slot] * alpha + p_exp * float(v_smem[i]);
                    slot += 1;
                }
                m_run[r] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every live row owns a slot for this chunk, including neutral chunks.
    // Rows beyond live_rows are never read or written.
    for (uint r = 0; r < 4u; ++r) {
        const uint row = tile_start + r;
        if (row < live_rows) {
            const uint state = (row * NQ + q_head) * NC + chunk;
            const bool neutral = !row_has_interval[r];
            if (lid == 0) {
                m_out[state] = neutral ? -INFINITY : m_run[r];
                d_out[state] = neutral ? 0.0f : d_run[r];
            }
            device float* o_row = o_out + state * HD;
            uint slot = 0;
            for (uint i = lid; i < HD; i += lsize) {
                o_row[i] = neutral ? 0.0f : o_local[r][slot];
                slot += 1;
            }
        }
    }
}

// Merge split-KV partials for a row-major batch. The group index covers
// (row, query head); rows at or beyond live_rows return before reading state.
// Neutral partials (-inf, 0, 0) are skipped so they contribute exactly zero.
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_batch_combine(
    device const float* m_in         [[buffer(0)]], // [rows * num_q_heads * max_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]], // [rows * num_q_heads * max_chunks * head_dim]
    device       half*  out          [[buffer(3)]], // [rows * num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_q_heads  [[buffer(5)]],
    constant     uint&  max_chunks   [[buffer(6)]],
    constant     uint&  live_rows    [[buffer(7)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]) {
    const uint row = tg_id / num_q_heads;
    const uint q_head = tg_id % num_q_heads;
    if (row >= live_rows) { return; }

    const uint HD = head_dim;
    const uint NC = attn_fc_num_chunks(max_chunks);
    const uint head_index = row * num_q_heads + q_head;
    const uint state_base = head_index * NC;
    device const float* m_row = m_in + state_base;
    device const float* d_row = d_in + state_base;
    device const float* o_base = o_in + state_base * HD;

    // The chunk count is small. Recompute the same row/head reduction per
    // thread so the combine needs no shared memory or threadgroup barriers.
    float m_glob = -INFINITY;
    bool has_contribution = false;
    for (uint c = 0; c < NC; ++c) {
        if (d_row[c] > 0.0f) {
            m_glob = max(m_glob, m_row[c]);
            has_contribution = true;
        }
    }

    float D = 0.0f;
    if (has_contribution) {
        for (uint c = 0; c < NC; ++c) {
            if (d_row[c] > 0.0f) {
                D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob);
            }
        }
    }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + head_index * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        if (has_contribution) {
            for (uint c = 0; c < NC; ++c) {
                if (d_row[c] > 0.0f) {
                    acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
                }
            }
        }
        out_row[i] = half(acc * inv_d);
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
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
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = float(K_row[i]);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            o_local[slot] += p_exp * float(V_row[i]);
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    device const bfloat* sinks       [[buffer(6), function_constant(FC_ATTN_HAS_SINKS)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    // THE SINK ENTERS THE MAX AND THE DENOMINATOR, AND NOTHING ELSE. It is a
    // logit for a key that has no VALUE row, so it takes probability mass
    // away from the real keys and contributes nothing to the numerator --
    // exactly what ggml's `ggml_soft_max_add_sinks` does
    // (`max = MAX(max, sk[head])`, then `sum += expf(sk[head] - max)`).
    // Adding it to `acc` below would be the plausible wrong version.
    float sink = 0.0f;
    if (FC_ATTN_HAS_SINKS) {
        sink = float(sinks[q_head]);
        m_glob = max(m_glob, sink);
    }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    if (FC_ATTN_HAS_SINKS) { D += attn_softmax_exp(sink - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}
