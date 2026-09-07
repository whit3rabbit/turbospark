#include <metal_stdlib>
using namespace metal;

// ============================================================================
// kv_quantize_tq — quantizes M rows of FP16 K or V into TurboQuant's packed
// format: one f32 norm plus `ceil(head_dim * bits / 32)` LSB-first packed
// u32 words, per (row, kv_head). Mirrors
// `turbospark_compute::kv_quant::quantize_row` exactly (see that module's
// doc for the codec itself and `docs/TRUBOQUANT.md` for what is and is not
// ported from mlx-vlm).
//
// One threadgroup per (row, kv_head); `head_dim` <= 512 threads, two
// elements per lane at the widest shape (Gemma's 512), matching
// `attention.metal`'s own per-lane budget.
//
// Steps: load the FP16 source row into threadgroup memory as f32, block-
// reduce the sum of squares for the norm, form the unit vector times the
// SIGN vector, run the Hadamard butterfly (log2(head_dim) stages, one
// barrier each) and the final 1/sqrt(D) scale, count how many midpoints
// each rotated coordinate exceeds, then pack LSB-first with one thread per
// destination word (never `|=` on threadgroup memory across threads --
// mlx-vlm's own fixed bug, per docs/TRUBOQUANT.md).
// ============================================================================

constant constexpr uint kTqMaxHeadDim = 512;
constant constexpr uint kTqThreads    = 256;
constant constexpr uint kTqMaxSimdGroups = 8;
constant constexpr uint kTqMaxPackedWords = (kTqMaxHeadDim * 4 + 31) / 32;

inline float tq_block_reduce_sum(float v,
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

[[kernel, max_total_threads_per_threadgroup(kTqThreads)]]
void kv_quantize_tq(
    device const half*  source          [[buffer(0)]],  // [rows, num_kv_heads, head_dim]
    device       uint*  dest            [[buffer(1)]],   // [rows, num_kv_heads, 1 + packed_words] (u32)
    device const float* signs           [[buffer(2)]],   // [head_dim]
    device const float* midpoints       [[buffer(3)]],   // [levels - 1]
    constant     uint&  head_dim        [[buffer(4)]],
    constant     uint&  num_kv_heads    [[buffer(5)]],
    constant     uint&  bits            [[buffer(6)]],
    constant     uint&  packed_words    [[buffer(7)]],
    constant     uint&  source_row_stride_elems [[buffer(8)]], // elements, one source row (all kv heads)
    constant     uint&  levels_minus_one [[buffer(9)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float buf[kTqMaxHeadDim];
    threadgroup float reduce_scratch[kTqMaxSimdGroups];
    threadgroup float bcast;

    const uint row = tg_id / num_kv_heads;
    const uint kv_head = tg_id % num_kv_heads;

    device const half* src = source + row * source_row_stride_elems + kv_head * head_dim;
    for (uint i = lid; i < head_dim; i += lsize) {
        buf[i] = float(src[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum of squares -> norm.
    float partial_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += lsize) {
        partial_sq = fma(buf[i], buf[i], partial_sq);
    }
    float sum_sq = tq_block_reduce_sum(partial_sq, simd_lane_id, simd_group_id, simdgroups,
                                        reduce_scratch, &bcast);
    const float norm = sqrt(sum_sq);
    const float inv_norm = 1.0f / max(norm, 1e-6f);

    // unit * sign, in place.
    for (uint i = lid; i < head_dim; i += lsize) {
        buf[i] = buf[i] * inv_norm * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hadamard butterfly, natural (Sylvester) order: identical linear map
    // to `turbospark_compute::wht::wht`'s recursive top-down halving (the
    // Walsh-Hadamard matrix is symmetric, so the iterative bottom-up
    // network and the recursive top-down one compute the same output for
    // any dimension, verified by hand at D=4 in the port's own notes).
    for (uint h = 1; h < head_dim; h <<= 1) {
        for (uint i = lid; i < head_dim / 2; i += lsize) {
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
    const float rht_scale = rsqrt(float(head_dim));

    // Pack LSB-first via THREADGROUP ATOMIC OR, one thread per SOURCE
    // INDEX -- the direct Metal translation of
    // `turbospark_compute::kv_quant::pack_lsb_first`'s per-index loop
    // (index `i` writes its low bits into word `i*bits/32` and, if it
    // straddles a word boundary, its spilled high bits into the next
    // word), with the race a plain `|=` would have between two lanes
    // touching the SAME word resolved by the atomic rather than by
    // deriving each word's own index range by hand. mlx-vlm's own fused
    // kernel shipped exactly that race (a non-atomic `|=` on threadgroup
    // memory) as a real bug, per docs/TRUBOQUANT.md -- the atomic is what
    // this port takes instead of "pack thread-per-word" boundary
    // arithmetic, which is easy to get off-by-one on a straddling index
    // and this port has no way to test-run to catch it.
    threadgroup atomic_uint packed_smem[kTqMaxPackedWords];
    for (uint w = lid; w < packed_words; w += lsize) {
        atomic_store_explicit(&packed_smem[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // AGENTS.md/CLAUDE.md S6: the codebook index (count of midpoints
    // exceeded) is computed and packed in the SAME loop iteration now,
    // rather than through a separate `idx_smem` threadgroup array written
    // by one pass and read back by another. That was dead: this loop and
    // the removed index-computation loop shared the identical iteration
    // space (`i = lid; i < head_dim; i += lsize`), so every lane only ever
    // read the index IT had just computed itself, never a neighbour's --
    // `idx_smem` bought a 2 KiB array and a `threadgroup_barrier` for a
    // cross-thread dependency that never existed. The comment it replaces
    // said otherwise.
    for (uint i = lid; i < head_dim; i += lsize) {
        const float rotated = buf[i] * rht_scale;
        uint value = 0;
        for (uint m = 0; m < levels_minus_one; ++m) {
            value += (rotated > midpoints[m]) ? 1u : 0u;
        }
        const uint bit_offset = i * bits;
        const uint word_idx = bit_offset / 32u;
        const uint offset = bit_offset % 32u;
        atomic_fetch_or_explicit(&packed_smem[word_idx], value << offset, memory_order_relaxed);
        const int spill = int(offset) + int(bits) - 32;
        if (spill > 0) {
            atomic_fetch_or_explicit(&packed_smem[word_idx + 1u], value >> (bits - uint(spill)),
                                      memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint* dst = dest + (row * num_kv_heads + kv_head) * (1u + packed_words);
    for (uint w = lid; w < packed_words; w += lsize) {
        dst[1u + w] = atomic_load_explicit(&packed_smem[w], memory_order_relaxed);
    }
    if (lid == 0) {
        dst[0] = as_type<uint>(norm);
    }
}
