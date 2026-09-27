#include <metal_stdlib>
using namespace metal;

// Exact ranks under compute::select_blocks' ordering: greater score first,
// then lower block index. A NaN writes a sentinel rank and sets the sticky
// status word; infinities remain valid ordered scores, like the CPU oracle.
constant constexpr uint kQsaTopKMaxSimdGroups = 8;

[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_topk_block_ranks_fp32(
    device const float* scores [[buffer(0)]],
    device uint* ranks [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    if (block >= num_blocks) { return; }

    const float score = scores[block];
    if (isnan(score)) {
        if (lid == 0) {
            ranks[block] = ~0u;
            atomic_store_explicit(&status[0], 1u, memory_order_relaxed);
        }
        return;
    }

    threadgroup uint partial[kQsaTopKMaxSimdGroups];
    uint rank = 0;
    for (uint other = lid; other < num_blocks; other += lsize) {
        const float candidate = scores[other];
        if (!isnan(candidate) &&
            (candidate > score || (candidate == score && other < block))) {
            rank += 1u;
        }
    }

    const uint simd_rank = simd_sum(rank);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_rank;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const uint group_rank = simd_lane_id < simdgroups ? partial[simd_lane_id] : 0u;
        const uint total_rank = simd_sum(group_rank);
        if (simd_lane_id == 0) {
            ranks[block] = total_rank;
        }
    }
}

// One GPU thread walks the ranks in block order, emitting selected positions
// in ascending order. If any score was NaN, it writes one safe position so
// attention can finish; the caller reports the sticky status after its normal
// command-buffer wait.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_topk_write_positions(
    device const uint* ranks [[buffer(0)]],
    device uint* positions [[buffer(1)]],
    device uint* selected_count [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    constant uint& visible [[buffer(4)]],
    constant uint& block_topk [[buffer(5)]],
    constant uint& compress_ratio [[buffer(6)]],
    uint block [[thread_position_in_grid]]
) {
    if (block != 0u) { return; }

    bool invalid = false;
    uint selected = 0;
    for (uint i = 0; i < num_blocks; ++i) {
        const uint rank = ranks[i];
        invalid = invalid || rank == ~0u;
        if (rank < block_topk) {
            const uint position_start = i * compress_ratio;
            const uint output_start = selected * compress_ratio;
            for (uint j = 0; j < compress_ratio; ++j) {
                positions[output_start + j] = position_start + j;
            }
            selected += 1u;
        }
    }

    if (invalid) {
        positions[0] = 0u;
        selected_count[0] = 1u;
    } else {
        const uint tail = visible % compress_ratio;
        const uint tail_start = num_blocks * compress_ratio;
        for (uint i = 0; i < tail; ++i) {
            positions[selected * compress_ratio + i] = tail_start + i;
        }
        selected_count[0] = selected * compress_ratio + tail;
    }
}
