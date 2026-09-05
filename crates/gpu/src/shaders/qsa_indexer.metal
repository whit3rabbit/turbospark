#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qsa_indexer -- `qwen4_exp`'s QSA (query-sparse attention) block indexer:
// pooling and scoring (`docs/QWEN4_PHASE0.md` section 5,
// `crates/compute/src/qsa_indexer.rs`'s CPU reference). PORT-LOCAL: the
// Swift engine has no architecture with this mechanism, and this port has
// no decode flow reading `self_attn.indexer.*` yet -- these two kernels are
// groundwork, matched to the CPU reference, not yet dispatched from any
// family's attention path. Block SELECTION (top-k over `scores`) stays
// host-side, matching this port's existing MoE router precedent (top-k is
// already a host round trip there); no kernel for it exists here either.
//
// Norm (`rms_norm_centered`, per head) and RoPE (`rope_neox_subdim` at
// `rotary_dim = 64`, `theta = 1e7` -- the SAME object the trunk's own QSA
// attention already dispatches) are NOT re-implemented here: both already
// exist as separate kernels/dispatches in this crate
// (`rmsnorm_bf16w_centered`, `rope_neox_subdim`) and are called once per
// head (query) or once per block (pooled key, since blocks do not share a
// position) by whatever wires this indexer into a real forward pass.
// ============================================================================

// pooled[b*D+d] = mean over t in [0, compress_ratio) of keys[(b*compress_ratio+t)*D+d]
//
// `keys` is `[visible * D]` RAW (un-normed, un-roped) rows, oldest token
// first. Only the COMPLETE blocks are pooled (`visible / compress_ratio`,
// rounded down); the ragged tail (`visible % compress_ratio` trailing rows)
// is never read by this kernel -- section 5 says the tail is ALWAYS
// selected rather than pooled or scored at all.
//
// ONE THREAD PER OUTPUT ELEMENT, NO THREADGROUP REDUCTION: `compress_ratio`
// is small (4 for the real checkpoint) and both the block's rows and the
// output are already materialized, the same reasoning `hc_mix_fp16` gives
// for its own small-C mix (`hyper.metal`). Accumulates in FP32, matching
// the CPU reference's "pooled = mean(block keys) in FP32" contract.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_pool_blocks_mean_fp16(
    device const half* keys           [[buffer(0)]],   // [visible * D] FP16
    device half*       pooled         [[buffer(1)]],   // [num_blocks * D] FP16
    constant uint&     compress_ratio [[buffer(2)]],
    constant uint&     d              [[buffer(3)]],
    constant uint&     num_blocks     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total = num_blocks * d;
    if (tid >= total) return;
    const uint b = tid / d;
    const uint dim = tid % d;

    float acc = 0.0f;
    const uint block_base = b * compress_ratio * d;
    for (uint t = 0; t < compress_ratio; t++) {
        acc += float(keys[block_base + t * d + dim]);
    }
    pooled[tid] = half(acc / float(compress_ratio));
}

// Threadgroup memory carries at most simdgroups_per_threadgroup = 256/32 = 8
// partial sums, the same shape `ple_gate_fp16` uses for its own dot-product
// reduction (`ple.metal`).
constant constexpr uint kQsaMaxSimdGroups = 8;

// scores[b] = relu(sum over (h, d) of q[h*D+d] * pooled[b*D+d]) / sqrt(D)
//
// `q` is `[num_heads * D]`, ALREADY normed and roped at the query's own
// current position; `pooled` is `[num_blocks * D]`, ALREADY normed and
// roped at each block's own first-token position. `index_kv_heads == 1`,
// so every one of the `num_heads` query heads reads the SAME pooled row
// for a given block.
//
// **The `relu` is OUTSIDE the head sum**, matching section 5's own
// parenthesization (`relu(q @ pooled^T).sum(over heads)`): every head's
// dot product against this block's pooled key accumulates into ONE total
// before relu is applied once, not once per head before summing -- the
// CPU reference's own doc explains why the two read differently whenever
// a head disagrees in sign with the total.
//
// One threadgroup per block, two-stage SIMD reduction (`rms_block_inv`'s
// and `ple_gate_fp16`'s shape): every thread strides over the full
// `num_heads * D` term count, reading `pooled` at `i % D` so every head's
// dot product against the one shared pooled row folds into the same
// running sum before the per-SIMD-group and cross-SIMD-group reductions.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_score_blocks_fp16(
    device const half*  q          [[buffer(0)]],   // [num_heads * D] FP16
    device const half*  pooled     [[buffer(1)]],   // [num_blocks * D] FP16
    device float*       scores     [[buffer(2)]],   // [num_blocks] FP32
    constant uint&      num_heads  [[buffer(3)]],
    constant uint&      d          [[buffer(4)]],
    uint  block             [[threadgroup_position_in_grid]],
    uint  lid               [[thread_position_in_threadgroup]],
    uint  lsize              [[threads_per_threadgroup]],
    uint  simd_lane_id      [[thread_index_in_simdgroup]],
    uint  simd_group_id     [[simdgroup_index_in_threadgroup]],
    uint  simdgroups        [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQsaMaxSimdGroups];
    const uint terms = num_heads * d;
    device const half* pooled_block = pooled + block * d;

    float acc = 0.0f;
    for (uint i = lid; i < terms; i += lsize) {
        const uint dim = i % d;
        acc = fma(float(q[i]), float(pooled_block[dim]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            scores[block] = max(v, 0.0f) / sqrt(float(d));
        }
    }
}
