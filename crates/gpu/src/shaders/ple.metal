#include <metal_stdlib>
using namespace metal;

// ============================================================================
// ple — `qwen4_exp`'s PLE (per-layer n-gram embedding) gate
// (`docs/QWEN4_PHASE0.md` section 4). PORT-LOCAL: the Swift engine has no
// architecture with this mechanism.
// ============================================================================

// Threadgroup memory carries at most simdgroups_per_threadgroup = 256/32 = 8
// partial sums. Slot 0 is reused after the merge to broadcast the sigmoid
// gate scalar.
constant constexpr uint kPleMaxSimdGroups = 8;

// gate[c]   = sign(dot[c]) * sqrt(max(|dot[c]|, 1e-6))     -- SIGNED sqrt
// dot[c]    = sum_h(key[c*H+h] * query[c*H+h]) / sqrt(H)
// gv[c*H+h] = sigmoid(gate[c]) * value[h]
//
// `key`/`query` are ALREADY grouped-centered-normed on entry
// (`rmsnorm_bf16w_grouped_centered`); this kernel does not norm them again.
// `value` is `[H]`, shared across every group -- `value_proj(emb)`'s raw
// output, no norm at all.
//
// One threadgroup per group `c`: reduces a DOT PRODUCT of `key`/`query`
// over `H` (the same two-stage SIMD-group shape `rmsnorm.metal`'s
// `rms_block_inv` uses for a sum-of-squares, not shared with it because the
// accumulator here is a product of two different buffers), computes the
// scalar gate, then every thread writes its share of the BROADCAST product
// against `value`.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_gate_fp16(
    device const half* key    [[buffer(0)]],   // [groups * H] FP16
    device const half* query  [[buffer(1)]],   // [groups * H] FP16
    device const half* value  [[buffer(2)]],   // [H] FP16
    device half*       gv     [[buffer(3)]],   // [groups * H] FP16
    constant uint&     H      [[buffer(4)]],
    uint  group            [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kPleMaxSimdGroups];
    const uint base = group * H;
    device const half* kg  = key   + base;
    device const half* qg  = query + base;
    device       half* gvg = gv    + base;

    float acc = 0.0f;
    for (uint i = lid; i < H; i += lsize) {
        acc = fma(float(kg[i]), float(qg[i]), acc);
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
            // Metal's `sign()` returns +/-0.0 at exactly zero (matching
            // NumPy/PyTorch's `sign(0) == 0`), so it is safe to use
            // directly here -- see `turbospark_compute::ple`'s CPU
            // reference for why the analogous Rust builtin (`f32::signum`,
            // which returns 1.0 at +0.0) is NOT used on that side.
            const float scaled = v / sqrt(float(H));
            const float mag = sqrt(max(fabs(scaled), 1e-6f));
            const float gate = sign(scaled) * mag;
            partial[0] = 1.0f / (1.0f + exp(-gate));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sig = partial[0];

    for (uint i = lid; i < H; i += lsize) {
        gvg[i] = half(sig * float(value[i]));
    }
}
