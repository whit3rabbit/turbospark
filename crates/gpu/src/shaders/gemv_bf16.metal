#include <metal_stdlib>
using namespace metal;

// ============================================================================
// gemv_bf16 -- GEMV over a resident BF16 matrix, `y[m] = sum_n W[m,n]*x[n]`.
//
// The unquantized-matrix arm the dense qwen GDN flow needed for the Bonsai-2
// line: `linear_attn.in_proj_a/in_proj_b` ship as raw F32 in that checkpoint
// (every other install quantizes them), the repack walk narrows them to BF16
// like every unquantized tensor (AGENTS.md Gotcha 45), and this is the reader
// that tag 1 lacked for MATRICES -- the tag was honoured for norms (bf16
// weight views) and the embedding before, never for a projection.
//
// Reduction shape follows `vision.metal`'s tower GEMM: one threadgroup per
// output row, 256 threads striding the input axis, FP32 accumulate, simd
// then threadgroup reduce. `vision_matmul_fp16` itself is not reusable: its
// weight pointer is `half`, and BF16 read as FP16 is wrong by up to 2^112
// with no error anywhere -- the exact hazard the resident-dtype gate exists
// for.
// ============================================================================

constant constexpr uint kBf16GemmThreads = 256;

[[kernel, max_total_threads_per_threadgroup(256)]]
void bf16_gemv_rows(device const bfloat* W [[buffer(0)]],
                    device const half* x [[buffer(1)]],
                    device half* y [[buffer(2)]],
                    constant uint& M [[buffer(3)]],
                    constant uint& N [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint lsize [[threads_per_threadgroup]],
                    uint simd_lane_id [[thread_index_in_simdgroup]],
                    uint simd_group_id [[simdgroup_index_in_threadgroup]],
                    uint simdgroups [[simdgroups_per_threadgroup]]) {
    if (tg >= M) return;
    threadgroup float partial[kBf16GemmThreads / 32];
    device const bfloat* wrow = W + uint64_t(tg) * N;
    float acc = 0.0f;
    for (uint i = lid; i < N; i += lsize) {
        acc = fma(float(wrow[i]), float(x[i]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) { y[tg] = half(v); }
    }
}
