// ============================================================================
// bf16_gemm_rows - `bf16_gemv_rows` with B right-hand sides, so the
// speculative verify can run B drafted tokens through one resident BF16
// matrix in ONE dispatch. Concatenated AFTER gemv_bf16.metal (crate
// Gotcha 4), which supplies the includes and `kBf16GemmThreads`.
//
// WHY IT EXISTS. llama.cpp's Q4_K_M and Q6_K `qwen35` GGUFs ship
// `ssm_alpha`/`ssm_beta` as F32, the walk narrows them to BF16, and the
// batched GDN verify sends `in_proj_a`/`in_proj_b` through
// `encode_gemm_any`, which had no BF16 arm. So those files could decode but
// never speculate. The matrices are tiny (`num_v_heads` rows), so this is
// about coverage, not throughput.
//
// BIT-IDENTICAL TO B CALLS OF `bf16_gemv_rows`, held by
// `crates/gpu/tests/gemv_bf16_gemm_parity.rs` rather than by this source
// (the library builds with `-ffast-math`, which may reassociate; see
// `dequant_kquant_batch.metal`). The structure is the GEMV's: one
// threadgroup per output row, 256 threads striding the input axis into one
// FP32 accumulator each, `simd_sum`, a per-simdgroup partial in threadgroup
// memory, and a final `simd_sum` over those partials in simdgroup 0. Only
// the per-right-hand-side loop is new, and each `bi` repeats the GEMV's
// arithmetic on its own row of `x`.
//
// B IS BAKED (function constant 102) so the `bi` loops unroll and the live
// accumulator set matches the batch in flight. The host key carries it.
// Layouts: `x` is [B, N] and `y` is [B, M], token-major.
// ============================================================================

constant constexpr uint kMaxBatchRowsBf16 = 16;

constant uint FC_BF16_GEMM_B      [[function_constant(102)]];
constant bool FC_BF16_GEMM_USE_FC [[function_constant(103)]];

[[kernel, max_total_threads_per_threadgroup(256)]]
void bf16_gemm_rows(device const bfloat* W [[buffer(0)]],
                    device const half* x [[buffer(1)]],
                    device half* y [[buffer(2)]],
                    constant uint& M [[buffer(3)]],
                    constant uint& N [[buffer(4)]],
                    constant uint& B [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint lsize [[threads_per_threadgroup]],
                    uint simd_lane_id [[thread_index_in_simdgroup]],
                    uint simd_group_id [[simdgroup_index_in_threadgroup]],
                    uint simdgroups [[simdgroups_per_threadgroup]]) {
    const uint b_dim = (is_function_constant_defined(FC_BF16_GEMM_USE_FC) && FC_BF16_GEMM_USE_FC &&
                        is_function_constant_defined(FC_BF16_GEMM_B)) ? FC_BF16_GEMM_B : B;
    if (tg >= M) return;
    threadgroup float partial[kBf16GemmThreads / 32][kMaxBatchRowsBf16];
    device const bfloat* wrow = W + uint64_t(tg) * N;

    float acc[kMaxBatchRowsBf16];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }
    for (uint i = lid; i < N; i += lsize) {
        const float w = float(wrow[i]);
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            acc[bi] = fma(w, float(x[uint64_t(bi) * N + i]), acc[bi]);
        }
    }
    for (uint bi = 0; bi < b_dim; ++bi) {
        const float s = simd_sum(acc[bi]);
        if (simd_lane_id == 0) { partial[simd_group_id][bi] = s; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint bi = 0; bi < b_dim; ++bi) {
            float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id][bi] : 0.0f;
            v = simd_sum(v);
            if (simd_lane_id == 0) { y[uint64_t(bi) * M + tg] = half(v); }
        }
    }
}
