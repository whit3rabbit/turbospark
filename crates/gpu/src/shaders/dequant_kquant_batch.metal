// ============================================================================
// dequant_kquant_batch - the GGUF Q4_K, Q6_K and Q8_0 GEMVs with B right-hand
// sides instead of one, so the speculative verify pass can run B drafted
// tokens through one K-quant matrix in ONE dispatch. PORT-LOCAL, like every
// GGUF kernel here: the Swift engine has no GGUF intake.
//
// Concatenated AFTER dequant_q4_k.metal, dequant_q6_k.metal and
// dequant_q8_0.metal (crate Gotcha 4), which supply the block constants and
// the includes. Never compile this file alone.
//
// WHY A KERNEL AND NOT A LOOP. `encode_gemm_any` refuses a sequential GEMV
// fallback by name: it is numerically identical, so it passes every parity
// test while the verify measures the unbatched engine. The amortization has
// to happen inside one dispatch: read each weight byte once, multiply it into
// B accumulators (`dequant_int4_batch.metal`'s header has the measurements).
//
// BIT-IDENTICAL TO B CALLS OF THE MATCHING GEMV, and the PARITY TEST is the
// evidence, not this source. Each kernel keeps its GEMV's row mapping (one
// SIMD group per output row, `row = tg * 8 + sg`), its per-lane walk over
// superblocks, its weight reconstruction expression, its per-element fma
// sequence into ONE FP32 accumulator, and its 32-lane `simd_sum` partition.
// Only the innermost `for (bi < B)` is new. That is what lets speculation
// stay bit-exact against plain decode on a K-quant trunk, whose decode runs
// these GEMVs (there are no fused K-quant decode kernels).
//
// SOURCE ORDER DOES NOT PIN MACHINE ORDER HERE. These libraries build with
// `-ffast-math`, and swapping Q4_K's two fma lines survived the parity test
// at 67x4096 (measured 2026-11-19): the compiler is free to reassociate, and
// did. So equality with the GEMV holds because both kernels compile to the
// same reduction, which is a property of the toolchain. Keep the expressions
// textually identical to the GEMV's, and treat a parity failure after a
// toolchain or GEMV change as real. `crates/gpu/tests/
// dequant_kquant_gemm_parity.rs` holds every width 1..16, in one shared
// context so a pipeline-key bug is visible too.
//
// B IS BAKED (function constant 102), M and N are NOT. Baked B lets the
// `bi` loop unroll and bounds the live accumulator set to the batch in
// flight. M and N stay runtime arguments because the GEMVs take them at
// runtime and nothing has measured a specialized-N variant; keeping the loop
// bounds the same as the GEMV's is the smallest change that keeps the bit
// contract obvious. The pipeline key MUST carry B (crate Gotcha 1).
//
// NO R AXIS and NO MEASURED TUNING. One row per SIMD group is the GEMV's
// mapping and the parity baseline. Any row blocking or activation staging
// needs its own c(M) sweep first.
//
// Layouts: `x` is [B, N] and `y` is [B, M], token-major, like the affine
// batched kernels, so one token's slice of either is contiguous.
// ============================================================================

constant constexpr uint kMaxBatchRowsKQuant = 16;

constant uint FC_KQ_GEMM_B      [[function_constant(102)]];
constant bool FC_KQ_GEMM_USE_FC [[function_constant(103)]];

static inline uint kq_batch(uint B) {
    return (is_function_constant_defined(FC_KQ_GEMM_USE_FC) && FC_KQ_GEMM_USE_FC &&
            is_function_constant_defined(FC_KQ_GEMM_B)) ? FC_KQ_GEMM_B : B;
}

// Q4_K: `dequant_q4_k_row_simd`'s walk, B accumulators.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q4_k_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    constant uint&        B      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint b_dim = kq_batch(B);
    const uint row = tg_idx * kRowsPerTGQ4_K + sg_idx;
    if (row >= M) return;

    device const uint8_t* W_row = W + uint(row) * q4_k_row_bytes(N);
    const uint n_blocks = N / kQ4_KBlockElems;

    float acc[kMaxBatchRowsKQuant];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ4_KBlockBytes;
        ushort d_raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        ushort dmin_raw = ushort(blk[2]) | (ushort(blk[3]) << 8);
        const float d = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));
        device const uint8_t* packed = blk + kQ4_KScalesAt;
        device const uint8_t* qs = blk + kQ4_KQuantsAt;

        for (uint g = 0; g < kQ4_KBlockElems / (2 * kQ4_KSubElems); ++g) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4_k_scale_min(2 * g, packed, sc_lo, m_lo);
            q4_k_scale_min(2 * g + 1, packed, sc_hi, m_hi);

            const uint8_t byte = qs[g * kQ4_KSubElems + lane];
            const float w_lo = d * sc_lo * float(byte & 0xF) - dmin * m_lo;
            const float w_hi = d * sc_hi * float(byte >> 4) - dmin * m_hi;

            const uint at = b * kQ4_KBlockElems + g * 2 * kQ4_KSubElems + lane;
            #pragma clang loop unroll_count(4)
            for (uint bi = 0; bi < b_dim; ++bi) {
                device const half* x_b = x + bi * N;
                acc[bi] = fma(w_lo, float(x_b[at]), acc[bi]);
                acc[bi] = fma(w_hi, float(x_b[at + kQ4_KSubElems]), acc[bi]);
            }
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float s = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * M + row] = half(s);
        }
    }
}

// Q5_K: `dequant_q5_k_row_simd`'s walk, B accumulators. Q4_K's two-level
// scales (`q4_k_scale_min`, from dequant_q4_k.metal) plus a fifth bit per
// element from `qh`, exactly as the GEMV assembles it.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q5_k_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    constant uint&        B      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint b_dim = kq_batch(B);
    const uint row = tg_idx * kRowsPerTGQ5_K + sg_idx;
    if (row >= M) return;

    device const uint8_t* W_row = W + uint(row) * q5_k_row_bytes(N);
    const uint n_blocks = N / kQ5_KBlockElems;

    float acc[kMaxBatchRowsKQuant];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ5_KBlockBytes;
        const ushort d_raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const ushort dmin_raw = ushort(blk[2]) | (ushort(blk[3]) << 8);
        const float d = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));
        device const uint8_t* packed = blk + kQ5_KScalesAt;
        device const uint8_t* qh = blk + kQ5_KQhAt;
        device const uint8_t* ql = blk + kQ5_KQuantsAt;

        const uint8_t bits = qh[lane];

        for (uint g = 0; g < kQ5_KBlockElems / (2 * kQ5_KSubElems); ++g) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4_k_scale_min(2 * g, packed, sc_lo, m_lo);
            q4_k_scale_min(2 * g + 1, packed, sc_hi, m_hi);

            const uint8_t byte = ql[g * kQ5_KSubElems + lane];
            const float q_lo = float((byte & 0xF) + (((bits >> (2 * g)) & 1) << 4));
            const float q_hi = float((byte >> 4) + (((bits >> (2 * g + 1)) & 1) << 4));
            const float w_lo = d * sc_lo * q_lo - dmin * m_lo;
            const float w_hi = d * sc_hi * q_hi - dmin * m_hi;

            const uint at = b * kQ5_KBlockElems + g * 2 * kQ5_KSubElems + lane;
            #pragma clang loop unroll_count(4)
            for (uint bi = 0; bi < b_dim; ++bi) {
                device const half* x_b = x + bi * N;
                acc[bi] = fma(w_lo, float(x_b[at]), acc[bi]);
                acc[bi] = fma(w_hi, float(x_b[at + kQ5_KSubElems]), acc[bi]);
            }
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float s = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * M + row] = half(s);
        }
    }
}

// Q6_K: `dequant_q6_k_row_simd`'s walk, B accumulators. The four weights are
// formed exactly as the GEMV forms its fma's first operand,
// `(d * sc) * q`, left to right.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q6_k_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    constant uint&        B      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint b_dim = kq_batch(B);
    const uint row = tg_idx * kRowsPerTGQ6_K + sg_idx;
    if (row >= M) return;

    const uint n_blocks = N / kQ6_KBlockElems;
    device const uint8_t* W_row = W + uint(row) * n_blocks * kQ6_KBlockBytes;

    float acc[kMaxBatchRowsKQuant];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ6_KBlockBytes;
        const ushort d_raw = ushort(blk[kQ6_KDAt]) | (ushort(blk[kQ6_KDAt + 1]) << 8);
        const float d = float(as_type<half>(d_raw));

        for (uint h = 0; h < kQ6_KBlockElems / kQ6_KHalfElems; ++h) {
            device const uint8_t* ql = blk + h * 64;
            device const uint8_t* qh = blk + kQ6_KQhAt + h * 32;
            device const uint8_t* sc = blk + kQ6_KScalesAt + h * 8;
            const uint is = lane / kQ6_KSubElems;

            const uint8_t lo = ql[lane];
            const uint8_t hi = ql[lane + 32];
            const uint8_t bits = qh[lane];
            const float q0 = float(int((lo & 0xF) | ((bits & 3) << 4)) - 32);
            const float q1 = float(int((hi & 0xF) | (((bits >> 2) & 3) << 4)) - 32);
            const float q2 = float(int((lo >> 4) | (((bits >> 4) & 3) << 4)) - 32);
            const float q3 = float(int((hi >> 4) | (((bits >> 6) & 3) << 4)) - 32);
            const float w0 = d * float(as_type<int8_t>(sc[is])) * q0;
            const float w1 = d * float(as_type<int8_t>(sc[is + 2])) * q1;
            const float w2 = d * float(as_type<int8_t>(sc[is + 4])) * q2;
            const float w3 = d * float(as_type<int8_t>(sc[is + 6])) * q3;

            const uint at = b * kQ6_KBlockElems + h * kQ6_KHalfElems + lane;
            #pragma clang loop unroll_count(4)
            for (uint bi = 0; bi < b_dim; ++bi) {
                device const half* x_b = x + bi * N;
                acc[bi] = fma(w0, float(x_b[at]), acc[bi]);
                acc[bi] = fma(w1, float(x_b[at + 32]), acc[bi]);
                acc[bi] = fma(w2, float(x_b[at + 64]), acc[bi]);
                acc[bi] = fma(w3, float(x_b[at + 96]), acc[bi]);
            }
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float s = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * M + row] = half(s);
        }
    }
}

// Q8_0: `dequant_q8_0_gemv_simd`'s walk, B accumulators.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q8_0_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    constant uint&        B      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint b_dim = kq_batch(B);
    const uint row = tg_idx * kRowsPerTGQ8_0 + sg_idx;
    if (row >= M) return;

    const uint n_blocks = N / kQ8_0BlockElems;
    device const uint8_t* W_row = W + uint(row) * n_blocks * kQ8_0BlockBytes;

    float acc[kMaxBatchRowsKQuant];
    for (uint bi = 0; bi < b_dim; ++bi) {
        acc[bi] = 0.0f;
    }

    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ8_0BlockBytes;
        ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        float d = float(as_type<half>(raw));
        float q = float(int(as_type<int8_t>(blk[2 + lane])));
        const float w = d * q;
        const uint at = b * kQ8_0BlockElems + lane;
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            acc[bi] = fma(w, float(x[bi * N + at]), acc[bi]);
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float s = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * M + row] = half(s);
        }
    }
}
