// PORT-LOCAL, not vendored: the Swift engine decodes one token at a time
// and has no batched verify, so there is no upstream kernel to mirror.
//
// `dequant_int4_gemm_simd` is `dequant_int4_gemv_simd` with B right-hand
// sides instead of one. It exists because ROADMAP Phase D2 measured that
// no arrangement of the GEMV can do this job:
//
//   - B dispatches on ONE matrix cost 0.65B, not ~1, because the weights
//     do not stay cached across them;
//   - encoding them CONCURRENTLY (no serial barrier) recovers only
//     1.07-1.40x, and 8 x the resulting rate lands at the ~355 GiB/s the
//     kernel saturates at, which says the hardware really did move the
//     weight bytes B times.
//
// So the amortization has to happen INSIDE one dispatch: read each packed
// nibble once, multiply it into B accumulators. That is the whole idea,
// and everything else here is `dequant_int4_gemv_simd` unchanged.
//
// TWO OPTIMIZATIONS WERE TRIED HERE AND BOTH LOST. Do not re-add either
// without re-measuring `c_of_m` first.
//
//   1. Staging `x` in threadgroup memory. It removes the per-batch device
//      reads, and it is slower on EVERY shape (0.39 -> 0.55 at M=16 on the
//      512-row routed-expert shape, 0.67 -> 0.86 on o_proj): the per-block
//      barriers serialize the eight SIMD groups for more than the saved
//      reads cost, and the cache already serves them.
//   2. Register blocking over rows, so one activation read serves R rows.
//      It is the right idea and it does not reorder any sum, but holding
//      the activations across rows needs `float e[8][kMaxBatchRows]` plus
//      `acc[R][kMaxBatchRows]`, about 208 floats of register array, which
//      spills. Measured WORSE even at R=1 (expert 0.45 -> 0.79 at M=8).
//
// Both failures say the same thing: the register file cannot hold an
// M-wide activation tile and threadgroup memory's barriers cost more than
// they save, so amortizing activations needs real matrix hardware
// (`simdgroup_matrix`) rather than a loop rearrangement.
//
// Layouts. `x` is [B, N] and `y` is [B, M], both token-major, so each
// token's vectors stay contiguous and a caller can hand one row of either
// to a kernel that still wants a single token. B is capped at
// kMaxBatchRows because the accumulators are a register array; the cap is
// a hard precondition, not a clamp, and the host asserts it.

// Concatenated AFTER dequant_int4.metal (crate Gotcha 4), which supplies
// the includes and kGroupSize. Never dispatch this file's own constant.

constant constexpr uint kMaxBatchRows = 16;

// M, N and B baked per shape, mirroring the GEMV's constants 20-22 next
// door. Indices 100-103 because 20-26 are that file's and this source is
// its concatenation.
//
// **B IS THE ONE THAT MATTERS HERE, and it is why this was worth doing
// separately rather than copying the GEMV's pair across.** With B a runtime
// argument the `for (bi < B)` loops cannot unroll and `acc[]` occupies all
// sixteen registers whatever the caller asked for, so a B=4 dispatch pays
// a B=16 register footprint on a kernel whose header already records that
// the register file is the binding constraint. Baking it sizes the live
// set to the batch actually in flight.
//
// The GEMV took this treatment on 2026-08-16 (`46617c6`) and this kernel
// did not, which quietly made every `c(M)` number worse: `c(M)` is the
// ratio of the two arms, so an optimization landing on the sequential one
// alone moves it the wrong way. See `docs/MTP_SPECULATIVE.md`.
constant uint FC_GEMM_M      [[function_constant(100)]];
constant uint FC_GEMM_N      [[function_constant(101)]];
constant uint FC_GEMM_B      [[function_constant(102)]];
constant bool FC_GEMM_USE_FC [[function_constant(103)]];

static inline uint gemm_m(constant uint& M) {
    return (is_function_constant_defined(FC_GEMM_USE_FC) &&
            FC_GEMM_USE_FC &&
            is_function_constant_defined(FC_GEMM_M)) ? FC_GEMM_M : M;
}

static inline uint gemm_n(constant uint& N) {
    return (is_function_constant_defined(FC_GEMM_USE_FC) &&
            FC_GEMM_USE_FC &&
            is_function_constant_defined(FC_GEMM_N)) ? FC_GEMM_N : N;
}

static inline uint gemm_b(constant uint& B) {
    return (is_function_constant_defined(FC_GEMM_USE_FC) &&
            FC_GEMM_USE_FC &&
            is_function_constant_defined(FC_GEMM_B)) ? FC_GEMM_B : B;
}

kernel void dequant_int4_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        B      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint m_dim = gemm_m(M);
    const uint n_dim = gemm_n(N);
    const uint b_dim = gemm_b(B);
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= m_dim) return;

    const uint n_groups  = n_dim / kGroupSize;
    const uint row_bytes = n_dim / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    // Still declared at the cap, because MSL needs a compile-time bound
    // here and a function constant is not one. What the baked `b_dim` buys
    // is that every loop below stops at it, so the unused accumulators are
    // never written and the optimizer drops them.
    float acc[kMaxBatchRows];
    for (uint i = 0; i < b_dim; ++i) {
        acc[i] = 0.0f;
    }

    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Same 2-byte-aligned pair load as the GEMV: the resident tensors
        // are 2-aligned but not 4-aligned, so a `uint*` load is undefined.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;

        const float q0 = float(  w4        & 0x0Fu);
        const float q1 = float(( w4        & 0xFFu) >> 4);
        const float q2 = float((w4 >> 8)   & 0x0Fu);
        const float q3 = float(((w4 >> 8)  & 0xFFu) >> 4);
        const float q4 = float((w4 >> 16)  & 0x0Fu);
        const float q5 = float(((w4 >> 16) & 0xFFu) >> 4);
        const float q6 = float((w4 >> 24)  & 0x0Fu);
        const float q7 = float(((w4 >> 24) & 0xFFu) >> 4);

        // THE UNROLL FACTOR IS SWEPT, NOT CHOSEN, and 4 is the optimum on
        // every shape. Baking `b_dim` lets the compiler unroll this loop
        // FULLY, and full unrolling is the difference between the best and
        // the worst numbers this kernel has ever produced -- at B=16 it
        // holds sixteen copies of `e0..e7` live at once, ~128 floats beside
        // `acc[16]`, and spills. That is note 2's register-blocking failure
        // arriving through a different door.
        //
        // `c(M)` on gate/up 17408x5120, one session, one binary:
        //
        //   unroll     M=2    M=4    M=8   M=16
        //   none      1.04   0.90   0.85   0.83   <- before any of this
        //   disable   0.95   0.81   0.77   0.73
        //   count(2)  0.46   0.64   0.59   0.57
        //   count(4)  0.50   0.55   0.46   0.44   <- shipped
        //   count(8)  0.51   0.55   0.64   0.89
        //   full      0.51   0.57   0.65   1.14   <- spilling
        //
        // Read the two ends together: the unrolling is what buys the win at
        // every M, and it is also what destroys M=16 if left unbounded. A
        // fixed count of 4 keeps ~32 activation floats live regardless of B,
        // which is what makes the row monotonic in M for the first time.
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const half4 xa = *((device const half4*)(x_b + elem));
            const half4 xb = *((device const half4*)(x_b + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(q0, e0, dot); dot = fma(q1, e1, dot);
            dot = fma(q2, e2, dot); dot = fma(q3, e3, dot);
            dot = fma(q4, e4, dot); dot = fma(q5, e5, dot);
            dot = fma(q6, e6, dot); dot = fma(q7, e7, dot);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            acc[bi] = fma(s, dot, acc[bi]);
            acc[bi] = fma(b, sum, acc[bi]);
        }
    }

    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float lo = float(uint(byte & 0x0Fu));
        const float hi = float(uint(byte >> 4));
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const float x0 = float(x_b[g * kGroupSize + lane * 2u]);
            const float x1 = float(x_b[g * kGroupSize + lane * 2u + 1u]);
            float dot = fma(lo, x0, 0.0f);
            dot = fma(hi, x1, dot);
            const float sum = x0 + x1;
            acc[bi] = fma(s, dot, acc[bi]);
            acc[bi] = fma(b, sum, acc[bi]);
        }
    }

    for (uint bi = 0; bi < b_dim; ++bi) {
        const float total = simd_sum(acc[bi]);
        if (lane == 0) {
            y[bi * m_dim + row] = half(total);
        }
    }
}
