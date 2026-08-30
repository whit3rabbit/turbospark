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
// THAT LAST SENTENCE IS THE ONE FC_GEMM_R RE-OPENS, and note 2 is why it is
// worth re-opening rather than a reason not to. Note 2 held
// `float e[8][kMaxBatchRows]` -- a whole M-WIDE activation tile, ~208 floats
// of register array, declared at the cap whatever R was, which is why it
// measured worse even at R=1. That is the same door the unroll table below
// records: the failure was the DECLARED live set, not the idea. Blocking
// rows without an activation tile costs `acc[R][B] + q[R][8]` and nothing
// per-row that scales with B, so the arithmetic is different enough to
// deserve its own measurement.
//
// What it buys, and both terms are per (block, lane):
//
//   - the two half4 activation loads drop from R*B to B, because R rows now
//     share one read of `x` instead of sitting in R separate SIMD groups;
//   - `sum = e0 + ... + e7` drops from R*B to B. It depends only on `bi`,
//     and the R=1 kernel recomputes it once per ROW even though no row can
//     change it -- 7 of the ~17 inner ALU ops.
//
// The irreducible term is the dot product's 8 fma per (r, bi), which is the
// actual arithmetic and does not shrink.
//
// IT DOES NOT MOVE A SINGLE BIT, and that is the property that lets it be
// gated by the parity test rather than by a quality gate. Each output
// `(row, bi)` still walks the same blocks in the same order into one FP32
// accumulator, and the 32-lane partition of K feeding `simd_sum` is
// untouched; only which SIMD GROUP owns the row changes.
//
// MEASURED ON AC, 2026-08-29, mean of four runs on gate/up 17408x5120
// (the other five QWEN38_SHAPES agree to 0.03; within-cell spread 0.00-0.05):
//
//   R      M=2    M=4    M=8   M=16
//   1     0.510  0.607  0.502  0.487   <- the shape shipped before this
//   2     0.505  0.307  0.448  0.425
//   4     0.617  0.365  0.427  0.375
//
// R=4 wins at M=16 (1.30x) and M=8, R=2 wins at M=4 (1.98x), and R=4 LOSES
// at M=2 -- so nothing selects a width by heuristic. Read the M=4 column
// with the unroll table below: `unroll_count(4)` makes `b_dim == 4` exactly
// one full unroll, and R=1's bump there (0.607 against 0.50 either side) is
// that interaction going badly rather than a property of the width.
//
// R=1 reproduces the pre-FC_GEMM_R kernel in BITS
// (`row_blocking_does_not_move_a_single_bit`) and, now, in SPEED: an
// interleaved A/B against `aa094b4^`, three pairs, agrees to 0.01 on every
// cell. That is what says `acc[kMaxRowBlock][kMaxBatchRows]` collapses at
// R=1, and it had to be a timing because no static instrument on this
// device can see register pressure --
// `maxTotalThreadsPerThreadgroup` reads 1024 even for an `acc[64][16]` that
// fits no register file anywhere, which
// `pipeline_reflection_cannot_see_this_kernels_register_pressure` records.
//
// THE `count(4)` TABLE BELOW IS FROM ANOTHER SESSION AND IS ~11% OPTIMISTIC
// AGAINST TODAY'S MACHINE. The identical code read 0.51 / 0.61 / 0.50 / 0.49
// here against its 0.50 / 0.55 / 0.46 / 0.44. Compare arms measured beside
// each other, never against these rows (AGENTS.md Gotcha 22).
//
// Layouts. `x` is [B, N] and `y` is [B, M], both token-major, so each
// token's vectors stay contiguous and a caller can hand one row of either
// to a kernel that still wants a single token. B is capped at
// kMaxBatchRows because the accumulators are a register array; the cap is
// a hard precondition, not a clamp, and the host asserts it.

// Concatenated AFTER dequant_int4.metal (crate Gotcha 4), which supplies
// the includes and kGroupSize. Never dispatch this file's own constant.

constant constexpr uint kMaxBatchRows = 16;
// Output rows one SIMD group may own. Four rather than eight because
// `acc[R][B]` is R*B floats of register array and the header's unroll table
// is a record of what happens past ~128 live floats.
constant constexpr uint kMaxRowBlock = 4;

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
// Output rows per SIMD group; see the header. Absent or 1 is the shape this
// kernel had before the constant existed, which is what every wired call
// site still dispatches. It MUST reach `specialized_constants`'s key like
// its three siblings -- an axis missing from that key is served whichever
// pipeline compiled first (crate Gotcha 1), and here that reads a different
// number of rows per SIMD group while producing finite, plausible output.
constant uint FC_GEMM_R      [[function_constant(104)]];

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

// Unlike its three siblings there is no runtime argument to fall back to:
// R is a shape of the DISPATCH (it decides the threadgroup count) rather
// than of the data, so an unspecialized pipeline is the 1-row kernel.
static inline uint gemm_r() {
    return (is_function_constant_defined(FC_GEMM_USE_FC) &&
            FC_GEMM_USE_FC &&
            is_function_constant_defined(FC_GEMM_R)) ? FC_GEMM_R : 1u;
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
    constexpr uint simdgroups_per_tg = 8;
    const uint m_dim = gemm_m(M);
    const uint n_dim = gemm_n(N);
    const uint b_dim = gemm_b(B);
    const uint r_dim = gemm_r();
    // Rows a SIMD group owns are CONTIGUOUS, so its R weight rows are
    // adjacent in `W` rather than `simdgroups_per_tg` apart. Every quantity
    // here is uniform across the SIMD group (`tg_idx` and `sg_idx` are), so
    // the guards below never make `simd_sum` divergent.
    const uint row0 = (tg_idx * simdgroups_per_tg + sg_idx) * r_dim;
    if (row0 >= m_dim) return;

    const uint n_groups  = n_dim / kGroupSize;
    const uint row_bytes = n_dim / 2;
    device const uint8_t* W_row = W      + uint(row0) * row_bytes;
    device const bfloat*  s_row = scales + uint(row0) * n_groups;
    device const bfloat*  b_row = biases + uint(row0) * n_groups;

    // Still declared at the caps, because MSL needs a compile-time bound
    // here and a function constant is not one. What the baked `b_dim` and
    // `r_dim` buy is that every loop below stops at them, so the unused
    // accumulators are never written and the optimizer drops them. THAT
    // COLLAPSE IS UNVERIFIED: no static instrument on this device can see
    // register pressure (see the header), so the R=1 column of a `c(R, B)`
    // sweep on AC is what says the shipped shape did not regress.
    float acc[kMaxRowBlock][kMaxBatchRows];
    for (uint r = 0; r < r_dim; ++r) {
        for (uint i = 0; i < b_dim; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g    = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;

        // One unpack per (row, block), which is where it already was. `qv`
        // is R*8 floats; keeping the packed `w4` and re-extracting inside
        // the `bi` loop would trade those for 8 ops per (r, bi, block),
        // i.e. B times the work this loop does.
        float qv[kMaxRowBlock][8];
        float sv[kMaxRowBlock];
        float bv[kMaxRowBlock];
        for (uint r = 0; r < r_dim; ++r) {
            if (row0 + r >= m_dim) {
                // Zeroed rather than skipped: the store below drops this
                // row anyway, and reading `W` past the last row is the
                // out-of-bounds access that would otherwise be taken.
                sv[r] = 0.0f;
                bv[r] = 0.0f;
                for (uint k = 0; k < 8; ++k) { qv[r][k] = 0.0f; }
                continue;
            }
            // Same 2-byte-aligned pair load as the GEMV: the resident
            // tensors are 2-aligned but not 4-aligned, so a `uint*` load is
            // undefined.
            device const ushort* wp =
                (device const ushort*)(W_row + uint(r) * row_bytes + byte_base);
            const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
            sv[r] = float(s_row[uint(r) * n_groups + g]);
            bv[r] = float(b_row[uint(r) * n_groups + g]);
            qv[r][0] = float(  w4        & 0x0Fu);
            qv[r][1] = float(( w4        & 0xFFu) >> 4);
            qv[r][2] = float((w4 >> 8)   & 0x0Fu);
            qv[r][3] = float(((w4 >> 8)  & 0xFFu) >> 4);
            qv[r][4] = float((w4 >> 16)  & 0x0Fu);
            qv[r][5] = float(((w4 >> 16) & 0xFFu) >> 4);
            qv[r][6] = float((w4 >> 24)  & 0x0Fu);
            qv[r][7] = float(((w4 >> 24) & 0xFFu) >> 4);
        }

        // THE UNROLL FACTOR IS SWEPT, NOT CHOSEN, and 4 is the optimum on
        // every shape. Baking `b_dim` lets the compiler unroll this loop
        // FULLY, and full unrolling is the difference between the best and
        // the worst numbers this kernel has ever produced -- at B=16 it
        // holds sixteen copies of `e0..e7` live at once, ~128 floats beside
        // `acc[16]`, and spills. That is note 2's register-blocking failure
        // arriving through a different door.
        //
        // `c(M)` on gate/up 17408x5120, one session, one binary, at R=1:
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
        //
        // **R AND THIS FACTOR ARE NOT INDEPENDENT AXES**: both spend the
        // same register file, so the count above is optimal at R=1 and is
        // not evidence about R=2 or R=4. It cannot be a function constant
        // (a `#pragma` needs a literal), so a joint sweep means editing this
        // number and re-running `c_of_r_and_m_for_the_batched_kernel`. There
        // is no cheaper filter: the pipeline reflection that looked like one
        // reads its ceiling on every shape (see the header).
        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const half4 xa = *((device const half4*)(x_b + elem));
            const half4 xb = *((device const half4*)(x_b + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            // Hoisted out of the row loop: it depends on `bi` alone, and
            // the one-row kernel recomputed it once per SIMD group.
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            for (uint r = 0; r < r_dim; ++r) {
                float dot = 0.0f;
                dot = fma(qv[r][0], e0, dot); dot = fma(qv[r][1], e1, dot);
                dot = fma(qv[r][2], e2, dot); dot = fma(qv[r][3], e3, dot);
                dot = fma(qv[r][4], e4, dot); dot = fma(qv[r][5], e5, dot);
                dot = fma(qv[r][6], e6, dot); dot = fma(qv[r][7], e7, dot);
                acc[r][bi] = fma(sv[r], dot, acc[r][bi]);
                acc[r][bi] = fma(bv[r], sum, acc[r][bi]);
            }
        }
    }

    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const float x0 = float(x_b[g * kGroupSize + lane * 2u]);
            const float x1 = float(x_b[g * kGroupSize + lane * 2u + 1u]);
            const float sum = x0 + x1;
            for (uint r = 0; r < r_dim; ++r) {
                if (row0 + r >= m_dim) { continue; }
                const float s = float(s_row[uint(r) * n_groups + g]);
                const float b = float(b_row[uint(r) * n_groups + g]);
                const uint8_t byte =
                    W_row[uint(r) * row_bytes + g * (kGroupSize / 2) + lane];
                const float lo = float(uint(byte & 0x0Fu));
                const float hi = float(uint(byte >> 4));
                float dot = fma(lo, x0, 0.0f);
                dot = fma(hi, x1, dot);
                acc[r][bi] = fma(s, dot, acc[r][bi]);
                acc[r][bi] = fma(b, sum, acc[r][bi]);
            }
        }
    }

    for (uint r = 0; r < r_dim; ++r) {
        const uint row = row0 + r;
        if (row >= m_dim) { continue; }
        for (uint bi = 0; bi < b_dim; ++bi) {
            const float total = simd_sum(acc[r][bi]);
            if (lane == 0) {
                y[bi * m_dim + row] = half(total);
            }
        }
    }
}
