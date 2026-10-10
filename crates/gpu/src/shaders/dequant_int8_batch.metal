// PORT-LOCAL, not vendored: the Swift engine decodes one token at a time
// and has no batched verify, so there is no upstream kernel to mirror --
// the same standing `dequant_int4_batch.metal` records, one width up.
//
// `dequant_int8_gemm_simd` is `dequant_int4_gemm_simd` with one BYTE per
// element instead of a packed nibble pair. The block structure, the lane
// partition, the accumulator layout, the unroll discipline and the row
// blocking are that kernel's UNCHANGED: 8 SIMD groups per threadgroup, R
// contiguous output rows per SIMD group, B right-hand sides with B baked as
// a function constant so the `bi` loop unrolls, `acc[R][B]` at the same two
// caps. Only two things differ:
//
//   - `row_bytes` is `n_dim`, not `n_dim / 2` (one byte per element);
//   - the per-block unpack reads two `ushort`s (8 weight bytes per lane,
//     the same 8 elements per (lane, block) the INT4 kernel unpacks from
//     4) and feeds them through as floats with no nibble extraction.
//
// The bias term is identical: the 8 elements one lane sees per block are
// the same 1/8 of one group of 64 in both widths, so `sum` and the group
// index `g = blk * 4 + (lane >> 3)` carry over exactly.
//
// Like its sibling it is held bit-exact against the per-token GEMV rather
// than against a tolerance: B rows of this kernel must equal B separate
// `dequant_int8_gemv_resident` calls (which reduce through
// `turbospark_compute::dequant_int8_gemv`), and that is what the parity
// test asserts. First carrier: the dense humanizer Gemma 4 12B, whose
// every projection is INT8 affine at group 64.
//
// MEASUREMENTS: none yet. Unlike the INT4 sibling -- whose header is a
// record of swept unroll counts and a row-block table -- this kernel ships
// with the sibling's SHAPES (unroll_count(4), best_row_block's steps) and
// no numbers of its own. The R table next door is a throughput choice
// proven not to move bits, so borrowing it is safe for correctness and
// unproven for speed; a `c(M)` sweep on the real shapes (gate/up 15360x3840,
// down 3840x15360, q 8192x3840) is the named follow-up before anyone
// quotes a prefill win.

// Concatenated AFTER dequant_int8.metal (crate Gotcha 4), which supplies
// the includes and kInt8GroupSize. Never dispatch this file's own constant.

constant constexpr uint kInt8MaxBatchRows = 16;
// Output rows one SIMD group may own. Same value and same reason as the
// INT4 sibling: `acc[R][B]` is R*B floats of register array.
constant constexpr uint kInt8MaxRowBlock = 4;

// M, N and B baked per shape, mirroring the INT4 sibling's constants
// 100-103. Indices do not collide with `dequant_int8.metal`'s (70-74)
// because this file is only ever compiled as that file's concatenation and
// the host sets 70-72 to their unspecialized values.
constant uint FC_INT8_GEMM_M      [[function_constant(100)]];
constant uint FC_INT8_GEMM_N      [[function_constant(101)]];
constant uint FC_INT8_GEMM_B      [[function_constant(102)]];
constant bool FC_INT8_GEMM_USE_FC [[function_constant(103)]];
// Output rows per SIMD group; see the INT4 sibling's header. Joins the
// pipeline key like its siblings -- an axis missing from the key is served
// whichever pipeline compiled first (crate Gotcha 1).
constant uint FC_INT8_GEMM_R      [[function_constant(104)]];

static inline uint int8_gemm_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT8_GEMM_USE_FC) &&
            FC_INT8_GEMM_USE_FC &&
            is_function_constant_defined(FC_INT8_GEMM_M)) ? FC_INT8_GEMM_M : M;
}

static inline uint int8_gemm_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT8_GEMM_USE_FC) &&
            FC_INT8_GEMM_USE_FC &&
            is_function_constant_defined(FC_INT8_GEMM_N)) ? FC_INT8_GEMM_N : N;
}

static inline uint int8_gemm_b(constant uint& B) {
    return (is_function_constant_defined(FC_INT8_GEMM_USE_FC) &&
            FC_INT8_GEMM_USE_FC &&
            is_function_constant_defined(FC_INT8_GEMM_B)) ? FC_INT8_GEMM_B : B;
}

static inline uint int8_gemm_r() {
    return (is_function_constant_defined(FC_INT8_GEMM_USE_FC) &&
            FC_INT8_GEMM_USE_FC &&
            is_function_constant_defined(FC_INT8_GEMM_R)) ? FC_INT8_GEMM_R : 1u;
}

kernel void dequant_int8_gemm_simd(
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
    const uint m_dim = int8_gemm_m(M);
    const uint n_dim = int8_gemm_n(N);
    const uint b_dim = int8_gemm_b(B);
    const uint r_dim = int8_gemm_r();
    const uint row0 = (tg_idx * simdgroups_per_tg + sg_idx) * r_dim;
    if (row0 >= m_dim) return;

    const uint n_groups  = n_dim / kInt8GroupSize;
    const uint row_bytes = n_dim;
    device const uint8_t* W_row = W      + uint(row0) * row_bytes;
    device const bfloat*  s_row = scales + uint(row0) * n_groups;
    device const bfloat*  b_row = biases + uint(row0) * n_groups;

    float acc[kInt8MaxRowBlock][kInt8MaxBatchRows];
    for (uint r = 0; r < r_dim; ++r) {
        for (uint i = 0; i < b_dim; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        // 8 bytes per lane x 32 lanes = 256 elements = 4 groups, the same
        // partition the INT4 kernel runs at half the byte width. The
        // loads are 2-byte-aligned pairs: resident tensors are 2-aligned
        // but not 4-aligned, so a `uint*` load is undefined.
        const uint byte_base = blk * 256u + lane * 8u;
        const uint g = blk * 4u + (lane >> 3);
        const uint elem = byte_base;

        float qv[kInt8MaxRowBlock][8];
        float sv[kInt8MaxRowBlock];
        float bv[kInt8MaxRowBlock];
        for (uint r = 0; r < r_dim; ++r) {
            if (row0 + r >= m_dim) {
                sv[r] = 0.0f;
                bv[r] = 0.0f;
                for (uint k = 0; k < 8; ++k) { qv[r][k] = 0.0f; }
                continue;
            }
            device const ushort* wp =
                (device const ushort*)(W_row + uint(r) * row_bytes + byte_base);
            sv[r] = float(s_row[uint(r) * n_groups + g]);
            bv[r] = float(b_row[uint(r) * n_groups + g]);
            // Eight weight bytes = four 2-byte-aligned ushort loads. Low
            // byte first: the resident layout is little-endian and the
            // element order within a group is byte order.
            qv[r][0] = float( wp[0]        & 0xFFu);
            qv[r][1] = float((wp[0] >> 8)  & 0xFFu);
            qv[r][2] = float( wp[1]        & 0xFFu);
            qv[r][3] = float((wp[1] >> 8)  & 0xFFu);
            qv[r][4] = float( wp[2]        & 0xFFu);
            qv[r][5] = float((wp[2] >> 8)  & 0xFFu);
            qv[r][6] = float( wp[3]        & 0xFFu);
            qv[r][7] = float((wp[3] >> 8)  & 0xFFu);
        }

        #pragma clang loop unroll_count(4)
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const half4 xa = *((device const half4*)(x_b + elem));
            const half4 xb = *((device const half4*)(x_b + elem + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
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
            const float x0 = float(x_b[g * kInt8GroupSize + lane * 2u]);
            const float x1 = float(x_b[g * kInt8GroupSize + lane * 2u + 1u]);
            const float sum = x0 + x1;
            for (uint r = 0; r < r_dim; ++r) {
                if (row0 + r >= m_dim) { continue; }
                const float s = float(s_row[uint(r) * n_groups + g]);
                const float b = float(b_row[uint(r) * n_groups + g]);
                // Two bytes = this lane's two elements of the group, the
                // INT8 twin of the INT4 tail's one byte holding two nibbles.
                device const ushort* wp =
                    (device const ushort*)(W_row + uint(r) * row_bytes
                                           + g * kInt8GroupSize + lane * 2u);
                const float lo = float(uint(wp[0] & 0xFFu));
                const float hi = float(uint(wp[0] >> 8));
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
