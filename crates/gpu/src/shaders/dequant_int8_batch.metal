// PORT-LOCAL, not vendored: the Swift engine decodes one token at a time
// and has no batched verify, so there is no upstream kernel to mirror --
// the same standing `dequant_int4_batch.metal` records, one width up.
//
// `dequant_int8_gemm_simd` keeps `dequant_int4_gemm_simd`'s outer shape: 8
// SIMD groups per threadgroup, R contiguous output rows per SIMD group, B
// right-hand sides with B baked as a function constant so the `bi` loop
// unrolls, `acc[R][B]` at the same two caps. The inner loop is NOT the INT4
// one: it walks the row group by group with each lane on two elements, as
// the INT8 GEMV does. An earlier version gave each lane 8 consecutive
// elements of a 256-element block, which summed in a different order and
// drifted from sequential decode on real weights.
//
// Like its sibling it is held bit-exact against the per-token GEMV rather
// than against a tolerance: B rows of this kernel must equal B separate
// `dequant_int8_gemv_resident` calls (which reduce through
// `turbospark_compute::dequant_int8_gemv`), and that is what the parity
// test asserts. First carrier: the dense humanizer Gemma 4 12B, whose
// every projection is INT8 affine at group 64.
//
// MEASUREMENTS: no sweep of its own. Unlike the INT4 sibling -- whose header
// is a record of swept unroll counts and a row-block table -- this kernel
// ships with the sibling's SHAPES (unroll_count(4), best_row_block's steps)
// and no numbers. The R table next door is a throughput choice
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

    // Group by group, two elements per lane: the GEMV's lane partition and
    // add order. A wider block load per lane (the INT4 kernel's shape) moves
    // both, and the last bits with them.
    for (uint g = 0; g < n_groups; ++g) {
        for (uint bi = 0; bi < b_dim; ++bi) {
            device const half* x_b = x + bi * n_dim;
            const float x0 = float(x_b[g * kInt8GroupSize + lane * 2u]);
            const float x1 = float(x_b[g * kInt8GroupSize + lane * 2u + 1u]);
            const float sum = x0 + x1;
            for (uint r = 0; r < r_dim; ++r) {
                if (row0 + r >= m_dim) { continue; }
                const float s = float(s_row[uint(r) * n_groups + g]);
                const float b = float(b_row[uint(r) * n_groups + g]);
                // Two bytes = this lane's two elements of the group, a
                // 2-byte-aligned load: resident tensors are not 4-aligned.
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
