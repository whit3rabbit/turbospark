// PORT-LOCAL, AND A MEASURED DEAD END. Kept, with its bench, because
// `simdgroup_matrix` is the lever every reader reaches for next and
// `docs/SPECULATIVE_DECODING.md` named it as "the one remaining lever".
// It is not one.
//
// Measured 2026-08-17 against the SIMD sibling, same session, same
// matrices, `c(M)` on gate/up 17408x5120 (and the same shape on every
// other row of QWEN38_SHAPES):
//
//   M      exact   matrix   matrix/exact
//   2      0.52     3.39      6.56x slower
//   4      0.58     1.69      2.92x
//   8      0.48     0.85      1.76x
//   16     0.47     0.62      1.33x
//   32       --     0.52      past the SIMD kernel's cap
//   64       --     0.58      past the SIMD kernel's cap
//
// **IT LOSES AT EVERY WIDTH, AND THE PLATEAU IS THE FINDING.** Past M=16 it
// stops improving at ~0.5 -- worse than what the exact kernel already
// reaches at M=16 -- so this is not an implementation that more batching
// would rescue. Two hypotheses were tested and killed on the way: widening
// the staged K block from 8 to 64 (cutting barriers eightfold) changed
// nothing, and running past the SIMD kernel's register cap to M=32 and 64
// changed nothing.
//
// The reason it cannot work here: a packed INT4 run cannot be
// `simdgroup_load`ed, so every weight element must be dequantized into
// threadgroup memory first. That work is proportional to rows x K and is
// INDEPENDENT OF B, while the MAC work matrix hardware accelerates is
// proportional to rows x K x B. At the widths this engine has, the kernel
// is dequant-bound, and matrix hardware is accelerating the term that is
// not the cost. It would take weights that are already in a loadable
// format, not a better tiling.
//
// **THAT LAST SENTENCE IS SCOPED TO THIS TILE, AND THE SCOPE WAS NOT
// STATED UNTIL 2026-08-29.** Every number above was taken at
// `kMmaTile = 8` with ONE SIMD group, and "independent of B" is a property
// of that shape rather than of the approach. MLX's production kernel for
// the same operation
// (`qmm_t_impl`, `mlx/backend/metal/kernels/quantized.h`) is the same
// algorithm at a different shape -- `WM = WN = 2` so 128 threads and FOUR
// SIMD groups, `BM` 16 to 128, and BOTH operands staged into threadgroup
// memory through `BlockLoader` / `QuantizedBlockLoader` before the
// `BlockMMA`. This kernel stages only the weights and `simdgroup_load`s
// `x` TRANSPOSED STRAIGHT FROM DEVICE, once per `(n0, kt)`, which is a
// strided device gather in the innermost loop.
//
// So the honest verdict is "this tile loses", not "matrix hardware
// loses". `scripts/mlx_qmm_reference.py` measures what the other shape
// reaches on this machine: `c` = 0.145 at M=32 on gate/up, flat to M=512,
// against this port's 0.375 at M=16 (`docs/BENCHMARKS.md`, "The reference
// curve, measured rather than inferred"). That is 3.0x, of which 2.0x is
// width this port cannot reach at `MAX_BATCH_ROWS = 16` and 1.50x is the
// kernel at equal width. Re-opening the line means changing the shape --
// stage `x`, widen the threadgroup, raise `kMmaTile`. Nothing above refutes
// that. **What HAS been tried is the first of the three ON ITS OWN, and the
// result below says that was the wrong experiment**: measure them together,
// not separately.
//
// The bit-exactness objection below is UNCHANGED by any of that and is the
// reason a winning re-tile would still be prefill-only.
//
// **THE FIRST OF THE THREE WAS BUILT AND MEASURED, 2026-08-29, AND IT IS A
// LARGE LOSS: staging `x` costs 3.3x to 5.9x.** `FC_MMA_STAGE_X` (110)
// reads each `x` row once per `n0` block into threadgroup memory, coalesced,
// instead of `simdgroup_load`ing it transposed from device once per
// `(n0, kt)`. That was the cheapest of the three deltas against MLX's shape
// and the one predicted most likely to be the cause. It is not the cause,
// and the prediction was backwards.
//
//   M      mma    mma+stageX   staged/plain      (gate/up 17408x5120)
//   2     3.54x     12.49x        3.52x
//   4     1.78x      6.27x        3.52x
//   8     0.89x      3.19x        3.58x
//   16    0.66x      2.53x        3.81x
//   32    0.54x      2.25x        4.14x
//   64    0.52x      2.71x        5.19x
//
// Every shape agrees to within a few tenths and the penalty GROWS with B.
// `c_of_m_matrix_staged_against_unstaged` in `gemv_bandwidth_bench.rs` is
// the table; the arms are bit-identical (`dequant_int4_mma_parity.rs`,
// mutation-checked), so this is pure throughput.
//
// **WHAT IT TEACHES IS THAT THE THREE DELTAS ARE NOT INDEPENDENT.** A
// transposed `simdgroup_load` from device is not the naive strided gather it
// reads as -- Apple's tile load handles it -- while hand-staging the same
// bytes through 32 LANES is a serial copy of `col_tiles * kMmaTile * kMmaK`
// halfs per n-block, and at B=64 that is 10,240 loads per lane over the
// whole K walk. MLX stages `x` too and wins, because it has 128 threads
// doing it and a `BM` of 32 to 128 to amortize it over. So staging is a
// consequence of the wider threadgroup rather than a separate lever, and
// trying it alone was the wrong experiment. The remaining deltas -- four
// SIMD groups, and a wider `kMmaTile` -- have to move TOGETHER or not at
// all, and neither has been tried.
//
// **AND THE DEQUANT-AMORTIZATION STORY IS REFUTED BY ARITHMETIC, which is
// worth stating because it was this file's own explanation and it was the
// obvious next lever.** One threadgroup here dequantizes `8 * N` elements
// (`N / kMmaK` blocks of `kMmaTile * kMmaK`) to produce `8 * B` outputs, so
// dequant per output is `N / B` -- a function of the TOKEN count, not of
// the weight rows per threadgroup. MLX's is `K / BM`, with `BM` also the
// token tile. **They are the same number at the same width**: 320 at 16
// tokens, 80 at 64. This kernel ALREADY runs at B=64, already sits at that
// 80, and still reads 0.52 against MLX's 0.147. So amortization is not the
// differentiator and raising `kMmaTile` cannot be the lever -- it scales
// dequant and outputs together and changes the ratio not at all.
//
// **AND THE DEQUANT IS NOT THE COST EITHER. MEASURED BY DELETION,
// 2026-08-29.** `FC_MMA_SKIP_DEQUANT` (111) fills the weight tile with a
// constant and leaves every barrier, `simdgroup_load` and
// `simdgroup_multiply_accumulate` in place, so what it times is the matrix
// path with the unpack removed (`c_of_m_matrix_with_and_without_the_dequant`):
//
//   M     mma    mma-no-dequant   nodq/mma      (gate/up 17408x5120)
//   2    3.59x       2.30x          0.64
//   8    0.90x       0.58x          0.65
//   16   0.68x       0.51x          0.76
//   32   0.56x       0.48x          0.86
//   64   0.56x       0.50x          0.89
//
// The dequant is 36% of this kernel at M=2 and **11% at M=64** -- its share
// SHRINKS as B grows, which is the opposite of the header's original story
// and consistent with the arithmetic above. All six shapes agree to 0.03.
//
// **THE DECISIVE NUMBER IS THE MIDDLE COLUMN, NOT THE RATIO: with the
// dequant entirely FREE this kernel still reads 0.46 to 0.50 past M=16,
// against MLX's 0.145.** So a perfect loader -- vectorized reads, hoisted
// scale lookups, anything -- leaves it 3.2x behind, and
// `QuantizedBlockLoader` is not what to copy. It is also still SLOWER with a
// free dequant (0.51 at M=16) than the plain scalar `dequant_int4_gemm_simd`
// is with a real one (0.38), which is the sharpest statement of the problem.
//
// What is left is the matrix path itself: ONE SIMD group per threadgroup,
// `simdgroup_barrier` twice per 64-element K block, and eight
// `simdgroup_float8x8` accumulators owned by 32 lanes. That is where a
// re-tile has to go, and the loader and `kMmaTile` are both settled dead
// ends now rather than untried levers.
//
// So the exact kernel still wins on BOTH axes: it is faster AND it is
// bit-identical to a sequential decode. Nothing dispatches this one.
//
// `dequant_int4_gemm_mma` is the matrix-hardware form of
// `dequant_int4_gemm_simd`, and it is a SEPARATE KERNEL rather than a
// function-constant variant for a reason that is not style:
//
//   **IT IS NOT BIT-EXACT AGAINST THE GEMV, BY CONSTRUCTION.**
//
// `simdgroup_multiply_accumulate` reduces its K dimension in hardware, in
// an order Apple does not document and this port cannot control, where both
// the GEMV and the SIMD GEMM walk K in a fixed sequence into one FP32
// accumulator. So a verify pass built on this kernel would stop agreeing
// bit-for-bit with a sequential decode, and speculative output would stop
// being provably identical to non-speculative output -- the property
// AGENTS.md Gotcha 27 exists to protect, the one `accept_length_probe.rs`
// is written around, and the bar `docs/BATCHED_PREFILL.md` sets for a
// chunked prefill ("byte-identity is the bar, not coherence").
//
// Keeping both kernels is therefore the whole design: the exact one stays
// the default and this one is reachable only from a caller that has
// accepted the trade in writing. Never make this a fast path selected by a
// heuristic -- that would make the generated bytes a function of a shape.
//
// Concatenated AFTER dequant_int4.metal (crate Gotcha 4) for kGroupSize.

#include <metal_simdgroup_matrix>

// One SIMD group owns an 8-row x 8-column output tile and there is exactly
// ONE SIMD group per threadgroup. That is deliberate: the dequantized
// weight tile has to be staged in threadgroup memory (a packed INT4 run
// cannot be `simdgroup_load`ed directly), and with one SIMD group the
// staging needs only `simdgroup_barrier`, not `threadgroup_barrier`. The
// header of the SIMD kernel next door records that full threadgroup
// barriers across eight SIMD groups cost more than the staging saves; this
// shape is what avoids re-learning that.
constant constexpr uint kMmaTile = 8;
// Eight column tiles covers B up to 64. The SIMD sibling stops at 16
// because its accumulators are a per-thread register array; this kernel's
// are `simdgroup_matrix` accumulators spread across the SIMD group, ~4
// registers per lane per tile, so it is not bound the same way. That
// difference is the reason to measure it past 16: see MMA_MAX_BATCH_ROWS.
constant constexpr uint kMmaMaxColTiles = 8;
// K elements staged per barrier pair. **This is the whole cost model of
// this kernel.** A packed INT4 run cannot be `simdgroup_load`ed, so the
// weight tile has to be dequantized into threadgroup memory first, and the
// first draft staged only kMmaTile (8) at a time -- 640 staging rounds and
// 1280 barriers per output tile at N=5120, which made it 5.5x SLOWER than
// the plain SIMD kernel at M=2. Staging kMmaK at once amortizes the
// barriers over kMmaK/8 matrix multiplies.
constant constexpr uint kMmaK = 64;
constant constexpr uint kMmaKTiles = kMmaK / kMmaTile;

// **STAGE `x` THROUGH THREADGROUP MEMORY (110).** The un-staged form
// `simdgroup_load`s `x` TRANSPOSED STRAIGHT FROM DEVICE with row stride N,
// once per `(n0, kt)` -- a strided device gather in the innermost loop, and
// the sharpest structural difference between this kernel and MLX's
// `qmm_t_impl`, which stages BOTH operands (`BlockLoader` for x,
// `QuantizedBlockLoader` for w) before its `BlockMMA`. Staged, the same rows
// are read once per `n0` block, coalesced along n (x is [B, N] row-major, so
// a row's `kMmaK` run is contiguous), and the transpose happens out of
// threadgroup memory where it is free.
//
// It is a FUNCTION CONSTANT rather than a replacement so both shapes live in
// one binary and can be interleaved pair by pair in ONE process, which is
// the only way a few-percent difference is readable here (AGENTS.md Gotchas
// 22 and 28). Its byte is in the pipeline-cache key; a shared key would hand
// back whichever compiled first (crate Gotcha 1).
//
// **IT DOES NOT CHANGE THE ARITHMETIC.** Same values, same `simdgroup_load`
// order, same `simdgroup_multiply_accumulate` sequence -- only where the
// bytes are read from. So the two arms must agree BIT FOR BIT with each
// other, which `dequant_int4_mma_parity.rs` asserts. That is a stronger
// claim than this kernel's tolerance against the GEMV and is deliberately
// separate from it: the tolerance is about hardware K reduction, this is
// about a memory path.
constant bool FC_MMA_STAGE_X [[function_constant(110)]];

// **SKIP THE DEQUANT (111). A DIAGNOSTIC, AND IT PRODUCES WRONG NUMBERS BY
// DESIGN.** The header above ends at an unidentified mechanism: total
// dequant work is `N * K` in both engines and MLX spreads it over fewer
// threads, so neither amortization nor parallelism explains a 3.5x. The two
// live candidates are the dequant inner loop's own efficiency and the
// matrix side's register reuse, and they are separable by DELETION: fill
// the weight tile with a constant instead of unpacking it, leaving every
// barrier, every `simdgroup_load` and every
// `simdgroup_multiply_accumulate` exactly where they were. What remains is
// the matrix path's cost with the dequant removed.
//
// Near the full kernel's time means the dequant is NOT the cost and the
// matrix path is; far below it means the opposite. This is the one
// measurement that tells a re-tiling attempt which half to change, and it
// is cheaper than either rewrite.
//
// It is NEVER dispatched by anything but the bench. `y` is meaningless
// under it, which `dequant_int4_mma_parity.rs` asserts rather than leaves
// implicit -- an unreachable diagnostic constant would read as measured
// evidence while measuring the unmodified kernel twice.
constant bool FC_MMA_SKIP_DEQUANT [[function_constant(111)]];

inline bool mma_skip_dequant() {
    return is_function_constant_defined(FC_MMA_SKIP_DEQUANT) && FC_MMA_SKIP_DEQUANT;
}

inline bool mma_stage_x() {
    return is_function_constant_defined(FC_MMA_STAGE_X) && FC_MMA_STAGE_X;
}

kernel void dequant_int4_gemm_mma(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        B      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row0 = tg_idx * kMmaTile;
    if (row0 >= M) return;

    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    const uint col_tiles = (B + kMmaTile - 1u) / kMmaTile;

    // Staging for the dequantized weight tile, and for the result on the
    // way out. The result staging is not optional: `y` is [B, M] and a
    // direct transposed store would write all eight batch rows, running
    // past the end of a caller's buffer whenever B is not a multiple of 8.
    threadgroup half w_tile[kMmaTile * kMmaK];
    // FLOAT, matching the accumulator: `simdgroup_store` deduces one type
    // for the matrix and the destination, so a half staging tile does not
    // compile against a `simdgroup_float8x8`. The narrowing to `y`'s half
    // happens in the copy-out below, which is where the GEMV does it too.
    threadgroup float y_tile[kMmaTile * kMmaTile];
    // Sized for the widest B this kernel accepts (kMmaMaxColTiles tiles of
    // kMmaTile rows) because threadgroup arrays need a compile-time bound;
    // only `col_tiles * kMmaTile` rows are ever written or read. 64 x 64
    // halfs is 8 KiB, well inside Metal's 32 KiB.
    threadgroup half x_tile[kMmaMaxColTiles * kMmaTile * kMmaK];

    simdgroup_float8x8 acc[kMmaMaxColTiles];
    for (uint t = 0; t < kMmaMaxColTiles; ++t) {
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Each lane dequantizes kMmaTile * kMmaK / 32 of the staged elements.
    constexpr uint per_lane = (kMmaTile * kMmaK) / 32u;

    for (uint n0 = 0; n0 < N; n0 += kMmaK) {
        // Dequantize W[row0 .. row0+8, n0 .. n0+kMmaK] into threadgroup
        // memory, row-major with stride kMmaK. A row past M is filled with
        // zeros rather than skipped, so the multiply stays uniform across
        // the SIMD group.
        for (uint slot = 0; slot < per_lane; ++slot) {
            const uint e = lane * per_lane + slot;
            const uint m = e / kMmaK;
            const uint k = e % kMmaK;
            const uint row = row0 + m;
            half value = 0.0h;
            if (mma_skip_dequant()) {
                // Same store, same tile, no unpack: isolates the matrix
                // path. Deliberately not `0.0h`, so a compiler cannot fold
                // the multiply away and report a floor that is really a
                // deleted kernel.
                value = half(0.5h);
            } else if (row < M) {
                const uint n = n0 + k;
                const uint8_t byte = W[uint(row) * row_bytes + (n >> 1)];
                const uint q = (n & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
                const uint g = n / kGroupSize;
                const float s = float(scales[uint(row) * n_groups + g]);
                const float b = float(biases[uint(row) * n_groups + g]);
                value = half(fma(float(q), s, b));
            }
            w_tile[e] = value;
        }
        // Stage this n-block's slice of `x` alongside the weight tile, so
        // the inner loop reads no device memory at all. Coalesced: adjacent
        // lanes take adjacent k within one row, and a row's kMmaK run is
        // contiguous in [B, N]. The rows past B are the same ones the
        // un-staged arm reads through `col_tiles`, which the caller sizes
        // for, so both arms touch the identical bytes.
        if (mma_stage_x()) {
            const uint staged = col_tiles * kMmaTile * kMmaK;
            for (uint e = lane; e < staged; e += 32u) {
                const uint b = e / kMmaK;
                const uint k = e % kMmaK;
                x_tile[e] = x[b * N + n0 + k];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < kMmaKTiles; ++kt) {
            simdgroup_half8x8 w_mat;
            simdgroup_load(w_mat, w_tile + kt * kMmaTile, kMmaK);
            for (uint t = 0; t < col_tiles; ++t) {
                // `x` is [B, N]; the tile at (t*8, n0 + kt*8) with row
                // stride N is [8 batch][8 k], and the multiply wants
                // [8 k][8 batch], so it is loaded transposed. Rows past B
                // are not read: `col_tiles` is ceil(B/8) and the caller
                // sizes `x` for a whole number of tiles, which the host
                // documents and the test honours.
                simdgroup_half8x8 x_mat;
                if (mma_stage_x()) {
                    // Same tile, same transpose, out of threadgroup memory:
                    // row stride is kMmaK here rather than N.
                    simdgroup_load(x_mat,
                                   x_tile + uint(t) * kMmaTile * kMmaK
                                          + kt * kMmaTile,
                                   kMmaK,
                                   ulong2(0, 0),
                                   true);
                } else {
                    simdgroup_load(x_mat,
                                   x + uint(t) * kMmaTile * N + n0 + kt * kMmaTile,
                                   N,
                                   ulong2(0, 0),
                                   true);
                }
                simdgroup_multiply_accumulate(acc[t], w_mat, x_mat, acc[t]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = 0; t < col_tiles; ++t) {
        // Store transposed so the staging tile is [8 batch][8 row], which
        // is `y`'s own order, then copy out only the rows and columns that
        // exist.
        simdgroup_store(acc[t], y_tile, kMmaTile, ulong2(0, 0), true);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < kMmaTile * kMmaTile; e += 32u) {
            const uint b = t * kMmaTile + e / kMmaTile;
            const uint m = row0 + e % kMmaTile;
            if (b < B && m < M) {
                y[b * M + m] = half(y_tile[e]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
