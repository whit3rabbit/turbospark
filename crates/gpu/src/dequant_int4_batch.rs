//! Host-side dispatch for `dequant_int4_gemm_simd`, the B-right-hand-side
//! form of the INT4 GEMV (ROADMAP Phase D2).
//!
//! PORT-LOCAL: the Swift engine decodes one token per forward pass, so
//! there is no upstream kernel and no vendoring question. Its only
//! contract is the GEMV's own -- B rows of this must equal B separate
//! `dequant_int4_gemv_resident` calls, which is what the parity test
//! asserts, and through that `turbospark_compute::dequant_int4_gemv`.
//!
//! Why it exists rather than looping the GEMV: measured in
//! `tests/gemv_bandwidth_bench.rs`, B dispatches of the GEMV over one
//! matrix cost 0.65B (the weights do not stay cached), and encoding them
//! concurrently recovers only 1.07-1.40x. The weight bytes really do move
//! B times. Amortizing them needs one dispatch holding B accumulators.

use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext};
use crate::dequant_int4_gemv::Int4ResidentMatrix;

/// Gotcha 4: one `&'static str` with one stable address, because the
/// pipeline cache keys on the address.
const SOURCE: &str = concat!(
    include_str!("shaders/dequant_int4.metal"),
    include_str!("shaders/dequant_int4_batch.metal")
);
/// A SECOND concatenation, and therefore a second `&'static str` with its
/// own stable address, which is what Gotcha 4's address-keyed cache needs.
/// The matrix kernel is kept out of `SOURCE` so a caller that never touches
/// it never compiles it.
const MMA_SOURCE: &str = concat!(
    include_str!("shaders/dequant_int4.metal"),
    include_str!("shaders/dequant_int4_mma.metal")
);
/// Public because `pipeline_reflection_cannot_see_this_kernels_register_pressure`
/// compares the compiled pipeline's own `maxTotalThreadsPerThreadgroup`
/// against it. That comparison is a HARD-FAILURE backstop and not a spill
/// gate: the metric reads 1024 on every shape of this kernel, including
/// deliberately impossible ones, so it cannot price register pressure. See
/// that test for the discrimination check that settled it.
pub const GEMM_THREADS_PER_GROUP: u64 = 256;
/// SIMD groups per threadgroup. Rows per threadgroup is this times the row
/// block, since a SIMD group owns `row_block` contiguous rows.
const SIMDGROUPS_PER_THREADGROUP: usize = 8;

/// The kernel's accumulator array is a fixed-size register array, so this
/// is a hard precondition rather than a clamp. Sixteen covers every
/// published DFlash `block_size`.
pub const MAX_BATCH_ROWS: usize = 16;

/// Output rows one SIMD group may own (`FC_GEMM_R`), mirroring
/// `kMaxRowBlock` in the shader.
///
/// **This is a THROUGHPUT axis and not a numerics one.** Every value
/// produces bit-identical output: a row's accumulation still walks the same
/// blocks in the same order into one FP32 accumulator and `simd_sum` still
/// reduces the same 32-lane partition of K, so only which SIMD group owns
/// the row changes. `dequant_int4_gemm_parity.rs` asserts that across the
/// whole `(row_block, batch)` grid rather than leaving it as an argument.
///
/// 1 was the shape every wired call site dispatched before the constant
/// existed; since the AC sweep of 2026-08-29 the width is chosen per batch
/// by [`best_row_block`], which is where the measurement lives. 4 is the cap
/// because `acc[R][B]` plus `qv[R][8]` is R*B + 8R floats of register array,
/// and this kernel's header is a record of what happens past ~128 live
/// floats; nothing above 4 has been measured.
pub const MAX_GEMM_ROW_BLOCK: usize = 4;

/// `dequant_int4_gemm_simd` declares `FC_GEMM_M` (100), `FC_GEMM_N` (101),
/// `FC_GEMM_B` (102) and `FC_GEMM_USE_FC` (103). Indices 20-26 belong to
/// `dequant_int4.metal`, which this source is concatenated onto, and are set
/// to their unspecialized values because the shared file declares them.
///
/// **B is the reason this exists.** The GEMV took the same treatment in
/// `46617c6` and this kernel did not, which moved every `c(M)` reading the
/// wrong way -- `c(M)` is the ratio of the two arms, so speeding only the
/// sequential one makes batching look worse (`docs/MTP_SPECULATIVE.md`).
/// Beyond restoring that, baking B lets the `for (bi < B)` loops unroll and
/// bounds the live accumulator set to the batch actually in flight, on a
/// kernel whose own header records the register file as the binding
/// constraint.
///
/// THE KEY MUST CARRY THE BAKED VALUES, for the reason `dequant_int4_gemv.rs`
/// spells out: `MetalContext::pipeline` caches on (source address, name,
/// key), so a shared key silently serves the first-compiled shape's pipeline
/// to every later one, and a wrong baked N reads the wrong fraction of every
/// row while producing finite, plausible output (crate Gotcha 1).
fn specialized_constants(m: u32, n: u32, b: u32) -> (FunctionConstantValues, [u8; 12]) {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let off = false;
    let on = true;
    // The concatenated GEMV file's own constants, left unspecialized.
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&off as *const bool).cast(), MTLDataType::Bool, 22);
    // This kernel's.
    values.set_constant_value_at_index((&m as *const u32).cast(), MTLDataType::UInt, 100);
    values.set_constant_value_at_index((&n as *const u32).cast(), MTLDataType::UInt, 101);
    values.set_constant_value_at_index((&b as *const u32).cast(), MTLDataType::UInt, 102);
    values.set_constant_value_at_index((&on as *const bool).cast(), MTLDataType::Bool, 103);
    let mut key = [0u8; 12];
    key[..4].copy_from_slice(&m.to_le_bytes());
    key[4..8].copy_from_slice(&n.to_le_bytes());
    key[8..].copy_from_slice(&b.to_le_bytes());
    (values, key)
}

/// The above plus `FC_GEMM_R` (104), for `dequant_int4_gemm_simd` alone.
///
/// **A SEPARATE FUNCTION RATHER THAN A FOURTH PARAMETER ON THE ONE ABOVE**,
/// because 104 is declared in `dequant_int4_batch.metal` and that file is
/// not part of `MMA_SOURCE`. Setting a value at an index the library does
/// not declare is at best ignored, and the matrix kernel's key would change
/// width for a constant it cannot read.
///
/// R joins the key for the reason the other three did. `MetalContext::pipeline`
/// caches on (source address, name, key), so an axis missing from the key is
/// served whichever shape compiled first -- and a pipeline baked for a
/// different R walks a different number of rows per SIMD group while
/// producing finite, plausible output (crate Gotcha 1).
fn specialized_constants_row_blocked(
    m: u32,
    n: u32,
    b: u32,
    r: u32,
) -> (FunctionConstantValues, [u8; 16]) {
    let (values, base) = specialized_constants(m, n, b);
    values.set_constant_value_at_index((&r as *const u32).cast(), MTLDataType::UInt, 104);
    let mut key = [0u8; 16];
    key[..12].copy_from_slice(&base);
    key[12..].copy_from_slice(&r.to_le_bytes());
    (values, key)
}

/// The above plus `FC_MMA_STAGE_X` (110), for `dequant_int4_gemm_mma` alone.
///
/// A separate function for `specialized_constants_row_blocked`'s reason, in
/// the other direction: 110 is declared in `dequant_int4_mma.metal`, which
/// is not part of the SIMD kernel's source, so setting it there would widen
/// a key for a constant that library cannot read.
fn specialized_constants_mma(
    m: u32,
    n: u32,
    b: u32,
    stage_x: bool,
    skip_dequant: bool,
) -> (FunctionConstantValues, [u8; 14]) {
    let (values, base) = specialized_constants(m, n, b);
    values.set_constant_value_at_index((&stage_x as *const bool).cast(), MTLDataType::Bool, 110);
    values.set_constant_value_at_index(
        (&skip_dequant as *const bool).cast(),
        MTLDataType::Bool,
        111,
    );
    let mut key = [0u8; 14];
    key[..12].copy_from_slice(&base);
    key[12] = u8::from(stage_x);
    key[13] = u8::from(skip_dequant);
    (values, key)
}

/// The measured-best `row_block` for a batch width.
///
/// **SELECTING A KERNEL SHAPE BY BATCH WIDTH IS SAFE HERE AND IS FORBIDDEN
/// ONE FILE OVER, so read the difference before copying either.**
/// `dequant_int4_mma.metal` says "never make this a fast path selected by a
/// heuristic -- that would make the generated bytes a function of a shape",
/// and that is correct OF THAT KERNEL: it reduces K in hardware, so it is
/// not bit-exact against the GEMV and picking it per shape would make output
/// depend on the batch width. `FC_GEMM_R` is bit-exact at EVERY width
/// (`row_blocking_does_not_move_a_single_bit`, on the hostile fixture with
/// the MMA positive control), so this table cannot move a byte. It is a
/// pure throughput lookup, which is exactly why the axis was built as a
/// function constant on one kernel rather than as a second kernel.
///
/// Measured on AC, 2026-08-29, three runs over three real shapes (gate/up
/// 17408x5120, down 5120x17408, packed_q 12288x5120), mean `c` per width:
///
/// |  M   |  R=1  |  R=2  |  R=4  | chosen | gain |
/// |---|---|---|---|---|---|
/// |  1   | 1.004 | 1.043 | 1.222 |   1    | --      |
/// |  2   | 0.539 | 0.504 | 0.650 |   2    | 1.07x   |
/// |  3   | 0.508 | 0.356 | 0.459 |   2    | 1.43x   |
/// |  4   | 0.623 | 0.313 | 0.374 |   2    | 1.99x   |
/// |  5   | 0.534 | 0.520 | 0.488 |   4    | 1.10x   |
/// |  8   | 0.504 | 0.456 | 0.414 |   4    | 1.22x   |
/// | 12   | 0.494 | 0.439 | 0.392 |   4    | 1.26x   |
/// | 16   | 0.493 | 0.427 | 0.370 |   4    | 1.33x   |
///
/// **THE M=1 ROW IS WHY THIS IS A TABLE AND NOT A CONSTANT.** At one row a
/// wider block is a straight LOSS (1.00 to 1.22), and R=4 loses at M=2 as
/// well, so a global `row_block = 4` would have regressed exactly the widths
/// the speculative verify runs at. The full 1..16 sweep is in
/// `docs/BATCHED_PREFILL.md`.
///
/// The steps sit at 2 and 5 because that is where the measurement puts them,
/// not because they are round. M=13 is the one width where R=2 and R=4 TIE
/// (0.449 each, nine samples apiece), so the step function costs nothing
/// there rather than being wrong there; encoding a one-width exception would
/// be fitting noise between two neighbours that both favour R=4.
pub fn best_row_block(batch: usize) -> usize {
    match batch {
        0 | 1 => 1,
        2..=4 => 2,
        _ => 4,
    }
}

/// `y[b, m] = sum_n W[m, n] * x[b, n]` for `b` in `0..batch`.
///
/// `x` holds `batch * w.cols` halfs and `y` `batch * w.rows`, both
/// TOKEN-MAJOR, so one token's slice of either is contiguous and can be
/// handed to a single-token kernel unchanged.
///
/// Dispatches at [`best_row_block`]'s width, which is a throughput choice
/// and provably not a numerics one; see that function for why a per-width
/// table is legitimate here and forbidden for the matrix kernel next door.
pub fn encode_dequant_int4_gemm_resident(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    encode_dequant_int4_gemm_resident_blocked(context, pass, w, x, y, batch, best_row_block(batch))
}

/// [`encode_dequant_int4_gemm_resident`] with `row_block` output rows per
/// SIMD group ([`MAX_GEMM_ROW_BLOCK`]).
///
/// **`row_block` cannot change the output**, so this is not a second
/// numerics path and no call site has to choose between two answers; see
/// [`MAX_GEMM_ROW_BLOCK`] for the argument and
/// `tests/dequant_int4_gemm_parity.rs` for the assertion.
///
/// Callers should prefer [`encode_dequant_int4_gemm_resident`], which picks
/// [`best_row_block`] for them. This one exists for the BENCH, which has to
/// dispatch a width the table did not choose in order to have measured the
/// table at all, and for the parity cases, which assert every width agrees
/// with the GEMV rather than only the chosen one.
///
/// Nothing selects a width automatically and nothing should until `c(R, B)`
/// has been measured on AC: the shipped `c(M)` row is flat in M, which is
/// what a compute-bound kernel looks like, and a width chosen off a
/// contaminated clock is a footprint cost for no throughput
/// (`crates/gpu/CLAUDE.md`, this file's bullet).
pub fn encode_dequant_int4_gemm_resident_blocked(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    row_block: usize,
) -> Result<(), GpuError> {
    assert_eq!(w.cols % 64, 0, "N must be a multiple of 64");
    assert!(w.rows > 0);
    assert!(
        (1..=MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MAX_BATCH_ROWS}"
    );
    assert!(
        (1..=MAX_GEMM_ROW_BLOCK).contains(&row_block),
        "row_block {row_block} outside 1..={MAX_GEMM_ROW_BLOCK}"
    );
    let (m, n, b, r) = (w.rows as u32, w.cols as u32, batch as u32, row_block as u32);
    let (constants, key) = specialized_constants_row_blocked(m, n, b, r);
    let pipeline = context.pipeline(SOURCE, "dequant_int4_gemm_simd", &constants, &key)?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[(u32_bytes(&m), 5), (u32_bytes(&n), 6), (u32_bytes(&b), 7)],
        gemm_threadgroups(w.rows, row_block),
        GEMM_THREADS_PER_GROUP,
    );
    Ok(())
}

/// Threadgroups for `rows` output rows at `row_block` rows per SIMD group.
///
/// **A NAMED FUNCTION BECAUSE NO PARITY TEST CAN SEE THIS ARITHMETIC GOING
/// WRONG IN THE DIRECTION IT ACTUALLY GOES WRONG.** Found by mutation,
/// 2026-08-29: replacing the `* row_block` here with nothing survived every
/// case in `dequant_int4_gemm_parity.rs`, including the row-block sweep. It
/// over-dispatches rather than under-dispatching -- the surplus threadgroups
/// compute a `row0` past `M` and return at the kernel's first branch -- so
/// the output stays bit-correct and only the COST moves, by a factor of
/// `row_block`.
///
/// That is the worst shape a bug can have here, because the whole reason
/// `row_block` exists is to be timed: a variant launching R times the
/// threadgroups it needs would read as "row blocking does not help" and the
/// axis would be closed on a measurement of the mistake. `gemm_threadgroups`
/// is asserted as arithmetic in this module's own tests instead, on the
/// precedent `steering.rs` set for its row partition.
fn gemm_threadgroups(rows: usize, row_block: usize) -> u64 {
    rows.div_ceil(SIMDGROUPS_PER_THREADGROUP * row_block) as u64
}

/// What the compiler made of a `dequant_int4_gemm_simd` shape, read off the
/// pipeline itself.
///
/// The register file is this kernel's binding constraint (its shader header
/// is a record of two optimizations that lost to it), and until now every
/// statement about it was an inference from a TIMING. These three numbers
/// are static: they need a Metal device but no model, no install and no
/// clean clock, which is what makes "does B=32 spill?" answerable in a
/// session that cannot benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPipelineLimits {
    /// Below [`GEMM_THREADS_PER_GROUP`] this shape cannot be dispatched at
    /// the width the encode function asks for. Apple lowers it when a
    /// kernel's register demand will not fit the threadgroup, so it is the
    /// spill signal.
    pub max_total_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
    pub static_threadgroup_memory_length: u64,
}

/// Compile one `(rows, cols, batch, row_block)` shape and report
/// [`GemmPipelineLimits`].
///
/// Goes through the same `specialized_constants_row_blocked` and
/// `MetalContext::pipeline` the dispatch does, so it cannot describe a
/// pipeline the encode function would not produce.
pub fn dequant_int4_gemm_pipeline_limits(
    context: &mut MetalContext,
    rows: usize,
    cols: usize,
    batch: usize,
    row_block: usize,
) -> Result<GemmPipelineLimits, GpuError> {
    let (constants, key) =
        specialized_constants_row_blocked(rows as u32, cols as u32, batch as u32, row_block as u32);
    let pipeline = context.pipeline(SOURCE, "dequant_int4_gemm_simd", &constants, &key)?;
    Ok(GemmPipelineLimits {
        max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
        thread_execution_width: pipeline.thread_execution_width(),
        static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
    })
}

/// The 8x8 output tile one SIMD group owns in the matrix-hardware kernel.
const MMA_TILE: usize = 8;

/// The matrix kernel's own batch cap, FOUR TIMES the SIMD kernel's.
///
/// `MAX_BATCH_ROWS` is 16 because that kernel's accumulators are a
/// per-thread `float[]` register array. This one accumulates into
/// `simdgroup_matrix` tiles spread across the SIMD group -- about four
/// registers per lane per tile -- so the same constraint does not apply,
/// and the widths where matrix hardware could plausibly pay are well above
/// 16. Kept as a separate constant rather than raising the shared one,
/// which would be a claim about the SIMD kernel that is not true.
pub const MMA_MAX_BATCH_ROWS: usize = 64;
/// One SIMD group per threadgroup; see the shader's header for why.
const MMA_THREADS_PER_GROUP: u64 = 32;

/// The `simdgroup_matrix` form of [`encode_dequant_int4_gemm_resident`].
///
/// **NOT BIT-EXACT AGAINST THE GEMV, and that is inherent rather than a
/// tolerance to be tightened.** `simdgroup_multiply_accumulate` reduces its
/// K dimension in hardware in an undocumented order, where both the GEMV and
/// the SIMD GEMM walk K in a fixed sequence into one FP32 accumulator. A
/// verify pass built on this therefore cannot claim that speculative output
/// is identical to non-speculative output (AGENTS.md Gotcha 27), and a
/// chunked prefill built on it fails `docs/BATCHED_PREFILL.md`'s stated bar.
///
/// It exists so the cost of that trade is a measured number rather than an
/// argument. `encode_dequant_int4_gemm_resident` remains the default and
/// nothing selects between them automatically; see
/// `docs/MTP_SPECULATIVE.md` for what it buys.
///
/// `x` must be sized for a whole number of 8-token column tiles
/// (`ceil(batch / 8) * cols` halfs), because the kernel loads its
/// right-hand side a tile at a time. `y` needs only `batch * rows`.
pub fn encode_dequant_int4_gemm_mma_resident(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    encode_dequant_int4_gemm_mma_resident_staged(context, pass, w, x, y, batch, false)
}

/// **DIAGNOSTIC ONLY: fills the weight tile with a constant instead of
/// unpacking it, so `y` IS MEANINGLESS.**
///
/// It exists to split this kernel's cost in two by deletion. The header's
/// account of the plateau (dequant work independent of B) is refuted by
/// arithmetic, and total dequant work is `N * K` in both this engine and
/// MLX, so what remains unidentified is whether the dequant inner loop or
/// the matrix path dominates. Removing the unpack while leaving every
/// barrier, `simdgroup_load` and `simdgroup_multiply_accumulate` in place
/// times the matrix path alone: near the full kernel means the dequant is
/// not the cost, far below means it is.
///
/// Nothing but `gemv_bandwidth_bench.rs` may call this, and
/// `dequant_int4_mma_parity.rs` asserts its output DIFFERS from the real
/// kernel -- an unreachable diagnostic constant would silently time the
/// unmodified kernel twice and read as evidence.
pub fn encode_dequant_int4_gemm_mma_resident_skip_dequant(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    encode_mma(context, pass, w, x, y, batch, false, true)
}

/// The same, with `x` optionally staged through threadgroup memory.
///
/// **THE TWO ARMS ARE BIT-IDENTICAL TO EACH OTHER** (asserted in
/// `dequant_int4_mma_parity.rs`): same values, same `simdgroup_load` order,
/// same accumulate sequence, only a different place to read the bytes from.
/// That is a stronger claim than this kernel's tolerance against the GEMV
/// and separate from it -- the tolerance is about hardware K reduction, this
/// is about a memory path.
///
/// It exists to price the sharpest structural difference between this kernel
/// and MLX's `qmm_t_impl`, which stages both operands. See the shader header
/// and `docs/BENCHMARKS.md`, "The reference curve, measured rather than
/// inferred". `stage_x` is a caller's parameter and never a heuristic on
/// `batch`, for the reason `best_row_block` spells out: this kernel is not
/// bit-exact against the GEMV, so selecting its shape by a runtime width
/// would make generated bytes a function of that width.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_int4_gemm_mma_resident_staged(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    stage_x: bool,
) -> Result<(), GpuError> {
    encode_mma(context, pass, w, x, y, batch, stage_x, false)
}

#[allow(clippy::too_many_arguments)]
fn encode_mma(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
    stage_x: bool,
    skip_dequant: bool,
) -> Result<(), GpuError> {
    assert_eq!(w.cols % 64, 0, "N must be a multiple of 64");
    assert!(w.rows > 0);
    assert!(
        (1..=MMA_MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MMA_MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    // Baked for the same reason as the sibling, and keyed the same way. The
    // staging flag joins the key: a shared one would hand back whichever
    // arm compiled first (crate Gotcha 1), which in an A/B would silently
    // measure one shape twice.
    let (constants, key) = specialized_constants_mma(m, n, b, stage_x, skip_dequant);
    let pipeline = context.pipeline(MMA_SOURCE, "dequant_int4_gemm_mma", &constants, &key)?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[(u32_bytes(&m), 5), (u32_bytes(&n), 6), (u32_bytes(&b), 7)],
        w.rows.div_ceil(MMA_TILE) as u64,
        MMA_THREADS_PER_GROUP,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is covered, and NO threadgroup is launched that has no row
    /// to do. The second half is the one a parity test cannot see: an
    /// over-dispatch is bit-correct and `row_block` times too expensive
    /// (see `gemm_threadgroups`).
    ///
    /// Mutation-checked: dropping `* row_block` from `gemm_threadgroups`
    /// reddens the surplus assertion at every `row_block > 1` and leaves the
    /// coverage one green, which is exactly the asymmetry that let the
    /// mutation survive the GPU suite.
    #[test]
    fn a_dispatch_covers_every_row_and_launches_no_idle_threadgroup() {
        for row_block in 1..=MAX_GEMM_ROW_BLOCK {
            let per_group = SIMDGROUPS_PER_THREADGROUP * row_block;
            // Exact multiples, one over, one under, plus the real model's
            // four shapes' row counts.
            for rows in [
                1,
                7,
                per_group - 1,
                per_group,
                per_group + 1,
                72,
                1024,
                5120,
                12288,
                17408,
            ] {
                let groups = gemm_threadgroups(rows, row_block) as usize;
                assert!(
                    groups * per_group >= rows,
                    "rows {rows} row_block {row_block}: {groups} groups cover only \
                     {} rows, so {} rows are never written",
                    groups * per_group,
                    rows - groups * per_group
                );
                assert!(
                    groups.saturating_sub(1) * per_group < rows,
                    "rows {rows} row_block {row_block}: {groups} groups, but {} would \
                     already cover them. The surplus returns at the kernel's first \
                     branch, so the output is correct and the dispatch costs {}x too \
                     much -- which would read as `row blocking does not help`",
                    groups - 1,
                    groups as f64 / (groups - 1).max(1) as f64
                );
            }
        }
    }

    /// The two caps must agree with the shader's `constexpr` bounds, which
    /// are the compile-time sizes of `acc[kMaxRowBlock][kMaxBatchRows]`. A
    /// host cap ABOVE the shader's writes past a register array; the host
    /// asserts its own value, so nothing else would catch it.
    #[test]
    fn the_host_caps_match_the_shaders_array_bounds() {
        assert!(SOURCE.contains(&format!(
            "constant constexpr uint kMaxBatchRows = {MAX_BATCH_ROWS};"
        )));
        assert!(SOURCE.contains(&format!(
            "constant constexpr uint kMaxRowBlock = {MAX_GEMM_ROW_BLOCK};"
        )));
    }
}

#[cfg(test)]
mod row_block_choice_tests {
    use super::*;

    /// The chosen width must never be one the MEASUREMENT says is a loss,
    /// and the two losses are both at the narrow end: R=4 at M=1 (1.22
    /// against R=1's 1.00) and at M=2 (0.650 against R=2's 0.504). A global
    /// `row_block = 4` -- the obvious reading of "R=4 is best" off the M=16
    /// row -- would have shipped both, and the widths it would have
    /// regressed are exactly the ones a speculative verify runs at
    /// (`docs/DFLASH2.md`'s published block sizes are 2 to 8).
    ///
    /// Pinned as a table rather than as a range check because the steps are
    /// measurements, not round numbers.
    #[test]
    fn the_chosen_width_is_never_one_the_measurement_calls_a_loss() {
        let expected = [
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (12, 4),
            (16, 4),
        ];
        for (batch, want) in expected {
            assert_eq!(
                best_row_block(batch),
                want,
                "batch {batch} chose {} against the measured best {want}",
                best_row_block(batch)
            );
        }
    }

    /// Whatever the table says, it must be dispatchable: the host asserts
    /// `1..=MAX_GEMM_ROW_BLOCK` and would panic mid-encode otherwise.
    #[test]
    fn every_reachable_batch_chooses_a_dispatchable_width() {
        for batch in 1..=MAX_BATCH_ROWS {
            let r = best_row_block(batch);
            assert!(
                (1..=MAX_GEMM_ROW_BLOCK).contains(&r),
                "batch {batch} chose row_block {r}, outside 1..={MAX_GEMM_ROW_BLOCK}"
            );
        }
    }
}
