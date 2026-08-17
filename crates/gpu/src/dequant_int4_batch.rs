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
const THREADS_PER_GROUP: u64 = 256;
const ROWS_PER_THREADGROUP: usize = 8;

/// The kernel's accumulator array is a fixed-size register array, so this
/// is a hard precondition rather than a clamp. Sixteen covers every
/// published DFlash `block_size`.
pub const MAX_BATCH_ROWS: usize = 16;

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

/// `y[b, m] = sum_n W[m, n] * x[b, n]` for `b` in `0..batch`.
///
/// `x` holds `batch * w.cols` halfs and `y` `batch * w.rows`, both
/// TOKEN-MAJOR, so one token's slice of either is contiguous and can be
/// handed to a single-token kernel unchanged.
pub fn encode_dequant_int4_gemm_resident(
    context: &mut MetalContext,
    pass: &crate::context::PassEncoder,
    w: &Int4ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert_eq!(w.cols % 64, 0, "N must be a multiple of 64");
    assert!(w.rows > 0);
    assert!(
        (1..=MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    let (constants, key) = specialized_constants(m, n, b);
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
        w.rows.div_ceil(ROWS_PER_THREADGROUP) as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
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
    assert_eq!(w.cols % 64, 0, "N must be a multiple of 64");
    assert!(w.rows > 0);
    assert!(
        (1..=MMA_MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MMA_MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    // Baked for the same reason as the sibling, and keyed the same way.
    let (constants, key) = specialized_constants(m, n, b);
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
