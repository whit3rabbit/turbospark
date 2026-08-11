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
const THREADS_PER_GROUP: u64 = 256;
const ROWS_PER_THREADGROUP: usize = 8;

/// The kernel's accumulator array is a fixed-size register array, so this
/// is a hard precondition rather than a clamp. Sixteen covers every
/// published DFlash `block_size`.
pub const MAX_BATCH_ROWS: usize = 16;

fn unused_function_constants() -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 20);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 21);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 22);
    values
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
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_int4_gemm_simd",
        &unused_function_constants(),
        b"",
    )?;
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
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
