//! Host-side dispatch for `dequant_int2_gemm_simd` (ROADMAP P3.3): the 2-bit
//! GEMV with B right-hand sides, the 1-bit batch wrapper's structure at the
//! ternary checkpoint's width.
//!
//! The kernel's header carries the design and parity arguments. What this
//! wrapper owns is the same pair the 1-bit one does: the pipeline-cache key
//! carries the baked `B` (crate Gotcha 1), and the batch cap is an `assert!`
//! because `acc[B]` is a per-thread register array.

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::dequant_2bit_gemv::{int2_row_bytes, Int2ResidentMatrix};
use crate::dequant_int4_batch::MAX_BATCH_ROWS;

const SOURCE: &str = include_str!("shaders/dequant_2bit.metal");
const THREADS_PER_GROUP: u64 = 256; // 8 rows/threadgroup * 32 lanes/SIMD group.
const ROWS_PER_THREADGROUP: u64 = 8;

/// The kernel's `FC_GEMM2_M` (100), `FC_GEMM2_N` (101), `FC_GEMM2_B` (102)
/// and `FC_GEMM2_USE_FC` (103), with the key carrying all three baked values
/// for `MetalContext::pipeline`'s cache (crate Gotcha 1).
fn specialized_constants(m: u32, n: u32, b: u32) -> (FunctionConstantValues, [u8; 12]) {
    let values = FunctionConstantValues::new();
    let on = true;
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

/// `y[b, m] = sum_n W[m, n] * x[b, n]` for `b` in `0..batch`, over 2-bit
/// affine rows: bit-exact against `batch` calls of
/// [`crate::encode_dequant_int2_gemv_resident`] at every width, which
/// `crates/gpu/tests/dequant_2bit_gemm_parity.rs` asserts.
pub fn encode_dequant_int2_gemm_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Int2ResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
    batch: usize,
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    assert_eq!(
        int2_row_bytes(w.cols),
        w.cols / 4,
        "the batch kernel shares the GEMV's row layout"
    );
    assert!(
        (1..=MAX_BATCH_ROWS).contains(&batch),
        "batch {batch} outside 1..={MAX_BATCH_ROWS}"
    );
    let (m, n, b) = (w.rows as u32, w.cols as u32, batch as u32);
    let (constants, key) = specialized_constants(m, n, b);
    let pipeline = context.pipeline(SOURCE, "dequant_int2_gemm_simd", &constants, &key)?;
    let threadgroups = w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (w.buffer, 1, w.scales_offset),
            (w.buffer, 2, w.biases_offset),
            (x.0, 3, x.1),
            (y.0, 4, y.1),
        ],
        &[
            (u32_bytes(&m), 5),
            (u32_bytes(&n), 6),
            (u32_bytes(&(w.group_size as u32)), 7),
            (u32_bytes(&b), 8),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// One-shot [`encode_dequant_int2_gemm_resident`] for the parity tests.
pub fn dequant_int2_gemm_resident(
    context: &mut MetalContext,
    w: &Int2ResidentMatrix<'_>,
    x: &[f16],
    batch: usize,
) -> Result<Vec<f16>, GpuError> {
    assert_eq!(x.len(), w.cols * batch);
    let x_buffer = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y_buffer = context.new_output_buffer((w.rows * batch * std::mem::size_of::<u16>()) as u64);
    let pass = context.begin_pass();
    encode_dequant_int2_gemm_resident(context, &pass, w, (&x_buffer, 0), (&y_buffer, 0), batch)?;
    pass.commit_and_wait();
    Ok(read_half_buffer(&y_buffer, w.rows * batch))
}
