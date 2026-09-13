//! Host dispatch for GGUF Q2_K resident GEMV.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};

const SOURCE: &str = concat!(
    include_str!("shaders/dequant_q4_k.metal"),
    "\n",
    include_str!("shaders/dequant_q2_k.metal"),
);
const THREADS_PER_GROUP: u64 = 256;
const ROWS_PER_THREADGROUP: u64 = 8;
pub const Q2_K_BLOCK_ELEMS: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = 84;

fn no_function_constants() -> FunctionConstantValues {
    FunctionConstantValues::new()
}

pub fn q2_k_row_bytes(n: usize) -> usize {
    assert_eq!(
        n % Q2_K_BLOCK_ELEMS,
        0,
        "N ({n}) is not a whole number of Q2_K blocks"
    );
    n / Q2_K_BLOCK_ELEMS * Q2_K_BLOCK_BYTES
}

pub struct Q2KResidentMatrix<'a> {
    pub buffer: &'a metal::Buffer,
    pub weights_offset: u64,
    pub rows: usize,
    pub cols: usize,
}

pub fn encode_dequant_q2_k_gemv_resident(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: &Q2KResidentMatrix<'_>,
    x: (&metal::Buffer, u64),
    y: (&metal::Buffer, u64),
) -> Result<(), GpuError> {
    assert!(w.rows > 0);
    let _ = q2_k_row_bytes(w.cols);
    let m = w.rows as u32;
    let n = w.cols as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q2_k_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (w.buffer, 0, w.weights_offset),
            (x.0, 1, x.1),
            (y.0, 2, y.1),
        ],
        &[(u32_bytes(&m), 3), (u32_bytes(&n), 4)],
        w.rows.div_ceil(ROWS_PER_THREADGROUP as usize) as u64,
        THREADS_PER_GROUP,
    );
    Ok(())
}

pub fn dequant_q2_k_gemv(
    context: &mut MetalContext,
    weight_rows: &[&[u8]],
    x: &[f16],
    n: usize,
) -> Result<Vec<f16>, GpuError> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    let row_bytes = q2_k_row_bytes(n);
    let mut bytes = Vec::with_capacity(weight_rows.len() * row_bytes);
    for row in weight_rows {
        assert_eq!(row.len(), row_bytes);
        bytes.extend_from_slice(row);
    }
    let w = context.new_buffer_with_data(&bytes);
    let x = context.new_buffer_with_data(&half_slice_to_le_bytes(x));
    let y = context.new_output_buffer((weight_rows.len() * 2) as u64);
    let m = weight_rows.len() as u32;
    let n = n as u32;
    let pipeline = context.pipeline(
        SOURCE,
        "dequant_q2_k_gemv_simd",
        &no_function_constants(),
        b"",
    )?;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[(&w, 0), (&x, 1), (&y, 2)],
        &[(u32_bytes(&m), 3), (u32_bytes(&n), 4)],
        weight_rows.len().div_ceil(ROWS_PER_THREADGROUP as usize) as u64,
        THREADS_PER_GROUP,
    );
    Ok(read_half_buffer(&y, weight_rows.len()))
}
