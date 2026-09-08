//! Quantization repack: turns a row-major FP32 weight matrix into the
//! affine int4/int8 groupwise rows the `.gturbo` format stores, reusing
//! `turbospark_compute`'s quantizer (the same math the runtime's dequant
//! GEMV kernels expect) rather than re-deriving it here.
//!
//! This module is the per-matrix quantization step; `gturbo_writer.rs`
//! assembles the byte-exact `.gturbo` directory around it, and
//! `hf_checkpoint.rs`'s `orchestrate_llama_checkpoint` calls both to walk
//! a real downloaded HF checkpoint's tensors end to end (proven against a
//! real Hugging Face Hub download — see `DEVIATIONS.md`).

use compute::quant::GROUP_SIZE;
use compute::{quantize_int4_affine, quantize_int8_affine, Int4AffineRow, Int8AffineRow};

use crate::resident_writer::ResidentTensorSpec;

#[derive(Debug, Clone, PartialEq)]
pub enum RepackError {
    /// A row's length is not a multiple of the affine quantizer's group
    /// size.
    RowNotGroupAligned { row: usize, len: usize },
    /// `data.len()` is not `rows * cols`.
    ShapeMismatch {
        rows: usize,
        cols: usize,
        data_len: usize,
    },
}

impl std::fmt::Display for RepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepackError::RowNotGroupAligned { row, len } => {
                write!(
                    f,
                    "row {row} has length {len}, not a multiple of {GROUP_SIZE}"
                )
            }
            RepackError::ShapeMismatch {
                rows,
                cols,
                data_len,
            } => {
                write!(
                    f,
                    "data length {data_len} does not match rows*cols ({rows}*{cols})"
                )
            }
        }
    }
}

impl std::error::Error for RepackError {}

/// Quantize a row-major `[rows, cols]` FP32 matrix to affine INT4, one
/// [`Int4AffineRow`] per row.
pub fn quantize_matrix_int4(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Int4AffineRow>, RepackError> {
    if rows.checked_mul(cols) != Some(data.len()) {
        return Err(RepackError::ShapeMismatch {
            rows,
            cols,
            data_len: data.len(),
        });
    }
    // `cols == 0` passes `0 % GROUP_SIZE == 0` and would otherwise reach
    // `chunks_exact(0)`, which panics.
    if cols == 0 || cols % GROUP_SIZE != 0 {
        return Err(RepackError::RowNotGroupAligned { row: 0, len: cols });
    }
    Ok(data.chunks_exact(cols).map(quantize_int4_affine).collect())
}

/// Quantize a row-major `[rows, cols]` FP32 matrix to affine INT8, one
/// [`Int8AffineRow`] per row.
pub fn quantize_matrix_int8(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Int8AffineRow>, RepackError> {
    if rows.checked_mul(cols) != Some(data.len()) {
        return Err(RepackError::ShapeMismatch {
            rows,
            cols,
            data_len: data.len(),
        });
    }
    // `cols == 0` passes `0 % GROUP_SIZE == 0` and would otherwise reach
    // `chunks_exact(0)`, which panics.
    if cols == 0 || cols % GROUP_SIZE != 0 {
        return Err(RepackError::RowNotGroupAligned { row: 0, len: cols });
    }
    Ok(data.chunks_exact(cols).map(quantize_int8_affine).collect())
}

/// Concatenates a per-row INT4-affine quantization into one resident tensor
/// spec: `packed`/`scales`/`biases` back to back across `rows`, in row
/// order.
///
/// The common tail of five near-identical call sites across this crate
/// (`hf_checkpoint.rs`, `gemma4_checkpoint/{mtp,dflash,narrow}.rs`,
/// `gguf_checkpoint/transcode.rs`), each of which quantized a matrix -- via
/// [`quantize_matrix_int4`]/[`quantize_matrix_int8`] or a per-row
/// `compute::quantize_int{4,8}_affine` loop -- and then hand-rolled this
/// exact concatenation and struct.
///
/// Panics if `rows.len()` or `cols` exceeds `u32::MAX`: `ResidentTensorSpec`
/// stores shape as `u32`, and a silent truncation here would write a
/// plausible, wrong shape with no error at any later read (the same writer
/// invariant every call site asserted for itself before this existed).
pub fn resident_spec_from_int4_rows(
    name: impl Into<String>,
    rows: &[Int4AffineRow],
    cols: usize,
) -> ResidentTensorSpec {
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for row in rows {
        packed.extend_from_slice(&row.packed);
        scales.extend_from_slice(&row.scales);
        biases.extend_from_slice(&row.biases);
    }
    assert!(
        rows.len() <= u32::MAX as usize && cols <= u32::MAX as usize,
        "shape {}x{cols} exceeds the 32-bit resident index shape fields",
        rows.len()
    );
    ResidentTensorSpec {
        name: name.into(),
        packed,
        scales,
        biases,
        rows: rows.len() as u32,
        cols: cols as u32,
    }
}

/// The INT8-affine sibling of [`resident_spec_from_int4_rows`].
pub fn resident_spec_from_int8_rows(
    name: impl Into<String>,
    rows: &[Int8AffineRow],
    cols: usize,
) -> ResidentTensorSpec {
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for row in rows {
        packed.extend_from_slice(&row.packed);
        scales.extend_from_slice(&row.scales);
        biases.extend_from_slice(&row.biases);
    }
    assert!(
        rows.len() <= u32::MAX as usize && cols <= u32::MAX as usize,
        "shape {}x{cols} exceeds the 32-bit resident index shape fields",
        rows.len()
    );
    ResidentTensorSpec {
        name: name.into(),
        packed,
        scales,
        biases,
        rows: rows.len() as u32,
        cols: cols as u32,
    }
}
