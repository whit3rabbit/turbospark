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
    if data.len() != rows * cols {
        return Err(RepackError::ShapeMismatch {
            rows,
            cols,
            data_len: data.len(),
        });
    }
    if cols % GROUP_SIZE != 0 {
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
    if data.len() != rows * cols {
        return Err(RepackError::ShapeMismatch {
            rows,
            cols,
            data_len: data.len(),
        });
    }
    if cols % GROUP_SIZE != 0 {
        return Err(RepackError::RowNotGroupAligned { row: 0, len: cols });
    }
    Ok(data.chunks_exact(cols).map(quantize_int8_affine).collect())
}

/// Total packed byte size of `rows` INT4-affine rows of `cols` elements
/// each: `rows * (cols/2 + 2 * 2 * cols/GROUP_SIZE)` (packed nibbles plus a
/// BF16 scale and bias per group).
pub fn int4_packed_bytes(rows: usize, cols: usize) -> usize {
    let groups_per_row = cols / GROUP_SIZE;
    rows * (cols / 2 + groups_per_row * 2 * 2)
}

/// Total packed byte size of `rows` INT8-affine rows of `cols` elements
/// each: `rows * (cols + 2 * 2 * cols/GROUP_SIZE)`.
pub fn int8_packed_bytes(rows: usize, cols: usize) -> usize {
    let groups_per_row = cols / GROUP_SIZE;
    rows * (cols + groups_per_row * 2 * 2)
}
