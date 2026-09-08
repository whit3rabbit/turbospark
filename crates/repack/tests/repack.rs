//! Tests for the row-major matrix quantization repack.

use turbospark_repack::{quantize_matrix_int4, quantize_matrix_int8, RepackError};

#[test]
fn quantize_matrix_int4_produces_one_row_per_matrix_row() {
    let rows = 3;
    let cols = 64;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 96.0) * 0.01).collect();
    let quantized = quantize_matrix_int4(&data, rows, cols).unwrap();
    assert_eq!(quantized.len(), rows);
    assert_eq!(quantized[0].packed.len(), cols / 2);
}

#[test]
fn quantize_matrix_int8_produces_one_row_per_matrix_row() {
    let rows = 2;
    let cols = 128;
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 - 128.0) * 0.02)
        .collect();
    let quantized = quantize_matrix_int8(&data, rows, cols).unwrap();
    assert_eq!(quantized.len(), rows);
    assert_eq!(quantized[0].packed.len(), cols);
}

#[test]
fn quantize_matrix_rejects_shape_mismatch() {
    let data = vec![0.0f32; 100];
    let err = quantize_matrix_int4(&data, 2, 64).unwrap_err();
    assert!(matches!(err, RepackError::ShapeMismatch { .. }));
}

#[test]
fn quantize_matrix_rejects_non_group_aligned_columns() {
    let data = vec![0.0f32; 30];
    let err = quantize_matrix_int4(&data, 1, 30).unwrap_err();
    assert!(matches!(err, RepackError::RowNotGroupAligned { .. }));
}

/// `cols == 0` passes `0 % GROUP_SIZE == 0`, so the shape a hostile or
/// corrupt header names (`[N, 0]`) must be refused explicitly rather than
/// reaching `chunks_exact(0)`, which panics.
#[test]
fn quantize_matrix_rejects_zero_columns_instead_of_panicking() {
    let err4 = quantize_matrix_int4(&[], 3, 0).unwrap_err();
    assert!(matches!(
        err4,
        RepackError::RowNotGroupAligned { len: 0, .. }
    ));

    let err8 = quantize_matrix_int8(&[], 3, 0).unwrap_err();
    assert!(matches!(
        err8,
        RepackError::RowNotGroupAligned { len: 0, .. }
    ));
}

/// `rows * cols` must not wrap: two shape values that overflow `usize`
/// multiplication but happen to produce a small wrapped product must not be
/// read as a matching shape.
#[test]
fn quantize_matrix_rejects_a_shape_whose_product_overflows() {
    let data = vec![0.0f32; 4];
    let err = quantize_matrix_int4(&data, usize::MAX / 2 + 1, 2).unwrap_err();
    assert!(matches!(err, RepackError::ShapeMismatch { .. }));
}
