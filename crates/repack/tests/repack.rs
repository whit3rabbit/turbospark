//! Tests for the row-major matrix quantization repack.

use mrefrust_repack::{
    int4_packed_bytes, int8_packed_bytes, quantize_matrix_int4, quantize_matrix_int8, RepackError,
};

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

#[test]
fn packed_byte_sizes_match_the_written_row_shapes() {
    let rows = 4;
    let cols = 128; // 2 groups of 64
    let data = vec![0.5f32; rows * cols];

    let int4_rows = quantize_matrix_int4(&data, rows, cols).unwrap();
    let int4_actual: usize = int4_rows
        .iter()
        .map(|r| r.packed.len() + r.scales.len() * 2 + r.biases.len() * 2)
        .sum();
    assert_eq!(int4_actual, int4_packed_bytes(rows, cols));

    let int8_rows = quantize_matrix_int8(&data, rows, cols).unwrap();
    let int8_actual: usize = int8_rows
        .iter()
        .map(|r| r.packed.len() + r.scales.len() * 2 + r.biases.len() * 2)
        .sum();
    assert_eq!(int8_actual, int8_packed_bytes(rows, cols));
}
