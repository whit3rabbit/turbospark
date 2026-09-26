//! GGUF Q2_0 block decoding and its FP32 matrix-vector reference.

use foundation::LogitValue as F16;

/// Elements in one Q2_0 block.
pub const Q2_0_BLOCK_ELEMS: usize = 64;
/// Bytes in one Q2_0 block: an f16 scale and sixteen packed two-bit bytes.
pub const Q2_0_BLOCK_BYTES: usize = 18;

/// Dequantize `n` elements from a GGUF Q2_0 byte run.
///
/// ggml stores four two-bit values per byte, least-significant pair first,
/// and reconstructs each value as `(q - 1) * d`.
pub fn dequantize_q2_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q2_0_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q2_0_BLOCK_ELEMS}-element Q2_0 blocks"
    );
    let n_blocks = n / Q2_0_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q2_0_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q2_0_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0.0; n];
    for block in 0..n_blocks {
        let base = block * Q2_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        for i in 0..Q2_0_BLOCK_ELEMS {
            let packed = blocks[base + 2 + i / 4];
            let q = (packed >> (2 * (i % 4))) & 3;
            out[block * Q2_0_BLOCK_ELEMS + i] = (i32::from(q) - 1) as f32 * d;
        }
    }
    out
}

/// FP32 reference for `y = W * x`, one Q2_0 byte run per output row.
pub fn dequant_q2_0_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q2_0(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
