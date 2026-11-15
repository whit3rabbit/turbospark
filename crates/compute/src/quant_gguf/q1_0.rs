//! GGUF Q1_0 block decoding and its FP32 matrix-vector reference.

use foundation::LogitValue as F16;

/// Elements in one Q1_0 block.
pub const Q1_0_BLOCK_ELEMS: usize = 128;
/// Bytes in one Q1_0 block: an f16 scale and sixteen sign bytes.
pub const Q1_0_BLOCK_BYTES: usize = 18;

/// Dequantize `n` elements from a GGUF Q1_0 byte run.
///
/// Element `i` of a block is `+d` when bit `i % 8` of sign byte `i / 8` is
/// set and `-d` when it is clear (LSB first). Checked against ggml's own
/// `to_float` over 384 random elements, not against a spec page.
pub fn dequantize_q1_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q1_0_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q1_0_BLOCK_ELEMS}-element Q1_0 blocks"
    );
    let n_blocks = n / Q1_0_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q1_0_BLOCK_BYTES,
        "Q1_0 run is {} bytes, {} needed",
        blocks.len(),
        n_blocks * Q1_0_BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for block in 0..n_blocks {
        let base = block * Q1_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        for i in 0..Q1_0_BLOCK_ELEMS {
            let bit = (blocks[base + 2 + i / 8] >> (i % 8)) & 1;
            out[block * Q1_0_BLOCK_ELEMS + i] = if bit == 1 { d } else { -d };
        }
    }
    out
}

/// FP32 reference for `y = W * x`, one Q1_0 byte run per output row.
pub fn dequant_q1_0_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q1_0(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
