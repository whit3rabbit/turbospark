//! GGUF TQ2_0 (ternary, 2 bits per element) block decoding and its FP32
//! matrix-vector reference.
//!
//! Block (ggml `block_tq2_0`): 64 bytes of packed 2-bit codes, THEN the f16
//! scale (the scale is LAST, as in Q2_K and unlike Q2_0). Element
//! `j * 4 + l * 32 + m` for chunk `j` in {0, 32}, shift `l` in 0..4 and byte
//! `m` in 0..32 is `((qs[j + m] >> (2 * l)) & 3) - 1` times `d`. Checked
//! against ggml's own `to_float` (`tests/fixtures/ggml/tq2_0.*`).

use foundation::LogitValue as F16;

/// Elements in one TQ2_0 block.
pub const TQ2_0_BLOCK_ELEMS: usize = 256;
/// Bytes in one TQ2_0 block: 64 code bytes and an f16 scale.
pub const TQ2_0_BLOCK_BYTES: usize = 66;

/// Dequantize `n` elements from a GGUF TQ2_0 byte run.
pub fn dequantize_tq2_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % TQ2_0_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {TQ2_0_BLOCK_ELEMS}-element TQ2_0 blocks"
    );
    let n_blocks = n / TQ2_0_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * TQ2_0_BLOCK_BYTES,
        "TQ2_0 run is {} bytes, {} needed",
        blocks.len(),
        n_blocks * TQ2_0_BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for block in 0..n_blocks {
        let base = block * TQ2_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + 64],
            blocks[base + 65],
        ])));
        let dst = &mut out[block * TQ2_0_BLOCK_ELEMS..(block + 1) * TQ2_0_BLOCK_ELEMS];
        for j in [0usize, 32] {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (blocks[base + j + m] >> (2 * l)) & 3;
                    dst[j * 4 + l * 32 + m] = (i32::from(q) - 1) as f32 * d;
                }
            }
        }
    }
    out
}

/// FP32 reference for `y = W * x`, one TQ2_0 byte run per output row.
pub fn dequant_tq2_0_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_tq2_0(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
