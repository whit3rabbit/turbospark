//! GGUF Q4_0 block dequantization (32 signed 4-bit values per block).

use foundation::LogitValue as F16;

pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Dequantize a Q4_0 byte run using ggml's `dequantize_row_q4_0` layout.
pub fn dequantize_q4_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(
        n % Q4_0_BLOCK_ELEMS,
        0,
        "Q4_0 length must contain whole 32-element blocks"
    );
    let n_blocks = n / Q4_0_BLOCK_ELEMS;
    assert!(blocks.len() >= n_blocks * Q4_0_BLOCK_BYTES);

    let mut out = vec![0.0; n];
    for block in 0..n_blocks {
        let base = block * Q4_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        for j in 0..Q4_0_BLOCK_ELEMS / 2 {
            let packed = blocks[base + 2 + j];
            out[block * Q4_0_BLOCK_ELEMS + j] = (i32::from(packed & 0x0f) - 8) as f32 * d;
            out[block * Q4_0_BLOCK_ELEMS + j + Q4_0_BLOCK_ELEMS / 2] =
                (i32::from(packed >> 4) - 8) as f32 * d;
        }
    }
    out
}
