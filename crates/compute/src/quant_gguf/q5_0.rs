//! GGUF Q5_0 block dequantization (32 signed 5-bit values per block).

use foundation::LogitValue as F16;

pub const Q5_0_BLOCK_ELEMS: usize = 32;
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Dequantize a Q5_0 byte run using ggml's `dequantize_row_q5_0` layout.
pub fn dequantize_q5_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(
        n % Q5_0_BLOCK_ELEMS,
        0,
        "Q5_0 length must contain whole 32-element blocks"
    );
    let n_blocks = n / Q5_0_BLOCK_ELEMS;
    assert!(blocks.len() >= n_blocks * Q5_0_BLOCK_BYTES);

    let mut out = vec![0.0; n];
    for block in 0..n_blocks {
        let base = block * Q5_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        let qh = u32::from_le_bytes([
            blocks[base + 2],
            blocks[base + 3],
            blocks[base + 4],
            blocks[base + 5],
        ]);
        let qs = &blocks[base + 6..base + Q5_0_BLOCK_BYTES];
        for j in 0..Q5_0_BLOCK_ELEMS / 2 {
            let low = (qs[j] & 0x0f) | (((qh >> j) as u8 & 1) << 4);
            let high = (qs[j] >> 4) | (((qh >> (j + 16)) as u8 & 1) << 4);
            out[block * Q5_0_BLOCK_ELEMS + j] = (i32::from(low) - 16) as f32 * d;
            out[block * Q5_0_BLOCK_ELEMS + j + Q5_0_BLOCK_ELEMS / 2] =
                (i32::from(high) - 16) as f32 * d;
        }
    }
    out
}
