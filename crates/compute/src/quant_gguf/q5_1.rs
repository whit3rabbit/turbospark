//! GGUF Q5_1 block dequantization (32 unsigned 5-bit values per block, with a
//! per-block scale and minimum).

use foundation::LogitValue as F16;

pub const Q5_1_BLOCK_ELEMS: usize = 32;
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Dequantize a Q5_1 byte run using ggml's `dequantize_row_q5_1` layout:
/// f16 `d`, f16 `m`, a little-endian u32 of fifth bits `qh`, then 16 nibble
/// bytes where byte `j` holds element `j` (low) and `j + 16` (high). The fifth
/// bit of element `e` is bit `e` of `qh`; `w = q * d + m`.
pub fn dequantize_q5_1(blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(
        n % Q5_1_BLOCK_ELEMS,
        0,
        "Q5_1 length must contain whole 32-element blocks"
    );
    let n_blocks = n / Q5_1_BLOCK_ELEMS;
    assert!(blocks.len() >= n_blocks * Q5_1_BLOCK_BYTES);

    let mut out = vec![0.0; n];
    for block in 0..n_blocks {
        let base = block * Q5_1_BLOCK_BYTES;
        let f16_at = |at: usize| {
            f32::from(F16::from_bits(u16::from_le_bytes([
                blocks[at],
                blocks[at + 1],
            ])))
        };
        let (d, m) = (f16_at(base), f16_at(base + 2));
        let qh = u32::from_le_bytes([
            blocks[base + 4],
            blocks[base + 5],
            blocks[base + 6],
            blocks[base + 7],
        ]);
        let qs = &blocks[base + 8..base + Q5_1_BLOCK_BYTES];
        for j in 0..Q5_1_BLOCK_ELEMS / 2 {
            let low = (qs[j] & 0x0f) | (((qh >> j) as u8 & 1) << 4);
            let high = (qs[j] >> 4) | (((qh >> (j + 16)) as u8 & 1) << 4);
            out[block * Q5_1_BLOCK_ELEMS + j] = f32::from(low) * d + m;
            out[block * Q5_1_BLOCK_ELEMS + j + Q5_1_BLOCK_ELEMS / 2] = f32::from(high) * d + m;
        }
    }
    out
}
