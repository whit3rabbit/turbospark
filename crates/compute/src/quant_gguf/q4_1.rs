//! GGUF Q4_1 block dequantization (32 unsigned 4-bit values per block, with a
//! per-block scale and minimum).

use foundation::LogitValue as F16;

pub const Q4_1_BLOCK_ELEMS: usize = 32;
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Dequantize a Q4_1 byte run using ggml's `dequantize_row_q4_1` layout:
/// f16 `d`, f16 `m`, then 16 nibble bytes where byte `j` holds element `j`
/// (low) and element `j + 16` (high); `w = q * d + m`.
pub fn dequantize_q4_1(blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(
        n % Q4_1_BLOCK_ELEMS,
        0,
        "Q4_1 length must contain whole 32-element blocks"
    );
    let n_blocks = n / Q4_1_BLOCK_ELEMS;
    assert!(blocks.len() >= n_blocks * Q4_1_BLOCK_BYTES);

    let mut out = vec![0.0; n];
    for block in 0..n_blocks {
        let base = block * Q4_1_BLOCK_BYTES;
        let f16_at = |at: usize| {
            f32::from(F16::from_bits(u16::from_le_bytes([
                blocks[at],
                blocks[at + 1],
            ])))
        };
        let (d, m) = (f16_at(base), f16_at(base + 2));
        for j in 0..Q4_1_BLOCK_ELEMS / 2 {
            let packed = blocks[base + 4 + j];
            out[block * Q4_1_BLOCK_ELEMS + j] = f32::from(packed & 0x0f) * d + m;
            out[block * Q4_1_BLOCK_ELEMS + j + Q4_1_BLOCK_ELEMS / 2] =
                f32::from(packed >> 4) * d + m;
        }
    }
    out
}
