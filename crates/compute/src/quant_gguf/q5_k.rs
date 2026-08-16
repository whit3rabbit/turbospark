use foundation::LogitValue as F16;

use super::q4_k::{q4_k_scale_min, Q4_K_QUANTS_AT, Q4_K_SCALES_AT};

/// Elements in one Q5_K superblock.
pub const Q5_K_BLOCK_ELEMS: usize = 256;

/// Elements in one Q5_K sub-block. Eight tile a superblock, exactly as in
/// Q4_K, and they share Q4_K's 6-bit scale-and-min packing.
pub const Q5_K_SUB_ELEMS: usize = 32;

/// Bytes in one Q5_K superblock: f16 `d`, f16 `dmin`, 12 bytes of packed
/// 6-bit sub-scales and sub-mins, 32 bytes of fifth bits, then 128 bytes of
/// nibble-packed low bits. Confirmed against `ggml_type_size` by
/// `scripts/ggml_q5_k_oracle.c` rather than recalled, and matched to
/// `ggml_type_block(13)` by `crates/repack`'s tests.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Byte offset of the fifth-bit run inside a superblock.
const Q5_K_QH_AT: usize = 16;
/// Byte offset of the nibble run inside a superblock.
const Q5_K_QUANTS_AT: usize = Q5_K_QH_AT + Q5_K_BLOCK_ELEMS / 8;

/// Dequantize `n` elements of a Q5_K byte run to FP32.
///
/// Q5_K is Q4_K plus one bit per element, and every part of that sentence is
/// a trap for someone carrying Q4_K over by habit:
///
/// 1. **The fifth bit lives in its own 32-byte run**, `qh`, between the
///    packed scales and the nibbles. It contributes 16 to the quant, so
///    dropping it halves the dynamic range and leaves values that are
///    correctly signed, correctly ordered and merely wrong.
/// 2. **`qh` is indexed by the element's position WITHIN a 32-element
///    sub-block, not by its position in the superblock.** Byte `qh[l]`
///    serves elements `l`, `l + 32`, `l + 64`, ... and the BIT it serves
///    them from advances with the 64-element group: bit `2g + h` for group
///    `g` and half `h`. So one `qh` byte is read eight times, at eight
///    different bit positions.
/// 3. The reconstruction keeps Q4_K's per-sub-block min, `w = d*sc*q -
///    dmin*m` with UNSIGNED `q` in `0..32`. Q6_K's symmetric bias-32 form is
///    the odd one out among the K-quants, not this.
///
/// The grouping of the arithmetic below (`d1 = d * sc`, then `d1 * q - m1`)
/// mirrors ggml's, so the oracle in `tests/generated` compares EXACTLY
/// rather than within a tolerance.
pub fn dequantize_q5_k(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q5_K_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q5_K_BLOCK_ELEMS}-element Q5_K superblocks"
    );
    let n_blocks = n / Q5_K_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q5_K_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q5_K_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q5_K_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        let dmin = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + 2],
            blocks[base + 3],
        ])));
        // The 6-bit scale/min packing is Q4_K's, byte for byte, so it is
        // read with Q4_K's unpacker rather than a second copy of it.
        let packed = &blocks[base + Q4_K_SCALES_AT..base + Q4_K_QUANTS_AT];
        let qh = &blocks[base + Q5_K_QH_AT..base + Q5_K_QUANTS_AT];
        let ql = &blocks[base + Q5_K_QUANTS_AT..base + Q5_K_BLOCK_BYTES];

        for g in 0..Q5_K_BLOCK_ELEMS / (2 * Q5_K_SUB_ELEMS) {
            let (sc_lo, m_lo) = q4_k_scale_min(2 * g, packed);
            let (sc_hi, m_hi) = q4_k_scale_min(2 * g + 1, packed);
            let (d_lo, min_lo) = (d * sc_lo as f32, dmin * m_lo as f32);
            let (d_hi, min_hi) = (d * sc_hi as f32, dmin * m_hi as f32);
            let group = &ql[g * Q5_K_SUB_ELEMS..(g + 1) * Q5_K_SUB_ELEMS];
            let (bit_lo, bit_hi) = (2 * g, 2 * g + 1);
            let at = b * Q5_K_BLOCK_ELEMS + g * 2 * Q5_K_SUB_ELEMS;
            for (l, &byte) in group.iter().enumerate() {
                let hi_bits = qh[l];
                let q_lo = (byte & 0xF) + (((hi_bits >> bit_lo) & 1) << 4);
                let q_hi = (byte >> 4) + (((hi_bits >> bit_hi) & 1) << 4);
                out[at + l] = d_lo * q_lo as f32 - min_lo;
                out[at + Q5_K_SUB_ELEMS + l] = d_hi * q_hi as f32 - min_hi;
            }
        }
    }
    out
}

/// FP32 reference for the Q5_K GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q5_k_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q5_k(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
