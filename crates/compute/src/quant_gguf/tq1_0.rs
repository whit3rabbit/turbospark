//! GGUF TQ1_0 (ternary, base-3 packed, 1.6875 bits per element) block
//! decoding and its FP32 matrix-vector reference.
//!
//! Block (ggml `block_tq1_0`): 48 bytes `qs`, 4 bytes `qh`, then the f16
//! scale LAST. Each code byte packs five trits as a base-3 number scaled to
//! 8 bits; trit `n` of byte `b` is `((b * 3^n) mod 256) * 3 >> 8`, minus 1.
//! Element order: `qs[0..32]` give 160 elements (trit-plane `n` of every
//! byte, planes in order), `qs[32..48]` give the next 80 the same way, and
//! the four `qh` bytes give the last 16 as four planes of four. Checked
//! against ggml's own `to_float` (`tests/fixtures/ggml/tq1_0.*`).

use foundation::LogitValue as F16;

/// Elements in one TQ1_0 block.
pub const TQ1_0_BLOCK_ELEMS: usize = 256;
/// Bytes in one TQ1_0 block: 48 `qs`, 4 `qh` and an f16 scale.
pub const TQ1_0_BLOCK_BYTES: usize = 54;

const POW3: [u32; 6] = [1, 3, 9, 27, 81, 243];

#[inline]
fn trit(byte: u8, plane: usize) -> i32 {
    let q = (u32::from(byte) * POW3[plane]) & 0xFF;
    ((q * 3) >> 8) as i32 - 1
}

/// Dequantize `n` elements from a GGUF TQ1_0 byte run.
pub fn dequantize_tq1_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % TQ1_0_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {TQ1_0_BLOCK_ELEMS}-element TQ1_0 blocks"
    );
    let n_blocks = n / TQ1_0_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * TQ1_0_BLOCK_BYTES,
        "TQ1_0 run is {} bytes, {} needed",
        blocks.len(),
        n_blocks * TQ1_0_BLOCK_BYTES
    );
    let mut out = Vec::with_capacity(n);
    for block in 0..n_blocks {
        let b = &blocks[block * TQ1_0_BLOCK_BYTES..(block + 1) * TQ1_0_BLOCK_BYTES];
        let d = f32::from(F16::from_bits(u16::from_le_bytes([b[52], b[53]])));
        let (qs, qh) = (&b[..48], &b[48..52]);
        for (bytes, planes) in [(&qs[..32], 5), (&qs[32..48], 5), (qh, 4)] {
            for plane in 0..planes {
                for &byte in bytes {
                    out.push(trit(byte, plane) as f32 * d);
                }
            }
        }
    }
    out
}

/// FP32 reference for `y = W * x`, one TQ1_0 byte run per output row.
pub fn dequant_tq1_0_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_tq1_0(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
