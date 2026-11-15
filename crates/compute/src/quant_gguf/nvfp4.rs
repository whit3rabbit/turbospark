//! GGUF NVFP4 block decoding and its FP32 matrix-vector reference.
//!
//! Block (ggml `block_nvfp4`): 4 sub-block scale bytes FIRST (unsigned E4M3),
//! then 32 bytes of packed FP4 (E2M1) codes. Sub-block `s` covers 16 elements:
//! element `s * 16 + j` is the LOW nibble of `qs[s * 8 + j]` and element
//! `s * 16 + j + 8` its HIGH nibble, for `j` in 0..8, times the sub-block's
//! scale. Checked against ggml's own `to_float` over every scale byte value
//! (`tests/fixtures/ggml/nvfp4.*` and the scale sweep in the test file).
//!
//! The scale byte is read with bit 7 IGNORED and `0x7F` (the E4M3 NaN code)
//! decoding to ZERO, both measured from ggml's output rather than assumed.

/// Elements in one NVFP4 block.
pub const NVFP4_BLOCK_ELEMS: usize = 64;
/// Bytes in one NVFP4 block: 4 scale bytes and 32 code bytes.
pub const NVFP4_BLOCK_BYTES: usize = 36;

/// The E2M1 code values, in the order of the 4-bit code. Code 8 is +0, not -0:
/// ggml decodes through an integer table, and the golden test compares bits.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one NVFP4 scale byte exactly as ggml does.
pub fn nvfp4_scale(byte: u8) -> f32 {
    if byte & 0x7F == 0x7F {
        return 0.0;
    }
    let e = i32::from((byte >> 3) & 15);
    let m = f32::from(byte & 7);
    if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Dequantize `n` elements from a GGUF NVFP4 byte run.
pub fn dequantize_nvfp4(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % NVFP4_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {NVFP4_BLOCK_ELEMS}-element NVFP4 blocks"
    );
    let n_blocks = n / NVFP4_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * NVFP4_BLOCK_BYTES,
        "NVFP4 run is {} bytes, {} needed",
        blocks.len(),
        n_blocks * NVFP4_BLOCK_BYTES
    );
    let mut out = vec![0.0f32; n];
    for block in 0..n_blocks {
        let b = &blocks[block * NVFP4_BLOCK_BYTES..(block + 1) * NVFP4_BLOCK_BYTES];
        let dst = &mut out[block * NVFP4_BLOCK_ELEMS..(block + 1) * NVFP4_BLOCK_ELEMS];
        for s in 0..4 {
            let d = nvfp4_scale(b[s]);
            for j in 0..8 {
                let q = b[4 + s * 8 + j];
                dst[s * 16 + j] = E2M1[usize::from(q & 15)] * d;
                dst[s * 16 + j + 8] = E2M1[usize::from(q >> 4)] * d;
            }
        }
    }
    out
}

/// FP32 reference for `y = W * x`, one NVFP4 byte run per output row.
pub fn dequant_nvfp4_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_nvfp4(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
