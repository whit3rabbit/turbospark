//! GGUF block-quantization reference (ROADMAP Phase G Stage 2).
//!
//! Sibling of [`crate::quant`], and deliberately a separate module because the
//! layout is not a variant of the affine one. MLX affine stores three planar
//! regions per tensor (nibbles, BF16 scales, BF16 biases) at group 64. A GGUF
//! block is self-contained and interleaved: its scale sits immediately before
//! the weights it scales, so a row is one contiguous byte run with no
//! companion arrays. Keeping the two apart is what stops a caller passing a
//! Q8_0 row to an affine helper, where the argument types would otherwise
//! agree.
//!
//! Q8_0 is also SYMMETRIC where affine is not: `w = q * d` with signed `q`,
//! against affine's `w = q * scale + bias` with unsigned `q`. There is no bias
//! to carry, and a value of zero is exactly representable, which affine's
//! min/max derivation does not guarantee.
//!
//! Q8_0 leads Q4_K here for the reason recorded in ROADMAP Phase G: a Q8_0
//! block is 34 bytes with one scale, so this reference proves the plumbing
//! rather than fighting Q4_K's 6-bit packed sub-scales.

use foundation::LogitValue as F16;

/// Elements in one Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Bytes in one Q8_0 block: an f16 scale (little-endian) then 32 signed
/// weights. Matches `ggml_type_block(8)` in `mrefrust_repack::gguf_header`;
/// the two are checked against each other in `crates/repack`'s tests rather
/// than one importing the other, since this crate must not depend on repack.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantize `n` elements of a Q8_0 byte run to FP32.
///
/// `blocks` is the raw GGUF bytes, exactly as they sit in the file and exactly
/// as the repack walk copies them through. `n` must be a whole number of
/// blocks: ggml pads a tensor's last dimension up rather than emitting a short
/// block, so a caller asking for a partial one has a shape bug, not a ragged
/// tensor.
pub fn dequantize_q8_0(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q8_0_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q8_0_BLOCK_ELEMS}-element Q8_0 blocks"
    );
    let n_blocks = n / Q8_0_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q8_0_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q8_0_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        for k in 0..Q8_0_BLOCK_ELEMS {
            // The weights are SIGNED. Reading them as u8 costs no crash and no
            // NaN, just a silently wrong sign on half the coefficients.
            let q = blocks[base + 2 + k] as i8;
            out[b * Q8_0_BLOCK_ELEMS + k] = q as f32 * d;
        }
    }
    out
}

/// Quantize a row to Q8_0 bytes, following ggml's `quantize_row_q8_0`.
///
/// This port never quantizes a real GGUF (bytes arrive already quantized and
/// are copied through verbatim, which is the lossless-repack rule). It exists
/// so a test can build a block run from known weights and check that
/// [`dequantize_q8_0`] recovers them, and so a future GPU parity test has
/// arbitrary weights to compare against.
pub fn quantize_q8_0(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % Q8_0_BLOCK_ELEMS == 0,
        "row length {} is not a whole number of {Q8_0_BLOCK_ELEMS}-element blocks",
        row.len()
    );
    let mut out = Vec::with_capacity(row.len() / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES);
    for block in row.chunks_exact(Q8_0_BLOCK_ELEMS) {
        let amax = block.iter().fold(0f32, |acc, &w| acc.max(w.abs()));
        // Round the scale through f16 before quantizing, so decode reproduces
        // the same q this row stores. Same discipline as the affine
        // quantizers rounding through BF16 first.
        let d = F16::from_f32(amax / 127.0);
        let d_f32 = f32::from(d);
        let inv = if d_f32 == 0.0 { 0.0 } else { 1.0 / d_f32 };
        out.extend_from_slice(&d.to_bits().to_le_bytes());
        for &w in block {
            out.push((((w * inv).round() as i32).clamp(-127, 127) as i8) as u8);
        }
    }
    out
}

/// FP32 reference for the Q8_0 GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q8_0_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q8_0(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}

/// Pearson correlation of two equal-length vectors.
///
/// Lives here rather than in a test because settling `FUSED_GATE_FIRST` needs
/// it (comparing a GGUF-dequantized expert against the same expert in the
/// MLX-derived install: two different quantizations of the same trained
/// weights, so they agree in direction and not to the bit), and because a
/// future quant kernel is more likely to be judged by correlation than by an
/// element-wise tolerance. Returns 0.0 when either side is constant, which is
/// the answer that makes a caller's threshold behave: an undefined
/// correlation must not read as a match.
pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "correlation needs equal lengths");
    assert!(!a.is_empty());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - mean_a, y as f64 - mean_b);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())) as f32
}
