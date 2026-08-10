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
//!
//! Q4_K is the second type, and it is NOT a wider Q8_0. It is a two-level
//! scheme: a 256-element superblock carries two f16 super-scales, then eight
//! 32-element sub-blocks each carry a 6-bit scale and a 6-bit min quantized
//! against those. It is also ASYMMETRIC (`w = d*sc*q - dmin*m`, unsigned
//! 4-bit `q`), which puts it closer to this port's affine layout than to
//! Q8_0 on that one axis while differing from both on every other.
//!
//! Q6_K is the third, and it is a third shape again rather than a wider Q4_K.
//! A 256-element superblock carries one f16 super-scale and SIXTEEN signed
//! int8 sub-block scales (plain bytes, no 6-bit packing), and an element's
//! six bits are split across two runs: four low bits in `ql`, two high bits
//! in `qh`. It is symmetric like Q8_0, with a fixed bias of 32 subtracted off
//! the stored value rather than a per-sub-block min. Qwen 3.6's Q4_K_M ships
//! exactly one Q6_K tensor (`output.weight`), which is why it is here.

use foundation::LogitValue as F16;

/// Elements in one Q8_0 block.
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// Bytes in one Q8_0 block: an f16 scale (little-endian) then 32 signed
/// weights. Matches `ggml_type_block(8)` in `turbospark_repack::gguf_header`;
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

/// Elements in one Q4_K superblock.
pub const Q4_K_BLOCK_ELEMS: usize = 256;

/// Elements in one Q4_K sub-block. Eight of these tile a superblock, and each
/// carries its own 6-bit scale and 6-bit min.
pub const Q4_K_SUB_ELEMS: usize = 32;

/// Bytes in one Q4_K superblock: f16 `d`, f16 `dmin`, 12 bytes of packed
/// 6-bit sub-scales and sub-mins, then 128 bytes of nibble-packed quants.
/// Matches `ggml_type_block(12)` in `turbospark_repack::gguf_header`, which is
/// held to this constant in `crates/repack`'s tests rather than by an import
/// (this crate must not depend on repack).
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Sub-blocks per superblock.
const Q4_K_SUBS: usize = Q4_K_BLOCK_ELEMS / Q4_K_SUB_ELEMS;
/// Bytes of packed sub-scales and sub-mins per superblock.
const Q4_K_SCALE_BYTES: usize = 12;
/// Byte offset of the packed sub-scale run inside a superblock.
const Q4_K_SCALES_AT: usize = 4;
/// Byte offset of the nibble run inside a superblock.
const Q4_K_QUANTS_AT: usize = Q4_K_SCALES_AT + Q4_K_SCALE_BYTES;

/// Unpack sub-block `j`'s 6-bit scale and 6-bit min out of the 12 packed
/// bytes, following ggml's `get_scale_min_k4`.
///
/// The packing is the part of Q4_K most likely to be got wrong, and wrong
/// here is finite and plausible rather than a crash, so it is spelled out:
///
/// - `packed[0..4]` hold the low 6 bits of the scales for sub-blocks 0..4.
/// - `packed[4..8]` hold the low 6 bits of the mins for sub-blocks 0..4.
/// - `packed[8..12]` hold, per sub-block 4..8, the LOW 4 bits of its scale in
///   the low nibble and the low 4 bits of its min in the high nibble.
/// - The remaining HIGH 2 bits for sub-blocks 4..8 are stowed in the top two
///   bits of the first eight bytes: scale bits in `packed[j - 4]`, min bits in
///   `packed[j]`.
///
/// So a sub-block in the second half has its 6 bits split across two bytes,
/// which is why an implementation that treats all eight uniformly reads
/// scales that are merely too small rather than obviously broken.
fn q4_k_scale_min(j: usize, packed: &[u8]) -> (u8, u8) {
    debug_assert!(j < Q4_K_SUBS);
    debug_assert!(packed.len() >= Q4_K_SCALE_BYTES);
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        (
            (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4),
            (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4),
        )
    }
}

/// Dequantize `n` elements of a Q4_K byte run to FP32.
///
/// `blocks` is the raw GGUF bytes. Two orderings inside a superblock are easy
/// to assume wrongly and neither faults:
///
/// 1. The 128 quant bytes are read in four groups of 32. Within a group,
///    every LOW nibble comes first (elements `64g .. 64g + 32`) and every HIGH
///    nibble second (`64g + 32 .. 64g + 64`). The nibbles of one byte are 32
///    elements apart, not adjacent.
/// 2. Those two halves use sub-blocks `2g` and `2g + 1`, so the sub-block
///    index advances with the nibble half, not with the byte.
pub fn dequantize_q4_k(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q4_K_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q4_K_BLOCK_ELEMS}-element Q4_K superblocks"
    );
    let n_blocks = n / Q4_K_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q4_K_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q4_K_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q4_K_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base],
            blocks[base + 1],
        ])));
        let dmin = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + 2],
            blocks[base + 3],
        ])));
        let packed = &blocks[base + Q4_K_SCALES_AT..base + Q4_K_QUANTS_AT];
        let qs = &blocks[base + Q4_K_QUANTS_AT..base + Q4_K_BLOCK_BYTES];

        for g in 0..Q4_K_SUBS / 2 {
            let (sc_lo, m_lo) = q4_k_scale_min(2 * g, packed);
            let (sc_hi, m_hi) = q4_k_scale_min(2 * g + 1, packed);
            let (d_lo, min_lo) = (d * sc_lo as f32, dmin * m_lo as f32);
            let (d_hi, min_hi) = (d * sc_hi as f32, dmin * m_hi as f32);
            let group = &qs[g * Q4_K_SUB_ELEMS..(g + 1) * Q4_K_SUB_ELEMS];
            let at = b * Q4_K_BLOCK_ELEMS + g * 2 * Q4_K_SUB_ELEMS;
            for (l, &byte) in group.iter().enumerate() {
                out[at + l] = d_lo * (byte & 0xF) as f32 - min_lo;
                out[at + Q4_K_SUB_ELEMS + l] = d_hi * (byte >> 4) as f32 - min_hi;
            }
        }
    }
    out
}

/// Quantize a row to Q4_K bytes, following ggml's `quantize_row_q4_K_ref`
/// with a plain min/max fit per sub-block in place of its `make_qkx2_quants`
/// search.
///
/// The search is an encoder-quality choice and this port never encodes a real
/// GGUF (bytes arrive quantized and are copied through verbatim, which is the
/// lossless-repack rule). What has to match ggml exactly is the BYTE LAYOUT
/// and the reconstruction formula, and both do. Like the Q8_0 sibling this
/// exists so tests have arbitrary known weights to hand the decoder and the
/// GPU kernel.
///
/// One detail carried over from ggml deliberately: the element quants are
/// derived by reading the sub-scales BACK through [`q4_k_scale_min`], not
/// from the pre-packing floats. That makes the round trip self-consistent
/// even where the 6-bit sub-scale rounded, and it means a packing bug shows
/// up as a bad round trip rather than being cancelled out.
pub fn quantize_q4_k(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % Q4_K_BLOCK_ELEMS == 0,
        "row length {} is not a whole number of {Q4_K_BLOCK_ELEMS}-element superblocks",
        row.len()
    );
    let mut out = Vec::with_capacity(row.len() / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES);

    for block in row.chunks_exact(Q4_K_BLOCK_ELEMS) {
        // Per sub-block: the range is fitted over [min, max] with min pinned
        // at or below zero, so a quant of 0 always reconstructs the minimum
        // and the stored min is non-negative (it is SUBTRACTED on decode).
        let mut scales = [0f32; Q4_K_SUBS];
        let mut mins = [0f32; Q4_K_SUBS];
        for j in 0..Q4_K_SUBS {
            let sub = &block[j * Q4_K_SUB_ELEMS..(j + 1) * Q4_K_SUB_ELEMS];
            let lo = sub.iter().fold(0f32, |acc, &v| acc.min(v));
            let hi = sub.iter().fold(lo, |acc, &v| acc.max(v));
            scales[j] = (hi - lo) / 15.0;
            mins[j] = -lo;
        }
        let max_scale = scales.iter().fold(0f32, |acc, &v| acc.max(v));
        let max_min = mins.iter().fold(0f32, |acc, &v| acc.max(v));
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        let mut packed = [0u8; Q4_K_SCALE_BYTES];
        for j in 0..Q4_K_SUBS {
            let ls = (scales[j] * inv_scale).round().clamp(0.0, 63.0) as u8;
            let lm = (mins[j] * inv_min).round().clamp(0.0, 63.0) as u8;
            if j < 4 {
                packed[j] |= ls;
                packed[j + 4] |= lm;
            } else {
                packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                packed[j - 4] |= (ls >> 4) << 6;
                packed[j] |= (lm >> 4) << 6;
            }
        }

        // Round the super-scales through f16 before deriving quants, so a
        // decode reproduces the integers chosen here. Same discipline as the
        // Q8_0 and affine quantizers.
        let d = F16::from_f32(max_scale / 63.0);
        let dmin = F16::from_f32(max_min / 63.0);

        let mut quants = [0u8; Q4_K_BLOCK_ELEMS];
        for j in 0..Q4_K_SUBS {
            let (sc, m) = q4_k_scale_min(j, &packed);
            let step = f32::from(d) * sc as f32;
            if step == 0.0 {
                continue;
            }
            let bias = f32::from(dmin) * m as f32;
            for (ii, &w) in block[j * Q4_K_SUB_ELEMS..(j + 1) * Q4_K_SUB_ELEMS]
                .iter()
                .enumerate()
            {
                quants[j * Q4_K_SUB_ELEMS + ii] =
                    (((w + bias) / step).round().clamp(0.0, 15.0)) as u8;
            }
        }

        out.extend_from_slice(&d.to_bits().to_le_bytes());
        out.extend_from_slice(&dmin.to_bits().to_le_bytes());
        out.extend_from_slice(&packed);
        for g in 0..Q4_K_SUBS / 2 {
            let at = g * 2 * Q4_K_SUB_ELEMS;
            for l in 0..Q4_K_SUB_ELEMS {
                out.push(quants[at + l] | (quants[at + Q4_K_SUB_ELEMS + l] << 4));
            }
        }
    }
    out
}

/// FP32 reference for the Q4_K GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q4_k_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q4_k(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}

/// Elements in one Q6_K superblock.
pub const Q6_K_BLOCK_ELEMS: usize = 256;

/// Elements sharing one Q6_K scale. Sixteen of these tile a superblock, half
/// the sub-block size Q4_K uses.
pub const Q6_K_SUB_ELEMS: usize = 16;

/// Bytes in one Q6_K superblock: 128 bytes of low nibbles, 64 bytes of high
/// bit-pairs, 16 SIGNED sub-block scales, then the f16 super-scale. Matches
/// `ggml_type_block(14)` in `turbospark_repack::gguf_header`, held to it by
/// `crates/repack`'s tests rather than by an import.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Byte offset of the high-bit run inside a superblock.
const Q6_K_QH_AT: usize = 128;
/// Byte offset of the 16 signed sub-block scales.
const Q6_K_SCALES_AT: usize = 192;
/// Byte offset of the f16 super-scale.
const Q6_K_D_AT: usize = 208;
/// Elements covered by one (ql, qh, scales) stride: a superblock is two.
const Q6_K_HALF_ELEMS: usize = 128;

/// Dequantize `n` elements of a Q6_K byte run to FP32.
///
/// Q6_K is the third layout in this module and shares its shape with neither
/// sibling. Four things about it are silently wrong rather than a fault if
/// carried over from Q4_K by habit:
///
/// 1. The six bits of an element are SPLIT ACROSS TWO RUNS: four low bits in
///    `ql`, two high bits in `qh`. Reading only `ql` gives values in `0..16`
///    where `0..64` was meant, which is finite and merely compressed.
/// 2. The quants are BIASED, not unsigned: the stored 6-bit value has 32
///    SUBTRACTED from it, so the reconstruction is symmetric and there is no
///    per-sub-block min. Dropping the bias shifts every weight positive.
/// 3. The sub-block scales are SIGNED int8, plain bytes rather than the 6-bit
///    packing Q4_K uses, and real files do carry negative ones (ggml derives
///    them through a negative `iscale`). Reading them as `u8` mirrors the
///    sign of whole 16-element runs.
/// 4. A superblock is walked in two halves of 128 elements. Within a half,
///    the four values a `qh` byte serves are 32 elements apart and each uses
///    a DIFFERENT scale pair (`is`, `is + 2`, `is + 4`, `is + 6`).
pub fn dequantize_q6_k(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q6_K_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q6_K_BLOCK_ELEMS}-element Q6_K superblocks"
    );
    let n_blocks = n / Q6_K_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q6_K_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q6_K_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + Q6_K_D_AT],
            blocks[base + Q6_K_D_AT + 1],
        ])));
        for h in 0..Q6_K_BLOCK_ELEMS / Q6_K_HALF_ELEMS {
            let ql = &blocks[base + h * 64..base + h * 64 + 64];
            let qh = &blocks[base + Q6_K_QH_AT + h * 32..base + Q6_K_QH_AT + h * 32 + 32];
            let sc = &blocks[base + Q6_K_SCALES_AT + h * 8..base + Q6_K_SCALES_AT + h * 8 + 8];
            let at = b * Q6_K_BLOCK_ELEMS + h * Q6_K_HALF_ELEMS;
            for l in 0..32 {
                let is = l / Q6_K_SUB_ELEMS;
                let q = [
                    ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32,
                    ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32,
                    ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32,
                    ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32,
                ];
                for (k, &qk) in q.iter().enumerate() {
                    out[at + l + k * 32] = d * (sc[is + k * 2] as i8) as f32 * qk as f32;
                }
            }
        }
    }
    out
}

/// Quantize a row to Q6_K bytes, following ggml's `quantize_row_q6_K_ref`
/// with a plain largest-magnitude fit per 16-element sub-block in place of
/// its `make_qx_quants` search.
///
/// Same standing as the other two quantizers here: this port never encodes a
/// real GGUF (bytes arrive quantized and are copied through verbatim), so
/// what has to match ggml exactly is the BYTE LAYOUT and the reconstruction
/// formula, not the encoder's search quality. ggml's negative `iscale` is
/// kept rather than simplified away, because it is what makes real files
/// carry negative sub-block scales, and a quantizer that only ever emitted
/// positive ones would leave the decoder's signed read untested.
pub fn quantize_q6_k(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % Q6_K_BLOCK_ELEMS == 0,
        "row length {} is not a whole number of {Q6_K_BLOCK_ELEMS}-element superblocks",
        row.len()
    );
    let subs = Q6_K_BLOCK_ELEMS / Q6_K_SUB_ELEMS;
    let mut out = Vec::with_capacity(row.len() / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES);

    for block in row.chunks_exact(Q6_K_BLOCK_ELEMS) {
        // Per sub-block: the SIGNED element of largest magnitude sets the
        // scale, so a sub-block whose extreme is negative gets a negative
        // scale, exactly as ggml's `make_qx_quants` does.
        let mut scales = [0f32; 16];
        let mut max_scale = 0f32;
        for (j, scale) in scales.iter_mut().enumerate().take(subs) {
            let sub = &block[j * Q6_K_SUB_ELEMS..(j + 1) * Q6_K_SUB_ELEMS];
            let mut extreme = 0f32;
            for &v in sub {
                if v.abs() > extreme.abs() {
                    extreme = v;
                }
            }
            *scale = -extreme / 32.0;
            if scale.abs() > max_scale.abs() {
                max_scale = *scale;
            }
        }

        let mut sc = [0i8; 16];
        let d = if max_scale == 0.0 {
            F16::from_f32(0.0)
        } else {
            let iscale = -128.0 / max_scale;
            for (j, s) in sc.iter_mut().enumerate().take(subs) {
                *s = ((iscale * scales[j]).round() as i32).clamp(-128, 127) as i8;
            }
            // Round the super-scale through f16 before deriving quants, so a
            // decode reproduces the integers chosen here. Same discipline as
            // the Q8_0 and Q4_K siblings.
            F16::from_f32(1.0 / iscale)
        };
        let d_f32 = f32::from(d);

        // Stored levels are the quant PLUS 32, which is what the packing
        // holds and what the decoder subtracts back off.
        let mut levels = [32u8; Q6_K_BLOCK_ELEMS];
        for j in 0..subs {
            let step = d_f32 * sc[j] as f32;
            if step == 0.0 {
                continue;
            }
            for (ii, &w) in block[j * Q6_K_SUB_ELEMS..(j + 1) * Q6_K_SUB_ELEMS]
                .iter()
                .enumerate()
            {
                let q = (w / step).round().clamp(-32.0, 31.0) as i32;
                levels[j * Q6_K_SUB_ELEMS + ii] = (q + 32) as u8;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for h in 0..Q6_K_BLOCK_ELEMS / Q6_K_HALF_ELEMS {
            let at = h * Q6_K_HALF_ELEMS;
            for l in 0..32 {
                let v = [
                    levels[at + l],
                    levels[at + l + 32],
                    levels[at + l + 64],
                    levels[at + l + 96],
                ];
                ql[h * 64 + l] = (v[0] & 0xF) | ((v[2] & 0xF) << 4);
                ql[h * 64 + l + 32] = (v[1] & 0xF) | ((v[3] & 0xF) << 4);
                qh[h * 32 + l] =
                    (v[0] >> 4) | ((v[1] >> 4) << 2) | ((v[2] >> 4) << 4) | ((v[3] >> 4) << 6);
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&s| s as u8));
        out.extend_from_slice(&d.to_bits().to_le_bytes());
    }
    out
}

/// FP32 reference for the Q6_K GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q6_k_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q6_k(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}

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
