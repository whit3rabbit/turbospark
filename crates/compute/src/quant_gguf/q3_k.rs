use foundation::LogitValue as F16;

/// Elements in one Q3_K superblock.
pub const Q3_K_BLOCK_ELEMS: usize = 256;

/// Elements in one Q3_K group. SIXTEEN of these tile a superblock, which is
/// twice the eight 32-element sub-blocks its Q4_K sibling uses, and each
/// carries one 6-bit scale with no separate min.
pub const Q3_K_SUB_ELEMS: usize = 16;

/// Bytes in one Q3_K superblock: 32 high-bit bytes, 64 two-bit quants, 12
/// bytes of packed 6-bit scales, then one closing f16 super-scale. The
/// super-scale is LAST, unlike Q4_K/Q5_K/Q6_K where it leads, and unlike
/// every sibling a wrong assumption here stays byte-aligned and silently
/// wrong rather than failing a length check.
/// Matches `ggml_type_block(11)` in `turbospark_repack::gguf_header`, which
/// is held to this constant in `crates/repack`'s tests rather than by an
/// import (this crate must not depend on repack).
pub const Q3_K_BLOCK_BYTES: usize = 110;

/// Byte offset of the high-bit run inside a superblock.
pub(crate) const Q3_K_HMASK_AT: usize = 0;
/// Byte offset of the 2-bit quant run inside a superblock.
pub(crate) const Q3_K_QUANTS_AT: usize = 32;
/// Byte offset of the packed 6-bit scale run inside a superblock.
pub(crate) const Q3_K_SCALES_AT: usize = 96;
/// Byte offset of the closing f16 super-scale.
pub(crate) const Q3_K_D_AT: usize = 108;

/// Decode the 12 packed scale bytes into the SIXTEEN 6-bit scale values,
/// minus the fixed bias of 32 the format stores them against.
///
/// ggml packs sixteen 6-bit values into 12 bytes by splitting each one: the
/// low nibble of values 0..8 into the low nibbles of bytes 0..8, the low
/// nibble of values 8..16 into the HIGH nibbles of the same bytes, and the
/// top two bits of every value into two-bit fields of bytes 8..12. The
/// decoder reconstructs that with a four-word shuffle (ggml's `kmask1`/
/// `kmask2` aux trick) rather than a bit-by-bit walk; this is the same
/// shuffle verbatim, with the f16 super-scale excluded because it sits
/// outside the packed run in this layout.
///
/// `out` receives 16 values in the bias-subtracted domain: the stored
/// 6-bit value s means the scale factor `stored_d * (s - 32)`.
pub fn q3_k_decode_scales(packed: &[u8], out: &mut [i32; 16]) {
    debug_assert!(packed.len() >= 12);
    let mut aux = [0u32; 4];
    for (i, word) in aux.iter_mut().enumerate().take(3) {
        for k in 0..4 {
            *word |= u32::from(packed[i * 4 + k]) << (8 * k);
        }
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    for (i, value) in out.iter_mut().enumerate() {
        let byte = (aux[i / 4] >> (8 * (i % 4))) & 0xFF;
        *value = i32::from(byte as u8) - 32;
    }
}

/// The scale group of one element, and the signed 3-bit level it decodes to.
///
/// Within a superblock, element `e` lives in half `e / 128`, group-of-32
/// `j = (e % 128) / 32` and lane `rem = e % 32`. Its 2 bits sit in quant
/// byte `half * 32 + rem` at shift `2 * j`; its high bit is bit `4 * half +
/// j` of high-bit byte `e % 32`; and its scale is `8 * half + 2 * j +
/// (rem / 16)`. A cleared high bit means the level is NEGATIVE: the stored
/// 2-bit value is `level + 4` for levels -4..-1, so the decoded value is
/// `q - 4` when the bit is clear and `q` when it is set.
#[inline]
pub(crate) fn q3_k_element(e: usize) -> (usize, usize, usize, usize, usize) {
    let half = e / 128;
    let j = (e % 128) / 32;
    let rem = e % 32;
    let quant_byte = half * 32 + rem;
    let shift = 2 * j;
    (half, j, rem, quant_byte, shift)
}

/// Dequantize `n` elements of a Q3_K byte run to FP32.
///
/// `blocks` is the raw GGUF bytes. Three orderings inside a superblock are
/// easy to assume wrongly and none of the three faults:
///
/// 1. The super-scale is the LAST field, not the first.
/// 2. The high-bit run is indexed by `e % 32` with the BIT chosen by
///    `e / 32`, so consecutive elements alternate bits, not bytes.
/// 3. The 16 scales consume in an order that interleaves the two 16-element
///    lanes of each group of 32, and a cleared high bit SUBTRACTS 4 from the
///    stored level rather than the value being unsigned.
pub fn dequantize_q3_k(blocks: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % Q3_K_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of {Q3_K_BLOCK_ELEMS}-element Q3_K superblocks"
    );
    let n_blocks = n / Q3_K_BLOCK_ELEMS;
    assert!(
        blocks.len() >= n_blocks * Q3_K_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        n_blocks * Q3_K_BLOCK_BYTES,
        blocks.len()
    );

    let mut out = vec![0f32; n];
    let mut scales = [0i32; 16];
    for b in 0..n_blocks {
        let base = b * Q3_K_BLOCK_BYTES;
        let d = f32::from(F16::from_bits(u16::from_le_bytes([
            blocks[base + Q3_K_D_AT],
            blocks[base + Q3_K_D_AT + 1],
        ])));
        let hmask = &blocks[base + Q3_K_HMASK_AT..base + Q3_K_HMASK_AT + 32];
        let qs = &blocks[base + Q3_K_QUANTS_AT..base + Q3_K_QUANTS_AT + 64];
        q3_k_decode_scales(
            &blocks[base + Q3_K_SCALES_AT..base + Q3_K_SCALES_AT + 12],
            &mut scales,
        );

        for e in 0..Q3_K_BLOCK_ELEMS {
            let (half, j, rem, quant_byte, shift) = q3_k_element(e);
            let q = (qs[quant_byte] >> shift) & 3;
            let high_bit_set = hmask[e % 32] & (1 << (4 * half + j)) != 0;
            let level = i32::from(q) - if high_bit_set { 0 } else { 4 };
            let dl = d * scales[8 * half + 2 * j + (rem / 16)] as f32;
            out[b * Q3_K_BLOCK_ELEMS + e] = dl * level as f32;
        }
    }
    out
}

/// Quantize a row to Q3_K bytes, following ggml's `quantize_row_q3_K_ref`
/// layout with a plain per-group max-magnitude fit in place of its
/// `make_qx_quants` search.
///
/// The search is an encoder-quality choice and this port never encodes a real
/// GGUF (bytes arrive quantized and are copied through verbatim, which is the
/// lossless-repack rule). What has to match ggml exactly is the BYTE LAYOUT,
/// the scale packing and the reconstruction formula, and all three do. Like
/// the Q4_K sibling this exists so tests have arbitrary known weights to hand
/// the decoder and the GPU kernel.
///
/// Two details are carried over from ggml deliberately:
///
/// - The element levels are derived by reading the packed scales BACK through
///   [`q3_k_decode_scales`], not from the pre-packing floats, so the round
///   trip is self-consistent even where a 6-bit scale rounded.
/// - The high bit marks NON-NEGATIVE levels, a cleared bit shifts the level
///   down by 4, and the hmask bit for element `e` is `1 << (e / 32)` in byte
///   `e % 32`.
pub fn quantize_q3_k(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % Q3_K_BLOCK_ELEMS == 0,
        "row length {} is not a whole number of {Q3_K_BLOCK_ELEMS}-element superblocks",
        row.len()
    );
    let mut out = Vec::with_capacity(row.len() / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES);
    let mut scales = [0i32; 16];

    for block in row.chunks_exact(Q3_K_BLOCK_ELEMS) {
        // Per group of 16: the step covers positive levels 0..3 and negative
        // levels -4..-1, so the largest negative magnitude needs a step of
        // its magnitude over 4, the largest positive value its magnitude
        // over 3.
        let mut steps = [0f32; 16];
        for g in 0..16 {
            let group = &block[g * Q3_K_SUB_ELEMS..(g + 1) * Q3_K_SUB_ELEMS];
            let max_pos = group.iter().fold(0f32, |acc, &v| acc.max(v));
            let max_neg = group.iter().fold(0f32, |acc, &v| acc.max(-v));
            steps[g] = (max_pos / 3.0).max(max_neg / 4.0);
        }
        let max_step = steps.iter().fold(0f32, |acc, &v| acc.max(v));
        let d = F16::from_f32(max_step / 31.0);
        let df = f32::from(d);

        // Pack the 16 group scales as 6-bit values biased by +32, inverting
        // the decoder's shuffle in the same order ggml's encoder uses.
        let mut packed = [0u8; 12];
        for (g, step) in steps.iter().enumerate() {
            let s = if max_step > 0.0 {
                ((step / df).round().clamp(0.0, 31.0) as u8) + 32
            } else {
                32
            };
            if g < 8 {
                packed[g] |= s & 0xF;
            } else {
                packed[g - 8] |= (s & 0xF) << 4;
            }
            packed[8 + g % 4] |= ((s >> 4) & 0x3) << (2 * (g / 4));
        }

        // Derive the element levels against the scales AS THEY WILL DECODE,
        // not against the floats above.
        q3_k_decode_scales(&packed, &mut scales);

        let mut hmask = [0u8; 32];
        let mut levels = [0u8; Q3_K_BLOCK_ELEMS];
        for (e, level) in levels.iter_mut().enumerate() {
            let (half, j, rem, _, _) = q3_k_element(e);
            let step = df * scales[8 * half + 2 * j + (rem / 16)] as f32;
            if step == 0.0 {
                continue;
            }
            let w = block[e];
            if w >= 0.0 {
                *level = (w / step).round().clamp(0.0, 3.0) as u8;
                hmask[e % 32] |= 1 << (e / 32);
            } else {
                // Stored value is level + 4 with the high bit CLEAR.
                *level = ((w / step).round().clamp(-4.0, -1.0) + 4.0) as u8;
            }
        }

        let mut qs = [0u8; 64];
        for (e, &level) in levels.iter().enumerate() {
            let (_, _, _, quant_byte, shift) = q3_k_element(e);
            qs[quant_byte] |= level << shift;
        }

        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&packed);
        out.extend_from_slice(&d.to_bits().to_le_bytes());
    }
    out
}

/// FP32 reference for the Q3_K GEMV `y = W * x`, one byte run per output row.
pub fn dequant_q3_k_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_q3_k(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
