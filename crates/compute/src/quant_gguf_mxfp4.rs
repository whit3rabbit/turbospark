//! GGUF MXFP4 reference (ROADMAP M5, the `gpt-oss` family).
//!
//! A third quantization family beside [`crate::quant_gguf`]'s arithmetic
//! reconstructions and [`crate::quant_gguf_iq`]'s codebook ones, and it is
//! genuinely a third rather than a variant of either. Like an IQ type it
//! stores an INDEX into a fixed 16-entry table, so a decoder either has the
//! table or produces plausible garbage. Unlike every type in this port so
//! far, its per-block scale is not a float at all: it is a bare 8-bit
//! EXPONENT (OCP Microscaling's E8M0), so the scale is always an exact power
//! of two and there is no mantissa to round.
//!
//! One block is 32 elements in 17 bytes: the exponent byte, then sixteen
//! bytes of nibble-packed indices. That is the whole format.
//!
//! Four things are silently wrong rather than a fault if carried over from a
//! neighbouring type by habit:
//!
//! 1. **The two nibbles of a byte are not adjacent elements.** Byte `j`
//!    serves elements `j` and `j + 16`, the same split-half layout IQ4_NL
//!    uses and the opposite of what a K-quant habit suggests. Reading them
//!    adjacent gives a correctly-scaled PERMUTATION of the right values,
//!    which correlates well and is wrong.
//! 2. **The codebook is not affine and is not symmetric in its indexing.**
//!    Values are `{0, 1, 2, 3, 4, 6, 8, 12}` and their negatives, with the
//!    sign carried by the index's high bit -- so index 8 is a SECOND zero,
//!    not `-0.5` or a continuation of the ramp. A `q - 8` affine read gets
//!    the sign right and the magnitude wrong everywhere past index 4.
//! 3. **The exponent byte is biased by 127 and then shifted once more.**
//!    ggml's own mapping is the E8M0-to-fp32 "half" variant, i.e. exponent
//!    `e` yields `2^(e - 128)` rather than `2^(e - 127)`, because the
//!    codebook above is the FP4 E2M1 grid scaled by two. Using the plain
//!    E8M0 bias doubles every weight in the model, which produces finite,
//!    correctly-ordered, entirely wrong output.
//! 4. **`e = 0` and `e = 1` are not `2^-128` and `2^-127` by the same
//!    expression.** Those two land in the subnormal range of the shifted
//!    encoding and ggml builds them by shifting a fixed bit pattern instead.
//!    They are far too small to matter numerically and cost nothing to get
//!    right, so [`mxfp4_scale`] does.
//!
//! **There is no `quantize_mxfp4` and there will not be one**, for the same
//! reason [`crate::quant_gguf_iq`] has no encoder: the lossless-repack rule
//! means real bytes arrive already quantized and are copied through verbatim,
//! so nothing in the product would call it, and a round trip through this
//! port's own encoder passes whenever encoder and decoder share a misreading.
//! The check that replaces it is
//! `crates/compute/tests/quant_gguf_mxfp4.rs`, which decodes bytes ggml
//! itself produced and compares with `==`
//! (`scripts/ggml_mxfp4_oracle.c`).

/// Elements in one MXFP4 block.
pub const MXFP4_BLOCK_ELEMS: usize = 32;

/// Bytes in one MXFP4 block: one E8M0 exponent byte then 16 nibble-packed
/// indices. Matches `ggml_type_block(39)` in
/// `turbospark_repack::gguf_header`; the two are checked against each other
/// in `crates/repack`'s tests rather than one importing the other, since
/// this crate must not depend on repack.
pub const MXFP4_BLOCK_BYTES: usize = 17;

/// The FP4 codebook an MXFP4 index expands to.
///
/// RECOVERED FROM GGML, not transcribed: `scripts/ggml_mxfp4_oracle.c`
/// decodes a block whose shared exponent yields a unit scale at each of the
/// sixteen indices, and `MXFP4_ORACLE_CODEBOOK` in the generated fixture is
/// what it read back. The test asserts this array equals that one, so a
/// transcription slip here reddens rather than scaling the model.
///
/// Indices 0 and 8 are BOTH zero, and both are POSITIVE zero. The FP4 E2M1
/// encoding this table stands for would put a sign bit on index 8 and make it
/// `-0.0`; ggml's table is `int8` and so has no signed zero at all, and the
/// oracle test is what settled which of the two this port must reproduce (it
/// reddened on `-0.0`). The distinction never reaches an arithmetic result --
/// both zeros multiply to zero -- but it is the format's answer rather than a
/// plausible one, and a decoder that reads the index as a SIGNED 4-bit
/// integer maps eight of the sixteen entries somewhere else entirely.
pub const MXFP4_VALUES: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

/// The scale an MXFP4 shared-exponent byte stands for.
///
/// Kept as an expression rather than as a 256-entry generated table, the same
/// division [`crate::quant_gguf_iq::iq3xxs_signs`] uses: a computed value
/// cannot go stale, and the generator checks the identity for all 256 bytes
/// so the expression is not merely asserted here.
///
/// `e >= 2` is an ordinary power of two, `2^(e - 128)`, built by writing
/// `e - 1` into the fp32 exponent field. The two bytes below that would need
/// a negative exponent field, so ggml builds them by shifting a fixed
/// subnormal pattern instead; they are reproduced rather than approximated
/// because "close enough at 1e-39" is a judgement this decoder should not be
/// making on the format's behalf.
#[inline]
#[must_use]
pub fn mxfp4_scale(exponent: u8) -> f32 {
    let bits: u32 = if exponent < 2 {
        0x0020_0000u32 << exponent
    } else {
        u32::from(exponent - 1) << 23
    };
    f32::from_bits(bits)
}

/// Dequantize `n` elements of an MXFP4 byte run to FP32.
///
/// `w = MXFP4_VALUES[q] * mxfp4_scale(e)`, where byte `j` of a block holds
/// element `j` in its low nibble and element `j + 16` in its high one.
///
/// A short trailing run is fine: `n` is honoured exactly and the caller's
/// last block may be partially consumed, which is what a GEMV over a row
/// whose width is not a multiple of 32 needs. Callers must still supply
/// whole blocks -- a partial block on disk is not a thing GGUF writes.
#[must_use]
pub fn dequantize_mxfp4(blocks: &[u8], n: usize) -> Vec<f32> {
    let block_count = n.div_ceil(MXFP4_BLOCK_ELEMS);
    assert!(
        blocks.len() >= block_count * MXFP4_BLOCK_BYTES,
        "need {} bytes for {n} elements, got {}",
        block_count * MXFP4_BLOCK_BYTES,
        blocks.len()
    );
    let mut out = vec![0.0f32; n];
    for b in 0..block_count {
        let base = b * MXFP4_BLOCK_BYTES;
        let d = mxfp4_scale(blocks[base]);
        let qs = &blocks[base + 1..base + MXFP4_BLOCK_BYTES];
        for (j, byte) in qs.iter().enumerate() {
            // The split-half layout, stated once: low nibble low half, high
            // nibble high half, never adjacent.
            let lo = b * MXFP4_BLOCK_ELEMS + j;
            let hi = lo + MXFP4_BLOCK_ELEMS / 2;
            if lo < n {
                out[lo] = MXFP4_VALUES[usize::from(byte & 0x0F)] * d;
            }
            if hi < n {
                out[hi] = MXFP4_VALUES[usize::from(byte >> 4)] * d;
            }
        }
    }
    out
}

/// FP32 reference for the MXFP4 GEMV `y = W * x`, one byte run per row.
///
/// Spelled out here rather than reaching for `quant_gguf_iq`'s private
/// `gemv`: five lines of dot product is not worth a cross-module dependency
/// between two quantization families that share no layout.
#[must_use]
pub fn dequant_mxfp4_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequantize_mxfp4(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}
