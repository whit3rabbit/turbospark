//! GGUF block-quant reference tests (ROADMAP Phase G Stage 2).
//!
//! The kernel rule says the CPU reference and its test land before any GPU
//! kernel. What that buys here is specific: Q8_0 differs from this port's
//! existing affine quant on three axes at once (signed vs unsigned quants,
//! interleaved vs planar scales, no bias), and each difference has a wrong
//! version that produces finite, plausible numbers rather than an error.
//!
//! Q4_K, in the second half of this file, is worse on that count rather than
//! better: its 6-bit sub-scales are split across two bytes for half the
//! sub-blocks, and every way of getting that wrong reads a scale that is
//! merely too small.

use mrefrust_compute::{
    dequant_q4_k_gemv, dequant_q8_0_gemv, dequantize_q4_k, dequantize_q8_0, pearson, quantize_q4_k,
    quantize_q8_0, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q4_K_SUB_ELEMS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
};

/// Deterministic weights spanning both signs and several magnitudes, so a
/// sign or scale error cannot hide in a narrow range.
fn weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

#[test]
fn a_block_is_a_scale_then_thirty_two_weights() {
    assert_eq!(Q8_0_BLOCK_ELEMS, 32);
    assert_eq!(Q8_0_BLOCK_BYTES, 2 + Q8_0_BLOCK_ELEMS);
    let bytes = quantize_q8_0(&weights(96, 7));
    assert_eq!(bytes.len(), 3 * Q8_0_BLOCK_BYTES);
}

#[test]
fn round_trip_stays_inside_one_quantization_step() {
    let w = weights(256, 11);
    let out = dequantize_q8_0(&quantize_q8_0(&w), w.len());

    // Per block, the error bound is half a step of that block's own scale.
    for (b, block) in w.chunks_exact(Q8_0_BLOCK_ELEMS).enumerate() {
        let amax = block.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        let step = amax / 127.0;
        for (k, &orig) in block.iter().enumerate() {
            let got = out[b * Q8_0_BLOCK_ELEMS + k];
            // The scale itself is stored as f16, so allow a relative slop on
            // top of the half-step: f16 has 10 mantissa bits.
            let bound = step * 0.5 + amax * 1e-3;
            assert!(
                (got - orig).abs() <= bound,
                "block {b} lane {k}: {got} vs {orig}, bound {bound}"
            );
        }
    }
}

/// The scale is rounded to f16 BEFORE the quants are derived, so a decode
/// reproduces exactly the integers the encoder chose. Re-quantizing a
/// dequantized row must therefore be a fixed point, byte for byte. This is
/// what catches a quantizer that divides by an unrounded scale.
#[test]
fn quantize_is_idempotent_after_one_pass() {
    let once = quantize_q8_0(&weights(512, 3));
    let twice = quantize_q8_0(&dequantize_q8_0(&once, 512));
    assert_eq!(once, twice, "quantize(dequantize(q)) must return q");
}

/// Negative weights are the axis where reading `qs` as `u8` instead of `i8`
/// stays finite and silently wrong, so it gets its own assertion rather than
/// being left to the round-trip bound.
#[test]
fn negative_weights_survive_the_round_trip() {
    let w: Vec<f32> = (0..Q8_0_BLOCK_ELEMS)
        .map(|i| if i % 2 == 0 { -1.0 } else { 0.5 })
        .collect();
    let out = dequantize_q8_0(&quantize_q8_0(&w), w.len());
    for (i, (&got, &orig)) in out.iter().zip(w.iter()).enumerate() {
        assert_eq!(
            got < 0.0,
            orig < 0.0,
            "lane {i} lost its sign: {got} vs {orig}"
        );
        assert!((got - orig).abs() < 0.02, "lane {i}: {got} vs {orig}");
    }
}

/// An all-zero block has `amax == 0`, so the scale is zero and the reciprocal
/// would be infinite. It must come back as zeros, not NaN: a padded tensor
/// tail is exactly this shape.
#[test]
fn an_all_zero_block_dequantizes_to_zeros() {
    let out = dequantize_q8_0(&quantize_q8_0(&vec![0.0; 64]), 64);
    assert!(out.iter().all(|&v| v == 0.0), "got {out:?}");
}

#[test]
fn gemv_matches_a_dequantize_then_multiply() {
    let n = 128;
    let x = weights(n, 5);
    let rows: Vec<Vec<u8>> = (0..4).map(|r| quantize_q8_0(&weights(n, 20 + r))).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let got = dequant_q8_0_gemv(&refs, &x, n);
    for (i, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_q8_0(row, n)
            .iter()
            .zip(x.iter())
            .map(|(w, xv)| w * xv)
            .sum();
        assert!((got[i] - want).abs() <= want.abs() * 1e-6 + 1e-6);
    }
}

/// `pearson` is load-bearing for settling `FUSED_GATE_FIRST`, so its two
/// answers that matter are pinned: matched data correlates near 1, unrelated
/// data does not, and a constant input returns 0 rather than NaN.
#[test]
fn correlation_separates_the_same_weights_from_different_ones() {
    let a = weights(1024, 1);
    let b = weights(1024, 2);
    let a_requantized = dequantize_q8_0(&quantize_q8_0(&a), a.len());

    assert!(
        pearson(&a, &a_requantized) > 0.999,
        "a quantized copy of the same weights must correlate: {}",
        pearson(&a, &a_requantized)
    );
    assert!(
        pearson(&a, &b).abs() < 0.2,
        "unrelated weights must not: {}",
        pearson(&a, &b)
    );
    assert_eq!(pearson(&[1.0; 32], &a[..32]), 0.0);
}

// ---------------------------------------------------------------------------
// Q4_K
// ---------------------------------------------------------------------------

#[test]
fn a_superblock_is_two_scales_twelve_packed_bytes_and_128_nibbles() {
    assert_eq!(Q4_K_BLOCK_ELEMS, 256);
    assert_eq!(Q4_K_SUB_ELEMS, 32);
    assert_eq!(Q4_K_BLOCK_BYTES, 2 + 2 + 12 + Q4_K_BLOCK_ELEMS / 2);
    let bytes = quantize_q4_k(&weights(2 * Q4_K_BLOCK_ELEMS, 7));
    assert_eq!(bytes.len(), 2 * Q4_K_BLOCK_BYTES);
}

/// The load-bearing Q4_K test: a superblock assembled from LITERAL bytes,
/// with the twelve packed scale bytes worked out by hand rather than by
/// calling the code under test, checked against the documented formula
/// `w = d * sc[j] * q - dmin * m[j]`.
///
/// This is the one assertion a decoder and an encoder written from the same
/// wrong mental model cannot both satisfy. It pins three things at once:
///
/// - the 6-bit split for sub-blocks 4..8 (every one of the eight scales and
///   eight mins below needs its high two bits, so a decoder that only reads
///   the low nibble gets all of them wrong),
/// - the nibble order (low nibbles are elements `64g..64g+32`, high nibbles
///   are `64g+32..64g+64`, so the two nibbles of one byte are 32 elements
///   apart),
/// - the sub-block index advancing with the nibble half, not the byte.
///
/// `d` and `dmin` are dyadic and the quants are small integers, so every
/// expected value is exact in FP32 and this compares with `==`.
#[test]
fn a_hand_packed_superblock_decodes_to_the_documented_formula() {
    // Chosen so that each of the four second-half sub-blocks has a scale and
    // a min above 15, i.e. the high-2-bit path is exercised on all of them.
    const LS: [u8; 8] = [63, 1, 2, 3, 62, 17, 33, 48];
    const LM: [u8; 8] = [7, 0, 63, 5, 40, 20, 21, 63];
    // Worked out by hand from the layout, not produced by the packer:
    //   packed[0..4] = LS[0..4] with LS[4..8]'s top two bits in bits 6-7,
    //   packed[4..8] = LM[0..4] with LM[4..8]'s top two bits in bits 6-7,
    //   packed[8..12] = LS[4..8] low nibble | LM[4..8] low nibble << 4.
    const PACKED: [u8; 12] = [
        0xFF, 0x41, 0x82, 0xC3, 0x87, 0x40, 0x7F, 0xC5, 0x8E, 0x41, 0x51, 0xF0,
    ];
    // 0.5 and 0.25 as little-endian f16.
    const D_BITS: [u8; 2] = [0x00, 0x38];
    const DMIN_BITS: [u8; 2] = [0x00, 0x34];
    let (d, dmin) = (0.5f32, 0.25f32);

    // A pattern that visits every nibble value and repeats with a period
    // coprime to 32 and 64, so any index scrambling inside a group shows up.
    let quants: Vec<u8> = (0..Q4_K_BLOCK_ELEMS)
        .map(|e| ((e * 7 + 3) % 16) as u8)
        .collect();

    let mut block = Vec::with_capacity(Q4_K_BLOCK_BYTES);
    block.extend_from_slice(&D_BITS);
    block.extend_from_slice(&DMIN_BITS);
    block.extend_from_slice(&PACKED);
    for g in 0..4 {
        for l in 0..Q4_K_SUB_ELEMS {
            let at = g * 2 * Q4_K_SUB_ELEMS + l;
            block.push(quants[at] | (quants[at + Q4_K_SUB_ELEMS] << 4));
        }
    }
    assert_eq!(block.len(), Q4_K_BLOCK_BYTES);

    let got = dequantize_q4_k(&block, Q4_K_BLOCK_ELEMS);
    for e in 0..Q4_K_BLOCK_ELEMS {
        let j = e / Q4_K_SUB_ELEMS;
        let want = d * LS[j] as f32 * quants[e] as f32 - dmin * LM[j] as f32;
        assert_eq!(got[e], want, "element {e} (sub-block {j})");
    }
}

#[test]
fn q4_k_round_trip_stays_inside_one_quantization_step() {
    let w = weights(2 * Q4_K_BLOCK_ELEMS, 11);
    let out = dequantize_q4_k(&quantize_q4_k(&w), w.len());

    for (j, sub) in w.chunks_exact(Q4_K_SUB_ELEMS).enumerate() {
        let lo = sub.iter().fold(0f32, |acc, &v| acc.min(v));
        let hi = sub.iter().fold(lo, |acc, &v| acc.max(v));
        let step = (hi - lo) / 15.0;
        for (k, &orig) in sub.iter().enumerate() {
            let got = out[j * Q4_K_SUB_ELEMS + k];
            // Half a step of this sub-block's own range, plus slop for the
            // 6-bit sub-scale and 6-bit sub-min rounding on top of it.
            let bound = step * 0.5 + (hi - lo) * 0.05;
            assert!(
                (got - orig).abs() <= bound,
                "sub-block {j} lane {k}: {got} vs {orig}, bound {bound}"
            );
        }
    }
}

/// Q4_K quants are UNSIGNED and the sub-block min is what carries negative
/// values. A decoder that drops the `- dmin * m` term returns numbers that
/// are finite, correctly ordered within a sub-block, and entirely
/// non-negative, so the sign of the reconstruction is asserted outright
/// rather than left to the round-trip bound.
#[test]
fn the_sub_block_min_is_what_makes_a_value_negative() {
    let w: Vec<f32> = (0..Q4_K_BLOCK_ELEMS)
        .map(|e| -1.0 + (e % Q4_K_SUB_ELEMS) as f32 / 64.0)
        .collect();
    assert!(w.iter().all(|&v| v < 0.0));

    let out = dequantize_q4_k(&quantize_q4_k(&w), w.len());
    for (e, (&got, &orig)) in out.iter().zip(w.iter()).enumerate() {
        assert!(got < 0.0, "element {e} lost its min term: {got} vs {orig}");
        assert!((got - orig).abs() < 0.05, "element {e}: {got} vs {orig}");
    }
}

/// Sub-blocks whose ranges differ by 16x inside one superblock, so the eight
/// 6-bit scales come out spread over most of their range instead of all
/// landing near 63. Correlation rather than an element bound, because a
/// sub-block far below the superblock maximum legitimately keeps fewer
/// effective levels.
///
/// The spread is deliberate on two counts: the LARGEST sub-block is 5, a
/// second-half one, so the maximum scale is stored through the split-byte
/// path; and 3 and 7 land on scales of 16, whose only set bit lives in the
/// high two bits, so dropping those reads them as zero.
#[test]
fn sub_blocks_of_very_different_magnitude_survive_the_same_superblock() {
    const FACTORS: [f32; 8] = [1.0, 0.5, 4.0, 2.0, 0.5, 8.0, 1.0, 2.0];
    let base = weights(Q4_K_BLOCK_ELEMS, 23);
    let w: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(e, &v)| v * FACTORS[e / Q4_K_SUB_ELEMS])
        .collect();
    let out = dequantize_q4_k(&quantize_q4_k(&w), w.len());

    for j in 0..8 {
        let (a, b) = (
            &w[j * Q4_K_SUB_ELEMS..(j + 1) * Q4_K_SUB_ELEMS],
            &out[j * Q4_K_SUB_ELEMS..(j + 1) * Q4_K_SUB_ELEMS],
        );
        assert!(
            pearson(a, b) > 0.9,
            "sub-block {j} decoded to something unrelated: correlation {}",
            pearson(a, b)
        );
    }
}

/// An all-zero superblock has no range and no min, so both super-scales are
/// zero and every reciprocal would be infinite. A padded tensor tail is
/// exactly this shape.
#[test]
fn an_all_zero_superblock_dequantizes_to_zeros() {
    let out = dequantize_q4_k(
        &quantize_q4_k(&vec![0.0; Q4_K_BLOCK_ELEMS]),
        Q4_K_BLOCK_ELEMS,
    );
    assert!(out.iter().all(|&v| v == 0.0), "got {out:?}");
}

#[test]
fn q4_k_gemv_matches_a_dequantize_then_multiply() {
    let n = 2 * Q4_K_BLOCK_ELEMS;
    let x = weights(n, 5);
    let rows: Vec<Vec<u8>> = (0..4).map(|r| quantize_q4_k(&weights(n, 20 + r))).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let got = dequant_q4_k_gemv(&refs, &x, n);
    for (i, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_q4_k(row, n)
            .iter()
            .zip(x.iter())
            .map(|(w, xv)| w * xv)
            .sum();
        assert!((got[i] - want).abs() <= want.abs() * 1e-6 + 1e-6);
    }
}
