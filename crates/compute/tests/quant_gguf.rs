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

use turbospark_compute::{
    dequant_q2_k_gemv, dequant_q4_k_gemv, dequant_q5_k_gemv, dequant_q6_k_gemv, dequant_q8_0_gemv,
    dequantize_q2_k, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, pearson,
    quantize_q2_k, quantize_q4_k, quantize_q6_k, quantize_q8_0, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS,
    Q2_K_SUB_ELEMS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q4_K_SUB_ELEMS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6_K_SUB_ELEMS, Q8_0_BLOCK_BYTES,
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
fn q2_k_hand_packed_superblock_obeys_ggmls_two_bit_plane_order() {
    let mut block = vec![0u8; Q2_K_BLOCK_BYTES];
    // d = 0.5, dmin = 0.25. Each scale byte uses low nibble scale, high
    // nibble min. The values exercise both halves and all four bit planes.
    for (sub, byte) in block.iter_mut().take(16).enumerate() {
        *byte = ((sub % 8) as u8 + 1) | (((sub % 4) as u8 + 1) << 4);
    }
    block[80..82].copy_from_slice(&0x3800u16.to_le_bytes());
    block[82..84].copy_from_slice(&0x3400u16.to_le_bytes());
    for half in 0..2 {
        for lane in 0..16 {
            block[16 + half * 32 + lane] = (lane as u8 & 3)
                | (((lane as u8 + 1) & 3) << 2)
                | (((lane as u8 + 2) & 3) << 4)
                | (((lane as u8 + 3) & 3) << 6);
        }
    }
    let got = dequantize_q2_k(&block, Q2_K_BLOCK_ELEMS);
    for e in 0..Q2_K_BLOCK_ELEMS {
        let sub = e / Q2_K_SUB_ELEMS;
        let q_byte = block[16 + (e / 128) * 32 + ((e % 32) / 16) * 16 + e % 16];
        let q = (q_byte >> (2 * ((e % 128) / 32))) & 3;
        let scale = (block[sub] & 15) as f32;
        let min = (block[sub] >> 4) as f32;
        assert_eq!(got[e], 0.5 * scale * q as f32 - 0.25 * min, "element {e}");
    }
}

#[test]
fn q2_k_fixture_encoder_has_the_declared_shape_and_gemv_contract() {
    let n = 2 * Q2_K_BLOCK_ELEMS;
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|seed| quantize_q2_k(&weights(n, seed)))
        .collect();
    assert!(rows.iter().all(|r| r.len() == 2 * Q2_K_BLOCK_BYTES));
    let x = weights(n, 17);
    let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
    let got = dequant_q2_k_gemv(&refs, &x);
    for (row, value) in rows.iter().zip(got) {
        let want: f32 = dequantize_q2_k(row, n)
            .iter()
            .zip(&x)
            .map(|(w, x)| w * x)
            .sum();
        assert_eq!(value, want);
    }
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

// ---------------------------------------------------------------------------
// Q6_K. A third layout rather than a wider Q4_K: six bits per element split
// across two runs, a fixed bias of 32 instead of a per-sub-block min, and
// sixteen SIGNED int8 sub-block scales stored as plain bytes.
// ---------------------------------------------------------------------------

#[test]
fn a_q6_k_superblock_is_two_quant_runs_sixteen_scales_and_a_super_scale() {
    assert_eq!(Q6_K_BLOCK_ELEMS, 256);
    assert_eq!(Q6_K_SUB_ELEMS, 16);
    // 128 low-nibble bytes + 64 high-bit bytes + 16 scales + one f16.
    assert_eq!(Q6_K_BLOCK_BYTES, 128 + 64 + 16 + 2);
    let bytes = quantize_q6_k(&weights(2 * Q6_K_BLOCK_ELEMS, 31));
    assert_eq!(bytes.len(), 2 * Q6_K_BLOCK_BYTES);
}

/// The load-bearing Q6_K test, and the sibling of the hand-packed Q4_K one: a
/// superblock assembled from LITERAL bytes, decoded against the documented
/// formula `w = d * sc[is + 2k] * (level - 32)` with every level worked out by
/// hand rather than by calling the code under test.
///
/// Every byte is one of two values, which is what makes the expectations
/// hand-checkable: `ql = 0x93` gives low nibble 3 and high nibble 9, and
/// `qh = 0xB1` gives the four bit-pairs 1, 0, 3, 2 from low to high. So the
/// four levels a lane serves are 3|16 = 19, 3|0 = 3, 9|48 = 57 and 9|32 = 41,
/// i.e. quants -13, -29, 25 and 9 after the bias.
///
/// That pins five things at once, each of which is finite and plausible when
/// got wrong:
///
/// - the six bits SPLIT across `ql` and `qh` (dropping `qh` turns 57 into 9),
/// - the fixed bias of 32 (dropping it makes every value non-negative),
/// - the SIGNED scales (half of the sixteen below are negative),
/// - the scale index striding by 2 per quarter, not by 1,
/// - the 32-element spacing between the four values one `qh` byte serves.
///
/// `d` is 0.5 and every scale and quant is a small integer, so each expected
/// value is exact in FP32 and this compares with `==`.
#[test]
fn a_hand_packed_q6_k_superblock_decodes_to_the_documented_formula() {
    const SCALES: [i8; 16] = [
        1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
    ];
    const QUANTS: [f32; 4] = [-13.0, -29.0, 25.0, 9.0];
    const D: f32 = 0.5;

    let mut block = Vec::with_capacity(Q6_K_BLOCK_BYTES);
    block.extend_from_slice(&[0x93u8; 128]);
    block.extend_from_slice(&[0xB1u8; 64]);
    block.extend(SCALES.iter().map(|&s| s as u8));
    // f16 0.5 is sign 0, exponent field 14, zero mantissa.
    block.extend_from_slice(&0x3800u16.to_le_bytes());
    assert_eq!(block.len(), Q6_K_BLOCK_BYTES);

    let out = dequantize_q6_k(&block, Q6_K_BLOCK_ELEMS);
    for h in 0..2 {
        for k in 0..4 {
            for l in 0..32 {
                let is = l / Q6_K_SUB_ELEMS;
                let want = D * SCALES[h * 8 + is + 2 * k] as f32 * QUANTS[k];
                let at = h * 128 + k * 32 + l;
                assert_eq!(out[at], want, "half {h} quarter {k} lane {l}");
            }
        }
    }
    // Spot-check two of them against numbers written out longhand, so the
    // loop above cannot be satisfied by an index scheme that is wrong in the
    // same way on both sides.
    assert_eq!(out[0], -6.5); // 0.5 * 1 * -13
    assert_eq!(out[128 + 32 + 16], 174.0); // 0.5 * -12 * -29
}

#[test]
fn q6_k_round_trip_stays_inside_one_quantization_step() {
    let w = weights(2 * Q6_K_BLOCK_ELEMS, 41);
    let out = dequantize_q6_k(&quantize_q6_k(&w), w.len());

    for (j, sub) in w.chunks_exact(Q6_K_SUB_ELEMS).enumerate() {
        let amax = sub.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
        // A sub-block keeps 32 levels of its own extreme, and its scale is
        // reached through an int8 quantized against the superblock maximum,
        // so allow that second rounding on top of the half-step.
        let bound = amax / 32.0 * 0.5 + amax * 2e-2;
        for (ii, &orig) in sub.iter().enumerate() {
            let got = out[j * Q6_K_SUB_ELEMS + ii];
            assert!(
                (got - orig).abs() <= bound,
                "sub-block {j} element {ii}: {got} vs {orig}, bound {bound}"
            );
        }
    }
}

/// ggml derives the sub-block scales through a NEGATIVE `iscale`, so a real
/// file carries negative scale bytes wherever a sub-block's extreme is
/// positive. Reading them as `u8` mirrors whole 16-element runs and stays
/// finite, so the sign is asserted on both the stored bytes and the decode.
#[test]
fn sub_block_scales_are_signed_and_both_signs_occur() {
    // Alternating sub-blocks whose extreme is positive then negative.
    let w: Vec<f32> = (0..Q6_K_BLOCK_ELEMS)
        .map(|e| {
            let sign = if (e / Q6_K_SUB_ELEMS) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            sign * (1.0 + (e % Q6_K_SUB_ELEMS) as f32 / 16.0)
        })
        .collect();
    let bytes = quantize_q6_k(&w);
    let scales: Vec<i8> = bytes[192..208].iter().map(|&b| b as i8).collect();
    assert!(
        scales.iter().any(|&s| s < 0) && scales.iter().any(|&s| s > 0),
        "expected both signs among {scales:?}"
    );

    let out = dequantize_q6_k(&bytes, w.len());
    for (e, (&got, &orig)) in out.iter().zip(w.iter()).enumerate() {
        assert!(
            got.signum() == orig.signum(),
            "element {e} flipped sign: {got} vs {orig}"
        );
        assert!((got - orig).abs() < 0.1, "element {e}: {got} vs {orig}");
    }
}

/// An all-zero superblock has no extreme, so `iscale` would divide by zero.
/// Real routed experts and vocab rows do contain all-zero runs (AGENTS.md
/// Gotcha 30), so this is the shape a padded or dead row takes.
#[test]
fn an_all_zero_q6_k_superblock_dequantizes_to_zeros() {
    let out = dequantize_q6_k(
        &quantize_q6_k(&vec![0.0; Q6_K_BLOCK_ELEMS]),
        Q6_K_BLOCK_ELEMS,
    );
    assert!(out.iter().all(|&v| v == 0.0), "got {out:?}");
}

#[test]
fn q6_k_gemv_matches_a_dequantize_then_multiply() {
    let n = 2 * Q6_K_BLOCK_ELEMS;
    let x = weights(n, 13);
    let rows: Vec<Vec<u8>> = (0..4).map(|r| quantize_q6_k(&weights(n, 50 + r))).collect();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    let got = dequant_q6_k_gemv(&refs, &x, n);
    for (i, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_q6_k(row, n)
            .iter()
            .zip(x.iter())
            .map(|(w, xv)| w * xv)
            .sum();
        assert!((got[i] - want).abs() <= want.abs() * 1e-6 + 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Q5_K (ROADMAP Phase M2). Mixtral's Q4_K_M puts it on `attn_output`.
// ---------------------------------------------------------------------------

// ggml quantized these bytes and decoded these floats
// (`scripts/ggml_q5_k_oracle.c`). Under `generated/` for the same reason the
// IQ oracles are: every top-level file in `tests/` is its own test binary.
include!("generated/quant_gguf_q5_k_oracle.rs");

/// The one test that can catch this decoder and its author sharing a
/// misreading, because ggml produced BOTH sides of it. Everything else in
/// this file about Q5_K is a property check on top.
///
/// Compared with `==` and not a tolerance: the arithmetic is grouped exactly
/// as ggml groups it (`d * sc` first, then `q * that - dmin * m`), so any
/// difference at all is a real one rather than float drift.
#[test]
fn q5_k_decodes_exactly_what_ggml_decodes() {
    let got = dequantize_q5_k(&Q5_K_ORACLE_BYTES, Q5_K_ORACLE_FLOATS.len());
    assert_eq!(got.len(), Q5_K_ORACLE_FLOATS.len());
    for (i, (&g, &want)) in got.iter().zip(Q5_K_ORACLE_FLOATS.iter()).enumerate() {
        assert_eq!(g, want, "element {i} disagrees with ggml");
    }
}

/// The oracle spans two superblocks so that block 1 cannot be decoded with
/// block 0's scales and still pass. Asserted rather than assumed, since a
/// shorter fixture would silently weaken the test above.
#[test]
fn the_q5_k_oracle_covers_more_than_one_superblock() {
    assert_eq!(Q5_K_ORACLE_BYTES.len() % Q5_K_BLOCK_BYTES, 0);
    assert!(Q5_K_ORACLE_BYTES.len() / Q5_K_BLOCK_BYTES >= 2);
    assert_eq!(
        Q5_K_ORACLE_FLOATS.len(),
        Q5_K_ORACLE_BYTES.len() / Q5_K_BLOCK_BYTES * Q5_K_BLOCK_ELEMS
    );
}

/// The fifth bit is the whole difference from Q4_K, and dropping it leaves
/// values that are correctly signed, correctly ordered and merely compressed.
/// So: clearing `qh` must MOVE the decode, and it must only ever move it
/// down by whole multiples of that sub-block's step.
#[test]
fn clearing_the_fifth_bit_run_changes_the_decode() {
    let full = dequantize_q5_k(&Q5_K_ORACLE_BYTES, Q5_K_ORACLE_FLOATS.len());

    let mut stripped = Q5_K_ORACLE_BYTES;
    for block in 0..stripped.len() / Q5_K_BLOCK_BYTES {
        let at = block * Q5_K_BLOCK_BYTES + 16;
        stripped[at..at + Q5_K_BLOCK_ELEMS / 8].fill(0);
    }
    let without = dequantize_q5_k(&stripped, Q5_K_ORACLE_FLOATS.len());

    assert_ne!(full, without, "qh contributes nothing: it is being ignored");
    for (a, b) in full.iter().zip(without.iter()) {
        assert!(a >= b, "clearing a high bit must never raise a weight");
    }
}

/// Q5_K keeps Q4_K's per-sub-block MIN, and a symmetric-quant habit carried
/// over from Q6_K drops it. Without the subtraction every decoded weight is
/// non-negative, so the presence of negatives is the cheap witness.
#[test]
fn q5_k_reconstructs_negative_weights() {
    let got = dequantize_q5_k(&Q5_K_ORACLE_BYTES, Q5_K_ORACLE_FLOATS.len());
    assert!(got.iter().any(|&v| v < 0.0), "no negative weights decoded");
    assert!(got.iter().any(|&v| v > 0.0));
}

#[test]
fn q5_k_gemv_matches_dequantize_then_dot() {
    let n = Q5_K_BLOCK_ELEMS * 2;
    let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();
    let rows: Vec<&[u8]> = vec![&Q5_K_ORACLE_BYTES, &Q5_K_ORACLE_BYTES];

    let got = dequant_q5_k_gemv(&rows, &x, n);
    let want: f32 = dequantize_q5_k(&Q5_K_ORACLE_BYTES, n)
        .iter()
        .zip(x.iter())
        .map(|(w, xv)| w * xv)
        .sum();
    for v in got {
        assert!((v - want).abs() <= want.abs() * 1e-6 + 1e-6);
    }
}
