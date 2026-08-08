//! Q8_0 reference tests (ROADMAP Phase G Stage 2).
//!
//! The kernel rule says the CPU reference and its test land before any GPU
//! kernel. What that buys here is specific: Q8_0 differs from this port's
//! existing affine quant on three axes at once (signed vs unsigned quants,
//! interleaved vs planar scales, no bias), and each difference has a wrong
//! version that produces finite, plausible numbers rather than an error.

use mrefrust_compute::{
    dequant_q8_0_gemv, dequantize_q8_0, pearson, quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
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
