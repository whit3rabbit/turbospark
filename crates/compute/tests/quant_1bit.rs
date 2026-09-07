//! 1-bit affine reference tests (the `prism-ml/Bonsai-27B-mlx-1bit` layout).
//!
//! Its own test binary rather than a section of `quant.rs`, matching how
//! `quant_gguf_iq.rs` and `quant_gguf_mxfp4.rs` are separated: this one
//! carries a generated fixture the others do not need.
//!
//! The ORACLE is the test that matters, and this decoder needs one more than
//! most: at one bit there is no arithmetic to sanity-check an unpacker
//! against. A wrong bit order, a wrong element-to-byte mapping and a wrong
//! group stride all produce weights of exactly the right MAGNITUDE and only
//! the wrong sign, which correlates near zero without ever looking
//! malformed. MLX produced both sides of the oracle, so this decoder is the
//! only side under test.
//!
//! Everything else is a property check on top of it, and the ones that can
//! be are written as MUTATIONS the decoder must not survive.

use turbospark_compute::quant_1bit::{
    asymmetric_group_count, dequant_int1_gemv, dequant_int1_gemv_symmetric, dequantize_int1_affine,
    embed_lookup_int1, f16_to_f32, f32_to_f16, is_symmetric, quantize_int1_affine_symmetric,
    Int1AffineRow, BONSAI_GROUP_SIZE,
};

// Ranged-read out of the real published checkpoint and decoded by MLX
// itself (`scripts/mlx_1bit_oracle.py`). Under `generated/` for the same
// reason the ggml oracles are.
include!("generated/quant_1bit_oracle.rs");

fn oracle_row() -> Int1AffineRow {
    Int1AffineRow {
        packed: INT1_ORACLE_PACKED.to_vec(),
        scales: INT1_ORACLE_SCALES.to_vec(),
        biases: INT1_ORACLE_BIASES.to_vec(),
        group_size: BONSAI_GROUP_SIZE,
    }
}

/// The one test that can catch this decoder and its author sharing a
/// misreading, because MLX produced both sides of it.
///
/// Compared with `==` and not a tolerance. Nothing here rounds: `q` is 0 or
/// 1, so the product is either `bias` or `bias + scale` and both are exact
/// in FP32 once the FP16 companions are widened.
#[test]
fn decodes_exactly_what_mlx_decodes() {
    let row = oracle_row();
    let got = dequantize_int1_affine(&row, INT1_ORACLE_FLOATS.len());
    assert_eq!(got.len(), INT1_ORACLE_FLOATS.len());
    for (i, (&g, &want)) in got.iter().zip(INT1_ORACLE_FLOATS.iter()).enumerate() {
        assert_eq!(g, want, "element {i} disagrees with MLX");
    }
}

/// MSB-first within the byte is the plausible alternative convention, and it
/// is not this one.
///
/// Stated as a test because the difference is invisible in every other
/// way: reversing the bits of each byte permutes elements within an
/// 8-element run, so the value SET per group, the magnitudes, the group
/// scales and the total popcount are all unchanged. Only an oracle can see
/// it, which is what the oracle is for.
#[test]
fn msb_first_within_the_byte_is_not_the_convention() {
    let mut row = oracle_row();
    row.packed = row.packed.iter().map(|b| b.reverse_bits()).collect();
    let got = dequantize_int1_affine(&row, INT1_ORACLE_FLOATS.len());
    assert_ne!(
        got.as_slice(),
        &INT1_ORACLE_FLOATS[..],
        "MSB-first decoded the same as MLX; the oracle cannot see bit order"
    );
}

/// The second group's scale must be read from the second group.
///
/// A decoder that resolves scale and bias once and reuses them for the whole
/// row passes every single-group test. The oracle row spans two groups with
/// different scales precisely so this is reachable; asserting the two groups
/// decode to different magnitudes is what makes that non-vacuous.
#[test]
fn each_group_is_decoded_with_its_own_scale() {
    let row = oracle_row();
    assert_ne!(
        INT1_ORACLE_SCALES[0], INT1_ORACLE_SCALES[1],
        "the fixture's two groups share a scale, so it cannot catch a group-stride bug"
    );
    let got = dequantize_int1_affine(&row, INT1_ORACLE_FLOATS.len());
    let mag = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert_ne!(
        mag(&got[..BONSAI_GROUP_SIZE]),
        mag(&got[BONSAI_GROUP_SIZE..]),
        "both groups decoded at one magnitude"
    );
}

/// The real checkpoint's groups are symmetric binary, which is the property
/// a `+/-1` GEMV needs and the one the module refuses to assume.
///
/// `scripts/mlx_1bit_oracle.py` re-measures this over all 1,920 groups of
/// the probed tensor every time it regenerates the fixture and prints the
/// result into the file's header; this asserts it on the two frozen here.
#[test]
fn the_probed_checkpoint_is_symmetric_binary() {
    let row = oracle_row();
    assert!(is_symmetric(&row));
    for g in 0..2 {
        let scale = f16_to_f32(INT1_ORACLE_SCALES[g]);
        let bias = f16_to_f32(INT1_ORACLE_BIASES[g]);
        assert_eq!(bias, -scale / 2.0, "group {g}");
    }
    // ...so every decoded weight is one of exactly two opposite values.
    let got = dequantize_int1_affine(&row, INT1_ORACLE_FLOATS.len());
    for (i, w) in got.iter().enumerate() {
        let half = f16_to_f32(INT1_ORACLE_SCALES[i / BONSAI_GROUP_SIZE]) / 2.0;
        assert_eq!(w.abs(), half, "element {i}");
    }
}

/// An asymmetric group is legal in the container, is reported, and is
/// refused by the fast path rather than silently decoded by it.
#[test]
fn an_asymmetric_group_is_reported_and_refused() {
    let mut row = oracle_row();
    // Leave the scale alone and move the bias off `-scale/2`: the group now
    // represents two same-signed values, which the affine decoder handles
    // and the `+/-1` one cannot.
    row.biases[1] = f32_to_f16(0.0);
    assert_eq!(asymmetric_group_count(&row), 1);
    assert!(!is_symmetric(&row));

    let x = vec![1.0f32; row.len()];
    let n = row.len();
    // The affine decoder still answers.
    let affine = dequant_int1_gemv(std::slice::from_ref(&row), &x, n);
    assert!(affine[0].is_finite());
    // The fast path refuses.
    let err = std::panic::catch_unwind(|| dequant_int1_gemv_symmetric(&[row], &x, n));
    assert!(
        err.is_err(),
        "the symmetric GEMV accepted an asymmetric row"
    );
}

/// The sign rule round-trips: every weight comes back as the representable
/// value on its own side of zero.
#[test]
fn the_sign_rule_round_trips() {
    let g = 64;
    let src: Vec<f32> = (0..4 * g)
        .map(|i| ((i as f32 * 0.37).sin() * 0.5) - 0.05)
        .collect();
    let row = quantize_int1_affine_symmetric(&src, g);
    assert_eq!(row.group_size, g);
    assert!(is_symmetric(&row));

    let back = dequantize_int1_affine(&row, src.len());
    for (i, (&w, &q)) in src.iter().zip(back.iter()).enumerate() {
        let half = f16_to_f32(row.scales[i / g]) / 2.0;
        assert_eq!(q.abs(), half, "element {i} left the two-value set");
        assert_eq!(
            q > 0.0,
            w >= 0.0,
            "element {i} ({w}) came back on the wrong side of zero"
        );
    }
}

/// A group of all zeros gets a zero scale and decodes back to zeros, rather
/// than to a pair of `+/-inf` or a NaN.
#[test]
fn an_all_zero_group_survives_the_round_trip() {
    let g = 64;
    let src = vec![0f32; g];
    let row = quantize_int1_affine_symmetric(&src, g);
    assert_eq!(f16_to_f32(row.scales[0]), 0.0);
    for (i, w) in dequantize_int1_affine(&row, g).iter().enumerate() {
        assert_eq!(*w, 0.0, "element {i}");
    }
}

/// The `+/-1` GEMV computes the same thing as the affine one, to FP32
/// rounding.
///
/// Not `==`: factoring the scale out of the group reassociates the sum (see
/// `dequant_int1_gemv_symmetric`'s doc). The tolerance is what separates
/// "reassociated" from "wrong", so it is tight rather than generous.
#[test]
fn the_symmetric_gemv_agrees_with_the_affine_one() {
    let g = 128;
    let n = 4 * g;
    let rows: Vec<Int1AffineRow> = (0..3)
        .map(|r| {
            let src: Vec<f32> = (0..n)
                .map(|i| ((i + 7 * r) as f32 * 0.11).cos() * 0.25)
                .collect();
            quantize_int1_affine_symmetric(&src, g)
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019).sin()).collect();

    let affine = dequant_int1_gemv(&rows, &x, n);
    let fast = dequant_int1_gemv_symmetric(&rows, &x, n);
    for (r, (&a, &f)) in affine.iter().zip(fast.iter()).enumerate() {
        assert!(
            (a - f).abs() <= 1e-6 * a.abs().max(1.0),
            "row {r}: affine {a} vs symmetric {f}"
        );
    }
}

/// The embedding lookup reads the row it was asked for, and the row stride
/// is `D / 8` rather than `D`.
///
/// The real checkpoint quantizes `embed_tokens` at one bit like everything
/// else, so this is a kernel the type genuinely needs. The bug it is written
/// for is the stride: at one bit a row is EIGHT times shorter than the
/// element count, so an off-by-a-factor read lands inside a neighbouring
/// token's weights, which is finite and plausible. The token is deliberately
/// not row 0, and the expected values come from `dequantize_int1_affine` of
/// that row alone.
#[test]
fn the_embedding_lookup_reads_the_right_row_and_scales_it() {
    let (vocab, d, g) = (7usize, 256usize, BONSAI_GROUP_SIZE);
    let rows: Vec<Int1AffineRow> = (0..vocab)
        .map(|t| {
            let src: Vec<f32> = (0..d)
                .map(|i| ((i + 13 * t) as f32 * 0.23).sin() * (1.0 + t as f32))
                .collect();
            quantize_int1_affine_symmetric(&src, g)
        })
        .collect();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in &rows {
        packed.extend_from_slice(&r.packed);
        scales.extend_from_slice(&r.scales);
        biases.extend_from_slice(&r.biases);
    }

    let token = 5usize;
    let out_scale = 4.0f32;
    let got = embed_lookup_int1(&packed, &scales, &biases, token, d, g, out_scale);
    let want: Vec<f32> = dequantize_int1_affine(&rows[token], d)
        .iter()
        .map(|v| v * out_scale)
        .collect();
    assert_eq!(got, want);

    // Non-vacuous on both counts: a neighbouring row decodes differently, so
    // a stride bug would show, and the scale is large enough that dropping
    // it could not pass as rounding.
    assert_ne!(dequantize_int1_affine(&rows[token - 1], d), want);
    let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 1.0, "the test row is too small to prove the scale");
}

/// A short `x` must be refused rather than silently truncated by `zip`,
/// which would return a partial dot product with no indication anything
/// was dropped.
#[test]
#[should_panic(expected = "x.len() must equal n")]
fn dequant_int1_gemv_refuses_a_short_x() {
    let row = oracle_row();
    let n = row.len();
    let short_x = vec![1.0f32; n - 1];
    let _ = dequant_int1_gemv(&[row], &short_x, n);
}

/// A row length that is not a whole number of groups is a caller error, not
/// a truncation.
#[test]
#[should_panic(expected = "not a multiple of the group size")]
fn a_partial_group_is_refused() {
    let row = Int1AffineRow {
        packed: vec![0u8; 24],
        scales: vec![0u16; 2],
        biases: vec![0u16; 2],
        group_size: BONSAI_GROUP_SIZE,
    };
    let _ = dequantize_int1_affine(&row, 192);
}

/// FP16 is not BF16, stated as an assertion because the two planes are the
/// same width and a misread is otherwise silent.
///
/// The oracle's first scale is `0x26f0`. Read as FP16 that is ~0.0271, the
/// magnitude a QAT binary weight has; read as BF16 it is ~1.67e-16.
#[test]
fn the_companions_are_fp16_and_reading_them_as_bf16_is_not_close() {
    let as_f16 = f16_to_f32(INT1_ORACLE_SCALES[0]);
    let as_bf16 = f32::from_bits((INT1_ORACLE_SCALES[0] as u32) << 16);
    assert!((as_f16 - 0.0271).abs() < 1e-3, "scale read {as_f16}");
    assert!(
        as_bf16 < 1e-10,
        "BF16 reading {as_bf16} is close enough to the FP16 one to hide the bug"
    );
}
