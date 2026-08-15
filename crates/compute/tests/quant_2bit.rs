//! 2-bit affine reference tests (the `prism-ml/Ternary-Bonsai-27B-mlx-2bit`
//! layout, ROADMAP's ternary entry).
//!
//! Its own test binary for the reason `quant_1bit.rs` is: it carries a
//! generated fixture the others do not need.
//!
//! The ORACLE is again the test that matters, and for a reason that survives
//! the extra bit. Permuting the two-bit fields inside a packed word permutes
//! the multiset of levels without changing it, so the group scales, the level
//! histogram and every magnitude in the row are all invariant under a wrong
//! field order. MLX produced both sides of the oracle, so this decoder is the
//! only side under test.
//!
//! Everything else is a property check on top of it, written as MUTATIONS the
//! decoder must not survive wherever it can be.

use turbospark_compute::quant_2bit::{
    asymmetric_group_count_int2, dequant_int2_gemv, dequantize_int2_affine, embed_lookup_int2,
    f16_to_f32, f32_to_f16, is_ternary_symmetric, quantize_int2_affine_ternary, uses_fourth_level,
    Int2AffineRow, TERNARY_GROUP_SIZE,
};

// Ranged-read out of the real published checkpoint and decoded by MLX itself
// (`scripts/mlx_2bit_oracle.py`). Under `generated/` for the same reason the
// ggml oracles are.
include!("generated/quant_2bit_oracle.rs");

fn oracle_row() -> Int2AffineRow {
    Int2AffineRow {
        packed: INT2_ORACLE_PACKED.to_vec(),
        scales: INT2_ORACLE_SCALES.to_vec(),
        biases: INT2_ORACLE_BIASES.to_vec(),
        group_size: TERNARY_GROUP_SIZE,
    }
}

/// Reverses the order of the four 2-bit fields inside every byte.
///
/// The plausible alternative convention, and the one mutation the oracle
/// exists to catch.
fn reverse_fields(b: u8) -> u8 {
    ((b & 3) << 6) | (((b >> 2) & 3) << 4) | (((b >> 4) & 3) << 2) | ((b >> 6) & 3)
}

/// The one test that can catch this decoder and its author sharing a
/// misreading, because MLX produced both sides of it.
///
/// Compared with `==` and not a tolerance. Nothing here rounds: `q` is a
/// small integer, so `q * scale + bias` is exact in FP32 once the FP16
/// companions are widened.
#[test]
fn decodes_exactly_what_mlx_decodes() {
    let row = oracle_row();
    let got = dequantize_int2_affine(&row, INT2_ORACLE_FLOATS.len());
    assert_eq!(got.len(), INT2_ORACLE_FLOATS.len());
    for (i, (&g, &want)) in got.iter().zip(INT2_ORACLE_FLOATS.iter()).enumerate() {
        assert_eq!(g, want, "element {i} disagrees with MLX");
    }
}

/// Reversing the four fields inside each byte is not the convention, and the
/// fixture can tell.
///
/// The `assert_ne` in the middle is the load-bearing half: it establishes
/// that these bytes DISCRIMINATE the two orders at all, which is not free.
/// A byte whose four fields are equal decodes identically under both, so a
/// tidier fixture could pass this test while proving nothing.
#[test]
fn reversed_field_order_within_the_byte_is_not_the_convention() {
    let discriminating = INT2_ORACLE_PACKED.iter().any(|&b| reverse_fields(b) != b);
    assert!(
        discriminating,
        "every fixture byte is palindromic in its fields; the oracle cannot see field order"
    );

    let mut row = oracle_row();
    row.packed = row.packed.iter().map(|&b| reverse_fields(b)).collect();
    let got = dequantize_int2_affine(&row, INT2_ORACLE_FLOATS.len());
    assert_ne!(
        got.as_slice(),
        &INT2_ORACLE_FLOATS[..],
        "the reversed order decoded the same as MLX; the oracle cannot see field order"
    );
}

/// The level histogram is invariant under field order, stated as a test so
/// nobody reads "level 3 never occurs" as evidence about the packing.
///
/// This is the 2-bit analogue of the popcount trap at one bit, and it is why
/// the previous test needs an oracle rather than a self-consistency check.
#[test]
fn the_level_histogram_cannot_see_field_order() {
    let row = oracle_row();
    let mut mutated = oracle_row();
    mutated.packed = mutated.packed.iter().map(|&b| reverse_fields(b)).collect();

    let histogram = |r: &Int2AffineRow| {
        let mut counts = [0usize; 4];
        for i in 0..r.len() {
            counts[r.quant(i) as usize] += 1;
        }
        counts
    };
    assert_eq!(histogram(&row), histogram(&mutated));
}

/// The second group's scale must be read from the second group.
///
/// A decoder that resolves scale and bias once and reuses them for the whole
/// row passes every single-group test. The oracle row spans two groups with
/// different scales precisely so this is reachable; asserting the two groups
/// decode to different magnitudes is what makes it non-vacuous.
#[test]
fn each_group_is_decoded_with_its_own_scale() {
    let row = oracle_row();
    assert_ne!(
        INT2_ORACLE_SCALES[0], INT2_ORACLE_SCALES[1],
        "the fixture's two groups share a scale, so it cannot catch a group-stride bug"
    );
    let got = dequantize_int2_affine(&row, INT2_ORACLE_FLOATS.len());
    let mag = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert_ne!(
        mag(&got[..TERNARY_GROUP_SIZE]),
        mag(&got[TERNARY_GROUP_SIZE..]),
        "both groups decoded at one magnitude"
    );
}

/// The real checkpoint is TERNARY: `bias == -scale`, and the fourth level is
/// never used, so every weight is one of `-scale`, `0`, `+scale`.
///
/// `scripts/mlx_2bit_oracle.py` re-measures both properties over all 1,920
/// groups and all 245,760 elements of the probed tensor every time it
/// regenerates the fixture, and prints the result into the file's header;
/// this asserts them on the slice frozen here. Note the two are INDEPENDENT
/// -- `bias == -scale` alone would still permit a `+2 * scale` fourth level.
#[test]
fn the_probed_checkpoint_is_ternary() {
    let row = oracle_row();
    assert!(is_ternary_symmetric(&row));
    for g in 0..2 {
        let scale = f16_to_f32(INT2_ORACLE_SCALES[g]);
        let bias = f16_to_f32(INT2_ORACLE_BIASES[g]);
        assert_eq!(bias, -scale, "group {g}");
    }
    assert!(!uses_fourth_level(&row));

    let got = dequantize_int2_affine(&row, INT2_ORACLE_FLOATS.len());
    let mut seen_zero = false;
    let mut seen_negative = false;
    let mut seen_positive = false;
    for (i, w) in got.iter().enumerate() {
        let scale = f16_to_f32(INT2_ORACLE_SCALES[i / TERNARY_GROUP_SIZE]);
        assert!(
            *w == 0.0 || w.abs() == scale,
            "element {i} ({w}) is off the ternary grid for scale {scale}"
        );
        seen_zero |= *w == 0.0;
        seen_negative |= *w < 0.0;
        seen_positive |= *w > 0.0;
    }
    // All three levels are actually present, or the assertion above is
    // satisfied by a row that only ever uses one of them.
    assert!(seen_zero && seen_negative && seen_positive);
}

/// An asymmetric group is legal in the container and is reported rather than
/// normalized away.
///
/// There is no fast path to refuse it here (see the module header), so what
/// this pins is that the affine decoder keeps decoding it correctly -- the
/// grid simply stops being centred on zero.
#[test]
fn an_asymmetric_group_is_reported_and_still_decodes() {
    let mut row = oracle_row();
    row.biases[1] = f32_to_f16(0.0);
    assert_eq!(asymmetric_group_count_int2(&row), 1);
    assert!(!is_ternary_symmetric(&row));

    let got = dequantize_int2_affine(&row, INT2_ORACLE_FLOATS.len());
    let scale = f16_to_f32(INT2_ORACLE_SCALES[1]);
    for (i, w) in got[TERNARY_GROUP_SIZE..].iter().enumerate() {
        assert!(*w >= 0.0, "element {i} of a zero-bias group went negative");
        assert!(w.abs() <= 3.0 * scale);
    }
}

/// A row that DOES use the fourth level decodes through the same path, and is
/// reported as using it.
///
/// The container permits it; only this checkpoint's QAT recipe does not. A
/// decoder that had baked in "three levels" would mis-decode this row rather
/// than refuse it, which is why the property is measured rather than assumed.
#[test]
fn the_fourth_level_is_permitted_and_reported() {
    let g = 4;
    let row = Int2AffineRow {
        // One byte, fields 0..4 in order: levels 0, 1, 2, 3.
        packed: vec![0b11_10_01_00],
        scales: vec![f32_to_f16(0.5)],
        biases: vec![f32_to_f16(-0.5)],
        group_size: g,
    };
    assert!(uses_fourth_level(&row));
    assert!(is_ternary_symmetric(&row), "bias == -scale still holds");
    assert_eq!(dequantize_int2_affine(&row, 4), vec![-0.5, 0.0, 0.5, 1.0]);
}

/// The ternary rule round-trips: every weight comes back on its own side of
/// zero, or at zero when it is inside the dead band.
#[test]
fn the_ternary_rule_round_trips() {
    let g = 64;
    let src: Vec<f32> = (0..4 * g)
        .map(|i| ((i as f32 * 0.37).sin() * 0.5) - 0.05)
        .collect();
    let row = quantize_int2_affine_ternary(&src, g);
    assert_eq!(row.group_size, g);
    assert!(is_ternary_symmetric(&row));
    assert!(!uses_fourth_level(&row), "the quantizer emitted level 3");

    let back = dequantize_int2_affine(&row, src.len());
    let mut zeros = 0;
    for (i, (&w, &q)) in src.iter().zip(back.iter()).enumerate() {
        let scale = f16_to_f32(row.scales[i / g]);
        assert!(
            q == 0.0 || q.abs() == scale,
            "element {i} left the three-value set"
        );
        if q == 0.0 {
            zeros += 1;
            assert!(
                w.abs() <= scale / 2.0,
                "element {i} ({w}) zeroed outside the dead band"
            );
        } else {
            assert_eq!(
                q > 0.0,
                w > 0.0,
                "element {i} ({w}) came back on the wrong side of zero"
            );
        }
    }
    // Non-vacuous: the dead band actually catches something, so the branch
    // above is exercised rather than merely present.
    assert!(zeros > 0, "no weight landed in the dead band");
}

/// A group of all zeros gets a zero scale and decodes back to zeros, rather
/// than to a NaN or an infinity.
#[test]
fn an_all_zero_group_survives_the_round_trip() {
    let g = 64;
    let src = vec![0f32; g];
    let row = quantize_int2_affine_ternary(&src, g);
    assert_eq!(f16_to_f32(row.scales[0]), 0.0);
    for (i, w) in dequantize_int2_affine(&row, g).iter().enumerate() {
        assert_eq!(*w, 0.0, "element {i}");
    }
}

/// The group size is a PARAMETER, not a constant: the same bytes decode
/// differently at 64 and at 128, and both are well-formed.
///
/// The 1-bit module's sibling assertion, kept because it is the axis
/// `quant.rs` gets to hardcode and these two do not.
#[test]
fn the_group_size_is_a_parameter_not_a_constant() {
    let at_128 = dequantize_int2_affine(&oracle_row(), 256);
    let mut narrow = oracle_row();
    narrow.group_size = 64;
    narrow.scales = vec![
        INT2_ORACLE_SCALES[0],
        INT2_ORACLE_SCALES[1],
        INT2_ORACLE_SCALES[0],
        INT2_ORACLE_SCALES[1],
    ];
    narrow.biases = vec![
        INT2_ORACLE_BIASES[0],
        INT2_ORACLE_BIASES[1],
        INT2_ORACLE_BIASES[0],
        INT2_ORACLE_BIASES[1],
    ];
    let at_64 = dequantize_int2_affine(&narrow, 256);
    assert_ne!(at_128, at_64);
}

/// The GEMV multiplies the row it decoded, in element order.
#[test]
fn the_gemv_matches_a_dequantize_and_dot() {
    let g = 128;
    let n = 2 * g;
    let rows: Vec<Int2AffineRow> = (0..3)
        .map(|r| {
            let src: Vec<f32> = (0..n)
                .map(|i| ((i + 7 * r) as f32 * 0.11).cos() * 0.25)
                .collect();
            quantize_int2_affine_ternary(&src, g)
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019).sin()).collect();

    let got = dequant_int2_gemv(&rows, &x, n);
    for (r, row) in rows.iter().enumerate() {
        let want: f32 = dequantize_int2_affine(row, n)
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(got[r], want, "row {r}");
    }
}

/// The embedding lookup reads the row it was asked for, and the row stride is
/// `D / 4` rather than `D`.
///
/// The real checkpoint quantizes `embed_tokens` at two bits like everything
/// else, so this is a kernel the type genuinely needs. The bug it is written
/// for is the stride: at two bits a row is FOUR times shorter than its
/// element count, so an off-by-a-factor read lands inside a neighbouring
/// token's weights, which is finite and plausible. The token is deliberately
/// not row 0.
#[test]
fn the_embedding_lookup_reads_the_right_row_and_scales_it() {
    let (vocab, d, g) = (7usize, 256usize, TERNARY_GROUP_SIZE);
    let rows: Vec<Int2AffineRow> = (0..vocab)
        .map(|t| {
            let src: Vec<f32> = (0..d)
                .map(|i| ((i + 13 * t) as f32 * 0.23).sin() * (1.0 + t as f32))
                .collect();
            quantize_int2_affine_ternary(&src, g)
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
    let got = embed_lookup_int2(&packed, &scales, &biases, token, d, g, out_scale);
    let want: Vec<f32> = dequantize_int2_affine(&rows[token], d)
        .iter()
        .map(|v| v * out_scale)
        .collect();
    assert_eq!(got, want);

    // Non-vacuous on both counts: a neighbouring row decodes differently, so
    // a stride bug would show, and the scale is large enough that dropping it
    // could not pass as rounding.
    assert_ne!(dequantize_int2_affine(&rows[token - 1], d), want);
    let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 1.0, "the test row is too small to prove the scale");
}

/// A row length that is not a whole number of groups is a caller error, not a
/// truncation.
#[test]
#[should_panic(expected = "not a multiple of the group size")]
fn a_partial_group_is_refused() {
    let row = Int2AffineRow {
        packed: vec![0u8; 48],
        scales: vec![0u16; 2],
        biases: vec![0u16; 2],
        group_size: TERNARY_GROUP_SIZE,
    };
    let _ = dequantize_int2_affine(&row, 192);
}

/// FP16 is not BF16, stated as an assertion because the two planes are the
/// same width and a misread is otherwise silent.
///
/// The oracle's first scale is `0x2300`. Read as FP16 that is 0.013671875,
/// the magnitude a QAT ternary weight has; read as BF16 it is ~7e-18.
#[test]
fn the_companions_are_fp16_and_reading_them_as_bf16_is_not_close() {
    let as_f16 = f16_to_f32(INT2_ORACLE_SCALES[0]);
    let as_bf16 = f32::from_bits((INT2_ORACLE_SCALES[0] as u32) << 16);
    assert!((as_f16 - 0.0137).abs() < 1e-3, "scale read {as_f16}");
    assert!(
        as_bf16 < 1e-10,
        "BF16 reading {as_bf16} is close enough to the FP16 one to hide the bug"
    );
}
