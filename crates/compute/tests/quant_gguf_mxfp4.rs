//! MXFP4 reference tests (ROADMAP M5, the `gpt-oss` family).
//!
//! Its own test binary rather than a section of `quant_gguf.rs`, matching
//! how `quant_gguf_iq.rs` is separated: each top-level file here is one
//! binary, and this one carries a generated fixture the others do not need.
//!
//! The ORACLE is the test that matters. Everything else is a property check
//! on top of it, and each of those is written as a MUTATION the decoder must
//! not survive -- the four traps `quant_gguf_mxfp4`'s module header names,
//! stated as assertions instead of prose.

use turbospark_compute::{
    dequant_mxfp4_gemv, dequantize_mxfp4, mxfp4_scale, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
    MXFP4_VALUES,
};

// ggml quantized these bytes, decoded these floats, and both tables were
// recovered from it by construction (`scripts/ggml_mxfp4_oracle.c`). Under
// `generated/` for the same reason the other oracles are.
include!("generated/quant_gguf_mxfp4_oracle.rs");

/// The one test that can catch this decoder and its author sharing a
/// misreading, because ggml produced BOTH sides of it.
///
/// Compared with `==` and not a tolerance. Nothing here rounds: the scale is
/// an exact power of two and the codebook holds small integers, so every
/// product is exact in FP32 and any difference at all is a real one.
#[test]
fn mxfp4_decodes_exactly_what_ggml_decodes() {
    let got = dequantize_mxfp4(&MXFP4_ORACLE_BYTES, MXFP4_ORACLE_FLOATS.len());
    assert_eq!(got.len(), MXFP4_ORACLE_FLOATS.len());
    for (i, (&g, &want)) in got.iter().zip(MXFP4_ORACLE_FLOATS.iter()).enumerate() {
        assert_eq!(g, want, "element {i} disagrees with ggml");
    }
}

/// The codebook in `src/` must be the one ggml decodes with.
///
/// This is the assertion that makes the literal table safe to carry: it was
/// recovered by construction from libggml rather than typed out of a spec,
/// and a transcription slip reddens here instead of scaling every weight in
/// the model by a constant nobody would notice.
///
/// `to_bits` rather than `==` because the two zeros at indices 0 and 8 are
/// exactly where a plausible reading and ggml's actual one differ, and
/// `+0.0 == -0.0` is true. This test is what caught the first version of
/// `MXFP4_VALUES` carrying `-0.0` at index 8, read off the FP4 E2M1 sign
/// bit; ggml's table is `int8`, so both zeros are positive. A float
/// comparison would have passed it.
#[test]
fn the_codebook_is_the_one_ggml_uses() {
    assert_eq!(MXFP4_VALUES.len(), MXFP4_ORACLE_CODEBOOK.len());
    for (i, (&got, &want)) in MXFP4_VALUES
        .iter()
        .zip(MXFP4_ORACLE_CODEBOOK.iter())
        .enumerate()
    {
        assert_eq!(got.to_bits(), want.to_bits(), "codebook index {i}");
    }
}

/// The shared-exponent expression must agree with ggml on all 256 bytes.
///
/// The generated table holds `codebook[1] * scale(e)`, so this compares the
/// same product rather than the scale alone: both halves of the
/// reconstruction stay under test, and a decoder that got the codebook and
/// the bias wrong in compensating directions cannot pass.
///
/// The two subnormal bytes (`e < 2`) are the reason this sweeps all 256
/// rather than spot-checking. They are ~1e-39 and could not matter
/// numerically; they are also the two a plain `2^(e - 128)` gets wrong, so
/// they are exactly where an expression written from the common case fails.
#[test]
fn every_shared_exponent_matches_ggml() {
    for e in 0..=u8::MAX {
        let got = MXFP4_VALUES[1] * mxfp4_scale(e);
        let want = MXFP4_ORACLE_E8M0[usize::from(e)];
        assert_eq!(got, want, "shared exponent byte {e}");
    }
}

/// The oracle must span more than one block AND more than one shared
/// exponent, or the test above cannot see a decoder that carries block 0's
/// scale forward. Asserted rather than assumed, since regenerating the
/// fixture with a narrower weight range would silently weaken it.
#[test]
fn the_oracle_covers_several_blocks_at_different_exponents() {
    assert_eq!(MXFP4_ORACLE_BYTES.len() % MXFP4_BLOCK_BYTES, 0);
    let blocks = MXFP4_ORACLE_BYTES.len() / MXFP4_BLOCK_BYTES;
    assert!(blocks >= 4, "only {blocks} blocks");
    assert_eq!(MXFP4_ORACLE_FLOATS.len(), blocks * MXFP4_BLOCK_ELEMS);

    let mut exponents: Vec<u8> = (0..blocks)
        .map(|b| MXFP4_ORACLE_BYTES[b * MXFP4_BLOCK_BYTES])
        .collect();
    exponents.sort_unstable();
    exponents.dedup();
    assert!(
        exponents.len() >= 3,
        "only {} distinct shared exponents; a decoder reusing block 0's would pass",
        exponents.len()
    );
}

/// MUTATION 1: the two nibbles of a byte are 16 elements apart, not
/// adjacent.
///
/// Written as a property rather than by re-implementing the wrong decoder,
/// because the wrong decoder produces a PERMUTATION of the right values and
/// a set comparison would call the two equal. What separates them is
/// POSITION: with the split-half layout, elements `j` and `j + 16` of a
/// block come from one byte, so zeroing every high nibble must leave the low
/// half of each block untouched and flatten the high half to zero. Under the
/// adjacent reading it would zero every odd element instead.
#[test]
fn the_nibble_halves_are_sixteen_apart() {
    let mut high_cleared = MXFP4_ORACLE_BYTES;
    for b in 0..high_cleared.len() / MXFP4_BLOCK_BYTES {
        for j in 0..MXFP4_BLOCK_ELEMS / 2 {
            high_cleared[b * MXFP4_BLOCK_BYTES + 1 + j] &= 0x0F;
        }
    }
    let got = dequantize_mxfp4(&high_cleared, MXFP4_ORACLE_FLOATS.len());

    for (i, (&g, &full)) in got.iter().zip(MXFP4_ORACLE_FLOATS.iter()).enumerate() {
        let within = i % MXFP4_BLOCK_ELEMS;
        if within < MXFP4_BLOCK_ELEMS / 2 {
            assert_eq!(g, full, "element {i} is a low nibble and must not move");
        } else {
            assert_eq!(g, 0.0, "element {i} is a high nibble and must be zeroed");
        }
    }
}

/// MUTATION 2: index 8 is a second ZERO, not a continuation of the ramp.
///
/// The affine habit reads the index as `q - 8` (or as a signed nibble),
/// which puts a large magnitude where the format puts zero. Setting every
/// element to index 8 must decode to all zeros whatever the shared exponent
/// is.
///
/// AND IT IS POSITIVE ZERO, which is the half of this that was WRONG on the
/// first pass and is why the codebook is checked by bits. Reading the table
/// as FP4 E2M1 -- index 8 being sign-bit-set, magnitude zero -- gives `-0.0`,
/// which is defensible from the encoding's name and is not what ggml does:
/// its `kvalues_mxfp4` is an `int8` array, so the sign never reaches a zero.
/// Nothing downstream can tell the difference arithmetically. That is exactly
/// why it needs an oracle rather than a review.
#[test]
fn index_eight_is_a_second_positive_zero_not_a_magnitude() {
    for exponent in [0u8, 1, 64, 128, 200, 255] {
        let mut block = [0u8; MXFP4_BLOCK_BYTES];
        block[0] = exponent;
        block[1..].fill(0x88);
        let got = dequantize_mxfp4(&block, MXFP4_BLOCK_ELEMS);
        for (i, &g) in got.iter().enumerate() {
            assert_eq!(
                g.to_bits(),
                0.0f32.to_bits(),
                "exponent {exponent} element {i}"
            );
        }
    }
    assert_eq!(MXFP4_VALUES[8].to_bits(), 0.0f32.to_bits());
    assert_eq!(MXFP4_VALUES[0].to_bits(), 0.0f32.to_bits());
}

/// MUTATION 3: the exponent bias is 128, not 127.
///
/// Getting it wrong doubles (or halves) every weight in the model, which is
/// finite, correctly ordered, and produces fluent wrong text rather than an
/// error -- the failure mode this repo has now hit on three separate axes.
/// Pinned against two absolute values that need no table: byte 128 is a unit
/// scale, and each step of the byte is exactly one octave.
#[test]
fn the_exponent_bias_is_one_twenty_eight() {
    assert_eq!(mxfp4_scale(128), 1.0);
    assert_eq!(mxfp4_scale(129), 2.0);
    assert_eq!(mxfp4_scale(127), 0.5);
    for e in 3..=254u8 {
        assert_eq!(
            mxfp4_scale(e + 1),
            mxfp4_scale(e) * 2.0,
            "byte {e} to {} is not one octave",
            e + 1
        );
    }
}

/// MUTATION 4: `e < 2` is not the same expression as the rest.
///
/// Both are ~1e-39 and could not change any output that matters. They are
/// checked because the ONLY reason to know they are special is to have read
/// ggml, and an expression written from the common case gets them wrong --
/// which is the tell that the rest was written from the source rather than
/// from a plausible guess. The identity is that the octave rule still holds
/// across the seam.
#[test]
fn the_two_subnormal_exponents_continue_the_octave_ladder() {
    assert_eq!(mxfp4_scale(1), mxfp4_scale(0) * 2.0);
    assert_eq!(mxfp4_scale(2), mxfp4_scale(1) * 2.0);
    assert!(mxfp4_scale(0) > 0.0);
}

/// The GEMV is the dot product of the decoded row with `x`, and its only job
/// is to not lose the row. Checked against a hand-summed decode rather than
/// against a second GEMV.
#[test]
fn the_gemv_is_the_dot_product_of_the_decoded_rows() {
    let n = MXFP4_BLOCK_ELEMS * 2;
    let row_a: Vec<u8> = MXFP4_ORACLE_BYTES[..2 * MXFP4_BLOCK_BYTES].to_vec();
    let row_b: Vec<u8> = MXFP4_ORACLE_BYTES[2 * MXFP4_BLOCK_BYTES..].to_vec();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 % 7.0) - 3.0).collect();

    let got = dequant_mxfp4_gemv(&[&row_a, &row_b], &x, n);
    assert_eq!(got.len(), 2);
    for (r, row) in [row_a.as_slice(), row_b.as_slice()].iter().enumerate() {
        let want: f32 = dequantize_mxfp4(row, n)
            .iter()
            .zip(x.iter())
            .map(|(w, v)| w * v)
            .sum();
        assert_eq!(got[r], want, "row {r}");
    }
}

/// A partial trailing block is honoured exactly, which a routed-expert row
/// whose width is not a multiple of 32 needs. gpt-oss's own widths are all
/// multiples of 32, so this is a contract check rather than a live case --
/// and it is here because the alternative, silently writing past `n`, is a
/// buffer overrun in the caller rather than a wrong number.
#[test]
fn a_partial_trailing_block_stops_at_n() {
    for n in [1usize, 15, 16, 17, 31, 32, 33, 47] {
        let got = dequantize_mxfp4(&MXFP4_ORACLE_BYTES, n);
        assert_eq!(got.len(), n, "n = {n}");
        for (i, &g) in got.iter().enumerate() {
            assert_eq!(g, MXFP4_ORACLE_FLOATS[i], "n = {n}, element {i}");
        }
    }
}
