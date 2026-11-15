//! Q1_0, TQ1_0, TQ2_0 and NVFP4 decoded against golden outputs produced by
//! ggml's own `to_float` (`scripts/ggml_dequant_fixture.c`) on three random
//! blocks each. The inputs are checked in beside the outputs so the test needs
//! no ggml; regenerate with the fixture tool if a layout is ever re-derived.

use turbospark_compute::{
    dequantize_iq1_xs, dequantize_iq1_xxs, dequantize_iq1_xxxs, dequantize_nvfp4, dequantize_q1_0,
    dequantize_tq1_0, dequantize_tq2_0, nvfp4_scale, IQ1_NARROW_BLOCK_ELEMS, IQ1_XS_BLOCK_BYTES,
    IQ1_XXS_BLOCK_BYTES, IQ1_XXXS_BLOCK_BYTES, NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMS,
    Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMS, TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES,
    TQ2_0_BLOCK_ELEMS,
};

fn load(name: &str) -> (Vec<u8>, Vec<f32>) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ggml");
    let input = std::fs::read(format!("{dir}/{name}.in.bin")).unwrap();
    let out = std::fs::read(format!("{dir}/{name}.out.f32")).unwrap();
    let want = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (input, want)
}

fn check(name: &str, elems: usize, bytes: usize, decode: fn(&[u8], usize) -> Vec<f32>) {
    check_blocks(name, 3, elems, bytes, decode);
}

fn check_blocks(
    name: &str,
    blocks: usize,
    elems: usize,
    bytes: usize,
    decode: fn(&[u8], usize) -> Vec<f32>,
) {
    let (input, want) = load(name);
    assert_eq!(input.len(), blocks * bytes, "{name}: fixture size");
    assert_eq!(want.len(), blocks * elems, "{name}: fixture output size");
    let got = decode(&input, blocks * elems);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{name} element {i}: {g} vs ggml {w}"
        );
    }
}

#[test]
fn q1_0_matches_ggml() {
    check("q1_0", Q1_0_BLOCK_ELEMS, Q1_0_BLOCK_BYTES, dequantize_q1_0);
}

#[test]
fn tq2_0_matches_ggml() {
    check(
        "tq2_0",
        TQ2_0_BLOCK_ELEMS,
        TQ2_0_BLOCK_BYTES,
        dequantize_tq2_0,
    );
}

#[test]
fn tq1_0_matches_ggml() {
    check(
        "tq1_0",
        TQ1_0_BLOCK_ELEMS,
        TQ1_0_BLOCK_BYTES,
        dequantize_tq1_0,
    );
}

#[test]
fn nvfp4_matches_ggml() {
    check(
        "nvfp4",
        NVFP4_BLOCK_ELEMS,
        NVFP4_BLOCK_BYTES,
        dequantize_nvfp4,
    );
}

/// The fixture draws scale bytes from a handful of values; this pins the rest
/// of the byte range to the values measured from ggml (code 2 is 1.0, so the
/// decoded element equals the scale). Only the special cases are asserted
/// numerically: the NaN code reads as zero and bit 7 is ignored.
#[test]
fn nvfp4_scale_special_cases_match_ggml() {
    assert_eq!(nvfp4_scale(0x7F), 0.0);
    assert_eq!(nvfp4_scale(0xFF), 0.0);
    assert_eq!(nvfp4_scale(0x00), 0.0);
    assert_eq!(nvfp4_scale(0x38), 1.0);
    assert_eq!(nvfp4_scale(0xB8), 1.0);
    assert_eq!(nvfp4_scale(0x7E), 448.0);
    assert_eq!(nvfp4_scale(0x01), 2f32.powi(-9));
}

/// Unsloth's IQ1_XS, IQ1_XXS and IQ1_XXXS (ggml ids 64 to 66), against the
/// fork's own `to_float` (branch `iq1-narrow-grids-b10775`, built from source
/// with `scripts/ggml_dequant_fixture.c`). Six random blocks each.
#[test]
fn iq1_narrow_types_match_the_forks_to_float() {
    check_blocks(
        "iq1_xs",
        6,
        IQ1_NARROW_BLOCK_ELEMS,
        IQ1_XS_BLOCK_BYTES,
        dequantize_iq1_xs,
    );
    check_blocks(
        "iq1_xxs",
        6,
        IQ1_NARROW_BLOCK_ELEMS,
        IQ1_XXS_BLOCK_BYTES,
        dequantize_iq1_xxs,
    );
    check_blocks(
        "iq1_xxxs",
        6,
        IQ1_NARROW_BLOCK_ELEMS,
        IQ1_XXXS_BLOCK_BYTES,
        dequantize_iq1_xxxs,
    );
}
