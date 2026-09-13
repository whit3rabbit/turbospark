//! IQ-quant reference tests (ROADMAP Phase S).
//!
//! The three `*_ORACLE` arrays below are the statement of each layout, and
//! they come from GGML, not from this port. `scripts/ggml_tables.c` builds the
//! same byte pattern [`oracle_bytes`] builds here and decodes it through
//! `ggml_get_type_traits(t)->to_float`. That matters more for these types than
//! it did for Q4_K: a codebook decoder and a fixture written from one mental
//! model agree while both are wrong, and there is no arithmetic identity to
//! fall back on -- an IQ3_XXS value is whatever the table says it is.
//!
//! The remaining tests each pin ONE of the traps named in
//! `quant_gguf_iq.rs`'s module doc, so a regression says which assumption
//! broke rather than just printing 256 mismatched floats.

use turbospark_compute::{
    dequant_iq1_m_gemv, dequant_iq1_s_gemv, dequant_iq2_s_gemv, dequant_iq2_xs_gemv,
    dequant_iq2_xxs_gemv, dequant_iq3_s_gemv, dequant_iq4_nl_gemv, dequantize_iq1_m,
    dequantize_iq1_s, dequantize_iq2_s, dequantize_iq2_xs, dequantize_iq2_xxs, dequantize_iq3_s,
    dequantize_iq3_xxs, dequantize_iq4_nl, dequantize_iq4_xs, iq3xxs_signs, IQ1_M_BLOCK_BYTES,
    IQ1_S_BLOCK_BYTES, IQ2_S_BLOCK_BYTES, IQ2_XS_BLOCK_BYTES, IQ2_XXS_BLOCK_BYTES, IQ3S_GRID,
    IQ3XXS_GRID, IQ3_S_BLOCK_BYTES, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS, IQ4NL_VALUES,
    IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS,
    IQ_LOWBIT_BLOCK_ELEMS,
};

/// FP16 1.0, the `d` every oracle block uses so the decoded value is the
/// table entry times the sub-block scale and nothing else.
const F16_ONE: u16 = 0x3C00;

/// The generator `scripts/ggml_tables.c` uses, so the Rust side reproduces
/// the same bytes without linking libggml. A plain LCG, chosen for exactly
/// that reason.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
}

/// Builds the byte pattern the C generator built, for one block of `kind`.
fn oracle_bytes(kind: &str) -> Vec<u8> {
    let mut lcg = Lcg(12345);
    let mut out = Vec::new();
    out.extend_from_slice(&F16_ONE.to_le_bytes());
    match kind {
        "iq4_nl" => {
            for _ in 0..16 {
                out.push((lcg.next() & 0xFF) as u8);
            }
        }
        "iq4_xs" => {
            out.extend_from_slice(&((lcg.next() & 0xFFFF) as u16).to_le_bytes());
            for _ in 0..4 {
                out.push((lcg.next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                out.push((lcg.next() & 0xFF) as u8);
            }
        }
        "iq3_xxs" => {
            for _ in 0..96 {
                out.push((lcg.next() & 0xFF) as u8);
            }
        }
        other => panic!("unknown kind {other}"),
    }
    out
}

#[test]
fn iq4_nl_decodes_to_what_ggml_decodes() {
    let got = dequantize_iq4_nl(&oracle_bytes("iq4_nl"), IQ4_NL_BLOCK_ELEMS);
    assert_eq!(got.len(), IQ4NL_ORACLE.len());
    // Exact: every value is an integer table entry times an exact FP16 1.0.
    assert_eq!(got.as_slice(), IQ4NL_ORACLE.as_slice());
}

#[test]
fn iq4_xs_decodes_to_what_ggml_decodes() {
    let got = dequantize_iq4_xs(&oracle_bytes("iq4_xs"), IQ4_XS_BLOCK_ELEMS);
    assert_eq!(got.as_slice(), IQ4XS_ORACLE.as_slice());
}

#[test]
fn iq3_xxs_decodes_to_what_ggml_decodes() {
    let got = dequantize_iq3_xxs(&oracle_bytes("iq3_xxs"), IQ3_XXS_BLOCK_ELEMS);
    // Not exact-comparable like the other two: the scale carries a factor of
    // 0.25, so the values are quarters and the C side printed four decimals.
    for (i, (g, e)) in got.iter().zip(IQ3XXS_ORACLE.iter()).enumerate() {
        assert!((g - e).abs() < 1e-3, "element {i}: got {g}, ggml says {e}");
    }
}

/// Trap 1: the two nibbles of a byte are 16 elements apart, not adjacent.
///
/// A block whose bytes all read `0xF0` decodes to sixteen copies of the
/// lowest table entry followed by sixteen of the highest. Read adjacent, it
/// would alternate.
#[test]
fn an_iq4_nl_byte_serves_two_elements_sixteen_apart() {
    let mut block = vec![0u8; IQ4_NL_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_ONE.to_le_bytes());
    for b in block[2..].iter_mut() {
        *b = 0xF0;
    }
    let got = dequantize_iq4_nl(&block, IQ4_NL_BLOCK_ELEMS);
    let low = f32::from(IQ4NL_VALUES[0]);
    let high = f32::from(IQ4NL_VALUES[15]);
    assert_ne!(low, high);
    assert!(got[..16].iter().all(|&v| v == low), "{:?}", &got[..16]);
    assert!(got[16..].iter().all(|&v| v == high), "{:?}", &got[16..]);
}

/// Trap 2: the levels are non-linear, so an affine read is not merely
/// rescaled. Pins the actual table rather than restating it: the check is
/// that consecutive steps are UNEQUAL, which `(q - 8) * d` cannot produce.
#[test]
fn the_iq4_levels_are_not_evenly_spaced() {
    let steps: Vec<i32> = IQ4NL_VALUES
        .windows(2)
        .map(|w| i32::from(w[1]) - i32::from(w[0]))
        .collect();
    assert!(
        steps.windows(2).any(|w| w[0] != w[1]),
        "an evenly-spaced table means the affine shortcut is correct, which it is not: {steps:?}"
    );
    // The tails are wider than the middle, which is the property the type
    // exists for.
    assert!(steps[0] > *steps.iter().min().unwrap());
    assert!(*steps.last().unwrap() > *steps.iter().min().unwrap());
}

/// Trap 3: IQ4_XS sub-block scales are biased by 32 and go negative.
///
/// Two sub-blocks with the same quants and scale indices 0 and 63 must decode
/// to opposite signs. Dropping the bias makes both positive.
#[test]
fn an_iq4_xs_sub_block_scale_can_be_negative() {
    let mut block = vec![0u8; IQ4_XS_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_ONE.to_le_bytes());
    // Sub-block 0: ls = 0 -> dl = -32. Sub-block 1: ls = 63 -> dl = +31.
    // Sub-block `ib` takes byte `ib / 2`, LOW nibble for even `ib`, so ib0 is
    // the low nibble of byte 0 and ib1 the high one.
    block[4] = 0xF0;
    let scales_h: u16 = 3 << 2; // ib1 high bits = 3, ib0 high bits = 0
    block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    for b in block[8..].iter_mut() {
        *b = 0xFF; // every quant index 15, the largest positive level
    }
    let got = dequantize_iq4_xs(&block, IQ4_XS_BLOCK_ELEMS);
    let level = f32::from(IQ4NL_VALUES[15]);
    assert_eq!(got[0], -32.0 * level);
    assert_eq!(got[32], 31.0 * level);
    assert!(got[0] < 0.0 && got[32] > 0.0);
}

/// Trap 4: the eighth IQ3_XXS sign is a parity bit, so every group of eight
/// carries an even number of negatives. Checked over the whole index space,
/// which is 128 values.
#[test]
fn iq3_xxs_sign_groups_always_have_even_parity() {
    for i in 0..128u32 {
        let signs = iq3xxs_signs(i);
        assert_eq!(
            signs.count_ones() % 2,
            0,
            "index {i} gives {signs:#010b}, odd parity"
        );
        assert_eq!(u32::from(signs & 127), i, "index {i} is not preserved");
    }
}

/// Trap 5: the IQ3_XXS scale is `(0.5 + nibble) * 0.5`, so nibble 0 is a real
/// scale rather than zero. A block with scale nibble 0 and a non-zero grid
/// entry must decode to something non-zero.
#[test]
fn an_iq3_xxs_scale_nibble_of_zero_is_not_a_zero_scale() {
    // Grid entry 0 is all fours, per the generated table; use it so the
    // expected value is arithmetic rather than a lookup.
    assert_eq!(IQ3XXS_GRID[0], [4, 4, 4, 4]);
    let mut block = vec![0u8; IQ3_XXS_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_ONE.to_le_bytes());
    // qs all zero -> grid entry 0 everywhere; aux all zero -> scale nibble 0,
    // sign index 0, which is all-positive.
    let got = dequantize_iq3_xxs(&block, IQ3_XXS_BLOCK_ELEMS);
    // db = 1.0 * (0.5 + 0) * 0.5 = 0.25, value = 0.25 * 4.
    assert!(
        got.iter().all(|&v| v == 1.0),
        "first eight: {:?}",
        &got[..8]
    );
}

/// The grid holds positive magnitudes drawn from an eight-value alphabet.
/// Pinned because it is the cheapest check that a regenerated table is the
/// real one: a table off by a scale factor (which is what a first run of the
/// generator produced) leaves the alphabet a different set of eight, and a
/// table off by a transposition leaves it the same set. This catches the
/// first and the `*_ORACLE` tests catch the second.
#[test]
fn the_iq3_xxs_grid_is_positive_and_drawn_from_ggmls_alphabet() {
    let mut alphabet: Vec<u8> = IQ3XXS_GRID.iter().flatten().copied().collect();
    alphabet.sort_unstable();
    alphabet.dedup();
    assert_eq!(alphabet, vec![4, 12, 20, 28, 36, 44, 52, 62]);
}

#[test]
fn a_short_run_is_refused_rather_than_read_past() {
    let block = vec![0u8; IQ4_NL_BLOCK_BYTES];
    assert!(std::panic::catch_unwind(|| dequantize_iq4_nl(&block, 16)).is_err());
    assert!(std::panic::catch_unwind(|| dequantize_iq4_nl(&block, 64)).is_err());
}

#[test]
fn the_iq4_nl_gemv_matches_a_dequantize_then_dot() {
    let row_a = oracle_bytes("iq4_nl");
    let mut row_b = oracle_bytes("iq4_nl");
    row_b[2] ^= 0xFF;
    let x: Vec<f32> = (0..IQ4_NL_BLOCK_ELEMS)
        .map(|i| (i as f32 * 0.125) - 2.0)
        .collect();
    let got = dequant_iq4_nl_gemv(&[&row_a, &row_b], &x, IQ4_NL_BLOCK_ELEMS);
    for (r, row) in [row_a.as_slice(), row_b.as_slice()].iter().enumerate() {
        let want: f32 = dequantize_iq4_nl(row, IQ4_NL_BLOCK_ELEMS)
            .iter()
            .zip(x.iter())
            .map(|(w, v)| w * v)
            .sum();
        assert_eq!(got[r], want);
    }
}

fn dense_iq_bytes(bytes: usize, iq1_m: bool) -> Vec<u8> {
    let mut lcg = Lcg(0x51_a7_92_31);
    let mut block = (0..bytes)
        .map(|_| (lcg.next() >> 24) as u8)
        .collect::<Vec<_>>();
    if iq1_m {
        // IQ1_M reconstructs d from four 16-bit scale words. This spells
        // finite f16 0x2c00, leaving all index, scale and delta bits live.
        block[48..56].copy_from_slice(&[0, 0, 0, 0, 0, 0xc0, 0, 0x20]);
    } else {
        block[..2].copy_from_slice(&F16_ONE.to_le_bytes());
    }
    block
}

#[test]
fn dense_gsq_rco_iq_references_refuse_short_blocks_and_match_their_gemvs() {
    type Decode = fn(&[u8], usize) -> Vec<f32>;
    type Gemv = fn(&[&[u8]], &[f32], usize) -> Vec<f32>;
    let cases: [(&str, usize, bool, Decode, Gemv); 6] = [
        (
            "IQ2_XXS",
            IQ2_XXS_BLOCK_BYTES,
            false,
            dequantize_iq2_xxs,
            dequant_iq2_xxs_gemv,
        ),
        (
            "IQ2_XS",
            IQ2_XS_BLOCK_BYTES,
            false,
            dequantize_iq2_xs,
            dequant_iq2_xs_gemv,
        ),
        (
            "IQ1_S",
            IQ1_S_BLOCK_BYTES,
            false,
            dequantize_iq1_s,
            dequant_iq1_s_gemv,
        ),
        (
            "IQ3_S",
            IQ3_S_BLOCK_BYTES,
            false,
            dequantize_iq3_s,
            dequant_iq3_s_gemv,
        ),
        (
            "IQ2_S",
            IQ2_S_BLOCK_BYTES,
            false,
            dequantize_iq2_s,
            dequant_iq2_s_gemv,
        ),
        (
            "IQ1_M",
            IQ1_M_BLOCK_BYTES,
            true,
            dequantize_iq1_m,
            dequant_iq1_m_gemv,
        ),
    ];
    let x: Vec<f32> = (0..IQ_LOWBIT_BLOCK_ELEMS)
        .map(|i| i as f32 / 127.0 - 1.0)
        .collect();
    for (name, bytes, iq1_m, decode, gemv) in cases {
        let a = dense_iq_bytes(bytes, iq1_m);
        let mut b = dense_iq_bytes(bytes, iq1_m);
        // Keep IQ1_M's reconstructed f16 scale finite; its final eight bytes
        // are not spare padding, unlike the other layouts' final field.
        b[0] ^= 0x5a;
        assert!(
            std::panic::catch_unwind(|| decode(&a[..bytes - 1], IQ_LOWBIT_BLOCK_ELEMS)).is_err(),
            "{name}"
        );
        let got = gemv(&[&a, &b], &x, IQ_LOWBIT_BLOCK_ELEMS);
        for (row, dot) in [&a, &b].into_iter().zip(got) {
            let want: f32 = decode(row, IQ_LOWBIT_BLOCK_ELEMS)
                .iter()
                .zip(&x)
                .map(|(w, x)| w * x)
                .sum();
            assert_eq!(dot, want, "{name}");
        }
    }
    assert!(
        IQ3S_GRID.iter().any(|&v| v != 0),
        "generated IQ3_S grid is empty"
    );
}

// Under `generated/` rather than beside this file because every top-level
// `.rs` in `tests/` is compiled as its own test binary, and a bare table of
// constants is not one.
include!("generated/quant_gguf_iq_oracles.rs");
