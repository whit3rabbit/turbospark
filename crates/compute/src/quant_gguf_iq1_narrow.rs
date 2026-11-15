//! Unsloth's IQ1_XS, IQ1_XXS and IQ1_XXXS (ggml type ids 64 to 66).
//!
//! These exist only in Unsloth's llama.cpp fork (branch
//! `iq1-narrow-grids-b10775`), which keeps ggml's reserved range 42..63 free
//! and starts its own types at 64 so a stock build rejects the files instead
//! of misreading them. Each is IQ1_S's 8-dimensional ternary codebook behind a
//! NARROWER index: dequantization is `dl * (grid_j + delta)` exactly as in
//! IQ1_S, with `d` the f16 super-block scale, `dl = d * (2 * ls + 1)` from a
//! 3-bit sub-block code and the sign bit choosing `delta = +/- 0.125`. Only
//! the index reassembly and where the scale code lives differ:
//!
//! | type      | block bytes | index    | layout after the f16 `d`                          |
//! |-----------|-------------|----------|---------------------------------------------------|
//! | IQ1_XS    | 46          | 10 bits  | `qs[32]`, `qh[8]` (4 x 2-bit high), `sc[4]` nibbles |
//! | IQ1_XXS   | 42          | 9 bits   | `qs[32]`, `qh[8]` (bit `l` high, bits 4-6 scale, bit 7 sign) |
//! | IQ1_XXXS  | 38          | 8 bits   | `qs[32]`, `sc[4]` nibbles                         |
//!
//! A scale nibble is `bits 0-2` the code and `bit 3` the delta sign; sub-block
//! `ib` reads nibble `(sc[ib / 2] >> (4 * (ib & 1))) & 15`. The decoders are
//! held bit-exact to the fork's own `to_float` by
//! `tests/quant_gguf_iq1_narrow.rs` (built from the fork at the commit named in
//! the generated tables' header).

use foundation::LogitValue as F16;

pub use super::quant_gguf_iq1_narrow_tables::{IQ1_XS_GRID, IQ1_XXS_GRID, IQ1_XXXS_GRID};

/// Elements per block of all three types.
pub const IQ1_NARROW_BLOCK_ELEMS: usize = 256;
/// IQ1_XS block bytes: f16 `d`, 32 `qs`, 8 `qh`, 4 `sc`.
pub const IQ1_XS_BLOCK_BYTES: usize = 46;
/// IQ1_XXS block bytes: f16 `d`, 32 `qs`, 8 `qh`.
pub const IQ1_XXS_BLOCK_BYTES: usize = 42;
/// IQ1_XXXS block bytes: f16 `d`, 32 `qs`, 4 `sc`.
pub const IQ1_XXXS_BLOCK_BYTES: usize = 38;

const DELTA: f32 = 0.125;

fn grid_i8(word: u64, lane: usize) -> f32 {
    (((word >> (8 * lane)) & 0xff) as u8 as i8) as f32
}

fn check(n: usize, bytes: usize, blocks: &[u8], name: &str) -> usize {
    assert!(
        n % IQ1_NARROW_BLOCK_ELEMS == 0,
        "n ({n}) is not a whole number of 256-element {name} blocks"
    );
    let nb = n / IQ1_NARROW_BLOCK_ELEMS;
    assert!(
        blocks.len() >= nb * bytes,
        "need {} bytes for {n} elements, got {}",
        nb * bytes,
        blocks.len()
    );
    nb
}

/// `(index, dl, delta)` for the four 8-element groups of sub-block `ib`.
fn emit(out: &mut [f32], d: f32, code: u32, negative: bool, indices: [usize; 4], grid: &[u64]) {
    let dl = d * (2 * code + 1) as f32;
    let delta = if negative { -DELTA } else { DELTA };
    for (l, &index) in indices.iter().enumerate() {
        for j in 0..8 {
            out[l * 8 + j] = dl * (grid_i8(grid[index], j) + delta);
        }
    }
}

fn d_of(blocks: &[u8], base: usize) -> f32 {
    f32::from(F16::from_bits(u16::from_le_bytes([
        blocks[base],
        blocks[base + 1],
    ])))
}

/// Dequantize `n` elements of IQ1_XS.
pub fn dequantize_iq1_xs(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = check(n, IQ1_XS_BLOCK_BYTES, blocks, "IQ1_XS");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ1_XS_BLOCK_BYTES;
        let d = d_of(blocks, base);
        let (qs, qh, sc) = (
            &blocks[base + 2..base + 34],
            &blocks[base + 34..base + 42],
            &blocks[base + 42..base + 46],
        );
        for ib in 0..8 {
            let nib = u32::from(sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
            let mut idx = [0usize; 4];
            for (l, slot) in idx.iter_mut().enumerate() {
                *slot = usize::from(qs[4 * ib + l]) | ((usize::from(qh[ib] >> (2 * l)) & 3) << 8);
            }
            emit(
                &mut out[b * 256 + ib * 32..],
                d,
                nib & 7,
                nib & 8 != 0,
                idx,
                &IQ1_XS_GRID,
            );
        }
    }
    out
}

/// Dequantize `n` elements of IQ1_XXS.
pub fn dequantize_iq1_xxs(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = check(n, IQ1_XXS_BLOCK_BYTES, blocks, "IQ1_XXS");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ1_XXS_BLOCK_BYTES;
        let d = d_of(blocks, base);
        let (qs, qh) = (&blocks[base + 2..base + 34], &blocks[base + 34..base + 42]);
        for ib in 0..8 {
            let h = qh[ib];
            let mut idx = [0usize; 4];
            for (l, slot) in idx.iter_mut().enumerate() {
                *slot = usize::from(qs[4 * ib + l]) | ((usize::from(h >> l) & 1) << 8);
            }
            emit(
                &mut out[b * 256 + ib * 32..],
                d,
                u32::from((h >> 4) & 7),
                h & 0x80 != 0,
                idx,
                &IQ1_XXS_GRID,
            );
        }
    }
    out
}

/// Dequantize `n` elements of IQ1_XXXS.
pub fn dequantize_iq1_xxxs(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = check(n, IQ1_XXXS_BLOCK_BYTES, blocks, "IQ1_XXXS");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ1_XXXS_BLOCK_BYTES;
        let d = d_of(blocks, base);
        let (qs, sc) = (&blocks[base + 2..base + 34], &blocks[base + 34..base + 38]);
        for ib in 0..8 {
            let nib = u32::from(sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
            let mut idx = [0usize; 4];
            for (l, slot) in idx.iter_mut().enumerate() {
                *slot = usize::from(qs[4 * ib + l]);
            }
            emit(
                &mut out[b * 256 + ib * 32..],
                d,
                nib & 7,
                nib & 8 != 0,
                idx,
                &IQ1_XXXS_GRID,
            );
        }
    }
    out
}

macro_rules! gemv {
    ($name:ident, $dequant:ident) => {
        /// FP32 reference for `y = W * x`, one byte run per output row.
        pub fn $name(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
            assert!(!weight_rows.is_empty());
            assert_eq!(x.len(), n);
            weight_rows
                .iter()
                .map(|row| $dequant(row, n).iter().zip(x).map(|(w, xv)| w * xv).sum())
                .collect()
        }
    };
}
gemv!(dequant_iq1_xs_gemv, dequantize_iq1_xs);
gemv!(dequant_iq1_xxs_gemv, dequantize_iq1_xxs);
gemv!(dequant_iq1_xxxs_gemv, dequantize_iq1_xxxs);
