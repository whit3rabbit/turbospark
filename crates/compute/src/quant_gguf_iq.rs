//! GGUF IQ-quant reference: IQ4_NL, IQ4_XS and IQ3_XXS (ROADMAP Phase S).
//!
//! Sibling of [`crate::quant_gguf`], and a separate module for the same
//! reason that one is separate from [`crate::quant`]: the layout is not a
//! variant of what is already here. Every type in `quant_gguf` reconstructs a
//! weight ARITHMETICALLY from its stored bits (`q * d`, `d*sc*q - dmin*m`,
//! `(q - 32) * d * sc`). These three do not. They store an INDEX into a fixed
//! table that ships with the format, so a decoder either has the table or it
//! produces plausible garbage; there is no formula to fall back on. That is
//! the whole difference between a K-quant and an IQ-quant, and it is why
//! Phase S was scoped as "an IQ-codebook phase or nothing".
//!
//! The tables live in [`crate::quant_gguf_iq_tables`] and are GENERATED from
//! ggml, not transcribed. Read that module's header before touching them.
//!
//! There is no `quantize_*` here, unlike the three types in `quant_gguf`.
//! Those have one so tests can hand a decoder arbitrary weights; an IQ
//! encoder would mean a codebook nearest-neighbour search that nothing in
//! this port will ever call. Tests instead pick valid indices, signs and
//! scales DIRECTLY and decode them, which covers the code space more evenly
//! than an encoder's output would and needs no search. The lossless-repack
//! rule means real bytes arrive already quantized and are copied through
//! verbatim, so no encoder is missing from the product either.
//!
//! Three properties are shared by all three types and are each silently wrong
//! rather than a fault if carried over from a K-quant by habit:
//!
//! 1. The two nibbles of a byte are NOT adjacent elements. IQ4_NL splits them
//!    16 apart, IQ4_XS 16 apart within each 32-element sub-block. Reading
//!    them adjacent gives a correctly-scaled permutation of the right values.
//! 2. IQ4_NL and IQ4_XS reconstruct through the non-linear table, not through
//!    `q - 8`. An affine read produces values with the right sign and roughly
//!    the right magnitude, wrong everywhere in the tails.
//! 3. IQ4_XS's sub-block scale is BIASED by 32 and can be negative
//!    (`dl = d * (ls - 32)` with `ls` in `0..64`), the same trap Q6_K's
//!    signed sub-scales carry. Dropping the bias mirrors whole sub-blocks.

use foundation::LogitValue as F16;

use crate::quant_gguf_iq_lowbit_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID,
};
use crate::quant_gguf_iq_tables::{IQ3XXS_GRID, IQ4NL_VALUES};

/// Elements in one IQ4_NL block.
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;

/// Bytes in one IQ4_NL block: an f16 scale then 16 nibble-packed indices.
/// Matches `ggml_type_block(20)` in `turbospark_repack::gguf_header`; the two
/// are checked against each other in `crates/repack`'s tests rather than one
/// importing the other, since this crate must not depend on repack.
pub const IQ4_NL_BLOCK_BYTES: usize = 18;

/// Elements in one IQ4_XS superblock.
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;

/// Elements in one IQ4_XS sub-block. Eight tile a superblock.
pub const IQ4_XS_SUB_ELEMS: usize = 32;

/// Bytes in one IQ4_XS superblock: f16 `d`, a u16 of high scale bits, four
/// bytes of low scale nibbles, then 128 nibble-packed indices. Matches
/// `ggml_type_block(23)`.
pub const IQ4_XS_BLOCK_BYTES: usize = 136;

/// Elements in one IQ3_XXS superblock.
pub const IQ3_XXS_BLOCK_ELEMS: usize = 256;

/// Elements in one IQ3_XXS sub-block. Eight tile a superblock, and each has
/// its own 4-bit scale packed into the same word as its sign indices.
pub const IQ3_XXS_SUB_ELEMS: usize = 32;

/// Bytes in one IQ3_XXS superblock: f16 `d`, 64 grid indices, then eight u32
/// sign-and-scale words. Matches `ggml_type_block(18)`.
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;

/// Byte offset of the sign-and-scale words inside an IQ3_XXS superblock.
const IQ3_XXS_AUX_OFFSET: usize = 2 + IQ3_XXS_BLOCK_ELEMS / 4;

/// Elements in each newly-supported low-bit IQ superblock.
pub const IQ_LOWBIT_BLOCK_ELEMS: usize = 256;
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;
pub const IQ2_XS_BLOCK_BYTES: usize = 74;
pub const IQ2_S_BLOCK_BYTES: usize = 82;
pub const IQ3_S_BLOCK_BYTES: usize = 110;
pub const IQ1_S_BLOCK_BYTES: usize = 50;
pub const IQ1_M_BLOCK_BYTES: usize = 56;
const IQ1_DELTA: f32 = 0.125;

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    f32::from(F16::from_bits(u16::from_le_bytes([
        bytes[at],
        bytes[at + 1],
    ])))
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn grid_u8(word: u64, lane: usize) -> f32 {
    ((word >> (8 * lane)) & 0xff) as f32
}

fn grid_i8(word: u64, lane: usize) -> f32 {
    (((word >> (8 * lane)) & 0xff) as u8 as i8) as f32
}

fn sign_byte(index: u32) -> u8 {
    // ggml's ksigns_iq2xs: seven stored sign bits plus an even-parity eighth.
    let low = (index & 127) as u8;
    low | (((low.count_ones() as u8) & 1) << 7)
}

fn assert_run(n: usize, elems: usize, bytes: usize, blocks: &[u8], name: &str) -> usize {
    assert!(
        n % elems == 0,
        "n ({n}) is not a whole number of {elems}-element {name} blocks"
    );
    let n_blocks = n / elems;
    assert!(
        blocks.len() >= n_blocks * bytes,
        "need {} bytes for {n} elements, got {}",
        n_blocks * bytes,
        blocks.len()
    );
    n_blocks
}

/// The eight sign bits an IQ3_XXS 7-bit sign index stands for.
///
/// ggml ships this as a 128-entry table (`ksigns_iq2xs`), but it is an
/// expression: the eighth sign is a PARITY bit over the seven stored ones, so
/// every group of eight elements has an even number of negatives. Kept as
/// code rather than as a third generated table because a computed byte cannot
/// go stale. `scripts/ggml_tables.c` checks the identity against ggml for all
/// 128 indices rather than this file asserting it.
///
/// Dropping the parity bit is the interesting failure: seven of every eight
/// elements keep the right sign, so the output correlates well and is wrong.
pub fn iq3xxs_signs(index: u32) -> u8 {
    debug_assert!(index < 128);
    let i = (index & 127) as u8;
    i | ((i.count_ones() as u8 & 1) << 7)
}

/// Dequantize `n` elements of an IQ4_NL byte run to FP32.
///
/// The simplest of the three and the one to read first. `w = d * table[q]`,
/// where `q` is a 4-bit index. Every byte serves two elements SIXTEEN apart:
/// low nibbles fill `0..16`, high nibbles `16..32`.
pub fn dequantize_iq4_nl(blocks: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = assert_run(n, IQ4_NL_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES, blocks, "IQ4_NL");
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * IQ4_NL_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let qs = &blocks[base + 2..base + IQ4_NL_BLOCK_BYTES];
        let dst = &mut out[b * IQ4_NL_BLOCK_ELEMS..(b + 1) * IQ4_NL_BLOCK_ELEMS];
        for (j, &byte) in qs.iter().enumerate() {
            dst[j] = d * f32::from(IQ4NL_VALUES[(byte & 0xF) as usize]);
            dst[j + IQ4_NL_BLOCK_ELEMS / 2] = d * f32::from(IQ4NL_VALUES[(byte >> 4) as usize]);
        }
    }
    out
}

/// Dequantize `n` elements of an IQ4_XS byte run to FP32.
///
/// IQ4_XS is IQ4_NL's table under a two-level scale, and the scale is where
/// it goes wrong quietly. Each of the eight 32-element sub-blocks has a
/// 6-bit scale SPLIT ACROSS TWO FIELDS: the low four bits are a nibble of
/// `scales_l` (sub-block `ib` uses byte `ib / 2`, low nibble for even `ib`),
/// and the high two bits are bits `2 * ib` of the u16 `scales_h`. The result
/// is BIASED: `dl = d * (ls - 32)`, so half the range is negative and a
/// decoder that skips the bias mirrors whole sub-blocks rather than failing.
pub fn dequantize_iq4_xs(blocks: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = assert_run(n, IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, blocks, "IQ4_XS");
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * IQ4_XS_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let scales_h = u16::from_le_bytes([blocks[base + 2], blocks[base + 3]]);
        let scales_l = &blocks[base + 4..base + 8];
        let qs = &blocks[base + 8..base + IQ4_XS_BLOCK_BYTES];
        for ib in 0..IQ4_XS_BLOCK_ELEMS / IQ4_XS_SUB_ELEMS {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = i32::from(low | (high << 4));
            let dl = d * (ls - 32) as f32;
            let src = &qs[ib * (IQ4_XS_SUB_ELEMS / 2)..(ib + 1) * (IQ4_XS_SUB_ELEMS / 2)];
            let at = b * IQ4_XS_BLOCK_ELEMS + ib * IQ4_XS_SUB_ELEMS;
            for (j, &byte) in src.iter().enumerate() {
                out[at + j] = dl * f32::from(IQ4NL_VALUES[(byte & 0xF) as usize]);
                out[at + j + IQ4_XS_SUB_ELEMS / 2] =
                    dl * f32::from(IQ4NL_VALUES[(byte >> 4) as usize]);
            }
        }
    }
    out
}

/// Dequantize `n` elements of an IQ3_XXS byte run to FP32.
///
/// The codebook type, and the one with no relative in this port. A 256-element
/// superblock holds an f16 `d`, 64 grid index bytes, and eight u32 words. Each
/// word serves one 32-element sub-block and packs FIVE fields into 32 bits:
/// four 7-bit sign indices in bits `0..28`, and the sub-block's 4-bit scale in
/// bits `28..32`.
///
/// Four things about it are silently wrong rather than a fault:
///
/// 1. The scale is `db = d * (0.5 + (aux >> 28)) * 0.5`, not `d * (aux >> 28)`.
///    The `0.5 +` means scale nibble 0 is a real, non-zero scale; dropping it
///    zeroes one sub-block in sixteen and merely dims the rest.
/// 2. Each grid index expands to FOUR elements, and a sub-block consumes eight
///    indices for its 32 elements. The pairing is interleaved: index `2l`
///    fills elements `8l..8l+4` and index `2l + 1` fills `8l+4..8l+8`.
/// 3. Signs come from [`iq3xxs_signs`], eight per 7-bit index, and the eighth
///    is parity. See that function.
/// 4. The grid holds MAGNITUDES only; every value in it is positive. A decoder
///    that treats a grid byte as signed reads a quarter of the table as large
///    negatives and still produces finite output.
pub fn dequantize_iq3_xxs(blocks: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = assert_run(
        n,
        IQ3_XXS_BLOCK_ELEMS,
        IQ3_XXS_BLOCK_BYTES,
        blocks,
        "IQ3_XXS",
    );
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * IQ3_XXS_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let qs = &blocks[base + 2..base + IQ3_XXS_AUX_OFFSET];
        let aux = &blocks[base + IQ3_XXS_AUX_OFFSET..base + IQ3_XXS_BLOCK_BYTES];
        for ib in 0..IQ3_XXS_BLOCK_ELEMS / IQ3_XXS_SUB_ELEMS {
            let word = u32::from_le_bytes([
                aux[4 * ib],
                aux[4 * ib + 1],
                aux[4 * ib + 2],
                aux[4 * ib + 3],
            ]);
            let db = d * (0.5 + (word >> 28) as f32) * 0.5;
            let at = b * IQ3_XXS_BLOCK_ELEMS + ib * IQ3_XXS_SUB_ELEMS;
            for l in 0..4 {
                let signs = iq3xxs_signs((word >> (7 * l)) & 127);
                let g1 = &IQ3XXS_GRID[qs[ib * 8 + 2 * l] as usize];
                let g2 = &IQ3XXS_GRID[qs[ib * 8 + 2 * l + 1] as usize];
                for j in 0..4 {
                    let s1 = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & (1 << (j + 4)) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[at + 8 * l + j] = db * f32::from(g1[j]) * s1;
                    out[at + 8 * l + 4 + j] = db * f32::from(g2[j]) * s2;
                }
            }
        }
    }
    out
}

/// Dequantize IQ2_XXS. Each 32-element sub-block has four 8-lane codebook
/// entries and four parity-expanded sign bytes in its second little-endian word.
pub fn dequantize_iq2_xxs(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(
        n,
        IQ_LOWBIT_BLOCK_ELEMS,
        IQ2_XXS_BLOCK_BYTES,
        blocks,
        "IQ2_XXS",
    );
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ2_XXS_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        for ib in 0..8 {
            let at = base + 2 + ib * 8;
            let a = u32::from_le_bytes(blocks[at..at + 4].try_into().unwrap());
            let aux = u32::from_le_bytes(blocks[at + 4..at + 8].try_into().unwrap());
            let scale = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[((a >> (8 * l)) & 0xff) as usize];
                let signs = sign_byte((aux >> (7 * l)) & 127);
                for j in 0..8 {
                    let sign = if signs & (1 << j) == 0 { 1.0 } else { -1.0 };
                    out[b * 256 + ib * 32 + l * 8 + j] = scale * grid_u8(grid, j) * sign;
                }
            }
        }
    }
    out
}

/// Dequantize IQ2_XS, with two 4-bit scales per 32-element sub-block.
pub fn dequantize_iq2_xs(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(
        n,
        IQ_LOWBIT_BLOCK_ELEMS,
        IQ2_XS_BLOCK_BYTES,
        blocks,
        "IQ2_XS",
    );
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ2_XS_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let qs = &blocks[base + 2..base + 66];
        let scales = &blocks[base + 66..base + 74];
        for ib in 0..8 {
            let s = scales[ib];
            let db = [
                d * (0.5 + (s & 15) as f32) * 0.25,
                d * (0.5 + (s >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = le_u16(qs, 2 * (4 * ib + l));
                let grid = IQ2XS_GRID[(q & 511) as usize];
                let signs = sign_byte((q >> 9) as u32);
                for j in 0..8 {
                    let sign = if signs & (1 << j) == 0 { 1.0 } else { -1.0 };
                    out[b * 256 + ib * 32 + l * 8 + j] = db[l / 2] * grid_u8(grid, j) * sign;
                }
            }
        }
    }
    out
}

/// Dequantize IQ2_S. Its high codebook bits are interleaved in `qh`; signs
/// follow the first 32 bytes of each block's `qs` region.
pub fn dequantize_iq2_s(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(n, IQ_LOWBIT_BLOCK_ELEMS, IQ2_S_BLOCK_BYTES, blocks, "IQ2_S");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ2_S_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let qs = &blocks[base + 2..base + 66];
        let qh = &blocks[base + 66..base + 74];
        let scales = &blocks[base + 74..base + 82];
        for ib in 0..8 {
            let s = scales[ib];
            let db = [
                d * (0.5 + (s & 15) as f32) * 0.25,
                d * (0.5 + (s >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = qs[4 * ib + l] as usize | (((qh[ib] as usize) << (8 - 2 * l)) & 0x300);
                let signs = qs[32 + 4 * ib + l];
                for j in 0..8 {
                    let sign = if signs & (1 << j) == 0 { 1.0 } else { -1.0 };
                    out[b * 256 + ib * 32 + l * 8 + j] =
                        db[l / 2] * grid_u8(IQ2S_GRID[q], j) * sign;
                }
            }
        }
    }
    out
}

/// Dequantize IQ3_S. Two 32-element blocks share a packed scale byte.
pub fn dequantize_iq3_s(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(n, IQ_LOWBIT_BLOCK_ELEMS, IQ3_S_BLOCK_BYTES, blocks, "IQ3_S");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ3_S_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let mut qs = base + 2;
        let mut qh = base + 66;
        let mut signs = base + 74;
        let scales = &blocks[base + 106..base + 110];
        for (pair, &packed_scale) in scales.iter().enumerate() {
            for half in 0..2 {
                let db = d
                    * (1 + 2 * if half == 0 {
                        (packed_scale & 15) as i32
                    } else {
                        (packed_scale >> 4) as i32
                    }) as f32;
                for l in 0..4 {
                    let q1 = blocks[qs + 2 * l] as usize
                        | (((blocks[qh + half] as usize) << (8 - 2 * l)) & 256);
                    let q2 = blocks[qs + 2 * l + 1] as usize
                        | (((blocks[qh + half] as usize) << (7 - 2 * l)) & 256);
                    let s = blocks[signs + l];
                    for j in 0..4 {
                        let sign1 = if s & (1 << j) == 0 { 1.0 } else { -1.0 };
                        let sign2 = if s & (1 << (j + 4)) == 0 { 1.0 } else { -1.0 };
                        let out_at = b * 256 + pair * 64 + half * 32 + l * 8;
                        out[out_at + j] = db * grid_u8(IQ3S_GRID[q1] as u64, j) * sign1;
                        out[out_at + 4 + j] = db * grid_u8(IQ3S_GRID[q2] as u64, j) * sign2;
                    }
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
    out
}

/// Dequantize IQ1_S, including its signed delta correction.
pub fn dequantize_iq1_s(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(n, IQ_LOWBIT_BLOCK_ELEMS, IQ1_S_BLOCK_BYTES, blocks, "IQ1_S");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ1_S_BLOCK_BYTES;
        let d = f16_at(blocks, base);
        let qs = &blocks[base + 2..base + 34];
        let qh = &blocks[base + 34..base + 50];
        for ib in 0..8 {
            let h = le_u16(qh, 2 * ib);
            let scale = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
            let delta = if h & 0x8000 == 0 {
                IQ1_DELTA
            } else {
                -IQ1_DELTA
            };
            for l in 0..4 {
                let index = qs[4 * ib + l] as usize | (((h >> (3 * l)) as usize & 7) << 8);
                for j in 0..8 {
                    out[b * 256 + ib * 32 + l * 8 + j] =
                        scale * (grid_i8(IQ1S_GRID[index], j) + delta);
                }
            }
        }
    }
    out
}

/// Dequantize IQ1_M. Its packed half-scale and per-grid delta bits differ
/// from IQ1_S, but use the same 2048-entry signed grid.
pub fn dequantize_iq1_m(blocks: &[u8], n: usize) -> Vec<f32> {
    let nb = assert_run(n, IQ_LOWBIT_BLOCK_ELEMS, IQ1_M_BLOCK_BYTES, blocks, "IQ1_M");
    let mut out = vec![0.0; n];
    for b in 0..nb {
        let base = b * IQ1_M_BLOCK_BYTES;
        let qs = &blocks[base..base + 32];
        let qh = &blocks[base + 32..base + 48];
        let scales = &blocks[base + 48..base + 56];
        let sc: [u16; 4] = core::array::from_fn(|i| le_u16(scales, 2 * i));
        let d_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f32::from(F16::from_bits(d_bits));
        for ib in 0..8 {
            let h0 = qh[2 * ib];
            let h1 = qh[2 * ib + 1];
            let word = sc[ib / 2];
            let scales = [
                d * (2.0 * ((word >> (6 * (ib % 2))) & 7) as f32 + 1.0),
                d * (2.0 * ((word >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0),
            ];
            let indices = [
                qs[4 * ib] as usize | (((h0 as usize) << 8) & 0x700),
                qs[4 * ib + 1] as usize | (((h0 as usize) << 4) & 0x700),
                qs[4 * ib + 2] as usize | (((h1 as usize) << 8) & 0x700),
                qs[4 * ib + 3] as usize | (((h1 as usize) << 4) & 0x700),
            ];
            let deltas = [
                if h0 & 0x08 == 0 {
                    IQ1_DELTA
                } else {
                    -IQ1_DELTA
                },
                if h0 & 0x80 == 0 {
                    IQ1_DELTA
                } else {
                    -IQ1_DELTA
                },
                if h1 & 0x08 == 0 {
                    IQ1_DELTA
                } else {
                    -IQ1_DELTA
                },
                if h1 & 0x80 == 0 {
                    IQ1_DELTA
                } else {
                    -IQ1_DELTA
                },
            ];
            for l in 0..4 {
                for j in 0..8 {
                    out[b * 256 + ib * 32 + l * 8 + j] =
                        scales[l / 2] * (grid_i8(IQ1S_GRID[indices[l]], j) + deltas[l]);
                }
            }
        }
    }
    out
}

/// FP32 reference for the IQ4_NL GEMV `y = W * x`, one byte run per row.
pub fn dequant_iq4_nl_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    gemv(weight_rows, x, n, dequantize_iq4_nl)
}

/// FP32 reference for the IQ4_XS GEMV `y = W * x`, one byte run per row.
pub fn dequant_iq4_xs_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    gemv(weight_rows, x, n, dequantize_iq4_xs)
}

/// FP32 reference for the IQ3_XXS GEMV `y = W * x`, one byte run per row.
pub fn dequant_iq3_xxs_gemv(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
    gemv(weight_rows, x, n, dequantize_iq3_xxs)
}

macro_rules! iq_gemv {
    ($name:ident, $dequant:ident) => {
        pub fn $name(weight_rows: &[&[u8]], x: &[f32], n: usize) -> Vec<f32> {
            gemv(weight_rows, x, n, $dequant)
        }
    };
}
iq_gemv!(dequant_iq2_xxs_gemv, dequantize_iq2_xxs);
iq_gemv!(dequant_iq2_xs_gemv, dequantize_iq2_xs);
iq_gemv!(dequant_iq2_s_gemv, dequantize_iq2_s);
iq_gemv!(dequant_iq3_s_gemv, dequantize_iq3_s);
iq_gemv!(dequant_iq1_s_gemv, dequantize_iq1_s);
iq_gemv!(dequant_iq1_m_gemv, dequantize_iq1_m);

fn gemv(
    weight_rows: &[&[u8]],
    x: &[f32],
    n: usize,
    dequant: fn(&[u8], usize) -> Vec<f32>,
) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    weight_rows
        .iter()
        .map(|row| {
            dequant(row, n)
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| w * xv)
                .sum()
        })
        .collect()
}
