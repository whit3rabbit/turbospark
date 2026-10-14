//! MLX `mode: "mxfp4"` tensors, re-encoded for this port's two consumers.
//!
//! MLX stores an mxfp4 matrix as two planes: `.weight`, u32 words holding
//! eight E2M1 codes each (element `i` of a row at bits `4 * (i % 8)` of word
//! `i / 8`, so byte `k` of a 16-byte group holds elements `2k` low and
//! `2k + 1` high), and `.scales`, one u8 E8M0 exponent per 32 elements,
//! read as `2^(e - 127)` against the OCP grid `{0, .5, 1, 1.5, 2, 3, 4, 6}`.
//! There is no `.biases` plane. The in-repo statement of that layout is
//! Music 3's CPU decoder (`crates/audio/src/music/minimax_music3/weights.rs`).
//!
//! Neither consumer reads that layout, and both conversions are EXACT:
//!
//! - [`reblock_to_gguf_mxfp4`] for routed experts. The MXFP4 MoE kernels
//!   (`moe_gguf.metal`) read GGUF's 17-byte block: the exponent byte, then 16
//!   bytes where byte `j` holds element `j` low and `j + 16` high, against
//!   ggml's doubled grid with `2^(e - 128)`. Since `E2M1[c] * 2^(e - 127) ==
//!   2 * E2M1[c] * 2^(e - 128)`, the 4-bit codes and the exponent byte carry
//!   over unchanged and only the nibble ORDER moves. Getting that order wrong
//!   is a permutation of correctly scaled values, which correlates well with
//!   the right answer and is wrong (`quant_gguf_mxfp4`'s header, trap 1).
//! - [`dequant_to_bf16`] for resident matrices, because there is no resident
//!   MXFP4 GEMV. E2M1 has at most two significant bits, so every value
//!   `E2M1 * 2^k` sits on BF16's grid and the BF16 word is the top half of
//!   the f32 one with nothing discarded. The function asserts that rather
//!   than assuming it.
//!
//! **Exponent byte 255 is refused by both.** MLX decodes it as NaN and ggml
//! as a finite `2^127`, so passing it through would turn a poisoned weight
//! into a large one silently. **So is any element that overflows f32**
//! (`|E2M1| >= 4` at exponent 253, `>= 2` at 254), which every decoder,
//! MLX's included, turns into infinity. That refusal is what keeps the BF16
//! exactness claim above true at the top of the exponent range.

use super::config::Gemma4Error;

/// Elements sharing one E8M0 exponent, in both layouts.
pub const MXFP4_GROUP: usize = 32;

/// GGUF MXFP4 bytes per 32-element block: exponent plus 16 nibble bytes.
/// Restated rather than imported so a drift between this and
/// `compute::MXFP4_BLOCK_BYTES` reddens `tests/mlx_mxfp4.rs`.
pub const GGUF_MXFP4_BLOCK_BYTES: usize = 17;

/// Bytes one row of `cols` elements occupies after [`reblock_to_gguf_mxfp4`].
#[must_use]
pub fn gguf_mxfp4_row_bytes(cols: usize) -> usize {
    cols / MXFP4_GROUP * GGUF_MXFP4_BLOCK_BYTES
}

/// The E8M0 exponent byte MLX decodes as NaN.
const E8M0_NAN: u8 = 0xFF;

fn check_planes(
    tensor: &str,
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), Gemma4Error> {
    let fail = |detail: String| {
        Err(Gemma4Error::ShapeMismatch {
            tensor: tensor.to_string(),
            detail,
        })
    };
    if cols == 0 || cols % MXFP4_GROUP != 0 {
        return fail(format!(
            "mxfp4 row width {cols} is not a positive multiple of {MXFP4_GROUP}"
        ));
    }
    if weight.len() != rows * cols / 2 {
        return fail(format!(
            "mxfp4 weight plane is {} bytes, expected {rows} x {cols} / 2",
            weight.len()
        ));
    }
    if scales.len() != rows * cols / MXFP4_GROUP {
        return fail(format!(
            "mxfp4 scale plane is {} bytes, expected {rows} x {cols} / {MXFP4_GROUP}",
            scales.len()
        ));
    }
    if let Some(at) = scales.iter().position(|&e| e == E8M0_NAN) {
        return fail(format!(
            "mxfp4 exponent byte {at} is 0xFF, which MLX reads as NaN and ggml as 2^127; \
             refusing rather than turning a poisoned weight into a finite one"
        ));
    }
    // Below 0xFF only the top two exponents can overflow f32: |E2M1| >= 4 at
    // 253 and >= 2 at 254 reach 2^128. MLX and both consumers here would all
    // decode those to infinity, so refuse them the same way.
    for (g, &exponent) in scales.iter().enumerate().filter(|(_, &e)| e >= 253) {
        let scale = 2.0f32.powi(i32::from(exponent) - 127);
        for (i, &byte) in weight[g * 16..g * 16 + 16].iter().enumerate() {
            for (half, code) in [byte & 0x0F, byte >> 4].into_iter().enumerate() {
                if (e2m1(code) * scale).is_infinite() {
                    return fail(format!(
                        "mxfp4 group {g} element {} (code {code:#x}, exponent {exponent}) \
                         overflows f32 to infinity; refusing rather than passing an \
                         infinite weight",
                        i * 2 + half
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Re-blocks an MLX mxfp4 matrix (`rows x cols`) into GGUF MXFP4 blocks, row
/// major, `gguf_mxfp4_row_bytes(cols)` bytes per row.
pub fn reblock_to_gguf_mxfp4(
    tensor: &str,
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, Gemma4Error> {
    check_planes(tensor, weight, scales, rows, cols)?;
    let groups = cols / MXFP4_GROUP;
    let mut out = Vec::with_capacity(rows * groups * GGUF_MXFP4_BLOCK_BYTES);
    for (g, &exponent) in scales.iter().enumerate() {
        // One MLX group is 16 consecutive bytes of the row-major weight plane.
        let src = &weight[g * 16..g * 16 + 16];
        let code = |i: usize| (src[i / 2] >> ((i % 2) * 4)) & 0x0F;
        out.push(exponent);
        for j in 0..16 {
            out.push(code(j) | (code(j + 16) << 4));
        }
    }
    debug_assert_eq!(out.len(), rows * groups * GGUF_MXFP4_BLOCK_BYTES);
    Ok(out)
}

/// The OCP E2M1 value of a 4-bit code, sign in bit 3.
fn e2m1(code: u8) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = MAGNITUDE[usize::from(code & 0x7)];
    if code & 0x8 == 0 {
        m
    } else {
        -m
    }
}

/// Dequantizes an MLX mxfp4 matrix to little-endian BF16, row major.
pub fn dequant_to_bf16(
    tensor: &str,
    weight: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, Gemma4Error> {
    check_planes(tensor, weight, scales, rows, cols)?;
    let mut out = Vec::with_capacity(rows * cols * 2);
    for (g, &exponent) in scales.iter().enumerate() {
        // `powi` is exact for every byte below 0xFF, including the f32
        // subnormals at exponents 0 and 1.
        let scale = 2.0f32.powi(i32::from(exponent) - 127);
        for &byte in &weight[g * 16..g * 16 + 16] {
            for code in [byte & 0x0F, byte >> 4] {
                let bits = (e2m1(code) * scale).to_bits();
                // The exactness claim in the module header, checked: an
                // E2M1 value times a power of two has no mantissa bits below
                // BF16's seven.
                debug_assert_eq!(
                    bits & 0xFFFF,
                    0,
                    "{tensor}: mxfp4 value not on the BF16 grid"
                );
                out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }
    }
    Ok(out)
}
