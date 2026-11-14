//! Bit-exact port of MLX's `mx.quantize` affine mode (4- and 8-bit).
//!
//! **Why this exists beside [`crate::quant::quantize_int4_affine`].** That
//! reference uses `scale = (max - min) / 15`, `bias = min`. MLX instead snaps
//! the scale so the larger-magnitude group edge is exactly representable, and
//! quantizes with the f32 scale while storing only its BF16 cast. The two
//! produce different bytes for the same weights. Quantize-on-install must
//! write what `mlx_lm.convert` writes, so an install from a BF16 checkpoint
//! is byte-identical to one from the mlx-community 4-bit conversion of the
//! same weights. The older function stays untouched because frozen fixtures
//! and other callers depend on its exact output.
//!
//! Ported from `mlx/backend/metal/kernels/quantized.h` (`affine_quantize`),
//! the kernel `mlx_lm.convert` runs on Apple GPUs. The CPU kernel
//! (`backend/cpu/quantized.cpp`) differs only in rounding: it uses `rint`
//! (half to even) where Metal's `round` is half away from zero, which is
//! what [`f32::round`] does.
//!
//! Layout is the existing one: groups of [`GROUP_SIZE`] along the row, 4-bit
//! low nibble first, BF16 scale and bias per group, `w = q * scale + bias`.

use crate::quant::{f32_to_bf16, Int4AffineRow, Int8AffineRow, GROUP_SIZE};

/// MLX's `eps` floor on the scale.
const EPS: f32 = 1e-7;

/// Per-group `(scale, bias)` in f32, exactly as the Metal kernel computes
/// them before the BF16 store. `n_bins` is `2^bits - 1`.
#[inline]
fn group_params(group: &[f32], n_bins: f32) -> (f32, f32) {
    let mut w_min = f32::INFINITY;
    let mut w_max = f32::NEG_INFINITY;
    for &v in group {
        w_min = w_min.min(v);
        w_max = w_max.max(v);
    }
    let mut scale = ((w_max - w_min) / n_bins).max(EPS);
    let side = w_min.abs() > w_max.abs();
    if !side {
        scale = -scale;
    }
    let edge = if side { w_min } else { w_max };
    let q0 = (edge / scale).round();
    if q0 == 0.0 {
        (scale, 0.0)
    } else {
        (edge / q0, edge)
    }
}

#[inline]
fn quantize_value(w: f32, scale: f32, bias: f32, n_bins: f32) -> u8 {
    // The Metal kernel clamps only the top. A value below zero can come only
    // from f32 error at the snapped edge and rounds to -0, so clamping at 0
    // matches its cast.
    ((w - bias) / scale).round().clamp(0.0, n_bins) as u8
}

fn check_len(row: &[f32]) -> usize {
    assert!(
        row.len() % GROUP_SIZE == 0,
        "row length {} is not a multiple of {GROUP_SIZE}",
        row.len()
    );
    row.len() / GROUP_SIZE
}

/// `mx.quantize(w, group_size=64, bits=4)` for one row.
pub fn quantize_int4_affine_mlx(row: &[f32]) -> Int4AffineRow {
    let n_groups = check_len(row);
    let mut packed = vec![0u8; row.len() / 2];
    let mut scales = vec![0u16; n_groups];
    let mut biases = vec![0u16; n_groups];
    for g in 0..n_groups {
        let group = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let (scale, bias) = group_params(group, 15.0);
        scales[g] = f32_to_bf16(scale);
        biases[g] = f32_to_bf16(bias);
        for (k, pair) in group.chunks_exact(2).enumerate() {
            let lo = quantize_value(pair[0], scale, bias, 15.0);
            let hi = quantize_value(pair[1], scale, bias, 15.0);
            packed[g * (GROUP_SIZE / 2) + k] = lo | (hi << 4);
        }
    }
    Int4AffineRow {
        packed,
        scales,
        biases,
    }
}

/// `mx.quantize(w, group_size=64, bits=8)` for one row.
pub fn quantize_int8_affine_mlx(row: &[f32]) -> Int8AffineRow {
    let n_groups = check_len(row);
    let mut packed = vec![0u8; row.len()];
    let mut scales = vec![0u16; n_groups];
    let mut biases = vec![0u16; n_groups];
    for g in 0..n_groups {
        let group = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let (scale, bias) = group_params(group, 255.0);
        scales[g] = f32_to_bf16(scale);
        biases[g] = f32_to_bf16(bias);
        for (k, &w) in group.iter().enumerate() {
            packed[g * GROUP_SIZE + k] = quantize_value(w, scale, bias, 255.0);
        }
    }
    Int8AffineRow {
        packed,
        scales,
        biases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{bf16_to_f32, dequantize_int4_affine, dequantize_int8_affine};

    fn row(seed: u32, n: usize) -> Vec<f32> {
        // Deterministic, mixed-sign, BF16-representable values.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.45) * 0.2;
                bf16_to_f32(f32_to_bf16(x))
            })
            .collect()
    }

    #[test]
    fn int4_round_trip_error_is_within_one_step() {
        let r = row(7, 256);
        let q = quantize_int4_affine_mlx(&r);
        let d = dequantize_int4_affine(&q, r.len());
        for g in 0..r.len() / GROUP_SIZE {
            let step = bf16_to_f32(q.scales[g]).abs();
            for k in 0..GROUP_SIZE {
                let i = g * GROUP_SIZE + k;
                // Half a step of rounding plus the BF16 cast of scale/bias.
                assert!((d[i] - r[i]).abs() <= step * 0.51 + 1e-3, "i={i}");
            }
        }
    }

    #[test]
    fn int8_round_trip_error_is_within_one_step() {
        let r = row(11, 192);
        let q = quantize_int8_affine_mlx(&r);
        let d = dequantize_int8_affine(&q, r.len());
        for g in 0..r.len() / GROUP_SIZE {
            let step = bf16_to_f32(q.scales[g]).abs();
            for k in 0..GROUP_SIZE {
                let i = g * GROUP_SIZE + k;
                // Half a step of rounding plus the stored scale's BF16 cast,
                // which MLX also takes: up to q * |scale| * 2^-9, q <= 255.
                assert!((d[i] - r[i]).abs() <= step * 1.0 + 1e-4, "i={i}");
            }
        }
    }

    #[test]
    fn the_larger_magnitude_edge_is_exact() {
        // MLX's defining property: the edge farther from zero dequantizes to
        // itself (up to the BF16 cast of the stored scale).
        let mut r = vec![0.01f32; 64];
        r[3] = -0.5;
        r[9] = 0.125;
        let q = quantize_int4_affine_mlx(&r);
        let d = dequantize_int4_affine(&q, 64);
        assert_eq!(bf16_to_f32(q.biases[0]), -0.5);
        assert!((d[3] + 0.5).abs() < 1e-6);
        // Positive side larger: negative scale, bias at the max.
        let mut p = vec![-0.01f32; 64];
        p[5] = 0.75;
        let q = quantize_int4_affine_mlx(&p);
        assert!(bf16_to_f32(q.scales[0]) < 0.0);
        assert_eq!(bf16_to_f32(q.biases[0]), 0.75);
    }

    #[test]
    fn a_constant_group_has_zero_bias_or_exact_edge() {
        let q = quantize_int4_affine_mlx(&[0.0f32; 64]);
        assert_eq!(bf16_to_f32(q.biases[0]), 0.0);
        assert!(q.packed.iter().all(|&b| b == 0));
    }

    #[test]
    fn differs_from_the_min_bias_reference() {
        // Guards the reason this module exists: if the two ever agree on a
        // mixed-sign row, one of them has been changed into the other.
        let r = row(3, 64);
        let a = quantize_int4_affine_mlx(&r);
        let b = crate::quant::quantize_int4_affine(&r);
        assert_ne!(a.biases, b.biases);
    }
}
