//! TurboQuant KV-cache codec reference: CPU ground truth for
//! `crates/gpu`'s `attention_tq.metal` and `kv_quantize_tq.metal` kernels.
//!
//! Ported from mlx-vlm's `_TurboQuantMSECodec` (`mlx_vlm/turboquant.py`),
//! the only codec that file actually ships live -- see `docs/TRUBOQUANT.md`
//! for what else lives in that source and why none of it is ported (QJL,
//! PolarQuant, the mixed-precision split codec, every fused prefill
//! kernel). Per K or V row of length `dim`:
//!
//! 1. `norm = ||x||` (stored separately), `unit = x / max(norm, eps)`.
//! 2. `rotated = rht_forward(unit, signs)`, a Randomized Hadamard
//!    Transform with a fixed +/-1 sign vector, ONE per (K or V, whole
//!    model) rather than per layer or head.
//! 3. Each rotated coordinate is quantized against a 1-D Lloyd-Max
//!    codebook fit to the sphere-marginal density `(1 - x^2)^((dim-3)/2)`
//!    -- the density a coordinate of a random rotated unit vector in
//!    `dim` dimensions follows. The index is the count of codebook
//!    midpoints the coordinate exceeds.
//! 4. Indices are packed LSB-first into `u32` words, `bits` wide each,
//!    straddling word boundaries as needed.
//!
//! **The convention below was measured against mlx-vlm's own output, not
//! read off the paper.** `crates/compute/tests/kv_quant.rs` feeds mlx-vlm's
//! own printed sign vectors into this module's quantize/dequantize path and
//! requires bit-identical packed words -- see [`sign_vector`]'s doc for why
//! this module's OWN sign generator does not have to (and does not)
//! reproduce numpy's PCG64 stream.

use crate::wht::wht;

/// Seed for the key rotation's sign vector. mlx-vlm's own default
/// (`KEY_SEED` in the model wrapper that constructs `_TurboQuantMSECodec`).
pub const KEY_SEED: u64 = 0;
/// Seed for the value rotation's sign vector. mlx-vlm's own default.
pub const VALUE_SEED: u64 = 1;

/// `max(norm, eps)` floor before forming the unit vector. mlx-vlm's `_EPS`.
pub const NORM_EPS: f32 = 1e-6;

const CODEBOOK_GRID: usize = 32768;

/// splitmix64, deterministic and dependency-free.
///
/// **NOT numpy's PCG64.** mlx-vlm's `_rht_sign_vector` seeds
/// `np.random.default_rng(seed + dim * 7919)` and this port deliberately
/// does not reproduce that stream: the codec needs a FIXED vector per
/// `(dim, seed)`, consistently used at write time and read time, and any
/// deterministic +/-1 vector serves that equally well -- the choice of RNG
/// is not part of the codec's numerics the way the codebook and the
/// packing order are. `sign_vector`'s own bytes are checked only for
/// "differs between K and V"; the oracle-backed correctness tests in
/// `tests/kv_quant.rs` feed mlx-vlm's OWN printed signs into
/// [`quantize_row`] / [`dequantize_row`] instead of regenerating them here.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic +/-1 sign vector for the randomized Hadamard rotation, one
/// per `(dim, seed)`. Combines `dim` into the seed the way mlx-vlm's own
/// generator does (`seed + dim * 7919`) purely so [`KEY_SEED`] and
/// [`VALUE_SEED`] can never collide across a mixed-head-dim model; see this
/// function's module-level note for why the underlying RNG differs from
/// mlx-vlm's on purpose.
pub fn sign_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add((dim as u64).wrapping_mul(7919));
    (0..dim)
        .map(|_| {
            if splitmix64_next(&mut state) & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

/// Randomized Hadamard Transform, forward: `hadamard(signs * x) / sqrt(D)`.
///
/// `x.len()` and `signs.len()` must both be the same power of two --
/// [`wht::wht`] asserts this. Every head_dim this port runs (64/128/256/
/// 512) already is one, so mlx-vlm's `_rht_forward` pad-to-next-power-of-
/// two branch for non-power-of-two dims is never reached and is not
/// ported; `model_io::kv_quant::rht_supported` refuses any install whose
/// full head_dim is not a power of two in `32..=512` before this is ever
/// called.
pub fn rht_forward(x: &[f32], signs: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), signs.len(), "x and signs must be the same length");
    let y: Vec<f32> = x.iter().zip(signs).map(|(a, b)| a * b).collect();
    wht(&y)
}

/// Inverse: `signs * hadamard(x) / sqrt(D)`. [`wht::wht`] is its own
/// inverse (orthogonal and symmetric), so this differs from
/// [`rht_forward`] only in when the sign multiply happens -- multiplying by
/// `signs` before or after a linear, sign-vector-diagonal-conjugated
/// transform are not the same operation in general, which is why the two
/// are separate functions rather than one called twice.
pub fn rht_inverse(x: &[f32], signs: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), signs.len(), "x and signs must be the same length");
    let y = wht(x);
    y.iter().zip(signs).map(|(a, b)| a * b).collect()
}

/// log-density (up to an additive constant that cancels after
/// normalization) of one coordinate of a random rotated unit vector in
/// `dim` dimensions: `((dim - 3) / 2) * ln(max(1 - x^2, 1e-30))`. mlx-vlm's
/// `_beta_pdf` computes an exact `lgamma`-based normalizing constant before
/// this term and then immediately subtracts `max(log_pdf)`, which removes
/// that constant (it is the same additive term at every grid point) -- so
/// it is dropped here rather than recomputed.
fn beta_log_density(x: f64, dim: usize) -> f64 {
    if dim <= 1 {
        return 0.0;
    }
    let one_minus_x2 = (1.0 - x * x).max(1e-30);
    ((dim as f64 - 3.0) / 2.0) * one_minus_x2.ln()
}

/// Linear interpolation of `cdf -> grid` at query point `q`, mirroring
/// `np.interp`: clamps outside `cdf`'s range, otherwise interpolates
/// between the bracketing samples. `cdf` must be nondecreasing.
fn interp(q: f64, cdf: &[f64], grid: &[f64]) -> f64 {
    if q <= cdf[0] {
        return grid[0];
    }
    if q >= cdf[cdf.len() - 1] {
        return grid[grid.len() - 1];
    }
    let idx = match cdf.binary_search_by(|probe| probe.partial_cmp(&q).unwrap()) {
        Ok(i) => return grid[i],
        Err(i) => i,
    };
    let (x0, x1) = (cdf[idx - 1], cdf[idx]);
    let (y0, y1) = (grid[idx - 1], grid[idx]);
    if x1 == x0 {
        return y0;
    }
    y0 + (y1 - y0) * (q - x0) / (x1 - x0)
}

/// Lloyd-Max codebook for the sphere-marginal density of a rotated unit
/// vector's single coordinate. Mirrors mlx-vlm's `_codebook`: a 32768-point
/// grid over `(-1 + 1e-6, 1 - 1e-6)`, inverse-CDF initialization at the
/// `levels` quantile midpoints, up to 100 fixed-point (weighted-centroid)
/// iterations at `1e-6` max-delta tolerance, and a right-inclusive last
/// bucket (`x <= boundary` rather than `x < boundary`) so the endpoint at
/// `+1 - 1e-6` always lands in the top bucket.
///
/// Computed in `f64` throughout for iteration stability and cast to `f32`
/// only at the end -- mlx-vlm's own arithmetic is a mix of `f32` and
/// Python-`float` (`f64`) promotions depending on numpy version, so nothing
/// here targets bit-exactness. `tests/kv_quant.rs` checks the result
/// against mlx-vlm's own printed output to a `1e-4` absolute tolerance --
/// mlx-vlm's own loop runs in `f32` throughout, and 100 iterations is
/// enough for that to separate from this `f64` one by more than a mere
/// rounding difference (observed up to 5.3e-5).
pub fn codebook(dim: usize, bits: u8) -> Vec<f32> {
    if bits == 0 {
        return Vec::new();
    }
    let levels = 1usize << bits;
    if dim <= 1 {
        if levels == 1 {
            return vec![0.0];
        }
        return (0..levels)
            .map(|i| -1.0 + 2.0 * i as f32 / (levels as f32 - 1.0))
            .collect();
    }

    let grid: Vec<f64> = (0..CODEBOOK_GRID)
        .map(|i| {
            let t = i as f64 / (CODEBOOK_GRID as f64 - 1.0);
            (-1.0 + 1e-6) + t * (2.0 - 2e-6)
        })
        .collect();

    let log_pdf: Vec<f64> = grid.iter().map(|&x| beta_log_density(x, dim)).collect();
    let max_log = log_pdf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = log_pdf.iter().map(|&lp| (lp - max_log).exp()).collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for w in weights.iter_mut() {
            *w /= total;
        }
    } else {
        let uniform = 1.0 / weights.len() as f64;
        weights.iter_mut().for_each(|w| *w = uniform);
    }

    let mut cdf = vec![0.0f64; grid.len()];
    let mut running = 0.0;
    for (i, w) in weights.iter().enumerate() {
        running += w;
        cdf[i] = running;
    }

    let mut centroids: Vec<f64> = (0..levels)
        .map(|i| {
            let q = (i as f64 + 0.5) / levels as f64;
            interp(q, &cdf, &grid)
        })
        .collect();

    for _ in 0..100 {
        let mut boundaries = vec![0.0f64; levels + 1];
        boundaries[0] = -1.0;
        boundaries[levels] = 1.0;
        for i in 1..levels {
            boundaries[i] = 0.5 * (centroids[i - 1] + centroids[i]);
        }

        let mut new_centroids = centroids.clone();
        for (i, new_centroid) in new_centroids.iter_mut().enumerate() {
            let lo = boundaries[i];
            let hi = boundaries[i + 1];
            let last = i == levels - 1;
            let mut num = 0.0;
            let mut den = 0.0;
            for (gi, &x) in grid.iter().enumerate() {
                let inside = if last {
                    x >= lo && x <= hi
                } else {
                    x >= lo && x < hi
                };
                if inside {
                    num += weights[gi] * x;
                    den += weights[gi];
                }
            }
            if den > 0.0 {
                *new_centroid = num / den;
            }
        }

        let max_delta = centroids
            .iter()
            .zip(&new_centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        centroids = new_centroids;
        if max_delta < 1e-6 {
            break;
        }
    }

    centroids.into_iter().map(|c| c as f32).collect()
}

/// Bucket boundaries between adjacent codebook entries:
/// `(cb[i] + cb[i + 1]) / 2`. [`quantize_index`] counts how many of these a
/// rotated coordinate exceeds.
pub fn midpoints(codebook: &[f32]) -> Vec<f32> {
    codebook.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
}

/// The codebook index for one rotated coordinate: the count of midpoints it
/// strictly exceeds. Mirrors mlx-vlm's
/// `indices = indices + (rotated > midpoint).astype(uint32)`, summed over
/// every midpoint.
pub fn quantize_index(value: f32, midpoints: &[f32]) -> u32 {
    midpoints.iter().filter(|&&m| value > m).count() as u32
}

/// Packed word count for `length` `bits`-wide indices, LSB-first,
/// `u32`-aligned: `ceil(length * bits / 32)`.
pub fn packed_words(length: usize, bits: u8) -> usize {
    if length == 0 || bits == 0 {
        0
    } else {
        (length * bits as usize).div_ceil(32)
    }
}

/// Packs `indices` (each `< 2^bits`) LSB-first into `u32` words. Mirrors
/// mlx-vlm's `_pack_lowbit`: index `i` starts at bit `i * bits`; an index
/// that straddles a word boundary spills its high bits into the next word
/// (`packed[w] |= v << offset; if offset + bits > 32: packed[w+1] |= v >>
/// (bits - spill)`).
pub fn pack_lsb_first(indices: &[u32], bits: u8) -> Vec<u32> {
    if bits == 0 {
        return Vec::new();
    }
    assert!((1..=16).contains(&bits), "bits {bits} must be in 1..=16");
    let width = packed_words(indices.len(), bits);
    let mut packed = vec![0u32; width];
    let bits_i = bits as i64;
    for (i, &value) in indices.iter().enumerate() {
        assert!(
            value >> bits == 0,
            "index {i} = {value} does not fit {bits} bits"
        );
        let bit_offset = i as u32 * bits as u32;
        let word_idx = (bit_offset / 32) as usize;
        let offset = bit_offset % 32;
        packed[word_idx] |= value << offset;
        let spill = offset as i64 + bits_i - 32;
        if spill > 0 {
            packed[word_idx + 1] |= value >> (bits_i - spill) as u32;
        }
    }
    packed
}

/// Inverse of [`pack_lsb_first`].
pub fn unpack_lsb_first(packed: &[u32], bits: u8, length: usize) -> Vec<u32> {
    if bits == 0 {
        return Vec::new();
    }
    assert!((1..=16).contains(&bits), "bits {bits} must be in 1..=16");
    let mask = (1u32 << bits) - 1;
    let bits_i = bits as i64;
    (0..length)
        .map(|i| {
            let bit_offset = i as u32 * bits as u32;
            let word_idx = (bit_offset / 32) as usize;
            let offset = bit_offset % 32;
            let mut value = packed[word_idx] >> offset;
            let spill = offset as i64 + bits_i - 32;
            if spill > 0 {
                value |= packed[word_idx + 1] << (bits_i - spill) as u32;
            }
            value & mask
        })
        .collect()
}

/// One quantized K or V row: the pre-rotation L2 norm plus its packed
/// per-coordinate codebook indices.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedRow {
    pub norm: f32,
    pub words: Vec<u32>,
}

/// Quantizes one row. Mirrors mlx-vlm's
/// `_TurboQuantMSECodec.quantize` / `_quantize_unit`: `norm = ||x||`,
/// `unit = x / max(norm, eps)`, `rotated = rht_forward(unit, signs)`, one
/// codebook index per coordinate, packed LSB-first.
pub fn quantize_row(x: &[f32], signs: &[f32], midpoints: &[f32], bits: u8) -> QuantizedRow {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let inv = 1.0 / norm.max(NORM_EPS);
    let unit: Vec<f32> = x.iter().map(|v| v * inv).collect();
    let rotated = rht_forward(&unit, signs);
    let indices: Vec<u32> = rotated
        .iter()
        .map(|&r| quantize_index(r, midpoints))
        .collect();
    QuantizedRow {
        norm,
        words: pack_lsb_first(&indices, bits),
    }
}

/// Dequantizes one row back to `dim` floats: codebook lookup per
/// coordinate, [`rht_inverse`], scaled by the stored norm. Mirrors
/// `_TurboQuantMSECodec.dequantize` / `_dequantize_unit`.
pub fn dequantize_row(
    row: &QuantizedRow,
    signs: &[f32],
    codebook: &[f32],
    bits: u8,
    dim: usize,
) -> Vec<f32> {
    let indices = unpack_lsb_first(&row.words, bits, dim);
    let rotated: Vec<f32> = indices.iter().map(|&i| codebook[i as usize]).collect();
    let unit = rht_inverse(&rotated, signs);
    unit.iter().map(|&u| u * row.norm).collect()
}
