//! PIL's `Image.resize(..., Image.BICUBIC)`, reproduced bit for bit on RGB8.
//!
//! Adapted from the sconce vision crate's PIL variant (see `NOTICE`). Only the
//! PIL spelling is ported. sconce carries a second one for torchvision, which
//! differs solely in the fixed-point coefficient precision, and a third
//! unclamped f64 path for resampling learned parameters; neither is reachable
//! from this family. The reference processor's resize is
//! `PIL.Image.resize(resample=Image.BICUBIC)` (mlx-vlm
//! `_resize_video_frames`), and `preprocessor_config.json` declares
//! `resample: 3`, which is `Image.BICUBIC`.
//!
//! The reason the fixed-point path is reproduced rather than accumulated in
//! f64 and rounded at the end is that the two disagree by whole 8-bit levels.
//! After this family's `2*(px/255) - 1` normalization one level is `2/255`,
//! i.e. `0.0078` on a signal in `[-1, 1]` -- four orders of magnitude over the
//! 1e-6 bar the parity tests hold, so it is not a tolerance question.
//!
//! PIL has no separate "antialias" switch here: its downscale path widens the
//! filter support in proportion to the scale factor, which IS the antialiasing,
//! and its upscale path leaves the support at the plain bicubic radius.

/// The Keys cubic-convolution filter support radius.
const CUBIC_SUPPORT: f64 = 2.0;

/// PIL's flat coefficient precision: `PRECISION_BITS` in `Resample.c`, spelled
/// there as `32 - 8 - 2` -- an `int32` accumulator, less 8 bits of `uint8`
/// pixel, less 2 bits of headroom for the cubic kernel's negative lobes. PIL
/// applies it to every axis regardless of the weights.
const PRECISION_BITS: u32 = 22;

/// Keys cubic-convolution kernel with `a = -0.5`, PIL's "bicubic" filter.
fn cubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One output index's resample window: the input range `[start, start + n)` it
/// blends, and the fixed-point weights, already normalized to sum to one before
/// quantization.
struct AxisFilter {
    start: usize,
    fixed: Vec<i64>,
}

/// `w * 2^PRECISION_BITS`, rounded half away from zero -- PIL's
/// `(int)(0.5 + w * (1 << p))` with the sign handled explicitly, since Rust's
/// `as i64` truncates toward zero the way the C cast does.
fn round_away(w: f64) -> i64 {
    let v = w * (1u64 << PRECISION_BITS) as f64;
    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i64
}

/// Arithmetic-shift the accumulator down, then clamp. The shift happens BEFORE
/// the clamp, which is why the accumulator is seeded with `1 << (bits - 1)`
/// rather than rounded afterwards.
fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Every output index's filter for one axis.
///
/// The window is clamped to the input range and the weights are renormalized
/// AFTER that clamp, which is PIL's edge handling: an output sample near the
/// border blends fewer inputs but its weights still sum to one, so the border
/// does not darken. Renormalizing before the clamp instead (or padding the
/// input) is the ordinary way to get this subtly wrong.
fn precompute_filters(in_size: usize, out_size: usize) -> Vec<AxisFilter> {
    if in_size == 0 || out_size == 0 {
        return Vec::new();
    }
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = CUBIC_SUPPORT * filterscale;
    let inv_filterscale = 1.0 / filterscale;

    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax =
                (((center + support + 0.5).floor()) as i64).clamp(0, in_size as i64) as usize;
            let xmax = xmax.max(xmin);
            let mut weights: Vec<f64> = (xmin..xmax)
                .map(|x| cubic_kernel((x as f64 - center + 0.5) * inv_filterscale))
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
            AxisFilter {
                start: xmin,
                fixed: weights.iter().map(|&w| round_away(w)).collect(),
            }
        })
        .collect()
}

/// Resize interleaved RGB8 pixels to `new_h x new_w`.
///
/// Separable: the horizontal pass runs first into a full-height intermediate,
/// then the vertical pass, with a shift-and-clamp to `0..255` after each. That
/// intermediate rounding is part of the reference and not an implementation
/// choice -- carrying full precision between the passes lands different bytes.
///
/// Returns the input untouched when the size already matches, matching the
/// reference's early-out. This is PURELY a speed path, which was checked
/// rather than assumed: at `scale == 1` the half-pixel centre puts every
/// sample on an integer offset, the Keys kernel is zero at every nonzero
/// integer, and the weights collapse to `[0, 1, 0, 0]` -- so the general path
/// returns the same bytes. `an_identity_resize_is_the_identity_either_way`
/// pins that, and it is why a mutation deleting this branch survives the
/// parity suite.
pub fn resize_bicubic_pil(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<u8> {
    if new_w == src_w && new_h == src_h {
        return src.to_vec();
    }
    resample_pil(src, src_w, src_h, new_w, new_h)
}

/// [`resize_bicubic_pil`] without the equal-size early-out.
///
/// Exists so a test can assert the two agree at equal size. Not a separate
/// implementation: the early-out calls straight into this, so the pair cannot
/// drift the way two copies of a filter loop would.
pub fn resample_pil(src: &[u8], src_w: usize, src_h: usize, new_w: usize, new_h: usize) -> Vec<u8> {
    const CHANNELS: usize = 3;
    let col = precompute_filters(src_w, new_w);
    let row = precompute_filters(src_h, new_h);
    let seed = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: src_h rows of new_w pixels.
    let mut horiz = vec![0u8; src_h * new_w * CHANNELS];
    for y in 0..src_h {
        for (ox, filt) in col.iter().enumerate() {
            for c in 0..CHANNELS {
                let mut acc = seed;
                for (k, &wgt) in filt.fixed.iter().enumerate() {
                    acc += src[(y * src_w + filt.start + k) * CHANNELS + c] as i64 * wgt;
                }
                horiz[(y * new_w + ox) * CHANNELS + c] = clip8(acc);
            }
        }
    }

    // Vertical pass: new_h rows of new_w pixels.
    let mut out = vec![0u8; new_h * new_w * CHANNELS];
    for (oy, filt) in row.iter().enumerate() {
        for ox in 0..new_w {
            for c in 0..CHANNELS {
                let mut acc = seed;
                for (k, &wgt) in filt.fixed.iter().enumerate() {
                    acc += horiz[((filt.start + k) * new_w + ox) * CHANNELS + c] as i64 * wgt;
                }
                out[(oy * new_w + ox) * CHANNELS + c] = clip8(acc);
            }
        }
    }
    out
}
