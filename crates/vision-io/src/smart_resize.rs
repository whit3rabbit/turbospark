//! `smart_resize`: round both edges to the patch/merge factor, then clamp the
//! pixel count into the checkpoint's budget by uniform rescaling.
//!
//! A direct port of `_smart_resize_image` in mlx-vlm's
//! `processing_qwen3_vl.py`, which is itself HF's `smart_resize`. Adapted from
//! the sconce vision crate (see `NOTICE`), with sconce's extra final clamp
//! dropped -- see [`resized_dims`].

use crate::error::VisionIoError;
use crate::params::{PreprocessParams, MAX_ASPECT_RATIO};
use crate::rounding::round_half_to_even;

/// Round to the nearest multiple of `factor` under Python's `round()`.
///
/// Deliberately unclamped, matching the reference. An edge under half a factor
/// rounds to ZERO, which then makes the product zero, which trips the
/// `< min_pixels` branch and is rescaled from the original dims. Clamping to
/// `factor` here instead would suppress that branch on exactly those inputs
/// and land a different answer.
fn round_by_factor(value: usize, factor: usize) -> usize {
    (round_half_to_even(value as f64 / factor as f64) as usize) * factor
}

/// Both take `f64` rather than a pre-truncated `usize`: truncating before the
/// divide discards the fractional pixel `ceil` exists to catch, and the
/// reference passes `height * beta` unrounded.
fn floor_by_factor(value: f64, factor: usize) -> usize {
    ((value / factor as f64).floor() as usize) * factor
}

fn ceil_by_factor(value: f64, factor: usize) -> usize {
    ((value / factor as f64).ceil() as usize) * factor
}

/// The resized `(height, width)` for a source image.
///
/// THREE THINGS HERE ARE EXACT REQUIREMENTS RATHER THAN STYLE, and each
/// produces a plausible wrong answer if changed:
///
/// - The aspect-ratio check runs BEFORE any rounding, on the original dims.
/// - Both rescale branches take `beta` from, and apply it to, the ORIGINAL
///   dims rather than the already-rounded pair. Rescaling the rounded pair
///   double-counts the rounding: a 19x19 source reads (56, 56) that way
///   against the reference's (84, 84), because `19 * sqrt(3136/361)` is
///   `56.000000000000014` and ceils to a third patch row.
/// - `max(factor, ...)` is applied in the max_pixels branch ONLY. The
///   reference does not apply it in the min_pixels branch, and it cannot
///   matter there: that branch is reached only when beta > 1, so the ceil
///   already lands at or above one factor. sconce applies a final clamp to
///   both; this does not, because the only inputs where the two differ are
///   the ones where the reference genuinely returns a zero edge -- and this
///   REFUSES those by name rather than inventing a one-patch image.
pub fn resized_dims(
    orig_h: usize,
    orig_w: usize,
    params: &PreprocessParams,
) -> Result<(usize, usize), VisionIoError> {
    if orig_h == 0 || orig_w == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("source image is {orig_w}x{orig_h}; both edges must be positive"),
        });
    }

    let (long, short) = if orig_h >= orig_w {
        (orig_h, orig_w)
    } else {
        (orig_w, orig_h)
    };
    let ratio = long as f64 / short as f64;
    if ratio > MAX_ASPECT_RATIO {
        return Err(VisionIoError::AspectRatioTooExtreme {
            ratio,
            limit: MAX_ASPECT_RATIO,
        });
    }

    let factor = params.spatial_factor();
    let mut h = round_by_factor(orig_h, factor);
    let mut w = round_by_factor(orig_w, factor);

    let orig_pixels = (orig_h * orig_w) as f64;
    if h * w > params.max_pixels {
        let beta = (orig_pixels / params.max_pixels as f64).sqrt();
        h = floor_by_factor(orig_h as f64 / beta, factor).max(factor);
        w = floor_by_factor(orig_w as f64 / beta, factor).max(factor);
    } else if h * w < params.min_pixels {
        let beta = (params.min_pixels as f64 / orig_pixels).sqrt();
        h = ceil_by_factor(orig_h as f64 * beta, factor);
        w = ceil_by_factor(orig_w as f64 * beta, factor);
    }

    if h == 0 || w == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{orig_w}x{orig_h} resizes to {w}x{h} under a min_pixels of {}; \
                 the budget cannot rescue an edge that rounded to zero",
                params.min_pixels
            ),
        });
    }
    Ok((h, w))
}
