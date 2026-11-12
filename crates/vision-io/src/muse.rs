//! Muse-Glimmer image preprocessing (`docs/MUSEGLIMMER_VISION_PHASE0.md`).
//!
//! A port of `MuseGlimmerImageProcessor._process_one` and its `smart_resize`
//! (mlx-vlm `models/muse_glimmer/processing_muse_glimmer.py`). It shares
//! nothing with the Qwen rule except the normalize step:
//!
//! - **The budget is a cap on merged tokens** (`max_image_tokens`, 4096), not a
//!   pixel count, and the size is chosen by a nearest-aspect search over the
//!   four floor/ceil grids, not by rounding to a factor.
//! - **The filter is Lanczos**, not bicubic.
//! - **Patches are in raster order** (row of patches by row of patches), not in
//!   merge-window order, and a row is `(T, C, P_h, P_w)`: the temporal copy
//!   outermost, then channel, then the patch's pixels. Neither Qwen's
//!   `(C, T, P, P)` nor this repo's Qwen row order `(T, P, P, C)`. The patch
//!   embedding is a plain bias-free `Linear`, so the weight is copied verbatim
//!   and the rows follow whatever order the processor emits.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::normalize::rescale_and_normalize;
use crate::params::PreprocessParams;
use crate::patchify::GridThw;
use crate::preprocess::PreprocessedImage;
use crate::resize::resize_lanczos_pil;
use crate::rounding::round_half_to_even;

/// The resized size in PIXELS, `(height, width)`, for an image of
/// `height x width`. `unit` is `patch_size * merge_size` (28) and
/// `max_tokens` the merged-token cap.
///
/// The reference's `smart_resize`, step for step. Ties between two grids with
/// the same aspect error (a 3.5 x 3.5 ideal, say) go to the first in the
/// reference's `itertools.product((floor_h, ceil_h), (floor_w, ceil_w))` order,
/// which is a deterministic choice here; the reference iterates a Python `set`
/// of tuples there, so on an exact tie the two can disagree. The test page and
/// every ordinary photograph have no tie.
pub fn muse_resized_dims(
    height: usize,
    width: usize,
    unit: usize,
    max_tokens: usize,
) -> Result<(usize, usize), VisionIoError> {
    if height == 0 || width == 0 || unit == 0 || max_tokens == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("muse resize of {height}x{width} (unit {unit}, cap {max_tokens})"),
        });
    }
    let mut ideal_h = height as f64 / unit as f64;
    let mut ideal_w = width as f64 / unit as f64;
    let ratio = if ideal_h > 0.0 {
        ideal_w / ideal_h
    } else {
        1.0
    };
    if ideal_h * ideal_w > max_tokens as f64 {
        ideal_h = (max_tokens as f64 / ratio).sqrt();
        ideal_w = ideal_h * ratio;
    }
    let hs = [ideal_h.floor() as i64, ideal_h.ceil() as i64];
    let ws = [ideal_w.floor() as i64, ideal_w.ceil() as i64];
    let mut candidates: Vec<(i64, i64)> = Vec::with_capacity(4);
    for &h in &hs {
        for &w in &ws {
            if !candidates.contains(&(h, w)) {
                candidates.push((h, w));
            }
        }
    }
    candidates.retain(|&(h, w)| h >= 1 && w >= 1 && h * w <= max_tokens as i64);
    if candidates.is_empty() {
        candidates.push((
            (round_half_to_even(ideal_h) as i64).max(1),
            (round_half_to_even(ideal_w) as i64).max(1),
        ));
    }
    let target = height as f64 / width as f64;
    let mut best = candidates[0];
    let mut best_key = f64::INFINITY;
    for &(h, w) in &candidates {
        let key = (h as f64 / w as f64 - target).abs();
        // Strictly less: the first of equal keys wins, like Python's `min`.
        if key < best_key {
            best_key = key;
            best = (h, w);
        }
    }
    Ok((best.0 as usize * unit, best.1 as usize * unit))
}

/// Patch rows in raster order, `(T, C, P_h, P_w)` inside a row, from
/// interleaved `(y, x, c)` normalized samples. A still image is repeated
/// across the temporal patch.
pub fn muse_patch_rows(
    normalized: &[f32],
    height: usize,
    width: usize,
    params: &PreprocessParams,
) -> Result<(Vec<f32>, GridThw), VisionIoError> {
    let p = params.patch_size;
    let c_n = params.in_channels;
    if p == 0 || height % p != 0 || width % p != 0 || normalized.len() != height * width * c_n {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("muse patchify of {height}x{width} samples with patch {p}"),
        });
    }
    let (gh, gw) = (height / p, width / p);
    let row_len = params.temporal_patch_size * c_n * p * p;
    let mut rows = vec![0.0f32; gh * gw * row_len];
    for gy in 0..gh {
        for gx in 0..gw {
            let row = &mut rows[(gy * gw + gx) * row_len..(gy * gw + gx + 1) * row_len];
            for c in 0..c_n {
                for py in 0..p {
                    for px in 0..p {
                        let v = normalized[((gy * p + py) * width + gx * p + px) * c_n + c];
                        for t in 0..params.temporal_patch_size {
                            row[((t * c_n + c) * p + py) * p + px] = v;
                        }
                    }
                }
            }
        }
    }
    Ok((rows, GridThw::new(1, gh, gw)))
}

/// The whole Muse pipeline: resize (cap and filter), normalize, patchify.
///
/// The cap is the LOWER of `max_tokens` and what `params.max_pixels` allows,
/// so the session's memory-budget clamp (which lowers `max_pixels`) bites.
pub fn muse_preprocess(
    image: &Rgb8Image,
    params: &PreprocessParams,
    max_tokens: usize,
) -> Result<PreprocessedImage, VisionIoError> {
    let unit = params.patch_size * params.merge_size;
    let by_pixels = (params.max_pixels / (unit * unit)).max(1);
    let cap = max_tokens.min(by_pixels);
    let (resized_h, resized_w) = muse_resized_dims(image.height, image.width, unit, cap)?;
    let resized = resize_lanczos_pil(&image.data, image.width, image.height, resized_w, resized_h);
    let normalized = rescale_and_normalize(&resized, params);
    let (patch_rows, grid) = muse_patch_rows(&normalized, resized_h, resized_w, params)?;
    Ok(PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (resized_h, resized_w),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ResizeRule;
    use crate::resize::{resample_pil_with, PilFilter};

    fn params() -> PreprocessParams {
        PreprocessParams {
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            in_channels: 3,
            min_pixels: 784,
            max_pixels: 4096 * 784,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
            resize: ResizeRule::MuseTokens { max_tokens: 4096 },
        }
    }

    #[test]
    fn the_test_page_resizes_to_the_nearest_aspect_grid() {
        // 1024x1280: ideal 36.57 x 45.71. Of the four floor/ceil grids,
        // 36x45 has aspect exactly 0.8 (the image's own), so it wins and the
        // image shrinks to 1008x1260 (1,620 merged tokens).
        assert_eq!(
            muse_resized_dims(1280, 1024, 28, 4096).unwrap(),
            (1260, 1008)
        );
        assert_eq!(
            muse_resized_dims(1024, 1280, 28, 4096).unwrap(),
            (1008, 1260)
        );
    }

    #[test]
    fn an_exact_multiple_keeps_its_size() {
        assert_eq!(muse_resized_dims(448, 448, 28, 4096).unwrap(), (448, 448));
    }

    #[test]
    fn a_huge_image_is_capped_to_the_token_budget() {
        let (h, w) = muse_resized_dims(8000, 6000, 28, 4096).unwrap();
        assert!((h / 28) * (w / 28) <= 4096);
        // And it is still close to the cap rather than collapsed.
        assert!((h / 28) * (w / 28) > 3500, "{h}x{w}");
    }

    #[test]
    fn a_sliver_falls_back_to_the_rounded_grid() {
        // Under the cap but so thin that floor(ideal_h) is 0: the reference's
        // `h >= 1` filter keeps only ceil, and a grid never has a zero side.
        let (h, w) = muse_resized_dims(5, 2000, 28, 4096).unwrap();
        assert!(h >= 28 && w >= 28);
    }

    #[test]
    fn patch_rows_are_raster_with_the_temporal_copy_outermost() {
        // A 28x28 image of distinct samples: 2x2 patches of 14x14.
        let p = params();
        let (h, w) = (28usize, 28usize);
        let normalized: Vec<f32> = (0..h * w * 3).map(|i| i as f32).collect();
        let (rows, grid) = muse_patch_rows(&normalized, h, w, &p).unwrap();
        assert_eq!((grid.t, grid.h, grid.w), (1, 2, 2));
        let row_len = 2 * 3 * 14 * 14;
        assert_eq!(rows.len(), 4 * row_len);
        // Patch (gy=1, gx=0) is raster index 2.
        let row = &rows[2 * row_len..3 * row_len];
        // Its first sample: channel 0 of pixel (14, 0).
        assert_eq!(row[0], ((14 * 28) * 3) as f32);
        // The next channel starts 14*14 samples later.
        assert_eq!(row[14 * 14], ((14 * 28) * 3 + 1) as f32);
        // The temporal copy is the SAME frame, a whole (C, P, P) block later.
        assert_eq!(row[3 * 14 * 14], row[0]);
        // Within a patch row, x moves fastest, then y.
        assert_eq!(row[1], ((14 * 28 + 1) * 3) as f32);
        assert_eq!(row[14], (((14 + 1) * 28) * 3) as f32);
    }

    #[test]
    fn lanczos_matches_a_hand_computed_downscale() {
        // 6 -> 3 on one row: scale 2, filterscale 2, support 6, so every
        // output sample blends all six inputs. For the middle output the
        // centre is 3.0 and the weights are lanczos((x - 3 + 0.5) / 2).
        let src: Vec<u8> = [10u8, 40, 90, 160, 200, 250]
            .iter()
            .flat_map(|&v| [v, v, v])
            .collect();
        let out = resample_pil_with(PilFilter::Lanczos, &src, 6, 1, 3, 1);
        let sinc = |x: f64| {
            if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            }
        };
        let lanczos = |x: f64| {
            if x.abs() < 3.0 {
                sinc(x) * sinc(x / 3.0)
            } else {
                0.0
            }
        };
        let centre = 1.5 * 2.0;
        let weights: Vec<f64> = (0..6)
            .map(|x| lanczos((x as f64 - centre + 0.5) / 2.0))
            .collect();
        let total: f64 = weights.iter().sum();
        let expected: f64 = weights
            .iter()
            .zip([10.0, 40.0, 90.0, 160.0, 200.0, 250.0])
            .map(|(w, v)| w / total * v)
            .sum();
        let got = out[3] as f64;
        assert!(
            (got - expected).abs() <= 1.0,
            "got {got}, expected {expected:.3}"
        );
        // Bicubic weights differ, so the same row resamples differently: the
        // fixture can tell the filters apart.
        let cubic = resample_pil_with(PilFilter::Bicubic, &src, 6, 1, 3, 1);
        assert!(
            out.iter().zip(&cubic).any(|(a, b)| a != b),
            "lanczos and bicubic agree on this row"
        );
    }

    #[test]
    fn lanczos_is_the_identity_at_equal_size() {
        let src: Vec<u8> = (0..4 * 4 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resample_pil_with(PilFilter::Lanczos, &src, 4, 4, 4, 4), src);
    }
}
