//! Preprocessing for the Gemma 4 vision tower (`docs/GEMMA4_VISION_PHASE0.md`).
//!
//! A different pipeline from the rest of this crate, and the differences are
//! the point. There is no smart resize, no temporal axis, no normalize, and
//! no merge-window order:
//!
//! - resize keeps the aspect ratio and fits a SOFT-TOKEN budget
//!   (`max_soft_tokens * 9` patches), both sides a multiple of 48 px
//!   (`pooling_kernel_size * patch_size`);
//! - samples are only rescaled to `[0, 1]`. The tower's own `2 * (x - 0.5)`
//!   happens in model code, not here, which is where the reference puts it;
//! - patches are emitted ROW-MAJOR over the grid, each one
//!   `[16 rows][16 cols][RGB]`, the order `input_proj.weight` `[1152, 768]`
//!   reads;
//! - positions are `(x, y)`, column first. The tower's axial rope rotates the
//!   first half of each head by x, and the pooler groups by both.
//!
//! References: transformers `image_processing_pil_gemma4.py` and
//! mlx-vlm `models/gemma4/processing_gemma4.py` / `vision.py`, at the pins in
//! the Phase 0 doc.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;

/// Pixels per patch side.
pub const GEMMA4_PATCH: usize = 16;
/// Patches per pooled soft token, per side.
pub const GEMMA4_POOL: usize = 3;
/// Values per patch row: `16 * 16 * 3`.
pub const GEMMA4_PATCH_DIM: usize = GEMMA4_PATCH * GEMMA4_PATCH * 3;
/// The budgets the reference processor accepts (`_SUPPORTED_SOFT_TOKENS`).
pub const GEMMA4_SOFT_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// The checkpoint's default, `processor_config.json` `max_soft_tokens`.
pub const GEMMA4_DEFAULT_SOFT_TOKENS: usize = 280;

/// `(height, width)` an image is resized to for `max_patches` patches:
/// `get_aspect_ratio_preserving_size`, line for line, including its edge
/// cases for a side that rounds to zero.
pub fn gemma4_resize_dims(
    height: usize,
    width: usize,
    max_patches: usize,
) -> Result<(usize, usize), VisionIoError> {
    let invalid = |detail: String| VisionIoError::InvalidDimensions { detail };
    if height == 0 || width == 0 {
        return Err(invalid("image has a zero dimension".into()));
    }
    let total = (height * width) as f64;
    let target_px = (max_patches * GEMMA4_PATCH * GEMMA4_PATCH) as f64;
    let factor = (target_px / total).sqrt();
    let side = GEMMA4_POOL * GEMMA4_PATCH;
    let mut h = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut w = ((factor * width as f64) / side as f64).floor() as usize * side;
    if h == 0 && w == 0 {
        return Err(invalid(format!(
            "{height}x{width} would resize to 0 x 0 at {max_patches} patches"
        )));
    }
    let max_side = (max_patches / (GEMMA4_POOL * GEMMA4_POOL)) * side;
    if h == 0 {
        h = side;
        w = ((width / height) * side).min(max_side);
    } else if w == 0 {
        w = side;
        h = ((height / width) * side).min(max_side);
    }
    if (h * w) as f64 > target_px {
        return Err(invalid(format!(
            "resizing {height}x{width} to {h}x{w} exceeds {max_patches} patches"
        )));
    }
    Ok((h, w))
}

/// One image as tower input.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Patches {
    /// `[n_patches, 768]`, each row `[16 rows][16 cols][RGB]` in `[0, 1]`.
    pub pixels: Vec<f32>,
    /// `(x, y)` patch coordinates, row-major over the grid.
    pub positions: Vec<(usize, usize)>,
    pub grid_w: usize,
    pub grid_h: usize,
}

impl Gemma4Patches {
    /// Patchify an image whose sides are already multiples of 48 px.
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Result<Self, VisionIoError> {
        let side = GEMMA4_POOL * GEMMA4_PATCH;
        if width == 0 || height == 0 || width % side != 0 || height % side != 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("{width}x{height} is not a nonzero multiple of {side} px"),
            });
        }
        if rgb.len() != width * height * 3 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!(
                    "{} samples for a {width}x{height} RGB image (expected {})",
                    rgb.len(),
                    width * height * 3
                ),
            });
        }
        let (gw, gh) = (width / GEMMA4_PATCH, height / GEMMA4_PATCH);
        let mut pixels = Vec::with_capacity(gw * gh * GEMMA4_PATCH_DIM);
        let mut positions = Vec::with_capacity(gw * gh);
        for py in 0..gh {
            for px in 0..gw {
                for r in 0..GEMMA4_PATCH {
                    let row = (py * GEMMA4_PATCH + r) * width + px * GEMMA4_PATCH;
                    for c in 0..GEMMA4_PATCH {
                        for ch in 0..3 {
                            pixels.push(f32::from(rgb[(row + c) * 3 + ch]) / 255.0);
                        }
                    }
                }
                positions.push((px, py));
            }
        }
        Ok(Self {
            pixels,
            positions,
            grid_w: gw,
            grid_h: gh,
        })
    }

    pub fn num_patches(&self) -> usize {
        self.positions.len()
    }

    /// Soft token patch `i` pools into: `Gemma4VisionPooler`'s
    /// `x / 3 + (grid_w / 3) * (y / 3)`, so soft tokens are row-major over
    /// the 3x3 blocks.
    pub fn pool_index(&self, i: usize) -> usize {
        let (x, y) = self.positions[i];
        x / GEMMA4_POOL + (self.grid_w / GEMMA4_POOL) * (y / GEMMA4_POOL)
    }

    pub fn soft_tokens(&self) -> usize {
        self.positions.len() / (GEMMA4_POOL * GEMMA4_POOL)
    }
}

/// Resize and patchify one decoded image at a soft-token budget.
///
/// A budget outside [`GEMMA4_SOFT_TOKEN_BUDGETS`] is refused, as the
/// reference refuses it. An image already at its target size is not
/// resampled (mlx-vlm returns it untouched, and a same-size PIL resize would
/// be a copy anyway), so that case cannot pick up filter rounding.
pub fn gemma4_preprocess(
    image: &Rgb8Image,
    max_soft_tokens: usize,
) -> Result<Gemma4Patches, VisionIoError> {
    if !GEMMA4_SOFT_TOKEN_BUDGETS.contains(&max_soft_tokens) {
        return Err(VisionIoError::BadConfig {
            field: "max_soft_tokens".into(),
            why: format!("{max_soft_tokens} is not one of {GEMMA4_SOFT_TOKEN_BUDGETS:?}"),
        });
    }
    let max_patches = max_soft_tokens * GEMMA4_POOL * GEMMA4_POOL;
    let (h, w) = gemma4_resize_dims(image.height, image.width, max_patches)?;
    if (h, w) == (image.height, image.width) {
        return Gemma4Patches::from_rgb(&image.data, w, h);
    }
    let resized = resize_bicubic_pil(&image.data, image.width, image.height, w, h);
    Gemma4Patches::from_rgb(&resized, w, h)
}

/// The three special ids an image occupies in a Gemma 4 prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4ImageIds {
    /// `boi`, emitted before an image's placeholders.
    pub boi: i32,
    /// The placeholder, one per soft token; the tower's rows replace these.
    pub image: i32,
    /// `eoi`, emitted after.
    pub eoi: i32,
}

/// Soft tokens an image of `height x width` becomes at `max_soft_tokens`,
/// from the header alone (no decode), for a request budget check before
/// any pixel buffer exists.
pub fn gemma4_soft_tokens_for(
    height: usize,
    width: usize,
    max_soft_tokens: usize,
) -> Result<usize, VisionIoError> {
    let max_patches = max_soft_tokens * GEMMA4_POOL * GEMMA4_POOL;
    let (h, w) = gemma4_resize_dims(height, width, max_patches)?;
    Ok((h / GEMMA4_PATCH) * (w / GEMMA4_PATCH) / (GEMMA4_POOL * GEMMA4_POOL))
}

/// Expand a rendered prompt's image markers the way `Gemma4Processor` does.
///
/// The chat template renders ONE `<|image|>` per image part. The processor
/// replaces each with `boi`, `soft_tokens[i]` placeholders and `eoi`
/// (`replace_image_token`), and the tower's rows replace the placeholders.
/// Returns the expanded ids and each image's placeholder span, in order.
///
/// Refused rather than guessed: a marker count that differs from the image
/// count (the nth image must pair with the nth marker), a marker already
/// next to `boi`/`eoi` or another marker (a prompt expanded twice would
/// otherwise grow nested spans), and an empty image.
pub fn gemma4_expand_image_placeholders(
    rendered: &[i32],
    ids: Gemma4ImageIds,
    soft_tokens: &[usize],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    let markers: Vec<usize> = rendered
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == ids.image)
        .map(|(i, _)| i)
        .collect();
    if markers.len() != soft_tokens.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: markers.len(),
            grids: soft_tokens.len(),
        });
    }
    for &i in &markers {
        let neighbour = |j: Option<usize>| j.and_then(|j| rendered.get(j)).copied();
        for t in [neighbour(i.checked_sub(1)), neighbour(Some(i + 1))]
            .into_iter()
            .flatten()
        {
            if t == ids.boi || t == ids.eoi || t == ids.image {
                return Err(VisionIoError::InvalidDimensions {
                    detail: format!(
                        "the image marker at {i} is already next to an image token ({t}); \
                         this prompt looks expanded already"
                    ),
                });
            }
        }
    }
    let extra: usize = soft_tokens.iter().map(|n| n + 1).sum();
    let mut out = Vec::with_capacity(rendered.len() + extra);
    let mut spans = Vec::with_capacity(soft_tokens.len());
    let mut next = 0usize;
    for &t in rendered {
        if t != ids.image {
            out.push(t);
            continue;
        }
        let n = soft_tokens[next];
        if n == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("image {next} has no soft tokens"),
            });
        }
        out.push(ids.boi);
        spans.push(crate::ImageSpan {
            start: out.len(),
            len: n,
        });
        out.extend(std::iter::repeat_n(ids.image, n));
        out.push(ids.eoi);
        next += 1;
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_matches_transformers_examples() {
        // Worked by hand from get_aspect_ratio_preserving_size, budget 280
        // (2520 patches, 645120 px): a 640x480 photo has factor ~1.4491.
        assert_eq!(gemma4_resize_dims(480, 640, 2520).unwrap(), (672, 912));
        // A square at budget 70 (630 patches): 401.6 -> 384 per side.
        assert_eq!(gemma4_resize_dims(1000, 1000, 630).unwrap(), (384, 384));
        // A very wide strip rounds the short side to zero and clamps.
        assert_eq!(gemma4_resize_dims(10, 4000, 630).unwrap(), (48, 3360));
        for (h, w, p) in [
            (480, 640, 2520),
            (1000, 1000, 630),
            (10, 4000, 630),
            (333, 777, 1260),
        ] {
            let (rh, rw) = gemma4_resize_dims(h, w, p).unwrap();
            assert_eq!(rh % 48, 0);
            assert_eq!(rw % 48, 0);
            assert!((rh / 16) * (rw / 16) <= p);
        }
    }

    #[test]
    fn patches_are_row_major_and_pool_by_3x3_blocks() {
        // A 96x48 image: a 6x3 patch grid, two soft tokens side by side.
        let (w, h) = (96, 48);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| [(i % 251) as u8, 0, 255]).collect();
        let p = Gemma4Patches::from_rgb(&rgb, w, h).unwrap();
        assert_eq!((p.grid_w, p.grid_h), (6, 3));
        assert_eq!(p.positions[0], (0, 0));
        assert_eq!(p.positions[1], (1, 0));
        assert_eq!(p.positions[6], (0, 1));
        assert_eq!(p.soft_tokens(), 2);
        assert_eq!(p.pool_index(0), 0);
        assert_eq!(p.pool_index(2), 0);
        assert_eq!(p.pool_index(3), 1);
        assert_eq!(p.pool_index(17), 1);
        // Patch 1 starts at pixel (0, 16); channel order is RGB.
        let patch1 = &p.pixels[GEMMA4_PATCH_DIM..GEMMA4_PATCH_DIM + 3];
        assert_eq!(patch1, &[16.0 / 255.0, 0.0, 1.0]);
        assert!(Gemma4Patches::from_rgb(&rgb[..3 * 50 * 48], 50, 48).is_err());
    }

    const IDS: Gemma4ImageIds = Gemma4ImageIds {
        boi: 7,
        image: 9,
        eoi: 8,
    };

    #[test]
    fn markers_expand_to_boi_placeholders_eoi_with_their_spans() {
        let (out, spans) =
            gemma4_expand_image_placeholders(&[1, 9, 2, 3, 9, 4], IDS, &[3, 2]).unwrap();
        assert_eq!(out, vec![1, 7, 9, 9, 9, 8, 2, 3, 7, 9, 9, 8, 4]);
        assert_eq!(
            spans,
            vec![
                crate::ImageSpan { start: 2, len: 3 },
                crate::ImageSpan { start: 9, len: 2 }
            ]
        );
    }

    #[test]
    fn a_marker_count_mismatch_or_an_expanded_prompt_is_refused() {
        assert!(gemma4_expand_image_placeholders(&[1, 9, 2], IDS, &[3, 2]).is_err());
        assert!(gemma4_expand_image_placeholders(&[1, 7, 9, 8], IDS, &[3]).is_err());
        assert!(gemma4_expand_image_placeholders(&[1, 9, 9], IDS, &[1, 1]).is_err());
        assert!(gemma4_expand_image_placeholders(&[1, 9, 2], IDS, &[0]).is_err());
    }

    #[test]
    fn the_header_count_matches_the_preprocessed_count() {
        let img = Rgb8Image::new(640, 480, vec![0; 640 * 480 * 3]).unwrap();
        let p = gemma4_preprocess(&img, 280).unwrap();
        assert_eq!(
            gemma4_soft_tokens_for(480, 640, 280).unwrap(),
            p.soft_tokens()
        );
        // The real gate's image: 2,394 patches, 266 soft tokens.
        assert_eq!(gemma4_soft_tokens_for(480, 640, 280).unwrap(), 266);
    }

    #[test]
    fn an_unsupported_budget_is_refused() {
        let img = Rgb8Image::new(48, 48, vec![0; 48 * 48 * 3]).unwrap();
        assert!(gemma4_preprocess(&img, 100).is_err());
        let p = gemma4_preprocess(&img, 70).unwrap();
        assert!(p.num_patches() <= 630);
    }
}
