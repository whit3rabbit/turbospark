//! Preprocessing, rope rows and prompt layout for the Pixtral vision tower
//! (`docs/PIXTRAL_VISION_PHASE0.md`).
//!
//! The reference is mlx-vlm's `PixtralImageProcessor` / `PixtralProcessor`
//! (read, never executed) over `mistral-community/pixtral-12b`'s processor
//! config. It is a third pipeline in this crate, unlike the Qwen smart resize
//! and unlike CLIP's crop:
//!
//! - the image is NEVER cropped or padded to a square. If the longer side is
//!   over `longest_edge` (1024) the image is scaled down by
//!   `ratio = max(h / 1024, w / 1024)` with each side floored, and then each
//!   side is rounded UP to a multiple of the patch (16) by RESIZING to that
//!   size (a slight stretch, PIL bicubic), so a 1000x1000 image is resized to
//!   1008x1008 and the patch grid has no ragged edge;
//! - rescale by `1/255`, then `(x - mean) / std` with the OpenAI CLIP
//!   constants;
//! - patches are emitted row-major over a `grid_h x grid_w` grid and one patch
//!   row is `(C, P_h, P_w)`, the layout `patch_conv.weight` `[out, C, 16, 16]`
//!   flattens to (the CLIP choice, for the same reason).
//!
//! The prompt layout is not a bare run either: each patch ROW becomes
//! `[IMG] * grid_w`, rows are separated by `[IMG_BREAK]`, and the last row is
//! followed by `[IMG_END]`. The tower's rows replace only the `[IMG]`
//! positions, so every row is its own injection span and the break and end
//! tokens keep their ordinary embeddings.

use crate::clip::ClipPatches;
use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;

/// Pixels per patch side.
pub const PIXTRAL_PATCH: usize = 16;
/// Values per patch row: `3 * 16 * 16`.
pub const PIXTRAL_PATCH_DIM: usize = 3 * PIXTRAL_PATCH * PIXTRAL_PATCH;

/// `get_resize_output_image_size`: scale down so neither side passes
/// `longest_edge`, flooring each side, then round each side UP to a whole
/// number of patches. Returns `(height, width)`.
pub fn pixtral_resize_target(
    height: usize,
    width: usize,
    longest_edge: usize,
    patch: usize,
) -> (usize, usize) {
    let (mut h, mut w) = (height as f64, width as f64);
    let ratio = (h / longest_edge as f64).max(w / longest_edge as f64);
    if ratio > 1.0 {
        h = (h / ratio).floor();
        w = (w / ratio).floor();
    }
    let up = |side: f64| -> usize {
        let side = side as usize;
        // `(side - 1) // patch + 1`, with a floor of one patch.
        (side.saturating_sub(1) / patch + 1) * patch
    };
    (up(h), up(w))
}

/// Resize, normalize and patchify one decoded image.
pub fn pixtral_preprocess(
    image: &Rgb8Image,
    longest_edge: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    rescale_factor: f32,
) -> Result<ClipPatches, VisionIoError> {
    if image.width == 0 || image.height == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    let (th, tw) = pixtral_resize_target(image.height, image.width, longest_edge, PIXTRAL_PATCH);
    let resized;
    let data: &[u8] = if (tw, th) == (image.width, image.height) {
        &image.data
    } else {
        resized = resize_bicubic_pil(&image.data, image.width, image.height, tw, th);
        &resized
    };
    let (gh, gw) = (th / PIXTRAL_PATCH, tw / PIXTRAL_PATCH);
    let mut pixels = Vec::with_capacity(gh * gw * PIXTRAL_PATCH_DIM);
    for py in 0..gh {
        for px in 0..gw {
            for c in 0..3 {
                for r in 0..PIXTRAL_PATCH {
                    let y = py * PIXTRAL_PATCH + r;
                    for k in 0..PIXTRAL_PATCH {
                        let x = px * PIXTRAL_PATCH + k;
                        let sample = data[(y * tw + x) * 3 + c];
                        pixels
                            .push((sample as f32 * rescale_factor - image_mean[c]) / image_std[c]);
                    }
                }
            }
        }
    }
    Ok(ClipPatches {
        pixels,
        grid: gh,
        grid_w: gw,
        anyres: None,
    })
}

/// The tower's rope angle rows, `[grid_h * grid_w, head_dim / 2]` F32,
/// row-major over the grid.
///
/// `PixtralRotaryEmbedding` builds `freqs[j] = theta^(-2j / head_dim)` for
/// `j in 0..head_dim/2`, gives the HEIGHT the even entries and the WIDTH the
/// odd ones (`freqs[::2]` and `freqs[1::2]`), lays the two halves side by side
/// and duplicates them over the head. This is the first half of that: the
/// kernel's NeoX half-split pairs element `i` with `i + head_dim / 2`, which
/// is the duplication. So a row is `[row * freqs[0], row * freqs[2], ...
/// (head_dim/4 of them) | col * freqs[1], col * freqs[3], ...]`.
///
/// The even/odd assignment is Pixtral's own: Qwen gives each axis a
/// contiguous run of the same frequency ladder, and Gemma's axial rope rotates
/// each half within itself.
pub fn pixtral_rope_freq_rows(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    theta: f32,
) -> Result<Vec<f32>, VisionIoError> {
    if head_dim == 0 || head_dim % 4 != 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("pixtral head_dim {head_dim} must be a positive multiple of 4"),
        });
    }
    // theta is the base of the frequency ladder: zero, negative or non-finite
    // gives inf or NaN rows the tower would rotate by without any error.
    if !(theta.is_finite() && theta > 0.0) {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("pixtral rope theta {theta} must be finite and positive"),
        });
    }
    let quarter = head_dim / 4;
    let freq = |j: usize| (theta as f64).powf(-2.0 * j as f64 / head_dim as f64);
    let mut out = Vec::with_capacity(grid_h * grid_w * (head_dim / 2));
    for r in 0..grid_h {
        for c in 0..grid_w {
            for k in 0..quarter {
                out.push((r as f64 * freq(2 * k)) as f32);
            }
            for k in 0..quarter {
                out.push((c as f64 * freq(2 * k + 1)) as f32);
            }
        }
    }
    Ok(out)
}

/// Expand a rendered prompt's image markers into Pixtral's row layout.
///
/// Each `marker` becomes, for its image's `(grid_h, grid_w)`: `grid_w` copies
/// of the marker per row, an `image_break` between rows (not after the last),
/// and one `image_end`. One span is returned per ROW, in order, because the
/// break and end tokens between rows are real tokens that keep their own
/// embeddings. The marker count must equal the image count and no two markers
/// may be adjacent (an already-expanded prompt), as for CLIP.
pub fn pixtral_expand_image_placeholders(
    rendered: &[i32],
    marker: i32,
    image_break: i32,
    image_end: i32,
    grids: &[(usize, usize)],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    let markers: Vec<usize> = rendered
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == marker)
        .map(|(i, _)| i)
        .collect();
    if markers.len() != grids.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: markers.len(),
            grids: grids.len(),
        });
    }
    if markers.windows(2).any(|w| w[1] == w[0] + 1) {
        return Err(VisionIoError::InvalidDimensions {
            detail: "two image markers are adjacent; this prompt looks expanded already".into(),
        });
    }
    let extra: usize = grids.iter().map(|&(h, w)| h * w + h).sum();
    let mut out = Vec::with_capacity(rendered.len() + extra);
    let mut spans = Vec::new();
    let mut next = 0usize;
    for &t in rendered {
        if t != marker {
            out.push(t);
            continue;
        }
        let (gh, gw) = grids[next];
        if gh == 0 || gw == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("image {next} has no tokens"),
            });
        }
        for row in 0..gh {
            spans.push(crate::ImageSpan {
                start: out.len(),
                len: gw,
            });
            out.extend(std::iter::repeat_n(marker, gw));
            if row + 1 < gh {
                out.push(image_break);
            }
        }
        out.push(image_end);
        next += 1;
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resize_target_scales_down_then_rounds_each_side_up() {
        // Over the cap: 2000 wide x 1000 high scales by 1.953125 to 1024 x 512,
        // already whole patches.
        assert_eq!(pixtral_resize_target(1000, 2000, 1024, 16), (512, 1024));
        // Over the cap and off the patch afterwards: 1500 x 1001 -> 1024 x 683
        // -> 1024 x 688.
        assert_eq!(pixtral_resize_target(1001, 1500, 1024, 16), (688, 1024));
        // Exactly on the cap and on the patch: unchanged.
        assert_eq!(pixtral_resize_target(1024, 1024, 1024, 16), (1024, 1024));
        // Under the cap, off the patch: rounded up, never down.
        assert_eq!(pixtral_resize_target(480, 640, 1024, 16), (480, 640));
        assert_eq!(pixtral_resize_target(481, 641, 1024, 16), (496, 656));
        // A tiny image still gets a whole patch.
        assert_eq!(pixtral_resize_target(1, 1, 1024, 16), (16, 16));
        // A square just over the cap: 1100 scales to floor(1100 / 1.074..) = 1024.
        assert_eq!(pixtral_resize_target(1100, 1100, 1024, 16), (1024, 1024));
    }

    #[test]
    fn a_patch_row_is_channel_major_and_the_grid_is_row_major() {
        // 32x16 image: two patches side by side, R = x, G = 100, B = y.
        let mut data = Vec::new();
        for y in 0..16u8 {
            for x in 0..32u8 {
                data.extend([x, 100, y]);
            }
        }
        let image = Rgb8Image {
            data,
            width: 32,
            height: 16,
        };
        let p = pixtral_preprocess(&image, 1024, [0.0; 3], [1.0; 3], 1.0).unwrap();
        assert_eq!((p.grid, p.grid_w), (1, 2));
        assert_eq!(p.pixels.len(), 2 * PIXTRAL_PATCH_DIM);
        // Patch 0, channel 0 (R), row 0: x = 0..16. Channel 1 starts at 256.
        assert_eq!(p.pixels[0], 0.0);
        assert_eq!(p.pixels[15], 15.0);
        assert_eq!(p.pixels[256], 100.0);
        // Channel 2 (B) row 3 of patch 0 is y = 3.
        assert_eq!(p.pixels[512 + 3 * 16], 3.0);
        // Patch 1 starts at x = 16.
        assert_eq!(p.pixels[PIXTRAL_PATCH_DIM], 16.0);
    }

    #[test]
    fn rope_rows_give_height_the_even_frequencies_and_width_the_odd_ones() {
        let rows = pixtral_rope_freq_rows(2, 3, 64, 10_000.0).unwrap();
        assert_eq!(rows.len(), 6 * 32);
        let row = |r: usize, c: usize| &rows[(r * 3 + c) * 32..(r * 3 + c + 1) * 32];
        // Position (0, 0) has no angle anywhere.
        assert!(row(0, 0).iter().all(|&v| v == 0.0));
        // Position (1, 0): only the height half is nonzero; entry 0 is
        // freqs[0] = 1, entry 1 is freqs[2] = theta^(-4/64).
        let a = row(1, 0);
        assert!((a[0] - 1.0).abs() < 1e-7);
        assert!((a[1] - 10_000f32.powf(-4.0 / 64.0)).abs() < 1e-6);
        assert!(a[16..].iter().all(|&v| v == 0.0));
        // Position (0, 2): only the width half; entry 16 is 2 * freqs[1].
        let b = row(0, 2);
        assert!(b[..16].iter().all(|&v| v == 0.0));
        assert!((b[16] - 2.0 * 10_000f32.powf(-2.0 / 64.0)).abs() < 1e-6);
        assert!(pixtral_rope_freq_rows(1, 1, 6, 1.0).is_err());
    }

    #[test]
    fn rows_are_separated_by_breaks_and_the_run_ends_with_an_end_token() {
        // ids: BOS, [INST], <marker>, [/INST]
        let rendered = [1, 3, 10, 4];
        let (ids, spans) =
            pixtral_expand_image_placeholders(&rendered, 10, 12, 13, &[(2, 3)]).unwrap();
        assert_eq!(ids, vec![1, 3, 10, 10, 10, 12, 10, 10, 10, 13, 4]);
        assert_eq!(
            spans,
            vec![
                crate::ImageSpan { start: 2, len: 3 },
                crate::ImageSpan { start: 6, len: 3 }
            ]
        );
        // No break after the last row, one end token.
        let (ids, _) = pixtral_expand_image_placeholders(&[10], 10, 12, 13, &[(1, 2)]).unwrap();
        assert_eq!(ids, vec![10, 10, 13]);
    }

    #[test]
    fn a_marker_count_that_disagrees_with_the_images_is_refused() {
        assert!(pixtral_expand_image_placeholders(&[10, 4, 10], 10, 12, 13, &[(1, 1)]).is_err());
        assert!(pixtral_expand_image_placeholders(&[4], 10, 12, 13, &[(1, 1)]).is_err());
        assert!(
            pixtral_expand_image_placeholders(&[10, 10], 10, 12, 13, &[(1, 1), (1, 1)]).is_err()
        );
        assert!(pixtral_expand_image_placeholders(&[10], 10, 12, 13, &[(0, 4)]).is_err());
    }
}
