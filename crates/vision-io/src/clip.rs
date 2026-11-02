//! Preprocessing for the CLIP vision tower LLaVA-1.5 carries
//! (`docs/LLAVA_VISION_PHASE0.md`).
//!
//! The reference is `CLIPImageProcessor` as `llava-hf/llava-1.5-7b-hf` ships
//! it, and it is a different pipeline from the rest of this crate:
//!
//! - resize so the SHORTEST edge is `size` (the long edge is
//!   `int(size * long / short)`, truncated), PIL bicubic;
//! - center crop to `size x size` with `top = (h - size) / 2` and
//!   `left = (w - size) / 2`, both floored, so an image is cropped and never
//!   padded or stretched (the original LLaVA repo pads to a square with the
//!   mean color; the Hugging Face processor does not, and this port follows
//!   the processor the checkpoint publishes);
//! - rescale by `1/255`, then `(x - mean) / std` per channel;
//! - patches are emitted ROW-MAJOR over the grid, and one patch row is
//!   `(C, P_h, P_w)`: the layout `patch_embedding.weight` `[out, C, P, P]`
//!   flattens to, so the repack copies that weight verbatim and the GEMM reads
//!   it row-major. This is the opposite choice from the qwen tower's
//!   `(T, P, P, C)` and for the same reason: no permutation anywhere.
//!
//! There is no merge window, no temporal axis and no smart resize.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;

/// Pixels per patch side. CLIP ViT-L/14.
pub const CLIP_PATCH: usize = 14;
/// Values per patch row: `3 * 14 * 14`.
pub const CLIP_PATCH_DIM: usize = 3 * CLIP_PATCH * CLIP_PATCH;

/// What `preprocessor_config.json` says, narrowed to what this pipeline acts
/// on. Every field is read off the file and nothing here is a baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipPreprocessParams {
    /// Both the resize target of the shortest edge and the square crop side.
    pub size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
}

impl ClipPreprocessParams {
    /// Parse a CLIP `preprocessor_config.json`.
    ///
    /// Refuses everything this pipeline does not implement rather than
    /// ignoring it: `do_resize`, `do_center_crop`, `do_rescale` and
    /// `do_normalize` must be true, `resample` must be 3 (bicubic), and the
    /// crop must be square and equal to the shortest edge. A processor that
    /// differs would be a different image reaching the tower.
    pub fn from_preprocessor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let root: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionIoError::BadConfig {
                field: "<root>".into(),
                why: e.to_string(),
            })?;
        let bad = |field: &str, why: String| VisionIoError::BadConfig {
            field: field.to_string(),
            why,
        };
        for flag in ["do_resize", "do_center_crop", "do_rescale", "do_normalize"] {
            match root.get(flag).and_then(serde_json::Value::as_bool) {
                Some(true) => {}
                other => return Err(bad(flag, format!("{other:?}; this pipeline needs true"))),
            }
        }
        if root.get("resample").and_then(serde_json::Value::as_i64) != Some(3) {
            return Err(bad("resample", "must be 3 (PIL bicubic)".into()));
        }
        let shortest = root
            .get("size")
            .and_then(|s| s.get("shortest_edge"))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| bad("size.shortest_edge", "missing".into()))? as usize;
        let crop = |axis: &str| {
            root.get("crop_size")
                .and_then(|s| s.get(axis))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
        };
        if crop("height") != Some(shortest) || crop("width") != Some(shortest) {
            return Err(bad(
                "crop_size",
                format!(
                    "{:?} x {:?} against shortest_edge {shortest}; only a square crop equal to \
                     the shortest edge is implemented",
                    crop("height"),
                    crop("width")
                ),
            ));
        }
        if shortest == 0 || shortest % CLIP_PATCH != 0 {
            return Err(bad(
                "size.shortest_edge",
                format!("{shortest} is not a nonzero multiple of {CLIP_PATCH}"),
            ));
        }
        let triple = |field: &str| -> Result<[f32; 3], VisionIoError> {
            let a = root
                .get(field)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| bad(field, "missing".into()))?;
            if a.len() != 3 {
                return Err(bad(field, format!("{} values, expected 3", a.len())));
            }
            let mut out = [0f32; 3];
            for (slot, v) in out.iter_mut().zip(a) {
                *slot = v
                    .as_f64()
                    .ok_or_else(|| bad(field, "not a number".into()))? as f32;
            }
            Ok(out)
        };
        let rescale_factor = root
            .get("rescale_factor")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| bad("rescale_factor", "missing".into()))? as f32;
        Ok(Self {
            size: shortest,
            image_mean: triple("image_mean")?,
            image_std: triple("image_std")?,
            rescale_factor,
        })
    }

    /// Patches per side of the cropped square.
    pub fn grid(&self) -> usize {
        self.size / CLIP_PATCH
    }

    /// Patch rows the tower sees and image tokens the prompt carries
    /// (`grid^2`; the class row is added and dropped inside the tower).
    pub fn tokens(&self) -> usize {
        self.grid() * self.grid()
    }
}

/// One image as tower input.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipPatches {
    /// `[grid * grid, 588]`, row-major over the grid, each row `(C, 14, 14)`.
    pub pixels: Vec<f32>,
    /// Patches per side.
    pub grid: usize,
}

impl ClipPatches {
    pub fn num_patches(&self) -> usize {
        self.grid * self.grid
    }
}

/// `(new_w, new_h)` after the shortest-edge resize, the way
/// `get_resize_output_image_size` computes it: the short side is exactly
/// `size` and the long side is `int(size * long / short)`.
pub fn clip_resize_dims(width: usize, height: usize, size: usize) -> (usize, usize) {
    if width <= height {
        (size, size * height / width)
    } else {
        (size * width / height, size)
    }
}

/// Resize, center-crop, normalize and patchify one decoded image.
pub fn clip_preprocess(
    image: &Rgb8Image,
    params: &ClipPreprocessParams,
) -> Result<ClipPatches, VisionIoError> {
    if image.width == 0 || image.height == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    let size = params.size;
    let (nw, nh) = clip_resize_dims(image.width, image.height, size);
    let resized;
    let (data, w, h): (&[u8], usize, usize) = if (nw, nh) == (image.width, image.height) {
        (&image.data, nw, nh)
    } else {
        resized = resize_bicubic_pil(&image.data, image.width, image.height, nw, nh);
        (&resized, nw, nh)
    };
    let top = (h - size) / 2;
    let left = (w - size) / 2;

    let grid = size / CLIP_PATCH;
    let mut pixels = Vec::with_capacity(grid * grid * CLIP_PATCH_DIM);
    for py in 0..grid {
        for px in 0..grid {
            for c in 0..3 {
                for r in 0..CLIP_PATCH {
                    let y = top + py * CLIP_PATCH + r;
                    for k in 0..CLIP_PATCH {
                        let x = left + px * CLIP_PATCH + k;
                        let sample = data[(y * w + x) * 3 + c];
                        pixels.push(
                            (sample as f32 * params.rescale_factor - params.image_mean[c])
                                / params.image_std[c],
                        );
                    }
                }
            }
        }
    }
    Ok(ClipPatches { pixels, grid })
}

/// Expand a rendered prompt's image markers: each `marker` id becomes
/// `counts[i]` copies of `fill`, and the span of each run is returned.
///
/// `fill` is the marker itself whenever the trunk's embedding table has a row
/// for it, and an in-vocabulary stand-in otherwise. LLaVA's `<image>` is id
/// 32000, a row only the 32,064-row Hugging Face checkpoint has: the GGUF
/// trunks carry 32,000, and the trunk refuses an out-of-range id BEFORE it
/// reaches the embedding blit that would have replaced the lookup anyway. The
/// run's content is never read as a token, so the stand-in costs nothing.
///
/// LLaVA has no start or end token around an image, so the run is bare. The
/// template renders ONE `<image>` per image part; the processor replaces it
/// with `image_token * num_image_tokens`, and the tower's rows replace those
/// positions. Refused rather than guessed: a marker count that differs from
/// the image count, two markers already adjacent (an expanded prompt would
/// otherwise grow a run), and an empty image.
pub fn clip_expand_image_placeholders(
    rendered: &[i32],
    marker: i32,
    fill: i32,
    counts: &[usize],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    let markers: Vec<usize> = rendered
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == marker)
        .map(|(i, _)| i)
        .collect();
    if markers.len() != counts.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: markers.len(),
            grids: counts.len(),
        });
    }
    if markers.windows(2).any(|w| w[1] == w[0] + 1) {
        return Err(VisionIoError::InvalidDimensions {
            detail: "two image markers are adjacent; this prompt looks expanded already".into(),
        });
    }
    let extra: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(rendered.len() + extra);
    let mut spans = Vec::with_capacity(counts.len());
    let mut next = 0usize;
    for &t in rendered {
        if t != marker {
            out.push(t);
            continue;
        }
        let n = counts[next];
        if n == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("image {next} has no tokens"),
            });
        }
        spans.push(crate::ImageSpan {
            start: out.len(),
            len: n,
        });
        out.extend(std::iter::repeat_n(fill, n));
        next += 1;
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
      "crop_size": {"height": 336, "width": 336},
      "do_center_crop": true, "do_convert_rgb": true, "do_normalize": true,
      "do_rescale": true, "do_resize": true,
      "image_mean": [0.48145466, 0.4578275, 0.40821073],
      "image_std": [0.26862954, 0.26130258, 0.27577711],
      "resample": 3, "rescale_factor": 0.00392156862745098,
      "size": {"shortest_edge": 336}
    }"#;

    fn params() -> ClipPreprocessParams {
        ClipPreprocessParams::from_preprocessor_config_json(CONFIG).unwrap()
    }

    #[test]
    fn the_published_config_parses_to_a_24_grid() {
        let p = params();
        assert_eq!(p.size, 336);
        assert_eq!(p.grid(), 24);
        assert_eq!(p.tokens(), 576);
    }

    #[test]
    fn a_config_this_pipeline_does_not_implement_is_refused() {
        for (needle, with) in [
            ("\"do_center_crop\": true", "\"do_center_crop\": false"),
            ("\"resample\": 3", "\"resample\": 2"),
            ("\"height\": 336", "\"height\": 224"),
        ] {
            assert!(
                ClipPreprocessParams::from_preprocessor_config_json(
                    &CONFIG.replace(needle, with)
                )
                .is_err(),
                "{with} must be refused"
            );
        }
    }

    #[test]
    fn the_short_edge_is_exact_and_the_long_edge_truncates() {
        assert_eq!(clip_resize_dims(1024, 1280, 336), (336, 420));
        assert_eq!(clip_resize_dims(1280, 1024, 336), (420, 336));
        assert_eq!(clip_resize_dims(500, 333, 336), (504, 336));
        assert_eq!(clip_resize_dims(336, 336, 336), (336, 336));
    }

    #[test]
    fn patches_are_row_major_with_channel_major_rows() {
        // A 336x336 image whose R channel is the column and G the row, so the
        // order inside one patch row is checkable by value.
        let mut data = Vec::with_capacity(336 * 336 * 3);
        for y in 0..336usize {
            for x in 0..336usize {
                data.extend_from_slice(&[(x % 251) as u8, (y % 251) as u8, 7]);
            }
        }
        let img = Rgb8Image::new(336, 336, data).unwrap();
        let mut p = params();
        p.image_mean = [0.0; 3];
        p.image_std = [1.0; 3];
        p.rescale_factor = 1.0;
        let out = clip_preprocess(&img, &p).unwrap();
        assert_eq!(out.grid, 24);
        assert_eq!(out.pixels.len(), 576 * CLIP_PATCH_DIM);
        // Patch (py=1, px=2): channel 0 is the first 196 values, R = column.
        let patch = &out.pixels[(24 + 2) * CLIP_PATCH_DIM..(24 + 3) * CLIP_PATCH_DIM];
        assert_eq!(patch[0], 28.0); // column 2 * 14, first row
        assert_eq!(patch[1], 29.0); // next column
        assert_eq!(patch[14], 28.0); // second row, same column
        // Channel 1 starts at 196: G = row 1 * 14.
        assert_eq!(patch[196], 14.0);
        assert_eq!(patch[196 + 14], 15.0);
        // Channel 2 is the constant.
        assert_eq!(patch[2 * 196], 7.0);
    }

    #[test]
    fn a_wide_image_is_center_cropped_not_squeezed() {
        // 672x336 resizes to itself (short edge already 336) and crops the
        // middle 336 columns: left = (672 - 336) / 2 = 168.
        let mut data = Vec::new();
        for _y in 0..336usize {
            for x in 0..672usize {
                data.extend_from_slice(&[(x / 3) as u8, 0, 0]);
            }
        }
        let img = Rgb8Image::new(672, 336, data).unwrap();
        let mut p = params();
        p.image_mean = [0.0; 3];
        p.image_std = [1.0; 3];
        p.rescale_factor = 1.0;
        let out = clip_preprocess(&img, &p).unwrap();
        assert_eq!(out.pixels[0], (168 / 3) as f32);
    }

    #[test]
    fn markers_expand_to_bare_runs_with_their_spans() {
        let (out, spans) = clip_expand_image_placeholders(&[1, 9, 2, 9, 4], 9, 9, &[3, 2]).unwrap();
        assert_eq!(out, vec![1, 9, 9, 9, 2, 9, 9, 4]);
        assert_eq!(
            spans,
            vec![
                crate::ImageSpan { start: 1, len: 3 },
                crate::ImageSpan { start: 5, len: 2 }
            ]
        );
    }

    #[test]
    fn the_run_carries_the_fill_id_not_the_marker() {
        let (out, spans) = clip_expand_image_placeholders(&[1, 32000, 2], 32000, 0, &[3]).unwrap();
        assert_eq!(out, vec![1, 0, 0, 0, 2]);
        assert_eq!(spans, vec![crate::ImageSpan { start: 1, len: 3 }]);
    }

    #[test]
    fn a_count_mismatch_an_expanded_prompt_or_an_empty_image_is_refused() {
        assert!(clip_expand_image_placeholders(&[1, 9, 2], 9, 9, &[3, 2]).is_err());
        assert!(clip_expand_image_placeholders(&[1, 9, 9], 9, 9, &[1, 1]).is_err());
        assert!(clip_expand_image_placeholders(&[1, 9, 2], 9, 9, &[0]).is_err());
    }
}
