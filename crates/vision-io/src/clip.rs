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
    /// Both the resize target of the shortest edge and the square crop side
    /// ([`ClipLayout::Crop`]), or the longest-edge cap ([`ClipLayout::Pixtral`]).
    pub size: usize,
    /// Which pipeline these numbers drive.
    pub layout: ClipLayout,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
    /// `image_grid_pinpoints` as `(height, width)` ([`ClipLayout::OneVision`]
    /// only; empty for the other layouts).
    pub pinpoints: Vec<(usize, usize)>,
}

/// Which image pipeline a flat-injection tower uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipLayout {
    /// LLaVA-1.5: shortest-edge resize and a square center crop to `size`.
    Crop,
    /// Pixtral: a longest-edge cap, no crop, sides rounded up to a patch
    /// (`crate::pixtral`).
    Pixtral,
    /// LLaVA-OneVision: a squashed base tile plus an anyres grid of 384-pixel
    /// tiles (`crate::onevision`). `size` is the tile.
    OneVision,
}

impl ClipPreprocessParams {
    /// Parse a CLIP or Pixtral `preprocessor_config.json`.
    ///
    /// A config whose `image_processor_type` is `PixtralImageProcessor` takes
    /// the Pixtral path ([`ClipLayout::Pixtral`]); anything else is read as a
    /// CLIP crop config.
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
        if root
            .get("image_processor_type")
            .and_then(serde_json::Value::as_str)
            == Some("PixtralImageProcessor")
        {
            return Self::from_pixtral_config(&root);
        }
        if root
            .get("image_processor_type")
            .and_then(serde_json::Value::as_str)
            == Some("LlavaOnevisionImageProcessor")
        {
            return Self::from_onevision_config(&root);
        }
        for flag in ["do_resize", "do_center_crop", "do_rescale", "do_normalize"] {
            match root.get(flag).and_then(serde_json::Value::as_bool) {
                Some(true) => {}
                other => return Err(bad(flag, format!("{other:?}; this pipeline needs true"))),
            }
        }
        if root.get("resample").and_then(serde_json::Value::as_i64) != Some(3) {
            return Err(bad("resample", "must be 3 (PIL bicubic)".into()));
        }
        let shortest =
            root.get("size")
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
        // The crop is what every image is resized to, so it is an
        // allocation the config controls.
        crate::params::check_config_edge("size.shortest_edge", shortest)?;
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
                    .ok_or_else(|| bad(field, "not a number".into()))?
                    as f32;
            }
            Ok(out)
        };
        let rescale_factor =
            root.get("rescale_factor")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| bad("rescale_factor", "missing".into()))? as f32;
        Ok(Self {
            size: shortest,
            layout: ClipLayout::Crop,
            image_mean: triple("image_mean")?,
            image_std: positive_std(triple("image_std")?)?,
            rescale_factor,
            pinpoints: Vec::new(),
        })
    }

    /// Patches per side of the cropped square (the Pixtral cap, for that
    /// layout: the largest side an image can have).
    pub fn grid(&self) -> usize {
        match self.layout {
            ClipLayout::Crop => self.size / CLIP_PATCH,
            ClipLayout::Pixtral => self.size / crate::pixtral::PIXTRAL_PATCH,
            ClipLayout::OneVision => self.size / crate::onevision::OV_PATCH,
        }
    }

    /// Patch rows the tower sees and image tokens the prompt carries for a
    /// CROP image (`grid^2`; the class row is added and dropped inside the
    /// tower). For Pixtral this is the WORST case, a square at the cap: the
    /// real count is per image, [`ClipPatches::num_patches`].
    pub fn tokens(&self) -> usize {
        match self.layout {
            // The worst case: the base tile plus the biggest pinpoint's grid.
            ClipLayout::OneVision => {
                crate::onevision::ov_max_tiles(&self.pinpoints) * self.grid() * self.grid()
            }
            _ => self.grid() * self.grid(),
        }
    }

    /// The patch grid `(rows, cols)` an image of `width x height` pixels
    /// becomes: the fixed crop for CLIP, the longest-edge rule for Pixtral.
    pub fn grid_for(&self, width: usize, height: usize) -> (usize, usize) {
        match self.layout {
            ClipLayout::Crop => (self.grid(), self.grid()),
            ClipLayout::Pixtral => {
                let p = crate::pixtral::PIXTRAL_PATCH;
                let (h, w) = crate::pixtral::pixtral_resize_target(height, width, self.size, p);
                (h / p, w / p)
            }
            // The placeholder run as a degenerate `(tokens, 1)` grid: the
            // prompt layer only multiplies the pair. A zero-sized image gives
            // zero tokens, which the expansion refuses by name.
            ClipLayout::OneVision => (
                crate::onevision::ov_image_tokens(height, width, &self.pinpoints).unwrap_or(0),
                1,
            ),
        }
    }

    /// Cross-check these numbers against the install's own tower for what
    /// must agree for the tower to read the rows at all: the patch edge and
    /// the position-table size (a CLIP crop has `grid^2 + 1` rows, a Pixtral
    /// tower none). A config describing a different checkpoint would
    /// otherwise fail at a shape check inside the tower, far from the file
    /// that caused it.
    pub fn check_against_tower(
        &self,
        patch_size: i64,
        num_position_embeddings: i64,
    ) -> Result<(), String> {
        let (want_patch, want_rows) = match self.layout {
            ClipLayout::Crop => (CLIP_PATCH as i64, self.tokens() as i64 + 1),
            ClipLayout::Pixtral => (crate::pixtral::PIXTRAL_PATCH as i64, 0),
            // No class row: one position per patch of a tile.
            ClipLayout::OneVision => (
                crate::onevision::OV_PATCH as i64,
                crate::onevision::OV_TILE_PATCHES as i64,
            ),
        };
        if patch_size != want_patch || num_position_embeddings != want_rows {
            return Err(format!(
                "a {}-pixel {} is patch {want_patch} with a {want_rows}-row position table, but \
                 this install's tower has patch {patch_size} and a {num_position_embeddings}-row \
                 one; the two describe different checkpoints",
                self.size,
                match self.layout {
                    ClipLayout::Crop => "crop",
                    ClipLayout::Pixtral => "longest-edge cap",
                    ClipLayout::OneVision => "tile",
                }
            ));
        }
        Ok(())
    }

    /// Values in one patch row.
    pub fn patch_dim(&self) -> usize {
        match self.layout {
            ClipLayout::Crop => CLIP_PATCH_DIM,
            ClipLayout::Pixtral => crate::pixtral::PIXTRAL_PATCH_DIM,
            ClipLayout::OneVision => crate::onevision::OV_PATCH_DIM,
        }
    }

    /// LLaVA-OneVision's `preprocessor_config.json`. Refuses what this pipeline
    /// does not implement: resize, rescale, normalize and pad must be on, the
    /// resample must be 3 (PIL bicubic), the tile must be 384 square and every
    /// pinpoint a whole number of tiles.
    fn from_onevision_config(root: &serde_json::Value) -> Result<Self, VisionIoError> {
        let bad = |field: &str, why: String| VisionIoError::BadConfig {
            field: field.to_string(),
            why,
        };
        for flag in ["do_resize", "do_rescale", "do_normalize", "do_pad"] {
            match root.get(flag).and_then(serde_json::Value::as_bool) {
                Some(true) => {}
                other => return Err(bad(flag, format!("{other:?}; this pipeline needs true"))),
            }
        }
        if root.get("resample").and_then(serde_json::Value::as_i64) != Some(3) {
            return Err(bad("resample", "must be 3 (PIL bicubic)".into()));
        }
        let side = |axis: &str| {
            root.get("size")
                .and_then(|s| s.get(axis))
                .and_then(serde_json::Value::as_u64)
        };
        let tile = crate::onevision::OV_TILE as u64;
        if side("height") != Some(tile) || side("width") != Some(tile) {
            return Err(bad(
                "size",
                format!(
                    "{:?} x {:?}; only the {tile}-pixel square tile is implemented",
                    side("height"),
                    side("width")
                ),
            ));
        }
        let pinpoints = crate::onevision::parse_pinpoints(root)?;
        if let Some(&(h, w)) = pinpoints.iter().find(|&&(h, w)| {
            h % crate::onevision::OV_TILE != 0 || w % crate::onevision::OV_TILE != 0
        }) {
            return Err(bad(
                "image_grid_pinpoints",
                format!("{h}x{w} is not a whole number of tiles"),
            ));
        }
        let triple = |field: &str| -> Result<[f32; 3], VisionIoError> {
            let a = root
                .get(field)
                .and_then(serde_json::Value::as_array)
                .filter(|a| a.len() == 3)
                .ok_or_else(|| bad(field, "missing or not three numbers".into()))?;
            let mut out = [0f32; 3];
            for (slot, v) in out.iter_mut().zip(a) {
                *slot = v
                    .as_f64()
                    .ok_or_else(|| bad(field, "not a number".into()))?
                    as f32;
            }
            Ok(out)
        };
        let rescale_factor =
            root.get("rescale_factor")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| bad("rescale_factor", "missing".into()))? as f32;
        Ok(Self {
            size: crate::onevision::OV_TILE,
            layout: ClipLayout::OneVision,
            image_mean: triple("image_mean")?,
            image_std: positive_std(triple("image_std")?)?,
            rescale_factor,
            pinpoints,
        })
    }

    /// Pixtral's `preprocessor_config.json`. Refuses what this pipeline does
    /// not implement: resize, rescale and normalize must be on, the resample
    /// must be 3 (PIL bicubic), the patch must be the tower's 16 and the size
    /// a `longest_edge` that is a whole number of patches.
    fn from_pixtral_config(root: &serde_json::Value) -> Result<Self, VisionIoError> {
        let bad = |field: &str, why: String| VisionIoError::BadConfig {
            field: field.to_string(),
            why,
        };
        for flag in ["do_resize", "do_rescale", "do_normalize"] {
            match root.get(flag).and_then(serde_json::Value::as_bool) {
                Some(true) => {}
                other => return Err(bad(flag, format!("{other:?}; this pipeline needs true"))),
            }
        }
        if root.get("resample").and_then(serde_json::Value::as_i64) != Some(3) {
            return Err(bad("resample", "must be 3 (PIL bicubic)".into()));
        }
        let patch = |axis: &str| {
            root.get("patch_size")
                .and_then(|p| p.get(axis))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
        };
        let want = crate::pixtral::PIXTRAL_PATCH;
        if patch("height") != Some(want) || patch("width") != Some(want) {
            return Err(bad(
                "patch_size",
                format!(
                    "{:?} x {:?}; the implemented tower is {want} x {want}",
                    patch("height"),
                    patch("width")
                ),
            ));
        }
        let longest =
            root.get("size")
                .and_then(|s| s.get("longest_edge"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| bad("size.longest_edge", "missing".into()))? as usize;
        // Only a downscale cap, but `tokens()` squares it for the worst-case
        // token count, which would wrap.
        crate::params::check_config_edge("size.longest_edge", longest)?;
        if longest == 0 || longest % want != 0 {
            return Err(bad(
                "size.longest_edge",
                format!("{longest} is not a nonzero multiple of {want}"),
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
                    .ok_or_else(|| bad(field, "not a number".into()))?
                    as f32;
            }
            Ok(out)
        };
        let rescale_factor =
            root.get("rescale_factor")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| bad("rescale_factor", "missing".into()))? as f32;
        Ok(Self {
            size: longest,
            layout: ClipLayout::Pixtral,
            image_mean: triple("image_mean")?,
            image_std: positive_std(triple("image_std")?)?,
            rescale_factor,
            pinpoints: Vec::new(),
        })
    }
}

/// The normalization divides by `image_std`, so a zero (or a value that
/// rounds to zero in f32) would make every pixel inf or NaN with no error.
/// Same rule as `DeepseekOcrParams`.
fn positive_std(std: [f32; 3]) -> Result<[f32; 3], VisionIoError> {
    if std.iter().any(|s| *s <= 0.0) {
        return Err(VisionIoError::BadConfig {
            field: "image_std".into(),
            why: "entries must be positive".into(),
        });
    }
    Ok(std)
}

/// One image as tower input.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipPatches {
    /// `[grid * grid_w, patch_dim]`, row-major over the grid, each row
    /// `(C, P, P)` (588 values for CLIP, 768 for Pixtral).
    pub pixels: Vec<f32>,
    /// Patch rows of the grid (its HEIGHT). The CLIP crop is square, so this
    /// is also its width.
    pub grid: usize,
    /// Patches per row (the grid's WIDTH).
    pub grid_w: usize,
    /// The anyres tiling, for [`ClipLayout::OneVision`] only. Then `pixels`
    /// holds the base tile and the grid's tiles (729 rows each), and `grid`
    /// and `grid_w` are the prompt's placeholder count as `(tokens, 1)`.
    pub anyres: Option<crate::onevision::Anyres>,
}

impl ClipPatches {
    pub fn num_patches(&self) -> usize {
        self.grid * self.grid_w
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
    if params.layout == ClipLayout::Pixtral {
        return crate::pixtral::pixtral_preprocess(
            image,
            params.size,
            params.image_mean,
            params.image_std,
            params.rescale_factor,
        );
    }
    if params.layout == ClipLayout::OneVision {
        return crate::onevision::ov_preprocess(
            image,
            &params.pinpoints,
            params.image_mean,
            params.image_std,
            params.rescale_factor,
        );
    }
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
    Ok(ClipPatches {
        pixels,
        grid,
        grid_w: grid,
        anyres: None,
    })
}

/// Open a rendered LLaVA prompt with the BOS token when the template did not.
///
/// The Hugging Face processor tokenizes the rendered text with
/// `add_special_tokens`, so the reference prefills `<s>` first. The llava-hf
/// chat template does not emit `<s>` itself, and this port's convention is
/// that the TEMPLATE carries the BOS (every call site encodes with
/// `add_bos = false`), so without this the prompt runs BOS-less. Vicuna was
/// trained with the BOS and degrades without it, quietly. `None` (a tokenizer
/// with no BOS prefix) and a prompt that already opens with it are left
/// untouched, so a template that does emit `<s>` is never doubled.
pub fn clip_prepend_bos(rendered: &[i32], bos: Option<i32>) -> Vec<i32> {
    match bos {
        Some(b) if rendered.first() != Some(&b) => {
            let mut out = Vec::with_capacity(rendered.len() + 1);
            out.push(b);
            out.extend_from_slice(rendered);
            out
        }
        _ => rendered.to_vec(),
    }
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

/// Expand a rendered prompt's image markers for a flat-injection tower:
/// CLIP's bare run (`break_end` is `None`) or Pixtral's rows with a break
/// between them and an end after the last. `grids` is each image's
/// `(rows, cols)`. Spans come back one per run (CLIP) or one per patch row
/// (Pixtral).
pub fn flat_expand_image_placeholders(
    rendered: &[i32],
    marker: i32,
    fill: i32,
    break_end: Option<(i32, i32)>,
    grids: &[(usize, usize)],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    match break_end {
        Some((image_break, image_end)) => crate::pixtral::pixtral_expand_image_placeholders(
            rendered,
            marker,
            image_break,
            image_end,
            grids,
        ),
        None => {
            let counts: Vec<usize> = grids.iter().map(|&(h, w)| h * w).collect();
            clip_expand_image_placeholders(rendered, marker, fill, &counts)
        }
    }
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
                ClipPreprocessParams::from_preprocessor_config_json(&CONFIG.replace(needle, with))
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
    fn the_bos_is_added_once_and_only_when_missing() {
        assert_eq!(clip_prepend_bos(&[5, 6], Some(1)), vec![1, 5, 6]);
        assert_eq!(clip_prepend_bos(&[1, 5, 6], Some(1)), vec![1, 5, 6]);
        assert_eq!(clip_prepend_bos(&[5, 6], None), vec![5, 6]);
        assert_eq!(clip_prepend_bos(&[], Some(1)), vec![1]);
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

    /// The normalization divides by `image_std`, so a zero (or a value that
    /// rounds to zero in f32) turns every pixel into inf or NaN with no error.
    /// Each of the three layouts reads its own triple, so each is checked.
    #[test]
    fn a_non_positive_image_std_is_refused_in_every_layout() {
        let pixtral = r#"{"do_resize": true, "do_rescale": true, "do_normalize": true,
          "resample": 3, "patch_size": {"height": 16, "width": 16},
          "size": {"longest_edge": 1024}, "image_processor_type": "PixtralImageProcessor",
          "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
          "rescale_factor": 0.00392156862745098}"#;
        let onevision = r#"{"do_convert_rgb":true,"do_normalize":true,"do_pad":true,
          "do_rescale":true,"do_resize":true,
          "image_grid_pinpoints":[[384,384],[384,768]],
          "image_mean":[0.5,0.5,0.5],"image_processor_type":"LlavaOnevisionImageProcessor",
          "image_std": [0.5, 0.5, 0.5],"resample":3,"rescale_factor":0.00392156862745098,
          "size":{"height":384,"width":384}}"#;
        let clip_std = "\"image_std\": [0.26862954, 0.26130258, 0.27577711]";
        for (config, good) in [
            (CONFIG, clip_std),
            (pixtral, "\"image_std\": [0.5, 0.5, 0.5]"),
            (onevision, "\"image_std\": [0.5, 0.5, 0.5]"),
        ] {
            assert!(ClipPreprocessParams::from_preprocessor_config_json(config).is_ok());
            for std in ["[0.0, 0.5, 0.5]", "[0.5, -0.5, 0.5]", "[0.5, 0.5, 1e-50]"] {
                let text = config.replace(good, &format!("\"image_std\": {std}"));
                assert_ne!(text, config, "{good} must appear in the fixture");
                assert!(
                    ClipPreprocessParams::from_preprocessor_config_json(&text).is_err(),
                    "image_std {std} must be refused"
                );
            }
        }
    }

    /// A config-declared resize target is the allocation a decoded image is
    /// stretched to, so each layout's own size field is held to the decode
    /// ceiling: a CLIP crop edge, a Pixtral longest edge, a OneVision
    /// pinpoint. The real values (336, 1024, 2304) sit well under it.
    #[test]
    fn a_declared_size_past_the_decode_ceiling_is_refused_in_every_layout() {
        let over = crate::decode::MAX_IMAGE_DIM as usize + 1;
        let crop = (over / CLIP_PATCH + 1) * CLIP_PATCH;
        let clip = CONFIG.replace("336", &crop.to_string());
        let pixtral = r#"{"do_resize": true, "do_rescale": true, "do_normalize": true,
          "resample": 3, "patch_size": {"height": 16, "width": 16},
          "size": {"longest_edge": 1048576}, "image_processor_type": "PixtralImageProcessor",
          "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
          "rescale_factor": 0.00392156862745098}"#;
        let onevision = r#"{"do_convert_rgb":true,"do_normalize":true,"do_pad":true,
          "do_rescale":true,"do_resize":true,
          "image_grid_pinpoints":[[384,384],[384,393216]],
          "image_mean":[0.5,0.5,0.5],"image_processor_type":"LlavaOnevisionImageProcessor",
          "image_std": [0.5, 0.5, 0.5],"resample":3,"rescale_factor":0.00392156862745098,
          "size":{"height":384,"width":384}}"#;
        for config in [clip.as_str(), pixtral, onevision] {
            assert!(
                ClipPreprocessParams::from_preprocessor_config_json(config).is_err(),
                "{config} must be refused"
            );
        }
        // The real edges still parse, so the ceiling is not refusing them.
        let real = onevision
            .replace("393216", "2304")
            .replace("[384,384],", "[2304,2304],");
        assert!(ClipPreprocessParams::from_preprocessor_config_json(&real).is_ok());
        let real = pixtral.replace("1048576", "1024");
        assert!(ClipPreprocessParams::from_preprocessor_config_json(&real).is_ok());
    }
}
