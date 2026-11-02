//! Preprocessing for DeepSeek-VL2's SigLIP tower (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`).
//!
//! The reference is `DeepseekVLV2Processor.tokenize_with_images`, which is
//! transcribed here from its source rather than run:
//!
//! - `select_best_resolution` picks, from `candidate_resolutions`, the
//!   `(width, height)` that keeps the most of the image (downscaled sizes are
//!   truncated with `int()`), breaking ties on the least wasted area;
//! - the image is padded, never cropped or stretched, with PIL
//!   `ImageOps.pad` (bicubic `contain`, then pasted centered on a canvas of
//!   `int(mean * 255)` per channel, which is 127 for the published 0.5 mean);
//! - one GLOBAL view (the image padded to `image_size x image_size`) comes
//!   first, then the LOCAL view (padded to the chosen resolution) cut into
//!   `image_size` tiles row-major;
//! - every tile is rescaled by `1/255` and normalized with `(x - mean) / std`;
//! - patch rows are `(P_h, P_w, C)`, the layout the converted checkpoint's
//!   `patch_embed.proj.weight [1152, 14, 14, 3]` flattens to, so the repack
//!   copies that weight verbatim and the GEMM reads it row-major.
//!
//! More than two images in one request turn tiling off (`cropping` is
//! `len(images) <= 2` in the reference): each image is then its global view
//! plus one 384 tile. The caller passes that as `crop`.
//!
//! `ImageOps.contain` rounds with Python's `round()` from Pillow 10 on and
//! truncated with `int()` before. This follows the newer behavior: it differs
//! from the older one only when `size * h / w` lands within one of an
//! integer, i.e. by at most one padded row.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;
use crate::rounding::round_half_to_even;

/// Pixels per patch side. SigLIP-so400m/14.
pub const DSVL2_PATCH: usize = 14;
/// Values per patch row: `14 * 14 * 3`, `(P_h, P_w, C)` order.
pub const DSVL2_PATCH_DIM: usize = DSVL2_PATCH * DSVL2_PATCH * 3;

/// What `processor_config.json` says, narrowed to what this pipeline acts on.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekVl2Params {
    /// Tile side: the first candidate's width, as the reference defines it.
    pub image_size: usize,
    /// `(width, height)` pairs, in file order (ties resolve to the first).
    pub candidates: Vec<(usize, usize)>,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Merge factor of the projector's unfold.
    pub downsample_ratio: usize,
}

impl DeepseekVl2Params {
    /// Parse `processor_config.json`.
    ///
    /// Refused rather than ignored: a patch size other than 14, a missing
    /// `normalize`, a candidate that is not a whole number of tiles, a
    /// downsample ratio the projector does not implement.
    pub fn from_processor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let root: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionIoError::BadConfig {
                field: "<root>".into(),
                why: e.to_string(),
            })?;
        let bad = |field: &str, why: String| VisionIoError::BadConfig {
            field: field.to_string(),
            why,
        };
        if root.get("patch_size").and_then(serde_json::Value::as_u64) != Some(DSVL2_PATCH as u64) {
            return Err(bad("patch_size", format!("must be {DSVL2_PATCH}")));
        }
        if root.get("normalize").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(bad("normalize", "this pipeline needs true".into()));
        }
        let downsample_ratio =
            root.get("downsample_ratio")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| bad("downsample_ratio", "missing".into()))? as usize;
        if downsample_ratio != 2 {
            return Err(bad(
                "downsample_ratio",
                format!("{downsample_ratio}; only the 2x2 unfold is implemented"),
            ));
        }
        let list = root
            .get("candidate_resolutions")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| bad("candidate_resolutions", "missing".into()))?;
        let mut candidates = Vec::with_capacity(list.len());
        for item in list {
            let pair = item
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| bad("candidate_resolutions", "an entry is not a pair".into()))?;
            let get = |i: usize| {
                pair[i]
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| bad("candidate_resolutions", "a non-integer entry".into()))
            };
            candidates.push((get(0)?, get(1)?));
        }
        let image_size = candidates
            .first()
            .map(|c| c.0)
            .ok_or_else(|| bad("candidate_resolutions", "empty".into()))?;
        // 384 is not a multiple of 14: the tower's stride-14 conv has no
        // padding, so it reads floor(384 / 14) = 27 patches a side and the
        // last 6 pixel columns and rows never reach it.
        if image_size < DSVL2_PATCH {
            return Err(bad(
                "candidate_resolutions",
                format!("tile side {image_size} is under one {DSVL2_PATCH} px patch"),
            ));
        }
        if let Some((w, h)) = candidates
            .iter()
            .find(|(w, h)| w % image_size != 0 || h % image_size != 0 || *w == 0 || *h == 0)
        {
            return Err(bad(
                "candidate_resolutions",
                format!("{w}x{h} is not a whole number of {image_size} tiles"),
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
        Ok(Self {
            image_size,
            candidates,
            image_mean: triple("image_mean")?,
            image_std: triple("image_std")?,
            downsample_ratio,
        })
    }

    /// Patches per tile side (floor: the conv drops a partial patch).
    pub fn patches_per_side(&self) -> usize {
        self.image_size / DSVL2_PATCH
    }

    /// Projector output positions per tile side: `ceil(patches / ratio)`.
    pub fn merged_per_side(&self) -> usize {
        self.patches_per_side().div_ceil(self.downsample_ratio)
    }

    /// The canvas fill, `int(mean * 255)` per channel.
    pub fn pad_color(&self) -> [u8; 3] {
        [
            (self.image_mean[0] * 255.0) as u8,
            (self.image_mean[1] * 255.0) as u8,
            (self.image_mean[2] * 255.0) as u8,
        ]
    }
}

/// `select_best_resolution`, over `(width, height)` pairs.
///
/// Python's `int()` is truncation toward zero, and the products stay exact in
/// f64 at these magnitudes, so the comparison below sees the same integers.
pub fn select_best_resolution(
    width: usize,
    height: usize,
    candidates: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let (ow, oh) = (width as f64, height as f64);
    let mut best = None;
    let mut max_effective = 0usize;
    let mut min_wasted = usize::MAX;
    for &(cw, ch) in candidates {
        let scale = (cw as f64 / ow).min(ch as f64 / oh);
        let dw = (ow * scale) as usize;
        let dh = (oh * scale) as usize;
        let effective = (dw * dh).min(width * height);
        let wasted = cw * ch - effective;
        if effective > max_effective || (effective == max_effective && wasted < min_wasted) {
            max_effective = effective;
            min_wasted = wasted;
            best = Some((cw, ch));
        }
    }
    best
}

/// `ImageOps.pad(image, (tw, th), color=color)` with bicubic and centered.
pub fn pil_pad(
    image: &Rgb8Image,
    tw: usize,
    th: usize,
    color: [u8; 3],
) -> Result<Rgb8Image, VisionIoError> {
    // `contain`: scale to fit inside the target, keeping the aspect ratio.
    let (w, h) = (image.width, image.height);
    let im_ratio = w as f64 / h as f64;
    let dest_ratio = tw as f64 / th as f64;
    let (mut rw, mut rh) = (tw, th);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            rh = round_half_to_even(h as f64 / w as f64 * tw as f64) as usize;
        } else {
            rw = round_half_to_even(w as f64 / h as f64 * th as f64) as usize;
        }
    }
    if rw == 0 || rh == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("{w}x{h} fits a {tw}x{th} canvas as {rw}x{rh}"),
        });
    }
    let resized = resize_bicubic_pil(&image.data, w, h, rw, rh);
    if (rw, rh) == (tw, th) {
        return Rgb8Image::new(tw, th, resized);
    }
    let mut out = Vec::with_capacity(tw * th * 3);
    for _ in 0..tw * th {
        out.extend_from_slice(&color);
    }
    // Exactly one axis is short (the other equals the target), so the paste
    // offset is zero on the other axis. `round()` is Python's, half to even.
    let (x0, y0) = if rw != tw {
        (round_half_to_even((tw - rw) as f64 * 0.5) as usize, 0)
    } else {
        (0, round_half_to_even((th - rh) as f64 * 0.5) as usize)
    };
    for y in 0..rh {
        let dst = ((y0 + y) * tw + x0) * 3;
        out[dst..dst + rw * 3].copy_from_slice(&resized[y * rw * 3..(y + 1) * rw * 3]);
    }
    Rgb8Image::new(tw, th, out)
}

/// One image as tower input: the global view then the local tiles.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekVl2Tiles {
    /// `[tiles * patches^2, 588]`, tile-major then row-major over each tile's
    /// patch grid, each row `(14, 14, 3)`.
    pub pixels: Vec<f32>,
    /// Local tiles across and down (the reference's `images_spatial_crop`).
    pub tiles_w: usize,
    pub tiles_h: usize,
    /// Patches per tile side.
    pub patches_per_side: usize,
    /// Projector output positions per tile side.
    pub merged_per_side: usize,
}

impl DeepseekVl2Tiles {
    /// Tiles the tower sees: the global view plus the local grid.
    pub fn tiles(&self) -> usize {
        1 + self.tiles_w * self.tiles_h
    }

    /// Prompt positions this image occupies.
    pub fn image_tokens(&self) -> usize {
        deepseek_vl2_image_tokens(self.tiles_w, self.tiles_h, self.merged_per_side)
    }
}

/// Positions an image spans in the prompt: the global view's `m * (m + 1)`
/// rows (a newline row-end per row), one view separator, then the local grid's
/// `(th * m) * (tw * m + 1)`.
pub fn deepseek_vl2_image_tokens(tiles_w: usize, tiles_h: usize, merged: usize) -> usize {
    merged * (merged + 1) + 1 + (tiles_h * merged) * (tiles_w * merged + 1)
}

/// Where each prompt position of one image comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekVl2Row {
    /// Projector row `row` of tile `tile` (tile 0 is the global view).
    Feature { tile: usize, row: usize },
    /// The learned `image_newline` embedding.
    Newline,
    /// The learned `view_separator` embedding.
    Separator,
}

/// The sequence the reference's `process_image_features` builds for one
/// image, as a source per position: global view rows with a newline after
/// each, the separator, then the local grid reassembled from tiles with a
/// newline after each assembled row.
pub fn deepseek_vl2_row_layout(
    tiles_w: usize,
    tiles_h: usize,
    merged: usize,
) -> Vec<DeepseekVl2Row> {
    let mut out = Vec::with_capacity(deepseek_vl2_image_tokens(tiles_w, tiles_h, merged));
    for r in 0..merged {
        for c in 0..merged {
            out.push(DeepseekVl2Row::Feature {
                tile: 0,
                row: r * merged + c,
            });
        }
        out.push(DeepseekVl2Row::Newline);
    }
    out.push(DeepseekVl2Row::Separator);
    for ty in 0..tiles_h {
        for r in 0..merged {
            for tx in 0..tiles_w {
                let tile = 1 + ty * tiles_w + tx;
                for c in 0..merged {
                    out.push(DeepseekVl2Row::Feature {
                        tile,
                        row: r * merged + c,
                    });
                }
            }
            out.push(DeepseekVl2Row::Newline);
        }
    }
    out
}

/// Pad, tile, normalize and patchify one decoded image.
///
/// `crop` is the reference's `len(images) <= 2`.
pub fn deepseek_vl2_preprocess(
    image: &Rgb8Image,
    params: &DeepseekVl2Params,
    crop: bool,
) -> Result<DeepseekVl2Tiles, VisionIoError> {
    if image.width == 0 || image.height == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    let size = params.image_size;
    let (bw, bh) = if crop {
        select_best_resolution(image.width, image.height, &params.candidates).ok_or_else(|| {
            VisionIoError::BadConfig {
                field: "candidate_resolutions".into(),
                why: "empty".into(),
            }
        })?
    } else {
        (size, size)
    };
    let color = params.pad_color();
    let global = pil_pad(image, size, size, color)?;
    let local = pil_pad(image, bw, bh, color)?;
    let (tiles_w, tiles_h) = (bw / size, bh / size);
    let per_side = params.patches_per_side();

    let mut pixels =
        Vec::with_capacity((1 + tiles_w * tiles_h) * per_side * per_side * DSVL2_PATCH_DIM);
    let mut emit = |src: &Rgb8Image, x0: usize, y0: usize| {
        for py in 0..per_side {
            for px in 0..per_side {
                for r in 0..DSVL2_PATCH {
                    for k in 0..DSVL2_PATCH {
                        let y = y0 + py * DSVL2_PATCH + r;
                        let x = x0 + px * DSVL2_PATCH + k;
                        for c in 0..3 {
                            let s = src.data[(y * src.width + x) * 3 + c];
                            pixels.push(
                                (s as f32 / 255.0 - params.image_mean[c]) / params.image_std[c],
                            );
                        }
                    }
                }
            }
        }
    };
    emit(&global, 0, 0);
    for ty in 0..tiles_h {
        for tx in 0..tiles_w {
            emit(&local, tx * size, ty * size);
        }
    }
    Ok(DeepseekVl2Tiles {
        pixels,
        tiles_w,
        tiles_h,
        patches_per_side: per_side,
        merged_per_side: params.merged_per_side(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
      "candidate_resolutions": [[384,384],[384,768],[768,384],[384,1152],[1152,384],
        [384,1536],[1536,384],[768,768],[384,1920],[1920,384],[384,2304],[2304,384],
        [768,1152],[1152,768],[384,2688],[2688,384],[384,3072],[3072,384],[768,1536],
        [1536,768],[384,3456],[3456,384],[1152,1152]],
      "downsample_ratio": 2, "image_mean": [0.5,0.5,0.5], "image_std": [0.5,0.5,0.5],
      "normalize": true, "patch_size": 14
    }"#;

    fn params() -> DeepseekVl2Params {
        DeepseekVl2Params::from_processor_config_json(CONFIG).unwrap()
    }

    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Rgb8Image {
        let data = (0..w * h).flat_map(|_| rgb).collect();
        Rgb8Image::new(w, h, data).unwrap()
    }

    #[test]
    fn the_published_config_parses() {
        let p = params();
        assert_eq!(p.image_size, 384);
        assert_eq!(p.candidates.len(), 23);
        assert_eq!(p.patches_per_side(), 27);
        // ceil(27 / 2): the projector zero-pads 27 up to 28 before the unfold.
        assert_eq!(p.merged_per_side(), 14);
        assert_eq!(p.pad_color(), [127, 127, 127]);
    }

    #[test]
    fn a_config_this_pipeline_does_not_implement_is_refused() {
        for (from, to) in [
            ("\"patch_size\": 14", "\"patch_size\": 16"),
            ("\"downsample_ratio\": 2", "\"downsample_ratio\": 3"),
            ("\"normalize\": true", "\"normalize\": false"),
        ] {
            let text = CONFIG.replace(from, to);
            assert_ne!(text, CONFIG, "{from} must appear in the fixture");
            assert!(
                DeepseekVl2Params::from_processor_config_json(&text).is_err(),
                "{to}"
            );
        }
        let ragged = CONFIG.replace("[1152,1152]", "[1100,1152]");
        assert!(DeepseekVl2Params::from_processor_config_json(&ragged).is_err());
    }

    #[test]
    fn best_resolution_keeps_the_most_pixels_then_wastes_the_least() {
        let c = params().candidates;
        // A square image fits the 1x1 tile only when it is small; a 1000 px
        // square keeps the most at 1152x1152 (a 1000x1000 downscale).
        assert_eq!(select_best_resolution(1000, 1000, &c), Some((1152, 1152)));
        // 100x100: everything keeps all 10_000 pixels, so the smallest area
        // wins the tie.
        assert_eq!(select_best_resolution(100, 100, &c), Some((384, 384)));
        // Wide: 2000x500 fits 1536x384 or 1920x384 without loss of the 4:1
        // aspect; the exact winner is the reference's arithmetic, pinned here.
        assert_eq!(select_best_resolution(2000, 500, &c), Some((1536, 384)));
        // Tall mirrors wide, with the pair ordered (width, height).
        assert_eq!(select_best_resolution(500, 2000, &c), Some((384, 1536)));
    }

    #[test]
    fn pad_centers_with_pythons_rounding_and_fills_with_the_mean() {
        // 2x1 into 4x4: contain gives 4x2, pasted at y = round((4 - 2) / 2) = 1.
        let img = solid(2, 1, [255, 0, 0]);
        let out = pil_pad(&img, 4, 4, [127, 127, 127]).unwrap();
        assert_eq!((out.width, out.height), (4, 4));
        assert_eq!(out.sample(0, 0, 0), 127);
        assert_eq!(out.sample(1, 0, 0), 255);
        assert_eq!(out.sample(2, 3, 0), 255);
        assert_eq!(out.sample(3, 0, 0), 127);
        let img = solid(1, 1, [0, 0, 255]);
        let out = pil_pad(&img, 5, 4, [9, 9, 9]).unwrap();
        // contain: ratio 1 < 1.25 so width = round(1 * 4) = 4; slack is 1 and
        // x = round(0.5) = 0 under half-to-even (rounding half up gives 1).
        assert_eq!(out.sample(0, 0, 2), 255);
        assert_eq!(out.sample(0, 4, 2), 9);
    }

    #[test]
    fn a_target_the_image_already_fills_is_only_resized() {
        let img = solid(8, 8, [10, 20, 30]);
        let out = pil_pad(&img, 4, 4, [0, 0, 0]).unwrap();
        assert!(out.data.chunks(3).all(|p| p == [10, 20, 30]));
    }

    #[test]
    fn the_row_layout_matches_the_reference_counts() {
        // A 1x1 grid: 14 * 15 global rows, the separator, 14 * 15 local rows.
        assert_eq!(deepseek_vl2_image_tokens(1, 1, 14), 14 * 15 + 1 + 14 * 15);
        assert_eq!(deepseek_vl2_image_tokens(2, 1, 14), 14 * 15 + 1 + 14 * 29);
        assert_eq!(deepseek_vl2_image_tokens(2, 3, 14), 14 * 15 + 1 + 42 * 29);
        for (tw, th) in [(1, 1), (2, 1), (1, 3), (3, 2)] {
            let layout = deepseek_vl2_row_layout(tw, th, 14);
            assert_eq!(layout.len(), deepseek_vl2_image_tokens(tw, th, 14));
            let newlines = layout
                .iter()
                .filter(|r| **r == DeepseekVl2Row::Newline)
                .count();
            assert_eq!(newlines, 14 + th * 14);
            assert_eq!(
                layout
                    .iter()
                    .filter(|r| **r == DeepseekVl2Row::Separator)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn local_rows_interleave_tiles_across_one_assembled_row() {
        // 2 tiles across, merged side 2: the first assembled local row is
        // tile 1 row 0, tile 2 row 0, newline; the second is tile 1 row 1 ...
        let layout = deepseek_vl2_row_layout(2, 1, 2);
        let global = 2 * 3 + 1;
        let f = |tile, row| DeepseekVl2Row::Feature { tile, row };
        assert_eq!(
            &layout[global..global + 5],
            &[f(1, 0), f(1, 1), f(2, 0), f(2, 1), DeepseekVl2Row::Newline]
        );
        assert_eq!(layout[global + 5], f(1, 2));
        // Every feature row of every tile is used exactly once.
        let mut seen = [0u8; 3 * 4];
        for r in &layout {
            if let DeepseekVl2Row::Feature { tile, row } = r {
                seen[tile * 4 + row] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn preprocessing_emits_the_global_view_then_row_major_local_tiles() {
        let p = params();
        // 1000x1000 picks 1152x1152: a 3x3 grid, so ten tiles in all.
        let img = solid(1000, 1000, [255, 255, 255]);
        let t = deepseek_vl2_preprocess(&img, &p, true).unwrap();
        assert_eq!((t.tiles_w, t.tiles_h, t.tiles()), (3, 3, 10));
        assert_eq!(t.pixels.len(), 10 * 729 * DSVL2_PATCH_DIM);
        assert_eq!(t.image_tokens(), 14 * 15 + 1 + 42 * 43);
        // White normalizes to (1 - 0.5) / 0.5 = 1.
        assert_eq!(t.pixels[0], 1.0);
        // Cropping off collapses to the global view plus one tile.
        let one = deepseek_vl2_preprocess(&img, &p, false).unwrap();
        assert_eq!((one.tiles_w, one.tiles_h, one.tiles()), (1, 1, 2));
    }

    #[test]
    fn padding_normalizes_to_nearly_zero_and_patch_rows_are_hwc() {
        let p = params();
        // A 384x192 red image: global = 384 wide, 192 tall, padded 96 above
        // and below with 127 (-> (127/255 - .5) / .5, about -0.0039).
        let img = solid(384, 192, [255, 0, 0]);
        let t = deepseek_vl2_preprocess(&img, &p, false).unwrap();
        let pad = (127.0f32 / 255.0 - 0.5) / 0.5;
        // First patch of the global view is all padding: (0,0,c) first.
        assert_eq!(&t.pixels[0..3], &[pad, pad, pad]);
        // A patch row inside the image: patch row index 14 * 27 is pixel row
        // 14*14 = 196 > 96, so it is image; channel 0 is red = 1, 1 and 2 = -1.
        let row = (14 * 27 + 5) * DSVL2_PATCH_DIM;
        assert_eq!(&t.pixels[row..row + 3], &[1.0, -1.0, -1.0]);
    }
}
