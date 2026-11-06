//! Preprocessing, anyres tiling and row packing for LLaVA-OneVision
//! (`docs/LLAVA_ONEVISION_VISION_PHASE0.md`).
//!
//! The reference is Hugging Face's `LlavaOnevisionImageProcessor` and
//! `LlavaOnevisionForConditionalGeneration.pack_image_features` as mlx-vlm
//! ports them (read, never executed). One image becomes:
//!
//! - a BASE tile: the whole image squashed to 384x384 (PIL bicubic, no aspect
//!   preservation);
//! - an anyres grid: [`ov_select_best_resolution`] picks a `(height, width)`
//!   from the checkpoint's pinpoint list, the image is resized to fit it with
//!   its aspect kept ([`ov_patch_output_size`]), centered on a ZERO canvas of
//!   that size (raw pixels, before normalization, so the pad is the value
//!   `-mean/std` after it), and cut into 384x384 tiles in row-major order.
//!
//! Every tile is 27x27 patches of 14 pixels; a patch row is `(C, 14, 14)`, the
//! layout the conv weight `[out, C, 14, 14]` flattens to (the CLIP choice).
//! [`ClipPatches::pixels`] holds the base tile first, then the grid's tiles.
//!
//! The tower returns 729 projected rows per tile. [`ov_pack_features`] turns
//! them into the prompt run: the base tile's 729 rows in raster order, then the
//! grid's features laid out as one `(H, W)` map, UNPADDED back to the image's
//! aspect, pooled down (bilinear, `align_corners=False`, no antialias) when
//! they exceed the `anyres_max_9` budget, with the learned `image_newline`
//! row after every map row. Token counts and packing share one plan
//! ([`ov_plan`]) so the prompt can never disagree with the rows.

use crate::clip::ClipPatches;
use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;

/// Side of one tile, in pixels.
pub const OV_TILE: usize = 384;
/// Pixels per patch side.
pub const OV_PATCH: usize = 14;
/// Patches per tile side.
pub const OV_SIDE: usize = OV_TILE / OV_PATCH;
/// Patches (and projected rows) per tile.
pub const OV_TILE_PATCHES: usize = OV_SIDE * OV_SIDE;
/// Values per patch row: `3 * 14 * 14`.
pub const OV_PATCH_DIM: usize = 3 * OV_PATCH * OV_PATCH;
/// `vision_aspect_ratio = "anyres_max_9"`: the feature map is pooled once it
/// holds more than 9 tiles' worth of patches (with the 1.1 slack below).
pub const OV_MAX_AREA_TILES: usize = 9;

/// What the packing needs from the original image to undo the padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Anyres {
    /// The original image's height and width in pixels.
    pub orig_h: usize,
    pub orig_w: usize,
    /// The tile grid's rows and columns.
    pub tiles_h: usize,
    pub tiles_w: usize,
}

/// `select_best_resolution`: over `(height, width)` candidates, the one that
/// keeps the most of the image (capped at its own pixel count) and, on a tie,
/// wastes the least canvas. Returns `(height, width)`.
pub fn ov_select_best_resolution(
    orig_h: usize,
    orig_w: usize,
    pinpoints: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let mut best = None;
    let mut max_effective = 0usize;
    let mut min_wasted = usize::MAX;
    for &(h, w) in pinpoints {
        let scale = (w as f64 / orig_w as f64).min(h as f64 / orig_h as f64);
        let down_w = (orig_w as f64 * scale) as usize;
        let down_h = (orig_h as f64 * scale) as usize;
        let effective = (down_w * down_h).min(orig_w * orig_h);
        let wasted = (w * h).saturating_sub(effective);
        if effective > max_effective || (effective == max_effective && wasted < min_wasted) {
            max_effective = effective;
            min_wasted = wasted;
            best = Some((h, w));
        }
    }
    best
}

/// `_get_patch_output_size`: the size the image is resized to before padding,
/// aspect kept, never past the target. Returns `(height, width)`.
pub fn ov_patch_output_size(
    orig_h: usize,
    orig_w: usize,
    target_h: usize,
    target_w: usize,
) -> (usize, usize) {
    let scale_w = target_w as f64 / orig_w as f64;
    let scale_h = target_h as f64 / orig_h as f64;
    if scale_w < scale_h {
        let new_h = ((orig_h as f64 * scale_w).ceil() as usize).min(target_h);
        (new_h, target_w)
    } else {
        let new_w = ((orig_w as f64 * scale_h).ceil() as usize).min(target_w);
        (target_h, new_w)
    }
}

/// Python's `int(round(x, 7))` for the non-negative values used here.
fn round7_trunc(x: f64) -> usize {
    ((x * 1e7).round() / 1e7) as usize
}

/// The packing geometry for one image: where the grid's feature map is
/// cropped, and the size it ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OvPlan {
    /// Tile grid rows and columns.
    pub tiles_h: usize,
    pub tiles_w: usize,
    /// The feature map's full size, `tiles * 27` per axis.
    pub map_h: usize,
    pub map_w: usize,
    /// The crop of the map that removes the aspect padding: rows
    /// `[y0, y1)`, columns `[x0, x1)`.
    pub y0: usize,
    pub y1: usize,
    pub x0: usize,
    pub x1: usize,
    /// The map's size after the crop and any pooling.
    pub out_h: usize,
    pub out_w: usize,
}

impl OvPlan {
    /// Rows the grid contributes: the map plus one newline per map row.
    pub fn grid_rows(&self) -> usize {
        self.out_h * self.out_w + self.out_h
    }

    /// Rows in the whole prompt run: the base tile and the grid.
    pub fn total_rows(&self) -> usize {
        OV_TILE_PATCHES + self.grid_rows()
    }
}

/// Plan one image's packing from its original size and the tile grid the
/// processor chose.
pub fn ov_plan(orig_h: usize, orig_w: usize, tiles_h: usize, tiles_w: usize) -> OvPlan {
    let map_h = tiles_h * OV_SIDE;
    let map_w = tiles_w * OV_SIDE;
    // `unpad_image`.
    let orig_aspect = orig_w as f64 / orig_h as f64;
    let cur_aspect = map_w as f64 / map_h as f64;
    let (mut y0, mut y1, mut x0, mut x1) = (0, map_h, 0, map_w);
    if orig_aspect > cur_aspect {
        let scale = map_w as f64 / orig_w as f64;
        let new_h = round7_trunc(orig_h as f64 * scale);
        let pad = map_h.saturating_sub(new_h) / 2;
        y0 = pad;
        y1 = map_h - pad;
    } else {
        let scale = map_h as f64 / orig_h as f64;
        let new_w = round7_trunc(orig_w as f64 * scale);
        let pad = map_w.saturating_sub(new_w) / 2;
        x0 = pad;
        x1 = map_w - pad;
    }
    let (cur_h, cur_w) = (y1 - y0, x1 - x0);
    let (mut out_h, mut out_w) = (cur_h, cur_w);
    let ratio = ((cur_h * cur_w) as f64 / (OV_MAX_AREA_TILES * OV_SIDE * OV_SIDE) as f64).sqrt();
    if ratio > 1.1 {
        out_h = (cur_h as f64 / ratio).floor() as usize;
        out_w = (cur_w as f64 / ratio).floor() as usize;
    }
    OvPlan {
        tiles_h,
        tiles_w,
        map_h,
        map_w,
        y0,
        y1,
        x0,
        x1,
        out_h,
        out_w,
    }
}

/// Tokens one image occupies in the prompt, from its size alone.
pub fn ov_image_tokens(
    orig_h: usize,
    orig_w: usize,
    pinpoints: &[(usize, usize)],
) -> Result<usize, VisionIoError> {
    let (th, tw) = best_grid(orig_h, orig_w, pinpoints)?;
    Ok(ov_plan(orig_h, orig_w, th, tw).total_rows())
}

/// The tile grid `(rows, cols)` for an image of the given size.
pub fn best_grid(
    orig_h: usize,
    orig_w: usize,
    pinpoints: &[(usize, usize)],
) -> Result<(usize, usize), VisionIoError> {
    if orig_h == 0 || orig_w == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    let (h, w) = ov_select_best_resolution(orig_h, orig_w, pinpoints).ok_or_else(|| {
        VisionIoError::BadConfig {
            field: "image_grid_pinpoints".into(),
            why: "empty".into(),
        }
    })?;
    if h % OV_TILE != 0 || w % OV_TILE != 0 {
        return Err(VisionIoError::BadConfig {
            field: "image_grid_pinpoints".into(),
            why: format!("{h}x{w} is not a whole number of {OV_TILE}-pixel tiles"),
        });
    }
    Ok((h / OV_TILE, w / OV_TILE))
}

/// The worst-case tile count (base included) any pinpoint produces.
pub fn ov_max_tiles(pinpoints: &[(usize, usize)]) -> usize {
    1 + pinpoints
        .iter()
        .map(|&(h, w)| (h / OV_TILE) * (w / OV_TILE))
        .max()
        .unwrap_or(1)
}

/// Read `image_grid_pinpoints` (`[[h, w], ...]`) out of a processor config.
pub fn parse_pinpoints(root: &serde_json::Value) -> Result<Vec<(usize, usize)>, VisionIoError> {
    let bad = |why: &str| VisionIoError::BadConfig {
        field: "image_grid_pinpoints".into(),
        why: why.into(),
    };
    let list = root
        .get("image_grid_pinpoints")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad("missing"))?;
    let mut out = Vec::with_capacity(list.len());
    for entry in list {
        let pair = entry.as_array().filter(|p| p.len() == 2);
        let (Some(h), Some(w)) = (
            pair.and_then(|p| p[0].as_u64()),
            pair.and_then(|p| p[1].as_u64()),
        ) else {
            return Err(bad("an entry is not a [height, width] pair of integers"));
        };
        // The chosen pinpoint is the canvas the image is resized and padded
        // to; an unbounded one is a multi-GB allocation from a config.
        for edge in [h, w] {
            crate::params::check_config_edge("image_grid_pinpoints", edge as usize)?;
        }
        out.push((h as usize, w as usize));
    }
    if out.is_empty() {
        return Err(bad("empty"));
    }
    Ok(out)
}

/// Normalize and patchify one 384x384 tile of `data` (an interleaved RGB
/// canvas `canvas_w` wide) whose top-left corner is `(top, left)`.
#[allow(clippy::too_many_arguments)]
fn push_tile(
    out: &mut Vec<f32>,
    data: &[u8],
    canvas_w: usize,
    top: usize,
    left: usize,
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
) {
    for py in 0..OV_SIDE {
        for px in 0..OV_SIDE {
            for c in 0..3 {
                for r in 0..OV_PATCH {
                    let y = top + py * OV_PATCH + r;
                    for k in 0..OV_PATCH {
                        let x = left + px * OV_PATCH + k;
                        let sample = data[(y * canvas_w + x) * 3 + c];
                        out.push((sample as f32 * rescale - mean[c]) / std[c]);
                    }
                }
            }
        }
    }
}

/// Resize, pad, tile, normalize and patchify one decoded image.
///
/// `pixels` is `(1 + tiles_h * tiles_w) * 729` patch rows of 588 values. The
/// returned `grid` and `grid_w` are the prompt's placeholder count as a
/// degenerate `(tokens, 1)` grid: the prompt layer only multiplies the pair.
pub fn ov_preprocess(
    image: &Rgb8Image,
    pinpoints: &[(usize, usize)],
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
) -> Result<ClipPatches, VisionIoError> {
    let (oh, ow) = (image.height, image.width);
    let (tiles_h, tiles_w) = best_grid(oh, ow, pinpoints)?;
    let (target_h, target_w) = (tiles_h * OV_TILE, tiles_w * OV_TILE);

    let mut pixels = Vec::with_capacity((1 + tiles_h * tiles_w) * OV_TILE_PATCHES * OV_PATCH_DIM);

    // The base tile: a squash to the tile size.
    let base;
    let base_data: &[u8] = if (ow, oh) == (OV_TILE, OV_TILE) {
        &image.data
    } else {
        base = resize_bicubic_pil(&image.data, ow, oh, OV_TILE, OV_TILE);
        &base
    };
    push_tile(&mut pixels, base_data, OV_TILE, 0, 0, mean, std, rescale);

    // The grid: aspect-kept resize, centered on a zero canvas.
    let (nh, nw) = ov_patch_output_size(oh, ow, target_h, target_w);
    let resized;
    let resized_data: &[u8] = if (nw, nh) == (ow, oh) {
        &image.data
    } else {
        resized = resize_bicubic_pil(&image.data, ow, oh, nw, nh);
        &resized
    };
    let (top, left) = ((target_h - nh) / 2, (target_w - nw) / 2);
    let mut canvas = vec![0u8; target_h * target_w * 3];
    for y in 0..nh {
        let dst = ((top + y) * target_w + left) * 3;
        canvas[dst..dst + nw * 3].copy_from_slice(&resized_data[y * nw * 3..(y + 1) * nw * 3]);
    }
    for ty in 0..tiles_h {
        for tx in 0..tiles_w {
            push_tile(
                &mut pixels,
                &canvas,
                target_w,
                ty * OV_TILE,
                tx * OV_TILE,
                mean,
                std,
                rescale,
            );
        }
    }

    let plan = ov_plan(oh, ow, tiles_h, tiles_w);
    Ok(ClipPatches {
        pixels,
        grid: plan.total_rows(),
        grid_w: 1,
        anyres: Some(Anyres {
            orig_h: oh,
            orig_w: ow,
            tiles_h,
            tiles_w,
        }),
    })
}

/// `F.interpolate(mode="bilinear", align_corners=False)` of a `(h, w, c)`
/// row-major map to `(out_h, out_w, c)`, no antialias.
fn bilinear(src: &[f32], h: usize, w: usize, c: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    let scale_h = h as f64 / out_h as f64;
    let scale_w = w as f64 / out_w as f64;
    let coords = |i: usize, scale: f64, limit: usize| -> (usize, usize, f32) {
        let s = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let lo = (s.floor() as usize).min(limit - 1);
        let hi = (lo + 1).min(limit - 1);
        (lo, hi, (s - lo as f64) as f32)
    };
    let mut out = vec![0f32; out_h * out_w * c];
    for y in 0..out_h {
        let (y0, y1, ly) = coords(y, scale_h, h);
        for x in 0..out_w {
            let (x0, x1, lx) = coords(x, scale_w, w);
            for ch in 0..c {
                let v00 = src[(y0 * w + x0) * c + ch];
                let v01 = src[(y0 * w + x1) * c + ch];
                let v10 = src[(y1 * w + x0) * c + ch];
                let v11 = src[(y1 * w + x1) * c + ch];
                out[(y * out_w + x) * c + ch] =
                    (1.0 - ly) * ((1.0 - lx) * v00 + lx * v01) + ly * ((1.0 - lx) * v10 + lx * v11);
            }
        }
    }
    out
}

/// Pack the tower's rows into the prompt run.
///
/// `rows` is `(1 + tiles_h * tiles_w) * 729` rows of `dim` values (the base
/// tile first), `newline` the learned `image_newline` row. Returns
/// `plan.total_rows()` rows of `dim` values.
pub fn ov_pack_features(
    rows: &[f32],
    dim: usize,
    newline: &[f32],
    plan: &OvPlan,
) -> Result<Vec<f32>, VisionIoError> {
    let tiles = plan.tiles_h * plan.tiles_w;
    if rows.len() != (1 + tiles) * OV_TILE_PATCHES * dim || newline.len() != dim {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{} values for {tiles} tiles of {dim}-wide rows and a {}-wide newline",
                rows.len(),
                newline.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(plan.total_rows() * dim);
    out.extend_from_slice(&rows[..OV_TILE_PATCHES * dim]);

    // The grid as one (map_h, map_w, dim) map, tiles in row-major order.
    let grid_rows = &rows[OV_TILE_PATCHES * dim..];
    let (cur_h, cur_w) = (plan.y1 - plan.y0, plan.x1 - plan.x0);
    let mut cropped = Vec::with_capacity(cur_h * cur_w * dim);
    for y in plan.y0..plan.y1 {
        let (ty, i) = (y / OV_SIDE, y % OV_SIDE);
        for x in plan.x0..plan.x1 {
            let (tx, j) = (x / OV_SIDE, x % OV_SIDE);
            let row = (ty * plan.tiles_w + tx) * OV_TILE_PATCHES + i * OV_SIDE + j;
            cropped.extend_from_slice(&grid_rows[row * dim..(row + 1) * dim]);
        }
    }
    let map = if (plan.out_h, plan.out_w) == (cur_h, cur_w) {
        cropped
    } else {
        bilinear(&cropped, cur_h, cur_w, dim, plan.out_h, plan.out_w)
    };
    for y in 0..plan.out_h {
        out.extend_from_slice(&map[y * plan.out_w * dim..(y + 1) * plan.out_w * dim]);
        out.extend_from_slice(newline);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinpoints() -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for h in 1..=6 {
            for w in 1..=6 {
                v.push((h * OV_TILE, w * OV_TILE));
            }
        }
        v
    }

    #[test]
    fn a_square_image_takes_the_single_tile_grid_and_1485_tokens() {
        let p = pinpoints();
        assert_eq!(ov_select_best_resolution(300, 300, &p), Some((384, 384)));
        assert_eq!(ov_image_tokens(300, 300, &p).unwrap(), 729 + 729 + 27);
    }

    #[test]
    fn the_best_resolution_keeps_the_aspect_and_wastes_least() {
        let p = pinpoints();
        // 448 wide x 336 high: pinned against the independent python
        // transcription of the HF rule (phase-0 doc).
        assert_eq!(ov_select_best_resolution(336, 448, &p), Some((384, 768)));
        // A tall page.
        assert_eq!(
            ov_select_best_resolution(1280, 1024, &p),
            Some((1536, 1152))
        );
    }

    #[test]
    fn patch_output_size_keeps_the_aspect_and_never_passes_the_target() {
        assert_eq!(ov_patch_output_size(336, 448, 384, 768), (384, 512));
        assert_eq!(ov_patch_output_size(1280, 1024, 1536, 1152), (1440, 1152));
    }

    #[test]
    fn the_plan_unpads_the_canvas_back_to_the_image_aspect() {
        let plan = ov_plan(336, 448, 1, 2);
        // map 27 x 54; 4:3 over a 1:2 canvas -> the width is cropped.
        assert_eq!((plan.map_h, plan.map_w), (27, 54));
        assert_eq!((plan.y0, plan.y1), (0, 27));
        assert_eq!(plan.x1 - plan.x0, 36);
        assert_eq!((plan.out_h, plan.out_w), (27, 36));
        assert_eq!(plan.total_rows(), 729 + 27 * 36 + 27);
        // The independent python transcription (phase-0 doc) gives the same
        // token counts for the sizes the real gate uses.
        let p = pinpoints();
        assert_eq!(ov_image_tokens(336, 448, &p).unwrap(), 1728);
        assert_eq!(ov_image_tokens(448, 448, &p).unwrap(), 3699);
        assert_eq!(ov_image_tokens(480, 640, &p).unwrap(), 2929);
        assert_eq!(ov_image_tokens(700, 560, &p).unwrap(), 3159);
        assert_eq!(ov_image_tokens(1280, 1024, &p).unwrap(), 7299);
        assert_eq!(ov_image_tokens(192, 384, &p).unwrap(), 1093);
    }

    #[test]
    fn a_large_map_is_pooled_below_the_nine_tile_budget() {
        let plan = ov_plan(1280, 1024, 4, 3);
        assert!(plan.out_h * plan.out_w < (plan.y1 - plan.y0) * (plan.x1 - plan.x0));
        assert!(plan.out_h * plan.out_w <= 9 * 729);
    }

    #[test]
    fn bilinear_matches_torch_half_pixel_centres() {
        // 1x4 -> 1x2: sources at x = 0.5 and 2.5 -> blends (0,1) and (2,3).
        let src = [0.0, 10.0, 20.0, 30.0];
        let out = bilinear(&src, 1, 4, 1, 1, 2);
        assert!(
            (out[0] - 5.0).abs() < 1e-5 && (out[1] - 25.0).abs() < 1e-5,
            "{out:?}"
        );
        // Up-sampling clamps the first source to 0.
        let up = bilinear(&[0.0, 8.0], 1, 2, 1, 1, 4);
        assert!(
            (up[0] - 0.0).abs() < 1e-5 && (up[3] - 8.0).abs() < 1e-5,
            "{up:?}"
        );
    }

    #[test]
    fn the_packed_run_is_base_then_rows_with_a_newline_each() {
        // 1x1 grid, square image: map 27x27, no crop, no pool.
        let dim = 2usize;
        let plan = ov_plan(384, 384, 1, 1);
        let mut rows = Vec::new();
        for tile in 0..2 {
            for r in 0..OV_TILE_PATCHES {
                rows.push(tile as f32 * 1000.0 + r as f32);
                rows.push(0.5);
            }
        }
        let newline = [-1.0, -2.0];
        let out = ov_pack_features(&rows, dim, &newline, &plan).unwrap();
        assert_eq!(out.len() / dim, 729 + 27 * 27 + 27);
        // Base first, raster order.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[(728) * dim], 728.0);
        // Then grid row 0, then the newline row, then grid row 1.
        let at = |row: usize| out[row * dim];
        assert_eq!(at(729), 1000.0);
        assert_eq!(at(729 + 26), 1000.0 + 26.0);
        assert_eq!(&out[(729 + 27) * dim..(729 + 28) * dim], &newline);
        assert_eq!(at(729 + 28), 1000.0 + 27.0);
    }

    #[test]
    fn tiles_are_stitched_row_major_before_the_unpad() {
        // Two tiles side by side, a 1:2 image: no crop; row 0 of the map reads
        // tile 0's first row then tile 1's first row.
        let dim = 1usize;
        let plan = ov_plan(384, 768, 1, 2);
        assert_eq!((plan.out_h, plan.out_w), (27, 54));
        let mut rows = vec![0.0f32; OV_TILE_PATCHES];
        for tile in 0..2 {
            for r in 0..OV_TILE_PATCHES {
                rows.push((tile * 10000 + r) as f32);
            }
        }
        let out = ov_pack_features(&rows, dim, &[-1.0], &plan).unwrap();
        let first = &out[729..729 + 54];
        assert_eq!(first[0], 0.0);
        assert_eq!(first[26], 26.0);
        assert_eq!(first[27], 10000.0);
        assert_eq!(first[53], 10026.0);
        assert_eq!(out[729 + 54], -1.0);
    }

    #[test]
    fn preprocess_emits_base_then_tiles_and_pads_with_black() {
        let image = Rgb8Image {
            width: 384,
            height: 192,
            data: vec![255u8; 384 * 192 * 3],
        };
        let p = vec![(384usize, 384usize)];
        let patches = ov_preprocess(&image, &p, [0.5; 3], [0.5; 3], 1.0 / 255.0).unwrap();
        assert_eq!(patches.pixels.len(), 2 * 729 * 588);
        // Base is the squash of a white image: every value (1 - .5)/.5 = 1.
        assert!(patches.pixels[..729 * 588]
            .iter()
            .all(|&v| (v - 1.0).abs() < 1e-3));
        // The tile is centered on black: its first patch row (top of canvas)
        // is the pad, (0 - .5)/.5 = -1; the middle is white.
        let tile = &patches.pixels[729 * 588..];
        assert!((tile[0] + 1.0).abs() < 1e-6);
        let mid_patch = (13 * OV_SIDE + 13) * 588;
        assert!((tile[mid_patch] - 1.0).abs() < 1e-3);
        let plan = ov_plan(192, 384, 1, 1);
        assert_eq!(patches.grid * patches.grid_w, plan.total_rows());
    }

    #[test]
    fn pinpoints_parse_and_a_malformed_list_is_refused() {
        let ok: serde_json::Value =
            serde_json::from_str(r#"{"image_grid_pinpoints": [[384, 768], [384, 384]]}"#).unwrap();
        assert_eq!(parse_pinpoints(&ok).unwrap(), vec![(384, 768), (384, 384)]);
        let bad: serde_json::Value =
            serde_json::from_str(r#"{"image_grid_pinpoints": [[384]]}"#).unwrap();
        assert!(parse_pinpoints(&bad).is_err());
        assert_eq!(ov_max_tiles(&[(384, 768), (768, 768)]), 5);
    }

    const PUBLISHED: &str = r#"{"do_convert_rgb":true,"do_normalize":true,"do_pad":true,
      "do_rescale":true,"do_resize":true,
      "image_grid_pinpoints":[[384,384],[384,768],[768,384],[768,768]],
      "image_mean":[0.5,0.5,0.5],"image_processor_type":"LlavaOnevisionImageProcessor",
      "image_std":[0.5,0.5,0.5],"resample":3,"rescale_factor":0.00392156862745098,
      "size":{"height":384,"width":384}}"#;

    #[test]
    fn the_published_config_parses_and_a_changed_one_is_refused() {
        use crate::clip::{ClipLayout, ClipPreprocessParams};
        let p = ClipPreprocessParams::from_preprocessor_config_json(PUBLISHED).unwrap();
        assert_eq!(p.layout, ClipLayout::OneVision);
        assert_eq!((p.size, p.grid(), p.patch_dim()), (384, 27, 588));
        assert_eq!(p.pinpoints.len(), 4);
        // Worst case: base + a 2x2 grid.
        assert_eq!(p.tokens(), 5 * 729);
        assert_eq!(p.grid_for(300, 300), (1485, 1));
        p.check_against_tower(14, 729).unwrap();
        assert!(p.check_against_tower(14, 730).is_err());
        let no_pad = PUBLISHED.replace("\"do_pad\":true", "\"do_pad\":false");
        assert!(ClipPreprocessParams::from_preprocessor_config_json(&no_pad).is_err());
        let odd = PUBLISHED.replace("[384,768]", "[384,700]");
        assert!(ClipPreprocessParams::from_preprocessor_config_json(&odd).is_err());
    }
}
