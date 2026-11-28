//! Preprocessing, pixel shuffle and prompt layout for the InternViT tower
//! the InternVL wrappers carry (`docs/INTERNVL_VISION_PHASE0.md`).
//!
//! The reference is mlx-vlm's `InternVLChatProcessor` / `dynamic_preprocess`
//! (`internvl_chat/processor.py`, read, never executed). It is a fourth
//! pipeline in this crate, and its shape is the anyres tiling OneVision does
//! without the packing:
//!
//! - the whole image is RESIZED (PIL bicubic) to `cols x rows` tiles of
//!   `image_size` (448) pixels, where `(cols, rows)` is the aspect-ratio
//!   pair whose product sits in `[min_num, max_num]` (1..12) and whose
//!   `cols / rows` is closest to the image's own ratio, ties broken toward
//!   the pair whose total area is closest to the image's;
//! - the resized image is cropped into exactly `cols * rows` tiles, and when
//!   there is more than one tile a THUMBNAIL (the original image resized to
//!   one tile) is appended, so a 3-tile image is four tower runs;
//! - each tile is scaled by `1/255` and normalized with the ImageNet
//!   constants the reference hardcodes (`0.485, 0.456, 0.406` /
//!   `0.229, 0.224, 0.225`; the published `preprocessor_config.json` states
//!   the same pair, so the file is read and the constants are the default);
//! - patches are emitted row-major over each tile's `32 x 32` grid and one
//!   patch row is `(P_h, P_w, C)`, the layout the checkpoint's conv weight
//!   `[out, kh, kw, C]` flattens to (MLX's own conv order -- the CLIP and
//!   Pixtral towers' `(C, P_h, P_w)` rows are the transpose of this, which
//!   is why this pipeline does not reuse [`crate::clip::ClipPatches`]).
//!
//! The tower consumes tiles as SEPARATE runs (no attention across tiles);
//! the projector's pixel shuffle mixes 2x2 patches WITHIN a tile only, so
//! the per-tile row sets are independent end to end. The prompt-side token
//! count is `tokens_per_tile` per tower run, all inside one `<img>` ...
//! `</img>` span per image.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::resize::resize_bicubic_pil;

/// The ImageNet normalization constants the reference processor hardcodes.
pub const INTERNVL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const INTERNVL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The reference processor's defaults for the dynamic-tiling controls, read
/// off `config.json`'s own defaults (`max_dynamic_patch` 12, `min_dynamic_patch`
/// 1, `use_thumbnail` true) and `force_image_size` / `vision_config.image_size`
/// 448.
const DEFAULT_TILE: usize = 448;
const DEFAULT_MAX_DYNAMIC: usize = 12;
const DEFAULT_MIN_DYNAMIC: usize = 1;

/// The tower preprocessing's settings, read off the install's
/// `preprocessor_config.json` (`size`, `image_mean`, `image_std`) and the
/// checkpoint config's tiling keys, with the reference defaults for
/// anything the file omits (the InternVL3.5 conversion ships a
/// `processor_config.json` that names only `image_seq_length`).
#[derive(Debug, Clone, PartialEq)]
pub struct InternVlParams {
    /// One tile's edge in pixels. 448 on every published InternVL tower.
    pub tile: usize,
    /// The largest tile grid the processor may pick (`max_dynamic_patch`).
    pub max_dynamic: usize,
    /// The smallest (`min_dynamic_patch`).
    pub min_dynamic: usize,
    /// Whether a multi-tile image appends the thumbnail run.
    pub use_thumbnail: bool,
    /// Normalization mean, per channel.
    pub mean: [f32; 3],
    /// Normalization standard deviation, per channel.
    pub std: [f32; 3],
}

impl Default for InternVlParams {
    fn default() -> Self {
        Self {
            tile: DEFAULT_TILE,
            max_dynamic: DEFAULT_MAX_DYNAMIC,
            min_dynamic: DEFAULT_MIN_DYNAMIC,
            use_thumbnail: true,
            mean: INTERNVL_MEAN,
            std: INTERNVL_STD,
        }
    }
}

impl InternVlParams {
    /// Read the settings the published `preprocessor_config.json` states.
    /// Unknown keys are ignored; missing ones take the reference defaults,
    /// which the published file's own values agree with.
    pub fn from_preprocessor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionIoError::InvalidDimensions {
                detail: format!("preprocessor config is not JSON: {e}"),
            })?;
        let mut params = Self::default();
        if let Some(size) = v.get("size").and_then(serde_json::Value::as_u64) {
            params.tile = size as usize;
        } else if let Some(crop) = v.get("crop_size").and_then(serde_json::Value::as_u64) {
            params.tile = crop as usize;
        }
        if let Some(mean) = float_triplet(v.get("image_mean")) {
            params.mean = mean;
        }
        if let Some(std) = float_triplet(v.get("image_std")) {
            params.std = std;
        }
        if params.tile == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: "preprocessor config states a zero tile edge".into(),
            });
        }
        Ok(params)
    }
}

fn float_triplet(v: Option<&serde_json::Value>) -> Option<[f32; 3]> {
    let items = v?.as_array()?;
    if items.len() != 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (i, item) in items.iter().enumerate() {
        out[i] = item.as_f64()? as f32;
    }
    Some(out)
}

/// The tower's tile grid for one image: the `(cols, rows)` pair the reference
/// picks. `target_ratios` is every `(i, j)` with `i * j` in
/// `[min_num, max_num]`, deduplicated and sorted by area; the winner is the
/// pair whose ratio is closest to the image's own, with the reference's
/// tie-break toward the pair whose total TARGET area
/// (`tile^2 * i * j`, which is why `tile` is a parameter) is closest to the
/// image's own pixel area.
pub fn internvl_tile_grid(
    width: usize,
    height: usize,
    tile: usize,
    min_num: usize,
    max_num: usize,
) -> Result<(usize, usize), VisionIoError> {
    if width == 0 || height == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    if tile == 0 || min_num == 0 || max_num < min_num {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("tile range [{min_num}, {max_num}] at edge {tile} is empty"),
        });
    }
    // A set in the reference: the loop over `n` regenerates the small pairs
    // once per n, and a duplicate row would let a later duplicate win the
    // area tie-break the first occurrence already settled.
    let mut seen = std::collections::BTreeSet::new();
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= max_num && i * j >= min_num && seen.insert((i, j)) {
                    ratios.push((i, j));
                }
            }
        }
    }
    ratios.sort_by_key(|&(i, j)| i * j);
    let aspect = width as f64 / height as f64;
    let area = (width * height) as f64;
    let mut best = (1usize, 1usize);
    let mut best_diff = f64::INFINITY;
    for &(i, j) in &ratios {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff {
            let target = (tile * tile * i * j) as f64;
            let best_target = (tile * tile * best.0 * best.1) as f64;
            if (area - target).abs() < (area - best_target).abs() {
                best = (i, j);
            }
        }
    }
    Ok(best)
}

/// One preprocessed image: every tower run's patch rows, back to back.
#[derive(Debug, Clone, PartialEq)]
pub struct InternVlTiles {
    /// `[runs * side_patches^2, patch_dim]`, run-major (tiles first, then the
    /// thumbnail), each run row-major over its `side_patches x side_patches`
    /// grid, one row `(P_h, P_w, C)`, normalized.
    pub rows: Vec<f32>,
    /// Tower runs: the grid tiles plus the thumbnail when there is one.
    pub tiles: usize,
    /// Patches per tile side (`448 / 14` = 32).
    pub side_patches: usize,
    /// Values per patch row (`3 * patch^2` = 588).
    pub patch_dim: usize,
}

impl InternVlTiles {
    /// Patch grid side of one run (`side_patches`).
    pub fn run_side(&self) -> usize {
        self.side_patches
    }
}

/// Resize, tile, normalize and patchify one decoded image.
///
/// `patch` is the tower's patch edge (14) and must divide `params.tile`
/// exactly; the tower runs are the grid's tiles and then the thumbnail when
/// the grid holds more than one tile and `params.use_thumbnail` is set.
pub fn internvl_preprocess(
    image: &Rgb8Image,
    patch: usize,
    params: &InternVlParams,
) -> Result<InternVlTiles, VisionIoError> {
    if params.tile % patch != 0 || params.tile == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "tile edge {} is not a whole number of {patch}-pixel patches",
                params.tile
            ),
        });
    }
    let (cols, rows) = internvl_tile_grid(
        image.width,
        image.height,
        params.tile,
        params.min_dynamic,
        params.max_dynamic,
    )?;
    let tw = cols * params.tile;
    let th = rows * params.tile;
    let resized = resize_bicubic_pil(&image.data, image.width, image.height, tw, th);
    let side_patches = params.tile / patch;
    let patch_dim = 3 * patch * patch;
    let runs = cols * rows;
    let total_runs = tile_runs(runs, params);
    let mut out = InternVlTiles {
        rows: Vec::new(),
        tiles: total_runs,
        side_patches,
        patch_dim,
    };
    out.rows
        .reserve_exact(total_runs * side_patches * side_patches * patch_dim);

    // The grid tiles: crop the resized image into `cols x rows` exact tiles,
    // row-major (the reference crops by `row_idx = i // cols`).
    for tile in 0..runs {
        let tx = (tile % cols) * params.tile;
        let ty = (tile / cols) * params.tile;
        patchify_tile(
            &resized,
            tw,
            th,
            tx,
            ty,
            params.tile,
            patch,
            &params.mean,
            &params.std,
            &mut out.rows,
        );
    }
    // The thumbnail: the ORIGINAL image resized to one tile, only when the
    // image actually split (the reference adds no thumbnail to a 1x1 grid).
    if total_runs > runs {
        let thumb = resize_bicubic_pil(
            &image.data,
            image.width,
            image.height,
            params.tile,
            params.tile,
        );
        patchify_tile(
            &thumb,
            params.tile,
            params.tile,
            0,
            0,
            params.tile,
            patch,
            &params.mean,
            &params.std,
            &mut out.rows,
        );
    }
    Ok(out)
}

/// Tower runs for a grid of `grid` tiles: a split image adds the thumbnail.
fn tile_runs(grid: usize, params: &InternVlParams) -> usize {
    grid + usize::from(grid > 1 && params.use_thumbnail)
}

/// What each image content part flattens to before the template renders. The
/// published `internvl_chat` templates concatenate `message.content` as a
/// string, and the reference writes `<img>` through its conversation module.
pub const INTERNVL_IMAGE_MARKER_TEXT: &str = "<img>\n";

/// One install's InternViT prompt layout: the processor settings, the tower
/// geometry and the three span ids. The CLI, the FFI and the server all place
/// and budget images through this, so they cannot disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct InternVlLayout {
    /// The processor settings.
    pub params: InternVlParams,
    /// Tower patch edge.
    pub patch: usize,
    /// Pixel-shuffle factor.
    pub merge: usize,
    /// `<IMG_CONTEXT>`.
    pub context: i32,
    /// `<img>`.
    pub img_start: i32,
    /// `</img>`.
    pub img_end: i32,
}

impl InternVlLayout {
    /// Projector rows one tower run contributes.
    pub fn tokens_per_tile(&self) -> usize {
        let side = self.params.tile / self.patch / self.merge;
        side * side
    }

    /// Context positions one `width x height` image takes, from its
    /// dimensions alone: the same count [`internvl_preprocess`] produces.
    pub fn tokens_for(&self, width: usize, height: usize) -> Result<usize, VisionIoError> {
        let (cols, rows) = internvl_tile_grid(
            width,
            height,
            self.params.tile,
            self.params.min_dynamic,
            self.params.max_dynamic,
        )?;
        Ok(tile_runs(cols * rows, &self.params) * self.tokens_per_tile())
    }

    /// [`internvl_expand_image_placeholders`] with whichever marker the
    /// template rendered: bare `<IMG_CONTEXT>` (the `internvl` wrapper) or
    /// `<img>` (`internvl_chat`). A prompt with neither is refused.
    pub fn expand(
        &self,
        rendered: &[i32],
        counts: &[usize],
    ) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
        let marker = if rendered.contains(&self.context) {
            self.context
        } else if rendered.contains(&self.img_start) {
            self.img_start
        } else {
            return Err(VisionIoError::PlaceholderMismatch {
                placeholders: 0,
                grids: counts.len(),
            });
        };
        internvl_expand_image_placeholders(
            rendered,
            marker,
            self.img_start,
            self.img_end,
            self.context,
            counts,
        )
    }
}

/// Normalizes and patchifies one `size x size` crop of `data` into `out`:
/// rows are `(P_h, P_w, C)` over a `(side_patches, side_patches)` grid.
#[allow(clippy::too_many_arguments)]
fn patchify_tile(
    data: &[u8],
    src_w: usize,
    src_h: usize,
    tx: usize,
    ty: usize,
    size: usize,
    patch: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    out: &mut Vec<f32>,
) {
    let side_patches = size / patch;
    for py in 0..side_patches {
        for px in 0..side_patches {
            for r in 0..patch {
                for k in 0..patch {
                    for c in 0..3 {
                        let x = tx + px * patch + k;
                        let y = ty + py * patch + r;
                        let sample = data[(y * src_w + x) * 3 + c] as f32 / 255.0;
                        out.push((sample - mean[c]) / std[c]);
                    }
                }
            }
        }
    }
    let _ = src_h;
}

/// The projector's pixel shuffle, on FP16 BITS (a pure permutation, so no
/// arithmetic and no rounding): `[H * W, C]` tower rows become
/// `[H/f * W/f, C * f * f]`, where row `(h2, w2)`'s channels are the four
/// (f = 2) source rows `(h2*f + dr, w2*f + dc)` concatenated in
/// row-major block order, `dr` outer.
///
/// This is mlx-vlm's `pixel_shuffle` at `shuffle_ratio = 1 / f`, derived:
/// `out[b, h2, w2, mc*C + c] = x[b, h2*f + mc/f, w2*f + mc%f, c]` with `mc`
/// the channel-block index. The reference's reshape/transpose chain reduces
/// to exactly this gather.
pub fn internvl_pixel_shuffle(
    rows: &[u16],
    side_patches: usize,
    channels: usize,
    merge: usize,
) -> Result<Vec<u16>, VisionIoError> {
    if merge == 0 || side_patches == 0 || channels == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "pixel shuffle of an empty grid".into(),
        });
    }
    if side_patches % merge != 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!("{side_patches} patches do not shuffle whole {merge}-wide blocks"),
        });
    }
    let grid = side_patches * side_patches;
    if rows.len() != grid * channels {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{} row values for a {grid}-patch grid of {channels} channels",
                rows.len()
            ),
        });
    }
    let f = merge;
    let side2 = side_patches / f;
    let blocks = f * f;
    let wide = channels * blocks;
    let mut out = vec![0u16; side2 * side2 * wide];
    let at = |h: usize, w: usize| (h * side_patches + w) * channels;
    for h2 in 0..side2 {
        for w2 in 0..side2 {
            let dst = (h2 * side2 + w2) * wide;
            for mc in 0..blocks {
                let src = at(h2 * f + mc / f, w2 * f + mc % f);
                out[dst + mc * channels..dst + (mc + 1) * channels]
                    .copy_from_slice(&rows[src..src + channels]);
            }
        }
    }
    Ok(out)
}

/// Expand a rendered prompt's image markers into the InternVL span:
/// each `marker` becomes `[img_start] + [img_context] * count + [img_end]`,
/// one span per image over the `img_context` run (the only positions the
/// tower's rows replace; the two ends keep their own embeddings).
///
/// `marker` is whichever id the checkpoint's own chat template renders --
/// the `internvl` wrapper's emits bare `<IMG_CONTEXT>` (`img_context`
/// itself), the `internvl_chat` one emits `<img>` (`img_start`), so this
/// accepts either and refuses a marker that names neither. A rendered
/// sequence carrying BOTH spellings is an already-expanded or malformed
/// prompt and is refused.
pub fn internvl_expand_image_placeholders(
    rendered: &[i32],
    marker: i32,
    img_start: i32,
    img_end: i32,
    img_context: i32,
    counts: &[usize],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    if marker != img_context && marker != img_start {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "image marker {marker} is neither the context id {img_context} nor the \
                 start id {img_start} this tower's spans are built from"
            ),
        });
    }
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
    let other = if marker == img_context {
        img_start
    } else {
        img_context
    };
    if rendered.contains(&other) {
        return Err(VisionIoError::InvalidDimensions {
            detail: "the rendered prompt carries both image span spellings; it looks expanded \
                     already"
                .into(),
        });
    }
    let extra: usize = counts
        .iter()
        .map(|&c| c + 2) // start, context run, end; the marker itself is replaced
        .sum();
    let mut out = Vec::with_capacity(rendered.len() + extra);
    let mut spans = Vec::new();
    let mut next = 0usize;
    for &t in rendered {
        if t != marker {
            out.push(t);
            continue;
        }
        let count = counts[next];
        if count == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("image {next} expands to zero tokens"),
            });
        }
        out.push(img_start);
        spans.push(crate::ImageSpan {
            start: out.len(),
            len: count,
        });
        out.extend(std::iter::repeat_n(img_context, count));
        out.push(img_end);
        next += 1;
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: usize, height: usize, rgb: [u8; 3]) -> Rgb8Image {
        Rgb8Image {
            data: vec![rgb; width * height].into_iter().flatten().collect(),
            width,
            height,
        }
    }

    #[test]
    fn the_tile_grid_follows_the_aspect_ratio_table() {
        // Square images: the 1x1 grid, whatever the size.
        assert_eq!(internvl_tile_grid(448, 448, 448, 1, 12).unwrap(), (1, 1));
        // 2:1 wide: the first pair whose ratio matches exactly, lowest area
        // first in the reference's sorted table -> (2, 1).
        assert_eq!(internvl_tile_grid(896, 448, 448, 1, 12).unwrap(), (2, 1));
        // 1:2 tall -> (1, 2).
        assert_eq!(internvl_tile_grid(448, 896, 448, 1, 12).unwrap(), (1, 2));
        // 2000x1000 (2.0) -> (2, 1); the max of 2 caps the table.
        assert_eq!(internvl_tile_grid(2000, 1000, 448, 1, 2).unwrap(), (2, 1));
        // 3:2-ish (1500x1000 = 1.5): exact match (3, 2) within max 12.
        assert_eq!(internvl_tile_grid(1500, 1000, 448, 1, 12).unwrap(), (3, 2));
        // An off ratio snaps to the nearest pair: 1000x751 (1.3315) is
        // closest to 4/3 among areas up to 12.
        assert_eq!(internvl_tile_grid(1000, 751, 448, 1, 12).unwrap(), (4, 3));
        // max_num 1 admits only the square.
        assert_eq!(internvl_tile_grid(896, 448, 448, 1, 1).unwrap(), (1, 1));
        assert!(internvl_tile_grid(0, 448, 448, 1, 12).is_err());
    }

    #[test]
    fn a_preprocess_splits_the_grid_and_appends_the_thumbnail() {
        // 896x448 at tile 448: a 2x1 grid plus a thumbnail = 3 runs of
        // 32x32 patches of 588 values.
        let image = solid(896, 448, [200, 100, 50]);
        let p = internvl_preprocess(&image, 14, &InternVlParams::default()).unwrap();
        assert_eq!(p.tiles, 3);
        assert_eq!((p.side_patches, p.patch_dim), (32, 588));
        assert_eq!(p.rows.len(), 3 * 32 * 32 * 588);
        // Every patch row of the uniform image carries the same normalized
        // triple, (P_h, P_w, C) order, so channel 0 of every row is equal.
        let mean = INTERNVL_MEAN[0];
        let std = INTERNVL_STD[0];
        let want = (200f32 / 255.0 - mean) / std;
        for row in 0..p.rows.len() / 588 {
            assert!((p.rows[row * 588] - want).abs() < 1e-6, "row {row}");
        }
        // use_thumbnail false: no third run.
        let params = InternVlParams {
            use_thumbnail: false,
            ..InternVlParams::default()
        };
        let p = internvl_preprocess(&image, 14, &params).unwrap();
        assert_eq!(p.tiles, 2);
    }

    #[test]
    fn the_tile_crops_are_disjoint_and_ordered() {
        // Left half red, right half blue, 2x1 grid: tile 0 is red, tile 1 is
        // blue, the thumbnail is a half-blend of both.
        let mut data = Vec::new();
        for _y in 0..448usize {
            for x in 0..896usize {
                if x < 448 {
                    data.extend([250u8, 0, 0]);
                } else {
                    data.extend([0u8, 0, 250]);
                }
            }
        }
        let image = Rgb8Image {
            data,
            width: 896,
            height: 448,
        };
        let p = internvl_preprocess(&image, 14, &InternVlParams::default()).unwrap();
        let r_norm = (250f32 / 255.0 - INTERNVL_MEAN[0]) / INTERNVL_STD[0];
        let b_norm = (250f32 / 255.0 - INTERNVL_MEAN[2]) / INTERNVL_STD[2];
        // Tile 0, first patch row, channel 0 (R): 250. Channel 2 (B): 0.
        assert!((p.rows[0] - r_norm).abs() < 1e-6);
        assert!((p.rows[2] - (0f32 - INTERNVL_MEAN[2]) / INTERNVL_STD[2]).abs() < 1e-6);
        // Tile 1 starts at run 1: channel 0 is 0, channel 2 is 250.
        let t1 = 32 * 32 * 588;
        assert!((p.rows[t1] - (0f32 - INTERNVL_MEAN[0]) / INTERNVL_STD[0]).abs() < 1e-6);
        assert!((p.rows[t1 + 2] - b_norm).abs() < 1e-6);
        // The thumbnail (run 2) is the ORIGINAL resized, not a tile crop: its
        // top-left corner patch is still red (bicubic corners sit close to
        // the source color, with overshoot allowed), and its far edge is
        // blue -- values neither tile alone carries at those offsets.
        let t2 = 2 * t1;
        let thumb_r = p.rows[t2];
        assert!(
            (thumb_r - r_norm).abs() < 0.05,
            "thumb corner {thumb_r} vs {r_norm}"
        );
        // Thumbnail patch (0, 15): thumbnail x 210..224 maps to source
        // x 420..448, still red. Patch (0, 31) maps to source x ~868..896,
        // blue. Both differ from the corner.
        let thumb_blue = p.rows[t2 + 31 * 588];
        assert!(
            thumb_blue < (thumb_r + b_norm) / 2.0,
            "thumb far {thumb_blue}"
        );
    }

    #[test]
    fn the_pixel_shuffle_is_the_block_gather_the_reference_derives_to() {
        // 2x2 grid of 1-channel rows: out row (0,0) = rows 0, 1, 2, 3.
        let rows: Vec<u16> = vec![100, 200, 300, 400];
        let out = internvl_pixel_shuffle(&rows, 2, 1, 2).unwrap();
        assert_eq!(out, vec![100, 200, 300, 400]);
        // 4x4 grid: out row (0, 0) takes (0,0),(0,1),(1,0),(1,1) in that
        // channel-block order; out row (0, 1) takes (0,2),(0,3),(1,2),(1,3).
        let rows: Vec<u16> = (0..16).map(|i| i as u16 * 10).collect();
        let out = internvl_pixel_shuffle(&rows, 4, 1, 2).unwrap();
        assert_eq!(out.len(), 4 * 4);
        assert_eq!(&out[0..4], &[0, 10, 40, 50]);
        assert_eq!(&out[4..8], &[20, 30, 60, 70]);
        assert_eq!(&out[8..12], &[80, 90, 120, 130]);
        // Channels survive inside their block: 2 channels, out row (0,0)
        // channels = r(0,0) then r(0,1) then r(1,0) then r(1,1).
        let rows: Vec<u16> = (0..8).map(|i| 1000 + i as u16).collect();
        let out = internvl_pixel_shuffle(&rows, 2, 2, 2).unwrap();
        assert_eq!(out, vec![1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]);
        // Refusals: a ragged grid and an odd side.
        assert!(internvl_pixel_shuffle(&rows, 3, 2, 2).is_err());
        assert!(internvl_pixel_shuffle(&[1, 2, 3], 2, 1, 2).is_err());
    }

    /// MUTATION GUARD: swap the channel-block order (column-major blocks
    /// instead of row-major) and the shuffle must disagree with the hand
    /// values. The 4x4 case above distinguishes (0,1)-before-(1,0) from
    /// (1,0)-before-(0,1) exactly, so this test fails on that mutation.
    #[test]
    fn the_shuffle_block_order_is_row_major_within_the_block() {
        let rows: Vec<u16> = (0..4).map(|i| i as u16).collect();
        let out = internvl_pixel_shuffle(&rows, 2, 1, 2).unwrap();
        // Row-major block order puts (0,1) = 1 BEFORE (1,0) = 2. A
        // column-major mutation would emit 2 before 1.
        assert_eq!(out[1], 1);
        assert_eq!(out[2], 2);
    }

    #[test]
    fn the_expansion_wraps_each_marker_in_the_img_span() {
        // ids: BOS, user text, IMG_CONTEXT, EOS.
        let rendered = [1, 7, 11, 4];
        let (ids, spans) =
            internvl_expand_image_placeholders(&rendered, 11, 10, 12, 11, &[3]).unwrap();
        assert_eq!(ids, vec![1, 7, 10, 11, 11, 11, 12, 4]);
        assert_eq!(spans, vec![crate::ImageSpan { start: 3, len: 3 }]);
        // The internvl_chat template's `<img>` marker: the same output from
        // a different trigger id.
        let rendered = [1, 7, 10, 4];
        let (ids, spans) =
            internvl_expand_image_placeholders(&rendered, 10, 10, 12, 11, &[2]).unwrap();
        assert_eq!(ids, vec![1, 7, 10, 11, 11, 12, 4]);
        assert_eq!(spans, vec![crate::ImageSpan { start: 3, len: 2 }]);
        // Two images: spans stay disjoint and ordered.
        let (ids, spans) =
            internvl_expand_image_placeholders(&[11, 5, 11], 11, 10, 12, 11, &[1, 2]).unwrap();
        assert_eq!(ids.len(), 1 + (1 + 2) + (2 + 2));
        assert_eq!(
            spans,
            vec![
                crate::ImageSpan { start: 1, len: 1 },
                crate::ImageSpan { start: 5, len: 2 }
            ]
        );
    }

    #[test]
    fn a_marker_count_mismatch_or_foreign_marker_is_refused() {
        assert!(internvl_expand_image_placeholders(&[11, 11], 11, 10, 12, 11, &[2]).is_err());
        assert!(internvl_expand_image_placeholders(&[11], 11, 10, 12, 11, &[1, 1]).is_err());
        assert!(internvl_expand_image_placeholders(&[11], 11, 10, 12, 11, &[0]).is_err());
        // A marker that is neither span end.
        assert!(internvl_expand_image_placeholders(&[9], 9, 10, 12, 11, &[1]).is_err());
        // Both spellings present: already expanded.
        assert!(internvl_expand_image_placeholders(&[10, 11], 11, 10, 12, 11, &[2]).is_err());
    }

    #[test]
    fn the_params_reader_takes_the_file_and_defaults_the_rest() {
        let p = InternVlParams::from_preprocessor_config_json(
            r#"{"crop_size": 448, "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225], "size": 448}"#,
        )
        .unwrap();
        assert_eq!(p.tile, 448);
        assert_eq!(
            (p.max_dynamic, p.min_dynamic, p.use_thumbnail),
            (12, 1, true)
        );
        assert_eq!(p.mean, INTERNVL_MEAN);
        // The InternVL3.5 conversion's processor_config names only the token
        // length: everything defaults.
        let p = InternVlParams::from_preprocessor_config_json(
            r#"{"image_seq_length": 256, "processor_class": "InternVLProcessor"}"#,
        )
        .unwrap();
        assert_eq!(p, InternVlParams::default());
        assert!(InternVlParams::from_preprocessor_config_json("not json").is_err());
        assert!(InternVlParams::from_preprocessor_config_json(r#"{"size": 0}"#).is_err());
    }
}
