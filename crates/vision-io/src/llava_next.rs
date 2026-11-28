//! Preprocessing, anyres tiling and row packing for LLaVA 1.6 (LLaVA-NeXT,
//! `docs/LLAVA_NEXT_VISION_PHASE0.md`).
//!
//! The reference is `transformers`' `LlavaNextImageProcessor` and
//! `LlavaNextForConditionalGeneration.pack_image_features` (read, never
//! executed). It shares the grid choice, the aspect-kept resize onto a zero
//! canvas and the unpad rule with [`crate::onevision`], and differs in:
//!
//! - the tile: 336 pixels, 24 x 24 patches (not 384 and 27 x 27);
//! - no pooling: the unpadded map is used at full size;
//! - the run: the base tile's 576 rows with NO newline after them, then the
//!   grid map with `image_newline` after every row.
//!
//! Tokens per image = 576 + H * W + H for the unpadded `(H, W)` map.
//! [`ClipPatches::pixels`] holds the base tile first, then the grid's tiles.

use crate::clip::{ClipPatches, CLIP_PATCH, CLIP_PATCH_DIM};
use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::onevision::{
    centered_canvas, ov_select_best_resolution, push_tile, unpad_window, Anyres,
};
use crate::resize::resize_bicubic_pil;

/// Side of one tile, in pixels.
pub const NEXT_TILE: usize = 336;
/// Patches per tile side.
pub const NEXT_SIDE: usize = NEXT_TILE / CLIP_PATCH;
/// Patches (and projected rows) per tile.
pub const NEXT_TILE_PATCHES: usize = NEXT_SIDE * NEXT_SIDE;

/// The tile grid `(rows, cols)` for an image of the given size.
fn next_grid(
    orig_h: usize,
    orig_w: usize,
    pinpoints: &[(usize, usize)],
) -> Result<(usize, usize), VisionIoError> {
    if orig_h == 0 || orig_w == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    let bad = |why: String| VisionIoError::BadConfig {
        field: "image_grid_pinpoints".into(),
        why,
    };
    let (h, w) = ov_select_best_resolution(orig_h, orig_w, pinpoints)
        .ok_or_else(|| bad("no pinpoint fits this image".into()))?;
    if h % NEXT_TILE != 0 || w % NEXT_TILE != 0 {
        return Err(bad(format!(
            "{h}x{w} is not a whole number of {NEXT_TILE}-pixel tiles"
        )));
    }
    Ok((h / NEXT_TILE, w / NEXT_TILE))
}

/// The unpadded map `(y0, y1, x0, x1)` for an image on a tile grid.
fn window(orig_h: usize, orig_w: usize, tiles_h: usize, tiles_w: usize) -> [usize; 4] {
    let (y0, y1, x0, x1) = unpad_window(orig_h, orig_w, tiles_h * NEXT_SIDE, tiles_w * NEXT_SIDE);
    [y0, y1, x0, x1]
}

fn rows_for([y0, y1, x0, x1]: [usize; 4]) -> usize {
    NEXT_TILE_PATCHES + (y1 - y0) * (x1 - x0) + (y1 - y0)
}

/// Tokens one image occupies in the prompt, from its size alone.
pub fn llava_next_image_tokens(
    orig_h: usize,
    orig_w: usize,
    pinpoints: &[(usize, usize)],
) -> Result<usize, VisionIoError> {
    let (th, tw) = next_grid(orig_h, orig_w, pinpoints)?;
    Ok(rows_for(window(orig_h, orig_w, th, tw)))
}

/// Resize, pad, tile, normalize and patchify one decoded image.
///
/// `grid` and `grid_w` are the prompt's placeholder count as `(tokens, 1)`,
/// as for the other anyres layout.
pub fn llava_next_preprocess(
    image: &Rgb8Image,
    pinpoints: &[(usize, usize)],
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
) -> Result<ClipPatches, VisionIoError> {
    let (oh, ow) = (image.height, image.width);
    let (tiles_h, tiles_w) = next_grid(oh, ow, pinpoints)?;
    let (target_h, target_w) = (tiles_h * NEXT_TILE, tiles_w * NEXT_TILE);
    let mut pixels =
        Vec::with_capacity((1 + tiles_h * tiles_w) * NEXT_TILE_PATCHES * CLIP_PATCH_DIM);

    // The base tile: a squash to the tile size (not 1.5's center crop).
    let base;
    let base_data: &[u8] = if (ow, oh) == (NEXT_TILE, NEXT_TILE) {
        &image.data
    } else {
        base = resize_bicubic_pil(&image.data, ow, oh, NEXT_TILE, NEXT_TILE);
        &base
    };
    push_tile(
        NEXT_SIDE,
        &mut pixels,
        base_data,
        NEXT_TILE,
        0,
        0,
        mean,
        std,
        rescale,
    );

    let canvas = centered_canvas(image, target_h, target_w);
    for ty in 0..tiles_h {
        for tx in 0..tiles_w {
            push_tile(
                NEXT_SIDE,
                &mut pixels,
                &canvas,
                target_w,
                ty * NEXT_TILE,
                tx * NEXT_TILE,
                mean,
                std,
                rescale,
            );
        }
    }

    Ok(ClipPatches {
        pixels,
        grid: rows_for(window(oh, ow, tiles_h, tiles_w)),
        grid_w: 1,
        anyres: Some(Anyres {
            orig_h: oh,
            orig_w: ow,
            tiles_h,
            tiles_w,
        }),
    })
}

/// Pack the tower's projected rows into the prompt run.
///
/// `rows` is `(1 + tiles_h * tiles_w) * 576` rows of `dim` values (the base
/// tile first, then the tiles row-major).
pub fn llava_next_pack_features<T: Copy>(
    rows: &[T],
    dim: usize,
    newline: &[T],
    anyres: &Anyres,
) -> Result<Vec<T>, VisionIoError> {
    let tiles = anyres.tiles_h * anyres.tiles_w;
    if rows.len() != (1 + tiles) * NEXT_TILE_PATCHES * dim || newline.len() != dim {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{} values for {tiles} tiles of {dim}-wide rows and a {}-wide newline",
                rows.len(),
                newline.len()
            ),
        });
    }
    let win = window(anyres.orig_h, anyres.orig_w, anyres.tiles_h, anyres.tiles_w);
    let [y0, y1, x0, x1] = win;
    let mut out = Vec::with_capacity(rows_for(win) * dim);
    out.extend_from_slice(&rows[..NEXT_TILE_PATCHES * dim]);
    let grid = &rows[NEXT_TILE_PATCHES * dim..];
    for y in y0..y1 {
        let (ty, i) = (y / NEXT_SIDE, y % NEXT_SIDE);
        for x in x0..x1 {
            let (tx, j) = (x / NEXT_SIDE, x % NEXT_SIDE);
            let row = (ty * anyres.tiles_w + tx) * NEXT_TILE_PATCHES + i * NEXT_SIDE + j;
            out.extend_from_slice(&grid[row * dim..(row + 1) * dim]);
        }
        out.extend_from_slice(newline);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{ClipLayout, ClipPreprocessParams};

    /// `llava-hf/llava-v1.6-vicuna-7b-hf`'s `preprocessor_config.json`.
    const PUBLISHED: &str = r#"{
        "aspect_ratio_setting": "anyres",
        "crop_size": {"height": 336, "width": 336},
        "do_center_crop": true, "do_convert_rgb": true, "do_normalize": true,
        "do_pad": true, "do_rescale": true, "do_resize": true,
        "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "LlavaNextImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3, "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 336}
    }"#;

    fn params() -> ClipPreprocessParams {
        ClipPreprocessParams::from_preprocessor_config_json(PUBLISHED).unwrap()
    }

    #[test]
    fn the_published_config_takes_the_next_layout_and_never_the_1_5_crop() {
        let p = params();
        assert_eq!(p.layout, ClipLayout::LlavaNext);
        assert_eq!((p.size, p.grid(), p.patch_dim()), (336, 24, 588));
        // Base plus a 2x2 grid of patch rows.
        assert_eq!(p.tokens(), 5 * 576);
        assert_eq!(p.max_source_pixels(), 672 * 672);
        assert!(p.check_against_tower(14, 577).is_ok());
        assert!(p.check_against_tower(14, 729).is_err());
        for (from, to) in [
            (r#""resample": 3"#, r#""resample": 2"#),
            (r#""do_pad": true"#, r#""do_pad": false"#),
            (r#""shortest_edge": 336"#, r#""shortest_edge": 224"#),
            (r#"[336, 1008]"#, r#"[336, 1000]"#),
            (r#""anyres""#, r#""pad""#),
        ] {
            let text = PUBLISHED.replace(from, to);
            assert_ne!(text, PUBLISHED, "{from}");
            assert!(
                ClipPreprocessParams::from_preprocessor_config_json(&text).is_err(),
                "{from} -> {to}"
            );
        }
    }

    /// Expected counts from an independent python transcription of the
    /// processor's `_get_number_of_features`, minus the class token.
    #[test]
    fn token_counts_match_the_reference_processor() {
        let p = params();
        for (h, w, want) in [
            (336, 336, 1176),
            (336, 672, 1752),
            (480, 640, 2340),
            (448, 448, 2928),
            (336, 448, 1368),
            (1280, 1024, 2448),
            (1000, 300, 2232),
            (300, 1000, 2182),
            (100, 1000, 1160),
            (1, 2000, 576),
            (700, 500, 2256),
        ] {
            assert_eq!(
                llava_next_image_tokens(h, w, &p.pinpoints).unwrap(),
                want,
                "{h}x{w}"
            );
            assert_eq!(p.grid_for(w, h), (want, 1), "{h}x{w}");
        }
        assert!(llava_next_image_tokens(0, 10, &p.pinpoints).is_err());
    }

    #[test]
    fn preprocessing_emits_the_base_then_the_tiles_row_major() {
        let p = params();
        // 672 x 672 is a 2x2 grid with no resize and no pad, so every tile is
        // one quadrant. One flat red value per quadrant names it.
        let side = 672usize;
        let mut data = vec![0u8; side * side * 3];
        for y in 0..side {
            for x in 0..side {
                data[(y * side + x) * 3] = 50 * (1 + 2 * (y / 336) + x / 336) as u8;
            }
        }
        let img = Rgb8Image {
            width: side,
            height: side,
            data,
        };
        let out = crate::clip::clip_preprocess(&img, &p).unwrap();
        let anyres = out.anyres.expect("anyres");
        assert_eq!((anyres.tiles_h, anyres.tiles_w), (2, 2));
        assert_eq!(out.num_patches(), 2928);
        assert_eq!(out.pixels.len(), 5 * 576 * 588);
        let red = |v: u8| (v as f32 * p.rescale_factor - p.image_mean[0]) / p.image_std[0];
        for (tile, level) in [50u8, 100, 150, 200].into_iter().enumerate() {
            // The center patch of tile `tile + 1`, away from resampling.
            let at = ((tile + 1) * 576 + 12 * 24 + 12) * 588;
            assert_eq!(out.pixels[at], red(level), "tile {tile}");
        }
    }

    #[test]
    fn the_base_tile_squashes_and_the_grid_pads_with_black() {
        let p = params();
        // 448 wide by 336 high, with the leftmost 56 columns marked.
        let (width, height) = (448usize, 336usize);
        let mut data = vec![200u8; width * height * 3];
        for y in 0..height {
            for x in 0..56 {
                data[(y * width + x) * 3] = 20;
            }
        }
        let out = crate::clip::clip_preprocess(
            &Rgb8Image {
                width,
                height,
                data,
            },
            &p,
        )
        .unwrap();
        let red = |v: u8| (v as f32 * p.rescale_factor - p.image_mean[0]) / p.image_std[0];
        // The whole width is squashed into the base, so the mark survives in
        // its first patch column. A center crop would have cut it off.
        assert_eq!(out.pixels[12 * 24 * 588], red(20));
        // The 1 x 2 canvas is 672 wide against a 448 wide image: 112 columns
        // of zero pixels, then the mark.
        let first_tile = &out.pixels[576 * 588..];
        assert_eq!(first_tile[0], red(0), "the pad is black before normalizing");
        assert_eq!(first_tile[9 * 588], red(20), "column 126 is the image");
    }

    #[test]
    fn packing_places_tiles_unpads_and_ends_each_row_with_the_newline() {
        // 336 x 448 on a 1 x 2 grid: the 24 x 48 map crops to columns 8..40.
        let anyres = Anyres {
            orig_h: 336,
            orig_w: 448,
            tiles_h: 1,
            tiles_w: 2,
        };
        // One value per row: tile * 1000 + row within the tile.
        let rows: Vec<f32> = (0..3 * 576)
            .map(|r| ((r / 576) * 1000 + r % 576) as f32)
            .collect();
        let out = llava_next_pack_features(&rows, 1, &[-1.0], &anyres).unwrap();
        assert_eq!(out.len(), 576 + 24 * 32 + 24);
        assert_eq!(&out[..576], &rows[..576], "base first, no newline");
        let line = &out[576..576 + 33];
        assert_eq!(line[0], 1008.0, "first tile, column 8");
        assert_eq!(line[15], 1023.0);
        assert_eq!(line[16], 2000.0, "second tile, column 0");
        assert_eq!(line[31], 2015.0);
        assert_eq!(line[32], -1.0);
        assert_eq!(out[576 + 33], 1000.0 + 24.0 + 8.0, "second map row");
        assert_eq!(*out.last().unwrap(), -1.0);
        assert!(llava_next_pack_features(&rows[1..], 1, &[-1.0], &anyres).is_err());
    }

    #[test]
    fn packing_a_two_by_two_grid_indexes_tiles_row_major() {
        let anyres = Anyres {
            orig_h: 672,
            orig_w: 672,
            tiles_h: 2,
            tiles_w: 2,
        };
        let rows: Vec<u32> = (0..5 * 576).map(|r| r as u32).collect();
        let out = llava_next_pack_features(&rows, 1, &[u32::MAX], &anyres).unwrap();
        // Map row 24 starts in grid tile 2 (row 1, column 0), which is the
        // fourth tile of `rows` once the base is counted.
        assert_eq!(out[576 + 24 * 49], 3 * 576);
        assert_eq!(out[576 + 24], 2 * 576, "map row 0 crosses into tile 1");
    }
}
