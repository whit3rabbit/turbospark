//! Preprocessing for the Inkling 2 Mini vision tower
//! (`docs/INKLING2_VISION_PHASE0.md`).
//!
//! The reference never resizes. An image is padded on the bottom and right
//! to a multiple of 40 px and cut into 40x40 tiles, one trunk token per tile.
//! Each tile is 2x2 native 20x20 patches, so the tower sees a `2R x 2C` patch
//! grid. What differs from the rest of this crate:
//!
//! - padding is -1 on the 0-255 scale, BEFORE normalization (it lands at
//!   -1.00784, not -1);
//! - normalization is the fused form `(x - 127.5) / 127.5`;
//! - an image with transparency is composited onto an 8 px checkerboard of
//!   204 and 153, not dropped to RGB;
//! - patch rows are emitted in **merge-block order** (tile row, tile col, row
//!   in tile, col in tile), each row `[20 rows][20 cols][RGB]`, the order
//!   `patch_embedding.weight` `[1024, 1200]` reads.
//!
//! The one departure from the reference is the cap: above
//! [`INKLING2_MAX_TOKENS`] tiles (or a smaller per-session budget) the image is
//! downscaled with its aspect ratio kept, PIL bicubic, each side floored to a
//! multiple of 40 px. llama.cpp caps at the same 4,096 tokens.

use crate::decode::{image_dimensions, Rgb8Image};
use crate::error::VisionIoError;
use crate::patchify::GridThw;
use crate::resize::resize_bicubic_pil;

/// Pixels per tile side; one tile is one trunk token.
pub const INKLING2_TILE: usize = 40;
/// Native patch side.
pub const INKLING2_PATCH: usize = 20;
/// Patches per tile side.
pub const INKLING2_MERGE: usize = INKLING2_TILE / INKLING2_PATCH;
/// Values per patch row: `20 * 20 * 3`.
pub const INKLING2_PATCH_DIM: usize = INKLING2_PATCH * INKLING2_PATCH * 3;
/// Tiles (trunk tokens) per image above which the image is downscaled.
pub const INKLING2_MAX_TOKENS: usize = 4_096;
/// The image placeholder, `<|unused_200054|>`.
pub const INKLING2_IMAGE_TOKEN_ID: i32 = 200_054;
/// `<|content_image|>`, which the template renders right before it.
pub const INKLING2_CONTENT_IMAGE_ID: i32 = 200_005;

/// The value padding takes on the 0-255 scale, before normalization.
const PAD_SAMPLE: f32 = -1.0;

/// The checkerboard a transparent pixel is composited onto: 204 where
/// `(y / 8 + x / 8)` is even, else 153.
fn checkerboard(y: usize, x: usize) -> u32 {
    if (y / 8 + x / 8) % 2 == 0 {
        204
    } else {
        153
    }
}

/// Composite interleaved RGBA8 onto the checkerboard:
/// `(rgb * a + bg * (255 - a) + 127) // 255`, integer arithmetic, as the
/// reference's `convert_to_rgb` does.
pub fn inkling2_composite_rgba(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Rgb8Image, VisionIoError> {
    if rgba.len() != width * height * 4 {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{} samples for a {width}x{height} RGBA image (expected {})",
                rgba.len(),
                width * height * 4
            ),
        });
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let px = &rgba[(y * width + x) * 4..][..4];
            let a = u32::from(px[3]);
            let bg = checkerboard(y, x);
            for &c in &px[..3] {
                rgb.push(((u32::from(c) * a + bg * (255 - a) + 127) / 255) as u8);
            }
        }
    }
    Rgb8Image::new(width, height, rgb)
}

/// Decode an image the way the Inkling 2 processor sees it: an image with an
/// alpha channel is composited onto the checkerboard, anything else converts
/// to RGB. The shared [`crate::decode_image_bytes`] drops alpha instead,
/// which matches the other towers' `convert("RGB")` but not this one.
pub fn inkling2_decode_image_bytes(bytes: &[u8]) -> Result<Rgb8Image, VisionIoError> {
    // Same dimension guard as the shared decoder, before any allocation.
    image_dimensions(bytes)?;
    let decoded = image::load_from_memory(bytes).map_err(|e| VisionIoError::Decode {
        detail: e.to_string(),
    })?;
    if decoded.color().has_alpha() {
        let rgba = decoded.to_rgba8();
        let (w, h) = rgba.dimensions();
        inkling2_composite_rgba(w as usize, h as usize, rgba.as_raw())
    } else {
        let rgb = decoded.to_rgb8();
        let (w, h) = rgb.dimensions();
        Rgb8Image::new(w as usize, h as usize, rgb.into_raw())
    }
}

/// `(rows, cols)` of tiles for an image: `ceil(h / 40)`, `ceil(w / 40)`.
pub fn inkling2_tile_grid(height: usize, width: usize) -> (usize, usize) {
    (
        height.div_ceil(INKLING2_TILE),
        width.div_ceil(INKLING2_TILE),
    )
}

/// The `(height, width)` an image is preprocessed at under a `max_tokens`
/// cap: unchanged when its tile grid fits, else downscaled with the aspect
/// ratio kept and each side floored to a multiple of 40 px (so the result
/// needs no padding), at least one tile per side.
pub fn inkling2_capped_dims(
    height: usize,
    width: usize,
    max_tokens: usize,
) -> Result<(usize, usize), VisionIoError> {
    if height == 0 || width == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "image has a zero dimension".into(),
        });
    }
    if max_tokens == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "a cap of zero image tokens admits no image".into(),
        });
    }
    let (r, c) = inkling2_tile_grid(height, width);
    if r * c <= max_tokens {
        return Ok((height, width));
    }
    // (h / beta) * (w / beta) == max_tokens * 40^2, so flooring each side to
    // a multiple of 40 can only land at or under the cap. One exception: a
    // side that floors below one tile is raised to one, and then the other
    // side is clamped so the product still fits.
    let tile = INKLING2_TILE as f64;
    let beta = ((height * width) as f64 / (max_tokens as f64 * tile * tile)).sqrt();
    let mut rows = ((height as f64 / beta) / tile).floor().max(1.0) as usize;
    let mut cols = ((width as f64 / beta) / tile).floor().max(1.0) as usize;
    if rows * cols > max_tokens {
        if rows <= cols {
            cols = (max_tokens / rows).max(1);
        } else {
            rows = (max_tokens / cols).max(1);
        }
    }
    debug_assert!(rows * cols <= max_tokens);
    Ok((rows * INKLING2_TILE, cols * INKLING2_TILE))
}

/// One image as tower input.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Patches {
    /// `[4 * rows * cols, 1200]` normalized patch rows, merge-block order,
    /// each `[20 rows][20 cols][RGB]`.
    pub pixels: Vec<f32>,
    /// Tile rows (trunk-token rows).
    pub rows: usize,
    /// Tile columns.
    pub cols: usize,
    /// The `(height, width)` actually tiled, after any cap downscale.
    pub tiled_dims: (usize, usize),
}

impl Inkling2Patches {
    /// Trunk tokens this image takes: one per tile.
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }

    /// Rows of the patch matrix: four per tile.
    pub fn patches(&self) -> usize {
        self.tokens() * INKLING2_MERGE * INKLING2_MERGE
    }

    /// The native patch grid, `[1, 2R, 2C]`.
    pub fn patch_grid(&self) -> GridThw {
        GridThw::new(1, self.rows * INKLING2_MERGE, self.cols * INKLING2_MERGE)
    }
}

/// Preprocess an RGB image: cap, pad, normalize, and cut into patch rows.
pub fn inkling2_preprocess(
    image: &Rgb8Image,
    max_tokens: usize,
) -> Result<Inkling2Patches, VisionIoError> {
    let (h, w) = inkling2_capped_dims(image.height, image.width, max_tokens)?;
    let resized;
    let src: &[u8] = if (h, w) == (image.height, image.width) {
        &image.data
    } else {
        resized = resize_bicubic_pil(&image.data, image.width, image.height, w, h);
        &resized
    };
    let (rows, cols) = inkling2_tile_grid(h, w);
    let norm = |v: f32| (v - 127.5) / 127.5;
    let pad = norm(PAD_SAMPLE);
    let mut pixels = Vec::with_capacity(rows * cols * 4 * INKLING2_PATCH_DIM);
    for tr in 0..rows {
        for tc in 0..cols {
            for pr in 0..INKLING2_MERGE {
                for pc in 0..INKLING2_MERGE {
                    let y0 = tr * INKLING2_TILE + pr * INKLING2_PATCH;
                    let x0 = tc * INKLING2_TILE + pc * INKLING2_PATCH;
                    for y in y0..y0 + INKLING2_PATCH {
                        for x in x0..x0 + INKLING2_PATCH {
                            if y < h && x < w {
                                let at = (y * w + x) * 3;
                                for ch in 0..3 {
                                    pixels.push(norm(f32::from(src[at + ch])));
                                }
                            } else {
                                pixels.extend_from_slice(&[pad; 3]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Inkling2Patches {
        pixels,
        rows,
        cols,
        tiled_dims: (h, w),
    })
}

/// Expand each rendered `<|unused_200054|>` to one copy per tile. The
/// template renders `<|content_image|>` right before every placeholder; a
/// placeholder without it is refused, since it would mean a template this
/// port has not read.
pub fn inkling2_expand_image_placeholders(
    rendered: &[i32],
    token_counts: &[usize],
) -> Result<(Vec<i32>, Vec<crate::ImageSpan>), VisionIoError> {
    for (i, &t) in rendered.iter().enumerate() {
        if t == INKLING2_IMAGE_TOKEN_ID && (i == 0 || rendered[i - 1] != INKLING2_CONTENT_IMAGE_ID)
        {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!(
                    "image placeholder {INKLING2_IMAGE_TOKEN_ID} at {i} is not preceded by \
                     <|content_image|> ({INKLING2_CONTENT_IMAGE_ID})"
                ),
            });
        }
    }
    crate::clip::clip_expand_image_placeholders(
        rendered,
        INKLING2_IMAGE_TOKEN_ID,
        INKLING2_IMAGE_TOKEN_ID,
        token_counts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, v: u8) -> Rgb8Image {
        Rgb8Image::new(w, h, vec![v; w * h * 3]).unwrap()
    }

    #[test]
    fn the_grid_is_a_ceil_over_40() {
        assert_eq!(inkling2_tile_grid(40, 40), (1, 1));
        assert_eq!(inkling2_tile_grid(41, 40), (2, 1));
        assert_eq!(inkling2_tile_grid(83, 121), (3, 4));
        assert_eq!(inkling2_tile_grid(1, 1), (1, 1));
    }

    #[test]
    fn padding_lands_below_minus_one() {
        let p = inkling2_preprocess(&solid(21, 21, 255), INKLING2_MAX_TOKENS).unwrap();
        assert_eq!((p.rows, p.cols, p.patches()), (1, 1, 4));
        assert_eq!(p.pixels.len(), 4 * INKLING2_PATCH_DIM);
        // Patch (0, 0) row 0 col 0 is a real white pixel.
        assert_eq!(p.pixels[0], 1.0);
        // Patch (0, 0) row 0 col 20 does not exist; patch (0, 1) col 0 is
        // x = 20, a real pixel; its col 1 (x = 21) is padding.
        let patch01 = &p.pixels[INKLING2_PATCH_DIM..][..INKLING2_PATCH_DIM];
        assert_eq!(patch01[0], 1.0);
        let pad = (-1.0f32 - 127.5) / 127.5;
        assert_eq!(patch01[3], pad);
        assert!((pad + 1.007_843).abs() < 1e-5);
    }

    #[test]
    fn rows_are_merge_block_order_and_hwc() {
        // A 80x40 image (one row of two tiles) whose red channel is x and
        // green is y: the third patch row is tile (0, 1), patch (0, 0).
        let (w, h) = (80usize, 40usize);
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[x as u8, y as u8, 7]);
            }
        }
        let img = Rgb8Image::new(w, h, data).unwrap();
        let p = inkling2_preprocess(&img, INKLING2_MAX_TOKENS).unwrap();
        assert_eq!((p.rows, p.cols), (1, 2));
        let denorm = |v: f32| (v * 127.5 + 127.5).round() as u8;
        let row = |i: usize| &p.pixels[i * INKLING2_PATCH_DIM..][..INKLING2_PATCH_DIM];
        // Patch order inside tile (0, 0): (0,0) x 0.., (0,1) x 20.., (1,0) y 20..
        assert_eq!(denorm(row(1)[0]), 20);
        assert_eq!(denorm(row(2)[1]), 20);
        // Patch row 4 is tile (0, 1) patch (0, 0): x starts at 40.
        assert_eq!(denorm(row(4)[0]), 40);
        // Inside a row: (row, col, C), so sample 3 is x + 1, sample 60 is y + 1.
        assert_eq!(denorm(row(0)[3]), 1);
        assert_eq!(denorm(row(0)[60 + 1]), 1);
        assert_eq!(denorm(row(0)[2]), 7);
    }

    #[test]
    fn the_cap_keeps_aspect_and_fits() {
        assert_eq!(inkling2_capped_dims(640, 640, 4096).unwrap(), (640, 640));
        let (h, w) = inkling2_capped_dims(4096, 4096, 4096).unwrap();
        let (r, c) = inkling2_tile_grid(h, w);
        assert!(r * c <= 4096, "{r}x{c}");
        assert_eq!((h % 40, w % 40), (0, 0));
        assert_eq!(h, w);
        // An extreme strip still fits and keeps one tile on its short side.
        let (h, w) = inkling2_capped_dims(30, 4096 * 40, 16).unwrap();
        let (r, c) = inkling2_tile_grid(h, w);
        assert_eq!(r, 1);
        assert!(r * c <= 16);
        // A small session budget clamps further.
        let (h, w) = inkling2_capped_dims(800, 1200, 100).unwrap();
        let (r, c) = inkling2_tile_grid(h, w);
        assert!(r * c <= 100 && r * c >= 90, "{r}x{c}");
        assert!(inkling2_capped_dims(0, 10, 10).is_err());
        assert!(inkling2_capped_dims(10, 10, 0).is_err());
    }

    #[test]
    fn transparency_composites_onto_the_checkerboard() {
        // Fully transparent pixels take the board; opaque ones keep their
        // colour; half alpha blends with the reference's integer rounding.
        let rgba = vec![
            10, 20, 30, 0, // (0, 0): board 204
            10, 20, 30, 255, // (0, 1): opaque
        ];
        let img = inkling2_composite_rgba(2, 1, &rgba).unwrap();
        assert_eq!(img.data, vec![204, 204, 204, 10, 20, 30]);
        let half = inkling2_composite_rgba(1, 1, &[0, 0, 0, 128]).unwrap();
        assert_eq!(half.data[0], ((204 * 127 + 127) / 255) as u8);
        // The board flips every 8 px.
        assert_eq!(checkerboard(0, 8), 153);
        assert_eq!(checkerboard(8, 8), 204);
    }

    #[test]
    fn placeholders_expand_to_one_per_tile_and_need_the_content_mark() {
        let rendered = [1, INKLING2_CONTENT_IMAGE_ID, INKLING2_IMAGE_TOKEN_ID, 2];
        let (ids, spans) = inkling2_expand_image_placeholders(&rendered, &[6]).unwrap();
        assert_eq!(ids.len(), 4 + 5);
        assert_eq!(spans[0].start, 2);
        assert_eq!(spans[0].len, 6);
        assert!(ids[2..8].iter().all(|&t| t == INKLING2_IMAGE_TOKEN_ID));
        let bare = [1, INKLING2_IMAGE_TOKEN_ID, 2];
        assert!(inkling2_expand_image_placeholders(&bare, &[1]).is_err());
        assert!(inkling2_expand_image_placeholders(&rendered, &[1, 2]).is_err());
    }
}
