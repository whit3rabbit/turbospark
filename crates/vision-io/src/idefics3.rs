//! Preprocessing, pixel shuffle and prompt layout for the variable-size SigLIP
//! tower the Idefics3 and SmolVLM wrappers carry
//! (`docs/IDEFICS3_VISION_PHASE0.md`).
//!
//! The reference is HF `Idefics3ImageProcessor` and the mlx-vlm
//! `processing_idefics3.py` prompt builder (read, never executed):
//!
//! 1. the LONGEST edge is set to `size.longest_edge` (small images are
//!    upscaled), the other edge truncated and then rounded up to even;
//! 2. both edges are rounded up to a multiple of the tile edge
//!    `max_image_size.longest_edge`, the shorter one derived from the longer;
//! 3. the result is cut into `rows x cols` tiles, row-major, and a "global"
//!    copy resized to one tile is appended LAST. An image that is exactly one
//!    tile (or `do_image_splitting: false`) is a single frame with
//!    `rows = cols = 0`.
//!
//! Every resize is PIL Lanczos, applied to the previous step's output.
//!
//! **Every frame is a full square tile**, so the tower never sees padding.
//! The reference rescales position ids by each frame's valid patch count (the
//! NaViT rule); for a full frame that is the identity, so no position
//! bucketing exists here or in the runtime.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::mrope::ImageSpan;
use crate::resize::resize_lanczos_pil;

/// PIL's resampling code for Lanczos, which every published
/// `preprocessor_config.json` of this processor states.
const PIL_LANCZOS: i64 = 1;
/// The reference's absolute cap on the resized long edge.
const MAX_LONG_EDGE: usize = 4096;
/// Most tiles along one edge this port will allocate for. The published
/// processors give 4 or 5.
const MAX_TILES_PER_EDGE: usize = 8;

/// The processor settings read off `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Idefics3Params {
    /// `size.longest_edge`.
    pub longest_edge: usize,
    /// `max_image_size.longest_edge`: one tile's edge.
    pub tile: usize,
    /// `do_image_splitting`.
    pub split: bool,
    /// Normalization mean, per channel.
    pub mean: [f32; 3],
    /// Normalization standard deviation, per channel.
    pub std: [f32; 3],
}

impl Idefics3Params {
    /// Read the published file. Both edges and the normalization pair are
    /// REQUIRED: the three published checkpoints disagree about them, so a
    /// default would silently preprocess for another checkpoint.
    pub fn from_preprocessor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let bad = |field: &str, why: String| VisionIoError::BadConfig {
            field: field.into(),
            why,
        };
        let v: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| bad("preprocessor_config", format!("not JSON: {e}")))?;
        let edge = |field: &str| -> Result<usize, VisionIoError> {
            v.get(field)
                .and_then(|s| s.get("longest_edge"))
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .filter(|&n| n > 0)
                .ok_or_else(|| bad(field, "missing a positive longest_edge".into()))
        };
        let triple = |field: &str| -> Result<[f32; 3], VisionIoError> {
            let items = v
                .get(field)
                .and_then(serde_json::Value::as_array)
                .filter(|a| a.len() == 3)
                .ok_or_else(|| bad(field, "missing three numbers".into()))?;
            let mut out = [0.0f32; 3];
            for (slot, item) in out.iter_mut().zip(items) {
                *slot = item
                    .as_f64()
                    .ok_or_else(|| bad(field, "not a number".into()))?
                    as f32;
            }
            Ok(out)
        };
        for field in ["do_resize", "do_rescale", "do_normalize"] {
            if v.get(field).and_then(serde_json::Value::as_bool) == Some(false) {
                return Err(bad(field, "false is not implemented".into()));
            }
        }
        if let Some(r) = v.get("resample").and_then(serde_json::Value::as_i64) {
            if r != PIL_LANCZOS {
                return Err(bad(
                    "resample",
                    format!("{r}; only PIL Lanczos (1) is implemented here"),
                ));
            }
        }
        if let Some(f) = v.get("rescale_factor").and_then(serde_json::Value::as_f64) {
            if (f - 1.0 / 255.0).abs() > 1e-9 {
                return Err(bad("rescale_factor", format!("{f}; only 1/255")));
            }
        }
        let params = Self {
            longest_edge: edge("size")?,
            tile: edge("max_image_size")?,
            split: v
                .get("do_image_splitting")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            mean: triple("image_mean")?,
            std: triple("image_std")?,
        };
        if params.longest_edge > MAX_LONG_EDGE
            || params.longest_edge.div_ceil(params.tile) > MAX_TILES_PER_EDGE
        {
            return Err(bad(
                "size",
                format!(
                    "longest edge {} against tile {} is past what this port tiles",
                    params.longest_edge, params.tile
                ),
            ));
        }
        if params.std.contains(&0.0) {
            return Err(bad("image_std", "zero".into()));
        }
        Ok(params)
    }
}

/// Step 1: `(width, height)` after setting the longest edge to `longest`.
pub fn idefics3_longest_dims(width: usize, height: usize, longest: usize) -> (usize, usize) {
    let aspect = width as f64 / height as f64;
    let even = |n: usize| n + n % 2;
    if width >= height {
        (longest, even((longest as f64 / aspect) as usize).max(1))
    } else {
        (even((longest as f64 * aspect) as usize).max(1), longest)
    }
}

/// Step 2: `(width, height)` rounded up to whole tiles.
pub fn idefics3_tile_dims(width: usize, height: usize, tile: usize) -> (usize, usize) {
    let aspect = width as f64 / height as f64;
    let up = |n: usize| n.div_ceil(tile) * tile;
    if width >= height {
        let w = up(width);
        (w, up((w as f64 / aspect) as usize))
    } else {
        let h = up(height);
        (up((h as f64 * aspect) as usize), h)
    }
}

/// One image's frames, ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct Idefics3Tiles {
    /// `[frames * patches_per_side^2, 3 * patch^2]` normalized values, frames
    /// in reference order (tiles row-major, then the global view), each patch
    /// row `(P_h, P_w, C)`.
    pub pixels: Vec<f32>,
    /// Tower runs for this image.
    pub frames: usize,
    /// Tile rows, 0 for a single-frame image.
    pub rows: usize,
    /// Tile columns, 0 for a single-frame image.
    pub cols: usize,
    /// `tile / patch`, floored: the unpadded conv drops the remainder
    /// (384 over 14 leaves 6 pixels).
    pub patches_per_side: usize,
}

/// Resize, tile and patchify one decoded image.
pub fn idefics3_preprocess(
    image: &Rgb8Image,
    patch: usize,
    params: &Idefics3Params,
) -> Result<Idefics3Tiles, VisionIoError> {
    let tile = params.tile;
    if image.width == 0 || image.height == 0 || patch == 0 || tile < patch {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{}x{} image against tile {tile}, patch {patch}",
                image.width, image.height
            ),
        });
    }
    let (w1, h1) = idefics3_longest_dims(image.width, image.height, params.longest_edge);
    let first = resize_lanczos_pil(&image.data, image.width, image.height, w1, h1);

    let mut frames: Vec<Vec<u8>> = Vec::new();
    let (mut rows, mut cols) = (0, 0);
    if params.split {
        let (w2, h2) = idefics3_tile_dims(w1, h1, tile);
        // A sliver can truncate its short edge to zero before the round-up.
        if w2 == 0 || h2 == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("{}x{} is too thin to tile", image.width, image.height),
            });
        }
        let second = resize_lanczos_pil(&first, w1, h1, w2, h2);
        if w2 > tile || h2 > tile {
            rows = h2 / tile;
            cols = w2 / tile;
            for r in 0..rows {
                for c in 0..cols {
                    let mut crop = Vec::with_capacity(tile * tile * 3);
                    for y in 0..tile {
                        let at = ((r * tile + y) * w2 + c * tile) * 3;
                        crop.extend_from_slice(&second[at..at + tile * 3]);
                    }
                    frames.push(crop);
                }
            }
        }
        frames.push(resize_lanczos_pil(&second, w2, h2, tile, tile));
    } else {
        frames.push(resize_lanczos_pil(&first, w1, h1, tile, tile));
    }

    let side = tile / patch;
    let mut pixels = Vec::with_capacity(frames.len() * side * side * 3 * patch * patch);
    for frame in &frames {
        for py in 0..side {
            for px in 0..side {
                for y in 0..patch {
                    let at = ((py * patch + y) * tile + px * patch) * 3;
                    for (i, &sample) in frame[at..at + patch * 3].iter().enumerate() {
                        let c = i % 3;
                        pixels.push((sample as f32 / 255.0 - params.mean[c]) / params.std[c]);
                    }
                }
            }
        }
    }
    Ok(Idefics3Tiles {
        pixels,
        frames: frames.len(),
        rows,
        cols,
        patches_per_side: side,
    })
}

/// The connector's pixel shuffle for ONE frame: `[side^2, hidden]` rows to
/// `[(side / scale)^2, hidden * scale^2]`. Output channel
/// `(dy * scale + dx) * hidden + c` of merged cell `(y, x)` is input row
/// `(y * scale + dy, x * scale + dx)`. This is the reference's
/// reshape/transpose chain written out; InternVL's shuffle orders the block
/// differently and is not interchangeable.
pub fn idefics3_pixel_shuffle(
    rows: &[u16],
    side: usize,
    hidden: usize,
    scale: usize,
) -> Result<Vec<u16>, VisionIoError> {
    if scale == 0 || side % scale != 0 || rows.len() != side * side * hidden {
        return Err(VisionIoError::InvalidDimensions {
            detail: format!(
                "{} values do not shuffle as a {side}x{side}x{hidden} grid by {scale}",
                rows.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(rows.len());
    let merged = side / scale;
    for y in 0..merged {
        for x in 0..merged {
            for dy in 0..scale {
                for dx in 0..scale {
                    let at = ((y * scale + dy) * side + x * scale + dx) * hidden;
                    out.extend_from_slice(&rows[at..at + hidden]);
                }
            }
        }
    }
    Ok(out)
}

/// Replace each rendered `<image>` id with the reference's span:
///
/// - split image: for every row, `fake <row_r_col_c> image*n` per column and
///   then `"\n"`; then `"\n" fake <global-img> image*n fake` (the two
///   newlines before the global view tokenize as ONE segment);
/// - single frame: `fake <global-img> image*n fake`.
///
/// `layouts` holds `(rows, cols)` per image. `encode` tokenizes one literal
/// segment without BOS; each segment sits between special tokens, so encoding
/// it alone gives the ids the reference's whole-string encode gives. Returns
/// the expanded ids and one span per FRAME, in tower output order.
pub fn idefics3_expand_prompt<E: std::fmt::Display>(
    rendered: &[i32],
    image_id: i32,
    fake_id: i32,
    layouts: &[(usize, usize)],
    seq_len: usize,
    mut encode: impl FnMut(&str) -> Result<Vec<i32>, E>,
) -> Result<(Vec<i32>, Vec<ImageSpan>), VisionIoError> {
    let found = rendered.iter().filter(|&&t| t == image_id).count();
    if found != layouts.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: found,
            grids: layouts.len(),
        });
    }
    let mut text = |s: &str| -> Result<Vec<i32>, VisionIoError> {
        let ids = encode(s).map_err(|e| VisionIoError::InvalidDimensions {
            detail: format!("tokenizing {s:?}: {e}"),
        })?;
        if ids.is_empty() || ids.contains(&image_id) {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!("{s:?} does not tokenize to plain separator ids"),
            });
        }
        Ok(ids)
    };
    let mut out = Vec::with_capacity(rendered.len());
    let mut spans = Vec::new();
    let mut layouts = layouts.iter();
    for &token in rendered {
        if token != image_id {
            out.push(token);
            continue;
        }
        // Counted above, so one layout exists per marker.
        let Some(&(rows, cols)) = layouts.next() else {
            break;
        };
        let mut run = |out: &mut Vec<i32>| {
            spans.push(ImageSpan {
                start: out.len(),
                len: seq_len,
            });
            out.resize(out.len() + seq_len, image_id);
        };
        for r in 0..rows {
            for c in 0..cols {
                out.push(fake_id);
                out.extend(text(&format!("<row_{}_col_{}>", r + 1, c + 1))?);
                run(&mut out);
            }
            // The last row's newline joins the one before the global view.
            out.extend(text(if r + 1 == rows { "\n\n" } else { "\n" })?);
        }
        out.push(fake_id);
        out.extend(text("<global-img>")?);
        run(&mut out);
        out.push(fake_id);
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOL: &str = r#"{"do_convert_rgb": true, "do_image_splitting": true,
        "do_normalize": true, "do_rescale": true, "do_resize": true,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "max_image_size": {"longest_edge": 512}, "resample": 1,
        "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 2048}}"#;

    #[test]
    fn the_published_config_reads_and_a_missing_edge_is_refused() {
        let p = Idefics3Params::from_preprocessor_config_json(SMOL).unwrap();
        assert_eq!((p.longest_edge, p.tile, p.split), (2048, 512, true));
        assert_eq!(p.mean, [0.5; 3]);
        for (from, to) in [
            ("\"max_image_size\"", "\"other\""),
            ("\"size\"", "\"other\""),
            ("\"resample\": 1", "\"resample\": 3"),
        ] {
            let broken = SMOL.replace(from, to);
            assert!(
                Idefics3Params::from_preprocessor_config_json(&broken).is_err(),
                "{to}"
            );
        }
    }

    #[test]
    fn the_two_resize_steps_match_the_reference_arithmetic() {
        // 1456 / (1000 / 667) = 971.15: truncated to 971, rounded up even.
        assert_eq!(idefics3_longest_dims(1000, 667, 1456), (1456, 972));
        assert_eq!(idefics3_longest_dims(300, 400, 1456), (1092, 1456));
        assert_eq!(idefics3_tile_dims(1456, 972, 364), (1456, 1092));
        assert_eq!(idefics3_tile_dims(1092, 1456, 364), (1092, 1456));
        // The short edge is derived from the long one AFTER it was rounded.
        assert_eq!(idefics3_tile_dims(700, 360, 364), (728, 728));
    }

    fn small() -> Idefics3Params {
        Idefics3Params {
            longest_edge: 16,
            tile: 8,
            split: true,
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }

    fn flat(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Rgb8Image {
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[f(x, y), 0, 0]);
            }
        }
        Rgb8Image {
            width,
            height,
            data,
        }
    }

    #[test]
    fn tiles_are_row_major_with_the_global_view_last() {
        // 16x16 needs no resize: 2x2 tiles of 8, then the global view.
        let image = flat(16, 16, |x, y| ((y / 8) * 2 + x / 8) as u8 * 60);
        let t = idefics3_preprocess(&image, 4, &small()).unwrap();
        assert_eq!((t.frames, t.rows, t.cols, t.patches_per_side), (5, 2, 2, 2));
        assert_eq!(t.pixels.len(), 5 * 4 * 48);
        for frame in 0..4 {
            let red = t.pixels[frame * 4 * 48];
            assert!((red - frame as f32 * 60.0 / 255.0).abs() < 1e-6, "{frame}");
        }
    }

    #[test]
    fn a_ragged_patch_grid_drops_the_remainder_and_no_split_is_one_frame() {
        let mut p = small();
        p.split = false;
        // Tile 8 over patch 3: a 2x2 grid, pixels 6 and 7 unused.
        let image = flat(16, 16, |_, _| 255);
        let t = idefics3_preprocess(&image, 3, &p).unwrap();
        assert_eq!((t.frames, t.rows, t.cols, t.patches_per_side), (1, 0, 0, 2));
        assert_eq!(t.pixels.len(), 4 * 27);
    }

    #[test]
    fn the_shuffle_puts_the_row_offset_outermost() {
        // A 4x4 grid of one channel, value = row * 4 + column.
        let rows: Vec<u16> = (0..16).collect();
        let out = idefics3_pixel_shuffle(&rows, 4, 1, 2).unwrap();
        assert_eq!(&out[0..4], &[0, 1, 4, 5]);
        assert_eq!(&out[4..8], &[2, 3, 6, 7]);
        assert_eq!(&out[8..12], &[8, 9, 12, 13]);
        assert!(idefics3_pixel_shuffle(&rows, 4, 1, 3).is_err());
    }

    fn encode(s: &str) -> Result<Vec<i32>, String> {
        Ok(match s {
            "<global-img>" => vec![100],
            "\n" => vec![101],
            "\n\n" => vec![102],
            "<row_1_col_1>" => vec![111],
            "<row_1_col_2>" => vec![112],
            "<row_2_col_1>" => vec![121],
            "<row_2_col_2>" => vec![122],
            other => return Err(format!("unexpected {other:?}")),
        })
    }

    #[test]
    fn a_split_image_expands_like_the_reference_prompt_string() {
        let (ids, spans) = idefics3_expand_prompt(&[1, 9, 2], 9, 7, &[(2, 2)], 2, encode).unwrap();
        #[rustfmt::skip]
        let want = vec![
            1,
            7, 111, 9, 9,  7, 112, 9, 9,  101,
            7, 121, 9, 9,  7, 122, 9, 9,  102,
            7, 100, 9, 9, 7,
            2,
        ];
        assert_eq!(ids, want);
        let starts: Vec<usize> = spans.iter().map(|s| s.start).collect();
        assert_eq!(starts, [3, 7, 12, 16, 21]);
        assert!(spans.iter().all(|s| s.len == 2));
    }

    #[test]
    fn a_single_frame_and_a_marker_count_mismatch() {
        let (ids, spans) = idefics3_expand_prompt(&[9], 9, 7, &[(0, 0)], 2, encode).unwrap();
        assert_eq!(ids, [7, 100, 9, 9, 7]);
        assert_eq!(spans, [ImageSpan { start: 2, len: 2 }]);
        assert!(idefics3_expand_prompt(&[9, 9], 9, 7, &[(0, 0)], 2, encode).is_err());
    }
}
