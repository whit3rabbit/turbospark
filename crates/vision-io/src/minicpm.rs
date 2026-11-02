//! MiniCPM-V 4.7 image preprocessing, placeholder expansion and canvas
//! positions (`docs/MINICPMV_VISION_PHASE0.md`).
//!
//! Ported from mlx-vlm's `minicpmv4_6/processing_minicpmv4_6.py` and
//! `minicpmv4_7/rope.py`, read and never executed. **Only the THUMBNAIL is
//! built**: the reference's `get_sliced_grid` returns `None` for
//! `max_slice_nums == 1`, and this module always takes that arm, so a large
//! page is one resized view rather than a global view plus up to nine detail
//! crops. A prompt carrying `<slice>` markers is refused by name, not skipped.
//!
//! # Differences from the Qwen path this crate otherwise serves
//!
//! - The size is NOT a pixel budget: [`minicpm_resize_dims`] fits the image to
//!   about `scale_resolution`^2 pixels (upscaling a small one) and rounds each
//!   side to a multiple of `patch * 4` = 56.
//! - Patch rows are RASTER order, `(P_h, P_w, C)` inside a row, with no
//!   temporal duplication (patch row = 588 values, not 1176).
//! - Normalization DIVIDES by 255 (`/ 255.0` in the reference), where the Qwen
//!   path multiplies by a stored reciprocal; the two can differ by an ulp.
//! - One `<|image_pad|>` in the rendered prompt expands to
//!   `<image_id>N</image_id><image>` + merged-token fill + `</image>`, and the
//!   fill token is the SAME id as the template marker.

use crate::decode::Rgb8Image;
use crate::error::VisionIoError;
use crate::mrope::{ImageSpan, MropePositions, SplicedPrompt};
use crate::params::PreprocessParams;
use crate::patchify::GridThw;
use crate::preprocess::PreprocessedImage;
use crate::resize::resize_bicubic_pil;
use crate::rounding::round_half_to_even;

/// The reference's `patch_size` and `scale_resolution`.
pub const MINICPM_PATCH: usize = 14;
pub const MINICPM_SCALE_RESOLUTION: usize = 448;
/// Per-axis downsample of the default `16x` mode: a 2x2 window merger then a
/// 2x2 merger, so one output token covers a 4x4 block of patches.
pub const MINICPM_MERGE: usize = 4;

/// Preprocessing parameters read off the install.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniCpmParams {
    pub patch_size: usize,
    pub scale_resolution: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl MiniCpmParams {
    /// From a checkpoint's `preprocessor_config.json`. `image_mean` and
    /// `image_std` default to 0.5 when absent, the reference's own default;
    /// the patch size and scale resolution are the reference's constants.
    pub fn from_preprocessor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let root: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionIoError::BadConfig {
                field: "<document>".into(),
                why: e.to_string(),
            })?;
        let triple = |key: &str| -> Result<[f32; 3], VisionIoError> {
            match root.get(key) {
                None => Ok([0.5; 3]),
                Some(v) => {
                    let list = v.as_array().filter(|a| a.len() == 3).ok_or_else(|| {
                        VisionIoError::BadConfig {
                            field: key.into(),
                            why: "expected an array of three numbers".into(),
                        }
                    })?;
                    let mut out = [0.0f32; 3];
                    for (slot, item) in out.iter_mut().zip(list) {
                        *slot = item.as_f64().ok_or_else(|| VisionIoError::BadConfig {
                            field: key.into(),
                            why: "non-numeric entry".into(),
                        })? as f32;
                    }
                    Ok(out)
                }
            }
        };
        let std = triple("image_std")?;
        if std.iter().any(|s| *s == 0.0) {
            return Err(VisionIoError::BadConfig {
                field: "image_std".into(),
                why: "a zero standard deviation".into(),
            });
        }
        Ok(Self {
            patch_size: MINICPM_PATCH,
            scale_resolution: MINICPM_SCALE_RESOLUTION,
            image_mean: triple("image_mean")?,
            image_std: std,
        })
    }

    /// The Qwen-shaped parameter block the runner's `encode_image` validates a
    /// tower against: patch 14, one frame, the TOTAL merge of 4. Not used to
    /// preprocess.
    pub fn as_preprocess_params(&self) -> PreprocessParams {
        PreprocessParams {
            patch_size: self.patch_size,
            temporal_patch_size: 1,
            merge_size: MINICPM_MERGE,
            in_channels: 3,
            min_pixels: self.patch_size * self.patch_size,
            max_pixels: self.scale_resolution * self.scale_resolution * 4,
            image_mean: self.image_mean,
            image_std: self.image_std,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

/// `max(round(length / patch) * patch, patch)` with Python's `round`
/// (half to even).
fn ensure_divide(length: usize, patch: usize) -> usize {
    let rounded = round_half_to_even(length as f64 / patch as f64) as usize * patch;
    rounded.max(patch)
}

/// The reference's `_find_best_resize(size, scale_resolution, patch_size,
/// allow_upscale=True)`, returned as `(height, width)`.
pub fn minicpm_resize_dims(width: usize, height: usize, params: &MiniCpmParams) -> (usize, usize) {
    let ratio = width as f64 / height.max(1) as f64;
    // `int()` truncates toward zero.
    let new_height = (params.scale_resolution as f64 / ratio.max(1e-6).sqrt()) as usize;
    let new_width = (new_height as f64 * ratio) as usize;
    let merge_factor = params.patch_size * MINICPM_MERGE;
    (
        ensure_divide(new_height, merge_factor),
        ensure_divide(new_width, merge_factor),
    )
}

/// Decode-side pipeline for one image: best-fit resize with PIL bicubic,
/// `(px / 255 - mean) / std`, and raster patch rows `(P_h, P_w, C)`.
pub fn minicpm_preprocess(
    image: &Rgb8Image,
    params: &MiniCpmParams,
) -> Result<PreprocessedImage, VisionIoError> {
    if image.width == 0 || image.height == 0 {
        return Err(VisionIoError::InvalidDimensions {
            detail: "an empty image".into(),
        });
    }
    let (h, w) = minicpm_resize_dims(image.width, image.height, params);
    let resized = resize_bicubic_pil(&image.data, image.width, image.height, w, h);
    let p = params.patch_size;
    let (gh, gw) = (h / p, w / p);
    let mut rows = Vec::with_capacity(gh * gw * p * p * 3);
    for py in 0..gh {
        for px in 0..gw {
            for dy in 0..p {
                for dx in 0..p {
                    let base = ((py * p + dy) * w + px * p + dx) * 3;
                    for c in 0..3 {
                        let sample = resized[base + c] as f32 / 255.0;
                        rows.push((sample - params.image_mean[c]) / params.image_std[c]);
                    }
                }
            }
        }
    }
    let grid = GridThw::new(1, gh, gw);
    Ok(PreprocessedImage {
        merged_tokens: grid.merged_tokens(MINICPM_MERGE),
        patch_rows: rows,
        grid,
        resized: (h, w),
    })
}

/// The ids the splice and the canvas positions key on, read off the install
/// (`VisionConfig` carries the first four; the rest are the kind's constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniCpmIds {
    /// `<|image_pad|>`: the template's marker AND the per-token fill id.
    pub image_pad: i32,
    /// `<image>` and `</image>`.
    pub image_start: i32,
    pub image_end: i32,
    pub slice_start: i32,
    pub slice_end: i32,
    /// `\n`, which the canvas rope absorbs into a frame when it follows one.
    pub newline: i32,
    /// `<image_id>` and `</image_id>`.
    pub image_id_start: i32,
    pub image_id_end: i32,
}

/// Expand every rendered `<|image_pad|>` to its full placeholder run and
/// compute the canvas positions, in one call so the counts cannot disagree.
///
/// `index_ids(k)` is the token ids of the image's ordinal as text (`"0"`,
/// `"1"`, ...), from the tokenizer, because this crate has none.
pub fn minicpm_splice_and_positions(
    rendered: &[i32],
    grids: &[GridThw],
    ids: &MiniCpmIds,
    index_ids: &dyn Fn(usize) -> Vec<i32>,
) -> Result<SplicedPrompt, VisionIoError> {
    let placeholders = rendered.iter().filter(|&&t| t == ids.image_pad).count();
    if placeholders != grids.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders,
            grids: grids.len(),
        });
    }
    if rendered
        .iter()
        .any(|&t| t == ids.slice_start || t == ids.slice_end)
    {
        return Err(VisionIoError::InvalidDimensions {
            detail: "the prompt carries <slice> markers; image slicing is not built (only the \
                     thumbnail is)"
                .into(),
        });
    }
    let mut out = Vec::with_capacity(rendered.len() + grids.len() * 80);
    let mut spans = Vec::with_capacity(grids.len());
    let mut canvas = Vec::with_capacity(grids.len());
    let mut image = 0usize;
    for &token in rendered {
        if token != ids.image_pad {
            out.push(token);
            continue;
        }
        let grid = grids[image];
        let merged = grid.merged_tokens(MINICPM_MERGE);
        if merged == 0 || grid.h % MINICPM_MERGE != 0 || grid.w % MINICPM_MERGE != 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: format!(
                    "image {image} grid {}x{} is not a whole number of 4x4 blocks",
                    grid.h, grid.w
                ),
            });
        }
        out.push(ids.image_id_start);
        out.extend(index_ids(image));
        out.push(ids.image_id_end);
        out.push(ids.image_start);
        spans.push(ImageSpan {
            start: out.len(),
            len: merged,
        });
        out.extend(std::iter::repeat_n(ids.image_pad, merged));
        out.push(ids.image_end);
        canvas.push((grid.h / MINICPM_MERGE, grid.w / MINICPM_MERGE));
        image += 1;
    }
    let (triples, rope_delta) = canvas_positions(&out, ids, &canvas)?;
    Ok(SplicedPrompt {
        ids: out,
        positions: MropePositions {
            triples,
            rope_delta,
            spans,
        },
    })
}

/// A frame as the reference groups it: `<image>` marker at `thumb_start - 1`,
/// the fill run `[thumb_start, thumb_end)`, then any marker tokens after it.
struct Frame {
    thumb_start: usize,
    thumb_end: usize,
    frame_end: usize,
}

/// The reference's `CanvasRoPE.get_rope_index` for a prompt of thumbnails
/// only. `canvas[k]` is image `k`'s merged grid `(rows, cols)`.
///
/// Returns one `(t, h, w)` per token and `max + 1 - len`. Ported literally,
/// including the two behaviors that are easy to "fix" into a different model:
/// marker tokens that FOLLOW `</image>` (a `\n`, the next `<image>`) are
/// absorbed into the frame and take the canvas origin on all three axes, and
/// a frame's `<image>` marker takes `origin - 1` on h and w.
pub fn canvas_positions(
    ids: &[i32],
    cfg: &MiniCpmIds,
    canvas: &[(usize, usize)],
) -> Result<(Vec<(i32, i32, i32)>, i32), VisionIoError> {
    let n = ids.len();
    let markers = [
        cfg.slice_start,
        cfg.slice_end,
        cfg.image_start,
        cfg.image_end,
        cfg.newline,
    ];
    let is_marker = |t: i32| markers.contains(&t);

    // Runs of fill tokens, in order.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if ids[i] == cfg.image_pad {
            let start = i;
            while i < n && ids[i] == cfg.image_pad {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    if runs.len() != canvas.len() {
        return Err(VisionIoError::PlaceholderMismatch {
            placeholders: runs.len(),
            grids: canvas.len(),
        });
    }

    // Group the runs into frames and the frames into groups.
    let mut groups: Vec<(usize, Vec<Frame>)> = Vec::new();
    let mut previous_end = 0usize;
    for &(start, end) in &runs {
        if start == 0 {
            return Err(VisionIoError::InvalidDimensions {
                detail: "an image run with no <image> marker before it".into(),
            });
        }
        let gap = &ids[previous_end.min(start - 1)..start - 1];
        let adjacent = !groups.is_empty() && gap.iter().all(|&t| is_marker(t));
        let mut frame_end = end;
        while frame_end < n && is_marker(ids[frame_end]) {
            frame_end += 1;
        }
        let frame = Frame {
            thumb_start: start,
            thumb_end: end,
            frame_end,
        };
        if adjacent {
            groups.last_mut().expect("checked").1.push(frame);
        } else {
            groups.push((start - 1, vec![frame]));
        }
        previous_end = end;
    }

    let mut pos: Vec<(i32, i32, i32)> = (0..n as i32).map(|i| (i, i, i)).collect();
    let mut current_pos = 0i32;
    let mut current_idx = 0usize;
    let mut image = 0usize;
    for (group_start, frames) in &groups {
        let group_end = frames.last().expect("a group has a frame").frame_end;
        if *group_start > current_idx {
            let text_len = group_start - current_idx;
            for (j, slot) in pos[current_idx..*group_start].iter_mut().enumerate() {
                let v = current_pos + j as i32;
                *slot = (v, v, v);
            }
            current_pos += text_len as i32;
        }
        let mut frame_cursor = *group_start;
        for frame in frames {
            let (ch, cw) = canvas[image];
            image += 1;
            let frame_start = frame.thumb_start - 1;
            if frame_start > frame_cursor {
                let gap_len = frame_start - frame_cursor;
                for (j, slot) in pos[frame_cursor..frame_start].iter_mut().enumerate() {
                    let v = current_pos + j as i32;
                    *slot = (v, v, v);
                }
                current_pos += gap_len as i32 + 1;
            }
            let frame_end = frame.frame_end.min(group_end);
            let origin = current_pos;
            let halo = (origin - 1).max(0);
            for slot in &mut pos[frame_start..frame_end] {
                *slot = (origin, origin, origin);
            }
            pos[frame_start].1 = halo;
            pos[frame_start].2 = halo;
            if frame.thumb_end < frame_end {
                pos[frame.thumb_end].1 = origin + ch as i32;
                pos[frame.thumb_end].2 = origin + cw as i32;
            }
            // `np.rint(np.linspace(0, n - 1, n))` is `0..n` exactly.
            for r in 0..ch {
                for c in 0..cw {
                    pos[frame.thumb_start + r * cw + c] =
                        (origin, origin + r as i32, origin + c as i32);
                }
            }
            current_pos = origin + ch.max(cw) as i32 + 1;
            frame_cursor = frame_end;
        }
        if frame_cursor < group_end {
            let trail_len = group_end - frame_cursor;
            for (j, slot) in pos[frame_cursor..group_end].iter_mut().enumerate() {
                let v = current_pos + j as i32;
                *slot = (v, v, v);
            }
            current_pos += trail_len as i32;
        }
        current_idx = group_end;
    }
    if current_idx < n {
        for (j, slot) in pos[current_idx..n].iter_mut().enumerate() {
            let v = current_pos + j as i32;
            *slot = (v, v, v);
        }
    }
    let max = pos
        .iter()
        .map(|&(a, b, c)| a.max(b).max(c))
        .max()
        .unwrap_or(-1);
    Ok((pos, max + 1 - n as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: i32 = 9000;
    const IDS: MiniCpmIds = MiniCpmIds {
        image_pad: PAD,
        image_start: 9001,
        image_end: 9002,
        slice_start: 9003,
        slice_end: 9004,
        newline: 198,
        image_id_start: 9005,
        image_id_end: 9006,
    };

    fn digits(k: usize) -> Vec<i32> {
        vec![500 + k as i32]
    }

    fn params() -> MiniCpmParams {
        MiniCpmParams::from_preprocessor_config_json("{}").unwrap()
    }

    #[test]
    fn resize_dims_match_the_reference_rule() {
        let p = params();
        // 448x448: ratio 1, height 448, width 448, both already multiples of 56.
        assert_eq!(minicpm_resize_dims(448, 448, &p), (448, 448));
        // 1024x1280 (w x h): ratio 0.8, height = int(448 / sqrt(0.8)) = 500,
        // width = int(500 * 0.8) = 400, rounded to 504 and 392.
        assert_eq!(minicpm_resize_dims(1024, 1280, &p), (504, 392));
        // A tiny image upscales to the same area (allow_upscale): 56x56 -> 448.
        assert_eq!(minicpm_resize_dims(56, 56, &p), (448, 448));
        // 16x1: the round never goes below one 56-pixel block.
        let (h, w) = minicpm_resize_dims(1600, 16, &p);
        assert!(h >= 56 && w >= 56 && h % 56 == 0 && w % 56 == 0, "{h}x{w}");
    }

    #[test]
    fn python_round_is_half_to_even_here() {
        // 28 / 56 = 0.5 rounds to 0 (then the floor of one block), 84 / 56 = 1.5
        // rounds to 2: a half-away-from-zero round would give 56 and 112.
        assert_eq!(ensure_divide(28, 56), 56);
        assert_eq!(ensure_divide(84, 56), 112);
        assert_eq!(ensure_divide(140, 56), 112, "2.5 rounds to 2, not 3");
    }

    #[test]
    fn preprocess_rows_are_raster_with_p_p_c_inside() {
        // A 448x448 image whose pixel value encodes its coordinates.
        let mut data = Vec::new();
        for y in 0..448usize {
            for x in 0..448usize {
                data.extend([(x % 251) as u8, (y % 251) as u8, ((x + y) % 251) as u8]);
            }
        }
        let image = Rgb8Image::new(448, 448, data).unwrap();
        let out = minicpm_preprocess(&image, &params()).unwrap();
        assert_eq!(out.grid, GridThw::new(1, 32, 32));
        assert_eq!(out.merged_tokens, 64);
        assert_eq!(out.patch_rows.len(), 1024 * 588);
        let norm = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;
        let at = |patch: usize, dy: usize, dx: usize, c: usize| {
            out.patch_rows[patch * 588 + (dy * 14 + dx) * 3 + c]
        };
        // Patch 1 is the SECOND patch of the first row (raster), not the first
        // patch of the second merge group.
        assert_eq!(at(1, 0, 0, 0), norm(14));
        // Patch 32 starts the second patch row.
        assert_eq!(at(32, 0, 0, 1), norm(14));
        // Inside a patch: (dy, dx, c) with c fastest.
        assert_eq!(at(0, 3, 5, 2), norm(8));
        assert_eq!(at(0, 3, 5, 0), norm(5));
        assert_eq!(at(0, 3, 5, 1), norm(3));
    }

    #[test]
    fn a_single_thumbnail_takes_the_canvas_origin_and_a_grid_of_offsets() {
        // text, text, <|image_pad|>, "\n", text.
        let rendered = [10, 11, PAD, 198, 12];
        let g = GridThw::new(1, 8, 12); // merged canvas 2 rows x 3 cols
        let s = minicpm_splice_and_positions(&rendered, &[g], &IDS, &digits).unwrap();
        // 10 11 <image_id> 500 </image_id> <image> fill*6 </image> \n 12
        let expect_ids: Vec<i32> = [10, 11, 9005, 500, 9006, 9001]
            .into_iter()
            .chain(std::iter::repeat_n(PAD, 6))
            .chain([9002, 198, 12])
            .collect();
        assert_eq!(s.ids, expect_ids);
        assert_eq!(s.positions.spans, vec![ImageSpan { start: 6, len: 6 }]);
        let t = &s.positions.triples;
        // Text before the group: 0, 1, then image_id tokens 2, 3, 4 (still text).
        assert_eq!(
            &t[..5],
            &[(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
        );
        // The frame starts at the <image> marker (index 5): origin = 5.
        assert_eq!(
            t[5],
            (5, 4, 4),
            "<image>: t at the origin, h and w one behind"
        );
        // Fill grid: t = origin, h = origin + row, w = origin + col.
        assert_eq!(t[6], (5, 5, 5));
        assert_eq!(t[8], (5, 5, 7));
        assert_eq!(t[9], (5, 6, 5));
        assert_eq!(t[11], (5, 6, 7));
        // </image> at (origin, origin + rows, origin + cols).
        assert_eq!(t[12], (5, 7, 8));
        // The trailing "\n" is a marker: absorbed at the origin on all axes.
        assert_eq!(t[13], (5, 5, 5));
        // The clock resumes at origin + max(rows, cols) + 1 = 9.
        assert_eq!(t[14], (9, 9, 9));
        // max position 9, 15 tokens: delta = 9 + 1 - 15.
        assert_eq!(s.positions.rope_delta, -5);
    }

    #[test]
    fn the_absorbed_marker_is_the_behavior_not_a_bug() {
        // The same prompt with a TEXT token (not a marker) after </image> keeps
        // the clock running from the end of the canvas instead.
        let g = GridThw::new(1, 8, 12);
        let s = minicpm_splice_and_positions(&[PAD, 77], &[g], &IDS, &digits).unwrap();
        let t = &s.positions.triples;
        let last = t.len() - 1;
        assert_eq!(t[last], (7, 7, 7), "origin 3 + max(2,3) + 1");
        // And a "\n" in that slot would have been (3,3,3): different positions
        // for what is otherwise the same sequence.
        let s2 = minicpm_splice_and_positions(&[PAD, 198], &[g], &IDS, &digits).unwrap();
        assert_eq!(s2.positions.triples[last], (3, 3, 3));
    }

    #[test]
    fn two_images_are_separate_groups_when_text_sits_between() {
        let g = GridThw::new(1, 4, 4); // 1x1 canvas, one token
        let s = minicpm_splice_and_positions(&[PAD, 33, PAD], &[g, g], &IDS, &digits).unwrap();
        // <id0> 500 </id> <image> fill </image> 33 <id> 501 </id> <image> fill </image>
        assert_eq!(s.positions.spans.len(), 2);
        let t = &s.positions.triples;
        // First frame: origin 3 (after the three image-id tokens), fill (3,3,3),
        // </image> at (3, 4, 4); clock resumes at 3 + 1 + 1 = 5 for "33".
        assert_eq!(t[3], (3, 2, 2));
        assert_eq!(t[4], (3, 3, 3));
        assert_eq!(t[5], (3, 4, 4));
        assert_eq!(t[6], (5, 5, 5), "33");
        // Second image's id tokens continue the text clock: 6, 7, 8; origin 9.
        assert_eq!(&t[7..10], &[(6, 6, 6), (7, 7, 7), (8, 8, 8)]);
        assert_eq!(t[10], (9, 8, 8));
        assert_eq!(t[11], (9, 9, 9));
    }

    #[test]
    fn counts_and_slices_are_checked() {
        let g = GridThw::new(1, 4, 4);
        assert!(minicpm_splice_and_positions(&[PAD], &[], &IDS, &digits).is_err());
        assert!(minicpm_splice_and_positions(&[1, 2], &[g], &IDS, &digits).is_err());
        assert!(minicpm_splice_and_positions(&[PAD, 9003], &[g], &IDS, &digits).is_err());
        // A grid that is not whole 4x4 blocks is refused.
        assert!(
            minicpm_splice_and_positions(&[PAD], &[GridThw::new(1, 6, 8)], &IDS, &digits).is_err()
        );
    }
}
