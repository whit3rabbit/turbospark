//! Preprocessing and prompt layout for the DeepSeek-OCR encoder, BASE mode
//! (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`).
//!
//! The reference is `DeepseekOCRProcessor.tokenize_with_images`, transcribed
//! from its source rather than run. Base mode is one global view:
//!
//! - the image is padded, never cropped or stretched, to `1024 x 1024` with
//!   PIL `ImageOps.pad` (bicubic `contain`, centered, canvas
//!   `int(mean * 255)` per channel, 127 for the published 0.5 mean). That is
//!   [`crate::pil_pad`], shared with the DeepSeek-VL2 processor;
//! - rescaled by `1/255` and normalized `(x - mean) / std` per channel;
//! - the tower reads the view as an HWC grid (its first conv is an im2col GEMM
//!   over it), so no patch rows are built here.
//!
//! The prompt run for one image is `16 * (16 + 1) + 1 = 273` positions: sixteen
//! rows of sixteen projected features each ended by the learned
//! `image_newline`, then one learned `view_separator`.
//!
//! Gundam mode (640 px local crops plus the global view) is NOT implemented:
//! it needs a bicubic-antialias resample of the CLIP position table and the
//! crop-grid layout. A caller always gets the base view.

use crate::decode::Rgb8Image;
use crate::deepseek_vl2::pil_pad;
use crate::error::VisionIoError;

/// The global view's side, in pixels.
pub const OCR_BASE_SIZE: usize = 1024;
/// Pixels per SAM patch.
pub const OCR_PATCH: usize = 16;
/// The processor's `downsample_ratio`: 64 patches wide, 16 projected rows wide.
pub const OCR_DOWNSAMPLE: usize = 4;
/// Projected feature rows per side of the global view.
pub const OCR_GRID: usize = OCR_BASE_SIZE / OCR_PATCH / OCR_DOWNSAMPLE;
/// Prompt positions one image occupies: a newline per row plus the separator.
pub const OCR_IMAGE_TOKENS: usize = OCR_GRID * (OCR_GRID + 1) + 1;

/// What `processor_config.json` says, narrowed to what base mode acts on.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcrParams {
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl DeepseekOcrParams {
    /// Parse `processor_config.json`.
    ///
    /// Refused rather than ignored: a patch size other than 16, a downsample
    /// ratio other than 4, a first candidate resolution other than 1024, or a
    /// missing or malformed mean or std (a wrong normalization is invisible in
    /// the output shape).
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
        let num = |key: &str| root.get(key).and_then(serde_json::Value::as_u64);
        if num("patch_size") != Some(OCR_PATCH as u64) {
            return Err(bad("patch_size", format!("must be {OCR_PATCH}")));
        }
        if num("downsample_ratio") != Some(OCR_DOWNSAMPLE as u64) {
            return Err(bad("downsample_ratio", format!("must be {OCR_DOWNSAMPLE}")));
        }
        let first = root
            .get("candidate_resolutions")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_array)
            .and_then(|p| p.first())
            .and_then(serde_json::Value::as_u64);
        if first != Some(OCR_BASE_SIZE as u64) {
            return Err(bad(
                "candidate_resolutions",
                format!("the first candidate must be {OCR_BASE_SIZE} wide"),
            ));
        }
        if root.get("normalize").and_then(serde_json::Value::as_bool) == Some(false) {
            return Err(bad("normalize", "must not be false".into()));
        }
        let triple = |key: &str| -> Result<[f32; 3], VisionIoError> {
            let array = root
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| bad(key, "missing or not an array".into()))?;
            if array.len() != 3 {
                return Err(bad(
                    key,
                    format!("expected 3 entries, found {}", array.len()),
                ));
            }
            let mut out = [0.0f32; 3];
            for (slot, value) in out.iter_mut().zip(array) {
                *slot = value
                    .as_f64()
                    .ok_or_else(|| bad(key, "entry is not a number".into()))?
                    as f32;
            }
            Ok(out)
        };
        let image_std = triple("image_std")?;
        if image_std.iter().any(|s| *s <= 0.0) {
            return Err(bad("image_std", "entries must be positive".into()));
        }
        Ok(Self {
            image_mean: triple("image_mean")?,
            image_std,
        })
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

/// One image as tower input.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekOcrView {
    /// `[1024 * 1024 * 3]`, row-major HWC, RGB, `(x / 255 - mean) / std`.
    pub pixels: Vec<f32>,
}

/// Pad an image to the 1024 px global view and normalize it.
pub fn deepseek_ocr_preprocess(
    image: &Rgb8Image,
    params: &DeepseekOcrParams,
) -> Result<DeepseekOcrView, VisionIoError> {
    let padded = pil_pad(image, OCR_BASE_SIZE, OCR_BASE_SIZE, params.pad_color())?;
    let mut pixels = Vec::with_capacity(OCR_BASE_SIZE * OCR_BASE_SIZE * 3);
    for px in padded.data.chunks_exact(3) {
        for c in 0..3 {
            pixels.push((f32::from(px[c]) / 255.0 - params.image_mean[c]) / params.image_std[c]);
        }
    }
    Ok(DeepseekOcrView { pixels })
}

/// One position of the image run, in prompt order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekOcrRow {
    /// Projected feature `row * 16 + col` of the global view.
    Feature(usize),
    /// The learned `image_newline`, closing each row of features.
    Newline,
    /// The learned `view_separator`, closing the run.
    Separator,
}

/// The base-mode run: sixteen rows of sixteen features each followed by a
/// newline, then the separator ([`OCR_IMAGE_TOKENS`] entries).
pub fn deepseek_ocr_row_layout() -> Vec<DeepseekOcrRow> {
    let mut out = Vec::with_capacity(OCR_IMAGE_TOKENS);
    for row in 0..OCR_GRID {
        for col in 0..OCR_GRID {
            out.push(DeepseekOcrRow::Feature(row * OCR_GRID + col));
        }
        out.push(DeepseekOcrRow::Newline);
    }
    out.push(DeepseekOcrRow::Separator);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = r#"{
        "add_special_token": false,
        "candidate_resolutions": [[1024, 1024]],
        "downsample_ratio": 4,
        "ignore_id": -100,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "image_token": "<image>",
        "mask_prompt": false,
        "normalize": true,
        "patch_size": 16,
        "processor_class": "DeepseekVLV2Processor",
        "sft_format": "deepseek"
    }"#;

    #[test]
    fn the_published_processor_config_parses() {
        let p = DeepseekOcrParams::from_processor_config_json(REAL).unwrap();
        assert_eq!(p.image_mean, [0.5; 3]);
        assert_eq!(p.pad_color(), [127, 127, 127]);
        assert_eq!(OCR_GRID, 16);
        assert_eq!(OCR_IMAGE_TOKENS, 273);
    }

    #[test]
    fn a_processor_this_pipeline_does_not_implement_is_refused() {
        let mut v: serde_json::Value = serde_json::from_str(REAL).unwrap();
        for (key, value) in [
            ("patch_size", serde_json::json!(14)),
            ("downsample_ratio", serde_json::json!(2)),
            ("candidate_resolutions", serde_json::json!([[640, 640]])),
            ("normalize", serde_json::json!(false)),
            ("image_mean", serde_json::json!([0.5, 0.5])),
            ("image_std", serde_json::json!([0.0, 0.5, 0.5])),
        ] {
            let saved = v[key].clone();
            v[key] = value;
            assert!(
                DeepseekOcrParams::from_processor_config_json(&v.to_string()).is_err(),
                "{key}"
            );
            v[key] = saved;
        }
        assert!(DeepseekOcrParams::from_processor_config_json("not json").is_err());
    }

    #[test]
    fn a_wide_image_is_padded_top_and_bottom_with_the_normalized_fill() {
        let p = DeepseekOcrParams::from_processor_config_json(REAL).unwrap();
        // 64 x 16 white: scales to 1024 x 256, so 384 pad rows above and below.
        let image = Rgb8Image::new(64, 16, vec![255u8; 64 * 16 * 3]).unwrap();
        let view = deepseek_ocr_preprocess(&image, &p).unwrap();
        assert_eq!(view.pixels.len(), OCR_BASE_SIZE * OCR_BASE_SIZE * 3);
        let at = |y: usize, x: usize, c: usize| view.pixels[(y * OCR_BASE_SIZE + x) * 3 + c];
        let fill = (127.0f32 / 255.0 - 0.5) / 0.5;
        assert!((at(0, 0, 0) - fill).abs() < 1e-6);
        assert!((at(383, 512, 1) - fill).abs() < 1e-6);
        // White maps to (1 - 0.5) / 0.5 = 1 inside the pasted band.
        assert!((at(512, 512, 0) - 1.0).abs() < 1e-3);
        // The band is rows 384..=639; row 384 is its first and 640 the first
        // pad row below it.
        assert!((at(384, 5, 2) - 1.0).abs() < 1e-3);
        assert!((at(639, 5, 2) - 1.0).abs() < 1e-3);
        assert!((at(640, 5, 2) - fill).abs() < 1e-6);
        assert!((at(640 + 200, 512, 0) - fill).abs() < 1e-6);
    }

    #[test]
    fn the_run_is_sixteen_rows_of_features_a_newline_each_then_one_separator() {
        let layout = deepseek_ocr_row_layout();
        assert_eq!(layout.len(), OCR_IMAGE_TOKENS);
        assert_eq!(layout[0], DeepseekOcrRow::Feature(0));
        assert_eq!(layout[15], DeepseekOcrRow::Feature(15));
        assert_eq!(layout[16], DeepseekOcrRow::Newline);
        assert_eq!(layout[17], DeepseekOcrRow::Feature(16));
        assert_eq!(layout[OCR_IMAGE_TOKENS - 2], DeepseekOcrRow::Newline);
        assert_eq!(layout[OCR_IMAGE_TOKENS - 1], DeepseekOcrRow::Separator);
        let features: Vec<usize> = layout
            .iter()
            .filter_map(|r| match r {
                DeepseekOcrRow::Feature(i) => Some(*i),
                _ => None,
            })
            .collect();
        assert_eq!(features, (0..256).collect::<Vec<_>>());
    }
}
