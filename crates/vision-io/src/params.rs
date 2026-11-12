//! The preprocessing parameters, and the `preprocessor_config.json` reader
//! that resolves them from a checkpoint.
//!
//! Everything here is READ from the checkpoint rather than recalled. The pixel
//! budget in particular is checkpoint-specific and far from the generic
//! Qwen2-VL library default: the qwen3_5 checkpoints declare 65,536 ..
//! 16,777,216 against the library's 3,136 .. 1,003,520, a factor of 16 on the
//! ceiling (`docs/VISION_PHASE0.md` item 5). Hardcoding the generic pair would
//! silently resize every image to a fraction of its intended resolution and
//! produce a correct-looking, lower-quality result.

use crate::decode::MAX_IMAGE_DIM;
use crate::error::VisionIoError;

/// Everything the preprocessing pipeline needs, resolved from one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessParams {
    /// Side of one square patch, in pixels.
    pub patch_size: usize,
    /// How many frames one temporal patch spans. A still image is repeated to
    /// fill it.
    pub temporal_patch_size: usize,
    /// Side of the spatial merge window, in patches.
    pub merge_size: usize,
    /// Channels per pixel. See [`PreprocessParams::from_preprocessor_config_json`]
    /// for why this is not read from the file.
    pub in_channels: usize,
    /// Lower bound on `resized_h * resized_w`.
    pub min_pixels: usize,
    /// Upper bound on `resized_h * resized_w`.
    pub max_pixels: usize,
    /// Per-channel mean, subtracted after the rescale.
    pub image_mean: [f32; 3],
    /// Per-channel standard deviation, divided out after the mean.
    pub image_std: [f32; 3],
    /// The rescale multiplier applied to the raw `0..255` sample.
    pub rescale_factor: f32,
    /// How the resized size, the resample filter and the patch row order are
    /// chosen. Defaults to the Qwen rule every existing checkpoint uses.
    pub resize: ResizeRule,
}

/// Which processor family's resize and patchify rule applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeRule {
    /// `Qwen2VLImageProcessor` / Qwen3-VL: smart resize on a pixel budget,
    /// PIL bicubic, patches in merge-window order with `(T, P_h, P_w, C)` rows.
    #[default]
    QwenPixels,
    /// Muse-Glimmer's processor: a cap on MERGED TOKENS with a nearest-aspect
    /// grid search, PIL Lanczos, patches in raster order with `(T, C, P_h,
    /// P_w)` rows (`crate::muse`).
    MuseTokens {
        /// `max_image_tokens`, merged tokens per image.
        max_tokens: usize,
    },
}

/// What an absent `rescale_factor` means: the reference processor's own
/// signature default (`processing_qwen3_vl.py:162`), i.e. the universal
/// 0-255 to 0-1 conversion.
pub const DEFAULT_RESCALE_FACTOR: f64 = 1.0 / 255.0;

/// The maximum accepted long-side/short-side aspect ratio, from the reference
/// processor's own `> 200` check.
pub const MAX_ASPECT_RATIO: f64 = 200.0;

/// The largest resize target, in pixels, a processor config may declare:
/// the largest image decode admits ([`MAX_IMAGE_DIM`] squared). A config is
/// checkpoint data and so untrusted, and a declared target is what a small
/// decoded image is stretched to, so without this a KB-sized file plus a
/// hostile config is a multi-GB allocation. The qwen3_5 checkpoints' own
/// 16,777,216 ceiling equals it exactly.
pub const MAX_CONFIG_PIXELS: usize = MAX_IMAGE_DIM as usize * MAX_IMAGE_DIM as usize;

/// Refuse a config-declared resize edge (a crop side, a tiling candidate's
/// side) past [`MAX_IMAGE_DIM`]. Bounding each edge also bounds every
/// `w * h` product the selection code forms, so none of them can wrap.
pub(crate) fn check_config_edge(field: &str, edge: usize) -> Result<(), VisionIoError> {
    if edge > MAX_IMAGE_DIM as usize {
        return Err(VisionIoError::BadConfig {
            field: field.into(),
            why: format!("{edge} exceeds the {MAX_IMAGE_DIM}-pixel decode ceiling"),
        });
    }
    Ok(())
}

impl PreprocessParams {
    /// The rounding granularity every resized edge is a multiple of:
    /// `patch_size * merge_size`. 32 for the qwen3_5 checkpoints (16 x 2), NOT
    /// the 28 a `patch_size` of 14 would give -- that is the generic Qwen2-VL
    /// value and this family does not use it.
    pub fn spatial_factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Elements in one patch row: `temporal_patch_size * patch_size^2 *
    /// in_channels`. 1536 for this family.
    pub fn patch_dim(&self) -> usize {
        self.temporal_patch_size * self.patch_size * self.patch_size * self.in_channels
    }

    /// Refuse a zero patch, temporal patch, or merge size. The config parser
    /// already refuses them, but the fields are pub, and the pipeline's entry
    /// points divide by them: a hand-built block with a zero would panic on a
    /// remainder by zero instead of returning an error.
    pub fn check_geometry(&self) -> Result<(), VisionIoError> {
        for (value, name) in [
            (self.patch_size, "patch_size"),
            (self.temporal_patch_size, "temporal_patch_size"),
            (self.merge_size, "merge_size"),
        ] {
            if value == 0 {
                return Err(VisionIoError::BadConfig {
                    field: name.into(),
                    why: "must be positive".into(),
                });
            }
        }
        Ok(())
    }

    /// Parse a checkpoint's `preprocessor_config.json`.
    ///
    /// THREE READING DECISIONS, each a claim about what silence means
    /// (AGENTS.md Gotcha 39), and each made the way the reference processor
    /// makes it rather than by picking a plausible default:
    ///
    /// - The pixel budget has TWO spellings and the file that ships with these
    ///   checkpoints uses the second. `min_pixels`/`max_pixels` win when
    ///   present and non-null; otherwise `size.shortest_edge` /
    ///   `size.longest_edge` are read (they are PIXEL COUNTS despite the
    ///   "edge" naming -- 65,536 is 256x256, not a 65,536-pixel side). With
    ///   neither present this REFUSES rather than falling back to the library
    ///   default, because that default is 16x smaller and the resulting images
    ///   would be quietly downsized.
    /// - `in_channels` is absent from every one of these files, and its
    ///   absence is not ambiguous: the processor sets `do_convert_rgb` and
    ///   converts unconditionally, so three is what the format's silence
    ///   MEANS rather than a value being guessed. It lives in `vision_config`
    ///   for the tower's own use and is not duplicated here.
    /// - `image_mean` / `image_std` are required. A wrong normalization is
    ///   invisible in the output shape, and no default is safe: these are
    ///   per-checkpoint and the shipped pair here (`0.5`) differs from the
    ///   ImageNet triples other families write.
    /// - `rescale_factor` is DEFAULTED to [`DEFAULT_RESCALE_FACTOR`], and the
    ///   asymmetry with the pixel budget is the point. This clause used to say
    ///   it was "written by every checkpoint" and
    ///   `mlx-community/Qwen3.8-27B-4bit` -- the very checkpoint this crate
    ///   was built against -- omits it, which the first real `--image` run
    ///   found. The reference's own processor declares
    ///   `rescale_factor: float = 1 / 255.0` as a signature default
    ///   (`processing_qwen3_vl.py:162`), so `1/255` is what the format's
    ///   SILENCE means rather than a guess: it is the universal 0-255 to 0-1
    ///   conversion, where the generic pixel budget is wrong for this family
    ///   by a factor of 16. AGENTS.md Gotcha 39's rule -- reach for the
    ///   FORMAT's default when a key is optional, and refuse only where the
    ///   format has none.
    pub fn from_preprocessor_config_json(json: &str) -> Result<Self, VisionIoError> {
        let root: serde_json::Value =
            serde_json::from_str(json).map_err(|e| VisionIoError::BadConfig {
                field: "<document>".into(),
                why: e.to_string(),
            })?;

        let usize_field = |name: &str| -> Result<usize, VisionIoError> {
            root.get(name)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| VisionIoError::BadConfig {
                    field: name.into(),
                    why: "missing or not a non-negative integer".into(),
                })
        };

        let patch_size = usize_field("patch_size")?;
        let temporal_patch_size = usize_field("temporal_patch_size")?;
        let merge_size = usize_field("merge_size")?;
        for (value, name) in [
            (patch_size, "patch_size"),
            (temporal_patch_size, "temporal_patch_size"),
            (merge_size, "merge_size"),
        ] {
            if value == 0 {
                return Err(VisionIoError::BadConfig {
                    field: name.into(),
                    why: "must be positive".into(),
                });
            }
        }
        // Every resized edge is at least one `patch_size * merge_size`, so a
        // factor past the decode ceiling is an upscale no pixel budget sees.
        let factor = patch_size.saturating_mul(merge_size);
        check_config_edge("patch_size * merge_size", factor)?;

        // Muse-Glimmer states its budget as a cap on merged tokens and has no
        // pixel budget at all. The pixel pair is DERIVED from the cap so the
        // session's memory-budget clamp (which lowers `max_pixels`) has
        // something to act on; `preprocess` honours the lower of the two.
        let muse_tokens = root.get("max_image_tokens").and_then(|v| v.as_u64());
        let (min_pixels, max_pixels, resize) = match muse_tokens {
            Some(tokens) if tokens > 0 => {
                let side = factor;
                (
                    side * side,
                    // Saturating, then clamped: Muse never upscales, so a
                    // token cap past what decode can produce is a no-op, but
                    // the plain product wraps silently in release.
                    (tokens as usize)
                        .saturating_mul(side * side)
                        .min(MAX_CONFIG_PIXELS),
                    ResizeRule::MuseTokens {
                        max_tokens: tokens as usize,
                    },
                )
            }
            Some(_) => {
                return Err(VisionIoError::BadConfig {
                    field: "max_image_tokens".into(),
                    why: "must be positive".into(),
                })
            }
            None => {
                let (min, max) = read_pixel_budget(&root)?;
                // The floor upscales every smaller image to it, so it is
                // refused past the decode ceiling. The ceiling only ever
                // shrinks an image, and decode already bounds the source,
                // so clamping it changes no real resize and refuses nothing.
                if min > MAX_CONFIG_PIXELS {
                    return Err(VisionIoError::BadConfig {
                        field: "min_pixels".into(),
                        why: format!("{min} exceeds the {MAX_CONFIG_PIXELS}-pixel decode ceiling"),
                    });
                }
                (min, max.min(MAX_CONFIG_PIXELS), ResizeRule::QwenPixels)
            }
        };
        if min_pixels > max_pixels {
            return Err(VisionIoError::BadConfig {
                field: "size".into(),
                why: format!("min_pixels {min_pixels} exceeds max_pixels {max_pixels}"),
            });
        }

        // The normalization divides by it: zero (or a value that rounds to
        // zero in f32) would be silent inf or NaN in every embedding.
        let image_std = read_triple(&root, "image_std")?;
        if image_std.iter().any(|s| *s <= 0.0) {
            return Err(VisionIoError::BadConfig {
                field: "image_std".into(),
                why: "entries must be positive".into(),
            });
        }
        Ok(Self {
            patch_size,
            temporal_patch_size,
            merge_size,
            in_channels: 3,
            min_pixels,
            max_pixels,
            image_mean: read_triple(&root, "image_mean")?,
            image_std,
            // DEFAULTED, unlike the pixel budget above it, and the two are
            // not inconsistent -- see the doc comment's third bullet.
            rescale_factor: root
                .get("rescale_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(DEFAULT_RESCALE_FACTOR) as f32,
            resize,
        })
    }
}

/// `min_pixels`/`max_pixels` if both are present and non-null, else
/// `size.shortest_edge`/`size.longest_edge`. Never a default.
fn read_pixel_budget(root: &serde_json::Value) -> Result<(usize, usize), VisionIoError> {
    let direct = |name: &str| root.get(name).and_then(|v| v.as_u64()).map(|v| v as usize);
    if let (Some(min), Some(max)) = (direct("min_pixels"), direct("max_pixels")) {
        return Ok((min, max));
    }
    let size = root.get("size").ok_or_else(|| VisionIoError::BadConfig {
        field: "size".into(),
        why: "absent, and neither min_pixels nor max_pixels was given; \
              the pixel budget is checkpoint-specific and has no safe default"
            .into(),
    })?;
    let edge = |name: &str| -> Result<usize, VisionIoError> {
        size.get(name)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| VisionIoError::BadConfig {
                field: format!("size.{name}"),
                why: "missing or not a non-negative integer".into(),
            })
    };
    Ok((edge("shortest_edge")?, edge("longest_edge")?))
}

fn read_triple(root: &serde_json::Value, field: &str) -> Result<[f32; 3], VisionIoError> {
    let bad = |why: &str| VisionIoError::BadConfig {
        field: field.into(),
        why: why.into(),
    };
    let array = root
        .get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| bad("missing or not an array"))?;
    if array.len() != 3 {
        return Err(bad(&format!("expected 3 entries, found {}", array.len())));
    }
    let mut out = [0.0f32; 3];
    for (slot, value) in out.iter_mut().zip(array) {
        *slot = value.as_f64().ok_or_else(|| bad("entry is not a number"))? as f32;
    }
    Ok(out)
}
