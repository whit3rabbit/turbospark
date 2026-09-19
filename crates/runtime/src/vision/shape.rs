//! The tower's dimensions, resolved once from `ArchConfig.vision` and passed
//! down rather than re-derived at each stage (ROADMAP M-V4).
//!
//! `head_dim` and `merger_input` are DERIVED and are not `VisionConfig`
//! fields, because the checkpoint states neither: the config carries
//! `hidden_size` and `num_heads`, and `spatial_merge_size` rather than the
//! product it implies. Deriving them in one place is what stops two call
//! sites from disagreeing about a number the file never wrote down.

use model_io::VisionConfig;

use crate::real_forward_types::RealForwardError;

/// The tower's layer normalization epsilon.
///
/// A family CONSTANT rather than an `ArchConfig` field, following the
/// `rms_eps` precedent (`crates/runtime` CLAUDE.md, the museGlimmer bullet):
/// it is not a binary fraction and `arch_validation` compares manifest floats
/// with `!=`, so carrying it in the manifest would make every install fail to
/// round-trip (AGENTS.md Gotcha 24). Read off the reference's
/// `nn.LayerNorm(config.hidden_size, eps=1e-6)`, which both the blocks and the
/// merger construct.
pub(crate) const VISION_LAYER_NORM_EPS: f32 = 1e-6;

// No `Copy`: the deepstack indexes are a `Vec`. Every use was `&self.shape`
// or a field read, which `Clone` serves.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisionShape {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) merge: usize,
    pub(crate) out_hidden: usize,
    /// Elements in one patch row: `temporal * patch^2 * channels`. 1536 on
    /// this family, and the `k` of the patch-embedding GEMM.
    pub(crate) patch_dim: usize,
    /// Rows in the learned position table (2304 = 48x48).
    pub(crate) pos_rows: usize,
    /// Its side, `sqrt(pos_rows)`. Checked to be exact at resolve time: a
    /// non-square table cannot be bilinearly indexed by `(h, w)` at all, and
    /// a rounded side would silently read the wrong rows.
    pub(crate) pos_side: usize,
    /// Spatial patch edge in pixels (16 on this family). NOT recoverable
    /// from `patch_dim`, which also folds in `temporal_patch_size` and
    /// `in_channels` -- kept separately for `vision::budget`'s pixel-budget
    /// conversion (Part B3), which needs `patch_size * merge` the same way
    /// `turbospark_vision_io::PreprocessParams::spatial_factor` does.
    pub(crate) patch_size: usize,
    /// Block indices whose outputs feed the deepstack mergers, as usize,
    /// ascending, each below [`Self::depth`]. EMPTY for a tower without
    /// deepstack (every `qwen3_5` tower); `[5, 11, 17]` on the `qwen3_vl`-4B
    /// tower. The merger at slot `k` of `VisionResident::deepstack_mergers`
    /// belongs to [`Self::deepstack`][`k`], and its output is injected after
    /// TRUNK layer `k` -- the injection depth is the slot order, never the
    /// block index.
    pub(crate) deepstack: Vec<usize>,
}

impl VisionShape {
    /// The merger's input width, `hidden * merge^2` (4608 on this family).
    ///
    /// NOT `intermediate`, which is the per-block MLP's 4304. The two are
    /// close, both are "the wide one", and the original planning table had
    /// this exact confusion in it (`docs/VISION_PHASE0.md` item 1).
    pub(crate) fn merger_input(&self) -> usize {
        self.hidden * self.merge * self.merge
    }

    /// Patches this many merged tokens came from.
    pub(crate) fn patches_per_token(&self) -> usize {
        self.merge * self.merge
    }

    /// `head_dim ** -0.5`, the attention scale.
    pub(crate) fn attention_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Resolve from the install's declared tower, refusing anything the
    /// kernels cannot address.
    pub(crate) fn resolve(vision: &VisionConfig) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| RealForwardError::Unsupported(detail);
        if !vision.is_active() {
            return Err(unsupported(
                "this install declares no vision tower (arch.vision.depth is 0)".to_string(),
            ));
        }
        let depth = usize::try_from(vision.depth).unwrap_or(0);
        let hidden = usize::try_from(vision.hidden_size).unwrap_or(0);
        let intermediate = usize::try_from(vision.intermediate_size).unwrap_or(0);
        let heads = usize::try_from(vision.num_heads).unwrap_or(0);
        let merge = usize::try_from(vision.spatial_merge_size).unwrap_or(0);
        let out_hidden = usize::try_from(vision.out_hidden_size).unwrap_or(0);
        let patch = usize::try_from(vision.patch_size).unwrap_or(0);
        let temporal = usize::try_from(vision.temporal_patch_size).unwrap_or(0);
        let channels = usize::try_from(vision.in_channels).unwrap_or(0);
        let pos_rows = usize::try_from(vision.num_position_embeddings).unwrap_or(0);

        for (label, value) in [
            ("depth", depth),
            ("hidden_size", hidden),
            ("intermediate_size", intermediate),
            ("num_heads", heads),
            ("spatial_merge_size", merge),
            ("out_hidden_size", out_hidden),
            ("patch_size", patch),
            ("temporal_patch_size", temporal),
            ("in_channels", channels),
            ("num_position_embeddings", pos_rows),
        ] {
            if value == 0 {
                return Err(unsupported(format!(
                    "vision config field {label} is zero, which describes no tower"
                )));
            }
        }

        if hidden % heads != 0 {
            return Err(unsupported(format!(
                "vision hidden_size {hidden} is not divisible by num_heads {heads}"
            )));
        }
        let head_dim = hidden / heads;
        // The rope kernel pairs `i` with `i + head_dim / 2` and the frequency
        // rows carry two axes of `head_dim / 4` each, so a head that is not a
        // multiple of four cannot be built at all. Refused rather than
        // rounded: `vision_rope_freq_rows` makes the same check and a
        // disagreement between the two would be a row-width mismatch at the
        // dispatch instead of a message here.
        if head_dim % 4 != 0 {
            return Err(unsupported(format!(
                "vision head_dim {head_dim} must be a multiple of 4 for the 2-D rope's two axes"
            )));
        }
        if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
            return Err(unsupported(format!(
                "vision head_dim {head_dim} exceeds the attention kernel's register tile \
                 ({}); it would be truncated silently rather than refused",
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }

        let pos_side = (pos_rows as f64).sqrt().round() as usize;
        if pos_side * pos_side != pos_rows {
            return Err(unsupported(format!(
                "vision num_position_embeddings {pos_rows} is not a square grid; the bilinear \
                 interpolation indexes it by (h, w) and has no meaning otherwise"
            )));
        }

        let mut deepstack = Vec::with_capacity(vision.deepstack_visual_indexes.len());
        for &idx in &vision.deepstack_visual_indexes {
            if idx < 0 || idx >= vision.depth {
                return Err(unsupported(format!(
                    "deepstack_visual_indexes holds {idx}, outside this tower's depth of {}",
                    vision.depth
                )));
            }
            deepstack.push(idx as usize);
        }

        Ok(Self {
            depth,
            hidden,
            intermediate,
            heads,
            head_dim,
            merge,
            out_hidden,
            patch_dim: temporal * patch * patch * channels,
            pos_rows,
            pos_side,
            patch_size: patch,
            deepstack,
        })
    }
}
