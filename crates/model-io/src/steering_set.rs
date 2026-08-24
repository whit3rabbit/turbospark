//! A loaded set of per-layer steering directions, and its validation against
//! the model it will be applied to.
//!
//! This is the plain DATA type. It lives here rather than in `crates/runtime`
//! for the reason `context_policy.rs` and `expert_cache_policy.rs` do: it is a
//! pure function of an [`ArchConfig`] and some bytes, it touches no GPU, and
//! two crates on opposite sides of a macOS-only dependency edge both need to
//! name it. `crates/repack` parses a file INTO one (it owns the GGUF parser)
//! and `crates/runtime` consumes one (it owns the dispatch), and
//! `crates/runtime` declares `model_io` and `gpu` under
//! `[target.'cfg(target_os = "macos")'.dependencies]`, so a type declared
//! there would make a portable question macOS-only (AGENTS.md Gotcha 8).
//!
//! # What a direction set is, and is not
//!
//! It is the FILE's content: one direction per layer, plus whatever the file
//! declares about itself. It is NOT the run's parameters -- `alpha`, `target`,
//! the gate threshold and the layer range come from the caller and can change
//! between two generations without reloading anything. That split is what
//! makes "activate and deactivate live" cheap: the expensive half is the file
//! and it is loaded once.

use crate::{ArchConfig, ModelError};
use foundation::SteeringMode;

/// One layer's direction, with its norm precomputed.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDirection {
    /// The direction itself, `hidden` elements, as the file stored it.
    ///
    /// NOT normalized: `SteeringMode::Add` reads the magnitude, because a
    /// llama.cpp control vector carries its strength IN the vector.
    pub values: Vec<f32>,
    /// `1 / ||values||` over the F32 values above, or `0.0` for a zero
    /// direction.
    ///
    /// Precomputed here because it is a property of the direction, constant
    /// across every layer, token and generation -- a per-token kernel
    /// deriving it would reduce over the whole vector on every dispatch to
    /// learn something known at load time. `turbospark_compute::steering`
    /// takes it by the same route so the two sides compute one expression.
    ///
    /// **IT IS NOT THE VALUE THE GPU DISPATCH USES, AND MUST NOT BE PASSED TO
    /// ONE.** `runtime::SteeringState::build` recomputes it from the FP16
    /// values it packs for the kernel, deliberately: the kernel reads FP16 and
    /// reports `c_hat = (d . x) * inv_norm`, so an `inv_norm` taken over a
    /// different `d` than the one being multiplied makes the reported
    /// coefficient disagree with the edit applied -- in the number whose whole
    /// job is to be the measurement. This one is the FILE's, correct for a
    /// caller reasoning in F32 about what the file contains (which is what
    /// `crates/repack`'s reporting target uses it for) and wrong for anything
    /// that dispatches.
    pub inv_norm: f32,
}

impl LayerDirection {
    /// Precomputes [`Self::inv_norm`] from `values`.
    ///
    /// A zero direction yields `0.0` rather than an infinity, which makes
    /// every steering mode the identity on it. That is the honest degenerate
    /// answer -- a direction with no magnitude names no subspace -- and it
    /// keeps the failure FINITE, where the alternative propagates a NaN into
    /// the residual stream and NaN is the one wrong value no downstream
    /// instrument reports as wrong (AGENTS.md Gotcha 59).
    pub fn new(values: Vec<f32>) -> Self {
        let sum_sq: f32 = values.iter().map(|v| v * v).sum();
        let inv_norm = if sum_sq > 0.0 {
            1.0 / sum_sq.sqrt()
        } else {
            0.0
        };
        Self { values, inv_norm }
    }
}

/// Per-layer directions loaded from a control vector file.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringSet {
    /// One entry per layer the file covers, `None` where it carries no
    /// direction for that layer.
    ///
    /// `Option` rather than a zero vector so an uncovered layer costs no
    /// DISPATCH at all, not merely a dispatch that computes nothing. On a
    /// 64-layer model steered at three layers that is the difference between
    /// 3 and 64 extra dispatches per token.
    pub layers: Vec<Option<LayerDirection>>,
    /// Element count of every direction present.
    pub hidden: usize,
    /// The mode the file declares, if it declares one. A caller's
    /// `--steering-mode` overrides it; this is the default when none is
    /// given.
    pub declared_mode: Option<SteeringMode>,
    /// `general.architecture` as stamped in the file, for the startup line
    /// and for the mismatch message.
    pub declared_arch: Option<String>,
}

impl SteeringSet {
    /// The direction for `layer`, or `None` if this set does not cover it.
    pub fn layer(&self, layer: usize) -> Option<&LayerDirection> {
        self.layers.get(layer).and_then(|d| d.as_ref())
    }

    /// How many layers actually carry a direction.
    pub fn covered_layers(&self) -> usize {
        self.layers.iter().filter(|d| d.is_some()).count()
    }

    /// Restrict this set to `start..=end`, dropping every direction outside.
    ///
    /// Applied at LOAD time rather than at each dispatch, so an excluded
    /// layer costs nothing per token and the runtime never has to carry the
    /// range. An empty or inverted range clears the set, which
    /// [`Self::validate`] then refuses -- a run that asked to steer and
    /// silently steered nothing would measure the unsteered engine and report
    /// it as the steered one (the argument `MtpState::build` makes for an
    /// explicitly-requested drafter).
    pub fn restrict_to_range(&mut self, start: usize, end: usize) {
        for (l, slot) in self.layers.iter_mut().enumerate() {
            if l < start || l > end {
                *slot = None;
            }
        }
    }

    /// Checks this set against the model it is about to steer.
    ///
    /// Every failure here is an error at OPEN rather than at the first token,
    /// which is the same contract `MtpState::build`'s `REQUIRED` loop has: a
    /// half-usable direction set should fail where the cause is visible, not
    /// four layers into a forward pass.
    pub fn validate(&self, arch: &ArchConfig) -> Result<(), ModelError> {
        let want = arch.hidden_size as usize;
        if self.hidden != want {
            return Err(ModelError::ArchMismatch {
                field: "steering direction width".to_string(),
                expected: want.to_string(),
                actual: self.hidden.to_string(),
            });
        }
        let layers = arch.num_layers as usize;
        if self.layers.len() > layers {
            return Err(ModelError::ArchMismatch {
                field: "steering direction layer count".to_string(),
                expected: format!("at most {layers}"),
                actual: self.layers.len().to_string(),
            });
        }
        // A set covering FEWER layers than the model is legitimate and common
        // -- steering 2-3 middle layers is often as effective as all of them
        // -- so only an empty one is refused, and it is refused because it
        // would make an asked-for edit a silent no-op.
        if self.covered_layers() == 0 {
            return Err(ModelError::ArchMismatch {
                field: "steering directions".to_string(),
                expected: "at least one layer".to_string(),
                actual: "none (an empty file, or a layer range that excludes every layer)"
                    .to_string(),
            });
        }
        for (l, dir) in self.layers.iter().enumerate() {
            if let Some(d) = dir {
                if d.values.len() != self.hidden {
                    return Err(ModelError::TensorSizeMismatch {
                        name: format!("direction.{}", l + 1),
                        expected: self.hidden as u64,
                        actual: d.values.len() as u64,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(layers: usize, hidden: usize) -> SteeringSet {
        SteeringSet {
            layers: (0..layers)
                .map(|l| Some(LayerDirection::new(vec![(l + 1) as f32; hidden])))
                .collect(),
            hidden,
            declared_mode: None,
            declared_arch: None,
        }
    }

    fn arch(layers: i64, hidden: i64) -> ArchConfig {
        let mut a = crate::arch_baselines::qwen_gdn_dense_27b();
        a.num_layers = layers;
        a.hidden_size = hidden;
        a
    }

    #[test]
    fn inv_norm_is_the_reciprocal_of_the_length() {
        let d = LayerDirection::new(vec![3.0, 4.0]);
        assert!((d.inv_norm - 0.2).abs() < 1e-6, "got {}", d.inv_norm);
    }

    /// The degenerate case must stay FINITE. `1 / 0` here would put an
    /// infinity into a scale that multiplies the residual stream.
    #[test]
    fn a_zero_direction_yields_a_finite_zero_rather_than_an_infinity() {
        let d = LayerDirection::new(vec![0.0; 16]);
        assert_eq!(d.inv_norm, 0.0);
        assert!(d.inv_norm.is_finite());
    }

    #[test]
    fn a_matching_set_validates() {
        assert!(set(4, 8).validate(&arch(4, 8)).is_ok());
    }

    /// A set covering fewer layers than the model is legitimate: steering a
    /// narrow band is the normal case, not a truncated file.
    #[test]
    fn a_short_set_is_accepted_and_a_long_one_is_not() {
        assert!(set(2, 8).validate(&arch(4, 8)).is_ok());
        assert!(set(6, 8).validate(&arch(4, 8)).is_err());
    }

    #[test]
    fn a_width_mismatch_is_refused() {
        let err = set(4, 16).validate(&arch(4, 8)).unwrap_err();
        assert!(
            format!("{err:?}").contains("width"),
            "message should name the width: {err:?}"
        );
    }

    /// The range is applied at load, so an excluded layer costs no dispatch.
    #[test]
    fn restricting_the_range_drops_every_layer_outside_it() {
        let mut s = set(6, 8);
        s.restrict_to_range(2, 4);
        assert_eq!(s.covered_layers(), 3);
        assert!(s.layer(1).is_none());
        assert!(s.layer(2).is_some());
        assert!(s.layer(4).is_some());
        assert!(s.layer(5).is_none());
    }

    /// A range that excludes everything must be an ERROR, never a silent
    /// no-op: a caller who asked to steer and quietly got nothing would
    /// measure the unsteered engine and report it as the steered one.
    #[test]
    fn a_range_that_excludes_every_layer_is_refused_rather_than_ignored() {
        let mut s = set(4, 8);
        s.restrict_to_range(10, 20);
        assert_eq!(s.covered_layers(), 0);
        assert!(s.validate(&arch(4, 8)).is_err());
    }
}
