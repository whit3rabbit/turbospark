//! The runtime half of directional steering: the policy a caller asks for,
//! and the GPU state that serves it.
//!
//! `docs/OBLITERATION.md` is the page; `turbospark_compute::steering` is the
//! numerical contract; `model_io::SteeringSet` is the loaded file.
//!
//! # The split, and why it is where it is
//!
//! [`SteeringPolicy`] is what a RUN asks for: a direction set plus the four
//! scalars that shape the edit. The set is the expensive half and is loaded
//! once, before open, by the front end -- the same shape `resolve_drafter`
//! uses to read a resident index before handing `open` a decision, and for the
//! same reason: `crates/runtime` cannot reach `crates/repack`, which owns the
//! GGUF parser (AGENTS.md Gotcha 8).
//!
//! [`SteeringState`] is what open BUILDS from that: one FP16 buffer holding
//! every direction, one FP32 buffer for the coefficients, and a per-layer
//! table of offsets. A layer the set does not cover has no entry and costs no
//! dispatch -- not a dispatch that computes nothing, none at all. On a
//! 64-layer model steered at three layers that is 3 extra dispatches per
//! token rather than 64.
//!
//! # Off allocates nothing
//!
//! [`SteeringPolicy::off`] returns before touching the set, so an engine with
//! steering off is identical in bytes and in footprint to the one that
//! shipped before this module existed. That is what lets `qwen38`'s frozen
//! quality-gate row and memory-oracle ceiling stand rather than needing new
//! ones, and it is the same guarantee `MtpState::build` gives for `Off`.

use foundation::SteeringMode;
use model_io::{ArchConfig, SteeringSet};

use crate::real_forward_types::RealForwardError;

/// What a run asks for. [`Self::off`] is the default and allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct SteeringPolicy {
    /// The loaded direction set. `None` is off.
    pub set: Option<SteeringSet>,
    /// Which edit to apply.
    pub mode: SteeringMode,
    /// Strength. `0.0` is the exact identity in every mode.
    pub alpha: f32,
    /// The coefficient [`SteeringMode::Clamp`] pins to; ignored otherwise.
    pub target: f32,
    /// Coefficient magnitude below which the edit does not fire; non-positive
    /// fires always. Evaluated INSIDE the kernel -- a host-side gate would
    /// cost a command-buffer synchronization per layer per token.
    pub gate_threshold: f32,
}

impl SteeringPolicy {
    /// Steering off.
    pub fn off() -> Self {
        Self {
            set: None,
            mode: SteeringMode::Ablate,
            alpha: 0.0,
            target: 0.0,
            gate_threshold: 0.0,
        }
    }

    /// Whether this policy would actually edit anything.
    pub fn is_active(&self) -> bool {
        self.set.is_some()
    }
}

/// One layer's resolved dispatch parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerSteer {
    /// Byte offset of this layer's direction in [`SteeringState::directions`].
    pub(crate) offset: u64,
    /// `1 / ||d||` computed from the FP16 values the kernel will actually
    /// read. See [`SteeringState::build`] on why not the file's own.
    pub(crate) inv_norm: f32,
}

/// The GPU-side steering state, built at open.
pub(crate) struct SteeringState {
    /// Every covered layer's direction, FP16, packed back to back.
    pub(crate) directions: gpu::MetalBuffer,
    /// One FP32 pre-edit coefficient per layer, rewritten every pass.
    pub(crate) coeff: gpu::MetalBuffer,
    /// Per model layer, `None` where this set steers nothing.
    pub(crate) layers: Vec<Option<LayerSteer>>,
    pub(crate) mode: SteeringMode,
    pub(crate) alpha: f32,
    pub(crate) target: f32,
    pub(crate) gate_threshold: f32,
}

impl SteeringState {
    /// Builds the GPU state, or `Ok(None)` when the policy is off.
    ///
    /// Failures here are errors at OPEN rather than at the first token, which
    /// is `MtpState::build`'s contract for the same reason: a caller who
    /// asked to steer and silently got nothing would measure the unsteered
    /// engine and report it as the steered one.
    pub(crate) fn build(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        policy: &SteeringPolicy,
    ) -> Result<Option<Self>, RealForwardError> {
        // Returns BEFORE touching anything, so `off` allocates nothing.
        let Some(set) = policy.set.as_ref() else {
            return Ok(None);
        };
        if !policy.alpha.is_finite()
            || !policy.target.is_finite()
            || !policy.gate_threshold.is_finite()
        {
            return Err(RealForwardError::Unsupported(
                "steering parameters must be finite; a non-finite alpha, target or gate \
                 puts a NaN into the residual stream, which reads as a PERFECT score on \
                 every rank instrument downstream"
                    .to_string(),
            ));
        }
        set.validate(arch)
            .map_err(|e| RealForwardError::Unsupported(format!("steering set: {e:?}")))?;

        let hidden = arch.hidden_size as usize;
        let num_layers = arch.num_layers as usize;
        let covered = set.covered_layers();

        // One buffer, directions packed in COVERAGE order rather than at
        // `layer * hidden`, so a set steering three layers of sixty-four
        // allocates three rows and not sixty-four.
        let mut packed: Vec<u8> = Vec::with_capacity(covered * hidden * 2);
        let mut layers: Vec<Option<LayerSteer>> = vec![None; num_layers];
        for (layer, slot) in layers.iter_mut().enumerate() {
            let Some(dir) = set.layer(layer) else {
                continue;
            };
            let offset = packed.len() as u64;

            // THE INV_NORM IS RECOMPUTED FROM THE ROUNDED VALUES, not taken
            // from the file's f32 ones. The kernel reads FP16 and reports
            // `c_hat = (d . x) * inv_norm`, so an `inv_norm` derived from a
            // different `d` than the one multiplied would make the reported
            // coefficient disagree with the edit that was applied -- by a
            // small amount, in a number whose whole job is to be the
            // measurement. Consistency is worth more here than the third
            // decimal place.
            let mut sum_sq = 0.0f32;
            for v in &dir.values {
                let h = half::f16::from_f32(*v);
                packed.extend_from_slice(&h.to_bits().to_le_bytes());
                let r = h.to_f32();
                sum_sq += r * r;
            }
            let inv_norm = if sum_sq > 0.0 {
                1.0 / sum_sq.sqrt()
            } else {
                0.0
            };
            *slot = Some(LayerSteer { offset, inv_norm });
        }

        Ok(Some(Self {
            directions: context.new_buffer_with_data(&packed),
            coeff: context.new_output_buffer((num_layers.max(1) * 4) as u64),
            layers,
            mode: policy.mode,
            alpha: policy.alpha,
            target: policy.target,
            gate_threshold: policy.gate_threshold,
        }))
    }

    /// This layer's dispatch parameters, or `None` if it is not steered.
    pub(crate) fn layer(&self, layer: usize) -> Option<LayerSteer> {
        self.layers.get(layer).copied().flatten()
    }

    /// How many layers this state edits.
    pub(crate) fn covered_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }

    /// The pre-edit coefficient each steered layer reported on the last pass.
    ///
    /// `None` for a layer that is not steered, so a caller cannot mistake an
    /// unwritten slot for a measured zero.
    pub(crate) fn coefficients(&self) -> Vec<Option<f32>> {
        let raw = gpu::read_f32_buffer(&self.coeff, self.layers.len());
        self.layers
            .iter()
            .enumerate()
            .map(|(l, slot)| slot.map(|_| raw[l]))
            .collect()
    }

    /// A one-line description for the startup line.
    pub(crate) fn summary(&self) -> String {
        format!(
            "steering: {} at alpha {} over {} of {} layers{}",
            self.mode.as_str(),
            self.alpha,
            self.covered_layers(),
            self.layers.len(),
            if self.gate_threshold > 0.0 {
                format!(", gated at |c| >= {}", self.gate_threshold)
            } else {
                String::new()
            }
        )
    }
}
