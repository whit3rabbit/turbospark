//! The runtime half of directional steering: the policy a caller asks for,
//! and the GPU state that serves it.
//!
//! `docs/OBLITERATION.md` is the page; `turbospark_compute::steering` is the
//! numerical contract; `model_io::SteeringSet` is the loaded file.
//!
//! # The split, and why it is where it is
//!
//! [`SteeringPolicy`] is what a RUN asks for: N direction vectors, each with
//! its own mode and alpha, plus the two scalars shared by every vector. Each
//! vector's set is the expensive half and is loaded once, before open, by the
//! front end -- the same shape `resolve_drafter` uses to read a resident
//! index before handing `open` a decision, and for the same reason:
//! `crates/runtime` cannot reach `crates/repack`, which owns the GGUF parser
//! (AGENTS.md Gotcha 8).
//!
//! [`SteeringState`] is what open BUILDS from that: one FP16 buffer holding
//! every vector's directions, one FP32 buffer for the coefficients, and a
//! per-layer table of per-vector dispatch parameters. A layer no vector
//! covers has no entry and costs no dispatch -- not a dispatch that computes
//! nothing, none at all. On a 64-layer model steered at three layers that is
//! 3 extra dispatches per token rather than 64.
//!
//! # Multi-vector semantics
//!
//! Vectors are applied IN ORDER at each steered layer: vector 0's dispatch
//! reads and edits the row, then vector 1's dispatch reads the EDITED row.
//! Composition is therefore sequential by construction -- `encode_steering`
//! literally issues one kernel dispatch per covering vector -- and the
//! reported coefficient of vector k is measured against the stream vector k-1
//! left behind, which is the honest measurement of the edit that actually
//! fired. There is no fused multi-direction kernel to keep parity with; two
//! dispatches of the one existing kernel IS the definition.
//!
//! Per-vector mode and alpha ride with the vector because the measured facts
//! in `docs/OBLITERATION.md` make a shared scalar wrong: usable alpha is a
//! property of the DIRECTION (ocean 0.4 against register 0.8 on the same
//! install), so one knob for two vectors would tune neither. The clamp
//! target and the gate threshold stay shared: the target pins a coefficient
//! that all vectors measure on the same stream, and the gate is evaluated
//! inside each dispatch against that dispatch's own coefficient.
//!
//! # Off allocates nothing
//!
//! [`SteeringPolicy::off`] returns before touching any vector, so an engine
//! with steering off is identical in bytes and in footprint to the one that
//! shipped before this module existed. That is what lets `qwen38`'s frozen
//! quality-gate row and memory-oracle ceiling stand rather than needing new
//! ones, and it is the same guarantee `MtpState::build` gives for `Off`.

use foundation::SteeringMode;
use model_io::{ArchConfig, SteeringSet};

use crate::real_forward_types::RealForwardError;

/// One direction set plus the per-vector parameters that shape its edit.
///
/// The set comes from one control-vector file; `mode` and `alpha` are the
/// run's choices for it. A caller's explicit mode wins over the file's
/// declared one, and the file's wins over the default -- that precedence is
/// applied by whoever loads the file, because only it has both in hand.
#[derive(Debug, Clone)]
pub struct SteeringVector {
    /// The loaded direction set.
    pub set: SteeringSet,
    /// Which edit this vector applies.
    pub mode: SteeringMode,
    /// Strength. `0.0` is the exact identity in every mode, which is what
    /// makes a zero-alpha vector the plumbing probe for the multi-vector
    /// path: N dispatches where one is a no-op must be bit-identical to the
    /// single dispatch.
    pub alpha: f32,
}

/// What a run asks for. [`Self::off`] is the default and allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct SteeringPolicy {
    /// The loaded vectors, in application order. Empty is off.
    pub vectors: Vec<SteeringVector>,
    /// The coefficient [`SteeringMode::Clamp`] pins to; ignored otherwise,
    /// and shared by every vector.
    pub target: f32,
    /// Coefficient magnitude below which an edit does not fire; non-positive
    /// fires always. Evaluated INSIDE the kernel -- a host-side gate would
    /// cost a command-buffer synchronization per layer per token.
    pub gate_threshold: f32,
}

impl SteeringPolicy {
    /// Steering off.
    pub fn off() -> Self {
        Self {
            vectors: Vec::new(),
            target: 0.0,
            gate_threshold: 0.0,
        }
    }

    /// One vector, in the shape every pre-multi-vector caller constructed by
    /// hand. Kept as a constructor so the single-vector sites read as
    /// single-vector intent rather than as a one-element list they have to
    /// see through.
    pub fn single(
        set: SteeringSet,
        mode: SteeringMode,
        alpha: f32,
        target: f32,
        gate_threshold: f32,
    ) -> Self {
        Self {
            vectors: vec![SteeringVector { set, mode, alpha }],
            target,
            gate_threshold,
        }
    }

    /// Whether this policy would actually edit anything.
    pub fn is_active(&self) -> bool {
        !self.vectors.is_empty()
    }

    /// The first vector, for readers that report single-vector facts.
    ///
    /// The FFI capability block is the reason this exists rather than every
    /// caller indexing `[0]`: its wire shape carries one mode and one scale,
    /// and until that shape grows an array the honest report for a
    /// multi-vector run is vector 0's, with the full list on the summary
    /// line.
    pub fn primary(&self) -> Option<&SteeringVector> {
        self.vectors.first()
    }

    /// How many vectors this policy applies.
    pub(crate) fn vector_count(&self) -> usize {
        self.vectors.len().max(1)
    }
}

/// One vector's dispatch parameters at one layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerSteer {
    /// Byte offset of this vector's direction for this layer in
    /// [`SteeringState::directions`].
    pub(crate) offset: u64,
    /// `1 / ||d||` computed from the FP16 values the kernel will actually
    /// read. See [`SteeringState::build`] on why not the file's own.
    pub(crate) inv_norm: f32,
    /// This vector's edit, carried per layer because the kernel reads it
    /// from the per-dispatch params rather than from any shared state.
    pub(crate) mode: SteeringMode,
    /// This vector's strength, for the same reason.
    pub(crate) alpha: f32,
}

/// The widest dispatch the coefficient buffer has room for.
///
/// The kernel writes `coeff[row]` for every row it steers, so a block of M
/// rows writes M floats from the layer's own base. One float per layer was
/// enough while the only caller was the per-token pass; the batched verify
/// dispatches M at once, and at the LAST layer those M writes would run off
/// the end of the buffer entirely.
///
/// So the buffer is `num_layers * vector_count * MAX_STEER_ROWS` and each
/// (layer, vector) pair owns a block. The batched driver REFUSES a wider
/// block by name rather than truncating it: a row left unsteered inside a
/// steered block is a token drawn from a different model than its
/// neighbours, which is fluent, wrong, and invisible in every downstream
/// number.
///
/// **It is the batched INT4 GEMM's own cap rather than an independent number,
/// and that is what makes the refusal a BACKSTOP that cannot currently
/// fire.** `gpu::MAX_BATCH_ROWS` bounds the block for an unrelated reason (its
/// accumulators are a per-thread register array), so a wider block is already
/// refused one level down, by name, before any layer is steered. Writing 16
/// here instead would be two independent constants that happen to agree --
/// and if the kernel's ever rose, the coefficient buffer would silently stop
/// being wide enough for the blocks the engine now accepts. Sized from the
/// thing that decides, so they cannot drift apart.
pub const MAX_STEER_ROWS: usize = gpu::MAX_BATCH_ROWS;

/// The GPU-side steering state, built at open.
pub(crate) struct SteeringState {
    /// Every vector's covered directions, FP16, packed back to back in
    /// (vector, coverage) order.
    pub(crate) directions: gpu::MetalBuffer,
    /// `vector_count * MAX_STEER_ROWS` FP32 pre-edit coefficients per layer,
    /// rewritten every pass. A per-token pass writes row 0 of each covered
    /// vector's block and leaves the rest of it alone.
    pub(crate) coeff: gpu::MetalBuffer,
    /// Per model layer, one slot per vector in application order, `None`
    /// where that vector does not cover the layer. The inner length is
    /// always the policy's vector count, so a slot's INDEX is the vector it
    /// belongs to -- a readback or dispatch that iterates with `enumerate`
    /// cannot attribute one vector's coefficient to another.
    pub(crate) layers: Vec<Vec<Option<LayerSteer>>>,
    /// The vector stride between two layers' coefficient blocks. Held rather
    /// than recomputed so the allocation and the offsets cannot disagree.
    pub(crate) coeff_stride: usize,
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
        if policy.vectors.is_empty() {
            return Ok(None);
        }
        if !policy.target.is_finite() || !policy.gate_threshold.is_finite() {
            return Err(RealForwardError::Unsupported(
                "steering parameters must be finite; a non-finite target or gate \
                 puts a NaN into the residual stream, which reads as a PERFECT score on \
                 every rank instrument downstream"
                    .to_string(),
            ));
        }
        for (k, vector) in policy.vectors.iter().enumerate() {
            if !vector.alpha.is_finite() {
                return Err(RealForwardError::Unsupported(format!(
                    "steering vector {k}'s alpha must be finite; a non-finite alpha \
                     puts a NaN into the residual stream, which reads as a PERFECT \
                     score on every rank instrument downstream"
                )));
            }
            vector.set.validate(arch).map_err(|e| {
                RealForwardError::Unsupported(format!("steering vector {k}: {e:?}"))
            })?;
        }

        let num_layers = arch.num_layers as usize;
        let stride = policy.vector_count();

        // One buffer, directions packed in (vector, coverage) order rather
        // than at `layer * hidden`, so a vector steering three layers of
        // sixty-four allocates three rows and not sixty-four.
        let mut packed: Vec<u8> = Vec::new();
        let mut layers: Vec<Vec<Option<LayerSteer>>> = vec![vec![None; stride]; num_layers];
        for (k, vector) in policy.vectors.iter().enumerate() {
            for (layer, slot) in layers.iter_mut().enumerate() {
                let Some(dir) = vector.set.layer(layer) else {
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
                let inv_norm = if sum_sq > 0.0 && sum_sq.is_finite() {
                    1.0 / sum_sq.sqrt()
                } else {
                    return Err(RealForwardError::Unsupported(format!(
                        "steering vector {k}'s direction for layer {layer} has zero norm; \
                         steering direction must be non-zero"
                    )));
                };
                slot[k] = Some(LayerSteer {
                    offset,
                    inv_norm,
                    mode: vector.mode,
                    alpha: vector.alpha,
                });
            }
        }

        Ok(Some(Self {
            directions: context.new_buffer_with_data(&packed),
            coeff: context.new_output_buffer(Self::coeff_bytes(num_layers, stride)),
            layers,
            coeff_stride: stride,
            target: policy.target,
            gate_threshold: policy.gate_threshold,
        }))
    }

    /// This layer's per-vector dispatch slots, indexed by vector. `None`
    /// where that vector does not cover the layer.
    pub(crate) fn layer(&self, layer: usize) -> &[Option<LayerSteer>] {
        self.layers.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// How many layers any vector edits.
    pub(crate) fn covered_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|v| v.iter().any(|s| s.is_some()))
            .count()
    }

    /// Bytes the coefficient buffer needs for `num_layers` layers and
    /// `vectors` vectors.
    ///
    /// Paired with [`Self::coeff_offset`] so the allocation and the write
    /// offsets are two views of one layout rather than two expressions that
    /// have to be kept in step by hand.
    pub(crate) fn coeff_bytes(num_layers: usize, vectors: usize) -> u64 {
        (num_layers.max(1) * vectors.max(1) * MAX_STEER_ROWS * 4) as u64
    }

    /// Byte offset of vector `k`'s coefficient block at `layer`.
    ///
    /// Both dispatch sites go through this rather than computing it, so the
    /// per-token and batched passes cannot disagree about the stride -- which
    /// they would silently, since a wrong stride still writes finite floats
    /// into a valid buffer and only the reported trace would be wrong.
    pub(crate) fn coeff_offset(layer: usize, k: usize, vectors: usize) -> u64 {
        ((layer * vectors.max(1)) + k) as u64 * (MAX_STEER_ROWS as u64) * 4
    }

    /// The pre-edit coefficient each steered (layer, vector) reported on the
    /// last pass.
    ///
    /// Outer index is the layer, inner index the vector, `None` where that
    /// vector does not cover the layer. Row 0 of each block: the single row
    /// of a per-token pass, and the FIRST row of a batched one.
    pub(crate) fn coefficients(&self) -> Vec<Vec<Option<f32>>> {
        let raw = gpu::read_f32_buffer(
            &self.coeff,
            self.layers.len() * self.coeff_stride * MAX_STEER_ROWS,
        );
        self.layers
            .iter()
            .enumerate()
            .map(|(l, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .map(|(k, entry)| {
                        entry.map(|_| raw[(l * self.coeff_stride + k) * MAX_STEER_ROWS])
                    })
                    .collect()
            })
            .collect()
    }

    /// A one-line description for the startup line.
    ///
    /// One vector keeps the exact line every release before multi-vector
    /// steering printed, because that line is rendered by the Swift
    /// capability surface and parsed by eyes that have read it for months.
    pub(crate) fn summary(&self) -> String {
        let gate = if self.gate_threshold > 0.0 {
            format!(", gated at |c| >= {}", self.gate_threshold)
        } else {
            String::new()
        };
        if self.coeff_stride == 1 {
            let e = self
                .layers
                .iter()
                .find(|v| v.iter().any(|s| s.is_some()))
                .and_then(|v| v.iter().flatten().next())
                .expect("one steered layer was found above");
            format!(
                "steering: {} at alpha {} over {} of {} layers{}",
                e.mode.as_str(),
                e.alpha,
                self.covered_layers(),
                self.layers.len(),
                gate
            )
        } else {
            // ONE clause per VECTOR with its coverage, never per (layer,
            // vector) pair: the first real run printed 46 clauses for what
            // was two vectors over 23 layers, and a startup line nobody
            // reads past the first screen is a startup line that hides a
            // misconfigured steer.
            let by_vector: Vec<String> = (0..self.coeff_stride)
                .map(|k| {
                    let mut entry = None;
                    let mut layers = 0usize;
                    for slots in &self.layers {
                        if let Some(e) = slots[k].as_ref() {
                            entry = Some(e);
                            layers += 1;
                        }
                    }
                    let e = entry.expect("a vector in an active policy covers a layer");
                    format!("{}@{} ({} layers)", e.mode.as_str(), e.alpha, layers)
                })
                .collect();
            format!(
                "steering: {} vectors over {} of {} layers: {}{}",
                self.coeff_stride,
                self.covered_layers(),
                self.layers.len(),
                by_vector.join(" then "),
                gate
            )
        }
    }
}

/// Does this family's decode flow dispatch the steering edit and feed the
/// capture?
///
/// **ONE predicate answers both questions, and that is the point.** The edit
/// and the capture land on the same boundary by construction (see
/// [`encode_steering`]), so a family wired for one and not the other is
/// either a direction extracted from a place nothing steers, or a direction
/// applied where nothing was measured. Two lists would let those drift; this
/// one cannot. `real_forward_open` refuses a set on a family this rejects and
/// `resid_capture` declines to open a capture there, both by name rather than
/// silently.
///
/// It is a MATCH and not a default-true test: a new family arrives with no
/// hook in its flow, so the safe answer for an unlisted one is `false` --
/// which fails loudly at open rather than writing a file of zeros
/// (`ffn_hist`'s reason, and `crates/runtime` Gotcha 7's).
pub fn family_dispatches_steering(family: model_io::ModelFamily) -> bool {
    use model_io::ModelFamily as F;
    match family {
        // `families/qwen/`, both halves, per-token and batched.
        F::QwenGdnMoe | F::QwenGdnDense => true,
        // `families/llama/`, both halves: Mixtral's routed experts and the
        // dense Llama / Mistral FFN join at the same boundary, and
        // `Qwen3Moe` runs the same flow. Dense `qwen3` runs the same flow
        // too (`docs/QWEN3_PHASE0.md`) and joins at the same dense-FFN
        // boundary `Llama`'s own dense half does.
        F::Llama | F::Qwen3Moe | F::Qwen3Dense | F::Qwen2Dense | F::MiniMaxM2 => true,
        // `families/gemma4/`, THREE call sites: the sequential/decode routed
        // tail (`mod.rs`), the chunked prefill driver's per-token routed
        // loop (`prefill.rs`, which calls the same tail function and so
        // needs no hook of its own), and the batched-routed prefill's
        // per-row tail (`moe_batch.rs`, reachable only under
        // `TURBOSPARK_ROUTED_BATCH`). All three land on the boundary AFTER
        // `layer_scalar`'s `encode_scalar_mul`, which is the true end of a
        // Gemma layer's contribution to the residual stream -- the residual
        // add alone is not the boundary, because the whole accumulated
        // stream is rescaled by `layer_scalar` immediately after it.
        F::Gemma4 => true,
        // `families/gptoss/`, ONE call site: the routed-MoE tail's raw
        // residual add (no shared expert here, so the routed sum IS the
        // whole join), on the post-mid-layer-commit "routed cb" pass.
        F::GptOss => true,
        // `families/museglimmer/`, ONE call site: the FFN-half sandwich
        // tail's residual add, the layer's true output. No router, so no
        // mid-layer commit -- the whole token runs on one pass and this is
        // simply wherever it currently is.
        F::MuseGlimmer => true,
        // `families/spark/`, ONE call site: the FFN residual add, the same
        // one-stream shape as muse's (dense, no router, no sandwich norm, so
        // the post-FFN add is the layer's whole contribution). Wired with the
        // flow.
        F::Spark25 => true,
        // No hook in the flow yet: DeepSeek-V4-Flash's compressed-attention
        // kernels are unported, so it is refused at open before any decode
        // flow -- there is no layer loop for a hook to sit in.
        F::DeepseekV4Flash => false,
        // No hook in the flow yet (the MLA flow is landing separately); the
        // joins are one-stream residual adds like the muse/spark shape, so
        // the boundary when it lands is a plain residual add.
        F::Deepseek2 => false,
        // Same answer, and the boundary this will need is not the same shape
        // as any `true` arm above. Every family here joins its sublayer
        // output to a ONE-stream residual, so a steering edit is one row at
        // one offset. `qwen4_exp`'s stream is `hc_count` streams wide and a
        // sublayer's output reaches it through a gated INJECT, so "the layer's
        // contribution to the residual" is a different expression and picking
        // the wrong side of it would steer a per-stream mix rather than the
        // stream. Decide it with the flow, not ahead of it.
        F::Qwen4Exp => false,
    }
}

/// Why a family cannot steer, in the ONE wording both readers use, or `None`
/// when it can.
///
/// [`family_dispatches_steering`] answers the question and this answers "say
/// why" -- and both the open-time refusal in `real_forward_open` and the
/// capability block `crates/ffi` reports call this rather than spelling their
/// own. Two spellings would be the same drift the predicate above exists to
/// prevent, one level out: a GUI could tell a user a family steers while the
/// open refuses it, which reads as a broken engine rather than as an
/// unsupported family.
pub fn steering_unsupported_reason(family: model_io::ModelFamily) -> Option<String> {
    if family_dispatches_steering(family) {
        return None;
    }
    Some(format!(
        "steering is not wired for family {family:?}: its flow does not dispatch the edit, \
         so a direction set here would be a silent no-op. Wired today: the qwen flow \
         (both halves), the llama flow (Mixtral, Qwen3-MoE, and the dense Llama / \
         Mistral half), Gemma 4 (sequential decode and chunked prefill, both \
         batched-routed and per-token), gpt-oss, and museGlimmer. Unwired: \
         DeepSeek-V4-Flash (no decode flow exists to hook), deepseek2 (MLA flow, \
         not hooked yet), and qwen4_exp (its residual is hc_count streams wide, \
         so the boundary is a different shape)"
    ))
}

/// Apply this layer's directional-steering edits, one dispatch per covering
/// vector in application order.
///
/// **ONE function serves every dispatch site, and that is the whole design.**
/// It lived in `families/qwen/produce.rs` while the qwen flow was the only
/// caller; `families/qwen/batched.rs` was the second and `families/llama/`
/// the third, so it moved to the module that owns the state rather than
/// staying in one family's file. The alternative is N dispatch sites that
/// must agree on the mode, the alpha, the direction offset, the row stride
/// and the coefficient block, and a disagreement in any one of them is a
/// fluent model that is not the one the caller asked for.
///
/// The BOUNDARY is the caller's to choose and every caller chooses the same
/// one: the residual stream after the FFN join, which is the layer's OUTPUT
/// and is where `resid_capture` lifts from. That is also where llama.cpp
/// applies a control vector (`build_cvec`, between the FFN residual add and
/// `l_out`), which is what makes a vector written here and a vector written
/// there the same edit. Steering at a different boundary than the capture
/// would be a different edit than the one measured.
///
/// Zero dispatches when steering is off, and zero for a layer no vector
/// covers: the per-layer table holds an empty vec there rather than zero
/// vectors, so a set steering three of sixty-four layers costs three
/// dispatches per token and not sixty-four. Multiple covering vectors issue
/// one dispatch each, IN ORDER, each reading the row the previous one left
/// -- see the module docs on why that ordering IS the semantics.
///
/// `rows` is 1 on a per-token path and the block size on a batched verify.
/// `x_off` is the byte offset of the first steered row inside `scratch.x`;
/// every caller before Gemma 4 had exactly one row and it always sat at 0,
/// so the parameter did not exist until a caller needed otherwise. Gemma
/// 4's chunked-prefill driver runs several tokens through one layer's
/// routed tail in a per-token loop, each token's row at its own slot
/// offset (`token * hidden * 2`) inside the SAME `scratch.x` buffer the
/// sequential path uses at offset 0 -- so a caller steering token `t` of a
/// micro-batch must steer ITS row, not row 0 of every call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_steering(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    scratch: &crate::real_forward_types::DecodeScratch,
    steering: Option<&SteeringState>,
    layer: usize,
    hidden: usize,
    rows: usize,
    x_off: u64,
) -> Result<(), RealForwardError> {
    let Some(s) = steering else {
        return Ok(());
    };
    // Checked before the coverage test, so the refusal does not depend on
    // whether THIS layer happens to be steered: a block too wide for the
    // coefficient buffer is a property of the block.
    if rows > MAX_STEER_ROWS {
        return Err(RealForwardError::Unsupported(format!(
            "steering a block of {rows} rows, but the coefficient buffer holds {MAX_STEER_ROWS} \
             per layer; steering fewer rows than the block would draw part of a block from the \
             unsteered model"
        )));
    }
    for (k, slot) in s.layer(layer).iter().enumerate() {
        let Some(entry) = slot else {
            continue;
        };
        gpu::encode_steer_direction(
            context,
            pass,
            (&scratch.x, x_off),
            (&s.directions, entry.offset),
            // A block of FP32 slots per (layer, vector), so the coefficients
            // of a whole pass survive to be read back together after the
            // commit.
            (
                &s.coeff,
                SteeringState::coeff_offset(layer, k, s.coeff_stride),
            ),
            &gpu::SteerParams {
                d_len: hidden as u32,
                rows: rows as u32,
                row_stride: hidden as u32,
                mode: entry.mode,
                alpha: entry.alpha,
                inv_norm: entry.inv_norm,
                target: s.target,
                gate_threshold: s.gate_threshold,
            },
        )
        .map_err(RealForwardError::Gpu)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "steering_tests.rs"]
mod steering_tests;
