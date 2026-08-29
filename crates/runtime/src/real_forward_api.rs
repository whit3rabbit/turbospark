//! Public constructors, property accessors, and runtime controls for [`RealForwardRunner`].

use std::path::Path;

use model_io::{ArchConfig, ExpertCacheSlots};

use crate::real_forward::{RealForwardRunner, DEFAULT_MAX_CONTEXT, EXPERT_CACHE_SLOTS};
use crate::real_forward_types::{PhaseCounters, RealForwardError};

impl RealForwardRunner {
    /// Opens a `.gturbo` install directory whose `manifest.json` matches
    /// `expecting` field-by-field, and whose weights are all resident
    /// (no packed experts): every layer must be full attention
    /// (`full_attention_layer_mask` all `1`). `num_experts` may be `0`
    /// (dense FFN) or positive (routed-expert FFN); see module docs for
    /// what MoE support here does and does not cover.
    pub fn open(dir: &Path, expecting: ArchConfig) -> Result<Self, RealForwardError> {
        Self::open_with_max_context(dir, expecting, DEFAULT_MAX_CONTEXT)
    }

    /// Metal buffers allocated so far by this runner's context. The dense
    /// decode hot path allocates none: all weights are the zero-copy
    /// resident buffer, KV and activation scratch are preallocated at
    /// open. Tests assert this stays flat across generated tokens.
    pub fn gpu_buffer_allocations(&self) -> u64 {
        self.context.buffer_allocation_count()
    }

    /// Whether this install declares a vision tower.
    ///
    /// Asks the ARCH rather than whether one has been opened: the tower is
    /// built lazily, so `self.vision.is_none()` means "no image yet" on an
    /// install that has one.
    pub fn has_vision_tower(&self) -> bool {
        self.arch.vision.is_active()
    }

    /// Bytes the vision tower's slot cache pins, `VISION_SLOTS x
    /// block_stride`. `None` until the tower has been opened.
    ///
    /// This plus [`Self::vision_last_scratch_bytes`] is the whole of the
    /// tower's residency: the blocks stream, so nothing else about the tower
    /// is held between images.
    pub fn vision_slot_bytes(&self) -> Option<u64> {
        self.vision.as_ref().map(|v| v.slot_bytes)
    }

    /// What the last page's scratch actually allocated.
    pub fn vision_last_scratch_bytes(&self) -> Option<u64> {
        self.vision.as_ref().map(|v| v.last_scratch_bytes)
    }

    /// What a page of `seq` patches WOULD cost in scratch, predicted rather
    /// than measured.
    ///
    /// The sizing a caller budgeting a page would use, and the thing
    /// [`Self::vision_last_scratch_bytes`] is worth checking against -- a
    /// formula compared only against itself asserts nothing.
    pub fn vision_scratch_bytes_for(&self, seq: usize) -> Option<u64> {
        self.vision.as_ref().map(|v| v.scratch_bytes(seq))
    }

    /// Run one preprocessed image through the vision tower (ROADMAP M-V4).
    ///
    /// Returns the `[merged_tokens, out_hidden_size]` FP16 rows the trunk's
    /// residual stream wants; M-V5 is what injects them at the image-pad
    /// positions. Nothing in the decode path reads them yet, so calling this
    /// changes no generated token.
    ///
    /// Opens the tower on first use and keeps it for the runner's life --
    /// `VISION_SLOTS x block_stride` of pinned host memory, ~58 MiB on the
    /// real 27B, which a text-only session on the same install never pays.
    /// The per-page scratch is allocated and dropped inside this call.
    ///
    /// An install with no tower is refused BY NAME rather than answering an
    /// empty embedding: a caller that passed an image and silently got no
    /// rows would build a prompt whose image spans are filled with the
    /// placeholder token's own embedding, which reads as a model ignoring the
    /// picture rather than as an install that cannot see one.
    pub fn encode_image(
        &mut self,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
    ) -> Result<crate::vision::VisionEmbedding, RealForwardError> {
        if !self.arch.vision.is_active() {
            return Err(RealForwardError::Unsupported(
                "this install declares no vision tower; repack the checkpoint with its \
                 vision_tower.* tensors to get one"
                    .to_string(),
            ));
        }
        if self.vision.is_none() {
            self.vision = Some(crate::vision::VisionTower::open(
                &self.install_dir,
                &self.context,
                &self.weights,
                &self.index,
                &self.arch,
            )?);
        }
        // Two disjoint fields of `self`, which is what lets the tower take
        // the context mutably while it is itself borrowed mutably.
        let tower = self.vision.as_mut().expect("opened just above");
        tower.run(&mut self.context, &self.weights, image, params)
    }

    /// Whether [`crate::producer::ChunkedPrefillRunner::prefill_chunk`] would
    /// serve this install rather than refuse it by name.
    ///
    /// The ONE place that decision is made: `prefill_chunk`'s own refusal
    /// calls this too, so a caller deciding whether to route through the
    /// chunked driver at all (`crates/cli`'s default `--prefill-chunk`
    /// wiring, `crates/server`'s automatic dispatch) and the driver's own
    /// hard refusal can never disagree. Gemma 4, BOTH halves of `llama`
    /// (Mistral, Llama 2/3.x, Mixtral, `qwen3moe`), `muse_glimmer` and
    /// `gpt-oss` today; only the qwen flow answers `false`.
    pub fn supports_chunked_prefill(&self) -> bool {
        self.real.is_some()
            || self.real_llama.is_some()
            || self.real_muse.is_some()
            || self.real_gpt_oss.is_some()
    }

    /// Rows this model's output head writes, i.e. the length every `produce`
    /// logits buffer must have.
    ///
    /// **This is a property of the MODEL, and callers used to take it from
    /// the TOKENIZER's dialect instead.** `MfTokenizer::vocab_size` is a
    /// per-dialect constant standing in for the checkpoint's padded
    /// embedding row count, which is correct only while one model uses a
    /// dialect: Qwen 3.6 and Qwen3-30B-A3B are both ChatML and pad to
    /// 248,320 and 151,936 rows respectively, so the second one failed at
    /// the first token with a vocab mismatch. Ask the runner.
    pub fn vocab_size(&self) -> usize {
        self.arch.vocab_size as usize
    }

    /// The per-layer routed-expert slot count this runner actually opened
    /// with, after [`ExpertCacheSlots::Auto`] resolved and after the cap at
    /// the install's own expert count.
    ///
    /// Worth printing next to any throughput or footprint number, because
    /// under `Auto` it is a property of the machine: `docs/DECODE_BUDGET.md`
    /// measures 44.2 tok/s at 16 slots against 51.2 at 32 on one install.
    pub fn expert_cache_slots(&self) -> usize {
        self.expert_cache_slots
    }

    /// Cumulative phase timings across every `produce` call so far. See
    /// [`PhaseCounters`] for what each bucket covers.
    pub fn phase_counters(&self) -> PhaseCounters {
        self.phases
    }

    /// Flips the shared-expert command buffer (`MFERENCE_SHARED_CB`) after
    /// open, so a test can A/B both states in one process. Setting the
    /// environment variable instead would race the other test threads.
    /// Both states must produce identical output; that is the whole
    /// correctness claim of the overlap.
    #[doc(hidden)]
    pub fn set_shared_cb_overlap(&mut self, on: bool) {
        self.shared_cb_overlap = on;
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `MFERENCE_ROUTED_PIPELINE`.
    #[doc(hidden)]
    pub fn set_routed_pipeline(&mut self, on: bool) {
        self.routed_pipeline = on;
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `MFERENCE_ROUTED_BATCH` (the chunked-prefill driver's batched
    /// routed half).
    #[doc(hidden)]
    pub fn set_routed_batch_prefill(&mut self, on: bool) {
        self.routed_batch_prefill = on;
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `MFERENCE_BATCHED_GEMV` (the chunked-prefill driver's batched
    /// resident GEMVs).
    #[doc(hidden)]
    pub fn set_batched_gemv_prefill(&mut self, on: bool) {
        self.batched_gemv_prefill = on;
    }

    /// [`RealForwardRunner::open`] with an explicit KV capacity: the
    /// per-layer K/V buffers are sized `max_context * kv_stride` up front
    /// (the decode hot path never allocates), so generation past
    /// `max_context` positions is a hard error, matching the loop's own
    /// admission check.
    pub fn open_with_max_context(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
    ) -> Result<Self, RealForwardError> {
        Self::open_with_options(dir, expecting, max_context, EXPERT_CACHE_SLOTS)
    }

    /// [`RealForwardRunner::open_with_max_context`] with the per-layer
    /// streamed-expert slot count too (`--expert-cache-slots`). More slots
    /// means a higher expert residency rate and fewer blocking `pread`s per
    /// token, paid for in pinned host memory: one `expert_stride` buffer
    /// per slot per layer.
    pub fn open_with_options(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: usize,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            ExpertCacheSlots::Fixed(expert_cache_slots),
            None,
            // Pinned OFF, for the same reason this form takes a slot COUNT:
            // every caller that measures something reaches the engine through
            // here, and speculation allocates a head KV plus an M-row scratch.
            // A frozen footprint row must not acquire either by detection.
            // `open_with_options_and_speculation` is the explicit way in.
            crate::families::qwen::DraftPolicies::off(),
            // Pinned OFF for the same reason, and the argument is stronger
            // here: steering CHANGES THE TOKENS. A measuring caller that
            // acquired one by detection would freeze a digest for an edited
            // model and report it as the model's.
            crate::steering::SteeringPolicy::off(),
        )
    }

    /// [`RealForwardRunner::open_with_options`] with the drafting policy
    /// named, for the two MTP probes and the speculative generation gate.
    ///
    /// Separate from `open_with_options` for AGENTS.md Gotcha 35's reason and
    /// separate from `open_with_slot_policy` because a probe pins its slot
    /// count while asking for a drafter, which is neither of the other two
    /// combinations.
    pub fn open_with_options_and_speculation(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: usize,
        speculation: crate::families::qwen::DraftPolicies,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            ExpertCacheSlots::Fixed(expert_cache_slots),
            None,
            speculation,
            crate::steering::SteeringPolicy::off(),
        )
    }

    /// [`RealForwardRunner::open_with_options`] taking a slot POLICY rather
    /// than a count, so a caller can ask for [`ExpertCacheSlots::Auto`] and
    /// have the count sized against this machine and this install at open.
    ///
    /// Deliberately a sibling rather than a widened `open_with_options`.
    /// Every caller that measures something -- `turbospark-bench`, both
    /// memory oracles, every quality gate, `logit_dump` -- passes a pinned
    /// `PROTOCOL_EXPERT_CACHE_SLOTS` through the count-taking form, and
    /// leaving that signature alone is what guarantees none of them can
    /// acquire an environment-sensing default by accident (AGENTS.md Gotcha
    /// 35). The user-facing binaries are the only callers of this one.
    pub fn open_with_slot_policy(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            slots,
            None,
            crate::families::qwen::DraftPolicies::from_env(),
            crate::steering::SteeringPolicy::off(),
        )
    }

    /// [`RealForwardRunner::open_with_slot_policy`] with the drafting policy
    /// named rather than read from the environment.
    ///
    /// What `turbospark-check` and `turbospark-server` call, because
    /// `--speculative` is a FLAG and a flag that lost to an environment
    /// variable would be a knob that silently does nothing.
    pub fn open_with_slot_policy_and_speculation(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
        speculation: crate::families::qwen::DraftPolicies,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            slots,
            None,
            speculation,
            crate::steering::SteeringPolicy::off(),
        )
    }

    /// [`RealForwardRunner::open_with_slot_policy_and_speculation`] carrying
    /// a directional-steering policy as well (`docs/OBLITERATION.md`).
    ///
    /// One entry point taking BOTH rather than a `..._and_steering` sibling
    /// beside the speculation one: the two are independent axes and the
    /// front ends set both, so separate entry points would need a third for
    /// the combination and the count would keep doubling.
    ///
    /// The direction set arrives already PARSED. `crates/runtime` cannot
    /// reach `crates/repack`, which owns the GGUF parser (AGENTS.md Gotcha
    /// 8), so the front end loads the file and hands over a plain
    /// `model_io::SteeringSet` -- the same shape `resolve_drafter` uses to
    /// read a resident index before open and pass in a decision.
    ///
    /// `SteeringPolicy::off()` allocates nothing and encodes nothing, so an
    /// engine opened this way with steering off is identical in bytes and in
    /// footprint to one opened through the function above.
    pub fn open_with_slot_policy_speculation_and_steering(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
        speculation: crate::families::qwen::DraftPolicies,
        steering: crate::steering::SteeringPolicy,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            slots,
            None,
            speculation,
            steering,
        )
    }

    /// A one-line description of the steering state, or `None` when off.
    ///
    /// Reported by the binaries on the startup line beside the resolved slot
    /// count, for that field's reason: an edit applied to every token has to
    /// be readable beside any number taken from the run.
    pub fn steering_line(&self) -> Option<String> {
        self.steering.as_ref().map(|s| s.summary())
    }

    /// The pre-edit coefficient each steered layer reported on the last
    /// forward pass, `None` per layer where nothing is steered.
    ///
    /// This is the measurement the edit produces for free: how much of the
    /// direction the residual stream carried at each layer. See
    /// `docs/OBLITERATION.md`.
    pub fn steering_coefficients(&self) -> Option<Vec<Option<f32>>> {
        self.steering.as_ref().map(|s| s.coefficients())
    }

    /// [`RealForwardRunner::open_with_options`] with an explicit SWA ring
    /// capacity override, for tests that need the ring to wrap after a
    /// handful of tokens instead of `sliding_window + 128`. Not part of
    /// the supported surface.
    #[doc(hidden)]
    pub fn open_with_kv_ring_override(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: usize,
        fp16_ring_capacity_override: Option<usize>,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            ExpertCacheSlots::Fixed(expert_cache_slots),
            fp16_ring_capacity_override,
            crate::families::qwen::DraftPolicies::off(),
            crate::steering::SteeringPolicy::off(),
        )
    }
}
