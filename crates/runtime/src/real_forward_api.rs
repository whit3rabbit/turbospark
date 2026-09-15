//! Public constructors, property accessors, and runtime controls for [`RealForwardRunner`].

use std::path::Path;

use model_io::{ArchConfig, ExpertCacheSlots, KvQuant};

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

    /// Let the NEXT turn continue from this turn's KV when its prompt begins
    /// with exactly the tokens that built it, instead of re-prefilling the
    /// whole transcript (`crate::kv_prefix`).
    ///
    /// Off by default. A reusing runner is not reset between generations, so
    /// it carries state across them; every frozen row here was measured
    /// without that, and the harnesses run one generation per process where
    /// this could never fire anyway. Multi-turn callers opt in.
    pub fn set_prefix_reuse(&mut self, enabled: bool) {
        self.prefix_reuse_enabled = enabled;
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
    /// `gpt-oss` today, plus the DENSE half of the qwen flow
    /// (`qwenGdnDense`) as long as no drafter is open --
    /// `prefill_chunk_real_qwen_dense`'s named refusal
    /// (`crates/runtime/CLAUDE.md`'s qwen chunked-prefill Gotcha).
    /// The MoE half of qwen (`qwenGdnMoe`) still answers `false`. `qwen4_exp`
    /// answers `true` unconditionally: unlike the dense qwen flow it has no
    /// drafter feature to refuse in the first place (`mod.rs`'s own scope
    /// doc), so there is no open-time condition to check.
    ///
    /// **AN IMAGE PROMPT USED TO BE A THIRD CONJUNCT HERE AND IS NOT ANY
    /// MORE** (2026-09-06): the dense qwen driver mirrors both halves of the
    /// sequential flow's vision handling now, so an install carrying a live
    /// `prompt_vision` map chunks like any other.
    ///
    /// **THAT DRIVER'S REMAINING VISION REFUSAL IS DELIBERATELY NOT MIRRORED
    /// HERE.** `TURBOSPARK_BATCHED_GEMV` plus an image prompt is refused by
    /// name inside the driver, and this predicate must keep answering `true`
    /// for that install: the question here is about the INSTALL, while the
    /// seam is a per-RUN choice the driver's own backstop catches. Folding it
    /// in would make one env var change what an install is reported to
    /// support.
    pub fn supports_chunked_prefill(&self) -> bool {
        self.real.is_some()
            || self.real_llama.is_some()
            || self.real_muse.is_some()
            || self.real_gpt_oss.is_some()
            || (self.real_qwen.as_ref().is_some_and(|s| s.dense)
                && self.real_mtp.is_none()
                && self.real_dflash.is_none())
            || self.real_qwen4.is_some()
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

    /// Which residency mode this runner actually opened with (ROADMAP P1 item 3).
    /// Worth printing next to any throughput or footprint number the same
    /// way [`Self::expert_cache_slots`] is, because a mapped row and a
    /// streamed row differ by the entire slot cache.
    pub fn resolved_expert_residency(&self) -> model_io::ResolvedExpertResidency {
        self.resolved_residency
    }

    /// How many distinct sessions this runner may hold reusable KV/
    /// recurrent state for at once: the one LIVE session plus however many
    /// this opened with in its parked pool (`--session-slots`,
    /// `crate::session_pool`). `1` when no pool was requested, matching
    /// the flag's own default and worth printing next to any throughput or
    /// footprint number the same way [`Self::expert_cache_slots`] is.
    pub fn session_pool_size(&self) -> usize {
        self.session_pool.capacity() + 1
    }

    /// Cumulative phase timings across every `produce` call so far. See
    /// [`PhaseCounters`] for what each bucket covers.
    ///
    /// The expert BYTE counters are summed from the streamers here rather
    /// than folded per layer beside `expert_io_nanos`: each streamer already
    /// keeps its own running total, so reading them once at report time
    /// costs nothing on the decode path and leaves all five routed call
    /// sites untouched.
    pub fn phase_counters(&self) -> PhaseCounters {
        let mut phases = self.phases;
        let mut io = streaming::ExpertIoStats::default();
        for streamer in self.streamers.iter().flatten() {
            io.accumulate(&streamer.io_stats());
        }
        phases.expert_io_bytes_requested = io.bytes_requested;
        phases.expert_io_bytes_physical = io.bytes_physical;
        phases.expert_io_samples = io.samples;
        phases
    }

    /// Flips the shared-expert command buffer (`TURBOSPARK_SHARED_CB`) after
    /// open, so a test can A/B both states in one process. Setting the
    /// environment variable instead would race the other test threads.
    /// Both states must produce identical output; that is the whole
    /// correctness claim of the overlap.
    #[doc(hidden)]
    pub fn set_shared_cb_overlap(&mut self, on: bool) {
        self.shared_cb_overlap = on;
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `TURBOSPARK_ROUTED_PIPELINE`.
    #[doc(hidden)]
    pub fn set_routed_pipeline(&mut self, on: bool) {
        self.routed_pipeline = on;
    }

    /// DIAGNOSTIC for `qwen4_exp`'s QSA: attend densely above the indexer
    /// budget as if every block were selected (`TURBOSPARK_QSA_FORCE_DENSE=1`
    /// at open sets the same flag). The KL between this arm and the sparse
    /// one past 2,051 tokens is the only quantitative instrument the sparse
    /// path has on a real install, since no reference engine for this
    /// checkpoint fits this machine. No-op on every other family.
    #[doc(hidden)]
    pub fn set_qsa_force_dense(&mut self, on: bool) {
        if let Some(qwen4) = self.real_qwen4.as_mut() {
            qwen4.qsa_force_dense = on;
        }
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `TURBOSPARK_ROUTED_BATCH` (the chunked-prefill driver's batched
    /// routed half).
    #[doc(hidden)]
    pub fn set_routed_batch_prefill(&mut self, on: bool) {
        self.routed_batch_prefill = on;
    }

    /// Sibling of [`Self::set_shared_cb_overlap`] for
    /// `TURBOSPARK_BATCHED_GEMV` (the chunked-prefill driver's batched
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
        Self::open_with_slot_policy(
            dir,
            expecting,
            max_context,
            ExpertCacheSlots::Fixed(EXPERT_CACHE_SLOTS),
        )
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
            // Pinned at 1 (no pool): every caller here measures something,
            // and a parked slot is real committed memory a frozen footprint
            // row must not acquire by detection, same reasoning as the two
            // pins above.
            1,
            // Pinned OFF for AGENTS.md Gotcha 35's reason, restated for a
            // third knob: `--kv-bits` moves KV bytes and (through the
            // codec) numerics, so a caller measuring a footprint or a
            // digest through this entry point must not acquire it by
            // detection. `open_with_kv_quant` is the explicit way in.
            KvQuant::Off,
            model_io::ExpertResidency::Streamed,
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
            1,
            KvQuant::Off,
            model_io::ExpertResidency::Streamed,
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
            1,
            KvQuant::Off,
            model_io::ExpertResidency::Auto,
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
            1,
            KvQuant::Off,
            model_io::ExpertResidency::Auto,
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
        Self::open_with_slot_policy_speculation_steering_and_sessions(
            dir,
            expecting,
            max_context,
            slots,
            speculation,
            steering,
            1,
        )
    }

    /// [`RealForwardRunner::open_with_slot_policy_speculation_and_steering`]
    /// carrying a `--session-slots` count as well, so more than one
    /// conversation can reuse its own KV/recurrent state against one runner
    /// instead of each turn discarding the others' (`crate::session_pool`).
    ///
    /// A SIBLING rather than a widened form of the six-argument function
    /// above, on the same reasoning [`Self::open_with_slot_policy`]'s own
    /// doc gives for staying separate from `open_with_options`: that
    /// function has around a dozen existing callers across this crate's own
    /// tests, `crates/cli` and `crates/ffi`, none of which has any reason to
    /// acquire a session pool, and `session_slots <= 1` costs nothing extra
    /// (`model_io::context_policy::session_pool_bytes`), so widening it
    /// would be a no-op diff at every one of those call sites for no
    /// benefit. `turbospark-server` is the only caller of this one, because
    /// it is the only front end whose one runner serves more than one
    /// conversation at a time.
    pub fn open_with_slot_policy_speculation_steering_and_sessions(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
        speculation: crate::families::qwen::DraftPolicies,
        steering: crate::steering::SteeringPolicy,
        session_slots: usize,
    ) -> Result<Self, RealForwardError> {
        Self::open_with_kv_quant(
            dir,
            expecting,
            max_context,
            slots,
            speculation,
            steering,
            session_slots,
            KvQuant::Off,
        )
    }

    /// [`RealForwardRunner::open_with_slot_policy_speculation_steering_and_sessions`]
    /// carrying a `--kv-bits` selection too. The widest sibling: every
    /// earlier `open_with_*` form pins [`KvQuant::Off`] explicitly (AGENTS.md
    /// Gotcha 35's reasoning, restated for this knob in
    /// [`Self::open_with_options`]'s doc), so this is the one entry point
    /// that can acquire TurboQuant KV quantization at all. `turbospark-check`
    /// and `turbospark-server` are its callers.
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_kv_quant(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
        speculation: crate::families::qwen::DraftPolicies,
        steering: crate::steering::SteeringPolicy,
        session_slots: usize,
        kv_quant: KvQuant,
    ) -> Result<Self, RealForwardError> {
        Self::open_with_residency(
            dir,
            expecting,
            max_context,
            slots,
            speculation,
            steering,
            session_slots,
            kv_quant,
            model_io::ExpertResidency::Auto,
        )
    }

    /// [`RealForwardRunner::open_with_kv_quant`] carrying an explicit
    /// [`model_io::ExpertResidency`] policy. The widest entry point, used by
    /// `turbospark-check` and `turbospark-server` to thread `--expert-residency`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_residency(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        slots: ExpertCacheSlots,
        speculation: crate::families::qwen::DraftPolicies,
        steering: crate::steering::SteeringPolicy,
        session_slots: usize,
        kv_quant: KvQuant,
        residency: model_io::ExpertResidency,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            slots,
            None,
            speculation,
            steering,
            session_slots,
            kv_quant,
            residency,
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

    /// The pre-edit coefficient each steered (layer, vector) reported on the
    /// last forward pass, `None` where that vector does not cover the layer.
    ///
    /// This is the measurement the edit produces for free: how much of the
    /// direction the residual stream carried at each layer. See
    /// `docs/OBLITERATION.md`.
    pub fn steering_coefficients(&self) -> Option<Vec<Vec<Option<f32>>>> {
        self.steering.as_ref().map(|s| s.coefficients())
    }

    /// Whether this session's family dispatches the steering edit at all.
    ///
    /// **NOT "is steering on"** -- that is `steering_line().is_some()`. This
    /// answers the question a caller has BEFORE it offers the control:
    /// requesting a direction set on a family that answers `false` is refused
    /// at open (`real_forward_open`), so a GUI that offers the knob anyway is
    /// offering one whose only outcome is a failed load.
    pub fn steering_supported(&self) -> bool {
        crate::steering::family_dispatches_steering(self.arch.family)
    }

    /// Why [`RealForwardRunner::steering_supported`] is false, in the exact
    /// words the open-time refusal would use, or `None` when it is true.
    pub fn steering_unsupported_reason(&self) -> Option<String> {
        crate::steering::steering_unsupported_reason(self.arch.family)
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
            1,
            KvQuant::Off,
            model_io::ExpertResidency::Streamed,
        )
    }
}
