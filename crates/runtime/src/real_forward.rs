//! A real (not scripted) [`LogitProducer`]: runs an actual dense
//! transformer forward pass through the real GPU kernels
//! `turbospark_gpu` wires (`rmsnorm_no_scale`, `rope_proportional_neox`,
//! `dequant_int4_gemv_simd`, `logit_softcap_softmax`, and the two-pass
//! split-KV decode `attention_decode`) against real, resident,
//! INT4-affine-quantized weights loaded through `turbospark_model_io`.
//! macOS/Metal only, matching `turbospark_gpu`'s own platform gate.
//!
//! Memory model (matching the Swift original):
//! - Weights: ONE zero-copy `MTLBuffer` over the mmap of
//!   `model_weights.bin` (`gpu::ResidentGpuWeights`); every projection
//!   binds weights/scales/biases as offsets into it. No weight bytes are
//!   staged or copied per dispatch.
//! - KV: persistent per-layer GPU buffers (`gpu::KvCacheManager`),
//!   allocated once; the K projection is written directly into its cache
//!   slot by the GEMV and RoPE'd there in place. `attention_k_eq_v`
//!   architectures bind the K buffer as V too, so V buffers stay
//!   untouched (their pages never become resident).
//! - Dispatch: the whole token is encoded into ONE command buffer with a
//!   serial compute encoder (`gpu::PassEncoder`) for dense architectures
//!   -- one commit + one wait per token. The FP16 residual stream and all
//!   intermediates live in preallocated GPU scratch (`DecodeScratch`);
//!   the hot path allocates no Metal buffers. Residual adds and the
//!   gated-FFN activation run on the GPU (`utility.metal`), FP16, as the
//!   Swift original does.
//!
//! MoE layers follow the Swift CB1/CB2 decode shape: the router GEMV
//! rides the first command buffer, its logits are the one host readback
//! per layer (top-k selection + the expert `pread` need them on the CPU),
//! then a second command buffer runs the real vendored `moe.metal`
//! decode kernels (`moe_phase1_gate_up_act_u16load` +
//! `moe_phase2_down_reduce_k8`), reading the expert blobs IN PLACE from
//! the streamer's zero-copy slot buffers through a `RoutedBlobs` argument
//! buffer. No expert byte reaches the host on streamed installs.
//!
//! What this is NOT yet: production parity with the Swift
//! `RealForwardRunner`. Resident-expert MoE installs (a synthetic-only
//! shape) still use the CPU `compute::run_ffn` bridge; top-k selection is
//! host arithmetic (`topk_softmax`), not the `router_topk_select_k8`
//! kernel; the router readback is a full command-buffer wait, not the
//! `MTLSharedEvent` passive wait + phase1-hit/pipelined-CB overlap the
//! Swift original layers on top; and no learned norm weights, separate V
//! projection, per-head q/k norms, or shared-expert branch are wired yet
//! (the `rmsnorm_bf16w` kernel is dispatched and parity-tested, awaiting
//! a real-checkpoint tensor mapping) -- see `DEVIATIONS.md`.
//!
//! Full-attention (mask 1) and sliding-window (mask 0, via attention
//! `kv_start` over the linear KV layout) layers are supported; linear
//! (Qwen GDN) and compressed (DeepSeek DSV4) layers are not (their
//! kernels are unported). FFN may be dense, routed-resident, or
//! routed-streamed. The synthetic installs in `turbospark_repack`
//! (`build_synthetic_gemma4_install` and its `_swa`/`_moe`/
//! `_moe_streamed` variants) are the shapes this runner is exercised
//! against, since no trained `.gturbo` checkpoint exists in this
//! environment.

use std::path::Path;

use foundation::LogitValue;
use model_io::{ArchConfig, ResidentBuffer, ResidentIndex};

use crate::producer::LogitProducer;
use crate::real_forward_layout::{
    moe_offsets_from_layout, readable_resident_dtype, routed_layouts_from_layout,
    RoutedLayerLayout, EXECUTABLE_GGUF_DTYPES, GGUF_BLOCK_DTYPES,
};
pub use crate::real_forward_types::{dispatch_profile_report, PhaseCounters, RealForwardError};
use crate::real_forward_types::{DecodeScratch, ROUTED_BANKS};
use model_io::ExpertCacheSlots;

/// Matches `RuntimeConfig`'s default `expert_cache_slots`; callers that
/// want another allowed value pass it to `open_with_options`.
const EXPERT_CACHE_SLOTS: usize = 16;
/// Mirrors the Swift RuntimeConfiguration default `prefillChunkTokens`
/// (128). Ring capacity per SWA layer is `min(max_context, sliding_window
/// plus this)`, i.e. 1152 for real Gemma 4 at the 4096 default. The chunk
/// headroom is no longer notional: `ChunkedPrefillRunner` writes M KV rows
/// before reading any of them back, and this is the budget that guarantees
/// a sliding-window ring still holds every row the chunk will attend over.
const MAX_PREFILL_CHUNK_TOKENS: usize = 128;
const PACKED_LAYOUT_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// KV capacity when the caller does not say otherwise; matches the CLI's
/// `--max-context` default. `open_with_max_context` overrides it.
const DEFAULT_MAX_CONTEXT: usize = 4096;

pub struct RealForwardRunner {
    pub(crate) context: gpu::MetalContext,
    /// The whole resident region as one zero-copy `MTLBuffer` over the
    /// mmap (see `gpu::ResidentGpuWeights`); every GPU projection binds
    /// weights/scales/biases as offsets into it, Swift-style. No weight
    /// bytes are staged or copied per dispatch.
    pub(crate) weights: gpu::ResidentGpuWeights,
    pub(crate) index: ResidentIndex,
    pub(crate) arch: ArchConfig,
    /// Persistent per-layer GPU K/V buffers (allocated once, written one
    /// token-stride per step, reset via `MADV_DONTNEED`) -- the Swift
    /// original's `KVCacheManager` shape, replacing the old host
    /// `Vec<f16>` history that was re-uploaded whole every token.
    pub(crate) kv: gpu::KvCacheManager,
    pub(crate) scratch: DecodeScratch,
    /// One zero-copy `MTLBuffer` per streamer slot, wrapped once at open
    /// over the slot's page-aligned allocation -- the GPU MoE kernels read
    /// expert weights straight out of these through the `RoutedBlobs`
    /// argument buffer. Declared BEFORE `streamers` so the buffers drop
    /// before the allocations they alias.
    pub(crate) slot_buffers: Vec<Vec<gpu::MetalBuffer>>,
    pub(crate) streamers: Vec<Option<streaming::PreadExpertStreamer>>,
    /// What the slot policy actually resolved to, kept so a caller can
    /// REPORT it. Under `ExpertCacheSlots::Auto` the number is a property of
    /// this machine and this install, so a startup line that echoed the
    /// request rather than the resolution would be describing nothing --
    /// and every throughput or footprint figure has to be read beside it.
    pub(crate) expert_cache_slots: usize,
    /// Blob-relative sub-tensor offsets, ONE ENTRY PER LAYER, and the
    /// reusable argument buffer. Empty when the install does not pack
    /// experts.
    pub(crate) moe_offsets: Vec<gpu::MoeExpertOffsets>,
    pub(crate) routed_blobs: Option<gpu::RoutedBlobsBuffer>,
    /// Banks 1.. of the routed argument buffer, for the chunked prefill
    /// driver alone; bank 0 is `routed_blobs` above, which every family
    /// and the sequential path use. A separate field rather than widening
    /// `routed_blobs` into a `Vec` because five families read that one and
    /// none of them has a second bank to choose from.
    pub(crate) routed_blobs_banks: Vec<gpu::RoutedBlobsBuffer>,
    /// Real-checkpoint (verbatim `language_model.` tensor naming) decode
    /// state: learned norms, INT8 router effective scales, per-expert
    /// scales, layer scalars. `None` for synthetic short-name installs,
    /// which keep the plain no-scale flow. See `real_forward_gemma4.rs`.
    pub(crate) real: Option<crate::families::gemma4::RealGemmaState>,
    /// Real-checkpoint Qwen decode state: the GDN recurrent buffers and
    /// the Qwen-only scratch, for BOTH the MoE `qwen36` family and the dense
    /// `qwen3_5` one (ROADMAP's 1-bit entry), which it tells apart by
    /// `num_experts`. Mutually exclusive with `real`; kept as its own
    /// `Option` rather than folded into an enum so the Gemma path's borrow
    /// shape is untouched.
    pub(crate) real_qwen: Option<crate::families::qwen::RealQwenState>,
    /// The multi-token-prediction head's draft state, for the dense
    /// `qwen3_5` family (`docs/MTP_SPECULATIVE.md`, step 2). `None` unless
    /// `MFERENCE_MTP_DRAFT` asked for a depth AND the install carries a
    /// head, which is read off the resident index rather than a manifest
    /// field so nothing can disagree with the bytes.
    pub(crate) real_mtp: Option<crate::families::qwen::MtpState>,
    /// Present for a `llama`-architecture install (ROADMAP Phase M2), which
    /// is Mixtral-style MoE only; a dense one is refused at build.
    pub(crate) real_llama: Option<crate::families::llama::RealLlamaState>,
    /// Present for a `gpt-oss` install (ROADMAP M5). Its own state rather
    /// than a flag on `real_llama` because all four of this architecture's
    /// differences are inside the layer: a YaRN frequency table, the
    /// per-layer router bias, attention sinks and per-projection biases.
    pub(crate) real_gpt_oss: Option<crate::families::gptoss::RealGptOssState>,
    pub(crate) real_muse: Option<crate::families::museglimmer::RealMuseState>,
    pub(crate) phases: PhaseCounters,
    /// Whether the shared-expert branch rides its own command buffer so it
    /// overlaps the host's expert `pread` (see `real_forward_gemma4.rs`).
    /// `MFERENCE_SHARED_CB=0` reverts to encoding it after the pread, the
    /// A/B seam the Swift original keeps as `MFERENCE_ROUTER_EVENT=0`:
    /// same kernels, same order, identical output, different overlap.
    pub(crate) shared_cb_overlap: bool,
    /// Whether a layer's routed-expert command buffer is committed at the
    /// END of that layer and retired one layer later (after the next
    /// layer's router wait), Swift's one-layer-pipelined routed CB.
    /// `MFERENCE_ROUTED_PIPELINE=0` reverts to folding the routed work
    /// uncommitted into the next layer's first command buffer: same
    /// kernels, same order, identical output, different overlap.
    pub(crate) routed_pipeline: bool,
    /// Which layout each layer's routed expert blobs use, PER PHASE.
    pub(crate) routed_layouts: Vec<RoutedLayerLayout>,
    /// Per-layer expert-selection histogram, `None` unless
    /// `MFERENCE_ROUTER_HIST=/path.json`. Written on drop; see
    /// `router_hist.rs`.
    pub(crate) router_hist: Option<crate::router_hist::RouterHistogram>,
    /// Dense-FFN activation census, `None` unless
    /// `MFERENCE_FFN_HIST=/path.json` on a family whose flow feeds the
    /// capture (museGlimmer today). Written on drop; see `ffn_hist.rs`.
    pub(crate) ffn_hist: Option<crate::ffn_hist::FfnActHist>,
    /// Set for the duration of one [`LogitProducer::produce_prefill`] call:
    /// the caller is discarding this token's logits, so the output head
    /// (final norm, full-vocab GEMV, softcap, host readback) is skipped.
    /// Everything else, including committing and waiting on the open command
    /// buffer and advancing the KV cache, still runs.
    pub(crate) skip_head: bool,
}

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
            crate::families::qwen::MtpDraftPolicy::Off,
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
        speculation: crate::families::qwen::MtpDraftPolicy,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(
            dir,
            expecting,
            max_context,
            ExpertCacheSlots::Fixed(expert_cache_slots),
            None,
            speculation,
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
            crate::families::qwen::MtpDraftPolicy::from_env(),
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
        speculation: crate::families::qwen::MtpDraftPolicy,
    ) -> Result<Self, RealForwardError> {
        Self::open_inner(dir, expecting, max_context, slots, None, speculation)
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
            crate::families::qwen::MtpDraftPolicy::Off,
        )
    }

    fn open_inner(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: ExpertCacheSlots,
        fp16_ring_capacity_override: Option<usize>,
        speculation: crate::families::qwen::MtpDraftPolicy,
    ) -> Result<Self, RealForwardError> {
        if expert_cache_slots == ExpertCacheSlots::Fixed(0) {
            return Err(RealForwardError::Unsupported(
                "expert_cache_slots must be positive".to_string(),
            ));
        }
        crate::real_forward_init::validate_arch_config(&expecting)?;

        model_io::load_manifest(dir, &expecting, model_io::DEFAULT_MAX_BYTES)
            .map_err(RealForwardError::Model)?;
        let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
            .map_err(RealForwardError::Model)?;

        if let Some(entry) = index.entries.values().find(|e| {
            GGUF_BLOCK_DTYPES.contains(&e.dtype) && !EXECUTABLE_GGUF_DTYPES.contains(&e.dtype)
        }) {
            // The message names the RESIDENT question, not a global one: a
            // type can have routed kernels and no GEMV (MXFP4 does), so
            // "has no kernel in this port" was about to become false while
            // the refusal stayed correct.
            return Err(RealForwardError::Unsupported(format!(
                "tensor {} carries GGUF block dtype {}, which has no RESIDENT kernel in this \
                 port (ROADMAP Phase G Stage 2; resident-executable tags: {:?})",
                entry.name, entry.dtype, EXECUTABLE_GGUF_DTYPES
            )));
        }
        // The same question for every OTHER tag, and the one that catches the
        // quiet half. The check above only looks at tags the writer calls
        // GGUF blocks; an unquantized tensor written FP16 (tag 2) or FP32
        // (tag 3) passes it and is then decoded as BF16 off its byte size,
        // because that is what every unquantized reader here does. FP16 is
        // the same width, so nothing fails and the values are wrong by up to
        // 2^112 -- fluent garbage from an install that opened cleanly.
        if let Some(entry) = index
            .entries
            .values()
            .find(|e| !readable_resident_dtype(e.dtype))
        {
            return Err(RealForwardError::Unsupported(format!(
                "tensor {} carries resident dtype {}, which no reader in this crate honours; \
                 unquantized tensors must be narrowed to BF16 (tag 1) at repack time",
                entry.name, entry.dtype
            )));
        }

        let buffer = ResidentBuffer::map(
            &dir.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .map_err(RealForwardError::Model)?;
        let mut context = gpu::MetalContext::new().map_err(RealForwardError::Gpu)?;
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer)
            .map_err(RealForwardError::Gpu)?;

        let kv = gpu::KvCacheManager::new(
            context.device(),
            &expecting,
            max_context,
            true,
            None,
            MAX_PREFILL_CHUNK_TOKENS,
            fp16_ring_capacity_override,
        )
        .map_err(RealForwardError::Gpu)?;
        let scratch = DecodeScratch::new(&context, &expecting);

        let (streamers, slot_buffers, experts_layout, resolved_slots) =
            crate::real_forward_init::open_expert_streamers(
                dir,
                &expecting,
                expert_cache_slots,
                index.header.resident_size,
                PACKED_LAYOUT_MAX_BYTES,
                &mut context,
            )?;

        let use_silu = expecting.hidden_activation.contains("silu");
        let (moe_offsets, routed_layouts, routed_blobs, routed_blobs_banks) = match &experts_layout
        {
            Some(layout) => {
                let offsets = moe_offsets_from_layout(layout)?;
                let layouts = routed_layouts_from_layout(layout)?;
                let routed = gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                    .map_err(RealForwardError::Gpu)?;
                let mut banks = Vec::with_capacity(ROUTED_BANKS - 1);
                for _ in 1..ROUTED_BANKS {
                    banks.push(
                        gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                            .map_err(RealForwardError::Gpu)?,
                    );
                }
                (offsets, layouts, Some(routed), banks)
            }
            None => (Vec::new(), Vec::new(), None, Vec::new()),
        };
        drop(experts_layout);

        let router_hist = crate::router_hist::RouterHistogram::from_env(
            expecting.num_layers as usize,
            expecting.num_experts.max(0) as usize,
        );
        let ffn_hist = crate::ffn_hist::FfnActHist::from_env(&context, &expecting);
        let mut runner = Self {
            context,
            weights,
            index,
            arch: expecting,
            kv,
            scratch,
            slot_buffers,
            expert_cache_slots: resolved_slots,
            streamers,
            moe_offsets,
            routed_blobs,
            routed_blobs_banks,
            real: None,
            real_qwen: None,
            real_mtp: None,
            real_llama: None,
            real_gpt_oss: None,
            real_muse: None,
            phases: PhaseCounters::default(),
            shared_cb_overlap: std::env::var("MFERENCE_SHARED_CB").as_deref() != Ok("0"),
            routed_pipeline: std::env::var("MFERENCE_ROUTED_PIPELINE").as_deref() != Ok("0"),
            routed_layouts,
            router_hist,
            ffn_hist,
            skip_head: false,
        };
        match runner.arch.family {
            model_io::ModelFamily::Gemma4 => {
                if runner
                    .index
                    .entries
                    .contains_key("language_model.model.embed_tokens.weight")
                {
                    runner.real = Some(crate::families::gemma4::RealGemmaState::build(
                        &mut runner.context,
                        &runner.weights,
                        &runner.index,
                        &runner.arch,
                    )?);
                }
            }
            // One flow for both, on the same footing `llama` and `qwen3moe`
            // share `families/llama/`'s: every BEHAVIOURAL field of
            // `qwen_gdn_dense_27b()` equals `qwen_gdn_moe_35b_a3b()`'s and every SHAPE field
            // differs, so `qwen3_5` is the DENSE half of this flow and not a
            // sixth one. `RealQwenState` carries the split, off `num_experts`.
            model_io::ModelFamily::QwenGdnMoe | model_io::ModelFamily::QwenGdnDense => {
                runner.real_qwen = Some(crate::families::qwen::RealQwenState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
                // The speculative drafter, and it builds NOTHING unless a
                // depth was asked for -- so an install that has a head is
                // byte-identical and footprint-identical to one that does
                // not until someone turns drafting on
                // (`docs/MTP_SPECULATIVE.md`, step 2).
                // The GDN shape comes from the state built immediately above:
                // step 4's batched scratch needs the recurrent widths, and
                // deriving them a second time here would be a second place
                // for them to be wrong.
                let gdn_shape = runner
                    .real_qwen
                    .as_ref()
                    .expect("real Qwen state built above")
                    .shape;
                runner.real_mtp = crate::families::qwen::MtpState::build(
                    &mut runner.context,
                    &runner.index,
                    &runner.arch,
                    max_context,
                    speculation,
                    gdn_shape,
                )?;
            }
            // One flow for both: `qwen3moe` is the same layer graph, and
            // `RealLlamaState` carries the two differences (per-head q/k
            // norms, a different RMS epsilon).
            model_io::ModelFamily::Llama | model_io::ModelFamily::Qwen3Moe => {
                runner.real_llama = Some(crate::families::llama::RealLlamaState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
            }
            model_io::ModelFamily::DeepseekV4Flash => {
                return Err(RealForwardError::Unsupported(
                    "the DeepSeek-V4-Flash family has no decode flow yet".to_string(),
                ));
            }
            // A FIFTH FLOW, not a sixth family on an existing one: all four
            // of `gpt-oss`'s differences (per-projection biases, attention
            // sinks, YaRN rope scaling, a clamped SwiGLU) are INSIDE the
            // layer, and each produces fluent wrong output rather than an
            // error if a neighbour's flow is used instead.
            model_io::ModelFamily::GptOss => {
                runner.real_gpt_oss = Some(crate::families::gptoss::RealGptOssState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
            }
            // A SIXTH FLOW, on the same reasoning `gpt-oss` got the fifth:
            // ten differences, every one inside the layer. See
            // `ModelFamily::MuseGlimmer` for the list.
            model_io::ModelFamily::MuseGlimmer => {
                runner.real_muse = Some(crate::families::museglimmer::RealMuseState::build(
                    &mut runner.context,
                    &runner.index,
                    &runner.arch,
                )?);
            }
        }
        Ok(runner)
    }

    /// Captures everything a forward pass mutates that is not addressed by
    /// position, so a later [`Self::rollback`] can undo an arbitrary run of
    /// tokens. The two halves need opposite treatment and that asymmetry is
    /// the whole content of this pair:
    ///
    /// - the KV cache keeps one row per position, so undoing it is moving a
    ///   cursor and nothing is copied here;
    /// - a gated-DeltaNet layer folds its whole history into a fixed-size
    ///   accumulator through a non-invertible update, so the only way back
    ///   is a copy taken beforehand (~60 MiB on Qwen 3.6, a memcpy).
    ///
    /// Take it BEFORE the tokens you may want to drop, and only when the
    /// previous pass has completed -- `produce` waits on its own command
    /// buffer before returning, so any point between calls is safe.
    pub fn checkpoint(&self) -> RollbackPoint {
        RollbackPoint {
            position: self.kv.position(),
            gdn: self.real_qwen.as_ref().map(|qwen| qwen.gdn.snapshot()),
        }
    }

    /// Returns the runner to a [`Self::checkpoint`]. Afterwards the next
    /// `produce` must be called at `point.position`, and any tokens to keep
    /// are replayed through it.
    ///
    /// Panics if the KV rewind is not safe, which on a sliding-window model
    /// means more tokens than the ring's slack; see
    /// `gpu::KvCacheManager::max_safe_rewind`. That is deliberate: the
    /// alternative is attending over rows this generation has already
    /// overwritten, which produces plausible text rather than an error.
    pub fn rollback(&mut self, point: &RollbackPoint) {
        assert!(
            point.position <= self.kv.position(),
            "rollback target is ahead of the cursor"
        );
        self.kv.rewind_by(self.kv.position() - point.position);
        if let (Some(qwen), Some(snapshot)) = (self.real_qwen.as_mut(), point.gdn.as_ref()) {
            qwen.gdn.restore(snapshot);
        }
    }

    /// The largest number of tokens [`Self::rollback`] can undo. Bounds the
    /// speculative block size on a sliding-window model; unbounded (in
    /// practice `usize::MAX`) on one whose layers all keep full history.
    pub fn max_rollback(&self) -> usize {
        self.kv.max_safe_rewind()
    }
}

/// An opaque restore point from [`RealForwardRunner::checkpoint`].
pub struct RollbackPoint {
    position: usize,
    gdn: Option<gpu::GdnSnapshot>,
}

impl RollbackPoint {
    /// The KV position this point restores to.
    pub fn position(&self) -> usize {
        self.position
    }
}

impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        self.kv.reset();
        if let Some(qwen) = self.real_qwen.as_mut() {
            qwen.reset();
        }
        // The head keeps its OWN KV, so the trunk's reset leaves it holding
        // the previous generation's context -- the same shape of leak
        // Gotcha 4 records for the GDN recurrent state, one cache over.
        if let Some(mtp) = self.real_mtp.as_mut() {
            mtp.reset();
        }
    }

    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        gpu::autorelease_pool(|| self.produce_inner(token, position, logits))
            .map_err(|e| e.to_string())
    }

    fn produce_prefill(
        &mut self,
        token: i32,
        position: usize,
        scratch: &mut [LogitValue],
    ) -> Result<(), String> {
        self.skip_head = true;
        let result = self.produce(token, position, scratch);
        self.skip_head = false;
        result
    }
}

/// The MTP head as a drafter and the M-row pass as the verify
/// (`docs/MTP.md`). Every method here is a thin forward to an inherent one
/// that already existed and was reachable only from
/// `crates/bench/tests/mtp_accept_length_probe.rs`; the trait is what lets
/// `run_raw_completion_speculative` drive them.
///
/// The three refusals the pieces carry are NOT re-stated here, deliberately.
/// `mtp_draft_step` errors on an install with no head or a step off the
/// drafter's cursor, and `produce_batched` refuses a non-dense, non-INT4 or
/// KV-wrapping block by name. Restating them would be a second copy of a
/// condition that has to agree with the first, and the failure mode of
/// disagreeing is a refusal message that names the wrong cause.
impl crate::producer::SpeculativeProducer for RealForwardRunner {
    type Checkpoint = RollbackPoint;

    fn prime_drafter(&mut self, next: i32, position: usize) -> Result<(), String> {
        self.mtp_prime_step(next, position)
            .map_err(|e| e.to_string())
    }

    fn draft_step(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.mtp_draft_step(token, position, logits)
            .map_err(|e| e.to_string())
    }

    fn rewind_drafter(&mut self, position: usize) -> Result<(), String> {
        self.mtp_rewind_to(position).map_err(|e| e.to_string())
    }

    fn checkpoint(&mut self) -> RollbackPoint {
        RealForwardRunner::checkpoint(self)
    }

    fn rollback(&mut self, point: &RollbackPoint) {
        RealForwardRunner::rollback(self, point)
    }

    fn verify(
        &mut self,
        feed: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.produce_batched(feed, base, logits)
            .map_err(|e| e.to_string())
    }
}

impl crate::producer::ChunkedPrefillRunner for RealForwardRunner {
    /// Gemma 4 only, and the refusal is BY NAME rather than a silent
    /// fallback to the sequential path. A caller that asked for chunked
    /// prefill and quietly got the token-at-a-time loop would measure the
    /// old engine and report it as the new one, which is the failure mode
    /// this whole phase exists to avoid.
    fn prefill_chunk(
        &mut self,
        tokens: &[i32],
        start_position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        if self.real.is_none() {
            return Err(format!(
                "chunked prefill is wired for the real Gemma 4 flow only; this install is {:?}",
                self.arch.family
            ));
        }
        gpu::autorelease_pool(|| self.prefill_chunk_real_gemma4(tokens, start_position, logits))
            .map_err(|e| e.to_string())
    }
}
