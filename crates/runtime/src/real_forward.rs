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
    moe_offsets_from_layout, routed_layouts_from_layout, RoutedLayerLayout, EXECUTABLE_GGUF_DTYPES,
    GGUF_BLOCK_DTYPES,
};
use crate::real_forward_types::DecodeScratch;
pub use crate::real_forward_types::{dispatch_profile_report, PhaseCounters, RealForwardError};

/// Matches `RuntimeConfig`'s default `expert_cache_slots`; callers that
/// want another allowed value pass it to `open_with_options`.
const EXPERT_CACHE_SLOTS: usize = 16;
/// Mirrors the Swift RuntimeConfiguration default `prefillChunkTokens`
/// (128). Ring capacity per SWA layer is `min(max_context, sliding_window
/// plus this)`, i.e. 1152 for real Gemma 4 at the 4096 default -- the
/// same KV sizing as the Swift runner even though this port's prefill is
/// still token-at-a-time (see `raw_completion.rs`); the chunk headroom is
/// reserved for the future chunked-prefill port.
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
    /// Blob-relative sub-tensor offsets, ONE ENTRY PER LAYER, and the
    /// reusable argument buffer. Empty when the install does not pack
    /// experts.
    pub(crate) moe_offsets: Vec<gpu::MoeExpertOffsets>,
    pub(crate) routed_blobs: Option<gpu::RoutedBlobsBuffer>,
    /// Real-checkpoint (verbatim `language_model.` tensor naming) decode
    /// state: learned norms, INT8 router effective scales, per-expert
    /// scales, layer scalars. `None` for synthetic short-name installs,
    /// which keep the plain no-scale flow. See `real_forward_gemma4.rs`.
    pub(crate) real: Option<crate::families::gemma4::RealGemmaState>,
    /// Real-checkpoint Qwen 3.6 decode state: the GDN recurrent buffers and
    /// the Qwen-only scratch. Mutually exclusive with `real`; kept as its
    /// own `Option` rather than folded into an enum so the Gemma path's
    /// borrow shape is untouched.
    pub(crate) real_qwen: Option<crate::families::qwen::RealQwenState>,
    /// Present for a `llama`-architecture install (ROADMAP Phase M2), which
    /// is Mixtral-style MoE only; a dense one is refused at build.
    pub(crate) real_llama: Option<crate::families::llama::RealLlamaState>,
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
        Self::open_inner(dir, expecting, max_context, expert_cache_slots, None)
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
            expert_cache_slots,
            fp16_ring_capacity_override,
        )
    }

    fn open_inner(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: usize,
        fp16_ring_capacity_override: Option<usize>,
    ) -> Result<Self, RealForwardError> {
        if expert_cache_slots == 0 {
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
            return Err(RealForwardError::Unsupported(format!(
                "tensor {} carries GGUF block dtype {}, which has no kernel in this port \
                 (ROADMAP Phase G Stage 2; executable so far: Q8_0, Q4_K, Q6_K)",
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

        let (streamers, slot_buffers, experts_layout) =
            crate::real_forward_init::open_expert_streamers(
                dir,
                &expecting,
                expert_cache_slots,
                PACKED_LAYOUT_MAX_BYTES,
                &mut context,
            )?;

        let use_silu = expecting.hidden_activation.contains("silu");
        let (moe_offsets, routed_layouts, routed_blobs) = match &experts_layout {
            Some(layout) => {
                let offsets = moe_offsets_from_layout(layout)?;
                let layouts = routed_layouts_from_layout(layout)?;
                let routed = gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                    .map_err(RealForwardError::Gpu)?;
                (offsets, layouts, Some(routed))
            }
            None => (Vec::new(), Vec::new(), None),
        };
        drop(experts_layout);

        let router_hist = crate::router_hist::RouterHistogram::from_env(
            expecting.num_layers as usize,
            expecting.num_experts.max(0) as usize,
        );
        let mut runner = Self {
            context,
            weights,
            index,
            arch: expecting,
            kv,
            scratch,
            slot_buffers,
            streamers,
            moe_offsets,
            routed_blobs,
            real: None,
            real_qwen: None,
            real_llama: None,
            phases: PhaseCounters::default(),
            shared_cb_overlap: std::env::var("MFERENCE_SHARED_CB").as_deref() != Ok("0"),
            routed_pipeline: std::env::var("MFERENCE_ROUTED_PIPELINE").as_deref() != Ok("0"),
            routed_layouts,
            router_hist,
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
            model_io::ModelFamily::Qwen36 => {
                runner.real_qwen = Some(crate::families::qwen::RealQwenState::build(
                    &mut runner.context,
                    &runner.weights,
                    &runner.index,
                    &runner.arch,
                )?);
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
        }
        Ok(runner)
    }
}

impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        self.kv.reset();
        if let Some(qwen) = self.real_qwen.as_mut() {
            qwen.reset();
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
