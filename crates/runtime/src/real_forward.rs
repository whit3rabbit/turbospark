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

use model_io::{ArchConfig, ResidentIndex};

use crate::real_forward_layout::RoutedLayerLayout;
pub use crate::real_forward_rollback::RollbackPoint;
use crate::real_forward_types::DecodeScratch;
pub use crate::real_forward_types::{dispatch_profile_report, PhaseCounters, RealForwardError};

/// Matches `RuntimeConfig`'s default `expert_cache_slots`; callers that
/// want another allowed value pass it to `open_with_options`.
pub(crate) const EXPERT_CACHE_SLOTS: usize = 16;
/// Mirrors the Swift RuntimeConfiguration default `prefillChunkTokens`
/// (128). Ring capacity per SWA layer is `min(max_context, sliding_window
/// plus this)`, i.e. 1152 for real Gemma 4 at the 4096 default. The chunk
/// headroom is no longer notional: `ChunkedPrefillRunner` writes M KV rows
/// before reading any of them back, and this is the budget that guarantees
/// a sliding-window ring still holds every row the chunk will attend over.
pub(crate) const MAX_PREFILL_CHUNK_TOKENS: usize = 128;
pub(crate) const PACKED_LAYOUT_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// KV capacity when the caller does not say otherwise; matches the CLI's
/// `--max-context` default. `open_with_max_context` overrides it.
pub(crate) const DEFAULT_MAX_CONTEXT: usize = 4096;

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
    pub(crate) real_dflash: Option<crate::families::qwen::DflashState>,
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
    /// Whether the chunked-prefill driver runs each layer's routed half
    /// as ONE batched dispatch pair over a route list
    /// (`docs/BATCHED_PREFILL.md` steps 2 and 3) instead of per token.
    /// `MFERENCE_ROUTED_BATCH=1` turns it on; UNSET keeps the per-token
    /// path, so the seam A/Bs the two halves of the chunk driver the way
    /// `MFERENCE_SHARED_CB` A/Bs decode overlap. Output is byte-identical
    /// either way (the batched kernels are bit-exact against M decode
    /// passes), so this is a throughput axis only. INT4-affine blobs
    /// only: a GGUF install is refused by layout when the seam is on,
    /// never silently looped.
    pub(crate) routed_batch_prefill: bool,
    /// Whether the chunked-prefill driver runs its RESIDENT GEMVs -- the
    /// four attention projections and the shared expert's three -- as
    /// M-row GEMMs instead of one dispatch per token
    /// (`docs/BATCHED_PREFILL.md`, the 29.7% row of the prefill dispatch
    /// ranking). `MFERENCE_BATCHED_GEMV=1` turns it on; UNSET keeps the
    /// per-token path, exactly as the routed seam above does for the other
    /// half of the layer.
    ///
    /// Output is byte-identical either way, and that is a MEASURED claim
    /// rather than a structural one: `dequant_int4_gemm_simd` and
    /// `dequant_int4_gemv_simd` agree bit-for-bit on a fixture built to
    /// see reassociation, against a positive control
    /// (`dequant_int4_gemm_mma`) that differs on ~39% of outputs on the
    /// same data (`crates/gpu/tests/dequant_int4_gemm_parity.rs`). So this
    /// is a throughput axis only. INT4-affine only: any other dtype is
    /// refused BY NAME when the seam is on, never silently looped.
    ///
    /// **WHICH DISPATCHES IT MOVES, EXACTLY, because "batched GEMVs" is
    /// wider than what it does.** The four attention projections always.
    /// The shared expert's three ONLY when [`Self::routed_batch_prefill`]
    /// is also on: the per-token routed pass reads a single-row `h1` at
    /// offset 0, and that read is on the DECODE path's signature
    /// (`families/gemma4/moe.rs`), so pointing it at an M-row buffer is a
    /// decode change rather than a prefill one. The batched routed half
    /// already writes `batch_h1` per token and needs no such change.
    pub(crate) batched_gemv_prefill: bool,
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
    /// Per-layer residual stream at the last prompt token, `None` unless
    /// `MFERENCE_RESID_CAPTURE=/path.json` on a family whose flow feeds the
    /// capture (the qwen flow today). Written on drop; see
    /// `resid_capture.rs`. This is what a steering direction is extracted
    /// FROM (ROADMAP item 9's prerequisite).
    pub(crate) resid_capture: Option<crate::resid_capture::ResidCapture>,
    /// The directional-steering state, `None` unless a caller passed a
    /// policy carrying a direction set. Built at open; see `steering.rs`.
    pub(crate) steering: Option<crate::steering::SteeringState>,
    /// The install's own directory, kept so the vision tower can open its
    /// `packed_vision/` streamer lazily. Nothing else needs it: every other
    /// file is read at `open` and mapped or parsed there.
    pub(crate) install_dir: std::path::PathBuf,
    /// The vision tower (ROADMAP M-V4), built on the FIRST image rather than
    /// at open.
    ///
    /// Lazy because `arch.vision` is read by nothing else in this crate, so
    /// an eager open would charge every text-only session on a vision install
    /// the tower's pinned slots and layout parse for a component it never
    /// touches. **`None` therefore means "not yet asked for", never "this
    /// install has none"** -- that question is `arch.vision.is_active()`.
    pub(crate) vision: Option<crate::vision::VisionTower>,
    /// Set for the duration of one [`LogitProducer::produce_prefill`] call:
    /// the caller is discarding this token's logits, so the output head
    /// (final norm, full-vocab GEMV, softcap, host readback) is skipped.
    /// Everything else, including committing and waiting on the open command
    /// buffer and advancing the KV cache, still runs.
    pub(crate) skip_head: bool,
}
