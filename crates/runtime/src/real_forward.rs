//! A real (not scripted) [`LogitProducer`]: runs an actual dense
//! transformer forward pass through the real GPU kernels
//! `mrefrust_gpu` wires (`rmsnorm_no_scale`, `rope_proportional_neox`,
//! `dequant_int4_gemv_simd`, `logit_softcap_softmax`, and the two-pass
//! split-KV decode `attention_decode`) against real, resident,
//! INT4-affine-quantized weights loaded through `mrefrust_model_io`.
//! macOS/Metal only, matching `mrefrust_gpu`'s own platform gate.
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
//!   — one commit + one wait per token. The FP16 residual stream and all
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
//! a real-checkpoint tensor mapping) — see `DEVIATIONS.md`.
//!
//! Full-attention (mask 1) and sliding-window (mask 0, via attention
//! `kv_start` over the linear KV layout) layers are supported; linear
//! (Qwen GDN) and compressed (DeepSeek DSV4) layers are not (their
//! kernels are unported). FFN may be dense, routed-resident, or
//! routed-streamed. The synthetic installs in `mrefrust_repack`
//! (`build_synthetic_gemma4_install` and its `_swa`/`_moe`/
//! `_moe_streamed` variants) are the shapes this runner is exercised
//! against, since no trained `.gturbo` checkpoint exists in this
//! environment.

use std::path::Path;

use foundation::LogitValue;
use half::f16;
use model_io::{ArchConfig, ResidentBuffer, ResidentIndex};

use crate::producer::LogitProducer;

#[derive(Debug)]
pub enum RealForwardError {
    Model(model_io::ModelError),
    Gpu(gpu::GpuError),
    MissingTensor(String),
    Unsupported(String),
}

impl std::fmt::Display for RealForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealForwardError::Model(e) => write!(f, "{e}"),
            RealForwardError::Gpu(e) => write!(f, "{e}"),
            RealForwardError::MissingTensor(name) => {
                write!(f, "missing resident tensor: {name}")
            }
            RealForwardError::Unsupported(detail) => write!(f, "unsupported: {detail}"),
        }
    }
}

impl std::error::Error for RealForwardError {}

const RMS_EPS: f32 = 1e-6;

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
    /// token-stride per step, reset via `MADV_DONTNEED`) — the Swift
    /// original's `KVCacheManager` shape, replacing the old host
    /// `Vec<f16>` history that was re-uploaded whole every token.
    pub(crate) kv: gpu::KvCacheManager,
    pub(crate) scratch: DecodeScratch,
    /// One zero-copy `MTLBuffer` per streamer slot, wrapped once at open
    /// over the slot's page-aligned allocation — the GPU MoE kernels read
    /// expert weights straight out of these through the `RoutedBlobs`
    /// argument buffer. Declared BEFORE `streamers` so the buffers drop
    /// before the allocations they alias.
    pub(crate) slot_buffers: Vec<Vec<gpu::MetalBuffer>>,
    pub(crate) streamers: Vec<Option<streaming::PreadExpertStreamer>>,
    /// Uniform blob-relative sub-tensor offsets + the reusable argument
    /// buffer, present when the install packs experts.
    pub(crate) moe_offsets: Option<gpu::MoeExpertOffsets>,
    pub(crate) routed_blobs: Option<gpu::RoutedBlobsBuffer>,
    /// A SECOND argument buffer, for the cache-hit phase-1 dispatch that
    /// rides its own command buffer across the expert `pread` (see
    /// `real_forward_gemma4.rs`). It exists only so that dispatch's
    /// pointer array is not the one the host rebinds for the misses while
    /// the GPU may still be reading it; 64 bytes, allocated once at open.
    pub(crate) routed_blobs_hits: Option<gpu::RoutedBlobsBuffer>,
    /// Real-checkpoint (verbatim `language_model.` tensor naming) decode
    /// state: learned norms, INT8 router effective scales, per-expert
    /// scales, layer scalars. `None` for synthetic short-name installs,
    /// which keep the plain no-scale flow. See `real_forward_gemma4.rs`.
    pub(crate) real: Option<crate::real_forward_gemma4::RealGemmaState>,
    pub(crate) phases: PhaseCounters,
    /// Whether the shared-expert branch rides its own command buffer so it
    /// overlaps the host's expert `pread` (see `real_forward_gemma4.rs`).
    /// `MFERENCE_SHARED_CB=0` reverts to encoding it after the pread, the
    /// A/B seam the Swift original keeps as `MFERENCE_ROUTER_EVENT=0`:
    /// same kernels, same order, identical output, different overlap.
    pub(crate) shared_cb_overlap: bool,
    /// Whether the cache-hit share of the routed experts gets its phase-1
    /// GEMV dispatched on its own command buffer BEFORE the blocking
    /// expert `pread`, so it runs while the host is in the read. The other
    /// A/B seam, `MFERENCE_HIT_CB=0`: same kernels, same slot order,
    /// identical output, different overlap.
    pub(crate) hit_cb_overlap: bool,
    /// Whether a layer's routed-expert command buffer is committed at the
    /// END of that layer and retired one layer later (after the next
    /// layer's router wait), Swift's one-layer-pipelined routed CB.
    /// `MFERENCE_ROUTED_PIPELINE=0` reverts to folding the routed work
    /// uncommitted into the next layer's first command buffer: same
    /// kernels, same commit-relative order of every host buffer write,
    /// identical output, different overlap.
    pub(crate) routed_pipeline: bool,
    /// Set for the duration of one [`LogitProducer::produce_prefill`] call:
    /// the caller is discarding this token's logits, so the output head
    /// (final norm, full-vocab GEMV, softcap, host readback) is skipped.
    /// Everything else, including committing and waiting on the open command
    /// buffer and advancing the KV cache, still runs.
    pub(crate) skip_head: bool,
}

/// The per-dispatch ranking inside each command buffer, or `None` unless
/// `MFERENCE_DISPATCH_PROFILE=1`. One level below [`PhaseCounters`]'s
/// per-buffer GPU busy numbers; `calls` is the forward-pass count those
/// counters cover, so every row reads per token. Re-exported here so
/// callers that already hold a runner do not need their own `gpu`
/// dependency. Read `gpu::dispatch_profile`'s module doc first: profiling
/// serializes the decode it measures.
pub fn dispatch_profile_report(calls: u64) -> Option<String> {
    gpu::dispatch_profile_report(calls)
}

/// Cumulative per-phase decode accounting, the port's answer to the Swift
/// original's `MFERENCE_PHASES=1` breakdown. Every field is summed over
/// every `produce` call this runner has served, prefill included, so a
/// caller reporting decode cost should generate enough tokens for decode
/// to dominate the prompt.
///
/// The buckets are disjoint and all lie on the critical path of one token:
/// `gpu_wait` is time blocked in `wait_until_completed` on a LAYER's
/// attention+router buffer (once per layer, ~30 times per token on real
/// Gemma 4), `final_wait` is the single end-of-token wait split out from
/// it so the two can be told apart, `router` is the
/// logit readback plus host top-k plus slot planning, `hit_cb` is binding
/// and encoding the cache-hit phase-1 command buffer, `expert_io` is the
/// blocking `pread` of missing expert blobs, `bind` is the routing
/// weight upload plus argument-buffer rebind, and `pipeline_wait` is time
/// blocked retiring the previous layer's pipelined routed command buffer
/// (expected ~0 per layer: it was committed before the buffer just
/// waited on, so it has already completed; the post-loop drain of the
/// LAST layer's routed work is the one real payer). What `total` minus
/// those leaves is CPU dispatch encoding plus the final logits readback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseCounters {
    pub calls: u64,
    pub total_nanos: u64,
    pub gpu_wait_nanos: u64,
    /// The end-of-token commit+wait only, disjoint from `gpu_wait_nanos`
    /// (which is the per-layer waits). Split out because the two answer
    /// different questions: this one is the entire ceiling on deferring
    /// the last wait, and under `skip_head` with the routed pipeline on
    /// it is a wait on an EMPTY command buffer (the last layer commits
    /// its routed pass and opens a fresh one it never encodes into), so
    /// what is left is commit plus completion latency, not device time.
    pub final_wait_nanos: u64,
    pub router_nanos: u64,
    pub hit_cb_nanos: u64,
    pub expert_io_nanos: u64,
    pub bind_nanos: u64,
    pub pipeline_wait_nanos: u64,
    /// GPU-side busy time (`GPUEndTime - GPUStartTime`) per command-buffer
    /// class: a SEPARATE axis from the wall-clock buckets above, never part
    /// of their sum (a wall-clock wait on one buffer pays for everything
    /// queued before it; these attribute the GPU's own time). `cb1` is the
    /// per-layer attention+router buffer (in the non-pipelined arm it also
    /// carries the previous layer's routed tail), `routed_cb` is the
    /// pipelined routed buffer (zero when `MFERENCE_ROUTED_PIPELINE=0`),
    /// `final_cb` is the end-of-token norm+head buffer. The shared-expert
    /// and hit-phase-1 buffers are dropped unwaited and stay unattributed.
    pub cb1_gpu_nanos: u64,
    pub routed_cb_gpu_nanos: u64,
    pub final_cb_gpu_nanos: u64,
    /// Expert slots asked for across every layer (`top_k` per layer per
    /// call) and how many were already resident. The miss rate is what
    /// `--expert-cache-slots` buys.
    pub expert_requests: u64,
    pub expert_hits: u64,
}

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

/// Activation scratch, allocated once at open (the decode hot path never
/// allocates a Metal buffer): the FP16 residual stream `x`, the normed /
/// projection / FFN intermediates, the attention partials, and the final
/// logits+probs. The whole token chains through these on the GPU inside
/// one (dense) or a few (MoE) command buffers.
pub(crate) struct DecodeScratch {
    pub(crate) x: gpu::MetalBuffer,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) o: gpu::MetalBuffer,
    pub(crate) o_normed: gpu::MetalBuffer,
    pub(crate) ffn_gate: gpu::MetalBuffer,
    pub(crate) ffn_up: gpu::MetalBuffer,
    pub(crate) ffn_act: gpu::MetalBuffer,
    pub(crate) ffn_out: gpu::MetalBuffer,
    pub(crate) ffn_normed: gpu::MetalBuffer,
    pub(crate) logits: gpu::MetalBuffer,
    /// Router logits (`num_experts` halfs), the per-slot activation rows
    /// (`top_k * moe_inter` halfs), the 8-slot routing-weight vector, and
    /// an all-zero residual (the phase-2 kernel fuses a residual add; the
    /// sandwich-norm path needs the raw combined output, so it feeds
    /// zeros) — MoE-only, allocated tiny for dense architectures.
    pub(crate) router_logits: gpu::MetalBuffer,
    pub(crate) moe_acts: gpu::MetalBuffer,
    pub(crate) routing_w: gpu::MetalBuffer,
    pub(crate) zero_hidden: gpu::MetalBuffer,
    pub(crate) attn: gpu::AttentionScratch,
}

impl DecodeScratch {
    fn new(context: &gpu::MetalContext, arch: &ArchConfig) -> Self {
        let hidden = arch.hidden_size as u64;
        // Mixed-attention architectures (real Gemma 4) project different
        // head dims on SWA vs full layers; size Q/attention scratch for
        // the widest.
        let max_head_dim = arch.head_dim.max(arch.full_head_dim);
        let qk_dim = (arch.num_heads * max_head_dim) as u64;
        let inter = arch.intermediate_size.max(arch.moe_intermediate_size) as u64;
        let vocab = arch.vocab_size as u64;
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * 2);
        Self {
            x: halfs(hidden),
            normed: halfs(hidden),
            q: halfs(qk_dim),
            attn_out: halfs(qk_dim),
            o: halfs(hidden),
            o_normed: halfs(hidden),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            ffn_out: halfs(hidden),
            ffn_normed: halfs(hidden),
            logits: halfs(vocab),
            router_logits: halfs(arch.num_experts.max(1) as u64),
            // Sized for ALL EIGHT kernel slots, not just top_k, and
            // zero-filled once: moe_phase2_down_reduce_k8 unconditionally
            // reads acts[slot * F] for slots 0..7, so padded slots must
            // read finite (zero) activations — a recycled-heap garbage row
            // can be NaN, and 0 * NaN = NaN would poison the whole reduce.
            moe_acts: {
                let n =
                    (gpu::MAX_STREAMED_EXPERTS as u64) * arch.moe_intermediate_size.max(1) as u64;
                let buffer = context.new_output_buffer(n * 2);
                gpu::write_buffer_bytes(&buffer, 0, &vec![0u8; (n * 2) as usize]);
                buffer
            },
            routing_w: {
                let buffer = context.new_output_buffer(gpu::MAX_STREAMED_EXPERTS as u64 * 2);
                gpu::write_buffer_bytes(&buffer, 0, &[0u8; gpu::MAX_STREAMED_EXPERTS * 2]);
                buffer
            },
            zero_hidden: {
                let buffer = context.new_output_buffer(hidden.max(1) * 2);
                gpu::write_buffer_bytes(&buffer, 0, &vec![0u8; hidden.max(1) as usize * 2]);
                buffer
            },
            attn: gpu::AttentionScratch::new(context, arch.num_heads as u32, max_head_dim as u32),
        }
    }
}

/// KV capacity when the caller does not say otherwise; matches the CLI's
/// `--max-context` default. `open_with_max_context` overrides it.
const DEFAULT_MAX_CONTEXT: usize = 4096;

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

    /// Cumulative phase timings across every `produce` call so far. See
    /// [`PhaseCounters`] for what each bucket covers.
    pub fn phase_counters(&self) -> PhaseCounters {
        self.phases
    }

    /// Flips the cache-hit phase-1 command buffer (`MFERENCE_HIT_CB`) after
    /// open, so a test can A/B both states in one process. Setting the
    /// environment variable instead would race the other test threads.
    /// Both states must produce identical output; that is the whole
    /// correctness claim of the overlap.
    #[doc(hidden)]
    pub fn set_hit_cb_overlap(&mut self, on: bool) {
        self.hit_cb_overlap = on;
    }

    /// Sibling of [`Self::set_hit_cb_overlap`] for `MFERENCE_SHARED_CB`.
    #[doc(hidden)]
    pub fn set_shared_cb_overlap(&mut self, on: bool) {
        self.shared_cb_overlap = on;
    }

    /// Sibling of [`Self::set_hit_cb_overlap`] for
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
        // Full attention (1) and sliding-window (0) layers are supported;
        // linear (2, Qwen GDN) and compressed (3/4, DeepSeek DSV4) layers
        // still are not (their compute kernels are unported).
        if expecting
            .full_attention_layer_mask
            .iter()
            .any(|&kind| kind > 1)
        {
            return Err(RealForwardError::Unsupported(
                "linear/compressed attention layers are not supported yet".to_string(),
            ));
        }
        if expecting.full_attention_layer_mask.contains(&0) && expecting.sliding_window <= 0 {
            return Err(RealForwardError::Unsupported(
                "sliding-window layers require a positive sliding_window".to_string(),
            ));
        }

        model_io::load_manifest(dir, &expecting, model_io::DEFAULT_MAX_BYTES)
            .map_err(RealForwardError::Model)?;
        let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
            .map_err(RealForwardError::Model)?;
        let buffer = ResidentBuffer::map(
            &dir.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .map_err(RealForwardError::Model)?;
        let mut context = gpu::MetalContext::new().map_err(RealForwardError::Gpu)?;
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer)
            .map_err(RealForwardError::Gpu)?;

        // Full layers get a linear max_context-capacity buffer; SWA layers
        // get the fp16 ring, `min(max_context, sliding_window + 128)` rows
        // (1152 for real Gemma 4 at 4K), matching the Swift runner's KV
        // sizing. Writes go through `k_slot`/`v_slot` (mod capacity from
        // token 0); reads switch to the ring pipeline only once `seq_len`
        // exceeds the ring (see the dispatch site below).
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

        // Streamed experts, when the install packs them: one streamer per
        // layer over its `packed_experts/layer_NN.bin` file.
        let layout = model_io::load_packed_experts_layout(dir, PACKED_LAYOUT_MAX_BYTES)
            .map_err(RealForwardError::Model)?;
        let num_layers = expecting.num_layers as usize;
        let mut streamers: Vec<Option<streaming::PreadExpertStreamer>> = Vec::new();
        let experts_layout = if layout.num_layers > 0 {
            for layer in 0..num_layers {
                let entry = layout
                    .layers
                    .iter()
                    .find(|l| l.layer == layer)
                    .ok_or_else(|| {
                        RealForwardError::Unsupported(format!(
                            "packed_experts layout missing layer {layer}"
                        ))
                    })?;
                let stream_layout = streaming::StreamLayout::from_packed_experts_layer(
                    entry,
                    dir,
                    layout.expert_stride,
                );
                let streamer = streaming::PreadExpertStreamer::open(
                    stream_layout,
                    expert_cache_slots,
                    streaming::ExpertCachePolicy::DEFAULT,
                )
                .map_err(|e| RealForwardError::Unsupported(format!("expert streamer: {e}")))?;
                streamers.push(Some(streamer));
            }
            Some(layout)
        } else {
            streamers.resize_with(num_layers, || None);
            None
        };

        // Wrap every streamer slot's aligned allocation in a zero-copy
        // Metal buffer once, and resolve the uniform expert sub-tensor
        // offsets the MoE kernels index blobs with.
        let mut slot_buffers: Vec<Vec<gpu::MetalBuffer>> = Vec::with_capacity(streamers.len());
        for streamer in &streamers {
            match streamer {
                Some(s) => {
                    let mut wrapped = Vec::with_capacity(expert_cache_slots);
                    for slot in 0..expert_cache_slots {
                        let (ptr, len) = s.slot_allocation(slot);
                        wrapped.push(
                            gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                                .map_err(RealForwardError::Gpu)?,
                        );
                    }
                    slot_buffers.push(wrapped);
                }
                None => slot_buffers.push(Vec::new()),
            }
        }
        let use_silu = expecting.hidden_activation.contains("silu");
        let (moe_offsets, routed_blobs, routed_blobs_hits) = match &experts_layout {
            Some(layout) => {
                let offsets = moe_offsets_from_layout(layout)?;
                let routed = gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                    .map_err(RealForwardError::Gpu)?;
                let hits = gpu::RoutedBlobsBuffer::new(&mut context, use_silu)
                    .map_err(RealForwardError::Gpu)?;
                (Some(offsets), Some(routed), Some(hits))
            }
            None => (None, None, None),
        };
        drop(experts_layout);

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
            routed_blobs_hits,
            real: None,
            phases: PhaseCounters::default(),
            shared_cb_overlap: std::env::var("MFERENCE_SHARED_CB").as_deref() != Ok("0"),
            hit_cb_overlap: std::env::var("MFERENCE_HIT_CB").as_deref() != Ok("0"),
            routed_pipeline: std::env::var("MFERENCE_ROUTED_PIPELINE").as_deref() != Ok("0"),
            skip_head: false,
        };
        // Real-checkpoint installs keep the source's verbatim tensor
        // naming; their presence selects the learned-weight decode flow.
        if runner
            .index
            .entries
            .contains_key("language_model.model.embed_tokens.weight")
        {
            runner.real = Some(crate::real_forward_gemma4::RealGemmaState::build(
                &mut runner.context,
                &runner.weights,
                &runner.index,
                &runner.arch,
            )?);
        }
        Ok(runner)
    }
}

/// Resolves the shader's uniform `ExpertOffsets` from the decoded blob
/// layout (first expert of the first layer; the writer packs every blob
/// identically). The phase-2 down projection reads its weight bytes with
/// 4-byte loads, so `down`'s offset must be 4-byte aligned.
fn moe_offsets_from_layout(
    layout: &model_io::PackedExpertsLayout,
) -> Result<gpu::MoeExpertOffsets, RealForwardError> {
    let entry = &layout
        .layers
        .first()
        .and_then(|l| l.experts.first())
        .ok_or_else(|| {
            RealForwardError::Unsupported("packed_experts layout has no experts".to_string())
        })?
        .sub_tensors;
    let get = |name: &str| -> Result<u32, RealForwardError> {
        entry
            .get(name)
            .map(|s| s.offset as u32)
            .ok_or_else(|| RealForwardError::MissingTensor(format!("expert blob {name}")))
    };
    let offsets = gpu::MoeExpertOffsets {
        gate_w: get("gate")?,
        gate_s: get("gate_scales")?,
        gate_b: get("gate_biases")?,
        up_w: get("up")?,
        up_s: get("up_scales")?,
        up_b: get("up_biases")?,
        down_w: get("down")?,
        down_s: get("down_scales")?,
        down_b: get("down_biases")?,
    };
    if offsets.down_w % 4 != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "down projection offset {} is not 4-byte aligned",
            offsets.down_w
        )));
    }
    Ok(offsets)
}

fn le_bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn f32_to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

pub(crate) fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

pub(crate) fn f16_slice_to_le_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn tensor_bytes<'a>(
    index: &'a ResidentIndex,
    data: &'a [u8],
    name: &str,
) -> Result<&'a [u8], RealForwardError> {
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let local = (entry.file_offset - index.header.index_size) as usize;
    Ok(&data[local..local + entry.size_bytes as usize])
}

/// Resolves `name` to an offset-bound matrix view into the one shared
/// resident `MTLBuffer` — the zero-copy binding every GPU projection
/// dispatches against. Validates the entry's packed size against the
/// caller's expected shape (the offsets are trusted after that; the index
/// was already bounds-validated at load).
pub(crate) fn resident_matrix<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<gpu::Int4ResidentMatrix<'a>, RealForwardError> {
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    if entry.size_bytes as usize != rows * cols / 2 {
        return Err(RealForwardError::Unsupported(format!(
            "tensor {name}: packed size {} does not match shape {rows}x{cols}",
            entry.size_bytes
        )));
    }
    let base = index.header.index_size;
    Ok(gpu::Int4ResidentMatrix {
        buffer: weights.buffer(),
        weights_offset: weights.gpu_offset(entry.file_offset - base),
        scales_offset: weights.gpu_offset(entry.scale_offset - base),
        biases_offset: weights.gpu_offset(entry.bias_offset - base),
        rows,
        cols,
    })
}

/// Host copies for the CPU FFN bridge only (`compute::run_ffn` has no GPU
/// counterpart yet — see module docs). Converts the tensor's BF16
/// scale/bias bytes on each call; goes away with the GPU FFN/MoE kernels.
fn owned_rows(
    index: &ResidentIndex,
    data: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<compute::quant::Int4AffineRow>, RealForwardError> {
    let packed_all = tensor_bytes(index, data, name)?;
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let base = index.header.index_size;
    let scale_local = (entry.scale_offset - base) as usize;
    let bias_local = (entry.bias_offset - base) as usize;
    let scales = le_bytes_to_u16(&data[scale_local..scale_local + entry.scale_size as usize]);
    let biases = le_bytes_to_u16(&data[bias_local..bias_local + entry.bias_size as usize]);
    let row_bytes = cols / 2;
    let groups = cols / 64;
    Ok((0..rows)
        .map(|r| compute::quant::Int4AffineRow {
            packed: packed_all[r * row_bytes..(r + 1) * row_bytes].to_vec(),
            scales: scales[r * groups..(r + 1) * groups].to_vec(),
            biases: biases[r * groups..(r + 1) * groups].to_vec(),
        })
        .collect())
}

fn layer_name(prefix: &str, layer: usize) -> String {
    format!("layer{layer}.{prefix}")
}

/// Softmax over all `logits`, then the top-`k` entries with their
/// probabilities renormalized to sum to `1` over just the survivors
/// (Gemma 4's `router_scaled` convention). Plain host arithmetic — cheap
/// enough at any real `num_experts` count that no kernel is warranted.
fn topk_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let selected: Vec<usize> = order.into_iter().take(k).collect();

    let selected_sum: f32 = selected.iter().map(|&i| probs[i]).sum();
    let weights: Vec<f32> = selected
        .iter()
        .map(|&i| {
            if selected_sum > 0.0 {
                probs[i] / selected_sum
            } else {
                0.0
            }
        })
        .collect();
    (selected, weights)
}

/// The routed-expert FFN: a real GPU router GEMV, host-side top-k
/// selection, then each selected expert's gate/up/down GEMVs and gated
/// activation via `compute::run_ffn` (see module docs for why this stays
/// CPU-bridged), weighted and summed. Only the selected experts' weights
/// are ever read, not the full expert table.
#[allow(clippy::too_many_arguments)]
impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        self.kv.reset();
    }

    /// One pool per token, not per process. Every command buffer and
    /// compute encoder this token creates is autoreleased; without a pool
    /// scoped here they would all stay alive until the process exits (see
    /// `gpu::autorelease_pool`).
    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        gpu::autorelease_pool(|| self.produce_inner(token, position, logits))
            .map_err(|e| e.to_string())
    }

    /// Same forward pass as [`Self::produce`] minus the output head, whose
    /// logits the caller has told us it will discard. On a long prompt this
    /// is the whole prefill but its last token, and the head is one
    /// full-vocab GEMV plus a vocab-sized host readback per token.
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

impl RealForwardRunner {
    /// The routed-expert FFN host bridge: a real GPU router GEMV, host
    /// top-k, then each selected expert's gate/up/down + gated activation
    /// via `compute::run_ffn`, weighted and summed. Expert weights come
    /// from the per-layer `PreadExpertStreamer` (LFU slot cache, parallel
    /// pread on misses) when the install packs them, or from the resident
    /// region by name otherwise. Phase B moves the expert math onto the
    /// GPU reading the slots directly.
    fn moe_ffn_host(&mut self, layer: usize, x: &[f32]) -> Result<Vec<f32>, RealForwardError> {
        let hidden = self.arch.hidden_size as usize;
        let moe_inter = self.arch.moe_intermediate_size as usize;
        let num_experts = self.arch.num_experts as usize;
        let top_k = self.arch.top_k_experts as usize;

        let x16 = f32_to_f16(x);
        let router = resident_matrix(
            &self.weights,
            &self.index,
            &layer_name("router", layer),
            num_experts,
            hidden,
        )?;
        let logits16 = gpu::dequant_int4_gemv_resident(&mut self.context, &router, &x16)
            .map_err(RealForwardError::Gpu)?;
        let (selected, weights) = topk_softmax(&f16_to_f32(&logits16), top_k);

        let mut combined = vec![0f32; hidden];
        let data = self.weights.data();
        for (&e, &w) in selected.iter().zip(weights.iter()) {
            let gate_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.gate_proj"),
                moe_inter,
                hidden,
            )?;
            let up_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.up_proj"),
                moe_inter,
                hidden,
            )?;
            let down_rows = owned_rows(
                &self.index,
                data,
                &format!("layer{layer}.expert{e}.down_proj"),
                hidden,
                moe_inter,
            )?;
            let out = compute::run_ffn(&gate_rows, &up_rows, &down_rows, x, hidden, moe_inter);
            for (c, o) in combined.iter_mut().zip(out.iter()) {
                *c += w * o;
            }
        }
        Ok(combined)
    }

    fn produce_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        if self.real.is_some() {
            return self.produce_real_gemma4(token, position, logits);
        }
        let hidden = self.arch.hidden_size as usize;
        let inter = self.arch.intermediate_size as usize;
        let num_heads = self.arch.num_heads as u32;
        let num_kv_heads = self.arch.num_full_kv_heads as u32;
        let head_dim = self.arch.full_head_dim as u32;
        let qk_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let vocab = self.arch.vocab_size as usize;
        let rotated_pairs =
            ((head_dim as f64 * self.arch.partial_rotary_factor) / 2.0).round() as u32;
        let theta = self.arch.full_rope_theta as f32;
        let attn_scale = self.arch.attention_scale as f32;
        let softcap = self.arch.final_logit_softcap as f32;
        let embed_scale = if self.arch.embedding_scaled_by_sqrt_hidden {
            (hidden as f32).sqrt()
        } else {
            1.0
        };
        let use_silu = self.arch.hidden_activation.contains("silu");
        let sandwich = self.arch.ffn_sandwich_norms;

        // The KV cache is positional: tokens must arrive in order from the
        // position the cache is at (reset() rewinds to 0).
        if position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if !self.arch.attention_k_eq_v {
            return Err(RealForwardError::Unsupported(
                "only attention_k_eq_v architectures are supported".to_string(),
            ));
        }
        let seq_len = (position + 1) as u32;

        let gpu_err = RealForwardError::Gpu;
        // The embedding row dequant runs on the GPU (embed_lookup_int4,
        // table/scales/biases bound as offsets into the resident buffer),
        // seeding the FP16 residual stream in place: the whole token is
        // GPU-side from the first byte.
        let embed = self
            .index
            .entries
            .get("embed_lm_head")
            .ok_or_else(|| RealForwardError::MissingTensor("embed_lm_head".to_string()))?;
        if (token as usize) >= self.arch.vocab_size as usize {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {}",
                self.arch.vocab_size
            )));
        }
        let base = self.index.header.index_size;
        let embed_table = self.weights.gpu_offset(embed.file_offset - base);
        let embed_scales = self.weights.gpu_offset(embed.scale_offset - base);
        let embed_biases = self.weights.gpu_offset(embed.bias_offset - base);

        let mut pass = self.context.begin_pass();
        gpu::encode_embed_lookup_int4(
            &mut self.context,
            &pass,
            (self.weights.buffer(), embed_table),
            (self.weights.buffer(), embed_scales),
            (self.weights.buffer(), embed_biases),
            (&self.scratch.x, 0),
            token as u32,
            hidden as u32,
            embed_scale,
        )
        .map_err(gpu_err)?;
        for layer in 0..self.arch.num_layers as usize {
            let q_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("q_proj", layer),
                qk_dim,
                hidden,
            )?;
            let k_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("k_proj", layer),
                kv_dim,
                hidden,
            )?;
            let o_proj = resident_matrix(
                &self.weights,
                &self.index,
                &layer_name("o_proj", layer),
                hidden,
                qk_dim,
            )?;
            let (k_buf, k_off) = self.kv.k_slot(layer, position);

            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                &pass,
                &q_proj,
                (&self.scratch.normed, 0),
                (&self.scratch.q, 0),
            )
            .map_err(gpu_err)?;
            // The K projection lands DIRECTLY in its persistent cache
            // slot, and RoPE rotates it there in place — no staging row.
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                &pass,
                &k_proj,
                (&self.scratch.normed, 0),
                (k_buf, k_off as u64),
            )
            .map_err(gpu_err)?;
            gpu::encode_rope_proportional_neox(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                position as u32,
                num_heads,
                head_dim,
                rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;
            gpu::encode_rope_proportional_neox(
                &mut self.context,
                &pass,
                (k_buf, k_off as u64),
                position as u32,
                num_kv_heads,
                head_dim,
                rotated_pairs,
                theta,
            )
            .map_err(gpu_err)?;

            // attention_k_eq_v: the K buffer is bound as both K and V, so
            // the V buffers are never written at all (their untouched
            // pages never become resident). A separate-V architecture
            // would write and bind `v_slot` here instead.
            //
            // Sliding-window layers (mask 0) attend only the trailing
            // `sliding_window` positions via kv_start, and switch to the
            // ring pipeline once seq_len outgrows the ring (the Swift
            // activation rule; below capacity the slot mapping is the
            // identity, so the linear pipeline is byte-identical).
            let (kv_start, active_ring) = if self.arch.full_attention_layer_mask[layer] == 0 {
                let ring = self.kv.ring_capacity(layer) as u32;
                (
                    seq_len.saturating_sub(self.arch.sliding_window as u32),
                    if ring > 0 && seq_len > ring { ring } else { 0 },
                )
            } else {
                (0, 0)
            };
            gpu::encode_attention_decode(
                &mut self.context,
                &pass,
                (&self.scratch.q, 0),
                k_buf,
                k_buf,
                &self.scratch.attn,
                (&self.scratch.attn_out, 0),
                head_dim,
                num_heads,
                num_kv_heads,
                seq_len,
                kv_start,
                active_ring,
                attn_scale,
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                &pass,
                &o_proj,
                (&self.scratch.attn_out, 0),
                (&self.scratch.o, 0),
            )
            .map_err(gpu_err)?;
            let attn_delta = if sandwich {
                gpu::encode_rms_norm_no_scale(
                    &mut self.context,
                    &pass,
                    (&self.scratch.o, 0),
                    (&self.scratch.o_normed, 0),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                &self.scratch.o_normed
            } else {
                &self.scratch.o
            };
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (attn_delta, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;

            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;

            if self.arch.num_experts > 0 && self.streamers[layer].is_some() {
                // Streamed MoE, the Swift CB1/CB2 shape: the router GEMV
                // rides the current pass; its logits are the one host
                // readback per layer (top-k + expert pread need them);
                // then a second pass runs the real GPU MoE kernels
                // reading the expert blobs in place from the slot
                // buffers. No expert byte ever reaches the host.
                let num_experts = self.arch.num_experts as usize;
                let top_k = self.arch.top_k_experts as usize;
                let moe_inter = self.arch.moe_intermediate_size as u32;
                let router = resident_matrix(
                    &self.weights,
                    &self.index,
                    &layer_name("router", layer),
                    num_experts,
                    hidden,
                )?;
                gpu::encode_dequant_int4_gemv_resident(
                    &mut self.context,
                    &pass,
                    &router,
                    (&self.scratch.normed, 0),
                    (&self.scratch.router_logits, 0),
                )
                .map_err(gpu_err)?;
                pass.commit_and_wait();

                let router_logits = f16_to_f32(&gpu::read_buffer_f16(
                    &self.scratch.router_logits,
                    0,
                    num_experts,
                ));
                let (selected, route_weights) = topk_softmax(&router_logits, top_k);
                let streamer = self.streamers[layer].as_mut().expect("checked above");
                let plan =
                    streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
                let slots = streamer
                    .execute_expert_cache_plan(&plan)
                    .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;

                let mut routing16 = vec![f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
                for (i, &w) in route_weights.iter().enumerate() {
                    routing16[i] = f16::from_f32(w);
                }
                gpu::write_buffer_bytes(
                    &self.scratch.routing_w,
                    0,
                    &f16_slice_to_le_bytes(&routing16),
                );

                let layer_slots = &self.slot_buffers[layer];
                let blob_refs: Vec<(&gpu::MetalBuffer, u64)> =
                    slots.iter().map(|&s| (&layer_slots[s], 0u64)).collect();
                let routed = self.routed_blobs.as_ref().expect("layout implies blobs");
                let offsets = self.moe_offsets.as_ref().expect("layout implies offsets");
                routed
                    .bind(&mut self.context, use_silu, &blob_refs)
                    .map_err(gpu_err)?;

                pass = self.context.begin_pass();
                for &(buffer, _) in &blob_refs {
                    pass.use_read_buffer(buffer);
                }
                gpu::encode_moe_phase1(
                    &mut self.context,
                    &pass,
                    routed,
                    offsets,
                    (&self.scratch.normed, 0),
                    (&self.scratch.moe_acts, 0),
                    hidden as u32,
                    moe_inter,
                    top_k as u32,
                    use_silu,
                )
                .map_err(gpu_err)?;
                gpu::encode_moe_phase2(
                    &mut self.context,
                    &pass,
                    routed,
                    offsets,
                    (&self.scratch.moe_acts, 0),
                    (&self.scratch.routing_w, 0),
                    (&self.scratch.zero_hidden, 0),
                    (&self.scratch.ffn_out, 0),
                    hidden as u32,
                    moe_inter,
                    use_silu,
                )
                .map_err(gpu_err)?;
            } else if self.arch.num_experts > 0 {
                // Resident-expert MoE (synthetic-only): the host bridge.
                pass.commit_and_wait();
                let pre_ffn32 = f16_to_f32(&gpu::read_buffer_f16(&self.scratch.normed, 0, hidden));
                let combined = self.moe_ffn_host(layer, &pre_ffn32)?;
                gpu::write_buffer_bytes(
                    &self.scratch.ffn_out,
                    0,
                    &f16_slice_to_le_bytes(&f32_to_f16(&combined)),
                );
                pass = self.context.begin_pass();
            } else {
                let gate_proj = resident_matrix(
                    &self.weights,
                    &self.index,
                    &layer_name("gate_proj", layer),
                    inter,
                    hidden,
                )?;
                let up_proj = resident_matrix(
                    &self.weights,
                    &self.index,
                    &layer_name("up_proj", layer),
                    inter,
                    hidden,
                )?;
                let down_proj = resident_matrix(
                    &self.weights,
                    &self.index,
                    &layer_name("down_proj", layer),
                    hidden,
                    inter,
                )?;
                gpu::encode_dequant_int4_gemv_resident(
                    &mut self.context,
                    &pass,
                    &gate_proj,
                    (&self.scratch.normed, 0),
                    (&self.scratch.ffn_gate, 0),
                )
                .map_err(gpu_err)?;
                gpu::encode_dequant_int4_gemv_resident(
                    &mut self.context,
                    &pass,
                    &up_proj,
                    (&self.scratch.normed, 0),
                    (&self.scratch.ffn_up, 0),
                )
                .map_err(gpu_err)?;
                let act = if use_silu {
                    gpu::encode_silu_mul
                } else {
                    gpu::encode_gelu_mul
                };
                act(
                    &mut self.context,
                    &pass,
                    (&self.scratch.ffn_gate, 0),
                    (&self.scratch.ffn_up, 0),
                    (&self.scratch.ffn_act, 0),
                    inter as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_dequant_int4_gemv_resident(
                    &mut self.context,
                    &pass,
                    &down_proj,
                    (&self.scratch.ffn_act, 0),
                    (&self.scratch.ffn_out, 0),
                )
                .map_err(gpu_err)?;
            }

            let ffn_delta = if sandwich {
                gpu::encode_rms_norm_no_scale(
                    &mut self.context,
                    &pass,
                    (&self.scratch.ffn_out, 0),
                    (&self.scratch.ffn_normed, 0),
                    hidden as u32,
                    RMS_EPS,
                )
                .map_err(gpu_err)?;
                &self.scratch.ffn_normed
            } else {
                &self.scratch.ffn_out
            };
            gpu::encode_residual_add(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (ffn_delta, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }

        // The head is pure waste on a prefill token whose logits the caller
        // discards; the commit and the KV advance below are not, so they stay
        // outside this guard.
        if !self.skip_head {
            let lm_head =
                resident_matrix(&self.weights, &self.index, "embed_lm_head", vocab, hidden)?;
            gpu::encode_rms_norm_no_scale(
                &mut self.context,
                &pass,
                (&self.scratch.x, 0),
                (&self.scratch.normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_dequant_int4_gemv_resident(
                &mut self.context,
                &pass,
                &lm_head,
                (&self.scratch.normed, 0),
                (&self.scratch.logits, 0),
            )
            .map_err(gpu_err)?;
            // Stop at the softcapped logits, not at probabilities: the softmax
            // is `selection::select`'s job (see real_forward_gemma4.rs's header).
            if softcap > 0.0 {
                gpu::encode_logit_softcap(
                    &mut self.context,
                    &pass,
                    (&self.scratch.logits, 0),
                    softcap,
                    vocab as u32,
                )
                .map_err(gpu_err)?;
            }
        }
        pass.commit_and_wait();
        self.kv.advance();

        if self.skip_head {
            return Ok(());
        }
        let head = gpu::read_buffer_f16(&self.scratch.logits, 0, vocab);
        if head.len() != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                head.len(),
                logits.len()
            )));
        }
        logits.copy_from_slice(&head);
        Ok(())
    }
}
