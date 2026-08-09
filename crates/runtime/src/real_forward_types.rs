//! Core runtime types, error definitions, phase accounting counters,
//! and GPU activation scratch allocation for real forward runner execution.

use model_io::ArchConfig;

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
/// logit readback plus host top-k plus slot planning, `expert_io` is the
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
    /// zeros) -- MoE-only, allocated tiny for dense architectures.
    pub(crate) router_logits: gpu::MetalBuffer,
    pub(crate) moe_acts: gpu::MetalBuffer,
    pub(crate) routing_w: gpu::MetalBuffer,
    pub(crate) zero_hidden: gpu::MetalBuffer,
    pub(crate) attn: gpu::AttentionScratch,
}

impl DecodeScratch {
    pub(crate) fn new(context: &gpu::MetalContext, arch: &ArchConfig) -> Self {
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
            // read finite (zero) activations -- a recycled-heap garbage row
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
