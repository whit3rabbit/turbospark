//! Metal buffer allocation for batched Qwen forward verification passes.

use model_io::ArchConfig;

/// The M-row siblings of the buffers `DecodeScratch` and `RealQwenState`
/// hold one row of.
///
/// **Allocated only when a draft depth was asked for**, beside [`super::mtp::MtpState`]
/// and for its reason: `MFERENCE_MTP_DRAFT` unset must allocate nothing, so
/// that `qwen38_memory_oracle`'s frozen row keeps describing the engine that
/// shipped before this module existed. At the block sizes that pay this is
/// ~10 MiB, almost all of it `logits` (`batch * vocab` halfs, and this
/// family's vocab is 248,320).
///
/// `x` is NOT here: `DecodeScratch::x` already holds `MAX_PREFILL_BATCH`
/// rows, because the residual stream crosses layers and the chunk driver
/// needed the same thing first.
pub(crate) struct BatchedScratch {
    /// Rows this was sized for. A verify of block M needs `M + 1`.
    pub(crate) batch: usize,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) q_packed: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) o: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,
    pub(crate) moe_x: gpu::MetalBuffer,
    pub(crate) ffn_gate: gpu::MetalBuffer,
    pub(crate) ffn_up: gpu::MetalBuffer,
    pub(crate) ffn_act: gpu::MetalBuffer,
    pub(crate) h2: gpu::MetalBuffer,
    pub(crate) logits: gpu::MetalBuffer,
    /// The routed half's M-row buffers, `None` on a DENSE install.
    ///
    /// Gated on `num_experts != 0` for the same reason the whole struct is
    /// gated on a draft depth: a dense install must allocate exactly what it
    /// allocated before the routed batched path existed, so that
    /// `qwen38_memory_oracle`'s frozen row keeps describing it. A dense
    /// checkpoint also has no `moe_intermediate_size` (it encodes 0 there,
    /// per `crates/runtime` Gotcha 12's note on the two widths), so the
    /// sizes below are not merely wasted, they are meaningless.
    pub(crate) routed: Option<BatchedRoutedScratch>,
}

/// The M-row siblings of the buffers the per-token routed pass
/// (`families/qwen/moe.rs`) holds one row of.
///
/// `ffn_gate` / `ffn_up` / `ffn_act` are NOT here: the shared expert's
/// width is `intermediate_size`, which is what the dense FFN buffers above
/// are already sized for, and the two never run in the same layer.
pub(crate) struct BatchedRoutedScratch {
    /// `[M * top_k, moe_inter]` FP16, phase 1's output over the route list.
    pub(crate) batch_acts: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, the fused phase 2's output.
    pub(crate) batch_y: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, the GATED shared-expert output per token, and
    /// phase 2's accumulator SEED. It cannot be the single-row `h1` the
    /// per-token path reuses: all M tokens' shared branches are encoded
    /// before any of them is consumed.
    pub(crate) batch_h1: gpu::MetalBuffer,
    /// `[M]` FP16, one `shared_expert_gate` logit per token.
    pub(crate) batch_gate_logit: gpu::MetalBuffer,
    /// `[M * top_k]` FP16 in PAIR order (token-major, rank within), which
    /// is the layout the fused phase 2 looks its routes up by -- not the
    /// decode path's slot-indexed `MAX_STREAMED_EXPERTS` row.
    pub(crate) batch_routing_w: gpu::MetalBuffer,
    /// The encoded route list, 16 bytes per `MoePrefillRoute`.
    pub(crate) batch_routes: gpu::MetalBuffer,
    /// `[M, num_experts]` FP32 router logits, read back for the WHOLE
    /// batch in one host wait. `RealQwenState::router_logits_f32` holds a
    /// single row and is not widened: that buffer is allocated by every
    /// MoE install whether or not a drafter is open, and this one is not.
    pub(crate) batch_router_logits_f32: gpu::MetalBuffer,
    /// The wide expert-blob argument buffer, bound once per layer with
    /// every cache slot.
    pub(crate) wide_blobs: gpu::RoutedBlobsWideBuffer,
}

impl BatchedScratch {
    pub(crate) fn new(
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
        qwen_shape: gpu::GdnShape,
        batch: usize,
    ) -> Result<Self, gpu::GpuError> {
        let hidden = arch.hidden_size as u64;
        let q_dim = (arch.num_heads * arch.full_head_dim) as u64;
        let inter = arch.intermediate_size as u64;
        let vocab = arch.vocab_size as u64;
        let qkv_dim = qwen_shape.qkv_dim() as u64;
        let value_dim = qwen_shape.value_dim() as u64;
        let v_heads = qwen_shape.num_v_heads as u64;
        let b = batch as u64;
        // The argument buffer needs `context` MUTABLY where every plain
        // allocation below needs it immutably through `halfs`, so it is
        // built first and its borrow ends here.
        let wide_blobs = if arch.num_experts == 0 {
            None
        } else {
            Some(gpu::RoutedBlobsWideBuffer::new(
                context,
                arch.hidden_activation.contains("silu"),
            )?)
        };
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * b * 2);
        let routed = wide_blobs.map(|wide_blobs| {
            let top_k = arch.top_k_experts as u64;
            let moe_inter = arch.moe_intermediate_size.max(1) as u64;
            BatchedRoutedScratch {
                batch_acts: halfs(top_k * moe_inter),
                batch_y: halfs(hidden),
                batch_h1: halfs(hidden),
                batch_gate_logit: halfs(1),
                batch_routing_w: halfs(top_k),
                // 16 bytes per encoded route (token, rank, slot, reserved),
                // matching `MoePrefillRoute::bytes` and the shader struct.
                batch_routes: context.new_output_buffer(b * top_k * 16),
                batch_router_logits_f32: context
                    .new_output_buffer(b * arch.num_experts.max(1) as u64 * 4),
                wide_blobs,
            }
        });
        Ok(Self {
            batch,
            normed: halfs(hidden),
            q_packed: halfs(2 * q_dim),
            q: halfs(q_dim),
            attn_gate: halfs(q_dim),
            attn_out: halfs(q_dim),
            o: halfs(hidden),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),
            moe_x: halfs(hidden),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            h2: halfs(hidden),
            logits: halfs(vocab),
            routed,
        })
    }
}
