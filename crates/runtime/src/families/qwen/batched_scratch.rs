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
}

impl BatchedScratch {
    pub(crate) fn new(
        context: &gpu::MetalContext,
        arch: &ArchConfig,
        qwen_shape: gpu::GdnShape,
        batch: usize,
    ) -> Self {
        let hidden = arch.hidden_size as u64;
        let q_dim = (arch.num_heads * arch.full_head_dim) as u64;
        let inter = arch.intermediate_size as u64;
        let vocab = arch.vocab_size as u64;
        let qkv_dim = qwen_shape.qkv_dim() as u64;
        let value_dim = qwen_shape.value_dim() as u64;
        let v_heads = qwen_shape.num_v_heads as u64;
        let b = batch as u64;
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * b * 2);
        Self {
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
        }
    }
}
