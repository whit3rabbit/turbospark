//! `RealQwenState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a Qwen 3.6 install -- the GDN recurrent
//! buffers, the unit router scales that stand in for Qwen's absent
//! `router.scale`/`per_expert_scale`, and every piece of per-token
//! scratch the flow in `mod.rs` writes.

use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

/// Per-open Qwen decode state: the recurrent GDN buffers plus every piece
/// of scratch the per-token flow writes.
pub(crate) struct RealQwenState {
    pub(crate) shape: gpu::GdnShape,
    pub(crate) gdn: gpu::GdnStateManager,
    /// `rotary_dim`: `full_head_dim * partial_rotary_factor`, the width of
    /// the rotated prefix of each head.
    pub(crate) rotary_dim: u32,
    /// BF16 `[hidden]` of ones. The INT8 router kernel multiplies `x[n]`
    /// by an effective scale per element; Qwen has no `router.scale` and
    /// `router_scaled` is false, so the scale is identically 1.
    pub(crate) router_ones: gpu::MetalBuffer,
    /// `[num_experts]` of ones, for the same reason: Qwen has no
    /// `per_expert_scale`, so `router_topk_gemma4`'s weighting is a no-op
    /// and the selection reduces to softmax-over-the-selected.
    pub(crate) per_expert_ones: Vec<f32>,
    pub(crate) router_logits_f32: gpu::MetalBuffer,

    /// `[2 * num_heads * full_head_dim]`: q_proj's packed query/gate rows.
    pub(crate) q_packed: gpu::MetalBuffer,
    /// `[num_heads * full_head_dim]`: the gate half, after the split.
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) gdn_qkv_raw: gpu::MetalBuffer,
    pub(crate) gdn_conv_out: gpu::MetalBuffer,
    pub(crate) gdn_z: gpu::MetalBuffer,
    pub(crate) gdn_a: gpu::MetalBuffer,
    pub(crate) gdn_b: gpu::MetalBuffer,
    pub(crate) gdn_y: gpu::MetalBuffer,
    pub(crate) gdn_out: gpu::MetalBuffer,
    /// `[hidden]`: the one post-attention norm that feeds the router, the
    /// shared expert, and the routed experts alike.
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the gated shared-expert output, which doubles as the
    /// residual phase 2 adds its routed sum onto.
    pub(crate) h1: gpu::MetalBuffer,
    /// `[hidden]`: shared + routed, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
    /// One half: the shared expert's scalar gate logit.
    pub(crate) shared_gate_logit: gpu::MetalBuffer,
}

impl RealQwenState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || !arch.attn_output_gate
            || !arch.shared_expert_gated
            || !arch.rope_neox_subdim
        {
            return unsupported(
                "the Qwen 3.6 flow needs attnOutputGate + sharedExpertGated + ropeNeoxSubdim \
                 and none of ffnSandwichNorms / routerScaled / embeddingScaledBySqrtHidden"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("Qwen 3.6 has no final logit softcap".to_string());
        }
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported("Qwen 3.6 installs are MoE on every layer".to_string());
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        if arch
            .full_attention_layer_mask
            .iter()
            .any(|&m| m != 1 && m != 2)
        {
            return unsupported(
                "Qwen 3.6 layers are full attention (1) or gated DeltaNet (2) only".to_string(),
            );
        }
        if !arch.has_linear_attention_layers() {
            return unsupported("a Qwen 3.6 install with no linear layers is not Qwen".to_string());
        }

        let shape = gpu::GdnShape {
            num_k_heads: arch.linear_attention.num_k_heads as u32,
            num_v_heads: arch.linear_attention.num_v_heads as u32,
            key_head_dim: arch.linear_attention.key_head_dim as u32,
            value_head_dim: arch.linear_attention.value_head_dim as u32,
            conv_kernel_size: arch.linear_attention.conv_kernel_size as u32,
        };
        shape.validate().map_err(RealForwardError::Gpu)?;

        let head_dim = arch.full_head_dim;
        let rotary_dim = (head_dim as f64 * arch.partial_rotary_factor).round() as i64;
        if rotary_dim <= 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return unsupported(format!(
                "rotary_dim {rotary_dim} (full_head_dim {head_dim} x partial_rotary_factor \
                 {}) must be positive, even, and at most full_head_dim",
                arch.partial_rotary_factor
            ));
        }

        // Fail at open, not at token 1: probe one layer of each kind for
        // the tensors this flow will bind.
        let hidden = arch.hidden_size as usize;
        let num_experts = arch.num_experts as usize;
        for layer in 0..arch.num_layers as usize {
            let mut probes = vec![
                layer_tensor(layer, "input_layernorm.weight"),
                layer_tensor(layer, "post_attention_layernorm.weight"),
                layer_tensor(layer, "mlp.gate.weight"),
                layer_tensor(layer, "mlp.shared_expert_gate.weight"),
                layer_tensor(layer, "mlp.shared_expert.gate_proj.weight"),
            ];
            probes.extend(if arch.layer_is_linear(layer) {
                [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.conv1d.weight",
                    // No `.weight` suffix on either of these.
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ]
                .iter()
                .map(|s| layer_tensor(layer, s))
                .collect::<Vec<_>>()
            } else {
                ["self_attn.q_proj.weight", "self_attn.q_norm.weight"]
                    .iter()
                    .map(|s| layer_tensor(layer, s))
                    .collect::<Vec<_>>()
            });
            for name in probes {
                entry(index, &name)?;
            }
        }
        let head_name = if arch.tie_word_embeddings {
            "language_model.model.embed_tokens.weight"
        } else {
            "language_model.lm_head.weight"
        };
        entry(index, head_name)?;
        entry(index, "language_model.model.norm.weight")?;
        let _ = weights;

        let ones: Vec<u8> = (0..hidden).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let q_dim = (arch.num_heads * head_dim) as usize;
        let qkv_dim = shape.qkv_dim() as usize;
        let value_dim = shape.value_dim() as usize;
        let v_heads = shape.num_v_heads as usize;
        Ok(Self {
            gdn: gpu::GdnStateManager::new(context.device(), arch),
            shape,
            rotary_dim: rotary_dim as u32,
            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            router_logits_f32: context.new_output_buffer((num_experts * 4) as u64),
            q_packed: halfs(2 * q_dim),
            attn_gate: halfs(q_dim),
            gdn_qkv_raw: halfs(qkv_dim),
            gdn_conv_out: halfs(qkv_dim),
            gdn_z: halfs(value_dim),
            gdn_a: halfs(v_heads),
            gdn_b: halfs(v_heads),
            gdn_y: halfs(value_dim),
            gdn_out: halfs(value_dim),
            moe_x: halfs(hidden),
            h1: halfs(hidden),
            h2: halfs(hidden),
            shared_gate_logit: halfs(1),
        })
    }

    /// Rewinds the recurrent state to empty context. Both the delta rule
    /// and the causal conv define that state as zero.
    pub(crate) fn reset(&mut self) {
        self.gdn.reset();
    }
}
