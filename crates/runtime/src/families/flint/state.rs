//! `RealFlintState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `flint` install, and the open-time proof
//! that every tensor the flow will read is present and the right size.
//!
//! Smaller than its `gpt-oss` ancestor: there is no YaRN table (plain RoPE
//! through the scalar-theta kernels), no projection biases and no router
//! bias. What it adds over `deepseek2`, the other dense-lead MoE flow, is
//! attention (per-head QK norm, learned sinks, NoPE by layer mask), carried
//! in `attn.rs`, and a sigmoid router, carried in `moe.rs`.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::flint::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

pub(crate) struct RealFlintState {
    /// Rotated PAIRS per head: `partial_rotary_factor` is 1.0, so this is
    /// `head_dim / 2`, the whole head.
    pub(crate) rotated_pairs: u32,
    /// RMS epsilon, 1e-6 from `config.json`. Not an `ArchConfig` field, so
    /// the family carries it as `RealLlamaState` and `RealDeepseek2State` do.
    pub(crate) rms_eps: f32,
    /// BF16 `[hidden]` of ones: the INT8 router GEMV scales `x[n]` per
    /// element and this architecture has no `router.scale`.
    pub(crate) router_ones: gpu::MetalBuffer,
    /// FP32 router logits, one `[num_experts]` row. Decode only: there is no
    /// chunked driver for this family yet, so one row is enough.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]`: the post-attention norm feeding the dense FFN, or the
    /// router, shared expert and routed experts.
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: shared output plus routed sum, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
    /// `[hidden]`: the shared expert's output, which seeds phase 2.
    pub(crate) shared_out: gpu::MetalBuffer,
    /// Gate and up activations, sized for the wider of the dense lead and
    /// the shared expert. The two never run in the same layer.
    pub(crate) ffn_a: gpu::MetalBuffer,
    pub(crate) ffn_b: gpu::MetalBuffer,
    /// Dense leading layers (`mlp_only_layers`), a prefix.
    pub(crate) lead: usize,
    /// The dense lead's FFN width (6144).
    pub(crate) dense_inter: usize,
    /// The shared expert's width (1024).
    pub(crate) shared_inter: usize,
}

impl RealFlintState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let _ = weights;
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        if arch.family != ModelFamily::Flint {
            return unsupported(format!(
                "the flint flow refuses family {:?}; flow selection keys on the family, never \
                 on tensor naming",
                arch.family
            ));
        }
        // Every behavioural extension Flint does NOT have, checked rather
        // than assumed: each is a manifest field with a GEMMA fallback
        // (AGENTS.md Gotcha 24).
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the flint flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 attentionKEqV"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("flint has no final logit softcap".to_string());
        }
        if arch.tie_word_embeddings {
            return unsupported("flint ships an untied lm_head".to_string());
        }
        if arch.router_scoring_func != "sigmoid" {
            return unsupported(format!(
                "flint scores experts with a sigmoid router; this install declares {:?}",
                arch.router_scoring_func
            ));
        }
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported(format!(
                "flint is MoE; got num_experts {} and top_k_experts {}",
                arch.num_experts, arch.top_k_experts
            ));
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        // Sliding (0) or full (1) only, with a usable window.
        if arch.full_attention_layer_mask.iter().any(|&m| m > 1) {
            return unsupported(
                "flint layers are sliding-window (0) or full (1); no linear or compressed \
                 layer kinds exist in this architecture"
                    .to_string(),
            );
        }
        if arch.sliding_window <= 0 {
            return unsupported(format!(
                "flint slides a window on its sliding layers; {} is not a usable one",
                arch.sliding_window
            ));
        }
        // NoPE is tied to the full-attention mask: a full layer with a rope
        // theta would be a different function that still produces fluent text.
        if arch.full_rope_theta != 0.0 {
            return unsupported(
                "flint's full-attention layers carry no positional encoding; the manifest \
                 must state full_rope_theta 0"
                    .to_string(),
            );
        }
        let head_dim = arch.full_head_dim;
        if arch.head_dim != head_dim || arch.num_kv_heads != arch.num_full_kv_heads {
            return unsupported(
                "flint requires identical sliding and full head dimensions and KV heads"
                    .to_string(),
            );
        }
        let rotated_pairs = (head_dim as f64 * arch.partial_rotary_factor / 2.0).round() as i64;
        if rotated_pairs <= 0 || rotated_pairs > head_dim / 2 {
            return unsupported(format!(
                "rotated pairs {rotated_pairs} must be positive and at most half the head"
            ));
        }
        let lead = arch.num_dense_leading_layers.max(0) as usize;
        let num_layers = arch.num_layers as usize;
        if lead == 0 || lead >= num_layers {
            return unsupported(format!(
                "flint has a dense lead followed by MoE layers; got lead {lead} of {num_layers}"
            ));
        }
        if arch.dense_lead_intermediate_size <= 0 || arch.intermediate_size <= 0 {
            return unsupported(
                "flint needs both a dense lead FFN width and a shared expert width".to_string(),
            );
        }

        let hidden = arch.hidden_size as usize;
        let num_experts = arch.num_experts as usize;
        // Fail at open, not at token 1.
        for layer in 0..num_layers {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.softmax_offset",
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            // ONE SINK PER QUERY HEAD, not per KV head: a short vector reads
            // whatever follows it in the resident mapping as a logit.
            let sinks = entry(index, &layer_tensor(layer, "self_attn.softmax_offset"))?;
            let want = arch.num_heads as usize * 2;
            if sinks.size_bytes as usize != want {
                return unsupported(format!(
                    "layer {layer} attention sinks are {} bytes; expected {want} (one BF16 \
                     logit per QUERY head, {} of them)",
                    sinks.size_bytes, arch.num_heads
                ));
            }
            let ffn: &[&str] = if layer < lead {
                &["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            } else {
                &[
                    "mlp.gate.weight",
                    "mlp.shared_expert.gate_proj.weight",
                    "mlp.shared_expert.up_proj.weight",
                    "mlp.shared_expert.down_proj.weight",
                ]
            };
            for suffix in ffn {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            if layer >= lead {
                let router_name = layer_tensor(layer, "mlp.gate.weight");
                let router = entry(index, &router_name)?;
                if router.dtype != 5 || router.size_bytes as usize != num_experts * hidden {
                    return unsupported(format!(
                        "{router_name}: expected INT8 (dtype 5) {num_experts}x{hidden}, got \
                         dtype {} with {} packed bytes",
                        router.dtype, router.size_bytes
                    ));
                }
            }
        }
        entry(index, "language_model.lm_head.weight")?;
        entry(index, "language_model.model.norm.weight")?;
        entry(index, "language_model.model.embed_tokens.weight")?;

        let ones: Vec<u8> = (0..hidden).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let dense_inter = arch.dense_lead_intermediate_size as usize;
        let shared_inter = arch.intermediate_size as usize;
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let ffn_width = dense_inter.max(shared_inter);
        Ok(Self {
            rotated_pairs: rotated_pairs as u32,
            rms_eps: 1e-6,
            router_ones,
            router_logits_f32: context.new_output_buffer((num_experts * 4) as u64),
            moe_x: halfs(hidden),
            h2: halfs(hidden),
            shared_out: halfs(hidden),
            ffn_a: halfs(ffn_width),
            ffn_b: halfs(ffn_width),
            lead,
            dense_inter,
            shared_inter,
        })
    }
}
