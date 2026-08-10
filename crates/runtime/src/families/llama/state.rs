//! `RealLlamaState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `llama`-architecture install (ROADMAP
//! Phase M2).
//!
//! Much smaller than its Gemma and Qwen siblings, and the absences are the
//! point: no recurrent state, no packed q/gate split, no shared expert, no
//! per-head norms, no logit softcap. What is left is the unit router scales
//! (this architecture has neither a `router.scale` nor a `per_expert_scale`,
//! exactly as Qwen does not) plus two pieces of per-token scratch.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::llama::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

pub(crate) struct RealLlamaState {
    /// Rotated PAIRS per head: `head_dim * partial_rotary_factor / 2`, which
    /// at this architecture's `partial_rotary_factor = 1.0` is `head_dim / 2`
    /// -- i.e. default full-head NeoX, the whole head rotated.
    pub(crate) rotated_pairs: u32,
    /// Norm q and k PER HEAD, with a learned `[head_dim]` weight, before
    /// RoPE. False for the `llama` architecture (Mixtral norms neither),
    /// true for `qwen3moe`. Derived from the FAMILY and never from whether
    /// the tensors happen to be present: a name probe cannot tell a model
    /// that has no q-norm from an install that lost one.
    pub(crate) qk_norm: bool,
    /// RMS epsilon. `llama` publishes 1e-5 and `qwen3moe` 1e-6, and this is
    /// not an `ArchConfig` field, so the family carries it. Small enough to
    /// be invisible in a smoke test and large enough to move a perplexity
    /// digit, which is exactly the class of difference this port freezes
    /// goldens over.
    pub(crate) rms_eps: f32,
    /// BF16 `[hidden]` of ones: the INT8 router kernel scales `x[n]` by an
    /// effective scale per element, and this architecture has no
    /// `router.scale`, so the scale is identically 1. Same reason Qwen's
    /// state carries one.
    pub(crate) router_ones: gpu::MetalBuffer,
    /// `[num_experts]` of ones, so `router_topk_gemma4`'s per-expert
    /// weighting is a no-op and selection reduces to softmax over the
    /// selected.
    pub(crate) per_expert_ones: Vec<f32>,
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]`: the post-attention norm that feeds the router and the
    /// routed experts. There is no shared expert to feed.
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the routed sum, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
}

impl RealLlamaState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        // Every behavioural extension this architecture does NOT have.
        // Checked rather than assumed, because each one is a manifest field
        // with a GEMMA fallback (AGENTS.md Gotcha 24), so an install that
        // omitted them would arrive here claiming Gemma's answers.
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.rope_neox_subdim
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the llama flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 ropeNeoxSubdim / attentionKEqV"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("the llama architecture has no final logit softcap".to_string());
        }
        if arch.full_attention_layer_mask.iter().any(|&m| m != 1) {
            return unsupported(
                "every llama layer is full attention (mask 1); a sliding-window \
                 Mistral is not wired"
                    .to_string(),
            );
        }
        // THE DENSE HALF IS REFUSED, DELIBERATELY AND BY NAME. One
        // `general.architecture` covers dense Llama 2/3.x, Mistral AND the
        // Mixtral MoEs, and only the MoE half reuses the routed-expert
        // streamer this engine's memory result comes from. A dense install
        // needs a GPU dense-FFN path (ROADMAP Phase M2 step 2); until that
        // lands, refusing is the honest answer, since the alternative is a
        // CPU-bridged FFN that would run and be slow for reasons no one
        // could see.
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported(
                "this is a DENSE llama install (no routed experts) and the llama flow is \
                 MoE-only so far: the dense FFN has no GPU path here yet. Mixtral-style \
                 checkpoints run; Llama 2/3.x and Mistral do not"
                    .to_string(),
            );
        }
        if arch.top_k_experts as usize > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {} exceeds the {}-slot MoE kernels",
                arch.top_k_experts,
                gpu::MAX_STREAMED_EXPERTS
            ));
        }

        let head_dim = arch.full_head_dim;
        let rotated_pairs = (head_dim as f64 * arch.partial_rotary_factor / 2.0).round() as i64;
        if rotated_pairs <= 0 || rotated_pairs > head_dim / 2 {
            return unsupported(format!(
                "rotated pairs {rotated_pairs} (full_head_dim {head_dim} x \
                 partial_rotary_factor {}) must be positive and at most half the head",
                arch.partial_rotary_factor
            ));
        }

        // The two places `qwen3moe` differs from `llama`. Both are read off
        // the FAMILY rather than sniffed, per Gotcha 12's rule: two
        // architectures that share a tensor-name contract cannot be told
        // apart by their tensor names.
        let qk_norm = arch.family == ModelFamily::Qwen3Moe;
        let rms_eps = if qk_norm { 1e-6 } else { 1e-5 };

        // Fail at open, not at token 1.
        let hidden = arch.hidden_size as usize;
        let num_experts = arch.num_experts as usize;
        for layer in 0..arch.num_layers as usize {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate.weight",
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            if qk_norm {
                for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
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
        Ok(Self {
            rotated_pairs: rotated_pairs as u32,
            qk_norm,
            rms_eps,
            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            router_logits_f32: context.new_output_buffer((num_experts * 4) as u64),
            moe_x: halfs(hidden),
            h2: halfs(hidden),
        })
    }
}
