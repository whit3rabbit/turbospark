//! `RealSparkState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `spark2_5` install, plus the one published
//! scalar this family carries as a constant.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::spark::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `rms_norm_eps`, 1e-6 on every norm in the stack (there is one epsilon;
/// the muse pair is that family's own quirk). Same value as `qwen3moe`'s,
/// carried as a constant on the same llama/qwen3moe precedent rather than
/// spent as an ArchConfig field. Pinned offline by the baseline tests and
/// `docs/SPARK_PHASE0.md`.
pub(crate) const RMS_EPS: f32 = 1e-6;

pub(crate) struct RealSparkState {
    /// Rotary PAIRS per head on a FULL layer: `round(head_dim *
    /// partial_rotary_factor / 2)` = 32 on the published checkpoint (64
    /// dims). The sliding layers rotate the whole head (128 pairs) at the
    /// other theta, which the attention block derives per layer.
    pub(crate) full_rotary_dim: u32,
    /// The fused QKV projection's row staging: `[q | k | v]` as ONE GEMV
    /// output of `num_heads * head_dim + 2 * num_kv_heads * head_dim`
    /// elements, split into `scratch.q` and the cache slots by
    /// `gpu::encode_split_qkv`. Lives here rather than in `DecodeScratch`
    /// because no other family reads a fused q/k/v tensor (Qwen's packing
    /// is `[q; gate]` and is split by `split_q_gate`).
    pub(crate) qkv: gpu::MetalBuffer,
    /// `[num_heads]`: the headwise attention output gate, projected from
    /// the layer's normed input by `self_attn.g_proj`. One SCALAR per head,
    /// broadcast over head_dim -- a shape no other family's gate has (Qwen
    /// packs full-width per-head into `q_proj`, muse is full-width).
    pub(crate) attn_gate: gpu::MetalBuffer,
}

impl RealSparkState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        if arch.family != ModelFamily::Spark25 {
            return unsupported(format!(
                "the spark2_5 flow refuses family {:?}; flow selection keys on the family, \
                 never on tensor naming",
                arch.family
            ));
        }

        // THE BEHAVIOURAL EXTENSIONS, checked rather than assumed: each is a
        // manifest field a wrong install could carry with a plausible value.
        // This family is a PLAIN pre-norm stack -- none of the tricks any
        // sibling here turns on -- except the three that ARE its own (the
        // per-class rope, the headwise gate, the erf GELU), which the flow
        // applies unconditionally and which the checks below therefore pin
        // by their absence flags.
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the spark2_5 flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 attentionKEqV. Note attnOutputGate is FALSE even though this family HAS \
                 an output gate: that field means `q_proj` emits packed [query; gate] \
                 rows, and this family's gate is its own headwise `g_proj` tensor"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported(format!(
                "spark2_5 applies no logit softcap (`final_logit_softcap` 0.0 on the \
                 published checkpoint); got {}",
                arch.final_logit_softcap
            ));
        }
        // TIED, the inverse of muse's axis: the GGUF ships no `output.weight`
        // and the head re-reads the embedding. An untied install claiming
        // this family has no head to read.
        if !arch.tie_word_embeddings {
            return unsupported(
                "spark2_5 ties its output head to the embedding (the GGUF carries no \
                 output.weight); an install declaring tieWordEmbeddings false was built \
                 from something else"
                    .to_string(),
            );
        }
        if arch.hidden_activation != "gelu" {
            return unsupported(format!(
                "spark2_5's FFN activation is exact-erf GELU (`hidden_act`, and the \
                 reference implementation refuses anything else); this install declares \
                 hiddenActivation {:?}. Note the string must be exactly \"gelu\": \
                 \"gelu_pytorch_tanh\" names the tanh approximation, which is a \
                 different function this flow must not silently run",
                arch.hidden_activation
            ));
        }
        // DENSE ONLY.
        if arch.num_experts != 0 || arch.top_k_experts != 0 {
            return unsupported(format!(
                "spark2_5 is dense; got num_experts {} and top_k_experts {}",
                arch.num_experts, arch.top_k_experts
            ));
        }
        // THE WINDOW MUST BE PRESENT AND THE MASK MUST BE 0/1, on the same
        // refusal axis as muse: an all-full mask decodes fine and is a
        // DIFFERENT model past the window.
        if arch.full_attention_layer_mask.iter().any(|&m| m > 1) {
            return unsupported(
                "spark2_5 layers are sliding-window (0) or full (1); this architecture has \
                 no linear or compressed layer kinds"
                    .to_string(),
            );
        }
        if !arch.full_attention_layer_mask.contains(&0) {
            return unsupported(
                "spark2_5 alternates three sliding layers to one full; a mask with no \
                 sliding layer would size every KV buffer at max_context and decode a \
                 different model"
                    .to_string(),
            );
        }
        if arch.sliding_window <= 0 {
            return unsupported(format!(
                "spark2_5 alternates a sliding window; {} is not a usable one",
                arch.sliding_window
            ));
        }
        // **PER-CLASS ROPE, AND BOTH CLASSES ROTATE.** The inverse of muse's
        // NoPE axis: here a zero theta on EITHER class is a malformed install,
        // because the reference builds a rope cache per layer type from
        // nonzero thetas (5e6 full, 1e4 sliding). The DIVERGENCE between the
        // two is the family's shape; a flow that applied one theta everywhere
        // would still decode fluently.
        if arch.full_rope_theta <= 0.0 || arch.rope_theta <= 0.0 {
            return unsupported(format!(
                "spark2_5 rotates every layer, per class: fullRopeTheta {} and ropeTheta {} \
                 must both be positive (the muse-style zero-means-NoPE convention does \
                 not apply to this family)",
                arch.full_rope_theta, arch.rope_theta
            ));
        }
        if !(arch.partial_rotary_factor > 0.0 && arch.partial_rotary_factor <= 1.0) {
            return unsupported(format!(
                "spark2_5's partialRotaryFactor {} must be in (0, 1]: it is the FULL \
                 layers' rotary fraction, with the sliding layers at 1.0 hard-coded",
                arch.partial_rotary_factor
            ));
        }
        let head_dim = arch.full_head_dim;
        if arch.head_dim != head_dim {
            return unsupported(format!(
                "spark2_5 uses one head dim for both layer kinds; got {} sliding and \
                 {head_dim} full",
                arch.head_dim
            ));
        }
        if arch.num_kv_heads != arch.num_full_kv_heads {
            return unsupported(format!(
                "spark2_5 uses one kv-head count for both layer kinds; got {} sliding and \
                 {} full",
                arch.num_kv_heads, arch.num_full_kv_heads
            ));
        }
        let full_rotary_dim = (head_dim as f64 * arch.partial_rotary_factor).round() as i64;
        if full_rotary_dim <= 0 || full_rotary_dim > head_dim || full_rotary_dim % 2 != 0 {
            return unsupported(format!(
                "full-layer rotary dim {full_rotary_dim} (head_dim {head_dim} x \
                 partial_rotary_factor {}) must be even, positive, and at most the head",
                arch.partial_rotary_factor
            ));
        }

        // Fail at open, not at token 1. `self_attn.g_proj` is the one a name
        // table would most plausibly drop (it collides in POSITION with
        // muse's `self_attn.gate_proj`, a different width); the fused
        // `q_k_v_proj` is the one a neighbour's flow would most plausibly
        // ask for under three other names.
        for layer in 0..arch.num_layers as usize {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_k_v_proj.weight",
                "self_attn.g_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
        }
        entry(index, "language_model.model.norm.weight")?;
        entry(index, "language_model.model.embed_tokens.weight")?;
        // AND THE ONE THAT MUST NOT EXIST: a separate head means an untied
        // install, which the arch check above already refuses; this names the
        // tensor rather than the field so the message points at a file.
        if index.entries.contains_key("language_model.lm_head.weight") {
            return unsupported(
                "the install carries language_model.lm_head.weight, but spark2_5 ties its \
                 head to the embedding; this install is not this family"
                    .to_string(),
            );
        }

        let num_heads = arch.num_heads as usize;
        let num_kv = arch.num_kv_heads as usize;
        let head_dim_u = head_dim as usize;
        // `new_output_buffer` takes BYTES (`f16` = 2 per element), the same
        // convention muse's gate allocation uses. The fused row is
        // `q + 2*kv` elements; undersizing it here would have the fused
        // GEMV write past the buffer into whatever the allocator placed
        // next, which corrupts neighbours rather than erroring.
        let qkv_elems = (num_heads * head_dim_u + 2 * num_kv * head_dim_u).max(1);
        Ok(Self {
            full_rotary_dim: full_rotary_dim as u32,
            qkv: context.new_output_buffer(qkv_elems as u64 * 2),
            // The gate is one scalar per head.
            attn_gate: context.new_output_buffer((num_heads.max(1) * 2) as u64),
        })
    }
}
