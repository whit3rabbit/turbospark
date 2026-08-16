//! `RealMuseState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `muse_glimmer` install, plus the FOUR
//! published scalars this family carries as constants.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::museglimmer::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `text_config.rms_norm_eps`. The input norm, the pre-FFN norm, the two
/// per-head q/k norms and the embedding norm.
pub(crate) const RMS_EPS: f32 = 1e-5;

/// `text_config.post_norm_eps`. The two POST norms ONLY
/// (`post_attention_layernorm`, `post_feedforward_layernorm`), and four
/// orders of magnitude smaller than its sibling.
///
/// **This family is the first here to need a PAIR**, where `llama` carries
/// 1e-5 and `qwen3moe` 1e-6 as one number each. Using [`RMS_EPS`] for all six
/// norms is finite, fluent and wrong; nothing but a perplexity row can see
/// it.
pub(crate) const POST_NORM_EPS: f32 = 1e-8;

/// `text_config.qk_scale_factor`. Multiplies Q AFTER its per-head no-scale
/// norm and BEFORE RoPE.
///
/// SEPARATE from `ArchConfig::attention_scale` (`128^-0.5`), and deliberately
/// not folded into it. Folding is algebraically equivalent and is rejected
/// twice over: the product is an arbitrary decimal on `arch_validation`'s
/// `!=`-on-`f64` path (AGENTS.md Gotcha 24), and it reassociates the
/// floating-point order the reference fixes by rounding back to the
/// activation dtype between the two multiplies.
pub(crate) const QK_SCALE_FACTOR: f32 = 3.87;

/// `text_config.output_multiplier`, which is `26^-0.5`. Multiplies the logits
/// BEFORE the softcap.
///
/// Order matters and is not recoverable afterwards: `softcap(z * m)` is not
/// `softcap(z) * m`, and at a softcap of 20 with `m < 1` the two differ over
/// most of the range.
pub(crate) const OUTPUT_MULTIPLIER: f32 = 0.196_116_13;

/// All four constants above are RECALLED values where `config.json` states
/// them, which is AGENTS.md Gotcha 38's shape. What keeps that honest is
/// `crates/repack/tests/museglimmer_config.rs`, which parses the real config
/// and asserts these exact numbers offline in milliseconds. Do not change one
/// here without changing it there; the test is the reason they may live here
/// at all.
pub(crate) struct RealMuseState {
    /// Rotated PAIRS per head on a ROTATING layer. `partial_rotary_factor` is
    /// 1.0, so this is `head_dim / 2`: the whole head, NeoX-style.
    pub(crate) rotated_pairs: u32,
    /// `[num_heads * head_dim]`: the attention output GATE, projected from
    /// the layer's normed input by `self_attn.gate_proj`.
    ///
    /// Lives here rather than in `DecodeScratch` because no other family has
    /// one at this width -- Qwen's gate arrives packed inside `q_proj` and is
    /// split into buffers that already exist.
    pub(crate) attn_gate: gpu::MetalBuffer,
}

impl RealMuseState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        if arch.family != ModelFamily::MuseGlimmer {
            return unsupported(format!(
                "the muse_glimmer flow refuses family {:?}; flow selection keys on the family, \
                 never on tensor naming",
                arch.family
            ));
        }

        // The behavioural extensions this architecture DOES have, and the
        // ones it does not, checked rather than assumed: each is a manifest
        // field with a GEMMA fallback (AGENTS.md Gotcha 24), so an install
        // that omitted them would arrive here claiming Gemma's answers.
        if !arch.ffn_sandwich_norms {
            return unsupported(
                "muse_glimmer normalizes both the attention and the FFN output before adding \
                 them back; an install declaring ffnSandwichNorms false was built from \
                 something else"
                    .to_string(),
            );
        }
        if arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.rope_neox_subdim
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the muse_glimmer flow takes none of routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 ropeNeoxSubdim / attentionKEqV. Note attnOutputGate is FALSE even though this \
                 family HAS an output gate: that field means `q_proj` emits packed [query; \
                 gate] rows, and this family's gate is its own tensor"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap <= 0.0 {
            return unsupported(format!(
                "muse_glimmer softcaps its logits (`final_logit_softcapping`, 20.0 on the \
                 published checkpoint); got {}",
                arch.final_logit_softcap
            ));
        }
        if arch.tie_word_embeddings {
            return unsupported(
                "muse_glimmer ships a separate `lm_head`; an install claiming tied embeddings \
                 was built from something else"
                    .to_string(),
            );
        }
        // DENSE ONLY. This architecture string has no MoE half, unlike
        // `llama`, so a nonzero expert count is a malformed install rather
        // than a second shape to serve.
        if arch.num_experts != 0 || arch.top_k_experts != 0 {
            return unsupported(format!(
                "muse_glimmer is dense; got num_experts {} and top_k_experts {}",
                arch.num_experts, arch.top_k_experts
            ));
        }
        // THE WINDOW MUST BE PRESENT AND THE MASK MUST BE 0/1. An all-full
        // mask decodes fine and is a DIFFERENT MODEL past the window, which
        // no short smoke reaches -- the same refusal `gpt-oss` makes, for the
        // same reason.
        if arch.full_attention_layer_mask.iter().any(|&m| m > 1) {
            return unsupported(
                "muse_glimmer layers are sliding-window (0) or full (1); this architecture has \
                 no linear or compressed layer kinds"
                    .to_string(),
            );
        }
        if !arch.full_attention_layer_mask.contains(&0) {
            return unsupported(
                "muse_glimmer alternates three sliding layers to one full; a mask with no \
                 sliding layer would size every KV buffer at max_context and decode a \
                 different model"
                    .to_string(),
            );
        }
        if arch.sliding_window <= 0 {
            return unsupported(format!(
                "muse_glimmer alternates a sliding window; {} is not a usable one",
                arch.sliding_window
            ));
        }
        // **NoPE IS A REFUSAL AXIS, NOT A DEFAULT.** The full-attention
        // layers rotate nothing, and this port expresses that as
        // `full_rope_theta == 0.0`. An install that carried a nonzero one
        // would rotate those layers -- fluent, wrong, and invisible to every
        // gate short of a perplexity row.
        if arch.full_rope_theta != 0.0 {
            return unsupported(format!(
                "muse_glimmer's full-attention layers are NoPE and this install declares \
                 fullRopeTheta {}; rotating them is a different model that still reads \
                 fluently",
                arch.full_rope_theta
            ));
        }
        if arch.rope_theta <= 0.0 {
            return unsupported(format!(
                "muse_glimmer rotates its SLIDING layers; ropeTheta {} is not a usable base",
                arch.rope_theta
            ));
        }

        let head_dim = arch.full_head_dim;
        if arch.head_dim != head_dim {
            return unsupported(format!(
                "muse_glimmer uses one head dim for both layer kinds; got {} sliding and \
                 {head_dim} full",
                arch.head_dim
            ));
        }
        let rotated_pairs = (head_dim as f64 * arch.partial_rotary_factor / 2.0).round() as i64;
        if rotated_pairs <= 0 || rotated_pairs > head_dim / 2 {
            return unsupported(format!(
                "rotated pairs {rotated_pairs} (full_head_dim {head_dim} x \
                 partial_rotary_factor {}) must be positive and at most half the head",
                arch.partial_rotary_factor
            ));
        }

        // Fail at open, not at token 1. `self_attn.gate_proj` is the one a
        // name table would most plausibly drop -- it collides in NAME with
        // `mlp.gate_proj`, which every family has.
        for layer in 0..arch.num_layers as usize {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.gate_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            // AND THE TWO THAT MUST NOT EXIST. This family's q/k norms are
            // NO-SCALE, so the checkpoint stores no weights for them. A
            // `q_norm.weight` here means the install came from a family whose
            // norms ARE learned, and dispatching this flow's no-scale norm
            // against it would silently ignore a real trained tensor.
            for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                if index.entries.contains_key(&layer_tensor(layer, suffix)) {
                    return unsupported(format!(
                        "layer {layer} carries {suffix}, but muse_glimmer's q/k norms are \
                         no-scale and store no weights; this install is not this family"
                    ));
                }
            }
        }
        entry(index, "language_model.lm_head.weight")?;
        entry(index, "language_model.model.norm.weight")?;
        entry(index, "language_model.model.embed_tokens.weight")?;

        let q_dim = (arch.num_heads * head_dim) as usize;
        Ok(Self {
            rotated_pairs: rotated_pairs as u32,
            attn_gate: context.new_output_buffer((q_dim.max(1) * 2) as u64),
        })
    }
}
