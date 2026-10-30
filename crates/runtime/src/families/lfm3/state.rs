//! `RealLfm3State`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens an `lfm3` install, plus the per-session
//! conv tails that `reset()` must rewind.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::lfm3::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `norm_eps`, 1e-5 on every norm in the stack, the per-head q/k norms
/// included. Carried as a constant on the spark / llama precedent; the
/// parsers refuse any other value (`docs/LFM3_PHASE0.md`).
pub(crate) const RMS_EPS: f32 = 1e-5;

pub(crate) struct RealLfm3State {
    /// Rotary PAIRS per head on a FULL layer: `partial_rotary_factor *
    /// head_dim / 2` = 32 on the published checkpoint. Sliding layers rotate
    /// all `head_dim / 2` pairs.
    pub(crate) full_rotated_pairs: u32,
    /// Conv taps (`short_conv_kernel_size`, 7).
    pub(crate) taps: u32,
    /// `[3 * hidden]`: one conv layer's `in_proj` output, `B | C | xx`.
    pub(crate) in_proj: gpu::MetalBuffer,
    /// `[hidden]`: `C * conv(B * xx)`, the `out_proj` GEMV's input.
    pub(crate) conv_out: gpu::MetalBuffer,
    /// `[num_heads]`: the headwise output gate's pre-sigmoid logits.
    pub(crate) attn_gate: gpu::MetalBuffer,
    /// The conv tails, one `[taps - 1, hidden]` FP16 window per conv layer.
    pub(crate) conv: gpu::ShortConvStateManager,
    /// Decode's FFN takes the fused gate+up+silu GEMV where the weights
    /// allow (`mlp.rs`); seeded from `TURBOSPARK_LFM3_FUSED_FFN`.
    pub(crate) fused_ffn: bool,
    /// The chunked prefill driver's M-row buffers (`prefill.rs`), allocated
    /// at its first chunk so an install that only decodes never pays for
    /// them.
    pub(crate) batched: Option<Lfm3Batched>,
}

/// The M-row siblings of the single-row buffers above and in
/// `DecodeScratch`, every one `[rows, width]` FP16 and row-major. `x` is not
/// here: `DecodeScratch::x` already holds `MAX_PREFILL_ROWS` rows.
pub(crate) struct Lfm3Batched {
    pub(crate) rows: usize,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) in_proj: gpu::MetalBuffer,
    pub(crate) conv_out: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    /// K and V rows before they reach the cache: the per-head k norm and
    /// rope run here, then a copy lands each run of rows in its slots, so a
    /// sliding layer's ring can wrap inside a micro-batch.
    pub(crate) k_stage: gpu::MetalBuffer,
    pub(crate) v_stage: gpu::MetalBuffer,
    pub(crate) attn_gate: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) o: gpu::MetalBuffer,
    pub(crate) ffn_normed: gpu::MetalBuffer,
    pub(crate) ffn_gate: gpu::MetalBuffer,
    pub(crate) ffn_up: gpu::MetalBuffer,
    pub(crate) ffn_act: gpu::MetalBuffer,
    pub(crate) ffn_out: gpu::MetalBuffer,
    /// The multi-row GQA partials for the FULL layers, `None` where the head
    /// shape is not one the kernel serves.
    pub(crate) rows_attention: Option<gpu::GqaRowsScratch>,
}

impl Lfm3Batched {
    pub(crate) fn new(context: &mut gpu::MetalContext, arch: &ArchConfig, rows: usize) -> Self {
        let hidden = arch.hidden_size as u64;
        let inter = arch.intermediate_size as u64;
        let heads = arch.num_heads as u32;
        let kv_heads = arch.num_kv_heads as u32;
        let head_dim = arch.full_head_dim as u32;
        let q_dim = u64::from(heads * head_dim);
        let kv_dim = u64::from(kv_heads * head_dim);
        let rows_attention =
            gpu::attention_decode_gqa_supported(head_dim, heads, kv_heads).then(|| {
                gpu::GqaRowsScratch::new(
                    context,
                    rows as u32,
                    heads,
                    head_dim,
                    gpu::GQA_ROWS_SCRATCH_CHUNKS,
                )
            });
        let halfs = |n: u64| context.new_output_buffer(n.max(1) * rows as u64 * 2);
        Self {
            rows,
            normed: halfs(hidden),
            in_proj: halfs(3 * hidden),
            conv_out: halfs(hidden),
            q: halfs(q_dim),
            k_stage: halfs(kv_dim),
            v_stage: halfs(kv_dim),
            attn_gate: halfs(u64::from(heads)),
            attn_out: halfs(q_dim),
            o: halfs(hidden),
            ffn_normed: halfs(hidden),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            ffn_out: halfs(hidden),
            rows_attention,
        }
    }
}

impl RealLfm3State {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if arch.family != ModelFamily::Lfm3 {
            return unsupported(format!(
                "the lfm3 flow refuses family {:?}; flow selection keys on the family, never \
                 on tensor naming",
                arch.family
            ));
        }
        // The behavioural extensions this flow applies unconditionally, pinned
        // by the fields that would name them. `ffn_sandwich_norms` must be ON:
        // the post-mixer and post-FFN norms are part of every layer.
        if !arch.ffn_sandwich_norms {
            return unsupported(
                "the lfm3 flow applies sandwich norms on every layer; an install declaring \
                 ffnSandwichNorms false is a different layer graph"
                    .to_string(),
            );
        }
        if arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.attention_k_eq_v
            || arch.rope_neox_subdim
        {
            return unsupported(
                "the lfm3 flow takes none of routerScaled / embeddingScaledBySqrtHidden / \
                 attnOutputGate / sharedExpertGated / attentionKEqV / ropeNeoxSubdim. \
                 attnOutputGate is FALSE although every attention layer has a gate: that \
                 field means a packed q_proj, and this gate is its own headwise tensor"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported(format!(
                "lfm3 applies no logit softcap; got {}",
                arch.final_logit_softcap
            ));
        }
        if !arch.tie_word_embeddings {
            return unsupported("lfm3 ties its head to the embedding".to_string());
        }
        if arch.hidden_activation != "silu" {
            return unsupported(format!(
                "lfm3's FFN is SwiGLU (silu); got hiddenActivation {:?}",
                arch.hidden_activation
            ));
        }
        if arch.num_experts != 0 || arch.top_k_experts != 0 {
            return unsupported("lfm3 is dense".to_string());
        }
        if arch
            .full_attention_layer_mask
            .iter()
            .any(|&m| !matches!(m, 0 | 1 | 6))
        {
            return unsupported(
                "lfm3 layers are short-conv (6), full (1) or sliding (0)".to_string(),
            );
        }
        if !arch.has_short_conv_layers() {
            return unsupported(
                "an lfm3 install with no short-conv layer is not this family".to_string(),
            );
        }
        if arch.full_attention_layer_mask.contains(&0) && arch.sliding_window <= 0 {
            return unsupported(format!(
                "lfm3's sliding layers need a positive window; got {}",
                arch.sliding_window
            ));
        }
        if arch.full_rope_theta <= 0.0 || arch.rope_theta <= 0.0 {
            return unsupported(format!(
                "lfm3 rotates every attention layer, per class: fullRopeTheta {} and \
                 ropeTheta {} must both be positive",
                arch.full_rope_theta, arch.rope_theta
            ));
        }
        let head_dim = arch.full_head_dim;
        if arch.head_dim != head_dim || arch.num_kv_heads != arch.num_full_kv_heads {
            return unsupported(
                "lfm3 uses one head dim and one kv-head count for both attention kinds".to_string(),
            );
        }
        let rotated = arch.partial_rotary_factor * head_dim as f64 / 2.0;
        if rotated <= 0.0 || rotated.fract() != 0.0 || rotated > (head_dim / 2) as f64 {
            return unsupported(format!(
                "full-layer rotation {} x head_dim {head_dim} is not a whole number of pairs",
                arch.partial_rotary_factor
            ));
        }
        let taps = arch.short_conv_kernel_size;
        if !(2..=16).contains(&taps) {
            return unsupported(format!("shortConvKernelSize {taps} is outside 2..=16"));
        }

        // Fail at open, not at token 1: each layer carries exactly its own
        // mixer's tensors, and the conv taps must be the BF16 `[hidden, taps]`
        // the kernel reads.
        let hidden = arch.hidden_size as usize;
        for layer in 0..arch.num_layers as usize {
            let mixer: &[&str] = if arch.layer_is_short_conv(layer) {
                &[
                    "conv.in_proj.weight",
                    "conv.conv.weight",
                    "conv.out_proj.weight",
                ]
            } else {
                &[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.gate_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                ]
            };
            for suffix in mixer.iter().chain(&[
                "operator_norm.weight",
                "post_operator_norm.weight",
                "ffn_norm.weight",
                "post_ffn_norm.weight",
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
            ]) {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            if arch.layer_is_short_conv(layer) {
                let w = entry(index, &layer_tensor(layer, "conv.conv.weight"))?;
                if w.size_bytes as usize != hidden * taps as usize * 2 {
                    return unsupported(format!(
                        "layer {layer} conv.conv.weight is {} bytes, not BF16 [{hidden}, {taps}]",
                        w.size_bytes
                    ));
                }
            }
        }
        entry(index, "language_model.model.embedding_norm.weight")?;
        entry(index, "language_model.model.embed_tokens.weight")?;
        if index.entries.contains_key("language_model.lm_head.weight") {
            return unsupported(
                "the install carries language_model.lm_head.weight, but lfm3 ties its head"
                    .to_string(),
            );
        }

        let conv_layers: Vec<bool> = (0..arch.num_layers as usize)
            .map(|l| arch.layer_is_short_conv(l))
            .collect();
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let in_proj = halfs(3 * hidden);
        let conv_out = halfs(hidden);
        let attn_gate = halfs(arch.num_heads as usize);
        Ok(Self {
            full_rotated_pairs: rotated as u32,
            taps: taps as u32,
            in_proj,
            conv_out,
            attn_gate,
            conv: gpu::ShortConvStateManager::new(
                context.device(),
                &conv_layers,
                hidden,
                taps as usize,
            ),
            fused_ffn: super::mlp::lfm3_fused_ffn(),
            batched: None,
        })
    }

    /// Zeroes every conv tail. Rewinding only the KV cache would leak the
    /// previous generation into every conv layer: finite, deterministic and
    /// wrong (`docs/NEW_MODEL.md` Phase 3).
    pub(crate) fn reset(&mut self) {
        self.conv.reset();
    }
}
