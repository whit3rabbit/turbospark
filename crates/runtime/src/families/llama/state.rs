//! `RealLlamaState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `llama`-architecture install (ROADMAP
//! Phase M2).
//!
//! Shared full-attention state, including family-selected Q/K normalization
//! and router scoring. No recurrent state, output gates, or shared expert.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::llama::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_types::MAX_PREFILL_BATCH;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum QkNorm {
    None,
    PerHead,
    Projection,
}

pub(crate) struct RealLlamaState {
    /// Rotated pairs per head. MiniMax rotates only the first 64 elements;
    /// the other families use the full head width.
    pub(crate) rotated_pairs: u32,
    /// Family-selected Q/K normalization before RoPE. MiniMax spans the
    /// entire projection; Qwen3 uses independent heads. Never inferred from whether
    /// the tensors happen to be present: a name probe cannot tell a model
    /// that has no q-norm from an install that lost one.
    pub(super) qk_norm: QkNorm,
    /// Qwen2 adds learned biases to q, k, and v before RoPE. The flag is
    /// family-selected rather than inferred from optional tensor presence.
    pub(super) qkv_bias: bool,
    pub(super) correction_bias: Vec<Vec<f32>>,
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
    /// selected. Empty on a dense install.
    pub(crate) per_expert_ones: Vec<f32>,
    /// DENSE, i.e. `num_experts == 0`: one `general.architecture = "llama"`
    /// covers Mistral and Llama 2/3.x as well as the Mixtral MoEs (ROADMAP
    /// M4). Read off `num_experts` and not off tensor naming, for the reason
    /// `qk_norm` is read off the family.
    ///
    /// It changes what the FFN half of a layer is and nothing above it:
    /// attention, both norms and the head are the same code either way.
    pub(crate) dense: bool,
    /// Capacity-sized Q/output and split-KV scratch reserved only for dense,
    /// full-attention Llama configurations. Other families share this state
    /// type but keep their existing allocation footprint.
    #[allow(dead_code)] // The next runtime task adds the row-staging reader.
    pub(crate) batch_attention_buffers: Option<BatchAttentionBuffers>,
    /// FP32 router logits, one `[num_experts]` row per token of a prefill
    /// micro-batch (`MAX_PREFILL_BATCH` rows, matching `RealGemmaState`'s
    /// field of the same name): the chunked-prefill driver's per-layer
    /// command buffer writes all M tokens' router GEMVs before the host
    /// reads any of them back. The sequential decode path always uses row 0.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]` per token of a prefill micro-batch: the post-attention
    /// norm that feeds the router and the routed experts. There is no
    /// shared expert to feed. Written in the attention half of a layer and
    /// read in the routed half, with a commit between them in the chunked
    /// driver, so it needs a row per token for the same reason
    /// `RealGemmaState::routed_x` does (`crates/runtime/CLAUDE.md`
    /// Gotcha 14).
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the routed sum, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
}

/// Persistent allocations for the dense-Llama batched-attention path.
/// Query/output byte lengths and GPU scratch all come from the same layout.
/// The later memory-oracle gate checks process peak separately.
pub(crate) struct BatchAttentionBuffers {
    #[allow(dead_code)] // Read when task 3.2 stages query rows.
    pub(crate) q: gpu::MetalBuffer,
    #[allow(dead_code)] // Read after task 3.3 completes batched attention.
    pub(crate) output: gpu::MetalBuffer,
    #[allow(dead_code)] // Bound by the batch encoder in task 3.4.
    pub(crate) scratch: gpu::BatchAttentionScratch,
}

impl BatchAttentionBuffers {
    fn allocate(
        context: &gpu::MetalContext,
        layout: gpu::BatchAttentionScratchLayout,
    ) -> Result<Self, RealForwardError> {
        let q_bytes = metal_buffer_byte_length(layout.q_buffer_bytes())?;
        let output_bytes = metal_buffer_byte_length(layout.output_buffer_bytes())?;
        let q = context.new_output_buffer(q_bytes);
        let output = context.new_output_buffer(output_bytes);
        let scratch =
            gpu::BatchAttentionScratch::new(context, layout).map_err(RealForwardError::Gpu)?;

        Ok(Self { q, output, scratch })
    }
}

fn metal_buffer_byte_length(bytes: usize) -> Result<u64, RealForwardError> {
    u64::try_from(bytes).map_err(|_| {
        RealForwardError::Gpu(gpu::GpuError::InvalidInput(
            "batch-attention buffer length exceeds Metal's byte range".into(),
        ))
    })
}

fn batch_attention_eligible(
    family: ModelFamily,
    num_experts: i64,
    num_layers: i64,
    full_attention_layer_mask: &[u8],
) -> bool {
    let Ok(num_layers) = usize::try_from(num_layers) else {
        return false;
    };

    family == ModelFamily::Llama
        && num_experts == 0
        && num_layers > 0
        && full_attention_layer_mask.len() == num_layers
        && full_attention_layer_mask.iter().all(|&mask| mask == 1)
}

fn arch_uses_batch_attention_buffers(arch: &ArchConfig) -> bool {
    batch_attention_eligible(
        arch.family,
        arch.num_experts,
        arch.num_layers,
        &arch.full_attention_layer_mask,
    )
}

fn batch_attention_layout(
    arch: &ArchConfig,
) -> Result<Option<gpu::BatchAttentionScratchLayout>, RealForwardError> {
    if !arch_uses_batch_attention_buffers(arch) {
        return Ok(None);
    }

    let num_q_heads = usize::try_from(arch.num_heads).map_err(|_| {
        RealForwardError::Unsupported(format!(
            "dense Llama batch attention requires positive query heads, got {}",
            arch.num_heads
        ))
    })?;
    let head_dim = usize::try_from(arch.full_head_dim).map_err(|_| {
        RealForwardError::Unsupported(format!(
            "dense Llama batch attention requires a positive full head dimension, got {}",
            arch.full_head_dim
        ))
    })?;
    let layout = gpu::BatchAttentionScratchLayout::new(num_q_heads, head_dim).map_err(|error| {
        RealForwardError::Unsupported(format!(
            "dense Llama batch-attention scratch layout is unsupported: {error:?}"
        ))
    })?;
    if layout.capacity() != MAX_PREFILL_BATCH {
        return Err(RealForwardError::Unsupported(format!(
            "GPU batch-attention capacity {} does not match runtime prefill capacity {}",
            layout.capacity(),
            MAX_PREFILL_BATCH
        )));
    }

    Ok(Some(layout))
}

impl RealLlamaState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if arch.family == ModelFamily::MiniMaxM2 && arch.tie_word_embeddings {
            return unsupported("MiniMax requires an untied output head".into());
        }

        // Every behavioural extension this architecture does NOT have.
        // Checked rather than assumed, because each one is a manifest field
        // with a GEMMA fallback (AGENTS.md Gotcha 24), so an install that
        // omitted them would arrive here claiming Gemma's answers.
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || (arch.rope_neox_subdim && arch.family != ModelFamily::MiniMaxM2)
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
        // DENSE AND MoE ARE BOTH THIS FLOW, and one `general.architecture`
        // really does cover both: Mistral and Llama 2/3.x report `llama`
        // exactly as the Mixtral MoEs do, and only `expert_count` tells them
        // apart (ROADMAP M2 finding 3, built out in M4).
        //
        // The two are mutually exclusive rather than a spectrum, so a file
        // claiming experts with no `top_k` (or the reverse) is malformed and
        // is refused rather than guessed at.
        let dense = arch.num_experts == 0;
        if dense != (arch.top_k_experts == 0) {
            return unsupported(format!(
                "num_experts {} and top_k_experts {} disagree about whether this install is \
                 dense; both must be zero or both positive",
                arch.num_experts, arch.top_k_experts
            ));
        }
        if arch.num_experts < 0 || arch.top_k_experts < 0 {
            return unsupported(format!(
                "negative expert counts: num_experts {}, top_k_experts {}",
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
        // A DENSE INSTALL'S WORKING SET IS ITS WHOLE WEIGHT FILE, and that is
        // worth saying at open rather than leaving to be discovered from a
        // footprint number. Routed experts are the only thing this engine
        // streams; everything else is mapped AND PINNED (AGENTS.md Gotcha
        // 19), so Gotcha 36's `slots x layers x expert_stride` does not apply
        // and no slot count moves the answer.
        if dense && arch.intermediate_size <= 0 {
            return unsupported(format!(
                "a dense llama install needs a positive ffnIntermediate, got {}",
                arch.intermediate_size
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
        let qk_norm = match arch.family {
            ModelFamily::MiniMaxM2 => QkNorm::Projection,
            // `qwen3_vl` norms q and k per head exactly as dense `qwen3`
            // does (`self_attn.{q,k}_norm.weight`, `[head_dim]`, before
            // RoPE); the tensors sit under the same names this flow's
            // per-head arm already reads.
            ModelFamily::Qwen3Moe | ModelFamily::Qwen3Dense | ModelFamily::Qwen3Vl => {
                QkNorm::PerHead
            }
            _ => QkNorm::None,
        };
        let rms_eps = match arch.family {
            ModelFamily::MiniMaxM2
            | ModelFamily::Qwen2Dense
            | ModelFamily::Qwen3Moe
            | ModelFamily::Qwen3Dense
            | ModelFamily::Qwen3Vl => 1e-6,
            _ => 1e-5,
        };
        let qkv_bias = arch.family == ModelFamily::Qwen2Dense;
        let mut correction_bias = Vec::new();

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
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            // The FFN half is the only thing the two shapes disagree on. A
            // dense layer's three names are exactly what `gguf_names.rs`
            // maps `ffn_gate` / `ffn_up` / `ffn_down` to, and exactly what
            // the Gemma flow's shared-expert branch already reads.
            let ffn: &[&str] = if dense {
                &[
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ]
            } else {
                &["mlp.gate.weight"]
            };
            for suffix in ffn {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            if qkv_bias {
                for suffix in [
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ] {
                    entry(index, &layer_tensor(layer, suffix))?;
                }
            }
            if qk_norm != QkNorm::None {
                for (suffix, heads) in [
                    ("self_attn.q_norm.weight", arch.num_heads),
                    ("self_attn.k_norm.weight", arch.num_full_kv_heads),
                ] {
                    let width = head_dim as usize
                        * if qk_norm == QkNorm::Projection {
                            heads as usize
                        } else {
                            1
                        };
                    crate::real_forward_utils::norm_view(
                        weights,
                        index,
                        &layer_tensor(layer, suffix),
                        width,
                    )?;
                }
            }
        }
        if arch.family == ModelFamily::MiniMaxM2 {
            if dense || arch.router_scoring_func != "sigmoid" || !arch.rope_neox_subdim {
                return unsupported("MiniMax requires sigmoid MoE and subdimension RoPE".into());
            }
            for layer in 0..arch.num_layers as usize {
                for (tail, count) in [
                    ("mlp.gate.weight", hidden * num_experts),
                    ("mlp.e_score_correction_bias", num_experts),
                ] {
                    let name = layer_tensor(layer, tail);
                    let e = entry(index, &name)?;
                    if e.dtype != 3 || e.size_bytes as usize != count * 4 {
                        return unsupported(format!("{name}: expected FP32 with {count} elements"));
                    }
                }
                let e = entry(index, &layer_tensor(layer, "mlp.e_score_correction_bias"))?;
                let off = weights.gpu_offset(e.file_offset - index.header.index_size) as usize;
                let bias = gpu::read_f32_buffer_at(weights.buffer(), off / 4, num_experts);
                if bias.iter().any(|v| !v.is_finite()) {
                    return unsupported("non-finite MiniMax correction bias".into());
                }
                correction_bias.push(bias);
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

        let batch_attention_buffers = batch_attention_layout(arch)?
            .map(|layout| BatchAttentionBuffers::allocate(context, layout))
            .transpose()?;

        let ones: Vec<u8> = (0..hidden).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            rotated_pairs: rotated_pairs as u32,
            qk_norm,
            qkv_bias,
            rms_eps,
            router_ones,
            correction_bias,
            dense,
            batch_attention_buffers,
            per_expert_ones: vec![1.0; num_experts],
            // `new_output_buffer(0)` is not a thing worth finding out about
            // at the first dispatch, and a dense flow never binds either of
            // these past row 0.
            router_logits_f32: context
                .new_output_buffer((num_experts.max(1) * 4 * MAX_PREFILL_BATCH) as u64),
            moe_x: halfs(hidden * MAX_PREFILL_BATCH),
            h2: halfs(hidden),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_attention_buffer_gate_requires_dense_full_attention_llama() {
        assert!(batch_attention_eligible(
            ModelFamily::Llama,
            0,
            3,
            &[1, 1, 1]
        ));

        // Mixtral shares the Llama architecture string but has routed experts.
        assert!(!batch_attention_eligible(
            ModelFamily::Llama,
            8,
            3,
            &[1, 1, 1]
        ));
        // Dense families sharing RealLlamaState keep their existing scratch.
        for family in [
            ModelFamily::Qwen3Moe,
            ModelFamily::Qwen2Dense,
            ModelFamily::Qwen3Dense,
            ModelFamily::Qwen3Vl,
        ] {
            assert!(!batch_attention_eligible(family, 0, 3, &[1, 1, 1]));
        }
        assert!(!batch_attention_eligible(
            ModelFamily::Llama,
            0,
            3,
            &[1, 0, 1]
        ));
        assert!(!batch_attention_eligible(ModelFamily::Llama, 0, 3, &[1, 1]));
        assert!(!batch_attention_eligible(ModelFamily::Llama, 0, 0, &[]));
    }

    #[test]
    fn batch_attention_buffers_match_shared_layout_byte_lengths() {
        let context = gpu::MetalContext::new().expect("Metal device");
        let layout = gpu::BatchAttentionScratchLayout::new(32, 128)
            .expect("supported Llama attention shape");
        assert_eq!(layout.capacity(), MAX_PREFILL_BATCH);

        let buffers = BatchAttentionBuffers::allocate(&context, layout)
            .expect("allocate layout-sized batch buffers");
        assert_eq!(buffers.scratch.layout(), layout);
        assert_eq!(buffers.q.length(), layout.q_buffer_bytes() as u64);
        assert_eq!(buffers.output.length(), layout.output_buffer_bytes() as u64);
        assert_eq!(
            buffers.scratch.row_plan.length(),
            layout.row_plan_bytes() as u64
        );
        let partial_bytes =
            buffers.scratch.m.length() + buffers.scratch.d.length() + buffers.scratch.o.length();
        assert_eq!(partial_bytes, layout.partial_state_bytes() as u64);
        assert_eq!(
            buffers.q.length()
                + buffers.output.length()
                + buffers.scratch.row_plan.length()
                + partial_bytes,
            layout.total_bytes() as u64
        );
    }
}
