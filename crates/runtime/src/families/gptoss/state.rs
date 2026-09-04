//! `RealGptOssState`: everything [`crate::real_forward::RealForwardRunner`]
//! allocates once when it opens a `gpt-oss` install (ROADMAP M5).
//!
//! Bigger than its `llama` sibling by exactly the four things that make this a
//! fifth flow: a YaRN frequency TABLE (built here rather than per token,
//! because it is position-independent), the per-layer router BIAS pulled to
//! the host (the top-k it feeds is host-side already), and open-time proof
//! that every bias and every sink tensor is present. The clamped SwiGLU needs
//! nothing here -- it rides inside the MXFP4 routed pair as a uniform.

use model_io::{ArchConfig, ModelFamily, ResidentIndex};

use crate::families::gptoss::layer_tensor;
use crate::real_forward::RealForwardError;
use crate::real_forward_types::MAX_PREFILL_BATCH;
use crate::real_forward_utils::entry;

/// BF16 bit pattern for 1.0.
const BF16_ONE: u16 = 0x3F80;

pub(crate) struct RealGptOssState {
    /// Rotated PAIRS per head. `partial_rotary_factor` is 1.0, so this is
    /// `head_dim / 2` -- the whole head, NeoX-style.
    pub(crate) rotated_pairs: u32,
    /// RMS epsilon. 1e-5, from `attention.layer_norm_rms_epsilon`, and not an
    /// `ArchConfig` field, so the family carries it exactly as
    /// `RealLlamaState` does.
    pub(crate) rms_eps: f32,
    /// YaRN's per-PAIR frequency table, `head_dim / 2` f32 entries, uploaded
    /// once.
    ///
    /// It is built at open rather than per token because it is a function of
    /// the ROPE CONFIG ALONE -- base, factor, the two betas and the original
    /// context -- and the position multiplies it inside the kernel. The three
    /// older rope kernels here derive every frequency from a scalar theta,
    /// which is precisely what YaRN's ramp between two correction dimensions
    /// cannot be written as, and is why `rope_neox_freqs` takes a table.
    pub(crate) rope_frequencies: gpu::MetalBuffer,
    /// Multiplies both cos and sin, i.e. scales q and k. **1.3465736, NOT
    /// 1.0.** `llama-context.cpp` computes `yarn_attn_factor = get_mscale(32,
    /// 1)` and then divides it by the same quantity, which reads as a
    /// cancellation and is there to stop `rope_yarn` applying it twice. Net,
    /// q and k are scaled once and `q.k` by 1.8133; dropping it leaves a model
    /// that is finite, fluent and wrong.
    pub(crate) rope_mscale: f32,
    /// Per layer, the router's `[num_experts]` bias, on the HOST.
    ///
    /// llama.cpp adds it to the logits BEFORE the top-k, and this port's top-k
    /// is already a host-side function of a read-back f32 buffer, so the add
    /// lands there and costs no kernel and no dispatch. Doing it on the GPU
    /// would need a bias-add over an F32 buffer, which `bias_add_bf16_fp16`
    /// is not -- it writes FP16.
    pub(crate) router_bias: Vec<Vec<f32>>,
    /// BF16 `[hidden]` of ones: the INT8 router GEMV scales `x[n]` per
    /// element and this architecture has no `router.scale`.
    pub(crate) router_ones: gpu::MetalBuffer,
    /// `[num_experts]` of ones, so `router_topk_gemma4`'s per-expert weighting
    /// is a no-op and selection reduces to a softmax over the selected --
    /// llama.cpp's `SOFTMAX_WEIGHT` with `norm_w = false`, exactly.
    pub(crate) per_expert_ones: Vec<f32>,
    /// FP32 router logits, one `[num_experts]` row per token of a prefill
    /// micro-batch (`MAX_PREFILL_BATCH` rows, matching
    /// `RealGemmaState`/`RealLlamaState`'s field of the same name): the
    /// chunked-prefill driver's per-layer command buffer writes all M
    /// tokens' router GEMVs before the host reads any of them back (and
    /// adds the bias). The sequential decode path always uses row 0.
    pub(crate) router_logits_f32: gpu::MetalBuffer,
    /// `[hidden]` per token of a prefill micro-batch: the post-attention
    /// norm feeding router and routed experts. There is no shared expert to
    /// feed. Written in the attention half of a layer and read in the
    /// routed half, with a commit between them in the chunked driver, so it
    /// needs a row per token for the same reason `RealLlamaState::moe_x`
    /// does (`crates/runtime/CLAUDE.md` Gotcha 14).
    pub(crate) moe_x: gpu::MetalBuffer,
    /// `[hidden]`: the routed sum, added back to the stream.
    pub(crate) h2: gpu::MetalBuffer,
    /// The batched routed half's scratch (`docs/BATCHED_PREFILL.md` step 5),
    /// allocated on first use by [`Self::ensure_batched`] and `None` on every
    /// run that never chunks its prefill with `TURBOSPARK_ROUTED_BATCH` set.
    pub(crate) batched: Option<BatchedRoutedScratch>,
}

/// Batched routed-expert scratch for `gpt-oss`, all sized for
/// [`MAX_PREFILL_BATCH`] rows and reached only from the chunk driver's
/// batched routed half.
///
/// It is `RealGemmaState`'s `BatchedPrefillScratch` MINUS two groups, and
/// both absences are this family's rather than an omission. There is no
/// `batch_h1`, because there is no shared expert to hold M rows of output
/// for -- phase 2's seed is `batch_zero` and the routed sum reaches the
/// stream through one raw residual add. And there are no batched-GEMV
/// fields (`batch_normed`, `batch_q`, ...), because `TURBOSPARK_BATCHED_GEMV`
/// is not wired for this family: that seam is INT4-affine only, and this
/// family's resident tensors are Q8_0.
///
/// **ALLOCATED ON FIRST USE**, per the rule `families/gemma4/state.rs`
/// states at length: an unasked-for feature must allocate nothing, so a
/// frozen memory-oracle row keeps describing the engine that shipped before
/// the feature existed. On the real 20B install (hidden 2880, moe_inter
/// 2880, top_k 4, `MAX_PREFILL_BATCH` 16) these five come to ~1.1 MiB,
/// against a 5,700 MiB ceiling.
pub(crate) struct BatchedRoutedScratch {
    /// `[M * top_k, moe_inter]` FP16, written by the route-list phase 1.
    pub(crate) batch_acts: gpu::MetalBuffer,
    /// `[M, hidden]` FP16, the fused phase-2 output the per-token residual
    /// add reads.
    pub(crate) batch_y: gpu::MetalBuffer,
    /// `[M * top_k]` FP16, one HOST write per sub-batch.
    pub(crate) batch_routing_w: gpu::MetalBuffer,
    /// The encoded route list, likewise host-written. 16 bytes per route,
    /// matching `MoePrefillRoute::bytes` and the shader struct.
    pub(crate) batch_routes: gpu::MetalBuffer,
    /// `[M, hidden]` FP16 of ZEROS, written once here and never again: the
    /// fused phase 2's accumulator seed, the batched twin of
    /// `DecodeScratch::zero_hidden`. Filled explicitly rather than trusting
    /// a fresh `MTLBuffer` to be zeroed, which is `zero_hidden`'s own
    /// precedent.
    pub(crate) batch_zero: gpu::MetalBuffer,
    /// The 32-pointer argument buffer the batched pair reads, encoded by
    /// the MXFP4 phase-1 function out of that pair's OWN shader library --
    /// never the affine pair's, though the two declare `RoutedBlobsWide`
    /// identically (`crates/gpu`'s `new_for`).
    pub(crate) wide_blobs: gpu::RoutedBlobsWideBuffer,
}

impl RealGptOssState {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));

        if arch.family != ModelFamily::GptOss {
            return unsupported(format!(
                "the gpt-oss flow refuses family {:?}; flow selection keys on the family, never \
                 on tensor naming",
                arch.family
            ));
        }
        // Every behavioural extension this architecture does NOT have, checked
        // rather than assumed: each is a manifest field with a GEMMA fallback
        // (AGENTS.md Gotcha 24), so an install that omitted them would arrive
        // here claiming Gemma's answers.
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.rope_neox_subdim
            || arch.attention_k_eq_v
        {
            return unsupported(
                "the gpt-oss flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 ropeNeoxSubdim / attentionKEqV"
                    .to_string(),
            );
        }
        if arch.final_logit_softcap != 0.0 {
            return unsupported("gpt-oss has no final logit softcap".to_string());
        }
        if arch.tie_word_embeddings {
            return unsupported(
                "gpt-oss ships an untied `output.weight`; an install claiming tied embeddings \
                 was built from something else"
                    .to_string(),
            );
        }
        // MoE ONLY. Unlike `llama`, this architecture string has no dense
        // half, so a zero expert count is a malformed install rather than a
        // second shape to serve.
        if arch.num_experts <= 0 || arch.top_k_experts <= 0 {
            return unsupported(format!(
                "gpt-oss is MoE-only; got num_experts {} and top_k_experts {}",
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
        // THE WINDOW MUST ALTERNATE, and this is worth refusing rather than
        // tolerating: a mask of all 1s decodes fine and is a DIFFERENT MODEL
        // past 128 tokens, which no short smoke reaches. Layer masks here are
        // 0 = sliding, 1 = full; llama.cpp's period-2 default makes the EVEN
        // layers slide.
        if arch.full_attention_layer_mask.iter().any(|&m| m > 1) {
            return unsupported(
                "gpt-oss layers are sliding-window (0) or full (1); no linear or compressed \
                 layer kinds exist in this architecture"
                    .to_string(),
            );
        }
        if arch.sliding_window <= 0 {
            return unsupported(format!(
                "gpt-oss alternates a sliding window; {} is not a usable one",
                arch.sliding_window
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
        if arch.rope_scaling.factor <= 0.0 {
            return unsupported(
                "gpt-oss declares YaRN rope scaling; an install with no `ropeScaling` block \
                 would decode with unscaled frequencies, which is fluent and wrong past the \
                 original context"
                    .to_string(),
            );
        }

        // THE ROPE BASE IS THE FULL-ATTENTION ONE FOR BOTH HALVES. The file
        // publishes `rope.freq_base` and no `freq_base_swa`, unlike Gemma
        // whose two bases are the reason `ArchConfig` carries the pair.
        let yarn = compute::yarn_frequencies(
            head_dim as usize,
            arch.full_rope_theta as f32,
            arch.rope_scaling.factor as f32,
            arch.rope_scaling.original_context,
            arch.rope_scaling.beta_fast as f32,
            arch.rope_scaling.beta_slow as f32,
        );
        let freq_bytes: Vec<u8> = yarn
            .frequencies
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let rope_frequencies = context.new_output_buffer(freq_bytes.len().max(4) as u64);
        gpu::write_buffer_bytes(&rope_frequencies, 0, &freq_bytes);

        // Fail at open, not at token 1. EVERY projection has a bias here,
        // which no other family in this port does, so a name table that
        // dropped one would otherwise surface as a missing entry mid-decode.
        let hidden = arch.hidden_size as usize;
        let num_experts = arch.num_experts as usize;
        let mut router_bias = Vec::with_capacity(arch.num_layers as usize);
        for layer in 0..arch.num_layers as usize {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.bias",
                "self_attn.sinks.weight",
                "mlp.gate.weight",
                "mlp.gate.bias",
            ] {
                entry(index, &layer_tensor(layer, suffix))?;
            }
            // ONE SINK PER QUERY HEAD, not per KV head. The two differ by 8x
            // on the real 20b, and a short sink vector reads whatever follows
            // it in the resident mapping as a logit -- finite, and wrong in
            // every softmax denominator.
            let sinks = entry(index, &layer_tensor(layer, "self_attn.sinks.weight"))?;
            let want = arch.num_heads as usize * 2;
            if sinks.size_bytes as usize != want {
                return unsupported(format!(
                    "layer {layer} attention sinks are {} bytes; expected {want} (one BF16 \
                     logit per QUERY head, {} of them)",
                    sinks.size_bytes, arch.num_heads
                ));
            }
            router_bias.push(read_bias(weights, index, layer, num_experts)?);
        }
        entry(index, "language_model.lm_head.weight")?;
        entry(index, "language_model.model.norm.weight")?;
        entry(index, "language_model.model.embed_tokens.weight")?;

        let ones: Vec<u8> = (0..hidden).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let router_ones = context.new_output_buffer(ones.len() as u64);
        gpu::write_buffer_bytes(&router_ones, 0, &ones);

        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            rotated_pairs: rotated_pairs as u32,
            rms_eps: 1e-5,
            rope_frequencies,
            rope_mscale: yarn.mscale,
            router_bias,
            router_ones,
            per_expert_ones: vec![1.0; num_experts],
            router_logits_f32: context
                .new_output_buffer((num_experts * 4 * MAX_PREFILL_BATCH) as u64),
            moe_x: halfs(hidden * MAX_PREFILL_BATCH),
            h2: halfs(hidden),
            batched: None,
        })
    }

    /// Allocates [`BatchedRoutedScratch`] the first time the chunk driver's
    /// batched routed half asks for it, and is a no-op afterwards.
    ///
    /// Idempotent rather than "call once", because the entry point that calls
    /// it is reached once per CHUNK, not once per run.
    pub(crate) fn ensure_batched(
        &mut self,
        context: &mut gpu::MetalContext,
        arch: &ArchConfig,
    ) -> Result<(), RealForwardError> {
        if self.batched.is_some() {
            return Ok(());
        }
        let top_k = arch.top_k_experts as usize;
        let hidden = arch.hidden_size as usize;
        let moe_inter = arch.moe_intermediate_size.max(1) as usize;
        // The silu specialization is a constant inside the MXFP4 batched
        // module itself, so the argument-buffer encoder and the dispatches
        // cannot disagree on the pipeline-cache key.
        let wide_blobs =
            gpu::new_routed_blobs_wide_mxfp4(context).map_err(RealForwardError::Gpu)?;
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        let batch_rows = |per_token: usize| halfs(MAX_PREFILL_BATCH * per_token);
        self.batched = Some(BatchedRoutedScratch {
            batch_acts: batch_rows(top_k * moe_inter),
            batch_y: batch_rows(hidden),
            batch_routing_w: batch_rows(top_k),
            batch_routes: context.new_output_buffer(
                (MAX_PREFILL_BATCH * top_k * gpu::MoePrefillRoute::STRIDE_BYTES) as u64,
            ),
            batch_zero: {
                let bytes = MAX_PREFILL_BATCH * hidden.max(1) * 2;
                let buffer = context.new_output_buffer(bytes as u64);
                gpu::write_buffer_bytes(&buffer, 0, &vec![0u8; bytes]);
                buffer
            },
            wide_blobs,
        });
        Ok(())
    }

    /// The batched scratch, reached only from inside the chunk driver's
    /// batched routed half -- downstream of the `ensure_batched` at its entry
    /// point, so the `expect` is a structural invariant rather than a hope.
    pub(crate) fn batched(&self) -> &BatchedRoutedScratch {
        self.batched
            .as_ref()
            .expect("prefill_chunk_real_gpt_oss calls ensure_batched before any layer runs")
    }
}

/// One layer's router bias, narrowed to BF16 by the repack transcode and read
/// back to f32 here.
fn read_bias(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    layer: usize,
    num_experts: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let name = layer_tensor(layer, "mlp.gate.bias");
    let bias = crate::real_forward_utils::read_bf16_host(weights, index, &name)?;
    if bias.len() != num_experts {
        return Err(RealForwardError::Unsupported(format!(
            "{name}: expected {num_experts} values, got {}",
            bias.len()
        )));
    }
    Ok(bias)
}

#[cfg(test)]
mod tests {
    /// The YaRN scalars this state feeds `yarn_frequencies`, in the order it
    /// feeds them.
    ///
    /// WHAT THIS PINS THAT THE INTEGRATION TESTS CANNOT. A synthetic install
    /// can show that the frequency TABLE reaches the rope kernel -- change
    /// the declared factor and the logits move -- but it cannot isolate
    /// `mscale`, because `mscale` is a function of the factor alone
    /// (`1 + 0.1 * ln(factor)`) and so every perturbation that moves one
    /// moves the other. Mutating `attn.rs` to pass a literal `1.0` therefore
    /// leaves all six of those tests green, which was measured rather than
    /// assumed.
    ///
    /// That matters because 1.0 is exactly the wrong answer a reader arrives
    /// at from llama.cpp's source: `llama-context.cpp` computes
    /// `yarn_attn_factor = get_mscale(32, 1)` and immediately DIVIDES it by
    /// the same quantity, which reads as a cancellation. It is not one --
    /// `rope_yarn` re-applies the factor internally, so the division exists
    /// to stop it being applied twice. Net, q and k are scaled by 1.3465736
    /// once and `q.k` by 1.8133, and dropping it leaves a model that is
    /// finite, fluent and wrong.
    ///
    /// So this pins the VALUE and the argument ORDER (a swapped beta pair or
    /// `rope_theta` for `full_rope_theta` is the other silent way to get here),
    /// and the end-to-end "the flow passes what it computed" property is left
    /// to the real-model quality gate, which is the instrument that can see a
    /// fluent wrong model.
    #[test]
    fn the_gpt_oss_yarn_scalars_produce_llama_cpps_mscale() {
        let arch = model_io::known_architecture(model_io::ModelFamily::GptOss);
        let yarn = compute::yarn_frequencies(
            arch.full_head_dim as usize,
            arch.full_rope_theta as f32,
            arch.rope_scaling.factor as f32,
            arch.rope_scaling.original_context,
            arch.rope_scaling.beta_fast as f32,
            arch.rope_scaling.beta_slow as f32,
        );
        assert!(
            (yarn.mscale - 1.346_573_6).abs() < 1e-6,
            "mscale is {}, expected 1.3465736; 1.0 is the value llama.cpp's apparent \
             cancellation invites and is wrong",
            yarn.mscale
        );
        assert_eq!(yarn.frequencies.len(), arch.full_head_dim as usize / 2);
        // The ramp really does run between two correction dimensions rather
        // than scaling everything: the fast end is left at its extrapolated
        // frequency and the slow end is divided by the factor. A swapped beta
        // pair inverts this.
        let first = yarn.frequencies[0];
        let last = yarn.frequencies[yarn.frequencies.len() - 1];
        assert!(
            first > last,
            "frequencies must fall across the head ({first} then {last})"
        );
        assert!(
            (first - 1.0).abs() < 1e-6,
            "pair 0 is the fastest dimension and YaRN leaves it unscaled, got {first}"
        );
    }
}
