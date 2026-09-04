//! `manifest.arch` field-by-field validation against a resolved
//! `ArchConfig`. Split out of `manifest.rs` to keep that file's decode/load
//! logic separate from this mechanical field-by-field check list.

use crate::arch_config::{ArchConfig, ModelFamily};
use crate::error::ModelError;
use crate::manifest::ManifestArch;

pub(crate) fn validate_arch(a: &ManifestArch, e: &ArchConfig) -> Result<(), ModelError> {
    macro_rules! check {
        ($field:literal, $actual:expr, $expected:expr) => {
            if $actual != $expected {
                return Err(ModelError::ArchMismatch {
                    field: $field.to_string(),
                    expected: format!("{:?}", $expected),
                    actual: format!("{:?}", $actual),
                });
            }
        };
    }
    check!("hiddenSize", a.hidden_size, e.hidden_size);
    check!("ffnIntermediate", a.ffn_intermediate, e.intermediate_size);
    check!(
        "moeIntermediateSize",
        a.moe_intermediate_size,
        e.moe_intermediate_size
    );
    check!("numHeads", a.num_heads, e.num_heads);
    check!("numKVHeads", a.num_kv_heads, e.num_kv_heads);
    check!("numFullKVHeads", a.num_full_kv_heads, e.num_full_kv_heads);
    check!("headDim", a.head_dim, e.head_dim);
    check!("fullHeadDim", a.full_head_dim, e.full_head_dim);
    check!("vocabSize", a.vocab_size, e.vocab_size);
    check!("slidingWindow", a.sliding_window, e.sliding_window);
    check!(
        "finalLogitSoftcap",
        a.final_logit_softcap,
        e.final_logit_softcap
    );
    check!("ropeTheta", a.rope_theta, e.rope_theta);
    check!("fullRopeTheta", a.full_rope_theta, e.full_rope_theta);
    check!(
        "partialRotaryFactor",
        a.partial_rotary_factor,
        e.partial_rotary_factor
    );
    check!("numLayers", a.num_layers, e.num_layers);
    check!("numExperts", a.num_experts, e.num_experts);
    check!("topKExperts", a.top_k_experts, e.top_k_experts);
    check!(
        "tieWordEmbeddings",
        a.tie_word_embeddings,
        e.tie_word_embeddings
    );
    check!("attentionKEqV", a.attention_k_eq_v, e.attention_k_eq_v);
    check!("hiddenActivation", a.hidden_activation, e.hidden_activation);
    let actual_mask: Vec<u8> = a
        .full_attention_layer_mask
        .iter()
        .map(|&v| v as u8)
        .collect();
    check!(
        "fullAttentionLayerMask",
        actual_mask,
        e.full_attention_layer_mask
    );

    // Family extensions: absent fields mean the Gemma defaults.
    let gemma_defaults = crate::arch_baselines::gemma4_26b_a4b();
    check!(
        "family",
        a.family
            .clone()
            .unwrap_or_else(|| ModelFamily::Gemma4.as_str().to_string()),
        e.family.as_str().to_string()
    );
    check!(
        "attnOutputGate",
        a.attn_output_gate
            .unwrap_or(gemma_defaults.attn_output_gate),
        e.attn_output_gate
    );
    check!(
        "attentionScale",
        a.attention_scale.unwrap_or(gemma_defaults.attention_scale),
        e.attention_scale
    );
    check!(
        "embeddingScaledBySqrtHidden",
        a.embedding_scaled_by_sqrt_hidden
            .unwrap_or(gemma_defaults.embedding_scaled_by_sqrt_hidden),
        e.embedding_scaled_by_sqrt_hidden
    );
    check!(
        "routerScaled",
        a.router_scaled.unwrap_or(gemma_defaults.router_scaled),
        e.router_scaled
    );
    check!(
        "ffnSandwichNorms",
        a.ffn_sandwich_norms
            .unwrap_or(gemma_defaults.ffn_sandwich_norms),
        e.ffn_sandwich_norms
    );
    check!(
        "sharedExpertGated",
        a.shared_expert_gated
            .unwrap_or(gemma_defaults.shared_expert_gated),
        e.shared_expert_gated
    );
    check!(
        "ropeNeoxSubdim",
        a.rope_neox_subdim
            .unwrap_or(gemma_defaults.rope_neox_subdim),
        e.rope_neox_subdim
    );
    check!(
        "linearNumKHeads",
        a.linear_num_k_heads.unwrap_or(0),
        e.linear_attention.num_k_heads
    );
    check!(
        "linearNumVHeads",
        a.linear_num_v_heads.unwrap_or(0),
        e.linear_attention.num_v_heads
    );
    check!(
        "linearKeyHeadDim",
        a.linear_key_head_dim.unwrap_or(0),
        e.linear_attention.key_head_dim
    );
    check!(
        "linearValueHeadDim",
        a.linear_value_head_dim.unwrap_or(0),
        e.linear_attention.value_head_dim
    );
    check!(
        "linearConvKernelSize",
        a.linear_conv_kernel_size.unwrap_or(0),
        e.linear_attention.conv_kernel_size
    );
    // Absent means SILU, which is every pre-`qwen4_exp` family's value and
    // what the kernel did unconditionally, so no install on disk moves.
    check!(
        "linearOutputGateSigmoid",
        a.linear_output_gate_sigmoid.unwrap_or(false),
        e.linear_attention.output_gate_sigmoid
    );
    check!(
        "caQLoraRank",
        a.ca_q_lora_rank.unwrap_or(0),
        e.compressed_attention.q_lora_rank
    );
    check!(
        "caOLoraRank",
        a.ca_o_lora_rank.unwrap_or(0),
        e.compressed_attention.o_lora_rank
    );
    check!(
        "caOGroups",
        a.ca_o_groups.unwrap_or(0),
        e.compressed_attention.o_groups
    );
    check!(
        "caRopeHeadDim",
        a.ca_rope_head_dim.unwrap_or(0),
        e.compressed_attention.rope_head_dim
    );
    check!(
        "caIndexNHeads",
        a.ca_index_n_heads.unwrap_or(0),
        e.compressed_attention.index_n_heads
    );
    check!(
        "caIndexHeadDim",
        a.ca_index_head_dim.unwrap_or(0),
        e.compressed_attention.index_head_dim
    );
    check!(
        "caIndexTopK",
        a.ca_index_top_k.unwrap_or(0),
        e.compressed_attention.index_top_k
    );
    check!(
        "caCSACompressRate",
        a.ca_csa_compress_rate.unwrap_or(0),
        e.compressed_attention.csa_compress_rate
    );
    check!(
        "caHCACompressRate",
        a.ca_hca_compress_rate.unwrap_or(0),
        e.compressed_attention.hca_compress_rate
    );
    check!(
        "caCompressRopeTheta",
        a.ca_compress_rope_theta.unwrap_or(0.0),
        e.compressed_attention.compress_rope_theta
    );
    check!(
        "caRopeScalingFactor",
        a.ca_rope_scaling_factor.unwrap_or(0.0),
        e.compressed_attention.rope_scaling_factor
    );
    check!(
        "caRopeScalingOriginalMax",
        a.ca_rope_scaling_original_max.unwrap_or(0),
        e.compressed_attention.rope_scaling_original_max
    );
    check!(
        "caRopeScalingBetaFast",
        a.ca_rope_scaling_beta_fast.unwrap_or(0.0),
        e.compressed_attention.rope_scaling_beta_fast
    );
    check!(
        "caRopeScalingBetaSlow",
        a.ca_rope_scaling_beta_slow.unwrap_or(0.0),
        e.compressed_attention.rope_scaling_beta_slow
    );
    check!(
        "caIndexKvHeads",
        a.ca_index_kv_heads.unwrap_or(0),
        e.compressed_attention.index_kv_heads
    );
    check!(
        "caIndexBudget",
        a.ca_index_budget.unwrap_or(0),
        e.compressed_attention.index_budget
    );
    check!("hcMult", a.hc_mult.unwrap_or(0), e.hyper_connections.mult);
    check!(
        "hcSinkhornIters",
        a.hc_sinkhorn_iters.unwrap_or(0),
        e.hyper_connections.sinkhorn_iters
    );
    check!("hcEps", a.hc_eps.unwrap_or(0.0), e.hyper_connections.eps);
    check!(
        "hcLowrank",
        a.hc_lowrank.unwrap_or(0),
        e.hyper_connections.lowrank
    );
    // The n-gram PLE table (`qwen4_exp`). Every field falls back to
    // `PleConfig::NONE`'s zero and NOT to `gemma_defaults.ple`, for the reason
    // the vision block below states: an absent field means the install
    // declares no table, and another family's answer about ITS table is not
    // evidence (AGENTS.md Gotcha 24).
    check!(
        "pleNgramSize",
        a.ple_ngram_size.unwrap_or(0),
        e.ple.ngram_size
    );
    check!(
        "pleHeadsPerNgram",
        a.ple_heads_per_ngram.unwrap_or(0),
        e.ple.heads_per_ngram
    );
    check!(
        "pleNgramVocabSizeBase",
        a.ple_ngram_vocab_size_base.unwrap_or(0),
        e.ple.ngram_vocab_size_base
    );
    check!(
        "pleMakeDivisibleBy",
        a.ple_make_divisible_by.unwrap_or(0),
        e.ple.make_divisible_by
    );
    check!(
        "pleSplitNgramParts",
        a.ple_split_ngram_parts.unwrap_or(0),
        e.ple.split_ngram_parts
    );
    check!(
        "pleEmbedDim",
        a.ple_embed_dim.unwrap_or(0),
        e.ple.ple_embed_dim
    );
    check!(
        "pleConvKernelSize",
        a.ple_conv_kernel_size.unwrap_or(0),
        e.ple.conv_kernel_size
    );
    check!("pleSeed", a.ple_seed.unwrap_or(0), e.ple.seed);
    check!(
        "pleEosTokenId",
        a.ple_eos_token_id.unwrap_or(0),
        e.ple.eos_token_id
    );
    // The id LIST, compared whole. An absent key is an empty list, which is
    // what `PleConfig::NONE` carries -- so a length check alone would pass an
    // install that declares the right NUMBER of PLE layers at the wrong
    // indices, which puts the block on a layer the checkpoint never trained.
    check!(
        "pleLayerIds",
        a.ple_layer_ids.clone().unwrap_or_default(),
        e.ple.layer_ids
    );
    check!(
        "numHashRoutedLayers",
        a.num_hash_routed_layers.unwrap_or(0),
        e.num_hash_routed_layers
    );
    check!(
        "routerScoringFunc",
        a.router_scoring_func
            .clone()
            .unwrap_or_else(|| "softmax".to_string()),
        e.router_scoring_func
    );
    check!(
        "routedScalingFactor",
        a.routed_scaling_factor.unwrap_or(1.0),
        e.routed_scaling_factor
    );
    check!("swigluLimit", a.swiglu_limit.unwrap_or(0.0), e.swiglu_limit);
    // ROADMAP M5. Each falls back to the "no scaling" value rather than to a
    // neighbouring family's, which is AGENTS.md Gotcha 39's rule: a default
    // is a claim about what SILENCE means, and silence here means the file
    // declares no rope scaling. gpt-oss's four are 32.0 / 4096 / 32.0 / 1.0,
    // all binary fractions, so Gotcha 24's f64 round-trip is not in play.
    check!(
        "ropeScalingFactor",
        a.rope_scaling_factor.unwrap_or(0.0),
        e.rope_scaling.factor
    );
    check!(
        "ropeScalingOriginalContext",
        a.rope_scaling_original_context.unwrap_or(0),
        e.rope_scaling.original_context
    );
    check!(
        "ropeScalingBetaFast",
        a.rope_scaling_beta_fast.unwrap_or(0.0),
        e.rope_scaling.beta_fast
    );
    check!(
        "ropeScalingBetaSlow",
        a.rope_scaling_beta_slow.unwrap_or(0.0),
        e.rope_scaling.beta_slow
    );
    // ROADMAP M-V3, the vision tower. Every field falls back to
    // `VisionConfig::NONE`'s zero and NOT to `gemma_defaults.vision`, which is
    // the same distinction the rope-scaling block above draws: silence here
    // means the install declares no tower, and Gemma's answer to a question
    // about a Qwen tower is not evidence (AGENTS.md Gotchas 24 and 39). The
    // two happen to agree today -- Gemma's baseline also carries `NONE` -- so
    // this is a case where writing the fallback correctly costs nothing now
    // and is what keeps the next multimodal baseline from silently validating
    // every other family's installs against ITS tower.
    check!("visionDepth", a.vision_depth.unwrap_or(0), e.vision.depth);
    check!(
        "visionHiddenSize",
        a.vision_hidden_size.unwrap_or(0),
        e.vision.hidden_size
    );
    check!(
        "visionIntermediateSize",
        a.vision_intermediate_size.unwrap_or(0),
        e.vision.intermediate_size
    );
    check!(
        "visionNumHeads",
        a.vision_num_heads.unwrap_or(0),
        e.vision.num_heads
    );
    check!(
        "visionPatchSize",
        a.vision_patch_size.unwrap_or(0),
        e.vision.patch_size
    );
    check!(
        "visionTemporalPatchSize",
        a.vision_temporal_patch_size.unwrap_or(0),
        e.vision.temporal_patch_size
    );
    check!(
        "visionInChannels",
        a.vision_in_channels.unwrap_or(0),
        e.vision.in_channels
    );
    check!(
        "visionSpatialMergeSize",
        a.vision_spatial_merge_size.unwrap_or(0),
        e.vision.spatial_merge_size
    );
    check!(
        "visionNumPositionEmbeddings",
        a.vision_num_position_embeddings.unwrap_or(0),
        e.vision.num_position_embeddings
    );
    check!(
        "visionOutHiddenSize",
        a.vision_out_hidden_size.unwrap_or(0),
        e.vision.out_hidden_size
    );
    check!(
        "visionMropeSection",
        a.vision_mrope_section.unwrap_or([0, 0, 0]),
        e.vision.mrope_section
    );
    check!(
        "visionStartTokenId",
        a.vision_start_token_id.unwrap_or(0),
        e.vision.vision_start_token_id
    );
    check!(
        "visionEndTokenId",
        a.vision_end_token_id.unwrap_or(0),
        e.vision.vision_end_token_id
    );
    check!(
        "visionImageTokenId",
        a.vision_image_token_id.unwrap_or(0),
        e.vision.image_token_id
    );
    check!(
        "visionVideoTokenId",
        a.vision_video_token_id.unwrap_or(0),
        e.vision.video_token_id
    );
    Ok(())
}
