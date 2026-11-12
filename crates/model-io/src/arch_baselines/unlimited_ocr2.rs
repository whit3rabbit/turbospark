use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Canonical `unlimited_ocr2` baseline: `mlx-community/Unlimited-OCR-2-4bit`'s
/// `config.json`, the checkpoint this family was brought up against
/// (`docs/UNLIMITED_OCR2_PHASE0.md`).
///
/// Two widths are worth a second read. `intermediate_size` is the fused shared
/// expert, `2 x 896 = 1792`, as the tensor's own `[1792, ...]` gate says. The
/// dense lead layer's 6848, which `config.json` calls `intermediate_size`,
/// travels in [`ArchConfig::dense_lead_intermediate_size`].
///
/// `rope_scaling` only serves a prompt with no page marker, which falls back
/// to upstream's 1-D yarn rope. An image prompt rotates by page2d angles from
/// `compute::dash`, which neither this table nor `rope_theta` describes.
///
/// `attention_scale` is `head_dim^-0.5`. `attention_factor` is 1.0, so yarn
/// contributes no magnitude and `mscale` stays 1.
pub fn unlimited_ocr2_12l() -> ArchConfig {
    ArchConfig {
        hidden_size: 1280,
        intermediate_size: 1792,
        moe_intermediate_size: 896,
        num_heads: 10,
        num_kv_heads: 10,
        num_full_kv_heads: 10,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 129_280,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 500_000.0,
        full_rope_theta: 500_000.0,
        partial_rotary_factor: 1.0,
        num_layers: 12,
        num_experts: 64,
        top_k_experts: 6,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        // Every layer keeps its whole KV: DASH reads it, it does not shrink it.
        full_attention_layer_mask: vec![1u8; 12],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::UnlimitedOcr2,
        attn_output_gate: false,
        attention_scale: 0.088_388_347_648_318_44,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        // Softmax over all 64 experts, top-6, no renormalization
        // (`norm_topk_prob: false`), scale 1.0: `deepseek2`'s router function.
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig {
            factor: 4.0,
            original_context: 32768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
        },
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
        dense_lead_intermediate_size: 6848,
        num_dense_leading_layers: 1,
    }
}
