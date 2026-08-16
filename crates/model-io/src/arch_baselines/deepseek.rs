use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

fn deepseek_v4_flash_layer_mask() -> Vec<u8> {
    // Layer kinds: 0 = sliding-window only (layers 0-1), then 3 = CSA on
    // even layers and 4 = HCA on odd layers.
    let mut mask = vec![0u8; 43];
    for (i, slot) in mask.iter_mut().enumerate().take(43).skip(2) {
        *slot = if i % 2 == 0 { 3 } else { 4 };
    }
    mask
}

/// Canonical DeepSeek-V4-Flash 284B-A13B baseline: 43 all-MoE layers,
/// shared-KV MQA attention, sliding window 128 on every layer, and
/// compressed long-range KV (CSA/HCA). The residual is 4 mHC streams.
/// Untied lm_head, no logit softcap.
pub fn deepseek_v4_flash_284b_a13b() -> ArchConfig {
    ArchConfig {
        hidden_size: 4096,
        intermediate_size: 2048,
        moe_intermediate_size: 2048,
        num_heads: 64,
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: 512,
        full_head_dim: 512,
        vocab_size: 129_280,
        sliding_window: 128,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 0.125,
        num_layers: 43,
        num_experts: 256,
        top_k_experts: 6,
        tie_word_embeddings: false,
        attention_k_eq_v: true,
        full_attention_layer_mask: deepseek_v4_flash_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::DeepseekV4Flash,
        attn_output_gate: false,
        attention_scale: 0.044_194_173_824_159_216, // 512^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig {
            q_lora_rank: 1024,
            o_lora_rank: 1024,
            o_groups: 8,
            rope_head_dim: 64,
            index_n_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            csa_compress_rate: 4,
            hca_compress_rate: 128,
            compress_rope_theta: 160_000.0,
            rope_scaling_factor: 16.0,
            rope_scaling_original_max: 65_536,
            rope_scaling_beta_fast: 32.0,
            rope_scaling_beta_slow: 1.0,
        },
        hyper_connections: HyperConnectionConfig {
            mult: 4,
            sinkhorn_iters: 20,
            eps: 1.0e-6,
        },
        num_hash_routed_layers: 3,
        router_scoring_func: "sqrtsoftplus".to_string(),
        routed_scaling_factor: 1.5,
        swiglu_limit: 10.0,
        rope_scaling: RopeScalingConfig::NONE,
    }
}
