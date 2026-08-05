//! The three canonical architecture baselines, checked against an installed
//! model's manifest at load time. Ported from the `ArchConfig` static
//! members in `Infrastructure/ModelIO/ModelTypes.swift`.

use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily,
};

fn gemma4_layer_mask() -> Vec<u8> {
    let mut mask = vec![0u8; 30];
    let mut i = 5;
    while i < 30 {
        mask[i] = 1;
        i += 6;
    }
    mask
}

/// Canonical Gemma 4 26B-A4B baseline.
/// `intermediate_size = 2112` is the shared-expert FFN width (3x moe).
pub fn gemma4_26b_a4b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2816,
        intermediate_size: 2112,
        moe_intermediate_size: 704,
        num_heads: 16,
        num_kv_heads: 8,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 512,
        vocab_size: 262_144,
        sliding_window: 1024,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 30,
        num_experts: 128,
        top_k_experts: 8,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: gemma4_layer_mask(),
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

fn qwen36_layer_mask() -> Vec<u8> {
    // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
    // 4th layer ((i + 1) % 4 == 0).
    let mut mask = vec![2u8; 40];
    let mut i = 3;
    while i < 40 {
        mask[i] = 1;
        i += 4;
    }
    mask
}

/// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
/// gated-DeltaNet linear-attention layers and 10 full-attention layers
/// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
/// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
pub fn qwen36_35b_a3b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2048,
        intermediate_size: 512,
        moe_intermediate_size: 512,
        num_heads: 16,
        num_kv_heads: 2,
        num_full_kv_heads: 2,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 248_320,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 10_000_000.0,
        full_rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 40,
        num_experts: 256,
        top_k_experts: 8,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: qwen36_layer_mask(),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Qwen36,
        attn_output_gate: true,
        attention_scale: 0.0625, // 256^-0.5
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: true,
        rope_neox_subdim: true,
        linear_attention: LinearAttentionConfig {
            num_k_heads: 16,
            num_v_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
        },
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
    }
}

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
    }
}

/// Registry keyed by `manifest.arch.family` for auto-detection at load.
pub fn known_architecture(family: ModelFamily) -> ArchConfig {
    match family {
        ModelFamily::Gemma4 => gemma4_26b_a4b(),
        ModelFamily::Qwen36 => qwen36_35b_a3b(),
        ModelFamily::DeepseekV4Flash => deepseek_v4_flash_284b_a13b(),
    }
}

pub fn all_known_architectures() -> Vec<ArchConfig> {
    vec![
        gemma4_26b_a4b(),
        qwen36_35b_a3b(),
        deepseek_v4_flash_284b_a13b(),
    ]
}
