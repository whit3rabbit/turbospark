use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig, VisionConfig,
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
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
    }
}
