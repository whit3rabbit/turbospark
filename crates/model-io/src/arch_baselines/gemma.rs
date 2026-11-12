use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
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

fn gemma4_12b_layer_mask() -> Vec<u8> {
    let mut mask = vec![0u8; 48];
    let mut i = 5;
    while i < 48 {
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
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
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
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
    }
}

/// The DENSE Gemma 4 12B trunk, read off `cowWhySo/humanizer-12b-mlx-8bit`
/// @ baea046b43eb77516e270ffea17869bbe9110fc8 (`text_config` of a
/// `gemma4_unified` checkpoint, `enable_moe_block: false`).
///
/// NOT the `known_architecture(Gemma4)` default -- the family's fallback
/// stays the MoE 26B shape every pre-existing manifest resolves against, and
/// `peek_manifest_arch` overwrites every shape field a manifest carries, so
/// a dense install never reads the fallback. This entry exists for the
/// config-parser's headline test and any test that wants the real dense
/// production values inline.
///
/// The intermediate width 15360 is the DENSE FFN (no shared-expert shrink:
/// that 3x-moe relationship is an MoE-only convention). Everything hybrid --
/// the 5:1 schedule, `head_dim` 256 against `full_head_dim` 512, one global
/// KV head, the thetas -- matches the 26B rows.
pub fn gemma4_12b_dense() -> ArchConfig {
    ArchConfig {
        hidden_size: 3840,
        intermediate_size: 15360,
        moe_intermediate_size: 0,
        num_heads: 16,
        num_kv_heads: 8,
        num_full_kv_heads: 1,
        head_dim: 256,
        full_head_dim: 512,
        vocab_size: 262_144,
        sliding_window: 1024,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
        num_layers: 48,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: gemma4_12b_layer_mask(),
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
        mla: MlaConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: RopeScalingConfig::NONE,
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
    }
}
