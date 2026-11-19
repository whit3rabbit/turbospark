//! Gemma 3 text baselines. Facts: `docs/GEMMA3_PHASE0.md`.

use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// Gemma 3's 5-local : 1-global schedule (`sliding_window_pattern 6`): layer
/// `i` is global when `(i + 1) % 6 == 0`. The GGUF does not carry it, so it is
/// an explicit constant that `gguf_config` guards by layer count.
pub fn gemma3_layer_mask(num_layers: usize) -> Vec<u8> {
    (0..num_layers)
        .map(|i| u8::from((i + 1) % 6 == 0))
        .collect()
}

/// `gemma-3-1b-it` read off the `ggml-org` GGUF header (rev f9c28bcd) and the
/// `mlx-community` config (`query_pre_attn_scalar 256`, no softcap, no rope
/// scaling). Local layers rotate at 10000 (`rope_local_base_freq`, not in the
/// GGUF), global layers at the file's `freq_base` 1e6, both over the whole
/// head.
pub fn gemma3_1b() -> ArchConfig {
    ArchConfig {
        hidden_size: 1152,
        intermediate_size: 6912,
        moe_intermediate_size: 0,
        num_heads: 4,
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 262_144,
        sliding_window: 512,
        final_logit_softcap: 0.0,
        rope_theta: 10_000.0,
        full_rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        num_layers: 26,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        full_attention_layer_mask: gemma3_layer_mask(26),
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma3,
        attn_output_gate: false,
        // query_pre_attn_scalar^-0.5 = 256^-0.5, a binary fraction.
        attention_scale: 0.0625,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: false,
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
        rope_freq_factors: Vec::new(),
        ple: PleConfig::NONE,
    }
}
