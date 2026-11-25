//! Gemma 2 text baseline. Facts read off `bartowski/gemma-2-2b-it-GGUF`'s
//! Q3_K_L header (2026-11-19): 26 layers, hidden 2304, ffn 9216, 8 q heads over
//! 4 kv at head_dim 256, `attn_logit_softcapping` 50, `final_logit_softcapping`
//! 30, `attention.sliding_window` 4096, RMS epsilon 1e-6, vocab 256000, tied
//! head, and NO per-head q/k norm tensors (Gemma 3 added them).

use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

/// The attention-score soft cap every Gemma 2 size publishes
/// (`attn_logit_softcapping`). A constant of the family, like Gemma 3's
/// schedule: `gguf_config` refuses a file that states another value rather
/// than decoding it with this one.
pub const GEMMA2_ATTN_LOGIT_SOFTCAP: f32 = 50.0;

/// Gemma 2 alternates local and global attention one for one, local first:
/// llama.cpp's `set_swa_pattern(2)` makes `il % 2 < 1` sliding, so EVEN layers
/// slide (mask 0) and ODD layers are global (mask 1). The GGUF does not carry
/// it.
pub fn gemma2_layer_mask(num_layers: usize) -> Vec<u8> {
    (0..num_layers).map(|i| u8::from(i % 2 == 1)).collect()
}

/// `gemma-2-2b-it`. Both layer kinds rotate over the whole head at 10000.
pub fn gemma2_2b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2304,
        intermediate_size: 9216,
        moe_intermediate_size: 0,
        num_heads: 8,
        num_kv_heads: 4,
        num_full_kv_heads: 4,
        head_dim: 256,
        full_head_dim: 256,
        vocab_size: 256_000,
        sliding_window: 4096,
        final_logit_softcap: 30.0,
        rope_theta: 10_000.0,
        full_rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        num_layers: 26,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: true,
        attention_k_eq_v: false,
        full_attention_layer_mask: gemma2_layer_mask(26),
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma2,
        attn_output_gate: false,
        // query_pre_attn_scalar 256 for the 2B and 9B: 256^-0.5, a binary fraction.
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
        rope_freq_factors: Vec::new(),
        vision: VisionConfig::NONE,
        short_conv_kernel_size: 0,
        ple: PleConfig::NONE,
    }
}
