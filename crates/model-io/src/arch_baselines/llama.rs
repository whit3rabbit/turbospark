use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

/// Canonical Mixtral-8x7B-Instruct baseline (ROADMAP Phase M2): 32 dense
/// full-attention layers with GQA (32 query heads over 8 KV heads), 8 routed
/// experts at top-2 and no shared expert, SwiGLU, untied lm_head, no logit
/// softcap and no sliding window.
///
/// **This baseline is Mixtral's, and the family covers dense Llama too.**
/// `general.architecture = "llama"` is what both report -- a dense Llama 3.1
/// differs from this only in shape fields (`num_experts = 0`, its own vocab
/// and thetas), which `arch_from_gguf` reads off the file. Every BEHAVIOURAL
/// field below is shared, which is what makes one baseline honest for both;
/// see `gguf_config.rs`'s module header for why that split matters.
///
/// `intermediate_size` equals `moe_intermediate_size` because the file
/// publishes one `feed_forward_length` and has no shared expert to size
/// separately.
pub fn mixtral_8x7b() -> ArchConfig {
    ArchConfig {
        hidden_size: 4096,
        intermediate_size: 14_336,
        moe_intermediate_size: 14_336,
        num_heads: 32,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 128,
        full_head_dim: 128,
        vocab_size: 32_000,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta: 1_000_000.0,
        full_rope_theta: 1_000_000.0,
        // Full rotary: `rope.dimension_count` is 128, the whole head.
        partial_rotary_factor: 1.0,
        num_layers: 32,
        num_experts: 8,
        top_k_experts: 2,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1u8; 32],
        hidden_activation: "silu".to_string(),
        family: ModelFamily::Llama,
        attn_output_gate: false,
        // 128^-0.5 = 2^-3.5. NOT a binary fraction, unlike Gemma's 1.0 and
        // Qwen's 0.0625, so AGENTS.md Gotcha 24's round-trip warning applies
        // to this field and `crates/model-io/tests/arch_config.rs` pins it.
        attention_scale: 0.088_388_347_648_318_45,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
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
    }
}
