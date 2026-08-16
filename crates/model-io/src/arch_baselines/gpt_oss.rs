use crate::arch_config::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, RopeScalingConfig,
};

/// `gpt-oss`'s alternating window, in this port's polarity (0 = sliding,
/// 1 = full attention).
///
/// EVEN LAYERS SLIDE, and that is read rather than guessed: the file
/// declares `attention.sliding_window = 128` and NO
/// `sliding_window_pattern`, so llama.cpp's default period of 2 applies and
/// `llama_hparams::set_swa_pattern` with `dense_first = false` computes
/// `is_swa[il] = (il % 2) < 1`. Getting the phase backwards gives a model
/// that is wrong only past 128 tokens of context, which no short smoke
/// reaches.
fn gpt_oss_layer_mask(layers: usize) -> Vec<u8> {
    (0..layers).map(|i| u8::from(i % 2 != 0)).collect()
}

/// Canonical `gpt-oss-20b` baseline (ROADMAP M5). 24 layers, hidden 2880,
/// 64 query heads over 8 KV heads at head_dim 64, 32 routed experts at top-4
/// with no shared expert, untied lm_head, no logit softcap, and an
/// alternating 128-token sliding window.
///
/// **Every shape here was read off the published
/// `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf` header**
/// (`gguf_checkpoint_network.rs::scopes_phase_m5_gpt_oss_layer`), and every
/// BEHAVIOURAL field off llama.cpp's `src/models/openai-moe.cpp`, which is
/// what wrote those bytes.
///
/// TWO FIELDS ARE NOT METADATA AND THAT ASYMMETRY IS THE POINT.
/// `swiglu_limit` is 7.0 and its companion alpha is 1.702, both HARDCODED in
/// llama.cpp's graph builder with its own "TODO: move to hparams?" beside
/// them, so they are baseline constants here and a file cannot override
/// them. The YaRN scalars in `rope_scaling` ARE published, so those are read.
/// Storing either the other way round would be a guess wearing the other's
/// clothes.
pub fn gpt_oss_20b() -> ArchConfig {
    ArchConfig {
        hidden_size: 2880,
        // One `feed_forward_length`, and `expert_feed_forward_length` equals
        // it. There is no shared expert to size separately.
        intermediate_size: 2880,
        moe_intermediate_size: 2880,
        num_heads: 64,
        num_kv_heads: 8,
        num_full_kv_heads: 8,
        head_dim: 64,
        full_head_dim: 64,
        vocab_size: 201_088,
        sliding_window: 128,
        final_logit_softcap: 0.0,
        // One base for both halves of the window: the file publishes
        // `rope.freq_base` and NO `rope.freq_base_swa`, unlike Gemma, which
        // is why these two agree.
        rope_theta: 150_000.0,
        full_rope_theta: 150_000.0,
        partial_rotary_factor: 1.0,
        num_layers: 24,
        num_experts: 32,
        top_k_experts: 4,
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        full_attention_layer_mask: gpt_oss_layer_mask(24),
        hidden_activation: "silu".to_string(),
        family: ModelFamily::GptOss,
        attn_output_gate: false,
        // `1/sqrt(64)` = 0.125 exactly, from llama.cpp's own
        // `1.0f/sqrtf(float(n_rot))`. A binary fraction, so unlike Mixtral's
        // and Qwen3-MoE's it survives AGENTS.md Gotcha 24's round-trip
        // without care.
        attention_scale: 0.125,
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: CompressedAttentionConfig::NONE,
        hyper_connections: HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        // llama.cpp's `SOFTMAX_WEIGHT`: top-k on the RAW logits, then a
        // softmax over the selected k alone and no renormalization. That is
        // `router_topk_gemma4`'s existing contract minus the per-expert
        // scale, so the spelling here is the same "softmax" the others use.
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 7.0,
        rope_scaling: RopeScalingConfig {
            factor: 32.0,
            original_context: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
        },
    }
}
