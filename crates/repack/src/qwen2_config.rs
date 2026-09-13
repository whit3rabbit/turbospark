//! Dense Qwen2/Qwen2.5 `config.json` parsing.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

/// Parse a dense Qwen2/Qwen2.5 configuration into the canonical architecture
/// shape. The parser accepts both the ordinary root form and a text-config
/// wrapper, but never accepts an active sliding window or a non-SwiGLU
/// activation because the shared Llama flow has no such behavior.
pub fn parse_qwen2_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Qwen2Dense)
        .map_err(Gemma4Error::Config)?;
    let tc = root.get("text_config").unwrap_or(&root);

    let i = |key: &str| -> Result<i64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };
    let f = |key: &str| -> Result<f64, Gemma4Error> {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| Gemma4Error::Config(format!("missing {key}")))
    };

    let hidden_size = i("hidden_size")?;
    let num_heads = i("num_attention_heads")?;
    let num_kv_heads = i("num_key_value_heads")?;
    let num_layers = i("num_hidden_layers")?;
    let intermediate_size = i("intermediate_size")?;
    let vocab_size = i("vocab_size")?;
    let max_position_embeddings = i("max_position_embeddings")?;
    if hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_layers <= 0 {
        return Err(Gemma4Error::Config(
            "hidden size, attention heads, and layer count must be positive".into(),
        ));
    }
    if hidden_size % num_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "hidden_size {hidden_size} is not divisible by num_attention_heads {num_heads}"
        )));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(Gemma4Error::Config(format!(
            "num_attention_heads {num_heads} is not divisible by num_key_value_heads {num_kv_heads}"
        )));
    }
    if intermediate_size <= 0 || vocab_size <= 0 || max_position_embeddings <= 0 {
        return Err(Gemma4Error::Config(
            "intermediate_size, vocab_size, and max_position_embeddings must be positive".into(),
        ));
    }

    let rms_norm_eps = f("rms_norm_eps")?;
    if (rms_norm_eps - 1e-6).abs() > 1e-12 {
        return Err(Gemma4Error::Config(format!(
            "Qwen2 requires rms_norm_eps 1e-6, got {rms_norm_eps}"
        )));
    }
    let rope_theta = f("rope_theta")?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(Gemma4Error::Config(format!(
            "rope_theta must be finite and positive, got {rope_theta}"
        )));
    }
    let hidden_activation = tc
        .get("hidden_act")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Gemma4Error::Config("missing hidden_act".into()))?;
    if hidden_activation != "silu" {
        return Err(Gemma4Error::Config(format!(
            "Qwen2 requires hidden_act silu, got {hidden_activation:?}"
        )));
    }
    if tc
        .get("use_sliding_window")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "Qwen2 sliding-window attention is not wired; use_sliding_window must be false".into(),
        ));
    }

    let head_dim = hidden_size / num_heads;
    Ok(ArchConfig {
        hidden_size,
        intermediate_size,
        moe_intermediate_size: 0,
        num_heads,
        num_kv_heads,
        num_full_kv_heads: num_kv_heads,
        head_dim,
        full_head_dim: head_dim,
        vocab_size,
        sliding_window: 0,
        final_logit_softcap: 0.0,
        rope_theta,
        full_rope_theta: rope_theta,
        partial_rotary_factor: 1.0,
        num_layers,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: tc
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1; num_layers as usize],
        hidden_activation: hidden_activation.to_string(),
        family: ModelFamily::Qwen2Dense,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
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
        vision: VisionConfig::NONE,
        ple: PleConfig::NONE,
    })
}
