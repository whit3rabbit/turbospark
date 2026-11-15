//! Plain dense `qwen3` `config.json` parsing (`Qwen/Qwen3-4B`,
//! `Qwen3-4B-Instruct-2507`, and their MLX conversions).
//!
//! The architecture is [`crate::parse_qwen3_vl_config`]'s text trunk without
//! the wrapper: plain full-attention GQA, per-head q/k RMSNorm, a dense
//! SwiGLU FFN and full-head NeoX RoPE. The differences from `qwen2` (whose
//! parser this one otherwise follows) are the ones the model card hides:
//!
//! 1. `head_dim` is a REQUIRED key. Qwen3-4B declares 128 where
//!    `hidden_size / num_attention_heads` is 2560 / 32 = 80, so deriving it
//!    (the `qwen2` shortcut) builds the wrong attention shape.
//! 2. Per-head q/k RMSNorm is a family constant, not a config key: the
//!    reference builds `self_attn.{q,k}_norm` unconditionally.
//! 3. There is NO attention bias (Qwen2 has one); a config that declares one
//!    is refused rather than run with the bias dropped.
//! 4. Sliding-window attention, a non-default RoPE scaling (YaRN) and any
//!    expert count are refused: the shared flow does not implement them, and
//!    accepting the file would produce fluent, wrong output.
//!
//! Tied embeddings follow `tie_word_embeddings` (true for 0.6B to 4B, false
//! from 8B up); the install walk reads the `lm_head` tensor's presence the
//! same way for both.

use model_io::{
    ArchConfig, CompressedAttentionConfig, HyperConnectionConfig, LinearAttentionConfig, MlaConfig,
    ModelFamily, PleConfig, RopeScalingConfig, VisionConfig,
};

use crate::gemma4_checkpoint::Gemma4Error;

const MAX_MODEL_LAYERS: i64 = 4096;

/// Parse a plain dense `qwen3` configuration into the canonical architecture
/// shape. See the module doc for what is required and what is refused.
pub fn parse_qwen3_config(json: &str) -> Result<ArchConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    crate::arch_registry::refuse_foreign_config(&root, ModelFamily::Qwen3Dense)
        .map_err(Gemma4Error::Config)?;
    // Plain `qwen3` is flat. A `text_config` here means a wrapper of some
    // other architecture whose own model_type the registry did not resolve.
    if root.get("text_config").is_some() {
        return Err(Gemma4Error::Config(
            "a qwen3 config carries no text_config wrapper; this is not plain dense qwen3".into(),
        ));
    }
    let tc = &root;

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
    // Independent of hidden/heads on this architecture; required, not derived.
    let head_dim = i("head_dim")?;
    if hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 || num_layers <= 0 || head_dim <= 0 {
        return Err(Gemma4Error::Config(
            "hidden size, attention heads, KV heads, layer count, and head_dim must be positive"
                .into(),
        ));
    }
    if num_layers > MAX_MODEL_LAYERS {
        return Err(Gemma4Error::Config(format!(
            "num_hidden_layers {num_layers} exceeds the maximum supported layer count of \
             {MAX_MODEL_LAYERS}"
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
            "qwen3 requires rms_norm_eps 1e-6, got {rms_norm_eps}"
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
            "qwen3 requires hidden_act silu, got {hidden_activation:?}"
        )));
    }

    if tc
        .get("use_sliding_window")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "qwen3 sliding-window attention is not wired; use_sliding_window must be false".into(),
        ));
    }
    // Newer transformers exports spell the per-layer mask out. Every entry
    // must be full attention, or the all-ones mask below would be a lie.
    if let Some(types) = tc.get("layer_types").and_then(|v| v.as_array()) {
        if types.iter().any(|t| t.as_str() != Some("full_attention")) {
            return Err(Gemma4Error::Config(
                "qwen3 layer_types names a non-full-attention layer; only full attention is wired"
                    .into(),
            ));
        }
    }
    // The shared flow carries plain RoPE only. YaRN or any other scaling
    // changes every position's rotation, so it is refused by name.
    if let Some(scaling) = tc.get("rope_scaling").filter(|v| !v.is_null()) {
        let rope_type = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        if rope_type != "default" {
            return Err(Gemma4Error::Config(format!(
                "qwen3 rope_scaling rope_type {rope_type:?} is not wired; only plain RoPE is"
            )));
        }
    }
    if tc
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Gemma4Error::Config(
            "qwen3 attention_bias true is not wired on this flow (plain qwen3 has none)".into(),
        ));
    }
    // A dense family: a config that declares experts is `qwen3_moe`, not this.
    if let Some(n) = tc.get("num_experts").and_then(serde_json::Value::as_i64) {
        if n > 0 {
            return Err(Gemma4Error::Config(format!(
                "a qwen3 config declares num_experts {n}; the family is dense, and a \
                 checkpoint with experts is not plain qwen3"
            )));
        }
    }

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
        // FULL rotary, as the qwen3 GGUF baseline and the reference.
        partial_rotary_factor: 1.0,
        num_layers,
        dense_lead_intermediate_size: 0,
        num_dense_leading_layers: 0,
        num_experts: 0,
        top_k_experts: 0,
        tie_word_embeddings: tc
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        attention_k_eq_v: false,
        full_attention_layer_mask: vec![1; num_layers as usize],
        hidden_activation: hidden_activation.to_string(),
        family: ModelFamily::Qwen3Dense,
        attn_output_gate: false,
        attention_scale: (head_dim as f64).powf(-0.5),
        embedding_scaled_by_sqrt_hidden: false,
        router_scaled: false,
        ffn_sandwich_norms: false,
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
    })
}
