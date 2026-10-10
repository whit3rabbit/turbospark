//! The qwen2 dense backbone.
//!
//! Config port of `mlx_audio/lm/models/qwen2.py` (vendored from mlx-lm) at
//! the pinned mlx-audio commit: the llama geometry with forced attention
//! q/k/v biases, no MLP bias, no q/k norms, no sliding window, and
//! `head_dim = hidden_size / num_attention_heads`.

use serde_json::Value;

use crate::nn::json as cfg;
use crate::Result;

use super::Weights;

use super::engine::{Backbone, DenseConfig};

/// Parses a qwen2 config into the shared engine configuration.
pub fn from_json(value: &Value) -> Result<DenseConfig> {
    let num_attention_heads = cfg::usize_field(value, "num_attention_heads")?;
    let num_key_value_heads = cfg::usize_field(value, "num_key_value_heads")?;
    let hidden = cfg::usize_field(value, "hidden_size")?;
    Ok(DenseConfig {
        model_type: cfg::text_field(value, "model_type")?,
        hidden_size: hidden,
        num_hidden_layers: cfg::usize_field(value, "num_hidden_layers")?,
        intermediate_size: cfg::usize_field(value, "intermediate_size")?,
        num_attention_heads,
        num_key_value_heads,
        head_dim: hidden / num_attention_heads,
        rms_norm_eps: cfg::required(value, "rms_norm_eps")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| crate::SpeechError::BadConfig {
                field: "rms_norm_eps".into(),
                why: "must be a positive number".into(),
            })? as f32,
        vocab_size: cfg::usize_field(value, "vocab_size")?,
        qk_norms: false,
        attention_bias: true,
        mlp_bias: false,
        rope_theta: value
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(1000000.0) as f32,
        rope_traditional: value
            .get("rope_traditional")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        max_position_embeddings: Some(
            value
                .get("max_position_embeddings")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(32768),
        ),
        tie_word_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        sliding_window: None,
    })
}

/// Loads a qwen2 backbone from `weights`.
pub fn load(weights: &Weights, value: &Value) -> Result<Backbone> {
    Backbone::load(weights, from_json(value)?)
}

/// The loaded qwen2 model type. Construction goes through [`load`].
pub type Qwen2Model = Backbone;
