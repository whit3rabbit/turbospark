//! The llama dense backbone.
//!
//! Config port of `mlx_audio/lm/models/llama.py` (vendored from mlx-lm) at
//! the pinned mlx-audio commit: `from_dict` keeps only known fields, applies
//! the llama defaults (no attention or MLP bias, kv heads defaulting to the
//! query heads, optional head_dim, layer types defaulting to full
//! attention), and forwards to the shared engine. Sliding attention layers
//! are refused by the engine until the rotating cache is ported.

use serde_json::Value;

use crate::nn::json as cfg;
use crate::Result;

use super::Weights;

use super::engine::{Backbone, DenseConfig};

/// Parses a llama `config.json` (or the config dict a TTS family builds)
/// into the shared engine configuration.
pub fn from_json(value: &Value) -> Result<DenseConfig> {
    const BAD_HEADS: &str = "must be a positive integer fitting usize";
    let num_attention_heads = cfg::usize_field(value, "num_attention_heads")?;
    let num_key_value_heads = value
        .get("num_key_value_heads")
        .and_then(Value::as_u64)
        .map(|n| {
            usize::try_from(n).map_err(|_| crate::SpeechError::BadConfig {
                field: "head count".into(),
                why: BAD_HEADS.into(),
            })
        })
        .transpose()?
        .unwrap_or(num_attention_heads);
    let head_dim = value
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(|n| {
            usize::try_from(n).map_err(|_| crate::SpeechError::BadConfig {
                field: "head count".into(),
                why: BAD_HEADS.into(),
            })
        })
        .transpose()?;
    let num_hidden_layers = cfg::usize_field(value, "num_hidden_layers")?;
    let layer_types = value.get("layer_types").and_then(Value::as_array);
    if let Some(types) = layer_types {
        if types.len() != num_hidden_layers {
            return Err(crate::SpeechError::BadConfig {
                field: "layer_types".into(),
                why: format!("has {} entries for {num_hidden_layers} layers", types.len()),
            });
        }
        if types
            .iter()
            .filter_map(Value::as_str)
            .any(|t| t != "full_attention")
        {
            return Err(crate::SpeechError::Unsupported {
                why: "layer_types other than full_attention are not ported yet".into(),
            });
        }
    }
    let hidden = cfg::usize_field(value, "hidden_size")?;
    Ok(DenseConfig {
        model_type: cfg::text_field(value, "model_type")?,
        hidden_size: hidden,
        num_hidden_layers,
        intermediate_size: cfg::usize_field(value, "intermediate_size")?,
        num_attention_heads,
        num_key_value_heads,
        head_dim: head_dim.unwrap_or(hidden / num_attention_heads).max(1),
        rms_norm_eps: cfg::required(value, "rms_norm_eps")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| crate::SpeechError::BadConfig {
                field: "rms_norm_eps".into(),
                why: "must be a positive number".into(),
            })? as f32,
        vocab_size: cfg::usize_field(value, "vocab_size")?,
        qk_norms: false,
        attention_bias: value
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        mlp_bias: value
            .get("mlp_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        rope_theta: value
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10000.0) as f32,
        rope_traditional: value
            .get("rope_traditional")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        max_position_embeddings: value
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
        tie_word_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        sliding_window: value
            .get("sliding_window")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
    })
}

/// Loads a llama backbone from `weights` (prefix `model.` for a full-model
/// save).
pub fn load(weights: &Weights, value: &Value) -> Result<Backbone> {
    Backbone::load(weights, from_json(value)?)
}

/// The loaded llama model type. Construction goes through [`load`] so the
/// flavor defaults apply.
pub type LlamaModel = Backbone;
