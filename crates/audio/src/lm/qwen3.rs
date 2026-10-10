//! The qwen3 dense backbone.
//!
//! Config port of `mlx_audio/lm/models/qwen3.py` (vendored from mlx-lm) at
//! the pinned mlx-audio commit: per-head q/k RMSNorm, bias-free
//! projections, every field required by the reference dataclass, and the
//! non-traditional rope construction.

use serde_json::Value;

use crate::nn::json as cfg;
use crate::Result;

use super::Weights;

use super::engine::{Backbone, DenseConfig};

/// Parses a qwen3 config into the shared engine configuration.
pub fn from_json(value: &Value) -> Result<DenseConfig> {
    let num_attention_heads = cfg::usize_field(value, "num_attention_heads")?;
    Ok(DenseConfig {
        model_type: cfg::text_field(value, "model_type")?,
        hidden_size: cfg::usize_field(value, "hidden_size")?,
        num_hidden_layers: cfg::usize_field(value, "num_hidden_layers")?,
        intermediate_size: cfg::usize_field(value, "intermediate_size")?,
        num_attention_heads,
        num_key_value_heads: cfg::usize_field(value, "num_key_value_heads")?,
        head_dim: cfg::usize_field(value, "head_dim")?,
        rms_norm_eps: cfg::required(value, "rms_norm_eps")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| crate::SpeechError::BadConfig {
                field: "rms_norm_eps".into(),
                why: "must be a positive number".into(),
            })? as f32,
        vocab_size: cfg::usize_field(value, "vocab_size")?,
        qk_norms: true,
        attention_bias: false,
        mlp_bias: false,
        rope_theta: cfg::required(value, "rope_theta")?
            .as_f64()
            .filter(|v| *v > 0.0)
            .ok_or_else(|| crate::SpeechError::BadConfig {
                field: "rope_theta".into(),
                why: "must be a positive number".into(),
            })? as f32,
        rope_traditional: false,
        rope_scaling: value.get("rope_scaling").cloned().filter(|v| !v.is_null()),
        max_position_embeddings: Some(cfg::usize_field(value, "max_position_embeddings")?),
        tie_word_embeddings: cfg::bool_field(value, "tie_word_embeddings")?,
        sliding_window: None,
    })
}

/// Loads a qwen3 backbone from `weights`.
pub fn load(weights: &Weights, value: &Value) -> Result<Backbone> {
    Backbone::load(weights, from_json(value)?)
}

/// The loaded qwen3 model type. Construction goes through [`load`].
pub type Qwen3Model = Backbone;
