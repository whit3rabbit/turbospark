//! `config.json` for the GTE ("new") encoder (`model_type = "gte"`).
//!
//! Parsed by hand so every refusal names the field. A field that is
//! absent takes the default of the pinned `GteConfig.__init__`
//! (configuration_hf_alibaba_nlp_gte.py:94-119), because that is what the
//! reference runs with; an explicit `null` is an error (the reference
//! would carry `None` into the model). A value this port does not
//! implement is an error, never ignored.
//!
//! Boolean flags are read with Python truthiness: the pinned checkpoint
//! stores `unpad_inputs` and `use_memory_efficient_attention` as the
//! string `"true"`, which the reference treats as true (and so would a
//! string `"false"`).

use serde_json::Value;

use crate::error::{bad_config, EmbeddingError};

/// The encoder hyperparameters this port consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct GteConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    /// RoPE base as written (Python int or float). The reference raises
    /// it to an f32 power, so tables cast it to f32 once.
    pub rope_theta: f64,
    /// Trained Matryoshka widths (`matryoshka_dimensions`), informational.
    pub matryoshka_dimensions: Vec<usize>,
}

/// Python truthiness of a JSON value: absent and `null` are false, a
/// string is true when non-empty, a number when non-zero.
pub(crate) fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn usize_field(v: &Value, key: &str, default: Option<usize>) -> Result<usize, EmbeddingError> {
    match v.get(key) {
        None => default.ok_or_else(|| bad_config(key, "missing")),
        Some(x) => {
            let n = x
                .as_u64()
                .ok_or_else(|| bad_config(key, "not a non-negative integer"))?;
            usize::try_from(n).map_err(|_| bad_config(key, "does not fit usize"))
        }
    }
}

fn f64_field(v: &Value, key: &str, default: f64) -> Result<f64, EmbeddingError> {
    let x = match v.get(key) {
        None => default,
        Some(x) => x.as_f64().ok_or_else(|| bad_config(key, "not a number"))?,
    };
    if !x.is_finite() || x <= 0.0 {
        return Err(bad_config(key, "must be a positive finite number"));
    }
    Ok(x)
}

fn str_field<'a>(v: &'a Value, key: &str, default: &'a str) -> Result<&'a str, EmbeddingError> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x.as_str().ok_or_else(|| bad_config(key, "not a string")),
    }
}

fn unsupported(why: String) -> EmbeddingError {
    EmbeddingError::Unsupported { why }
}

impl GteConfig {
    pub fn from_json(v: &Value) -> Result<Self, EmbeddingError> {
        match v.get("model_type").and_then(Value::as_str) {
            Some("gte") => {}
            other => {
                return Err(bad_config(
                    "model_type",
                    format!("expected gte, got {other:?}"),
                ))
            }
        }
        // Absolute positions take a different embedding path
        // (modeling:289-292, 409-412).
        let pos = str_field(v, "position_embedding_type", "rope")?;
        if pos != "rope" {
            return Err(unsupported(format!(
                "position_embedding_type {pos:?}; only rope is implemented"
            )));
        }
        // NTK scaling rewrites inv_freq past the trained length
        // (modeling:218-250); refuse rather than run unscaled.
        if !v.get("rope_scaling").is_none_or(Value::is_null) {
            return Err(unsupported(
                "rope_scaling is set; only null is implemented".into(),
            ));
        }
        let act = str_field(v, "hidden_act", "gelu")?;
        if act != "gelu" {
            return Err(unsupported(format!(
                "hidden_act {act:?}; only gelu (erf) is implemented"
            )));
        }
        let norm = str_field(v, "layer_norm_type", "layer_norm")?;
        if norm != "layer_norm" {
            return Err(unsupported(format!(
                "layer_norm_type {norm:?}; only layer_norm is implemented"
            )));
        }
        // pack_qkv defaults to true when absent; false or null loads
        // separate q/k/v_proj (modeling:435-444).
        if v.get("pack_qkv").is_some() && !truthy(v.get("pack_qkv")) {
            return Err(unsupported(
                "pack_qkv = false (separate q/k/v projections) is not implemented".into(),
            ));
        }
        // The pinned code only logs a TODO for logn scaling
        // (modeling:927-937), so it is a no-op there; a checkpoint that
        // sets it expects a behavior nobody implements.
        for key in ["logn_attention_scale", "logn_attention_clip1"] {
            if truthy(v.get(key)) {
                return Err(unsupported(format!("{key} is set; not implemented")));
            }
        }

        let num_hidden_layers = usize_field(v, "num_hidden_layers", Some(12))?;
        let matryoshka_dimensions = match v.get("matryoshka_dimensions") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| bad_config("matryoshka_dimensions", "not an integer list"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(bad_config("matryoshka_dimensions", "not a list")),
        };
        let config = Self {
            vocab_size: usize_field(v, "vocab_size", Some(30528))?,
            hidden_size: usize_field(v, "hidden_size", Some(768))?,
            num_hidden_layers,
            num_attention_heads: usize_field(v, "num_attention_heads", Some(12))?,
            intermediate_size: usize_field(v, "intermediate_size", Some(3072))?,
            max_position_embeddings: usize_field(v, "max_position_embeddings", Some(2048))?,
            type_vocab_size: usize_field(v, "type_vocab_size", Some(1))?,
            layer_norm_eps: f64_field(v, "layer_norm_eps", 1e-12)? as f32,
            rope_theta: f64_field(v, "rope_theta", 10000.0)?,
            matryoshka_dimensions,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), EmbeddingError> {
        for (k, n) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if n == 0 {
                return Err(bad_config(k, "must be positive"));
            }
        }
        // type_vocab_size 0 drops the token-type add (modeling:299); the
        // pinned checkpoint has 1. Only the add path is implemented.
        if self.type_vocab_size == 0 {
            return Err(unsupported(
                "type_vocab_size = 0 (no token-type embedding) is not implemented".into(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(bad_config(
                "num_attention_heads",
                format!(
                    "hidden_size {} is not a multiple of {} heads",
                    self.hidden_size, self.num_attention_heads
                ),
            ));
        }
        if self.head_dim() % 2 != 0 {
            return Err(bad_config(
                "num_attention_heads",
                "head_dim must be even for RoPE",
            ));
        }
        if let Some(&d) = self
            .matryoshka_dimensions
            .iter()
            .find(|&&d| d == 0 || d > self.hidden_size)
        {
            return Err(bad_config(
                "matryoshka_dimensions",
                format!("{d} outside 1..={}", self.hidden_size),
            ));
        }
        Ok(())
    }

    /// `int(hidden_size / num_attention_heads)` (modeling:313).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `1 / sqrt(head_dim)`, the eager path's divisor and the sdpa
    /// default scale. Exact in f32 for head_dim 16 and 64.
    pub fn attn_scale(&self) -> f32 {
        (1.0 / (self.head_dim() as f64).sqrt()) as f32
    }
}
