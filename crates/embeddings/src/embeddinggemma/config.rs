//! `config.json` for the Gemma3 text backbone.
//!
//! Parsed by hand from `serde_json::Value` so every refusal names the
//! field. Only the configuration EmbeddingGemma ships is accepted: a
//! value this port does not implement (causal attention, logit
//! softcapping, RoPE scaling, another activation, attention bias) is an
//! error, because ignoring it would produce fluent but wrong embeddings.

use serde_json::Value;
use turbospark_audio::quant::QuantScheme;

use crate::error::{bad_config, EmbeddingError};
use crate::weights::quant_scheme;

/// Attention span of one layer (`config.layer_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Bidirectional band `|q - k| < sliding_window`, local RoPE base.
    Sliding,
    /// Every real token, global RoPE base.
    Full,
}

/// The backbone hyperparameters this port consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub query_pre_attn_scalar: f32,
    /// MLX affine quantization, when the checkpoint is an MLX conversion.
    pub quantization: Option<QuantScheme>,
}

/// The only MLX quantization schemes checked against the reference (the
/// pinned mlx-community 8-bit and 4-bit conversions). 5- and 6-bit are
/// pinned but unmeasured, so they are refused.
pub const VERIFIED_QUANT_BITS: &[u32] = &[4, 8];
pub const VERIFIED_QUANT_GROUP: usize = 64;

fn usize_field(v: &Value, key: &str) -> Result<usize, EmbeddingError> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| bad_config(key, "missing or not a non-negative integer"))?;
    usize::try_from(n).map_err(|_| bad_config(key, "does not fit usize"))
}

fn f32_field(v: &Value, key: &str) -> Result<f32, EmbeddingError> {
    let x = v
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| bad_config(key, "missing or not a number"))?;
    if !x.is_finite() {
        return Err(bad_config(key, "not finite"));
    }
    Ok(x as f32)
}

/// Absent and `null` both mean "off".
fn is_null_or_absent(v: &Value, key: &str) -> bool {
    v.get(key).is_none_or(Value::is_null)
}

impl GemmaConfig {
    pub fn from_json(v: &Value) -> Result<Self, EmbeddingError> {
        match v.get("model_type").and_then(Value::as_str) {
            Some("gemma3_text") => {}
            other => {
                return Err(bad_config(
                    "model_type",
                    format!("expected gemma3_text, got {other:?}"),
                ))
            }
        }
        // EmbeddingGemma is an encoder; a causal Gemma3 decoder loaded
        // here would embed without error but with the wrong mask.
        if v.get("use_bidirectional_attention")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(bad_config(
                "use_bidirectional_attention",
                "must be true (causal Gemma3 is not an EmbeddingGemma encoder)",
            ));
        }
        for key in [
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "rope_scaling",
        ] {
            if !is_null_or_absent(v, key) {
                return Err(EmbeddingError::Unsupported {
                    why: format!("config field {key} is set; only null is implemented"),
                });
            }
        }
        match v.get("hidden_activation").and_then(Value::as_str) {
            Some("gelu_pytorch_tanh") => {}
            other => {
                return Err(EmbeddingError::Unsupported {
                    why: format!(
                        "hidden_activation {other:?}; only gelu_pytorch_tanh is implemented"
                    ),
                })
            }
        }
        if v.get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(EmbeddingError::Unsupported {
                why: "attention_bias = true is not implemented".into(),
            });
        }

        let num_hidden_layers = usize_field(v, "num_hidden_layers")?;
        let types = v
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| bad_config("layer_types", "missing or not an array"))?;
        if types.len() != num_hidden_layers {
            return Err(bad_config(
                "layer_types",
                format!("{} entries for {num_hidden_layers} layers", types.len()),
            ));
        }
        let layer_types = types
            .iter()
            .map(|t| match t.as_str() {
                Some("sliding_attention") => Ok(LayerType::Sliding),
                Some("full_attention") => Ok(LayerType::Full),
                other => Err(bad_config(
                    "layer_types",
                    format!("unknown layer type {other:?}"),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let config = Self {
            vocab_size: usize_field(v, "vocab_size")?,
            hidden_size: usize_field(v, "hidden_size")?,
            intermediate_size: usize_field(v, "intermediate_size")?,
            num_attention_heads: usize_field(v, "num_attention_heads")?,
            num_key_value_heads: usize_field(v, "num_key_value_heads")?,
            head_dim: usize_field(v, "head_dim")?,
            layer_types,
            sliding_window: usize_field(v, "sliding_window")?,
            max_position_embeddings: usize_field(v, "max_position_embeddings")?,
            rms_norm_eps: f32_field(v, "rms_norm_eps")?,
            rope_theta: f32_field(v, "rope_theta")?,
            rope_local_base_freq: f32_field(v, "rope_local_base_freq")?,
            query_pre_attn_scalar: f32_field(v, "query_pre_attn_scalar")?,
            quantization: verified_quant(v)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), EmbeddingError> {
        let positive = [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
            ("num_hidden_layers", self.layer_types.len()),
        ];
        for (k, n) in positive {
            if n == 0 {
                return Err(bad_config(k, "must be positive"));
            }
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(bad_config(
                "num_key_value_heads",
                format!(
                    "{} query heads are not a multiple of {} kv heads",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(bad_config("head_dim", "must be even for RoPE"));
        }
        for (k, x) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
            ("rope_local_base_freq", self.rope_local_base_freq),
            ("query_pre_attn_scalar", self.query_pre_attn_scalar),
        ] {
            if x <= 0.0 {
                return Err(bad_config(k, "must be positive"));
            }
        }
        Ok(())
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.layer_types.len()
    }

    /// `query_pre_attn_scalar ** -0.5`, in f64 then rounded once like the
    /// Python float the reference passes to sdpa.
    pub fn attn_scale(&self) -> f32 {
        (f64::from(self.query_pre_attn_scalar)).powf(-0.5) as f32
    }

    /// Exclusive half-width of the sliding layers' band: a query sees a
    /// key when `|q - k| < band_width()`. transformers 4.57.1
    /// `Gemma3TextConfig.__init__` rewrites `sliding_window` to
    /// `sliding_window // 2 + 1` whenever `use_bidirectional_attention`
    /// is set (257 for the shipped 512), so `config.json`'s value is the
    /// full two-sided span, not the reach in each direction.
    pub fn band_width(&self) -> usize {
        self.sliding_window / 2 + 1
    }

    /// `hidden_size ** 0.5` as the Python float the reference casts to
    /// the weight dtype.
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f64).sqrt() as f32
    }
}

/// The MLX `quantization` block, refused unless it is a measured scheme
/// with no per-layer overrides.
fn verified_quant(v: &Value) -> Result<Option<QuantScheme>, EmbeddingError> {
    let Some(scheme) = quant_scheme(v)? else {
        return Ok(None);
    };
    if let Some(obj) = v.get("quantization").and_then(Value::as_object) {
        if let Some(k) = obj
            .keys()
            .find(|k| !matches!(k.as_str(), "group_size" | "bits" | "mode"))
        {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "quantization.{k}: per-layer quantization overrides are not implemented"
                ),
            });
        }
    }
    if !VERIFIED_QUANT_BITS.contains(&scheme.bits) || scheme.group_size != VERIFIED_QUANT_GROUP {
        return Err(EmbeddingError::Unsupported {
            why: format!(
                "quantization {} bits / group {} has not been verified for EmbeddingGemma \
                 (verified: bits {VERIFIED_QUANT_BITS:?}, group {VERIFIED_QUANT_GROUP})",
                scheme.bits, scheme.group_size
            ),
        });
    }
    Ok(Some(scheme))
}
