//! `config.json` for BERT and XLM-RoBERTa encoders.
//!
//! Parsed by hand from `serde_json::Value` so every refusal names the
//! field. Only what the port implements is accepted: absolute positions,
//! an encoder (no decoder, no cross-attention), and erf GELU. Anything
//! else is an error, because ignoring it would produce fluent but wrong
//! embeddings.

use serde_json::Value;
use turbospark_audio::quant::QuantScheme;

use crate::error::{bad_config, EmbeddingError};
use crate::weights::quant_scheme;

/// Which embedding-position rule applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// `model_type = "bert"`: positions `0..n`.
    Bert,
    /// `model_type = "xlm-roberta"`: `create_position_ids_from_input_ids`
    /// (a pad id keeps `pad_token_id`, others count from `pad + 1`).
    XlmRoberta,
}

/// The encoder hyperparameters this port consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct BertConfig {
    pub arch: Arch,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    /// Matryoshka widths the checkpoint advertises (`matryoshka_dimensions`,
    /// a Snowflake extension). Informational: any truncation is allowed.
    pub matryoshka_dimensions: Vec<usize>,
    /// MLX affine quantization, when the checkpoint is an MLX conversion.
    pub quantization: Option<QuantScheme>,
}

/// MLX schemes measured against a same-weights golden (the pinned
/// mlx-community arctic-embed-l-v2.0 4/6/8-bit conversions).
pub const VERIFIED_QUANT_BITS: &[u32] = &[4, 6, 8];
pub const VERIFIED_QUANT_GROUP: usize = 64;

fn usize_field(v: &Value, key: &str) -> Result<usize, EmbeddingError> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| bad_config(key, "missing or not a non-negative integer"))?;
    usize::try_from(n).map_err(|_| bad_config(key, "does not fit usize"))
}

impl BertConfig {
    pub fn from_json(v: &Value) -> Result<Self, EmbeddingError> {
        let arch = match v.get("model_type").and_then(Value::as_str) {
            Some("bert") => Arch::Bert,
            Some("xlm-roberta") => Arch::XlmRoberta,
            other => {
                return Err(bad_config(
                    "model_type",
                    format!("expected bert or xlm-roberta, got {other:?}"),
                ))
            }
        };
        // transformers defaults to "absolute" when the key is absent.
        match v.get("position_embedding_type") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s == "absolute" => {}
            Some(other) => {
                return Err(EmbeddingError::Unsupported {
                    why: format!("position_embedding_type {other}; only absolute is implemented"),
                })
            }
        }
        for key in ["is_decoder", "add_cross_attention"] {
            if v.get(key).and_then(Value::as_bool).unwrap_or(false) {
                return Err(EmbeddingError::Unsupported {
                    why: format!("{key} = true; only a bidirectional encoder is implemented"),
                });
            }
        }
        // ACT2FN["gelu"] is torch's exact erf GELU. The tanh variants
        // (gelu_new, gelu_pytorch_tanh) are not exercised by any pin.
        match v.get("hidden_act").and_then(Value::as_str) {
            Some("gelu") => {}
            other => {
                return Err(EmbeddingError::Unsupported {
                    why: format!("hidden_act {other:?}; only gelu (erf) is implemented"),
                })
            }
        }
        let eps = v
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .ok_or_else(|| bad_config("layer_norm_eps", "missing or not a number"))?;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(bad_config("layer_norm_eps", "must be positive and finite"));
        }
        let pad = usize_field(v, "pad_token_id")?;
        let matryoshka_dimensions = match v.get("matryoshka_dimensions") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| bad_config("matryoshka_dimensions", "not an integer"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(bad_config("matryoshka_dimensions", "not an array")),
        };
        let config = Self {
            arch,
            vocab_size: usize_field(v, "vocab_size")?,
            hidden_size: usize_field(v, "hidden_size")?,
            num_hidden_layers: usize_field(v, "num_hidden_layers")?,
            num_attention_heads: usize_field(v, "num_attention_heads")?,
            intermediate_size: usize_field(v, "intermediate_size")?,
            max_position_embeddings: usize_field(v, "max_position_embeddings")?,
            type_vocab_size: usize_field(v, "type_vocab_size")?,
            layer_norm_eps: eps as f32,
            pad_token_id: u32::try_from(pad)
                .map_err(|_| bad_config("pad_token_id", "too large"))?,
            matryoshka_dimensions,
            quantization: verified_quant(v)?,
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
            ("type_vocab_size", self.type_vocab_size),
        ] {
            if n == 0 {
                return Err(bad_config(k, "must be positive"));
            }
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
        if self.pad_token_id as usize >= self.vocab_size {
            return Err(bad_config("pad_token_id", "outside the vocabulary"));
        }
        if self.max_tokens() == 0 {
            return Err(bad_config(
                "max_position_embeddings",
                "leaves no position for a real token",
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

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Longest non-pad token run the position table covers. XLM-R spends
    /// rows `0..=pad` on the offset, so 8194 rows hold 8192 tokens.
    pub fn max_tokens(&self) -> usize {
        match self.arch {
            Arch::Bert => self.max_position_embeddings,
            Arch::XlmRoberta => self
                .max_position_embeddings
                .saturating_sub(self.pad_token_id as usize + 1),
        }
    }

    /// Position ids for `ids` under this architecture's rule.
    pub fn position_ids(&self, ids: &[u32]) -> Vec<u32> {
        match self.arch {
            Arch::Bert => (0..ids.len() as u32).collect(),
            Arch::XlmRoberta => xlmr_position_ids(ids, self.pad_token_id),
        }
    }
}

/// transformers `create_position_ids_from_input_ids`:
/// `cumsum(ids != pad) * (ids != pad) + pad`. A pad id keeps position
/// `pad` and does not advance the count.
pub fn xlmr_position_ids(ids: &[u32], pad: u32) -> Vec<u32> {
    let mut count = 0u32;
    ids.iter()
        .map(|&id| {
            if id == pad {
                pad
            } else {
                count += 1;
                count + pad
            }
        })
        .collect()
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
                "quantization {} bits / group {} has not been verified for BERT \
                 (verified: bits {VERIFIED_QUANT_BITS:?}, group {VERIFIED_QUANT_GROUP})",
                scheme.bits, scheme.group_size
            ),
        });
    }
    Ok(Some(scheme))
}
