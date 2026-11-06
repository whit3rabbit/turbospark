//! Configuration schema for BERT and XLM-RoBERTa encoder models.
//!
//! Deserializes directly from Hugging Face `config.json` for encoder models
//! such as BGE (`model_type: bert`) and Snowflake Arctic Embed (`model_type: xlm-roberta`).

use serde::{Deserialize, Serialize};

/// Quantization specification inside an encoder's `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderQuantization {
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    #[serde(default = "default_bits")]
    pub bits: usize,
}

fn default_group_size() -> usize {
    64
}

fn default_bits() -> usize {
    8
}

/// Parsed architecture configuration for an encoder model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub type_vocab_size: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub quantization: Option<EncoderQuantization>,
}

fn default_model_type() -> String {
    "bert".to_string()
}

fn default_max_positions() -> usize {
    512
}

fn default_vocab_size() -> usize {
    30522
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

/// Ceiling on `hidden_size` and `intermediate_size`. The largest published
/// XLM-RoBERTa (XXL) is 4096 wide with a 16384 MLP, so this only refuses
/// values that would make the loader's zero-filled default bias (a weight
/// file may omit biases) a multi-gigabyte allocation.
pub const MAX_ENCODER_WIDTH: usize = 1 << 18;

/// Ceiling on `num_hidden_layers`, which sizes a `Vec::with_capacity` before
/// any layer tensor is checked. XLM-RoBERTa XXL has 48.
pub const MAX_ENCODER_LAYERS: usize = 1024;

impl EncoderConfig {
    /// Parse from a `config.json` string or bytes, then [`validate`](Self::validate).
    ///
    /// The config comes from a downloaded model dir, so a value that would
    /// panic or over-allocate downstream is a parse error here rather than a
    /// crash later.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        let config: Self = serde_json::from_str(s)?;
        config
            .validate()
            .map_err(<serde_json::Error as serde::de::Error>::custom)?;
        Ok(config)
    }

    /// Structural checks, matching the embeddings crate's BERT config:
    /// positive dims, heads dividing `hidden_size`, `pad_token_id` inside the
    /// vocabulary (which also keeps `position_offset` from overflowing), plus
    /// the allocation ceilings above. `type_vocab_size` may be 0 (no token
    /// type table).
    pub fn validate(&self) -> Result<(), String> {
        for (field, n, max) in [
            ("hidden_size", self.hidden_size, MAX_ENCODER_WIDTH),
            (
                "intermediate_size",
                self.intermediate_size,
                MAX_ENCODER_WIDTH,
            ),
            (
                "num_hidden_layers",
                self.num_hidden_layers,
                MAX_ENCODER_LAYERS,
            ),
            ("num_attention_heads", self.num_attention_heads, usize::MAX),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                usize::MAX,
            ),
            ("vocab_size", self.vocab_size, usize::MAX),
        ] {
            if n == 0 || n > max {
                return Err(format!("{field} {n} outside 1..={max}"));
            }
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(format!(
                "num_attention_heads {} does not divide hidden_size {}",
                self.num_attention_heads, self.hidden_size
            ));
        }
        if self.pad_token_id >= self.vocab_size {
            return Err(format!(
                "pad_token_id {} outside vocab_size {}",
                self.pad_token_id, self.vocab_size
            ));
        }
        Ok(())
    }

    /// Head dimension: `hidden_size / num_attention_heads`.
    ///
    /// Configs from [`from_json_str`](Self::from_json_str) are validated, so
    /// the assert only fires for a hand-built config with zero heads.
    pub fn head_dim(&self) -> usize {
        assert!(
            self.num_attention_heads > 0,
            "num_attention_heads must be > 0"
        );
        self.hidden_size / self.num_attention_heads
    }

    /// Position offset for 1D position embeddings:
    /// - XLM-RoBERTa uses `pad_token_id + 1` (offset 2).
    /// - Standard BERT uses 0.
    pub fn position_offset(&self) -> usize {
        if self.is_roberta() {
            self.pad_token_id + 1
        } else {
            0
        }
    }

    /// Whether this is an XLM-RoBERTa or RoBERTa variant.
    pub fn is_roberta(&self) -> bool {
        self.model_type.eq_ignore_ascii_case("xlm-roberta")
            || self.model_type.eq_ignore_ascii_case("roberta")
            || self
                .architectures
                .iter()
                .any(|a| a.contains("Roberta") || a.contains("XLMRoberta"))
    }

    /// Whether linear weights are 8-bit affine quantized.
    pub fn is_int8_quantized(&self) -> bool {
        self.quantization.is_some_and(|q| q.bits == 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bge_small_config() {
        let json = r#"{
            "architectures": ["BertModel"],
            "hidden_size": 384,
            "intermediate_size": 1536,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        }"#;
        let cfg = EncoderConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.head_dim(), 32);
        assert_eq!(cfg.position_offset(), 0);
        assert!(!cfg.is_roberta());
        assert!(!cfg.is_int8_quantized());
    }

    #[test]
    fn parse_snowflake_arctic_config() {
        let json = r#"{
            "architectures": ["XLMRobertaModel"],
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 8194,
            "model_type": "xlm-roberta",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 1,
            "quantization": {
                "group_size": 64,
                "bits": 8
            },
            "type_vocab_size": 1,
            "vocab_size": 250002
        }"#;
        let cfg = EncoderConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.position_offset(), 2);
        assert!(cfg.is_roberta());
        assert!(cfg.is_int8_quantized());
        assert_eq!(cfg.quantization.unwrap().group_size, 64);
    }

    /// A downloaded `config.json` is untrusted. Each case would otherwise
    /// parse and reach a panic (remainder by zero in the compute block, the
    /// `head_dim` assert, `pad_token_id + 1` overflow) or a huge zero-filled
    /// bias allocation when the weight file omits that bias.
    #[test]
    fn rejects_dims_that_panic_or_overallocate_downstream() {
        let base = |patch: &str| {
            format!(
                r#"{{"model_type":"bert","hidden_size":384,"num_hidden_layers":12,
                "num_attention_heads":12,"intermediate_size":1536,{patch}"vocab_size":30522}}"#
            )
        };
        assert!(EncoderConfig::from_json_str(&base("")).is_ok());
        for (patch, field) in [
            (r#""num_attention_heads":0,"#, "num_attention_heads"),
            (r#""num_attention_heads":7,"#, "num_attention_heads"),
            (r#""hidden_size":0,"#, "hidden_size"),
            (r#""intermediate_size":0,"#, "intermediate_size"),
            (r#""num_hidden_layers":0,"#, "num_hidden_layers"),
            (r#""max_position_embeddings":0,"#, "max_position_embeddings"),
            (r#""hidden_size":4294967296,"#, "hidden_size"),
            (r#""intermediate_size":4294967296,"#, "intermediate_size"),
            (r#""num_hidden_layers":4294967296,"#, "num_hidden_layers"),
            (r#""pad_token_id":18446744073709551615,"#, "pad_token_id"),
        ] {
            // A later duplicate key wins in serde_json, so the patch must
            // replace the field rather than repeat it.
            let json = base(patch);
            let json = dedupe_first(&json, field);
            let err = EncoderConfig::from_json_str(&json)
                .expect_err(&format!("{patch} should be refused"));
            assert!(err.to_string().contains(field), "{patch}: {err}");
        }
    }

    /// Drops the first `"field":value,` occurrence so the patched one is the
    /// only one left.
    fn dedupe_first(json: &str, field: &str) -> String {
        let key = format!("\"{field}\":");
        let first = json.find(&key).unwrap();
        match json[first + key.len()..].find(&key) {
            None => json.to_string(),
            Some(_) => {
                let end = first + json[first..].find(',').unwrap() + 1;
                format!("{}{}", &json[..first], &json[end..])
            }
        }
    }
}
