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

impl EncoderConfig {
    /// Parse from a `config.json` string or bytes.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Head dimension: `hidden_size / num_attention_heads`.
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
}
