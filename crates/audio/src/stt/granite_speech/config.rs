//! Strict config parsing for Granite Speech 1B.
//!
//! Reference: `mlx_audio/stt/models/granite_speech/config.py` at mlx-audio
//! 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The pinned
//! checkpoint `ibm-granite/granite-4.0-1b-speech` ships bfloat16 shards; the
//! loader refuses quantization and any scaling or bias configuration the
//! reference path has not been verified against.

use serde_json::Value;

use crate::{Result, SpeechError};

/// Encoder geometry (`encoder_config` in config.json, mirroring the
/// reference `EncoderConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub input_dim: usize,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub feedforward_mult: usize,
    pub num_heads: usize,
    pub dim_head: usize,
    pub output_dim: usize,
    pub context_size: usize,
    pub max_pos_emb: usize,
    pub conv_kernel_size: usize,
    pub conv_expansion_factor: usize,
}

/// QFormer projector geometry (`projector_config`, mirroring the reference
/// `ProjectorConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectorConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub encoder_hidden_size: usize,
}

/// Granite text backbone geometry (`text_config`, mirroring the reference
/// `TextConfig` including the four Granite multipliers).
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    pub tie_word_embeddings: bool,
}

/// The full model config (`ModelConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechConfig {
    pub encoder: EncoderConfig,
    pub projector: ProjectorConfig,
    pub text: TextConfig,
    pub audio_token_index: i32,
    pub downsample_rate: usize,
    pub window_size: usize,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| bad(field, "must be a positive integer fitting usize"))
}

fn finite_positive(value: &Value, field: &str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite() && *n > 0.0)
        .map(|n| n as f32)
        .ok_or_else(|| bad(field, "must be a finite positive number"))
}

fn finite(value: &Value, field: &str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .map(|n| n as f32)
        .ok_or_else(|| bad(field, "must be a finite number"))
}

impl EncoderConfig {
    fn from_value(value: &Value) -> Result<Self> {
        let config = Self {
            input_dim: positive(value, "input_dim")?,
            num_layers: positive(value, "num_layers")?,
            hidden_dim: positive(value, "hidden_dim")?,
            feedforward_mult: positive(value, "feedforward_mult")?,
            num_heads: positive(value, "num_heads")?,
            dim_head: positive(value, "dim_head")?,
            output_dim: positive(value, "output_dim")?,
            context_size: positive(value, "context_size")?,
            max_pos_emb: positive(value, "max_pos_emb")?,
            conv_kernel_size: positive(value, "conv_kernel_size")?,
            conv_expansion_factor: positive(value, "conv_expansion_factor")?,
        };
        if config.hidden_dim % config.num_heads != 0
            || config.hidden_dim != config.num_heads * config.dim_head
        {
            return Err(bad(
                "encoder_config",
                "hidden_dim must equal num_heads * dim_head",
            ));
        }
        if config.conv_kernel_size % 2 == 0 {
            return Err(bad(
                "encoder_config",
                "conv_kernel_size must be odd for the reference symmetric padding",
            ));
        }
        Ok(config)
    }
}

impl ProjectorConfig {
    fn from_value(value: &Value) -> Result<Self> {
        if value
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|name| name != "gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "the Granite Speech QFormer supports only GELU".into(),
            });
        }
        let config = Self {
            hidden_size: positive(value, "hidden_size")?,
            num_hidden_layers: positive(value, "num_hidden_layers")?,
            num_attention_heads: positive(value, "num_attention_heads")?,
            intermediate_size: positive(value, "intermediate_size")?,
            layer_norm_eps: finite_positive(value, "layer_norm_eps")?,
            encoder_hidden_size: positive(value, "encoder_hidden_size")?,
        };
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(bad(
                "projector_config",
                "hidden_size must divide evenly across attention heads",
            ));
        }
        Ok(config)
    }
}

impl TextConfig {
    fn from_value(value: &Value) -> Result<Self> {
        if value
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|name| name != "silu")
        {
            return Err(SpeechError::Unsupported {
                why: "the Granite text backbone requires the SiLU MLP".into(),
            });
        }
        if value
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(SpeechError::Unsupported {
                why: "the Granite text backbone requires bias-free attention".into(),
            });
        }
        if value
            .get("mlp_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(SpeechError::Unsupported {
                why: "the Granite text backbone requires a bias-free MLP".into(),
            });
        }
        if value
            .get("rope_scaling")
            .is_some_and(|scaling| !scaling.is_null())
        {
            return Err(SpeechError::Unsupported {
                why: "the Granite text backbone is verified without RoPE scaling".into(),
            });
        }
        // rope_parameters is the Granite 4 spelling of the same default rope;
        // anything but the plain default type is refused.
        if let Some(rope) = value.get("rope_parameters") {
            let rope_type = rope
                .get("rope_type")
                .and_then(Value::as_str)
                .unwrap_or("default");
            if rope_type != "default" {
                return Err(SpeechError::Unsupported {
                    why: format!("unsupported rope_type {rope_type} in rope_parameters"),
                });
            }
        }
        let config = Self {
            vocab_size: positive(value, "vocab_size")?,
            hidden_size: positive(value, "hidden_size")?,
            intermediate_size: positive(value, "intermediate_size")?,
            num_hidden_layers: positive(value, "num_hidden_layers")?,
            num_attention_heads: positive(value, "num_attention_heads")?,
            num_key_value_heads: positive(value, "num_key_value_heads")?,
            rms_norm_eps: finite_positive(value, "rms_norm_eps")?,
            rope_theta: finite_positive(value, "rope_theta")?,
            attention_multiplier: finite(value, "attention_multiplier")?,
            embedding_multiplier: finite(value, "embedding_multiplier")?,
            residual_multiplier: finite(value, "residual_multiplier")?,
            logits_scaling: finite_positive(value, "logits_scaling")?,
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        };
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(bad(
                "text_config",
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(bad(
                "text_config",
                "hidden_size must divide evenly across attention heads",
            ));
        }
        Ok(config)
    }
}

impl GraniteSpeechConfig {
    /// Parses the pinned checkpoint's `config.json`.
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("granite_speech") {
            return Err(bad("model_type", "expected granite_speech"));
        }
        if root.get("quantization_config").is_some() {
            return Err(SpeechError::Unsupported {
                why: "no quantized Granite Speech checkpoint is verified; the port refuses a \
                      quantization_config"
                    .into(),
            });
        }
        if root.get("has_lora_adapter").and_then(Value::as_bool) == Some(true) {
            return Err(SpeechError::Unsupported {
                why: "LoRA-adapted Granite Speech checkpoints are not verified".into(),
            });
        }
        let encoder = EncoderConfig::from_value(
            root.get("encoder_config")
                .ok_or_else(|| bad("encoder_config", "missing from config.json"))?,
        )?;
        let projector = ProjectorConfig::from_value(
            root.get("projector_config")
                .ok_or_else(|| bad("projector_config", "missing from config.json"))?,
        )?;
        let text = TextConfig::from_value(
            root.get("text_config")
                .ok_or_else(|| bad("text_config", "missing from config.json"))?,
        )?;
        let audio_token_index = root
            .get("audio_token_index")
            .and_then(Value::as_i64)
            .and_then(|n| i32::try_from(n).ok())
            .filter(|&n| n >= 0)
            .ok_or_else(|| bad("audio_token_index", "must be a non-negative token id"))?;
        let downsample_rate = positive(root, "downsample_rate")?;
        let window_size = positive(root, "window_size")?;
        if window_size % downsample_rate != 0 {
            return Err(bad(
                "window_size",
                "window_size must be divisible by downsample_rate",
            ));
        }
        if encoder.input_dim % 2 != 0 || encoder.input_dim % 80 != 0 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "the paired-mel frontend requires input_dim to be 2 x mel bins, got {}",
                    encoder.input_dim
                ),
            });
        }
        Ok(Self {
            encoder,
            projector,
            text,
            audio_token_index,
            downsample_rate,
            window_size,
        })
    }

    /// Audio placeholder rows per encoder window
    /// (`window_size / downsample_rate`, the reference `num_queries`).
    pub fn num_queries(&self) -> usize {
        self.window_size / self.downsample_rate
    }

    /// Total audio placeholder tokens for `encoder_rows` encoder frames
    /// (reference `_extract_features` tail).
    pub fn audio_token_count(&self, encoder_rows: usize) -> usize {
        encoder_rows.div_ceil(self.window_size) * self.num_queries()
    }
}

#[cfg(test)]
mod tests {
    use super::GraniteSpeechConfig;
    use serde_json::json;

    fn pinned_root() -> serde_json::Value {
        json!({
            "model_type": "granite_speech",
            "audio_token_index": 100352,
            "downsample_rate": 5,
            "window_size": 15,
            "has_lora_adapter": false,
            "encoder_config": {
                "input_dim": 160,
                "num_layers": 16,
                "hidden_dim": 1024,
                "feedforward_mult": 4,
                "num_heads": 8,
                "dim_head": 128,
                "output_dim": 348,
                "context_size": 200,
                "max_pos_emb": 512,
                "conv_kernel_size": 15,
                "conv_expansion_factor": 2
            },
            "projector_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-12,
                "encoder_hidden_size": 1024
            },
            "text_config": {
                "model_type": "granite",
                "vocab_size": 100353,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "attention_bias": false,
                "mlp_bias": false,
                "attention_multiplier": 0.0078125,
                "embedding_multiplier": 12.0,
                "residual_multiplier": 0.22,
                "logits_scaling": 8.0,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "rope_scaling": null,
                "tie_word_embeddings": false
            },
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn parses_the_pinned_profile() {
        let config = GraniteSpeechConfig::from_json(&pinned_root()).unwrap();
        assert_eq!(config.encoder.num_layers, 16);
        assert_eq!(config.encoder.output_dim, 348);
        assert_eq!(config.encoder.context_size, 200);
        assert_eq!(config.encoder.max_pos_emb, 512);
        assert_eq!(config.projector.num_hidden_layers, 2);
        assert_eq!(config.projector.layer_norm_eps, 1e-12);
        assert_eq!(config.text.num_hidden_layers, 40);
        assert_eq!(config.text.num_key_value_heads, 4);
        assert_eq!(config.text.attention_multiplier, 0.0078125);
        assert_eq!(config.text.embedding_multiplier, 12.0);
        assert_eq!(config.text.residual_multiplier, 0.22);
        assert_eq!(config.text.logits_scaling, 8.0);
        assert!(!config.text.tie_word_embeddings);
        assert_eq!(config.audio_token_index, 100_352);
        assert_eq!(config.window_size, 15);
        assert_eq!(config.downsample_rate, 5);
        assert_eq!(config.num_queries(), 3);
        assert_eq!(config.audio_token_count(140), 30);
        assert_eq!(config.audio_token_count(1), 3);
    }

    #[test]
    fn rejects_wrong_family_quantization_lora_and_geometry() {
        let mut root = pinned_root();
        root["model_type"] = json!("granite_speech_nar");
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["quantization_config"] = json!({"bits": 4, "group_size": 64});
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["has_lora_adapter"] = json!(true);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["encoder_config"]["num_layers"] = json!(0);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["encoder_config"]["conv_kernel_size"] = json!(16);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["encoder_config"]["input_dim"] = json!(120);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["projector_config"]["hidden_act"] = json!("relu");
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["attention_bias"] = json!(true);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["mlp_bias"] = json!(true);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["rope_scaling"] = json!({"rope_type": "linear", "factor": 4.0});
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["rope_parameters"] =
            json!({"rope_theta": 10000.0, "rope_type": "llama3"});
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["logits_scaling"] = json!(0.0);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["window_size"] = json!(16);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["audio_token_index"] = json!(-3);
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root.as_object_mut().unwrap().remove("encoder_config");
        assert!(matches!(
            GraniteSpeechConfig::from_json(&root),
            Err(crate::SpeechError::BadConfig { .. })
        ));
    }
}
