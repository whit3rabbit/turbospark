//! Strict config parsing for the Qwen3 audio tower and text decoder.

use serde_json::Value;

use crate::{Result, SpeechError};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AudioEncoderConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub d_model: usize,
    pub max_source_positions: usize,
    pub n_window: usize,
    pub output_dim: usize,
    pub n_window_infer: usize,
    pub downsample_hidden_size: usize,
    pub scale_embedding: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3Config {
    pub audio: AudioEncoderConfig,
    pub text: TextConfig,
    pub audio_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub supported_languages: Vec<String>,
    pub quant_bits: u32,
    pub quant_group_size: usize,
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

fn number(value: &Value, field: &str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite() && *n > 0.0 && *n <= f32::MAX as f64)
        .map(|n| n as f32)
        .ok_or_else(|| bad(field, "must be a finite positive number"))
}

fn bool_value(value: &Value, field: &str) -> Result<bool> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| bad(field, "must be a boolean"))
}

fn thinker_config(root: &Value) -> &Value {
    root.get("thinker_config").unwrap_or(root)
}

impl AudioEncoderConfig {
    pub(crate) fn from_root(root: &Value) -> Result<Self> {
        let thinker = thinker_config(root);
        let value = thinker
            .get("audio_config")
            .or_else(|| root.get("audio_config"))
            .ok_or_else(|| bad("audio_config", "missing from config.json"))?;
        let config = Self {
            num_mel_bins: positive(value, "num_mel_bins")?,
            encoder_layers: value
                .get("encoder_layers")
                .and_then(Value::as_u64)
                .or_else(|| value.get("num_hidden_layers").and_then(Value::as_u64))
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| bad("audio_config.encoder_layers", "must be positive"))?,
            encoder_attention_heads: positive(value, "encoder_attention_heads")?,
            encoder_ffn_dim: positive(value, "encoder_ffn_dim")?,
            d_model: positive(value, "d_model")?,
            max_source_positions: positive(value, "max_source_positions")?,
            n_window: positive(value, "n_window")?,
            output_dim: positive(value, "output_dim")?,
            n_window_infer: positive(value, "n_window_infer")?,
            downsample_hidden_size: positive(value, "downsample_hidden_size")?,
            scale_embedding: value
                .get("scale_embedding")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        if config.num_mel_bins != 128
            || config.d_model % config.encoder_attention_heads != 0
            || config.d_model % 2 != 0
            || config.n_window.checked_mul(2).is_none()
            || config.n_window_infer % (2 * config.n_window) != 0
            || config.max_source_positions < config.n_window_infer / 2
        {
            return Err(bad(
                "audio_config",
                "unsupported Qwen3 audio dimensions or window geometry",
            ));
        }
        if value
            .get("activation_function")
            .and_then(Value::as_str)
            .is_some_and(|name| name != "gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "Qwen3 audio encoder supports only GELU activation".into(),
            });
        }
        Ok(config)
    }
}

impl TextConfig {
    fn from_root(root: &Value) -> Result<Self> {
        let thinker = thinker_config(root);
        let value = thinker
            .get("text_config")
            .or_else(|| root.get("text_config"))
            .ok_or_else(|| bad("text_config", "missing from config.json"))?;
        let config = Self {
            vocab_size: positive(value, "vocab_size")?,
            hidden_size: positive(value, "hidden_size")?,
            intermediate_size: positive(value, "intermediate_size")?,
            num_hidden_layers: positive(value, "num_hidden_layers")?,
            num_attention_heads: positive(value, "num_attention_heads")?,
            num_key_value_heads: positive(value, "num_key_value_heads")?,
            head_dim: positive(value, "head_dim")?,
            rms_norm_eps: number(value, "rms_norm_eps")?,
            rope_theta: number(value, "rope_theta")?,
            tie_word_embeddings: bool_value(value, "tie_word_embeddings")?,
        };
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(bad("text_config", "unsupported Qwen3 decoder dimensions"));
        }
        if value.get("attention_bias").and_then(Value::as_bool) == Some(true)
            || value
                .get("hidden_act")
                .and_then(Value::as_str)
                .is_some_and(|name| name != "silu")
        {
            return Err(SpeechError::Unsupported {
                why: "Qwen3 decoder requires bias-free projections and SiLU MLP".into(),
            });
        }
        Ok(config)
    }
}

impl Qwen3Config {
    pub(crate) fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("qwen3_asr") {
            return Err(bad("model_type", "expected qwen3_asr"));
        }
        let nested = thinker_config(root);
        let quant = root
            .get("quantization_config")
            .ok_or_else(|| bad("quantization_config", "missing from config.json"))?;
        let bits = quant
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|bits| matches!(bits, 2 | 3 | 4 | 5 | 6 | 8))
            .ok_or_else(|| bad("quantization_config.bits", "unsupported affine bit width"))?;
        let group_size = positive(quant, "group_size")?;
        if quant.get("mode").and_then(Value::as_str) != Some("affine") {
            return Err(SpeechError::Unsupported {
                why: "Qwen3 supports MLX affine groupwise quantization only".into(),
            });
        }
        let supported_languages = nested
            .get("support_languages")
            .or_else(|| root.get("support_languages"))
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let token_id = |key: &str| -> Result<i32> {
            nested
                .get(key)
                .or_else(|| root.get(key))
                .and_then(Value::as_i64)
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| bad(key, "must be a signed 32-bit token id"))
        };
        let config = Self {
            audio: AudioEncoderConfig::from_root(root)?,
            text: TextConfig::from_root(root)?,
            audio_token_id: token_id("audio_token_id")?,
            audio_start_token_id: token_id("audio_start_token_id")?,
            audio_end_token_id: token_id("audio_end_token_id")?,
            supported_languages,
            quant_bits: bits,
            quant_group_size: group_size,
        };
        if config.audio.output_dim != config.text.hidden_size
            || config.audio_token_id < 0
            || config.audio_start_token_id < 0
            || config.audio_end_token_id < 0
        {
            return Err(bad(
                "audio_config.output_dim",
                "audio output width must equal decoder hidden size and token ids must be valid",
            ));
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_the_pinned_profile_and_rejects_unverified_audio_geometry() {
        let root = json!({
            "model_type":"qwen3_asr",
            "thinker_config":{
                "audio_token_id":151676,"audio_start_token_id":151669,"audio_end_token_id":151670,
                "support_languages":["English"],
                "audio_config":{"num_mel_bins":128,"encoder_layers":18,"encoder_attention_heads":14,"encoder_ffn_dim":3584,"d_model":896,"max_source_positions":1500,"n_window":50,"output_dim":1024,"n_window_infer":800,"downsample_hidden_size":480,"scale_embedding":false,"activation_function":"gelu"},
                "text_config":{"vocab_size":151936,"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":true,"attention_bias":false,"hidden_act":"silu"}
            },
            "quantization_config":{"bits":8,"group_size":64,"mode":"affine"}
        });
        let parsed = Qwen3Config::from_json(&root).unwrap();
        assert_eq!(parsed.audio.d_model, 896);
        assert_eq!(parsed.audio.output_dim, parsed.text.hidden_size);
        assert_eq!(parsed.quant_bits, 8);
        assert_eq!(parsed.quant_group_size, 64);

        let mut unsupported = root;
        unsupported["thinker_config"]["audio_config"]["num_mel_bins"] = json!(80);
        assert!(matches!(
            Qwen3Config::from_json(&unsupported),
            Err(SpeechError::BadConfig { .. })
        ));
    }
}
