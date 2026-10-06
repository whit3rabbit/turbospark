//! Strict config parsing for Higgs Audio v3 STT.
//!
//! Reference: `mlx_audio/stt/models/higgs_audio_3/config.py` at mlx-audio
//! 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The pinned
//! checkpoint is an unquantized bfloat16 conversion of
//! `bosonai/higgs-audio-v3-stt`; the loader refuses any quantization block
//! because no quantized variant has been verified.

use serde_json::Value;

use crate::stt::qwen3_asr::config::TextConfig;
use crate::{Result, SpeechError};

#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub d_model: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsConfig {
    pub audio: AudioEncoderConfig,
    pub text: TextConfig,
    pub audio_in_token_idx: i32,
    pub audio_out_token_idx: i32,
    pub audio_eos_token_id: i32,
    pub projector_temporal_downsample: usize,
    pub chunk_size_seconds: f32,
    pub sample_rate: usize,
    pub vad_cut: bool,
    pub split_vads: bool,
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

fn token_id(value: &Value, field: &str) -> Result<i32> {
    value
        .get(field)
        .and_then(Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| bad(field, "must be a signed 32-bit token id"))
}

impl AudioEncoderConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        let config = Self {
            num_mel_bins: positive(value, "num_mel_bins")?,
            encoder_layers: positive(value, "encoder_layers")?,
            encoder_attention_heads: positive(value, "encoder_attention_heads")?,
            encoder_ffn_dim: positive(value, "encoder_ffn_dim")?,
            d_model: positive(value, "d_model")?,
            max_source_positions: positive(value, "max_source_positions")?,
        };
        if config.d_model % config.encoder_attention_heads != 0 {
            return Err(bad(
                "audio_encoder_config",
                "d_model must divide evenly across encoder attention heads",
            ));
        }
        if config.num_mel_bins != 128 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Higgs Audio v3 pins the 128-band whisper frontend, got {} mel bins",
                    config.num_mel_bins
                ),
            });
        }
        if value
            .get("activation_function")
            .and_then(Value::as_str)
            .is_some_and(|name| name != "gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "Higgs Audio v3 encoder supports only GELU activation".into(),
            });
        }
        Ok(config)
    }
}

/// Parses the `text_config` block into the shared qwen3 decoder config. The
/// checkpoint carries a separate LM head, and the reference forces
/// `tie_word_embeddings` off for the backbone regardless of the HF config
/// flag, so this loader always builds an untied decoder config.
fn text_config(value: &Value) -> Result<TextConfig> {
    if value
        .get("hidden_act")
        .and_then(Value::as_str)
        .is_some_and(|name| name != "silu")
    {
        return Err(SpeechError::Unsupported {
            why: "Higgs Audio v3 decoder requires the SiLU MLP".into(),
        });
    }
    if value
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(SpeechError::Unsupported {
            why: "Higgs Audio v3 decoder requires bias-free projections".into(),
        });
    }
    if value
        .get("rope_scaling")
        .is_some_and(|scaling| !scaling.is_null())
    {
        return Err(SpeechError::Unsupported {
            why: "Higgs Audio v3 decoder is verified without RoPE scaling".into(),
        });
    }
    let head_dim = positive(value, "head_dim")?;
    let config = TextConfig {
        vocab_size: positive(value, "vocab_size")?,
        hidden_size: positive(value, "hidden_size")?,
        intermediate_size: positive(value, "intermediate_size")?,
        num_hidden_layers: positive(value, "num_hidden_layers")?,
        num_attention_heads: positive(value, "num_attention_heads")?,
        num_key_value_heads: positive(value, "num_key_value_heads")?,
        head_dim,
        rotary_dim: head_dim,
        rms_norm_eps: value
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| {
                bad(
                    "text_config.rms_norm_eps",
                    "must be a finite positive number",
                )
            })? as f32,
        rope_theta: value
            .get("rope_theta")
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| bad("text_config.rope_theta", "must be a finite positive number"))?
            as f32,
        qk_norm: true,
        // The reference passes tie_word_embeddings: False to the Qwen3
        // backbone and the checkpoint ships the separate text LM head.
        tie_word_embeddings: false,
    };
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err(bad("text_config", "unsupported decoder dimensions"));
    }
    Ok(config)
}

impl HiggsConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("higgs_audio_3") {
            return Err(bad("model_type", "expected higgs_audio_3"));
        }
        if root.get("quantization_config").is_some() {
            return Err(SpeechError::Unsupported {
                why: "no quantized Higgs Audio v3 checkpoint is verified; the port refuses a \
                      quantization_config"
                    .into(),
            });
        }
        let audio_value = root
            .get("audio_encoder_config")
            .ok_or_else(|| bad("audio_encoder_config", "missing from config.json"))?;
        let audio = AudioEncoderConfig::from_value(audio_value)?;
        let text_value = root
            .get("text_config")
            .ok_or_else(|| bad("text_config", "missing from config.json"))?;
        let text = text_config(text_value)?;

        let downsample = positive(root, "projector_temporal_downsample")?;
        if downsample != 2 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Higgs Audio v3 is verified with projector_temporal_downsample 2, got \
                     {downsample}"
                ),
            });
        }
        // The reference ModelConfig defaults sample_rate to 16000 when the
        // checkpoint config omits it, as the pinned one does.
        let sample_rate = root
            .get("sample_rate")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| bad("sample_rate", "must be a positive integer"))
            })
            .transpose()?
            .unwrap_or(16_000);
        if sample_rate != 16_000 {
            return Err(SpeechError::Unsupported {
                why: format!("Higgs Audio v3 expects 16 kHz audio, config declares {sample_rate}"),
            });
        }
        let chunk_size_seconds = root
            .get("chunk_size_seconds")
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| bad("chunk_size_seconds", "must be a finite positive number"))?
            as f32;
        let config = Self {
            audio,
            text,
            audio_in_token_idx: token_id(root, "audio_in_token_idx")?,
            audio_out_token_idx: token_id(root, "audio_out_token_idx")?,
            audio_eos_token_id: token_id(root, "audio_eos_token_id")?,
            projector_temporal_downsample: downsample,
            chunk_size_seconds,
            sample_rate,
            vad_cut: root
                .get("vad_cut")
                .and_then(Value::as_bool)
                .ok_or_else(|| bad("vad_cut", "must be a boolean"))?,
            split_vads: root
                .get("split_vads")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        if config.audio_in_token_idx < 0
            || config.audio_out_token_idx < 0
            || config.audio_eos_token_id < 0
        {
            return Err(bad("audio token ids", "must be non-negative"));
        }
        Ok(config)
    }

    /// VAD chunk length in samples at the configured sample rate
    /// (`chunk_size_seconds * sample_rate`, truncated).
    pub fn chunk_samples(&self) -> usize {
        (self.chunk_size_seconds * self.sample_rate as f32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pinned_root() -> Value {
        json!({
            "model_type": "higgs_audio_3",
            "audio_in_token_idx": 151672,
            "audio_out_token_idx": 151673,
            "audio_eos_token_id": 151670,
            "projector_temporal_downsample": 2,
            "chunk_size_seconds": 4.0,
            "sample_rate": 16000,
            "vad_cut": true,
            "split_vads": false,
            "audio_encoder_config": {
                "num_mel_bins": 128,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "d_model": 1280,
                "max_source_positions": 1500,
                "activation_function": "gelu"
            },
            "text_config": {
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "rope_scaling": null,
                "attention_bias": false,
                "tie_word_embeddings": true
            }
        })
    }

    #[test]
    fn parses_the_pinned_profile() {
        let config = HiggsConfig::from_json(&pinned_root()).unwrap();
        assert_eq!(config.audio.d_model, 1280);
        assert_eq!(config.audio.encoder_layers, 32);
        assert_eq!(config.audio.encoder_attention_heads, 20);
        assert_eq!(config.audio.encoder_ffn_dim, 5120);
        assert_eq!(config.audio.max_source_positions, 1500);
        assert_eq!(config.text.hidden_size, 2048);
        assert_eq!(config.text.num_hidden_layers, 28);
        assert_eq!(config.text.head_dim, 128);
        assert_eq!(config.text.rotary_dim, 128);
        assert!(config.text.qk_norm);
        // The reference forces the untied backbone even though the HF config
        // flag says tied; the separate LM head tensor is load-bearing.
        assert!(!config.text.tie_word_embeddings);
        assert_eq!(config.text.rope_theta, 1_000_000.0);
        assert_eq!(config.audio_in_token_idx, 151_672);
        assert_eq!(config.audio_out_token_idx, 151_673);
        assert_eq!(config.audio_eos_token_id, 151_670);
        assert_eq!(config.chunk_samples(), 64_000);
        assert!(config.vad_cut);
        assert!(!config.split_vads);
    }

    #[test]
    fn defaults_sample_rate_and_split_vads_like_the_reference() {
        let mut root = pinned_root();
        root.as_object_mut().unwrap().remove("sample_rate");
        root.as_object_mut().unwrap().remove("split_vads");
        let config = HiggsConfig::from_json(&root).unwrap();
        assert_eq!(config.sample_rate, 16_000);
        assert!(!config.split_vads);
    }

    #[test]
    fn rejects_wrong_model_type_quantization_and_geometry() {
        let mut root = pinned_root();
        root["model_type"] = json!("higgs_audio_2");
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["quantization_config"] = json!({"bits": 4, "group_size": 64});
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_encoder_config"]["num_mel_bins"] = json!(80);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_encoder_config"]["activation_function"] = json!("relu");
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_encoder_config"]["d_model"] = json!(1281);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["projector_temporal_downsample"] = json!(1);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["sample_rate"] = json!(24000);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["chunk_size_seconds"] = json!(0.0);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["vad_cut"] = json!("yes");
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["audio_in_token_idx"] = json!(-1);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["hidden_act"] = json!("gelu");
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["attention_bias"] = json!(true);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["rope_scaling"] = json!({"type": "linear", "factor": 2.0});
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["text_config"]["rms_norm_eps"] = json!(0.0);
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root.as_object_mut().unwrap().remove("audio_encoder_config");
        assert!(matches!(
            HiggsConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));
    }
}
