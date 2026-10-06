//! Strict config parsing for MOSS-Transcribe-Diarize.
//!
//! Reference: `mlx_audio/stt/models/moss_transcribe_diarize/config.py` at
//! mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//! The pinned checkpoint quantizes only the text backbone (the whisper
//! encoder and VQ adaptor stay bfloat16), so the quantization block carries
//! a `text_backbone_only` scope this loader verifies explicitly.

use serde_json::Value;

use crate::stt::qwen3_asr::config::TextConfig;
use crate::{Result, SpeechError};

/// SHA-256 of the pinned `chat_template.jinja`. The reference builds its
/// prompt by rendering this Jinja template through the tokenizer; this port
/// rebuilds the rendered output for the verified no-tools single-user-turn
/// shape and refuses any other template.
pub const PINNED_CHAT_TEMPLATE_SHA256: &str =
    "8641466a16b184ebaf7c4903391e607cfd532ab937e81a64120628ab79d827f4";

#[derive(Debug, Clone, PartialEq)]
pub struct WhisperAudioConfig {
    pub num_mel_bins: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MossConfig {
    pub audio: WhisperAudioConfig,
    pub text: TextConfig,
    pub audio_token_id: i32,
    pub audio_merge_size: usize,
    pub adaptor_input_dim: usize,
    pub sample_rate: usize,
    pub quant_bits: u32,
    pub quant_group_size: usize,
}

/// The `processor_config.json` values that steer time-marker injection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    pub audio_tokens_per_second: f32,
    pub time_marker_every_seconds: usize,
    pub enable_time_marker: bool,
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

impl WhisperAudioConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        let config = Self {
            num_mel_bins: positive(value, "num_mel_bins")?,
            d_model: positive(value, "d_model")?,
            encoder_layers: positive(value, "encoder_layers")?,
            encoder_attention_heads: positive(value, "encoder_attention_heads")?,
            encoder_ffn_dim: positive(value, "encoder_ffn_dim")?,
            max_source_positions: positive(value, "max_source_positions")?,
        };
        if config.d_model % config.encoder_attention_heads != 0 {
            return Err(bad(
                "audio_config",
                "d_model must divide evenly across encoder attention heads",
            ));
        }
        if value
            .get("activation_function")
            .and_then(Value::as_str)
            .is_some_and(|name| name != "gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "MOSS whisper encoder supports only GELU activation".into(),
            });
        }
        if config.num_mel_bins != 80 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "MOSS pins the 80-band whisper frontend, got {} mel bins",
                    config.num_mel_bins
                ),
            });
        }
        Ok(config)
    }
}

impl MossConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("moss_transcribe_diarize") {
            return Err(bad("model_type", "expected moss_transcribe_diarize"));
        }
        let audio_value = root
            .get("audio_config")
            .ok_or_else(|| bad("audio_config", "missing from config.json"))?;
        let audio = WhisperAudioConfig::from_value(audio_value)?;
        let text = TextConfig::from_root(root)?;

        let audio_token_id = root
            .get("audio_token_id")
            .and_then(Value::as_i64)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| bad("audio_token_id", "must be a signed 32-bit token id"))?;

        let audio_merge_size = root
            .get("audio_merge_size")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| bad("audio_merge_size", "must be a positive integer"))
            })
            .transpose()?
            .unwrap_or(4);
        if audio_merge_size != 4 {
            return Err(SpeechError::Unsupported {
                why: format!("MOSS is verified with audio_merge_size 4, got {audio_merge_size}"),
            });
        }

        let adaptor_input_dim = match root.get("adaptor_input_dim") {
            Some(value) => value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad("adaptor_input_dim", "must be a positive integer"))?,
            None => audio.d_model * audio_merge_size,
        };
        if adaptor_input_dim != audio.d_model * audio_merge_size {
            return Err(bad(
                "adaptor_input_dim",
                "must equal d_model * audio_merge_size",
            ));
        }

        if root.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true) {
            return Err(SpeechError::Unsupported {
                why: "MOSS requires tied token embeddings in this port".into(),
            });
        }
        if audio_token_id < 0 {
            return Err(bad("audio_token_id", "must be a non-negative token id"));
        }

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
                why: format!("MOSS expects 16 kHz audio, config declares {sample_rate}"),
            });
        }

        let quant = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"))
            .ok_or_else(|| bad("quantization", "missing from config.json"))?;
        if quant.get("mode").and_then(Value::as_str) != Some("affine") {
            return Err(SpeechError::Unsupported {
                why: "MOSS supports MLX affine groupwise quantization only".into(),
            });
        }
        let bits = positive(quant, "bits")? as u32;
        let quant_group_size = positive(quant, "group_size")?;
        if bits != 4 || quant_group_size != 64 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "MOSS supports the pinned affine 4-bit group-64 backbone, got \
                     {bits}-bit group {quant_group_size}"
                ),
            });
        }
        if let Some(scope) = quant.get("scope").and_then(Value::as_str) {
            if scope != "text_backbone_only" {
                return Err(SpeechError::Unsupported {
                    why: format!("MOSS is verified with scope text_backbone_only, got {scope}"),
                });
            }
        }
        if let Some(excluded) = quant.get("excluded_prefixes").and_then(Value::as_array) {
            let excluded: Vec<&str> = excluded.iter().filter_map(Value::as_str).collect();
            for prefix in ["model.whisper_encoder", "model.vq_adaptor"] {
                if !excluded.contains(&prefix) {
                    return Err(SpeechError::Unsupported {
                        why: format!(
                            "the quantization excluded_prefixes must keep {prefix} \
                             unquantized"
                        ),
                    });
                }
            }
        }

        Ok(Self {
            audio,
            text,
            audio_token_id,
            audio_merge_size,
            adaptor_input_dim,
            sample_rate,
            quant_bits: bits,
            quant_group_size,
        })
    }

    /// Audio frames per merged adaptor row before the whisper encoder
    /// stride: hop 160 times the encoder stride 2 times the merge size.
    pub fn samples_per_audio_token(&self) -> usize {
        160 * 2 * self.audio_merge_size
    }
}

impl ProcessorConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        let audio_tokens_per_second = match value.get("audio_tokens_per_second") {
            Some(raw) => raw
                .as_f64()
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| {
                    bad(
                        "audio_tokens_per_second",
                        "must be a finite positive number",
                    )
                })? as f32,
            None => 12.5,
        };
        let time_marker_every_seconds = match value.get("time_marker_every_seconds") {
            Some(raw) => raw
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| bad("time_marker_every_seconds", "must be a positive integer"))?,
            None => 5,
        };
        let enable_time_marker = value
            .get("enable_time_marker")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        Ok(Self {
            audio_tokens_per_second,
            time_marker_every_seconds,
            enable_time_marker,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pinned_root() -> Value {
        json!({
            "model_type": "moss_transcribe_diarize",
            "text_config": {
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "rope_theta": 1000000,
                "attention_bias": false
            },
            "audio_config": {
                "model_type": "whisper",
                "num_mel_bins": 80,
                "d_model": 1024,
                "encoder_layers": 24,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "max_source_positions": 1500,
                "activation_function": "gelu"
            },
            "audio_token_id": 151671,
            "audio_merge_size": 4,
            "adaptor_input_dim": 4096,
            "tie_word_embeddings": true,
            "sample_rate": 16000,
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "scope": "text_backbone_only",
                "excluded_prefixes": ["model.whisper_encoder", "model.vq_adaptor"]
            }
        })
    }

    #[test]
    fn parses_the_pinned_profile() {
        let config = MossConfig::from_json(&pinned_root()).unwrap();
        assert_eq!(config.audio.d_model, 1024);
        assert_eq!(config.audio.encoder_layers, 24);
        assert_eq!(config.text.hidden_size, 1024);
        assert!(config.text.qk_norm);
        assert!(config.text.tie_word_embeddings);
        assert_eq!(config.audio_token_id, 151671);
        assert_eq!(config.audio_merge_size, 4);
        assert_eq!(config.adaptor_input_dim, 4096);
        assert_eq!(config.quant_bits, 4);
        assert_eq!(config.quant_group_size, 64);
        assert_eq!(config.samples_per_audio_token(), 1280);
    }

    #[test]
    fn defaults_adaptor_input_dim_and_sample_rate() {
        let mut root = pinned_root();
        root.as_object_mut().unwrap().remove("adaptor_input_dim");
        root.as_object_mut().unwrap().remove("sample_rate");
        let config = MossConfig::from_json(&root).unwrap();
        assert_eq!(config.adaptor_input_dim, 4096);
        assert_eq!(config.sample_rate, 16_000);
    }

    #[test]
    fn rejects_wrong_model_type_quantization_and_geometry() {
        let mut root = pinned_root();
        root["model_type"] = json!("qwen3_asr");
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["quantization"]["bits"] = json!(8);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["quantization"]["scope"] = json!("all");
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["quantization"]["excluded_prefixes"] = json!(["model.whisper_encoder"]);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_config"]["num_mel_bins"] = json!(128);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_config"]["activation_function"] = json!("relu");
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["audio_merge_size"] = json!(2);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["tie_word_embeddings"] = json!(false);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["sample_rate"] = json!(24000);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::Unsupported { .. })
        ));

        let mut root = pinned_root();
        root["adaptor_input_dim"] = json!(2048);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root["audio_token_id"] = json!(-1);
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));

        let mut root = pinned_root();
        root.as_object_mut().unwrap().remove("quantization");
        assert!(matches!(
            MossConfig::from_json(&root),
            Err(SpeechError::BadConfig { .. })
        ));
    }

    #[test]
    fn processor_config_defaults_and_rejects_nonpositive_rates() {
        let parsed = ProcessorConfig::from_value(&json!({
            "audio_tokens_per_second": 12.5,
            "audio_merge_size": 4,
            "time_marker_every_seconds": 5,
            "enable_time_marker": true
        }))
        .unwrap();
        assert_eq!(
            parsed,
            ProcessorConfig {
                audio_tokens_per_second: 12.5,
                time_marker_every_seconds: 5,
                enable_time_marker: true
            }
        );
        let defaults = ProcessorConfig::from_value(&json!({})).unwrap();
        assert_eq!(defaults.audio_tokens_per_second, 12.5);
        assert_eq!(defaults.time_marker_every_seconds, 5);
        assert!(defaults.enable_time_marker);
        assert!(matches!(
            ProcessorConfig::from_value(&json!({"audio_tokens_per_second": 0.0})),
            Err(SpeechError::BadConfig { .. })
        ));
        assert!(matches!(
            ProcessorConfig::from_value(&json!({"time_marker_every_seconds": 0})),
            Err(SpeechError::BadConfig { .. })
        ));
    }
}
