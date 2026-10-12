//! Dia configuration (`config.py`): the data, encoder, decoder, model,
//! and training sections with the reference defaults.

use serde_json::Value;

use super::layers::Act;
use crate::{Result, SpeechError};

/// `DataConfig`: sequence budgets, special audio token values, and the
/// per-channel delay pattern. `text_length` and `audio_length` round up
/// to a multiple of 128 exactly like the reference `__post_init__`.
#[derive(Debug, Clone)]
pub struct DataConfig {
    pub text_length: usize,
    pub audio_length: usize,
    pub channels: usize,
    pub text_pad_value: i32,
    pub audio_eos_value: i32,
    pub audio_pad_value: i32,
    pub audio_bos_value: i32,
    pub delay_pattern: Vec<usize>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            text_length: 0,
            audio_length: 0,
            channels: 9,
            text_pad_value: 0,
            audio_eos_value: 1024,
            audio_pad_value: 1025,
            audio_bos_value: 1026,
            delay_pattern: vec![0, 8, 9, 10, 11, 12, 13, 14, 15],
        }
    }
}

impl DataConfig {
    fn from_json(value: &Value) -> Result<Self> {
        let mut config = Self::default();
        let usize_of = |key: &str| -> Result<usize> {
            crate::nn::json::usize_field(value, key).map_err(|_| SpeechError::BadConfig {
                field: key.to_owned(),
                why: "data config field is missing or not an integer".into(),
            })
        };
        config.text_length = round_128(usize_of("text_length")?);
        config.audio_length = round_128(usize_of("audio_length")?);
        if let Some(channels) = opt_usize(value, "channels") {
            config.channels = channels;
        }
        if let Some(pad) = opt_i64(value, "text_pad_value") {
            config.text_pad_value = pad as i32;
        }
        if let Some(v) = opt_i64(value, "audio_eos_value") {
            config.audio_eos_value = v as i32;
        }
        if let Some(v) = opt_i64(value, "audio_pad_value") {
            config.audio_pad_value = v as i32;
        }
        if let Some(v) = opt_i64(value, "audio_bos_value") {
            config.audio_bos_value = v as i32;
        }
        if let Some(list) = value.get("delay_pattern").and_then(Value::as_array) {
            let pattern: Vec<usize> = list
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "delay_pattern".into(),
                    why: "expected a list of integers".into(),
                })?;
            if pattern.len() != config.channels {
                return Err(SpeechError::BadConfig {
                    field: "delay_pattern".into(),
                    why: format!("{} delays for {} channels", pattern.len(), config.channels),
                });
            }
            config.delay_pattern = pattern;
        }
        Ok(config)
    }
}

/// `EncoderConfig` / `DecoderConfig` shared shape fields.
#[derive(Debug, Clone)]
pub struct StackConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_hidden: usize,
    pub mlp_activations: [Act; 2],
    pub use_pre_norm: bool,
}

impl StackConfig {
    fn base(value: &Value) -> Result<Self> {
        let layer = |key: &str| -> Result<usize> {
            crate::nn::json::usize_field(value, key).map_err(|_| SpeechError::BadConfig {
                field: key.to_owned(),
                why: "stack config field is missing or not an integer".into(),
            })
        };
        let mut activations = [Act::Silu, Act::Linear];
        if let Some(list) = value.get("mlp_activations").and_then(Value::as_array) {
            let names: Vec<&str> = list.iter().filter_map(Value::as_str).collect();
            if names.len() == 2 {
                activations = [Act::parse(names[0])?, Act::parse(names[1])?];
            }
        }
        Ok(Self {
            n_layer: layer("n_layer")?,
            n_embd: layer("n_embd")?,
            n_hidden: layer("n_hidden")?,
            mlp_activations: activations,
            use_pre_norm: value
                .get("use_pre_norm")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// The encoder attention head geometry.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub stack: StackConfig,
    pub n_head: usize,
    pub head_dim: usize,
}

impl EncoderConfig {
    fn from_json(value: &Value) -> Result<Self> {
        let layer = |key: &str| -> Result<usize> {
            crate::nn::json::usize_field(value, key).map_err(|_| SpeechError::BadConfig {
                field: key.to_owned(),
                why: "encoder config field is missing or not an integer".into(),
            })
        };
        Ok(Self {
            stack: StackConfig::base(value)?,
            n_head: layer("n_head")?,
            head_dim: layer("head_dim")?,
        })
    }
}

/// The decoder attention head geometry (grouped-query self attention and
/// full multi-head cross attention).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub stack: StackConfig,
    pub gqa_query_heads: usize,
    pub kv_heads: usize,
    pub gqa_head_dim: usize,
    pub cross_query_heads: usize,
    pub cross_head_dim: usize,
}

impl DecoderConfig {
    fn from_json(value: &Value) -> Result<Self> {
        let layer = |key: &str| -> Result<usize> {
            crate::nn::json::usize_field(value, key).map_err(|_| SpeechError::BadConfig {
                field: key.to_owned(),
                why: "decoder config field is missing or not an integer".into(),
            })
        };
        Ok(Self {
            stack: StackConfig::base(value)?,
            gqa_query_heads: layer("gqa_query_heads")?,
            kv_heads: layer("kv_heads")?,
            gqa_head_dim: layer("gqa_head_dim")?,
            cross_query_heads: layer("cross_query_heads")?,
            cross_head_dim: layer("cross_head_dim")?,
        })
    }
}

/// `ModelConfig`: the two stacks plus the shared vocabulary, rope, and
/// normalization settings.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
    pub dropout: f32,
    pub normalization_layer_epsilon: f32,
    pub rope_min_timescale: i32,
    pub rope_max_timescale: i32,
    pub sample_rate: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig {
                stack: StackConfig {
                    n_layer: 0,
                    n_embd: 0,
                    n_hidden: 0,
                    mlp_activations: [Act::Silu, Act::Linear],
                    use_pre_norm: false,
                },
                n_head: 0,
                head_dim: 0,
            },
            decoder: DecoderConfig {
                stack: StackConfig {
                    n_layer: 0,
                    n_embd: 0,
                    n_hidden: 0,
                    mlp_activations: [Act::Silu, Act::Linear],
                    use_pre_norm: false,
                },
                gqa_query_heads: 0,
                kv_heads: 0,
                gqa_head_dim: 0,
                cross_query_heads: 0,
                cross_head_dim: 0,
            },
            src_vocab_size: 128,
            tgt_vocab_size: 1028,
            dropout: 0.0,
            normalization_layer_epsilon: 1.0e-5,
            rope_min_timescale: 1,
            rope_max_timescale: 10_000,
            sample_rate: 44_100,
        }
    }
}

impl ModelConfig {
    fn from_json(value: &Value) -> Result<Self> {
        let mut config = Self::default();
        let model = value.get("model").ok_or_else(|| SpeechError::BadConfig {
            field: "model".into(),
            why: "config has no model section".into(),
        })?;
        let encoder = model.get("encoder").ok_or_else(|| SpeechError::BadConfig {
            field: "model.encoder".into(),
            why: "model section has no encoder".into(),
        })?;
        let decoder = model.get("decoder").ok_or_else(|| SpeechError::BadConfig {
            field: "model.decoder".into(),
            why: "model section has no decoder".into(),
        })?;
        config.encoder = EncoderConfig::from_json(encoder)?;
        config.decoder = DecoderConfig::from_json(decoder)?;
        if let Some(v) = opt_usize(model, "src_vocab_size") {
            config.src_vocab_size = v;
        }
        if let Some(v) = opt_usize(model, "tgt_vocab_size") {
            config.tgt_vocab_size = v;
        }
        if let Some(v) = model.get("dropout").and_then(Value::as_f64) {
            config.dropout = v as f32;
        }
        if let Some(v) = model
            .get("normalization_layer_epsilon")
            .and_then(Value::as_f64)
        {
            config.normalization_layer_epsilon = v as f32;
        }
        if let Some(v) = opt_i64(model, "rope_min_timescale") {
            config.rope_min_timescale = v as i32;
        }
        if let Some(v) = opt_i64(model, "rope_max_timescale") {
            config.rope_max_timescale = v as i32;
        }
        if let Some(v) = opt_usize(model, "sample_rate") {
            config.sample_rate = v as u32;
        }
        Ok(config)
    }
}

/// The master `DiaConfig`. The `training` section is parsed but the port
/// computes in f32 only: a `training.dtype` other than `float32` is a
/// refusal (the reference would compute activations in that dtype, which
/// a portable f32 port cannot reproduce).
#[derive(Debug, Clone)]
pub struct DiaConfig {
    pub model: ModelConfig,
    pub data: DataConfig,
    pub version: String,
}

impl DiaConfig {
    /// Parses the checkpoint `config.json` (`DiaConfig.load_dict`).
    pub fn from_json(value: &Value) -> Result<Self> {
        let model = ModelConfig::from_json(value)?;
        let data_value = value.get("data").ok_or_else(|| SpeechError::BadConfig {
            field: "data".into(),
            why: "config has no data section".into(),
        })?;
        let data = DataConfig::from_json(data_value)?;
        if let Some(training) = value.get("training") {
            let dtype = training
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("bfloat16");
            if !dtype.eq_ignore_ascii_case("float32") {
                return Err(SpeechError::BadConfig {
                    field: "training.dtype".into(),
                    why: format!(
                        "checkpoint declares {dtype} activations; the portable port computes float32 only"
                    ),
                });
            }
        }
        Ok(Self {
            model,
            data,
            version: value
                .get("version")
                .and_then(Value::as_str)
                .unwrap_or("1.0")
                .to_owned(),
        })
    }
}

fn round_128(len: usize) -> usize {
    len.div_ceil(128) * 128
}

fn opt_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

fn opt_i64(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
}
