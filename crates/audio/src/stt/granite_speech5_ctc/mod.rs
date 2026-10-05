//! Encoder-only Granite Speech 5.0 TurboCTC transcription.
//!
//! Reference: `mlx_audio/stt/models/granite_speech5_ctc/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer;

use crate::{Result, SpeechError};

mod encoder;
mod frontend;

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeech5Profile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const GRANITE_SPEECH5_TURBOCTC: GraniteSpeech5Profile = GraniteSpeech5Profile {
    name: "Granite Speech 5.0 TurboCTC 470M",
    repository: "ibm-granite/granite-speech-5.0-470m-turboctc",
    revision: "947f59af40db9791170a0628cf0f3f4812d720f1",
};

/// Verified subset of the fixed Granite Speech 5.0 TurboCTC graph.
#[derive(Debug, Clone)]
pub struct GraniteSpeech5Config {
    pub vocab_size: usize,
    pub pad_token_id: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub head_dim: usize,
    pub context_size: usize,
    pub conv_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub max_position_embeddings: usize,
    pub subsample_layers: Vec<usize>,
}

fn config_error(field: &str, why: &str) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.to_owned(),
    }
}

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| config_error(field, "must be a positive integer"))
}

impl GraniteSpeech5Config {
    pub fn from_json(value: &Value) -> Result<Self> {
        if value.get("model_type").and_then(Value::as_str) != Some("granite_speech5_ctc")
            || value
                .get("architectures")
                .and_then(Value::as_array)
                .is_none_or(|architectures| {
                    !architectures
                        .iter()
                        .any(|item| item.as_str() == Some("GraniteSpeech5ForCTC"))
                })
        {
            return Err(config_error(
                "model_type",
                "expected the GraniteSpeech5ForCTC architecture",
            ));
        }
        if value.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true) {
            return Err(config_error(
                "tie_word_embeddings",
                "the TurboCTC output heads must be tied",
            ));
        }
        let encoder = value
            .get("encoder_config")
            .and_then(Value::as_object)
            .ok_or_else(|| config_error("encoder_config", "must be an object"))?;
        let number = |field: &str| positive(&Value::Object(encoder.clone()), field);
        let subsample_layers = encoder
            .get("subsample_layers")
            .and_then(Value::as_array)
            .ok_or_else(|| config_error("encoder_config.subsample_layers", "must be an array"))?
            .iter()
            .map(|item| {
                item.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        config_error("encoder_config.subsample_layers", "invalid layer index")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let config = Self {
            vocab_size: positive(value, "vocab_size")?,
            pad_token_id: value
                .get("pad_token_id")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| config_error("pad_token_id", "must be an unsigned integer"))?,
            hidden_size: number("hidden_size")?,
            intermediate_size: number("intermediate_size")?,
            num_hidden_layers: number("num_hidden_layers")?,
            num_attention_heads: number("num_attention_heads")?,
            num_mel_bins: number("num_mel_bins")?,
            head_dim: number("head_dim")?,
            context_size: number("context_size")?,
            conv_expansion_factor: number("conv_expansion_factor")?,
            conv_kernel_size: number("conv_kernel_size")?,
            max_position_embeddings: number("max_position_embeddings")?,
            subsample_layers,
        };
        if config.vocab_size != positive(&Value::Object(encoder.clone()), "vocab_size")?
            || config.hidden_size % config.num_attention_heads != 0
            || config.hidden_size / config.num_attention_heads != config.head_dim
            || config.conv_kernel_size % 2 == 0
            || !matches!(config.num_mel_bins, 80 | 128)
            || config.subsample_layers != [0, 1]
            || config
                .subsample_layers
                .iter()
                .any(|&index| index >= config.num_hidden_layers)
            || config.pad_token_id != 0
        {
            return Err(config_error(
                "encoder_config",
                "unsupported TurboCTC dimensions or subsampling layout",
            ));
        }
        if encoder.get("model_type").and_then(Value::as_str) != Some("granite_speech5_encoder")
            || encoder.get("hidden_act").and_then(Value::as_str) != Some("silu")
            || encoder
                .get("attention_dropout")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                != 0.0
            || encoder
                .get("activation_dropout")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                != 0.0
            || encoder.get("attention_bias").and_then(Value::as_bool) != Some(true)
        {
            return Err(SpeechError::Unsupported {
                why: "only the inference-mode Granite Speech 5.0 TurboCTC encoder is supported"
                    .into(),
            });
        }
        Ok(config)
    }
}

/// Loaded encoder-only CTC model. Inputs are mono 16 kHz samples.
pub struct GraniteSpeech5Ctc {
    config: GraniteSpeech5Config,
    encoder: encoder::Encoder,
    tokenizer: Tokenizer,
}

impl GraniteSpeech5Ctc {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json: Value = serde_json::from_slice(
            &std::fs::read(&config_path)
                .map_err(|error| config_error("config.json", &error.to_string()))?,
        )
        .map_err(|error| config_error("config.json", &error.to_string()))?;
        let config = GraniteSpeech5Config::from_json(&config_json)?;
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = encoder::Encoder::load(&weights, &config)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            config_error("tokenizer.json", &format!("cannot load tokenizer: {error}"))
        })?;
        Ok(Self {
            config,
            encoder,
            tokenizer,
        })
    }

    pub fn config(&self) -> &GraniteSpeech5Config {
        &self.config
    }

    /// Transcribes mono PCM already sampled at 16 kHz.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let (features, rows) = frontend::compute_features(samples, self.config.num_mel_bins)?;
        let token_ids = self.encoder.forward(&features, rows, &self.config);
        let ids = token_ids
            .into_iter()
            .map(|token| {
                u32::try_from(token).map_err(|_| SpeechError::Input {
                    why: "CTC token id does not fit tokenizer's u32 interface".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.tokenizer
            .decode(&ids, true)
            .map_err(|error| SpeechError::Input {
                why: format!("tokenizer decode failed: {error}"),
            })
            .map(|text| text.trim().to_owned())
    }
}

#[cfg(test)]
fn ctc_collapse(logits: &[f32], rows: usize, vocab_size: usize, blank_id: usize) -> Vec<usize> {
    let mut output = Vec::new();
    let mut previous = None;
    for row in logits.chunks_exact(vocab_size).take(rows) {
        let token = row
            .iter()
            .enumerate()
            .fold(
                (blank_id, f32::NEG_INFINITY),
                |(best_id, best), (id, &value)| {
                    if value > best {
                        (id, value)
                    } else {
                        (best_id, best)
                    }
                },
            )
            .0;
        if Some(token) != previous && token != blank_id {
            output.push(token);
        }
        previous = Some(token);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{ctc_collapse, GraniteSpeech5Config, GRANITE_SPEECH5_TURBOCTC};
    use serde_json::json;

    #[test]
    fn ctc_collapse_removes_blanks_and_adjacent_repeats() {
        let logits = [
            0.0, 3.0, 0.0, // token 1
            0.0, 2.0, 0.0, // repeated 1
            3.0, 0.0, 0.0, // blank
            0.0, 0.0, 4.0, // token 2
            0.0, 2.0, 0.0, // token 1 after 2
        ];
        assert_eq!(ctc_collapse(&logits, 5, 3, 0), vec![1, 2, 1]);
    }

    #[test]
    fn fixed_profile_and_checkpoint_config_are_validated() {
        assert_eq!(GRANITE_SPEECH5_TURBOCTC.revision.len(), 40);
        let config = json!({
            "architectures": ["GraniteSpeech5ForCTC"],
            "model_type": "granite_speech5_ctc",
            "vocab_size": 16384,
            "pad_token_id": 0,
            "tie_word_embeddings": true,
            "encoder_config": {
                "model_type": "granite_speech5_encoder",
                "vocab_size": 16384,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 16,
                "num_attention_heads": 8,
                "num_mel_bins": 80,
                "head_dim": 128,
                "context_size": 128,
                "conv_expansion_factor": 2,
                "conv_kernel_size": 7,
                "max_position_embeddings": 512,
                "subsample_layers": [0, 1],
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "attention_bias": true
            }
        });
        let parsed = GraniteSpeech5Config::from_json(&config).unwrap();
        assert_eq!(parsed.num_hidden_layers, 16);
        let mut unsupported = config;
        unsupported["encoder_config"]["subsample_layers"] = json!([1, 2]);
        assert!(GraniteSpeech5Config::from_json(&unsupported).is_err());
    }
}
