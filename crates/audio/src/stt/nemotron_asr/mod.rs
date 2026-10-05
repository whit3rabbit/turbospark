//! Nemotron 3.5 ASR RNN-T transcription.
//!
//! Reference: `mlx_audio/stt/models/nemotron_asr/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The Python reference
//! smoke-tested the pinned bf16 checkpoint. This port follows its causal
//! FastConformer, language prompt, and greedy RNN-T path.

use std::path::Path;

use serde_json::Value;
use turbospark_audio::nemo_mel::{
    nemo_log_mel_spectrogram_with_padding, NemoMelNormalization, NemoMelOptions,
};
use turbospark_audio::stft::{StftOptions, StftPaddingMode};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

mod encoder;
mod rnnt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronAsrProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const NEMOTRON_3_5_ASR: NemotronAsrProfile = NemotronAsrProfile {
    name: "Nemotron 3.5 ASR",
    repository: "mlx-community/nemotron-3.5-asr-streaming-0.6b",
    revision: "e550040c0478027ed679b2b6b0d055502c103663",
};

#[derive(Debug, Clone, PartialEq)]
pub struct NemotronAsrConfig {
    pub sample_rate: u32,
    pub num_mels: usize,
    pub n_fft: usize,
    pub window_samples: usize,
    pub hop_samples: usize,
    pub preemphasis: f32,
    pub log_zero_guard_value: f32,
    pub vocabulary: Vec<String>,
    pub prompt_indices: Vec<(String, usize)>,
    pub num_prompts: usize,
    pub prompt_hidden: usize,
    pub encoder_layers: usize,
    pub encoder_hidden: usize,
    pub encoder_heads: usize,
    pub encoder_expansion: usize,
    pub pos_emb_max_len: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub conv_kernel: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub joint_hidden: usize,
    pub default_language: String,
    pub default_attention_context: [usize; 2],
    pub max_symbols: usize,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.into(),
    }
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| bad(key, "missing from config.json"))
}

fn positive(value: &Value, key: &str) -> Result<usize> {
    required(value, key)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| bad(key, "must be a positive integer fitting usize"))
}

fn boolean(value: &Value, key: &str) -> Result<bool> {
    required(value, key)?
        .as_bool()
        .ok_or_else(|| bad(key, "must be a boolean"))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    required(value, key)?
        .as_str()
        .ok_or_else(|| bad(key, "must be a string"))
}

fn context_pair(value: &Value, field: &str) -> Result<[usize; 2]> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| bad(field, "must contain [left_context, right_context]"))?;
    let mut out = [0; 2];
    for (index, item) in pair.iter().enumerate() {
        let context = item
            .as_i64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad(field, "context values must be nonnegative integers"))?;
        out[index] = context;
    }
    Ok(out)
}

impl NemotronAsrConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        if string(value, "model_type")? != "nemotron_asr" {
            return Err(SpeechError::Unsupported {
                why: "config is not a Nemotron ASR checkpoint".into(),
            });
        }
        let pre = required(value, "preprocessor")?;
        let encoder = required(value, "encoder")?;
        let prompt = required(value, "prompt")?;
        let decoder = required(value, "decoder")?;
        let joint = required(value, "joint")?;

        let sample_rate = positive(pre, "sample_rate")?;
        let num_mels = positive(pre, "features")?;
        let n_fft = positive(pre, "n_fft")?;
        let window_samples = (required(pre, "window_size")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_size", "must be a number"))?
            * sample_rate as f64) as usize;
        let hop_samples = (required(pre, "window_stride")?
            .as_f64()
            .ok_or_else(|| bad("preprocessor.window_stride", "must be a number"))?
            * sample_rate as f64) as usize;
        let preemphasis = required(pre, "preemph")?
            .as_f64()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| bad("preprocessor.preemph", "must be finite and nonnegative"))?
            as f32;
        let log_zero_guard_value = required(pre, "log_zero_guard_value")?
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| {
                bad(
                    "preprocessor.log_zero_guard_value",
                    "must be finite and positive",
                )
            })? as f32;
        if sample_rate != 16_000
            || string(pre, "window")? != "hann"
            || string(pre, "normalize")? != "NA"
            || string(encoder, "conv_norm_type")? != "layer_norm"
            || string(encoder, "conv_context_size")? != "causal"
            || string(encoder, "self_attention_model")? != "rel_pos"
            || string(encoder, "att_context_style")? != "chunked_limited"
            || !boolean(encoder, "causal_downsampling")?
            || boolean(encoder, "use_bias")?
            || boolean(encoder, "xscaling")?
        {
            return Err(SpeechError::Unsupported {
                why: "unsupported Nemotron frontend or encoder graph variant".into(),
            });
        }
        if window_samples == 0 || hop_samples == 0 || n_fft < window_samples {
            return Err(bad(
                "preprocessor",
                "window, hop, and FFT dimensions are inconsistent",
            ));
        }

        let encoder_hidden = positive(encoder, "d_model")?;
        let encoder_heads = positive(encoder, "n_heads")?;
        let encoder_expansion = positive(encoder, "ff_expansion_factor")?;
        let subsampling_factor = positive(encoder, "subsampling_factor")?;
        let conv_kernel = positive(encoder, "conv_kernel_size")?;
        if encoder_hidden % encoder_heads != 0
            || !subsampling_factor.is_power_of_two()
            || conv_kernel % 2 == 0
            || positive(encoder, "feat_in")? != num_mels
            || positive(encoder, "d_model")?
                .checked_mul(encoder_expansion)
                .is_none()
        {
            return Err(bad(
                "encoder",
                "dimensions cannot form the configured FastConformer",
            ));
        }

        let context_values = required(encoder, "att_context_size")?
            .as_array()
            .filter(|contexts| !contexts.is_empty())
            .ok_or_else(|| bad("encoder.att_context_size", "must be a nonempty array"))?;
        let contexts = context_values
            .iter()
            .map(|pair| context_pair(pair, "encoder.att_context_size"))
            .collect::<Result<Vec<_>>>()?;
        let default_context = context_pair(
            required(value, "default_att_context_size")?,
            "default_att_context_size",
        )?;
        if !contexts.contains(&default_context) {
            return Err(bad(
                "default_att_context_size",
                "must be one of encoder.att_context_size",
            ));
        }

        let prompt_indices_object = required(prompt, "prompt_dictionary")?
            .as_object()
            .ok_or_else(|| bad("prompt.prompt_dictionary", "must be an object"))?;
        let num_prompts = positive(prompt, "num_prompts")?;
        let mut prompt_indices = Vec::with_capacity(prompt_indices_object.len());
        for (language, index) in prompt_indices_object {
            let index = index
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n < num_prompts)
                .ok_or_else(|| bad("prompt.prompt_dictionary", "index exceeds num_prompts"))?;
            prompt_indices.push((language.clone(), index));
        }
        let default_language = required(value, "default_language")?
            .as_str()
            .ok_or_else(|| bad("default_language", "must be a string"))?
            .to_string();
        if !prompt_indices
            .iter()
            .any(|(language, _)| language == &default_language)
        {
            return Err(bad(
                "default_language",
                "must be present in prompt.prompt_dictionary",
            ));
        }

        let vocabulary = required(value, "vocabulary")?
            .as_array()
            .ok_or_else(|| bad("vocabulary", "must be an array"))?
            .iter()
            .map(|piece| {
                piece
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad("vocabulary", "entries must be strings"))
            })
            .collect::<Result<Vec<_>>>()?;
        let vocab_size = positive(decoder, "vocab_size")?;
        let decoder_hidden = positive(decoder, "pred_hidden")?;
        let decoder_layers = positive(decoder, "pred_rnn_layers")?;
        let joint_hidden = positive(joint, "joint_hidden")?;
        if boolean(decoder, "blank_as_pad")? != true
            || vocabulary.len() != vocab_size
            || positive(joint, "num_classes")? != vocab_size
            || positive(joint, "encoder_hidden")? != encoder_hidden
            || positive(joint, "pred_hidden")? != decoder_hidden
            || joint_hidden != decoder_hidden
            || string(joint, "activation")? != "relu"
        {
            return Err(bad(
                "decoder/joint",
                "prediction network, joint network, and vocabulary dimensions disagree",
            ));
        }

        Ok(Self {
            sample_rate: u32::try_from(sample_rate)
                .map_err(|_| bad("preprocessor.sample_rate", "does not fit u32"))?,
            num_mels,
            n_fft,
            window_samples,
            hop_samples,
            preemphasis,
            log_zero_guard_value,
            vocabulary,
            prompt_indices,
            num_prompts,
            prompt_hidden: positive(prompt, "prompt_hidden")?,
            encoder_layers: positive(encoder, "n_layers")?,
            encoder_hidden,
            encoder_heads,
            encoder_expansion,
            pos_emb_max_len: positive(encoder, "pos_emb_max_len")?,
            subsampling_factor,
            subsampling_channels: positive(encoder, "subsampling_conv_channels")?,
            conv_kernel,
            decoder_hidden,
            decoder_layers,
            joint_hidden,
            default_language,
            default_attention_context: default_context,
            max_symbols: positive(value, "max_symbols")?,
        })
    }

    pub fn prompt_index(&self, language: &str) -> usize {
        self.prompt_indices
            .iter()
            .find_map(|(name, index)| (name == language).then_some(*index))
            .or_else(|| {
                self.prompt_indices
                    .iter()
                    .find_map(|(name, index)| (name == &self.default_language).then_some(*index))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny_config() -> Value {
        json!({
            "model_type": "nemotron_asr",
            "preprocessor": {
                "sample_rate": 16000,
                "features": 4,
                "n_fft": 8,
                "window_size": 0.00025,
                "window_stride": 0.000125,
                "window": "hann",
                "normalize": "NA",
                "preemph": 0.97,
                "log_zero_guard_value": 0.000000059604644775390625
            },
            "encoder": {
                "feat_in": 4,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 2,
                "ff_expansion_factor": 2,
                "subsampling_factor": 2,
                "subsampling_conv_channels": 2,
                "conv_kernel_size": 3,
                "causal_downsampling": true,
                "conv_context_size": "causal",
                "conv_norm_type": "layer_norm",
                "att_context_style": "chunked_limited",
                "att_context_size": [[4, 1], [4, 0]],
                "use_bias": false,
                "self_attention_model": "rel_pos",
                "xscaling": false,
                "pos_emb_max_len": 64
            },
            "prompt": {
                "num_prompts": 2,
                "prompt_hidden": 16,
                "prompt_dictionary": {"en-US": 0, "auto": 1}
            },
            "decoder": {
                "vocab_size": 3,
                "pred_hidden": 4,
                "pred_rnn_layers": 2,
                "blank_as_pad": true
            },
            "joint": {
                "num_classes": 3,
                "joint_hidden": 4,
                "encoder_hidden": 8,
                "pred_hidden": 4,
                "activation": "relu"
            },
            "vocabulary": ["<unk>", "▁hi", "!"],
            "default_language": "auto",
            "default_att_context_size": [4, 1],
            "max_symbols": 5
        })
    }

    #[test]
    fn profile_pin_is_immutable_and_language_aliases_resolve() {
        assert_eq!(
            NEMOTRON_3_5_ASR.repository,
            "mlx-community/nemotron-3.5-asr-streaming-0.6b"
        );
        assert_eq!(
            NEMOTRON_3_5_ASR.revision,
            "e550040c0478027ed679b2b6b0d055502c103663"
        );
        let config = NemotronAsrConfig::from_json(&tiny_config()).unwrap();
        assert_eq!(config.prompt_index("en-US"), 0);
        assert_eq!(config.prompt_index("unknown"), 1);
    }

    #[test]
    fn rejects_encoder_or_language_prompt_variants_outside_the_checkpoint_graph() {
        let mut value = tiny_config();
        value["encoder"]["causal_downsampling"] = json!(false);
        assert!(NemotronAsrConfig::from_json(&value).is_err());

        let mut value = tiny_config();
        value["prompt"]["prompt_dictionary"]["en-US"] = json!(2);
        assert!(NemotronAsrConfig::from_json(&value).is_err());
    }

    #[test]
    fn filters_language_tags_like_the_reference_tokenizer() {
        assert!(is_special_piece("<en-US>"));
        assert!(is_special_piece("<zh-Hant>"));
        assert!(is_special_piece("<unk>"));
        assert!(!is_special_piece("<unknown>"));
        assert!(!is_special_piece("▁hello"));
    }
}

struct PromptLinear {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct PromptKernel {
    first: PromptLinear,
    second: PromptLinear,
}

/// Loaded Nemotron 3.5 ASR checkpoint.
pub struct NemotronAsr {
    config: NemotronAsrConfig,
    encoder: encoder::NemotronEncoder,
    prompt: PromptKernel,
    predictor: rnnt::NemotronPredictor,
    joint: rnnt::NemotronJoint,
}

impl NemotronAsr {
    /// Opens a local Hugging Face snapshot containing config.json and model.safetensors.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_value: Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| {
                bad(
                    "config.json",
                    format!("failed to read {}: {error}", config_path.display()),
                )
            })?)
            .map_err(|error| bad("config.json", format!("invalid JSON: {error}")))?;
        let config = NemotronAsrConfig::from_json(&config_value)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = encoder::NemotronEncoder::load(&file, &config)?;
        let prompt = PromptKernel::load(
            &file,
            config.encoder_hidden,
            config.num_prompts,
            config.prompt_hidden,
        )?;
        let predictor = rnnt::NemotronPredictor::load(
            &file,
            config.decoder_hidden,
            config.decoder_layers,
            config.vocabulary.len(),
        )?;
        let joint = rnnt::NemotronJoint::load(
            &file,
            config.encoder_hidden,
            config.decoder_hidden,
            config.joint_hidden,
            config.vocabulary.len() + 1,
        )?;
        Ok(Self {
            config,
            encoder,
            prompt,
            predictor,
            joint,
        })
    }

    pub fn config(&self) -> &NemotronAsrConfig {
        &self.config
    }

    /// Transcribes finite mono PCM at the configured sample rate.
    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        self.transcribe_with_language(samples, sample_rate, None)
    }

    /// Transcribes with a language prompt such as "en-US" or "auto".
    pub fn transcribe_with_language(
        &self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        if sample_rate != self.config.sample_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "Nemotron expects {} Hz PCM, received {sample_rate} Hz",
                    self.config.sample_rate
                ),
            });
        }
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "Nemotron input must be nonempty finite mono PCM".into(),
            });
        }

        let options = NemoMelOptions {
            stft: StftOptions {
                fft_size: self.config.n_fft,
                hop: self.config.hop_samples,
                window: symmetric_hann(self.config.window_samples),
                center: true,
            },
            sample_rate: self.config.sample_rate,
            num_mels: self.config.num_mels,
            normalize: NemoMelNormalization::None,
            preemphasis: self.config.preemphasis,
            log_zero_guard_value: self.config.log_zero_guard_value,
            pad_to: 0,
            pad_value: 0.0,
            normalize_valid_frames: false,
        };
        let mel =
            nemo_log_mel_spectrogram_with_padding(samples, &options, StftPaddingMode::Reflect)?;
        let frames = mel.len();
        let flat_mel: Vec<f32> = mel.into_iter().flatten().collect();
        let (encoded, encoded_frames) =
            self.encoder
                .encode(&flat_mel, frames, self.config.num_mels)?;
        let prompt_index = self
            .config
            .prompt_index(language.unwrap_or(&self.config.default_language));
        let features = self.prompt.apply(
            &encoded,
            encoded_frames,
            self.config.encoder_hidden,
            self.config.num_prompts,
            prompt_index,
        );
        self.greedy_decode(&features, encoded_frames)
    }

    fn greedy_decode(&self, features: &[f32], frames: usize) -> Result<String> {
        let cfg = &self.config;
        let blank_id = cfg.vocabulary.len();
        let mut last_token = blank_id;
        let mut hidden = vec![vec![0.0; cfg.decoder_hidden]; cfg.decoder_layers];
        let mut cell = vec![vec![0.0; cfg.decoder_hidden]; cfg.decoder_layers];
        let mut pieces = Vec::new();
        let mut frame = 0;
        let mut symbols = 0;
        while frame < frames {
            let start = frame * cfg.encoder_hidden;
            let encoder_frame = &features[start..start + cfg.encoder_hidden];
            let (prediction, proposed_hidden, proposed_cell) = self.predictor.step(
                (last_token != blank_id).then_some(last_token),
                &hidden,
                &cell,
            )?;
            let logits = self.joint.logits(encoder_frame, &prediction);
            let token = argmax(&logits);
            if token == blank_id {
                frame += 1;
                symbols = 0;
                continue;
            }

            last_token = token;
            hidden = proposed_hidden;
            cell = proposed_cell;
            if let Some(piece) = cfg.vocabulary.get(token) {
                if !is_special_piece(piece) {
                    pieces.push(piece.replace('▁', " "));
                }
            }
            symbols += 1;
            if symbols >= cfg.max_symbols {
                frame += 1;
                symbols = 0;
            }
        }
        Ok(pieces.concat().trim().to_string())
    }
}

impl PromptKernel {
    fn load(
        file: &SafetensorsFile,
        hidden: usize,
        prompts: usize,
        prompt_hidden: usize,
    ) -> Result<Self> {
        let input = hidden
            .checked_add(prompts)
            .ok_or_else(|| bad("prompt", "input width overflow"))?;
        Ok(Self {
            first: load_prompt_linear(file, "prompt_kernel.0", prompt_hidden, input)?,
            second: load_prompt_linear(file, "prompt_kernel.2", hidden, prompt_hidden)?,
        })
    }

    fn apply(
        &self,
        encoded: &[f32],
        frames: usize,
        hidden: usize,
        prompts: usize,
        prompt_index: usize,
    ) -> Vec<f32> {
        let mut input = vec![0.0; frames * (hidden + prompts)];
        for frame in 0..frames {
            input[frame * (hidden + prompts)..frame * (hidden + prompts) + hidden]
                .copy_from_slice(&encoded[frame * hidden..(frame + 1) * hidden]);
            input[frame * (hidden + prompts) + hidden + prompt_index] = 1.0;
        }
        let mut projected = crate::ops::linear(
            &input,
            &self.first.weight,
            Some(&self.first.bias),
            frames,
            hidden + prompts,
            self.first.bias.len(),
        );
        for value in &mut projected {
            *value = value.max(0.0);
        }
        crate::ops::linear(
            &projected,
            &self.second.weight,
            Some(&self.second.bias),
            frames,
            projected.len() / frames,
            hidden,
        )
    }
}

fn load_prompt_linear(
    file: &SafetensorsFile,
    prefix: &str,
    output: usize,
    input: usize,
) -> Result<PromptLinear> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = load_f32_tensor(file, &weight_name, &[output, input])?;
    let bias = load_f32_tensor(file, &bias_name, &[output])?;
    Ok(PromptLinear { weight, bias })
}

fn load_f32_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file
        .descriptor(name)
        .ok_or_else(|| bad(name, "missing from safetensors"))?;
    if desc.shape != shape {
        return Err(bad(
            name,
            format!("expected shape {shape:?}, found {:?}", desc.shape),
        ));
    }
    if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(bad(
            name,
            format!("expected floating point weights, found {}", desc.dtype),
        ));
    }
    file.load_as_f32(name)
        .map_err(|error| bad(name, format!("load failed: {error}")))
}

fn symmetric_hann(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|index| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * index as f64 / (size - 1) as f64).cos()))
                as f32
        })
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn is_special_piece(piece: &str) -> bool {
    if matches!(piece, "<unk>" | "<pad>" | "<s>" | "</s>") {
        return true;
    }
    let Some(inner) = piece
        .strip_prefix('<')
        .and_then(|value| value.strip_suffix('>'))
    else {
        return false;
    };
    let Some((language, region)) = inner.split_once('-') else {
        return false;
    };
    (2..=3).contains(&language.len())
        && language.bytes().all(|byte| byte.is_ascii_lowercase())
        && (2..=4).contains(&region.len())
        && region.bytes().all(|byte| byte.is_ascii_alphabetic())
}
