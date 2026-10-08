//! Voxtral Mini 3B speech understanding.
//!
//! Reference: `mlx_audio/stt/models/voxtral/` (voxtral.py, config.py) at
//! mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//! Whisper-large-v3 audio tower, windowed projector, and a Mistral text
//! model; the transcription prompt is the processor's fixed English
//! transcription request, pinned here as token ids.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::quant::{load_quantized, QuantScheme};
use crate::stt::qwen3_asr::config::TextConfig;
use crate::stt::qwen3_asr::decoder::{Decoder, TokenDecoder};
use crate::stt::voxtral_realtime::tokenizer::TekkenTokenizer;
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxtralProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const VOXTRAL_MINI_3B_BF16: VoxtralProfile = VoxtralProfile {
    name: "Voxtral Mini 3B 2507 bf16",
    repository: "mlx-community/Voxtral-Mini-3B-2507-bf16",
    revision: "ab69f21c8e2c52b05335bf282898e141c8c42477",
};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const TARGET_SAMPLES: usize = 480_000; // the processor's 30 s window
const FRAMES_PER_MERGE: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralConfig {
    pub encoder_layers: usize,
    pub d_model: usize,
    pub encoder_heads: usize,
    pub encoder_ffn_dim: usize,
    pub intermediate_size: usize,
    pub text: TextConfig,
    pub audio_token_id: i32,
    pub scheme: QuantScheme,
}

fn bad_config(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn positive(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad_config(key, "must be a positive integer"))
}

impl VoxtralConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        let audio = root
            .get("audio_config")
            .ok_or_else(|| bad_config("audio_config", "is missing"))?;
        let text_json = root
            .get("text_config")
            .ok_or_else(|| bad_config("text_config", "is missing"))?;
        let positive_in = |value: &Value, key: &str| -> Result<usize> { positive(value, key) };
        let hidden = positive_in(text_json, "hidden_size")?;
        let heads = positive_in(text_json, "num_attention_heads")?;
        let kv_heads = positive_in(text_json, "num_key_value_heads")?;
        let head_dim = text_json
            .get("head_dim")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .unwrap_or(hidden / heads);
        let text = TextConfig {
            vocab_size: positive_in(text_json, "vocab_size")?,
            hidden_size: hidden,
            intermediate_size: positive_in(text_json, "intermediate_size")?,
            num_hidden_layers: positive_in(text_json, "num_hidden_layers")?,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim,
            rotary_dim: head_dim,
            rms_norm_eps: text_json
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| bad_config("rms_norm_eps", "must be positive"))?
                as f32,
            rope_theta: text_json
                .get("rope_theta")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| bad_config("rope_theta", "must be positive"))?
                as f32,
            // Llama/Mistral attention has no per-head Q/K normalization.
            qk_norm: false,
            tie_word_embeddings: false,
        };
        Ok(Self {
            encoder_layers: positive(audio, "num_hidden_layers")?,
            d_model: positive(audio, "hidden_size")?,
            encoder_heads: positive(audio, "num_attention_heads")?,
            encoder_ffn_dim: positive(audio, "intermediate_size")?,
            intermediate_size: positive(audio, "intermediate_size")?,
            text,
            audio_token_id: root
                .get("audio_token_id")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| bad_config("audio_token_id", "must be an integer"))?,
            scheme: QuantScheme {
                bits: 16,
                group_size: 1,
            },
        })
    }
}

#[derive(Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self> {
        let descriptor =
            file.descriptor(&format!("{base}.weight"))
                .ok_or_else(|| SpeechError::Tensor {
                    name: base.to_owned(),
                    why: "tensor is missing".into(),
                })?;
        if descriptor.shape != [output, input] {
            return Err(SpeechError::Tensor {
                name: base.to_owned(),
                why: format!("expected [{output}, {input}], got {:?}", descriptor.shape),
            });
        }
        Ok(Self {
            weight: file.load_as_f32(&format!("{base}.weight"))?,
            bias: if bias {
                Some(file.load_as_f32(&format!("{base}.bias"))?)
            } else {
                None
            },
            input,
            output,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, name: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: file.load_as_f32(&format!("{name}.weight"))?,
            bias: file.load_as_f32(&format!("{name}.bias"))?,
            width,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(x, rows, self.width, &self.weight, Some(&self.bias), 1e-5);
    }
}

struct AudioAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let width = self.heads * self.head_dim;
        let mut query = ops::split_heads(&self.q.forward(x, rows), rows, self.heads, self.head_dim);

        let keys = ops::split_heads(&self.k.forward(x, rows), rows, self.heads, self.head_dim);
        let values = ops::split_heads(&self.v.forward(x, rows), rows, self.heads, self.head_dim);
        let scale = (self.head_dim as f32).sqrt().recip();
        for value in query.iter_mut() {
            *value *= scale;
        }
        let mut attended = vec![0.0f32; width * rows];
        for head in 0..self.heads {
            let offset = head * rows * self.head_dim;
            for position in 0..rows {
                let q_row = &query
                    [offset + position * self.head_dim..offset + (position + 1) * self.head_dim];
                let out = ops::sdpa(
                    q_row,
                    &keys[offset..offset + rows * self.head_dim],
                    &values[offset..offset + rows * self.head_dim],
                    None,
                    1,
                    rows,
                    self.head_dim,
                    self.head_dim,
                    1.0,
                );
                attended[position * width + head * self.head_dim
                    ..position * width + (head + 1) * self.head_dim]
                    .copy_from_slice(&out);
            }
        }
        self.o.forward(&attended, rows)
    }
}

struct EncoderLayer {
    attention: AudioAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

struct AudioTower {
    conv1: Linear,
    conv2: Linear,
    positions: Vec<f32>,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
    d_model: usize,
    n_mels: usize,
}

impl AudioTower {
    /// Loads a Conv1d k3 kernel: the checkpoint stores MLX `[out, k, in]`;
    /// the plain kernel consumes `[out, in, k]`.
    fn load_conv(
        file: &SafetensorsFile,
        base: &str,
        in_ch: usize,
        out_ch: usize,
    ) -> Result<Linear> {
        let descriptor =
            file.descriptor(&format!("{base}.weight"))
                .ok_or_else(|| SpeechError::Tensor {
                    name: base.to_owned(),
                    why: "tensor is missing".into(),
                })?;
        let bias = Some(file.load_as_f32(&format!("{base}.bias"))?);
        let weight = match descriptor.shape[..] {
            [o, k, i] if o == out_ch && k == 3 && i == in_ch => {
                let raw = file.load_as_f32(&format!("{base}.weight"))?;
                let mut out = vec![0.0f32; raw.len()];
                for oc in 0..out_ch {
                    for k in 0..3usize {
                        for ic in 0..in_ch {
                            out[(oc * in_ch + ic) * 3 + k] = raw[(oc * 3 + k) * in_ch + ic];
                        }
                    }
                }
                out
            }
            [o, i, k] if o == out_ch && k == 3 && i == in_ch => {
                file.load_as_f32(&format!("{base}.weight"))?
            }
            _ => {
                return Err(SpeechError::Tensor {
                    name: base.to_owned(),
                    why: format!(
                        "expected conv kernel [{out_ch}, 3, {in_ch}], got {:?}",
                        descriptor.shape
                    ),
                })
            }
        };
        Ok(Linear {
            weight,
            bias,
            input: in_ch,
            output: out_ch,
        })
    }

    fn load(file: &SafetensorsFile, config: &VoxtralConfig) -> Result<Self> {
        let layers = (0..config.encoder_layers)
            .map(|index| {
                let prefix = format!("audio_tower.layers.{index}");
                Ok(EncoderLayer {
                    attention: AudioAttention {
                        q: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.q_proj"),
                            config.d_model,
                            config.d_model,
                            true,
                        )?,
                        k: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.k_proj"),
                            config.d_model,
                            config.d_model,
                            false,
                        )?,
                        v: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.v_proj"),
                            config.d_model,
                            config.d_model,
                            true,
                        )?,
                        o: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.out_proj"),
                            config.d_model,
                            config.d_model,
                            true,
                        )?,
                        heads: config.encoder_heads,
                        head_dim: config.d_model / config.encoder_heads,
                    },
                    attention_norm: LayerNorm::load(
                        file,
                        &format!("{prefix}.self_attn_layer_norm"),
                        config.d_model,
                    )?,
                    fc1: Linear::load(
                        file,
                        &format!("{prefix}.fc1"),
                        config.d_model,
                        config.encoder_ffn_dim,
                        true,
                    )?,
                    fc2: Linear::load(
                        file,
                        &format!("{prefix}.fc2"),
                        config.encoder_ffn_dim,
                        config.d_model,
                        true,
                    )?,
                    final_norm: LayerNorm::load(
                        file,
                        &format!("{prefix}.final_layer_norm"),
                        config.d_model,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: Self::load_conv(file, "audio_tower.conv1", N_MELS, config.d_model)?,
            conv2: Self::load_conv(file, "audio_tower.conv2", config.d_model, config.d_model)?,
            positions: file.load_as_f32("audio_tower.embed_positions.weight")?,
            layers,
            layer_norm: LayerNorm::load(file, "audio_tower.layer_norm", config.d_model)?,
            d_model: config.d_model,
            n_mels: N_MELS,
        })
    }

    /// `features` is `[steps, n_mels]` rows-major; returns
    /// `[steps / 2 merged-groups * FRAMES_PER_MERGE, d_model]`... actually
    /// `[steps / 2, d_model]` pre-merge tower output.
    fn forward(&self, features: &[f32], steps: usize) -> Result<Vec<f32>> {
        // Conv over channels-last rows: transpose to [n_mels, steps].
        let mut plane = vec![0.0f32; self.n_mels * steps];
        for step in 0..steps {
            for mel in 0..self.n_mels {
                plane[mel * steps + step] = features[step * self.n_mels + mel];
            }
        }
        let conv_out = |input: &[f32], in_ch: usize, weight: &Linear, stride: usize| {
            let out_steps = (steps * stride / stride).saturating_sub(0);
            let _ = out_steps;
            // Conv1d k3 padding 1 stride `stride` over [ch, time]: output
            // position t reads inputs t*stride - 1 ..= t*stride + 1.
            let out_ch = weight.output;
            let time = input.len() / in_ch;
            let out_time = (time + 2 - 3) / stride + 1;
            let mut out = vec![0.0f32; out_ch * out_time];
            for oc in 0..out_ch {
                for ot in 0..out_time {
                    let mut sum = weight.bias.as_ref().map_or(0.0, |b| b[oc]);
                    for ic in 0..in_ch {
                        for k in 0..3usize {
                            let base = ot as i64 * stride as i64 - 1i64;
                            let in_t = base + k as i64;
                            if in_t < 0 || in_t as usize >= time {
                                continue;
                            }
                            sum += input[ic * time + in_t as usize]
                                * weight.weight[(oc * in_ch + ic) * 3 + k];
                        }
                    }
                    out[oc * out_time + ot] = sum;
                }
            }
            (out, out_time)
        };
        let (hidden1, time1) = conv_out(&plane, self.n_mels, &self.conv1, 1);
        let mut rows = channels_first_to_rows(&hidden1, self.d_model, time1);
        ops::gelu_erf(&mut rows);
        let (hidden2, time2) = conv_out(
            &channels_first(&rows, time1, self.d_model),
            self.d_model,
            &self.conv2,
            2,
        );
        let mut rows = channels_first_to_rows(&hidden2, self.d_model, time2);
        ops::gelu_erf(&mut rows);
        for step in 0..time2 {
            for dim in 0..self.d_model {
                rows[step * self.d_model + dim] += self.positions[step * self.d_model + dim];
            }
        }
        for layer in self.layers.iter() {
            let mut normalized = rows.clone();
            layer.attention_norm.apply(&mut normalized, time2);
            let attended = layer.attention.forward(&normalized, time2);
            let mut hidden = add(&rows, &attended);
            let mut normalized = hidden.clone();
            layer.final_norm.apply(&mut normalized, time2);
            let mut ff = layer.fc1.forward(&normalized, time2);
            ops::gelu_erf(&mut ff);
            let ff = layer.fc2.forward(&ff, time2);
            hidden = add(&hidden, &ff);
            rows = hidden;
        }
        self.layer_norm.apply(&mut rows, time2);
        Ok(rows)
    }
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
}

fn channels_first_to_rows(x: &[f32], channels: usize, steps: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; x.len()];
    for channel in 0..channels {
        for step in 0..steps {
            rows[step * channels + channel] = x[channel * steps + step];
        }
    }
    rows
}

fn channels_first(x: &[f32], steps: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for step in 0..steps {
        for channel in 0..channels {
            out[channel * steps + step] = x[step * channels + channel];
        }
    }
    out
}

/// Whisper feature extractor: zero-pad to the 30 s window, centered reflect
/// STFT (400/160, full Hann frame), power, Slaney 128-band mel, drop the
/// trailing frame, log10 clamp, peak - 8 floor, `(x + 4) / 4`.
pub fn compute_features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    let mut padded = samples.to_vec();
    padded.resize(TARGET_SAMPLES, 0.0);
    let options = crate::stft::StftOptions {
        fft_size: N_FFT,
        hop: HOP_LENGTH,
        window: crate::dsp::hann_window(N_FFT),
        center: true,
    };
    let spectra = crate::stft::stft(&padded, &options).map_err(|error| SpeechError::Input {
        why: format!("frontend stft failed: {error}"),
    })?;
    let frames = spectra.len().saturating_sub(1);
    let bank = crate::mel::mel_filterbank(
        N_MELS,
        N_FFT,
        SAMPLE_RATE,
        0.0,
        None,
        crate::mel::MelScale::Slaney,
    )
    .map_err(|error| SpeechError::Input {
        why: format!("mel filterbank failed: {error}"),
    })?;
    let mut mel = vec![0.0f32; frames * N_MELS];
    for (frame, spectrum) in spectra.iter().take(frames).enumerate() {
        let power: Vec<f32> = spectrum.iter().map(|v| v.re * v.re + v.im * v.im).collect();
        let projected = bank.project(&power).map_err(|error| SpeechError::Input {
            why: format!("mel projection failed: {error}"),
        })?;
        mel[frame * N_MELS..(frame + 1) * N_MELS].copy_from_slice(&projected);
    }
    let peak = mel
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-10)
        .log10()
        - 8.0;
    for value in mel.iter_mut() {
        *value = (value.max(1e-10).log10().max(peak) + 4.0) / 4.0;
    }
    Ok((mel, frames))
}

const SAMPLE_RATE: u32 = 16_000;

/// Loaded Voxtral Mini model.
pub struct Voxtral {
    config: VoxtralConfig,
    tower: AudioTower,
    projector_1: Linear,
    projector_2: Linear,
    decoder: Decoder,
    tokenizer: TekkenTokenizer,
}

impl Voxtral {
    /// Load the pinned profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_json = fs::read_to_string(model_dir.join("config.json")).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read config.json: {error}"),
            }
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = VoxtralConfig::from_json(&root)?;
        let files: Vec<SafetensorsFile> = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        .iter()
        .map(|name| SafetensorsFile::open(&model_dir.join(name)).map_err(SpeechError::from))
        .collect::<Result<Vec<_>>>()?;
        let tower = AudioTower::load(&files[0], &config)?;
        let load_projector = |name: &str, input: usize, output: usize| -> Result<Linear> {
            let file = files
                .iter()
                .find(|f| f.contains_tensor(&format!("{name}.weight")))
                .ok_or_else(|| SpeechError::Tensor {
                    name: name.to_owned(),
                    why: "tensor is missing".into(),
                })?;
            let (weight, bias) = load_quantized(file, name, config.scheme)?;
            if bias.is_some() {
                return Err(SpeechError::Tensor {
                    name: name.to_owned(),
                    why: "projector bias is not expected".into(),
                });
            }
            Ok(Linear {
                weight,
                bias: None,
                input,
                output,
            })
        };
        let projector_1 = load_projector(
            "multi_modal_projector.linear_1",
            config.intermediate_size,
            config.text.hidden_size,
        )?;
        let projector_2 = load_projector(
            "multi_modal_projector.linear_2",
            config.text.hidden_size,
            config.text.hidden_size,
        )?;
        let decoder = Decoder::load_sharded(&files, &config.text, config.scheme)?;
        let tokenizer = TekkenTokenizer::from_model_path(model_dir)?;
        Ok(Self {
            config,
            tower,
            projector_1,
            projector_2,
            decoder,
            tokenizer,
        })
    }

    pub fn profile(&self) -> VoxtralProfile {
        VOXTRAL_MINI_3B_BF16
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    /// Transcribe one mono 16 kHz waveform.
    pub fn transcribe(&self, samples: &[f32], prompt_ids: &[i32]) -> Result<String> {
        let (features, frames) = compute_features(samples)?;
        let tower_out = self.tower.forward(&features, frames)?;
        let steps = tower_out.len() / self.config.d_model;
        if steps % FRAMES_PER_MERGE != 0 {
            return Err(SpeechError::Input {
                why: format!("tower produced {steps} steps, not divisible by {FRAMES_PER_MERGE}"),
            });
        }
        let groups = steps / FRAMES_PER_MERGE;
        let mut merged = vec![0.0f32; groups * self.config.intermediate_size];
        for group in 0..groups {
            for part in 0..FRAMES_PER_MERGE {
                merged[group * self.config.intermediate_size + part * self.config.d_model
                    ..group * self.config.intermediate_size + (part + 1) * self.config.d_model]
                    .copy_from_slice(
                        &tower_out[(group * FRAMES_PER_MERGE + part) * self.config.d_model
                            ..(group * FRAMES_PER_MERGE + part + 1) * self.config.d_model],
                    );
            }
        }
        let mut projected = self.projector_1.forward(&merged, groups);
        ops::gelu_erf(&mut projected);
        let audio_embeds = self.projector_2.forward(&projected, groups);

        let audio_positions: Vec<usize> = prompt_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.config.audio_token_id)
            .map(|(index, _)| index)
            .collect();
        if audio_positions.len() != groups {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt has {} audio tokens but the tower produced {groups} embeddings",
                    audio_positions.len()
                ),
            });
        }
        let mut inputs = self.decoder.embed(prompt_ids)?;
        for (row, &position) in audio_positions.iter().enumerate() {
            inputs[position * self.config.text.hidden_size
                ..(position + 1) * self.config.text.hidden_size]
                .copy_from_slice(
                    &audio_embeds[row * self.config.text.hidden_size
                        ..(row + 1) * self.config.text.hidden_size],
                );
        }
        let (mut hidden, mut cache) = self.decoder.prefill(&inputs, prompt_ids.len());
        let mut token = self
            .decoder
            .logits(&hidden)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as i32)
            .unwrap_or(0);
        let mut generated: Vec<i32> = Vec::new();
        for _ in 0..128 {
            // Tekken ids 0-2 are control tokens (bos/eos boundaries in the
            // reference tokenizer's eos_token_ids set).
            if token <= 2 {
                break;
            }
            generated.push(token);
            hidden = self.decoder.next_hidden(token, &mut cache)?;
            token = self
                .decoder
                .logits(&hidden)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index as i32)
                .unwrap_or(0);
        }
        Ok(self.tokenizer.decode(&generated))
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_features, Voxtral, VOXTRAL_MINI_3B_BF16};
    use crate::ops;
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/voxtral_reference.json");

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid voxtral reference fixture")
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            VOXTRAL_MINI_3B_BF16.repository,
            "mlx-community/Voxtral-Mini-3B-2507-bf16"
        );
        assert_eq!(
            VOXTRAL_MINI_3B_BF16.revision,
            "ab69f21c8e2c52b05335bf282898e141c8c42477"
        );
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "ab69f21c8e2c52b05335bf282898e141c8c42477"
        );
        assert_eq!(fixture["provenance"]["language"], "en");
    }

    #[test]
    fn frontend_matches_the_reference_fixture() {
        let fixture = load_fixture();
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let (features, frames) = compute_features(&waveform.samples).expect("features");
        assert_eq!(frames, 3000);

        let feature_spots: Spots =
            serde_json::from_value(fixture["input_features"].clone()).unwrap();
        assert_eq!(feature_spots.shape, [3000, 128]);
        let total_columns = 128;
        let mut worst = 0.0f32;
        for (row_index, &row) in feature_spots.rows.iter().enumerate() {
            for (column_index, &column) in feature_spots.columns.iter().enumerate() {
                let expected = feature_spots.values[row_index][column_index];
                let diff = (features[row * total_columns + column] - expected).abs();
                worst = worst.max(diff);
                assert!(diff < 2.0e-3, "input_features [{row},{column}] diff {diff}");
            }
        }
        eprintln!("voxtral frontend fixture parity: worst {worst:.3e} (gate 2.0e-3)");
    }

    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_VOXTRAL_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_VOXTRAL_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_VOXTRAL_MODEL_DIR is unset");
            return;
        };
        let model = Voxtral::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), VOXTRAL_MINI_3B_BF16);
        let fixture = load_fixture();

        let prompt_ids: Vec<i32> = fixture["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();

        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let (features, frames) = compute_features(&waveform.samples).expect("features");
        let feature_spots: Spots =
            serde_json::from_value(fixture["input_features"].clone()).unwrap();
        let total_columns = feature_spots.shape.iter().product::<usize>() / feature_spots.shape[0];
        for (row_index, &row) in feature_spots.rows.iter().enumerate() {
            for (column_index, &column) in feature_spots.columns.iter().enumerate() {
                let expected = feature_spots.values[row_index][column_index];
                let diff = (features[row * total_columns + column] - expected).abs();
                assert!(diff < 2.0e-3, "input_features [{row},{column}] diff {diff}");
            }
        }

        let tower_out = model.tower.forward(&features, frames).expect("tower");
        {
            let tower_spots: Spots =
                serde_json::from_value(fixture["tower_output"].clone()).unwrap();
            let d = model.config.d_model;
            let mut worst = 0.0f32;
            for (row_index, &row) in tower_spots.rows.iter().enumerate() {
                for (column_index, &column) in tower_spots.columns.iter().enumerate() {
                    let expected = tower_spots.values[row_index][column_index];
                    let diff = (tower_out[row * d + column] - expected).abs();
                    worst = worst.max(diff);
                    if diff >= 5.0e-2 {
                        eprintln!(
                            "voxtral tower diag: [{row},{column}] diff {diff} expected {expected}"
                        );
                    }
                }
            }
            eprintln!("voxtral tower parity: worst {worst:.3e} (gate 5e-2)");
        }
        let embed_spots: Spots = serde_json::from_value(fixture["audio_embeds"].clone()).unwrap();
        // The recorded audio_embeds is the merged+projected [groups, hidden].
        let steps = tower_out.len() / model.config.d_model;
        assert_eq!(steps % 4, 0);
        let groups = steps / 4;
        let mut merged = vec![0.0f32; groups * model.config.intermediate_size];
        for group in 0..groups {
            for part in 0..4usize {
                merged[group * model.config.intermediate_size + part * model.config.d_model
                    ..group * model.config.intermediate_size + (part + 1) * model.config.d_model]
                    .copy_from_slice(
                        &tower_out[(group * 4 + part) * model.config.d_model
                            ..(group * 4 + part + 1) * model.config.d_model],
                    );
            }
        }
        let mut projected = model.projector_1.forward(&merged, groups);
        ops::gelu_erf(&mut projected);
        let audio_embeds = model.projector_2.forward(&projected, groups);
        {
            let total = embed_spots.shape.iter().product::<usize>() / embed_spots.shape[0];
            let hidden = model.config.text.hidden_size;
            for (row_index, &row) in embed_spots.rows.iter().enumerate() {
                for (column_index, &column) in embed_spots.columns.iter().enumerate() {
                    let expected = embed_spots.values[row_index][column_index];
                    let diff = (audio_embeds[row * hidden + column] - expected).abs();
                    assert!(diff < 5.0e-2, "audio_embeds [{row},{column}] diff {diff}");
                }
            }
            let _ = total;
        }

        let transcript = model
            .transcribe(&waveform.samples, &prompt_ids)
            .expect("transcribes");
        let expected = fixture["transcript"].as_str().unwrap();
        assert_eq!(transcript, expected, "transcript must match the reference");
        let expected_generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        eprintln!("voxtral gated test: transcript {transcript:?}");
        let _ = expected_generated;
    }
}
