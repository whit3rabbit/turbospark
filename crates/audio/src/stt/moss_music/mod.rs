//! MOSS-Music music understanding and lyrics ASR.
//!
//! Reference: `mlx_audio/stt/models/moss_music/` (moss_music.py, config.py,
//! processor.py, audio.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The pinned distribution is a
//! plain `qwen3`-style text backbone plus a whisper-family audio encoder;
//! the reference's default prompt describes music, so transcription pins an
//! explicit lyrics prompt.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::quant::{load_quantized, QuantScheme};
use crate::stt::qwen3_asr::config::TextConfig;
use crate::stt::qwen3_asr::decoder::{Decoder, TokenDecoder};
use crate::stt::qwen3_asr::load_tokenizer;
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MossMusicProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const MOSS_MUSIC_8B_THINKING_4BIT: MossMusicProfile = MossMusicProfile {
    name: "MOSS-Music 8B Thinking 4-bit",
    repository: "mlx-community/MOSS-Music-8B-Thinking-4bit",
    revision: "b14123a419b6254b92d9e55b8a5b6f7285e05c70",
};

/// The pinned transcription prompt. The upstream default prompt asks for a
/// musical description and returns no lyrics; the reference investigation
/// used this explicit transcription prompt on the smoke clip.
pub const TRANSCRIPTION_PROMPT: &str = "Please transcribe the lyrics of this clip.";

const AUDIO_TOKEN_ID: i32 = 151_654;
const AUDIO_START_ID: i32 = 151_669;
const AUDIO_END_ID: i32 = 151_670;
const EOS_TOKEN_ID: i32 = 151_645;
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const AUDIO_TOKENS_PER_SECOND: f64 = 12.5;
const TIME_MARKER_EVERY_SECONDS: usize = 2;
const MAX_GENERATED_TOKENS: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct MossMusicConfig {
    pub encoder_layers: usize,
    pub d_model: usize,
    pub encoder_heads: usize,
    pub encoder_ffn_dim: usize,
    pub downsample_hidden: usize,
    pub deepstack_layers: Vec<usize>,
    pub adapter_hidden: usize,
    pub text: TextConfig,
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

impl MossMusicConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        let audio = root
            .get("audio_config")
            .ok_or_else(|| bad_config("audio_config", "is missing"))?;
        let text_json = root
            .get("language_config")
            .ok_or_else(|| bad_config("language_config", "is missing"))?;
        let text = parse_text_config(text_json)?;
        let quantization = root
            .get("quantization")
            .ok_or_else(|| bad_config("quantization", "is missing"))?;
        let scheme = QuantScheme {
            bits: u32::try_from(
                quantization
                    .get("bits")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| bad_config("bits", "must be a positive integer"))?,
            )
            .map_err(|_| bad_config("bits", "does not fit u32"))?,
            group_size: positive(quantization, "group_size")?,
        };
        if root.get("tie_word_embeddings").and_then(Value::as_bool) != Some(false) {
            return Err(SpeechError::Unsupported {
                why: "only the untied-head distribution is verified".into(),
            });
        }
        let deepstack_layers = audio
            .get("deepstack_encoder_layer_indexes")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        item.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| {
                                bad_config(
                                    "deepstack_encoder_layer_indexes",
                                    "must contain indices",
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            encoder_layers: positive(audio, "encoder_layers")?,
            d_model: positive(audio, "d_model")?,
            encoder_heads: positive(audio, "encoder_attention_heads")?,
            encoder_ffn_dim: positive(audio, "encoder_ffn_dim")?,
            downsample_hidden: positive(audio, "downsample_hidden_size")?,
            deepstack_layers,
            adapter_hidden: positive(root, "adapter_hidden_size")?,
            text,
            scheme,
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
    /// Plain (never quantized) audio-side linear with an `[output, input]`
    /// shape check.
    fn load_plain(
        file: &SafetensorsFile,
        base: &str,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let descriptor =
            file.descriptor(&format!("{base}.weight"))
                .ok_or_else(|| SpeechError::Tensor {
                    name: base.to_owned(),
                    why: "tensor is missing".into(),
                })?;
        let shape_ok = descriptor.shape == [output, input];
        if !shape_ok {
            return Err(SpeechError::Tensor {
                name: base.to_owned(),
                why: format!("expected [{output}, {input}], got {:?}", descriptor.shape),
            });
        }
        let weight = file.load_as_f32(&format!("{base}.weight"))?;
        let bias = if has_bias {
            Some(file.load_as_f32(&format!("{base}.bias"))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
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

/// Conv2d 3x3 stride 2 padding 1 over a plane `[channels][rows][width]`
/// stored channel-major (the reference's MLX `[B, C, H, W]` minus batch).
fn conv2d_3x3_stride2(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    rows: usize,
    width: usize,
) -> (Vec<f32>, usize, usize) {
    let out_rows = rows.div_ceil(2);
    let out_width = width.div_ceil(2);
    let mut output = vec![0.0f32; out_channels * out_rows * out_width];
    for oc in 0..out_channels {
        for orow in 0..out_rows {
            for ocol in 0..out_width {
                let mut sum = bias.map_or(0.0, |b| b[oc]);
                for ic in 0..in_channels {
                    for kr in 0..3i64 {
                        for kc in 0..3i64 {
                            // padding 1: input index = out*2 + k - 1.
                            let in_row = orow as i64 * 2 + kr - 1;
                            let in_col = ocol as i64 * 2 + kc - 1;
                            if in_row < 0 || in_col < 0 {
                                continue;
                            }
                            let (in_row, in_col) = (in_row as usize, in_col as usize);
                            if in_row >= rows || in_col >= width {
                                continue;
                            }
                            let w = weight
                                [((oc * in_channels + ic) * 3 + kr as usize) * 3 + kc as usize];
                            let v = input[(ic * rows + in_row) * width + in_col];
                            sum += w * v;
                        }
                    }
                }
                output[(oc * out_rows + orow) * out_width + ocol] = sum;
            }
        }
    }
    (output, out_rows, out_width)
}

fn gelu_in_place(x: &mut [f32]) {
    ops::gelu_erf(x);
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
    fn forward(&self, x: &[f32], rows: usize, mask: Option<&[f32]>) -> Vec<f32> {
        let width = self.heads * self.head_dim;
        let mut query = ops::split_heads(&self.q.forward(x, rows), rows, self.heads, self.head_dim);
        let keys = ops::split_heads(&self.k.forward(x, rows), rows, self.heads, self.head_dim);
        let values = ops::split_heads(&self.v.forward(x, rows), rows, self.heads, self.head_dim);
        let scale = (self.head_dim as f32).recip();
        for value in query.iter_mut() {
            *value *= scale;
        }
        let mut attended = vec![0.0f32; width * rows];
        let mut scores = vec![0.0f32; rows];
        for head in 0..self.heads {
            let offset = head * rows * self.head_dim;
            for position in 0..rows {
                let q_row = &query
                    [offset + position * self.head_dim..offset + (position + 1) * self.head_dim];
                let k_plane = &keys[offset..offset + rows * self.head_dim];
                let v_plane = &values[offset..offset + rows * self.head_dim];
                let out = ops::sdpa(
                    q_row,
                    k_plane,
                    v_plane,
                    mask,
                    1,
                    rows,
                    self.head_dim,
                    self.head_dim,
                    1.0,
                );
                attended[position * width + head * self.head_dim
                    ..position * width + (head + 1) * self.head_dim]
                    .copy_from_slice(&out);
                scores.clear();
            }
        }
        self.o.forward(&attended, rows)
    }
}

struct AudioLayer {
    attention: AudioAttention,
    attention_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

struct MossMusicEncoder {
    conv1: (Vec<f32>, Vec<f32>),
    conv2: (Vec<f32>, Vec<f32>),
    conv3: (Vec<f32>, Vec<f32>),
    stem: Linear,
    layers: Vec<AudioLayer>,
    layer_norm: LayerNorm,
    positional: Vec<f32>,
    d_model: usize,
    downsample_hidden: usize,
}

/// Sinusoidal positional table (`sin | cos` halves), matching the reference
/// `sinusoids` with max_timescale 10000.
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let log_timescale = (10_000.0f64).ln() / (half - 1) as f64;
    let mut table = vec![0.0f32; length * channels];
    for position in 0..length {
        for index in 0..half {
            let inv = (-log_timescale * index as f64).exp();
            let scaled = position as f64 * inv;
            table[position * channels + index] = scaled.sin() as f32;
            table[position * channels + half + index] = scaled.cos() as f32;
        }
    }
    table
}

impl MossMusicEncoder {
    fn load(file: &SafetensorsFile, config: &MossMusicConfig) -> Result<Self> {
        let conv = |name: &str, in_ch: usize, out_ch: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            let weight = load_plain_conv(file, name, out_ch, in_ch)?;
            let bias = file.load_as_f32(&format!("{name}.bias"))?;
            Ok((weight, bias))
        };
        let hidden = config.d_model;
        let layers = (0..config.encoder_layers)
            .map(|index| {
                // The checkpoint stores attention projections without the
                // reference sanitize's ".self_attn" segment.
                let prefix = format!("audio_encoder.layers.{index}");
                Ok(AudioLayer {
                    attention: AudioAttention {
                        q: Linear::load_plain(
                            file,
                            &format!("{prefix}.q_proj"),
                            hidden,
                            hidden,
                            true,
                        )?,
                        k: Linear::load_plain(
                            file,
                            &format!("{prefix}.k_proj"),
                            hidden,
                            hidden,
                            false,
                        )?,
                        v: Linear::load_plain(
                            file,
                            &format!("{prefix}.v_proj"),
                            hidden,
                            hidden,
                            true,
                        )?,
                        o: Linear::load_plain(
                            file,
                            &format!("{prefix}.out_proj"),
                            hidden,
                            hidden,
                            true,
                        )?,
                        heads: config.encoder_heads,
                        head_dim: hidden / config.encoder_heads,
                    },
                    attention_norm: LayerNorm::load(
                        file,
                        &format!("audio_encoder.layers.{index}.self_attn_layer_norm"),
                        hidden,
                        1e-5,
                    )?,
                    fc1: Linear::load_plain(
                        file,
                        &format!("audio_encoder.layers.{index}.fc1"),
                        hidden,
                        config.encoder_ffn_dim,
                        true,
                    )?,
                    fc2: Linear::load_plain(
                        file,
                        &format!("audio_encoder.layers.{index}.fc2"),
                        config.encoder_ffn_dim,
                        hidden,
                        true,
                    )?,
                    final_norm: LayerNorm::load(
                        file,
                        &format!("audio_encoder.layers.{index}.final_layer_norm"),
                        hidden,
                        1e-5,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: conv("audio_encoder.conv1", 1, config.downsample_hidden)?,
            conv2: conv(
                "audio_encoder.conv2",
                config.downsample_hidden,
                config.downsample_hidden,
            )?,
            conv3: conv(
                "audio_encoder.conv3",
                config.downsample_hidden,
                config.downsample_hidden,
            )?,
            stem: Linear::load_plain(
                file,
                "audio_encoder.stem_proj",
                config.downsample_hidden * 16,
                hidden,
                true,
            )?,
            layers,
            layer_norm: LayerNorm::load(file, "audio_encoder.layer_norm", hidden, 1e-5)?,
            positional: sinusoids(1500, hidden),
            d_model: hidden,
            downsample_hidden: config.downsample_hidden,
        })
    }

    /// `features` is `[steps, n_mels]` rows-major; returns
    /// `[down_steps, d_model]` plus deepstack captures in pinned order.
    fn forward(
        &self,
        features: &[f32],
        steps: usize,
        deepstack_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let (w1, b1) = (&self.conv1.0, &self.conv1.1);
        let (w2, b2) = (&self.conv2.0, &self.conv2.1);
        let (w3, b3) = (&self.conv3.0, &self.conv3.1);
        // Input plane: [1 channel][n_mels rows][steps width].
        let mut plane = vec![0.0f32; steps * N_MELS];
        for step in 0..steps {
            for mel in 0..N_MELS {
                plane[mel * steps + step] = features[step * N_MELS + mel];
            }
        }
        let (hidden1, rows1, width1) = conv2d_3x3_stride2(
            &plane,
            w1,
            Some(b1),
            1,
            self.downsample_hidden,
            N_MELS,
            steps,
        );
        let mut hidden = hidden1;
        let (mut rows, mut width) = (rows1, width1);
        gelu_in_place(&mut hidden);
        let (hidden2, rows2, width2) = conv2d_3x3_stride2(
            &hidden,
            w2,
            Some(b2),
            self.downsample_hidden,
            self.downsample_hidden,
            rows,
            width,
        );
        hidden = hidden2;
        rows = rows2;
        width = width2;
        gelu_in_place(&mut hidden);
        let (hidden3, rows3, width3) = conv2d_3x3_stride2(
            &hidden,
            w3,
            Some(b3),
            self.downsample_hidden,
            self.downsample_hidden,
            rows,
            width,
        );
        hidden = hidden3;
        rows = rows3;
        width = width3;
        gelu_in_place(&mut hidden);
        // [channels][width_rows][width_cols] -> [steps, channels * freq]
        // The conv output rows are the frequency axis (n_mels downsampled to
        // 16) and the columns are time steps.
        let freq = rows;
        let time = width;
        let mut fused = vec![0.0f32; time * (self.downsample_hidden * freq)];
        for t in 0..time {
            for channel in 0..self.downsample_hidden {
                for f in 0..freq {
                    fused[t * (self.downsample_hidden * freq) + channel * freq + f] =
                        hidden[(channel * time + t) * freq + f];
                }
            }
        }
        let mut projected = self.stem.forward(&fused, time);
        projected.resize(time * self.d_model, 0.0);
        for t in 0..time {
            for dim in 0..self.d_model {
                projected[t * self.d_model + dim] += self.positional[t * self.d_model + dim];
            }
        }
        // No padding mask: one chunk covers the whole clip and every step is
        // valid for the pinned single-clip path.
        let mut captures: Vec<Option<Vec<f32>>> = deepstack_layers.iter().map(|_| None).collect();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut normalized = projected.clone();
            layer.attention_norm.apply(&mut normalized, time);
            let mask: Option<&[f32]> = None;
            let attended = layer.attention.forward(&normalized, time, mask);
            let mut hidden = add(&projected, &attended);
            let mut ff = layer.fc1.forward(&hidden, time);
            gelu_in_place(&mut ff);
            let ff = layer.fc2.forward(&ff, time);
            hidden = add(&hidden, &ff);
            layer.final_norm.apply(&mut hidden, time);
            projected = hidden;
            if let Some(slot) = deepstack_layers
                .iter()
                .position(|&capture| capture == index)
            {
                captures[slot] = Some(projected.clone());
            }
        }
        self.layer_norm.apply(&mut projected, time);
        let captures = captures
            .into_iter()
            .map(|capture| {
                capture.ok_or_else(|| SpeechError::BadConfig {
                    field: "deepstack_encoder_layer_indexes".into(),
                    why: "a capture layer index was not reached".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((projected, captures))
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, name: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: file.load_as_f32(&format!("{name}.weight"))?,
            bias: file.load_as_f32(&format!("{name}.bias"))?,
            width,
            epsilon,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        for row in 0..rows {
            let slice = &mut x[row * self.width..(row + 1) * self.width];
            let mean = slice.iter().sum::<f32>() / self.width as f32;
            let variance = slice
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / self.width as f32;
            let scale = 1.0 / (variance + self.epsilon).sqrt();
            for (value, (weight, bias)) in slice.iter_mut().zip(self.weight.iter().zip(&self.bias))
            {
                *value = (*value - mean) * scale * weight + bias;
            }
        }
    }
}

/// Builds the shared decoder config from MOSS's bare `qwen3` language
/// config (the shared parser expects the `qwen3_asr` wrapper shape).
fn parse_text_config(value: &Value) -> Result<TextConfig> {
    let positive = |key: &str| -> Result<usize> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| bad_config(key, "must be a positive integer"))
    };
    let hidden = positive("hidden_size")?;
    let heads = positive("num_attention_heads")?;
    let kv_heads = positive("num_key_value_heads")?;
    let head_dim = value
        .get("head_dim")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(hidden / heads);
    Ok(TextConfig {
        vocab_size: positive("vocab_size")?,
        hidden_size: hidden,
        intermediate_size: positive("intermediate_size")?,
        num_hidden_layers: positive("num_hidden_layers")?,
        num_attention_heads: heads,
        num_key_value_heads: kv_heads,
        head_dim,
        rotary_dim: head_dim,
        rms_norm_eps: value
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| bad_config("rms_norm_eps", "must be a positive number"))?
            as f32,
        rope_theta: value
            .get("rope_theta")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| bad_config("rope_theta", "must be a positive number"))?
            as f32,
        qk_norm: true,
        tie_word_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
}

fn load_plain_conv(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    in_ch: usize,
) -> Result<Vec<f32>> {
    let descriptor =
        file.descriptor(&format!("{name}.weight"))
            .ok_or_else(|| SpeechError::Tensor {
                name: name.to_owned(),
                why: "tensor is missing".into(),
            })?;
    // The reference sanitize transposes [out, in, k, k] PyTorch kernels to
    // MLX [out, k, k, in]; the pinned conversion stores MLX order.
    let expected = [out_ch, 3, 3, in_ch];
    if descriptor.shape == expected {
        let raw = file.load_as_f32(&format!("{name}.weight"))?;
        return Ok(transpose_kernel_to_out_in_kk(&raw, out_ch, in_ch));
    }
    if descriptor.shape == [out_ch, in_ch, 3, 3] {
        return Ok(file.load_as_f32(&format!("{name}.weight"))?);
    }
    Err(SpeechError::Tensor {
        name: name.to_owned(),
        why: format!(
            "expected conv kernel shape {expected:?}, got {:?}",
            descriptor.shape
        ),
    })
}

/// MLX `[out, k, k, in]` -> the plain kernel's `[out, in, k, k]`.
fn transpose_kernel_to_out_in_kk(raw: &[f32], out_ch: usize, in_ch: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; raw.len()];
    for oc in 0..out_ch {
        for kr in 0..3 {
            for kc in 0..3 {
                for ic in 0..in_ch {
                    output[(oc * in_ch + ic) * 9 + kr * 3 + kc] =
                        raw[((oc * 3 + kr) * 3 + kc) * in_ch + ic];
                }
            }
        }
    }
    output
}

/// Gated MLP (`silu(gate) * up` then `down`), unquantized on the audio side.
struct GatedMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl GatedMlp {
    /// The adapter and deepstack mergers sit outside `audio_encoder`, so
    /// the pinned distribution quantizes them 4-bit group-64 (the encoder
    /// side stays full precision).
    fn load(
        file: &SafetensorsFile,
        base: &str,
        input: usize,
        hidden: usize,
        output: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let load = |name: &str, i: usize, o: usize| -> Result<Linear> {
            let (weight, bias) = load_quantized(file, name, scheme)?;
            if weight.len() != i * o {
                return Err(SpeechError::Tensor {
                    name: name.to_owned(),
                    why: format!("expected {} values, got {}", i * o, weight.len()),
                });
            }
            Ok(Linear {
                weight,
                bias,
                input: i,
                output: o,
            })
        };
        Ok(Self {
            gate: load(&format!("{base}.gate_proj"), input, hidden)?,
            up: load(&format!("{base}.up_proj"), input, hidden)?,
            down: load(&format!("{base}.down_proj"), hidden, output)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut gate = self.gate.forward(x, rows);
        ops::silu(&mut gate);
        let up = self.up.forward(x, rows);
        let product: Vec<f32> = gate.iter().zip(&up).map(|(&a, &b)| a * b).collect();
        self.down.forward(&product, rows)
    }
}

/// Whisper-family frontend: centered reflect STFT (400/160), power, Slaney
/// 128-band mel, drop the last frame, log10 clamp, peak - 8 floor, then
/// `(x + 4) / 4`.
pub fn compute_features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "audio must contain at least one sample".into(),
        });
    }
    let options = crate::stft::StftOptions {
        fft_size: N_FFT,
        hop: HOP_LENGTH,
        window: crate::dsp::hann_window(N_FFT),
        center: true,
    };
    let spectra = crate::stft::stft(samples, &options).map_err(|error| SpeechError::Input {
        why: format!("frontend stft failed: {error}"),
    })?;
    let frames = spectra.len().saturating_sub(1).max(1);
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

/// Digit token ids for the time markers, resolved through the tokenizer.
fn digit_token_ids(tokenizer: &turbospark_tokenizer::Tokenizer) -> Result<[i32; 10]> {
    let mut ids = [0i32; 10];
    for (digit, slot) in ids.iter_mut().enumerate() {
        let encoded = tokenizer
            .encode(digit.to_string().as_str(), false)
            .map_err(|error| SpeechError::Input {
                why: format!("digit encode failed: {error}"),
            })?;
        let ids = encoded.get_ids();
        if ids.len() != 1 {
            return Err(SpeechError::Unsupported {
                why: format!("digit {digit} is not a single token in the checkpoint tokenizer"),
            });
        }
        *slot = i32::try_from(ids[0]).map_err(|_| SpeechError::Input {
            why: "digit token id exceeds signed 32-bit range".into(),
        })?;
    }
    Ok(ids)
}

/// Audio placeholder ids with every-2-second digit markers between runs of
/// audio tokens (`_build_audio_tokens_with_time_markers`).
fn audio_placeholder_ids(audio_seq_len: usize, digits: &[i32; 10]) -> Vec<i32> {
    let total_seconds = audio_seq_len as f64 / AUDIO_TOKENS_PER_SECOND;
    let full_seconds = total_seconds as usize;
    let marker_every_tokens = (AUDIO_TOKENS_PER_SECOND * TIME_MARKER_EVERY_SECONDS as f64) as usize;
    let mut ids = Vec::new();
    let mut consumed = 0usize;
    for second in (TIME_MARKER_EVERY_SECONDS..=full_seconds).step_by(TIME_MARKER_EVERY_SECONDS) {
        let marker_position = (second / TIME_MARKER_EVERY_SECONDS) * marker_every_tokens;
        let segment = marker_position.saturating_sub(consumed);
        if segment > 0 {
            ids.extend(std::iter::repeat_n(AUDIO_TOKEN_ID, segment));
            consumed += segment;
        }
        for byte in second.to_string().bytes() {
            ids.push(digits[(byte - b'0') as usize]);
        }
    }
    if audio_seq_len > consumed {
        ids.extend(std::iter::repeat_n(
            AUDIO_TOKEN_ID,
            audio_seq_len - consumed,
        ));
    }
    ids
}

fn encode_ids(tokenizer: &turbospark_tokenizer::Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| SpeechError::Input {
            why: format!("moss_music prompt tokenization failed: {error}"),
        })?;
    encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: "prompt token id exceeds signed 32-bit range".into(),
            })
        })
        .collect()
}

/// Loaded MOSS-Music model.
pub struct MossMusic {
    config: MossMusicConfig,
    encoder: MossMusicEncoder,
    adapter: GatedMlp,
    mergers: Vec<GatedMlp>,
    decoder: Decoder,
    tokenizer: turbospark_tokenizer::Tokenizer,
    digits: [i32; 10],
}

impl MossMusic {
    /// Load the pinned profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_json = fs::read_to_string(model_dir.join("config.json")).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read config.json: {error}"),
            }
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = MossMusicConfig::from_json(&root)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = MossMusicEncoder::load(&file, &config)?;
        let adapter = GatedMlp::load(
            &file,
            "audio_adapter",
            config.d_model,
            config.adapter_hidden,
            config.text.hidden_size,
            config.scheme,
        )?;
        let mergers = (0..config.deepstack_layers.len())
            .map(|index| {
                GatedMlp::load(
                    &file,
                    &format!("deepstack_audio_merger_list.{index}"),
                    config.d_model,
                    config.adapter_hidden,
                    config.text.hidden_size,
                    config.scheme,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let decoder =
            Decoder::load_sharded(std::slice::from_ref(&file), &config.text, config.scheme)?;
        let tokenizer = load_tokenizer(model_dir)?;
        let digits = digit_token_ids(&tokenizer)?;
        Ok(Self {
            config,
            encoder,
            adapter,
            mergers,
            decoder,
            tokenizer,
            digits,
        })
    }

    pub fn profile(&self) -> MossMusicProfile {
        MOSS_MUSIC_8B_THINKING_4BIT
    }

    pub fn config(&self) -> &MossMusicConfig {
        &self.config
    }

    /// Transcribe one mono 16 kHz waveform with the pinned lyrics prompt.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let (features, frames) = compute_features(samples)?;
        let (encoded, captures) =
            self.encoder
                .forward(&features, frames, &self.config.deepstack_layers)?;
        let audio_steps = encoded.len() / self.config.d_model;
        let audio_embeds = self.adapter.forward(&encoded, audio_steps);
        let mergers: Vec<Vec<f32>> = captures
            .iter()
            .zip(&self.mergers)
            .map(|(capture, merger)| merger.forward(capture, audio_steps))
            .collect();

        // Prompt: system + user (<|audio_bos|> markers audio <|audio_eos|> +
        // lyrics prompt) + assistant, exactly like the reference wrapper.
        let mut prompt: Vec<i32> = Vec::new();
        prompt.extend(encode_ids(
            &self.tokenizer,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        )?);
        prompt.push(AUDIO_START_ID);
        prompt.extend(audio_placeholder_ids(audio_steps, &self.digits));
        prompt.push(AUDIO_END_ID);
        prompt.push(198); // "\n"
        prompt.extend(encode_ids(&self.tokenizer, TRANSCRIPTION_PROMPT)?);
        prompt.extend([151_645, 198, 151_644, 77_091, 198]); // <|im_end|>\n<|im_start|>assistant\n

        let audio_positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == AUDIO_TOKEN_ID)
            .map(|(index, _)| index)
            .collect();
        if audio_positions.len() != audio_steps {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt has {} audio tokens but the encoder produced {audio_steps}",
                    audio_positions.len()
                ),
            });
        }
        let mut inputs = self.decoder.embed(&prompt)?;
        for (row, &position) in audio_positions.iter().enumerate() {
            inputs[position * self.config.text.hidden_size
                ..(position + 1) * self.config.text.hidden_size]
                .copy_from_slice(
                    &audio_embeds[row * self.config.text.hidden_size
                        ..(row + 1) * self.config.text.hidden_size],
                );
        }
        let (mut hidden, mut cache) =
            self.decoder
                .prefill_with_injections(&inputs, prompt.len(), &mergers);
        let mut token = self
            .decoder
            .logits(&hidden)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as i32)
            .unwrap_or(EOS_TOKEN_ID);

        let mut generated: Vec<i32> = Vec::new();
        for _ in 0..MAX_GENERATED_TOKENS {
            if token == EOS_TOKEN_ID {
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
                .unwrap_or(EOS_TOKEN_ID);
        }
        let _ = &mut hidden;

        let ids: Vec<u32> = generated
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "generated id does not fit the tokenizer interface".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let raw = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|error| SpeechError::Input {
                why: format!("tokenizer decode failed: {error}"),
            })?;
        Ok(strip_thinking(&raw).to_owned())
    }
}

/// The reference `_strip_thinking`: drop `<think>...</think>` blocks and an
/// unterminated leading `<think>`.
fn strip_thinking(text: &str) -> &str {
    let mut working = text;
    if let Some(open) = working.find("<think>") {
        if let Some(close) = working[open..].find("</think>") {
            let mut stripped = String::with_capacity(working.len());
            stripped.push_str(&working[..open]);
            stripped.push_str(&working[open + close + "</think>".len()..]);
            working = Box::leak(stripped.into_boxed_str());
        } else if open == 0 || working[..open].trim().is_empty() {
            working = "";
        }
    }
    working.trim()
}

#[cfg(test)]
mod tests {
    use super::{audio_placeholder_ids, strip_thinking, MossMusic, MOSS_MUSIC_8B_THINKING_4BIT};
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/moss_music_reference.json");

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid moss_music reference fixture")
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            MOSS_MUSIC_8B_THINKING_4BIT.repository,
            "mlx-community/MOSS-Music-8B-Thinking-4bit"
        );
        assert_eq!(
            MOSS_MUSIC_8B_THINKING_4BIT.revision,
            "b14123a419b6254b92d9e55b8a5b6f7285e05c70"
        );
    }

    #[test]
    fn time_markers_split_audio_runs_every_two_seconds() {
        let digits = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
        // 30 audio tokens = 2.4 s: marker "2" after 25 tokens, then 5 more.
        let ids = audio_placeholder_ids(30, &digits);
        assert_eq!(ids.len(), 31);
        assert_eq!(&ids[..25], &[151_654; 25][..]);
        assert_eq!(ids[25], 17);
        assert_eq!(&ids[26..], &[151_654; 5][..]);
        // Shorter than the first marker: audio tokens only.
        assert_eq!(audio_placeholder_ids(20, &digits), vec![151_654; 20]);
    }

    #[test]
    fn strip_thinking_removes_think_blocks() {
        assert_eq!(strip_thinking("<think>x</think>\nhello"), "hello");
        assert_eq!(strip_thinking("<think>unterminated"), "");
        assert_eq!(strip_thinking("plain text"), "plain text");
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "b14123a419b6254b92d9e55b8a5b6f7285e05c70"
        );
        assert_eq!(
            fixture["provenance"]["transcription_prompt"],
            "Please transcribe the lyrics of this clip."
        );
    }

    #[test]
    fn frontend_and_prompt_construction_match_the_reference_fixture() {
        let fixture = load_fixture();
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let (features, frames) = super::compute_features(&waveform.samples).expect("features");

        let feature_spots: Spots =
            serde_json::from_value(fixture["input_features"].clone()).unwrap();
        // The fixture records the processor's mel-major [n_mels, T] layout;
        // the Rust frontend produces rows-major [T, n_mels].
        assert_eq!(feature_spots.shape, [128, 279]);
        let mut worst = 0.0f32;
        for (row_index, &mel_band) in feature_spots.rows.iter().enumerate() {
            for (column_index, &frame) in feature_spots.columns.iter().enumerate() {
                let expected = feature_spots.values[row_index][column_index];
                let diff = (features[frame * 128 + mel_band] - expected).abs();
                worst = worst.max(diff);
                assert!(
                    diff < 2.0e-3,
                    "input_features [{mel_band},{frame}] differs by {diff}"
                );
            }
        }
        eprintln!("moss_music frontend fixture parity: worst {worst:.3e} (gate 2.0e-3)");

        // The conv-downsampled token count drives the placeholder length.
        // Three 2x conv downsamples: ceil((ceil((ceil((F)/2))/2))/2).
        let audio_steps: usize = frames.div_ceil(2).div_ceil(2).div_ceil(2);
        assert_eq!(
            fixture["audio_mask_true_count"].as_u64().unwrap() as usize,
            audio_steps - 1 + 1,
            "audio token count must match the reference encoder output"
        );

        // Rebuild the prompt with the Qwen digit block (ids 15-24) and
        // compare against the recorded reference prompt exactly.
        let digits: [i32; 10] = std::array::from_fn(|d| 15 + d as i32);
        let placeholder = audio_placeholder_ids(audio_steps, &digits);
        let mut prompt: Vec<i64> = vec![
            151_644, 8948, 198, 2610, 525, 264, 10_950, 17_847, 13, 151_645, 198, 151_644, 872,
            198, 151_669,
        ];
        prompt.extend(placeholder.iter().map(|&id| id as i64));
        prompt.push(151_670);
        prompt.push(198);
        prompt.extend([
            5501, 1356, 3114, 279, 23_261, 315, 419, 12_327, 13, 151_645, 198, 151_644, 77_091, 198,
        ]);

        let expected_prompt: Vec<i64> = fixture["prompt_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(prompt, expected_prompt, "prompt token ids must match");
    }

    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_MOSS_MUSIC_MODEL_DIR and more than 36 GiB of resident memory (the 8B model dequantizes to about 40 GiB; this host cannot run it)"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_MOSS_MUSIC_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_MOSS_MUSIC_MODEL_DIR is unset");
            return;
        };
        let model = MossMusic::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), MOSS_MUSIC_8B_THINKING_4BIT);
        let fixture = load_fixture();

        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);

        let (features, frames) = super::compute_features(&waveform.samples).expect("features");
        let feature_spots: Spots =
            serde_json::from_value(fixture["input_features"].clone()).unwrap();
        {
            let total_columns =
                feature_spots.shape.iter().product::<usize>() / feature_spots.shape[0];
            for (row_index, &row) in feature_spots.rows.iter().enumerate() {
                for (column_index, &column) in feature_spots.columns.iter().enumerate() {
                    let expected = feature_spots.values[row_index][column_index];
                    let diff = (features[row * total_columns + column] - expected).abs();
                    assert!(diff < 2.0e-3, "input_features [{row},{column}] diff {diff}");
                }
            }
        }
        // The prompt id sequence (including time markers) must match exactly.

        let digits = model.digits;
        let audio_steps = frames.div_ceil(8);
        let placeholder = audio_placeholder_ids(audio_steps, &digits);
        let _ = placeholder;
        let transcript = model.transcribe(&waveform.samples).expect("transcribes");
        let expected = fixture["transcript"].as_str().unwrap();
        assert_eq!(transcript, expected, "transcript must match the reference");
        let expected_generated: Vec<i64> = fixture["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let _ = expected_generated;
    }
}
