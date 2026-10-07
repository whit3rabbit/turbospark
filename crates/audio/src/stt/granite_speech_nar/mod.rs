//! Granite Speech 4.1 2B NAR: non-autoregressive CTC plus a bidirectional
//! Granite editor.
//!
//! Reference: `mlx_audio/stt/models/granite_speech_nar/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The pinned MLX
//! conversion ships MLX-layout conv weights; the official IBM checkpoint
//! (PyTorch conv layout) is refused.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::stt::qwen3_asr::load_tokenizer;
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechNarProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const GRANITE_SPEECH_4_1_2B_NAR: GraniteSpeechNarProfile = GraniteSpeechNarProfile {
    name: "Granite Speech 4.1 2B NAR (MLX)",
    repository: "mlx-community/granite-speech-4.1-2b-nar-mlx",
    revision: "6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c",
};

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechNarConfig {
    pub encoder: EncoderConfig,
    pub projector: ProjectorConfig,
    pub text: TextConfig,
    pub encoder_layer_indices: Vec<usize>,
    pub blank_token_id: usize,
    pub min_edit_sequence_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dim_head: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub bpe_output_dim: usize,
    pub bpe_pooling_window: usize,
    pub conv_kernel_size: usize,
    pub conv_expansion_factor: usize,
    pub feedforward_mult: usize,
    pub max_pos_emb: usize,
    pub context_size: usize,
    pub self_conditioning_layer: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectorConfig {
    pub num_layers: usize,
    pub num_encoder_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub block_size: usize,
    pub downsample_rate: usize,
    pub encoder_dim: usize,
    pub llm_dim: usize,
    pub mlp_ratio: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub logits_scaling: f32,
    pub residual_multiplier: f32,
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

fn number(value: &Value, key: &str) -> Result<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .ok_or_else(|| bad_config(key, "must be a finite number"))
}

impl GraniteSpeechNarConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        if root
            .get("architectures")
            .and_then(Value::as_array)
            .is_none_or(|items| {
                !items
                    .iter()
                    .any(|item| item.as_str() == Some("GraniteSpeechNarForASR"))
            })
        {
            return Err(bad_config(
                "architectures",
                "expected the GraniteSpeechNarForASR architecture",
            ));
        }
        let encoder_json = root
            .get("encoder_config")
            .ok_or_else(|| bad_config("encoder_config", "is missing"))?;
        let encoder = EncoderConfig {
            num_layers: positive(encoder_json, "num_layers")?,
            hidden_dim: positive(encoder_json, "hidden_dim")?,
            num_heads: positive(encoder_json, "num_heads")?,
            dim_head: positive(encoder_json, "dim_head")?,
            input_dim: positive(encoder_json, "input_dim")?,
            output_dim: positive(encoder_json, "output_dim")?,
            bpe_output_dim: positive(encoder_json, "bpe_output_dim")?,
            bpe_pooling_window: positive(encoder_json, "bpe_pooling_window")?,
            conv_kernel_size: positive(encoder_json, "conv_kernel_size")?,
            conv_expansion_factor: positive(encoder_json, "conv_expansion_factor")?,
            feedforward_mult: positive(encoder_json, "feedforward_mult")?,
            max_pos_emb: positive(encoder_json, "max_pos_emb")?,
            context_size: positive(encoder_json, "context_size")?,
            self_conditioning_layer: positive(encoder_json, "self_conditioning_layer")?,
        };
        if encoder.hidden_dim != encoder.num_heads * encoder.dim_head {
            return Err(bad_config(
                "encoder_config",
                "hidden_dim must equal num_heads * dim_head",
            ));
        }
        let projector_json = root
            .get("projector_config")
            .ok_or_else(|| bad_config("projector_config", "is missing"))?;
        let projector = ProjectorConfig {
            num_layers: positive(projector_json, "num_layers")?,
            num_encoder_layers: positive(projector_json, "num_encoder_layers")?,
            hidden_size: positive(projector_json, "hidden_size")?,
            num_heads: positive(projector_json, "num_heads")?,
            block_size: positive(projector_json, "block_size")?,
            downsample_rate: positive(projector_json, "downsample_rate")?,
            encoder_dim: positive(projector_json, "encoder_dim")?,
            llm_dim: positive(projector_json, "llm_dim")?,
            mlp_ratio: positive(projector_json, "mlp_ratio")?,
        };
        if projector.block_size % projector.downsample_rate != 0
            || projector.hidden_size % projector.num_heads != 0
        {
            return Err(bad_config(
                "projector_config",
                "block_size must divide by downsample_rate and hidden_size by num_heads",
            ));
        }
        let text_json = root
            .get("text_config")
            .ok_or_else(|| bad_config("text_config", "is missing"))?;
        let rope_theta = text_json
            .get("rope_parameters")
            .and_then(|rope| rope.get("rope_theta"))
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| bad_config("rope_theta", "must be a positive number"))?;
        let text = TextConfig {
            hidden_size: positive(text_json, "hidden_size")?,
            intermediate_size: positive(text_json, "intermediate_size")?,
            num_hidden_layers: positive(text_json, "num_hidden_layers")?,
            num_attention_heads: positive(text_json, "num_attention_heads")?,
            num_key_value_heads: positive(text_json, "num_key_value_heads")?,
            vocab_size: positive(text_json, "vocab_size")?,
            rms_norm_eps: number(text_json, "rms_norm_eps")?,
            rope_theta: rope_theta as f32,
            attention_multiplier: number(text_json, "attention_multiplier")?,
            embedding_multiplier: number(text_json, "embedding_multiplier")?,
            logits_scaling: number(text_json, "logits_scaling")?,
            residual_multiplier: number(text_json, "residual_multiplier")?,
        };
        if text.hidden_size % text.num_attention_heads != 0
            || text.num_attention_heads % text.num_key_value_heads != 0
        {
            return Err(bad_config(
                "text_config",
                "attention head geometry is inconsistent",
            ));
        }
        let raw_indices = root
            .get("encoder_layer_indices")
            .and_then(Value::as_array)
            .ok_or_else(|| bad_config("encoder_layer_indices", "must be an integer array"))?;
        let mut encoder_layer_indices = Vec::with_capacity(raw_indices.len());
        for index in raw_indices {
            let raw = index
                .as_i64()
                .ok_or_else(|| bad_config("encoder_layer_indices", "must contain integers"))?;
            let resolved = if raw < 0 {
                (encoder.num_layers + 1)
                    .checked_sub(raw.unsigned_abs() as usize)
                    .ok_or_else(|| bad_config("encoder_layer_indices", "index out of range"))?
            } else {
                usize::try_from(raw)
                    .map_err(|_| bad_config("encoder_layer_indices", "index out of range"))?
            };
            if resolved > encoder.num_layers {
                return Err(bad_config(
                    "encoder_layer_indices",
                    "index past the recorded hidden states",
                ));
            }
            encoder_layer_indices.push(resolved);
        }
        if root
            .get("scale_projected_embeddings")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(SpeechError::Unsupported {
                why: "only the scale_projected_embeddings=true distribution is verified".into(),
            });
        }
        Ok(Self {
            encoder,
            projector,
            text,
            encoder_layer_indices,
            blank_token_id: positive(root, "blank_token_id")?,
            min_edit_sequence_length: positive(root, "min_edit_sequence_length")?,
        })
    }
}

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const LOG_FLOOR_DB: f32 = 8.0;

/// HTK triangular filterbank in f64 (the reference `precise=True` path),
/// returned columns-major as `[n_freqs, n_mels]` like `moveaxis(0, 1)`.
fn htk_mel_filterbank(sample_rate: f64, n_mels: usize) -> Vec<f64> {
    fn hz_to_mel(freq: f64) -> f64 {
        2595.0 * (1.0 + freq / 700.0).log10()
    }
    fn mel_to_hz(mels: f64) -> f64 {
        700.0 * (10.0f64.powf(mels / 2595.0) - 1.0)
    }
    let n_freqs = N_FFT / 2 + 1;
    let f_max = sample_rate / 2.0;
    let mut all_freqs = Vec::with_capacity(n_freqs);
    for index in 0..n_freqs {
        all_freqs.push(f_max * index as f64 / (n_freqs - 1) as f64);
    }
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(f_max);
    let m_pts: Vec<f64> = (0..n_mels + 2)
        .map(|index| m_min + (m_max - m_min) * index as f64 / (n_mels + 1) as f64)
        .collect();
    let f_pts: Vec<f64> = m_pts.iter().map(|&mel| mel_to_hz(mel)).collect();
    let mut filterbank = vec![0.0f64; n_freqs * n_mels];
    for freq_index in 0..n_freqs {
        let freq = all_freqs[freq_index];
        for mel_index in 0..n_mels {
            let down = -(f_pts[mel_index] - freq) / (f_pts[mel_index + 1] - f_pts[mel_index]);
            let up = (f_pts[mel_index + 2] - freq) / (f_pts[mel_index + 2] - f_pts[mel_index + 1]);
            let value = down.min(up).max(0.0);
            filterbank[freq_index * n_mels + mel_index] = value;
        }
    }
    filterbank
}

/// Mono 16 kHz waveform to `[T/2, 160]` stacked paired log-mel features,
/// following the reference `_compute_features`.
pub fn compute_features(samples: &[f32]) -> Result<Vec<f32>> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "audio must contain at least one sample".into(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "audio samples must be finite".into(),
        });
    }
    // The reference builds zeros(56) + periodic Hann(400) + zeros(56) inside
    // the 512 frame; the shared centered STFT produces the same product.
    let options = crate::stft::StftOptions {
        fft_size: N_FFT,
        hop: HOP_LENGTH,
        window: crate::dsp::hann_window(WIN_LENGTH),
        center: true,
    };
    let spectra = crate::stft::stft_with_modes(
        samples,
        &options,
        crate::stft::StftPaddingMode::Reflect,
        crate::stft::StftWindowPlacement::Center,
    )
    .map_err(|error| SpeechError::Input {
        why: format!("frontend stft failed: {error}"),
    })?;
    let filterbank = htk_mel_filterbank(16_000.0, N_MELS);
    let mut mel = vec![0.0f32; spectra.len() * N_MELS];
    for (frame, spectrum) in spectra.iter().enumerate() {
        for (bin, value) in spectrum.iter().enumerate() {
            let power = value.re * value.re + value.im * value.im;
            for mel_index in 0..N_MELS {
                mel[frame * N_MELS + mel_index] +=
                    power * filterbank[bin * N_MELS + mel_index] as f32;
            }
        }
    }
    // The reference truncates to `l = 2 * (n_samples // (2 * HOP))` frames
    // before the log; guard only against a short spectrum list.
    let keep = (2 * (samples.len() / (2 * HOP_LENGTH))).min(2 * (spectra.len() / 2));
    let mut mel = mel[..keep * N_MELS].to_vec();
    for value in mel.iter_mut() {
        *value = value.max(1e-10).log10();
    }
    let peak = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for value in mel.iter_mut() {
        *value = value.max(peak - LOG_FLOOR_DB) / 4.0 + 1.0;
    }
    Ok(mel)
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
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = load_tensor(file, &format!("{prefix}.weight"), &[output, input])?;
        let bias = if bias {
            Some(load_tensor(file, &format!("{prefix}.bias"), &[output])?)
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

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl LayerNorm {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(file, &format!("{prefix}.weight"), &[width])?,
            bias: load_tensor(file, &format!("{prefix}.bias"), &[width])?,
            width,
            epsilon,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(
            x,
            rows,
            self.width,
            &self.weight,
            Some(&self.bias),
            self.epsilon,
        );
    }
}

fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

#[allow(clippy::too_many_arguments)]
/// Conv1d over rows-major input `[steps, channels]` with MLX-layout weight
/// `[out, kernel, in/groups]`, returning rows-major output.
fn conv1d_rows(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    steps: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    padding: usize,
    groups: usize,
) -> Vec<f32> {
    let channels_first = rows_to_channels_first(x, steps, in_ch);
    let out = ops::conv1d(
        &channels_first,
        weight,
        bias,
        in_ch,
        out_ch,
        kernel,
        1,
        padding,
        1,
        groups,
    );
    let out_steps = channels_first.len() / in_ch + 2 * padding - kernel + 1;
    channels_first_to_rows(&out, out_ch, out_steps)
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

fn rows_to_channels_first(x: &[f32], steps: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; x.len()];
    for step in 0..steps {
        for channel in 0..channels {
            output[channel * steps + step] = x[step * channels + channel];
        }
    }
    output
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
}

fn scale_add(left: &[f32], right: &[f32], factor: f32) -> Vec<f32> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&a, &b)| a + factor * b)
        .collect()
}

fn gelu_in_place(x: &mut [f32]) {
    ops::gelu_erf(x);
}

struct FeedForward {
    pre_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        hidden: usize,
        mult: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            pre_norm: LayerNorm::load(file, &format!("{prefix}.pre_norm"), hidden, eps)?,
            up: Linear::load(
                file,
                &format!("{prefix}.up_proj"),
                hidden,
                hidden * mult,
                true,
            )?,
            down: Linear::load(
                file,
                &format!("{prefix}.down_proj"),
                hidden * mult,
                hidden,
                true,
            )?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = x.to_vec();
        self.pre_norm.apply(&mut hidden, rows);
        hidden = self.up.forward(&hidden, rows);
        ops::silu(&mut hidden);
        self.down.forward(&hidden, rows)
    }
}

struct ShawAttention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Vec<f32>,
    heads: usize,
    dim_head: usize,
    context: usize,
    max_pos_emb: usize,
    hidden: usize,
}

impl ShawAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &EncoderConfig,
        eps: f32,
    ) -> Result<Self> {
        let inner = config.num_heads * config.dim_head;
        Ok(Self {
            pre_norm: LayerNorm::load(file, &format!("{prefix}.pre_norm"), config.hidden_dim, eps)?,
            to_q: Linear::load(
                file,
                &format!("{prefix}.to_q"),
                config.hidden_dim,
                inner,
                false,
            )?,
            to_kv: Linear::load(
                file,
                &format!("{prefix}.to_kv"),
                config.hidden_dim,
                2 * inner,
                false,
            )?,
            to_out: Linear::load(
                file,
                &format!("{prefix}.to_out"),
                inner,
                config.hidden_dim,
                true,
            )?,
            rel_pos_emb: load_tensor(
                file,
                &format!("{prefix}.rel_pos_emb.weight"),
                &[2 * config.max_pos_emb + 1, config.dim_head],
            )?,
            heads: config.num_heads,
            dim_head: config.dim_head,
            context: config.context_size,
            max_pos_emb: config.max_pos_emb,
            hidden: config.hidden_dim,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.pre_norm.apply(&mut normalized, steps);
        let ctx = self.context;
        let padded_steps = steps + (ctx - steps % ctx) % ctx;
        let mut padded = normalized;
        padded.resize(padded_steps * self.hidden, 0.0);
        let q = self.to_q.forward(&padded, padded_steps);
        let kv = self.to_kv.forward(&padded, padded_steps);
        let (k, v) = {
            let mut key = Vec::with_capacity(padded_steps * self.hidden);
            let mut value = Vec::with_capacity(padded_steps * self.hidden);
            for row in 0..padded_steps {
                let base = row * 2 * self.hidden;
                key.extend_from_slice(&kv[base..base + self.hidden]);
                value.extend_from_slice(&kv[base + self.hidden..base + 2 * self.hidden]);
            }
            (key, value)
        };
        let n_blocks = padded_steps / ctx;
        let mut attended = vec![0.0f32; padded_steps * self.hidden];
        let scale = (self.dim_head as f32).sqrt().recip();
        let mut logits = vec![0.0f32; ctx];
        for block in 0..n_blocks {
            for head in 0..self.heads {
                for query_step in 0..ctx {
                    let global_q = (block * ctx + query_step) * self.hidden + head * self.dim_head;
                    for (key_step, logit) in logits.iter_mut().enumerate() {
                        let global_k =
                            (block * ctx + key_step) * self.hidden + head * self.dim_head;
                        let mut score = 0.0f32;
                        for dim in 0..self.dim_head {
                            score += q[global_q + dim] * k[global_k + dim];
                        }
                        // Shaw rel-pos: dot the query with the distance table.
                        let distance = query_step as i64 - key_step as i64;
                        let clipped =
                            distance.clamp(-(ctx as i64), ctx as i64) as usize + self.max_pos_emb;
                        let rel_base = clipped * self.dim_head;
                        for dim in 0..self.dim_head {
                            score += q[global_q + dim] * self.rel_pos_emb[rel_base + dim];
                        }
                        *logit = score * scale;
                    }
                    ops::softmax_row(&mut logits);
                    let global_out =
                        (block * ctx + query_step) * self.hidden + head * self.dim_head;
                    for dim in 0..self.dim_head {
                        let mut sum = 0.0f32;
                        for (key_step, logit) in logits.iter().enumerate() {
                            let global_k =
                                (block * ctx + key_step) * self.hidden + head * self.dim_head;
                            sum += logit * v[global_k + dim];
                        }
                        attended[global_out + dim] = sum;
                    }
                }
            }
        }
        attended.truncate(steps * self.hidden);
        self.to_out.forward(&attended, steps)
    }
}

struct EvalBatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

impl EvalBatchNorm {
    fn load(file: &SafetensorsFile, prefix: &str, features: usize) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(file, &format!("{prefix}.weight"), &[features])?,
            bias: load_tensor(file, &format!("{prefix}.bias"), &[features])?,
            running_mean: load_tensor(file, &format!("{prefix}.running_mean"), &[features])?,
            running_var: load_tensor(file, &format!("{prefix}.running_var"), &[features])?,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize, channels: usize, eps: f32) {
        for row in 0..rows {
            for channel in 0..channels {
                let slot = &mut x[row * channels + channel];
                *slot = (*slot - self.running_mean[channel])
                    / (self.running_var[channel] + eps).sqrt()
                    * self.weight[channel]
                    + self.bias[channel];
            }
        }
    }
}

struct ConvModule {
    norm: LayerNorm,
    up: Vec<f32>,
    up_bias: Vec<f32>,
    depth: Vec<f32>,
    bn: EvalBatchNorm,
    down: Vec<f32>,
    down_bias: Vec<f32>,
    inner: usize,
    kernel: usize,
    hidden: usize,
    pad: usize,
}

impl ConvModule {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &EncoderConfig,
        eps: f32,
    ) -> Result<Self> {
        let inner = config.hidden_dim * config.conv_expansion_factor;
        Ok(Self {
            norm: LayerNorm::load(file, &format!("{prefix}.norm"), config.hidden_dim, eps)?,
            up: load_tensor(
                file,
                &format!("{prefix}.up_conv.weight"),
                &[2 * inner, 1, config.hidden_dim],
            )?,
            up_bias: load_tensor(file, &format!("{prefix}.up_conv.bias"), &[2 * inner])?,
            depth: load_tensor(
                file,
                &format!("{prefix}.depth_conv.weight"),
                &[inner, config.conv_kernel_size, 1],
            )?,
            bn: EvalBatchNorm::load(file, &format!("{prefix}.bn"), inner)?,
            down: load_tensor(
                file,
                &format!("{prefix}.down_conv.weight"),
                &[config.hidden_dim, 1, inner],
            )?,
            down_bias: load_tensor(
                file,
                &format!("{prefix}.down_conv.bias"),
                &[config.hidden_dim],
            )?,
            inner,
            kernel: config.conv_kernel_size,
            hidden: config.hidden_dim,
            pad: config.conv_kernel_size / 2,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, eps: f32) -> Vec<f32> {
        let mut hidden = x.to_vec();
        self.norm.apply(&mut hidden, rows);
        let up = conv1d_rows(
            &hidden,
            &self.up,
            Some(&self.up_bias),
            rows,
            self.hidden,
            2 * self.inner,
            1,
            0,
            1,
        );
        // GLU: first half times sigmoid of second half.
        let mut gated = vec![0.0f32; rows * self.inner];
        for row in 0..rows {
            for channel in 0..self.inner {
                let a = up[row * 2 * self.inner + channel];
                let gate = up[row * 2 * self.inner + self.inner + channel];
                gated[row * self.inner + channel] = a / (1.0 + (-gate).exp());
            }
        }
        let depth = conv1d_rows(
            &gated,
            &self.depth,
            None,
            rows,
            self.inner,
            self.inner,
            self.kernel,
            self.pad,
            self.inner,
        );
        let mut normalized = depth;
        self.bn.apply(&mut normalized, rows, self.inner, eps);
        ops::silu(&mut normalized);
        conv1d_rows(
            &normalized,
            &self.down,
            Some(&self.down_bias),
            rows,
            self.inner,
            self.hidden,
            1,
            0,
            1,
        )
    }
}

struct ConformerBlock {
    ff1: FeedForward,
    attention: ShawAttention,
    conv: ConvModule,
    ff2: FeedForward,
    post_norm: LayerNorm,
    eps: f32,
}

impl ConformerBlock {
    fn load(
        file: &SafetensorsFile,
        index: usize,
        config: &EncoderConfig,
        eps: f32,
    ) -> Result<Self> {
        let prefix = format!("encoder.layers.{index}");
        Ok(Self {
            ff1: FeedForward::load(
                file,
                &format!("{prefix}.ff1"),
                config.hidden_dim,
                config.feedforward_mult,
                eps,
            )?,
            attention: ShawAttention::load(file, &format!("{prefix}.attn"), config, eps)?,
            conv: ConvModule::load(file, &format!("{prefix}.conv"), config, eps)?,
            ff2: FeedForward::load(
                file,
                &format!("{prefix}.ff2"),
                config.hidden_dim,
                config.feedforward_mult,
                eps,
            )?,
            post_norm: LayerNorm::load(
                file,
                &format!("{prefix}.post_norm"),
                config.hidden_dim,
                eps,
            )?,
            eps,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = scale_add(x, &self.ff1.forward(x, rows), 0.5);
        hidden = add(&hidden, &self.attention.forward(&hidden, rows));
        hidden = add(&hidden, &self.conv.forward(&hidden, rows, self.eps));
        hidden = scale_add(&hidden, &self.ff2.forward(&hidden, rows), 0.5);
        self.post_norm.apply(&mut hidden, rows);
        hidden
    }
}

struct ConformerEncoder {
    input_linear: Linear,
    blocks: Vec<ConformerBlock>,
    out: Linear,
    out_mid: Linear,
    out_bpe: Linear,
    layer_indices: Vec<usize>,
    self_conditioning: usize,
    pooling_window: usize,
}

struct EncoderOutput {
    bpe_logits: Vec<f32>,
    bpe_steps: usize,
    hidden_states: Vec<Vec<f32>>,
}

impl ConformerEncoder {
    fn load(file: &SafetensorsFile, config: &GraniteSpeechNarConfig) -> Result<Self> {
        let encoder_config = &config.encoder;
        let blocks = (0..encoder_config.num_layers)
            .map(|index| ConformerBlock::load(file, index, encoder_config, 1e-5))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_linear: Linear::load(
                file,
                "encoder.input_linear",
                encoder_config.input_dim,
                encoder_config.hidden_dim,
                true,
            )?,
            blocks,
            out: Linear::load(
                file,
                "encoder.out",
                encoder_config.hidden_dim,
                encoder_config.output_dim,
                true,
            )?,
            out_mid: Linear::load(
                file,
                "encoder.out_mid",
                encoder_config.output_dim,
                encoder_config.hidden_dim,
                true,
            )?,
            out_bpe: Linear::load(
                file,
                "encoder.out_bpe",
                encoder_config.hidden_dim,
                encoder_config.bpe_output_dim,
                true,
            )?,
            layer_indices: config.encoder_layer_indices.clone(),
            self_conditioning: encoder_config.self_conditioning_layer,
            pooling_window: encoder_config.bpe_pooling_window,
        })
    }

    fn forward(&self, features: &[f32], steps: usize) -> Result<EncoderOutput> {
        let hidden_dim = self
            .blocks
            .first()
            .map(|block| block.ff1.down.output)
            .unwrap_or(0);
        let mut hidden = self.input_linear.forward(features, steps);
        let mut all_hidden = vec![hidden.clone()];
        let mut char_logits = None;
        let mut blank_probs = None;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, steps);
            if index + 1 == self.self_conditioning {
                let logits = self.out.forward(&hidden, steps);
                let mut probabilities = logits.clone();
                for row in 0..steps {
                    let row_slice = &mut probabilities[row * 348..(row + 1) * 348];
                    ops::softmax_row(row_slice);
                }
                blank_probs = Some(
                    (0..steps)
                        .map(|row| probabilities[row * 348])
                        .collect::<Vec<_>>(),
                );
                let projected = self.out_mid.forward(&probabilities, steps);
                hidden = add(&hidden, &projected);
                char_logits = Some(logits);
            }
            all_hidden.push(hidden.clone());
        }
        let Some(blank_probs) = blank_probs else {
            return Err(SpeechError::BadConfig {
                field: "self_conditioning_layer".into(),
                why: "was not reached; config mismatch".into(),
            });
        };
        let hidden_states = self
            .layer_indices
            .iter()
            .map(|&index| all_hidden[index].clone())
            .collect::<Vec<_>>();
        // Posterior-weighted pooling in windows of `pooling_window`.
        let pooled_steps = steps.div_ceil(self.pooling_window);
        let mut pooled = vec![0.0f32; pooled_steps * hidden_dim];
        for window in 0..pooled_steps {
            let start = window * self.pooling_window;
            let end = (start + self.pooling_window).min(steps);
            let mut weight_sum = 0.0f32;
            for probability in &blank_probs[start..end] {
                weight_sum += 1.0 - probability;
            }
            let denominator = weight_sum.max(1e-6);
            for (offset, probability) in blank_probs[start..end].iter().enumerate() {
                let step = start + offset;
                let weight = (1.0 - probability) / denominator;
                for dim in 0..hidden_dim {
                    pooled[window * hidden_dim + dim] += weight * hidden[step * hidden_dim + dim];
                }
            }
        }
        let bpe_steps = pooled_steps;
        let bpe_logits = self.out_bpe.forward(&pooled, bpe_steps);
        let _ = char_logits;
        Ok(EncoderOutput {
            bpe_logits,
            bpe_steps,
            hidden_states,
        })
    }
}

/// CTC greedy collapse: dedup adjacent repeats, then drop blanks.
fn ctc_collapse_decode(tokens: &[i32], blank_id: usize) -> Vec<i32> {
    let mut output = Vec::new();
    let mut previous: Option<i32> = None;
    for &token in tokens {
        if Some(token) != previous && token as usize != blank_id {
            output.push(token);
        }
        previous = Some(token);
    }
    output
}

/// Interleave blanks as editing slots: CTC tokens at odd indices, blanks at
/// even indices and any padding up to `max(2n+1, min_len)`.
fn add_insertion_slots(token_ids: &[i32], blank_id: i32, min_len: usize) -> Vec<i32> {
    let total = (2 * token_ids.len() + 1).max(min_len);
    let mut output = Vec::with_capacity(total);
    for &token in token_ids {
        output.push(blank_id);
        output.push(token);
    }
    while output.len() < total {
        output.push(blank_id);
    }
    output
}

struct QFormerLayer {
    attn_norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    mlp_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    head_dim: usize,
    hidden: usize,
}

impl QFormerLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &ProjectorConfig,
        eps: f32,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_heads;
        Ok(Self {
            attn_norm: LayerNorm::load(file, &format!("{prefix}.attn_norm"), hidden, eps)?,
            q: Linear::load(
                file,
                &format!("{prefix}.cross_attention.q_proj"),
                hidden,
                hidden,
                true,
            )?,
            k: Linear::load(
                file,
                &format!("{prefix}.cross_attention.k_proj"),
                hidden,
                hidden,
                true,
            )?,
            v: Linear::load(
                file,
                &format!("{prefix}.cross_attention.v_proj"),
                hidden,
                hidden,
                true,
            )?,
            o: Linear::load(
                file,
                &format!("{prefix}.cross_attention.o_proj"),
                hidden,
                hidden,
                true,
            )?,
            mlp_norm: LayerNorm::load(file, &format!("{prefix}.mlp_norm"), hidden, eps)?,
            fc1: Linear::load(
                file,
                &format!("{prefix}.mlp.fc1"),
                hidden,
                hidden * config.mlp_ratio,
                true,
            )?,
            fc2: Linear::load(
                file,
                &format!("{prefix}.mlp.fc2"),
                hidden * config.mlp_ratio,
                hidden,
                true,
            )?,
            heads: config.num_heads,
            head_dim,
            hidden,
        })
    }

    /// `query`: [q_steps, hidden]; `kv`: [kv_steps, hidden]; both rows-major.
    fn forward(&self, query: &[f32], kv: &[f32], q_steps: usize, kv_steps: usize) -> Vec<f32> {
        let mut normed_query = query.to_vec();
        self.attn_norm.apply(&mut normed_query, q_steps);
        let q = self.q.forward(&normed_query, q_steps);
        let k = self.k.forward(kv, kv_steps);
        let v = self.v.forward(kv, kv_steps);
        let mut attended = vec![0.0f32; q_steps * self.hidden];
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut scores = vec![0.0f32; kv_steps];
        for head in 0..self.heads {
            for q_step in 0..q_steps {
                let q_offset = q_step * self.hidden + head * self.head_dim;
                for (k_step, score_slot) in scores.iter_mut().enumerate() {
                    let k_offset = k_step * self.hidden + head * self.head_dim;
                    let mut score = 0.0f32;
                    for dim in 0..self.head_dim {
                        score += q[q_offset + dim] * k[k_offset + dim];
                    }
                    *score_slot = score * scale;
                }
                ops::softmax_row(&mut scores);
                let out_offset = q_step * self.hidden + head * self.head_dim;
                for dim in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for k_step in 0..kv_steps {
                        sum +=
                            scores[k_step] * v[k_step * self.hidden + head * self.head_dim + dim];
                    }
                    attended[out_offset + dim] = sum;
                }
            }
        }
        let projected = self.o.forward(&attended, q_steps);
        let hidden = add(query, &projected);
        let mut normed = hidden.clone();
        self.mlp_norm.apply(&mut normed, q_steps);
        let mut mlp = self.fc1.forward(&normed, q_steps);
        gelu_in_place(&mut mlp);
        let mlp = self.fc2.forward(&mlp, q_steps);
        add(&hidden, &mlp)
    }
}

struct Projector {
    layer_norms: Vec<LayerNorm>,
    layer_projector: Linear,
    query: Vec<f32>,
    window_positions: Vec<f32>,
    qformer: Vec<QFormerLayer>,
    out_norm: LayerNorm,
    out_linear: Linear,
    config: ProjectorConfig,
}

impl Projector {
    fn load(file: &SafetensorsFile, config: &GraniteSpeechNarConfig) -> Result<Self> {
        let projector = &config.projector;
        let layer_norms = (0..projector.num_encoder_layers)
            .map(|index| {
                LayerNorm::load(
                    file,
                    &format!("projector.layer_norms.{index}"),
                    projector.encoder_dim,
                    1e-6,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let query = load_tensor(
            file,
            "projector.query",
            &[
                1,
                projector.block_size / projector.downsample_rate,
                projector.hidden_size,
            ],
        )?;
        let window_positions = load_tensor(
            file,
            "projector.window_positions",
            &[1, projector.block_size, projector.hidden_size],
        )?;
        let qformer = (0..projector.num_layers)
            .map(|index| {
                QFormerLayer::load(
                    file,
                    &format!("projector.qformer.layers.{index}"),
                    projector,
                    1e-6,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layer_norms,
            layer_projector: Linear::load(
                file,
                "projector.layer_projector",
                projector.num_encoder_layers * projector.encoder_dim,
                projector.hidden_size,
                true,
            )?,
            query,
            window_positions,
            qformer,
            out_norm: LayerNorm::load(file, "projector.out_norm", projector.hidden_size, 1e-6)?,
            out_linear: Linear::load(
                file,
                "projector.out_linear",
                projector.hidden_size,
                projector.llm_dim,
                true,
            )?,
            config: projector.clone(),
        })
    }

    /// `hidden_states`: [steps, num_encoder_layers * encoder_dim] rows-major.
    fn forward(&self, hidden_states: &[f32], steps: usize) -> Vec<f32> {
        let projector = &self.config;
        let mut fused = vec![0.0f32; steps * projector.num_encoder_layers * projector.encoder_dim];
        for layer in 0..projector.num_encoder_layers {
            let slice_width = projector.encoder_dim;
            let mut normed = vec![0.0f32; steps * slice_width];
            for step in 0..steps {
                normed[step * slice_width..(step + 1) * slice_width].copy_from_slice(
                    &hidden_states[step * projector.num_encoder_layers * slice_width
                        + layer * slice_width
                        ..step * projector.num_encoder_layers * slice_width
                            + (layer + 1) * slice_width],
                );
            }
            self.layer_norms[layer].apply(&mut normed, steps);
            for step in 0..steps {
                let target =
                    step * projector.num_encoder_layers * slice_width + layer * slice_width;
                fused[target..target + slice_width]
                    .copy_from_slice(&normed[step * slice_width..(step + 1) * slice_width]);
            }
        }
        let mut projected = self.layer_projector.forward(&fused, steps);
        gelu_in_place(&mut projected);

        let block = projector.block_size;
        let padded_steps = steps + (block - steps % block) % block;
        projected.resize(padded_steps * projector.hidden_size, 0.0);
        let nblocks = padded_steps / block;
        let query_length = block / projector.downsample_rate;
        let mut audio_embeddings = vec![0.0f32; nblocks * query_length * projector.llm_dim];
        for window in 0..nblocks {
            // Mean-pool groups of downsample_rate frames as the query init.
            let mut query = vec![0.0f32; query_length * projector.hidden_size];
            for slot in 0..query_length {
                for offset in 0..projector.downsample_rate {
                    let step = window * block + slot * projector.downsample_rate + offset;
                    for dim in 0..projector.hidden_size {
                        query[slot * projector.hidden_size + dim] += projected
                            [step * projector.hidden_size + dim]
                            / projector.downsample_rate as f32;
                    }
                }
                for dim in 0..projector.hidden_size {
                    query[slot * projector.hidden_size + dim] += self.query[dim];
                    // broadcast learned query, batch dim dropped
                }
            }
            let window_start = window * block;
            let mut kv = vec![0.0f32; block * projector.hidden_size];
            for step in 0..block {
                for dim in 0..projector.hidden_size {
                    kv[step * projector.hidden_size + dim] = projected
                        [(window_start + step) * projector.hidden_size + dim]
                        + self.window_positions[step * projector.hidden_size + dim];
                }
            }
            let mut out = query;
            for layer in &self.qformer {
                out = layer.forward(&out, &kv, query_length, block);
            }
            self.out_norm.apply(&mut out, query_length);
            let projected_out = self.out_linear.forward(&out, query_length);
            audio_embeddings[window * query_length * projector.llm_dim
                ..(window + 1) * query_length * projector.llm_dim]
                .copy_from_slice(&projected_out);
        }
        audio_embeddings
    }
}

struct RmsNorm {
    weight: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl RmsNorm {
    fn load(file: &SafetensorsFile, prefix: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: load_tensor(file, &format!("{prefix}.weight"), &[width])?,
            width,
            epsilon,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        for row in 0..rows {
            let slice = &mut x[row * self.width..(row + 1) * self.width];
            let sum = slice.iter().map(|v| v * v).sum::<f32>();
            let scale = (1.0 / (sum / self.width as f32 + self.epsilon).sqrt()) * 1.0;
            for (value, weight) in slice.iter_mut().zip(&self.weight) {
                *value = *value * scale * weight;
            }
        }
    }
}

/// Half-rotation (NeoX) RoPE over packed heads: each row holds
/// `heads` segments of `head_dim`, and each segment's first half rotates
/// against its second half at the row's position.
fn rope_half(values: &mut [f32], rows: usize, heads: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let row_width = heads * head_dim;
    for row in 0..rows {
        for head in 0..heads {
            let base = row * row_width + head * head_dim;
            for index in 0..half {
                let angle = (row as f32) / theta.powf((2 * index) as f32 / head_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let a = values[base + index];
                let b = values[base + half + index];
                values[base + index] = a * cos - b * sin;
                values[base + half + index] = a * sin + b * cos;
            }
        }
    }
}

struct EditorAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_theta: f32,
    hidden: usize,
}

impl EditorAttention {
    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let q = self.q.forward(x, steps);
        let k = self.k.forward(x, steps);
        let v = self.v.forward(x, steps);
        let mut q_headed = q;
        let mut k_headed = k;
        rope_half(
            &mut q_headed,
            steps,
            self.heads,
            self.head_dim,
            self.rope_theta,
        );
        rope_half(
            &mut k_headed,
            steps,
            self.kv_heads,
            self.head_dim,
            self.rope_theta,
        );
        let kv_group = self.heads / self.kv_heads;
        let mut attended = vec![0.0f32; steps * self.hidden];
        let mut scores = vec![0.0f32; steps];
        for head in 0..self.heads {
            let kv_head = head / kv_group;
            for q_step in 0..steps {
                let q_offset = q_step * self.heads * self.head_dim + head * self.head_dim;
                for (k_step, score_slot) in scores.iter_mut().enumerate() {
                    let k_offset = k_step * self.kv_heads * self.head_dim + kv_head * self.head_dim;
                    let mut score = 0.0f32;
                    for dim in 0..self.head_dim {
                        score += q_headed[q_offset + dim] * k_headed[k_offset + dim];
                    }
                    *score_slot = score * self.scale;
                }
                ops::softmax_row(&mut scores);
                let out_offset = q_step * self.heads * self.head_dim + head * self.head_dim;
                for dim in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for k_step in 0..steps {
                        sum += scores[k_step]
                            * v[k_step * self.kv_heads * self.head_dim
                                + kv_head * self.head_dim
                                + dim];
                    }
                    attended[out_offset + dim] = sum;
                }
            }
        }
        self.o.forward(&attended, steps)
    }
}

struct EditorLayer {
    input_norm: RmsNorm,
    attention: EditorAttention,
    post_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    residual: f32,
}

impl EditorLayer {
    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.input_norm.apply(&mut normalized, steps);
        let attention = self.attention.forward(&normalized, steps);
        let mut hidden = scale_add(x, &attention, self.residual);
        normalized.copy_from_slice(&hidden);
        self.post_norm.apply(&mut normalized, steps);
        let mut gate = self.gate.forward(&normalized, steps);
        ops::silu(&mut gate);
        let up = self.up.forward(&normalized, steps);
        let mut product = vec![0.0f32; gate.len()];
        for (out, (gate_value, up_value)) in product.iter_mut().zip(gate.iter().zip(&up)) {
            *out = gate_value * up_value;
        }
        let mlp = self.down.forward(&product, steps);
        hidden = scale_add(&hidden, &mlp, self.residual);
        hidden
    }
}

struct Editor {
    embedding: Vec<f32>,
    layers: Vec<EditorLayer>,
    norm: RmsNorm,
    embedding_multiplier: f32,
    logits_scaling: f32,
    hidden: usize,
    vocab: usize,
}

impl Editor {
    fn load(file: &SafetensorsFile, config: &GraniteSpeechNarConfig) -> Result<Self> {
        let text = &config.text;
        let head_dim = text.hidden_size / text.num_attention_heads;
        let layers = (0..text.num_hidden_layers)
            .map(|index| {
                let prefix = format!("editor.layers.{index}");
                Ok(EditorLayer {
                    input_norm: RmsNorm::load(
                        file,
                        &format!("{prefix}.input_layernorm"),
                        text.hidden_size,
                        text.rms_norm_eps,
                    )?,
                    attention: EditorAttention {
                        q: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.q_proj"),
                            text.hidden_size,
                            text.num_attention_heads * head_dim,
                            false,
                        )?,
                        k: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.k_proj"),
                            text.hidden_size,
                            text.num_key_value_heads * head_dim,
                            false,
                        )?,
                        v: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.v_proj"),
                            text.hidden_size,
                            text.num_key_value_heads * head_dim,
                            false,
                        )?,
                        o: Linear::load(
                            file,
                            &format!("{prefix}.self_attn.o_proj"),
                            text.num_attention_heads * head_dim,
                            text.hidden_size,
                            false,
                        )?,
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim,
                        scale: text.attention_multiplier,
                        rope_theta: text.rope_theta,
                        hidden: text.hidden_size,
                    },
                    post_norm: RmsNorm::load(
                        file,
                        &format!("{prefix}.post_attention_layernorm"),
                        text.hidden_size,
                        text.rms_norm_eps,
                    )?,
                    gate: Linear::load(
                        file,
                        &format!("{prefix}.mlp.gate_proj"),
                        text.hidden_size,
                        text.intermediate_size,
                        false,
                    )?,
                    up: Linear::load(
                        file,
                        &format!("{prefix}.mlp.up_proj"),
                        text.hidden_size,
                        text.intermediate_size,
                        false,
                    )?,
                    down: Linear::load(
                        file,
                        &format!("{prefix}.mlp.down_proj"),
                        text.intermediate_size,
                        text.hidden_size,
                        false,
                    )?,
                    residual: text.residual_multiplier,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: load_tensor(
                file,
                "editor.embed_tokens.weight",
                &[text.vocab_size, text.hidden_size],
            )?,
            layers,
            norm: RmsNorm::load(file, "editor.norm", text.hidden_size, text.rms_norm_eps)?,
            embedding_multiplier: text.embedding_multiplier,
            logits_scaling: text.logits_scaling,
            hidden: text.hidden_size,
            vocab: text.vocab_size,
        })
    }

    fn embed(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; ids.len() * self.hidden];
        for (row, &id) in ids.iter().enumerate() {
            if id < 0 || id as usize >= self.vocab {
                return Err(SpeechError::Input {
                    why: "editor token id outside the vocabulary".into(),
                });
            }
            output[row * self.hidden..(row + 1) * self.hidden].copy_from_slice(
                &self.embedding[id as usize * self.hidden..(id as usize + 1) * self.hidden],
            );
        }
        Ok(output)
    }

    /// Full bidirectional forward; returns logits for rows >= `logits_start`.
    fn forward(&self, inputs: &[f32], steps: usize, logits_start: usize) -> Vec<f32> {
        let mut hidden = vec![0.0f32; inputs.len()];
        for (value, input) in hidden.iter_mut().zip(inputs) {
            *value = input * self.embedding_multiplier;
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden, steps);
        }
        self.norm.apply(&mut hidden, steps);
        let tail_steps = steps - logits_start;
        let mut logits = vec![0.0f32; tail_steps * self.vocab];
        for row in 0..tail_steps {
            let source =
                &hidden[(logits_start + row) * self.hidden..(logits_start + row + 1) * self.hidden];
            let out_row = &mut logits[row * self.vocab..(row + 1) * self.vocab];
            for (index, out) in out_row.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (value, weight) in source
                    .iter()
                    .zip(&self.embedding[index * self.hidden..(index + 1) * self.hidden])
                {
                    sum += value * weight;
                }
                *out = sum / self.logits_scaling;
            }
        }
        logits
    }
}

/// Loaded Granite Speech NAR model.
pub struct GraniteSpeechNar {
    config: GraniteSpeechNarConfig,
    encoder: ConformerEncoder,
    projector: Projector,
    editor: Editor,
    tokenizer: turbospark_tokenizer::Tokenizer,
}

impl GraniteSpeechNar {
    /// Load the pinned profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_json = fs::read_to_string(model_dir.join("config.json")).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read config.json: {error}"),
            }
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = GraniteSpeechNarConfig::from_json(&root)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = ConformerEncoder::load(&file, &config)?;
        let projector = Projector::load(&file, &config)?;
        let editor = Editor::load(&file, &config)?;
        let tokenizer = load_tokenizer(model_dir)?;
        Ok(Self {
            config,
            encoder,
            projector,
            editor,
            tokenizer,
        })
    }

    pub fn profile(&self) -> GraniteSpeechNarProfile {
        GRANITE_SPEECH_4_1_2B_NAR
    }

    pub fn config(&self) -> &GraniteSpeechNarConfig {
        &self.config
    }

    /// Transcribe one mono 16 kHz waveform.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let features = compute_features(samples)?;
        let steps = features.len() / self.config.encoder.input_dim;
        let enc_out = self.encoder.forward(&features, steps)?;
        let bpe_vocab = self.config.encoder.bpe_output_dim;
        let hypothesis_ids: Vec<i32> = (0..enc_out.bpe_steps)
            .map(|row| {
                enc_out.bpe_logits[row * bpe_vocab..(row + 1) * bpe_vocab]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as i32
            })
            .collect();
        let hypothesis = ctc_collapse_decode(&hypothesis_ids, self.config.blank_token_id);

        let mut fused_steps = enc_out.hidden_states[0].len() / self.config.projector.encoder_dim;
        let mut fused = vec![
            0.0f32;
            fused_steps
                * self.config.projector.num_encoder_layers
                * self.config.projector.encoder_dim
        ];
        for (index, state) in enc_out.hidden_states.iter().enumerate() {
            fused_steps = state.len() / self.config.projector.encoder_dim;
            let width = self.config.projector.encoder_dim;
            for step in 0..fused_steps {
                fused[step * self.config.projector.num_encoder_layers * width + index * width
                    ..step * self.config.projector.num_encoder_layers * width
                        + (index + 1) * width]
                    .copy_from_slice(&state[step * width..(step + 1) * width]);
            }
        }
        let mut audio_embeds = self.projector.forward(&fused, fused_steps);
        let multiplier = self.config.text.embedding_multiplier;
        for value in audio_embeds.iter_mut() {
            *value /= multiplier;
        }

        let text_ids = add_insertion_slots(
            &hypothesis,
            self.config.blank_token_id as i32,
            self.config.min_edit_sequence_length,
        );
        let text_embeds = self.editor.embed(&text_ids)?;

        let mut inputs = audio_embeds.clone();
        inputs.extend_from_slice(&text_embeds);
        let total_steps = inputs.len() / self.config.text.hidden_size;
        let logits = self.editor.forward(
            &inputs,
            total_steps,
            audio_embeds.len() / self.config.text.hidden_size,
        );
        let edited: Vec<i32> = (0..logits.len() / self.config.text.vocab_size)
            .map(|row| {
                logits[row * self.config.text.vocab_size..(row + 1) * self.config.text.vocab_size]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as i32
            })
            .collect();
        let final_ids = ctc_collapse_decode(&edited, self.config.blank_token_id);
        let token_ids: Vec<u32> = final_ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "token id does not fit the tokenizer interface".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.tokenizer
            .decode(&token_ids, true)
            .map(|text| text.trim().to_owned())
            .map_err(|error| SpeechError::Input {
                why: format!("tokenizer decode failed: {error}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        add_insertion_slots, ctc_collapse_decode, GraniteSpeechNar, GRANITE_SPEECH_4_1_2B_NAR,
    };
    use serde_json::Value;
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/granite_speech_nar_reference.json");

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid granite_speech_nar reference fixture")
    }

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn compare_spots(actual: &[f32], spots: &Spots, label: &str, gate: f32) -> f32 {
        let total_columns = spots.shape.iter().product::<usize>() / spots.shape[0];
        assert!(
            actual.len() >= spots.shape[0] * total_columns,
            "{label} size"
        );
        let mut worst = 0.0f32;
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (column_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                let diff = (actual[row * total_columns + column] - expected).abs();
                worst = worst.max(diff);
                assert!(
                    diff < gate,
                    "{label} [{row},{column}] differs by {diff} (gate {gate})"
                );
            }
        }
        worst
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(
            GRANITE_SPEECH_4_1_2B_NAR.repository,
            "mlx-community/granite-speech-4.1-2b-nar-mlx"
        );
        assert_eq!(
            GRANITE_SPEECH_4_1_2B_NAR.revision,
            "6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c"
        );
    }

    #[test]
    fn ctc_collapse_dedups_then_drops_blanks() {
        let tokens = [5, 5, 100257, 7, 7, 7, 100257, 9];
        assert_eq!(ctc_collapse_decode(&tokens, 100257), [5, 7, 9]);
        assert!(ctc_collapse_decode(&[100257, 100257], 100257).is_empty());
    }

    #[test]
    fn insertion_slots_interleave_blanks_at_even_positions() {
        let slots = add_insertion_slots(&[5, 7], 100257, 8);
        assert_eq!(
            slots,
            [100257, 5, 100257, 7, 100257, 100257, 100257, 100257]
        );
        assert_eq!(add_insertion_slots(&[], 100257, 8), [100257; 8]);
        let long = add_insertion_slots(&[1, 2, 3, 4, 5], 100257, 8);
        assert_eq!(long.len(), 11);
        assert_eq!(long[1], 1);
        assert_eq!(long[9], 5);
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio granite_speech_nar at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
    }

    #[test]
    #[ignore = "requires the pinned checkpoint in TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model_dir) = std::env::var_os("TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR") else {
            eprintln!("skipping: TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR is unset");
            return;
        };
        let model = GraniteSpeechNar::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), GRANITE_SPEECH_4_1_2B_NAR);
        let fixture = load_fixture();

        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);

        let features = super::compute_features(&waveform.samples).expect("features");
        let feature_spots: Spots =
            serde_json::from_value(fixture["input_features"].clone()).unwrap();
        compare_spots(&features, &feature_spots, "input_features", 2.0e-4);

        let steps = features.len() / model.config.encoder.input_dim;
        let enc_out = model.encoder.forward(&features, steps).expect("encode");
        let bpe_spots: Spots = serde_json::from_value(fixture["bpe_logits"].clone()).unwrap();
        compare_spots(&enc_out.bpe_logits, &bpe_spots, "bpe_logits", 3.0e-1);
        let hypothesis_ids: Vec<i32> = fixture["hypothesis_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        let computed: Vec<i32> = (0..enc_out.bpe_steps)
            .map(|row| {
                let vocab = model.config.encoder.bpe_output_dim;
                enc_out.bpe_logits[row * vocab..(row + 1) * vocab]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as i32
            })
            .collect();
        let collapsed = ctc_collapse_decode(&computed, model.config.blank_token_id);
        assert_eq!(collapsed, hypothesis_ids, "initial CTC hypothesis");

        let projector_state0: Spots =
            serde_json::from_value(fixture["projector_hidden_state0"].clone()).unwrap();
        compare_spots(
            &enc_out.hidden_states[0],
            &projector_state0,
            "projector_hidden_state0",
            2.0e-2,
        );

        let mut fused = vec![
            0.0f32;
            steps
                * model.config.projector.num_encoder_layers
                * model.config.projector.encoder_dim
        ];
        let width = model.config.projector.encoder_dim;
        for (index, state) in enc_out.hidden_states.iter().enumerate() {
            for step in 0..steps {
                fused[step * model.config.projector.num_encoder_layers * width + index * width
                    ..step * model.config.projector.num_encoder_layers * width
                        + (index + 1) * width]
                    .copy_from_slice(&state[step * width..(step + 1) * width]);
            }
        }
        let mut audio_embeds = model.projector.forward(&fused, steps);
        let multiplier = model.config.text.embedding_multiplier;
        for value in audio_embeds.iter_mut() {
            *value /= multiplier;
        }
        let embedding_spots: Spots =
            serde_json::from_value(fixture["audio_embeddings"].clone()).unwrap();
        compare_spots(&audio_embeds, &embedding_spots, "audio_embeddings", 1.0);

        let text_ids: Vec<i32> = fixture["editor_text_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        let text_embeds = model.editor.embed(&text_ids).expect("embed");
        let mut inputs = audio_embeds.clone();
        inputs.extend_from_slice(&text_embeds);
        let total_steps = inputs.len() / model.config.text.hidden_size;
        let logits_start = audio_embeds.len() / model.config.text.hidden_size;
        let logits = model.editor.forward(&inputs, total_steps, logits_start);
        let logit_spots: Spots = serde_json::from_value(fixture["editor_logits"].clone()).unwrap();
        compare_spots(&logits, &logit_spots, "editor_logits", 1.0e1);

        let edited: Vec<i32> = (0..logits.len() / model.config.text.vocab_size)
            .map(|row| {
                logits[row * model.config.text.vocab_size..(row + 1) * model.config.text.vocab_size]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as i32
            })
            .collect();
        let final_ids = ctc_collapse_decode(&edited, model.config.blank_token_id);
        let expected_final: Vec<i64> = fixture["final_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap())
            .collect();
        let computed_final: Vec<i64> = final_ids.iter().map(|&id| id as i64).collect();
        assert_eq!(computed_final, expected_final, "final token ids");

        let transcript = model.transcribe(&waveform.samples).expect("transcribes");
        let expected = fixture["transcript"].as_str().unwrap();
        assert_eq!(transcript, expected, "transcript must match the reference");
    }
}
