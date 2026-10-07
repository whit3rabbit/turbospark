//! Nemotron VoiceChat audio codec (22.05 kHz, 31 scalar codebooks).
//!
//! Reference: `mlx_audio/codec/models/nemotron_voicechat/codec.py` and
//! `config.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/nemotron_voicechat).
//! A channels-last ConvNeXt encoder/decoder pair over 16-point STFT
//! real/imag features (hop 4), a probabilistic residual vector
//! quantizer whose codebooks are plain Gaussian means (Euclidean
//! argmin, no normalization), and a magnitude/phase decoder head with
//! a softplus-bounded magnitude and an envelope-clamped iSTFT.
//!
//! Scope: the batch `encode` / `decode` contract. The streaming
//! `decode_step` path (per-layer `CausalConv1dCache` plus the
//! spectrogram-overlap cache bookkeeping around iSTFT) is not ported;
//! the batch path left-pads each causal conv, which is the cache's
//! reset state. The checkpoint's `variance_list` tensors are loaded
//! (and fixtures seed them) because the reference keeps them in the
//! parameter tree, but they only matter to training/generative heads
//! outside this contract.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{
    load_bias, load_mlx_conv_weight, load_mlx_convt_weight, BiasMode, Conv1d, ConvTranspose1d,
};
use crate::codec::wnconv::load_f32_shaped;
use crate::fft::{ComplexF32, RealFftPlan};
use crate::ops;
use crate::{dsp, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Codec geometry, one-to-one with the reference dataclass defaults.
#[derive(Debug, Clone)]
pub struct NemotronVoiceChatConfig {
    pub sample_rate: u32,
    pub base_channels: usize,
    pub channel_multipliers: Vec<usize>,
    pub downsample_rates: Vec<usize>,
    pub blocks_per_stage: usize,
    pub block_kernel_size: usize,
    pub latent_dim: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
}

impl NemotronVoiceChatConfig {
    /// Parses a `config.json`-style object with the dataclass
    /// defaults.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u =
            |field: &str| -> Result<Option<u64>> { Ok(value.get(field).and_then(|v| v.as_u64())) };
        let usize_vec = |field: &str| -> Result<Option<Vec<usize>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_u64().map(|n| n as usize))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of integers".to_string(),
                            })?,
                    ))
                }
            }
        };
        Ok(NemotronVoiceChatConfig {
            sample_rate: u("sample_rate")?.unwrap_or(22_050) as u32,
            base_channels: u("base_channels")?.unwrap_or(384) as usize,
            channel_multipliers: usize_vec("channel_multipliers")?.unwrap_or_else(|| vec![1, 2, 4]),
            downsample_rates: usize_vec("downsample_rates")?.unwrap_or_else(|| vec![7, 7, 9]),
            blocks_per_stage: u("blocks_per_stage")?.unwrap_or(3) as usize,
            block_kernel_size: u("block_kernel_size")?.unwrap_or(7) as usize,
            latent_dim: u("latent_dim")?.unwrap_or(512) as usize,
            n_fft: u("n_fft")?.unwrap_or(16) as usize,
            hop_length: u("hop_length")?.unwrap_or(4) as usize,
            num_quantizers: u("num_quantizers")?.unwrap_or(31) as usize,
            codebook_size: u("codebook_size")?.unwrap_or(1024) as usize,
        })
    }

    /// Samples per emitted token (hop times every stage stride).
    pub fn waveform_to_token_ratio(&self) -> usize {
        self.hop_length * self.downsample_rates.iter().product::<usize>()
    }

    /// Frame rate in tokens per second.
    pub fn frame_rate(&self) -> f32 {
        self.sample_rate as f32 / self.waveform_to_token_ratio() as f32
    }

    /// Encoder input width: real and imaginary rFFT bins.
    pub fn stft_channels(&self) -> usize {
        2 * (self.n_fft / 2 + 1)
    }

    fn stage_channels(&self) -> Vec<usize> {
        self.channel_multipliers
            .iter()
            .map(|m| self.base_channels * m)
            .collect()
    }
}

/// Channels-last reference, channel-major kernel: pointwise 1x1 conv
/// `out[o, t] = sum_i w[o, i] * x[i, t] + b[o]` with the PyTorch
/// `[out, in, 1]` weight.
#[derive(Debug, Clone)]
struct NvcPointwise {
    in_ch: usize,
    out_ch: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl NvcPointwise {
    /// Loads a stored MLX 1x1 conv `[out, 1, in]`.
    fn load(file: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Self> {
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, 1, in_ch])?;
        let mut weight = vec![0.0f32; out_ch * in_ch];
        for o in 0..out_ch {
            for i in 0..in_ch {
                weight[o * in_ch + i] = stored[o * in_ch + i];
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(NvcPointwise {
            in_ch,
            out_ch,
            weight,
            bias,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_ch * frames];
        for t in 0..frames {
            for o in 0..self.out_ch {
                let mut acc = 0.0f32;
                let w_row = &self.weight[o * self.in_ch..(o + 1) * self.in_ch];
                for i in 0..self.in_ch {
                    acc += w_row[i] * x[i * frames + t];
                }
                out[o * frames + t] = acc + self.bias.as_deref().map_or(0.0, |b| b[o]);
            }
        }
        out
    }
}

/// Strided (or plain) Conv1d over channel-major activations, loaded
/// from the MLX layout `[out, K, in/groups]`. The depthwise variant
/// (`groups = channels`) stores `[C, K, 1]`.
///
/// `depthwise` selects the `[C, K, 1]` stored layout and
/// `groups = channels`; otherwise the layout is `[out, K, in]`
/// with groups 1.
fn load_nvc_conv(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    depthwise: bool,
) -> Result<Conv1d> {
    let (groups, stored_in) = if depthwise { (in_ch, 1) } else { (1, in_ch) };
    let weight =
        load_mlx_conv_weight(file, &format!("{prefix}.weight"), out_ch, kernel, stored_in)?;
    let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: 0,
        dilation: 1,
        groups,
        weight,
        bias,
    })
}

/// ConvTranspose1d stored PyTorch `[in, out, K]` (`kernel = stride =
/// rate`). No bias, output_padding 0, groups 1. MLX stores the
/// transposed conv out-first `[out, K, in]`; converts to the PyTorch
/// `[in, out, K]` the crate kernel takes.
fn load_nvc_convtr(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    rate: usize,
) -> Result<ConvTranspose1d> {
    let weight = load_mlx_convt_weight(file, &format!("{prefix}.weight"), out_ch, rate, in_ch)?;
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel: rate,
        stride: rate,
        padding: 0,
        output_padding: 0,
        groups: 1,
        weight,
        bias: None,
    })
}

/// LayerNorm over the channel axis (per frame), eps 1e-6.
#[derive(Debug, Clone)]
struct ChannelLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ChannelLayerNorm {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Self> {
        Ok(ChannelLayerNorm {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[channels])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[channels])?,
        })
    }

    /// `x [channels, frames]` normalized in place.
    fn forward(&self, x: &mut [f32], channels: usize, frames: usize) {
        for t in 0..frames {
            let col = &mut x[t..];
            let mean: f32 = (0..channels).map(|c| col[c * frames]).sum::<f32>() / channels as f32;
            let var = (0..channels)
                .map(|c| {
                    let d = col[c * frames] - mean;
                    d * d
                })
                .sum::<f32>()
                / channels as f32;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for c in 0..channels {
                col[c * frames] = (col[c * frames] - mean) * inv * self.weight[c] + self.bias[c];
            }
        }
    }
}

/// ConvNeXt block: causal depthwise conv (left pad `k - 1`), channel
/// norm, pwconv1, exact GELU, pwconv2, residual.
#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: ChannelLayerNorm,
    pwconv1: NvcPointwise,
    pwconv2: NvcPointwise,
    channels: usize,
}

impl ConvNeXtBlock {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize, kernel: usize) -> Result<Self> {
        Ok(ConvNeXtBlock {
            dwconv: load_nvc_conv(
                file,
                &format!("{prefix}.dwconv"),
                channels,
                channels,
                kernel,
                1,
                true,
            )?,
            norm: ChannelLayerNorm::load(file, &format!("{prefix}.norm"), channels)?,
            pwconv1: NvcPointwise::load(
                file,
                &format!("{prefix}.pwconv1"),
                channels,
                4 * channels,
            )?,
            pwconv2: NvcPointwise::load(
                file,
                &format!("{prefix}.pwconv2"),
                4 * channels,
                channels,
            )?,
            channels,
        })
    }

    fn forward(&self, x: &mut [f32], frames: usize) {
        let residual = x.to_vec();
        let padded = ops::pad_left(x, self.channels, self.dwconv.kernel - 1);
        let mut h = self.dwconv.forward(&padded);
        self.norm.forward(&mut h, self.channels, frames);
        let mut h = self.pwconv1.forward(&h, frames);
        ops::gelu_erf(&mut h);
        let h = self.pwconv2.forward(&h, frames);
        for ((v, &hv), &rv) in x.iter_mut().zip(h.iter()).zip(residual.iter()) {
            *v = rv + hv;
        }
    }
}

struct EncoderStage {
    blocks: Vec<ConvNeXtBlock>,
    downsample: Conv1d,
}

struct Encoder {
    conv_in: NvcPointwise,
    stages: Vec<EncoderStage>,
}

struct DecoderStage {
    upsample: ConvTranspose1d,
    blocks: Vec<ConvNeXtBlock>,
}

struct Decoder {
    stages: Vec<DecoderStage>,
    conv_out: NvcPointwise,
}

/// Probabilistic RVQ: per-quantizer Gaussian means, Euclidean argmin,
/// residual subtraction on encode, sum on decode.
struct Prvq {
    /// `[num_quantizers][codebook_size * latent_dim]`.
    mus: Vec<Vec<f32>>,
    size: usize,
    dim: usize,
}

impl Prvq {
    fn load(
        file: &SafetensorsFile,
        num_quantizers: usize,
        size: usize,
        dim: usize,
    ) -> Result<Self> {
        let mut mus = Vec::with_capacity(num_quantizers);
        for q in 0..num_quantizers {
            mus.push(load_f32_shaped(
                file,
                &format!("prvq.mus_list.{q}"),
                &[size, dim],
            )?);
        }
        Ok(Prvq { mus, size, dim })
    }

    /// `(B, T, D)` latents, batch-of-one -> `(Q, T)` codes.
    fn encode(&self, latents: &[f32], frames: usize) -> Vec<Vec<i32>> {
        let dim = self.dim;
        let mut residual = latents.to_vec();
        let mut codes = Vec::with_capacity(self.mus.len());
        for means in &self.mus {
            // Precompute squared norms of the means.
            let mut idx = vec![0i32; frames];
            for t in 0..frames {
                let row = &residual[t * dim..(t + 1) * dim];
                let r2: f32 = row.iter().map(|v| v * v).sum();
                let mut best = f32::INFINITY;
                let mut best_idx = 0usize;
                for c in 0..self.size {
                    let m = &means[c * dim..(c + 1) * dim];
                    let m2: f32 = m.iter().map(|v| v * v).sum();
                    let mut dot = 0.0f32;
                    for (a, &b) in row.iter().zip(m) {
                        dot += a * b;
                    }
                    let dist = r2 - 2.0 * dot + m2;
                    if dist < best {
                        best = dist;
                        best_idx = c;
                    }
                }
                idx[t] = best_idx as i32;
            }
            for t in 0..frames {
                let c = idx[t] as usize;
                for (d, v) in residual[t * dim..(t + 1) * dim].iter_mut().enumerate() {
                    *v -= means[c * dim + d];
                }
            }
            codes.push(idx);
        }
        codes
    }

    /// `(Q, T)` codes -> `(T, D)` latents.
    fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.len() > self.mus.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "received {} quantizers, maximum is {}",
                    codes.len(),
                    self.mus.len()
                ),
            });
        }
        let dim = self.dim;
        let mut latents = vec![0.0f32; frames * dim];
        for (q, book) in codes.iter().enumerate() {
            let means = &self.mus[q];
            for (t, &c) in book.iter().enumerate() {
                let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                    why: format!("negative code {c}"),
                })?;
                if idx >= self.size {
                    return Err(SpeechError::Input {
                        why: format!("code {c} out of range for {} entries", self.size),
                    });
                }
                for (d, v) in latents[t * dim..(t + 1) * dim].iter_mut().enumerate() {
                    *v += means[idx * dim + d];
                }
            }
        }
        Ok(latents)
    }
}

/// Loaded Nemotron VoiceChat codec, batch-of-one.
pub struct NemotronVoiceChatCodec {
    pub config: NemotronVoiceChatConfig,
    encoder: Encoder,
    decoder: Decoder,
    prvq: Prvq,
    fft: RealFftPlan,
    window: Vec<f32>,
}

impl NemotronVoiceChatCodec {
    /// Opens a checkpoint directory with `config.json` plus a
    /// safetensors file.
    pub fn open(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = NemotronVoiceChatConfig::from_json(&value)?;
        let file =
            SafetensorsFile::open(&dir.join("model.safetensors")).map_err(SpeechError::from)?;
        Self::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: NemotronVoiceChatConfig, file: &SafetensorsFile) -> Result<Self> {
        if config.channel_multipliers.len() != config.downsample_rates.len() {
            return Err(SpeechError::BadConfig {
                field: "channel_multipliers".to_string(),
                why: "channel_multipliers and downsample_rates must have equal lengths".to_string(),
            });
        }
        let channels = config.stage_channels();
        let kernel = config.block_kernel_size;

        // Encoder: 1x1 stem, per stage `blocks_per_stage` ConvNeXt
        // blocks then a strided downsample conv.
        let conv_in = NvcPointwise::load(
            file,
            "encoder.layers.0",
            config.stft_channels(),
            channels[0],
        )?;
        let mut encoder_stages = Vec::with_capacity(channels.len());
        let mut layer = 1usize;
        for (index, &stage_channels) in channels.iter().enumerate() {
            let mut blocks = Vec::with_capacity(config.blocks_per_stage);
            for _ in 0..config.blocks_per_stage {
                blocks.push(ConvNeXtBlock::load(
                    file,
                    &format!("encoder.layers.{layer}"),
                    stage_channels,
                    kernel,
                )?);
                layer += 1;
            }
            let next = if index + 1 < channels.len() {
                channels[index + 1]
            } else {
                config.latent_dim
            };
            let downsample = load_nvc_conv(
                file,
                &format!("encoder.layers.{layer}"),
                stage_channels,
                next,
                config.downsample_rates[index],
                config.downsample_rates[index],
                false,
            )?;
            layer += 1;
            encoder_stages.push(EncoderStage { blocks, downsample });
        }

        // Decoder: per stage (reversed) a transposed upsample conv then
        // blocks, and a 1x1 output conv.
        let mut decoder_stages = Vec::with_capacity(channels.len());
        let reversed: Vec<(usize, usize)> = channels
            .iter()
            .enumerate()
            .rev()
            .map(|(i, &c)| (i, c))
            .collect();
        let mut source = config.latent_dim;
        let mut layer = 0usize;
        for (stage_index, stage_channels) in reversed {
            let upsample = load_nvc_convtr(
                file,
                &format!("decoder.layers.{layer}"),
                source,
                stage_channels,
                config.downsample_rates[stage_index],
            )?;
            layer += 1;
            let mut blocks = Vec::with_capacity(config.blocks_per_stage);
            for _ in 0..config.blocks_per_stage {
                blocks.push(ConvNeXtBlock::load(
                    file,
                    &format!("decoder.layers.{layer}"),
                    stage_channels,
                    kernel,
                )?);
                layer += 1;
            }
            source = stage_channels;
            decoder_stages.push(DecoderStage { upsample, blocks });
        }
        let conv_out = NvcPointwise::load(
            file,
            &format!("decoder.layers.{layer}"),
            channels[0],
            config.stft_channels(),
        )?;

        let prvq = Prvq::load(
            file,
            config.num_quantizers,
            config.codebook_size,
            config.latent_dim,
        )?;
        // The reference keeps per-quantizer variances in the tree;
        // strict loading refuses a checkpoint missing them.
        for q in 0..config.num_quantizers {
            load_f32_shaped(file, &format!("prvq.variance_list.{q}.variance"), &[])?;
        }
        let fft = RealFftPlan::new(config.n_fft)?;
        let window = dsp::hann_window(config.n_fft);
        Ok(NemotronVoiceChatCodec {
            config,
            encoder: Encoder {
                conv_in,
                stages: encoder_stages,
            },
            decoder: Decoder {
                stages: decoder_stages,
                conv_out,
            },
            prvq,
            fft,
            window,
        })
    }

    /// Latents `(latent_dim, T)` from mono samples.
    pub fn encode_latents(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let features = self.spectrogram(samples)?;
        let frames = features.len() / self.config.stft_channels();
        let mut x = self.encoder.conv_in.forward(&features, frames);
        for stage in &self.encoder.stages {
            let channels = stage.blocks[0].channels;
            let frames = x.len() / channels;
            for block in &stage.blocks {
                block.forward(&mut x, frames);
            }
            x = stage.downsample.forward(&x);
        }
        // The reference works channels-last: hand the quantizer rows.
        let channels = self.config.latent_dim;
        let frames = x.len() / channels;
        let mut rows = vec![0.0f32; x.len()];
        for t in 0..frames {
            for d in 0..channels {
                rows[t * channels + d] = x[d * frames + t];
            }
        }
        Ok(rows)
    }

    /// Mono samples -> codes `(Q, T)`.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
        let latents = self.encode_latents(samples)?;
        let frames = latents.len() / self.config.latent_dim;
        Ok(self.prvq.encode(&latents, frames))
    }

    /// Codes `(Q, T)` -> mono samples.
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let frames = codes.first().map_or(0, |book| book.len());
        let latents = self.prvq.decode(codes, frames)?;
        // Latents are [T, D] rows; transpose to channel-major.
        let dim = self.config.latent_dim;
        let mut x = vec![0.0f32; latents.len()];
        for t in 0..frames {
            for d in 0..dim {
                x[d * frames + t] = latents[t * dim + d];
            }
        }
        for stage in &self.decoder.stages {
            x = stage.upsample.forward(&x);
            let frames = x.len() / stage.upsample.out_ch;
            for block in &stage.blocks {
                block.forward(&mut x, frames);
            }
        }
        let frames = x.len() / self.decoder.conv_out.in_ch;
        let features = self.decoder.conv_out.forward(&x, frames);
        self.spectrogram_to_wave(&features)
    }

    /// The reference `_spectrogram`: periodic Hann, zero pad
    /// `(n_fft - hop) / 2` on both sides, center-free framing,
    /// `[2 * bins, T]` (real rows then imag rows).
    fn spectrogram(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "waveform must contain at least one sample".to_string(),
            });
        }
        let pad_left = (self.config.n_fft - self.config.hop_length) / 2;
        let pad_right = self.config.n_fft - self.config.hop_length - pad_left;
        let mut padded = vec![0.0f32; samples.len() + pad_left + pad_right];
        padded[pad_left..pad_left + samples.len()].copy_from_slice(samples);
        let bins = self.config.n_fft / 2 + 1;
        let padded_len = padded.len();
        let frames = 1 + (padded_len - self.config.n_fft) / self.config.hop_length;
        // Channel-major `[2 * bins, T]`: real rows then imag rows,
        // matching the reference's `concat([real, imag], axis=1)`.
        let mut out = vec![0.0f32; 2 * bins * frames];
        for (f, start) in (0..padded_len)
            .step_by(self.config.hop_length)
            .take_while(|&s| s + self.config.n_fft <= padded_len)
            .enumerate()
        {
            let mut frame = vec![0.0f32; self.config.n_fft];
            for (t, slot) in frame.iter_mut().enumerate() {
                *slot = padded[start + t] * self.window[t];
            }
            let spectrum = self.fft.forward(&frame)?;
            for (b, c) in spectrum.iter().enumerate() {
                out[b * frames + f] = c.re;
                out[(bins + b) * frames + f] = c.im;
            }
        }
        Ok(out)
    }

    /// The reference decode head: softplus-bounded magnitude, phase,
    /// Hermitian fix-ups, envelope-clamped iSTFT, and the pad crop.
    fn spectrogram_to_wave(&self, features: &[f32]) -> Result<Vec<f32>> {
        let bins = self.config.n_fft / 2 + 1;
        let channels = self.config.stft_channels();
        let frames = features.len() / channels;
        let max_magnitude = 100.0f32;
        let log_max = max_magnitude.ln();
        let mut ola = vec![0.0f32; (frames - 1) * self.config.hop_length + self.config.n_fft];
        let mut norm = vec![0.0f32; ola.len()];
        // Norm buffer: window^2 overlap-add, floored at 1e-10.
        for f in 0..frames {
            let start = f * self.config.hop_length;
            for (t, &w) in self.window.iter().enumerate() {
                norm[start + t] += w * w;
            }
        }
        let mut spectrum = vec![ComplexF32::new(0.0, 0.0); bins];
        for f in 0..frames {
            for b in 0..bins {
                let logit = features[b * frames + f];
                let phase = features[(bins + b) * frames + f];
                // magnitude = max * exp(-softplus(-logit + log(max)))
                let z = -logit + log_max;
                let softplus = if z > 0.0 {
                    z + (-z).exp().ln_1p()
                } else {
                    z.exp().ln_1p()
                };
                let magnitude = max_magnitude * (-softplus).exp();
                let mut im = magnitude * phase.sin();
                // DC and Nyquist bins are real-valued.
                if b == 0 || b == bins - 1 {
                    im = 0.0;
                }
                spectrum[b] = ComplexF32::new(magnitude * phase.cos(), im);
            }
            let mut frame = self.fft.inverse(&spectrum)?;
            // constrain_value_range: clamp to the window envelope.
            for (v, &w) in frame.iter_mut().zip(&self.window) {
                *v = v.clamp(-w, w) * w;
            }
            let start = f * self.config.hop_length;
            for (t, v) in frame.iter().enumerate() {
                ola[start + t] += v;
            }
        }
        let pad_left = (self.config.n_fft - self.config.hop_length) / 2;
        let pad_right = self.config.n_fft - self.config.hop_length - pad_left;
        let mut out = Vec::with_capacity(ola.len() - pad_left - pad_right);
        for (t, &v) in ola.iter().enumerate().skip(pad_left) {
            if t >= ola.len() - pad_right {
                break;
            }
            out.push(v / norm[t].max(1e-10));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
