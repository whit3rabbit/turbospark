//! The soprano decoder: hidden-state upscale, ConvNeXt backbone, and
//! iSTFT head.
//!
//! Port of `mlx_audio/tts/models/soprano/decoder.py` (SopranoDecoder,
//! ISTFTHead) over `mlx_audio/codec/models/vocos/vocos.py` (VocosBackbone)
//! and `mlx_audio/dsp.py` (`hanning`, `istft`) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/soprano/decoder.py).
//! The backbone math is the same ConvNeXt + iSTFT pair the
//! [`crate::codec::vocos`] mel vocoder ports; it is re-implemented here
//! because the soprano checkpoint names, input width (the LM hidden
//! size), and the pre-backbone linear upsample are family specific, and
//! the codec module keeps its backbone private to its own loaders.
//!
//! Checkpoint layout (the reference `Model.sanitize` output for an
//! HF-style save): `decoder.decoder.embed` / `decoder.decoder.norm` /
//! `decoder.decoder.convnext.{i}.*` / `decoder.decoder.final_layer_norm`
//! under the VocosBackbone, and `decoder.head.out` under the ISTFTHead,
//! all stored float32 (the reference casts them). Conv weights stay in
//! the PyTorch `[out, in, K]` layout the mlx `nn.Conv1d` loader reads.

use crate::fft::{ComplexF32, ComplexFftPlan};
use crate::lm::Weights;
use crate::ops;
use crate::{Result, SpeechError};

/// The `DecoderConfig` dataclass fields and defaults.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub decoder_num_layers: usize,
    pub decoder_dim: usize,
    pub decoder_intermediate_dim: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub upscale: usize,
    pub input_kernel: usize,
    pub dw_kernel: usize,
    /// Samples per audio token (`upscale * hop_length` in every shipped
    /// configuration; kept as its own field like the reference).
    pub token_size: usize,
    pub receptive_field: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            decoder_num_layers: 8,
            decoder_dim: 768,
            decoder_intermediate_dim: 2304,
            hop_length: 512,
            n_fft: 2048,
            upscale: 4,
            input_kernel: 1,
            dw_kernel: 3,
            token_size: 2048,
            receptive_field: 4,
        }
    }
}

impl DecoderConfig {
    /// Parses the nested `decoder_config` mapping with the dataclass
    /// defaults; unknown keys are ignored.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let mut config = Self::default();
        let pick_usize = |field: &str, slot: &mut usize| {
            if let Some(v) = value.get(field).and_then(serde_json::Value::as_u64) {
                *slot = v as usize;
            }
        };
        pick_usize("decoder_num_layers", &mut config.decoder_num_layers);
        pick_usize("decoder_dim", &mut config.decoder_dim);
        pick_usize(
            "decoder_intermediate_dim",
            &mut config.decoder_intermediate_dim,
        );
        pick_usize("hop_length", &mut config.hop_length);
        pick_usize("n_fft", &mut config.n_fft);
        pick_usize("upscale", &mut config.upscale);
        pick_usize("input_kernel", &mut config.input_kernel);
        pick_usize("dw_kernel", &mut config.dw_kernel);
        pick_usize("token_size", &mut config.token_size);
        pick_usize("receptive_field", &mut config.receptive_field);
        Ok(config)
    }
}

struct ConvNeXtBlock {
    dwconv_weight: Vec<f32>,
    dwconv_bias: Option<Vec<f32>>,
    dim: usize,
    dw_kernel: usize,
    norm_weight: Vec<f32>,
    norm_bias: Option<Vec<f32>>,
    pwconv1: crate::nn::Linear,
    pwconv2: crate::nn::Linear,
    gamma: Option<Vec<f32>>,
}

impl ConvNeXtBlock {
    /// `x` is channel-major `[dim, frames]`; residual add in place.
    fn forward(&self, x: &mut [f32], frames: usize) {
        let dim = self.dim;
        let mut h = ops::conv1d(
            x,
            &self.dwconv_weight,
            self.dwconv_bias.as_deref(),
            dim,
            dim,
            self.dw_kernel,
            1,
            self.dw_kernel / 2,
            1,
            dim,
        );
        ops::layernorm_cm(
            &mut h,
            dim,
            frames,
            Some(&self.norm_weight),
            self.norm_bias.as_deref(),
            1e-6,
        );
        // pwconv1 + exact-erf GELU + pwconv2, per frame.
        let mut fed = vec![0.0f32; frames * dim];
        for frame in 0..frames {
            for channel in 0..dim {
                fed[frame * dim + channel] = h[channel * frames + frame];
            }
        }
        let hidden = self.pwconv1.forward(&fed, frames);
        let mut activated = hidden;
        for value in &mut activated {
            ops::gelu_erf(std::slice::from_mut(value));
        }
        let projected = self.pwconv2.forward(&activated, frames);
        for frame in 0..frames {
            for channel in 0..dim {
                let mut value = projected[frame * dim + channel];
                if let Some(gamma) = &self.gamma {
                    value *= gamma[channel];
                }
                x[channel * frames + frame] += value;
            }
        }
    }
}

/// The loaded decoder, batch-of-one.
pub struct SopranoDecoder {
    embed_weight: Vec<f32>,
    embed_bias: Option<Vec<f32>>,
    input_channels: usize,
    dim: usize,
    input_kernel: usize,
    norm_weight: Vec<f32>,
    norm_bias: Option<Vec<f32>>,
    blocks: Vec<ConvNeXtBlock>,
    final_norm_weight: Vec<f32>,
    final_norm_bias: Option<Vec<f32>>,
    head: crate::nn::Linear,
    n_fft: usize,
    hop_length: usize,
    upscale: usize,
}

/// Loads a plain tensor, treating "missing" as `None` for the optional
/// biases (the reference constructs every block with `bias=True`, so a
/// present-tensor check keeps the loader tolerant without guessing).
fn optional_tensor(weights: &Weights, name: &str, shape: &[usize]) -> Result<Option<Vec<f32>>> {
    match weights.raw_tensor(name, shape) {
        Ok(values) => Ok(Some(values)),
        Err(error) if error.to_string().contains("tensor is missing") => Ok(None),
        Err(error) => Err(error),
    }
}

fn required_tensor(weights: &Weights, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    weights.raw_tensor(name, shape)
}

/// MLX `nn.Conv1d` stores its weight `[out, kernel, in]` (channels
/// last); `ops::conv1d` wants `[out, in, kernel]`, so the family loader
/// owns the transpose (the same checkpoint-layout rule the vocos codec
/// loader documents in reverse).
fn transposed_conv_weight(
    mut weight: Vec<f32>,
    out: usize,
    in_ch: usize,
    kernel: usize,
) -> Vec<f32> {
    let source = weight.clone();
    weight.resize(out * in_ch * kernel, 0.0);
    for o in 0..out {
        for i in 0..in_ch {
            for k in 0..kernel {
                weight[(o * in_ch + i) * kernel + k] = source[(o * kernel + k) * in_ch + i];
            }
        }
    }
    weight
}

impl SopranoDecoder {
    /// Loads the decoder tensors from a prefix-free weights handle (the
    /// `decoder.*` names sit beside the `model.*` backbone tensors).
    pub fn load(weights: &Weights, config: &DecoderConfig, input_channels: usize) -> Result<Self> {
        let layers = config.decoder_num_layers;
        let dim = config.decoder_dim;
        let intermediate = if config.decoder_intermediate_dim > 0 {
            config.decoder_intermediate_dim
        } else {
            dim * 3
        };
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            let prefix = format!("decoder.decoder.convnext.{index}");
            let dw_weight = transposed_conv_weight(
                required_tensor(
                    weights,
                    &format!("{prefix}.dwconv.weight"),
                    &[dim, config.dw_kernel, 1],
                )?,
                dim,
                1,
                config.dw_kernel,
            );
            let dw_bias = optional_tensor(weights, &format!("{prefix}.dwconv.bias"), &[dim])?;
            let norm_weight = required_tensor(weights, &format!("{prefix}.norm.weight"), &[dim])?;
            let norm_bias = optional_tensor(weights, &format!("{prefix}.norm.bias"), &[dim])?;
            let pwconv1 = weights.linear_absolute(
                &format!("{prefix}.pwconv1"),
                dim,
                intermediate,
                crate::lm::Bias::Optional,
            )?;
            let pwconv2 = weights.linear_absolute(
                &format!("{prefix}.pwconv2"),
                intermediate,
                dim,
                crate::lm::Bias::Optional,
            )?;
            let gamma = match weights.raw_tensor(&format!("{prefix}.gamma"), &[dim]) {
                Ok(values) => Some(values),
                Err(error) if error.to_string().contains("tensor is missing") => None,
                Err(error) => return Err(error),
            };
            blocks.push(ConvNeXtBlock {
                dwconv_weight: dw_weight,
                dwconv_bias: dw_bias,
                dim,
                dw_kernel: config.dw_kernel,
                norm_weight,
                norm_bias,
                pwconv1,
                pwconv2,
                gamma,
            });
        }
        Ok(Self {
            embed_weight: transposed_conv_weight(
                required_tensor(
                    weights,
                    "decoder.decoder.embed.weight",
                    &[dim, config.input_kernel, input_channels],
                )?,
                dim,
                input_channels,
                config.input_kernel,
            ),
            embed_bias: optional_tensor(weights, "decoder.decoder.embed.bias", &[dim])?,
            input_channels,
            dim,
            input_kernel: config.input_kernel,
            norm_weight: required_tensor(weights, "decoder.decoder.norm.weight", &[dim])?,
            norm_bias: optional_tensor(weights, "decoder.decoder.norm.bias", &[dim])?,
            blocks,
            final_norm_weight: required_tensor(
                weights,
                "decoder.decoder.final_layer_norm.weight",
                &[dim],
            )?,
            final_norm_bias: optional_tensor(
                weights,
                "decoder.decoder.final_layer_norm.bias",
                &[dim],
            )?,
            head: weights.linear_absolute(
                "decoder.head.out",
                dim,
                config.n_fft + 2,
                crate::lm::Bias::Optional,
            )?,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            upscale: config.upscale,
        })
    }

    /// Reference `SopranoDecoder.__call__`: hidden states
    /// `[positions, hidden]` row-major, linear upsample to
    /// `upscale * (positions - 1) + 1` frames with aligned corners,
    /// backbone, and the iSTFT head. Returns mono samples.
    pub fn forward(&self, hidden: &[f32], positions: usize) -> Result<Vec<f32>> {
        if positions == 0 || hidden.len() != positions * self.input_channels {
            return Err(SpeechError::Input {
                why: format!(
                    "decoder input of {} values does not cover {positions} positions x {} channels",
                    hidden.len(),
                    self.input_channels
                ),
            });
        }
        // [positions, C] -> [C, positions].
        let mut x = vec![0.0f32; positions * self.input_channels];
        for frame in 0..positions {
            for channel in 0..self.input_channels {
                x[channel * positions + frame] = hidden[frame * self.input_channels + channel];
            }
        }
        // Linear upsample along frames, `align_corners=True`: source
        // coordinate `i * (positions - 1) / (target - 1)` (== i/upscale).
        let target = self.upscale * (positions - 1) + 1;
        let mut upsampled = vec![0.0f32; self.input_channels * target];
        for out_frame in 0..target {
            let source = out_frame as f64 * (positions as f64 - 1.0) / (target as f64 - 1.0);
            let low = (source as usize).min(positions - 1);
            let high = (low + 1).min(positions - 1);
            let frac = (source - low as f64) as f32;
            for channel in 0..self.input_channels {
                let a = x[channel * positions + low];
                let b = x[channel * positions + high];
                upsampled[channel * target + out_frame] = a * (1.0 - frac) + b * frac;
            }
        }
        let frames = target;

        // Backbone: embed conv, LayerNorm, ConvNeXt blocks, final norm.
        let mut h = ops::conv1d(
            &upsampled,
            &self.embed_weight,
            self.embed_bias.as_deref(),
            self.input_channels,
            self.dim,
            self.input_kernel,
            1,
            self.input_kernel / 2,
            1,
            1,
        );
        ops::layernorm_cm(
            &mut h,
            self.dim,
            frames,
            Some(&self.norm_weight),
            self.norm_bias.as_deref(),
            1e-6,
        );
        for block in &self.blocks {
            block.forward(&mut h, frames);
        }
        ops::layernorm_cm(
            &mut h,
            self.dim,
            frames,
            Some(&self.final_norm_weight),
            self.final_norm_bias.as_deref(),
            1e-6,
        );

        // ISTFTHead: linear to [n_fft + 2] per frame, exp/clamp magnitude,
        // complex spectrum, and the mlx `istft` overlap-add
        // (normalized=False: the window multiplies once and the divide is
        // by the summed window). The activations are channel-major
        // `[dim, frames]`; the head linear takes row-major frames.
        let n_fft = self.n_fft;
        let rows = n_fft + 2;
        let half = n_fft / 2 + 1;
        let mut head_in = vec![0.0f32; frames * self.dim];
        for frame in 0..frames {
            for channel in 0..self.dim {
                head_in[frame * self.dim + channel] = h[channel * frames + frame];
            }
        }
        let head_out = self.head.forward(&head_in, frames);
        let plan = ComplexFftPlan::new(n_fft)?;
        let window = hanning(n_fft);
        let total = (frames - 1) * self.hop_length + n_fft;
        let mut acc = vec![0.0f64; total];
        let mut window_sum = vec![0.0f64; total];
        let mut extended = vec![ComplexF32::default(); n_fft];
        for frame in 0..frames {
            let mut bins = Vec::with_capacity(half);
            for bin in 0..half {
                let mag = head_out[frame * rows + bin].exp().min(1e2);
                let phase = head_out[frame * rows + half + bin];
                bins.push(ComplexF32::new(mag * phase.cos(), mag * phase.sin()));
            }
            // Hermitian extension: the spectra are arbitrary complex, so
            // the inverse is the numpy irfft of the extended half.
            extended[..half].copy_from_slice(&bins);
            for k in half..n_fft {
                extended[k] = bins[n_fft - k].conj();
            }
            let frame_time = plan.inverse(&extended)?;
            let offset = frame * self.hop_length;
            for (index, sample) in frame_time.iter().enumerate() {
                acc[offset + index] += f64::from(sample.re) * f64::from(window[index]);
                window_sum[offset + index] += f64::from(window[index]);
            }
        }
        let start = n_fft / 2;
        let end = total - start;
        let mut out = Vec::with_capacity(end - start);
        for index in start..end {
            let sum = window_sum[index];
            out.push(if sum > 1e-10 {
                (acc[index] / sum) as f32
            } else {
                0.0
            });
        }
        Ok(out)
    }
}

/// `mlx_audio.dsp.hanning` (symmetric, `periodic=False`).
fn hanning(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|index| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * index as f64 / (size as f64 - 1.0)).cos()))
                as f32
        })
        .collect()
}
