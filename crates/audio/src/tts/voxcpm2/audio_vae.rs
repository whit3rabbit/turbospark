//! The VoxCPM2 AudioVAE: the voxcpm causal-conv stack plus per-block
//! sample-rate conditioning.
//!
//! Port of `mlx_audio/tts/models/voxcpm2/audio_vae.py` at the pinned
//! mlx-audio commit. The encoder matches the voxcpm `CausalEncoder`
//! (reused from the sibling); the decoder inserts a
//! `SampleRateConditionLayer` before every `CausalDecoderBlock`
//! (`scale_bias` conditioning: `x * scale_embed[sr] +
//! bias_embed[sr]`, the bucket of the output sample rate under
//! `sr_bin_boundaries`). The conv loaders, weight layouts, and the
//! causal padding conventions are the voxcpm ones; the noise block
//! variant is refused by the family (`use_noise_block` default
//! false). Encoding runs at `sample_rate` (16 kHz) with hop
//! `prod(encoder_rates)`; decoding runs at `out_sample_rate`
//! (48 kHz).

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::fft::{ComplexF32, RealFftPlan};
use crate::{Result, SpeechError};

use super::super::voxcpm::audio_vae::{
    self as vae, CausalConv, CausalConvTranspose, CausalEncoder, ResidualUnit,
};

/// The `AudioVAEConfig` dataclass (V2 fields).
#[derive(Debug, Clone)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub depthwise: bool,
    pub sample_rate: u32,
    pub out_sample_rate: u32,
    pub use_noise_block: bool,
    pub sr_bin_boundaries: Option<Vec<u32>>,
    pub cond_type: String,
    pub cond_dim: usize,
    pub cond_out_layer: bool,
}

impl AudioVaeConfig {
    pub fn from_json(value: &Value) -> Self {
        let rates = |key: &str, default: &[usize]| -> Vec<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_else(|| default.to_vec())
        };
        let u32_rates = |key: &str| -> Option<Vec<u32>> {
            match value.get(key) {
                Some(Value::Array(list)) if !list.is_empty() => Some(
                    list.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect(),
                ),
                _ => None,
            }
        };
        Self {
            encoder_dim: value
                .get("encoder_dim")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(128),
            encoder_rates: rates("encoder_rates", &[2, 5, 8, 8]),
            latent_dim: value
                .get("latent_dim")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(64),
            decoder_dim: value
                .get("decoder_dim")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(2048),
            decoder_rates: rates("decoder_rates", &[8, 6, 5, 2, 2, 2]),
            depthwise: value
                .get("depthwise")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            sample_rate: value
                .get("sample_rate")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(16000),
            out_sample_rate: value
                .get("out_sample_rate")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(48000),
            use_noise_block: value
                .get("use_noise_block")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            sr_bin_boundaries: u32_rates("sr_bin_boundaries"),
            cond_type: value
                .get("cond_type")
                .and_then(Value::as_str)
                .unwrap_or("scale_bias")
                .to_owned(),
            cond_dim: value
                .get("cond_dim")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(128),
            cond_out_layer: value
                .get("cond_out_layer")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }

    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    /// `chunk_size`: the encode patch alignment.
    pub fn chunk_size(&self) -> usize {
        self.hop_length()
    }

    /// `decode_chunk_size`: samples per latent patch at the output
    /// rate.
    pub fn decode_chunk_size(&self) -> usize {
        self.decoder_rates.iter().product()
    }
}

/// The `scipy.signal.resample` Fourier resample the reference applies
/// to array inputs before encoding. With `s_fac = len / num` and
/// `m = min(num, len)`: truncate the rfft to `m/2 + 1` bins, double
/// the new-length Nyquist bin when `m` is even and `num != len` (the
/// unpaired-bin correction), inverse-transform at the new length, and
/// divide by `s_fac`.
pub(crate) fn fourier_resample(x: &[f32], num: usize) -> Result<Vec<f32>> {
    let plan = RealFftPlan::new(x.len()).map_err(|e| SpeechError::Audio(e.to_string()))?;
    let spectrum = plan
        .forward(x)
        .map_err(|e| SpeechError::Audio(e.to_string()))?;
    let m = num.min(x.len());
    let kept_bins = m / 2 + 1;
    let mut kept = spectrum[..kept_bins.min(spectrum.len())].to_vec();
    if m % 2 == 0 && num != x.len() {
        let nyquist = m / 2;
        if let Some(bin) = kept.get_mut(nyquist) {
            // The unpaired-bin correction doubles the REAL part; the
            // imaginary part is unobservable to numpy's irfft but the
            // shared real-IFFT plan reads it, so it is zeroed.
            *bin = ComplexF32::new(bin.re * 2.0, 0.0);
        }
    }
    let out_plan = RealFftPlan::new(num).map_err(|e| SpeechError::Audio(e.to_string()))?;
    let mut out = out_plan
        .inverse(&kept)
        .map_err(|e| SpeechError::Audio(e.to_string()))?;
    let s_fac = x.len() as f32 / num as f32;
    for value in &mut out {
        *value /= s_fac;
    }
    Ok(out)
}

/// `SampleRateConditionLayer` (`scale_bias` only; the `add`, `concat`,
/// and out-layer variants wait for a checkpoint that configures them).
pub struct SampleRateConditionLayer {
    scale_embed: Vec<f32>,
    bias_embed: Vec<f32>,
    dim: usize,
}

impl SampleRateConditionLayer {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        buckets: usize,
        dim: usize,
        config: &AudioVaeConfig,
    ) -> Result<Self> {
        if config.cond_type != "scale_bias" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "audio_vae cond_type {:?}; the port implements the reference scale_bias path",
                    config.cond_type
                ),
            });
        }
        if config.cond_out_layer {
            return Err(SpeechError::Unsupported {
                why: "audio_vae cond_out_layer is enabled; no checkpoint at this commit uses it"
                    .into(),
            });
        }
        let table = |name: &str| -> Result<Vec<f32>> {
            let (file, shape) = vae::locate(files, name).ok_or_else(|| vae::missing(name))?;
            if shape != vec![buckets, dim] {
                return Err(SpeechError::Tensor {
                    name: name.to_owned(),
                    why: format!("expected [{buckets}, {dim}], got {shape:?}"),
                });
            }
            file.load_as_f32(name).map_err(SpeechError::from)
        };
        Ok(Self {
            scale_embed: table(&format!("{prefix}.scale_embed.weight"))?,
            bias_embed: table(&format!("{prefix}.bias_embed.weight"))?,
            dim,
        })
    }

    /// `x * scale[sr] + bias[sr]` over channel-major `[ch, seq]`.
    fn apply(&self, x: &mut [f32], sr: usize, frames: usize) {
        let scale = &self.scale_embed[sr * self.dim..(sr + 1) * self.dim];
        let bias = &self.bias_embed[sr * self.dim..(sr + 1) * self.dim];
        for ch in 0..self.dim {
            for frame in 0..frames {
                let index = ch * frames + frame;
                x[index] = x[index] * scale[ch] + bias[ch];
            }
        }
    }
}

/// One V2 decoder block: the conditioning layer, snake, transposed
/// conv, and the residual units.
struct CondDecoderBlock {
    cond: Option<SampleRateConditionLayer>,
    snake_alpha: Vec<f32>,
    conv_t: CausalConvTranspose,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    input_dim: usize,
}

impl CondDecoderBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        cond_prefix: &str,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
        buckets: Option<usize>,
        config: &AudioVaeConfig,
    ) -> Result<Self> {
        if files.iter().any(|file| {
            file.descriptor(&format!("{prefix}.noise.linear.weight"))
                .is_some()
        }) {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "{prefix} carries a noise block; use_noise_block checkpoints have no deterministic Rust decode at this commit"
                ),
            });
        }
        let cond = match buckets {
            Some(buckets) => Some(SampleRateConditionLayer::load(
                files,
                cond_prefix,
                buckets,
                input_dim,
                config,
            )?),
            None => None,
        };
        let conv_t = vae::load_conv_transpose(
            files,
            &format!("{prefix}.conv_t"),
            input_dim,
            output_dim,
            2 * stride,
            stride,
        )?;
        Ok(Self {
            cond,
            snake_alpha: vae::alpha(files, &format!("{prefix}.snake"), input_dim)?,
            conv_t: CausalConvTranspose {
                conv: conv_t,
                trim: 2 * stride.div_ceil(2) - stride % 2,
            },
            res1: ResidualUnit::load(files, &format!("{prefix}.res1"), output_dim, 1, groups)?,
            res2: ResidualUnit::load(files, &format!("{prefix}.res2"), output_dim, 3, groups)?,
            res3: ResidualUnit::load(files, &format!("{prefix}.res3"), output_dim, 9, groups)?,
            input_dim,
        })
    }

    fn forward(&self, x: &mut [f32], sr: usize) -> Vec<f32> {
        let dim = self.input_dim;
        let frames = x.len() / dim;
        if let Some(cond) = &self.cond {
            cond.apply(x, sr, frames);
        }
        vae::snake(x, &self.snake_alpha, dim, frames);
        let mut h = self.conv_t.forward(x);
        self.res1.forward(&mut h);
        self.res2.forward(&mut h);
        self.res3.forward(&mut h);
        h
    }
}

/// The V2 `AudioVAE`.
pub struct AudioVae {
    encoder: CausalEncoder,
    conv_in_dense: CausalConv,
    conv_in_point: CausalConv,
    blocks: Vec<CondDecoderBlock>,
    snake_out_alpha: Vec<f32>,
    conv_out: CausalConv,
    sr_boundaries: Option<Vec<u32>>,
    config: AudioVaeConfig,
}

impl AudioVae {
    /// Loads from the audio_vae weights of the opened checkpoint.
    pub fn load(files: &[SafetensorsFile], prefix: &str, config: &AudioVaeConfig) -> Result<Self> {
        // The encoder matches the voxcpm CausalEncoder geometry.
        let encoder = CausalEncoder::load(
            files,
            &format!("{prefix}encoder"),
            config.encoder_dim,
            config.latent_dim,
            &config.encoder_rates,
        )?;
        let channels = config.decoder_dim;
        let latent = config.latent_dim;
        let conv_in_dense = CausalConv {
            conv: vae::load_conv(
                files,
                &format!("{prefix}decoder.conv_in.layers.0"),
                latent,
                latent,
                7,
                1,
                1,
                latent,
            )?,
            pad: 6,
        };
        let conv_in_point = CausalConv {
            conv: vae::load_conv(
                files,
                &format!("{prefix}decoder.conv_in.layers.1"),
                latent,
                channels,
                1,
                1,
                1,
                1,
            )?,
            pad: 0,
        };
        let buckets = config
            .sr_bin_boundaries
            .as_ref()
            .map(|boundaries| boundaries.len() + 1);
        let mut blocks = Vec::new();
        for (index, stride) in config.decoder_rates.iter().enumerate() {
            let input_dim = channels / (1 << index);
            let output_dim = channels / (1 << (index + 1));
            blocks.push(CondDecoderBlock::load(
                files,
                &format!("{prefix}decoder.blocks.layers.{index}"),
                &format!("{prefix}decoder.sr_cond_layers.{index}"),
                input_dim,
                output_dim,
                *stride,
                output_dim,
                buckets,
                config,
            )?);
        }
        let final_dim = channels / (1 << config.decoder_rates.len());
        Ok(Self {
            encoder,
            conv_in_dense,
            conv_in_point,
            blocks,
            snake_out_alpha: vae::alpha(files, &format!("{prefix}decoder.snake_out"), final_dim)?,
            conv_out: CausalConv {
                conv: vae::load_conv(
                    files,
                    &format!("{prefix}decoder.conv_out"),
                    final_dim,
                    1,
                    7,
                    1,
                    1,
                    1,
                )?,
                pad: 6,
            },
            sr_boundaries: config.sr_bin_boundaries.clone(),
            config: config.clone(),
        })
    }

    pub fn out_sample_rate(&self) -> u32 {
        self.config.out_sample_rate
    }

    pub fn chunk_size(&self) -> usize {
        self.config.chunk_size()
    }

    pub fn decode_chunk_size(&self) -> usize {
        self.config.decode_chunk_size()
    }

    /// `get_sr_idx`: the count of boundaries the sample rate reaches.
    fn sr_index(&self, sample_rate: u32) -> usize {
        match &self.sr_boundaries {
            None => 0,
            Some(boundaries) => boundaries.iter().filter(|&&b| sample_rate >= b).count(),
        }
    }

    /// `AudioVAE.encode` for one waveform: right-pad to a hop multiple
    /// and encode. Returns `[latent_dim, frames]`.
    pub fn encode(&self, waveform: &[f32]) -> Vec<f32> {
        let hop = self.config.hop_length();
        let right_pad = hop * waveform.len().div_ceil(hop) - waveform.len();
        let mut input = waveform.to_vec();
        input.extend(std::iter::repeat_n(0.0f32, right_pad));
        self.encoder.forward(&input)
    }

    /// `AudioVAE.decode` at the output sample rate: `[latent, frames]`
    /// latents to the tanh waveform.
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        let sr = self.sr_index(self.config.out_sample_rate);
        let mut h = self.conv_in_dense.forward(z);
        h = self.conv_in_point.forward(&h);
        for block in &self.blocks {
            h = block.forward(&mut h, sr);
        }
        let in_ch = self.conv_out.conv.in_ch;
        let frames = h.len() / in_ch;
        vae::snake(&mut h, &self.snake_out_alpha, in_ch, frames);
        let mut out = self.conv_out.forward(&h);
        for value in &mut out {
            *value = value.tanh();
        }
        out
    }
}
