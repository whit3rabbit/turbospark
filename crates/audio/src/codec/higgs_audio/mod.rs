//! Higgs Audio acoustic tokenizer (HiggsAudioV2).
//!
//! Reference: `mlx_audio/codec/models/higgs_audio/` (higgs_audio.py,
//! dac.py, config.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/higgs_audio).
//! A DAC-shaped plain-conv (norm "none" everywhere) acoustic encoder
//! and decoder with Snake activations, an 8-book residual vector
//! quantizer with per-book `project_in` / codebook / `project_out`,
//! and the `fc2` bridge from the 1024-dim quantizer space to the
//! 256-dim decoder input.
//!
//! Scope: the codec surface -- token decode (`tokens -> waveform`),
//! acoustic encoding (`waveform -> 256-dim features`), and RVQ
//! encode/decode over embeddings. The full `encode` path additionally
//! fuses a wav2vec2 (HuBERT) semantic tower through `encoder_semantic`
//! and `fc`; that tower is an STT model outside the codec crate and is
//! not ported here (the reference's `encode` hard-requires it; see the
//! family README). All convolutions run without weight normalization
//! (`norm="none"`), unlike SNAC/Descript/DACVAE.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{
    load_bias, load_mlx_conv_weight, load_mlx_convt_weight, BiasMode, Conv1d, ConvTranspose1d,
};
use crate::codec::dac::{
    load_alpha as dac_load_alpha, load_alpha_mid as dac_load_alpha_mid, DecoderBlock, EncoderBlock,
    ResidualUnit,
};
use crate::codec::wnconv::{load_f32_shaped, snake1d};
use crate::ops;
use crate::{Result, SpeechError};

mod semantic;

pub(crate) use semantic::sinc_resample;
use semantic::SemanticEncoder;

/// How the checkpoint stores its convolutions. The mlx-audio conversions
/// keep MLX layouts (`[out, K, in]`, snake `[1, 1, C]`, codebook
/// `.weight`); the released OmniVoice `audio_tokenizer/` folder ships
/// PyTorch layouts (`[out, in, K]`, ConvTranspose `[in, out, K]`, snake
/// `[1, C, 1]`, codebook `.embed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceLayout {
    Mlx,
    PyTorch,
}

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Tokenizer geometry, one-to-one with the reference dataclass
/// defaults (semantic-path fields kept for config parity; the semantic
/// tower itself loads through [`crate::stt::wav2vec`]).
#[derive(Debug, Clone)]
pub struct HiggsAudioConfig {
    pub sample_rate: u32,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub downsample_factor: usize,
    pub dac_num_codebooks: usize,
    pub dac_encoder_ratios: Vec<usize>,
    pub dac_encoder_hidden: usize,
    pub dac_decoder_hidden: usize,
    pub semantic_sample_rate: u32,
    /// Quantizer latent width. The reference hardcodes this as the
    /// `VectorQuantizer(latent_dim=1024)` constructor default and the
    /// `fc2` input; a config field here keeps tiny fixtures honest.
    pub quantizer_latent_dim: usize,
    /// Decoder input width. The reference hardcodes 256 (`fc2` output
    /// and `AcousticDecoder.conv1` input).
    pub decoder_input_dim: usize,
    /// Semantic-fusion adaptor geometry (the config root's `strides`,
    /// `block_dilations`, `channel_ratios`, `kernel_size`, and
    /// `unit_kernel_size`; defaults match the reference constructor).
    pub strides: Vec<usize>,
    pub block_dilations: Vec<usize>,
    pub channel_ratios: Vec<usize>,
    pub kernel_size: usize,
    pub unit_kernel_size: usize,
    /// The inline Wav2Vec2 dict (`semantic_model_config`); present only
    /// in checkpoints that carry the encode path.
    pub semantic_model_config: Option<serde_json::Value>,
}

impl HiggsAudioConfig {
    /// Reference dataclass defaults.
    pub fn defaults() -> Self {
        HiggsAudioConfig {
            sample_rate: 24_000,
            codebook_size: 1024,
            codebook_dim: 64,
            downsample_factor: 320,
            dac_num_codebooks: 8,
            dac_encoder_ratios: vec![8, 5, 4, 2, 3],
            dac_encoder_hidden: 64,
            dac_decoder_hidden: 1024,
            semantic_sample_rate: 16_000,
            quantizer_latent_dim: 1024,
            decoder_input_dim: 256,
            strides: vec![1, 1],
            block_dilations: vec![1, 1],
            channel_ratios: vec![1, 1],
            kernel_size: 3,
            unit_kernel_size: 3,
            semantic_model_config: None,
        }
    }

    /// Parses a `config.json`-style object over the defaults.
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
        let d = Self::defaults();
        Ok(HiggsAudioConfig {
            sample_rate: u("sample_rate")?.unwrap_or(d.sample_rate as u64) as u32,
            codebook_size: u("codebook_size")?.unwrap_or(d.codebook_size as u64) as usize,
            codebook_dim: u("codebook_dim")?.unwrap_or(d.codebook_dim as u64) as usize,
            downsample_factor: u("downsample_factor")?.unwrap_or(d.downsample_factor as u64)
                as usize,
            dac_num_codebooks: u("dac_num_codebooks")?.unwrap_or(d.dac_num_codebooks as u64)
                as usize,
            dac_encoder_ratios: usize_vec("dac_encoder_ratios")?.unwrap_or(d.dac_encoder_ratios),
            dac_encoder_hidden: u("dac_encoder_hidden")?.unwrap_or(d.dac_encoder_hidden as u64)
                as usize,
            dac_decoder_hidden: u("dac_decoder_hidden")?.unwrap_or(d.dac_decoder_hidden as u64)
                as usize,
            semantic_sample_rate: u("semantic_sample_rate")?
                .unwrap_or(d.semantic_sample_rate as u64) as u32,
            quantizer_latent_dim: u("quantizer_latent_dim")?
                .unwrap_or(d.quantizer_latent_dim as u64)
                as usize,
            decoder_input_dim: u("decoder_input_dim")?.unwrap_or(d.decoder_input_dim as u64)
                as usize,
            strides: usize_vec("strides")?.unwrap_or(d.strides),
            block_dilations: usize_vec("block_dilations")?.unwrap_or(d.block_dilations),
            channel_ratios: usize_vec("channel_ratios")?.unwrap_or(d.channel_ratios),
            kernel_size: u("kernel_size")?.unwrap_or(d.kernel_size as u64) as usize,
            unit_kernel_size: u("unit_kernel_size")?.unwrap_or(d.unit_kernel_size as u64) as usize,
            semantic_model_config: value
                .get("semantic_model_config")
                .cloned()
                .filter(|v| !v.is_null()),
        })
    }

    /// Parses the released Transformers-format config: the DAC geometry
    /// lives under `acoustic_model_config` and the rest at the root.
    pub fn from_pytorch_json(value: &serde_json::Value) -> Result<Self> {
        let mut config = Self::from_json(value)?;
        let acoustic =
            value
                .get("acoustic_model_config")
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "acoustic_model_config".into(),
                    why: "the PyTorch-format Higgs tokenizer config must carry the DAC geometry"
                        .into(),
                })?;
        let usize_field = |name: &str, default: usize| -> usize {
            acoustic
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        if let Some(ratios) = acoustic
            .get("downsampling_ratios")
            .and_then(|v| v.as_array())
        {
            let parsed: Option<Vec<usize>> = ratios
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            if let Some(ratios) = parsed {
                config.dac_encoder_ratios = ratios;
            }
        }
        config.dac_encoder_hidden = usize_field("encoder_hidden_size", config.dac_encoder_hidden);
        config.dac_decoder_hidden = usize_field("decoder_hidden_size", config.dac_decoder_hidden);
        config.decoder_input_dim = usize_field("hidden_size", config.decoder_input_dim);
        Ok(config)
    }

    /// Product of the DAC encoder strides (960 on the default config).
    pub fn acoustic_hop(&self) -> usize {
        self.dac_encoder_ratios.iter().product()
    }

    /// The semantic tower runs at `semantic_sample_rate / downsample_factor`
    /// frames per second; the acoustic path runs at
    /// `sample_rate / acoustic_hop`. The fusion stride-slices the tower
    /// output by their rounded ratio (2 on the default config).
    pub fn semantic_downsample_factor(&self) -> usize {
        let tower_fps = self.semantic_sample_rate as f64 / self.downsample_factor as f64;
        let acoustic_fps = self.sample_rate as f64 / self.acoustic_hop() as f64;
        ((tower_fps / acoustic_fps).round() as usize).max(1)
    }

    /// Encoder channel ladder `hidden x 2^i` (class constant
    /// `_CHANNELS = [64, 128, 256, 512, 1024, 2048]` at the defaults).
    fn encoder_channels(&self) -> Vec<usize> {
        (0..self.dac_encoder_ratios.len() + 1)
            .map(|i| self.dac_encoder_hidden << i)
            .collect()
    }

    /// Quantizer latent width.
    pub fn latent_dim(&self) -> usize {
        self.quantizer_latent_dim
    }
}

/// Plain Conv1d (the reference uses `norm="none"`). MLX source tensors
/// are `[out, K, in]`; PyTorch source tensors are already in the
/// `[out, in, K]` layout the conv consumes.
#[allow(clippy::too_many_arguments)]
fn load_higgs_conv(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    with_bias: bool,
    layout: SourceLayout,
) -> Result<Conv1d> {
    let weight = match layout {
        SourceLayout::Mlx => {
            load_mlx_conv_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)?
        }
        SourceLayout::PyTorch => {
            load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, in_ch, kernel])?
        }
    };
    let bias = load_bias(
        file,
        &format!("{prefix}.bias"),
        out_ch,
        BiasMode::required(with_bias),
    )?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        dilation,
        groups: 1,
        weight,
        bias,
    })
}

/// Plain ConvTranspose1d (`nn.ConvTranspose1d`). MLX source tensors are
/// `[out, K, in]`; PyTorch source tensors are `[in, out, K]`, which is
/// the layout the transposed conv consumes. The reference builds the
/// decoder's transposed convs with `kernel_size = 2 * stride`,
/// `padding = stride // 2`.
#[allow(clippy::too_many_arguments)]
fn load_higgs_convtr(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    rate: usize,
    layout: SourceLayout,
) -> Result<ConvTranspose1d> {
    let kernel = 2 * rate;
    let weight = match layout {
        SourceLayout::Mlx => {
            load_mlx_convt_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)?
        }
        SourceLayout::PyTorch => {
            load_f32_shaped(file, &format!("{prefix}.weight"), &[in_ch, out_ch, kernel])?
        }
    };
    let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel,
        stride: rate,
        padding: rate / 2,
        output_padding: 0,
        groups: 1,
        weight,
        bias,
    })
}

/// Snake1d alpha: MLX source `[1, 1, C]`, PyTorch source `[1, C, 1]`;
/// both flatten to one gain per channel.
fn load_higgs_alpha(
    file: &SafetensorsFile,
    prefix: &str,
    channels: usize,
    layout: SourceLayout,
) -> Result<Vec<f32>> {
    match layout {
        SourceLayout::Mlx => dac_load_alpha(file, &format!("{prefix}.alpha"), channels),
        SourceLayout::PyTorch => dac_load_alpha_mid(file, &format!("{prefix}.alpha"), channels),
    }
}

/// DAC residual unit: Snake, dilated k7 conv (symmetric pad `3 *
/// dilation`), Snake, k1 conv, symmetric crop-add.
fn load_higgs_res_unit(
    file: &SafetensorsFile,
    prefix: &str,
    channels: usize,
    dilation: usize,
    layout: SourceLayout,
) -> Result<ResidualUnit> {
    Ok(ResidualUnit {
        snake1: load_higgs_alpha(file, &format!("{prefix}.snake1"), channels, layout)?,
        conv1: load_higgs_conv(
            file,
            &format!("{prefix}.conv1"),
            channels,
            channels,
            7,
            1,
            3 * dilation,
            dilation,
            true,
            layout,
        )?,
        snake2: load_higgs_alpha(file, &format!("{prefix}.snake2"), channels, layout)?,
        conv2: load_higgs_conv(
            file,
            &format!("{prefix}.conv2"),
            channels,
            channels,
            1,
            1,
            0,
            1,
            true,
            layout,
        )?,
        ch: channels,
    })
}

fn load_higgs_encoder_block(
    file: &SafetensorsFile,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
    layout: SourceLayout,
) -> Result<EncoderBlock> {
    Ok(EncoderBlock {
        units: [
            load_higgs_res_unit(file, &format!("{prefix}.res_unit1"), in_dim, 1, layout)?,
            load_higgs_res_unit(file, &format!("{prefix}.res_unit2"), in_dim, 3, layout)?,
            load_higgs_res_unit(file, &format!("{prefix}.res_unit3"), in_dim, 9, layout)?,
        ],
        snake: load_higgs_alpha(file, &format!("{prefix}.snake1"), in_dim, layout)?,
        // The reference forces padding to ceil(stride / 2) after
        // construction (the pad_mode "none" default s / 2 is wrong
        // for odd strides).
        down: load_higgs_conv(
            file,
            &format!("{prefix}.conv1"),
            in_dim,
            out_dim,
            2 * stride,
            stride,
            stride.div_ceil(2),
            1,
            true,
            layout,
        )?,
        ch: in_dim,
    })
}

struct HiggsDecoderBlock {
    block: DecoderBlock,
    stride: usize,
}

impl HiggsDecoderBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
        layout: SourceLayout,
    ) -> Result<Self> {
        Ok(HiggsDecoderBlock {
            block: DecoderBlock {
                snake: load_higgs_alpha(file, &format!("{prefix}.snake1"), in_dim, layout)?,
                up: load_higgs_convtr(
                    file,
                    &format!("{prefix}.conv_t1"),
                    in_dim,
                    out_dim,
                    stride,
                    layout,
                )?,
                units: [
                    load_higgs_res_unit(file, &format!("{prefix}.res_unit1"), out_dim, 1, layout)?,
                    load_higgs_res_unit(file, &format!("{prefix}.res_unit2"), out_dim, 3, layout)?,
                    load_higgs_res_unit(file, &format!("{prefix}.res_unit3"), out_dim, 9, layout)?,
                ],
            },
            stride,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.block.forward_with(x, |h, t_in| {
            // Trim to the exact upsampled length (odd strides overshoot by
            // one). Channel-major storage means this is a per-channel time
            // crop, not a tail truncate.
            let expected = t_in * self.stride;
            let out_ch = self.block.up.out_ch;
            let len = h.len() / out_ch;
            if len > expected {
                let mut trimmed = vec![0.0f32; expected * out_ch];
                for c in 0..out_ch {
                    trimmed[c * expected..(c + 1) * expected]
                        .copy_from_slice(&h[c * len..c * len + expected]);
                }
                trimmed
            } else {
                h
            }
        })
    }
}

/// DAC-style acoustic encoder: mono `[1, S]` -> features `[256, T']`
/// channel-major (reference output is channels-last `[T', 256]`).
pub struct AcousticEncoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    alpha: Vec<f32>,
    conv2: Conv1d,
    out_ch: usize,
}

impl AcousticEncoder {
    fn load(
        config: &HiggsAudioConfig,
        file: &SafetensorsFile,
        layout: SourceLayout,
    ) -> Result<Self> {
        let channels = config.encoder_channels();
        let conv1 = load_higgs_conv(
            file,
            "acoustic_encoder.conv1",
            1,
            channels[0],
            7,
            1,
            3,
            1,
            true,
            layout,
        )?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.dac_encoder_ratios.iter().enumerate() {
            blocks.push(load_higgs_encoder_block(
                file,
                &format!("acoustic_encoder.block.{i}"),
                channels[i],
                channels[i + 1],
                stride,
                layout,
            )?);
        }
        let last = *channels.last().unwrap();
        let latent = config.decoder_input_dim;
        Ok(AcousticEncoder {
            conv1,
            blocks,
            alpha: load_higgs_alpha(file, "acoustic_encoder.snake1", last, layout)?,
            conv2: load_higgs_conv(
                file,
                "acoustic_encoder.conv2",
                last,
                latent,
                3,
                1,
                1,
                1,
                true,
                layout,
            )?,
            out_ch: latent,
        })
    }

    /// Mono samples -> features channel-major `[latent, frames]`.
    pub fn forward(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut x = self.conv1.forward(samples);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let seq = x.len() / self.out_ch_max();
        snake1d(&mut x, &self.alpha, self.out_ch_max(), seq);
        Ok(self.conv2.forward(&x))
    }

    fn out_ch_max(&self) -> usize {
        self.conv2.in_ch
    }

    /// Frame count helper (channels-last width).
    pub fn frames(&self, features: &[f32]) -> usize {
        features.len() / self.out_ch
    }
}

/// DAC-style acoustic decoder: `[256, T]` channel-major -> mono
/// samples (`T * hop`).
pub struct AcousticDecoder {
    conv1: Conv1d,
    blocks: Vec<HiggsDecoderBlock>,
    alpha: Vec<f32>,
    conv2: Conv1d,
    out_ch: usize,
}

impl AcousticDecoder {
    fn load(
        config: &HiggsAudioConfig,
        file: &SafetensorsFile,
        layout: SourceLayout,
    ) -> Result<Self> {
        let hidden = config.dac_decoder_hidden;
        let latent = config.decoder_input_dim;
        let conv1 = load_higgs_conv(
            file,
            "acoustic_decoder.conv1",
            latent,
            hidden,
            7,
            1,
            3,
            1,
            true,
            layout,
        )?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.dac_encoder_ratios.iter().enumerate() {
            blocks.push(HiggsDecoderBlock::load(
                file,
                &format!("acoustic_decoder.block.{i}"),
                hidden >> i,
                hidden >> (i + 1),
                stride,
                layout,
            )?);
        }
        let final_ch = hidden >> config.dac_encoder_ratios.len();
        Ok(AcousticDecoder {
            conv1,
            blocks,
            alpha: load_higgs_alpha(file, "acoustic_decoder.snake1", final_ch, layout)?,
            conv2: load_higgs_conv(
                file,
                "acoustic_decoder.conv2",
                final_ch,
                1,
                7,
                1,
                3,
                1,
                true,
                layout,
            )?,
            out_ch: final_ch,
        })
    }

    /// Features channel-major `[latent, T]` -> mono samples.
    pub fn forward(&self, features: &[f32]) -> Vec<f32> {
        let mut x = self.conv1.forward(features);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let seq = x.len() / self.out_ch;
        snake1d(&mut x, &self.alpha, self.out_ch, seq);
        self.conv2.forward(&x)
    }
}

/// One VQ book: `project_in` -> codebook -> `project_out`.
struct HiggsVq {
    project_in_in: usize,
    project_in: Vec<f32>,
    project_in_bias: Vec<f32>,
    codebook: Vec<f32>,
    codebook_sq: Vec<f32>,
    codebook_size: usize,
    codebook_dim: usize,
    project_out: Vec<f32>,
    project_out_bias: Vec<f32>,
    latent_dim: usize,
}

impl HiggsVq {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        latent_dim: usize,
        size: usize,
        dim: usize,
        layout: SourceLayout,
    ) -> Result<Self> {
        let pin = load_f32_shaped(
            file,
            &format!("{prefix}.project_in.weight"),
            &[dim, latent_dim],
        )?;
        let pin_bias = load_f32_shaped(file, &format!("{prefix}.project_in.bias"), &[dim])?;
        // The MLX conversions rename the PyTorch `.codebook.embed` to
        // `.codebook.weight`; both flatten to `[size, dim]`.
        let codebook_name = match layout {
            SourceLayout::Mlx => format!("{prefix}.codebook.weight"),
            SourceLayout::PyTorch => format!("{prefix}.codebook.embed"),
        };
        let codebook = load_f32_shaped(file, &codebook_name, &[size, dim])?;
        let pout = load_f32_shaped(
            file,
            &format!("{prefix}.project_out.weight"),
            &[latent_dim, dim],
        )?;
        let pout_bias =
            load_f32_shaped(file, &format!("{prefix}.project_out.bias"), &[latent_dim])?;
        let mut codebook_sq = vec![0.0f32; size];
        for (r, sq) in codebook_sq.iter_mut().enumerate() {
            *sq = codebook[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
        }
        Ok(HiggsVq {
            project_in_in: latent_dim,
            project_in: pin,
            project_in_bias: pin_bias,
            codebook,
            codebook_sq,
            codebook_size: size,
            codebook_dim: dim,
            project_out: pout,
            project_out_bias: pout_bias,
            latent_dim,
        })
    }

    /// Rows `[rows, latent]` -> indices (Euclidean argmin over the
    /// projected frames, first-wins ties).
    fn encode(&self, z: &[f32], rows: usize) -> Vec<i32> {
        let projected = ops::linear(
            z,
            &self.project_in,
            Some(&self.project_in_bias),
            rows,
            self.project_in_in,
            self.codebook_dim,
        );
        let mut idx = vec![0i32; rows];
        for (r, code) in idx.iter_mut().enumerate() {
            let row = &projected[r * self.codebook_dim..(r + 1) * self.codebook_dim];
            let z2: f32 = row.iter().map(|v| v * v).sum();
            let mut best = f32::INFINITY;
            let mut best_idx = 0usize;
            for c in 0..self.codebook_size {
                let e = &self.codebook[c * self.codebook_dim..(c + 1) * self.codebook_dim];
                let mut dot = 0.0f32;
                for (a, &b) in row.iter().zip(e) {
                    dot += a * b;
                }
                let dist = z2 - 2.0 * dot + self.codebook_sq[c];
                if dist < best {
                    best = dist;
                    best_idx = c;
                }
            }
            *code = best_idx as i32;
        }
        idx
    }

    /// Indices -> recon rows `[rows, latent]` (project_out of the
    /// codebook rows).
    fn decode_codes(&self, codes: &[i32], rows: usize) -> Result<Vec<f32>> {
        let mut embedded = vec![0.0f32; rows * self.codebook_dim];
        for (r, &c) in codes.iter().enumerate() {
            let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                why: format!("negative code {c}"),
            })?;
            if idx >= self.codebook_size {
                return Err(SpeechError::Input {
                    why: format!("code {c} out of range for {} entries", self.codebook_size),
                });
            }
            embedded[r * self.codebook_dim..(r + 1) * self.codebook_dim].copy_from_slice(
                &self.codebook[idx * self.codebook_dim..(idx + 1) * self.codebook_dim],
            );
        }
        Ok(ops::linear(
            &embedded,
            &self.project_out,
            Some(&self.project_out_bias),
            rows,
            self.codebook_dim,
            self.latent_dim,
        ))
    }
}

/// Greedy residual VQ over the books.
pub struct HiggsRvq {
    books: Vec<HiggsVq>,
    latent_dim: usize,
}

impl HiggsRvq {
    fn load(
        config: &HiggsAudioConfig,
        file: &SafetensorsFile,
        layout: SourceLayout,
    ) -> Result<Self> {
        let latent = config.latent_dim();
        let mut books = Vec::with_capacity(config.dac_num_codebooks);
        for q in 0..config.dac_num_codebooks {
            books.push(HiggsVq::load(
                file,
                &format!("quantizer.quantizers.{q}"),
                latent,
                config.codebook_size,
                config.codebook_dim,
                layout,
            )?);
        }
        Ok(HiggsRvq {
            books,
            latent_dim: latent,
        })
    }

    /// Rows `[T, latent]` -> codes per book (greedy residual).
    pub fn encode(&self, z: &[f32], frames: usize) -> Result<Vec<Vec<i32>>> {
        let mut residual = z.to_vec();
        let mut tokens = Vec::with_capacity(self.books.len());
        for book in &self.books {
            let idx = book.encode(&residual, frames);
            let recon = book.decode_codes(&idx, frames)?;
            for (v, r) in residual.iter_mut().zip(&recon) {
                *v -= r;
            }
            tokens.push(idx);
        }
        Ok(tokens)
    }

    /// Codes per book -> rows `[T, latent]`.
    pub fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.len() > self.books.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "received {} books, maximum is {}",
                    codes.len(),
                    self.books.len()
                ),
            });
        }
        let mut z = vec![0.0f32; frames * self.latent_dim];
        for (q, book) in codes.iter().enumerate() {
            let recon = self.books[q].decode_codes(book, frames)?;
            for (v, r) in z.iter_mut().zip(recon) {
                *v += r;
            }
        }
        Ok(z)
    }
}

/// The encode-side modules a checkpoint needs for voice-clone reference
/// encoding: the Wav2Vec2 semantic tower, the `encoder_semantic` CNN,
/// and the fusion `fc`.
struct EncodePath {
    tower: crate::stt::wav2vec::SemanticTower,
    encoder_semantic: SemanticEncoder,
    fc: (usize, usize, Vec<f32>, Vec<f32>),
}

/// Loaded Higgs Audio tokenizer, batch-of-one decode contract.
pub struct HiggsAudioTokenizer {
    pub config: HiggsAudioConfig,
    acoustic_encoder: AcousticEncoder,
    acoustic_decoder: AcousticDecoder,
    rvq: HiggsRvq,
    fc2: (usize, usize, Vec<f32>, Vec<f32>),
    encode_path: Option<EncodePath>,
}

impl HiggsAudioTokenizer {
    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`. Both the mlx-audio conversion layout and the
    /// released Transformers-format layout (DAC geometry nested under
    /// `acoustic_model_config`, PyTorch conv layouts) load through here.
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
        let (config, layout) = if value.get("acoustic_model_config").is_some() {
            (
                HiggsAudioConfig::from_pytorch_json(&value)?,
                SourceLayout::PyTorch,
            )
        } else {
            (HiggsAudioConfig::from_json(&value)?, SourceLayout::Mlx)
        };
        let file =
            SafetensorsFile::open(&dir.join("model.safetensors")).map_err(SpeechError::from)?;
        Self::load_with_layout(config, &file, layout)
    }

    pub fn load(config: HiggsAudioConfig, file: &SafetensorsFile) -> Result<Self> {
        Self::load_with_layout(config, file, SourceLayout::Mlx)
    }

    fn load_with_layout(
        config: HiggsAudioConfig,
        file: &SafetensorsFile,
        layout: SourceLayout,
    ) -> Result<Self> {
        let acoustic_encoder = AcousticEncoder::load(&config, file, layout)?;
        let acoustic_decoder = AcousticDecoder::load(&config, file, layout)?;
        let rvq = HiggsRvq::load(&config, file, layout)?;
        let latent = config.latent_dim();
        let dec_in = config.decoder_input_dim;
        let w = load_f32_shaped(file, "fc2.weight", &[dec_in, latent])?;
        let b = load_f32_shaped(file, "fc2.bias", &[dec_in])?;
        let encode_path = match &config.semantic_model_config {
            Some(semantic_config) => Some(Self::load_encode_path(&config, file, semantic_config)?),
            None => None,
        };
        Ok(HiggsAudioTokenizer {
            config,
            acoustic_encoder,
            acoustic_decoder,
            rvq,
            fc2: (latent, dec_in, w, b),
            encode_path,
        })
    }

    /// Loads the encode path over the inline `semantic_model_config`
    /// dict; a checkpoint without it cannot encode references.
    fn load_encode_path(
        config: &HiggsAudioConfig,
        file: &SafetensorsFile,
        semantic_config: &serde_json::Value,
    ) -> Result<EncodePath> {
        let tower = crate::stt::wav2vec::SemanticTower::load(file, semantic_config)?;
        let hidden = tower.config().hidden_size;
        let encoder_semantic = SemanticEncoder::load(
            file,
            hidden,
            &config.strides,
            &config.block_dilations,
            &config.channel_ratios,
            config.kernel_size,
            config.unit_kernel_size,
        )?;
        let fusion_dim = hidden + config.decoder_input_dim;
        let w = load_f32_shaped(file, "fc.weight", &[fusion_dim, fusion_dim])?;
        let b = load_f32_shaped(file, "fc.bias", &[fusion_dim])?;
        Ok(EncodePath {
            tower,
            encoder_semantic,
            fc: (fusion_dim, fusion_dim, w, b),
        })
    }

    /// Codes per book -> mono samples (`frames * hop`). This is the
    /// reference `decode` path: RVQ sum, `fc2`, acoustic decoder.
    pub fn decode_tokens(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let frames = codes.first().map_or(0, |book| book.len());
        let z = self.rvq.decode(codes, frames)?;
        let (in_dim, out_dim, w, b) = &self.fc2;
        let projected = ops::linear(&z, w, Some(b), frames, *in_dim, *out_dim);
        // Rows [T, 256] -> channel-major for the decoder.
        let mut features = vec![0.0f32; projected.len()];
        for t in 0..frames {
            for d in 0..*out_dim {
                features[d * frames + t] = projected[t * out_dim + d];
            }
        }
        Ok(self.acoustic_decoder.forward(&features))
    }

    /// Mono samples -> acoustic features, rows `[T', latent_out]`
    /// (channels-last, matching the reference output shape). These are
    /// the inputs RVQ quantize after semantic fusion.
    pub fn encode_acoustic(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let features = self.acoustic_encoder.forward(samples)?;
        let frames = self.acoustic_encoder.frames(&features);
        let ch = self.acoustic_encoder.out_ch;
        let mut rows = vec![0.0f32; features.len()];
        for t in 0..frames {
            for d in 0..ch {
                rows[t * ch + d] = features[d * frames + t];
            }
        }
        Ok(rows)
    }

    /// RVQ encode over embedding rows `[T, 1024]` (the reference
    /// `quantizer.encode`; the caller fuses semantic features first on
    /// the full encode path).
    pub fn rvq_encode(&self, z_rows: &[f32], frames: usize) -> Result<Vec<Vec<i32>>> {
        self.rvq.encode(z_rows, frames)
    }

    /// RVQ decode to embedding rows `[T, 1024]` (the reference
    /// `quantizer.decode`).
    pub fn rvq_decode_rows(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let frames = codes.first().map_or(0, |book| book.len());
        self.rvq.decode(codes, frames)
    }

    /// The full reference `encode`: mono samples at the checkpoint's
    /// `sample_rate` -> code rows `[T, books]` (frames-major, the shape
    /// `tokenizer.encode(wav)[0]` returns). Requires the checkpoint's
    /// semantic encode path: sinc resample to the tower rate, HuBERT
    /// edge padding, the hidden-state mean, the downsample to the
    /// acoustic frame rate, `encoder_semantic`, the acoustic features,
    /// the `fc` fusion, and greedy residual quantization.
    pub fn encode_tokens(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
        let encode = self.encode_path.as_ref().ok_or_else(|| SpeechError::Unsupported {
            why: "this checkpoint carries no semantic encode path; voice-clone reference encoding is unavailable".into(),
        })?;
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "reference audio is empty".into(),
            });
        }
        // The tower input: sinc resample to the semantic rate, then the
        // HuBERT conv stack's edge pad (downsample_factor / 2 per side).
        let resampled = sinc_resample(
            samples,
            self.config.sample_rate as usize,
            self.config.semantic_sample_rate as usize,
        );
        let pad = self.config.downsample_factor / 2;
        let mut tower_input = vec![0.0f32; pad];
        tower_input.extend_from_slice(&resampled);
        tower_input.extend(std::iter::repeat_n(0.0f32, pad));

        let (mean_states, _final, tower_steps) = encode.tower.forward_hidden_mean(&tower_input)?;
        if tower_steps == 0 {
            return Err(SpeechError::Input {
                why: "reference audio is too short for the semantic tower".into(),
            });
        }
        // Stride-slice the tower frames to the acoustic frame rate
        // (`semantic_features[:, ::dsf, :]`).
        let dsf = self.config.semantic_downsample_factor();
        let hidden = encode.tower.config().hidden_size;
        let kept = tower_steps.div_ceil(dsf);
        let mut semantic = vec![0.0f32; kept * hidden];
        for t in 0..kept {
            let source = t * dsf;
            debug_assert!(
                source < tower_steps,
                "downsample index leaves the tower frames"
            );
            semantic[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&mean_states[source * hidden..(source + 1) * hidden]);
        }
        let semantic_steps = kept;
        let semantic = encode.encoder_semantic.forward(&semantic, semantic_steps)?;
        let semantic_steps = encode.encoder_semantic.output_steps(semantic_steps);

        let acoustic_dim = self.config.decoder_input_dim;
        let acoustic_rows = self.encode_acoustic(samples)?;
        let acoustic_frames = acoustic_rows.len() / acoustic_dim;

        let time_steps = semantic_steps.min(acoustic_frames);
        if time_steps == 0 {
            return Err(SpeechError::Input {
                why: "reference audio produced no aligned frames".into(),
            });
        }
        let semantic_dim = hidden;
        let fusion_in = acoustic_dim + semantic_dim;
        let mut embeddings = vec![0.0f32; time_steps * fusion_in];
        for t in 0..time_steps {
            embeddings[t * fusion_in..t * fusion_in + acoustic_dim]
                .copy_from_slice(&acoustic_rows[t * acoustic_dim..(t + 1) * acoustic_dim]);
            embeddings[t * fusion_in + acoustic_dim..(t + 1) * fusion_in]
                .copy_from_slice(&semantic[t * semantic_dim..(t + 1) * semantic_dim]);
        }
        let (in_dim, out_dim, w, b) = &encode.fc;
        let fused = ops::linear(&embeddings, w, Some(b), time_steps, *in_dim, *out_dim);
        let books = self.rvq.encode(&fused, time_steps)?;
        // Books [C][T] -> frames [T][C], the reference token layout.
        let mut rows = vec![vec![0i32; books.len()]; time_steps];
        for (book, codes) in books.iter().enumerate() {
            for (t, &code) in codes.iter().enumerate() {
                rows[t][book] = code;
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests;
