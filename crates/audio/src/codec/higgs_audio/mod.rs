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

use crate::codec::wnconv::{load_f32_shaped, snake1d};
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Tokenizer geometry, one-to-one with the reference dataclass
/// defaults (semantic-path fields kept for config parity; the semantic
/// tower itself is out of scope).
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
        })
    }

    /// Product of the DAC encoder strides (960 on the default config).
    pub fn acoustic_hop(&self) -> usize {
        self.dac_encoder_ratios.iter().product()
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

/// Plain Conv1d (the reference uses `norm="none"`), stored MLX layout
/// `[out, K, in]`, explicit padding.
#[derive(Debug, Clone)]
struct HiggsConv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl HiggsConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kernel, in_ch])?;
        let mut weight = vec![0.0f32; stored.len()];
        for oc in 0..out_ch {
            for kk in 0..kernel {
                for ic in 0..in_ch {
                    weight[oc * in_ch * kernel + ic * kernel + kk] =
                        stored[oc * kernel * in_ch + kk * in_ch + ic];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            if with_bias {
                return Err(SpeechError::Tensor {
                    name: bias_name,
                    why: "required by the reference layer".to_string(),
                });
            }
            None
        };
        Ok(HiggsConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            dilation,
            weight,
            bias,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            1,
        )
    }
}

/// Plain ConvTranspose1d (`nn.ConvTranspose1d`), stored MLX layout
/// `[out, K, in]`, converted to the PyTorch `[in, out, K]`.
#[derive(Debug, Clone)]
struct HiggsConvTr {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl HiggsConvTr {
    /// The reference builds the decoder's transposed convs with
    /// `kernel_size = 2 * stride`, `padding = stride // 2`.
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        rate: usize,
    ) -> Result<Self> {
        let kernel = 2 * rate;
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kernel, in_ch])?;
        let mut weight = vec![0.0f32; stored.len()];
        for ic in 0..in_ch {
            for kk in 0..kernel {
                for oc in 0..out_ch {
                    weight[ic * out_ch * kernel + oc * kernel + kk] =
                        stored[oc * kernel * in_ch + kk * in_ch + ic];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(HiggsConvTr {
            in_ch,
            out_ch,
            kernel,
            stride: rate,
            padding: rate / 2,
            weight,
            bias,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            0,
            1,
        )
    }
}

fn load_alpha(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Vec<f32>> {
    let stored = load_f32_shaped(file, &format!("{prefix}.alpha"), &[1, 1, channels])?;
    Ok(stored)
}

/// DAC residual unit: Snake, dilated k7 conv (symmetric pad `3 *
/// dilation`), Snake, k1 conv, symmetric crop-add.
struct HiggsResidualUnit {
    alpha1: Vec<f32>,
    conv1: HiggsConv1d,
    alpha2: Vec<f32>,
    conv2: HiggsConv1d,
    channels: usize,
}

impl HiggsResidualUnit {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(HiggsResidualUnit {
            alpha1: load_alpha(file, &format!("{prefix}.snake1"), channels)?,
            conv1: HiggsConv1d::load(
                file,
                &format!("{prefix}.conv1"),
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
                true,
            )?,
            alpha2: load_alpha(file, &format!("{prefix}.snake2"), channels)?,
            conv2: HiggsConv1d::load(
                file,
                &format!("{prefix}.conv2"),
                channels,
                channels,
                1,
                1,
                0,
                1,
                true,
            )?,
            channels,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.channels;
        let mut y = x.to_vec();
        snake1d(&mut y, &self.alpha1, self.channels, seq);
        let y = self.conv1.forward(&y);
        let seq2 = y.len() / self.channels;
        let mut y = y;
        snake1d(&mut y, &self.alpha2, self.channels, seq2);
        let y = self.conv2.forward(&y);
        let out_seq = y.len() / self.channels;
        let pad = seq.saturating_sub(out_seq) / 2;
        let mut out = vec![0.0f32; self.channels * out_seq];
        for c in 0..self.channels {
            for t in 0..out_seq {
                let r = if t + pad < seq {
                    x[c * seq + t + pad]
                } else {
                    0.0
                };
                out[c * out_seq + t] = r + y[c * out_seq + t];
            }
        }
        out
    }
}

struct HiggsEncoderBlock {
    res_units: [HiggsResidualUnit; 3],
    alpha: Vec<f32>,
    conv: HiggsConv1d,
    in_ch: usize,
}

impl HiggsEncoderBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(HiggsEncoderBlock {
            res_units: [
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit1"), in_dim, 1)?,
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit2"), in_dim, 3)?,
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit3"), in_dim, 9)?,
            ],
            alpha: load_alpha(file, &format!("{prefix}.snake1"), in_dim)?,
            // The reference forces padding to ceil(stride / 2) after
            // construction (the pad_mode "none" default s / 2 is wrong
            // for odd strides).
            conv: HiggsConv1d::load(
                file,
                &format!("{prefix}.conv1"),
                in_dim,
                out_dim,
                2 * stride,
                stride,
                stride.div_ceil(2),
                1,
                true,
            )?,
            in_ch: in_dim,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let x = self.res_units[0].forward(x);
        let x = self.res_units[1].forward(&x);
        let x = self.res_units[2].forward(&x);
        let seq = x.len() / self.in_ch;
        let mut x = x.to_vec();
        snake1d(&mut x, &self.alpha, self.in_ch, seq);
        self.conv.forward(&x)
    }
}

struct HiggsDecoderBlock {
    alpha: Vec<f32>,
    conv_t: HiggsConvTr,
    res_units: [HiggsResidualUnit; 3],
    in_ch: usize,
    stride: usize,
}

impl HiggsDecoderBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(HiggsDecoderBlock {
            alpha: load_alpha(file, &format!("{prefix}.snake1"), in_dim)?,
            conv_t: HiggsConvTr::load(file, &format!("{prefix}.conv_t1"), in_dim, out_dim, stride)?,
            res_units: [
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit1"), out_dim, 1)?,
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit2"), out_dim, 3)?,
                HiggsResidualUnit::load(file, &format!("{prefix}.res_unit3"), out_dim, 9)?,
            ],
            in_ch: in_dim,
            stride,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let t_in = x.len() / self.in_ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &self.alpha, self.in_ch, t_in);
        let mut h = self.conv_t.forward(&h);
        // Trim to the exact upsampled length (odd strides overshoot by
        // one). Channel-major storage means this is a per-channel time
        // crop, not a tail truncate.
        let expected = t_in * self.stride;
        let out_ch = self.conv_t.out_ch;
        let len = h.len() / out_ch;
        if len > expected {
            let mut trimmed = vec![0.0f32; expected * out_ch];
            for c in 0..out_ch {
                trimmed[c * expected..(c + 1) * expected]
                    .copy_from_slice(&h[c * len..c * len + expected]);
            }
            h = trimmed;
        }
        let h = self.res_units[0].forward(&h);
        let h = self.res_units[1].forward(&h);
        self.res_units[2].forward(&h)
    }
}

/// DAC-style acoustic encoder: mono `[1, S]` -> features `[256, T']`
/// channel-major (reference output is channels-last `[T', 256]`).
pub struct AcousticEncoder {
    conv1: HiggsConv1d,
    blocks: Vec<HiggsEncoderBlock>,
    alpha: Vec<f32>,
    conv2: HiggsConv1d,
    out_ch: usize,
}

impl AcousticEncoder {
    fn load(config: &HiggsAudioConfig, file: &SafetensorsFile) -> Result<Self> {
        let channels = config.encoder_channels();
        let conv1 = HiggsConv1d::load(
            file,
            "acoustic_encoder.conv1",
            1,
            channels[0],
            7,
            1,
            3,
            1,
            true,
        )?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.dac_encoder_ratios.iter().enumerate() {
            blocks.push(HiggsEncoderBlock::load(
                file,
                &format!("acoustic_encoder.block.{i}"),
                channels[i],
                channels[i + 1],
                stride,
            )?);
        }
        let last = *channels.last().unwrap();
        let latent = config.decoder_input_dim;
        Ok(AcousticEncoder {
            conv1,
            blocks,
            alpha: load_alpha(file, "acoustic_encoder.snake1", last)?,
            conv2: HiggsConv1d::load(
                file,
                "acoustic_encoder.conv2",
                last,
                latent,
                3,
                1,
                1,
                1,
                true,
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
    conv1: HiggsConv1d,
    blocks: Vec<HiggsDecoderBlock>,
    alpha: Vec<f32>,
    conv2: HiggsConv1d,
    out_ch: usize,
}

impl AcousticDecoder {
    fn load(config: &HiggsAudioConfig, file: &SafetensorsFile) -> Result<Self> {
        let hidden = config.dac_decoder_hidden;
        let latent = config.decoder_input_dim;
        let conv1 = HiggsConv1d::load(
            file,
            "acoustic_decoder.conv1",
            latent,
            hidden,
            7,
            1,
            3,
            1,
            true,
        )?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.dac_encoder_ratios.iter().enumerate() {
            blocks.push(HiggsDecoderBlock::load(
                file,
                &format!("acoustic_decoder.block.{i}"),
                hidden >> i,
                hidden >> (i + 1),
                stride,
            )?);
        }
        let final_ch = hidden >> config.dac_encoder_ratios.len();
        Ok(AcousticDecoder {
            conv1,
            blocks,
            alpha: load_alpha(file, "acoustic_decoder.snake1", final_ch)?,
            conv2: HiggsConv1d::load(
                file,
                "acoustic_decoder.conv2",
                final_ch,
                1,
                7,
                1,
                3,
                1,
                true,
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
    ) -> Result<Self> {
        let pin = load_f32_shaped(
            file,
            &format!("{prefix}.project_in.weight"),
            &[dim, latent_dim],
        )?;
        let pin_bias = load_f32_shaped(file, &format!("{prefix}.project_in.bias"), &[dim])?;
        let codebook = load_f32_shaped(file, &format!("{prefix}.codebook.weight"), &[size, dim])?;
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
    fn load(config: &HiggsAudioConfig, file: &SafetensorsFile) -> Result<Self> {
        let latent = config.latent_dim();
        let mut books = Vec::with_capacity(config.dac_num_codebooks);
        for q in 0..config.dac_num_codebooks {
            books.push(HiggsVq::load(
                file,
                &format!("quantizer.quantizers.{q}"),
                latent,
                config.codebook_size,
                config.codebook_dim,
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

/// Loaded Higgs Audio tokenizer, batch-of-one decode contract.
pub struct HiggsAudioTokenizer {
    pub config: HiggsAudioConfig,
    acoustic_encoder: AcousticEncoder,
    acoustic_decoder: AcousticDecoder,
    rvq: HiggsRvq,
    fc2: (usize, usize, Vec<f32>, Vec<f32>),
}

impl HiggsAudioTokenizer {
    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`.
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
        let config = HiggsAudioConfig::from_json(&value)?;
        let file =
            SafetensorsFile::open(&dir.join("model.safetensors")).map_err(SpeechError::from)?;
        Self::load(config, &file)
    }

    pub fn load(config: HiggsAudioConfig, file: &SafetensorsFile) -> Result<Self> {
        let acoustic_encoder = AcousticEncoder::load(&config, file)?;
        let acoustic_decoder = AcousticDecoder::load(&config, file)?;
        let rvq = HiggsRvq::load(&config, file)?;
        let latent = config.latent_dim();
        let dec_in = config.decoder_input_dim;
        let w = load_f32_shaped(file, "fc2.weight", &[dec_in, latent])?;
        let b = load_f32_shaped(file, "fc2.bias", &[dec_in])?;
        Ok(HiggsAudioTokenizer {
            config,
            acoustic_encoder,
            acoustic_decoder,
            rvq,
            fc2: (latent, dec_in, w, b),
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
}

#[cfg(test)]
mod tests;
