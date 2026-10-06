//! DACVAE: VAE-style continuous audio codec for SAM-Audio.
//!
//! Reference: `mlx_audio/codec/models/dacvae/codec.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/dacvae).
//! A DAC-shaped weight-norm conv encoder (Snake activations, residual
//! units with dilations 1/3/9, static symmetric padding), a
//! continuous-latent projection to mean/logvar halves (the mean is the
//! code), and a mirrored transposed-conv decoder with tanh output.
//!
//! Scope: the batch `encode` / `decode` contract. The watermarking
//! subsystem (StakedLSTM message encoder/decoder, MsgProcessor) is not
//! part of that contract -- `DACVAE.decode` never touches it -- and is
//! not ported; the checkpoint's watermark tensors are simply unused.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::{load_f32_shaped, WnConv1d};
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Codec geometry, one-to-one with the reference dataclass.
#[derive(Debug, Clone)]
pub struct DacvaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub codebook_dim: usize,
    pub sample_rate: u32,
}

impl DacvaeConfig {
    /// Parses the checkpoint `config.json` with the dataclass defaults.
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
        Ok(DacvaeConfig {
            encoder_dim: u("encoder_dim")?.unwrap_or(64) as usize,
            encoder_rates: usize_vec("encoder_rates")?.unwrap_or_else(|| vec![2, 8, 10, 12]),
            latent_dim: u("latent_dim")?.unwrap_or(1024) as usize,
            decoder_dim: u("decoder_dim")?.unwrap_or(1536) as usize,
            decoder_rates: usize_vec("decoder_rates")?.unwrap_or_else(|| vec![12, 10, 8, 2]),
            codebook_dim: u("codebook_dim")?.unwrap_or(128) as usize,
            sample_rate: u("sample_rate")?.unwrap_or(48000) as u32,
        })
    }

    /// Total encoder downsampling.
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
}

/// Snake1d alpha, stored `[1, 1, C]` in the dacvae layout.
fn load_alpha(file: &SafetensorsFile, name: &str, channels: usize) -> Result<Vec<f32>> {
    load_f32_shaped(file, name, &[1, 1, channels])
}

fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    for c in 0..channels {
        let a = alpha[c];
        let recip = 1.0 / (a + 1e-9);
        for f in 0..frames {
            let v = &mut x[c * frames + f];
            *v += recip * (a * *v).sin().powi(2);
        }
    }
}

/// ConvTranspose1d with the descript-style out-first stored weight and
/// static symmetric padding `(stride + 1) // 2` (pad_mode "none", no
/// unpad). Weight norm is folded at load.
struct DacvaeConvT {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    /// PyTorch layout `[in, out, K]`, weight norm folded.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl DacvaeConvT {
    fn load(file: &SafetensorsFile, prefix: &str, stride: usize) -> Result<Self> {
        let v_name = format!("{prefix}.weight_v");
        let desc = file
            .descriptor(&v_name)
            .ok_or_else(|| SpeechError::Tensor {
                name: v_name.clone(),
                why: "missing convtr weight".to_string(),
            })?;
        if desc.shape.len() != 3 {
            return Err(SpeechError::Tensor {
                name: v_name.clone(),
                why: format!("expected 3-D weight, got {:?}", desc.shape),
            });
        }
        let (out_ch, kernel, in_ch) = (desc.shape[0], desc.shape[1], desc.shape[2]);
        let v = load_f32_shaped(file, &v_name, &[out_ch, kernel, in_ch])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[1, 1, in_ch])?;
        let bias = load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?;
        // normalize_weight(v, except_dim=2): per-in-channel norm over
        // (out, K) in row-major order.
        let mut weight = vec![0.0f32; in_ch * out_ch * kernel];
        for (ic, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            for o in 0..out_ch {
                for k in 0..kernel {
                    acc += v[o * kernel * in_ch + k * in_ch + ic]
                        * v[o * kernel * in_ch + k * in_ch + ic];
                }
            }
            let norm = acc.sqrt();
            for o in 0..out_ch {
                for k in 0..kernel {
                    weight[ic * out_ch * kernel + o * kernel + k] =
                        gv * v[o * kernel * in_ch + k * in_ch + ic] / norm;
                }
            }
        }
        Ok(DacvaeConvT {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding: (stride + 1) / 2,
            weight,
            bias: Some(bias),
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

/// Static-padding conv (pad_mode "none"): MLX symmetric padding
/// `(K - stride) * dilation // 2`.
struct DacvaeConv {
    conv: WnConv1d,
}

impl DacvaeConv {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let padding = (kernel - stride) * dilation / 2;
        Ok(DacvaeConv {
            conv: WnConv1d::load(
                file, prefix, in_ch, out_ch, kernel, stride, padding, dilation, 1,
            )?,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.conv.forward(x)
    }
}

/// ResidualUnit (main path): Snake, dilated conv, Snake, 1x1 conv,
/// symmetric crop-add (true_skip=false variant).
struct DacvaeRu {
    alpha1: Vec<f32>,
    conv1: DacvaeConv,
    alpha2: Vec<f32>,
    conv2: WnConv1d,
    ch: usize,
}

impl DacvaeRu {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.ch;
        let mut y = x.to_vec();
        snake(&mut y, &self.alpha1, self.ch, seq);
        let y = self.conv1.forward(&y);
        let seq2 = y.len() / self.ch;
        let mut y = y;
        snake(&mut y, &self.alpha2, self.ch, seq2);
        let y = self.conv2.forward(&y);
        // pad = (x_len - y_len) / 2, crop the residual symmetrically.
        let out_seq = y.len() / self.ch;
        let pad = seq.saturating_sub(out_seq) / 2;
        let mut out = vec![0.0f32; self.ch * out_seq];
        for c in 0..self.ch {
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

struct DacvaeEncoderBlock {
    res: [DacvaeRu; 3],
    alpha: Vec<f32>,
    down: DacvaeConv,
    ch_in: usize,
}

impl DacvaeEncoderBlock {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h = self.res[0].forward(x);
        let h = self.res[1].forward(&h);
        let h = self.res[2].forward(&h);
        let seq = h.len() / self.ch_in;
        let mut h = h;
        snake(&mut h, &self.alpha, self.ch_in, seq);
        self.down.forward(&h)
    }
}

struct DacvaeEncoder {
    conv_in: DacvaeConv,
    blocks: Vec<DacvaeEncoderBlock>,
    alpha_out: Vec<f32>,
    conv_out: WnConv1d,
}

struct DacvaeDecoderBlock {
    alpha: Vec<f32>,
    up: DacvaeConvT,
    res: [DacvaeRu; 3],
}

impl DacvaeDecoderBlock {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.up.in_ch;
        let mut h = x.to_vec();
        snake(&mut h, &self.alpha, self.up.in_ch, seq);
        let mut h = self.up.forward(&h);
        let ch = self.up.out_ch;
        h = self.res[0].forward(&h);
        let h = self.res[1].forward(&h);
        let h = self.res[2].forward(&h);
        let _ = ch;
        h
    }
}

struct DacvaeDecoder {
    conv_in: DacvaeConv,
    blocks: Vec<DacvaeDecoderBlock>,
    alpha_out: Vec<f32>,
    conv_out: WnConv1d,
}

/// Loaded DACVAE codec, batch-of-one.
pub struct Dacvae {
    pub config: DacvaeConfig,
    encoder: DacvaeEncoder,
    decoder: DacvaeDecoder,
    in_proj: WnConv1d,
    out_proj: WnConv1d,
}

impl Dacvae {
    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`.
    pub fn open(dir: &Path) -> Result<Dacvae> {
        let config_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&config_text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = DacvaeConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Dacvae::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: DacvaeConfig, file: &SafetensorsFile) -> Result<Dacvae> {
        let encoder = Self::load_encoder(&config, file)?;
        let decoder = Self::load_decoder(&config, file)?;
        let in_proj = WnConv1d::load(
            file,
            "quantizer_in_proj",
            config.latent_dim,
            config.codebook_dim * 2,
            1,
            1,
            0,
            1,
            1,
        )?;
        let out_proj = WnConv1d::load(
            file,
            "quantizer_out_proj",
            config.codebook_dim,
            config.latent_dim,
            1,
            1,
            0,
            1,
            1,
        )?;
        Ok(Dacvae {
            config,
            encoder,
            decoder,
            in_proj,
            out_proj,
        })
    }

    fn load_encoder(config: &DacvaeConfig, file: &SafetensorsFile) -> Result<DacvaeEncoder> {
        let d = config.encoder_dim;
        let conv_in = DacvaeConv::load(file, "encoder.conv_in", 1, d, 7, 1, 1)?;
        let mut blocks = Vec::new();
        let mut ch = d;
        for (i, &stride) in config.encoder_rates.iter().enumerate() {
            ch *= 2;
            let half = ch / 2;
            let base = format!("encoder.blocks.{i}");
            let ru = |name: &str, dilation: usize| -> Result<DacvaeRu> {
                Ok(DacvaeRu {
                    alpha1: load_alpha(file, &format!("{base}.{name}.act1.alpha"), half)?,
                    conv1: DacvaeConv::load(
                        file,
                        &format!("{base}.{name}.conv1"),
                        half,
                        half,
                        7,
                        1,
                        dilation,
                    )?,
                    alpha2: load_alpha(file, &format!("{base}.{name}.act2.alpha"), half)?,
                    conv2: WnConv1d::load(
                        file,
                        &format!("{base}.{name}.conv2"),
                        half,
                        half,
                        1,
                        1,
                        0,
                        1,
                        1,
                    )?,
                    ch: half,
                })
            };
            let res = [ru("res1", 1)?, ru("res2", 3)?, ru("res3", 9)?];
            let alpha = load_alpha(file, &format!("{base}.snake.alpha"), half)?;
            let down = DacvaeConv::load(
                file,
                &format!("{base}.conv"),
                half,
                ch,
                2 * stride,
                stride,
                1,
            )?;
            blocks.push(DacvaeEncoderBlock {
                res,
                alpha,
                down,
                ch_in: half,
            });
        }
        let alpha_out = load_alpha(file, "encoder.snake_out.alpha", ch)?;
        let conv_out = WnConv1d::load(
            file,
            "encoder.conv_out",
            ch,
            config.latent_dim,
            3,
            1,
            1,
            1,
            1,
        )?;
        Ok(DacvaeEncoder {
            conv_in,
            blocks,
            alpha_out,
            conv_out,
        })
    }

    fn load_decoder(config: &DacvaeConfig, file: &SafetensorsFile) -> Result<DacvaeDecoder> {
        let d = config.decoder_dim;
        let conv_in = DacvaeConv::load(file, "decoder.conv_in", config.latent_dim, d, 7, 1, 1)?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = d >> i;
            let out_dim = d >> (i + 1);
            let base = format!("decoder.blocks.{i}");
            let alpha = load_alpha(file, &format!("{base}.block_0.alpha"), in_dim)?;
            let up = DacvaeConvT::load(file, &format!("{base}.block_1"), stride)?;
            let ru = |name: &str, dilation: usize| -> Result<DacvaeRu> {
                Ok(DacvaeRu {
                    alpha1: load_alpha(file, &format!("{base}.{name}.act1.alpha"), out_dim)?,
                    conv1: DacvaeConv::load(
                        file,
                        &format!("{base}.{name}.conv1"),
                        out_dim,
                        out_dim,
                        7,
                        1,
                        dilation,
                    )?,
                    alpha2: load_alpha(file, &format!("{base}.{name}.act2.alpha"), out_dim)?,
                    conv2: WnConv1d::load(
                        file,
                        &format!("{base}.{name}.conv2"),
                        out_dim,
                        out_dim,
                        1,
                        1,
                        0,
                        1,
                        1,
                    )?,
                    ch: out_dim,
                })
            };
            // Main path blocks 4 (d1), 5 (d3), 8 (d9); the watermark
            // blocks 2/3/6/7/10/11 are unused by the batch contract.
            let res = [ru("block_4", 1)?, ru("block_5", 3)?, ru("block_8", 9)?];
            blocks.push(DacvaeDecoderBlock { alpha, up, res });
        }
        let final_dim = d >> config.decoder_rates.len();
        let alpha_out = load_alpha(file, "decoder.snake_out.alpha", final_dim)?;
        let conv_out = WnConv1d::load(file, "decoder.conv_out", final_dim, 1, 7, 1, 3, 1, 1)?;
        Ok(DacvaeDecoder {
            conv_in,
            blocks,
            alpha_out,
            conv_out,
        })
    }

    /// Right-pads to a multiple of the hop length.
    pub fn padded_len(&self, len: usize) -> usize {
        let hop = self.config.hop_length();
        len.div_ceil(hop) * hop
    }

    /// Encodes mono samples to the VAE mean latents
    /// `[codebook_dim, frames]`.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        let padded = self.padded_len(samples.len());
        let mut x = vec![0.0f32; padded];
        x[..samples.len()].copy_from_slice(samples);
        let mut h = self.encoder.conv_in.forward(&x);
        for block in &self.encoder.blocks {
            h = block.forward(&h);
        }
        let ch = self.encoder.conv_out.in_ch;
        let frames = h.len() / ch;
        snake(&mut h, &self.encoder.alpha_out, ch, frames);
        let z = self.encoder.conv_out.forward(&h);
        // in_proj then the mean half (first codebook_dim channels).
        let proj = self.in_proj.forward(&z);
        let frames = proj.len() / (self.config.codebook_dim * 2);
        Ok(proj[..self.config.codebook_dim * frames].to_vec())
    }

    /// Decodes `[codebook_dim, frames]` latents to mono samples.
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.codebook_dim;
        if latents.len() % dim != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "latents length {} not a multiple of codebook dim {dim}",
                    latents.len()
                ),
            });
        }
        let emb = self.out_proj.forward(latents);
        let mut h = self.decoder.conv_in.forward(&emb);
        for block in &self.decoder.blocks {
            h = block.forward(&h);
        }
        let ch = self.decoder.conv_out.in_ch;
        let frames = h.len() / ch;
        snake(&mut h, &self.decoder.alpha_out, ch, frames);
        let mut out = self.decoder.conv_out.forward(&h);
        for v in &mut out {
            *v = v.tanh();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
