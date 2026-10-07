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

use crate::codec::conv::{permute_okc_to_cok, wn_fold_in_place, ConvTranspose1d, WnFold};
use crate::codec::dac::{load_alpha, DecoderBlock, EncoderBlock, ResidualUnit};
use crate::codec::wnconv::{load_f32_shaped, snake1d, WnConv1d};
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

/// ConvTranspose1d with the descript-style out-first stored weight and
/// static symmetric padding `(stride + 1) // 2` (pad_mode "none", no
/// unpad). Weight norm is folded at load; the bias is required.
fn load_dacvae_convt(
    file: &SafetensorsFile,
    prefix: &str,
    stride: usize,
) -> Result<ConvTranspose1d> {
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
    // (out, K) in row-major order, which is the contiguous row order
    // once the weight is in the PyTorch [in, out, K] layout.
    let mut weight = permute_okc_to_cok(&v, out_ch, kernel, in_ch);
    wn_fold_in_place(&mut weight, &g, out_ch * kernel, WnFold::Div);
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: (stride + 1) / 2,
        output_padding: 0,
        groups: 1,
        weight,
        bias: Some(bias),
    })
}

/// Static-padding conv (pad_mode "none"): MLX symmetric padding
/// `(K - stride) * dilation // 2`.
fn load_dacvae_conv(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<WnConv1d> {
    let padding = (kernel - stride) * dilation / 2;
    WnConv1d::load(
        file, prefix, in_ch, out_ch, kernel, stride, padding, dilation, 1,
    )
}

struct DacvaeEncoder {
    conv_in: WnConv1d,
    blocks: Vec<EncoderBlock>,
    alpha_out: Vec<f32>,
    conv_out: WnConv1d,
}

struct DacvaeDecoder {
    conv_in: WnConv1d,
    blocks: Vec<DecoderBlock>,
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
        let conv_in = load_dacvae_conv(file, "encoder.conv_in", 1, d, 7, 1, 1)?;
        let mut blocks = Vec::new();
        let mut ch = d;
        for (i, &stride) in config.encoder_rates.iter().enumerate() {
            ch *= 2;
            let half = ch / 2;
            let base = format!("encoder.blocks.{i}");
            let ru = |name: &str, dilation: usize| -> Result<ResidualUnit> {
                Ok(ResidualUnit {
                    snake1: load_alpha(file, &format!("{base}.{name}.act1.alpha"), half)?,
                    conv1: load_dacvae_conv(
                        file,
                        &format!("{base}.{name}.conv1"),
                        half,
                        half,
                        7,
                        1,
                        dilation,
                    )?,
                    snake2: load_alpha(file, &format!("{base}.{name}.act2.alpha"), half)?,
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
            let units = [ru("res1", 1)?, ru("res2", 3)?, ru("res3", 9)?];
            let alpha = load_alpha(file, &format!("{base}.snake.alpha"), half)?;
            let down = load_dacvae_conv(
                file,
                &format!("{base}.conv"),
                half,
                ch,
                2 * stride,
                stride,
                1,
            )?;
            blocks.push(EncoderBlock {
                units,
                snake: alpha,
                down,
                ch: half,
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
        let conv_in = load_dacvae_conv(file, "decoder.conv_in", config.latent_dim, d, 7, 1, 1)?;
        let mut blocks = Vec::new();
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = d >> i;
            let out_dim = d >> (i + 1);
            let base = format!("decoder.blocks.{i}");
            let alpha = load_alpha(file, &format!("{base}.block_0.alpha"), in_dim)?;
            let up = load_dacvae_convt(file, &format!("{base}.block_1"), stride)?;
            let ru = |name: &str, dilation: usize| -> Result<ResidualUnit> {
                Ok(ResidualUnit {
                    snake1: load_alpha(file, &format!("{base}.{name}.act1.alpha"), out_dim)?,
                    conv1: load_dacvae_conv(
                        file,
                        &format!("{base}.{name}.conv1"),
                        out_dim,
                        out_dim,
                        7,
                        1,
                        dilation,
                    )?,
                    snake2: load_alpha(file, &format!("{base}.{name}.act2.alpha"), out_dim)?,
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
            let units = [ru("block_4", 1)?, ru("block_5", 3)?, ru("block_8", 9)?];
            blocks.push(DecoderBlock {
                snake: alpha,
                up,
                units,
            });
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
        snake1d(&mut h, &self.encoder.alpha_out, ch, frames);
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
        snake1d(&mut h, &self.decoder.alpha_out, ch, frames);
        let mut out = self.decoder.conv_out.forward(&h);
        for v in &mut out {
            *v = v.tanh();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
