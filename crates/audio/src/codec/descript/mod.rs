//! Descript Audio Codec (DAC).
//!
//! Reference: `mlx_audio/codec/models/descript/` (dac.py, base.py,
//! nn/layers.py, nn/quantize.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/descript).
//! An all-dense conv codec (no depthwise groups, no attention, no noise
//! blocks): Snake activations, residual units with dilations 1/3/9, a
//! stride-1 residual vector quantizer stack (up to 32 levels of
//! 1024x8 codebooks), and a transposed-conv decoder with tanh output.
//! Serves as the acoustic tokenizer / vocoder for voxcpm2, zonos2,
//! omnivoice, irodori, and chatterbox conversions upstream.
//!
//! Deviations from the pinned reference (see the family README):
//! - The `CodecMixin` chunked compress/decompress machinery matches no
//!   layers at this commit (all convs are custom WN-wrapped modules, so
//!   `isinstance(layer, nn.Conv1d)` is never true), which collapses the
//!   delay to 0 and the chunk hop to the window length. This port
//!   exposes the single-window compress/decompress semantics and the
//!   recorded delay value; the .dac file container is out of scope.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::dac::{load_alpha, DecoderBlock, EncoderBlock, ResidualUnit};
use crate::codec::vq::CodebookIndex;
use crate::codec::wnconv::{snake1d, WnConv1d, WnConvTranspose1d};
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Checkpoint geometry, one-to-one with the reference constructor.
#[derive(Debug, Clone)]
pub struct DacConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    /// Derived from `encoder_dim * 2^len(encoder_rates)` when null.
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    /// One codebook width per quantizer level.
    pub codebook_dims: Vec<usize>,
    pub sample_rate: u32,
}

impl DacConfig {
    /// Parses the checkpoint `config.json` with the reference
    /// constructor defaults; unknown keys are ignored.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u =
            |field: &str| -> Result<Option<u64>> { Ok(value.get(field).and_then(|v| v.as_u64())) };
        let usize_vec = |field: &str| -> Result<Option<Vec<usize>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list of integers".to_string(),
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
        let encoder_rates = usize_vec("encoder_rates")?.unwrap_or_else(|| vec![2, 4, 5, 8]);
        let encoder_dim = u("encoder_dim")?.unwrap_or(64) as usize;
        let latent_dim = match u("latent_dim")? {
            None | Some(0) => encoder_dim * (1usize << encoder_rates.len()),
            Some(n) => n as usize,
        };
        let n_codebooks = u("n_codebooks")?.unwrap_or(32) as usize;
        let codebook_dims = match value.get("codebook_dim") {
            Some(serde_json::Value::Array(arr)) => {
                let dims: Vec<usize> = arr
                    .iter()
                    .map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "codebook_dim".to_string(),
                        why: "expected an integer or a list of integers".to_string(),
                    })?;
                if dims.len() != n_codebooks {
                    return Err(SpeechError::BadConfig {
                        field: "codebook_dim".to_string(),
                        why: format!(
                            "config lists {} widths for {n_codebooks} codebooks",
                            dims.len()
                        ),
                    });
                }
                dims
            }
            _ => vec![u("codebook_dim")?.unwrap_or(8) as usize; n_codebooks],
        };
        Ok(DacConfig {
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim: u("decoder_dim")?.unwrap_or(1536) as usize,
            decoder_rates: usize_vec("decoder_rates")?.unwrap_or_else(|| vec![8, 5, 4, 2]),
            n_codebooks,
            codebook_size: u("codebook_size")?.unwrap_or(1024) as usize,
            codebook_dims,
            sample_rate: u("sample_rate")?.unwrap_or(44100) as u32,
        })
    }

    /// Total encoder downsampling.
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
}

fn load_residual_unit(
    file: &SafetensorsFile,
    prefix: &str,
    dim: usize,
    dilation: usize,
) -> Result<ResidualUnit> {
    let pad = ((7 - 1) * dilation) / 2;
    Ok(ResidualUnit {
        ch: dim,
        snake1: load_alpha(file, &format!("{prefix}.block.layers.0.alpha"), dim)?,
        conv1: WnConv1d::load(
            file,
            &format!("{prefix}.block.layers.1"),
            dim,
            dim,
            7,
            1,
            pad,
            dilation,
            1,
        )?,
        snake2: load_alpha(file, &format!("{prefix}.block.layers.2.alpha"), dim)?,
        conv2: WnConv1d::load(
            file,
            &format!("{prefix}.block.layers.3"),
            dim,
            dim,
            1,
            1,
            0,
            1,
            1,
        )?,
    })
}

struct Encoder {
    first: WnConv1d,
    blocks: Vec<EncoderBlock>,
    out_snake: Vec<f32>,
    out_conv: WnConv1d,
}

impl Encoder {
    fn forward(&self, samples: &[f32]) -> Vec<f32> {
        let mut h = self.first.forward(samples);
        let mut ch = self.first.out_ch;
        for block in &self.blocks {
            h = block.forward(&h);
            ch = block.down.out_ch;
        }
        let seq = h.len() / ch;
        snake1d(&mut h, &self.out_snake, ch, seq);
        self.out_conv.forward(&h)
    }
}

/// One stride-1 quantizer level (the descript quantizer has no pooling
/// and no per-level strides).
struct DacQuantizer {
    in_proj: WnConv1d,
    out_proj: WnConv1d,
    index: CodebookIndex,
}

impl DacQuantizer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input_dim: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let in_proj = WnConv1d::load(
            file,
            &format!("{prefix}.in_proj"),
            input_dim,
            codebook_dim,
            1,
            1,
            0,
            1,
            1,
        )?;
        let out_proj = WnConv1d::load(
            file,
            &format!("{prefix}.out_proj"),
            codebook_dim,
            input_dim,
            1,
            1,
            0,
            1,
            1,
        )?;
        let index = CodebookIndex::load(file, prefix, codebook_dim)?;
        Ok(DacQuantizer {
            in_proj,
            out_proj,
            index,
        })
    }

    /// Encode one level over `z [latent, frames]`.
    fn encode(&self, z: &[f32], latent: usize) -> (Vec<f32>, Vec<i32>) {
        let frames = z.len() / latent;
        let projected = self.in_proj.forward(z);
        let dim = self.index.dim();
        let mut codes = vec![0i32; frames];
        let mut quantized = vec![0.0f32; dim * frames];
        for f in 0..frames {
            let col: Vec<f32> = (0..dim).map(|d| projected[d * frames + f]).collect();
            let (idx, _) = self.index.nearest(&col);
            codes[f] = idx as i32;
            for (d, &v) in self.index.raw_row(idx).iter().enumerate() {
                quantized[d * frames + f] = v;
            }
        }
        (self.out_proj.forward(&quantized), codes)
    }

    /// Codes to quantized latents (the `from_codes` level path).
    fn decode(&self, codes: &[i32], _latent: usize) -> Vec<f32> {
        let frames = codes.len();
        let dim = self.index.dim();
        let mut quantized = vec![0.0f32; dim * frames];
        for (f, &code) in codes.iter().enumerate() {
            for (d, &v) in self.index.raw_row(code as usize).iter().enumerate() {
                quantized[d * frames + f] = v;
            }
        }
        self.out_proj.forward(&quantized)
    }
}

struct Decoder {
    pre: WnConv1d,
    blocks: Vec<DecoderBlock>,
    out_snake: Vec<f32>,
    out_conv: WnConv1d,
}

impl Decoder {
    fn forward(&self, z: &[f32]) -> Vec<f32> {
        let mut h = self.pre.forward(z);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        let ch = self.out_conv.in_ch;
        let seq = h.len() / ch;
        snake1d(&mut h, &self.out_snake, ch, seq);
        let mut h = self.out_conv.forward(&h);
        for v in &mut h {
            *v = v.tanh();
        }
        h
    }
}

/// Loaded Descript Audio Codec, batch-of-one.
pub struct Dac {
    pub config: DacConfig,
    encoder: Encoder,
    quantizers: Vec<DacQuantizer>,
    decoder: Decoder,
    /// The reference `CodecMixin.get_delay()` value; 0 at the pinned
    /// commit because its layer walk matches no custom WN convs.
    pub delay: usize,
}

impl Dac {
    /// Opens a checkpoint directory holding `config.json` plus
    /// `model.safetensors` (the `from_pretrained` layout).
    pub fn open(dir: &Path) -> Result<Dac> {
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
        let config = DacConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Dac::load(config, &file)
    }

    /// Loads the model from a parsed config and a safetensors file.
    pub fn load(config: DacConfig, file: &SafetensorsFile) -> Result<Dac> {
        if config.codebook_dims.len() != config.n_codebooks {
            return Err(SpeechError::BadConfig {
                field: "codebook_dim".to_string(),
                why: format!(
                    "{} widths for {} codebooks",
                    config.codebook_dims.len(),
                    config.n_codebooks
                ),
            });
        }
        if config.latent_dim != config.encoder_dim << config.encoder_rates.len() {
            return Err(SpeechError::BadConfig {
                field: "latent_dim".to_string(),
                why: format!(
                    "checkpoint latent {} contradicts encoder_dim {} with rates {:?}",
                    config.latent_dim, config.encoder_dim, config.encoder_rates
                ),
            });
        }
        let encoder = Self::load_encoder(&config, file)?;
        let mut quantizers = Vec::with_capacity(config.n_codebooks);
        for (i, &dim) in config.codebook_dims.iter().enumerate() {
            quantizers.push(DacQuantizer::load(
                file,
                &format!("quantizer.quantizers.{i}"),
                config.latent_dim,
                dim,
            )?);
        }
        let decoder = Self::load_decoder(&config, file)?;
        Ok(Dac {
            config,
            encoder,
            quantizers,
            decoder,
            // get_delay folds an empty layer list at the pinned commit
            // (all convs are custom WN modules the isinstance walk
            // misses), so the delay is 0.
            delay: 0,
        })
    }

    fn load_encoder(config: &DacConfig, file: &SafetensorsFile) -> Result<Encoder> {
        let first = WnConv1d::load(
            file,
            "encoder.block.layers.0",
            1,
            config.encoder_dim,
            7,
            1,
            3,
            1,
            1,
        )?;
        let mut ch = config.encoder_dim;
        let mut blocks = Vec::with_capacity(config.encoder_rates.len());
        for (i, &stride) in config.encoder_rates.iter().enumerate() {
            ch *= 2;
            let half = ch / 2;
            let base = format!("encoder.block.layers.{}", i + 1);
            let unit = |j: usize, dilation: usize| -> Result<ResidualUnit> {
                load_residual_unit(file, &format!("{base}.block.layers.{j}"), half, dilation)
            };
            let units = [unit(0, 1)?, unit(1, 3)?, unit(2, 9)?];
            let snake = load_alpha(file, &format!("{base}.block.layers.3.alpha"), half)?;
            let pad = stride.div_ceil(2);
            let down = WnConv1d::load(
                file,
                &format!("{base}.block.layers.4"),
                half,
                ch,
                2 * stride,
                stride,
                pad,
                1,
                1,
            )?;
            blocks.push(EncoderBlock {
                units,
                snake,
                down,
                ch: half,
            });
        }
        let final_index = config.encoder_rates.len() + 1;
        let out_snake = load_alpha(
            file,
            &format!("encoder.block.layers.{final_index}.alpha"),
            config.latent_dim,
        )?;
        let out_conv = WnConv1d::load(
            file,
            &format!("encoder.block.layers.{}", final_index + 1),
            config.latent_dim,
            config.latent_dim,
            3,
            1,
            1,
            1,
            1,
        )?;
        Ok(Encoder {
            first,
            blocks,
            out_snake,
            out_conv,
        })
    }

    fn load_decoder(config: &DacConfig, file: &SafetensorsFile) -> Result<Decoder> {
        let pre = WnConv1d::load(
            file,
            "decoder.model.layers.0",
            config.latent_dim,
            config.decoder_dim,
            7,
            1,
            3,
            1,
            1,
        )?;
        let mut blocks = Vec::with_capacity(config.decoder_rates.len());
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = config.decoder_dim >> i;
            let out_dim = config.decoder_dim >> (i + 1);
            let base = format!("decoder.model.layers.{}", i + 1);
            let snake = load_alpha(file, &format!("{base}.block.layers.0.alpha"), in_dim)?;
            let pad = stride.div_ceil(2);
            let up = WnConvTranspose1d::load_out_first(
                file,
                &format!("{base}.block.layers.1"),
                in_dim,
                out_dim,
                2 * stride,
                stride,
                pad,
            )?;
            let unit = |j: usize, dilation: usize| -> Result<ResidualUnit> {
                load_residual_unit(
                    file,
                    &format!("{base}.block.layers.{}", j + 2),
                    out_dim,
                    dilation,
                )
            };
            let units = [unit(0, 1)?, unit(1, 3)?, unit(2, 9)?];
            blocks.push(DecoderBlock { snake, up, units });
        }
        let final_index = config.decoder_rates.len() + 1;
        let out_dim = config.decoder_dim >> config.decoder_rates.len();
        let out_snake = load_alpha(
            file,
            &format!("decoder.model.layers.{final_index}.alpha"),
            out_dim,
        )?;
        let out_conv = WnConv1d::load(
            file,
            &format!("decoder.model.layers.{}", final_index + 1),
            out_dim,
            1,
            7,
            1,
            3,
            1,
            1,
        )?;
        Ok(Decoder {
            pre,
            blocks,
            out_snake,
            out_conv,
        })
    }

    /// Right pad applied by `preprocess`: input is padded to a multiple
    /// of `hop_length`.
    pub fn padded_len(&self, len: usize) -> usize {
        len.div_ceil(self.config.hop_length()) * self.config.hop_length()
    }

    /// Encodes mono samples into per-level code vectors (all levels).
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
        let (codes, _) = self.encode_partial(samples, self.config.n_codebooks)?;
        Ok(codes)
    }

    /// Encodes with at most `n_quantizers` levels (the reference
    /// `encode(..., n_quantizers)` truncation).
    pub fn encode_partial(
        &self,
        samples: &[f32],
        n_quantizers: usize,
    ) -> Result<(Vec<Vec<i32>>, Vec<f32>)> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        let padded = self.padded_len(samples.len());
        let mut x = vec![0.0f32; padded];
        x[..samples.len()].copy_from_slice(samples);
        let z = self.encoder.forward(&x);
        let latent = self.config.latent_dim;
        let mut residual = z;
        let mut z_q_sum = vec![0.0f32; residual.len()];
        let mut codes = Vec::with_capacity(self.quantizers.len());
        for q in self.quantizers.iter().take(n_quantizers) {
            let (z_q, level) = q.encode(&residual, latent);
            for (dst, src) in z_q_sum.iter_mut().zip(&z_q) {
                *dst += src;
            }
            for (r, s) in residual.iter_mut().zip(&z_q) {
                *r -= s;
            }
            codes.push(level);
        }
        Ok((codes, z_q_sum))
    }

    /// Decodes quantized latents `[latent, frames]` to mono samples.
    pub fn decode_latents(&self, z: &[f32]) -> Vec<f32> {
        self.decoder.forward(z)
    }

    /// Decodes code levels to mono samples (the `from_codes` path).
    pub fn decode_codes(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        if codes.len() > self.config.n_codebooks {
            return Err(SpeechError::Input {
                why: format!(
                    "{} code levels exceed the {} configured",
                    codes.len(),
                    self.config.n_codebooks
                ),
            });
        }
        let latent = self.config.latent_dim;
        let frames = codes.first().map(|c| c.len()).unwrap_or(0);
        for (level, level_codes) in codes.iter().enumerate() {
            if level_codes.len() != frames {
                return Err(SpeechError::Input {
                    why: format!(
                        "level {level} has {} codes, level 0 has {frames}",
                        level_codes.len()
                    ),
                });
            }
            for &code in level_codes {
                if code < 0 || code as usize >= self.config.codebook_size {
                    return Err(SpeechError::Input {
                        why: format!(
                            "level {level} code {code} outside codebook of {}",
                            self.config.codebook_size
                        ),
                    });
                }
            }
        }
        let mut z_q = vec![0.0f32; latent * frames];
        for (q, level_codes) in self.quantizers.iter().zip(codes) {
            let part = q.decode(level_codes, latent);
            for (dst, src) in z_q.iter_mut().zip(&part) {
                *dst += src;
            }
        }
        Ok(self.decoder.forward(&z_q))
    }

    /// The single-window compress: peak-normalize to `normalize_db`
    /// (reference default -16), encode, and return the codes with the
    /// recorded input loudness for decompress.
    pub fn compress(
        &self,
        samples: &[f32],
        normalize_db: Option<f32>,
    ) -> Result<(Vec<Vec<i32>>, f32)> {
        let normalize_db = normalize_db.unwrap_or(-16.0);
        let rms =
            (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32 + 1e-12).sqrt();
        let input_db = 20.0 * (rms / 1.0 + 1e-12).log10();
        let gain = 10f32.powf((normalize_db - input_db) / 20.0);
        let normalized: Vec<f32> = samples.iter().map(|v| v * gain).collect();
        let codes = self.encode(&normalized)?;
        Ok((codes, input_db))
    }

    /// The single-window decompress: decode codes and undo the
    /// compress gain.
    pub fn decompress(&self, codes: &[Vec<i32>], input_db: f32) -> Result<Vec<f32>> {
        let recons = self.decode_codes(codes)?;
        let gain = 10f32.powf((input_db - -16.0) / 20.0);
        Ok(recons.iter().map(|v| v * gain).collect())
    }
}

#[cfg(test)]
mod tests;
