//! SNAC: multi-scale neural audio codec (SoundStream / EnCodec lineage).
//!
//! Reference: `mlx_audio/codec/models/snac/` (snac.py, layers.py, vq.py,
//! attention.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/snac).
//! An encoder of weight-norm conv blocks with Snake activations and
//! doubling channels, a residual vector quantizer whose levels are
//! average-pooled by `vq_strides`, and a mirrored decoder with transposed
//! convs, optional per-channel noise terms, and a tanh output.
//!
//! Deviations from the pinned reference, both upstream defects at this
//! commit (see the family README for evidence):
//! - `LocalMHA` is refused at load. The reference's attention block
//!   expects channels-first input while the surrounding convs pass
//!   channels-last, and its `rotate_half` helper also changes rank, so
//!   every attention-enabled flow errors out; upstream's own test
//!   config and the deployed `mlx-community/snac_24khz` checkpoint use
//!   `attn_window_size: null`.
//! - The noise terms draw one standard normal per channel (not per
//!   frame) because the reference unpacks channels-last input as
//!   `B, C, T`. This port keeps that behavior and takes the draws from
//!   a caller-supplied source so decode stays deterministic.
//!
//! The deployed checkpoint layout is `mlx-community/snac_24khz`
//! (24 kHz, encoder rates [2, 4, 8, 8], three 4096x8 codebooks with
//! strides [4, 2, 1], noise on, depthwise on, f32 safetensors).

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{wn_fold_in_place, WnFold};
use crate::codec::dac::{self, load_alpha_mid, EncoderBlock, ResidualUnit};
use crate::codec::vq::CodebookIndex;
use crate::codec::wnconv::{load_f32_shaped, snake1d, WnConv1d, WnConvTranspose1d};
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Checkpoint geometry, one-to-one with the reference constructor.
#[derive(Debug, Clone)]
pub struct SnacConfig {
    pub sampling_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    /// Derived from `encoder_dim * 2^len(encoder_rates)` when the
    /// checkpoint config stores null.
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub attn_window_size: Option<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vq_strides: Vec<usize>,
    pub noise: bool,
    pub depthwise: bool,
}

impl SnacConfig {
    /// Parses the checkpoint `config.json`. Defaults mirror the
    /// reference constructor, so a config that omits fields still
    /// yields the reference geometry.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u = |field: &str| -> Result<Option<u64>> {
            Ok(value
                .get(field)
                .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64))))
        };
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
        let encoder_rates = usize_vec("encoder_rates")?.unwrap_or_else(|| vec![3, 3, 7, 7]);
        let encoder_dim = u("encoder_dim")?.unwrap_or(64) as usize;
        let latent_dim = match u("latent_dim")? {
            None | Some(0) => encoder_dim * (1usize << encoder_rates.len()),
            Some(n) => n as usize,
        };
        let decoder_rates = usize_vec("decoder_rates")?.unwrap_or_else(|| vec![7, 7, 3, 3]);
        let attn_window_size = match value.get("attn_window_size") {
            None => Some(32),
            Some(serde_json::Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| SpeechError::BadConfig {
                field: "attn_window_size".to_string(),
                why: "expected an integer or null".to_string(),
            })? as usize),
        };
        Ok(SnacConfig {
            sampling_rate: u("sampling_rate")?.unwrap_or(44100) as u32,
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim: u("decoder_dim")?.unwrap_or(1536) as usize,
            decoder_rates,
            attn_window_size,
            codebook_size: u("codebook_size")?.unwrap_or(4096) as usize,
            codebook_dim: u("codebook_dim")?.unwrap_or(8) as usize,
            vq_strides: usize_vec("vq_strides")?.unwrap_or_else(|| vec![8, 4, 2, 1]),
            noise: value.get("noise").and_then(|v| v.as_bool()).unwrap_or(true),
            depthwise: value
                .get("depthwise")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }

    /// Total encoder downsampling; one latent frame per `hop_length`
    /// samples.
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    /// Samples per codebook level after `from_codes` repeat: the frame
    /// count the decoder consumes.
    pub fn frames_per_level(&self, codes: &[&[i32]]) -> Result<usize> {
        if codes.len() != self.vq_strides.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {} code levels, got {}",
                    self.vq_strides.len(),
                    codes.len()
                ),
            });
        }
        let mut frames = None;
        for (level, (stride, level_codes)) in self.vq_strides.iter().zip(codes).enumerate() {
            let repeated = level_codes.len() * stride;
            match frames {
                None => frames = Some(repeated),
                Some(f) if f != repeated => {
                    return Err(SpeechError::Input {
                        why: format!(
                            "level {level} repeats to {repeated} frames, level 0 repeats to {f}"
                        ),
                    })
                }
                _ => {}
            }
        }
        frames.ok_or_else(|| SpeechError::Input {
            why: "empty code levels".to_string(),
        })
    }
}

fn load_residual_unit(
    file: &SafetensorsFile,
    prefix: &str,
    dim: usize,
    dilation: usize,
    groups: usize,
) -> Result<ResidualUnit> {
    let pad = ((7 - 1) * dilation) / 2;
    Ok(ResidualUnit {
        ch: dim,
        snake1: load_alpha_mid(file, &format!("{prefix}.block.layers.0.alpha"), dim)?,
        conv1: WnConv1d::load(
            file,
            &format!("{prefix}.block.layers.1"),
            dim,
            dim,
            7,
            1,
            pad,
            dilation,
            groups,
        )?,
        snake2: load_alpha_mid(file, &format!("{prefix}.block.layers.2.alpha"), dim)?,
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
    last: WnConv1d,
}

impl Encoder {
    fn forward(&self, samples: &[f32]) -> Vec<f32> {
        let mut h = self.first.forward(samples);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.last.forward(&h)
    }
}

struct NoiseBlock {
    /// Linear map stored `[dim, dim]` (a kernel-1 conv, weight norm
    /// folded, no bias).
    weight: Vec<f32>,
    dim: usize,
}

impl NoiseBlock {
    /// Adds `noise[c] * (weight @ x)[c, t]` per channel. The reference
    /// draws `dim` normals, one per channel, broadcast across frames.
    fn forward(&self, x: &mut [f32], noise: &mut dyn FnMut() -> f32) {
        let seq = x.len() / self.dim;
        let draws: Vec<f32> = (0..self.dim).map(|_| noise()).collect();
        let mut h = vec![0.0f32; x.len()];
        for t in 0..seq {
            for oc in 0..self.dim {
                let mut acc = 0.0f32;
                for ic in 0..self.dim {
                    acc += x[ic * seq + t] * self.weight[oc * self.dim + ic];
                }
                h[oc * seq + t] = acc;
            }
        }
        for (c, &n) in draws.iter().enumerate() {
            for t in 0..seq {
                x[c * seq + t] += n * h[c * seq + t];
            }
        }
    }
}

struct DecoderBlock {
    block: dac::DecoderBlock,
    noise: Option<NoiseBlock>,
}

impl DecoderBlock {
    fn forward(&self, x: &[f32], noise: &mut dyn FnMut() -> f32) -> Vec<f32> {
        self.block.forward_with(x, |mut h, _| {
            if let Some(nb) = &self.noise {
                nb.forward(&mut h, noise);
            }
            h
        })
    }
}

struct Decoder {
    pre: Vec<WnConv1d>,
    blocks: Vec<DecoderBlock>,
    out_snake: Vec<f32>,
    out_conv: WnConv1d,
}

impl Decoder {
    fn forward(&self, z: &[f32], noise: &mut dyn FnMut() -> f32) -> Vec<f32> {
        let mut h = z.to_vec();
        for conv in &self.pre {
            h = conv.forward(&h);
        }
        for block in &self.blocks {
            h = block.forward(&h, noise);
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

/// One quantizer level: channel average pooling by stride, factorized
/// projections, L2-normalized nearest-code lookup.
struct Quantizer {
    in_proj: WnConv1d,
    out_proj: WnConv1d,
    index: CodebookIndex,
    stride: usize,
}

impl Quantizer {
    fn load(file: &SafetensorsFile, prefix: &str, input_dim: usize, stride: usize) -> Result<Self> {
        let codebook_dim = if file.contains_tensor(&format!("{prefix}.in_proj.bias")) {
            file.descriptor(&format!("{prefix}.in_proj.bias"))
                .map(|d| d.shape[0])
                .unwrap_or(0)
        } else {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.in_proj.bias"),
                why: "missing; codebook_dim cannot be derived".to_string(),
            });
        };
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
        Ok(Quantizer {
            in_proj,
            out_proj,
            index,
            stride,
        })
    }

    /// Encode one level: returns `(quantized [latent, frames'], codes)`.
    /// `z` is `[latent, frames]`.
    fn encode(&self, z: &[f32], latent: usize) -> (Vec<f32>, Vec<i32>) {
        // Stride path: per-channel box average pool, kernel == stride.
        let pooled: Vec<f32> = if self.stride > 1 {
            let frames = z.len() / latent;
            let out_frames = frames / self.stride;
            let recip = 1.0 / self.stride as f32;
            let mut out = vec![0.0f32; latent * out_frames];
            for c in 0..latent {
                for f in 0..out_frames {
                    let mut acc = 0.0f32;
                    for j in 0..self.stride {
                        acc += z[c * frames + f * self.stride + j] * recip;
                    }
                    out[c * out_frames + f] = acc;
                }
            }
            out
        } else {
            z.to_vec()
        };
        let frames = pooled.len() / latent;
        // in_proj: kernel-1 conv, [latent, frames] -> [codebook_dim, frames].
        let projected = self.in_proj.forward(&pooled);
        let dim = self.index.dim();
        let mut codes = vec![0i32; frames];
        // Raw codebook rows live in codebook_dim space; out_proj maps
        // them back to latent.
        let mut quantized = vec![0.0f32; dim * frames];
        for f in 0..frames {
            let col: Vec<f32> = (0..dim).map(|d| projected[d * frames + f]).collect();
            let (idx, _) = self.index.nearest(&col);
            codes[f] = idx as i32;
            for (d, &v) in self.index.raw_row(idx).iter().enumerate() {
                quantized[d * frames + f] = v;
            }
        }
        // out_proj back to latent, then repeat_interleave by stride.
        let expanded = self.out_proj.forward(&quantized);
        (self.repeat(&expanded, latent), codes)
    }

    /// Codes to quantized latents (the `from_codes` level path).
    fn decode(&self, codes: &[i32], latent: usize) -> Vec<f32> {
        let frames = codes.len();
        let _ = latent;
        let dim = self.index.dim();
        let mut quantized = vec![0.0f32; dim * frames];
        for (f, &code) in codes.iter().enumerate() {
            let idx = code as usize;
            for (d, &v) in self.index.raw_row(idx).iter().enumerate() {
                quantized[d * frames + f] = v;
            }
        }
        let expanded = self.out_proj.forward(&quantized);
        self.repeat(&expanded, latent)
    }

    fn repeat(&self, x: &[f32], ch: usize) -> Vec<f32> {
        if self.stride <= 1 {
            return x.to_vec();
        }
        let frames = x.len() / ch;
        let mut out = vec![0.0f32; ch * frames * self.stride];
        for c in 0..ch {
            for f in 0..frames {
                for j in 0..self.stride {
                    out[c * frames * self.stride + f * self.stride + j] = x[c * frames + f];
                }
            }
        }
        out
    }
}

/// Loaded SNAC codec, batch-of-one.
pub struct Snac {
    pub config: SnacConfig,
    encoder: Encoder,
    quantizers: Vec<Quantizer>,
    decoder: Decoder,
}

impl Snac {
    /// Opens a checkpoint directory holding `config.json` plus
    /// `model.safetensors` (the `from_pretrained` layout).
    pub fn open(dir: &Path) -> Result<Snac> {
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
        let config = SnacConfig::from_json(&value)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Snac::load(config, &file)
    }

    /// Loads the model from a parsed config and a safetensors file.
    pub fn load(config: SnacConfig, file: &SafetensorsFile) -> Result<Snac> {
        if config.attn_window_size.is_some() {
            return Err(SpeechError::Unsupported {
                why: "attn_window_size is enabled; the pinned reference's \
                      LocalMHA is shape-broken in the encoder/decoder flow and \
                      the deployed SNAC checkpoints disable it (null). Refusing \
                      rather than silently decoding with the wrong architecture."
                    .to_string(),
            });
        }
        if config.vq_strides.is_empty() {
            return Err(SpeechError::BadConfig {
                field: "vq_strides".to_string(),
                why: "at least one level is required".to_string(),
            });
        }
        if config.vq_strides.contains(&0) {
            return Err(SpeechError::BadConfig {
                field: "vq_strides".to_string(),
                why: "strides must be positive".to_string(),
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
        let mut quantizers = Vec::with_capacity(config.vq_strides.len());
        for (i, &stride) in config.vq_strides.iter().enumerate() {
            quantizers.push(Quantizer::load(
                file,
                &format!("quantizer.quantizers.{i}"),
                config.latent_dim,
                stride,
            )?);
        }
        let decoder = Self::load_decoder(&config, file)?;
        Ok(Snac {
            config,
            encoder,
            quantizers,
            decoder,
        })
    }

    fn load_encoder(config: &SnacConfig, file: &SafetensorsFile) -> Result<Encoder> {
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
            let in_dim = ch / 2;
            let groups = if config.depthwise { in_dim } else { 1 };
            let base = format!("encoder.block.layers.{}", i + 1);
            let unit = |j: usize, dilation: usize| -> Result<ResidualUnit> {
                load_residual_unit(
                    file,
                    &format!("{base}.block.layers.{j}"),
                    in_dim,
                    dilation,
                    groups,
                )
            };
            let units = [unit(0, 1)?, unit(1, 3)?, unit(2, 9)?];
            let snake = load_alpha_mid(file, &format!("{base}.block.layers.3.alpha"), in_dim)?;
            let pad = stride.div_ceil(2);
            let down = WnConv1d::load(
                file,
                &format!("{base}.block.layers.4"),
                in_dim,
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
                ch: in_dim,
            });
        }
        let groups = if config.depthwise {
            config.latent_dim
        } else {
            1
        };
        let last = WnConv1d::load(
            file,
            &format!("encoder.block.layers.{}", config.encoder_rates.len() + 1),
            config.latent_dim,
            config.latent_dim,
            7,
            1,
            3,
            1,
            groups,
        )?;
        Ok(Encoder {
            first,
            blocks,
            last,
        })
    }

    fn load_decoder(config: &SnacConfig, file: &SafetensorsFile) -> Result<Decoder> {
        let mut pre = Vec::new();
        let mut next_layer = 0usize;
        if config.depthwise {
            pre.push(WnConv1d::load(
                file,
                "decoder.model.layers.0",
                config.latent_dim,
                config.latent_dim,
                7,
                1,
                3,
                1,
                config.latent_dim,
            )?);
            pre.push(WnConv1d::load(
                file,
                "decoder.model.layers.1",
                config.latent_dim,
                config.decoder_dim,
                1,
                1,
                0,
                1,
                1,
            )?);
            next_layer = 2;
        } else {
            pre.push(WnConv1d::load(
                file,
                "decoder.model.layers.0",
                config.latent_dim,
                config.decoder_dim,
                7,
                1,
                3,
                1,
                1,
            )?);
            next_layer += 1;
        }
        let mut blocks = Vec::with_capacity(config.decoder_rates.len());
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = config.decoder_dim >> i;
            let out_dim = config.decoder_dim >> (i + 1);
            let base = format!("decoder.model.layers.{}", next_layer + i);
            let snake = load_alpha_mid(file, &format!("{base}.block.layers.0.alpha"), in_dim)?;
            let pad = stride.div_ceil(2);
            let up = WnConvTranspose1d::load(
                file,
                &format!("{base}.block.layers.1"),
                in_dim,
                out_dim,
                2 * stride,
                stride,
                pad,
            )?;
            let mut cursor = 2;
            let noise = if config.noise {
                let nb_prefix = format!("{base}.block.layers.{cursor}.linear");
                let dim = out_dim;
                let mut weight =
                    load_f32_shaped(file, &format!("{nb_prefix}.weight_v"), &[dim, 1, dim])?;
                let g = load_f32_shaped(file, &format!("{nb_prefix}.weight_g"), &[dim, 1, 1])?;
                // Weight-norm folded for the kernel-1 linear map.
                wn_fold_in_place(&mut weight, &g, dim, WnFold::Div);
                cursor += 1;
                Some(NoiseBlock { weight, dim })
            } else {
                None
            };
            let groups = if config.depthwise { out_dim } else { 1 };
            let unit = |j: usize, dilation: usize| -> Result<ResidualUnit> {
                load_residual_unit(
                    file,
                    &format!("{base}.block.layers.{}", cursor + j),
                    out_dim,
                    dilation,
                    groups,
                )
            };
            let units = [unit(0, 1)?, unit(1, 3)?, unit(2, 9)?];
            blocks.push(DecoderBlock {
                block: dac::DecoderBlock { snake, up, units },
                noise,
            });
        }
        let final_index = next_layer + config.decoder_rates.len();
        let out_dim = config.decoder_dim >> config.decoder_rates.len();
        let out_snake = load_alpha_mid(
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
    /// of `hop_length * lcm(vq_strides)`.
    pub fn padded_len(&self, len: usize) -> usize {
        let mut lcm = self.config.vq_strides[0];
        for &s in &self.config.vq_strides[1..] {
            lcm = lcm * s / gcd(lcm, s);
        }
        let pad_to = self.config.hop_length() * lcm;
        len.div_ceil(pad_to) * pad_to
    }

    /// Encodes mono samples into per-level code vectors.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
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
        let mut codes = Vec::with_capacity(self.quantizers.len());
        for q in &self.quantizers {
            let (z_q, level) = q.encode(&residual, latent);
            let frames = z_q.len() / latent;
            for c in 0..latent {
                for f in 0..frames {
                    residual[c * frames + f] -= z_q[c * frames + f];
                }
            }
            codes.push(level);
        }
        Ok(codes)
    }

    /// Decodes code levels to mono samples. Refuses checkpoints whose
    /// config enables the noise terms (the deployed 24 kHz checkpoint
    /// does); use [`Snac::decode_with_noise`].
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        if self.config.noise {
            return Err(SpeechError::Unsupported {
                why: "checkpoint enables the noise terms; call \
                      decode_with_noise with a normal source (or || 0.0 for \
                      the deterministic zero-noise backbone)"
                    .to_string(),
            });
        }
        let refs: Vec<&[i32]> = codes.iter().map(|c| c.as_slice()).collect();
        self.decode_impl(&refs, &mut || 0.0)
    }

    /// Decodes with a standard-normal source for the noise terms. The
    /// source is consumed channel-major per decoder block in order.
    /// Pass `|| 0.0` for the deterministic zero-noise backbone.
    pub fn decode_with_noise(
        &self,
        codes: &[Vec<i32>],
        noise: &mut dyn FnMut() -> f32,
    ) -> Result<Vec<f32>> {
        let refs: Vec<&[i32]> = codes.iter().map(|c| c.as_slice()).collect();
        self.decode_impl(&refs, noise)
    }

    fn decode_impl(&self, codes: &[&[i32]], noise: &mut dyn FnMut() -> f32) -> Result<Vec<f32>> {
        let frames = self.config.frames_per_level(codes)?;
        for (level, level_codes) in codes.iter().enumerate() {
            for &code in *level_codes {
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
        let latent = self.config.latent_dim;
        let mut z_q = vec![0.0f32; latent * frames];
        for (q, level_codes) in self.quantizers.iter().zip(codes) {
            let part = q.decode(level_codes, latent);
            for (dst, src) in z_q.iter_mut().zip(&part) {
                *dst += src;
            }
        }
        let audio = self.decoder.forward(&z_q, noise);
        Ok(audio)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests;
