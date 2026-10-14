//! The causal-convolutional AudioVAE vocoder (encoder and decoder).
//!
//! Port of `mlx_audio/tts/models/voxcpm/audio_vae.py` at the pinned
//! mlx-audio commit: a DAC-shaped fully convolutional VAE with Snake
//! activations, residual units at dilations 1/3/9, depthwise residual
//! convolutions (`depthwise=True` is hardcoded in the reference
//! constructor), left-padded causal convolutions, right-trimmed
//! transposed convolutions, and a tanh output. The noise block exists
//! in the reference file but `use_noise_block=False` is hardcoded in
//! `AudioVAE.__init__`, so it is not ported; a checkpoint that
//! carries `noise.` tensors under a decoder block is refused.
//!
//! Layouts follow the codec convention: activations are channel-major
//! `[ch, seq]` and weights PyTorch-layout `[out, in/groups, K]`
//! (transposed conv `[in, out/groups, K]`), converted from the MLX
//! checkpoint layout `[out, K, in]` at load time. The reference
//! `CausalConv1d` left-pads `2 * padding` and runs a valid conv, which
//! keeps the sequence length; the transposed conv runs zero-padded and
//! trims `2 * padding - output_padding` from the right.

use crate::codec::conv::{Conv1d, ConvTranspose1d};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

/// Finds a tensor across the opened checkpoint shards; returns the file
/// and the tensor shape.
pub(crate) fn locate<'a>(
    files: &'a [SafetensorsFile],
    name: &str,
) -> Option<(&'a SafetensorsFile, Vec<usize>)> {
    files
        .iter()
        .find_map(|file| file.descriptor(name).map(|d| (file, d.shape.clone())))
}

pub(crate) fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing from checkpoint".into(),
    }
}

/// The `AudioVAEConfig` dataclass.
#[derive(Debug, Clone)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sample_rate: u32,
}

impl AudioVaeConfig {
    pub fn from_json(value: &serde_json::Value) -> Self {
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
        Self {
            encoder_dim: value
                .get("encoder_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(64),
            encoder_rates: rates("encoder_rates", &[2, 3, 6, 7, 7]),
            latent_dim: value
                .get("latent_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(64),
            decoder_dim: value
                .get("decoder_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(2048),
            decoder_rates: rates("decoder_rates", &[7, 7, 6, 3, 2]),
            sample_rate: value
                .get("sample_rate")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u32)
                .unwrap_or(44100),
        }
    }

    /// `hop_length = prod(encoder_rates)`.
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
}

/// Snake activation `x + sin(alpha * x)^2 / (alpha + 1e-9)` over
/// channel-major activations.
pub(crate) fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    for ch in 0..channels {
        let a = alpha[ch];
        let recip = 1.0 / (a + 1e-9);
        for f in 0..frames {
            let v = &mut x[ch * frames + f];
            *v += recip * (a * *v).sin().powi(2);
        }
    }
}

/// Loads a conv weight as PyTorch-layout `[out, in, K]` (transposed
/// conv: `[in, out, K]`; the arguments below are relabeled
/// accordingly). Accepts the MLX `[out, K, in]` stored layout, the
/// already-PyTorch layout, or an unfused `weight_g` / `weight_v` pair,
/// which folds exactly like the reference sanitize
/// (`g * (v / (||v|| + 1e-9))`, the norm over each output channel).
pub(crate) fn conv_weight(
    files: &[SafetensorsFile],
    prefix: &str,
    out: usize,
    in_g: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let total = out * in_g * kernel;
    let weight_name = format!("{prefix}.weight");
    let (data, stored_shape) = match locate(files, &weight_name) {
        Some((file, shape)) => {
            let data = file.load_as_f32(&weight_name).map_err(SpeechError::from)?;
            (data, shape)
        }
        None => {
            // Unfused weight normalization pair; the fold is layout
            // independent (the norm is over each output channel), so
            // the v descriptor decides the layout below.
            let g_name = format!("{prefix}.weight_g");
            let v_name = format!("{prefix}.weight_v");
            let (g_file, g_shape) = locate(files, &g_name).ok_or_else(|| missing(&weight_name))?;
            let (v_file, v_shape) = locate(files, &v_name).ok_or_else(|| missing(&weight_name))?;
            if g_shape != vec![out, 1, 1] {
                return Err(SpeechError::Tensor {
                    name: g_name,
                    why: format!("expected per-channel scale shape [{out}, 1, 1], got {g_shape:?}"),
                });
            }
            let g = g_file.load_as_f32(&g_name).map_err(SpeechError::from)?;
            let v = v_file.load_as_f32(&v_name).map_err(SpeechError::from)?;
            if v.len() != total {
                return Err(SpeechError::Tensor {
                    name: v_name,
                    why: format!("expected {total} values, got {}", v.len()),
                });
            }
            let mut fused = vec![0.0f32; total];
            let per_channel = total / out;
            let mut sq = vec![0.0f32; out];
            for (channel, chunk) in v.chunks(per_channel).enumerate() {
                let sum: f32 = chunk.iter().map(|value| value * value).sum();
                sq[channel] = sum;
            }
            for (channel, chunk) in v.chunks(per_channel).enumerate() {
                let norm = sq[channel].sqrt() + 1e-9;
                let scale = g[channel];
                for (index, value) in chunk.iter().enumerate() {
                    fused[channel * per_channel + index] = scale * (value / norm);
                }
            }
            (fused, v_shape)
        }
    };
    if data.len() != total {
        return Err(SpeechError::Tensor {
            name: weight_name,
            why: format!("expected {total} values, got {}", data.len()),
        });
    }
    // The stored shape decides the layout; the ambiguous square
    // in_g == kernel case follows the descriptor order, which mirrors
    // the reference (equal shapes are used as-is, i.e. the MLX
    // layout).
    let square = in_g == kernel;
    let mut weight = vec![0.0f32; total];
    match stored_shape.as_slice() {
        // MLX layout [out, K, in]: rotate to [out, in, K].
        [o, k, i] if *o == out && *k == kernel && *i == in_g && !square => {
            for oc in 0..out {
                for kx in 0..kernel {
                    for ix in 0..in_g {
                        weight[(oc * in_g + ix) * kernel + kx] =
                            data[(oc * kernel + kx) * in_g + ix];
                    }
                }
            }
        }
        // PyTorch layout [out, in, K].
        [o, i, k] if *o == out && *i == in_g && *k == kernel => {
            weight.copy_from_slice(&data);
        }
        other => {
            return Err(SpeechError::Tensor {
                name: weight_name,
                why: format!("expected [{out}, {kernel}, {in_g}] or its transpose, got {other:?}"),
            })
        }
    }
    Ok(weight)
}

pub(crate) fn bias(
    files: &[SafetensorsFile],
    prefix: &str,
    out: usize,
) -> Result<Option<Vec<f32>>> {
    let name = format!("{prefix}.bias");
    match locate(files, &name) {
        Some((file, shape)) => {
            if shape != vec![out] {
                return Err(SpeechError::Tensor {
                    name,
                    why: format!("expected [{out}], got {shape:?}"),
                });
            }
            Ok(Some(file.load_as_f32(&name).map_err(SpeechError::from)?))
        }
        None => Ok(None),
    }
}

/// Loads a causal `CausalConv1d` from a fused checkpoint weight. The
/// `padding` argument is the reference constructor value; the forward
/// left-pads twice that amount, so the stored kernel runs with zero
/// padding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_conv(
    files: &[SafetensorsFile],
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Conv1d> {
    let weight = conv_weight(files, prefix, out_ch, in_ch / groups, kernel)?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: 0,
        dilation,
        groups,
        weight,
        bias: bias(files, prefix, out_ch)?,
    })
}

/// Loads a `CausalTransposeConv1d`; the forward runs zero-padded with
/// zero output padding and trims the tail.
///
/// The MLX `ConvTranspose1d` parameter is stored `[out_ch, K, in_ch]`
/// (confirmed on the fixture: decoder block 0 maps `in 32 -> out 16`
/// with `K 8` and the tensor is `[16, 8, 32]`); a PyTorch-layout
/// `[in_ch, out_ch, K]` tensor or an unfused `weight_g` / `weight_v`
/// pair (`g` `[in, 1, 1]`) is accepted and normalized to the crate
/// layout `[in_ch, out_ch, K]`.
pub(crate) fn load_conv_transpose(
    files: &[SafetensorsFile],
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
) -> Result<ConvTranspose1d> {
    let total = in_ch * out_ch * kernel;
    let weight_name = format!("{prefix}.weight");
    let (data, stored_shape) = match locate(files, &weight_name) {
        Some((file, shape)) => {
            let data = file.load_as_f32(&weight_name).map_err(SpeechError::from)?;
            (data, shape)
        }
        None => {
            let g_name = format!("{prefix}.weight_g");
            let v_name = format!("{prefix}.weight_v");
            let (g_file, g_shape) = locate(files, &g_name).ok_or_else(|| missing(&weight_name))?;
            let (v_file, v_shape) = locate(files, &v_name).ok_or_else(|| missing(&weight_name))?;
            if g_shape != vec![in_ch, 1, 1] {
                return Err(SpeechError::Tensor {
                    name: g_name,
                    why: format!(
                        "expected per-channel scale shape [{in_ch}, 1, 1], got {g_shape:?}"
                    ),
                });
            }
            let g = g_file.load_as_f32(&g_name).map_err(SpeechError::from)?;
            let v = v_file.load_as_f32(&v_name).map_err(SpeechError::from)?;
            if v.len() != total {
                return Err(SpeechError::Tensor {
                    name: v_name,
                    why: format!("expected {total} values, got {}", v.len()),
                });
            }
            let mut fused = vec![0.0f32; total];
            let per_channel = total / in_ch;
            let mut sq = vec![0.0f32; in_ch];
            for (channel, chunk) in v.chunks(per_channel).enumerate() {
                let sum: f32 = chunk.iter().map(|value| value * value).sum();
                sq[channel] = sum;
            }
            for (channel, chunk) in v.chunks(per_channel).enumerate() {
                let norm = sq[channel].sqrt() + 1e-9;
                let scale = g[channel];
                for (index, value) in chunk.iter().enumerate() {
                    fused[channel * per_channel + index] = scale * (value / norm);
                }
            }
            (fused, v_shape)
        }
    };
    if data.len() != total {
        return Err(SpeechError::Tensor {
            name: weight_name,
            why: format!("expected {total} values, got {}", data.len()),
        });
    }
    let mut weight = vec![0.0f32; total];
    match stored_shape.as_slice() {
        // MLX layout [out, K, in].
        [o, k, i] if *o == out_ch && *k == kernel && *i == in_ch => {
            for ic in 0..in_ch {
                for oc in 0..out_ch {
                    for kx in 0..kernel {
                        weight[(ic * out_ch + oc) * kernel + kx] =
                            data[(oc * kernel + kx) * in_ch + ic];
                    }
                }
            }
        }
        // PyTorch layout [in, out, K].
        [i, o, k] if *i == in_ch && *o == out_ch && *k == kernel => {
            weight.copy_from_slice(&data);
        }
        other => {
            return Err(SpeechError::Tensor {
                name: weight_name,
                why: format!(
                    "expected [{out_ch}, {kernel}, {in_ch}] or [{in_ch}, {out_ch}, {kernel}], got {other:?}"
                ),
            })
        }
    }
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: 0,
        output_padding: 0,
        groups: 1,
        weight,
        bias: bias(files, prefix, out_ch)?,
    })
}

pub(crate) fn alpha(files: &[SafetensorsFile], prefix: &str, channels: usize) -> Result<Vec<f32>> {
    let name = format!("{prefix}.alpha");
    let (file, shape) = locate(files, &name).ok_or_else(|| missing(&name))?;
    if shape != vec![1, 1, channels] {
        return Err(SpeechError::Tensor {
            name,
            why: format!("expected [1, 1, {channels}], got {shape:?}"),
        });
    }
    file.load_as_f32(&name).map_err(SpeechError::from)
}

/// Left-pads `[in_ch, seq]` by `pad` frames (the reference pads
/// `2 * padding` because its constructor value means the symmetric
/// amount).
pub(crate) fn left_pad(x: &[f32], channels: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / channels;
    let mut out = vec![0.0f32; channels * (seq + pad)];
    for ch in 0..channels {
        out[ch * (seq + pad) + pad..ch * (seq + pad) + pad + seq]
            .copy_from_slice(&x[ch * seq..(ch + 1) * seq]);
    }
    out
}

/// A causal convolution: left-pad then valid conv.
pub(crate) struct CausalConv {
    pub(crate) conv: Conv1d,
    pub(crate) pad: usize,
}

impl CausalConv {
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        if self.pad > 0 {
            let padded = left_pad(x, self.conv.in_ch, self.pad);
            self.conv.forward(&padded)
        } else {
            self.conv.forward(x)
        }
    }
}

/// A causal transposed conv: run, then trim `2 * padding -
/// output_padding` from the right.
pub(crate) struct CausalConvTranspose {
    pub(crate) conv: ConvTranspose1d,
    pub(crate) trim: usize,
}

impl CausalConvTranspose {
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.conv.forward(x);
        if self.trim > 0 {
            let out_ch = self.conv.out_ch;
            let seq = out.len() / out_ch;
            let kept = seq - self.trim;
            for ch in 0..out_ch {
                out.copy_within(ch * seq..ch * seq + kept, ch * kept);
            }
            out.truncate(kept * out_ch);
        }
        out
    }
}

pub(crate) struct ResidualUnit {
    snake1_alpha: Vec<f32>,
    conv1: CausalConv,
    snake2_alpha: Vec<f32>,
    conv2: CausalConv,
    dim: usize,
}

impl ResidualUnit {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        dim: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1_alpha: alpha(files, &format!("{prefix}.snake1"), dim)?,
            conv1: CausalConv {
                conv: load_conv(
                    files,
                    &format!("{prefix}.conv1"),
                    dim,
                    dim,
                    7,
                    1,
                    dilation,
                    groups,
                )?,
                pad: pad * 2,
            },
            snake2_alpha: alpha(files, &format!("{prefix}.snake2"), dim)?,
            conv2: CausalConv {
                conv: load_conv(files, &format!("{prefix}.conv2"), dim, dim, 1, 1, 1, 1)?,
                pad: 0,
            },
            dim,
        })
    }

    pub(crate) fn forward(&self, x: &mut [f32]) {
        let frames = x.len() / self.dim;
        let mut h = x.to_vec();
        snake(&mut h, &self.snake1_alpha, self.dim, frames);
        h = self.conv1.forward(&h);
        snake(&mut h, &self.snake2_alpha, self.dim, frames);
        h = self.conv2.forward(&h);
        for (value, residual) in x.iter_mut().zip(&h) {
            *value += residual;
        }
    }
}

pub(crate) struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake_alpha: Vec<f32>,
    conv: CausalConv,
    input_dim: usize,
}

impl EncoderBlock {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            res1: ResidualUnit::load(files, &format!("{prefix}.res1"), input_dim, 1, groups)?,
            res2: ResidualUnit::load(files, &format!("{prefix}.res2"), input_dim, 3, groups)?,
            res3: ResidualUnit::load(files, &format!("{prefix}.res3"), input_dim, 9, groups)?,
            snake_alpha: alpha(files, &format!("{prefix}.snake"), input_dim)?,
            conv: CausalConv {
                conv: load_conv(
                    files,
                    &format!("{prefix}.conv"),
                    input_dim,
                    output_dim,
                    2 * stride,
                    stride,
                    1,
                    1,
                )?,
                pad: 2 * stride.div_ceil(2),
            },
            input_dim,
        })
    }

    fn forward(&self, x: &mut [f32]) -> Vec<f32> {
        let frames = x.len() / self.input_dim;
        self.res1.forward(x);
        self.res2.forward(x);
        self.res3.forward(x);
        snake(x, &self.snake_alpha, self.input_dim, frames);
        self.conv.forward(x)
    }
}

struct DecoderBlock {
    snake_alpha: Vec<f32>,
    conv_t: CausalConvTranspose,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    input_dim: usize,
}

impl DecoderBlock {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let conv_t = load_conv_transpose(
            files,
            &format!("{prefix}.conv_t"),
            input_dim,
            output_dim,
            2 * stride,
            stride,
        )?;
        if locate(files, &format!("{prefix}.noise.linear.weight")).is_some() {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "{prefix} carries a noise block; the reference AudioVAE constructs the decoder with use_noise_block=False"
                ),
            });
        }
        Ok(Self {
            snake_alpha: alpha(files, &format!("{prefix}.snake"), input_dim)?,
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

    fn forward(&self, x: &mut [f32]) -> Vec<f32> {
        let frames = x.len() / self.input_dim;
        snake(x, &self.snake_alpha, self.input_dim, frames);
        let mut h = self.conv_t.forward(x);
        self.res1.forward(&mut h);
        self.res2.forward(&mut h);
        self.res3.forward(&mut h);
        h
    }
}

/// The `CausalEncoder`: conv_in, one block per stride (each doubling
/// the channels), fc_mu. Depthwise groups ride the residual conv1
/// layers.
pub(crate) struct CausalEncoder {
    conv_in: CausalConv,
    blocks: Vec<EncoderBlock>,
    fc_mu: CausalConv,
}

impl CausalEncoder {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        d_model: usize,
        latent_dim: usize,
        strides: &[usize],
    ) -> Result<Self> {
        let conv_in = CausalConv {
            conv: load_conv(files, &format!("{prefix}.conv_in"), 1, d_model, 7, 1, 1, 1)?,
            pad: 6,
        };
        let mut blocks = Vec::new();
        let mut curr = d_model;
        for (index, stride) in strides.iter().enumerate() {
            let next = curr * 2;
            blocks.push(EncoderBlock::load(
                files,
                &format!("{prefix}.blocks.layers.{index}"),
                curr,
                next,
                *stride,
                next / 2,
            )?);
            curr = next;
        }
        let fc_mu = CausalConv {
            conv: load_conv(
                files,
                &format!("{prefix}.fc_mu"),
                curr,
                latent_dim,
                3,
                1,
                1,
                1,
            )?,
            pad: 2,
        };
        Ok(Self {
            conv_in,
            blocks,
            fc_mu,
        })
    }

    /// `[1, seq]` waveform to `[latent_dim, frames]` latents.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.conv_in.forward(x);
        for block in &self.blocks {
            h = block.forward(&mut h);
        }
        self.fc_mu.forward(&h)
    }
}

struct CausalDecoder {
    conv_in_dense: CausalConv,
    conv_in_point: CausalConv,
    blocks: Vec<DecoderBlock>,
    snake_out_alpha: Vec<f32>,
    conv_out: CausalConv,
}

impl CausalDecoder {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        input_channel: usize,
        channels: usize,
        rates: &[usize],
    ) -> Result<Self> {
        // depthwise conv_in: depthwise 7-tap then a 1x1 mixer, stored
        // as `conv_in.layers.0` / `conv_in.layers.1`.
        let conv_in_dense = CausalConv {
            conv: load_conv(
                files,
                &format!("{prefix}.conv_in.layers.0"),
                input_channel,
                input_channel,
                7,
                1,
                1,
                input_channel,
            )?,
            pad: 6,
        };
        let conv_in_point = CausalConv {
            conv: load_conv(
                files,
                &format!("{prefix}.conv_in.layers.1"),
                input_channel,
                channels,
                1,
                1,
                1,
                1,
            )?,
            pad: 0,
        };
        let mut blocks = Vec::new();
        for (index, stride) in rates.iter().enumerate() {
            let input_dim = channels / (1 << index);
            let output_dim = channels / (1 << (index + 1));
            blocks.push(DecoderBlock::load(
                files,
                &format!("{prefix}.blocks.layers.{index}"),
                input_dim,
                output_dim,
                *stride,
                output_dim,
            )?);
        }
        let final_dim = channels / (1 << rates.len());
        Ok(Self {
            conv_in_dense,
            conv_in_point,
            blocks,
            snake_out_alpha: alpha(files, &format!("{prefix}.snake_out"), final_dim)?,
            conv_out: CausalConv {
                conv: load_conv(
                    files,
                    &format!("{prefix}.conv_out"),
                    final_dim,
                    1,
                    7,
                    1,
                    1,
                    1,
                )?,
                pad: 6,
            },
        })
    }

    /// `[latent, frames]` latents to `[1, seq]` tanh waveform.
    fn forward(&self, z: &[f32]) -> Vec<f32> {
        let mut h = self.conv_in_dense.forward(z);
        h = self.conv_in_point.forward(&h);
        for block in &self.blocks {
            h = block.forward(&mut h);
        }
        let frames = h.len() / self.conv_out.conv.in_ch;
        snake(
            &mut h,
            &self.snake_out_alpha,
            self.conv_out.conv.in_ch,
            frames,
        );
        let mut out = self.conv_out.forward(&h);
        for value in &mut out {
            *value = value.tanh();
        }
        out
    }
}

/// The `AudioVAE`: the latent encoder and the waveform decoder.
pub struct AudioVae {
    config: AudioVaeConfig,
    encoder: CausalEncoder,
    decoder: CausalDecoder,
}

impl AudioVae {
    /// Loads from the audio_vae weights of one opened checkpoint file.
    pub fn load(files: &[SafetensorsFile], prefix: &str, config: &AudioVaeConfig) -> Result<Self> {
        Ok(Self {
            encoder: CausalEncoder::load(
                files,
                &format!("{prefix}encoder"),
                config.encoder_dim,
                config.latent_dim,
                &config.encoder_rates,
            )?,
            decoder: CausalDecoder::load(
                files,
                &format!("{prefix}decoder"),
                config.latent_dim,
                config.decoder_dim,
                &config.decoder_rates,
            )?,
            config: config.clone(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn hop_length(&self) -> usize {
        self.config.hop_length()
    }

    /// `AudioVAE.encode` for one waveform `[1, seq]`: right-pad to a
    /// hop multiple and run the latent encoder. Returns
    /// `[latent_dim, frames]`.
    pub fn encode(&self, waveform: &[f32]) -> Vec<f32> {
        let hop = self.hop_length();
        let right_pad = hop * waveform.len().div_ceil(hop) - waveform.len();
        let mut input = waveform.to_vec();
        input.extend(std::iter::repeat_n(0.0f32, right_pad));
        self.encoder.forward(&input)
    }

    /// `AudioVAE.decode` for one latent sequence `[latent_dim,
    /// frames]`. Returns the waveform as `[samples]`.
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        self.decoder.forward(z)
    }
}
