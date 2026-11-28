//! Oobleck VAE: the 48 kHz stereo waveform autoencoder of ACE-Step 1.5.
//!
//! Reference: diffusers 0.43.0 `AutoencoderOobleck`
//! (`diffusers/models/autoencoders/autoencoder_oobleck.py`), which
//! upstream ACE-Step loads for `vae/`. The checkpoint keeps the
//! PyTorch `weight_norm` split (`weight_g`, `weight_v`) and Snake
//! parameters shaped `[1, C, 1]`; both are folded once at load.
//!
//! Layout: activations are channel-major `[ch, frames]`, one batch item.
//! The encoder maps `[audio_channels, samples]` to moments
//! `[2 * latent, samples / hop]`; the first half is the Gaussian mean,
//! the second the pre-softplus scale. The decoder maps
//! `[latent, frames]` to `[audio_channels, frames * hop]`.
//!
//! Scope: the untiled encode / decode contract plus upstream's
//! overlap-discard tiled decode ([`OobleckVae::decode_tiled`]). The
//! pipeline-level input clamp and output peak normalization live above
//! this module.
//!
//! Device seam: [`OobleckVae::attach_backend`] uploads every folded conv
//! weight as F32 to a [`ComputeBackend`] and runs the convolutions there.
//! Snake and the residual adds stay on the host (Snake is threaded across
//! channels; the per-element arithmetic is unchanged).

use std::rc::Rc;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::backend::{ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding};

use crate::codec::conv::{load_bias, wn_fold_in_place, BiasMode, Conv1d, ConvTranspose1d, WnFold};
use crate::codec::wnconv::load_f32_shaped;
use crate::{Result, SpeechError};

/// Bounds on config values read from an untrusted `config.json`. The
/// shipped VAE uses 5 stages, stride at most 10, width 128.
const MAX_STAGES: usize = 16;
const MAX_STRIDE: usize = 64;
const MAX_WIDTH: usize = 8192;
const MAX_MULTIPLE: usize = 64;

/// Residual-unit dilations, fixed by the reference blocks.
const DILATIONS: [usize; 3] = [1, 3, 9];

/// VAE geometry, one-to-one with the diffusers config fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OobleckConfig {
    pub encoder_hidden_size: usize,
    pub downsampling_ratios: Vec<usize>,
    pub channel_multiples: Vec<usize>,
    pub decoder_channels: usize,
    pub decoder_input_channels: usize,
    pub audio_channels: usize,
    pub sampling_rate: u32,
}

impl Default for OobleckConfig {
    /// The diffusers constructor defaults.
    fn default() -> Self {
        Self {
            encoder_hidden_size: 128,
            downsampling_ratios: vec![2, 4, 4, 8, 8],
            channel_multiples: vec![1, 2, 4, 8, 16],
            decoder_channels: 128,
            decoder_input_channels: 64,
            audio_channels: 2,
            sampling_rate: 44100,
        }
    }
}

impl OobleckConfig {
    /// Parses a diffusers `config.json`, filling absent fields with the
    /// constructor defaults and refusing out-of-range geometry.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let d = Self::default();
        let bad = |field: &str, why: &str| SpeechError::BadConfig {
            field: field.to_string(),
            why: why.to_string(),
        };
        let num = |field: &str, default: usize, max: usize| -> Result<usize> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(default),
                Some(v) => {
                    let n = v
                        .as_u64()
                        .ok_or_else(|| bad(field, "expected an integer"))?
                        as usize;
                    if n == 0 || n > max {
                        return Err(bad(field, &format!("{n} outside 1..={max}")));
                    }
                    Ok(n)
                }
            }
        };
        let list = |field: &str, default: &[usize], max: usize| -> Result<Vec<usize>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(default.to_vec()),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| bad(field, "expected a list"))?;
                    if arr.is_empty() || arr.len() > MAX_STAGES {
                        return Err(bad(field, &format!("length outside 1..={MAX_STAGES}")));
                    }
                    arr.iter()
                        .map(|x| match x.as_u64() {
                            Some(n) if n >= 1 && n as usize <= max => Ok(n as usize),
                            _ => Err(bad(field, &format!("entries must be in 1..={max}"))),
                        })
                        .collect()
                }
            }
        };
        let config = Self {
            encoder_hidden_size: num("encoder_hidden_size", d.encoder_hidden_size, MAX_WIDTH)?,
            downsampling_ratios: list("downsampling_ratios", &d.downsampling_ratios, MAX_STRIDE)?,
            channel_multiples: list("channel_multiples", &d.channel_multiples, MAX_MULTIPLE)?,
            decoder_channels: num("decoder_channels", d.decoder_channels, MAX_WIDTH)?,
            decoder_input_channels: num(
                "decoder_input_channels",
                d.decoder_input_channels,
                MAX_WIDTH,
            )?,
            audio_channels: num("audio_channels", d.audio_channels, 8)?,
            sampling_rate: num("sampling_rate", d.sampling_rate as usize, 384_000)? as u32,
        };
        if config.channel_multiples.len() != config.downsampling_ratios.len() {
            return Err(bad(
                "channel_multiples",
                "must have one entry per downsampling ratio",
            ));
        }
        // The encoder's last conv emits `encoder_hidden_size` channels
        // that the posterior splits into mean and scale halves, so the
        // latent width the decoder consumes must be exactly half.
        if config.encoder_hidden_size != 2 * config.decoder_input_channels {
            return Err(bad(
                "decoder_input_channels",
                "must be half of encoder_hidden_size (mean and scale halves)",
            ));
        }
        let widest = config.channel_multiples.iter().max().copied().unwrap_or(1);
        if config.encoder_hidden_size.max(config.decoder_channels) * widest > MAX_WIDTH {
            return Err(bad(
                "channel_multiples",
                "widest stage exceeds the width bound",
            ));
        }
        Ok(config)
    }

    /// Waveform samples per latent frame.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }

    /// Latent channels (the mean half of the encoder output).
    pub fn latent_channels(&self) -> usize {
        self.decoder_input_channels
    }

    /// `[1] + channel_multiples`, as both reference stacks index it.
    fn multiples(&self) -> Vec<usize> {
        let mut m = Vec::with_capacity(self.channel_multiples.len() + 1);
        m.push(1);
        m.extend_from_slice(&self.channel_multiples);
        m
    }
}

/// diffusers `Snake1d` with `logscale=True`:
/// `x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x)^2`.
#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    inv_beta: Vec<f32>,
}

impl Snake {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Self> {
        let alpha = load_f32_shaped(file, &format!("{prefix}.alpha"), &[1, channels, 1])?;
        let beta = load_f32_shaped(file, &format!("{prefix}.beta"), &[1, channels, 1])?;
        Ok(Self {
            alpha: alpha.iter().map(|a| a.exp()).collect(),
            inv_beta: beta.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect(),
        })
    }

    fn forward(&self, x: &mut [f32], frames: usize) {
        debug_assert_eq!(x.len(), self.alpha.len() * frames);
        let row = |ch: usize, row: &mut [f32]| {
            let (a, inv) = (self.alpha[ch], self.inv_beta[ch]);
            for v in row {
                let s = (a * *v).sin();
                *v += inv * (s * s);
            }
        };
        let channels = self.alpha.len();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        // Small tensors are not worth a thread spawn.
        if threads == 1 || channels * frames < (1 << 18) {
            for (ch, r) in x.chunks_exact_mut(frames).enumerate() {
                row(ch, r);
            }
            return;
        }
        let per = channels.div_ceil(threads);
        std::thread::scope(|scope| {
            for (i, chunk) in x.chunks_mut(per * frames).enumerate() {
                let row = &row;
                scope.spawn(move || {
                    for (j, r) in chunk.chunks_exact_mut(frames).enumerate() {
                        row(i * per + j, r);
                    }
                });
            }
        });
    }
}

/// A convolution on the host, optionally mirrored on a device.
#[derive(Clone)]
enum Host {
    Forward(Conv1d),
    Transpose(ConvTranspose1d),
}

/// Device form of a convolution. A transposed conv with `kernel = 2 *
/// stride` runs as `stride` kernel-2 forward convs, one per output phase
/// (`phases[r]` produces the outputs `t` with `(t + padding) % stride ==
/// r`), so the device multiplies only taps that touch the input.
#[derive(Clone)]
enum Device {
    Forward(Rc<dyn DeviceWeight>),
    Phases(Vec<Rc<dyn DeviceWeight>>),
}

#[derive(Clone)]
struct Conv {
    host: Host,
    device: Option<Device>,
}

impl std::fmt::Debug for Conv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.shape();
        write!(
            f,
            "Conv({}->{} k{} s{} t{} device={})",
            s.input_channels,
            s.output_channels,
            s.kernel,
            s.stride,
            s.transpose,
            self.device.is_some()
        )
    }
}

impl Conv {
    fn forward_conv(c: Conv1d) -> Self {
        Self {
            host: Host::Forward(c),
            device: None,
        }
    }

    fn transpose_conv(c: ConvTranspose1d) -> Self {
        Self {
            host: Host::Transpose(c),
            device: None,
        }
    }

    fn shape(&self) -> ConvShape {
        match &self.host {
            Host::Forward(c) => ConvShape {
                input_channels: c.in_ch,
                output_channels: c.out_ch,
                kernel: c.kernel,
                stride: c.stride,
                padding: c.padding,
                dilation: c.dilation,
                transpose: false,
            },
            Host::Transpose(c) => ConvShape {
                input_channels: c.in_ch,
                output_channels: c.out_ch,
                kernel: c.kernel,
                stride: c.stride,
                padding: c.padding,
                dilation: 1,
                transpose: true,
            },
        }
    }

    fn out_len(&self, len: usize) -> usize {
        match &self.host {
            Host::Forward(c) => {
                let span = c.dilation * (c.kernel - 1) + 1;
                (len + 2 * c.padding).saturating_sub(span) / c.stride + 1
            }
            Host::Transpose(c) => {
                ((len - 1) * c.stride + c.kernel + c.output_padding).saturating_sub(2 * c.padding)
            }
        }
    }

    fn out_channels(&self) -> usize {
        self.shape().output_channels
    }

    /// Uploads the folded weight (F32) to `backend`: a forward conv as is
    /// (`[out, in, K]`), a transposed conv as its per-phase kernel-2
    /// forward convs.
    fn attach(&mut self, name: &str, backend: &dyn ComputeBackend) -> Result<()> {
        let upload = |weight: &[f32], shape: [usize; 3]| {
            let bytes: Vec<u8> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
            backend.load_weight(WeightData {
                name,
                shape: &shape,
                bytes: &bytes,
                encoding: WeightEncoding::F32,
                dtype: DType::F32,
                scales_dtype: None,
                offsets_dtype: None,
                scales: &[],
                offsets: &[],
                block_scales: &[],
                planned_rows: None,
                tiled_f32: false,
            })
        };
        self.device = Some(match &self.host {
            Host::Forward(c) => Device::Forward(upload(&c.weight, [c.out_ch, c.in_ch, c.kernel])?),
            Host::Transpose(c) => {
                if c.output_padding != 0 || c.groups != 1 || c.kernel != 2 * c.stride {
                    return Err(SpeechError::Unsupported {
                        why: "device transposed conv needs kernel = 2 x stride, groups 1, \
                              no output padding"
                            .into(),
                    });
                }
                let (ci, co, k, st) = (c.in_ch, c.out_ch, c.kernel, c.stride);
                // Phase r: out[m] = w[.., r + s] x[m - 1] + w[.., r] x[m],
                // i.e. a forward conv with taps [w(r + s), w(r)] and pad 1.
                let mut phases = Vec::with_capacity(st);
                for r in 0..st {
                    let mut w = vec![0.0f32; co * ci * 2];
                    for o in 0..co {
                        for i in 0..ci {
                            w[(o * ci + i) * 2] = c.weight[(i * co + o) * k + r + st];
                            w[(o * ci + i) * 2 + 1] = c.weight[(i * co + o) * k + r];
                        }
                    }
                    phases.push(upload(&w, [co, ci, 2])?);
                }
                Device::Phases(phases)
            }
        });
        Ok(())
    }

    /// `[in_ch, len]` -> (`[out_ch, out_len]`, out_len).
    fn forward(&self, x: &[f32], len: usize) -> Result<(Vec<f32>, usize)> {
        let out_len = self.out_len(len);
        let out = match (&self.device, &self.host) {
            (Some(Device::Forward(w)), Host::Forward(c)) => {
                w.convolution(x, c.bias.as_deref(), self.shape(), DType::F32)?
            }
            (Some(Device::Phases(phases)), Host::Transpose(c)) => {
                let shape = ConvShape {
                    input_channels: c.in_ch,
                    output_channels: c.out_ch,
                    kernel: 2,
                    stride: 1,
                    padding: 1,
                    dilation: 1,
                    transpose: false,
                };
                let (st, pad, co) = (c.stride, c.padding, c.out_ch);
                let plen = len + 1;
                let mut parts = Vec::with_capacity(phases.len());
                for w in phases {
                    // `len + 1` phase outputs; output m lands at t = m s + r - pad.
                    let part = w.convolution(x, c.bias.as_deref(), shape, DType::F32)?;
                    if part.len() != co * plen {
                        return Err(SpeechError::Unsupported {
                            why: format!("phase conv returned {} values", part.len()),
                        });
                    }
                    parts.push(part);
                }
                let mut y = vec![0.0f32; co * out_len];
                let fill = |o0: usize, rows: &mut [f32]| {
                    for (j, row) in rows.chunks_exact_mut(out_len).enumerate() {
                        let o = o0 + j;
                        for (r, part) in parts.iter().enumerate() {
                            let src = &part[o * plen..(o + 1) * plen];
                            for (m, &v) in src.iter().enumerate() {
                                if let Some(t) =
                                    (m * st + r).checked_sub(pad).filter(|&t| t < out_len)
                                {
                                    row[t] = v;
                                }
                            }
                        }
                    }
                };
                let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
                let per = co.div_ceil(threads.max(1));
                std::thread::scope(|scope| {
                    for (i, rows) in y.chunks_mut(per * out_len).enumerate() {
                        let fill = &fill;
                        scope.spawn(move || fill(i * per, rows));
                    }
                });
                y
            }
            (Some(_), _) => {
                return Err(SpeechError::Unsupported {
                    why: "device form does not match the host convolution".into(),
                })
            }
            (None, Host::Forward(c)) => c.forward(x),
            (None, Host::Transpose(c)) => c.forward(x),
        };
        let want = self.out_channels() * out_len;
        if out.len() != want {
            return Err(SpeechError::Unsupported {
                why: format!("convolution returned {} values, expected {want}", out.len()),
            });
        }
        Ok((out, out_len))
    }
}

/// Folds a PyTorch `weight_norm(dim=0)` pair: `weight_g` is `[rows, 1,
/// 1]` and each row of `weight_v` is contiguous, so no permute is needed
/// for either Conv1d `[out, in, K]` or ConvTranspose1d `[in, out, K]`.
fn load_wn(file: &SafetensorsFile, prefix: &str, shape: [usize; 3]) -> Result<Vec<f32>> {
    let mut v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &shape)?;
    let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
    wn_fold_in_place(&mut v, &g, shape[1] * shape[2], WnFold::Div);
    Ok(v)
}

#[allow(clippy::too_many_arguments)]
fn conv(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    bias: bool,
) -> Result<Conv1d> {
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        dilation,
        groups: 1,
        weight: load_wn(file, prefix, [out_ch, in_ch, kernel])?,
        bias: load_bias(
            file,
            &format!("{prefix}.bias"),
            out_ch,
            BiasMode::required(bias),
        )?,
    })
}

/// Snake, k=7 dilated conv, Snake, k=1 conv, residual add (with the
/// reference's symmetric trim, a no-op at these paddings).
#[derive(Debug, Clone)]
struct ResidualUnit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, dilation: usize) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1: Snake::load(file, &format!("{prefix}.snake1"), dim)?,
            conv1: Conv::forward_conv(conv(
                file,
                &format!("{prefix}.conv1"),
                dim,
                dim,
                7,
                1,
                pad,
                dilation,
                true,
            )?),
            snake2: Snake::load(file, &format!("{prefix}.snake2"), dim)?,
            conv2: Conv::forward_conv(conv(
                file,
                &format!("{prefix}.conv2"),
                dim,
                dim,
                1,
                1,
                0,
                1,
                true,
            )?),
        })
    }

    fn forward(&self, x: Vec<f32>, frames: usize) -> Result<(Vec<f32>, usize)> {
        let mut h = x.clone();
        self.snake1.forward(&mut h, frames);
        let (mut h, mid) = self.conv1.forward(&h, frames)?;
        self.snake2.forward(&mut h, mid);
        let (mut out, out_len) = self.conv2.forward(&h, mid)?;
        let trim = frames.saturating_sub(out_len) / 2;
        let dim = self.conv2.out_channels();
        for ch in 0..dim {
            let skip = &x[ch * frames + trim..ch * frames + trim + out_len];
            for (o, s) in out[ch * out_len..(ch + 1) * out_len].iter_mut().zip(skip) {
                // Reference is hidden_state + output; IEEE addition is
                // commutative, so this is the same bits.
                *o += *s;
            }
        }
        Ok((out, out_len))
    }

    fn attach(&mut self, prefix: &str, backend: &dyn ComputeBackend) -> Result<()> {
        self.conv1.attach(&format!("{prefix}.conv1"), backend)?;
        self.conv2.attach(&format!("{prefix}.conv2"), backend)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    res: [ResidualUnit; 3],
    snake: Snake,
    down: Conv,
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: Snake,
    up: Conv,
    res: [ResidualUnit; 3],
}

fn res_units(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<[ResidualUnit; 3]> {
    Ok([
        ResidualUnit::load(file, &format!("{prefix}.res_unit1"), dim, DILATIONS[0])?,
        ResidualUnit::load(file, &format!("{prefix}.res_unit2"), dim, DILATIONS[1])?,
        ResidualUnit::load(file, &format!("{prefix}.res_unit3"), dim, DILATIONS[2])?,
    ])
}

/// The full autoencoder in f32, convolutions on the host or a device.
#[derive(Debug, Clone)]
pub struct OobleckVae {
    config: OobleckConfig,
    enc_in: Conv,
    enc_blocks: Vec<EncoderBlock>,
    enc_snake: Snake,
    enc_out: Conv,
    dec_in: Conv,
    dec_blocks: Vec<DecoderBlock>,
    dec_snake: Snake,
    dec_out: Conv,
}

/// Posterior moments for one item, `[latent, frames]` each.
#[derive(Debug, Clone)]
pub struct Moments {
    pub frames: usize,
    pub mean: Vec<f32>,
    /// `softplus(scale) + 1e-4`.
    pub std: Vec<f32>,
}

impl Moments {
    /// `mean + std * noise`, the reference `latent_dist.sample()` with
    /// caller-supplied standard-normal noise of the same shape.
    pub fn sample(&self, noise: &[f32]) -> Result<Vec<f32>> {
        if noise.len() != self.mean.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "noise has {} values, posterior has {}",
                    noise.len(),
                    self.mean.len()
                ),
            });
        }
        Ok(self
            .mean
            .iter()
            .zip(&self.std)
            .zip(noise)
            .map(|((m, s), n)| m + s * n)
            .collect())
    }
}

/// torch `softplus` (beta 1, threshold 20).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

impl OobleckVae {
    /// Loads both stacks from a diffusers-layout safetensors file.
    pub fn load(config: OobleckConfig, file: &SafetensorsFile) -> Result<Self> {
        let m = config.multiples();
        let strides = &config.downsampling_ratios;
        let eh = config.encoder_hidden_size;
        let enc_in = Conv::forward_conv(conv(
            file,
            "encoder.conv1",
            config.audio_channels,
            eh,
            7,
            1,
            3,
            1,
            true,
        )?);
        let mut enc_blocks = Vec::with_capacity(strides.len());
        for (i, &s) in strides.iter().enumerate() {
            let p = format!("encoder.block.{i}");
            let (cin, cout) = (eh * m[i], eh * m[i + 1]);
            enc_blocks.push(EncoderBlock {
                res: res_units(file, &p, cin)?,
                snake: Snake::load(file, &format!("{p}.snake1"), cin)?,
                down: Conv::forward_conv(conv(
                    file,
                    &format!("{p}.conv1"),
                    cin,
                    cout,
                    2 * s,
                    s,
                    s.div_ceil(2),
                    1,
                    true,
                )?),
            });
        }
        let d_model = eh * m[m.len() - 1];
        let enc_snake = Snake::load(file, "encoder.snake1", d_model)?;
        let enc_out =
            Conv::forward_conv(conv(file, "encoder.conv2", d_model, eh, 3, 1, 1, 1, true)?);

        let dc = config.decoder_channels;
        let n = strides.len();
        let dec_in = Conv::forward_conv(conv(
            file,
            "decoder.conv1",
            config.decoder_input_channels,
            dc * m[n],
            7,
            1,
            3,
            1,
            true,
        )?);
        let mut dec_blocks = Vec::with_capacity(n);
        for (i, &s) in strides.iter().rev().enumerate() {
            let p = format!("decoder.block.{i}");
            let (cin, cout) = (dc * m[n - i], dc * m[n - i - 1]);
            let prefix = format!("{p}.conv_t1");
            let up = ConvTranspose1d {
                in_ch: cin,
                out_ch: cout,
                kernel: 2 * s,
                stride: s,
                padding: s.div_ceil(2),
                output_padding: 0,
                groups: 1,
                weight: load_wn(file, &prefix, [cin, cout, 2 * s])?,
                bias: load_bias(file, &format!("{prefix}.bias"), cout, BiasMode::Required)?,
            };
            dec_blocks.push(DecoderBlock {
                snake: Snake::load(file, &format!("{p}.snake1"), cin)?,
                up: Conv::transpose_conv(up),
                res: res_units(file, &p, cout)?,
            });
        }
        let dec_snake = Snake::load(file, "decoder.snake1", dc)?;
        let dec_out = Conv::forward_conv(conv(
            file,
            "decoder.conv2",
            dc,
            config.audio_channels,
            7,
            1,
            3,
            1,
            false,
        )?);
        Ok(Self {
            config,
            enc_in,
            enc_blocks,
            enc_snake,
            enc_out,
            dec_in,
            dec_blocks,
            dec_snake,
            dec_out,
        })
    }

    pub fn config(&self) -> &OobleckConfig {
        &self.config
    }

    /// Moves every convolution to `backend` (F32 weights, host layout).
    pub fn attach_backend(&mut self, backend: &dyn ComputeBackend) -> Result<()> {
        self.enc_in.attach("encoder.conv1", backend)?;
        for (i, b) in self.enc_blocks.iter_mut().enumerate() {
            for (j, u) in b.res.iter_mut().enumerate() {
                u.attach(&format!("encoder.block.{i}.res_unit{}", j + 1), backend)?;
            }
            b.down
                .attach(&format!("encoder.block.{i}.conv1"), backend)?;
        }
        self.enc_out.attach("encoder.conv2", backend)?;
        self.dec_in.attach("decoder.conv1", backend)?;
        for (i, b) in self.dec_blocks.iter_mut().enumerate() {
            b.up.attach(&format!("decoder.block.{i}.conv_t1"), backend)?;
            for (j, u) in b.res.iter_mut().enumerate() {
                u.attach(&format!("decoder.block.{i}.res_unit{}", j + 1), backend)?;
            }
        }
        self.dec_out.attach("decoder.conv2", backend)
    }

    /// Encodes `[audio_channels, samples]` (channel-major) to posterior
    /// moments. Trailing samples past the last whole hop are dropped by
    /// the strided convs, as in the reference.
    pub fn encode(&self, audio: &[f32], samples: usize) -> Result<Moments> {
        let ch = self.config.audio_channels;
        if samples == 0 || audio.len() != ch * samples {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {ch} x {samples} samples, got {} values",
                    audio.len()
                ),
            });
        }
        if samples < self.config.hop_length() {
            return Err(SpeechError::Input {
                why: format!(
                    "{samples} samples is shorter than one latent hop ({})",
                    self.config.hop_length()
                ),
            });
        }
        let (mut h, mut len) = self.enc_in.forward(audio, samples)?;
        for block in &self.enc_blocks {
            for unit in &block.res {
                (h, len) = unit.forward(h, len)?;
            }
            block.snake.forward(&mut h, len);
            (h, len) = block.down.forward(&h, len)?;
        }
        self.enc_snake.forward(&mut h, len);
        let (h, len) = self.enc_out.forward(&h, len)?;
        let half = self.config.latent_channels() * len;
        let mean = h[..half].to_vec();
        let std = h[half..].iter().map(|&s| softplus(s) + 1e-4).collect();
        Ok(Moments {
            frames: len,
            mean,
            std,
        })
    }

    /// Decodes `[latent, frames]` (channel-major) to
    /// `[audio_channels, samples]`; returns the samples per channel.
    pub fn decode(&self, latents: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        let lc = self.config.latent_channels();
        if frames == 0 || latents.len() != lc * frames {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {lc} x {frames} latents, got {} values",
                    latents.len()
                ),
            });
        }
        let (mut h, mut len) = self.dec_in.forward(latents, frames)?;
        for block in &self.dec_blocks {
            block.snake.forward(&mut h, len);
            (h, len) = block.up.forward(&h, len)?;
            for unit in &block.res {
                (h, len) = unit.forward(h, len)?;
            }
        }
        self.dec_snake.forward(&mut h, len);
        self.dec_out.forward(&h, len)
    }

    /// Upstream's overlap-discard tiled decode (`_tiled_decode_inner` /
    /// `_tiled_decode_gpu` at the pinned commit): windows of `chunk`
    /// latent frames that step by `chunk - 2 * overlap`, each decoded on
    /// its own and trimmed back to its core. The overlap shrinks by
    /// halves (not below 4) until the stride is positive, as upstream
    /// does. A latent shorter than one chunk decodes untiled.
    pub fn decode_tiled(
        &self,
        latents: &[f32],
        frames: usize,
        chunk: usize,
        overlap: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let lc = self.config.latent_channels();
        if frames == 0 || latents.len() != lc * frames || chunk == 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {lc} x {frames} latents and a positive chunk, got {} values",
                    latents.len()
                ),
            });
        }
        if frames <= chunk {
            return self.decode(latents, frames);
        }
        let mut overlap = overlap;
        let original = overlap;
        while chunk <= 2 * overlap && overlap > 4 {
            overlap /= 2;
        }
        if overlap < 4 && original >= 4 {
            overlap = 4;
        }
        let stride = chunk
            .checked_sub(2 * overlap)
            .filter(|&s| s > 0)
            .ok_or_else(|| SpeechError::Input {
                why: format!("chunk {chunk} must exceed 2 x overlap {overlap}"),
            })?;
        let hop = self.config.hop_length();
        let ch = self.config.audio_channels;
        let mut cores: Vec<Vec<f32>> = vec![Vec::with_capacity(frames * hop); ch];
        for i in 0..frames.div_ceil(stride) {
            let core_start = i * stride;
            let core_end = (core_start + stride).min(frames);
            let win_start = core_start.saturating_sub(overlap);
            let win_end = (core_end + overlap).min(frames);
            let win = win_end - win_start;
            let mut part = Vec::with_capacity(lc * win);
            for c in 0..lc {
                part.extend_from_slice(&latents[c * frames + win_start..c * frames + win_end]);
            }
            let (audio, samples) = self.decode(&part, win)?;
            // The decoder upsamples by exactly `hop`, so upstream's
            // round(added * factor) is an integer product here.
            let trim_start = (core_start - win_start) * hop;
            let trim_end = (win_end - core_end) * hop;
            for (c, core) in cores.iter_mut().enumerate() {
                core.extend_from_slice(
                    &audio[c * samples + trim_start..(c + 1) * samples - trim_end],
                );
            }
        }
        let samples = cores[0].len();
        Ok((cores.concat(), samples))
    }
}
