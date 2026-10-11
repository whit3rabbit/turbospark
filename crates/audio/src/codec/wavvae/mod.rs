//! WavVAE: the MegaTTS3 latent audio codec (24 kHz waveform <-> 25 Hz latent).
//!
//! Reference: `tts/modules/wavvae/{decoder,encoder}` of
//! [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3) (Apache-2.0),
//! `WavVAE_V3`. There is no mlx-audio port; parity is against the PyTorch
//! classes through `tools/gen_wavvae_fixtures.py`.
//!
//! - Encoder: Encodec-style SEANet (weight-norm convs, ELU, reflect padding,
//!   two-layer LSTM with a skip) down to 25 Hz, `proj_to_z`, then a diagonal
//!   Gaussian whose first half is the mean.
//! - Decoder: `proj_to_decoder`, a 4x `Upsample` to 100 Hz, then a HiFiGAN-style
//!   generator (`Upsample` stages with a sin pre-activation and two parallel
//!   branches, `ResBlockMRFV2Inter` multi-receptive-field blocks).
//!
//! Weights come from `tools/convert_megatts3.py` (MLX layout, weight norm
//! unfolded). The official ByteDance checkpoint has no encoder, so
//! [`WavVae::has_encoder`] reports whether the community encoder is present.
//!
//! Layout: activations are channel-major `[ch, seq]`; latents are time-major
//! `[frames, z_dim]` (what the DiT consumes). Encoding is stochastic upstream
//! (`posterior.sample()`), so [`Posterior`] exposes the mean and takes the
//! normal draw from the caller.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::backend::{ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding};
use crate::codec::conv::{Conv1d, ConvTranspose1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::{ops, Result, SpeechError};

#[cfg(test)]
mod tests;

/// Bounds applied to every config value, so an untrusted config cannot request
/// absurd allocations.
const MAX_STAGES: usize = 8;
const MAX_RATIO: usize = 16;
const MAX_CHANNELS: usize = 4096;

/// Architecture of one WavVAE checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct WavVaeConfig {
    /// SEANet base width (`n_filters`).
    pub enc_filters: usize,
    /// Downsampling ratios in the reference's order (`[6,5,4,4,2]`); the
    /// encoder applies them reversed.
    pub enc_ratios: Vec<usize>,
    /// Encoder output width, the input of `proj_to_z`.
    pub enc_dim: usize,
    pub lstm_layers: usize,
    /// Latent channels (mean half of `proj_to_z`).
    pub z_dim: usize,
    /// Width of `proj_to_decoder` and the latent upsampler input.
    pub dec_in: usize,
    /// Upsampling factor of the latent upsampler (25 Hz -> 100 Hz).
    pub latent_up: usize,
    pub ngf: usize,
    pub dec_ratios: Vec<usize>,
    pub sample_rate: u32,
}

impl WavVaeConfig {
    /// The shipped MegaTTS3 checkpoint.
    pub fn megatts3() -> Self {
        Self {
            enc_filters: 32,
            enc_ratios: vec![6, 5, 4, 4, 2],
            enc_dim: 512,
            lstm_layers: 2,
            z_dim: 32,
            dec_in: 320,
            latent_up: 4,
            ngf: 128,
            dec_ratios: vec![5, 4, 4, 3],
            sample_rate: 24_000,
        }
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u = |f: &str| -> Result<usize> {
            value
                .get(f)
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .ok_or_else(|| bad(f, "missing or not an unsigned integer"))
        };
        let list = |f: &str| -> Result<Vec<usize>> {
            value
                .get(f)
                .and_then(|v| v.as_array())
                .ok_or_else(|| bad(f, "missing or not a list"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| bad(f, "non-integer entry"))
                })
                .collect()
        };
        let config = Self {
            enc_filters: u("enc_filters")?,
            enc_ratios: list("enc_ratios")?,
            enc_dim: u("enc_dim")?,
            lstm_layers: u("lstm_layers")?,
            z_dim: u("z_dim")?,
            dec_in: u("dec_in")?,
            latent_up: u("latent_up")?,
            ngf: u("ngf")?,
            dec_ratios: list("dec_ratios")?,
            sample_rate: u("sample_rate")? as u32,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let ratios = |f: &str, r: &[usize]| -> Result<()> {
            if r.is_empty() || r.len() > MAX_STAGES || r.iter().any(|&x| x == 0 || x > MAX_RATIO) {
                return Err(bad(f, "need 1..=8 ratios in 1..=16"));
            }
            Ok(())
        };
        ratios("enc_ratios", &self.enc_ratios)?;
        ratios("dec_ratios", &self.dec_ratios)?;
        let pos = |f: &str, v: usize| -> Result<()> {
            if v == 0 || v > MAX_CHANNELS {
                return Err(bad(f, "must be in 1..=4096"));
            }
            Ok(())
        };
        pos("enc_filters", self.enc_filters)?;
        pos("enc_dim", self.enc_dim)?;
        pos("z_dim", self.z_dim)?;
        pos("dec_in", self.dec_in)?;
        pos("ngf", self.ngf)?;
        if self.latent_up == 0 || self.latent_up > MAX_RATIO {
            return Err(bad("latent_up", "must be in 1..=16"));
        }
        if self.lstm_layers > 4 {
            return Err(bad("lstm_layers", "at most 4"));
        }
        if self.dec_in % 2 != 0 {
            return Err(bad("dec_in", "must be even (the upsampler halves it)"));
        }
        // Widest activation: enc_filters * 2^stages and ngf * 2^stages.
        if self.enc_filters << self.enc_ratios.len() > MAX_CHANNELS
            || self.ngf << self.dec_ratios.len() > MAX_CHANNELS
        {
            return Err(bad("enc_ratios/dec_ratios", "channel width exceeds 4096"));
        }
        if self.sample_rate == 0 {
            return Err(bad("sample_rate", "must be positive"));
        }
        Ok(())
    }

    /// Samples consumed per latent frame (960 for the shipped checkpoint).
    pub fn hop(&self) -> usize {
        self.enc_ratios.iter().product()
    }
}

fn bad(field: &str, why: &str) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.to_string(),
    }
}

// ----------------------------------------------------------------- helpers

/// PyTorch `pad1d(..., mode="reflect")` as the reference defines it: inputs no
/// longer than the pad are zero-extended first, then the extension is cut off.
fn pad_reflect(x: &[f32], ch: usize, len: usize, left: usize, right: usize) -> Vec<f32> {
    let max_pad = left.max(right);
    let extra = if len <= max_pad { max_pad - len + 1 } else { 0 };
    let src_len = len + extra;
    let out_len = src_len + left + right - extra;
    let mut out = vec![0.0f32; ch * out_len];
    for c in 0..ch {
        let row = &x[c * len..(c + 1) * len];
        for (i, o) in out[c * out_len..(c + 1) * out_len].iter_mut().enumerate() {
            let mut idx = i as isize - left as isize;
            if idx < 0 {
                idx = -idx;
            } else if idx >= src_len as isize {
                idx = 2 * (src_len as isize - 1) - idx;
            }
            // Indices past the real samples fall in the zero extension.
            *o = if (idx as usize) < len {
                row[idx as usize]
            } else {
                0.0
            };
        }
    }
    out
}

fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

fn add_in_place(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

// ----------------------------------------------------------------- encoder

/// `SConv1d`: a weight-norm conv with Encodec's padding rule (reflect, with
/// extra right padding so the last frame is complete).
struct SConv {
    conv: Conv1d,
    eff_kernel: usize,
}

impl SConv {
    fn load(
        file: &SafetensorsFile,
        module: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let conv = Conv1d::load(
            file,
            &format!("{module}.conv.conv"),
            in_ch,
            out_ch,
            kernel,
            stride,
            0,
            dilation,
            1,
        )?;
        Ok(Self {
            conv,
            eff_kernel: (kernel - 1) * dilation + 1,
        })
    }

    fn forward(&self, x: &[f32], len: usize) -> (Vec<f32>, usize) {
        let stride = self.conv.stride;
        let total = self.eff_kernel - stride;
        // ceil((len - k + total) / stride) without going through floats.
        let num = len as isize - self.eff_kernel as isize + total as isize;
        let frames =
            num.div_euclid(stride as isize) + isize::from(num.rem_euclid(stride as isize) != 0);
        let ideal = frames * stride as isize + (self.eff_kernel as isize - total as isize);
        let extra = (ideal - len as isize).max(0) as usize;
        let right = total / 2;
        let left = total - right;
        let padded = pad_reflect(x, self.conv.in_ch, len, left, right + extra);
        let out = self.conv.forward(&padded);
        let out_len = out.len() / self.conv.out_ch;
        (out, out_len)
    }
}

struct ResBlock {
    c1: SConv,
    c2: SConv,
    shortcut: SConv,
}

impl ResBlock {
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut t = x.to_vec();
        ops::elu(&mut t);
        let (mut t, l) = self.c1.forward(&t, len);
        ops::elu(&mut t);
        let (t, _) = self.c2.forward(&t, l);
        let (mut sc, _) = self.shortcut.forward(x, len);
        add_in_place(&mut sc, &t);
        sc
    }
}

/// One `nn.LSTM` layer in PyTorch layout (gate order i, f, g, o).
struct LstmLayer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    /// `bias_ih + bias_hh`, which only ever appear summed.
    bias: Vec<f32>,
    in_dim: usize,
    hidden: usize,
}

impl LstmLayer {
    fn load(file: &SafetensorsFile, prefix: &str, layer: usize, dim: usize) -> Result<Self> {
        let w_ih = load_f32_shaped(
            file,
            &format!("{prefix}.weight_ih_l{layer}"),
            &[4 * dim, dim],
        )?;
        let w_hh = load_f32_shaped(
            file,
            &format!("{prefix}.weight_hh_l{layer}"),
            &[4 * dim, dim],
        )?;
        let b_ih = load_f32_shaped(file, &format!("{prefix}.bias_ih_l{layer}"), &[4 * dim])?;
        let b_hh = load_f32_shaped(file, &format!("{prefix}.bias_hh_l{layer}"), &[4 * dim])?;
        let bias = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
        Ok(Self {
            w_ih,
            w_hh,
            bias,
            in_dim: dim,
            hidden: dim,
        })
    }

    /// `seq [T, in] -> [T, hidden]`, zero initial state.
    fn forward(&self, seq: &[f32], frames: usize) -> Vec<f32> {
        let h4 = 4 * self.hidden;
        // The input projection has no recurrence, so do it for all frames at once.
        let gx = ops::linear(seq, &self.w_ih, Some(&self.bias), frames, self.in_dim, h4);
        let mut out = vec![0.0f32; frames * self.hidden];
        let mut h = vec![0.0f32; self.hidden];
        let mut c = vec![0.0f32; self.hidden];
        for t in 0..frames {
            let gh = ops::linear(&h, &self.w_hh, None, 1, self.hidden, h4);
            let g = &gx[t * h4..(t + 1) * h4];
            for j in 0..self.hidden {
                let i_g = ops::sigmoid(g[j] + gh[j]);
                let f_g = ops::sigmoid(g[self.hidden + j] + gh[self.hidden + j]);
                let g_g = (g[2 * self.hidden + j] + gh[2 * self.hidden + j]).tanh();
                let o_g = ops::sigmoid(g[3 * self.hidden + j] + gh[3 * self.hidden + j]);
                c[j] = f_g * c[j] + i_g * g_g;
                h[j] = o_g * c[j].tanh();
            }
            out[t * self.hidden..(t + 1) * self.hidden].copy_from_slice(&h);
        }
        out
    }
}

/// `SLSTM`: stacked LSTM over `[ch, T]` with a residual add of the input.
struct SLstm {
    layers: Vec<LstmLayer>,
    dim: usize,
}

impl SLstm {
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut seq = transpose(x, self.dim, len);
        for layer in &self.layers {
            seq = layer.forward(&seq, len);
        }
        let mut out = transpose(&seq, len, self.dim);
        add_in_place(&mut out, x);
        out
    }
}

enum EncLayer {
    Conv(SConv),
    Res(Box<ResBlock>),
    Elu,
    Lstm(SLstm),
}

struct Encoder {
    layers: Vec<EncLayer>,
    proj_to_z: Linear,
}

impl Encoder {
    fn load(cfg: &WavVaeConfig, file: &SafetensorsFile) -> Result<Self> {
        let m = |i: usize| format!("encoder.encoder.model.{i}");
        let f = cfg.enc_filters;
        let mut layers = vec![EncLayer::Conv(SConv::load(file, &m(0), 1, f, 7, 1, 1)?)];
        let mut mult = 1;
        let mut idx = 1;
        for &ratio in cfg.enc_ratios.iter().rev() {
            let dim = mult * f;
            let hidden = dim / 2;
            layers.push(EncLayer::Res(Box::new(ResBlock {
                c1: SConv::load(file, &format!("{}.block.1", m(idx)), dim, hidden, 3, 1, 1)?,
                c2: SConv::load(file, &format!("{}.block.3", m(idx)), hidden, dim, 1, 1, 1)?,
                shortcut: SConv::load(file, &format!("{}.shortcut", m(idx)), dim, dim, 1, 1, 1)?,
            })));
            layers.push(EncLayer::Elu);
            layers.push(EncLayer::Conv(SConv::load(
                file,
                &m(idx + 2),
                dim,
                dim * 2,
                ratio * 2,
                ratio,
                1,
            )?));
            mult *= 2;
            idx += 3;
        }
        let dim = mult * f;
        if cfg.lstm_layers > 0 {
            let lstm = (0..cfg.lstm_layers)
                .map(|l| LstmLayer::load(file, &format!("{}.lstm", m(idx)), l, dim))
                .collect::<Result<Vec<_>>>()?;
            layers.push(EncLayer::Lstm(SLstm { layers: lstm, dim }));
            idx += 1;
        }
        layers.push(EncLayer::Elu);
        layers.push(EncLayer::Conv(SConv::load(
            file,
            &m(idx + 1),
            dim,
            cfg.enc_dim,
            7,
            1,
            1,
        )?));
        let proj_to_z = Linear::load(file, "proj_to_z", cfg.enc_dim, 2 * cfg.z_dim, true)?;
        Ok(Self { layers, proj_to_z })
    }
}

/// Diagonal Gaussian over the latent, time-major `[frames, z_dim]`.
#[derive(Debug, Clone)]
pub struct Posterior {
    pub frames: usize,
    pub z_dim: usize,
    pub mean: Vec<f32>,
    /// Already clamped to `[-30, 20]` like the reference.
    pub logvar: Vec<f32>,
}

impl Posterior {
    /// `mean + exp(0.5 * logvar) * eps`, with `eps` the caller's standard
    /// normal draw (same shape as `mean`).
    pub fn sample(&self, eps: &[f32]) -> Result<Vec<f32>> {
        if eps.len() != self.mean.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "eps has {} values, latent has {}",
                    eps.len(),
                    self.mean.len()
                ),
            });
        }
        Ok(self
            .mean
            .iter()
            .zip(&self.logvar)
            .zip(eps)
            .map(|((m, lv), e)| m + (0.5 * lv).exp() * e)
            .collect())
    }
}

// ----------------------------------------------------------------- decoder

/// Where the decoder's convolutions run. With a backend (the runtime's Metal
/// device) every ungrouped conv goes there, and so does every transposed conv
/// without output padding, which the device contract cannot express; the
/// elementwise work and the encoder stay on the host.
#[derive(Default)]
struct ConvExec {
    backend: Option<Rc<dyn ComputeBackend>>,
    /// Uploaded weights keyed by their host buffer, which the model owns.
    weights: RefCell<HashMap<usize, Rc<dyn DeviceWeight>>>,
}

impl ConvExec {
    fn weight(
        &self,
        backend: &dyn ComputeBackend,
        values: &[f32],
        dims: &[usize],
    ) -> Result<Rc<dyn DeviceWeight>> {
        let key = values.as_ptr() as usize;
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = backend.load_weight(WeightData {
            name: "wavvae_decoder",
            shape: dims,
            bytes: &bytes,
            encoding: WeightEncoding::F32,
            dtype: DType::F32,
            scales_dtype: None,
            offsets_dtype: None,
            scales: &[],
            offsets: &[],
            block_scales: &[],
        })?;
        self.weights.borrow_mut().insert(key, weight.clone());
        Ok(weight)
    }

    fn conv(&self, c: &Conv1d, x: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(backend) if c.groups == 1 => {
                let shape = ConvShape {
                    input_channels: c.in_ch,
                    output_channels: c.out_ch,
                    kernel: c.kernel,
                    stride: c.stride,
                    padding: c.padding,
                    dilation: c.dilation,
                    transpose: false,
                };
                self.weight(backend.as_ref(), &c.weight, &[c.out_ch, c.in_ch, c.kernel])?
                    .convolution_grouped(x, c.bias.as_deref(), shape, 1, DType::F32)
            }
            _ => Ok(c.forward(x)),
        }
    }

    fn conv_transpose(&self, c: &ConvTranspose1d, x: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(backend) if c.groups == 1 && c.output_padding == 0 => {
                let shape = ConvShape {
                    input_channels: c.in_ch,
                    output_channels: c.out_ch,
                    kernel: c.kernel,
                    stride: c.stride,
                    padding: c.padding,
                    dilation: 1,
                    transpose: true,
                };
                self.weight(backend.as_ref(), &c.weight, &[c.in_ch, c.out_ch, c.kernel])?
                    .convolution_grouped(x, c.bias.as_deref(), shape, 1, DType::F32)
            }
            _ => Ok(c.forward(x)),
        }
    }
}

/// Padding of a "same" dilated conv: `(k*d - d) / 2`.
fn same_pad(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

struct ResBlockMrf {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlockMrf {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize, kernel: usize) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        for (i, d) in [1usize, 3, 5].into_iter().enumerate() {
            convs1.push(Conv1d::load(
                file,
                &format!("{prefix}.convs1.{i}"),
                ch,
                ch,
                kernel,
                1,
                same_pad(kernel, d),
                d,
                1,
            )?);
            convs2.push(Conv1d::load(
                file,
                &format!("{prefix}.convs2.{i}"),
                ch,
                ch,
                kernel,
                1,
                same_pad(kernel, 1),
                1,
                1,
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, exec: &ConvExec, x: &[f32]) -> Result<Vec<f32>> {
        let mut x = x.to_vec();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let mut xt = x.clone();
            leaky_relu(&mut xt, 0.2);
            let mut xt = exec.conv(c1, &xt)?;
            leaky_relu(&mut xt, 0.2);
            let xt = exec.conv(c2, &xt)?;
            add_in_place(&mut x, &xt);
        }
        Ok(x)
    }
}

/// `ResBlockMRFV2Inter`: kernels 3, 7, 11 in parallel, averaged.
struct ResInter {
    blocks: Vec<ResBlockMrf>,
}

impl ResInter {
    fn load(file: &SafetensorsFile, prefix: &str, ch: usize) -> Result<Self> {
        let blocks = [3usize, 7, 11]
            .into_iter()
            .enumerate()
            .map(|(i, k)| ResBlockMrf::load(file, &format!("{prefix}.block{}", i + 1), ch, k))
            .collect::<Result<_>>()?;
        Ok(Self { blocks })
    }

    fn forward(&self, exec: &ConvExec, x: &[f32]) -> Result<Vec<f32>> {
        let mut acc = self.blocks[0].forward(exec, x)?;
        for b in &self.blocks[1..] {
            add_in_place(&mut acc, &b.forward(exec, x)?);
        }
        for v in acc.iter_mut() {
            *v /= 3.0;
        }
        Ok(acc)
    }
}

/// The reference `Upsample(mult, r)`: sin pre-activation, then a nearest-
/// upsample + conv branch and a transposed-conv branch, summed.
struct UpBlock {
    ratio: usize,
    conv: Conv1d,
    convt: ConvTranspose1d,
}

impl UpBlock {
    fn load(file: &SafetensorsFile, prefix: &str, in_ch: usize, ratio: usize) -> Result<Self> {
        let out_ch = in_ch / 2;
        let conv = Conv1d::load(
            file,
            &format!("{prefix}.upsample.3"),
            in_ch,
            out_ch,
            7,
            1,
            0,
            1,
            1,
        )?;
        let r_kernel = ratio.max(5);
        let mut convt = ConvTranspose1d::load(
            file,
            &format!("{prefix}.trans_upsample.1"),
            in_ch,
            out_ch,
            r_kernel * 2,
            ratio,
            r_kernel - ratio / 2,
        )?;
        // The shared loader assumes the DAC-family output_padding of 1; this
        // layer's reference sets `output_padding = r % 2` explicitly.
        convt.output_padding = ratio % 2;
        Ok(Self { ratio, conv, convt })
    }

    fn forward(&self, exec: &ConvExec, x: &[f32], len: usize) -> Result<(Vec<f32>, usize)> {
        let ch = self.conv.in_ch;
        let x: Vec<f32> = x.iter().map(|v| v + v.sin()).collect();
        let up_len = len * self.ratio;
        let mut nearest = vec![0.0f32; ch * up_len];
        for c in 0..ch {
            for t in 0..up_len {
                nearest[c * up_len + t] = x[c * len + t / self.ratio];
            }
        }
        leaky_relu(&mut nearest, 0.2);
        let padded = pad_reflect(&nearest, ch, up_len, 3, 3);
        let mut out = exec.conv(&self.conv, &padded)?;
        let mut t2 = x;
        leaky_relu(&mut t2, 0.2);
        let b2 = exec.conv_transpose(&self.convt, &t2)?;
        debug_assert_eq!(out.len(), b2.len());
        add_in_place(&mut out, &b2);
        Ok((out, up_len))
    }
}

struct Decoder {
    proj_to_decoder: Linear,
    latent_up: UpBlock,
    conv_in: Conv1d,
    stages: Vec<(UpBlock, ResInter)>,
    conv_out: Conv1d,
}

impl Decoder {
    fn load(cfg: &WavVaeConfig, file: &SafetensorsFile) -> Result<Self> {
        let proj_to_decoder = Linear::load(file, "proj_to_decoder", cfg.z_dim, cfg.dec_in, true)?;
        let latent_up = UpBlock::load(file, "latent_upsampler", cfg.dec_in, cfg.latent_up)?;
        let gen_in = cfg.dec_in / 2;
        let mut mult = 1usize << cfg.dec_ratios.len();
        let conv_in = Conv1d::load(
            file,
            "decoder.model_up.1",
            gen_in,
            mult * cfg.ngf,
            7,
            1,
            0,
            1,
            1,
        )?;
        let mut stages = Vec::new();
        for (i, &r) in cfg.dec_ratios.iter().enumerate() {
            let ch = mult * cfg.ngf;
            let up = UpBlock::load(file, &format!("decoder.model_up.{}", 2 + 2 * i), ch, r)?;
            let res = ResInter::load(file, &format!("decoder.model_up.{}", 3 + 2 * i), ch / 2)?;
            stages.push((up, res));
            mult /= 2;
        }
        let n = cfg.dec_ratios.len();
        let conv_out = Conv1d::load(
            file,
            &format!("decoder.model_up.{}", 4 + 2 * n),
            cfg.ngf,
            1,
            7,
            1,
            0,
            1,
            1,
        )?;
        Ok(Self {
            proj_to_decoder,
            latent_up,
            conv_in,
            stages,
            conv_out,
        })
    }
}

// ------------------------------------------------------------------- model

/// A loaded WavVAE. The encoder is optional: the official release ships the
/// decoder only.
pub struct WavVae {
    config: WavVaeConfig,
    encoder: Option<Encoder>,
    decoder: Decoder,
    exec: ConvExec,
}

impl WavVae {
    /// Opens `dir/model.safetensors` for the shipped MegaTTS3 geometry.
    pub fn open(dir: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::load(WavVaeConfig::megatts3(), &file)
    }

    pub fn load(config: WavVaeConfig, file: &SafetensorsFile) -> Result<Self> {
        config.validate()?;
        let encoder = if file.contains_tensor("encoder.encoder.model.0.conv.conv.weight_v") {
            Some(Encoder::load(&config, file)?)
        } else {
            None
        };
        let decoder = Decoder::load(&config, file)?;
        Ok(Self {
            config,
            encoder,
            decoder,
            exec: ConvExec::default(),
        })
    }

    /// Runs the decoder's convolutions on `backend` (the runtime's device);
    /// `None` keeps the host path. The encoder always runs on the host.
    pub fn with_backend(mut self, backend: Option<Rc<dyn ComputeBackend>>) -> Self {
        self.exec = ConvExec {
            backend,
            weights: RefCell::new(HashMap::new()),
        };
        self
    }

    pub fn config(&self) -> &WavVaeConfig {
        &self.config
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Mono samples at the codec sample rate -> latent posterior.
    pub fn encode(&self, samples: &[f32]) -> Result<Posterior> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "this WavVAE checkpoint has no encoder (official decoder-only release)"
                    .to_string(),
            })?;
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        let mut x = samples.to_vec();
        let mut len = x.len();
        for layer in &enc.layers {
            match layer {
                EncLayer::Conv(c) => (x, len) = c.forward(&x, len),
                EncLayer::Res(r) => x = r.forward(&x, len),
                EncLayer::Elu => ops::elu(&mut x),
                EncLayer::Lstm(l) => x = l.forward(&x, len),
            }
        }
        let tm = transpose(&x, self.config.enc_dim, len);
        let proj = enc.proj_to_z.forward(&tm, len);
        let z = self.config.z_dim;
        let mut mean = vec![0.0f32; len * z];
        let mut logvar = vec![0.0f32; len * z];
        for t in 0..len {
            mean[t * z..(t + 1) * z].copy_from_slice(&proj[t * 2 * z..t * 2 * z + z]);
            for j in 0..z {
                logvar[t * z + j] = proj[t * 2 * z + z + j].clamp(-30.0, 20.0);
            }
        }
        Ok(Posterior {
            frames: len,
            z_dim: z,
            mean,
            logvar,
        })
    }

    /// Time-major latent `[frames, z_dim]` -> mono samples
    /// (`frames * latent_up * prod(dec_ratios)` of them).
    pub fn decode(&self, latent: &[f32]) -> Result<Vec<f32>> {
        let z = self.config.z_dim;
        if latent.is_empty() || latent.len() % z != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "latent length {} is not a positive multiple of z_dim {z}",
                    latent.len()
                ),
            });
        }
        let frames = latent.len() / z;
        let d = &self.decoder;
        let proj = d.proj_to_decoder.forward(latent, frames);
        let x = transpose(&proj, frames, self.config.dec_in);
        let exec = &self.exec;
        let (mut x, mut len) = d.latent_up.forward(exec, &x, frames)?;
        let padded = pad_reflect(&x, self.config.dec_in / 2, len, 3, 3);
        x = exec.conv(&d.conv_in, &padded)?;
        for (up, res) in &d.stages {
            (x, len) = up.forward(exec, &x, len)?;
            x = res.forward(exec, &x)?;
        }
        leaky_relu(&mut x, 0.2);
        let padded = pad_reflect(&x, self.config.ngf, len, 3, 3);
        let mut out = exec.conv(&d.conv_out, &padded)?;
        for v in out.iter_mut() {
            *v = v.tanh();
        }
        Ok(out)
    }

    /// Decoder-input width after the latent upsampler: `[dec_in / 2, 4 * frames]`.
    /// Exposed for trace comparison in tests.
    #[cfg(test)]
    fn latent_upsample(&self, latent: &[f32]) -> (Vec<f32>, usize) {
        let z = self.config.z_dim;
        let frames = latent.len() / z;
        let d = &self.decoder;
        let proj = d.proj_to_decoder.forward(latent, frames);
        let x = transpose(&proj, frames, self.config.dec_in);
        d.latent_up.forward(&self.exec, &x, frames).unwrap()
    }
}
