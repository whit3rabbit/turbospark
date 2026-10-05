//! Kokoro 82M: StyleTTS2-family TTS with an iSTFTNet decoder.
//!
//! Reference: `mlx_audio/tts/models/kokoro/` (kokoro.py, modules.py,
//! istftnet.py, pipeline.py) at the cloned v0.5.7 revision.
//!
//! Data flow for one segment (batch size 1 throughout, as in the
//! reference): phoneme ids -> PLBert (ALBERT with weight-shared layers)
//! -> duration predictor (BLSTMs + AdaLayerNorm) -> duration alignment
//! -> prosody (F0/noise) predictor -> text encoder -> iSTFTNet decoder
//! with a harmonic sine source -> 24 kHz audio.
//!
//! Determinism note: the reference's SineGen injects random initial
//! phase (harmonics only) and additive noise at inference, so final
//! waveforms are stochastic by construction; this port uses a fixed-seed
//! LCG. Verification pins every deterministic stage (durations, F0/N,
//! alignment, decoder features) and treats the waveform statistically.
//!
//! Weight layouts: every `weight_v` in the checkpoint is PyTorch order
//! (`[out, in, kernel]` for convs, `[in, out, kernel]` for the
//! generator's transpose convs, `[in, 1, kernel]` for the depthwise
//! pool); the reference's sanitize transposes at load time, so this
//! port reads the PyTorch order directly.

pub mod frontend;
pub use frontend::{
    EnglishFrontend, PhonemeVocabulary, SynthesisRequest, SynthesisSegment, Voice, VoicePack,
};

use std::collections::HashMap;
use std::path::Path;

use turbospark_audio::fft::ComplexF32;
use turbospark_audio::stft::{istft, stft, StftOptions};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

const SAMPLE_RATE: f32 = 24_000.0;
const ISTFT_N_FFT: usize = 20;
const ISTFT_HOP: usize = 5;
/// 10 * 6 * hop 5: waveform samples per prosody frame.
const TOTAL_UPSAMPLE: usize = 300;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const HARMONICS: usize = 8;
const MAX_FRAMES_PER_PHONEME: i32 = 100;

// ---------------------------------------------------------------------------
// Weight-normalized convolutions
// ---------------------------------------------------------------------------

/// Weight-normalized strided conv1d. The reference normalizes per call,
/// but g and v are constants, so folding it into load time is exact.
struct WnConv1d {
    weight: Vec<f32>, // [out, in, kernel]
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_ch: usize,
    in_ch: usize,
}

/// `v [out, in, kernel]` (the checkpoint's PyTorch order) scaled per
/// output channel by `g / (L2(v channel) + 1e-7)`; the layout is
/// already the kernel's, so the weight is a scaled copy.
#[allow(
    clippy::needless_range_loop,
    reason = "the loop indexes mirror the packed tensor geometry"
)]
fn weight_norm_oki(v: &[f32], g: &[f32], out: usize, in_ch: usize, kernel: usize) -> Vec<f32> {
    let mut weight = Vec::with_capacity(v.len());
    let per = in_ch * kernel;
    for o in 0..out {
        let norm = v[o * per..(o + 1) * per]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let scale = g[o] / (norm + 1e-7);
        for i in o * per..(o + 1) * per {
            weight.push(v[i] * scale);
        }
    }
    weight
}

fn tensor_shape(file: &SafetensorsFile, name: &str) -> Result<(usize, usize, usize)> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    })?;
    if desc.shape.len() != 3 {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected 3-D, got {:?}", desc.shape),
        });
    }
    Ok((desc.shape[0], desc.shape[1], desc.shape[2]))
}

fn load_wn_conv(
    file: &SafetensorsFile,
    base: &str,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<WnConv1d> {
    let v = file.load_as_f32(&format!("{base}.weight_v"))?;
    let g = file.load_as_f32(&format!("{base}.weight_g"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let (out, in_ch, kernel) = tensor_shape(file, &format!("{base}.weight_v"))?;
    let weight = weight_norm_oki(&v, &g, out, in_ch, kernel);
    Ok(WnConv1d {
        weight,
        bias,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        out_ch: out,
        in_ch,
    })
}

impl WnConv1d {
    /// Channel-major `[in, seq]` -> `[out, seq']`.
    fn forward(&self, x: &[f32], _seq: usize) -> Vec<f32> {
        let bias = if self.bias.is_empty() {
            None
        } else {
            Some(&self.bias[..])
        };
        ops::conv1d(
            x,
            &self.weight,
            bias,
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

/// Weight-normalized transpose conv. Two checkpoint layouts occur:
/// - generator `ups.*`: `[in, out, kernel]` (PyTorch order), normalized
///   per input channel, groups 1;
/// - the AdaIN upsample `pool.*`: `[out, 1, kernel]` depthwise, one tap
///   triple per channel, normalized per channel.
struct WnConvTranspose1d {
    weight: Vec<f32>, // [in, out, kernel]
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl WnConvTranspose1d {
    fn forward(&self, x: &[f32], _seq: usize) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            Some(&self.bias),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            0,
            self.groups,
        )
    }
}

/// Loads a plain (non-weight-norm) conv whose checkpoint weight is
/// already `[out, in, kernel]` (the 1x1 projections).
fn load_plain_conv(file: &SafetensorsFile, base: &str) -> Result<WnConv1d> {
    let weight = file.load_as_f32(&format!("{base}.weight"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let (out, in_ch, kernel) = tensor_shape(file, &format!("{base}.weight"))?;
    Ok(WnConv1d {
        weight,
        bias,
        kernel,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        out_ch: out,
        in_ch,
    })
}

fn load_wn_conv_transpose(
    file: &SafetensorsFile,
    base: &str,
    stride: usize,
    padding: usize,
) -> Result<WnConvTranspose1d> {
    let v = file.load_as_f32(&format!("{base}.weight_v"))?;
    let g = file.load_as_f32(&format!("{base}.weight_g"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let (d0, d1, d2) = tensor_shape(file, &format!("{base}.weight_v"))?;
    let mut weight = v.clone();
    if d1 == 1 && d2 > 1 {
        // Depthwise pool: [out_ch, 1, kernel].
        let (ch, kernel) = (d0, d2);
        for c in 0..ch {
            let slice = &v[c * kernel..(c + 1) * kernel];
            let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            let scale = g[c] / (norm + 1e-7);
            for (k, &x) in slice.iter().enumerate() {
                weight[c * kernel + k] = x * scale;
            }
        }
        return Ok(WnConvTranspose1d {
            weight,
            bias,
            in_ch: ch,
            out_ch: ch,
            kernel,
            stride,
            padding,
            groups: ch,
        });
    }
    // [in, out, kernel], normalized per input channel.
    let (in_ch, out_ch, kernel) = (d0, d1, d2);
    for i in 0..in_ch {
        let slice = &v[i * out_ch * kernel..(i + 1) * out_ch * kernel];
        let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = g[i] / (norm + 1e-7);
        for (j, &x) in slice.iter().enumerate() {
            weight[i * out_ch * kernel + j] = x * scale;
        }
    }
    Ok(WnConvTranspose1d {
        weight,
        bias,
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        groups: 1,
    })
}

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// LayerNorm across channels at each timestep of a channel-major
/// `[channels, seq]` buffer (the reference normalizes the last axis of
/// its channels-last tensors).
fn layernorm_timesteps(x: &mut [f32], seq: usize, channels: usize, w: &[f32], b: &[f32], eps: f32) {
    let mut row = vec![0.0f32; channels];
    for t in 0..seq {
        for c in 0..channels {
            row[c] = x[c * seq + t];
        }
        let mean = row.iter().sum::<f32>() / channels as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..channels {
            x[c * seq + t] = (row[c] - mean) * inv * w[c] + b[c];
        }
    }
}

fn leaky_relu(v: f32, slope: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        v * slope
    }
}

/// Linear 1-D interpolation matching the reference `interpolate1d` with
/// `align_corners = None`: source position `(i + 0.5) * in/size - 0.5`,
/// clamped at zero, linear blend between neighbors. `input` is
/// `[in_width, channels]`; the result is `[size, channels]`.
fn interpolate_linear(input: &[f32], channels: usize, size: usize) -> Vec<f32> {
    let in_width = input.len() / channels;
    let mut out = vec![0.0f32; size * channels];
    if in_width == 1 {
        for i in 0..size {
            out[i * channels..(i + 1) * channels].copy_from_slice(&input[..channels]);
        }
        return out;
    }
    let scale = in_width as f32 / size as f32;
    for i in 0..size {
        let pos = (i as f32 * scale + 0.5 * scale - 0.5).max(0.0);
        let i0 = (pos.floor() as usize).min(in_width - 1);
        let i1 = (i0 + 1).min(in_width - 1);
        let frac = pos - i0 as f32;
        for c in 0..channels {
            let a = input[i0 * channels + c];
            let b = input[i1 * channels + c];
            out[i * channels + c] = a * (1.0 - frac) + b * frac;
        }
    }
    out
}

/// Deterministic-seed LCG for the vocoder's stochastic terms.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform() + 1e-7;
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

/// Bidirectional LSTM over `[seq, in_dim]`, gate order i, f, g, o with
/// tanh on g. Returns `[seq, 2 * hidden]`.
struct Lstm {
    wx_f: Vec<f32>,
    wh_f: Vec<f32>,
    b_f: Vec<f32>,
    wx_b: Vec<f32>,
    wh_b: Vec<f32>,
    b_b: Vec<f32>,
    in_dim: usize,
    hidden: usize,
}

impl Lstm {
    fn load(file: &SafetensorsFile, base: &str) -> Result<Self> {
        // Checkpoints carry PyTorch names; the reference sanitize maps
        // them to Wx/Wh/bias pairs. Accept either spelling.
        let named = |torch: &str, mlx: &str| -> Result<Vec<f32>> {
            let n = format!("{base}.{mlx}");
            if file.contains_tensor(&n) {
                return Ok(file.load_as_f32(&n)?);
            }
            Ok(file.load_as_f32(&format!("{base}.{torch}"))?)
        };
        let wx_f = named("weight_ih_l0", "Wx_forward")?;
        let wh_f = named("weight_hh_l0", "Wh_forward")?;
        let b_ih_f = named("bias_ih_l0", "bias_ih_forward")?;
        let b_hh_f = named("bias_hh_l0", "bias_hh_forward")?;
        let wx_b = named("weight_ih_l0_reverse", "Wx_backward")?;
        let wh_b = named("weight_hh_l0_reverse", "Wh_backward")?;
        let b_ih_b = named("bias_ih_l0_reverse", "bias_ih_backward")?;
        let b_hh_b = named("bias_hh_l0_reverse", "bias_hh_backward")?;
        // wx [4*hidden, in_dim], wh [4*hidden, hidden].
        let wx_shape = file
            .descriptor(&format!("{base}.weight_ih_l0"))
            .or_else(|| file.descriptor(&format!("{base}.Wx_forward")))
            .map(|d| (d.shape[0], d.shape[1]))
            .ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.weight_ih_l0"),
                why: "missing".to_string(),
            })?;
        let hidden = wx_shape.0 / 4;
        let in_dim = wx_shape.1;
        let b_f = b_ih_f.iter().zip(&b_hh_f).map(|(a, b)| a + b).collect();
        let b_b = b_ih_b.iter().zip(&b_hh_b).map(|(a, b)| a + b).collect();
        Ok(Lstm {
            wx_f,
            wh_f,
            b_f,
            wx_b,
            wh_b,
            b_b,
            in_dim,
            hidden,
        })
    }

    fn dir_forward(
        &self,
        x: &[f32],
        w_x: &[f32],
        w_h: &[f32],
        b: &[f32],
        backward: bool,
    ) -> Vec<f32> {
        let seq = x.len() / self.in_dim;
        let h = self.hidden;
        let mut proj = vec![0.0f32; seq * 4 * h];
        for (t, p_row) in proj.chunks_mut(4 * h).enumerate() {
            let x_row = &x[t * self.in_dim..(t + 1) * self.in_dim];
            for (g, p) in p_row.iter_mut().enumerate() {
                let w_row = &w_x[g * self.in_dim..(g + 1) * self.in_dim];
                *p = b[g] + dot(x_row, w_row);
            }
        }
        let mut hidden = vec![0.0f32; h];
        let mut cell = vec![0.0f32; h];
        let mut out = vec![0.0f32; seq * h];
        for step in 0..seq {
            let t = if backward { seq - 1 - step } else { step };
            let p_row = &proj[t * 4 * h..(t + 1) * 4 * h];
            let mut new_h = vec![0.0f32; h];
            for d in 0..h {
                let i = sigmoid(p_row[d] + dot(&hidden, &w_h[d * h..(d + 1) * h]));
                let f = sigmoid(p_row[h + d] + dot(&hidden, &w_h[(h + d) * h..(h + d + 1) * h]));
                let g = (p_row[2 * h + d]
                    + dot(&hidden, &w_h[(2 * h + d) * h..(2 * h + d + 1) * h]))
                .tanh();
                let o = sigmoid(
                    p_row[3 * h + d] + dot(&hidden, &w_h[(3 * h + d) * h..(3 * h + d + 1) * h]),
                );
                cell[d] = f * cell[d] + i * g;
                new_h[d] = o * cell[d].tanh();
            }
            hidden = new_h;
            out[t * h..(t + 1) * h].copy_from_slice(&hidden);
        }
        out
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.in_dim;
        let f = self.dir_forward(x, &self.wx_f, &self.wh_f, &self.b_f, false);
        let b = self.dir_forward(x, &self.wx_b, &self.wh_b, &self.b_b, true);
        let mut out = vec![0.0f32; seq * 2 * self.hidden];
        for t in 0..seq {
            let dst = &mut out[t * 2 * self.hidden..(t + 1) * 2 * self.hidden];
            dst[..self.hidden].copy_from_slice(&f[t * self.hidden..(t + 1) * self.hidden]);
            dst[self.hidden..].copy_from_slice(&b[t * self.hidden..(t + 1) * self.hidden]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Style-modulated norms and blocks
// ---------------------------------------------------------------------------

/// AdaLayerNorm on `[seq, channels]`: LayerNorm per step, then
/// `(1 + gamma) * x + beta` from the style.
struct AdaLayerNorm {
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
    channels: usize,
}

impl AdaLayerNorm {
    fn load(file: &SafetensorsFile, base: &str, channels: usize) -> Result<Self> {
        Ok(AdaLayerNorm {
            fc_w: file.load_as_f32(&format!("{base}.fc.weight"))?,
            fc_b: file.load_as_f32(&format!("{base}.fc.bias"))?,
            channels,
        })
    }

    fn forward(&self, x: &mut [f32], seq: usize, s: &[f32]) {
        let c = self.channels;
        let gb = ops::linear(s, &self.fc_w, Some(&self.fc_b), 1, s.len(), 2 * c);
        for row in x.chunks_mut(c).take(seq) {
            let mean = row.iter().sum::<f32>() / c as f32;
            let var = row.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / c as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for (i, a) in row.iter_mut().enumerate() {
                *a = (1.0 + gb[i]) * ((*a - mean) * inv) + gb[c + i];
            }
        }
    }
}

/// AdaIN1d on channel-major `[channels, seq]`: per-channel instance
/// norm over time, then style-modulated affine.
struct AdaIn1d {
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
    channels: usize,
}

impl AdaIn1d {
    fn load(file: &SafetensorsFile, base: &str, channels: usize) -> Result<Self> {
        Ok(AdaIn1d {
            fc_w: file.load_as_f32(&format!("{base}.fc.weight"))?,
            fc_b: file.load_as_f32(&format!("{base}.fc.bias"))?,
            channels,
        })
    }

    fn forward(&self, x: &mut [f32], seq: usize, s: &[f32]) {
        let c = self.channels;
        let gb = ops::linear(s, &self.fc_w, Some(&self.fc_b), 1, s.len(), 2 * c);
        for ch in 0..c {
            let row = &mut x[ch * seq..(ch + 1) * seq];
            let mean = row.iter().sum::<f32>() / seq as f32;
            let var = row.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / seq as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for a in row.iter_mut() {
                *a = (1.0 + gb[ch]) * ((*a - mean) * inv) + gb[c + ch];
            }
        }
    }
}

/// StyleTTS2 AdaIN residual block; channel-major `[channels, seq]`.
struct AdainResBlk1d {
    conv1: WnConv1d,
    conv2: WnConv1d,
    conv1x1: Option<WnConv1d>,
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    pool: Option<WnConvTranspose1d>,
    dim_in: usize,
}

impl AdainResBlk1d {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        dim_in: usize,
        dim_out: usize,
        _style_dim: usize,
        upsample: bool,
    ) -> Result<Self> {
        let conv1x1 = if dim_in != dim_out {
            Some(load_wn_conv(file, &format!("{base}.conv1x1"), 1, 0, 1, 1)?)
        } else {
            None
        };
        let pool = if upsample {
            Some(load_wn_conv_transpose(file, &format!("{base}.pool"), 2, 0)?)
        } else {
            None
        };
        Ok(AdainResBlk1d {
            conv1: load_wn_conv(file, &format!("{base}.conv1"), 1, 1, 1, 1)?,
            conv2: load_wn_conv(file, &format!("{base}.conv2"), 1, 1, 1, 1)?,
            conv1x1,
            norm1: AdaIn1d::load(file, &format!("{base}.norm1"), dim_in)?,
            norm2: AdaIn1d::load(file, &format!("{base}.norm2"), dim_out)?,
            pool,
            dim_in,
        })
    }

    fn forward(&self, x: &[f32], seq: usize, s: &[f32]) -> Vec<f32> {
        let out_seq = if self.pool.is_some() { seq * 2 } else { seq };
        // shortcut: nearest x2 upsample, then the 1x1 conv when the
        // channel count changes.
        let mut sc;
        if self.pool.is_some() {
            sc = vec![0.0f32; self.dim_in * out_seq];
            for c in 0..self.dim_in {
                for t in 0..seq {
                    let v = x[c * seq + t];
                    sc[c * out_seq + 2 * t] = v;
                    sc[c * out_seq + 2 * t + 1] = v;
                }
            }
        } else {
            sc = x.to_vec();
        }
        if let Some(c1x1) = &self.conv1x1 {
            sc = c1x1.forward(&sc, out_seq);
        }
        // residual
        let mut r = x.to_vec();
        self.norm1.forward(&mut r, seq, s);
        for v in r.iter_mut() {
            *v = leaky_relu(*v, 0.2);
        }
        if let Some(pool) = &self.pool {
            // Depthwise transpose conv k=3 s=2 unpadded gives 2T + 2
            // frames; the reference trims ONE FRAME off the left of
            // every channel (torch padding=1, output_padding=1).
            let up = pool.forward(&r, seq);
            let up_seq = (seq - 1) * pool.stride + pool.kernel;
            let mut trimmed = vec![0.0f32; self.dim_in * (up_seq - 1)];
            for c in 0..self.dim_in {
                trimmed[c * (up_seq - 1)..(c + 1) * (up_seq - 1)]
                    .copy_from_slice(&up[c * up_seq + 1..(c + 1) * up_seq]);
            }
            r = trimmed;
        }
        r = self.conv1.forward(&r, out_seq);
        self.norm2.forward(&mut r, out_seq, s);
        for v in r.iter_mut() {
            *v = leaky_relu(*v, 0.2);
        }
        r = self.conv2.forward(&r, out_seq);
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        for (a, b) in r.iter_mut().zip(&sc) {
            *a = (*a + b) * inv_sqrt2;
        }
        r
    }
}

/// iSTFTNet generator AdaIN residual block: three snake-conv pairs with
/// residual accumulation.
struct AdaInResBlock1 {
    convs1: Vec<WnConv1d>,
    convs2: Vec<WnConv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Vec<f32>>, // [block][channel]
    alpha2: Vec<Vec<f32>>,
}

impl AdaInResBlock1 {
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel geometry mirrors the checkpoint"
    )]
    fn load(
        file: &SafetensorsFile,
        base: &str,
        channels: usize,
        kernel: usize,
        dilations: &[usize; 3],
        _style_dim: usize,
    ) -> Result<Self> {
        let get_padding = |k: usize, d: usize| (k * d - d) / 2;
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        #[allow(
            clippy::needless_range_loop,
            reason = "the three snake-conv pairs are checkpoint-indexed"
        )]
        for i in 0..3 {
            convs1.push(load_wn_conv(
                file,
                &format!("{base}.convs1.{i}"),
                1,
                get_padding(kernel, dilations[i]),
                dilations[i],
                1,
            )?);
            convs2.push(load_wn_conv(
                file,
                &format!("{base}.convs2.{i}"),
                1,
                get_padding(kernel, 1),
                1,
                1,
            )?);
            adain1.push(AdaIn1d::load(
                file,
                &format!("{base}.adain1.{i}"),
                channels,
            )?);
            adain2.push(AdaIn1d::load(
                file,
                &format!("{base}.adain2.{i}"),
                channels,
            )?);
            // alphas are per-channel [1, C, 1] tensors.
            alpha1.push(file.load_as_f32(&format!("{base}.alpha1.{i}"))?);
            alpha2.push(file.load_as_f32(&format!("{base}.alpha2.{i}"))?);
        }
        Ok(AdaInResBlock1 {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    fn forward(&self, x: &[f32], seq: usize, s: &[f32]) -> Vec<f32> {
        let mut base = x.to_vec();
        let c = self.adain1[0].channels;
        for i in 0..3 {
            let mut xt = base.clone();
            self.adain1[i].forward(&mut xt, seq, s);
            let dump = std::env::var("KOKORO_BLK_DUMP").is_ok();
            if dump && i == 0 {
                std::env::remove_var("KOKORO_BLK_DUMP");
                let bytes: Vec<u8> = xt.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write("/tmp/rust_blk_a10.raw", bytes);
            }
            for ch in 0..c {
                let a1 = self.alpha1[i][ch];
                for v in &mut xt[ch * seq..(ch + 1) * seq] {
                    *v += (a1 * *v).sin().powi(2) / a1;
                }
            }
            if dump {
                let bytes: Vec<u8> = xt.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("/tmp/rust_blk_sn1{i}.raw"), bytes);
            }
            xt = self.convs1[i].forward(&xt, seq);
            if dump {
                let bytes: Vec<u8> = xt.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("/tmp/rust_blk_c1{i}.raw"), bytes);
            }
            self.adain2[i].forward(&mut xt, seq, s);
            for ch in 0..c {
                let a2 = self.alpha2[i][ch];
                for v in &mut xt[ch * seq..(ch + 1) * seq] {
                    *v += (a2 * *v).sin().powi(2) / a2;
                }
            }
            xt = self.convs2[i].forward(&xt, seq);
            if dump && i == 0 {
                let bytes: Vec<u8> = xt.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write("/tmp/rust_blk_c20.raw", bytes);
            }
            for (a, b) in xt.iter_mut().zip(&base) {
                *a += b;
            }
            base = xt;
        }
        std::env::remove_var("KOKORO_BLK_DUMP");
        base
    }
}

// ---------------------------------------------------------------------------
// PLBert (ALBERT)
// ---------------------------------------------------------------------------

#[cfg_attr(test, derive(Default))]
struct Albert {
    word_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    type_emb: Vec<f32>,
    emb_ln_w: Vec<f32>,
    emb_ln_b: Vec<f32>,
    mapping_w: Vec<f32>,
    mapping_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    attn_dense_w: Vec<f32>,
    attn_dense_b: Vec<f32>,
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    ffn_w: Vec<f32>,
    ffn_b: Vec<f32>,
    ffn_out_w: Vec<f32>,
    ffn_out_b: Vec<f32>,
    ffn_ln_w: Vec<f32>,
    ffn_ln_b: Vec<f32>,
    embedding_size: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    layers: usize,
    #[allow(dead_code, reason = "bounds the position table")]
    max_pos: usize,
}

impl Albert {
    fn load(file: &SafetensorsFile, plbert: &serde_json::Value) -> Result<Self> {
        let p = |k: &str| -> String {
            if file.contains_tensor(&format!("bert.{k}")) {
                format!("bert.{k}")
            } else {
                k.to_string()
            }
        };
        let f = |k: &str| -> Result<Vec<f32>> {
            let n = p(k);
            file.load_as_f32(&n).map_err(|e| SpeechError::Tensor {
                name: n,
                why: e.to_string(),
            })
        };
        let num = |key: &str, default: usize| {
            plbert
                .get(key)
                .and_then(|v| v.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let hidden = num("hidden_size", 768);
        let intermediate = num("intermediate_size", 2048);
        let layer = "encoder.albert_layer_groups.0.albert_layers.0";
        Ok(Albert {
            embedding_size: num("embedding_size", 128),
            hidden,
            intermediate,
            heads: num("num_attention_heads", 12),
            layers: num("num_hidden_layers", 12),
            max_pos: num("max_position_embeddings", 512),
            word_emb: f("embeddings.word_embeddings.weight")?,
            pos_emb: f("embeddings.position_embeddings.weight")?,
            type_emb: f("embeddings.token_type_embeddings.weight")?,
            emb_ln_w: f("embeddings.LayerNorm.weight")?,
            emb_ln_b: f("embeddings.LayerNorm.bias")?,
            mapping_w: f("encoder.embedding_hidden_mapping_in.weight")?,
            mapping_b: f("encoder.embedding_hidden_mapping_in.bias")?,
            q_w: f(&format!("{layer}.attention.query.weight"))?,
            q_b: f(&format!("{layer}.attention.query.bias"))?,
            k_w: f(&format!("{layer}.attention.key.weight"))?,
            k_b: f(&format!("{layer}.attention.key.bias"))?,
            v_w: f(&format!("{layer}.attention.value.weight"))?,
            v_b: f(&format!("{layer}.attention.value.bias"))?,
            attn_dense_w: f(&format!("{layer}.attention.dense.weight"))?,
            attn_dense_b: f(&format!("{layer}.attention.dense.bias"))?,
            attn_ln_w: f(&format!("{layer}.attention.LayerNorm.weight"))?,
            attn_ln_b: f(&format!("{layer}.attention.LayerNorm.bias"))?,
            ffn_w: f(&format!("{layer}.ffn.weight"))?,
            ffn_b: f(&format!("{layer}.ffn.bias"))?,
            ffn_out_w: f(&format!("{layer}.ffn_output.weight"))?,
            ffn_out_b: f(&format!("{layer}.ffn_output.bias"))?,
            ffn_ln_w: f(&format!("{layer}.full_layer_layer_norm.weight"))?,
            ffn_ln_b: f(&format!("{layer}.full_layer_layer_norm.bias"))?,
        })
    }

    /// Token ids -> sequence output `[seq, hidden]`.
    fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let seq = ids.len();
        let e = self.embedding_size;
        if seq == 0
            || seq > self.max_pos
            || e == 0
            || seq.checked_mul(e).is_none_or(|n| n > self.pos_emb.len())
            || self.type_emb.len() < e
            || ids.iter().any(|&id| {
                (id as usize)
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(e))
                    .is_none_or(|n| n > self.word_emb.len())
            })
        {
            return Err(SpeechError::Input {
                why: "phoneme IDs must fit the embedding and position tables".to_string(),
            });
        }
        let mut x = vec![0.0f32; seq * e];
        for (t, &id) in ids.iter().enumerate() {
            for d in 0..e {
                x[t * e + d] =
                    self.word_emb[id as usize * e + d] + self.pos_emb[t * e + d] + self.type_emb[d];
            }
        }
        ops::layernorm(&mut x, seq, e, &self.emb_ln_w, Some(&self.emb_ln_b), 1e-12);
        let mut x = ops::linear(
            &x,
            &self.mapping_w,
            Some(&self.mapping_b),
            seq,
            e,
            self.hidden,
        );
        let head_dim = self.hidden / self.heads;
        let scale = (head_dim as f32).sqrt().recip();
        let mut q_head = vec![0.0f32; seq * head_dim];
        let mut k_head = vec![0.0f32; seq * head_dim];
        let mut v_head = vec![0.0f32; seq * head_dim];
        for _li in 0..self.layers {
            let q = ops::linear(
                &x,
                &self.q_w,
                Some(&self.q_b),
                seq,
                self.hidden,
                self.hidden,
            );
            let k = ops::linear(
                &x,
                &self.k_w,
                Some(&self.k_b),
                seq,
                self.hidden,
                self.hidden,
            );
            let v = ops::linear(
                &x,
                &self.v_w,
                Some(&self.v_b),
                seq,
                self.hidden,
                self.hidden,
            );
            let mut attn = vec![0.0f32; seq * self.hidden];
            for h in 0..self.heads {
                for t in 0..seq {
                    let base = t * self.hidden + h * head_dim;
                    q_head[t * head_dim..(t + 1) * head_dim]
                        .copy_from_slice(&q[base..base + head_dim]);
                    k_head[t * head_dim..(t + 1) * head_dim]
                        .copy_from_slice(&k[base..base + head_dim]);
                    v_head[t * head_dim..(t + 1) * head_dim]
                        .copy_from_slice(&v[base..base + head_dim]);
                }
                let o = ops::sdpa(
                    &q_head, &k_head, &v_head, None, seq, seq, head_dim, head_dim, scale,
                );
                for t in 0..seq {
                    let dst = t * self.hidden + h * head_dim;
                    attn[dst..dst + head_dim].copy_from_slice(&o[t * head_dim..(t + 1) * head_dim]);
                }
            }
            let mut attn = ops::linear(
                &attn,
                &self.attn_dense_w,
                Some(&self.attn_dense_b),
                seq,
                self.hidden,
                self.hidden,
            );
            for (a, r) in attn.iter_mut().zip(&x) {
                *a += r;
            }
            ops::layernorm(
                &mut attn,
                seq,
                self.hidden,
                &self.attn_ln_w,
                Some(&self.attn_ln_b),
                1e-12,
            );
            let mut ffn = ops::linear(
                &attn,
                &self.ffn_w,
                Some(&self.ffn_b),
                seq,
                self.hidden,
                self.intermediate,
            );
            ops::gelu_erf(&mut ffn);
            let mut out = ops::linear(
                &ffn,
                &self.ffn_out_w,
                Some(&self.ffn_out_b),
                seq,
                self.intermediate,
                self.hidden,
            );
            for (o, a) in out.iter_mut().zip(&attn) {
                *o += a;
            }
            ops::layernorm(
                &mut out,
                seq,
                self.hidden,
                &self.ffn_ln_w,
                Some(&self.ffn_ln_b),
                1e-12,
            );
            x = out;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Text / prosody encoders
// ---------------------------------------------------------------------------

/// DurationEncoder inside the prosody predictor: alternating
/// BLSTM / AdaLayerNorm with style-concatenated LSTM input.
struct DurationEncoder {
    lstms: Vec<Lstm>,
    adalns: Vec<AdaLayerNorm>,
    d_model: usize,
    style_dim: usize,
}

impl DurationEncoder {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        style_dim: usize,
        d_model: usize,
        nlayers: usize,
    ) -> Result<Self> {
        let mut lstms = Vec::new();
        let mut adalns = Vec::new();
        for i in 0..nlayers {
            lstms.push(Lstm::load(file, &format!("{base}.lstms.{}", 2 * i))?);
            adalns.push(AdaLayerNorm::load(
                file,
                &format!("{base}.lstms.{}", 2 * i + 1),
                d_model,
            )?);
        }
        Ok(DurationEncoder {
            lstms,
            adalns,
            d_model,
            style_dim,
        })
    }

    /// `x` channel-major `[d_model, seq]`, style `[style_dim]` ->
    /// `[seq, d_model]`.
    fn forward(&self, x: &[f32], seq: usize, s: &[f32]) -> Result<Vec<f32>> {
        let d = self.d_model;
        let sty = self.style_dim;
        let mut feats = vec![0.0f32; seq * d];
        for t in 0..seq {
            for c in 0..d {
                feats[t * d + c] = x[c * seq + t];
            }
        }
        for i in 0..self.lstms.len() {
            let mut inp = vec![0.0f32; seq * (d + sty)];
            for t in 0..seq {
                inp[t * (d + sty)..t * (d + sty) + d].copy_from_slice(&feats[t * d..(t + 1) * d]);
                inp[t * (d + sty) + d..(t + 1) * (d + sty)].copy_from_slice(s);
            }
            let mut out = self.lstms[i].forward(&inp);
            self.adalns[i].forward(&mut out, seq, s);
            feats = out;
        }
        // The reference's final concat re-appends the style features,
        // so the returned width is d_model + style_dim.
        let mut out = vec![0.0f32; seq * (d + sty)];
        for t in 0..seq {
            out[t * (d + sty)..t * (d + sty) + d].copy_from_slice(&feats[t * d..(t + 1) * d]);
            out[t * (d + sty) + d..(t + 1) * (d + sty)].copy_from_slice(s);
        }
        Ok(out)
    }
}

/// Text encoder: embedding, depth-3 weighted-conv / LayerNorm /
/// LeakyReLU blocks, BLSTM. Output channel-major `[channels, seq]`.
struct TextEncoder {
    embedding: Vec<f32>,
    channels: usize,
    convs: Vec<WnConv1d>,
    lns: Vec<(Vec<f32>, Vec<f32>)>,
    lstm: Lstm,
}

impl TextEncoder {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        channels: usize,
        kernel: usize,
        depth: usize,
    ) -> Result<Self> {
        let padding = (kernel - 1) / 2;
        let mut convs = Vec::new();
        let mut lns = Vec::new();
        for i in 0..depth {
            convs.push(load_wn_conv(
                file,
                &format!("{base}.cnn.{i}.0"),
                1,
                padding,
                1,
                1,
            )?);
            // Checkpoints carry the gamma/beta spellings; accept both.
            let ln_base = format!("{base}.cnn.{i}.1");
            let (w_name, b_name) = if file.contains_tensor(&format!("{ln_base}.weight")) {
                (format!("{ln_base}.weight"), format!("{ln_base}.bias"))
            } else {
                (format!("{ln_base}.gamma"), format!("{ln_base}.beta"))
            };
            lns.push((file.load_as_f32(&w_name)?, file.load_as_f32(&b_name)?));
        }
        Ok(TextEncoder {
            embedding: file.load_as_f32(&format!("{base}.embedding.weight"))?,
            channels,
            convs,
            lns,
            lstm: Lstm::load(file, &format!("{base}.lstm"))?,
        })
    }

    fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let seq = ids.len();
        let c = self.channels;
        let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let emb = ops::embedding(&self.embedding, c, &ids_i32);
        let mut xc = vec![0.0f32; seq * c];
        for (t, row) in emb.chunks(c).enumerate() {
            for (ch, &v) in row.iter().enumerate() {
                xc[ch * seq + t] = v;
            }
        }
        for i in 0..self.convs.len() {
            let mut h = self.convs[i].forward(&xc, seq);
            // The reference LayerNorms run on channels-last tensors, so
            // they normalize across the 512 channels per timestep; on
            // the channel-major buffer that is a column-wise norm.
            layernorm_timesteps(&mut h, seq, c, &self.lns[i].0, &self.lns[i].1, 1e-5);
            for v in h.iter_mut() {
                *v = leaky_relu(*v, 0.2);
            }
            xc = h;
        }
        let mut feats = vec![0.0f32; seq * c];
        for t in 0..seq {
            for ch in 0..c {
                feats[t * c + ch] = xc[ch * seq + t];
            }
        }
        let out = self.lstm.forward(&feats);
        let row_width = 2 * self.lstm.hidden;
        let mut outc = vec![0.0f32; seq * c];
        for (t, row) in out.chunks(row_width).enumerate() {
            for (ch, &v) in row.iter().enumerate() {
                outc[ch * seq + t] = v;
            }
        }
        Ok(outc)
    }
}

/// Prosody predictor: duration path plus the shared F0/noise path.
struct ProsodyPredictor {
    text_encoder: DurationEncoder,
    lstm: Lstm,
    duration_proj_w: Vec<f32>,
    duration_proj_b: Vec<f32>,
    shared: Lstm,
    f0_blocks: Vec<AdainResBlk1d>,
    n_blocks: Vec<AdainResBlk1d>,
    f0_proj: WnConv1d,
    n_proj: WnConv1d,
    d_hid: usize,
    style_dim: usize,
}

impl ProsodyPredictor {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        style_dim: usize,
        d_hid: usize,
        nlayers: usize,
    ) -> Result<Self> {
        Ok(ProsodyPredictor {
            text_encoder: DurationEncoder::load(
                file,
                &format!("{base}.text_encoder"),
                style_dim,
                d_hid,
                nlayers,
            )?,
            lstm: Lstm::load(file, &format!("{base}.lstm"))?,
            duration_proj_w: file
                .load_as_f32(&format!("{base}.duration_proj.linear_layer.weight"))?,
            duration_proj_b: file
                .load_as_f32(&format!("{base}.duration_proj.linear_layer.bias"))?,
            shared: Lstm::load(file, &format!("{base}.shared"))?,
            f0_blocks: vec![
                AdainResBlk1d::load(
                    file,
                    &format!("{base}.F0.0"),
                    d_hid,
                    d_hid,
                    style_dim,
                    false,
                )?,
                AdainResBlk1d::load(
                    file,
                    &format!("{base}.F0.1"),
                    d_hid,
                    d_hid / 2,
                    style_dim,
                    true,
                )?,
                AdainResBlk1d::load(
                    file,
                    &format!("{base}.F0.2"),
                    d_hid / 2,
                    d_hid / 2,
                    style_dim,
                    false,
                )?,
            ],
            n_blocks: vec![
                AdainResBlk1d::load(file, &format!("{base}.N.0"), d_hid, d_hid, style_dim, false)?,
                AdainResBlk1d::load(
                    file,
                    &format!("{base}.N.1"),
                    d_hid,
                    d_hid / 2,
                    style_dim,
                    true,
                )?,
                AdainResBlk1d::load(
                    file,
                    &format!("{base}.N.2"),
                    d_hid / 2,
                    d_hid / 2,
                    style_dim,
                    false,
                )?,
            ],
            f0_proj: load_plain_conv(file, &format!("{base}.F0_proj"))?,
            n_proj: load_plain_conv(file, &format!("{base}.N_proj"))?,
            d_hid,
            style_dim,
        })
    }

    /// Duration path: `d_en` channel-major `[d_hid, seq]` + style ->
    /// per-phoneme durations (sigmoid sum / speed, rounded, clipped).
    fn durations(&self, d_en: &[f32], seq: usize, s: &[f32], speed: f32) -> Result<Vec<i32>> {
        let d = self.text_encoder.forward(d_en, seq, s)?;
        let lstm_out = self.lstm.forward(&d);
        let classes = self.duration_proj_w.len() / self.d_hid;
        let logits = ops::linear(
            &lstm_out,
            &self.duration_proj_w,
            Some(&self.duration_proj_b),
            seq,
            self.d_hid,
            classes,
        );
        let mut out = Vec::with_capacity(seq);
        for t in 0..seq {
            let row = &logits[t * classes..(t + 1) * classes];
            let sum: f32 = row.iter().map(|v| sigmoid(*v)).sum::<f32>() / speed;
            let v = if sum.is_nan() {
                1.0
            } else if sum.is_infinite() {
                MAX_FRAMES_PER_PHONEME as f32
            } else {
                sum
            };
            out.push(v.round().clamp(1.0, MAX_FRAMES_PER_PHONEME as f32) as i32);
        }
        Ok(out)
    }

    /// F0/noise prediction from aligned features `en` channel-major
    /// `[d_hid, frames]` and style. Returns `(F0, N)`, each
    /// `[2 * frames]` (one upsample stage doubles the length).
    fn f0_n_train(&self, en: &[f32], frames: usize, s: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.d_hid;
        let sty = self.style_dim;
        let mut feats = vec![0.0f32; frames * (d + sty)];
        for t in 0..frames {
            for c in 0..d {
                feats[t * (d + sty) + c] = en[c * frames + t];
            }
            feats[t * (d + sty) + d..(t + 1) * (d + sty)].copy_from_slice(s);
        }
        let shared_out = self.shared.forward(&feats); // [frames, d]
        let branch = |blocks: &[AdainResBlk1d], proj: &WnConv1d| -> Result<Vec<f32>> {
            let mut cm = vec![0.0f32; frames * d];
            for t in 0..frames {
                for (c, v) in shared_out[t * d..(t + 1) * d].iter().enumerate() {
                    cm[c * frames + t] = *v;
                }
            }
            let mut cur = cm;
            for b in blocks.iter() {
                let seq = cur.len() / b.dim_in;
                cur = b.forward(&cur, seq, s);
            }
            Ok(proj.forward(&cur, frames))
        };
        let f0 = branch(&self.f0_blocks, &self.f0_proj)?;
        let n = branch(&self.n_blocks, &self.n_proj)?;
        Ok((f0, n))
    }
}

// ---------------------------------------------------------------------------
// Harmonic source + generator + decoder
// ---------------------------------------------------------------------------

/// SineGen + SourceModuleHnNSF: harmonic excitation from upsampled F0.
struct HarmonicSource {
    l_linear_w: Vec<f32>, // [1, 9]
    l_linear_b: f32,
}

impl HarmonicSource {
    fn load(file: &SafetensorsFile, base: &str) -> Result<Self> {
        Ok(HarmonicSource {
            l_linear_w: file.load_as_f32(&format!("{base}.l_linear.weight"))?,
            l_linear_b: file.load_as_f32(&format!("{base}.l_linear.bias"))?[0],
        })
    }

    /// `f0_up` nearest-upsampled F0 `[n]` -> sine merge `[n]`.
    fn forward(&self, f0_up: &[f32], rng: &mut Lcg) -> Vec<f32> {
        let n = f0_up.len();
        let dim = HARMONICS + 1;
        // rad = (harmonic f0 / sr) % 1; frame 0 gets random initial
        // phase on harmonics only (the fundamental stays clean).
        let mut rad = vec![0.0f32; n * dim];
        for t in 0..n {
            for h in 0..dim {
                // Python-style floor modulus: MLX's % wraps negatives
                // into [0, 1), which the raw (sometimes negative) F0
                // predictions rely on in unvoiced regions.
                rad[t * dim + h] = (f0_up[t] * (h + 1) as f32 / SAMPLE_RATE).rem_euclid(1.0);
            }
        }
        #[allow(
            clippy::needless_range_loop,
            reason = "the harmonics are tensor lanes, not a collection"
        )]
        for h in 1..dim {
            rad[h] += rng.uniform();
        }
        // Down to frame rate (linear), cumsum, phase ramp, back up.
        // The reference's interpolate derives the size with float math:
        // ceil(n * (1/300.0)) which rounds to 127 frames for n = 37800
        // (the f64 product lands just above 126.0), so reproduce the
        // float ceil rather than integer division.
        let small = (((n as f64) * (1.0f64 / TOTAL_UPSAMPLE as f64)).ceil() as usize).max(1);
        let rad_small = interpolate_linear(&rad, dim, small);
        let mut phase = vec![0.0f32; small * dim];
        for h in 0..dim {
            let mut acc = 0.0f32;
            for s in 0..small {
                acc += rad_small[s * dim + h];
                phase[s * dim + h] = acc * 2.0 * std::f32::consts::PI;
            }
        }
        // The upsample leg targets ceil(small * 300) (38100 here); the
        // reference then truncates the sines to the f0 length. The
        // phase is scaled by the upsample factor BEFORE interpolation:
        // linear interpolation then ramps the phase across each frame
        // at the sample rate (the frame-rate cumsum alone only advances
        // one step per 300 samples).
        let up_n = ((small as f64) * (TOTAL_UPSAMPLE as f64)).ceil() as usize;
        for v in phase.iter_mut() {
            *v *= TOTAL_UPSAMPLE as f32;
        }
        let phase_up_full = interpolate_linear(&phase, dim, up_n);
        let phase_up = &phase_up_full[..(n * dim).min(phase_up_full.len())];
        // uv gating, noise, harmonic merge.
        let mut merge_in = vec![0.0f32; n * dim];
        for t in 0..n {
            let uv = (f0_up[t] > VOICED_THRESHOLD) as u8 as f32;
            let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
            for h in 0..dim {
                let wave = phase_up[t * dim + h].sin() * SINE_AMP;
                merge_in[t * dim + h] = wave * uv + noise_amp * rng.normal();
            }
        }
        let mut merge = vec![0.0f32; n];
        for t in 0..n {
            merge[t] =
                (dot(&merge_in[t * dim..(t + 1) * dim], &self.l_linear_w) + self.l_linear_b).tanh();
        }
        merge
    }
}

fn reflection_pad_left(x: &[f32], ch: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / ch;
    let mut out = vec![0.0f32; ch * (seq + pad)];
    for c in 0..ch {
        for t in 0..pad {
            let src = (pad - t).min(seq - 1);
            out[c * (seq + pad) + t] = x[c * seq + src];
        }
        out[c * (seq + pad) + pad..c * (seq + pad) + pad + seq]
            .copy_from_slice(&x[c * seq..(c + 1) * seq]);
    }
    out
}

/// Numpy-style phase unwrap along the frame axis of `[bins, frames]`.
fn unwrap_phase(phase: &[f32], bins: usize, frames: usize) -> Vec<f32> {
    let period = 2.0 * std::f32::consts::PI;
    let discont = period / 2.0;
    let mut out = phase.to_vec();
    let mut acc_all = vec![0.0f32; bins * frames];
    for b in 0..bins {
        let mut acc = 0.0f32;
        for t in 1..frames {
            let dd = out[b * frames + t] - out[b * frames + t - 1];
            let mut dd_mod = dd - period * ((dd + period / 2.0) / period).floor();
            if (dd - period / 2.0).abs() < 1e-10 && dd > 0.0 {
                dd_mod = period / 2.0;
            }
            let mut ph = dd_mod - dd;
            if dd.abs() < discont {
                ph = 0.0;
            }
            acc += ph;
            acc_all[b * frames + t] = acc;
        }
    }
    for (o, c) in out.iter_mut().zip(&acc_all) {
        *o += c;
    }
    out
}

/// iSTFTNet generator.
struct Generator {
    m_source: HarmonicSource,
    ups: Vec<WnConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<WnConv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: WnConv1d,
    #[allow(
        dead_code,
        reason = "mirrors the config; channel math derives from weights"
    )]
    upsample_initial_channel: usize,
}

impl Generator {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        style_dim: usize,
        istftnet: &serde_json::Value,
    ) -> Result<Self> {
        let nums = |key: &str| -> Vec<usize> {
            istftnet
                .get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default()
        };
        let resblock_kernel_sizes = nums("resblock_kernel_sizes");
        // [[1,3,5],[1,3,5],[1,3,5]] flattens to nine entries.
        let resblock_dilation_sizes: Vec<usize> = istftnet
            .get("resblock_dilation_sizes")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_array())
                    .flat_map(|inner| inner.iter().filter_map(|x| x.as_u64()).map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_else(|| vec![1, 3, 5, 1, 3, 5, 1, 3, 5]);
        let upsample_rates = nums("upsample_rates");
        let upsample_kernel_sizes = nums("upsample_kernel_sizes");
        let upsample_initial_channel = istftnet
            .get("upsample_initial_channel")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let mut ups = Vec::new();
        for (i, k) in upsample_kernel_sizes.iter().enumerate() {
            ups.push(load_wn_conv_transpose(
                file,
                &format!("{base}.ups.{i}"),
                upsample_rates[i],
                (k - upsample_rates[i]) / 2,
            )?);
        }
        let mut resblocks = Vec::new();
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..ups.len() {
            let ch = upsample_initial_channel / (1usize << (i + 1));
            for (j, k) in resblock_kernel_sizes.iter().enumerate() {
                let dils = [
                    resblock_dilation_sizes[j * 3],
                    resblock_dilation_sizes[j * 3 + 1],
                    resblock_dilation_sizes[j * 3 + 2],
                ];
                resblocks.push(AdaInResBlock1::load(
                    file,
                    &format!("{base}.resblocks.{}", i * resblock_kernel_sizes.len() + j),
                    ch,
                    *k,
                    &dils,
                    style_dim,
                )?);
            }
            // Noise-source branch convs are plain (non-weight-norm)
            // convs; checkpoints carry PyTorch [out, in, kernel], the
            // layout the kernel consumes directly.
            let name = format!("{base}.noise_convs.{i}");
            let weight = file.load_as_f32(&format!("{name}.weight"))?;
            let b = file.load_as_f32(&format!("{name}.bias"))?;
            let (o, in_ch, kernel) = tensor_shape(file, &format!("{name}.weight"))?;
            let (conv_stride, conv_padding) = if i + 1 < upsample_rates.len() {
                let stride_f0: usize = upsample_rates[i + 1..].iter().product();
                (stride_f0, (stride_f0 + 1).div_ceil(2))
            } else {
                (1, 0)
            };
            noise_convs.push(WnConv1d {
                weight,
                bias: b,
                kernel,
                stride: conv_stride,
                padding: conv_padding,
                dilation: 1,
                groups: 1,
                out_ch: o,
                in_ch,
            });
            let (rk, rd) = if i + 1 < upsample_rates.len() {
                (7, [1usize, 3, 5])
            } else {
                (11, [1usize, 3, 5])
            };
            noise_res.push(AdaInResBlock1::load(
                file,
                &format!("{base}.noise_res.{i}"),
                ch,
                rk,
                &rd,
                style_dim,
            )?);
        }
        Ok(Generator {
            m_source: HarmonicSource::load(file, &format!("{base}.m_source"))?,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post: load_wn_conv(file, &format!("{base}.conv_post"), 1, 3, 1, 1)?,
            upsample_initial_channel,
        })
    }

    /// `x` decoder features channel-major `[512, seq]`, `f0` prosody
    /// curve `[2 * seq]` -> waveform samples.
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        s: &[f32],
        f0: &[f32],
        rng: &mut Lcg,
    ) -> Result<Vec<f32>> {
        // Nearest-upsample F0 by TOTAL_UPSAMPLE.
        let mut f0_up = vec![0.0f32; f0.len() * TOTAL_UPSAMPLE];
        for (t, &v) in f0.iter().enumerate() {
            for u in 0..TOTAL_UPSAMPLE {
                f0_up[t * TOTAL_UPSAMPLE + u] = v;
            }
        }
        let har_source = self.m_source.forward(&f0_up, rng);
        // STFT of the source: periodic Hann, center reflect.
        let opts = StftOptions {
            fft_size: ISTFT_N_FFT,
            hop: ISTFT_HOP,
            window: turbospark_audio::hann_window(ISTFT_N_FFT),
            center: true,
        };
        let spectra = stft(&har_source, &opts).map_err(|e| SpeechError::Audio(e.to_string()))?;
        let frames_t = spectra.len();
        let bins = spectra[0].len();
        // har channel-major [2 * bins, frames_t]: magnitude then phase.
        let mut har = vec![0.0f32; 2 * bins * frames_t];
        for (t, frame) in spectra.iter().enumerate() {
            for (b, z) in frame.iter().enumerate() {
                har[b * frames_t + t] = (z.re * z.re + z.im * z.im).sqrt();
                har[(bins + b) * frames_t + t] = z.im.atan2(z.re);
            }
        }
        let mut cur = x.to_vec();
        let mut cur_seq = seq;
        for i in 0..self.ups.len() {
            for v in cur.iter_mut() {
                *v = leaky_relu(*v, 0.1);
            }
            let x_source = self.noise_convs[i].forward(&har, frames_t);
            // The noise branch runs at the STFT frame rate, not the
            // decoder feature rate.
            let xs_seq = x_source.len() / self.noise_convs[i].out_ch;
            let x_source = self.noise_res[i].forward(&x_source, xs_seq, s);
            let up = self.ups[i].forward(&cur, cur_seq);
            cur_seq =
                (cur_seq - 1) * self.ups[i].stride + self.ups[i].kernel - 2 * self.ups[i].padding;
            let mut next = up;
            if i == self.ups.len() - 1 {
                next = reflection_pad_left(&next, self.ups[i].out_ch, 1);
                cur_seq += 1;
            }
            for (a, b) in next.iter_mut().zip(&x_source) {
                *a += b;
            }
            let mut acc = vec![0.0f32; next.len()];
            for j in 0..3 {
                let out = self.resblocks[i * 3 + j].forward(&next, cur_seq, s);
                for (a, b) in acc.iter_mut().zip(&out) {
                    *a += b;
                }
            }
            for a in acc.iter_mut() {
                *a /= 3.0;
            }
            cur = acc;
        }
        for v in cur.iter_mut() {
            *v = leaky_relu(*v, 0.01);
        }
        let post = self.conv_post.forward(&cur, cur_seq);
        // spec = exp(first half), phase = sin(second half).
        let mut spec = vec![0.0f32; bins * cur_seq];
        let mut phase = vec![0.0f32; bins * cur_seq];
        for b in 0..bins {
            for t in 0..cur_seq {
                spec[b * cur_seq + t] = post[b * cur_seq + t].exp();
                phase[b * cur_seq + t] = post[(bins + b) * cur_seq + t].sin();
            }
        }
        let phase_u = unwrap_phase(&phase, bins, cur_seq);
        let mut spectra_inv = Vec::with_capacity(cur_seq);
        for t in 0..cur_seq {
            let mut frame = Vec::with_capacity(bins);
            for b in 0..bins {
                let m = spec[b * cur_seq + t];
                let p = phase_u[b * cur_seq + t];
                // The DC and Nyquist bins of a real signal are purely
                // real; mx.fft.irfft discards their imaginary parts, so
                // match that here.
                if b == 0 || b == bins - 1 {
                    frame.push(ComplexF32::new(m * p.cos(), 0.0));
                } else {
                    frame.push(ComplexF32::new(m * p.cos(), m * p.sin()));
                }
            }
            spectra_inv.push(frame);
        }
        // length None in the reference: the full centered region,
        // (frames - 1) * hop samples.
        let original_len = (spectra_inv.len() - 1) * ISTFT_HOP;
        istft(&spectra_inv, &opts, original_len).map_err(|e| SpeechError::Audio(e.to_string()))
    }
}

/// The decoder: prosody conditioning convs, AdaIN encode/decode stack,
/// then the generator.
struct KokoroDecoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: WnConv1d,
    n_conv: WnConv1d,
    asr_res: WnConv1d,
    generator: Generator,
}

impl KokoroDecoder {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        style_dim: usize,
        dim_in: usize,
        istftnet: &serde_json::Value,
    ) -> Result<Self> {
        // Blocks: three 1090 -> 1024, then 1090 -> 512 with upsample.
        let decode_dims = [
            (1090usize, 1024usize),
            (1090, 1024),
            (1090, 1024),
            (1090, 512),
        ];
        let mut decode = Vec::new();
        for (i, (din, dout)) in decode_dims.iter().enumerate() {
            decode.push(AdainResBlk1d::load(
                file,
                &format!("{base}.decode.{i}"),
                *din,
                *dout,
                style_dim,
                i == 3,
            )?);
        }
        Ok(KokoroDecoder {
            encode: AdainResBlk1d::load(
                file,
                &format!("{base}.encode"),
                dim_in + 2,
                1024,
                style_dim,
                false,
            )?,
            decode,
            f0_conv: load_wn_conv(file, &format!("{base}.F0_conv"), 2, 1, 1, 1)?,
            n_conv: load_wn_conv(file, &format!("{base}.N_conv"), 2, 1, 1, 1)?,
            asr_res: load_wn_conv(file, &format!("{base}.asr_res.0"), 1, 0, 1, 1)?,
            generator: Generator::load(file, &format!("{base}.generator"), style_dim, istftnet)?,
        })
    }

    fn forward(
        &self,
        asr: &[f32],
        frames: usize,
        f0_curve: &[f32],
        n_curve: &[f32],
        s: &[f32],
        rng: &mut Lcg,
    ) -> Result<Vec<f32>> {
        let f0_c = self.f0_conv.forward(f0_curve, f0_curve.len());
        let n_c = self.n_conv.forward(n_curve, n_curve.len());
        let cond_seq = f0_c.len();
        // concat [asr (512), F0 (1), N (1)] channel-major
        let mut x = vec![0.0f32; asr.len() + 2 * cond_seq];
        x[..asr.len()].copy_from_slice(asr);
        x[asr.len()..asr.len() + cond_seq].copy_from_slice(&f0_c);
        x[asr.len() + cond_seq..].copy_from_slice(&n_c);
        x = self.encode.forward(&x, cond_seq, s);
        let asr_res = self.asr_res.forward(asr, frames);
        let mut res = true;
        let mut cur_seq = cond_seq;
        for block in &self.decode {
            if res {
                let mut joined = vec![0.0f32; x.len() + asr_res.len() + 2 * cond_seq];
                joined[..x.len()].copy_from_slice(&x);
                joined[x.len()..x.len() + asr_res.len()].copy_from_slice(&asr_res);
                joined[x.len() + asr_res.len()..x.len() + asr_res.len() + cond_seq]
                    .copy_from_slice(&f0_c);
                joined[x.len() + asr_res.len() + cond_seq..].copy_from_slice(&n_c);
                x = joined;
            }
            x = block.forward(&x, cur_seq, s);
            if block.pool.is_some() {
                res = false;
                cur_seq *= 2;
            }
        }
        self.generator.forward(&x, cur_seq, s, f0_curve, rng)
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// The loaded Kokoro model.
pub struct Kokoro {
    bert: Albert,
    bert_encoder_w: Vec<f32>,
    bert_encoder_b: Vec<f32>,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: KokoroDecoder,
    hidden_dim: usize,
    style_dim: usize,
    #[allow(dead_code, reason = "recorded from config for loader symmetry")]
    n_layer: usize,
    plbert_hidden: usize,
    vocab: HashMap<char, u32>,
    /// Fixed-seed generator for the vocoder's stochastic terms.
    pub seed: u64,
}

impl Kokoro {
    /// Opens a model directory: `config.json` plus the weights
    /// (`kokoro-v1_0.safetensors` or `model.safetensors`).
    pub fn open(dir: &Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(
                |e| SpeechError::BadConfig {
                    field: "config.json".to_string(),
                    why: e.to_string(),
                },
            )?)
            .map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let weights_name = ["kokoro-v1_0.safetensors", "model.safetensors"]
            .iter()
            .find(|n| dir.join(n).exists())
            .ok_or_else(|| SpeechError::BadConfig {
                field: "weights".to_string(),
                why: "no kokoro safetensors in the model directory".to_string(),
            })?
            .to_string();
        let file = SafetensorsFile::open(&dir.join(&weights_name))?;
        let plbert = config
            .get("plbert")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let istftnet = config
            .get("istftnet")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let hidden_dim = config
            .get("hidden_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let style_dim = config
            .get("style_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let n_layer = config.get("n_layer").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let kernel = config
            .get("text_encoder_kernel_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        let mut vocab = HashMap::new();
        if let Some(v) = config.get("vocab").and_then(|v| v.as_object()) {
            for (k, id) in v {
                if let (Some(ch), Some(id)) = (k.chars().next(), id.as_u64()) {
                    vocab.insert(ch, id as u32);
                }
            }
        }
        let bert = Albert::load(&file, &plbert)?;
        let plbert_hidden = bert.hidden;
        Ok(Kokoro {
            bert,
            bert_encoder_w: file.load_as_f32("bert_encoder.weight")?,
            bert_encoder_b: file.load_as_f32("bert_encoder.bias")?,
            predictor: ProsodyPredictor::load(&file, "predictor", style_dim, hidden_dim, n_layer)?,
            text_encoder: TextEncoder::load(&file, "text_encoder", hidden_dim, kernel, n_layer)?,
            decoder: KokoroDecoder::load(&file, "decoder", style_dim, hidden_dim, &istftnet)?,
            hidden_dim,
            style_dim,
            n_layer,
            plbert_hidden,
            vocab,
            seed: 0,
        })
    }

    /// Maps a phoneme string to input ids the way the reference does:
    /// keep characters present in the vocab, wrap with the blank id 0.
    pub fn phoneme_ids(&self, phonemes: &str) -> Vec<u32> {
        let mut ids = vec![0u32];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()));
        ids.push(0);
        ids
    }

    /// Synthesizes one segment: `phoneme_ids` (wrapped here if raw),
    /// `ref_s` the 256-dim style vector from the voice pack row for
    /// this phoneme length, `speed` the rate divisor. Returns 24 kHz
    /// mono samples.
    pub fn generate(&self, phoneme_ids: &[u32], ref_s: &[f32], speed: f32) -> Result<Vec<f32>> {
        if !speed.is_finite() || speed <= 0.0 || ref_s.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "speed must be finite and positive, and voice style must be finite"
                    .to_string(),
            });
        }
        let mut rng = Lcg(self.seed.wrapping_mul(2685821657736338717).wrapping_add(1));
        let seq = phoneme_ids.len();
        if ref_s.len() != 2 * self.style_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "ref_s must be {} values, got {}",
                    2 * self.style_dim,
                    ref_s.len()
                ),
            });
        }
        // PLBert duration features.
        let bert_out = self.bert.forward(phoneme_ids)?;
        let d_en = ops::linear(
            &bert_out,
            &self.bert_encoder_w,
            Some(&self.bert_encoder_b),
            seq,
            self.plbert_hidden,
            self.hidden_dim,
        );
        // transpose [seq, hidden] -> channel-major [hidden, seq]
        let mut d_en_c = vec![0.0f32; d_en.len()];
        for t in 0..seq {
            for c in 0..self.hidden_dim {
                d_en_c[c * seq + t] = d_en[t * self.hidden_dim + c];
            }
        }
        let s = &ref_s[self.style_dim..];
        let durations = self.predictor.durations(&d_en_c, seq, s, speed)?;
        // Alignment one-hot: rows = phonemes, cols = total frames.
        let frames: usize = durations.iter().map(|&d| d as usize).sum();
        let mut indices = Vec::with_capacity(frames);
        for (i, &n) in durations.iter().enumerate() {
            for _ in 0..n {
                indices.push(i);
            }
        }
        // en = predictor-text-encoder features aligned to frames.
        let d = self.predictor.text_encoder.forward(&d_en_c, seq, s)?;
        let d_channels = self.hidden_dim + self.style_dim;
        let mut en = vec![0.0f32; d_channels * frames];
        for (frame, &ph) in indices.iter().enumerate() {
            let row = &d[ph * d_channels..(ph + 1) * d_channels];
            for (c, &v) in row.iter().enumerate() {
                en[c * frames + frame] = v;
            }
        }
        let (f0_pred, n_pred) = self.predictor.f0_n_train(&en, frames, s)?;
        // Text encoder + alignment.
        let t_en = self.text_encoder.forward(phoneme_ids)?;
        let mut asr = vec![0.0f32; self.hidden_dim * frames];
        for (frame, &ph) in indices.iter().enumerate() {
            for c in 0..self.hidden_dim {
                asr[c * frames + frame] = t_en[c * seq + ph];
            }
        }
        // Decoder with the first half of the style.
        self.decoder.forward(
            &asr,
            frames,
            &f0_pred,
            &n_pred,
            &ref_s[..self.style_dim],
            &mut rng,
        )
    }
}

#[cfg(test)]
mod input_regression {
    use super::*;

    #[test]
    fn bert_refuses_empty_out_of_vocab_and_overlong_phoneme_inputs() {
        let bert = Albert {
            embedding_size: 2,
            hidden: 2,
            heads: 1,
            max_pos: 1,
            word_emb: vec![1.0, 3.0, 2.0, 4.0],
            pos_emb: vec![0.0; 2],
            type_emb: vec![0.0; 2],
            emb_ln_w: vec![1.0; 2],
            emb_ln_b: vec![0.0; 2],
            mapping_w: vec![1.0, 0.0, 0.0, 1.0],
            mapping_b: vec![0.0; 2],
            ..Default::default()
        };
        assert_eq!(bert.forward(&[0]).unwrap(), vec![-1.0, 1.0]);
        for ids in [&[][..], &[2][..], &[0, 0][..]] {
            assert!(bert.forward(ids).is_err());
        }
    }
}
