//! Shared layers and tensor plumbing for the chatterbox S3Gen submodels
//! (flow conformer, flow-matching decoder, HiFT vocoder, CAMPPlus
//! speaker encoder): the checkpoint-shaped linear/conv/LayerNorm loads
//! and the channel-major <-> row-major conversions. Named after the
//! `chatterbox/s3gen` reference directory they were transcribed from;
//! the layout conventions match the stepaudio2 codec port, which shares
//! the same reference modules.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::{Result, SpeechError};

/// Plain linear layer, `[out, in]` row-major weights, row-major
/// activations `[rows, in]` -> `[rows, out]`.
#[derive(Debug, Clone)]
pub(crate) struct SaLinear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaLinear {
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_dim])?)
        } else {
            None
        };
        Ok(SaLinear {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    pub fn load_no_bias(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        Ok(SaLinear {
            in_dim,
            out_dim,
            weight,
            bias: None,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        crate::ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.in_dim,
            self.out_dim,
        )
    }
}

/// Plain Conv1d loaded from the MLX checkpoint layout `[out, K, in]`
/// and stored PyTorch `[out, in, K]` for the crate kernel.
/// Channel-major `x [in_ch, seq]` -> `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub(crate) struct SaConv1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_g = in_ch / groups;
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kernel, in_g])?;
        let mut weight = vec![0.0f32; stored.len()];
        for oc in 0..out_ch {
            for kk in 0..kernel {
                for ic in 0..in_g {
                    weight[oc * in_g * kernel + ic * kernel + kk] =
                        stored[oc * kernel * in_g + kk * in_g + ic];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(SaConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        if std::env::var("TURBOSPARK_DEBUG_CONV").is_ok() {
            eprintln!(
                "[conv1d] {}ch->{}ch k{} s{} p{} in_len {}",
                self.in_ch,
                self.out_ch,
                self.kernel,
                self.stride,
                self.padding,
                x.len()
            );
        }
        crate::ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
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

/// ConvTranspose1d loaded from the MLX layout `[out, K, in]` (the
/// out-first transposed-conv layout, matching the reference sanitize
/// transpose `(1, 2, 0)` of the PyTorch `[in, out, K]` weight) and
/// stored PyTorch `[in, out, K]` for the crate kernel. Channel-major
/// `x [in_ch, seq]` -> `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub(crate) struct SaConvTr1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaConvTr1d {
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
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
        Ok(SaConvTr1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        crate::ops::conv_transpose1d(
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

/// Affine LayerNorm state.
#[derive(Debug, Clone)]
pub(crate) struct SaLayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl SaLayerNorm {
    pub fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        Ok(SaLayerNorm {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[dim])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[dim])?,
        })
    }

    /// In-place over row-major `[rows, dim]`.
    pub fn forward(&self, x: &mut [f32], rows: usize, eps: f32) {
        let dim = self.weight.len();
        crate::ops::layernorm(x, rows, dim, &self.weight, Some(&self.bias), eps);
    }
}

/// Affine-free LayerNorm over row-major `[rows, dim]` (the decoder
/// transformer blocks do not use this; the reference reads affine
/// LayerNorms there. Kept for the CausalBlock1D-free decoder variant
/// parity with the reference Block1D which always carries an affine
/// norm; unused fallback).
#[allow(dead_code)]
pub(crate) fn layernorm_noaffine(x: &mut [f32], rows: usize, dim: usize, eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv;
        }
    }
}

/// Channel-major <-> row-major conversions used across the submodels.
pub(crate) fn to_cm(rows: &[f32], rows_n: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows_n * cols];
    for r in 0..rows_n {
        for c in 0..cols {
            out[c * rows_n + r] = rows[r * cols + c];
        }
    }
    out
}

pub(crate) fn to_rows(cm: &[f32], ch: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * ch];
    for c in 0..ch {
        for f in 0..frames {
            out[f * ch + c] = cm[c * frames + f];
        }
    }
    out
}

pub(crate) fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

pub(crate) fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// Mish activation (`x * tanh(softplus(x))`), numerically stable.
pub(crate) fn mish(x: &mut [f32]) {
    for v in x.iter_mut() {
        let sp = v.max(0.0) + (1.0 + (-v.abs()).exp()).ln();
        *v *= sp.tanh();
    }
}

/// Torch-style reflect padding along the sequence axis of a
/// channel-major `[ch, seq]` signal (edge sample not repeated).
pub(crate) fn reflect_pad_cm(x: &[f32], ch: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / ch;
    let mut out = vec![0.0f32; ch * (seq + 2 * pad)];
    for c in 0..ch {
        let src = &x[c * seq..(c + 1) * seq];
        let dst = &mut out[c * (seq + 2 * pad)..(c + 1) * (seq + 2 * pad)];
        for p in 0..pad {
            dst[p] = src[pad - p];
            dst[pad + seq + p] = src[seq - 2 - p];
        }
        dst[pad..pad + seq].copy_from_slice(src);
    }
    out
}

/// The chatterbox s3gen 24 kHz prompt mel
/// (`s3gen/mel.py::mel_spectrogram` defaults): reflect-pad by
/// `(n_fft - hop) / 2`, uncentered STFT with a symmetric Hann window,
/// magnitude spectrum, Slaney filterbank, natural-log compression
/// floored at 1e-5. Returns frames as rows `[T', num_mels]`.
pub(crate) fn prompt_mel(
    samples: &[f32],
    num_mels: usize,
    sample_rate: u32,
) -> Result<Vec<Vec<f32>>> {
    use crate::mel::{self, MelScale};
    use crate::stft;
    const N_FFT: usize = 1920;
    const HOP: usize = 480;
    const WIN: usize = 1920;
    const FMIN: f32 = 0.0;
    const FMAX: f32 = 8000.0;
    let _ = sample_rate;
    let pad = (N_FFT - HOP) / 2;
    if samples.len() < pad + 2 {
        return Err(SpeechError::Input {
            why: format!("prompt waveform needs at least {pad} samples for the mel reflect pad"),
        });
    }
    let padded = reflect_pad_cm(samples, 1, pad);
    let options = stft::StftOptions {
        fft_size: N_FFT,
        hop: HOP,
        window: hann_symmetric(WIN),
        center: false,
    };
    let spectra = stft::stft(&padded, &options)?;
    let filterbank =
        mel::mel_filterbank(num_mels, N_FFT, 24_000, FMIN, Some(FMAX), MelScale::Slaney)?;
    Ok(spectra
        .into_iter()
        .map(|spectrum| {
            let mag: Vec<f32> = spectrum.iter().map(|c| c.re.hypot(c.im)).collect();
            filterbank
                .project(&mag)
                .map(|row| row.into_iter().map(|v| v.max(1e-5).ln()).collect())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

/// Symmetric Hann window (the `periodic=False` convention the
/// chatterbox mel front end uses; `dsp::hann_window` is the periodic
/// variant the STFT reconstruction paths need).
pub(crate) fn hann_symmetric(size: usize) -> Vec<f32> {
    match size {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..size)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos()))
                    as f32
            })
            .collect(),
    }
}
