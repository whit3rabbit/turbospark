//! The Higgs tokenizer's encode-side semantic path: the sinc resampler
//! (`_sinc_resample`, torchaudio-compatible `sinc_interp_hann`) and the
//! `SemanticEncoder` CNN that processes the Wav2Vec2 hidden-state mean
//! before fusion (`encoder_semantic.*` weights). Ported from
//! `mlx_audio/codec/models/higgs_audio/{higgs_audio,semantic}.py` at the
//! pinned commit; weights are the PyTorch layout the released
//! `audio_tokenizer/model.safetensors` ships.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::{Result, SpeechError};

/// Hann-windowed sinc interpolation, torchaudio
/// `functional.resample(..., method="sinc_interp_hann")` compatible:
/// `lowpass_filter_width 6`, `rolloff 0.99`. `waveform` is f32; the
/// kernel math runs in f64 up to the reference's single f32 cast.
pub(crate) fn sinc_resample(waveform: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    if orig_freq == new_freq {
        return waveform.to_vec();
    }
    let lowpass_filter_width = 6.0f64;
    let rolloff = 0.99f64;
    let gcd = gcd(orig_freq, new_freq);
    let orig_r = orig_freq / gcd;
    let new_r = new_freq / gcd;

    let base_freq = orig_r.min(new_r) as f64 * rolloff;
    let width = (lowpass_filter_width * orig_r as f64 / base_freq).ceil() as usize;

    // The kernel matrix is [new_r, 2 * width + orig_r]: one resampling
    // kernel per output phase (`np.arange(-width, width + orig_r)`).
    let kernel_len = 2 * width + orig_r;
    let mut kernel = vec![0.0f32; new_r * kernel_len];
    for phase in 0..new_r {
        for column in 0..kernel_len {
            let idx = (column as f64 - width as f64) / orig_r as f64;
            let t = (-(phase as f64) / new_r as f64 + idx) * base_freq;
            let t = t.clamp(-lowpass_filter_width, lowpass_filter_width);
            let window = (t * std::f64::consts::PI / lowpass_filter_width / 2.0)
                .cos()
                .powi(2);
            let t_pi = t * std::f64::consts::PI;
            let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };
            kernel[phase * kernel_len + column] =
                (sinc * window * (base_freq / orig_r as f64)) as f32;
        }
    }

    let length = waveform.len();
    let mut padded = vec![0.0f32; length + width + width + orig_r];
    padded[width..width + length].copy_from_slice(waveform);

    let out_len = (length as f64 * new_r as f64 / orig_r as f64).ceil() as usize;
    let mut result = vec![0.0f32; out_len];
    // `np.convolve(padded, kernel[phase, ::-1], mode="valid")` reduces to
    // the unreversed kernel sum `sum_q padded[step + q] * kernel[phase, q]`
    // accumulated in f32, then every `orig_r`-th valid step lands on the
    // `phase + i * new_r` output grid.
    for phase in 0..new_r.min(out_len) {
        let valid_len = padded.len() - kernel_len + 1;
        let samples_count = valid_len.div_ceil(orig_r);
        let n = samples_count.min((out_len - phase).div_ceil(new_r));
        for i in 0..n {
            let pos = phase + i * new_r;
            if pos >= out_len {
                continue;
            }
            let step = i * orig_r;
            let mut sum = 0.0f32;
            for q in 0..kernel_len {
                sum += padded[step + q] * kernel[phase * kernel_len + q];
            }
            result[pos] = sum;
        }
    }
    result
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// ELU residual unit: ELU, dilated k3 conv (bias-free, symmetric pad),
/// ELU, k1 conv (bias-free), skip-add. Weights
/// `encoder_semantic.conv_blocks.N.res_units.M.conv{1,2}.weight`.
struct SemanticResidualUnit {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SemanticResidualUnit {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        dilation: usize,
        kernel: usize,
    ) -> Result<Self> {
        let pad = (kernel - 1) * dilation / 2;
        Ok(SemanticResidualUnit {
            conv1: Conv1d {
                in_ch: channels,
                out_ch: channels,
                kernel,
                stride: 1,
                padding: pad,
                dilation,
                groups: 1,
                weight: load_f32_shaped(
                    file,
                    &format!("{prefix}.conv1.weight"),
                    &[channels, channels, kernel],
                )?,
                bias: None,
            },
            conv2: Conv1d {
                in_ch: channels,
                out_ch: channels,
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                weight: load_f32_shaped(
                    file,
                    &format!("{prefix}.conv2.weight"),
                    &[channels, channels, 1],
                )?,
                bias: None,
            },
        })
    }

    /// Rows `[T, C]` in, rows `[T, C]` out.
    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut activated = x.to_vec();
        crate::ops::elu(&mut activated);
        let convolved = conv_rows(&activated, steps, &self.conv1);
        let mut activated = convolved;
        crate::ops::elu(&mut activated);
        let mut y = conv_rows(&activated, steps, &self.conv2);
        for (slot, value) in y.iter_mut().zip(x) {
            *slot += value;
        }
        y
    }
}

/// One semantic block: two residual units then a strided bias conv
/// (`encoder_semantic.conv_blocks.N.{conv,res_units.*}`).
struct SemanticConvBlock {
    res_units: [SemanticResidualUnit; 2],
    conv: Conv1d,
}

impl SemanticConvBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
        dilation: usize,
        kernel: usize,
        unit_kernel_size: usize,
    ) -> Result<Self> {
        Ok(SemanticConvBlock {
            res_units: [
                SemanticResidualUnit::load(
                    file,
                    &format!("{prefix}.res_units.0"),
                    in_dim,
                    dilation,
                    unit_kernel_size,
                )?,
                SemanticResidualUnit::load(
                    file,
                    &format!("{prefix}.res_units.1"),
                    in_dim,
                    dilation,
                    unit_kernel_size,
                )?,
            ],
            conv: Conv1d {
                in_ch: in_dim,
                out_ch: out_dim,
                kernel,
                stride,
                padding: (kernel - 1) / 2,
                dilation: 1,
                groups: 1,
                weight: load_f32_shaped(
                    file,
                    &format!("{prefix}.conv.weight"),
                    &[out_dim, in_dim, kernel],
                )?,
                bias: load_bias(
                    file,
                    &format!("{prefix}.conv.bias"),
                    out_dim,
                    BiasMode::Required,
                )?,
            },
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let x = self.res_units[0].forward(x, steps);
        let x = self.res_units[1].forward(&x, steps);
        conv_rows(&x, steps, &self.conv)
    }
}

/// `encoder_semantic`: the entry conv (bias-free k3) followed by the
/// blocks. Everything runs channels-last `[T, hidden]`, matching the
/// reference tensors.
pub(crate) struct SemanticEncoder {
    conv: Conv1d,
    blocks: Vec<SemanticConvBlock>,
    hidden: usize,
}

impl SemanticEncoder {
    pub(crate) fn load(
        file: &SafetensorsFile,
        hidden: usize,
        strides: &[usize],
        dilations: &[usize],
        channel_ratios: &[usize],
        kernel_size: usize,
        unit_kernel_size: usize,
    ) -> Result<Self> {
        let conv = Conv1d {
            in_ch: hidden,
            out_ch: hidden,
            kernel: kernel_size,
            stride: 1,
            padding: (kernel_size - 1) / 2,
            dilation: 1,
            groups: 1,
            weight: load_f32_shaped(
                file,
                "encoder_semantic.conv.weight",
                &[hidden, hidden, kernel_size],
            )?,
            bias: None,
        };
        let mut blocks = Vec::new();
        for (index, &stride) in strides.iter().enumerate() {
            let ratio = channel_ratios.get(index).copied().unwrap_or(1);
            blocks.push(SemanticConvBlock::load(
                file,
                &format!("encoder_semantic.conv_blocks.{index}"),
                hidden * ratio,
                hidden * ratio,
                stride,
                dilations.get(index).copied().unwrap_or(1),
                kernel_size,
                unit_kernel_size,
            )?);
        }
        Ok(SemanticEncoder {
            conv,
            blocks,
            hidden,
        })
    }

    /// Rows `[T, hidden]` in, rows `[T', hidden]` out.
    pub(crate) fn forward(&self, x: &[f32], steps: usize) -> Result<Vec<f32>> {
        if x.len() != steps * self.hidden {
            return Err(SpeechError::Input {
                why: format!(
                    "semantic encoder expected {steps} x {} values, got {}",
                    self.hidden,
                    x.len()
                ),
            });
        }
        let x = conv_rows(x, steps, &self.conv);
        let mut steps = conv_out_steps(steps, &self.conv);
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, steps);
            steps = conv_out_steps(steps, &block.conv);
        }
        Ok(x)
    }

    pub(crate) fn output_steps(&self, steps: usize) -> usize {
        let mut steps = conv_out_steps(steps, &self.conv);
        for block in &self.blocks {
            steps = conv_out_steps(steps, &block.conv);
        }
        steps
    }
}

fn conv_out_steps(steps: usize, conv: &Conv1d) -> usize {
    (steps + 2 * conv.padding - conv.dilation * (conv.kernel - 1) - 1) / conv.stride + 1
}

/// Runs one channels-last `[T, C]` row buffer through a [`Conv1d`] by
/// transposing to channel-major for the kernel walk and back.
fn conv_rows(x: &[f32], steps: usize, conv: &Conv1d) -> Vec<f32> {
    let channels = conv.in_ch;
    let mut channel_major = vec![0.0f32; x.len()];
    for step in 0..steps {
        for channel in 0..channels {
            channel_major[channel * steps + step] = x[step * channels + channel];
        }
    }
    let out = conv.forward(&channel_major);
    let out_steps = conv_out_steps(steps, conv);
    let out_channels = conv.out_ch;
    let mut rows = vec![0.0f32; out.len()];
    for step in 0..out_steps {
        for channel in 0..out_channels {
            rows[step * out_channels + channel] = out[channel * out_steps + step];
        }
    }
    rows
}
