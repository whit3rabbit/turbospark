//! LASR CTC speech recognition (Google MedASR family).
//!
//! Reference: `mlx_audio/stt/models/lasr_ctc/` (lasr.py, config.py) at
//! mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! This family has no working upstream end-to-end path. The mlx-audio stock
//! `load()` runs `LasrForCTC.sanitize`, which transposes every 3-axis conv
//! weight to MLX `(out, kernel, in)` layout, but the public MLX MedASR
//! conversion already stores conv weights in that layout, so the stock path
//! double-transposes them and the model silently decodes garbage. The
//! upstream `LasrForCTC` also defines no `generate` method (the mlx-audio STT
//! generate entry point raises AttributeError) and its `decode` returns an
//! empty text. This port instead reads the checkpoint tensors in their stored
//! MLX layout directly, squeezes only the stored `(out, in, 1)` CTC head to a
//! linear, implements its own greedy CTC decode, and uses the transformers
//! `LASRProcessor` frontend contract (the only working feature path for this
//! family). These divergences are deliberate; see the family README.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::bad_config;
use crate::ops;
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
///
/// The official `google/medasr` repository is access gated (HTTP 403 in the
/// verification environment) and remains unverified. This public MLX fp32
/// conversion is the pinned candidate profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LasrCtcProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const MEDASR_MLX_FP32: LasrCtcProfile = LasrCtcProfile {
    name: "Google MedASR (MLX fp32 conversion)",
    repository: "drankush-ai/medasr-mlx-fp32",
    revision: "3b967580b5176144bc633fac60420d1b122dfba8",
};

/// LASR feature extractor geometry, pinned by the checkpoint's
/// `preprocessor_config.json` and the transformers `LASRFeatureExtractor`.
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const MEL_LOWER_HZ: f64 = 125.0;
const MEL_UPPER_HZ: f64 = 7500.0;
const MEL_CLAMP: f64 = 1e-5;

/// Batch normalization epsilon. Neither the checkpoint config nor the
/// upstream dataclass carries it; both MLX `nn.BatchNorm` and the
/// transformers `BatchNorm1d` the conversion mirrors default to 1e-5.
const BATCH_NORM_EPS: f32 = 1e-5;

fn positive(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|&number| number > 0)
        .ok_or_else(|| bad_config(field, "must be a positive integer"))
}

fn residual_weights(value: &Value, field: &str, default: [f32; 2]) -> Result<[f32; 2]> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Array(items)) if items.len() == 2 => {
            let mut out = [0.0f32; 2];
            for (slot, item) in out.iter_mut().zip(items) {
                *slot = item
                    .as_f64()
                    .filter(|number| number.is_finite())
                    .ok_or_else(|| bad_config(field, "must contain finite numbers"))?
                    as f32;
            }
            Ok(out)
        }
        Some(_) => Err(bad_config(field, "must be a pair of finite numbers")),
    }
}

/// Inference settings from the pinned LASR CTC checkpoint config.
#[derive(Debug, Clone, PartialEq)]
pub struct LasrCtcConfig {
    pub vocab_size: usize,
    pub pad_token_id: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub num_mel_bins: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub conv_residual_weights: [f32; 2],
    pub feed_forward_residual_weights: [f32; 2],
}

impl LasrCtcConfig {
    /// Parses the pinned `lasr_ctc` config and refuses anything this port has
    /// not verified against the reference: non-SiLU activations, biased
    /// attention or convolution linears, grouped-query attention, and
    /// non-default RoPE.
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("lasr_ctc") {
            return Err(bad_config(
                "model_type",
                "expected the lasr_ctc architecture",
            ));
        }
        let encoder = root
            .get("encoder_config")
            .filter(|value| value.is_object())
            .ok_or_else(|| bad_config("encoder_config", "must be an object"))?;
        let field = |name: &str| -> Result<&Value> {
            encoder
                .get(name)
                .ok_or_else(|| bad_config(name, "missing from encoder_config"))
        };
        if encoder.get("hidden_act").and_then(Value::as_str) != Some("silu") {
            return Err(SpeechError::Unsupported {
                why: "the lasr_ctc port implements the SiLU convolution and feed-forward activation only".into(),
            });
        }
        if encoder.get("attention_bias").and_then(Value::as_bool) != Some(false) {
            return Err(SpeechError::Unsupported {
                why: "the lasr_ctc port implements bias-free attention projections only".into(),
            });
        }
        if encoder.get("convolution_bias").and_then(Value::as_bool) != Some(false) {
            return Err(SpeechError::Unsupported {
                why: "the lasr_ctc port implements bias-free convolution module projections only"
                    .into(),
            });
        }
        let heads = positive(encoder, "num_attention_heads")?;
        if let Some(kv_heads) = encoder
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
        {
            if kv_heads != heads {
                return Err(SpeechError::Unsupported {
                    why: "the lasr_ctc port implements multi-head attention only, not grouped-query attention".into(),
                });
            }
        }
        let rope_type = encoder
            .get("rope_parameters")
            .and_then(|rope| rope.get("rope_type"))
            .and_then(Value::as_str);
        if let Some(rope_type) = rope_type {
            if rope_type != "default" {
                return Err(SpeechError::Unsupported {
                    why: format!(
                        "the lasr_ctc port implements the default RoPE only, not {rope_type}"
                    ),
                });
            }
        }
        let rope_theta = encoder
            .get("rope_parameters")
            .and_then(|rope| rope.get("rope_theta"))
            .and_then(Value::as_f64)
            .filter(|number| number.is_finite() && *number > 0.0)
            .unwrap_or(10_000.0) as f32;
        let layer_norm_eps = field("layer_norm_eps")?
            .as_f64()
            .filter(|number| number.is_finite() && *number > 0.0)
            .ok_or_else(|| bad_config("layer_norm_eps", "must be a positive number"))?
            as f32;
        let config = Self {
            vocab_size: positive(root, "vocab_size")?,
            pad_token_id: root
                .get("pad_token_id")
                .and_then(Value::as_u64)
                .and_then(|number| usize::try_from(number).ok())
                .unwrap_or(0),
            hidden_size: positive(encoder, "hidden_size")?,
            num_hidden_layers: positive(encoder, "num_hidden_layers")?,
            num_attention_heads: heads,
            intermediate_size: positive(encoder, "intermediate_size")?,
            conv_kernel_size: positive(encoder, "conv_kernel_size")?,
            subsampling_conv_channels: positive(encoder, "subsampling_conv_channels")?,
            subsampling_conv_kernel_size: positive(encoder, "subsampling_conv_kernel_size")?,
            subsampling_conv_stride: positive(encoder, "subsampling_conv_stride")?,
            num_mel_bins: positive(encoder, "num_mel_bins")?,
            layer_norm_eps,
            rope_theta,
            conv_residual_weights: residual_weights(encoder, "conv_residual_weights", [2.0, 1.0])?,
            feed_forward_residual_weights: residual_weights(
                encoder,
                "feed_forward_residual_weights",
                [1.5, 0.5],
            )?,
        };
        if config.pad_token_id >= config.vocab_size {
            return Err(bad_config(
                "pad_token_id",
                "must be smaller than vocab_size",
            ));
        }
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(bad_config("num_attention_heads", "must divide hidden_size"));
        }
        if config.subsampling_conv_stride == 0 || config.subsampling_conv_stride > 8 {
            return Err(bad_config(
                "subsampling_conv_stride",
                "must be between 1 and 8",
            ));
        }
        Ok(config)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Builds the lingvo-style kaldi-mel slope weight matrix the transformers
/// `LASRFeatureExtractor` projects power spectra onto, in float64 exactly
/// like the reference. Shape `[num_spectrogram_bins, num_mel_bins]` with the
/// DC bin row forced to zero.
fn lasr_mel_matrix(num_spectrogram_bins: usize, num_mel_bins: usize) -> Vec<f64> {
    let kaldi_mel = |hertz: f64| 1127.0 * (1.0 + hertz / 700.0).ln();
    // numpy linspace includes the endpoint by overwriting the last entry
    // with the exact stop value.
    let linspace = |start: f64, stop: f64, count: usize| -> Vec<f64> {
        let step = (stop - start) / (count - 1) as f64;
        (0..count)
            .map(|index| {
                if index + 1 == count {
                    stop
                } else {
                    start + step * index as f64
                }
            })
            .collect()
    };
    let nyquist = 16_000.0 / 2.0;
    // HTK excludes the spectrogram DC bin, then the zero row is padded back.
    let bin_mel: Vec<f64> = linspace(0.0, nyquist, num_spectrogram_bins)[1..]
        .iter()
        .map(|&hertz| kaldi_mel(hertz))
        .collect();
    let edges = linspace(
        kaldi_mel(MEL_LOWER_HZ),
        kaldi_mel(MEL_UPPER_HZ),
        num_mel_bins + 2,
    );
    let mut matrix = vec![0.0f64; num_spectrogram_bins * num_mel_bins];
    for (bin, &mel) in bin_mel.iter().enumerate() {
        for band in 0..num_mel_bins {
            let lower = edges[band];
            let center = edges[band + 1];
            let upper = edges[band + 2];
            let lower_slope = (mel - lower) / (center - lower);
            let upper_slope = (upper - mel) / (upper - center);
            matrix[(bin + 1) * num_mel_bins + band] = lower_slope.min(upper_slope).max(0.0);
        }
    }
    matrix
}

/// Symmetric Hann window (`torch.hann_window(periodic=False)`) in float64.
fn lasr_hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / (size - 1) as f64).cos())
        .collect()
}

/// Iterative radix-2 float64 FFT for the forward real-spectrum convention
/// `X[k] = sum_n x[n] * exp(-2 pi i k n / N)`.
struct FftF64 {
    n: usize,
    reversal: Vec<usize>,
    stages: Vec<Vec<(f64, f64)>>,
}

impl FftF64 {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let reversal = (0..n)
            .map(|index| index.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let mut stages = Vec::new();
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            stages.push(
                (0..half)
                    .map(|j| {
                        let angle = std::f64::consts::TAU * j as f64 / len as f64;
                        (angle.cos(), -angle.sin())
                    })
                    .collect(),
            );
            len *= 2;
        }
        Self {
            n,
            reversal,
            stages,
        }
    }

    fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        for index in 0..self.n {
            let target = self.reversal[index];
            if target > index {
                re.swap(index, target);
                im.swap(index, target);
            }
        }
        for (stage, twiddles) in self.stages.iter().enumerate() {
            let len = 2 << stage;
            let half = len / 2;
            for start in (0..self.n).step_by(len) {
                for (j, &(wr, wi)) in twiddles.iter().enumerate() {
                    let even = start + j;
                    let odd = even + half;
                    let vr = re[odd] * wr - im[odd] * wi;
                    let vi = re[odd] * wi + im[odd] * wr;
                    re[odd] = re[even] - vr;
                    im[odd] = im[even] - vi;
                    re[even] += vr;
                    im[even] += vi;
                }
            }
        }
    }
}

/// The LASR log-mel frontend: unfold framing (win 400, hop 160, no
/// centering), symmetric float64 Hann window, 512-point rfft, power
/// spectrum, kaldi-mel slope projection, `log(clamp(power, min=1e-5))`,
/// then a float32 cast. The float64 pipeline mirrors the transformers
/// `LASRFeatureExtractor` exactly.
struct LasrFrontend {
    mel_bins: usize,
    window: Vec<f64>,
    mel_matrix: Vec<f64>,
    fft: FftF64,
}

impl LasrFrontend {
    fn new(mel_bins: usize) -> Self {
        Self {
            mel_bins,
            window: lasr_hann_window(WIN_LENGTH),
            mel_matrix: lasr_mel_matrix(N_FFT / 2 + 1, mel_bins),
            fft: FftF64::new(N_FFT),
        }
    }

    /// Returns row-major `[steps, mel_bins]` log-mel features.
    fn extract(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() < WIN_LENGTH {
            return Err(SpeechError::Input {
                why: format!(
                    "audio must hold at least {WIN_LENGTH} samples for the LASR window, got {}",
                    samples.len()
                ),
            });
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "audio samples must be finite".into(),
            });
        }
        let steps = (samples.len() - WIN_LENGTH) / HOP_LENGTH + 1;
        let bins = N_FFT / 2 + 1;
        let mut frame_re = vec![0.0f64; N_FFT];
        let mut frame_im = vec![0.0f64; N_FFT];
        let mut power = vec![0.0f64; bins];
        let mut features = vec![0.0f32; steps * self.mel_bins];
        for step in 0..steps {
            let offset = step * HOP_LENGTH;
            for (n, &sample) in samples[offset..offset + WIN_LENGTH].iter().enumerate() {
                frame_re[n] = f64::from(sample) * self.window[n];
                frame_im[n] = 0.0;
            }
            frame_re[WIN_LENGTH..].fill(0.0);
            frame_im[WIN_LENGTH..].fill(0.0);
            self.fft.forward(&mut frame_re, &mut frame_im);
            for bin in 0..bins {
                power[bin] = frame_re[bin] * frame_re[bin] + frame_im[bin] * frame_im[bin];
            }
            let row = &mut features[step * self.mel_bins..(step + 1) * self.mel_bins];
            for (band, slot) in row.iter_mut().enumerate() {
                let mut energy = 0.0f64;
                for (bin, &p) in power.iter().enumerate() {
                    energy += p * self.mel_matrix[bin * self.mel_bins + band];
                }
                *slot = energy.max(MEL_CLAMP).ln() as f32;
            }
        }
        Ok(features)
    }
}

fn load_tensor_f32(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.dtype.to_uppercase() != "F32" {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!(
                "expected the fp32 pinned profile, got dtype {}",
                descriptor.dtype
            ),
        });
    }
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

#[derive(Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    /// Loads an HF `[out, in]` linear weight, optionally with bias.
    fn load(
        file: &SafetensorsFile,
        name: &str,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let weight = load_tensor_f32(file, &format!("{name}.weight"), &[output, input])?;
        let bias = if has_bias {
            Some(load_tensor_f32(file, &format!("{name}.bias"), &[output])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    width: usize,
    epsilon: f32,
}

impl LayerNorm {
    /// Loads a layer norm. The pinned checkpoint ships weight-only norms
    /// (the transformers reference builds them with `bias=False`, and the
    /// MLX reference leaves the affine bias at its zero init), so the bias
    /// stays optional and defaults to none.
    fn load(file: &SafetensorsFile, name: &str, width: usize, epsilon: f32) -> Result<Self> {
        let bias = if file.contains_tensor(&format!("{name}.bias")) {
            Some(load_tensor_f32(file, &format!("{name}.bias"), &[width])?)
        } else {
            None
        };
        Ok(Self {
            weight: load_tensor_f32(file, &format!("{name}.weight"), &[width])?,
            bias,
            width,
            epsilon,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(
            x,
            rows,
            self.width,
            &self.weight,
            self.bias.as_deref(),
            self.epsilon,
        );
    }
}

/// Conv1d over channels-last rows, matching the MLX `nn.Conv1d` contract the
/// conversion stores weights in: weight `(out, kernel, in / groups)`, input
/// `[steps, input]`, no padding (callers pad explicitly like the reference).
struct ConvSpec {
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
}

struct Conv1dLast {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    spec: ConvSpec,
}

impl Conv1dLast {
    fn load(file: &SafetensorsFile, name: &str, spec: ConvSpec, has_bias: bool) -> Result<Self> {
        let in_per_group = spec.input / spec.groups;
        let weight = load_tensor_f32(
            file,
            &format!("{name}.weight"),
            &[spec.output, spec.kernel, in_per_group],
        )?;
        let bias = if has_bias {
            Some(load_tensor_f32(
                file,
                &format!("{name}.bias"),
                &[spec.output],
            )?)
        } else {
            None
        };
        Ok(Self { weight, bias, spec })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let spec = &self.spec;
        assert!(
            steps >= spec.kernel,
            "conv input steps {steps} smaller than kernel {}",
            spec.kernel
        );
        let out_steps = (steps - spec.kernel) / spec.stride + 1;
        let in_per_group = spec.input / spec.groups;
        let out_per_group = spec.output / spec.groups;
        let mut out = vec![0.0f32; out_steps * spec.output];
        for out_channel in 0..spec.output {
            let group = out_channel / out_per_group;
            let weight_base = out_channel * spec.kernel * in_per_group;
            for step in 0..out_steps {
                let source_step = step * spec.stride;
                let mut acc = 0.0f32;
                for k in 0..spec.kernel {
                    let input_row = (source_step + k) * spec.input + group * in_per_group;
                    let weight_row = weight_base + k * in_per_group;
                    for (i, input_value) in
                        x[input_row..input_row + in_per_group].iter().enumerate()
                    {
                        acc += input_value * self.weight[weight_row + i];
                    }
                }
                let slot = step * spec.output + out_channel;
                out[slot] = acc + self.bias.as_deref().map_or(0.0, |b| b[out_channel]);
            }
        }
        out
    }
}

/// Inference batch normalization: per-channel affine with the checkpoint's
/// running statistics, `y = (x - mean) * rsqrt(var + eps) * weight + bias`.
struct BatchNormInference {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    epsilon: f32,
}

impl BatchNormInference {
    fn load(file: &SafetensorsFile, name: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: load_tensor_f32(file, &format!("{name}.weight"), &[channels])?,
            bias: load_tensor_f32(file, &format!("{name}.bias"), &[channels])?,
            running_mean: load_tensor_f32(file, &format!("{name}.running_mean"), &[channels])?,
            running_var: load_tensor_f32(file, &format!("{name}.running_var"), &[channels])?,
            epsilon: BATCH_NORM_EPS,
        })
    }

    fn apply(&self, x: &mut [f32], steps: usize, channels: usize) {
        for step in 0..steps {
            let row = &mut x[step * channels..(step + 1) * channels];
            for (channel, value) in row.iter_mut().enumerate() {
                let normalized = (*value - self.running_mean[channel])
                    / (self.running_var[channel] + self.epsilon).sqrt();
                *value = normalized * self.weight[channel] + self.bias[channel];
            }
        }
    }
}

fn relu(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.max(0.0);
    }
}

/// Dense-ReLU-conv-conv-dense subsampling front end. Conv weights are stored
/// in MLX layout and consumed over channels-last rows.
struct Subsampler {
    dense_0: Linear,
    conv_0: Conv1dLast,
    conv_1: Conv1dLast,
    dense_1: Linear,
}

impl Subsampler {
    fn load(file: &SafetensorsFile, config: &LasrCtcConfig) -> Result<Self> {
        let prefix = "encoder.subsampler";
        let hidden = config.hidden_size;
        Ok(Self {
            dense_0: Linear::load(
                file,
                &format!("{prefix}.dense_0"),
                config.num_mel_bins,
                hidden,
                true,
            )?,
            conv_0: Conv1dLast::load(
                file,
                &format!("{prefix}.conv_0"),
                ConvSpec {
                    input: hidden,
                    output: hidden,
                    kernel: config.subsampling_conv_kernel_size,
                    stride: config.subsampling_conv_stride,
                    groups: 1,
                },
                true,
            )?,
            conv_1: Conv1dLast::load(
                file,
                &format!("{prefix}.conv_1"),
                ConvSpec {
                    input: hidden,
                    output: config.subsampling_conv_channels,
                    kernel: config.subsampling_conv_kernel_size,
                    stride: config.subsampling_conv_stride,
                    groups: 1,
                },
                true,
            )?,
            dense_1: Linear::load(
                file,
                &format!("{prefix}.dense_1"),
                config.subsampling_conv_channels,
                hidden,
                true,
            )?,
        })
    }

    fn forward(&self, features: &[f32], steps: usize) -> Vec<f32> {
        let mut hidden = self.dense_0.forward(features, steps);
        relu(&mut hidden);
        let steps_0 = (steps - self.conv_0.spec.kernel) / self.conv_0.spec.stride + 1;
        let mut convolved = self.conv_0.forward(&hidden, steps);
        relu(&mut convolved);
        let steps_1 = (steps_0 - self.conv_1.spec.kernel) / self.conv_1.spec.stride + 1;
        let mut downsampled = self.conv_1.forward(&convolved, steps_0);
        relu(&mut downsampled);
        self.dense_1.forward(&downsampled, steps_1)
    }
}

/// Per-position RoPE tables mirroring the MLX reference exactly:
/// `inv_freq[d] = 1 / theta ** (2 d / head_dim)` in f32, then
/// `arg = position * inv_freq`. Returns `(cos, sin)` flattened
/// `[steps, head_dim / 2]`.
fn rope_tables(steps: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; steps * half];
    let mut sin = vec![0.0f32; steps * half];
    for d in 0..half {
        let inv_freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
        for t in 0..steps {
            let arg = t as f32 * inv_freq;
            cos[t * half + d] = arg.cos();
            sin[t * half + d] = arg.sin();
        }
    }
    (cos, sin)
}

/// Applies the half-rotation RoPE convention (`q * cos + rotate_half(q) * sin`
/// with the cos/sin tables duplicated across the full head dim) in place on
/// channels-last rows `[steps, heads * head_dim]`.
fn apply_rope_rows(
    x: &mut [f32],
    steps: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    let hidden = heads * head_dim;
    for t in 0..steps {
        for head in 0..heads {
            let base = t * hidden + head * head_dim;
            for d in 0..half {
                let c = cos[t * half + d];
                let s = sin[t * half + d];
                let a = x[base + d];
                let b = x[base + half + d];
                x[base + d] = a * c - b * s;
                x[base + half + d] = b * c + a * s;
            }
        }
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(file: &SafetensorsFile, prefix: &str, config: &LasrCtcConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            query: Linear::load(file, &format!("{prefix}.q_proj"), hidden, hidden, false)?,
            key: Linear::load(file, &format!("{prefix}.k_proj"), hidden, hidden, false)?,
            value: Linear::load(file, &format!("{prefix}.v_proj"), hidden, hidden, false)?,
            output: Linear::load(file, &format!("{prefix}.o_proj"), hidden, hidden, false)?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        })
    }

    fn forward(&self, x: &[f32], steps: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let hidden = self.heads * self.head_dim;
        let mut q = self.query.forward(x, steps);
        let mut k = self.key.forward(x, steps);
        let v = self.value.forward(x, steps);
        apply_rope_rows(&mut q, steps, self.heads, self.head_dim, cos, sin);
        apply_rope_rows(&mut k, steps, self.heads, self.head_dim, cos, sin);
        let scale = (self.head_dim as f64).powf(-0.5) as f32;
        let mut attended = vec![0.0f32; steps * hidden];
        let mut scores = vec![0.0f32; steps];
        for head in 0..self.heads {
            for query_step in 0..steps {
                let q_offset = query_step * hidden + head * self.head_dim;
                for (key_step, slot) in scores.iter_mut().enumerate() {
                    let k_offset = key_step * hidden + head * self.head_dim;
                    let mut score = 0.0f32;
                    for d in 0..self.head_dim {
                        score += q[q_offset + d] * k[k_offset + d];
                    }
                    *slot = score * scale;
                }
                ops::softmax_row(&mut scores);
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for key_step in 0..steps {
                        sum += scores[key_step] * v[key_step * hidden + head * self.head_dim + d];
                    }
                    attended[q_offset + d] = sum;
                }
            }
        }
        self.output.forward(&attended, steps)
    }
}

/// Conformer-style convolution module: pointwise-glu, padded depthwise conv,
/// batch norm, SiLU, pointwise.
struct ConvModule {
    pointwise_conv1: Conv1dLast,
    depthwise_conv: Conv1dLast,
    pointwise_conv2: Conv1dLast,
    norm: BatchNormInference,
    kernel_size: usize,
    channels: usize,
}

impl ConvModule {
    fn load(file: &SafetensorsFile, prefix: &str, config: &LasrCtcConfig) -> Result<Self> {
        let channels = config.hidden_size;
        Ok(Self {
            pointwise_conv1: Conv1dLast::load(
                file,
                &format!("{prefix}.pointwise_conv1"),
                ConvSpec {
                    input: channels,
                    output: 2 * channels,
                    kernel: 1,
                    stride: 1,
                    groups: 1,
                },
                false,
            )?,
            depthwise_conv: Conv1dLast::load(
                file,
                &format!("{prefix}.depthwise_conv"),
                ConvSpec {
                    input: channels,
                    output: channels,
                    kernel: config.conv_kernel_size,
                    stride: 1,
                    groups: channels,
                },
                false,
            )?,
            pointwise_conv2: Conv1dLast::load(
                file,
                &format!("{prefix}.pointwise_conv2"),
                ConvSpec {
                    input: channels,
                    output: channels,
                    kernel: 1,
                    stride: 1,
                    groups: 1,
                },
                false,
            )?,
            norm: BatchNormInference::load(file, &format!("{prefix}.norm"), channels)?,
            kernel_size: config.conv_kernel_size,
            channels,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let channels = self.channels;
        let mut hidden = self.pointwise_conv1.forward(x, steps);
        // GLU: first half gated by the sigmoid of the second half, written
        // back into the front of the buffer. Row t reads slots at and after
        // t * 2 * channels while writes land at t * channels, so the write
        // front never overtakes the read front.
        for t in 0..steps {
            for c in 0..channels {
                let a = hidden[t * 2 * channels + c];
                let b = hidden[(2 * t + 1) * channels + c];
                hidden[t * channels + c] = a / (1.0 + (-b).exp());
            }
        }
        hidden.truncate(steps * channels);
        // Manual asymmetric "same" padding: left (k-1)/2, right k-1-(k-1)/2.
        let pad_left = (self.kernel_size - 1) / 2;
        let pad_right = self.kernel_size - 1 - pad_left;
        let padded_steps = steps + pad_left + pad_right;
        let mut padded = vec![0.0f32; padded_steps * channels];
        padded[pad_left * channels..(pad_left + steps) * channels].copy_from_slice(&hidden);
        let mut convolved = self.depthwise_conv.forward(&padded, padded_steps);
        self.norm.apply(&mut convolved, steps, channels);
        ops::silu(&mut convolved);
        self.pointwise_conv2.forward(&convolved, steps)
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, config: &LasrCtcConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            linear1: Linear::load(
                file,
                &format!("{prefix}.linear1"),
                hidden,
                config.intermediate_size,
                false,
            )?,
            linear2: Linear::load(
                file,
                &format!("{prefix}.linear2"),
                config.intermediate_size,
                hidden,
                false,
            )?,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut hidden = self.linear1.forward(x, steps);
        ops::silu(&mut hidden);
        self.linear2.forward(&hidden, steps)
    }
}

struct EncoderBlock {
    feed_forward1: FeedForward,
    self_attn: Attention,
    conv: ConvModule,
    feed_forward2: FeedForward,
    norm_feed_forward1: LayerNorm,
    norm_self_att: LayerNorm,
    norm_conv: LayerNorm,
    norm_feed_forward2: LayerNorm,
    norm_out: LayerNorm,
    conv_residual_weights: [f32; 2],
    feed_forward_residual_weights: [f32; 2],
}

impl EncoderBlock {
    fn load(file: &SafetensorsFile, index: usize, config: &LasrCtcConfig) -> Result<Self> {
        let prefix = format!("encoder.layers.{index}");
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        Ok(Self {
            feed_forward1: FeedForward::load(file, &format!("{prefix}.feed_forward1"), config)?,
            self_attn: Attention::load(file, &format!("{prefix}.self_attn"), config)?,
            conv: ConvModule::load(file, &format!("{prefix}.conv"), config)?,
            feed_forward2: FeedForward::load(file, &format!("{prefix}.feed_forward2"), config)?,
            norm_feed_forward1: LayerNorm::load(
                file,
                &format!("{prefix}.norm_feed_forward1"),
                hidden,
                eps,
            )?,
            norm_self_att: LayerNorm::load(file, &format!("{prefix}.norm_self_att"), hidden, eps)?,
            norm_conv: LayerNorm::load(file, &format!("{prefix}.norm_conv"), hidden, eps)?,
            norm_feed_forward2: LayerNorm::load(
                file,
                &format!("{prefix}.norm_feed_forward2"),
                hidden,
                eps,
            )?,
            norm_out: LayerNorm::load(file, &format!("{prefix}.norm_out"), hidden, eps)?,
            conv_residual_weights: config.conv_residual_weights,
            feed_forward_residual_weights: config.feed_forward_residual_weights,
        })
    }

    fn forward(&self, x: &[f32], steps: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let [ff_residual, ff_scaled] = self.feed_forward_residual_weights;
        let [conv_residual, conv_scaled] = self.conv_residual_weights;
        // Feed forward 1 with the scaled residual pair.
        let residual = x.to_owned();
        let mut normalized = x.to_owned();
        self.norm_feed_forward1.apply(&mut normalized, steps);
        let mut hidden = self.feed_forward1.forward(&normalized, steps);
        for (slot, value) in hidden.iter_mut().enumerate() {
            *value = ff_residual * residual[slot] + ff_scaled * *value;
        }
        // Self attention on the normalized state, plain residual.
        {
            let mut normalized = hidden.clone();
            self.norm_self_att.apply(&mut normalized, steps);
            let attended = self.self_attn.forward(&normalized, steps, cos, sin);
            for (slot, value) in attended.iter().enumerate() {
                hidden[slot] += *value;
            }
        }
        // Convolution module with the scaled residual pair.
        {
            let mut normalized = hidden.clone();
            self.norm_conv.apply(&mut normalized, steps);
            let convolved = self.conv.forward(&normalized, steps);
            for (slot, value) in convolved.iter().enumerate() {
                hidden[slot] = conv_residual * hidden[slot] + conv_scaled * *value;
            }
        }
        // Feed forward 2 with the scaled residual pair, then the block norm.
        let residual = hidden.clone();
        {
            let mut normalized = hidden.clone();
            self.norm_feed_forward2.apply(&mut normalized, steps);
            let projected = self.feed_forward2.forward(&normalized, steps);
            for (slot, value) in projected.iter().enumerate() {
                hidden[slot] = ff_residual * residual[slot] + ff_scaled * *value;
            }
        }
        self.norm_out.apply(&mut hidden, steps);
        hidden
    }
}

/// Tokenizer piece table for greedy CTC text decoding, materialized from the
/// checkpoint's `tokenizer.json` (a Unigram piece list where the index is the
/// token id, plus the added special tokens).
pub struct LasrVocab {
    pieces: Vec<String>,
    special: Vec<bool>,
}

impl LasrVocab {
    fn from_json(text: &str, vocab_size: usize) -> Result<Self> {
        let document: Value = serde_json::from_str(text)
            .map_err(|error| bad_config("tokenizer.json", error.to_string()))?;
        let model = document
            .get("model")
            .ok_or_else(|| bad_config("tokenizer.json", "missing model section"))?;
        let vocab = model
            .get("vocab")
            .and_then(Value::as_array)
            .ok_or_else(|| bad_config("tokenizer.json", "model.vocab must be a piece list"))?;
        if vocab.len() != vocab_size {
            return Err(bad_config(
                "tokenizer.json",
                format!(
                    "vocab holds {} pieces but the model vocab_size is {vocab_size}",
                    vocab.len()
                ),
            ));
        }
        let mut pieces = Vec::with_capacity(vocab_size);
        for entry in vocab {
            let pair = entry
                .as_array()
                .and_then(|items| items.first())
                .and_then(Value::as_str)
                .ok_or_else(|| bad_config("tokenizer.json", "vocab entries must be pairs"))?;
            pieces.push(pair.to_owned());
        }
        let mut special = vec![false; vocab_size];
        if let Some(added) = document.get("added_tokens").and_then(Value::as_array) {
            for token in added {
                let id = token.get("id").and_then(Value::as_u64);
                let marked = token.get("special").and_then(Value::as_bool);
                if let (Some(id), Some(true)) = (id, marked) {
                    if let Ok(id) = usize::try_from(id) {
                        if id < vocab_size {
                            special[id] = true;
                        }
                    }
                }
            }
        }
        Ok(Self { pieces, special })
    }

    fn piece(&self, id: usize) -> Result<&str> {
        self.pieces
            .get(id)
            .map(String::as_str)
            .ok_or_else(|| SpeechError::Tensor {
                name: "tokenizer.json".into(),
                why: format!(
                    "token id {id} is outside the {}-piece vocabulary",
                    self.pieces.len()
                ),
            })
    }
}

/// Greedy CTC decoding following the transformers `LasrForCTC.generate` plus
/// `LasrTokenizer._decode` contract: per-frame argmax, collapse consecutive
/// repeats, drop the blank (pad) id, drop special tokens
/// (`skip_special_tokens=True`), join the pieces with the sentencepiece
/// metaspace marker turned into a space, and strip the one leading space the
/// metaspace decoder prepends.
pub fn ctc_decode(argmax_ids: &[i32], blank: usize, vocab: &LasrVocab) -> Result<String> {
    let mut pieces: Vec<&str> = Vec::new();
    let mut previous: Option<i32> = None;
    for &token in argmax_ids {
        if Some(token) == previous {
            continue;
        }
        previous = Some(token);
        let id = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: "greedy token ids must be non-negative".into(),
        })?;
        if id == blank || vocab.special.get(id).copied().unwrap_or(false) {
            continue;
        }
        pieces.push(vocab.piece(id)?);
    }
    let joined = pieces.concat().replace('\u{2581}', " ");
    Ok(joined.strip_prefix(' ').unwrap_or(&joined).to_owned())
}

/// Collapses consecutive repeats and drops the blank id, the id sequence the
/// text decoder consumes.
fn collapse_ctc_ids(argmax_ids: &[i32], blank: usize) -> Vec<i32> {
    let mut ids = Vec::new();
    let mut previous: Option<i32> = None;
    for &token in argmax_ids {
        if Some(token) == previous {
            continue;
        }
        previous = Some(token);
        if token >= 0 && token as usize == blank {
            continue;
        }
        ids.push(token);
    }
    ids
}

/// Loaded LASR CTC model with its frontend, encoder, CTC head, and
/// vocabulary.
pub struct LasrCtc {
    config: LasrCtcConfig,
    frontend: LasrFrontend,
    subsampler: Subsampler,
    blocks: Vec<EncoderBlock>,
    out_norm: LayerNorm,
    ctc_head: Linear,
    vocab: LasrVocab,
}

impl LasrCtc {
    /// Load the pinned profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json = fs::read_to_string(&config_path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", config_path.display()),
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = LasrCtcConfig::from_json(&root)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let blocks = (0..config.num_hidden_layers)
            .map(|index| EncoderBlock::load(&file, index, &config))
            .collect::<Result<Vec<_>>>()?;
        // The pinned conversion stores the head as a 1x1 Conv1d weight
        // (out, in, 1); the trailing kernel axis is dropped into the HF
        // (out, in) linear layout at load, and anything else is refused.
        let head_weight = load_tensor_f32(
            &file,
            "ctc_head.weight",
            &[config.vocab_size, config.hidden_size, 1],
        )?;
        let ctc_head = Linear {
            weight: head_weight,
            bias: Some(load_tensor_f32(
                &file,
                "ctc_head.bias",
                &[config.vocab_size],
            )?),
            input: config.hidden_size,
            output: config.vocab_size,
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer_json =
            fs::read_to_string(&tokenizer_path).map_err(|error| SpeechError::Input {
                why: format!("cannot read {}: {error}", tokenizer_path.display()),
            })?;
        let vocab = LasrVocab::from_json(&tokenizer_json, config.vocab_size)?;
        Ok(Self {
            frontend: LasrFrontend::new(config.num_mel_bins),
            subsampler: Subsampler::load(&file, &config)?,
            blocks,
            out_norm: LayerNorm::load(
                &file,
                "encoder.out_norm",
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            ctc_head,
            vocab,
            config,
        })
    }

    pub fn profile(&self) -> LasrCtcProfile {
        MEDASR_MLX_FP32
    }

    pub fn config(&self) -> &LasrCtcConfig {
        &self.config
    }

    /// CTC logits over one mono 16 kHz waveform: `[steps, vocab_size]`
    /// row-major.
    pub fn logits(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let (features, steps) = self.features(samples)?;
        Ok(self.ctc_head.forward(&self.encode(&features, steps), steps))
    }

    /// Collapsed greedy token ids for one mono 16 kHz waveform.
    pub fn greedy_token_ids(&self, samples: &[f32]) -> Result<Vec<i32>> {
        let logits = self.logits(samples)?;
        let argmax = argmax_rows(&logits, self.config.vocab_size);
        Ok(collapse_ctc_ids(&argmax, self.config.pad_token_id))
    }

    /// Transcribe one mono 16 kHz waveform with greedy CTC decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let logits = self.logits(samples)?;
        let argmax = argmax_rows(&logits, self.config.vocab_size);
        ctc_decode(&argmax, self.config.pad_token_id, &self.vocab)
    }

    fn features(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let features = self.frontend.extract(samples)?;
        let steps = features.len() / self.config.num_mel_bins;
        Ok((features, steps))
    }

    fn encode(&self, features: &[f32], mel_steps: usize) -> Vec<f32> {
        let hidden = self.subsampler.forward(features, mel_steps);
        let steps = hidden.len() / self.config.hidden_size;
        let (cos, sin) = rope_tables(steps, self.config.head_dim(), self.config.rope_theta);
        let mut encoded = hidden;
        for block in &self.blocks {
            encoded = block.forward(&encoded, steps, &cos, &sin);
        }
        self.out_norm.apply(&mut encoded, steps);
        encoded
    }
}

/// Per-row argmax with numpy semantics: the lowest index wins ties.
fn argmax_rows(logits: &[f32], vocab_size: usize) -> Vec<i32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            let mut best_index = 0;
            let mut best = f32::NEG_INFINITY;
            for (index, &score) in row.iter().enumerate() {
                if score > best {
                    best = score;
                    best_index = index;
                }
            }
            best_index as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        collapse_ctc_ids, ctc_decode, lasr_mel_matrix, rope_tables, LasrCtc, LasrCtcConfig,
        LasrVocab, MEDASR_MLX_FP32,
    };
    use crate::wav::read_wav_f32;
    use serde_json::{json, Value};
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/lasr_ctc_reference.json");

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid lasr_ctc reference fixture")
    }

    fn load_pinned_model() -> Option<LasrCtc> {
        let model_dir = std::env::var_os("TURBOSPARK_LASR_CTC_MODEL_DIR")?;
        Some(LasrCtc::load(Path::new(&model_dir)).expect("pinned lasr_ctc checkpoint loads"))
    }

    fn reference_waveform() -> Vec<f32> {
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);
        waveform.samples
    }

    fn fixture_config() -> Value {
        let fixture = load_fixture();
        let config = &fixture["config"];
        json!({
            "model_type": "lasr_ctc",
            "vocab_size": config["vocab_size"],
            "pad_token_id": config["pad_token_id"],
            "encoder_config": {
                "hidden_size": config["hidden_size"],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "intermediate_size": config["intermediate_size"],
                "hidden_act": config["hidden_act"],
                "conv_kernel_size": config["conv_kernel_size"],
                "conv_residual_weights": config["conv_residual_weights"],
                "feed_forward_residual_weights": config["feed_forward_residual_weights"],
                "num_mel_bins": config["num_mel_bins"],
                "subsampling_conv_channels": config["subsampling_conv_channels"],
                "subsampling_conv_kernel_size": config["subsampling_conv_kernel_size"],
                "subsampling_conv_stride": config["subsampling_conv_stride"],
                "layer_norm_eps": config["layer_norm_eps"],
                "attention_bias": false,
                "convolution_bias": false,
                "num_key_value_heads": config["num_attention_heads"],
                "rope_parameters": {"rope_theta": config["rope_theta"], "rope_type": "default"},
            },
        })
    }

    fn compare_spots(
        actual: &[f32],
        rows: usize,
        columns: usize,
        spots: &Spots,
        label: &str,
        tolerance: f32,
    ) -> f32 {
        compare_spots_scaled(actual, rows, columns, spots, label, tolerance, 0.0)
    }

    /// Spot comparison with a combined absolute plus relative gate; the
    /// relative term covers this family's huge intermediate activations
    /// (sub-sampled magnitudes reach 1e5 or more) where f32 accumulation
    /// order alone produces larger absolute spread.
    fn compare_spots_scaled(
        actual: &[f32],
        rows: usize,
        columns: usize,
        spots: &Spots,
        label: &str,
        absolute: f32,
        relative: f32,
    ) -> f32 {
        assert_eq!([rows, columns], spots.shape.as_slice(), "{label} shape");
        let mut worst = 0.0f32;
        let mut worst_allowed = 0.0f32;
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (column_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                let diff = (actual[row * columns + column] - expected).abs();
                let allowed = absolute + relative * expected.abs();
                worst_allowed = worst_allowed.max(allowed);
                worst = worst.max(diff);
                assert!(
                    diff <= allowed,
                    "{label} [{row},{column}] differs by {diff} (gate {allowed})"
                );
            }
        }
        eprintln!("{label}: worst {worst:.3e} (max gate {worst_allowed:.3e})");
        worst
    }

    fn parse_spots(value: &Value) -> Spots {
        serde_json::from_value(value.clone()).expect("valid spots block")
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(MEDASR_MLX_FP32.repository, "drankush-ai/medasr-mlx-fp32");
        assert_eq!(
            MEDASR_MLX_FP32.revision,
            "3b967580b5176144bc633fac60420d1b122dfba8"
        );
    }

    #[test]
    fn parses_the_pinned_config_and_rejects_unverified_variants() {
        let root = fixture_config();
        let config = LasrCtcConfig::from_json(&root).unwrap();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_hidden_layers, 17);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.vocab_size, 512);
        assert_eq!(config.pad_token_id, 0);
        assert_eq!(config.conv_residual_weights, [2.0, 1.0]);
        assert_eq!(config.feed_forward_residual_weights, [1.5, 0.5]);
        assert_eq!(config.head_dim(), 64);

        let mut foreign = root.clone();
        foreign["model_type"] = json!("whisper");
        assert!(LasrCtcConfig::from_json(&foreign).is_err());

        let mut relu = root.clone();
        relu["encoder_config"]["hidden_act"] = json!("relu");
        assert!(LasrCtcConfig::from_json(&relu).is_err());

        let mut biased = root.clone();
        biased["encoder_config"]["attention_bias"] = json!(true);
        assert!(LasrCtcConfig::from_json(&biased).is_err());

        let mut conv_biased = root.clone();
        conv_biased["encoder_config"]["convolution_bias"] = json!(true);
        assert!(LasrCtcConfig::from_json(&conv_biased).is_err());

        let mut gqa = root.clone();
        gqa["encoder_config"]["num_key_value_heads"] = json!(4);
        assert!(LasrCtcConfig::from_json(&gqa).is_err());

        let mut rope = root.clone();
        rope["encoder_config"]["rope_parameters"]["rope_type"] = json!("linear");
        assert!(LasrCtcConfig::from_json(&rope).is_err());

        let mut missing = root;
        missing["encoder_config"]["hidden_size"] = Value::Null;
        assert!(LasrCtcConfig::from_json(&missing).is_err());
    }

    /// Compares a Rust half rope table `[steps, head_dim / 2]` against the
    /// fixture's full-dim MLX table `[steps, head_dim]`, including the
    /// convention that the model duplicates the half across the head dim.
    fn compare_rope_table(table: &[f32], steps: usize, half: usize, spots: &Spots, label: &str) {
        assert_eq!(spots.shape, vec![steps, 2 * half], "{label} shape");
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (value_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][value_index];
                let half_column = if column < half { column } else { column - half };
                if let Some(position) = spots.columns.iter().position(|&other| other == half_column)
                {
                    assert_eq!(
                        expected, spots.values[row_index][position],
                        "rope {label} table must duplicate its halves at column {column}"
                    );
                }
                let diff = (table[row * half + half_column] - expected).abs();
                assert!(
                    diff < 2.0e-4,
                    "rope {label} [{row},{column}] differs by {diff}"
                );
            }
        }
    }

    #[test]
    fn rope_tables_match_the_pinned_reference() {
        let fixture = load_fixture();
        let config = LasrCtcConfig::from_json(&fixture_config()).unwrap();
        let steps = fixture["rope"]["cos"]["shape"][0].as_u64().unwrap() as usize;
        let (cos, sin) = rope_tables(steps, config.head_dim(), config.rope_theta);
        let half = config.head_dim() / 2;
        compare_rope_table(
            &cos,
            steps,
            half,
            &parse_spots(&fixture["rope"]["cos"]),
            "cos",
        );
        compare_rope_table(
            &sin,
            steps,
            half,
            &parse_spots(&fixture["rope"]["sin"]),
            "sin",
        );
        eprintln!("lasr rope parity: cos/sin spot gate 2.0e-4 passed");
    }

    #[test]
    fn mel_matrix_matches_the_pinned_reference() {
        let fixture = load_fixture();
        let matrix = lasr_mel_matrix(257, 128);
        let spots = parse_spots(&fixture["mel_matrix"]["spots"]);
        compare_spots(
            &matrix.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            257,
            128,
            &spots,
            "mel_matrix",
            1.0e-6,
        );
    }

    #[test]
    fn frontend_matches_the_pinned_log_mel() {
        let fixture = load_fixture();
        let config = LasrCtcConfig::from_json(&fixture_config()).unwrap();
        let frontend = super::LasrFrontend::new(config.num_mel_bins);
        let features = frontend.extract(&reference_waveform()).unwrap();
        let steps = features.len() / config.num_mel_bins;
        let worst = compare_spots(
            &features,
            steps,
            config.num_mel_bins,
            &parse_spots(&fixture["input_features"]),
            "input_features",
            2.0e-4,
        );
        eprintln!("lasr frontend parity: log-mel worst {worst:.3e} (gate 2.0e-4)");
    }

    #[test]
    fn ctc_decode_collapses_repeats_and_drops_blank_and_specials() {
        let vocab = LasrVocab {
            pieces: vec![
                "<epsilon>".to_owned(),
                "</s>".to_owned(),
                "\u{2581}".to_owned(),
                "a".to_owned(),
                "\u{2581}b".to_owned(),
            ],
            special: vec![true, true, false, false, false],
        };
        // Repeat collapse first, then blank (0) drop, then special drop:
        // [3, 3, 0, 4, 3, 1, 2, 3] -> collapse [3, 0, 4, 3, 1, 2, 3] ->
        // drop blank and specials -> pieces "a", "\u{2581}b", "a",
        // "\u{2581}", "a" -> "a ba a" after the metaspace marker becomes a
        // space.
        assert_eq!(
            ctc_decode(&[3, 3, 0, 4, 3, 1, 2, 3], 0, &vocab).unwrap(),
            "a ba a"
        );
        // A leading metaspace run contributes one leading space per piece
        // minus the one space the metaspace decoder strips.
        assert_eq!(ctc_decode(&[2, 4], 0, &vocab).unwrap(), " b");
        assert_eq!(ctc_decode(&[4], 0, &vocab).unwrap(), "b");
        assert_eq!(ctc_decode(&[2], 0, &vocab).unwrap(), "");
        assert_eq!(ctc_decode(&[0, 0, 1], 0, &vocab).unwrap(), "");
        assert_eq!(collapse_ctc_ids(&[3, 3, 0, 4, 3], 0), vec![3, 4, 3]);
        assert!(ctc_decode(&[-1], 0, &vocab).is_err());
    }

    #[test]
    fn fixture_argmax_ids_decode_to_the_reference_transcript() {
        let fixture = load_fixture();
        let config = LasrCtcConfig::from_json(&fixture_config()).unwrap();
        let vocab = LasrVocab {
            pieces: fixture["vocab"]
                .as_array()
                .unwrap()
                .iter()
                .map(|piece| piece.as_str().unwrap().to_owned())
                .collect(),
            special: {
                let mut special = vec![false; config.vocab_size];
                for id in fixture["special_token_ids"].as_array().unwrap() {
                    special[id.as_u64().unwrap() as usize] = true;
                }
                special
            },
        };
        let argmax: Vec<i32> = fixture["argmax_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        let collapsed = collapse_ctc_ids(&argmax, config.pad_token_id);
        let expected: Vec<i32> = fixture["greedy_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(collapsed, expected, "collapsed greedy ids must match");
        assert_eq!(
            ctc_decode(&argmax, config.pad_token_id, &vocab).unwrap(),
            fixture["transcript"].as_str().unwrap(),
            "fixture argmax ids must decode to the pinned transcript"
        );
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "3b967580b5176144bc633fac60420d1b122dfba8"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio lasr_ctc at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
        let load_path = fixture["provenance"]["load_path"].as_str().unwrap();
        assert!(
            load_path.contains("no conv transposition") && load_path.contains("double-transposes"),
            "the load-path divergence note must be recorded: {load_path}"
        );
    }

    #[test]
    fn short_audio_is_refused() {
        let config = LasrCtcConfig::from_json(&fixture_config()).unwrap();
        let frontend = super::LasrFrontend::new(config.num_mel_bins);
        assert!(frontend.extract(&[0.0; 128]).is_err());
    }

    #[test]
    #[ignore = "requires the pinned MedASR MLX fp32 checkpoint in TURBOSPARK_LASR_CTC_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model) = load_pinned_model() else {
            eprintln!("skipping: TURBOSPARK_LASR_CTC_MODEL_DIR is unset");
            return;
        };
        let fixture = load_fixture();
        let samples = reference_waveform();

        let (features, mel_steps) = model.features(&samples).unwrap();
        let feature_worst = compare_spots(
            &features,
            mel_steps,
            model.config.num_mel_bins,
            &parse_spots(&fixture["input_features"]),
            "input_features",
            2.0e-4,
        );

        let hidden = model.subsampler.forward(&features, mel_steps);
        let subsampled_steps = hidden.len() / model.config.hidden_size;
        let subsampler_worst = compare_spots_scaled(
            &hidden,
            subsampled_steps,
            model.config.hidden_size,
            &parse_spots(&fixture["subsampler"]),
            "subsampler",
            2.0e-4,
            1.0e-5,
        );

        let (cos, sin) = super::rope_tables(
            subsampled_steps,
            model.config.head_dim(),
            model.config.rope_theta,
        );
        let half = model.config.head_dim() / 2;
        compare_rope_table(
            &cos,
            subsampled_steps,
            half,
            &parse_spots(&fixture["rope"]["cos"]),
            "cos",
        );
        compare_rope_table(
            &sin,
            subsampled_steps,
            half,
            &parse_spots(&fixture["rope"]["sin"]),
            "sin",
        );

        let mut encoded = hidden;
        let mut layer_worst = 0.0f32;
        for (index, block) in model.blocks.iter().enumerate() {
            encoded = block.forward(&encoded, subsampled_steps, &cos, &sin);
            for entry in fixture["hidden_states"].as_array().unwrap() {
                if entry["name"] == format!("encoder_layer_{index}") {
                    layer_worst = layer_worst.max(compare_spots_scaled(
                        &encoded,
                        subsampled_steps,
                        model.config.hidden_size,
                        &parse_spots(&entry["spots"]),
                        &format!("encoder_layer_{index}"),
                        2.0e-4,
                        1.0e-5,
                    ));
                }
            }
        }

        let mut final_state = encoded;
        model.out_norm.apply(&mut final_state, subsampled_steps);
        let final_worst = compare_spots(
            &final_state,
            subsampled_steps,
            model.config.hidden_size,
            &parse_spots(&fixture["encoder_final"]),
            "encoder_final",
            2.0e-4,
        );

        let logits = model.ctc_head.forward(&final_state, subsampled_steps);
        let logit_spots = parse_spots(&fixture["ctc_logits"]);
        let logit_worst = compare_spots_scaled(
            &logits,
            subsampled_steps,
            model.config.vocab_size,
            &logit_spots,
            "ctc_logits",
            2.0e-4,
            1.0e-5,
        );

        let argmax = super::argmax_rows(&logits, model.config.vocab_size);
        let expected_argmax: Vec<i32> = fixture["argmax_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(argmax, expected_argmax, "argmax token ids must match");
        let transcript = super::ctc_decode(&argmax, model.config.pad_token_id, &model.vocab)
            .expect("greedy decode succeeds");
        let expected = fixture["transcript"].as_str().unwrap();
        assert_eq!(transcript, expected, "transcript must match the reference");

        eprintln!(
            "lasr_ctc fixture parity: log-mel {feature_worst:.3e}, subsampler \
             {subsampler_worst:.3e}, layers {layer_worst:.3e}, final {final_worst:.3e}, \
             logits {logit_worst:.3e} (stage gates 2.0e-4)"
        );
        eprintln!("lasr_ctc transcript: {transcript:?}");
    }
}
