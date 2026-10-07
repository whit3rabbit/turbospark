//! Higgs Audio v3 whisper-style audio encoder, feature projector, and the
//! 128-band log-mel frontend.
//!
//! Reference: `mlx_audio/stt/models/higgs_audio_3/higgs_audio_3.py`
//! (`HiggsAudioEncoder`, `AudioEncoderLayer`, `AudioAttention`,
//! `HiggsAudioFeatureProjector`) and `audio.py`
//! (`AudioFeatureExtractor`) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Layout conventions: all weights load in the checkpoint's `[out, in]`
//! linear and `[out, in, kernel]` conv1d layout (the MLX reference
//! transposes conv weights to `[out, kernel, in]` at load; this port keeps
//! the torch layout its kernels use). The convolutions run channel-major
//! `[channel, time]`; the encoder body, pooling, and projector MLP run
//! row-major `[time, channel]`, and the projector's depthwise conv transposes
//! back to channel-major for its stride-2 pass.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::mel::{mel_spectrogram, MelScale, MelSpectrogramOptions};
use crate::nn::LayerNorm;
use crate::ops;
use crate::stft::StftOptions;
use crate::{Result, SpeechError};

/// The reference feature extractor constants (`audio.py`).
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;
pub(crate) const SAMPLE_RATE: u32 = 16_000;
/// Highest frequency a mel band covers (`mel_filters` default `f_max`).
const FMAX: f32 = 8_000.0;
/// MLX `nn.LayerNorm` default epsilon, used by every encoder norm.
const LAYER_NORM_EPS: f32 = 1e-5;

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing from the checkpoint shards".to_owned(),
    }
}

fn shard_of<'a>(files: &'a [SafetensorsFile], name: &str) -> Result<&'a SafetensorsFile> {
    files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| missing(name))
}

fn load_vector(files: &[SafetensorsFile], name: &str, len: usize) -> Result<Vec<f32>> {
    let values = shard_of(files, name)?.load_as_f32(name)?;
    if values.len() != len {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {len} values, got {}", values.len()),
        });
    }
    Ok(values)
}

/// Loads one linear weight in `[out, in]` layout plus its required bias.
fn load_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let weight_name = format!("{base}.weight");
    let weight = load_vector(files, &weight_name, input * output)?;
    let bias = load_vector(files, &format!("{base}.bias"), output)?;
    Ok((weight, bias))
}

/// Loads one linear weight plus its optional bias, for projections the
/// reference declares bias-free (`k_proj`).
fn load_convless_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let weight_name = format!("{base}.weight");
    let weight = load_vector(files, &weight_name, input * output)?;
    let bias_name = format!("{base}.bias");
    let bias = if files.iter().any(|f| f.contains_tensor(&bias_name)) {
        Some(load_vector(files, &bias_name, output)?)
    } else {
        None
    };
    Ok((weight, bias))
}

/// Loads one conv1d weight in `[out, in / groups, kernel]` layout plus bias.
fn load_conv(
    files: &[SafetensorsFile],
    base: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    groups: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let weight_name = format!("{base}.weight");
    let weight = load_vector(files, &weight_name, out_ch * (in_ch / groups) * kernel)?;
    let bias = load_vector(files, &format!("{base}.bias"), out_ch)?;
    Ok((weight, bias))
}

fn load_layer_norm(files: &[SafetensorsFile], base: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        load_vector(files, &format!("{base}.weight"), width)?,
        Some(load_vector(files, &format!("{base}.bias"), width)?),
        LAYER_NORM_EPS,
    ))
}

struct AudioAttention {
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    v_weight: Vec<f32>,
    v_bias: Vec<f32>,
    out_weight: Vec<f32>,
    out_bias: Vec<f32>,
    heads: usize,
    head_dim: usize,
    dim: usize,
}

impl AudioAttention {
    fn load(files: &[SafetensorsFile], base: &str, dim: usize, heads: usize) -> Result<Self> {
        let head_dim = dim / heads;
        // The reference AudioAttention carries a q bias but no k bias.
        let (k_weight, k_bias) = load_convless_linear(files, &format!("{base}.k_proj"), dim, dim)?;
        if k_bias.is_some() {
            return Err(SpeechError::Unsupported {
                why: format!("{base}.k_proj must stay bias-free"),
            });
        }
        Ok(Self {
            q_weight: load_vector(files, &format!("{base}.q_proj.weight"), dim * dim)?,
            q_bias: load_vector(files, &format!("{base}.q_proj.bias"), dim)?,
            k_weight,
            v_weight: load_vector(files, &format!("{base}.v_proj.weight"), dim * dim)?,
            v_bias: load_vector(files, &format!("{base}.v_proj.bias"), dim)?,
            out_weight: load_vector(files, &format!("{base}.out_proj.weight"), dim * dim)?,
            out_bias: load_vector(files, &format!("{base}.out_proj.bias"), dim)?,
            heads,
            head_dim,
            dim,
        })
    }

    /// Full (non-causal) attention over `x [time, dim]` row-major, returning
    /// `[time, dim]`. Query scaling folds in before the softmax exactly as
    /// the reference does (`q * scaling`, then scale 1.0).
    fn forward(&self, x: &[f32], time: usize) -> Vec<f32> {
        let mut q = ops::linear(
            x,
            &self.q_weight,
            Some(&self.q_bias),
            time,
            self.dim,
            self.dim,
        );
        let scaling = (self.head_dim as f32).recip().sqrt();
        for value in &mut q {
            *value *= scaling;
        }
        let k = ops::linear(x, &self.k_weight, None, time, self.dim, self.dim);
        let v = ops::linear(
            x,
            &self.v_weight,
            Some(&self.v_bias),
            time,
            self.dim,
            self.dim,
        );

        let mut attended = vec![0.0f32; time * self.dim];
        let mut q_head = vec![0.0f32; time * self.head_dim];
        let mut k_head = vec![0.0f32; time * self.head_dim];
        let mut v_head = vec![0.0f32; time * self.head_dim];
        for head in 0..self.heads {
            let base = head * self.head_dim;
            for t in 0..time {
                let source = t * self.dim + base;
                q_head[t * self.head_dim..(t + 1) * self.head_dim]
                    .copy_from_slice(&q[source..source + self.head_dim]);
                k_head[t * self.head_dim..(t + 1) * self.head_dim]
                    .copy_from_slice(&k[source..source + self.head_dim]);
                v_head[t * self.head_dim..(t + 1) * self.head_dim]
                    .copy_from_slice(&v[source..source + self.head_dim]);
            }
            let out = ops::sdpa(
                &q_head,
                &k_head,
                &v_head,
                None,
                time,
                time,
                self.head_dim,
                self.head_dim,
                1.0,
            );
            for t in 0..time {
                let target = t * self.dim + base;
                attended[target..target + self.head_dim]
                    .copy_from_slice(&out[t * self.head_dim..(t + 1) * self.head_dim]);
            }
        }
        ops::linear(
            &attended,
            &self.out_weight,
            Some(&self.out_bias),
            time,
            self.dim,
            self.dim,
        )
    }
}

struct EncoderLayer {
    attention: AudioAttention,
    attn_norm: LayerNorm,
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    ffn: usize,
    final_norm: LayerNorm,
    dim: usize,
}

impl EncoderLayer {
    fn load(
        files: &[SafetensorsFile],
        index: usize,
        d_model: usize,
        heads: usize,
        ffn: usize,
    ) -> Result<Self> {
        let base = format!("audio_tower.layers.{index}");
        Ok(Self {
            attention: AudioAttention::load(files, &format!("{base}.self_attn"), d_model, heads)?,
            attn_norm: load_layer_norm(files, &format!("{base}.self_attn_layer_norm"), d_model)?,
            fc1_weight: load_vector(files, &format!("{base}.fc1.weight"), ffn * d_model)?,
            fc1_bias: load_vector(files, &format!("{base}.fc1.bias"), ffn)?,
            fc2_weight: load_vector(files, &format!("{base}.fc2.weight"), d_model * ffn)?,
            fc2_bias: load_vector(files, &format!("{base}.fc2.bias"), d_model)?,
            ffn,
            final_norm: load_layer_norm(files, &format!("{base}.final_layer_norm"), d_model)?,
            dim: d_model,
        })
    }

    /// Pre-norm residual block over `x [time, dim]`: norm, attention, add;
    /// norm, GELU fc1, fc2, add.
    fn forward(&self, x: &[f32], time: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.attn_norm.apply(&mut normalized, time);
        let attended = self.attention.forward(&normalized, time);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(&attended) {
            *value += add;
        }

        let mut normalized = residual.clone();
        self.final_norm.apply(&mut normalized, time);
        let mut hidden = ops::linear(
            &normalized,
            &self.fc1_weight,
            Some(&self.fc1_bias),
            time,
            self.dim,
            self.ffn,
        );
        ops::gelu_erf(&mut hidden);
        let out = ops::linear(
            &hidden,
            &self.fc2_weight,
            Some(&self.fc2_bias),
            time,
            self.ffn,
            self.dim,
        );
        for (value, add) in residual.iter_mut().zip(out) {
            *value += add;
        }
        residual
    }
}

/// The whisper-style audio tower: GELU conv frontend, learned positions,
/// pre-norm encoder blocks, pairwise time pooling, final layer norm.
pub struct HiggsAudioEncoder {
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    positions: Vec<f32>,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    d_model: usize,
    mel_bins: usize,
    max_positions: usize,
}

impl HiggsAudioEncoder {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        layers: usize,
        heads: usize,
        ffn: usize,
        d_model: usize,
        mel_bins: usize,
        max_positions: usize,
    ) -> Result<Self> {
        let (conv1_weight, conv1_bias) =
            load_conv(files, "audio_tower.conv1", mel_bins, d_model, 3, 1)?;
        let (conv2_weight, conv2_bias) =
            load_conv(files, "audio_tower.conv2", d_model, d_model, 3, 1)?;
        Ok(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            positions: load_vector(
                files,
                "audio_tower.embed_positions.weight",
                max_positions * d_model,
            )?,
            layers: (0..layers)
                .map(|index| EncoderLayer::load(files, index, d_model, heads, ffn))
                .collect::<Result<Vec<_>>>()?,
            final_norm: load_layer_norm(files, "audio_tower.layer_norm", d_model)?,
            d_model,
            mel_bins,
            max_positions,
        })
    }

    /// Encodes one chunk's log-mel features (`[frames, mel_bins]`
    /// frame-major) into the pooled, normalized `[frames / 2, d_model]`
    /// plane in row-major order.
    pub fn forward(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        let d = self.d_model;
        let bins = self.mel_bins;
        if features.len() != frames * bins {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {frames} x {bins} mel features, got {}",
                    features.len()
                ),
            });
        }
        // Channel-major [mel_bins, frames] for the convolutions.
        let mut x = vec![0.0f32; bins * frames];
        for channel in 0..bins {
            for time in 0..frames {
                x[channel * frames + time] = features[time * bins + channel];
            }
        }
        let mut x = ops::conv1d(
            &x,
            &self.conv1_weight,
            Some(&self.conv1_bias),
            bins,
            d,
            3,
            1,
            1,
            1,
            1,
        );
        ops::gelu_erf(&mut x);
        let mut x = ops::conv1d(
            &x,
            &self.conv2_weight,
            Some(&self.conv2_bias),
            d,
            d,
            3,
            2,
            1,
            1,
            1,
        );
        ops::gelu_erf(&mut x);
        let t2 = x.len() / d;
        if t2 == 0 || t2 > self.max_positions {
            return Err(SpeechError::Input {
                why: format!(
                    "conv frontend produced {t2} frames for {} learned positions",
                    self.max_positions
                ),
            });
        }
        // Transpose to row-major [t2, d], adding the learned positions
        // (position table row-major [time, channel]) in the same pass.
        let mut rows = vec![0.0f32; t2 * d];
        for channel in 0..d {
            for time in 0..t2 {
                rows[time * d + channel] =
                    x[channel * t2 + time] + self.positions[time * d + channel];
            }
        }

        for layer in &self.layers {
            rows = layer.forward(&rows, t2);
        }

        // Pairwise time pooling: reshape [t2, d] to [t2/2, 2, d] and average
        // the pair axis; an odd trailing frame is dropped.
        let pooled = t2 / 2;
        let mut out = vec![0.0f32; pooled * d];
        for time in 0..pooled {
            let even = (2 * time) * d;
            let odd = even + d;
            for channel in 0..d {
                out[time * d + channel] = (rows[even + channel] + rows[odd + channel]) / 2.0;
            }
        }
        self.final_norm.apply(&mut out, pooled);
        Ok(out)
    }
}

/// The MLP projector with temporal downsample: depthwise stride-2 conv,
/// Linear(1280, 2048), ReLU, Linear(2048, hidden).
pub struct FeatureProjector {
    temporal_weight: Vec<f32>,
    temporal_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
    d_model: usize,
    wide: usize,
    hidden: usize,
}

impl FeatureProjector {
    pub(crate) fn load(
        files: &[SafetensorsFile],
        d_model: usize,
        wide: usize,
        hidden: usize,
    ) -> Result<Self> {
        let (temporal_weight, temporal_bias) = load_conv(
            files,
            "audio_encoder_proj.temporal",
            d_model,
            d_model,
            3,
            d_model,
        )?;
        let (linear1_weight, linear1_bias) =
            load_linear(files, "audio_encoder_proj.linear1", d_model, wide)?;
        let (linear2_weight, linear2_bias) =
            load_linear(files, "audio_encoder_proj.linear2", wide, hidden)?;
        Ok(Self {
            temporal_weight,
            temporal_bias,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
            d_model,
            wide,
            hidden,
        })
    }

    /// Projects the pooled encoder plane (`[rows, d_model]` row-major) to
    /// `[projected_rows, hidden]`, halving the time axis through the
    /// stride-2 depthwise conv.
    pub fn forward(&self, pooled: &[f32], rows: usize) -> Vec<f32> {
        let d = self.d_model;
        // Channel-major [d_model, rows] for the depthwise conv.
        let mut x = vec![0.0f32; d * rows];
        for channel in 0..d {
            for time in 0..rows {
                x[channel * rows + time] = pooled[time * d + channel];
            }
        }
        let x = ops::conv1d(
            &x,
            &self.temporal_weight,
            Some(&self.temporal_bias),
            d,
            d,
            3,
            2,
            1,
            1,
            d,
        );
        let projected_rows = x.len() / d;
        // Row-major [projected_rows, d_model] for the MLP.
        let mut rows_major = vec![0.0f32; projected_rows * d];
        for channel in 0..d {
            for time in 0..projected_rows {
                rows_major[time * d + channel] = x[channel * projected_rows + time];
            }
        }
        let mut hidden = ops::linear(
            &rows_major,
            &self.linear1_weight,
            Some(&self.linear1_bias),
            projected_rows,
            d,
            self.wide,
        );
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        ops::linear(
            &hidden,
            &self.linear2_weight,
            Some(&self.linear2_bias),
            projected_rows,
            self.wide,
            self.hidden,
        )
    }
}

/// The reference `hanning(size)` default: the symmetric Hann window
/// `0.5 * (1 - cos(2 pi n / (size - 1)))` (periodic=False).
fn hanning_symmetric(size: usize) -> Vec<f32> {
    match size {
        0 | 1 => vec![1.0; size],
        _ => (0..size)
            .map(|n| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (size - 1) as f64).cos()))
                    as f32
            })
            .collect(),
    }
}

/// The reference 128-band log-mel frontend over one chunk: centered STFT
/// with reflect padding and the symmetric Hann window, power spectrum, the
/// Slaney filterbank, the final STFT frame dropped, then
/// `log10(max(x, 1e-10))`, the global peak minus 8 clamp, and
/// `(x + 4) / 4`. Returns the frame-major `[frames, 128]` plane.
pub(crate) fn log_mel_128(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    let options = MelSpectrogramOptions {
        stft: StftOptions {
            fft_size: N_FFT,
            hop: HOP_LENGTH,
            window: hanning_symmetric(N_FFT),
            center: true,
        },
        num_mels: 128,
        sample_rate: SAMPLE_RATE,
        fmin: 0.0,
        fmax: Some(FMAX),
        scale: MelScale::Slaney,
        power: 2.0,
    };
    let mut mel = mel_spectrogram(samples, &options)?;
    // `freqs[:-1, :]`: the reference drops the final centered frame before
    // the global peak is taken.
    mel.pop();
    if mel.is_empty() {
        return Err(SpeechError::Audio(
            "log-mel frontend produced no frames".into(),
        ));
    }
    let mut peak = f32::NEG_INFINITY;
    for frame in &mel {
        for &x in frame {
            peak = peak.max(x.max(1e-10).log10());
        }
    }
    let clamp_at = peak - 8.0;
    let frames = mel.len();
    let mut values = Vec::with_capacity(frames * 128);
    for frame in &mel {
        for &x in frame {
            values.push((x.max(1e-10).log10().max(clamp_at) + 4.0) / 4.0);
        }
    }
    Ok((values, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_hann_matches_the_reference_endpoints() {
        let w = hanning_symmetric(8);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[7], 0.0);
        // An even-length symmetric window never reaches 1.0; its peak sits
        // between samples 3 and 4 (0.5 * (1 - cos(2 pi 3 / 7))).
        let peak = w.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (peak - 0.5 * (1.0 - (2.0 * std::f64::consts::PI * 3.0 / 7.0).cos()) as f32).abs()
                < 1e-6
        );
        assert!(
            (w[3] - 0.5 * (1.0 - (2.0 * std::f64::consts::PI * 3.0 / 7.0).cos()) as f32).abs()
                < 1e-6
        );
        // An odd-length symmetric window peaks at exactly 1.0.
        let w9 = hanning_symmetric(9);
        assert!((w9[4] - 1.0).abs() < 1e-6);
        // A size-1 window degenerates to the constant window.
        assert_eq!(hanning_symmetric(1), vec![1.0]);
    }
}
