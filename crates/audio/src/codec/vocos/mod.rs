//! Vocos: iSTFT-based neural vocoder (ConvNeXt backbone + iSTFT head).
//!
//! Reference: `mlx_audio/codec/models/vocos/` (vocos.py, mel.py) at
//! mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/vocos).
//! The mel variant front end is a whisper-style log-mel (symmetric
//! Hann centered reflect STFT, final frame dropped, HTK mel, log with
//! a 1e-5 floor); the backbone is ConvNeXt blocks with a depthwise
//! conv, LayerNorm (or AdaLayerNorm), and two linears; the head is a
//! linear to magnitude + phase with an inverse STFT whose overlap-add
//! divides by the summed window (the mlx-audio `istft` default
//! `normalized=False`, unlike torch.istft's window-squared divide).
//!
//! Checkpoint notes: vocos `model.safetensors` keeps the PyTorch conv
//! layout `[out, in, K]` for `backbone.embed` and `dwconv` (the
//! reference transposes on load); all other weights load as stored.
//! The reference deletes the recorded `window` tensors and recomputes.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::fft::RealFftPlan;
use crate::mel::{mel_filterbank, MelScale};
use crate::ops;
use crate::{AudioError, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Backbone geometry (`VocosBackbone` init args).
#[derive(Debug, Clone)]
pub struct VocosBackboneConfig {
    pub input_channels: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub layer_scale_init_value: Option<f32>,
    pub adanorm_num_embeddings: Option<usize>,
    pub bias: bool,
    pub input_kernel_size: usize,
    pub dw_kernel_size: usize,
}

/// Head geometry (`ISTFTHead` init args).
#[derive(Debug, Clone)]
pub struct VocosHeadConfig {
    pub dim: usize,
    pub n_fft: usize,
    pub hop_length: usize,
}

/// Mel feature-extractor geometry (`MelSpectrogramFeatures` init
/// args).
#[derive(Debug, Clone)]
pub struct VocosMelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    /// The reference accepts "center" or "same" but both run the same
    /// centered-STFT math.
    pub padding: String,
}

/// Which feature extractor the config selects.
#[derive(Debug, Clone)]
pub enum VocosFeatureConfig {
    Mel(VocosMelConfig),
    Encodec {
        encodec_model: String,
        bandwidths: Vec<f32>,
    },
}

/// Parsed `config.yaml` (the `from_hparams` structure).
#[derive(Debug, Clone)]
pub struct VocosConfig {
    pub feature_extractor: VocosFeatureConfig,
    pub backbone: VocosBackboneConfig,
    pub head: VocosHeadConfig,
}

impl VocosConfig {
    /// Parses the vocos `config.yaml` (a `class_path` + `init_args`
    /// mapping per section). Only the subset of YAML those files use
    /// is supported: nested mappings and scalar values.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let root = parse_yaml_mapping(text)?;
        let section = |name: &str| -> Result<&serde_json::Value> {
            root.get(name).ok_or_else(|| SpeechError::BadConfig {
                field: name.to_string(),
                why: "missing section in config.yaml".to_string(),
            })
        };
        let fe = section("feature_extractor")?;
        let class_path = fe
            .get("class_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let empty = serde_json::Value::Object(serde_json::Map::new());
        let fe_args = fe.get("init_args").unwrap_or(&empty);
        let feature_extractor = if class_path.contains("MelSpectrogramFeatures") {
            VocosFeatureConfig::Mel(VocosMelConfig {
                sample_rate: fe_args
                    .get("sample_rate")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(24000) as u32,
                n_fft: fe_args
                    .get("n_fft")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1024) as usize,
                hop_length: fe_args
                    .get("hop_length")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as usize,
                n_mels: fe_args
                    .get("n_mels")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100) as usize,
                padding: fe_args
                    .get("padding")
                    .and_then(|v| v.as_str())
                    .unwrap_or("center")
                    .to_string(),
            })
        } else if class_path.contains("EncodecFeatures") {
            let bandwidths = fe_args
                .get("bandwidths")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_else(|| vec![1.5, 3.0, 6.0, 12.0]);
            VocosFeatureConfig::Encodec {
                encodec_model: fe_args
                    .get("encodec_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("encodec_24khz")
                    .to_string(),
                bandwidths,
            }
        } else {
            return Err(SpeechError::Unsupported {
                why: format!("feature_extractor class {class_path:?} not supported"),
            });
        };
        let bb = section("backbone")?;
        let bb_args = bb.get("init_args").unwrap_or(&empty);
        let backbone = VocosBackboneConfig {
            input_channels: bb_args
                .get("input_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(100) as usize,
            dim: bb_args.get("dim").and_then(|v| v.as_u64()).unwrap_or(512) as usize,
            intermediate_dim: bb_args
                .get("intermediate_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(1536) as usize,
            num_layers: bb_args
                .get("num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            layer_scale_init_value: bb_args
                .get("layer_scale_init_value")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            adanorm_num_embeddings: bb_args
                .get("adanorm_num_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            bias: bb_args
                .get("bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            input_kernel_size: bb_args
                .get("input_kernel_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(7) as usize,
            dw_kernel_size: bb_args
                .get("dw_kernel_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(7) as usize,
        };
        let hd = section("head")?;
        let hd_args = hd.get("init_args").unwrap_or(&empty);
        let head = VocosHeadConfig {
            dim: hd_args.get("dim").and_then(|v| v.as_u64()).unwrap_or(512) as usize,
            n_fft: hd_args
                .get("n_fft")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024) as usize,
            hop_length: hd_args
                .get("hop_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize,
        };
        Ok(VocosConfig {
            feature_extractor,
            backbone,
            head,
        })
    }
}

/// Minimal YAML subset parser for the vocos config: nested mappings by
/// indentation, `key: value` scalars (int, float, bool, null, quoted
/// or bare strings), and `#` comments. Values come back as
/// `serde_json::Value`s so the config readers stay uniform.
pub(crate) fn parse_yaml_mapping(text: &str) -> Result<serde_json::Value> {
    let mut lines = parse_yaml_lines(text);
    for raw in text.lines() {
        let no_comment = match raw.find('#') {
            // '#' inside quotes is unusual in these configs; strip the
            // first comment marker outside quotes.
            Some(i) if !raw[..i].ends_with('"') => &raw[..i],
            _ => raw,
        };
        if no_comment.trim().is_empty() || no_comment.trim() == "---" {
            continue;
        }
        let indent = no_comment.len() - no_comment.trim_start().len();
        let body = no_comment.trim();
        let (key, value) = match body.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        lines.push(Line {
            indent,
            key: key.to_string(),
            value: if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            },
        });
    }
    let mut pos = 0usize;
    parse_block(
        &lines,
        &mut pos,
        lines.first().map(|l| l.indent).unwrap_or(0),
    )
}

#[derive(Debug)]
struct Line {
    indent: usize,
    key: String,
    value: Option<String>,
}

fn parse_yaml_lines(text: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    for raw in text.lines() {
        let no_comment = match raw.find('#') {
            Some(i) if !raw[..i].ends_with('"') => &raw[..i],
            _ => raw,
        };
        if no_comment.trim().is_empty() || no_comment.trim() == "---" {
            continue;
        }
        let indent = no_comment.len() - no_comment.trim_start().len();
        let body = no_comment.trim();
        let (key, value) = match body.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        lines.push(Line {
            indent,
            key: key.to_string(),
            value: if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            },
        });
    }
    lines
}

fn parse_block(lines: &[Line], pos: &mut usize, indent: usize) -> Result<serde_json::Value> {
    let mut map = serde_json::Map::new();
    while *pos < lines.len() {
        let line_indent = lines[*pos].indent;
        if line_indent < indent {
            break;
        }
        if line_indent > indent {
            return Err(SpeechError::BadConfig {
                field: lines[*pos].key.clone(),
                why: "unexpected indentation in config.yaml".to_string(),
            });
        }
        let line = &lines[*pos];
        let key = line.key.clone();
        match &line.value {
            Some(v) => {
                map.insert(key, scalar(v));
                *pos += 1;
            }
            None => {
                *pos += 1;
                let child_indent = lines.get(*pos).map(|l| l.indent).unwrap_or(0);
                if child_indent <= indent {
                    map.insert(key, serde_json::Value::Object(serde_json::Map::new()));
                } else {
                    let child = parse_block(lines, pos, child_indent)?;
                    map.insert(key, child);
                }
            }
        }
    }
    Ok(serde_json::Value::Object(map))
}

fn scalar(raw: &str) -> serde_json::Value {
    let v = raw.trim().trim_matches('"').trim_matches('\'');
    if v == "null" || v == "~" {
        serde_json::Value::Null
    } else if v == "true" {
        serde_json::Value::Bool(true)
    } else if v == "false" {
        serde_json::Value::Bool(false)
    } else if let Ok(i) = v.parse::<i64>() {
        serde_json::Value::Number(i.into())
    } else if let Ok(f) = v.parse::<f64>() {
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::String(v.to_string())
    }
}

/// Symmetric Hann (the reference `hanning` default, `periodic=False`).
fn symmetric_hann(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())) as f32
        })
        .collect()
}

/// Whisper-style log-mel front end (the reference `log_mel_
/// spectrogram`): centered reflect STFT with the symmetric Hann window,
/// final frame dropped, HTK mel without normalization, log with a
/// 1e-5 floor. Returns `[n_mels, frames]`.
pub fn log_mel_spectrogram(samples: &[f32], cfg: &VocosMelConfig) -> Result<Vec<f32>> {
    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    if !n_fft.is_power_of_two() {
        return Err(SpeechError::Audio(
            AudioError::NonPowerOfTwoSize { size: n_fft }.to_string(),
        ));
    }
    // The reference passes a precomputed hanning(n_fft) array to its
    // stft, so the win_length argument is never applied to it: the
    // window is the full symmetric Hann.
    let window = symmetric_hann(n_fft);
    // Center reflect padding of n_fft / 2 on both sides.
    let pad = n_fft / 2;
    if samples.len() < 2 {
        return Err(SpeechError::Input {
            why: "log-mel needs at least two samples".to_string(),
        });
    }
    if samples.len() <= pad + 1 {
        return Err(SpeechError::Input {
            why: "signal too short for the centered reflect pad".to_string(),
        });
    }
    let padded_len = samples.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    // prefix = x[1 .. pad + 1] reversed; suffix = x[len - (pad + 1) ..
    // len - 1] reversed (the edge sample is not repeated).
    for i in 0..pad {
        padded[i] = samples[1 + pad - 1 - i];
        padded[pad + samples.len() + i] = samples[samples.len() - 2 - i];
    }
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    let num_frames = 1 + (padded_len - n_fft) / hop;
    if num_frames < 2 {
        return Err(SpeechError::Input {
            why: "not enough frames for a log-mel".to_string(),
        });
    }
    let plan = RealFftPlan::new(n_fft)?;
    let bins = n_fft / 2 + 1;
    // magnitudes[bin, frame]
    let mut magnitudes = vec![0.0f32; bins * num_frames];
    let mut frame = vec![0.0f32; n_fft];
    for f in 0..num_frames {
        for i in 0..n_fft {
            frame[i] = padded[f * hop + i] * window[i];
        }
        let spectrum = plan.forward(&frame)?;
        for b in 0..bins {
            let c = &spectrum[b];
            magnitudes[b * num_frames + f] = (c.re * c.re + c.im * c.im).sqrt();
        }
    }
    // Drop the final frame (the reference slices freqs[:-1]).
    let frames_kept = num_frames - 1;
    let filterbank = mel_filterbank(cfg.n_mels, n_fft, cfg.sample_rate, 0.0, None, MelScale::Htk)?;
    let mut out = vec![0.0f32; cfg.n_mels * frames_kept];
    for m in 0..cfg.n_mels {
        let weights = &filterbank.weights[m * filterbank.num_bins..(m + 1) * filterbank.num_bins];
        for f in 0..frames_kept {
            let mut acc = 0.0f32;
            for b in 0..bins {
                acc += magnitudes[b * num_frames + f] * weights[b];
            }
            let v = acc.max(1e-5);
            out[m * frames_kept + f] = v.ln();
        }
    }
    Ok(out)
}

struct ConvNeXtBlock {
    /// Depthwise conv, PyTorch layout `[dim, 1, K]`.
    dwconv_weight: Vec<f32>,
    dwconv_bias: Option<Vec<f32>>,
    dim: usize,
    dw_kernel: usize,
    norm_w: Vec<f32>,
    norm_b: Option<Vec<f32>>,
    adanorm: Option<AdaLayerNorm>,
    pwconv1_w: Vec<f32>,
    pwconv1_b: Option<Vec<f32>>,
    intermediate: usize,
    pwconv2_w: Vec<f32>,
    pwconv2_b: Option<Vec<f32>>,
    gamma: Option<Vec<f32>>,
}

struct AdaLayerNorm {
    eps: f32,
    dim: usize,
    num_embeddings: usize,
    /// The reference replaces the scale weight with ones and the shift
    /// weight with zeros; the biases stay learned.
    scale_bias: Vec<f32>,
    shift_bias: Vec<f32>,
}

impl AdaLayerNorm {
    fn forward(&self, x: &mut [f32], embedding: &[f32], frames: usize) {
        // scale[c] = sum(embedding) + scale_bias[c]; shift[c] = 0 * ... + shift_bias[c].
        let mut scale = vec![0.0f32; self.dim];
        let mut shift = vec![0.0f32; self.dim];
        for c in 0..self.dim {
            let mut s = self.scale_bias[c];
            for v in embedding {
                s += v;
            }
            scale[c] = s;
            shift[c] = self.shift_bias[c];
        }
        layernorm_affine_free(x, self.dim, frames, self.eps);
        for f in 0..frames {
            for c in 0..self.dim {
                x[c * frames + f] = x[c * frames + f] * scale[c] + shift[c];
            }
        }
        let _ = self.num_embeddings;
    }
}

/// LayerNorm over the channel dimension of `x [ch, frames]` with no
/// affine parameters (`mx.fast.layer_norm` with weight=None).
fn layernorm_affine_free(x: &mut [f32], ch: usize, frames: usize, eps: f32) {
    for f in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * frames + f];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * frames + f] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..ch {
            x[c * frames + f] = (x[c * frames + f] - mean) * inv;
        }
    }
}

/// LayerNorm over channels of `x [ch, frames]` with affine weights.
fn layernorm_channels(
    x: &mut [f32],
    ch: usize,
    frames: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) {
    for f in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * frames + f];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * frames + f] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..ch {
            x[c * frames + f] = (x[c * frames + f] - mean) * inv * w[c] + b.map_or(0.0, |bb| bb[c]);
        }
    }
}

impl ConvNeXtBlock {
    fn forward(&self, x: &mut [f32], frames: usize, bandwidth_embedding: Option<&[f32]>) {
        let dim = self.dim;
        // Depthwise conv over frames.
        let mut h = ops::conv1d(
            x,
            &self.dwconv_weight,
            self.dwconv_bias.as_deref(),
            dim,
            dim,
            self.dw_kernel,
            1,
            self.dw_kernel / 2,
            1,
            dim,
        );
        if let Some(adanorm) = &self.adanorm {
            let emb = bandwidth_embedding
                .map(|e| e.to_vec())
                .unwrap_or_else(|| vec![0.0; adanorm.num_embeddings]);
            adanorm.forward(&mut h, &emb, frames);
        } else {
            layernorm_channels(
                &mut h,
                dim,
                frames,
                &self.norm_w,
                self.norm_b.as_deref(),
                1e-6,
            );
        }
        // pwconv1 + GELU + pwconv2, per frame.
        let inter = self.intermediate;
        let mut h2 = vec![0.0f32; inter * frames];
        for f in 0..frames {
            for o in 0..inter {
                let mut acc = self.pwconv1_b.as_deref().map_or(0.0, |b| b[o]);
                let w_row = &self.pwconv1_w[o * dim..(o + 1) * dim];
                for c in 0..dim {
                    acc += h[c * frames + f] * w_row[c];
                }
                h2[o * frames + f] = acc;
            }
        }
        for v in &mut h2 {
            ops::gelu_erf(std::slice::from_mut(v));
        }
        let mut h3 = vec![0.0f32; dim * frames];
        for f in 0..frames {
            for o in 0..dim {
                let mut acc = self.pwconv2_b.as_deref().map_or(0.0, |b| b[o]);
                let w_row = &self.pwconv2_w[o * inter..(o + 1) * inter];
                for c in 0..inter {
                    acc += h2[c * frames + f] * w_row[c];
                }
                h3[o * frames + f] = acc;
            }
        }
        if let Some(gamma) = &self.gamma {
            for c in 0..dim {
                for f in 0..frames {
                    h3[c * frames + f] *= gamma[c];
                }
            }
        }
        for (dst, src) in x.iter_mut().zip(&h3) {
            *dst += src;
        }
    }
}

struct Backbone {
    /// Embed conv weight, PyTorch layout `[dim, in_ch, K]`.
    embed_weight: Vec<f32>,
    embed_bias: Option<Vec<f32>>,
    in_ch: usize,
    dim: usize,
    input_kernel: usize,
    adanorm: Option<AdaLayerNorm>,
    norm_w: Vec<f32>,
    norm_b: Option<Vec<f32>>,
    blocks: Vec<ConvNeXtBlock>,
    final_norm_w: Vec<f32>,
    final_norm_b: Option<Vec<f32>>,
}

impl Backbone {
    fn forward(&self, x: &[f32], frames: usize, bandwidth_embedding: Option<&[f32]>) -> Vec<f32> {
        let mut h = ops::conv1d(
            x,
            &self.embed_weight,
            self.embed_bias.as_deref(),
            self.in_ch,
            self.dim,
            self.input_kernel,
            1,
            self.input_kernel / 2,
            1,
            1,
        );
        if let Some(adanorm) = &self.adanorm {
            let emb = bandwidth_embedding
                .map(|e| e.to_vec())
                .unwrap_or_else(|| vec![0.0; adanorm.num_embeddings]);
            adanorm.forward(&mut h, &emb, frames);
        } else {
            layernorm_channels(
                &mut h,
                self.dim,
                frames,
                &self.norm_w,
                self.norm_b.as_deref(),
                1e-6,
            );
        }
        for block in &self.blocks {
            block.forward(&mut h, frames, bandwidth_embedding);
        }
        layernorm_channels(
            &mut h,
            self.dim,
            frames,
            &self.final_norm_w,
            self.final_norm_b.as_deref(),
            1e-6,
        );
        h
    }
}

struct Head {
    /// `[n_fft + 2, dim]` linear weight.
    out_w: Vec<f32>,
    out_b: Option<Vec<f32>>,
    dim: usize,
    n_fft: usize,
    hop_length: usize,
}

impl Head {
    /// `x [dim, frames]` -> mono samples.
    fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let n_fft = self.n_fft;
        let rows = n_fft + 2;
        let half = n_fft / 2 + 1;
        // Linear per frame.
        let mut spec = vec![0.0f32; rows * frames];
        for f in 0..frames {
            for o in 0..rows {
                let mut acc = self.out_b.as_deref().map_or(0.0, |b| b[o]);
                let w_row = &self.out_w[o * self.dim..(o + 1) * self.dim];
                for c in 0..self.dim {
                    acc += x[c * frames + f] * w_row[c];
                }
                spec[o * frames + f] = acc;
            }
        }
        // mag = exp(clamp(max log-mag)); spectrum = mag * e^{i p}.
        let _plan = RealFftPlan::new(n_fft)?;
        let mut spectra = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut bins = Vec::with_capacity(half);
            for b in 0..half {
                let mag = spec[b * frames + f].exp().min(1e2);
                let p = spec[(half + b) * frames + f];
                bins.push(crate::fft::ComplexF32::new(mag * p.cos(), mag * p.sin()));
            }
            spectra.push(bins);
        }
        // Overlap-add iSTFT with window-sum normalization (the
        // reference istft default normalized=False), center crop. The
        // spectra are arbitrary complex, so the inverse is the numpy
        // irfft: Hermitian-extend the half and run a complex inverse
        // (RealFftPlan::inverse is only equivalent for real-FFT input).
        let cfft = crate::fft::ComplexFftPlan::new(n_fft)?;
        let mut ext = vec![crate::fft::ComplexF32::default(); n_fft];
        let window = symmetric_hann(n_fft);
        let total = (frames - 1) * self.hop_length + n_fft;
        let mut acc = vec![0.0f64; total];
        let mut win_sum = vec![0.0f64; total];
        for (f, bins) in spectra.iter().enumerate() {
            ext[..half].copy_from_slice(bins);
            // half == n_fft / 2 + 1, so the conjugate part starts at
            // index n_fft / 2 + 1 == half with ext[k] = conj(bins[n - k]).
            for k in half..n_fft {
                ext[k] = bins[n_fft - k].conj();
            }
            let frame_time = cfft.inverse(&ext)?;
            let off = f * self.hop_length;
            for i in 0..n_fft {
                acc[off + i] += (frame_time[i].re * window[i]) as f64;
                win_sum[off + i] += window[i] as f64;
            }
        }
        let start = n_fft / 2;
        let end = total - start;
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            let w = win_sum[i];
            out.push(if w > 1e-10 { (acc[i] / w) as f32 } else { 0.0 });
        }
        Ok(out)
    }
}

/// Which front end was loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    Mel,
    Encodec,
}

/// Loaded Vocos vocoder, batch-of-one.
pub struct Vocos {
    pub config: VocosConfig,
    pub kind: FeatureKind,
    backbone: Backbone,
    head: Head,
}

impl Vocos {
    /// Opens a vocos checkpoint directory: `config.yaml` plus
    /// `model.safetensors` (the `from_pretrained` layout). The mel
    /// variant is self-contained; the encodec variant loads the
    /// backbone and head and takes the encodec features through
    /// [`Vocos::decode_from_codes`].
    pub fn open(dir: &Path) -> Result<Vocos> {
        let text = std::fs::read_to_string(dir.join("config.yaml")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.yaml".to_string(),
                why: e.to_string(),
            }
        })?;
        let config = VocosConfig::from_yaml(&text)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Vocos::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: VocosConfig, file: &SafetensorsFile) -> Result<Vocos> {
        let kind = match &config.feature_extractor {
            VocosFeatureConfig::Mel(_) => FeatureKind::Mel,
            VocosFeatureConfig::Encodec { .. } => FeatureKind::Encodec,
        };
        let bb = &config.backbone;
        let bb_prefix = "backbone";
        let embed_weight = load_f32_shaped(
            file,
            &format!("{bb_prefix}.embed.weight"),
            &[bb.dim, bb.input_channels, bb.input_kernel_size],
        )?;
        let embed_bias = if file.contains_tensor(&format!("{bb_prefix}.embed.bias")) {
            Some(load_f32_shaped(
                file,
                &format!("{bb_prefix}.embed.bias"),
                &[bb.dim],
            )?)
        } else {
            None
        };
        let (adanorm, norm_w, norm_b) = if let Some(num_emb) = bb.adanorm_num_embeddings {
            let scale_bias =
                load_f32_shaped(file, &format!("{bb_prefix}.norm.scale.bias"), &[bb.dim])?;
            let shift_bias =
                load_f32_shaped(file, &format!("{bb_prefix}.norm.shift.bias"), &[bb.dim])?;
            // The scale/shift weights are overridden to ones/zeros and
            // are absent from checkpoints; the biases carry the terms.
            (
                Some(AdaLayerNorm {
                    eps: 1e-6,
                    dim: bb.dim,
                    num_embeddings: num_emb,
                    scale_bias,
                    shift_bias,
                }),
                Vec::new(),
                None,
            )
        } else {
            (
                None,
                load_f32_shaped(file, &format!("{bb_prefix}.norm.weight"), &[bb.dim])?,
                Some(load_f32_shaped(
                    file,
                    &format!("{bb_prefix}.norm.bias"),
                    &[bb.dim],
                )?),
            )
        };
        let init_value = bb
            .layer_scale_init_value
            .unwrap_or(1.0 / bb.num_layers as f32);
        let mut blocks = Vec::with_capacity(bb.num_layers);
        for i in 0..bb.num_layers {
            let prefix = format!("{bb_prefix}.convnext.{i}");
            let dwconv_weight = load_f32_shaped(
                file,
                &format!("{prefix}.dwconv.weight"),
                &[bb.dim, 1, bb.dw_kernel_size],
            )?;
            let dwconv_bias = if file.contains_tensor(&format!("{prefix}.dwconv.bias")) {
                Some(load_f32_shaped(
                    file,
                    &format!("{prefix}.dwconv.bias"),
                    &[bb.dim],
                )?)
            } else {
                None
            };
            let (block_adanorm, norm_w, norm_b) = if let Some(num_emb) = bb.adanorm_num_embeddings {
                (
                    Some(AdaLayerNorm {
                        eps: 1e-6,
                        dim: bb.dim,
                        num_embeddings: num_emb,
                        scale_bias: load_f32_shaped(
                            file,
                            &format!("{prefix}.norm.scale.bias"),
                            &[bb.dim],
                        )?,
                        shift_bias: load_f32_shaped(
                            file,
                            &format!("{prefix}.norm.shift.bias"),
                            &[bb.dim],
                        )?,
                    }),
                    Vec::new(),
                    None,
                )
            } else {
                (
                    None,
                    load_f32_shaped(file, &format!("{prefix}.norm.weight"), &[bb.dim])?,
                    Some(load_f32_shaped(
                        file,
                        &format!("{prefix}.norm.bias"),
                        &[bb.dim],
                    )?),
                )
            };
            let pwconv1_w = load_f32_shaped(
                file,
                &format!("{prefix}.pwconv1.weight"),
                &[bb.intermediate_dim, bb.dim],
            )?;
            let pwconv1_b = if file.contains_tensor(&format!("{prefix}.pwconv1.bias")) {
                Some(load_f32_shaped(
                    file,
                    &format!("{prefix}.pwconv1.bias"),
                    &[bb.intermediate_dim],
                )?)
            } else {
                None
            };
            let pwconv2_w = load_f32_shaped(
                file,
                &format!("{prefix}.pwconv2.weight"),
                &[bb.dim, bb.intermediate_dim],
            )?;
            let pwconv2_b = if file.contains_tensor(&format!("{prefix}.pwconv2.bias")) {
                Some(load_f32_shaped(
                    file,
                    &format!("{prefix}.pwconv2.bias"),
                    &[bb.dim],
                )?)
            } else {
                None
            };
            let gamma = load_f32_shaped(file, &format!("{prefix}.gamma"), &[bb.dim])?;
            let gamma = if init_value > 0.0 { Some(gamma) } else { None };
            blocks.push(ConvNeXtBlock {
                dwconv_weight,
                dwconv_bias,
                dim: bb.dim,
                dw_kernel: bb.dw_kernel_size,
                norm_w,
                norm_b,
                adanorm: block_adanorm,
                pwconv1_w,
                pwconv1_b,
                intermediate: bb.intermediate_dim,
                pwconv2_w,
                pwconv2_b,
                gamma,
            });
        }
        let final_norm_w = load_f32_shaped(
            file,
            &format!("{bb_prefix}.final_layer_norm.weight"),
            &[bb.dim],
        )?;
        let final_norm_b = if config.backbone.bias
            && file.contains_tensor(&format!("{bb_prefix}.final_layer_norm.bias"))
        {
            Some(load_f32_shaped(
                file,
                &format!("{bb_prefix}.final_layer_norm.bias"),
                &[bb.dim],
            )?)
        } else {
            None
        };
        let backbone = Backbone {
            embed_weight,
            embed_bias,
            in_ch: bb.input_channels,
            dim: bb.dim,
            input_kernel: bb.input_kernel_size,
            adanorm,
            norm_w,
            norm_b,
            blocks,
            final_norm_w,
            final_norm_b,
        };
        let hd = &config.head;
        let out_w = load_f32_shaped(file, "head.out.weight", &[hd.n_fft + 2, hd.dim])?;
        let out_b = if file.contains_tensor("head.out.bias") {
            Some(load_f32_shaped(file, "head.out.bias", &[hd.n_fft + 2])?)
        } else {
            None
        };
        let head = Head {
            out_w,
            out_b,
            dim: hd.dim,
            n_fft: hd.n_fft,
            hop_length: hd.hop_length,
        };
        Ok(Vocos {
            config,
            kind,
            backbone,
            head,
        })
    }

    /// Decodes mel features `[n_mels, frames]` to mono samples.
    pub fn decode(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        if self.kind != FeatureKind::Mel {
            return Err(SpeechError::Unsupported {
                why: "decode(features) requires the mel front end; use \
                      decode_from_codes for the encodec variant"
                    .to_string(),
            });
        }
        self.decode_features(features, frames, None)
    }

    fn decode_features(
        &self,
        features: &[f32],
        frames: usize,
        bandwidth_embedding: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let x = self.backbone.forward(features, frames, bandwidth_embedding);
        self.head.forward(&x, frames)
    }

    /// Decodes encodec code levels through the codebook-sum features
    /// path (`get_features_from_codes`) and the backbone. The
    /// `codebooks [num_q][codebook_size][dim]` slice comes from the
    /// loaded encodec quantizer layers for the target bandwidth; see
    /// the family README.
    pub fn decode_from_codes(
        &self,
        codes: &[Vec<i32>],
        codebooks: &[&[f32]],
        _codebook_size: usize,
        codebook_dim: usize,
        bandwidth_embedding: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Err(SpeechError::Input {
                why: "no code levels".to_string(),
            });
        }
        if codebooks.len() < codes.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} code levels but {} codebooks",
                    codes.len(),
                    codebooks.len()
                ),
            });
        }
        let frames = codes[0].len();
        let mut features = vec![0.0f32; codebook_dim * frames];
        for (level, level_codes) in codes.iter().enumerate() {
            if level_codes.len() != frames {
                return Err(SpeechError::Input {
                    why: "ragged code levels".to_string(),
                });
            }
            let book = codebooks[level];
            for (f, &code) in level_codes.iter().enumerate() {
                let idx = code as usize;
                for d in 0..codebook_dim {
                    features[d * frames + f] += book[idx * codebook_dim + d];
                }
            }
        }
        self.decode_features(&features, frames, bandwidth_embedding)
    }

    /// Mel log-mel features for the configured mel front end.
    pub fn mel_features(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let VocosFeatureConfig::Mel(cfg) = &self.config.feature_extractor else {
            return Err(SpeechError::Unsupported {
                why: "model does not use the mel front end".to_string(),
            });
        };
        let features = log_mel_spectrogram(samples, cfg)?;
        let frames = features.len() / cfg.n_mels;
        Ok((features, frames))
    }
}

#[cfg(test)]
mod tests;
