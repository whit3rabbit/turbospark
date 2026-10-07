//! Sortformer speaker diarization (offline v1 path): who-spoke-when
//! probabilities for up to four speakers.
//!
//! Reference: `mlx_audio/vad/models/sortformer/` (sortformer.py, config.py)
//! at the cloned v0.5.7 revision, itself ported from NVIDIA NeMo's
//! `SortformerEncLabelModel`. The architecture is a FastConformer encoder
//! (depthwise-striding conv subsampling, then Conformer layers with
//! relative positional attention), a BART-style post-LN Transformer
//! encoder over a learned positional table, and sortformer output modules
//! ending in per-speaker sigmoids.
//!
//! Layout conventions for this port:
//! - Encoder activations are t-major `[time, d_model]` row-major, matching
//!   the reference's channels-last `(b, t, c)`.
//! - `ops::conv1d` / `ops::conv2d` consume channel-major inputs, so the
//!   Conformer convolution module transposes in and out.
//! - Checkpoint conv tensors stay in PyTorch layout (Conv2d `[O, I, KH,
//!   KW]`, Conv1d `[O, I, K]`), which is what the v1 HuggingFace
//!   safetensors ship after the reference's `sanitize` remap; `ops`
//!   consumes them directly. The converted v2.1 layout (`layers_N` keys,
//!   MLX conv layouts) is accepted as-is by loading the tensors it names.
//!
//! Batch size one is assumed throughout: every entry point in this crate
//! diarizes a single waveform.

use std::path::Path;

use serde_json::Value;
use turbospark_audio::mel::{mel_filterbank, MelFilterbank, MelScale};
use turbospark_audio::stft::{stft, StftOptions};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

/// The log guard NeMo's `FilterbankFeatures` adds before the mel log.
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
/// The epsilon in the per-feature normalization denominator.
const NORM_CONSTANT: f32 = 1e-5;
/// The Conformer feed-forward residual factor.
const FC_FACTOR: f32 = 0.5;

// =============================================================================
// Configuration
// =============================================================================

/// FastConformer encoder configuration (`fc_encoder_config`).
#[derive(Debug, Clone)]
pub struct FcEncoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub attention_bias: bool,
    pub scale_input: bool,
}

/// Transformer encoder configuration (`tf_encoder_config`).
#[derive(Debug, Clone)]
pub struct TfEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub layer_norm_eps: f32,
    pub max_source_positions: usize,
    pub k_proj_bias: bool,
}

/// Sortformer head and streaming configuration (`modules_config`).
///
/// The AOSC fields are only consulted when `use_aosc` is set (v2.1);
/// defaults mirror the reference `config.py`.
#[derive(Debug, Clone)]
pub struct ModulesConfig {
    pub num_speakers: usize,
    pub fc_d_model: usize,
    pub tf_d_model: usize,
    pub subsampling_factor: usize,
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub spkcache_update_period: usize,
    pub chunk_left_context: usize,
    pub chunk_right_context: usize,
    pub spkcache_sil_frames_per_spk: usize,
    pub pred_score_threshold: f32,
    pub max_index: usize,
    pub scores_boost_latest: f32,
    pub sil_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
    pub use_aosc: bool,
}

/// Feature extractor configuration (`processor_config`).
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub preemphasis: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        ProcessorConfig {
            feature_size: 80,
            sampling_rate: 16_000,
            hop_length: 160,
            n_fft: 512,
            win_length: 400,
            preemphasis: 0.97,
        }
    }
}

/// Full model configuration as shipped in `config.json`.
#[derive(Debug, Clone)]
pub struct SortformerConfig {
    pub fc_encoder: FcEncoderConfig,
    pub tf_encoder: TfEncoderConfig,
    pub modules: ModulesConfig,
    pub processor: ProcessorConfig,
}

fn field_u64(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn field_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn field_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

impl SortformerConfig {
    /// Parses `config.json`; every field falls back to the reference
    /// `config.py` default when absent, so the v1 checkpoint (which omits
    /// `processor_config` and `use_aosc`) loads unchanged.
    pub fn from_json(v: &Value) -> Result<Self> {
        let section = |name: &str| -> Result<&Value> {
            v.get(name).ok_or_else(|| SpeechError::BadConfig {
                field: name.to_string(),
                why: "missing".into(),
            })
        };
        let fc = section("fc_encoder_config")?;
        let tf = section("tf_encoder_config")?;
        let modules = section("modules_config")?;
        let u = |x: &Value, k: &str, d: u64| field_u64(x, k, d) as usize;
        Ok(SortformerConfig {
            fc_encoder: FcEncoderConfig {
                hidden_size: u(fc, "hidden_size", 512),
                num_hidden_layers: u(fc, "num_hidden_layers", 18),
                num_attention_heads: u(fc, "num_attention_heads", 8),
                intermediate_size: u(fc, "intermediate_size", 2048),
                num_mel_bins: u(fc, "num_mel_bins", 80),
                conv_kernel_size: u(fc, "conv_kernel_size", 9),
                subsampling_conv_channels: u(fc, "subsampling_conv_channels", 256),
                subsampling_conv_kernel_size: u(fc, "subsampling_conv_kernel_size", 3),
                subsampling_conv_stride: u(fc, "subsampling_conv_stride", 2),
                attention_bias: field_bool(fc, "attention_bias", true),
                scale_input: field_bool(fc, "scale_input", true),
            },
            tf_encoder: TfEncoderConfig {
                d_model: u(tf, "d_model", 192),
                encoder_layers: u(tf, "encoder_layers", 18),
                encoder_attention_heads: u(tf, "encoder_attention_heads", 8),
                encoder_ffn_dim: u(tf, "encoder_ffn_dim", 768),
                layer_norm_eps: field_f64(tf, "layer_norm_eps", 1e-5) as f32,
                max_source_positions: u(tf, "max_source_positions", 1500),
                k_proj_bias: field_bool(tf, "k_proj_bias", false),
            },
            modules: ModulesConfig {
                num_speakers: u(modules, "num_speakers", 4),
                fc_d_model: u(modules, "fc_d_model", 512),
                tf_d_model: u(modules, "tf_d_model", 192),
                subsampling_factor: u(modules, "subsampling_factor", 8),
                chunk_len: u(modules, "chunk_len", 188),
                fifo_len: u(modules, "fifo_len", 0),
                spkcache_len: u(modules, "spkcache_len", 188),
                spkcache_update_period: u(modules, "spkcache_update_period", 188),
                chunk_left_context: u(modules, "chunk_left_context", 1),
                chunk_right_context: u(modules, "chunk_right_context", 1),
                spkcache_sil_frames_per_spk: u(modules, "spkcache_sil_frames_per_spk", 5),
                pred_score_threshold: field_f64(modules, "pred_score_threshold", 1e-6) as f32,
                max_index: u(modules, "max_index", 10_000),
                scores_boost_latest: field_f64(modules, "scores_boost_latest", 0.5) as f32,
                sil_threshold: field_f64(modules, "sil_threshold", 0.1) as f32,
                strong_boost_rate: field_f64(modules, "strong_boost_rate", 0.3) as f32,
                weak_boost_rate: field_f64(modules, "weak_boost_rate", 0.7) as f32,
                min_pos_scores_rate: field_f64(modules, "min_pos_scores_rate", 0.5) as f32,
                use_aosc: field_bool(modules, "use_aosc", false),
            },
            processor: match v.get("processor_config") {
                Some(p) => ProcessorConfig {
                    feature_size: u(p, "feature_size", 80),
                    sampling_rate: u(p, "sampling_rate", 16_000) as u32,
                    hop_length: u(p, "hop_length", 160),
                    n_fft: u(p, "n_fft", 512),
                    win_length: u(p, "win_length", 400),
                    preemphasis: field_f64(p, "preemphasis", 0.97) as f32,
                },
                None => ProcessorConfig::default(),
            },
        })
    }
}

// =============================================================================
// Weights
// =============================================================================

/// A Conv2d in PyTorch `[O, I, KH, KW]` layout with its bias.
struct Conv2d {
    w: Vec<f32>,
    b: Vec<f32>,
}

/// The depthwise-striding conv subsampling front end (factor 8).
struct Subsampling {
    conv0: Conv2d,
    dw2: Conv2d,
    pw3: Conv2d,
    dw5: Conv2d,
    pw6: Conv2d,
    linear_w: Vec<f32>,
    linear_b: Vec<f32>,
    conv_channels: usize,
    kernel_size: usize,
    stride: usize,
}

/// Relative-position multi-head attention weights for one Conformer layer.
struct RelPosAttention {
    q_w: Vec<f32>,
    q_b: Option<Vec<f32>>,
    k_w: Vec<f32>,
    k_b: Option<Vec<f32>>,
    v_w: Vec<f32>,
    v_b: Option<Vec<f32>>,
    o_w: Vec<f32>,
    o_b: Option<Vec<f32>>,
    rel_k_w: Vec<f32>,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

struct ConformerFeedForward {
    l1_w: Vec<f32>,
    l1_b: Option<Vec<f32>>,
    l2_w: Vec<f32>,
    l2_b: Option<Vec<f32>>,
}

/// Inference batch norm: running statistics baked at training time.
struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

/// GLU + depthwise conv + batch norm + pointwise, per Conformer layer.
struct ConformerConvolution {
    pw1_w: Vec<f32>,
    pw1_b: Option<Vec<f32>>,
    dw_w: Vec<f32>,
    dw_b: Option<Vec<f32>>,
    norm: BatchNorm,
    pw2_w: Vec<f32>,
    pw2_b: Option<Vec<f32>>,
}

struct ConformerLayer {
    norm_ff1_w: Vec<f32>,
    norm_ff1_b: Vec<f32>,
    ff1: ConformerFeedForward,
    norm_att_w: Vec<f32>,
    norm_att_b: Vec<f32>,
    attn: RelPosAttention,
    norm_conv_w: Vec<f32>,
    norm_conv_b: Vec<f32>,
    conv: ConformerConvolution,
    norm_ff2_w: Vec<f32>,
    norm_ff2_b: Vec<f32>,
    ff2: ConformerFeedForward,
    norm_out_w: Vec<f32>,
    norm_out_b: Vec<f32>,
}

pub(crate) struct FastConformer {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
}

/// Standard multi-head attention for one Transformer encoder layer.
struct TfAttention {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Option<Vec<f32>>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
}

struct TfLayer {
    attn: TfAttention,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

struct TfEncoder {
    /// Learned positional table `[max_source_positions, d_model]`.
    embed_positions: Vec<f32>,
    layers: Vec<TfLayer>,
}

/// Sortformer output modules.
struct SortformerModules {
    encoder_proj_w: Vec<f32>,
    encoder_proj_b: Vec<f32>,
    hidden_to_hidden_w: Vec<f32>,
    hidden_to_hidden_b: Vec<f32>,
    hidden_to_spks_w: Vec<f32>,
    hidden_to_spks_b: Vec<f32>,
    /// Loaded because the checkpoint ships it, but the reference only
    /// uses it in the training-time joint forward, never at inference.
    #[allow(dead_code)]
    cache_to_spks_w: Vec<f32>,
    #[allow(dead_code)]
    cache_to_spks_b: Vec<f32>,
}

/// A loaded Sortformer diarization model.
pub struct Sortformer {
    config: SortformerConfig,
    fc: FastConformer,
    tf: TfEncoder,
    modules: SortformerModules,
    mel_fb: MelFilterbank,
}

fn load_vec(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>> {
    file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
        name: name.to_string(),
        why: format!("load failed: {e}"),
    })
}

fn load_optional_vec(file: &SafetensorsFile, name: &str) -> Result<Option<Vec<f32>>> {
    if file.contains_tensor(name) {
        load_vec(file, name).map(Some)
    } else {
        Ok(None)
    }
}

/// Loads one subsampling conv (weight + bias) with layout detection.
fn load_conv2d_pair(
    file: &SafetensorsFile,
    base: &str,
    out: usize,
    in_ch: usize,
    k1: usize,
    k2: usize,
) -> Result<Conv2d> {
    Ok(Conv2d {
        w: load_conv_weight(file, &format!("{base}.weight"), out, in_ch, k1, k2)?,
        b: load_vec(file, &format!("{base}.bias"))?,
    })
}

/// Loads a conv weight, accepting either the PyTorch layout the v1
/// HuggingFace conversion ships (Conv2d `[O, I, KH, KW]`, Conv1d
/// `[O, I, K]`) or the MLX layout convert.py writes for v2.1 (`[O, KH,
/// KW, I]`, `[O, K, I]`), and returning PyTorch order for `ops`.
fn load_conv_weight(
    file: &SafetensorsFile,
    name: &str,
    out: usize,
    in_ch: usize,
    k1: usize,
    k2: usize,
) -> Result<Vec<f32>> {
    let raw = load_vec(file, name)?;
    let shape = file
        .descriptor(name)
        .ok_or_else(|| SpeechError::Tensor {
            name: name.to_string(),
            why: "missing descriptor".into(),
        })?
        .shape
        .clone();
    let permute_4d = |raw: &[f32]| {
        let mut w = vec![0.0f32; raw.len()];
        for o in 0..out {
            for i in 0..in_ch {
                for a in 0..k1 {
                    for b in 0..k2 {
                        w[o * in_ch * k1 * k2 + i * k1 * k2 + a * k2 + b] =
                            raw[o * k1 * k2 * in_ch + a * k2 * in_ch + b * in_ch + i];
                    }
                }
            }
        }
        w
    };
    let permute_3d = |raw: &[f32]| {
        let mut w = vec![0.0f32; raw.len()];
        for o in 0..out {
            for i in 0..in_ch {
                for k in 0..k1 {
                    w[o * in_ch * k1 + i * k1 + k] = raw[o * k1 * in_ch + k * in_ch + i];
                }
            }
        }
        w
    };
    match shape.as_slice() {
        [o, i, a, b] if *o == out && *i == in_ch && *a == k1 && *b == k2 => Ok(raw),
        [o, a, b, i] if *o == out && *i == in_ch && *a == k1 && *b == k2 => Ok(permute_4d(&raw)),
        [o, i, k] if *o == out && *i == in_ch && *k == k1 => Ok(raw),
        [o, k, i] if *o == out && *i == in_ch && *k == k1 => Ok(permute_3d(&raw)),
        other => Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("unexpected conv shape {other:?} for {out}x{in_ch}x{k1}x{k2}"),
        }),
    }
}

impl Subsampling {
    fn load(file: &SafetensorsFile, cfg: &FcEncoderConfig) -> Result<Self> {
        // v1 HuggingFace keys use `subsampling.layers.N`; the converted
        // v2.1 layout uses `subsampling.layers_N`. Either way the conv
        // tensors are PyTorch-layout `[O, I, KH, KW]` weights.
        let converted = file
            .load_as_f32("fc_encoder.subsampling.layers_0.weight")
            .is_ok();
        let layer = |idx: usize| {
            if converted {
                format!("fc_encoder.subsampling.layers_{idx}")
            } else {
                format!("fc_encoder.subsampling.layers.{idx}")
            }
        };
        let ch = cfg.subsampling_conv_channels;
        let ks = cfg.subsampling_conv_kernel_size;
        let conv = |idx: usize, out: usize, inp: usize| {
            load_conv2d_pair(file, &layer(idx), out, inp, ks, ks)
        };
        Ok(Subsampling {
            conv0: conv(0, ch, 1)?,
            dw2: conv(2, ch, 1)?,
            pw3: load_conv2d_pair(file, &layer(3), ch, ch, 1, 1)?,
            dw5: conv(5, ch, 1)?,
            pw6: load_conv2d_pair(file, &layer(6), ch, ch, 1, 1)?,
            linear_w: load_vec(file, "fc_encoder.subsampling.linear.weight")?,
            linear_b: load_vec(file, "fc_encoder.subsampling.linear.bias")?,
            conv_channels: ch,
            kernel_size: ks,
            stride: cfg.subsampling_conv_stride,
        })
    }

    fn load_parakeet(file: &SafetensorsFile, cfg: &FcEncoderConfig) -> Result<Self> {
        let ch = cfg.subsampling_conv_channels;
        let ks = cfg.subsampling_conv_kernel_size;
        let base = "encoder.pre_encode.conv";
        let conv = |idx: usize, out: usize, inp: usize| {
            load_conv2d_pair(file, &format!("{base}.{idx}"), out, inp, ks, ks)
        };
        Ok(Subsampling {
            conv0: conv(0, ch, 1)?,
            dw2: conv(2, ch, 1)?,
            pw3: load_conv2d_pair(file, &format!("{base}.3"), ch, ch, 1, 1)?,
            dw5: conv(5, ch, 1)?,
            pw6: load_conv2d_pair(file, &format!("{base}.6"), ch, ch, 1, 1)?,
            linear_w: load_vec(file, "encoder.pre_encode.out.weight")?,
            linear_b: load_vec(file, "encoder.pre_encode.out.bias")?,
            conv_channels: ch,
            kernel_size: ks,
            stride: cfg.subsampling_conv_stride,
        })
    }

    /// Canary variant of `load_parakeet`: identical tensor names, but the
    /// output projection is a groupwise-affine quantized linear whose bias
    /// may or may not exist.
    fn load_canary(
        file: &SafetensorsFile,
        cfg: &FcEncoderConfig,
        scheme: crate::quant::QuantScheme,
    ) -> Result<Self> {
        let ch = cfg.subsampling_conv_channels;
        let ks = cfg.subsampling_conv_kernel_size;
        let base = "encoder.pre_encode.conv";
        let conv = |idx: usize, out: usize, inp: usize| {
            load_conv2d_pair(file, &format!("{base}.{idx}"), out, inp, ks, ks)
        };
        let (linear_w, linear_b) =
            crate::quant::load_quantized(file, "encoder.pre_encode.out", scheme)?;
        Ok(Subsampling {
            conv0: conv(0, ch, 1)?,
            dw2: conv(2, ch, 1)?,
            pw3: load_conv2d_pair(file, &format!("{base}.3"), ch, ch, 1, 1)?,
            dw5: conv(5, ch, 1)?,
            pw6: load_conv2d_pair(file, &format!("{base}.6"), ch, ch, 1, 1)?,
            linear_w,
            linear_b: linear_b.ok_or_else(|| SpeechError::Tensor {
                name: "encoder.pre_encode.out.bias".into(),
                why: "missing from the Canary checkpoint".into(),
            })?,
            conv_channels: ch,
            kernel_size: ks,
            stride: cfg.subsampling_conv_stride,
        })
    }
}

impl RelPosAttention {
    fn load(file: &SafetensorsFile, prefix: &str, cfg: &FcEncoderConfig) -> Result<Self> {
        let linear = |name: &str| -> Result<(Vec<f32>, Option<Vec<f32>>)> {
            let w = load_vec(file, &format!("{prefix}.{name}.weight"))?;
            let b = match cfg.attention_bias {
                true => Some(load_vec(file, &format!("{prefix}.{name}.bias"))?),
                false => None,
            };
            Ok((w, b))
        };
        let (q_w, q_b) = linear("q_proj")?;
        let (k_w, k_b) = linear("k_proj")?;
        let (v_w, v_b) = linear("v_proj")?;
        let (o_w, o_b) = linear("o_proj")?;
        Ok(RelPosAttention {
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            o_w,
            o_b,
            rel_k_w: load_vec(file, &format!("{prefix}.relative_k_proj.weight"))?,
            bias_u: load_vec(file, &format!("{prefix}.bias_u"))?,
            bias_v: load_vec(file, &format!("{prefix}.bias_v"))?,
        })
    }
}

impl ConformerConvolution {
    fn load(file: &SafetensorsFile, prefix: &str, d: usize, kernel: usize) -> Result<Self> {
        Ok(ConformerConvolution {
            pw1_w: load_conv_weight(
                file,
                &format!("{prefix}.pointwise_conv1.weight"),
                2 * d,
                d,
                1,
                1,
            )?,
            pw1_b: Some(load_vec(file, &format!("{prefix}.pointwise_conv1.bias"))?),
            dw_w: load_conv_weight(
                file,
                &format!("{prefix}.depthwise_conv.weight"),
                d,
                1,
                kernel,
                1,
            )?,
            dw_b: Some(load_vec(file, &format!("{prefix}.depthwise_conv.bias"))?),
            norm: BatchNorm {
                weight: load_vec(file, &format!("{prefix}.norm.weight"))?,
                bias: load_vec(file, &format!("{prefix}.norm.bias"))?,
                running_mean: load_vec(file, &format!("{prefix}.norm.running_mean"))?,
                running_var: load_vec(file, &format!("{prefix}.norm.running_var"))?,
            },
            pw2_w: load_conv_weight(
                file,
                &format!("{prefix}.pointwise_conv2.weight"),
                d,
                d,
                1,
                1,
            )?,
            pw2_b: Some(load_vec(file, &format!("{prefix}.pointwise_conv2.bias"))?),
        })
    }

    fn load_parakeet(
        file: &SafetensorsFile,
        prefix: &str,
        d: usize,
        kernel: usize,
    ) -> Result<Self> {
        let pointwise = |name: &str, out: usize| {
            load_conv_weight(file, &format!("{prefix}.{name}.weight"), out, d, 1, 1)
        };
        Ok(ConformerConvolution {
            pw1_w: pointwise("pointwise_conv1", 2 * d)?,
            pw1_b: load_optional_vec(file, &format!("{prefix}.pointwise_conv1.bias"))?,
            dw_w: load_conv_weight(
                file,
                &format!("{prefix}.depthwise_conv.weight"),
                d,
                1,
                kernel,
                1,
            )?,
            dw_b: load_optional_vec(file, &format!("{prefix}.depthwise_conv.bias"))?,
            norm: BatchNorm {
                weight: load_vec(file, &format!("{prefix}.batch_norm.weight"))?,
                bias: load_vec(file, &format!("{prefix}.batch_norm.bias"))?,
                running_mean: load_vec(file, &format!("{prefix}.batch_norm.running_mean"))?,
                running_var: load_vec(file, &format!("{prefix}.batch_norm.running_var"))?,
            },
            pw2_w: pointwise("pointwise_conv2", d)?,
            pw2_b: load_optional_vec(file, &format!("{prefix}.pointwise_conv2.bias"))?,
        })
    }
}

impl ConformerLayer {
    fn load(file: &SafetensorsFile, prefix: &str, cfg: &FcEncoderConfig) -> Result<Self> {
        let ff = |name: &str| -> Result<ConformerFeedForward> {
            Ok(ConformerFeedForward {
                l1_w: load_vec(file, &format!("{prefix}.{name}.linear1.weight"))?,
                l1_b: Some(load_vec(file, &format!("{prefix}.{name}.linear1.bias"))?),
                l2_w: load_vec(file, &format!("{prefix}.{name}.linear2.weight"))?,
                l2_b: Some(load_vec(file, &format!("{prefix}.{name}.linear2.bias"))?),
            })
        };
        let norm = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                load_vec(file, &format!("{prefix}.{name}.weight"))?,
                load_vec(file, &format!("{prefix}.{name}.bias"))?,
            ))
        };
        let (norm_ff1_w, norm_ff1_b) = norm("norm_feed_forward1")?;
        let (norm_att_w, norm_att_b) = norm("norm_self_att")?;
        let (norm_conv_w, norm_conv_b) = norm("norm_conv")?;
        let (norm_ff2_w, norm_ff2_b) = norm("norm_feed_forward2")?;
        let (norm_out_w, norm_out_b) = norm("norm_out")?;
        Ok(ConformerLayer {
            norm_ff1_w,
            norm_ff1_b,
            ff1: ff("feed_forward1")?,
            norm_att_w,
            norm_att_b,
            attn: RelPosAttention::load(file, &format!("{prefix}.self_attn"), cfg)?,
            norm_conv_w,
            norm_conv_b,
            conv: ConformerConvolution::load(
                file,
                &format!("{prefix}.conv"),
                cfg.hidden_size,
                cfg.conv_kernel_size,
            )?,
            norm_ff2_w,
            norm_ff2_b,
            ff2: ff("feed_forward2")?,
            norm_out_w,
            norm_out_b,
        })
    }

    fn load_parakeet(file: &SafetensorsFile, prefix: &str, cfg: &FcEncoderConfig) -> Result<Self> {
        let ff = |name: &str| -> Result<ConformerFeedForward> {
            let base = format!("{prefix}.{name}");
            Ok(ConformerFeedForward {
                l1_w: load_vec(file, &format!("{base}.linear1.weight"))?,
                l1_b: load_optional_vec(file, &format!("{base}.linear1.bias"))?,
                l2_w: load_vec(file, &format!("{base}.linear2.weight"))?,
                l2_b: load_optional_vec(file, &format!("{base}.linear2.bias"))?,
            })
        };
        let norm = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                load_vec(file, &format!("{prefix}.{name}.weight"))?,
                load_vec(file, &format!("{prefix}.{name}.bias"))?,
            ))
        };
        let (norm_ff1_w, norm_ff1_b) = norm("norm_feed_forward1")?;
        let (norm_att_w, norm_att_b) = norm("norm_self_att")?;
        let (norm_conv_w, norm_conv_b) = norm("norm_conv")?;
        let (norm_ff2_w, norm_ff2_b) = norm("norm_feed_forward2")?;
        let (norm_out_w, norm_out_b) = norm("norm_out")?;
        let attn_prefix = format!("{prefix}.self_attn");
        let projection =
            |name: &str| load_vec(file, &format!("{attn_prefix}.linear_{name}.weight"));
        Ok(ConformerLayer {
            norm_ff1_w,
            norm_ff1_b,
            ff1: ff("feed_forward1")?,
            norm_att_w,
            norm_att_b,
            attn: RelPosAttention {
                q_w: projection("q")?,
                q_b: None,
                k_w: projection("k")?,
                k_b: None,
                v_w: projection("v")?,
                v_b: None,
                o_w: projection("out")?,
                o_b: None,
                rel_k_w: load_vec(file, &format!("{attn_prefix}.linear_pos.weight"))?,
                bias_u: load_vec(file, &format!("{attn_prefix}.pos_bias_u"))?,
                bias_v: load_vec(file, &format!("{attn_prefix}.pos_bias_v"))?,
            },
            norm_conv_w,
            norm_conv_b,
            conv: ConformerConvolution::load_parakeet(
                file,
                &format!("{prefix}.conv"),
                cfg.hidden_size,
                cfg.conv_kernel_size,
            )?,
            norm_ff2_w,
            norm_ff2_b,
            ff2: ff("feed_forward2")?,
            norm_out_w,
            norm_out_b,
        })
    }

    /// Canary variant of `load_parakeet`: identical tensor names, but every
    /// linear is a groupwise-affine quantized tensor and the attention and
    /// feed-forward projections carry biases (the reference constructs the
    /// Parakeet Conformer with `use_bias: true`; `linear_pos` stays
    /// bias-free).
    fn load_canary(
        file: &SafetensorsFile,
        prefix: &str,
        cfg: &FcEncoderConfig,
        scheme: crate::quant::QuantScheme,
    ) -> Result<Self> {
        let linear =
            |name: &str| crate::quant::load_quantized(file, &format!("{prefix}.{name}"), scheme);
        let ff = |name: &str| -> Result<ConformerFeedForward> {
            let (l1_w, l1_b) = linear(&format!("{name}.linear1"))?;
            let (l2_w, l2_b) = linear(&format!("{name}.linear2"))?;
            Ok(ConformerFeedForward {
                l1_w,
                l1_b: Some(l1_b.ok_or_else(|| SpeechError::Tensor {
                    name: format!("{prefix}.{name}.linear1.bias"),
                    why: "missing from the Canary checkpoint".into(),
                })?),
                l2_w,
                l2_b: Some(l2_b.ok_or_else(|| SpeechError::Tensor {
                    name: format!("{prefix}.{name}.linear2.bias"),
                    why: "missing from the Canary checkpoint".into(),
                })?),
            })
        };
        let norm = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                load_vec(file, &format!("{prefix}.{name}.weight"))?,
                load_vec(file, &format!("{prefix}.{name}.bias"))?,
            ))
        };
        let (norm_ff1_w, norm_ff1_b) = norm("norm_feed_forward1")?;
        let (norm_att_w, norm_att_b) = norm("norm_self_att")?;
        let (norm_conv_w, norm_conv_b) = norm("norm_conv")?;
        let (norm_ff2_w, norm_ff2_b) = norm("norm_feed_forward2")?;
        let (norm_out_w, norm_out_b) = norm("norm_out")?;
        let attn_prefix = format!("{prefix}.self_attn");
        let attention = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            let (w, b) = linear(&format!("self_attn.linear_{name}"))?;
            Ok((
                w,
                b.ok_or_else(|| SpeechError::Tensor {
                    name: format!("{attn_prefix}.linear_{name}.bias"),
                    why: "missing from the Canary checkpoint".into(),
                })?,
            ))
        };
        let (q_w, q_b) = attention("q")?;
        let (k_w, k_b) = attention("k")?;
        let (v_w, v_b) = attention("v")?;
        let (o_w, o_b) = attention("out")?;
        let (rel_k_w, rel_k_b) = linear("self_attn.linear_pos")?;
        if rel_k_b.is_some() {
            return Err(SpeechError::Tensor {
                name: format!("{attn_prefix}.linear_pos.bias"),
                why: "the reference constructs linear_pos without a bias".into(),
            });
        }
        Ok(ConformerLayer {
            norm_ff1_w,
            norm_ff1_b,
            ff1: ff("feed_forward1")?,
            norm_att_w,
            norm_att_b,
            attn: RelPosAttention {
                q_w,
                q_b: Some(q_b),
                k_w,
                k_b: Some(k_b),
                v_w,
                v_b: Some(v_b),
                o_w,
                o_b: Some(o_b),
                rel_k_w,
                bias_u: load_vec(file, &format!("{attn_prefix}.pos_bias_u"))?,
                bias_v: load_vec(file, &format!("{attn_prefix}.pos_bias_v"))?,
            },
            norm_conv_w,
            norm_conv_b,
            conv: ConformerConvolution::load_parakeet(
                file,
                &format!("{prefix}.conv"),
                cfg.hidden_size,
                cfg.conv_kernel_size,
            )?,
            norm_ff2_w,
            norm_ff2_b,
            ff2: ff("feed_forward2")?,
            norm_out_w,
            norm_out_b,
        })
    }
}

impl TfLayer {
    fn load(file: &SafetensorsFile, prefix: &str, cfg: &TfEncoderConfig) -> Result<Self> {
        let k_b = match cfg.k_proj_bias {
            true => Some(load_vec(file, &format!("{prefix}.self_attn.k_proj.bias"))?),
            false => None,
        };
        Ok(TfLayer {
            attn: TfAttention {
                q_w: load_vec(file, &format!("{prefix}.self_attn.q_proj.weight"))?,
                q_b: load_vec(file, &format!("{prefix}.self_attn.q_proj.bias"))?,
                k_w: load_vec(file, &format!("{prefix}.self_attn.k_proj.weight"))?,
                k_b,
                v_w: load_vec(file, &format!("{prefix}.self_attn.v_proj.weight"))?,
                v_b: load_vec(file, &format!("{prefix}.self_attn.v_proj.bias"))?,
                o_w: load_vec(file, &format!("{prefix}.self_attn.out_proj.weight"))?,
                o_b: load_vec(file, &format!("{prefix}.self_attn.out_proj.bias"))?,
            },
            ln1_w: load_vec(file, &format!("{prefix}.self_attn_layer_norm.weight"))?,
            ln1_b: load_vec(file, &format!("{prefix}.self_attn_layer_norm.bias"))?,
            fc1_w: load_vec(file, &format!("{prefix}.fc1.weight"))?,
            fc1_b: load_vec(file, &format!("{prefix}.fc1.bias"))?,
            fc2_w: load_vec(file, &format!("{prefix}.fc2.weight"))?,
            fc2_b: load_vec(file, &format!("{prefix}.fc2.bias"))?,
            ln2_w: load_vec(file, &format!("{prefix}.final_layer_norm.weight"))?,
            ln2_b: load_vec(file, &format!("{prefix}.final_layer_norm.bias"))?,
        })
    }
}

impl Sortformer {
    /// Opens a model directory holding `config.json` and
    /// `model.safetensors` (the v1 HuggingFace conversion or the v2.1
    /// converted layout).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| {
                SpeechError::BadConfig {
                    field: "config.json".into(),
                    why: format!("unreadable: {e}"),
                }
            })?,
        )
        .map_err(|e| SpeechError::BadConfig {
            field: "config.json".into(),
            why: format!("invalid JSON: {e}"),
        })?;
        let config = SortformerConfig::from_json(&config)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;

        let fc_cfg = &config.fc_encoder;
        let layers = (0..fc_cfg.num_hidden_layers)
            .map(|i| ConformerLayer::load(&file, &format!("fc_encoder.layers.{i}"), fc_cfg))
            .collect::<Result<Vec<_>>>()?;
        let fc = FastConformer {
            subsampling: Subsampling::load(&file, fc_cfg)?,
            layers,
        };

        let tf_layers = (0..config.tf_encoder.encoder_layers)
            .map(|i| TfLayer::load(&file, &format!("tf_encoder.layers.{i}"), &config.tf_encoder))
            .collect::<Result<Vec<_>>>()?;
        let tf = TfEncoder {
            embed_positions: load_vec(&file, "tf_encoder.embed_positions.weight")?,
            layers: tf_layers,
        };

        let modules = SortformerModules {
            encoder_proj_w: load_vec(&file, "sortformer_modules.encoder_proj.weight")?,
            encoder_proj_b: load_vec(&file, "sortformer_modules.encoder_proj.bias")?,
            hidden_to_hidden_w: load_vec(
                &file,
                "sortformer_modules.first_hidden_to_hidden.weight",
            )?,
            hidden_to_hidden_b: load_vec(&file, "sortformer_modules.first_hidden_to_hidden.bias")?,
            hidden_to_spks_w: load_vec(&file, "sortformer_modules.single_hidden_to_spks.weight")?,
            hidden_to_spks_b: load_vec(&file, "sortformer_modules.single_hidden_to_spks.bias")?,
            cache_to_spks_w: load_vec(&file, "sortformer_modules.hidden_to_spks.weight")?,
            cache_to_spks_b: load_vec(&file, "sortformer_modules.hidden_to_spks.bias")?,
        };

        let mel_fb = mel_filterbank(
            config.processor.feature_size,
            config.processor.n_fft,
            config.processor.sampling_rate,
            0.0,
            None,
            MelScale::Slaney,
        )
        .map_err(|e| SpeechError::Audio(format!("mel filterbank: {e}")))?;

        Ok(Sortformer {
            config,
            fc,
            tf,
            modules,
            mel_fb,
        })
    }

    pub fn config(&self) -> &SortformerConfig {
        &self.config
    }
}

// =============================================================================
// Feature extraction (NeMo FilterbankFeatures conventions)
// =============================================================================

/// Symmetric Hann window (`hanning(size, periodic=False)` in the
/// reference), zero-padded to `n_fft` the way the reference pads it when
/// `win_length < n_fft`.
fn nemo_window(proc: &ProcessorConfig) -> Vec<f32> {
    let size = proc.win_length;
    let mut window: Vec<f32> = (0..size)
        .map(|n| {
            let denom = (size - 1) as f64;
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / denom).cos())) as f32
        })
        .collect();
    if size < proc.n_fft {
        let left = (proc.n_fft - size) / 2;
        let right = proc.n_fft - size - left;
        let mut padded = vec![0.0f32; left];
        padded.append(&mut window);
        padded.extend(std::iter::repeat_n(0.0f32, right));
        window = padded;
    }
    window
}

/// Log-mel features matching the reference `extract_mel_features`.
///
/// Returns channel-major `[n_mels, frames]` and the frame count after
/// `pad_to` (0 disables padding). `normalize` selects the per-mel-bin
/// z-score over time (Bessel-corrected variance) the v1 offline path
/// uses; the v2.1 streaming path skips it.
pub fn extract_mel_features(
    samples: &[f32],
    proc: &ProcessorConfig,
    normalize: bool,
    pad_to: usize,
    fb: &MelFilterbank,
) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "empty waveform".into(),
        });
    }
    // Preemphasis keeps the first sample: y = [x0, x[n] - c x[n-1]].
    let mut x = Vec::with_capacity(samples.len());
    x.push(samples[0]);
    for w in samples.windows(2) {
        x.push(w[1] - proc.preemphasis * w[0]);
    }

    let window = nemo_window(proc);
    // The reference centers with constant (zero) padding, unlike the
    // reflect-padded torch.stft default, so pad here and frame uncentered.
    let half = proc.n_fft / 2;
    let mut padded = vec![0.0f32; half];
    padded.append(&mut x);
    padded.extend(std::iter::repeat_n(0.0f32, half));
    let spectra = stft(
        &padded,
        &StftOptions {
            fft_size: proc.n_fft,
            hop: proc.hop_length,
            window,
            center: false,
        },
    )
    .map_err(|e| SpeechError::Audio(format!("stft: {e}")))?;

    let frames = spectra.len();
    let mut feats = vec![0.0f32; proc.feature_size * frames];
    for (t, spec) in spectra.iter().enumerate() {
        let power: Vec<f32> = spec.iter().map(|c| c.re * c.re + c.im * c.im).collect();
        let mel = fb
            .project(&power)
            .map_err(|e| SpeechError::Audio(format!("mel project: {e}")))?;
        for (m, v) in mel.into_iter().enumerate() {
            feats[m * frames + t] = (v + LOG_GUARD).ln();
        }
    }

    if normalize {
        let n = frames;
        for row in feats.chunks_exact_mut(n) {
            // The mean is accumulated in f64: short clips contain rows
            // that are exactly constant (zero-energy mel bins sit at
            // log(2^-24)), where an f32 mean can round off the constant
            // by an ulp and the division by NORM_CONSTANT then amplifies
            // that to a visible error. The reference's mean of a constant
            // row is the constant itself, so the z-score must be exactly
            // 0 there.
            let mean = (row.iter().map(|v| *v as f64).sum::<f64>() / n as f64) as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1) as f32;
            let std = var.sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean) / (std + NORM_CONSTANT);
            }
        }
    }

    if pad_to > 0 {
        let rem = frames % pad_to;
        if rem > 0 {
            let add = pad_to - rem;
            let total = frames + add;
            let mut out = vec![0.0f32; proc.feature_size * total];
            for m in 0..proc.feature_size {
                out[m * total..m * total + frames]
                    .copy_from_slice(&feats[m * frames..(m + 1) * frames]);
            }
            return Ok((out, total));
        }
    }
    Ok((feats, frames))
}

// =============================================================================
// FastConformer encoder
// =============================================================================

fn relu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// One stage of the stride-2 subsampling length formula
/// `floor((L - 1) / 2) + 1`.
fn subsample_len(len: usize) -> usize {
    (len - 1) / 2 + 1
}

impl Subsampling {
    /// `[n_mels, frames]` mel features -> `[frames/8, hidden]` embeddings,
    /// returned t-major `[time, hidden]`.
    fn forward(&self, feats: &[f32], n_mels: usize, frames: usize, hidden: usize) -> Vec<f32> {
        let ks = self.kernel_size;
        let stride = self.stride;
        let pad = (ks - 1) / 2;
        let ch = self.conv_channels;
        // Channel-major [1, frames, n_mels] image for conv2d.
        let mut img = vec![0.0f32; frames * n_mels];
        for m in 0..n_mels {
            for t in 0..frames {
                img[t * n_mels + m] = feats[m * frames + t];
            }
        }
        let mut x = ops::conv2d(
            &img,
            &self.conv0.w,
            Some(&self.conv0.b),
            1,
            ch,
            frames,
            n_mels,
            ks,
            ks,
            stride,
            pad,
            1,
        );
        let (f1, m1) = (
            (frames + 2 * pad - ks) / stride + 1,
            (n_mels + 2 * pad - ks) / stride + 1,
        );
        relu_in_place(&mut x);
        x = ops::conv2d(
            &x,
            &self.dw2.w,
            Some(&self.dw2.b),
            ch,
            ch,
            f1,
            m1,
            ks,
            ks,
            stride,
            pad,
            ch,
        );
        let (f2, m2) = (
            (f1 + 2 * pad - ks) / stride + 1,
            (m1 + 2 * pad - ks) / stride + 1,
        );
        x = ops::conv2d(
            &x,
            &self.pw3.w,
            Some(&self.pw3.b),
            ch,
            ch,
            f2,
            m2,
            1,
            1,
            1,
            0,
            1,
        );
        relu_in_place(&mut x);
        x = ops::conv2d(
            &x,
            &self.dw5.w,
            Some(&self.dw5.b),
            ch,
            ch,
            f2,
            m2,
            ks,
            ks,
            stride,
            pad,
            ch,
        );
        let (f3, m3) = (
            (f2 + 2 * pad - ks) / stride + 1,
            (m2 + 2 * pad - ks) / stride + 1,
        );
        x = ops::conv2d(
            &x,
            &self.pw6.w,
            Some(&self.pw6.b),
            ch,
            ch,
            f3,
            m3,
            1,
            1,
            1,
            0,
            1,
        );
        relu_in_place(&mut x);
        // Flatten (c, f) per time step in channel-major order, matching
        // the reference's (b, t, c, f) reshape, then project to hidden.
        let mut flat = vec![0.0f32; f3 * ch * m3];
        for t in 0..f3 {
            for c in 0..ch {
                for f in 0..m3 {
                    flat[t * ch * m3 + c * m3 + f] = x[c * f3 * m3 + t * m3 + f];
                }
            }
        }
        ops::linear(
            &flat,
            &self.linear_w,
            Some(&self.linear_b),
            f3,
            ch * m3,
            hidden,
        )
    }
}

/// Transformer-XL style relative positional encoding: `[2*seq - 1,
/// d_model]`, even columns sin, odd columns cos, positions counting down
/// from `seq - 1` to `-(seq - 1)`.
fn rel_position_encoding(seq: usize, d_model: usize) -> Vec<f32> {
    let n = 2 * seq - 1;
    let scale = -(10_000f64.ln() / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model / 2)
        .map(|j| ((2 * j) as f32 * scale).exp())
        .collect();
    let mut pe = vec![0.0f32; n * d_model];
    for i in 0..n {
        let pos = (seq as isize - 1 - i as isize) as f32;
        for (j, div) in div_term.iter().enumerate() {
            let angle = pos * div;
            pe[i * d_model + 2 * j] = angle.sin();
            pe[i * d_model + 2 * j + 1] = angle.cos();
        }
    }
    pe
}

/// Splits `[t, h * dk]` into head-major `[h, t, dk]`.
fn split_heads(x: &[f32], t: usize, h: usize, dk: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * t * dk];
    for tt in 0..t {
        for hh in 0..h {
            out[hh * t * dk + tt * dk..hh * t * dk + (tt + 1) * dk]
                .copy_from_slice(&x[tt * h * dk + hh * dk..tt * h * dk + (hh + 1) * dk]);
        }
    }
    out
}

impl RelPosAttention {
    fn forward(
        &self,
        x: &[f32],
        t: usize,
        pos_emb: &[f32],
        d: usize,
        h: usize,
        dk: usize,
    ) -> Vec<f32> {
        let q = ops::linear(x, &self.q_w, self.q_b.as_deref(), t, d, d);
        let k = ops::linear(x, &self.k_w, self.k_b.as_deref(), t, d, d);
        let v = ops::linear(x, &self.v_w, self.v_b.as_deref(), t, d, d);
        let pos_len = 2 * t - 1;
        let p = ops::linear(pos_emb, &self.rel_k_w, None, pos_len, d, d);

        let kh = split_heads(&k, t, h, dk);
        let vh = split_heads(&v, t, h, dk);
        let ph = split_heads(&p, pos_len, h, dk);
        // q stays t-major for the bias add (the reference's q_t); the
        // biases are added before the matmuls, matching the reference's
        // (q_t + bias) @ ... operation order.
        let mut qbu = q.clone();
        let mut qbv = q;
        for tt in 0..t {
            for hh in 0..h {
                for j in 0..dk {
                    let i = tt * d + hh * dk + j;
                    qbu[i] += self.bias_u[hh * dk + j];
                    qbv[i] += self.bias_v[hh * dk + j];
                }
            }
        }

        // Relative shift selects position t - 1 + key - query. Compute
        // only that position for each score, avoiding two head*time*time
        // matrices and the larger head*time*(2*time-1) matrix.
        let scale = (dk as f32).sqrt();
        let mut out = vec![0.0f32; h * t * dk];
        for hh in 0..h {
            let k_head = &kh[hh * t * dk..(hh + 1) * t * dk];
            let v_head = &vh[hh * t * dk..(hh + 1) * t * dk];
            let p_head = &ph[hh * pos_len * dk..(hh + 1) * pos_len * dk];
            let mut scores = vec![0.0f32; t];
            for tt in 0..t {
                let q_u = &qbu[tt * d + hh * dk..tt * d + (hh + 1) * dk];
                let q_v = &qbv[tt * d + hh * dk..tt * d + (hh + 1) * dk];
                for ss in 0..t {
                    let k_row = &k_head[ss * dk..(ss + 1) * dk];
                    let relative = t - 1 + ss - tt;
                    let p_row = &p_head[relative * dk..(relative + 1) * dk];
                    let content: f32 = q_u.iter().zip(k_row).map(|(a, b)| a * b).sum();
                    let position: f32 = q_v.iter().zip(p_row).map(|(a, b)| a * b).sum();
                    scores[ss] = (content + position) / scale;
                }
                ops::softmax_row(&mut scores);
                for ss in 0..t {
                    let w = scores[ss];
                    for j in 0..dk {
                        out[hh * t * dk + tt * dk + j] += w * v_head[ss * dk + j];
                    }
                }
            }
        }
        // Merge heads back to t-major and project.
        let mut merged = vec![0.0f32; t * d];
        for tt in 0..t {
            for hh in 0..h {
                merged[tt * d + hh * dk..tt * d + (hh + 1) * dk]
                    .copy_from_slice(&out[hh * t * dk + tt * dk..hh * t * dk + (tt + 1) * dk]);
            }
        }
        ops::linear(&merged, &self.o_w, self.o_b.as_deref(), t, d, d)
    }
}

impl ConformerFeedForward {
    fn forward(&self, x: &[f32], t: usize, d: usize, d_ff: usize) -> Vec<f32> {
        let mut h = ops::linear(x, &self.l1_w, self.l1_b.as_deref(), t, d, d_ff);
        for v in h.iter_mut() {
            *v *= ops::sigmoid(*v);
        }
        ops::linear(&h, &self.l2_w, self.l2_b.as_deref(), t, d_ff, d)
    }
}

impl ConformerConvolution {
    /// `[t, d]` in, `[t, d]` out; transposes to channel-major for the
    /// convs and back.
    fn forward(&self, x: &[f32], t: usize, d: usize, kernel: usize) -> Vec<f32> {
        let mut xc = vec![0.0f32; d * t];
        for tt in 0..t {
            for c in 0..d {
                xc[c * t + tt] = x[tt * d + c];
            }
        }
        let pw1 = ops::conv1d(
            &xc,
            &self.pw1_w,
            self.pw1_b.as_deref(),
            d,
            2 * d,
            1,
            1,
            0,
            1,
            1,
        );
        // GLU over the channel axis: first d channels gated by the rest.
        let mut g = vec![0.0f32; d * t];
        for c in 0..d {
            for tt in 0..t {
                g[c * t + tt] = pw1[c * t + tt] * ops::sigmoid(pw1[(d + c) * t + tt]);
            }
        }
        let mut dw = ops::conv1d(
            &g,
            &self.dw_w,
            self.dw_b.as_deref(),
            d,
            d,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            d,
        );
        for c in 0..d {
            let mean = self.norm.running_mean[c];
            let inv = 1.0 / (self.norm.running_var[c] + 1e-5).sqrt();
            let w = self.norm.weight[c];
            let b = self.norm.bias[c];
            for tt in 0..t {
                dw[c * t + tt] = (dw[c * t + tt] - mean) * inv * w + b;
            }
        }
        for v in dw.iter_mut() {
            *v *= ops::sigmoid(*v);
        }
        let pw2 = ops::conv1d(&dw, &self.pw2_w, self.pw2_b.as_deref(), d, d, 1, 1, 0, 1, 1);
        let mut out = vec![0.0f32; t * d];
        for c in 0..d {
            for tt in 0..t {
                out[tt * d + c] = pw2[c * t + tt];
            }
        }
        out
    }
}

impl ConformerLayer {
    fn forward(&self, x: &[f32], pos_emb: &[f32], t: usize, cfg: &FcEncoderConfig) -> Vec<f32> {
        let d = cfg.hidden_size;
        let h = cfg.num_attention_heads;
        let dk = d / h;
        let d_ff = cfg.intermediate_size;

        // FF1 with the half-weight residual
        let mut residual = x.to_vec();
        let mut y = x.to_vec();
        ops::layernorm(&mut y, t, d, &self.norm_ff1_w, Some(&self.norm_ff1_b), 1e-5);
        let y = self.ff1.forward(&y, t, d, d_ff);
        for (r, v) in residual.iter_mut().zip(&y) {
            *r += v * FC_FACTOR;
        }

        // Self-attention
        let mut y = residual.clone();
        ops::layernorm(&mut y, t, d, &self.norm_att_w, Some(&self.norm_att_b), 1e-5);
        let y = self.attn.forward(&y, t, pos_emb, d, h, dk);
        for (r, v) in residual.iter_mut().zip(&y) {
            *r += v;
        }

        // Convolution module
        let mut y = residual.clone();
        ops::layernorm(
            &mut y,
            t,
            d,
            &self.norm_conv_w,
            Some(&self.norm_conv_b),
            1e-5,
        );
        let y = self.conv.forward(&y, t, d, cfg.conv_kernel_size);
        for (r, v) in residual.iter_mut().zip(&y) {
            *r += v;
        }

        // FF2 with the half-weight residual, then the final norm
        let mut y = residual.clone();
        ops::layernorm(&mut y, t, d, &self.norm_ff2_w, Some(&self.norm_ff2_b), 1e-5);
        let y = self.ff2.forward(&y, t, d, d_ff);
        for (r, v) in residual.iter_mut().zip(&y) {
            *r += v * FC_FACTOR;
        }
        let mut out = residual;
        ops::layernorm(
            &mut out,
            t,
            d,
            &self.norm_out_w,
            Some(&self.norm_out_b),
            1e-5,
        );
        out
    }
}

impl FastConformer {
    /// Loads the FastConformer encoder from the MLX Parakeet checkpoint
    /// layout. The encoder is shared with Sortformer, while Parakeet uses
    /// `encoder.pre_encode` and omits the disabled linear/convolution biases.
    pub(crate) fn load_parakeet(file: &SafetensorsFile, cfg: &FcEncoderConfig) -> Result<Self> {
        if cfg.subsampling_conv_kernel_size != 3 || cfg.subsampling_conv_stride != 2 {
            return Err(SpeechError::Unsupported {
                why: "Parakeet encoder requires 3x3 stride-2 subsampling convolutions".into(),
            });
        }
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| ConformerLayer::load_parakeet(file, &format!("encoder.layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling: Subsampling::load_parakeet(file, cfg)?,
            layers,
        })
    }

    /// Loads the FastConformer encoder from the Canary MLX conversion. The
    /// tensor names match the Parakeet layout, but the conversion quantizes
    /// every linear layer to the groupwise affine `scheme` (the reference is
    /// `mlx_audio/stt/models/canary/canary.py`, which reuses the Parakeet
    /// `Conformer` with `use_bias: true`). The convolution and batch norm
    /// tensors stay unquantized and load through the Parakeet path.
    pub(crate) fn load_canary(
        file: &SafetensorsFile,
        cfg: &FcEncoderConfig,
        scheme: crate::quant::QuantScheme,
    ) -> Result<Self> {
        if cfg.subsampling_conv_kernel_size != 3 || cfg.subsampling_conv_stride != 2 {
            return Err(SpeechError::Unsupported {
                why: "Canary encoder requires 3x3 stride-2 subsampling convolutions".into(),
            });
        }
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| ConformerLayer::load_canary(file, &format!("encoder.layers.{i}"), cfg, scheme))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling: Subsampling::load_canary(file, cfg, scheme)?,
            layers,
        })
    }

    /// Encodes NeMo mel features stored as `[mel, frame]` and returns
    /// `[valid frame, hidden]` features plus their valid length.
    pub(crate) fn encode_mel(
        &self,
        features: &[f32],
        mel_frames: usize,
        mel_length: usize,
        cfg: &FcEncoderConfig,
    ) -> Result<(Vec<f32>, usize)> {
        if mel_frames == 0
            || mel_length == 0
            || mel_length > mel_frames
            || features.len() != cfg.num_mel_bins.saturating_mul(mel_frames)
        {
            return Err(SpeechError::Input {
                why:
                    "Parakeet mel features must be nonempty [mel, frame] with a valid frame length"
                        .into(),
            });
        }
        let mut valid_len = mel_length;
        for _ in 0..3 {
            valid_len = subsample_len(valid_len);
        }
        let mut hidden =
            self.subsampling
                .forward(features, cfg.num_mel_bins, mel_frames, cfg.hidden_size);
        if hidden.len() != valid_len.saturating_mul(cfg.hidden_size) {
            return Err(SpeechError::Tensor {
                name: "encoder.pre_encode".into(),
                why: format!(
                    "subsampling produced {} frames, expected {valid_len}",
                    hidden.len() / cfg.hidden_size
                ),
            });
        }
        self.encode(&mut hidden, valid_len, cfg);
        Ok((hidden, valid_len))
    }

    /// Conformer stack over pre-encoded embeddings (the reference's
    /// `encode`); input and output are t-major `[t, hidden]`.
    pub(crate) fn encode(&self, embs: &mut Vec<f32>, t: usize, cfg: &FcEncoderConfig) {
        if cfg.scale_input {
            let scale = (cfg.hidden_size as f64).sqrt() as f32;
            for v in embs.iter_mut() {
                *v *= scale;
            }
        }
        let pos_emb = rel_position_encoding(t, cfg.hidden_size);
        for layer in &self.layers {
            *embs = layer.forward(embs, &pos_emb, t, cfg);
        }
    }
}

// =============================================================================
// Transformer encoder (BART-style, post-LN)
// =============================================================================

impl TfAttention {
    fn forward(&self, x: &[f32], t: usize, valid_len: usize, d: usize, heads: usize) -> Vec<f32> {
        let dk = d / heads;
        let q = ops::linear(x, &self.q_w, Some(&self.q_b), t, d, d);
        let k = ops::linear(x, &self.k_w, self.k_b.as_deref(), t, d, d);
        let v = ops::linear(x, &self.v_w, Some(&self.v_b), t, d, d);
        let qh = split_heads(&q, t, heads, dk);
        let kh = split_heads(&k, t, heads, dk);
        let vh = split_heads(&v, t, heads, dk);
        let scale = (dk as f64).powf(-0.5) as f32;
        let mut out = vec![0.0f32; heads * t * dk];
        for hh in 0..heads {
            for tt in 0..t {
                let q_row = &qh[hh * t * dk + tt * dk..hh * t * dk + (tt + 1) * dk];
                let mut scores = vec![0.0f32; t];
                for ss in 0..t {
                    let k_row = &kh[hh * t * dk + ss * dk..hh * t * dk + (ss + 1) * dk];
                    let mut s = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
                    if ss >= valid_len {
                        s += -1e4;
                    }
                    scores[ss] = s;
                }
                ops::softmax_row(&mut scores);
                for ss in 0..t {
                    let w = scores[ss];
                    for j in 0..dk {
                        out[hh * t * dk + tt * dk + j] += w * vh[hh * t * dk + ss * dk + j];
                    }
                }
            }
        }
        let mut merged = vec![0.0f32; t * d];
        for tt in 0..t {
            for hh in 0..heads {
                merged[tt * d + hh * dk..tt * d + (hh + 1) * dk]
                    .copy_from_slice(&out[hh * t * dk + tt * dk..hh * t * dk + (tt + 1) * dk]);
            }
        }
        ops::linear(&merged, &self.o_w, Some(&self.o_b), t, d, d)
    }
}

impl TfLayer {
    fn forward(&self, x: &[f32], t: usize, valid_len: usize, cfg: &TfEncoderConfig) -> Vec<f32> {
        let d = cfg.d_model;
        let heads = cfg.encoder_attention_heads;
        let a = self.attn.forward(x, t, valid_len, d, heads);
        let mut y = x.to_vec();
        for (yy, aa) in y.iter_mut().zip(&a) {
            *yy += aa;
        }
        ops::layernorm(
            &mut y,
            t,
            d,
            &self.ln1_w,
            Some(&self.ln1_b),
            cfg.layer_norm_eps,
        );
        let mut z = ops::linear(
            &y,
            &self.fc1_w,
            Some(&self.fc1_b),
            t,
            d,
            cfg.encoder_ffn_dim,
        );
        relu_in_place(&mut z);
        let z = ops::linear(
            &z,
            &self.fc2_w,
            Some(&self.fc2_b),
            t,
            cfg.encoder_ffn_dim,
            d,
        );
        for (yy, zz) in y.iter_mut().zip(&z) {
            *yy += zz;
        }
        ops::layernorm(
            &mut y,
            t,
            d,
            &self.ln2_w,
            Some(&self.ln2_b),
            cfg.layer_norm_eps,
        );
        y
    }
}

impl TfEncoder {
    fn forward(&self, x: &[f32], t: usize, valid_len: usize, cfg: &TfEncoderConfig) -> Vec<f32> {
        assert!(
            self.embed_positions.len() >= t * cfg.d_model,
            "sequence length {t} exceeds the positional table"
        );
        let mut out = x.to_vec();
        for tt in 0..t {
            for c in 0..cfg.d_model {
                out[tt * cfg.d_model + c] += self.embed_positions[tt * cfg.d_model + c];
            }
        }
        for layer in &self.layers {
            out = layer.forward(&out, t, valid_len, cfg);
        }
        out
    }
}

// =============================================================================
// Sortformer output modules and the offline forward
// =============================================================================

impl SortformerModules {
    /// The speaker sigmoid head: relu -> hidden -> relu -> per-speaker.
    fn forward_speaker_sigmoids(&self, x: &[f32], t: usize, cfg: &ModulesConfig) -> Vec<f32> {
        let din = cfg.tf_d_model;
        let mut h = x.to_vec();
        relu_in_place(&mut h);
        let h = ops::linear(
            &h,
            &self.hidden_to_hidden_w,
            Some(&self.hidden_to_hidden_b),
            t,
            din,
            din,
        );
        let mut h = h;
        relu_in_place(&mut h);
        let s = ops::linear(
            &h,
            &self.hidden_to_spks_w,
            Some(&self.hidden_to_spks_b),
            t,
            din,
            cfg.num_speakers,
        );
        s.iter().map(|v| ops::sigmoid(*v)).collect()
    }
}

impl Sortformer {
    /// Offline forward over log-mel features: `[n_mels, frames]`
    /// channel-major with `mel_length` valid frames, returning
    /// `[out_frames, num_speakers]` probabilities.
    ///
    /// The reference multiplies the predictions by the length mask; here
    /// the output length always equals the subsampled valid length (the
    /// conv subsampling length formula matches the conv output exactly),
    /// so the mask is all-valid and the multiply is an identity.
    pub fn forward(&self, features: &[f32], mel_frames: usize, mel_length: usize) -> Vec<f32> {
        let fc_cfg = self.config.fc_encoder.clone();
        let mc = self.config.modules.clone();
        let mut embs = self.fc.subsampling.forward(
            features,
            fc_cfg.num_mel_bins,
            mel_frames,
            fc_cfg.hidden_size,
        );
        let emb_len = {
            let mut l = mel_length;
            for _ in 0..3 {
                l = subsample_len(l);
            }
            l
        };
        debug_assert_eq!(embs.len() / fc_cfg.hidden_size, emb_len);
        self.fc.encode(&mut embs, emb_len, &fc_cfg);

        let proj = ops::linear(
            &embs,
            &self.modules.encoder_proj_w,
            Some(&self.modules.encoder_proj_b),
            emb_len,
            mc.fc_d_model,
            mc.tf_d_model,
        );
        let tf_out = self
            .tf
            .forward(&proj, emb_len, emb_len, &self.config.tf_encoder);
        self.modules.forward_speaker_sigmoids(&tf_out, emb_len, &mc)
    }
}

// =============================================================================
// Diarization output, segment conversion, and the generate() front end
// =============================================================================

/// One speaker segment in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// Offline diarization result for one waveform.
#[derive(Debug, Clone)]
pub struct DiarizationOutput {
    pub segments: Vec<DiarizationSegment>,
    /// Per-diarization-frame probabilities, `[frames, num_speakers]`.
    pub speaker_probs: Vec<f32>,
    pub num_speakers: usize,
}

/// Thresholds and post-processing for `generate`.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub threshold: f32,
    /// Minimum segment duration in seconds.
    pub min_duration: f64,
    /// Maximum gap in seconds between segments that get merged.
    pub merge_gap: f64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        GenerateOptions {
            threshold: 0.5,
            min_duration: 0.0,
            merge_gap: 0.0,
        }
    }
}

/// Energy-based leading/trailing silence trim matching the reference
/// `_trim_silence`: 30 ms frames, silence below 1% of peak RMS, at least
/// 0.5 s of consecutive speech required on both ends. Returns the trimmed
/// waveform and the trim offset in samples.
pub fn trim_silence(samples: &[f32], sample_rate: u32) -> (Vec<f32>, usize) {
    let frame_len = sample_rate as usize * 30 / 1000;
    // `max(3, int(min_speech_sec * 1000 / frame_ms))` with the reference's
    // 30 ms frames and 0.5 s minimum: 16 frames.
    let min_speech_frames = 16;
    let num_frames = samples.len() / frame_len;
    if num_frames < min_speech_frames * 2 {
        return (samples.to_vec(), 0);
    }
    let mut energy = vec![0.0f32; num_frames];
    for (i, e) in energy.iter_mut().enumerate() {
        let frame = &samples[i * frame_len..(i + 1) * frame_len];
        *e = (frame.iter().map(|v| v * v).sum::<f32>() / frame_len as f32).sqrt();
    }
    let threshold = energy.iter().cloned().fold(0.0f32, f32::max) * 0.01;
    let speech: Vec<bool> = energy.iter().map(|e| *e > threshold).collect();

    let mut start_frame = 0;
    for i in 0..=(num_frames - min_speech_frames) {
        if speech[i..i + min_speech_frames].iter().all(|s| *s) {
            start_frame = i;
            break;
        }
    }
    let mut end_frame = num_frames;
    for i in (min_speech_frames - 1..num_frames).rev() {
        if speech[i + 1 - min_speech_frames..=i].iter().all(|s| *s) {
            end_frame = i + 1;
            break;
        }
    }
    let start_sample = start_frame * frame_len;
    let end_sample = (end_frame * frame_len).min(samples.len());
    if start_sample == 0 && end_sample == samples.len() {
        return (samples.to_vec(), 0);
    }
    (samples[start_sample..end_sample].to_vec(), start_sample)
}

/// Converts frame-level probabilities to time segments (the reference
/// `_preds_to_segments`): threshold, minimum duration, optional gap merge,
/// segments sorted by start time.
pub fn preds_to_segments(
    preds: &[f32],
    num_speakers: usize,
    frame_duration: f64,
    threshold: f32,
    min_duration: f64,
    merge_gap: f64,
) -> Vec<DiarizationSegment> {
    let frames = preds.len() / num_speakers;
    let mut segments = Vec::new();
    for spk in 0..num_speakers {
        let activity: Vec<bool> = (0..frames)
            .map(|t| preds[t * num_speakers + spk] > threshold)
            .collect();
        if !activity.iter().any(|a| *a) {
            continue;
        }
        // Padded first-difference to find run boundaries.
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut prev = false;
        for (t, cur) in activity.iter().enumerate() {
            if *cur && !prev {
                starts.push(t);
            }
            if !*cur && prev {
                ends.push(t);
            }
            prev = *cur;
        }
        if prev {
            ends.push(frames);
        }
        let mut spk_segments: Vec<DiarizationSegment> = starts
            .iter()
            .zip(&ends)
            .map(|(&s, &e)| DiarizationSegment {
                start: s as f64 * frame_duration,
                end: e as f64 * frame_duration,
                speaker: spk,
            })
            .filter(|seg| seg.end - seg.start >= min_duration)
            .collect();
        if merge_gap > 0.0 && spk_segments.len() > 1 {
            let mut merged = vec![spk_segments[0].clone()];
            for seg in &spk_segments[1..] {
                let last = merged.last_mut().unwrap();
                if seg.start - last.end <= merge_gap {
                    last.end = seg.end;
                } else {
                    merged.push(seg.clone());
                }
            }
            spk_segments = merged;
        }
        segments.extend(spk_segments);
    }
    segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    segments
}

impl Sortformer {
    /// Full offline diarization matching the reference `generate`: trim
    /// silence, peak-normalize, extract per-feature-normalized log-mel
    /// features padded to 16 frames, run the forward pass, and threshold
    /// into segments. Input must be mono at the processor sample rate.
    pub fn generate(
        &self,
        samples: &[f32],
        sample_rate: u32,
        opts: &GenerateOptions,
    ) -> Result<DiarizationOutput> {
        let proc = self.config.processor.clone();
        if sample_rate != proc.sampling_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "sample rate must be {} Hz; resample before calling",
                    proc.sampling_rate
                ),
            });
        }
        let (waveform, trim_offset) = trim_silence(samples, proc.sampling_rate);
        let trim_offset_sec = trim_offset as f64 / proc.sampling_rate as f64;

        let peak = waveform.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let gain = 1.0 / (peak + 1e-3);
        let waveform: Vec<f32> = waveform.iter().map(|v| v * gain).collect();

        let (features, frames) = extract_mel_features(&waveform, &proc, true, 16, &self.mel_fb)?;
        let preds = self.forward(&features, frames, frames);

        let subsampling_factor = self.config.modules.subsampling_factor;
        let frame_duration =
            (proc.hop_length * subsampling_factor) as f64 / proc.sampling_rate as f64;
        let mut segments = preds_to_segments(
            &preds,
            self.config.modules.num_speakers,
            frame_duration,
            opts.threshold,
            opts.min_duration,
            opts.merge_gap,
        );
        if trim_offset > 0 {
            for seg in &mut segments {
                seg.start += trim_offset_sec;
                seg.end += trim_offset_sec;
            }
        }
        let num_speakers = segments
            .iter()
            .map(|s| s.speaker)
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok(DiarizationOutput {
            segments,
            speaker_probs: preds,
            num_speakers,
        })
    }
}

// =============================================================================
// Streaming inference
// =============================================================================

/// State carried between streaming chunks: the long-term speaker cache
/// and the recent-context FIFO, both holding pre-encoded (subsampled)
/// embeddings plus the predictions the encoder assigned them, the number
/// of diarization frames emitted so far, and the running silence profile
/// the AOSC compression maintains.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// `[spkcache_frames, emb_dim]` pre-encoded embeddings.
    pub spkcache: Vec<f32>,
    /// `[spkcache_frames, num_speakers]`.
    pub spkcache_preds: Vec<f32>,
    /// `[fifo_frames, emb_dim]`.
    pub fifo: Vec<f32>,
    /// `[fifo_frames, num_speakers]`.
    pub fifo_preds: Vec<f32>,
    /// Total diarization frames emitted so far.
    pub frames_processed: usize,
    /// Running mean silence embedding `[emb_dim]` (AOSC only).
    pub mean_sil_emb: Vec<f32>,
    /// Count of silence frames seen (AOSC only).
    pub n_sil_frames: f32,
}

impl StreamingState {
    fn spkcache_len(&self, emb_dim: usize) -> usize {
        self.spkcache.len() / emb_dim
    }
    fn fifo_len(&self, emb_dim: usize) -> usize {
        self.fifo.len() / emb_dim
    }
}

/// Options for the streaming entry points (`feed` and the file-mode
/// `generate_stream`), mirroring the reference streaming parameters.
#[derive(Debug, Clone)]
pub struct StreamingOptions {
    pub threshold: f32,
    pub min_duration: f64,
    pub merge_gap: f64,
    /// Maximum speaker-cache size in diarization frames.
    pub spkcache_max: usize,
    /// Maximum FIFO size in diarization frames.
    pub fifo_max: usize,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        StreamingOptions {
            threshold: 0.5,
            min_duration: 0.0,
            merge_gap: 0.0,
            spkcache_max: 188,
            fifo_max: 188,
        }
    }
}

/// One chunk result from the streaming entry points.
#[derive(Debug, Clone)]
pub struct StreamingChunk {
    pub segments: Vec<DiarizationSegment>,
    /// `[chunk_frames, num_speakers]` probabilities for this chunk.
    pub speaker_probs: Vec<f32>,
    pub num_speakers: usize,
}

impl Sortformer {
    /// An empty streaming state.
    pub fn init_streaming_state(&self) -> StreamingState {
        let emb_dim = self.config.fc_encoder.hidden_size;
        StreamingState {
            spkcache: Vec::new(),
            spkcache_preds: Vec::new(),
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            frames_processed: 0,
            mean_sil_emb: vec![0.0; emb_dim],
            n_sil_frames: 0.0,
        }
    }

    /// Pre-encodes one chunk's mel features through the conv subsampling.
    fn pre_encode(
        &self,
        chunk_features: &[f32],
        chunk_mel_frames: usize,
        chunk_mel_length: usize,
    ) -> (Vec<f32>, usize) {
        let fc_cfg = &self.config.fc_encoder;
        let embs = self.fc.subsampling.forward(
            chunk_features,
            fc_cfg.num_mel_bins,
            chunk_mel_frames,
            fc_cfg.hidden_size,
        );
        let mut len = chunk_mel_length;
        for _ in 0..3 {
            len = subsample_len(len);
        }
        (embs, len)
    }

    /// One streaming step over
    /// `[spkcache | fifo | left_ctx | chunk | right_ctx]`: runs the full
    /// encoder stack on the assembled sequence but returns predictions
    /// for the new chunk only, plus the state with the chunk folded into
    /// the FIFO and the context predictions refreshed.
    pub fn streaming_step(
        &self,
        chunk_features: &[f32],
        chunk_mel_frames: usize,
        chunk_mel_length: usize,
        state: &StreamingState,
        right_context_embs: Option<&[f32]>,
    ) -> (Vec<f32>, StreamingState) {
        let mc = self.config.modules.clone();
        let emb_dim = self.config.fc_encoder.hidden_size;
        let use_context = mc.use_aosc;
        let lc = if use_context {
            mc.chunk_left_context
        } else {
            0
        };
        let _rc = if use_context {
            mc.chunk_right_context
        } else {
            0
        };

        let (chunk_embs, chunk_diar_len) =
            self.pre_encode(chunk_features, chunk_mel_frames, chunk_mel_length);

        // Left context copies the last fifo frames (v2.1 only).
        let fifo_len = state.fifo_len(emb_dim);
        let spkcache_len = state.spkcache_len(emb_dim);
        let left_ctx_len = if lc > 0 && fifo_len > 0 {
            lc.min(fifo_len)
        } else {
            0
        };
        let left_ctx: &[f32] = if left_ctx_len > 0 {
            &state.fifo[(fifo_len - left_ctx_len) * emb_dim..]
        } else {
            &[]
        };
        let right_len = match right_context_embs {
            Some(r) if _rc > 0 => r.len() / emb_dim,
            _ => 0,
        };

        let total_len = spkcache_len + fifo_len + left_ctx_len + chunk_diar_len + right_len;
        let mut all_embs = Vec::with_capacity(total_len * emb_dim);
        all_embs.extend_from_slice(&state.spkcache);
        all_embs.extend_from_slice(&state.fifo);
        all_embs.extend_from_slice(left_ctx);
        all_embs.extend_from_slice(&chunk_embs);
        if right_len > 0 {
            all_embs.extend_from_slice(right_context_embs.unwrap());
        }

        // Full encoder pass over the assembled sequence.
        let fc_cfg = self.config.fc_encoder.clone();
        let mut encoded = all_embs;
        if fc_cfg.scale_input {
            let scale = (fc_cfg.hidden_size as f64).sqrt() as f32;
            for v in encoded.iter_mut() {
                *v *= scale;
            }
        }
        let pos_emb = rel_position_encoding(total_len, fc_cfg.hidden_size);
        for layer in &self.fc.layers {
            encoded = layer.forward(&encoded, &pos_emb, total_len, &fc_cfg);
        }
        let proj = ops::linear(
            &encoded,
            &self.modules.encoder_proj_w,
            Some(&self.modules.encoder_proj_b),
            total_len,
            mc.fc_d_model,
            mc.tf_d_model,
        );
        let tf_out = self
            .tf
            .forward(&proj, total_len, total_len, &self.config.tf_encoder);
        let all_preds = self
            .modules
            .forward_speaker_sigmoids(&tf_out, total_len, &mc);

        // Slice out the new chunk and the refreshed context predictions.
        let chunk_start = spkcache_len + fifo_len + left_ctx_len;
        let chunk_preds = all_preds
            [chunk_start * mc.num_speakers..(chunk_start + chunk_diar_len) * mc.num_speakers]
            .to_vec();
        let updated_cache_preds = all_preds[..spkcache_len * mc.num_speakers].to_vec();
        let updated_fifo_preds = all_preds
            [spkcache_len * mc.num_speakers..(spkcache_len + fifo_len) * mc.num_speakers]
            .to_vec();

        let mut new_state = state.clone();
        if spkcache_len > 0 {
            new_state.spkcache_preds = updated_cache_preds;
        }
        if fifo_len > 0 {
            new_state.fifo_preds = updated_fifo_preds;
        }
        new_state.fifo.extend_from_slice(&chunk_embs);
        new_state.fifo_preds.extend_from_slice(&chunk_preds);
        new_state.frames_processed = state.frames_processed + chunk_diar_len;
        (chunk_preds, new_state)
    }

    /// Feeds one chunk of raw mono audio (the real-time `feed` API): the
    /// chunk is peak-normalized (v1), feature-extracted without padding,
    /// run through `streaming_step`, and the state compressed. Returns
    /// this chunk's diarization and the updated streaming state.
    pub fn feed(
        &self,
        chunk: &[f32],
        sample_rate: u32,
        state: &StreamingState,
        opts: &StreamingOptions,
    ) -> Result<(StreamingChunk, StreamingState)> {
        let proc = self.config.processor.clone();
        if sample_rate != proc.sampling_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "sample rate must be {} Hz; resample before calling",
                    proc.sampling_rate
                ),
            });
        }
        let frame_duration = self.frame_duration();
        let chunk_time_offset = state.frames_processed as f64 * frame_duration;

        // v2.1 streaming skips the per-chunk peak normalization.
        let use_aosc = self.config.modules.use_aosc;
        let chunk_mx: Vec<f32> = if use_aosc {
            chunk.to_vec()
        } else {
            let peak = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let gain = 1.0 / (peak + 1e-3);
            chunk.iter().map(|v| v * gain).collect()
        };
        let (features, frames) =
            extract_mel_features(&chunk_mx, &proc, !use_aosc, 0, &self.mel_fb)?;
        let (chunk_preds, advanced) = self.streaming_step(&features, frames, frames, state, None);
        let state = self.maybe_compress_state(advanced, opts.spkcache_max, opts.fifo_max);

        let segments = preds_to_segments(
            &chunk_preds,
            self.config.modules.num_speakers,
            frame_duration,
            opts.threshold,
            opts.min_duration,
            opts.merge_gap,
        );
        let segments: Vec<DiarizationSegment> = segments
            .into_iter()
            .map(|mut seg| {
                seg.start += chunk_time_offset;
                seg.end += chunk_time_offset;
                seg
            })
            .collect();

        let num_speakers = segments
            .iter()
            .map(|s| s.speaker)
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok((
            StreamingChunk {
                segments,
                speaker_probs: chunk_preds,
                num_speakers,
            },
            state,
        ))
    }

    /// File-mode streaming (the reference `generate_stream` over a full
    /// waveform): global feature extraction, fixed-duration chunks, and
    /// per-chunk results. Returns every chunk's result in order.
    pub fn generate_stream(
        &self,
        samples: &[f32],
        sample_rate: u32,
        chunk_duration: f64,
        opts: &StreamingOptions,
    ) -> Result<Vec<StreamingChunk>> {
        let proc = self.config.processor.clone();
        if sample_rate != proc.sampling_rate {
            return Err(SpeechError::Input {
                why: format!(
                    "sample rate must be {} Hz; resample before calling",
                    proc.sampling_rate
                ),
            });
        }
        let mc = self.config.modules.clone();
        let frame_duration = self.frame_duration();
        let use_aosc = mc.use_aosc;

        let mut spkcache_max = opts.spkcache_max;
        let mut fifo_max = opts.fifo_max;
        if use_aosc {
            spkcache_max = mc.spkcache_len;
            if mc.fifo_len > 0 {
                fifo_max = mc.fifo_len;
            }
        }

        let (waveform_full, trim_offset_sec) = if use_aosc {
            // v2.1 streaming skips silence trimming, peak norm, and the
            // per-feature normalization.
            (samples.to_vec(), 0.0)
        } else {
            let (waveform, trim_offset) = trim_silence(samples, proc.sampling_rate);
            let sec = trim_offset as f64 / proc.sampling_rate as f64;
            let peak = waveform.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let gain = 1.0 / (peak + 1e-3);
            (waveform.iter().map(|v| v * gain).collect::<Vec<f32>>(), sec)
        };

        let (features, total_mel_frames) = extract_mel_features(
            &waveform_full,
            &proc,
            !use_aosc,
            if use_aosc { 0 } else { 16 },
            &self.mel_fb,
        )?;

        let subsampling_factor = mc.subsampling_factor;
        // Python round() is half-to-even.
        let chunk_mel = {
            let raw = chunk_duration * proc.sampling_rate as f64
                / proc.hop_length as f64
                / subsampling_factor as f64;
            let rounded = raw.round_ties_even();
            (rounded as usize).max(subsampling_factor) * subsampling_factor
        };
        let chunk_mel = chunk_mel.max(subsampling_factor);

        // v2.1 file mode: pre-encode everything once for right context.
        let rc = if use_aosc { mc.chunk_right_context } else { 0 };
        let all_pre_embs: Option<Vec<f32>> = if use_aosc && rc > 0 {
            let (embs, _) = self.pre_encode(&features, total_mel_frames, total_mel_frames);
            Some(embs)
        } else {
            None
        };
        let total_emb_frames = all_pre_embs
            .as_ref()
            .map(|e| e.len() / self.config.fc_encoder.hidden_size)
            .unwrap_or(0);

        let mut results = Vec::new();
        let mut state = self.init_streaming_state();
        let mut offset_mel = 0;
        let mut emb_offset = 0;
        while offset_mel < total_mel_frames {
            let end_mel = (offset_mel + chunk_mel).min(total_mel_frames);
            let chunk_frames = end_mel - offset_mel;
            // The features are channel-major [n_mels, total_frames]; a
            // chunk takes each mel row's [offset, end) segment.
            let mut chunk_feat = vec![0.0f32; self.config.processor.feature_size * chunk_frames];
            for m in 0..self.config.processor.feature_size {
                let row = &features[m * total_mel_frames..(m + 1) * total_mel_frames];
                chunk_feat[m * chunk_frames..(m + 1) * chunk_frames]
                    .copy_from_slice(&row[offset_mel..end_mel]);
            }

            let right_ctx: Option<Vec<f32>> = match &all_pre_embs {
                Some(pre) if rc > 0 => {
                    let mut chunk_emb_len = chunk_frames;
                    for _ in 0..3 {
                        chunk_emb_len = subsample_len(chunk_emb_len);
                    }
                    let rc_start = emb_offset + chunk_emb_len;
                    let rc_end = (rc_start + rc).min(total_emb_frames);
                    emb_offset += chunk_emb_len;
                    if rc_end > rc_start {
                        Some(
                            pre[rc_start * self.config.fc_encoder.hidden_size
                                ..rc_end * self.config.fc_encoder.hidden_size]
                                .to_vec(),
                        )
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let (chunk_preds, new_state) = self.streaming_step(
                &chunk_feat,
                chunk_frames,
                chunk_frames,
                &state,
                right_ctx.as_deref(),
            );
            state = new_state;

            let chunk_time_offset = offset_mel as f64 * proc.hop_length as f64
                / proc.sampling_rate as f64
                + trim_offset_sec;
            let segments = preds_to_segments(
                &chunk_preds,
                mc.num_speakers,
                frame_duration,
                opts.threshold,
                opts.min_duration,
                opts.merge_gap,
            );
            let segments: Vec<DiarizationSegment> = segments
                .into_iter()
                .map(|mut seg| {
                    seg.start += chunk_time_offset;
                    seg.end += chunk_time_offset;
                    seg
                })
                .collect();
            let num_speakers = segments
                .iter()
                .map(|s| s.speaker)
                .collect::<std::collections::HashSet<_>>()
                .len();
            results.push(StreamingChunk {
                segments,
                speaker_probs: chunk_preds,
                num_speakers,
            });

            state = self.maybe_compress_state(state, spkcache_max, fifo_max);
            offset_mel = end_mel;
        }
        Ok(results)
    }

    fn frame_duration(&self) -> f64 {
        let proc = &self.config.processor;
        (proc.hop_length * self.config.modules.subsampling_factor) as f64
            / proc.sampling_rate as f64
    }

    /// Moves FIFO overflow into the speaker cache, compressing when the
    /// cache overflows (the reference `_maybe_compress_state`).
    fn maybe_compress_state(
        &self,
        state: StreamingState,
        spkcache_max: usize,
        fifo_max: usize,
    ) -> StreamingState {
        let mc = self.config.modules.clone();
        let emb_dim = self.config.fc_encoder.hidden_size;
        let n_spk = mc.num_speakers;
        let fifo_len = state.fifo_len(emb_dim);
        if fifo_len <= fifo_max {
            return state;
        }
        let mut pop_len = fifo_len - fifo_max;
        if mc.use_aosc {
            pop_len = pop_len.min(mc.spkcache_update_period);
        }

        let popped_embs = state.fifo[..pop_len * emb_dim].to_vec();
        let popped_preds = state.fifo_preds[..pop_len * n_spk].to_vec();

        let mut mean_sil_emb = state.mean_sil_emb.clone();
        let mut n_sil_frames = state.n_sil_frames;
        if mc.use_aosc {
            update_silence_profile(
                &mut mean_sil_emb,
                &mut n_sil_frames,
                &popped_embs,
                &popped_preds,
                pop_len,
                mc.sil_threshold,
            );
        }

        let mut cache = state.spkcache.clone();
        let mut cache_preds = state.spkcache_preds.clone();
        cache.extend_from_slice(&popped_embs);
        cache_preds.extend_from_slice(&popped_preds);

        let cache_len = cache.len() / emb_dim;
        if cache_len > spkcache_max {
            if mc.use_aosc {
                let (c, p) =
                    compress_spkcache_aosc(&cache, &cache_preds, cache_len, &mean_sil_emb, &mc);
                cache = c;
                cache_preds = p;
            } else {
                let (c, p) =
                    compress_spkcache_simple(&cache, &cache_preds, cache_len, spkcache_max, n_spk);
                cache = c;
                cache_preds = p;
            }
        }

        StreamingState {
            spkcache: cache,
            spkcache_preds: cache_preds,
            fifo: state.fifo[pop_len * emb_dim..].to_vec(),
            fifo_preds: state.fifo_preds[pop_len * n_spk..].to_vec(),
            frames_processed: state.frames_processed,
            mean_sil_emb,
            n_sil_frames,
        }
    }
}

/// Simple v1 compression: keep the `target_len` frames with the highest
/// total log speaker activity, in temporal order.
fn compress_spkcache_simple(
    embs: &[f32],
    preds: &[f32],
    frames: usize,
    target_len: usize,
    num_speakers: usize,
) -> (Vec<f32>, Vec<f32>) {
    let emb_dim = embs.len() / frames;
    let frame_scores: Vec<f32> = (0..frames)
        .map(|t| {
            (0..num_speakers)
                .map(|k| (preds[t * num_speakers + k].clamp(1e-7, 1.0)).ln())
                .sum()
        })
        .collect();
    let mut idx: Vec<usize> = (0..frames).collect();
    idx.sort_by(|&a, &b| {
        frame_scores[b]
            .partial_cmp(&frame_scores[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    idx.truncate(target_len);
    idx.sort_unstable();
    let mut out_embs = Vec::with_capacity(target_len * emb_dim);
    let mut out_preds = Vec::with_capacity(target_len * num_speakers);
    for &i in &idx {
        out_embs.extend_from_slice(&embs[i * emb_dim..(i + 1) * emb_dim]);
        out_preds.extend_from_slice(&preds[i * num_speakers..(i + 1) * num_speakers]);
    }
    (out_embs, out_preds)
}

/// AOSC silence profile update: frames whose summed speaker probability
/// falls below `sil_threshold` fold into the running mean embedding.
fn update_silence_profile(
    mean_sil_emb: &mut [f32],
    n_sil_frames: &mut f32,
    embs: &[f32],
    preds: &[f32],
    frames: usize,
    sil_threshold: f32,
) {
    let emb_dim = embs.len() / frames;
    let num_speakers = preds.len() / frames;
    let mut sil_count = 0.0f32;
    let mut sil_sum = vec![0.0f32; emb_dim];
    for t in 0..frames {
        let total: f32 = (0..num_speakers).map(|k| preds[t * num_speakers + k]).sum();
        if total < sil_threshold {
            sil_count += 1.0;
            for e in 0..emb_dim {
                sil_sum[e] += embs[t * emb_dim + e];
            }
        }
    }
    let upd_n = *n_sil_frames + sil_count;
    for e in 0..emb_dim {
        let old_sum = mean_sil_emb[e] * *n_sil_frames;
        mean_sil_emb[e] = (old_sum + sil_sum[e]) / upd_n.max(1.0);
    }
    *n_sil_frames = upd_n;
}

/// Per-frame per-speaker log-likelihood-ratio scores; high when a speaker
/// is confidently active alone.
fn log_pred_scores(preds: &[f32], frames: usize, num_speakers: usize, threshold: f32) -> Vec<f32> {
    let mut scores = vec![0.0f32; frames * num_speakers];
    for t in 0..frames {
        let mut log_1_sum = 0.0f32;
        for k in 0..num_speakers {
            let p = preds[t * num_speakers + k];
            log_1_sum += (1.0 - p).max(threshold).ln();
        }
        for k in 0..num_speakers {
            let p = preds[t * num_speakers + k];
            scores[t * num_speakers + k] =
                p.max(threshold).ln() - (1.0 - p).max(threshold).ln() + log_1_sum - 0.5f32.ln();
        }
    }
    scores
}

/// The top-`k` indices by value (descending), ties broken by the lower
/// index. The reference uses `mx.argpartition`, whose tie order is
/// unspecified; exact float ties only arise in degenerate inputs.
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// AOSC (Arrival-Order Speaker Cache) compression: score, filter,
/// boost, pad with silence slots, and gather the top frames. Shared with
/// the Nemotron diarization port, which calls it with its own config.
pub(crate) fn compress_spkcache_aosc(
    embs: &[f32],
    preds: &[f32],
    frames: usize,
    mean_sil_emb: &[f32],
    mc: &ModulesConfig,
) -> (Vec<f32>, Vec<f32>) {
    let emb_dim = embs.len() / frames;
    let n_spk = mc.num_speakers;
    let sil_per_spk = mc.spkcache_sil_frames_per_spk;
    let per_spk = mc.spkcache_len / n_spk - sil_per_spk;
    let strong_boost = (per_spk as f64 * mc.strong_boost_rate as f64).floor() as usize;
    let weak_boost = (per_spk as f64 * mc.weak_boost_rate as f64).floor() as usize;
    let min_pos = (per_spk as f64 * mc.min_pos_scores_rate as f64).floor() as usize;

    // 1. Score.
    let mut scores = log_pred_scores(preds, frames, n_spk, mc.pred_score_threshold);
    // 2. Disable non-speech frames, then overlapped speech. The
    // positive-score count runs over all frames of the already-masked
    // scores, matching the reference's ordering.
    for t in 0..frames {
        for k in 0..n_spk {
            if preds[t * n_spk + k] <= 0.5 {
                scores[t * n_spk + k] = f32::NEG_INFINITY;
            }
        }
    }
    let pos_count: Vec<usize> = (0..n_spk)
        .map(|k| (0..frames).filter(|&t| scores[t * n_spk + k] > 0.0).count())
        .collect();
    for t in 0..frames {
        for k in 0..n_spk {
            if scores[t * n_spk + k] <= 0.0 && pos_count[k] >= min_pos {
                scores[t * n_spk + k] = f32::NEG_INFINITY;
            }
        }
    }
    // 3. Boost newly added frames.
    if mc.scores_boost_latest > 0.0 && frames > mc.spkcache_len {
        for t in mc.spkcache_len..frames {
            for k in 0..n_spk {
                scores[t * n_spk + k] += mc.scores_boost_latest;
            }
        }
    }
    // 4. Strong boost, then 5. weak boost (per speaker, top-k).
    for (n_boost, scale) in [(strong_boost, 2.0f32), (weak_boost, 1.0f32)] {
        if n_boost == 0 {
            continue;
        }
        let boost_val = -scale * 0.5f32.ln();
        let k = n_boost.min(frames);
        for spk in 0..n_spk {
            let flat: Vec<f32> = (0..frames).map(|t| scores[t * n_spk + spk]).collect();
            for &t in top_k_indices(&flat, k).iter() {
                if flat[t] > f32::NEG_INFINITY {
                    scores[t * n_spk + spk] += boost_val;
                }
            }
        }
    }
    // 6. Append silence padding with +inf scores.
    let padded_frames = frames + sil_per_spk;
    let mut padded_scores = vec![0.0f32; padded_frames * n_spk];
    padded_scores[..frames * n_spk].copy_from_slice(&scores);
    for t in frames..padded_frames {
        for s in padded_scores[t * n_spk..(t + 1) * n_spk].iter_mut() {
            *s = f32::INFINITY;
        }
    }
    // 7. Select the top frames globally across speakers. The flatten is
    // speaker-major: flat index = spk * padded_frames + frame.
    let flat: Vec<f32> = {
        let mut f = vec![0.0f32; n_spk * padded_frames];
        for t in 0..padded_frames {
            for k in 0..n_spk {
                f[k * padded_frames + t] = padded_scores[t * n_spk + k];
            }
        }
        f
    };
    let k = mc.spkcache_len.min(flat.len());
    let mut chosen: Vec<usize> = top_k_indices(&flat, k)
        .into_iter()
        .map(|i| {
            if flat[i] > f32::NEG_INFINITY {
                i
            } else {
                mc.max_index
            }
        })
        .collect();
    chosen.sort_unstable();
    let n_frames_no_sil = padded_frames - sil_per_spk;
    let indices: Vec<usize> = chosen
        .iter()
        .map(|&i| {
            let disabled = i == mc.max_index || i % padded_frames >= n_frames_no_sil;
            if disabled {
                0
            } else {
                i % padded_frames
            }
        })
        .collect();
    // 8. Gather, replacing disabled slots with the silence embedding.
    let mut out_embs = Vec::with_capacity(mc.spkcache_len * emb_dim);
    let mut out_preds = Vec::with_capacity(mc.spkcache_len * n_spk);
    for (slot, &i) in indices.iter().enumerate() {
        let disabled =
            chosen[slot] == mc.max_index || chosen[slot] % padded_frames >= n_frames_no_sil;
        let _ = i;
        if disabled {
            out_embs.extend_from_slice(mean_sil_emb);
            out_preds.extend(std::iter::repeat_n(0.0, n_spk));
        } else {
            let frame = chosen[slot] % padded_frames;
            out_embs.extend_from_slice(&embs[frame * emb_dim..(frame + 1) * emb_dim]);
            out_preds.extend_from_slice(&preds[frame * n_spk..(frame + 1) * n_spk]);
        }
    }
    (out_embs, out_preds)
}

/// The relative-shift of the positional score matrix: left-pad each row,
/// reinterpret `[h, t, L+1]` as `[h, L+1, t]`, drop the first row of the
/// middle axis, reinterpret back, and keep the first `t` columns.
#[cfg(test)]
fn rel_shift(bd: &[f32], h: usize, t: usize, pos_len: usize) -> Vec<f32> {
    let lp = pos_len + 1;
    let mut padded = vec![0.0f32; h * t * lp];
    for hh in 0..h {
        for tt in 0..t {
            padded[hh * t * lp + tt * lp + 1..hh * t * lp + (tt + 1) * lp].copy_from_slice(
                &bd[hh * t * pos_len + tt * pos_len..hh * t * pos_len + (tt + 1) * pos_len],
            );
        }
    }
    let mut out = vec![0.0f32; h * t * t];
    for hh in 0..h {
        let p_base = hh * t * lp;
        let o_base = hh * t * t;
        for a in 0..t {
            for b in 0..t {
                let s = a * pos_len + b;
                let ridx = (s / t + 1) * t + s % t;
                out[o_base + a * t + b] = padded[p_base + ridx];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect(path);
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        let shape: Vec<usize> = {
            let start = header.find('(').unwrap() + 1;
            let end = header[start..].find(')').unwrap() + start;
            header[start..end]
                .split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect()
        };
        let data_start = 10 + header_len;
        let data = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (shape, data)
    }

    fn testdata(name: &str) -> String {
        format!("{}/testdata/sortformer/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// rel_shift known-answer for t=2, pos_len=3: shifting a 2x3 matrix
    /// left-pads and re-slices to the lower-triangular-aligned view.
    #[test]
    fn rel_shift_known_answer() {
        let bd = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Padded rows: [0 1 2 3 | 0 4 5 6]; reinterpreted (4,2): rows
        // [0,1],[2,3],[0,4],[5,6]; drop first: [[2,3],[0,4],[5,6]];
        // reinterpreted (2,3): [2 3 0 | 4 5 6]; keep first 2 columns.
        let out = rel_shift(&bd, 1, 2, 3);
        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn direct_relative_scores_match_dense_shift() {
        for t in [2, 3, 6] {
            let pos_len = 2 * t - 1;
            let dk = 3;
            let q: Vec<f32> = (0..t * dk).map(|i| ((i * 7) % 11) as f32 * 0.1).collect();
            let p: Vec<f32> = (0..pos_len * dk)
                .map(|i| ((i * 5 + 2) % 13) as f32 * 0.07)
                .collect();
            let mut dense = vec![0.0f32; t * pos_len];
            for query in 0..t {
                for position in 0..pos_len {
                    dense[query * pos_len + position] = q[query * dk..(query + 1) * dk]
                        .iter()
                        .zip(&p[position * dk..(position + 1) * dk])
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            let shifted = rel_shift(&dense, 1, t, pos_len);
            for query in 0..t {
                for key in 0..t {
                    let relative = t - 1 + key - query;
                    let direct: f32 = q[query * dk..(query + 1) * dk]
                        .iter()
                        .zip(&p[relative * dk..(relative + 1) * dk])
                        .map(|(a, b)| a * b)
                        .sum();
                    assert_eq!(direct, shifted[query * t + key]);
                }
            }
        }
    }

    /// The mel frontend matches the reference's NeMo conventions on a
    /// synthetic clip: golden tensors from the Python reference.
    #[test]
    fn mel_frontend_matches_reference() {
        let (shape, samples) = read_npy(&testdata("synthetic_samples.npy"));
        assert_eq!(shape.len(), 1);
        let (mel_shape, golden) = read_npy(&testdata("synthetic_mel.npy"));
        assert_eq!(mel_shape, vec![1, 80, 64]);

        let proc = ProcessorConfig::default();
        let fb = mel_filterbank(
            proc.feature_size,
            proc.n_fft,
            proc.sampling_rate,
            0.0,
            None,
            MelScale::Slaney,
        )
        .unwrap();
        let (feats, frames) = extract_mel_features(&samples, &proc, true, 16, &fb).unwrap();
        assert_eq!(frames, 64);
        for (a, b) in feats.iter().zip(&golden) {
            assert!(close(*a, *b, 1e-3), "mel mismatch {a} vs {b}");
        }
    }

    /// Segment conversion: threshold runs, minimum duration, and gap
    /// merging on hand-computable predictions.
    #[test]
    fn preds_to_segments_known_answer() {
        // 10 frames, 2 speakers: spk 0 active frames 1..6, spk 1 frame 8.
        let mut preds = vec![0.0f32; 20];
        for t in 1..6 {
            preds[t * 2] = 0.9;
        }
        preds[8 * 2 + 1] = 0.7;
        let segs = preds_to_segments(&preds, 2, 0.08, 0.5, 0.0, 0.0);
        assert_eq!(
            segs,
            vec![
                DiarizationSegment {
                    start: 0.08,
                    end: 0.48,
                    speaker: 0
                },
                DiarizationSegment {
                    start: 0.64,
                    end: 0.72,
                    speaker: 1
                },
            ]
        );
        // min_duration 0.1 drops spk 1's 0.08 s segment.
        let segs = preds_to_segments(&preds, 2, 0.08, 0.5, 0.1, 0.0);
        assert_eq!(segs.len(), 1);
        // merge_gap 0.2 would merge across the gap only within a speaker;
        // the two speakers stay separate.
        let segs = preds_to_segments(&preds, 2, 0.08, 0.5, 0.0, 0.2);
        assert_eq!(segs.len(), 2);
        // Same-speaker merging: spk 0 active at 1 and 3 (gap of one frame).
        let mut preds2 = vec![0.0f32; 20];
        preds2[2] = 0.9;
        preds2[3 * 2] = 0.9;
        let segs = preds_to_segments(&preds2, 2, 0.08, 0.5, 0.0, 0.2);
        assert_eq!(
            segs,
            vec![DiarizationSegment {
                start: 0.08,
                end: 0.32,
                speaker: 0
            }]
        );
    }

    /// Silence trimming: 0.25 s silence + 1 s tone + 0.5 s silence
    /// trims to exactly the tone at a 7680-sample offset.
    #[test]
    fn trim_silence_known_answer() {
        let mut samples = vec![0.0f32; 7680 + 15840 + 9600];
        for (i, s) in samples.iter_mut().enumerate().skip(7680).take(15840) {
            *s = 0.5 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / 16_000.0).sin() as f32;
        }
        let (trimmed, offset) = trim_silence(&samples, 16_000);
        assert_eq!(offset, 7680);
        assert_eq!(trimmed.len(), 15840);
        assert!(trimmed.iter().any(|v| v.abs() > 0.4));
    }

    /// AOSC compression against the Python reference on a deterministic
    /// fixture (200 frames, 4 speakers, 8-dim embeddings).
    #[test]
    fn aosc_compression_matches_reference_fixture() {
        let (_, in_embs) = read_npy(&testdata("aosc_fixture_in_embs.npy"));
        let (_, in_preds) = read_npy(&testdata("aosc_fixture_in_preds.npy"));
        let (_, in_sil) = read_npy(&testdata("aosc_fixture_in_mean_sil.npy"));
        let (_, out_embs) = read_npy(&testdata("aosc_fixture_out_embs.npy"));
        let (_, out_preds) = read_npy(&testdata("aosc_fixture_out_preds.npy"));

        let mc = ModulesConfig {
            num_speakers: 4,
            fc_d_model: 512,
            tf_d_model: 192,
            subsampling_factor: 8,
            chunk_len: 188,
            fifo_len: 0,
            spkcache_len: 188,
            spkcache_update_period: 188,
            chunk_left_context: 1,
            chunk_right_context: 1,
            spkcache_sil_frames_per_spk: 5,
            pred_score_threshold: 1e-6,
            max_index: 10_000,
            scores_boost_latest: 0.5,
            sil_threshold: 0.1,
            strong_boost_rate: 0.3,
            weak_boost_rate: 0.7,
            min_pos_scores_rate: 0.5,
            use_aosc: true,
        };
        let (embs, preds) = compress_spkcache_aosc(&in_embs, &in_preds, 200, &in_sil, &mc);
        assert_eq!(embs.len(), out_embs.len());
        assert_eq!(preds.len(), out_preds.len());
        for (a, b) in embs.iter().zip(&out_embs) {
            assert!(close(*a, *b, 1e-4), "aosc emb mismatch {a} vs {b}");
        }
        for (a, b) in preds.iter().zip(&out_preds) {
            assert!(close(*a, *b, 1e-4), "aosc pred mismatch {a} vs {b}");
        }
    }

    /// Full offline diarization on the synthetic clip against the Python
    /// reference's predictions; skipped when the checkpoint is not
    /// installed (set TURBOSPEECH_SORTFORMER_MODEL to override).
    #[test]
    fn offline_forward_matches_reference() {
        let dir = std::env::var("TURBOSPEECH_SORTFORMER_MODEL").unwrap_or_else(|_| {
            format!(
                "{}/diar_sortformer_4spk-v1-fp32",
                std::env::var("HOME").unwrap()
            )
        });
        if !std::path::Path::new(&dir).join("config.json").exists() {
            eprintln!("skipping: checkpoint not present at {dir}");
            return;
        }
        let model = Sortformer::load(std::path::Path::new(&dir)).unwrap();
        let (_, samples) = read_npy(&testdata("synthetic_samples.npy"));
        let out = model
            .generate(&samples, 16_000, &GenerateOptions::default())
            .unwrap();

        let (pred_shape, golden) = read_npy(&testdata("synthetic_preds.npy"));
        assert_eq!(pred_shape, vec![1, 8, 4]);
        assert_eq!(out.speaker_probs.len(), 8 * 4);
        for (a, b) in out.speaker_probs.iter().zip(&golden) {
            assert!(close(*a, *b, 1e-5), "preds mismatch {a} vs {b}");
        }
        assert_eq!(out.segments.len(), 1);
        let seg = &out.segments[0];
        assert_eq!(seg.speaker, 1);
        assert!(close(seg.start as f32, 0.56, 1e-6));
        assert!(close(seg.end as f32, 0.64, 1e-6));
    }
}
