//! KittenTTS: StyleTTS2-family TTS with an ALBERT text encoder, an
//! iSTFTNet decoder, and a quantized-ONNX heritage.
//!
//! Port of `mlx_audio/tts/models/kitten_tts/` (kitten_tts.py, modules.py,
//! istftnet.py, preprocess.py, quant.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kitten_tts).
//! Data flow for one segment (batch size 1, non-streaming, as in the
//! reference): phoneme ids -> KittenAlbert (the reference's "bert") ->
//! duration predictor (BLSTMs + AdaLayerNorm) -> duration alignment ->
//! F0/noise prosody predictor -> text encoder -> iSTFTNet decoder with a
//! harmonic sine source -> 24 kHz audio. See [README.md](README.md) for
//! the verification evidence and the deviations (espeak G2P intake,
//! dropped dropout/masking no-ops).
//!
//! Determinism note: the reference's SineGen injects a random initial
//! phase (harmonics only) and additive noise at inference, so waveforms
//! are stochastic by construction. Requests reset the pinned MLX key
//! sequence, matching `mx.random.seed`.
//!
//! Weight layouts: checkpoints carry the raw MLX parameter tree
//! (`bert.*`, `bert_encoder.*`, `predictor.*`, `text_encoder.*`,
//! `decoder.*`; LSTMs under `Wx_forward`/...). Every conv `weight_v` is
//! MLX order `[out, kernel, in]`; the loader transposes to the
//! `[out, in, kernel]` (conv) or `[in, out, kernel]` (transpose conv)
//! layout `ops` wants. Weight normalization folds at load time (g and v
//! are constants).

mod frontend;
#[cfg(test)]
mod tests;

pub use frontend::{
    basic_english_tokenize, chunk_text, ensure_punctuation, symbol_table, TextCleaner,
    TextPreprocessor,
};

use std::collections::HashMap;
use std::path::Path;

use turbospark_audio::fft::ComplexF32;
use turbospark_audio::stft::{istft, stft, StftOptions};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{wn_fold_in_place, WnFold};
use crate::music::minimax_music3::rng::{self, KeySequence};
use crate::{ops, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Reference sample rate (`ModelConfig.sample_rate` default).
pub const SAMPLE_RATE: u32 = 24_000;

/// The reference splits the 256-dim style vector at a hard-coded 128.
const STYLE_SPLIT: usize = 128;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The `plbert` sub-config (`AlbertModelArgs`), defaults per the dataclass.
#[derive(Debug, Clone)]
pub struct AlbertConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub embedding_size: usize,
    pub inner_group_num: usize,
    pub num_hidden_groups: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        Self {
            num_hidden_layers: 12,
            num_attention_heads: 12,
            hidden_size: 768,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            embedding_size: 128,
            inner_group_num: 1,
            num_hidden_groups: 1,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }
}

impl AlbertConfig {
    fn from_json(value: &serde_json::Value) -> Self {
        let num = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(default as u64) as usize
        };
        Self {
            num_hidden_layers: num("num_hidden_layers", 12),
            num_attention_heads: num("num_attention_heads", 12),
            hidden_size: num("hidden_size", 768),
            intermediate_size: num("intermediate_size", 3072),
            max_position_embeddings: num("max_position_embeddings", 512),
            embedding_size: num("embedding_size", 128),
            inner_group_num: num("inner_group_num", 1),
            num_hidden_groups: num("num_hidden_groups", 1),
            type_vocab_size: num("type_vocab_size", 2),
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-12) as f32,
        }
    }
}

/// The `istftnet` sub-config (the reference `Generator` args).
#[derive(Debug, Clone)]
pub struct IstftnetConfig {
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gen_istft_n_fft: usize,
    pub gen_istft_hop_size: usize,
}

impl IstftnetConfig {
    fn from_json(value: &serde_json::Value) -> Result<Self> {
        let nums = |key: &str| -> Vec<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_default()
        };
        let dilation_rows = value
            .get("resblock_dilation_sizes")
            .and_then(serde_json::Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()
                            .map(|r| {
                                r.iter()
                                    .filter_map(|v| v.as_u64())
                                    .map(|v| v as usize)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let get_usize = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: key.to_string(),
                    why: "istftnet config field missing".to_string(),
                })
        };
        Ok(Self {
            resblock_kernel_sizes: nums("resblock_kernel_sizes"),
            upsample_rates: nums("upsample_rates"),
            upsample_initial_channel: get_usize("upsample_initial_channel")?,
            resblock_dilation_sizes: dilation_rows,
            upsample_kernel_sizes: nums("upsample_kernel_sizes"),
            gen_istft_n_fft: get_usize("gen_istft_n_fft")?,
            gen_istft_hop_size: get_usize("gen_istft_hop_size")?,
        })
    }
}

/// The TTS `ModelConfig`, defaults per the reference dataclass.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub max_conv_dim: usize,
    pub max_dur: usize,
    pub n_layer: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub asr_res_dim: usize,
    /// `decoder_out_dim or max_conv_dim` after `from_json`.
    pub decoder_out_dim: usize,
    pub sample_rate: u32,
    pub voices_path: String,
    pub speed_priors: HashMap<String, f32>,
    pub voice_aliases: HashMap<String, String>,
    pub plbert: AlbertConfig,
    pub istftnet: IstftnetConfig,
    /// Module paths whose activations run through the reference's dynamic
    /// per-tensor uint8 fake quantization (the ONNX heritage).
    pub activation_quant_modules: Vec<String>,
}

impl ModelConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let num = |key: &str, _default: usize| -> Result<usize> {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: key.to_string(),
                    why: "model config field missing".to_string(),
                })
        };
        let max_conv_dim = num("max_conv_dim", 512)?;
        Ok(Self {
            hidden_dim: num("hidden_dim", 512)?,
            max_conv_dim,
            max_dur: num("max_dur", 50)?,
            n_layer: num("n_layer", 3)?,
            n_token: num("n_token", 256)?,
            style_dim: num("style_dim", 128)?,
            text_encoder_kernel_size: num("text_encoder_kernel_size", 5)?,
            asr_res_dim: num("asr_res_dim", 64)?,
            decoder_out_dim: value
                .get("decoder_out_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(max_conv_dim),
            sample_rate: value
                .get("sample_rate")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(SAMPLE_RATE as u64) as u32,
            voices_path: value
                .get("voices_path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("voices.npz")
                .to_string(),
            speed_priors: value
                .get("speed_priors")
                .and_then(serde_json::Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v as f32)))
                        .collect()
                })
                .unwrap_or_default(),
            voice_aliases: value
                .get("voice_aliases")
                .and_then(serde_json::Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                        .collect()
                })
                .unwrap_or_default(),
            plbert: value
                .get("plbert")
                .map(AlbertConfig::from_json)
                .unwrap_or_default(),
            istftnet: match value.get("istftnet") {
                Some(nested) => IstftnetConfig::from_json(nested)?,
                None => {
                    return Err(SpeechError::BadConfig {
                        field: "istftnet".to_string(),
                        why: "istftnet config missing".to_string(),
                    })
                }
            },
            activation_quant_modules: value
                .get("activation_quant_modules")
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Fake quantization (quant.py)
// ---------------------------------------------------------------------------

/// `maybe_fake_quant` with the reference's enabled-set check: a module is
/// quantized when a configured name equals its path or sits under it.
#[derive(Clone, Default)]
pub struct ActivationQuant {
    prefixes: Vec<String>,
}

impl ActivationQuant {
    pub fn new(module_names: &[String]) -> Self {
        Self {
            prefixes: module_names.to_vec(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.prefixes.is_empty()
    }

    pub fn enabled(&self, module_path: &str) -> bool {
        self.prefixes
            .iter()
            .any(|q| q == module_path || q.starts_with(&format!("{module_path}.")))
    }
}

/// `fake_quant_dynamic_u8`: dynamic per-tensor uint8 fake quantization
/// (the ONNX DynamicQuantizeLinear semantics), including the
/// round-half-to-even behavior of `mx.round`.
pub fn fake_quant_dynamic_u8(x: &[f32]) -> Vec<f32> {
    let mut x_min = 0.0f32;
    let mut x_max = 0.0f32;
    for &v in x {
        if !v.is_finite() {
            continue;
        }
        x_min = x_min.min(v);
        x_max = x_max.max(v);
    }
    let scale = (x_max - x_min) / 255.0;
    let scale_safe = if scale == 0.0 { 1.0 } else { scale };
    let zero_point = round_ties_even(-x_min / scale_safe).clamp(0.0, 255.0);
    let dequantized: Vec<f32> = x
        .iter()
        .map(|&v| {
            let q = round_ties_even(v / scale_safe + zero_point).clamp(0.0, 255.0);
            (q - zero_point) * scale_safe
        })
        .collect();
    if scale == 0.0 {
        vec![0.0; x.len()]
    } else {
        dequantized
    }
}

/// `maybe_fake_quant`.
pub fn maybe_fake_quant(x: &[f32], enabled: bool) -> Vec<f32> {
    if enabled {
        fake_quant_dynamic_u8(x)
    } else {
        x.to_vec()
    }
}

fn round_ties_even(v: f32) -> f32 {
    // mx.round rounds half to even.
    v.round_ties_even()
}

// ---------------------------------------------------------------------------
// RNG (MLX key sequence)
// ---------------------------------------------------------------------------

/// The vocoder's stochastic terms, in MLX global-key order.
struct Rng(KeySequence);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(KeySequence::new(seed))
    }

    fn uniform(&mut self, count: usize) -> Vec<f32> {
        rng::uniform01(self.0.next(), count)
    }

    fn normal(&mut self, count: usize) -> Vec<f32> {
        rng::normal(self.0.next(), count)
    }
}

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------

fn tensor_shape(file: &SafetensorsFile, name: &str) -> Result<Vec<usize>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    })?;
    Ok(desc.shape.clone())
}

/// Transposes MLX conv order `[out, kernel, in]` to `ops` order
/// `[out, in, kernel]` and folds the weight norm: `w = g * v / ||v||` per
/// output channel with the reference's `+ 1e-7` epsilon.
fn load_wn_conv(file: &SafetensorsFile, base: &str) -> Result<WnConv1d> {
    let v = file.load_as_f32(&format!("{base}.weight_v"))?;
    let g = file.load_as_f32(&format!("{base}.weight_g"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let shape = tensor_shape(file, &format!("{base}.weight_v"))?;
    let (out, kernel, in_ch) = match shape.as_slice() {
        [o, k, i] => (*o, *k, *i),
        other => {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight_v"),
                why: format!("expected 3-D [out, kernel, in], got {other:?}"),
            })
        }
    };
    let mut weight = vec![0.0f32; out * in_ch * kernel];
    for o in 0..out {
        for k in 0..kernel {
            for i in 0..in_ch {
                weight[o * in_ch * kernel + i * kernel + k] = v[o * kernel * in_ch + k * in_ch + i];
            }
        }
    }
    wn_fold_in_place(
        &mut weight,
        &g[..out],
        in_ch * kernel,
        WnFold::ScaleAdd(1e-7),
    );
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

/// A plain (non-weight-norm) conv whose checkpoint weight is MLX order
/// `[out, kernel, in]` (the noise-source projections).
fn load_plain_conv(file: &SafetensorsFile, base: &str) -> Result<WnConv1d> {
    let weight_v = file.load_as_f32(&format!("{base}.weight"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let shape = tensor_shape(file, &format!("{base}.weight"))?;
    let (out, kernel, in_ch) = match shape.as_slice() {
        [o, k, i] => (*o, *k, *i),
        other => {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: format!("expected 3-D [out, kernel, in], got {other:?}"),
            })
        }
    };
    let mut weight = vec![0.0f32; out * in_ch * kernel];
    for o in 0..out {
        for k in 0..kernel {
            for i in 0..in_ch {
                weight[o * in_ch * kernel + i * kernel + k] =
                    weight_v[o * kernel * in_ch + k * in_ch + i];
            }
        }
    }
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

/// Loads a weight-normalized transpose conv. Two shapes occur, and both
/// transpose with `(0, 2, 1)`:
/// - the generator `ups.*` `[in_effective, kernel, out_effective]`
///   (the reference reads it transposed into `mx.conv_transpose1d`), so
///   `ops` gets `[in, out, kernel]` per input channel;
/// - the AdaIN upsample `pool.*` `[channels, kernel, 1]` depthwise
///   (`groups = channels`), one tap triple per channel.
fn load_wn_conv_transpose(file: &SafetensorsFile, base: &str) -> Result<WnConvTranspose1d> {
    let v = file.load_as_f32(&format!("{base}.weight_v"))?;
    let g = file.load_as_f32(&format!("{base}.weight_g"))?;
    let bias = file
        .load_as_f32(&format!("{base}.bias"))
        .unwrap_or_default();
    let shape = tensor_shape(file, &format!("{base}.weight_v"))?;
    let (d0, d1, d2) = match shape.as_slice() {
        [a, b, c] => (*a, *b, *c),
        other => {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight_v"),
                why: format!("expected 3-D, got {other:?}"),
            })
        }
    };
    if d2 == 1 {
        // Depthwise pool: [channels, kernel, 1] -> [channels, 1, kernel].
        let channels = d0;
        let kernel = d1;
        let mut weight = vec![0.0f32; channels * kernel];
        for c in 0..channels {
            for k in 0..kernel {
                weight[c * kernel + k] = v[c * kernel + k];
            }
        }
        wn_fold_in_place(&mut weight, &g[..channels], kernel, WnFold::ScaleAdd(1e-7));
        Ok(WnConvTranspose1d {
            weight,
            bias,
            in_ch: channels,
            out_ch: channels,
            kernel,
            stride: 2,
            padding: 0,
            groups: channels,
        })
    } else {
        // [in_effective, kernel, out_effective] -> [in, out, kernel].
        let (in_ch, out_ch, kernel) = (d0, d2, d1);
        let mut weight = vec![0.0f32; in_ch * out_ch * kernel];
        for a in 0..in_ch {
            for k in 0..kernel {
                for c in 0..out_ch {
                    weight[a * out_ch * kernel + c * kernel + k] =
                        v[a * kernel * out_ch + k * out_ch + c];
                }
            }
        }
        wn_fold_in_place(
            &mut weight,
            &g[..in_ch],
            out_ch * kernel,
            WnFold::ScaleAdd(1e-7),
        );
        Ok(WnConvTranspose1d {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride: 2,
            padding: 0,
            groups: 1,
        })
    }
}

struct WnConv1d {
    weight: Vec<f32>, // [out, in / groups, kernel]
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_ch: usize,
    in_ch: usize,
}

impl WnConv1d {
    /// Channel-major `[in, seq]` -> `[out, out_seq]`.
    fn forward(&self, x: &[f32], quant: &ActivationQuant, module_path: &str) -> Vec<f32> {
        let input = maybe_fake_quant(x, quant.enabled(module_path));
        ops::conv1d(
            &input,
            &self.weight,
            if self.bias.is_empty() {
                None
            } else {
                Some(&self.bias)
            },
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

struct WnConvTranspose1d {
    weight: Vec<f32>, // [in, out / groups, kernel]
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl WnConvTranspose1d {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            if self.bias.is_empty() {
                None
            } else {
                Some(&self.bias)
            },
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

fn leaky_relu(v: f32, slope: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        v * slope
    }
}

/// LayerNorm across channels at each timestep of a channel-major
/// `[channels, seq]` buffer (the reference normalizes the last axis of its
/// channels-last tensors).
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

/// Linear 1-D interpolation matching the reference `interpolate1d` with
/// `align_corners = None`: source position `(i + 0.5) * in/size - 0.5`,
/// clamped at zero. `input` is `[in_width, channels]`; the result is
/// `[size, channels]`.
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
    /// Checkpoints carry the reference `Wx_forward` spellings (the MLX
    /// tree); the PyTorch `weight_ih_l0` spellings are accepted too.
    fn load(file: &SafetensorsFile, base: &str) -> Result<Self> {
        let named = |torch: &str, mlx: &str| -> Result<Vec<f32>> {
            let name = format!("{base}.{mlx}");
            if file.contains_tensor(&name) {
                return Ok(file.load_as_f32(&name)?);
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
        let shape = tensor_shape(file, &format!("{base}.Wx_forward"))
            .or_else(|_| tensor_shape(file, &format!("{base}.weight_ih_l0")))?;
        let hidden = shape[0] / 4;
        let in_dim = shape[1];
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
        quant: &ActivationQuant,
        module_path: &str,
    ) -> Vec<f32> {
        let seq = x.len() / self.in_dim;
        let h = self.hidden;
        let input = maybe_fake_quant(x, quant.enabled(module_path));
        let proj = ops::linear(&input, w_x, Some(b), seq, self.in_dim, 4 * h);
        let mut hidden = vec![0.0f32; h];
        let mut cell = vec![0.0f32; h];
        let mut out = vec![0.0f32; seq * h];
        for step in 0..seq {
            let t = if backward { seq - 1 - step } else { step };
            let p_row = &proj[t * 4 * h..(t + 1) * 4 * h];
            let hidden_q = maybe_fake_quant(&hidden, quant.enabled(module_path));
            let mut new_h = vec![0.0f32; h];
            for d in 0..h {
                let dot = |row: usize| -> f32 {
                    hidden_q
                        .iter()
                        .zip(&w_h[row * h..(row + 1) * h])
                        .map(|(x, w)| x * w)
                        .sum()
                };
                let i = ops::sigmoid(p_row[d] + dot(d));
                let f = ops::sigmoid(p_row[h + d] + dot(h + d));
                let g = (p_row[2 * h + d] + dot(2 * h + d)).tanh();
                let o = ops::sigmoid(p_row[3 * h + d] + dot(3 * h + d));
                cell[d] = f * cell[d] + i * g;
                new_h[d] = o * cell[d].tanh();
            }
            hidden = new_h;
            out[t * h..(t + 1) * h].copy_from_slice(&hidden);
        }
        out
    }

    fn forward(&self, x: &[f32], quant: &ActivationQuant, module_path: &str) -> Vec<f32> {
        let seq = x.len() / self.in_dim;
        let f = self.dir_forward(
            x,
            &self.wx_f,
            &self.wh_f,
            &self.b_f,
            false,
            quant,
            module_path,
        );
        let b = self.dir_forward(
            x,
            &self.wx_b,
            &self.wh_b,
            &self.b_b,
            true,
            quant,
            module_path,
        );
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

    fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        s: &[f32],
        quant: &ActivationQuant,
        module_path: &str,
    ) {
        let c = self.channels;
        let s_q = maybe_fake_quant(s, quant.enabled(module_path));
        let gb = ops::linear(&s_q, &self.fc_w, Some(&self.fc_b), 1, s.len(), 2 * c);
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

/// AdaIN1d on channel-major `[channels, seq]`: per-channel instance norm
/// over time, then the style-modulated affine.
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

    fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        s: &[f32],
        quant: &ActivationQuant,
        module_path: &str,
    ) {
        let c = self.channels;
        let s_q = maybe_fake_quant(s, quant.enabled(module_path));
        let gb = ops::linear(&s_q, &self.fc_w, Some(&self.fc_b), 1, s.len(), 2 * c);
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
        upsample: bool,
    ) -> Result<Self> {
        let conv1x1 = if dim_in != dim_out {
            Some(load_wn_conv(file, &format!("{base}.conv1x1"))?)
        } else {
            None
        };
        let pool = if upsample {
            Some(load_wn_conv_transpose(file, &format!("{base}.pool"))?)
        } else {
            None
        };
        Ok(AdainResBlk1d {
            conv1: load_wn_conv(file, &format!("{base}.conv1"))?,
            conv2: load_wn_conv(file, &format!("{base}.conv2"))?,
            conv1x1,
            norm1: AdaIn1d::load(file, &format!("{base}.norm1"), dim_in)?,
            norm2: AdaIn1d::load(file, &format!("{base}.norm2"), dim_out)?,
            pool,
            dim_in,
        })
    }

    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        s: &[f32],
        quant: &ActivationQuant,
        path: &str,
    ) -> Vec<f32> {
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
            sc = c1x1.forward(&sc, quant, &format!("{path}.conv1x1"));
        }
        // residual
        let mut r = x.to_vec();
        self.norm1
            .forward(&mut r, seq, s, quant, &format!("{path}.norm1"));
        for v in r.iter_mut() {
            *v = leaky_relu(*v, 0.2);
        }
        if let Some(pool) = &self.pool {
            // Depthwise transpose conv k=3 s=2 unpadded gives 2T + 2
            // frames; the reference trims ONE FRAME off the left of every
            // channel (torch padding=1, output_padding=1).
            let up = pool.forward(&r);
            let up_seq = (seq - 1) * pool.stride + pool.kernel;
            let mut trimmed = vec![0.0f32; self.dim_in * (up_seq - 1)];
            for c in 0..self.dim_in {
                trimmed[c * (up_seq - 1)..(c + 1) * (up_seq - 1)]
                    .copy_from_slice(&up[c * up_seq + 1..(c + 1) * up_seq]);
            }
            r = trimmed;
        }
        r = self.conv1.forward(&r, quant, &format!("{path}.conv1"));
        self.norm2
            .forward(&mut r, out_seq, s, quant, &format!("{path}.norm2"));
        for v in r.iter_mut() {
            *v = leaky_relu(*v, 0.2);
        }
        r = self.conv2.forward(&r, quant, &format!("{path}.conv2"));
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        for (a, b) in r.iter_mut().zip(&sc) {
            *a = (*a + b) * inv_sqrt2;
        }
        r
    }
}

/// iSTFTNet generator AdaIN residual block: three snake-conv pairs with
/// residual accumulation; channel-major `[channels, seq]`.
struct AdaInResBlock1 {
    convs1: Vec<WnConv1d>,
    convs2: Vec<WnConv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Vec<f32>>,
    alpha2: Vec<Vec<f32>>,
    channels: usize,
}

impl AdaInResBlock1 {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        channels: usize,
        kernel: usize,
        dilations: &[usize; 3],
    ) -> Result<Self> {
        let get_padding = |k: usize, d: usize| (k * d - d) / 2;
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        for i in 0..3 {
            let mut conv = load_wn_conv(file, &format!("{base}.convs1.{i}"))?;
            conv.padding = get_padding(kernel, dilations[i]);
            conv.dilation = dilations[i];
            convs1.push(conv);
            let mut conv = load_wn_conv(file, &format!("{base}.convs2.{i}"))?;
            conv.padding = get_padding(kernel, 1);
            conv.dilation = 1;
            convs2.push(conv);
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
            // Alphas are per-channel [1, C, 1] tensors; the ONNX-converted
            // checkpoints may carry alpha1_0-style or alpha1.0-style names.
            let a1 = file
                .load_as_f32(&format!("{base}.alpha1_{i}"))
                .or_else(|_| file.load_as_f32(&format!("{base}.alpha1.{i}")))?;
            let a2 = file
                .load_as_f32(&format!("{base}.alpha2_{i}"))
                .or_else(|_| file.load_as_f32(&format!("{base}.alpha2.{i}")))?;
            alpha1.push(a1);
            alpha2.push(a2);
        }
        Ok(AdaInResBlock1 {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
            channels,
        })
    }

    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        s: &[f32],
        quant: &ActivationQuant,
        path: &str,
    ) -> Vec<f32> {
        let mut base = x.to_vec();
        let c = self.channels;
        for i in 0..3 {
            let mut xt = base.clone();
            self.adain1[i].forward(&mut xt, seq, s, quant, &format!("{path}.adain1.{i}"));
            ops::snake(&mut xt, &self.alpha1[i][..c], c, seq);
            xt = self.convs1[i].forward(&xt, quant, &format!("{path}.convs1.{i}"));
            self.adain2[i].forward(&mut xt, seq, s, quant, &format!("{path}.adain2.{i}"));
            ops::snake(&mut xt, &self.alpha2[i][..c], c, seq);
            xt = self.convs2[i].forward(&xt, quant, &format!("{path}.convs2.{i}"));
            for (a, b) in xt.iter_mut().zip(&base) {
                *a += b;
            }
            base = xt;
        }
        base
    }
}

// ---------------------------------------------------------------------------
// KittenAlbert
// ---------------------------------------------------------------------------

/// The ALBERT stack the reference uses as its "bert": embeddings, a layer
/// group walked `num_hidden_layers` times, and a (computed but unused)
/// pooler. The additive attention mask is all-zero on the batch-1 full
/// length path, so it is not materialized.
struct KittenAlbert {
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
    inner_group_num: usize,
    max_pos: usize,
    layer_norm_eps: f32,
    quant: ActivationQuant,
}

/// The reference FFN activation: the ONNX tanh GELU approximation with
/// its truncated constants (`0.7978846`, `0.044715`).
fn gelu_tanh_onnx(x: &mut [f32]) {
    const GELU_CONST: f32 = 0.7978846;
    const GELU_CONST_2: f32 = 0.044715;
    for v in x.iter_mut() {
        let x = *v;
        *v = 0.5 * x * (1.0 + (GELU_CONST * (x + GELU_CONST_2 * x * x * x)).tanh());
    }
}

impl KittenAlbert {
    fn load(
        file: &SafetensorsFile,
        config: &AlbertConfig,
        quant: &ActivationQuant,
    ) -> Result<Self> {
        let layer = "encoder.albert_layer_groups.0.albert_layers.0";
        let f = |name: &str| -> Result<Vec<f32>> {
            file.load_as_f32(&format!("bert.{name}"))
                .or_else(|_| file.load_as_f32(name))
                .map_err(|e| SpeechError::Tensor {
                    name: name.to_string(),
                    why: e.to_string(),
                })
        };
        Ok(KittenAlbert {
            embedding_size: config.embedding_size,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            heads: config.num_attention_heads,
            layers: config.num_hidden_layers,
            inner_group_num: config.inner_group_num,
            max_pos: config.max_position_embeddings,
            layer_norm_eps: config.layer_norm_eps,
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
            quant: quant.clone(),
        })
    }

    /// Token ids -> sequence output `[seq, hidden]` (the pooled output the
    /// reference also computes is unused by the TTS path).
    fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let seq = ids.len();
        if seq == 0
            || seq > self.max_pos
            || ids
                .iter()
                .any(|&id| id as usize * self.embedding_size >= self.word_emb.len())
        {
            return Err(SpeechError::Input {
                why: "token ids must fit the embedding and position tables".to_string(),
            });
        }
        let e = self.embedding_size;
        let layer_path = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let mut x = vec![0.0f32; seq * e];
        for (t, &id) in ids.iter().enumerate() {
            let row = &self.word_emb[id as usize * e..(id as usize + 1) * e];
            for d in 0..e {
                x[t * e + d] = row[d] + self.pos_emb[t * e + d] + self.type_emb[d];
            }
        }
        ops::layernorm(
            &mut x,
            seq,
            e,
            &self.emb_ln_w,
            Some(&self.emb_ln_b),
            self.layer_norm_eps,
        );
        let mapping_input = maybe_fake_quant(&x, self.quant.enabled("bert.encoder"));
        let mut x = ops::linear(
            &mapping_input,
            &self.mapping_w,
            Some(&self.mapping_b),
            seq,
            e,
            self.hidden,
        );
        let head_dim = self.hidden / self.heads;
        for _ in 0..self.layers {
            for _ in 0..self.inner_group_num {
                // Attention (the additive mask is zero on the full path).
                let hidden_q =
                    maybe_fake_quant(&x, self.quant.enabled(&format!("{layer_path}.attention")));
                let q = ops::linear(
                    &hidden_q,
                    &self.q_w,
                    Some(&self.q_b),
                    seq,
                    self.hidden,
                    self.hidden,
                );
                let k = ops::linear(
                    &hidden_q,
                    &self.k_w,
                    Some(&self.k_b),
                    seq,
                    self.hidden,
                    self.hidden,
                );
                let v = ops::linear(
                    &hidden_q,
                    &self.v_w,
                    Some(&self.v_b),
                    seq,
                    self.hidden,
                    self.hidden,
                );
                let mut attn = vec![0.0f32; seq * self.hidden];
                for h in 0..self.heads {
                    let mut q_head = vec![0.0f32; seq * head_dim];
                    let mut k_head = vec![0.0f32; seq * head_dim];
                    let mut v_head = vec![0.0f32; seq * head_dim];
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
                        &q_head,
                        &k_head,
                        &v_head,
                        None,
                        seq,
                        seq,
                        head_dim,
                        head_dim,
                        (head_dim as f32).sqrt().recip(),
                    );
                    for t in 0..seq {
                        let dst = t * self.hidden + h * head_dim;
                        attn[dst..dst + head_dim]
                            .copy_from_slice(&o[t * head_dim..(t + 1) * head_dim]);
                    }
                }
                let context_q = maybe_fake_quant(
                    &attn,
                    self.quant.enabled(&format!("{layer_path}.attention")),
                );
                let mut out = ops::linear(
                    &context_q,
                    &self.attn_dense_w,
                    Some(&self.attn_dense_b),
                    seq,
                    self.hidden,
                    self.hidden,
                );
                for (o, a) in out.iter_mut().zip(&x) {
                    *o += a;
                }
                ops::layernorm(
                    &mut out,
                    seq,
                    self.hidden,
                    &self.attn_ln_w,
                    Some(&self.attn_ln_b),
                    self.layer_norm_eps,
                );
                // FFN with the tanh GELU approximation.
                let ffn_q = maybe_fake_quant(&out, self.quant.enabled(layer_path));
                let mut ffn = ops::linear(
                    &ffn_q,
                    &self.ffn_w,
                    Some(&self.ffn_b),
                    seq,
                    self.hidden,
                    self.intermediate,
                );
                gelu_tanh_onnx(&mut ffn);
                let ffn_out_q = maybe_fake_quant(&ffn, self.quant.enabled(layer_path));
                let mut result = ops::linear(
                    &ffn_out_q,
                    &self.ffn_out_w,
                    Some(&self.ffn_out_b),
                    seq,
                    self.intermediate,
                    self.hidden,
                );
                for (o, a) in result.iter_mut().zip(&out) {
                    *o += a;
                }
                ops::layernorm(
                    &mut result,
                    seq,
                    self.hidden,
                    &self.ffn_ln_w,
                    Some(&self.ffn_ln_b),
                    self.layer_norm_eps,
                );
                x = result;
            }
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
    /// `[seq, d_model + style_dim]`.
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        s: &[f32],
        quant: &ActivationQuant,
        path: &str,
    ) -> Vec<f32> {
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
            let mut out = self.lstms[i].forward(&inp, quant, &format!("{path}.lstms.{}", 2 * i));
            self.adalns[i].forward(
                &mut out,
                seq,
                s,
                quant,
                &format!("{path}.lstms.{}", 2 * i + 1),
            );
            feats = out;
        }
        // The reference's final concat re-appends the style features, so
        // the returned width is d_model + style_dim.
        let mut out = vec![0.0f32; seq * (d + sty)];
        for t in 0..seq {
            out[t * (d + sty)..t * (d + sty) + d].copy_from_slice(&feats[t * d..(t + 1) * d]);
            out[t * (d + sty) + d..(t + 1) * (d + sty)].copy_from_slice(s);
        }
        out
    }
}

/// Text encoder: embedding, depth-n weighted-conv / LayerNorm /
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
        let mut convs = Vec::new();
        let mut lns = Vec::new();
        for i in 0..depth {
            let mut conv = load_wn_conv(file, &format!("{base}.cnn.{i}.0"))?;
            conv.padding = (kernel - 1) / 2;
            convs.push(conv);
            lns.push((
                file.load_as_f32(&format!("{base}.cnn.{i}.1.weight"))?,
                file.load_as_f32(&format!("{base}.cnn.{i}.1.bias"))?,
            ));
        }
        Ok(TextEncoder {
            embedding: file.load_as_f32(&format!("{base}.embedding.weight"))?,
            channels,
            convs,
            lns,
            lstm: Lstm::load(file, &format!("{base}.lstm"))?,
        })
    }

    fn forward(&self, ids: &[u32], quant: &ActivationQuant) -> Result<Vec<f32>> {
        let seq = ids.len();
        let c = self.channels;
        if seq == 0
            || ids
                .iter()
                .any(|&id| id as usize * c >= self.embedding.len())
        {
            return Err(SpeechError::Input {
                why: "token ids do not fit the text encoder embedding".to_string(),
            });
        }
        let mut xc = vec![0.0f32; seq * c];
        for (t, &id) in ids.iter().enumerate() {
            for (ch, &v) in self.embedding[id as usize * c..(id as usize + 1) * c]
                .iter()
                .enumerate()
            {
                xc[ch * seq + t] = v;
            }
        }
        for (i, conv) in self.convs.iter().enumerate() {
            let mut h = conv.forward(&xc, quant, &format!("text_encoder.cnn.{i}.0"));
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
        let out = self.lstm.forward(&feats, quant, "text_encoder.lstm");
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
                AdainResBlk1d::load(file, &format!("{base}.F0.0"), d_hid, d_hid, false)?,
                AdainResBlk1d::load(file, &format!("{base}.F0.1"), d_hid, d_hid / 2, true)?,
                AdainResBlk1d::load(file, &format!("{base}.F0.2"), d_hid / 2, d_hid / 2, false)?,
            ],
            n_blocks: vec![
                AdainResBlk1d::load(file, &format!("{base}.N.0"), d_hid, d_hid, false)?,
                AdainResBlk1d::load(file, &format!("{base}.N.1"), d_hid, d_hid / 2, true)?,
                AdainResBlk1d::load(file, &format!("{base}.N.2"), d_hid / 2, d_hid / 2, false)?,
            ],
            f0_proj: load_wn_conv(file, &format!("{base}.F0_proj"))?,
            n_proj: load_wn_conv(file, &format!("{base}.N_proj"))?,
            d_hid,
            style_dim,
        })
    }

    /// Duration path: `d_en` channel-major `[d_hid, seq]` + style ->
    /// per-phoneme durations (sigmoid sum / speed, rounded, min 1).
    fn durations(
        &self,
        d_en: &[f32],
        seq: usize,
        s: &[f32],
        speed: f32,
        quant: &ActivationQuant,
    ) -> Vec<i32> {
        let d = self
            .text_encoder
            .forward(d_en, seq, s, quant, "predictor.text_encoder");
        let lstm_out = self.lstm.forward(&d, quant, "predictor.lstm");
        let classes = self.duration_proj_w.len() / self.d_hid;
        // LinearNorm quantizes its input when enabled.
        let lstm_q = maybe_fake_quant(&lstm_out, quant.enabled("predictor.duration_proj"));
        let logits = ops::linear(
            &lstm_q,
            &self.duration_proj_w,
            Some(&self.duration_proj_b),
            seq,
            self.d_hid,
            classes,
        );
        let mut out = Vec::with_capacity(seq);
        for t in 0..seq {
            let row = &logits[t * classes..(t + 1) * classes];
            let sum: f32 = row.iter().map(|v| ops::sigmoid(*v)).sum::<f32>() / speed;
            out.push(round_ties_even(sum).max(1.0) as i32);
        }
        out
    }

    /// F0/noise prediction from aligned features `en` channel-major
    /// `[d_hid + style, frames]` and style. Returns `(F0, N)`, each
    /// `[2 * frames]` (one upsample stage doubles the length).
    fn f0_n_train(
        &self,
        en: &[f32],
        frames: usize,
        s: &[f32],
        quant: &ActivationQuant,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.d_hid;
        let sty = self.style_dim;
        let mut feats = vec![0.0f32; frames * (d + sty)];
        for t in 0..frames {
            for c in 0..d {
                feats[t * (d + sty) + c] = en[c * frames + t];
            }
            feats[t * (d + sty) + d..(t + 1) * (d + sty)].copy_from_slice(s);
        }
        let shared_out = self.shared.forward(&feats, quant, "predictor.shared");
        let branch =
            |blocks: &[AdainResBlk1d], proj: &WnConv1d, path: &str, proj_path: &str| -> Vec<f32> {
                let mut cm = vec![0.0f32; frames * d];
                for t in 0..frames {
                    for (c, v) in shared_out[t * d..(t + 1) * d].iter().enumerate() {
                        cm[c * frames + t] = *v;
                    }
                }
                let mut cur = cm;
                for (i, b) in blocks.iter().enumerate() {
                    let seq = cur.len() / b.dim_in;
                    cur = b.forward(&cur, seq, s, quant, &format!("{path}.{i}"));
                }
                // The 1x1 projection runs channels-last: a per-frame linear
                // over the channel axis.
                let ch = cur.len() / frames;
                let input = maybe_fake_quant(&cur, quant.enabled(proj_path));
                ops::linear(&input, &proj.weight, Some(&proj.bias), frames, ch, 1)
            };
        let f0 = branch(
            &self.f0_blocks,
            &self.f0_proj,
            "predictor.F0",
            "predictor.F0_proj",
        );
        let n = branch(
            &self.n_blocks,
            &self.n_proj,
            "predictor.N",
            "predictor.N_proj",
        );
        (f0, n)
    }
}

// ---------------------------------------------------------------------------
// Harmonic source + generator + decoder
// ---------------------------------------------------------------------------

/// SineGen + SourceModuleHnNSF: harmonic excitation from upsampled F0.
struct HarmonicSource {
    l_linear_w: Vec<f32>, // [1, dim]
    l_linear_b: f32,
    dim: usize,
    upsample_scale: usize,
}

const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;

impl HarmonicSource {
    fn load(file: &SafetensorsFile, base: &str, upsample_scale: usize) -> Result<Self> {
        let w = file.load_as_f32(&format!("{base}.l_linear.weight"))?;
        Ok(HarmonicSource {
            dim: w.len(),
            l_linear_w: w,
            l_linear_b: file.load_as_f32(&format!("{base}.l_linear.bias"))?[0],
            upsample_scale,
        })
    }

    /// `f0_up` nearest-upsampled F0 `[n]` -> sine merge `[n]`. Draws the
    /// MLX stream in reference order: the uniform initial phase, the
    /// SineGen noise, and the source-module noise (unused branch, but it
    /// advances the stream).
    fn forward(&self, f0_up: &[f32], rng: &mut Rng, quant: &ActivationQuant) -> Vec<f32> {
        let n = f0_up.len();
        let dim = self.dim;
        // rad = (harmonic f0 / sr) % 1; frame 0 gets a random initial
        // phase on the harmonics (the fundamental stays clean).
        let mut rad = vec![0.0f32; n * dim];
        for t in 0..n {
            for h in 0..dim {
                // Python-style floor modulus: MLX's % wraps negatives into
                // [0, 1), which the raw (sometimes negative) F0
                // predictions rely on in unvoiced regions.
                rad[t * dim + h] = (f0_up[t] * (h + 1) as f32 / SAMPLE_RATE as f32).rem_euclid(1.0);
            }
        }
        let initial = rng.uniform(dim);
        for h in 1..dim {
            rad[h] += initial[h];
        }
        // Down to frame rate (linear), cumsum, phase ramp, back up. The
        // reference's interpolate derives the size with float math, so
        // reproduce the float ceil rather than integer division.
        let small = (((n as f64) * (1.0f64 / self.upsample_scale as f64)).ceil() as usize).max(1);
        let rad_small = interpolate_linear(&rad, dim, small);
        let mut phase = vec![0.0f32; small * dim];
        for t in 0..small {
            for h in 0..dim {
                let mut sum = 0.0f32;
                for t2 in 0..=t {
                    sum += rad_small[t2 * dim + h];
                }
                phase[t * dim + h] = sum * 2.0 * std::f32::consts::PI;
            }
        }
        let up_n = ((small as f64) * (self.upsample_scale as f64)).ceil() as usize;
        for v in phase.iter_mut() {
            *v *= self.upsample_scale as f32;
        }
        let phase_up_full = interpolate_linear(&phase, dim, up_n);
        let phase_up = &phase_up_full[..(n * dim).min(phase_up_full.len())];
        let noise = rng.normal(n * dim);
        let mut merge_in = vec![0.0f32; n * dim];
        for t in 0..n {
            let uv = (f0_up[t] > VOICED_THRESHOLD) as u8 as f32;
            let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
            for h in 0..dim {
                let sine = (phase_up[t * dim + h]).sin() * SINE_AMP;
                merge_in[t * dim + h] = sine * uv + noise_amp * noise[t * dim + h];
            }
        }
        // The source's unused noise branch still advances the global
        // stream (it has the same shape as the used one here).
        let _unused = rng.normal(n * dim);
        let merge_in = maybe_fake_quant(
            &merge_in,
            quant.enabled("decoder.generator.m_source.l_linear"),
        );
        let merge = ops::linear(
            &merge_in,
            &self.l_linear_w,
            Some(&[self.l_linear_b]),
            n,
            dim,
            1,
        );
        merge.into_iter().map(f32::tanh).collect()
    }
}

fn constant_pad_left(x: &[f32], ch: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / ch;
    let mut out = vec![0.0f32; ch * (seq + pad)];
    for c in 0..ch {
        out[c * (seq + pad) + pad..(c + 1) * (seq + pad)]
            .copy_from_slice(&x[c * seq..(c + 1) * seq]);
    }
    out
}

/// iSTFTNet generator. `TOTAL_UPSAMPLE = prod(upsample_rates) * hop`.
struct Generator {
    m_source: HarmonicSource,
    ups: Vec<WnConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<WnConv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: WnConv1d,
    bins: usize,
    hop: usize,
    total_upsample: usize,
}

impl Generator {
    fn load(file: &SafetensorsFile, base: &str, istftnet: &IstftnetConfig) -> Result<Self> {
        let resblock_kernel_sizes = &istftnet.resblock_kernel_sizes;
        let resblock_dilation_sizes = &istftnet.resblock_dilation_sizes;
        let upsample_rates = &istftnet.upsample_rates;
        let upsample_kernel_sizes = &istftnet.upsample_kernel_sizes;
        let total_upsample: usize =
            upsample_rates.iter().product::<usize>() * istftnet.gen_istft_hop_size;
        let mut ups = Vec::new();
        for (i, k) in upsample_kernel_sizes.iter().enumerate() {
            let mut up = load_wn_conv_transpose(file, &format!("{base}.ups.{i}"))?;
            // Effective geometry: the reference reads the weight
            // transposed, so the input channels are the first weight dim
            // and the stride comes from the config.
            up.stride = upsample_rates[i];
            up.padding = (k - upsample_rates[i]) / 2;
            ups.push(up);
        }
        let mut resblocks = Vec::new();
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..ups.len() {
            let ch = upsample_initial_channel_at(istftnet, i + 1);
            for (j, k) in resblock_kernel_sizes.iter().enumerate() {
                let dils = [
                    resblock_dilation_sizes[j][0],
                    resblock_dilation_sizes[j][1],
                    resblock_dilation_sizes[j][2],
                ];
                resblocks.push(AdaInResBlock1::load(
                    file,
                    &format!("{base}.resblocks.{}", i * resblock_kernel_sizes.len() + j),
                    ch,
                    *k,
                    &dils,
                )?);
            }
            // Noise-source branch convs are plain convs on the harmonic
            // spectrum; checkpoints carry MLX [out, kernel, in].
            let name = format!("{base}.noise_convs.{i}");
            let mut conv = load_plain_conv(file, &name)?;
            if i + 1 < upsample_rates.len() {
                let stride_f0: usize = upsample_rates[i + 1..].iter().product();
                conv.stride = stride_f0;
                conv.padding = (stride_f0 + 1) / 2;
            } else {
                conv.stride = 1;
                conv.padding = 0;
            }
            noise_convs.push(conv);
            let (rk, rd) = if i + 1 < upsample_rates.len() {
                (7usize, [1usize, 3, 5])
            } else {
                (11usize, [1usize, 3, 5])
            };
            noise_res.push(AdaInResBlock1::load(
                file,
                &format!("{base}.noise_res.{i}"),
                ch,
                rk,
                &rd,
            )?);
        }
        let mut conv_post = load_wn_conv(file, &format!("{base}.conv_post"))?;
        conv_post.padding = 3;
        Ok(Generator {
            m_source: HarmonicSource::load(file, &format!("{base}.m_source"), total_upsample)?,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            bins: istftnet.gen_istft_n_fft / 2 + 1,
            hop: istftnet.gen_istft_hop_size,
            total_upsample,
        })
    }

    /// `x` decoder features channel-major `[ch, seq]`, `f0` prosody curve
    /// `[frames]` -> waveform samples.
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        s: &[f32],
        f0: &[f32],
        rng: &mut Rng,
        quant: &ActivationQuant,
    ) -> Vec<f32> {
        // Nearest-upsample F0 by total_upsample.
        let mut f0_up = vec![0.0f32; f0.len() * self.total_upsample];
        for (t, &v) in f0.iter().enumerate() {
            for u in 0..self.total_upsample {
                f0_up[t * self.total_upsample + u] = v;
            }
        }
        let har_source = self.m_source.forward(&f0_up, rng, quant);
        // STFT of the source: periodic Hann, center reflect.
        let opts = StftOptions {
            fft_size: (self.bins - 1) * 2,
            hop: self.hop,
            window: turbospark_audio::hann_window((self.bins - 1) * 2),
            center: true,
        };
        let frames_t = 1 + har_source.len() / self.hop;
        let spectra = stft(&har_source, &opts).expect("harmonic source STFT");
        let mut har = vec![0.0f32; 2 * self.bins * frames_t];
        for (t, frame) in spectra.iter().enumerate() {
            for (b, z) in frame.iter().enumerate() {
                har[b * frames_t + t] = (z.re * z.re + z.im * z.im).sqrt();
                har[(self.bins + b) * frames_t + t] = z.im.atan2(z.re);
            }
        }
        let mut cur = x.to_vec();
        let mut cur_seq = seq;
        for i in 0..self.ups.len() {
            for v in cur.iter_mut() {
                *v = leaky_relu(*v, 0.1);
            }
            let x_source = self.noise_convs[i].forward(
                &har,
                quant,
                &format!("decoder.generator.noise_convs.{i}"),
            );
            // The noise branch runs at the STFT frame rate.
            let xs_seq = x_source.len() / self.noise_convs[i].out_ch;
            let x_source = self.noise_res[i].forward(
                &x_source,
                xs_seq,
                s,
                quant,
                &format!("decoder.generator.noise_res.{i}"),
            );
            cur = self.ups[i].forward(&cur);
            cur_seq =
                (cur_seq - 1) * self.ups[i].stride + self.ups[i].kernel - 2 * self.ups[i].padding;
            if i == self.ups.len() - 1 {
                cur = constant_pad_left(&cur, self.ups[i].out_ch, 1);
                cur_seq += 1;
            }
            for (a, b) in cur.iter_mut().zip(&x_source) {
                *a += b;
            }
            let mut acc = vec![0.0f32; cur.len()];
            for j in 0..self.resblock_kernels() {
                let out = self.resblocks[i * self.resblock_kernels() + j].forward(
                    &cur,
                    cur_seq,
                    s,
                    quant,
                    &format!(
                        "decoder.generator.resblocks.{}",
                        i * self.resblock_kernels() + j
                    ),
                );
                for (a, b) in acc.iter_mut().zip(&out) {
                    *a += b;
                }
            }
            for a in acc.iter_mut() {
                *a /= self.resblock_kernels() as f32;
            }
            cur = acc;
        }
        for v in cur.iter_mut() {
            *v = leaky_relu(*v, 0.01);
        }
        let post = self
            .conv_post
            .forward(&cur, quant, "decoder.generator.conv_post");
        // spec = exp(first half), phase = sin(second half).
        let bins = self.bins;
        let mut spectra_inv = Vec::with_capacity(cur_seq);
        for t in 0..cur_seq {
            let mut frame = Vec::with_capacity(bins);
            for b in 0..bins {
                let m = post[b * cur_seq + t].exp();
                let p = post[(bins + b) * cur_seq + t].sin();
                // The DC and Nyquist bins of a real signal are purely
                // real; the inverse FFT discards their imaginary parts.
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
        let original_len = (spectra_inv.len() - 1) * self.hop;
        istft(&spectra_inv, &opts, original_len).expect("generator iSTFT")
    }

    fn resblock_kernels(&self) -> usize {
        self.resblocks.len() / self.ups.len()
    }
}

fn upsample_initial_channel_at(istftnet: &IstftnetConfig, stage: usize) -> usize {
    istftnet.upsample_initial_channel / (1usize << stage)
}

/// The decoder: prosody conditioning convs, AdaIN encode/decode stack,
/// then the generator.
struct KittenDecoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: WnConv1d,
    n_conv: WnConv1d,
    asr_res: WnConv1d,
    generator: Generator,
}

impl KittenDecoder {
    fn load(file: &SafetensorsFile, base: &str, config: &ModelConfig) -> Result<Self> {
        let mut decode = Vec::new();
        for i in 0..3 {
            decode.push(AdainResBlk1d::load(
                file,
                &format!("{base}.decode.{i}"),
                config.max_conv_dim + 2 + config.asr_res_dim,
                config.max_conv_dim,
                false,
            )?);
        }
        decode.push(AdainResBlk1d::load(
            file,
            &format!("{base}.decode.3"),
            config.max_conv_dim + 2 + config.asr_res_dim,
            config.decoder_out_dim,
            true,
        )?);
        let mut f0_conv = load_wn_conv(file, &format!("{base}.F0_conv"))?;
        f0_conv.stride = 2;
        f0_conv.padding = 1;
        let mut n_conv = load_wn_conv(file, &format!("{base}.N_conv"))?;
        n_conv.stride = 2;
        n_conv.padding = 1;
        let mut asr_res = load_wn_conv(file, &format!("{base}.asr_res.0"))?;
        asr_res.padding = 0;
        Ok(KittenDecoder {
            encode: AdainResBlk1d::load(
                file,
                &format!("{base}.encode"),
                config.hidden_dim + 2,
                config.max_conv_dim,
                false,
            )?,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator: Generator::load(file, &format!("{base}.generator"), &config.istftnet)?,
        })
    }

    /// `asr` channel-major `[hidden, frames]`, `f0`/`n` curves `[frames]`.
    fn forward(
        &self,
        asr: &[f32],
        _frames: usize,
        f0_curve: &[f32],
        n_curve: &[f32],
        s: &[f32],
        rng: &mut Rng,
        quant: &ActivationQuant,
    ) -> Vec<f32> {
        let f0_c = self.f0_conv.forward(f0_curve, quant, "decoder.F0_conv");
        let n_c = self.n_conv.forward(n_curve, quant, "decoder.N_conv");
        let cond_seq = f0_c.len();
        // concat [asr, F0, N] channel-major
        let mut x = vec![0.0f32; asr.len() + 2 * cond_seq];
        x[..asr.len()].copy_from_slice(asr);
        x[asr.len()..asr.len() + cond_seq].copy_from_slice(&f0_c);
        x[asr.len() + cond_seq..].copy_from_slice(&n_c);
        x = self
            .encode
            .forward(&x, cond_seq, s, quant, "decoder.encode");
        let asr_res = self.asr_res.forward(asr, quant, "decoder.asr_res.0");
        let mut res = true;
        let mut cur_seq = cond_seq;
        for (index, block) in self.decode.iter().enumerate() {
            if res {
                let mut joined = vec![0.0f32; x.len() + asr_res.len() + 2 * cond_seq];
                joined[..x.len()].copy_from_slice(&x);
                joined[x.len()..x.len() + asr_res.len()].copy_from_slice(&asr_res);
                joined[x.len() + asr_res.len()..x.len() + asr_res.len() + cond_seq]
                    .copy_from_slice(&f0_c);
                joined[x.len() + asr_res.len() + cond_seq..].copy_from_slice(&n_c);
                x = joined;
            }
            x = block.forward(&x, cur_seq, s, quant, &format!("decoder.decode.{index}"));
            if block.pool.is_some() {
                res = false;
                cur_seq *= 2;
            }
        }
        self.generator.forward(&x, cur_seq, s, f0_curve, rng, quant)
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// The loaded KittenTTS model.
pub struct Model {
    config: ModelConfig,
    albert: KittenAlbert,
    bert_encoder_w: Vec<f32>,
    bert_encoder_b: Vec<f32>,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: KittenDecoder,
    quant: ActivationQuant,
    /// Voice style rows: name -> `[rows, 2 * style_dim]`.
    voices: HashMap<String, Vec<f32>>,
    voice_rows: HashMap<String, usize>,
    /// Fixed-seed generator for the vocoder's stochastic terms.
    pub seed: u64,
}

/// One prepared segment: the ids, the style vector, and the resolved
/// speed (the reference `_prepare_inputs` output).
#[derive(Debug, Clone)]
pub struct PreparedSegment {
    pub input_ids: Vec<u32>,
    pub ref_s: Vec<f32>,
    pub speed: f32,
}

impl Model {
    /// Opens a model directory: `config.json`, the weights
    /// (`model.safetensors`), and the voice table (`voices.safetensors`
    /// with a `[rows, 256]` `voices` tensor, or the reference
    /// `voices.npz` name pointing at one).
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
        let config = ModelConfig::from_json(&config)?;
        let weights_name = ["model.safetensors"]
            .iter()
            .find(|n| dir.join(n).exists())
            .ok_or_else(|| SpeechError::BadConfig {
                field: "weights".to_string(),
                why: "no kitten_tts safetensors in the model directory".to_string(),
            })?;
        let file = SafetensorsFile::open(&dir.join(weights_name))?;
        let quant = ActivationQuant::new(&config.activation_quant_modules);
        let albert = KittenAlbert::load(&file, &config.plbert, &quant)?;
        let predictor = ProsodyPredictor::load(
            &file,
            "predictor",
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
        )?;
        let text_encoder = TextEncoder::load(
            &file,
            "text_encoder",
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
        )?;
        let decoder = KittenDecoder::load(&file, "decoder", &config)?;
        // Voice table: a safetensors file named after `voices_path` (with
        // the .npz spelling mapped to .safetensors) carrying a 2-D
        // `voices` tensor of `[rows, 2 * style_dim]`.
        let voice_file_name = config.voices_path.replace(".npz", ".safetensors");
        let voice_path = dir.join(&voice_file_name);
        let mut voices = HashMap::new();
        let mut voice_rows = HashMap::new();
        if voice_path.exists() {
            let voice_file = SafetensorsFile::open(&voice_path)?;
            let names: Vec<&str> = ["voices", "voice"]
                .iter()
                .copied()
                .filter(|n| voice_file.contains_tensor(n))
                .collect();
            let name = names.first().copied().ok_or_else(|| SpeechError::Tensor {
                name: voice_file_name.clone(),
                why: "no voices tensor".to_string(),
            })?;
            let shape = voice_file
                .descriptor(name)
                .map(|d| d.shape.clone())
                .unwrap_or_default();
            let rows = shape.first().copied().unwrap_or(0);
            let values = voice_file.load_as_f32(name)?;
            let voice = "test-voice".to_string();
            voices.insert(voice.clone(), values);
            voice_rows.insert(voice, rows);
        }
        Ok(Model {
            bert_encoder_w: file.load_as_f32("bert_encoder.weight")?,
            bert_encoder_b: file.load_as_f32("bert_encoder.bias")?,
            config,
            albert,
            predictor,
            text_encoder,
            decoder,
            quant,
            voices,
            voice_rows,
            seed: 0,
        })
    }

    /// Installs a voice table directly (`name -> [rows, 2 * style_dim]`).
    pub fn install_voice(&mut self, name: &str, values: Vec<f32>, rows: usize) {
        self.voices.insert(name.to_string(), values);
        self.voice_rows.insert(name.to_string(), rows);
    }

    /// Maps a phonemized string to ids the way the reference does:
    /// whitespace-tokenize, keep characters present in the vocab, wrap
    /// with the pad id 0.
    pub fn phoneme_ids(&self, phonemes: &str) -> Vec<u32> {
        let cleaner = TextCleaner::new();
        let token_string = basic_english_tokenize(phonemes).join(" ");
        let mut ids = vec![0u32];
        ids.extend(cleaner.clean(&token_string));
        ids.push(0);
        ids
    }

    /// The reference `_prepare_inputs`: resolve the voice alias and speed
    /// prior, pick the style row for the chunk's length, and build the
    /// ids from the phonemized string. `text_len` is the chunk's
    /// character length (before preprocessing), which selects the row.
    pub fn prepare_segment(
        &self,
        phonemes: &str,
        text_len: usize,
        voice: &str,
        speed: f32,
    ) -> Result<PreparedSegment> {
        let voice = self
            .config
            .voice_aliases
            .get(voice)
            .cloned()
            .unwrap_or_else(|| voice.to_string());
        let values = self.voices.get(&voice).ok_or_else(|| SpeechError::Input {
            why: format!("voice '{voice}' not available; install the voice table"),
        })?;
        let rows = self.voice_rows.get(&voice).copied().unwrap_or(0);
        let mut speed = speed;
        if let Some(prior) = self.config.speed_priors.get(&voice) {
            speed *= prior;
        }
        let mut ids = vec![0u32];
        ids.extend(TextCleaner::new().clean(&basic_english_tokenize(phonemes).join(" ")));
        ids.push(0);
        let cols = self.config.style_dim * 2;
        if rows == 0 || values.len() < cols {
            return Err(SpeechError::Tensor {
                name: "voices".to_string(),
                why: "voice table is empty".to_string(),
            });
        }
        let ref_id = text_len.min(rows - 1);
        let ref_s = values[ref_id * cols..(ref_id + 1) * cols].to_vec();
        Ok(PreparedSegment {
            input_ids: ids,
            ref_s,
            speed,
        })
    }

    /// Synthesizes one segment. `ref_s` is the `[2 * style_dim]` style
    /// vector; returns mono samples at [`SAMPLE_RATE`]. The stochastic
    /// vocoder terms reset the pinned MLX key sequence at `seed`.
    pub fn generate_segment(
        &self,
        input_ids: &[u32],
        ref_s: &[f32],
        speed: f32,
    ) -> Result<Vec<f32>> {
        if !speed.is_finite() || speed <= 0.0 || ref_s.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "speed must be finite and positive, and the style must be finite".to_string(),
            });
        }
        if ref_s.len() != 2 * self.config.style_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "ref_s must be {} values, got {}",
                    2 * self.config.style_dim,
                    ref_s.len()
                ),
            });
        }
        let mut rng = Rng::new(self.seed);
        self.generate_with_rng(input_ids, ref_s, speed, &mut rng)
    }

    /// The durations and audio for one segment (the reference
    /// `Model.__call__` with `return_output=True`).
    pub fn generate_segment_with_durations(
        &self,
        input_ids: &[u32],
        ref_s: &[f32],
        speed: f32,
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        let mut rng = Rng::new(self.seed);
        let audio = self.generate_with_rng(input_ids, ref_s, speed, &mut rng)?;
        let durations = self.durations_for(input_ids, ref_s, speed)?;
        Ok((audio, durations))
    }

    fn durations_for(&self, input_ids: &[u32], ref_s: &[f32], speed: f32) -> Result<Vec<i32>> {
        let seq = input_ids.len();
        let bert_out = self.albert.forward(input_ids)?;
        let d_en = ops::linear(
            &bert_out,
            &self.bert_encoder_w,
            Some(&self.bert_encoder_b),
            seq,
            self.albert.hidden,
            self.config.hidden_dim,
        );
        // [seq, hidden] -> channel-major [hidden, seq]
        let mut d_en_c = vec![0.0f32; d_en.len()];
        for t in 0..seq {
            for c in 0..self.config.hidden_dim {
                d_en_c[c * seq + t] = d_en[t * self.config.hidden_dim + c];
            }
        }
        let s = &ref_s[STYLE_SPLIT..];
        Ok(self
            .predictor
            .durations(&d_en_c, seq, s, speed, &self.quant))
    }

    fn generate_with_rng(
        &self,
        input_ids: &[u32],
        ref_s: &[f32],
        speed: f32,
        rng: &mut Rng,
    ) -> Result<Vec<f32>> {
        let _ = ids_checked(input_ids)?;
        let seq = input_ids.len();
        // KittenAlbert duration features.
        let bert_out = self.albert.forward(input_ids)?;
        let bert_out = maybe_fake_quant(&bert_out, self.quant.enabled("bert_encoder"));
        let d_en = ops::linear(
            &bert_out,
            &self.bert_encoder_w,
            Some(&self.bert_encoder_b),
            seq,
            self.albert.hidden,
            self.config.hidden_dim,
        );
        // [seq, hidden] -> channel-major [hidden, seq]
        let mut d_en_c = vec![0.0f32; d_en.len()];
        for t in 0..seq {
            for c in 0..self.config.hidden_dim {
                d_en_c[c * seq + t] = d_en[t * self.config.hidden_dim + c];
            }
        }
        let s = &ref_s[STYLE_SPLIT..];
        let durations = self
            .predictor
            .durations(&d_en_c, seq, s, speed, &self.quant);
        // Alignment one-hot: rows = phonemes, cols = total frames.
        let frames: usize = durations.iter().map(|&d| d as usize).sum();
        let mut indices = Vec::with_capacity(frames);
        for (i, &n) in durations.iter().enumerate() {
            for _ in 0..n {
                indices.push(i);
            }
        }
        // en = predictor-text-encoder features aligned to frames.
        let d = self.predictor.text_encoder.forward(
            &d_en_c,
            seq,
            s,
            &self.quant,
            "predictor.text_encoder",
        );
        let d_channels = self.config.hidden_dim + self.config.style_dim;
        let mut en = vec![0.0f32; d_channels * frames];
        for (frame, &ph) in indices.iter().enumerate() {
            let row = &d[ph * d_channels..(ph + 1) * d_channels];
            for (c, &v) in row.iter().enumerate() {
                en[c * frames + frame] = v;
            }
        }
        let (f0_pred, n_pred) = self.predictor.f0_n_train(&en, frames, s, &self.quant);
        // Text encoder + alignment.
        let t_en = self.text_encoder.forward(input_ids, &self.quant)?;
        let mut asr = vec![0.0f32; self.config.hidden_dim * frames];
        for (frame, &ph) in indices.iter().enumerate() {
            for c in 0..self.config.hidden_dim {
                asr[c * frames + frame] = t_en[c * seq + ph];
            }
        }
        // Decoder with the first half of the style.
        Ok(self.decoder.forward(
            &asr,
            frames,
            &f0_pred,
            &n_pred,
            &ref_s[..STYLE_SPLIT],
            rng,
            &self.quant,
        ))
    }

    /// The config this model loaded.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The reference `generate` glue: strip, chunk, phonemize (via the
    /// caller-owned `phonemize` closure), prepare, decode, crossfade the
    /// chunk boundaries, and apply the tail treatment to the last pending
    /// segment. Returns the segments in yield order (N chunks yield N
    /// segments: the first N-1 are crossfaded merges, the last is the
    /// tail-treated final chunk). An empty text yields nothing.
    pub fn synthesize(
        &self,
        text: &str,
        phonemize: &dyn Fn(&str) -> String,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<Vec<f32>>> {
        self.synthesize_with(text, phonemize, voice, speed, 400, 20, 200, 200)
    }

    /// [`Model::synthesize`] with the reference's full parameter set.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the reference generate signature"
    )]
    pub fn synthesize_with(
        &self,
        text: &str,
        phonemize: &dyn Fn(&str) -> String,
        voice: &str,
        speed: f32,
        chunk_size: usize,
        crossfade_ms: i64,
        fade_out_ms: i64,
        tail_silence_ms: i64,
    ) -> Result<Vec<Vec<f32>>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let chunks = if text.chars().count() <= chunk_size {
            vec![ensure_punctuation(text)]
        } else {
            chunk_text(text, chunk_size)
        };
        let preprocessor = TextPreprocessor::new();
        let mut rng = Rng::new(self.seed);
        let mut pending: Option<Vec<f32>> = None;
        let mut segments: Vec<Vec<f32>> = Vec::new();
        for chunk in &chunks {
            let cleaned = preprocessor.process(chunk);
            let phonemes = phonemize(&cleaned);
            let segment = self.prepare_segment(&phonemes, cleaned.chars().count(), voice, speed)?;
            let audio = self.generate_with_rng(
                &segment.input_ids,
                &segment.ref_s,
                segment.speed,
                &mut rng,
            )?;
            match pending.take() {
                None => pending = Some(audio),
                Some(prev) => {
                    let (merged, rest) =
                        crossfade(&prev, &audio, crossfade_ms, self.config.sample_rate);
                    pending = Some(rest);
                    segments.push(merged);
                }
            }
        }
        if let Some(pending) = pending {
            segments.push(self.apply_tail(&pending, crossfade_ms, fade_out_ms, tail_silence_ms));
        }
        Ok(segments)
    }

    /// The reference `_apply_tail`: trim trailing spurts after a silence
    /// run, apply the dynamic fade-out, and append the tail silence.
    pub fn apply_tail(
        &self,
        audio: &[f32],
        _crossfade_ms: i64,
        fade_out_ms: i64,
        tail_silence_ms: i64,
    ) -> Vec<f32> {
        let rate = self.config.sample_rate as i64;
        let fade_out_samples = (rate * fade_out_ms.max(0) / 1000) as usize;
        let tail_silence_samples = (rate * tail_silence_ms.max(0) / 1000) as usize;
        let mut audio = audio.to_vec();
        // Trim spurious trailing spurts after a brief silence (a common
        // end artifact): scan the last second in 10 ms frames backwards.
        let hop = (rate / 100).max(1) as usize; // 10 ms
        let tail_len = audio.len().min(rate as usize);
        if tail_len > hop * 3 {
            let tail = &audio[audio.len() - tail_len..];
            let n_frames = tail.len() / hop;
            if n_frames > 3 {
                let tail = &tail[tail.len() - n_frames * hop..];
                let mut rms = vec![0.0f32; n_frames];
                for (f, frame) in tail.chunks_exact(hop).enumerate() {
                    let energy: f32 = frame.iter().map(|v| v * v).sum::<f32>() / hop as f32;
                    rms[f] = energy.sqrt();
                }
                let rms_max = rms.iter().cloned().fold(0.0f32, f32::max);
                if rms_max > 1e-6 {
                    let silence_rel = 0.1f32;
                    let resume_rel = 0.2f32;
                    let min_silence_frames = ((0.03 / 0.01) as usize).max(3);
                    let mut run = 0usize;
                    for i in (0..n_frames).rev() {
                        if rms[i] / (rms_max + 1e-9) < silence_rel {
                            run += 1;
                        } else {
                            if run >= min_silence_frames {
                                let low_end = i + run;
                                // If audio resumes after the low-energy
                                // span, trim it.
                                if rms[low_end + 1..]
                                    .iter()
                                    .any(|v| v / (rms_max + 1e-9) > resume_rel)
                                {
                                    let cut_idx = audio.len() - tail_len + (low_end + 1) * hop;
                                    audio.truncate(cut_idx);
                                }
                                break;
                            }
                            run = 0;
                        }
                    }
                }
            }
        }
        if fade_out_samples > 0 {
            // Dynamic fade: find the last energetic 10 ms frame near the
            // end and fade from there.
            let hop = (rate / 100).max(1) as usize;
            let tail_len = audio
                .len()
                .min((rate as f64 * (fade_out_ms.max(0).max(400) as f64) / 1000.0) as usize);
            let mut fade_start = audio.len().saturating_sub(fade_out_samples);
            if tail_len > hop {
                let tail = &audio[audio.len() - tail_len..];
                let n_frames = tail.len() / hop;
                if n_frames > 0 {
                    let tail = &tail[tail.len() - n_frames * hop..];
                    let mut rms = vec![0.0f32; n_frames];
                    for (f, frame) in tail.chunks_exact(hop).enumerate() {
                        let energy: f32 = frame.iter().map(|v| v * v).sum::<f32>() / hop as f32;
                        rms[f] = energy.sqrt();
                    }
                    if !rms.is_empty() {
                        let rms_max = rms.iter().cloned().fold(0.0f32, f32::max);
                        let thr = (rms_max * 0.05).max(1e-4);
                        if let Some(last) = rms.iter().rposition(|v| *v > thr) {
                            fade_start = audio.len() - tail_len + last * hop;
                        }
                    }
                }
            }
            let mut fade_len = audio.len() - fade_start.min(audio.len());
            if fade_len < fade_out_samples {
                fade_start = audio.len().saturating_sub(fade_out_samples);
                fade_len = audio.len() - fade_start;
            }
            if fade_len > 0 {
                let mut faded = audio[..fade_start].to_vec();
                for (i, v) in audio[fade_start..].iter().enumerate() {
                    let t = i as f32 / fade_len as f32;
                    faded.push(v * (1.0 - t));
                }
                audio = faded;
            }
        }
        if tail_silence_samples > 0 {
            audio.extend(std::iter::repeat(0.0).take(tail_silence_samples));
        }
        audio
    }
}

/// The reference `_crossfade`: overlap the tail of `prev` with the head
/// of `next`, returning the merged audio and the next remainder.
pub fn crossfade(
    prev: &[f32],
    next: &[f32],
    crossfade_ms: i64,
    sample_rate: u32,
) -> (Vec<f32>, Vec<f32>) {
    let crossfade_samples = (sample_rate as i64 * crossfade_ms.max(0) / 1000) as usize;
    if crossfade_samples == 0 {
        return (prev.to_vec(), next.to_vec());
    }
    let fade = crossfade_samples.min(prev.len()).min(next.len());
    if fade == 0 {
        return (prev.to_vec(), next.to_vec());
    }
    let mut out = prev[..prev.len() - fade].to_vec();
    for (i, (a, b)) in prev[prev.len() - fade..]
        .iter()
        .zip(&next[..fade])
        .enumerate()
    {
        let t = i as f32 / fade as f32;
        out.push(a * (1.0 - t) + b * t);
    }
    (out, next[fade..].to_vec())
}

fn ids_checked(input_ids: &[u32]) -> Result<()> {
    if input_ids.is_empty() {
        return Err(SpeechError::Input {
            why: "input ids are empty".to_string(),
        });
    }
    Ok(())
}
