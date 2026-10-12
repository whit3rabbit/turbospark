//! MeloTTS: VITS2-family TTS with a transformer flow and a HiFi-GAN
//! vocoder.
//!
//! Port of `mlx_audio/tts/models/melotts/` (melotts.py, modules.py,
//! attentions.py, hifigan.py, transforms.py, bert.py, text.py) at
//! mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/melotts).
//! Inference path (batch size 1, non-streaming): phones -> embeddings
//! (phone, tone, language, the 1024-dim bert projection, the 768-dim
//! ja_bert projection) -> VITS transformer encoder (window-4 relative
//! attention, speaker conditioning) -> deterministic duration predictor
//! -> monotonic alignment -> mean/log-variance alignment -> latent sample
//! -> transformer coupling flow (reverse, mean-only) -> HiFi-GAN decoder
//! with speaker conditioning -> 44.1 kHz audio. The 12-layer BERT for the
//! ja_bert features is ported in `bert.rs`; see the family README for the
//! G2P/tokenizer intake.
//!
//! Weight layouts: checkpoints carry the raw MLX parameter tree; conv
//! weights are MLX order `[out, kernel, in]` and transpose-conv weights
//! `[out, kernel, in]` as well (the loader transposes to the
//! `[out, in, kernel]` / `[in, out, kernel]` layouts `ops` wants).

mod bert;
mod text;

#[cfg(test)]
mod tests;

pub use bert::BertModel;
pub use text::{
    add_blank_interleave, cleaned_text_to_sequence, default_symbols, distribute_phone,
    number_to_words, post_replace_ph, refine_ph, text_normalize,
};

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::music::minimax_music3::rng::{self, KeySequence};
use crate::{ops, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// The reference ja_bert input width (bert-base's hidden size).
pub const REFERENCE_BERT_DIM: usize = 768;

/// The reference `ModelConfig.sampling_rate` default.
pub const SAMPLE_RATE: u32 = 44_100;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The BERT sub-dimensions: the reference hard-codes the bert-base dims
/// (`BertConfig` defaults) and the 768-dim ja_bert projection; the
/// checkpoint `config.json` may override them (a fixture-only extension).
#[derive(Debug, Clone)]
pub struct BertDims {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl Default for BertDims {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }
}

impl BertDims {
    fn from_json(value: Option<&serde_json::Value>) -> Self {
        let Some(value) = value else {
            return Self::default();
        };
        let defaults = Self::default();
        let num = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(default as u64) as usize
        };
        Self {
            vocab_size: num("vocab_size", defaults.vocab_size),
            hidden_size: num("hidden_size", defaults.hidden_size),
            num_hidden_layers: num("num_hidden_layers", defaults.num_hidden_layers),
            num_attention_heads: num("num_attention_heads", defaults.num_attention_heads),
            intermediate_size: num("intermediate_size", defaults.intermediate_size),
            max_position_embeddings: num(
                "max_position_embeddings",
                defaults.max_position_embeddings,
            ),
            type_vocab_size: num("type_vocab_size", defaults.type_vocab_size),
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-12) as f32,
        }
    }
}

/// The TTS `ModelConfig`, defaults per the reference dataclass.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub sampling_rate: u32,
    pub filter_length: usize,
    pub hop_length: usize,
    pub add_blank: bool,
    pub n_speakers: usize,
    /// The `spk2id` table; voice resolution tries the requested voice,
    /// then the lang code, then "EN-Default", then id 0.
    pub spk2id: std::collections::HashMap<String, i32>,
    pub inter_channels: usize,
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_layers_trans_flow: usize,
    pub kernel_size: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gin_channels: usize,
    pub use_spk_conditioned_encoder: bool,
    pub use_transformer_flow: bool,
    pub num_tones: usize,
    pub num_languages: usize,
    pub n_vocab: usize,
    pub bert_hidden_size: usize,
    /// The symbol table from the checkpoint config.json.
    pub symbols: Vec<String>,
    pub bert: BertDims,
}

impl ModelConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let num = |key: &str, default: u64| -> usize {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(default) as usize
        };
        let list = |key: &str| -> Vec<usize> {
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
        let list_list = |key: &str| -> Vec<Vec<usize>> {
            value
                .get(key)
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
                .unwrap_or_default()
        };
        let symbols = match value.get("symbols").and_then(serde_json::Value::as_array) {
            Some(rows) => rows
                .iter()
                .filter_map(|v| v.as_str())
                .map(str::to_string)
                .collect(),
            None => {
                return Err(SpeechError::BadConfig {
                    field: "symbols".to_string(),
                    why: "the melotts symbol table is required".to_string(),
                })
            }
        };
        Ok(Self {
            sampling_rate: num("sampling_rate", SAMPLE_RATE as u64) as u32,
            filter_length: num("filter_length", 2048),
            hop_length: num("hop_length", 512),
            add_blank: value
                .get("add_blank")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            n_speakers: num("n_speakers", 256),
            spk2id: value
                .get("spk2id")
                .and_then(serde_json::Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_i64().map(|v| (k.clone(), v as i32)))
                        .collect()
                })
                .unwrap_or_default(),
            inter_channels: num("inter_channels", 192),
            hidden_channels: num("hidden_channels", 192),
            filter_channels: num("filter_channels", 768),
            n_heads: num("n_heads", 2),
            n_layers: num("n_layers", 6),
            n_layers_trans_flow: num("n_layers_trans_flow", 3),
            kernel_size: num("kernel_size", 3),
            resblock_kernel_sizes: list("resblock_kernel_sizes"),
            resblock_dilation_sizes: list_list("resblock_dilation_sizes"),
            upsample_rates: list("upsample_rates"),
            upsample_initial_channel: num("upsample_initial_channel", 512),
            upsample_kernel_sizes: list("upsample_kernel_sizes"),
            gin_channels: num("gin_channels", 256),
            use_spk_conditioned_encoder: value
                .get("use_spk_conditioned_encoder")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            use_transformer_flow: value
                .get("use_transformer_flow")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            num_tones: num("num_tones", 16),
            num_languages: num("num_languages", 10),
            n_vocab: num("n_vocab", 219),
            bert_hidden_size: num("bert_hidden_size", 1024),
            symbols,
            bert: BertDims::from_json(value.get("bert")),
        })
    }

    /// The sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sampling_rate
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

fn tensor_shape(file: &SafetensorsFile, name: &str) -> Result<Vec<usize>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    })?;
    Ok(desc.shape.clone())
}

/// Loads a plain conv from the MLX `[out, kernel, in]` order and
/// transposes to the `[out, in, kernel]` layout `ops::conv1d` wants.
fn load_conv(file: &SafetensorsFile, base: &str) -> Result<Conv1d> {
    let weight_mlx = file.load_as_f32(&format!("{base}.conv.weight"))?;
    let bias = file
        .load_as_f32(&format!("{base}.conv.bias"))
        .unwrap_or_default();
    let shape = tensor_shape(file, &format!("{base}.conv.weight"))?;
    let (out, kernel, in_ch) = match shape.as_slice() {
        [o, k, i] => (*o, *k, *i),
        other => {
            return Err(SpeechError::Tensor {
                name: format!("{base}.conv.weight"),
                why: format!("expected 3-D [out, kernel, in], got {other:?}"),
            })
        }
    };
    let mut weight = vec![0.0f32; out * in_ch * kernel];
    for o in 0..out {
        for k in 0..kernel {
            for i in 0..in_ch {
                weight[o * in_ch * kernel + i * kernel + k] =
                    weight_mlx[o * kernel * in_ch + k * in_ch + i];
            }
        }
    }
    Ok(Conv1d {
        weight,
        bias,
        kernel,
        stride: 1,
        padding: 0,
        dilation: 1,
        out_ch: out,
        in_ch,
    })
}

/// Loads a transpose conv from the MLX `[out, kernel, in]` order into the
/// `[in, out, kernel]` layout `ops::conv_transpose1d` wants.
fn load_conv_transpose(file: &SafetensorsFile, base: &str) -> Result<ConvTranspose1d> {
    let weight_mlx = file.load_as_f32(&format!("{base}.conv_t.weight"))?;
    let bias = file
        .load_as_f32(&format!("{base}.conv_t.bias"))
        .unwrap_or_default();
    let shape = tensor_shape(file, &format!("{base}.conv_t.weight"))?;
    let (out, kernel, in_ch) = match shape.as_slice() {
        [o, k, i] => (*o, *k, *i),
        other => {
            return Err(SpeechError::Tensor {
                name: format!("{base}.conv_t.weight"),
                why: format!("expected 3-D [out, kernel, in], got {other:?}"),
            })
        }
    };
    let mut weight = vec![0.0f32; in_ch * out * kernel];
    for o in 0..out {
        for k in 0..kernel {
            for i in 0..in_ch {
                weight[i * out * kernel + o * kernel + k] =
                    weight_mlx[o * kernel * in_ch + k * in_ch + i];
            }
        }
    }
    Ok(ConvTranspose1d {
        weight,
        bias,
        kernel,
        stride: 1,
        padding: 0,
        in_ch,
        out_ch: out,
    })
}

/// A 1x1 projection expressed as a per-timestep linear over the channel
/// axis (`Conv1dPT` with kernel 1).
struct PointwiseLinear {
    weight: Vec<f32>, // [out, in]
    bias: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
}

impl PointwiseLinear {
    /// Loads a `Conv1dPT` kernel-1 projection (`.conv.weight`).
    fn load(file: &SafetensorsFile, base: &str) -> Result<Self> {
        let conv = load_conv(file, base)?;
        debug_assert_eq!(conv.kernel, 1, "{base} must be a 1x1 projection");
        Ok(PointwiseLinear {
            weight: conv.weight,
            bias: conv.bias,
            out_ch: conv.out_ch,
            in_ch: conv.in_ch,
        })
    }

    /// Loads an `nn.Linear` (`weight` `[out, in]`), the speaker
    /// projection the encoder conditioning uses.
    fn load_linear(file: &SafetensorsFile, base: &str) -> Result<Self> {
        let weight = file.load_as_f32(&format!("{base}.weight"))?;
        let bias = file
            .load_as_f32(&format!("{base}.bias"))
            .unwrap_or_default();
        let shape = tensor_shape(file, &format!("{base}.weight"))?;
        let (out, in_ch) = match shape.as_slice() {
            [o, i] => (*o, *i),
            other => {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.weight"),
                    why: format!("expected 2-D [out, in], got {other:?}"),
                })
            }
        };
        Ok(PointwiseLinear {
            weight,
            bias,
            out_ch: out,
            in_ch,
        })
    }

    /// Channel-major `[in, rows]` -> `[out, rows]` (a per-row linear over
    /// the channel axis, the `Conv1dPT` kernel-1 case). The channel-major
    /// input is transposed into the row-major layout `ops::linear` reads,
    /// and the result transposed back, so the accumulation order is the
    /// shared kernel's.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.in_ch;
        let mut row_major = vec![0.0f32; rows * self.in_ch];
        for r in 0..rows {
            for c in 0..self.in_ch {
                row_major[r * self.in_ch + c] = x[c * rows + r];
            }
        }
        let out = ops::linear(
            &row_major,
            &self.weight,
            if self.bias.is_empty() {
                None
            } else {
                Some(&self.bias)
            },
            rows,
            self.in_ch,
            self.out_ch,
        );
        let mut out_c = vec![0.0f32; self.out_ch * rows];
        for r in 0..rows {
            for o in 0..self.out_ch {
                out_c[o * rows + r] = out[r * self.out_ch + o];
            }
        }
        out_c
    }
}

struct Conv1d {
    weight: Vec<f32>, // [out, in, kernel]
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_ch: usize,
    in_ch: usize,
}

impl Conv1d {
    /// Channel-major `[in, seq]` -> `[out, out_seq]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv1d(
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
            self.dilation,
            1,
        )
    }
}

struct ConvTranspose1d {
    weight: Vec<f32>, // [in, out, kernel]
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    padding: usize,
    in_ch: usize,
    out_ch: usize,
}

impl ConvTranspose1d {
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
            1,
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
/// `[channels, seq]` buffer (the melotts `LayerNorm` normalizes the
/// channel axis of its `[B, C, T]` tensors).
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

// ---------------------------------------------------------------------------
// VITS transformer pieces (attentions.py)
// ---------------------------------------------------------------------------

/// Multi-head attention with window-4 relative position representations
/// (the reference `MultiHeadAttention`, heads sharing one relative table).
struct MultiHeadAttention {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    emb_rel_k: Vec<f32>, // [1, 2 * window + 1, k_channels]
    emb_rel_v: Vec<f32>,
    n_heads: usize,
    k_channels: usize,
    window_size: usize,
}

impl MultiHeadAttention {
    fn load(file: &SafetensorsFile, base: &str) -> Result<Self> {
        Ok(MultiHeadAttention {
            conv_q: load_conv(file, &format!("{base}.conv_q"))?,
            conv_k: load_conv(file, &format!("{base}.conv_k"))?,
            conv_v: load_conv(file, &format!("{base}.conv_v"))?,
            conv_o: load_conv(file, &format!("{base}.conv_o"))?,
            emb_rel_k: file.load_as_f32(&format!("{base}.emb_rel_k"))?,
            emb_rel_v: file.load_as_f32(&format!("{base}.emb_rel_v"))?,
            n_heads: 0,
            k_channels: 0,
            window_size: 4,
        })
    }

    fn with_heads(mut self, n_heads: usize, k_channels: usize) -> Self {
        self.n_heads = n_heads;
        self.k_channels = k_channels;
        self
    }

    /// `x` and `c` channel-major `[channels, T]`; `mask` is the `[T, T]`
    /// attention mask the reference builds from `x_mask` (0.0 or -1e4).
    fn forward(&self, x: &[f32], c: &[f32], t: usize, mask: &[f32]) -> Vec<f32> {
        let q = self.conv_q.forward(x);
        let k = self.conv_k.forward(c);
        let v = self.conv_v.forward(c);
        let heads = self.n_heads;
        let d_k = self.k_channels;
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut output = vec![0.0f32; t * heads * d_k];
        for h in 0..heads {
            // Per-head views, rows = timesteps.
            let mut q_head = vec![0.0f32; t * d_k];
            let mut k_head = vec![0.0f32; t * d_k];
            let mut v_head = vec![0.0f32; t * d_k];
            for ti in 0..t {
                for d in 0..d_k {
                    q_head[ti * d_k + d] = q[(h * d_k + d) * t + ti] * scale;
                    k_head[ti * d_k + d] = k[(h * d_k + d) * t + ti];
                    v_head[ti * d_k + d] = v[(h * d_k + d) * t + ti];
                }
            }
            // scores = q_head @ k_head^T, [t, t].
            let mut scores = vec![0.0f32; t * t];
            for (ti, row) in scores.chunks_mut(t).enumerate() {
                for (si, slot) in row.iter_mut().enumerate() {
                    *slot = q_head[ti * d_k..(ti + 1) * d_k]
                        .iter()
                        .zip(&k_head[si * d_k..(si + 1) * d_k])
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            // Window-4 relative positions.
            let rel_k = relative_embeddings(&self.emb_rel_k, t, self.window_size, d_k);
            let mut rel_scores = vec![0.0f32; t * (2 * t - 1)];
            for (ti, row) in rel_scores.chunks_mut(2 * t - 1).enumerate() {
                for (ri, slot) in row.iter_mut().enumerate() {
                    *slot = q_head[ti * d_k..(ti + 1) * d_k]
                        .iter()
                        .zip(&rel_k[ri * d_k..(ri + 1) * d_k])
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            let scores_local = relative_position_to_absolute(&rel_scores, t);
            for (score, local) in scores.iter_mut().zip(&scores_local) {
                *score += local;
            }
            for (score, m) in scores.iter_mut().zip(mask) {
                if *m == 0.0 {
                    *score = -1e4;
                }
            }
            softmax_last_dim(&mut scores, t);
            let mut attn_out = vec![0.0f32; t * d_k];
            for ti in 0..t {
                for d in 0..d_k {
                    attn_out[ti * d_k + d] = scores[ti * t..(ti + 1) * t]
                        .iter()
                        .enumerate()
                        .map(|(si, p)| p * v_head[si * d_k + d])
                        .sum();
                }
            }
            // Relative values.
            let rel_v = relative_embeddings(&self.emb_rel_v, t, self.window_size, d_k);
            let rel_weights = absolute_position_to_relative(&scores, t);
            for ti in 0..t {
                for d in 0..d_k {
                    let rel: f32 = rel_weights[ti * (2 * t - 1)..(ti + 1) * (2 * t - 1)]
                        .iter()
                        .enumerate()
                        .map(|(ri, w)| w * rel_v[ri * d_k + d])
                        .sum();
                    attn_out[ti * d_k + d] += rel;
                }
            }
            // Back to the channel-major concat.
            for ti in 0..t {
                for d in 0..d_k {
                    output[(h * d_k + d) * t + ti] = attn_out[ti * d_k + d];
                }
            }
        }
        self.conv_o.forward(&output)
    }
}

/// `_get_relative_embeddings`: pad to `rows + 2 * pad_length` then slice
/// `2 * length - 1` rows starting at `slice_start`.
fn relative_embeddings(emb: &[f32], length: usize, window_size: usize, d_k: usize) -> Vec<f32> {
    let pad_length = length.saturating_sub(window_size + 1);
    let slice_start = (window_size + 1).saturating_sub(length);
    let rows = window_size * 2 + 1;
    let padded_rows = rows + 2 * pad_length;
    let mut padded = vec![0.0f32; padded_rows * d_k];
    for r in 0..rows {
        padded[(r + pad_length) * d_k..(r + pad_length + 1) * d_k]
            .copy_from_slice(&emb[r * d_k..(r + 1) * d_k]);
    }
    let mut out = vec![0.0f32; (2 * length - 1) * d_k];
    for (r, slot) in out.chunks_mut(d_k).enumerate() {
        let src = slice_start + r;
        slot.copy_from_slice(&padded[src * d_k..(src + 1) * d_k]);
    }
    out
}

/// `_relative_position_to_absolute_position`: `[t, 2t - 1]` -> `[t, t]`.
/// The reference pads one zero column, flattens, appends `t - 1` zeros,
/// reshapes to `(t + 1, 2t - 1)` and takes `[:t, t - 1:]`; the index math
/// crosses source rows through the pad column, so it is reproduced
/// literally.
fn relative_position_to_absolute(rel: &[f32], length: usize) -> Vec<f32> {
    let wide = 2 * length;
    let flat = |r: usize, c: usize| -> f32 {
        if c < wide - 1 {
            rel[r * (wide - 1) + c]
        } else {
            0.0 // the appended zero column
        }
    };
    let mut out = vec![0.0f32; length * length];
    for r in 0..length {
        for c in 0..length {
            let idx = r * (wide - 1) + (length - 1) + c;
            out[r * length + c] = if idx < length * wide {
                flat(idx / wide, idx % wide)
            } else {
                0.0 // the appended tail zeros
            };
        }
    }
    out
}

/// `_absolute_position_to_relative_position`: `[t, t]` -> `[t, 2t - 1]`
/// where `rel[r][j] = abs[r][j - (L - 1 - r)]` (zero outside the band).
fn absolute_position_to_relative(abs: &[f32], length: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; length * (2 * length - 1)];
    for r in 0..length {
        for c in 0..(2 * length - 1) {
            let abs_col = c as isize - (length as isize - 1 - r as isize);
            if abs_col >= 0 && (abs_col as usize) < length {
                out[r * (2 * length - 1) + c] = abs[r * length + abs_col as usize];
            }
        }
    }
    out
}

fn softmax_last_dim(values: &mut [f32], columns: usize) {
    for row in values.chunks_mut(columns) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// The reference `FFN`: two k-size convs with relu and the mask.
struct Ffn {
    conv_1: Conv1d,
    conv_2: Conv1d,
}

impl Ffn {
    fn load(file: &SafetensorsFile, base: &str, kernel: usize) -> Result<Self> {
        let mut conv_1 = load_conv(file, &format!("{base}.conv_1"))?;
        conv_1.padding = (kernel - 1) / 2;
        let mut conv_2 = load_conv(file, &format!("{base}.conv_2"))?;
        conv_2.padding = (kernel - 1) / 2;
        Ok(Ffn { conv_1, conv_2 })
    }

    fn forward(&self, x: &[f32], t: usize, mask: &[f32]) -> Vec<f32> {
        let mut y = self.conv_1.forward(x);
        for v in y.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        y = self.conv_2.forward(&y);
        // The x_mask broadcasts over channels: element (c, ti) takes
        // mask[ti] in the channel-major layout.
        for (i, v) in y.iter_mut().enumerate() {
            *v *= mask[i % t];
        }
        y
    }
}

/// The VITS transformer `Encoder`: window-4 attention, channel-first
/// LayerNorms, FFN, and the optional speaker-conditioning injection at
/// `cond_layer_idx` (2).
struct VitEncoder {
    attn_layers: Vec<MultiHeadAttention>,
    norm_layers_1: Vec<(Vec<f32>, Vec<f32>)>,
    ffn_layers: Vec<Ffn>,
    norm_layers_2: Vec<(Vec<f32>, Vec<f32>)>,
    spk_emb_linear: Option<PointwiseLinear>,
    cond_layer_idx: usize,
}

impl VitEncoder {
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature mirrors the reference constructor"
    )]
    fn load(
        file: &SafetensorsFile,
        base: &str,
        n_layers: usize,
        kernel: usize,
        n_heads: usize,
        k_channels: usize,
        gin_channels: usize,
        gin_conditioned: bool,
    ) -> Result<Self> {
        let mut attn_layers = Vec::new();
        let mut norm_layers_1 = Vec::new();
        let mut ffn_layers = Vec::new();
        let mut norm_layers_2 = Vec::new();
        for i in 0..n_layers {
            attn_layers.push(
                MultiHeadAttention::load(file, &format!("{base}.attn_layers.{i}"))?
                    .with_heads(n_heads, k_channels),
            );
            norm_layers_1.push((
                file.load_as_f32(&format!("{base}.norm_layers_1.{i}.weight"))?,
                file.load_as_f32(&format!("{base}.norm_layers_1.{i}.bias"))?,
            ));
            ffn_layers.push(Ffn::load(file, &format!("{base}.ffn_layers.{i}"), kernel)?);
            norm_layers_2.push((
                file.load_as_f32(&format!("{base}.norm_layers_2.{i}.weight"))?,
                file.load_as_f32(&format!("{base}.norm_layers_2.{i}.bias"))?,
            ));
        }
        // The conditioning layer index is min(2, n_layers); the speaker
        // projection only exists when the encoder is gin-conditioned.
        let spk_emb_linear = if gin_conditioned && gin_channels > 0 {
            Some(PointwiseLinear::load_linear(
                file,
                &format!("{base}.spk_emb_linear"),
            )?)
        } else {
            None
        };
        let cond_layer_idx = if gin_conditioned && gin_channels > 0 {
            2.min(n_layers)
        } else {
            n_layers
        };
        Ok(VitEncoder {
            attn_layers,
            norm_layers_1,
            ffn_layers,
            norm_layers_2,
            spk_emb_linear,
            cond_layer_idx,
        })
    }

    /// `x` channel-major `[channels, T]`, `g` the speaker vector
    /// `[gin, 1]`, `mask` the `[T, T]` attention mask (from `x_mask`).
    fn forward(
        &self,
        x: &[f32],
        t: usize,
        channels: usize,
        mask: &[f32],
        x_mask: &[f32],
        g: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut cur = x.to_vec();
        for v in cur.iter_mut() {
            *v *= x_mask_first(x_mask);
        }
        for i in 0..self.attn_layers.len() {
            if i == self.cond_layer_idx {
                if let Some(spk) = &self.spk_emb_linear {
                    if let Some(g) = g {
                        let proj = spk.forward(g);
                        for c in 0..proj.len() {
                            let p = proj[c];
                            for ti in 0..t {
                                cur[c * t + ti] = (cur[c * t + ti] + p) * x_mask[ti];
                            }
                        }
                    }
                }
            }
            let y = self.attn_layers[i].forward(&cur, &cur, t, mask);
            let mut sum = cur.iter().zip(&y).map(|(a, b)| a + b).collect::<Vec<_>>();
            layernorm_timesteps(
                &mut sum,
                t,
                channels,
                &self.norm_layers_1[i].0,
                &self.norm_layers_1[i].1,
                1e-5,
            );
            cur = sum;
            let mut y = self.ffn_layers[i].forward(&cur, t, x_mask);
            for (v, a) in y.iter_mut().zip(&cur) {
                *v += a;
            }
            layernorm_timesteps(
                &mut y,
                t,
                channels,
                &self.norm_layers_2[i].0,
                &self.norm_layers_2[i].1,
                1e-5,
            );
            cur = y;
        }
        for v in cur.iter_mut() {
            *v *= x_mask_first(x_mask);
        }
        cur
    }
}

fn x_mask_first(x_mask: &[f32]) -> f32 {
    x_mask.first().copied().unwrap_or(1.0)
}

// ---------------------------------------------------------------------------
// Text encoder, duration predictor, flow, decoder
// ---------------------------------------------------------------------------

/// The reference `TextEncoder`.
struct TextEncoder {
    emb: Vec<f32>,          // [n_vocab, hidden]
    tone_emb: Vec<f32>,     // [num_tones, hidden]
    language_emb: Vec<f32>, // [num_languages, hidden]
    bert_proj: PointwiseLinear,
    ja_bert_proj: PointwiseLinear,
    encoder: VitEncoder,
    proj: PointwiseLinear,
    out_channels: usize,
    hidden_channels: usize,
}

impl TextEncoder {
    fn load(file: &SafetensorsFile, base: &str, config: &ModelConfig) -> Result<Self> {
        let k_channels = config.hidden_channels / config.n_heads;
        Ok(TextEncoder {
            emb: file.load_as_f32(&format!("{base}.emb.weight"))?,
            tone_emb: file.load_as_f32(&format!("{base}.tone_emb.weight"))?,
            language_emb: file.load_as_f32(&format!("{base}.language_emb.weight"))?,
            bert_proj: PointwiseLinear::load(file, &format!("{base}.bert_proj"))?,
            ja_bert_proj: PointwiseLinear::load(file, &format!("{base}.ja_bert_proj"))?,
            encoder: VitEncoder::load(
                file,
                &format!("{base}.encoder"),
                config.n_layers,
                config.kernel_size,
                config.n_heads,
                k_channels,
                config.gin_channels,
                config.use_spk_conditioned_encoder,
            )?,
            proj: PointwiseLinear::load(file, &format!("{base}.proj"))?,
            out_channels: config.inter_channels,
            hidden_channels: config.hidden_channels,
        })
    }

    /// Returns `(x, m, logs, t)` channel-major.
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature mirrors the reference call"
    )]
    #[allow(
        clippy::type_complexity,
        reason = "the tuple mirrors the reference return"
    )]
    fn forward(
        &self,
        phone_ids: &[u32],
        tone_ids: &[u32],
        lang_ids: &[u32],
        bert: &[f32],
        ja_bert: &[f32],
        t: usize,
        g: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
        let hidden = self.hidden_channels;
        let mut x = vec![0.0f32; t * hidden];
        for (ti, &id) in phone_ids.iter().enumerate() {
            if id as usize * hidden >= self.emb.len() {
                return Err(SpeechError::Input {
                    why: "phone ids do not fit the phone embedding".to_string(),
                });
            }
            for d in 0..hidden {
                x[ti * hidden + d] = self.emb[id as usize * hidden + d];
            }
        }
        for (ti, &id) in tone_ids.iter().enumerate() {
            if id as usize * hidden >= self.tone_emb.len() {
                return Err(SpeechError::Input {
                    why: "tone ids do not fit the tone embedding".to_string(),
                });
            }
            for d in 0..hidden {
                x[ti * hidden + d] += self.tone_emb[id as usize * hidden + d];
            }
        }
        for (ti, &id) in lang_ids.iter().enumerate() {
            if id as usize * hidden >= self.language_emb.len() {
                return Err(SpeechError::Input {
                    why: "language ids do not fit the language embedding".to_string(),
                });
            }
            for d in 0..hidden {
                x[ti * hidden + d] += self.language_emb[id as usize * hidden + d];
            }
        }
        // The bert features are channel-major `[in_ch, t]`. `zip` below
        // would silently truncate a short slice (and a projection whose
        // output width differs from `hidden` would be truncated the same
        // way), so require exact shapes.
        for (name, proj, feats) in [
            ("bert", &self.bert_proj, bert),
            ("ja_bert", &self.ja_bert_proj, ja_bert),
        ] {
            if proj.out_ch != hidden {
                return Err(SpeechError::BadConfig {
                    field: format!("{name}_proj"),
                    why: format!("output width {} != hidden {hidden}", proj.out_ch),
                });
            }
            if feats.len() != proj.in_ch * t {
                return Err(SpeechError::Input {
                    why: format!(
                        "{name} features ({}) != {} channels x {t} tokens",
                        feats.len(),
                        proj.in_ch
                    ),
                });
            }
        }
        // The projections are channel-major `[hidden, t]` (the
        // `Conv1dPT` output layout), so transpose the row-major
        // embedding sum first and add them in the channel-major layout
        // the reference uses.
        let bert_out = self.bert_proj.forward(bert);
        let ja_out = self.ja_bert_proj.forward(ja_bert);
        let mut xc = vec![0.0f32; t * hidden];
        for ti in 0..t {
            for d in 0..hidden {
                xc[d * t + ti] = x[ti * hidden + d];
            }
        }
        for (v, b) in xc.iter_mut().zip(&bert_out) {
            *v += b;
        }
        for (v, j) in xc.iter_mut().zip(&ja_out) {
            *v += j;
        }
        let t_mask = vec![1.0f32; t * t];
        let x_mask = vec![1.0f32; t];
        let encoded = self.encoder.forward(&xc, t, hidden, &t_mask, &x_mask, g);
        let stats = self.proj.forward(&encoded);
        let m = stats[..self.out_channels * t].to_vec();
        let logs = stats[self.out_channels * t..].to_vec();
        Ok((encoded, m, logs, t))
    }
}

/// The reference `DurationPredictor`.
struct DurationPredictor {
    cond: PointwiseLinear,
    conv_1: Conv1d,
    norm_1: (Vec<f32>, Vec<f32>),
    conv_2: Conv1d,
    norm_2: (Vec<f32>, Vec<f32>),
    proj: PointwiseLinear,
}

impl DurationPredictor {
    fn load(file: &SafetensorsFile, base: &str, config: &ModelConfig) -> Result<Self> {
        let padding = (config.kernel_size - 1) / 2;
        let mut conv_1 = load_conv(file, &format!("{base}.conv_1"))?;
        conv_1.padding = padding;
        let mut conv_2 = load_conv(file, &format!("{base}.conv_2"))?;
        conv_2.padding = padding;
        Ok(DurationPredictor {
            cond: PointwiseLinear::load(file, &format!("{base}.cond"))?,
            conv_1,
            norm_1: (
                file.load_as_f32(&format!("{base}.norm_1.weight"))?,
                file.load_as_f32(&format!("{base}.norm_1.bias"))?,
            ),
            conv_2,
            norm_2: (
                file.load_as_f32(&format!("{base}.norm_2.weight"))?,
                file.load_as_f32(&format!("{base}.norm_2.bias"))?,
            ),
            proj: PointwiseLinear::load(file, &format!("{base}.proj"))?,
        })
    }

    /// `x` channel-major `[in, t]`, `g` the speaker `[gin, 1]` ->
    /// log-durations `[1, t]`.
    fn forward(&self, x: &[f32], t: usize, g: Option<&[f32]>) -> Vec<f32> {
        let mut cur = x.to_vec();
        if let Some(g) = g {
            let bias = self.cond.forward(g);
            for c in 0..self.conv_1.in_ch {
                let b = bias[c];
                for ti in 0..t {
                    cur[c * t + ti] += b;
                }
            }
        }
        cur = self.conv_1.forward(&cur);
        for v in cur.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        // The norms run on the transposed (channels-last) tensor.
        let mut rows = vec![0.0f32; t * self.conv_1.out_ch];
        for ti in 0..t {
            for c in 0..self.conv_1.out_ch {
                rows[ti * self.conv_1.out_ch + c] = cur[c * t + ti];
            }
        }
        ops::layernorm(
            &mut rows,
            t,
            self.conv_1.out_ch,
            &self.norm_1.0,
            Some(&self.norm_1.1),
            1e-5,
        );
        let mut back = vec![0.0f32; self.conv_1.out_ch * t];
        for ti in 0..t {
            for c in 0..self.conv_1.out_ch {
                back[c * t + ti] = rows[ti * self.conv_1.out_ch + c];
            }
        }
        cur = self.conv_2.forward(&back);
        for v in cur.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        let mut rows = vec![0.0f32; t * self.conv_2.out_ch];
        for ti in 0..t {
            for c in 0..self.conv_2.out_ch {
                rows[ti * self.conv_2.out_ch + c] = cur[c * t + ti];
            }
        }
        ops::layernorm(
            &mut rows,
            t,
            self.conv_2.out_ch,
            &self.norm_2.0,
            Some(&self.norm_2.1),
            1e-5,
        );
        let mut back = vec![0.0f32; self.conv_2.out_ch * t];
        for ti in 0..t {
            for c in 0..self.conv_2.out_ch {
                back[c * t + ti] = rows[ti * self.conv_2.out_ch + c];
            }
        }
        self.proj.forward(&back)
    }
}

/// The transformer coupling layer (reverse, mean-only) plus the channel
/// flip; `flow_layers` walks them in reverse order.
struct TransformerCouplingLayer {
    pre: PointwiseLinear,
    enc: VitEncoder,
    post: PointwiseLinear,
    half_channels: usize,
}

impl TransformerCouplingLayer {
    fn load(file: &SafetensorsFile, base: &str, config: &ModelConfig) -> Result<Self> {
        let half = config.inter_channels / 2;
        let k_channels = config.hidden_channels / config.n_heads;
        Ok(TransformerCouplingLayer {
            pre: PointwiseLinear::load(file, &format!("{base}.pre"))?,
            // The reference creates the coupling layers with kernel 5
            // regardless of the model kernel_size.
            enc: VitEncoder::load(
                file,
                &format!("{base}.enc"),
                config.n_layers_trans_flow,
                5,
                config.n_heads,
                k_channels,
                config.gin_channels,
                config.use_spk_conditioned_encoder,
            )?,
            post: PointwiseLinear::load(file, &format!("{base}.post"))?,
            half_channels: half,
        })
    }

    /// Reverse pass (mean-only: the log-scale is zero). `x` channel-major
    /// `[2 * half, t]`.
    fn reverse(&self, x: &[f32], t: usize, g: Option<&[f32]>) -> Vec<f32> {
        let half = self.half_channels;
        let x0 = &x[..half * t];
        let x1 = &x[half * t..];
        // The reference multiplies the pre output by x_mask (ones on the
        // full-length batch-1 path).
        let h = self.pre.forward(x0);
        let t_mask = vec![1.0f32; t * t];
        let x_mask = vec![1.0f32; t];
        let h = self
            .enc
            .forward(&h, t, self.pre.out_ch, &t_mask, &x_mask, g);
        let m = self.post.forward(&h);
        let mut out = x.to_vec();
        for (o, (x1v, mv)) in out[half * t..].iter_mut().zip(x1.iter().zip(&m)) {
            *o = x1v - mv;
        }
        out
    }
}

/// HiFi-GAN `ResBlock1`.
struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn load(
        file: &SafetensorsFile,
        base: &str,
        kernel: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let get_padding = |k: usize, d: usize| (k * d - d) / 2;
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        for d in dilations {
            let mut conv = load_conv(file, &format!("{base}.convs1.{}", convs1.len()))?;
            conv.padding = get_padding(kernel, *d);
            conv.dilation = *d;
            convs1.push(conv);
        }
        for _ in dilations {
            let mut conv = load_conv(file, &format!("{base}.convs2.{}", convs2.len()))?;
            conv.padding = get_padding(kernel, 1);
            conv.dilation = 1;
            convs2.push(conv);
        }
        Ok(ResBlock1 { convs1, convs2 })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let mut xt = cur.iter().map(|v| leaky_relu(*v, 0.1)).collect::<Vec<_>>();
            xt = c1.forward(&xt);
            for v in xt.iter_mut() {
                *v = leaky_relu(*v, 0.1);
            }
            xt = c2.forward(&xt);
            for (v, base) in xt.iter_mut().zip(&cur) {
                *v += base;
            }
            cur = xt;
        }
        cur
    }
}

/// The HiFi-GAN `Generator` (the reference `dec`).
struct HifiGan {
    conv_pre: Conv1d,
    cond: Option<PointwiseLinear>,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1d,
    num_kernels: usize,
}

/// Checks the HiFi-GAN list lengths that `HifiGan::load` indexes by
/// position. A short list used to panic, `k - u` underflowed when a
/// kernel was smaller than its stride, and an empty resblock kernel list
/// divided by zero (NaN audio).
fn check_hifigan_lists(config: &ModelConfig) -> Result<()> {
    let bad = |field: &str, why: String| SpeechError::BadConfig {
        field: field.to_string(),
        why,
    };
    if config.resblock_kernel_sizes.is_empty() {
        return Err(bad(
            "resblock_kernel_sizes",
            "at least one resblock kernel is required".to_string(),
        ));
    }
    if config.upsample_kernel_sizes.len() < config.upsample_rates.len() {
        return Err(bad(
            "upsample_kernel_sizes",
            format!(
                "{} kernels for {} upsample rates",
                config.upsample_kernel_sizes.len(),
                config.upsample_rates.len()
            ),
        ));
    }
    for (k, u) in config
        .upsample_kernel_sizes
        .iter()
        .zip(&config.upsample_rates)
    {
        if k < u {
            return Err(bad(
                "upsample_kernel_sizes",
                format!("kernel {k} is smaller than its stride {u}"),
            ));
        }
    }
    if config.resblock_dilation_sizes.len() < config.resblock_kernel_sizes.len() {
        return Err(bad(
            "resblock_dilation_sizes",
            format!(
                "{} dilation lists for {} resblock kernels",
                config.resblock_dilation_sizes.len(),
                config.resblock_kernel_sizes.len()
            ),
        ));
    }
    Ok(())
}

impl HifiGan {
    fn load(file: &SafetensorsFile, base: &str, config: &ModelConfig) -> Result<Self> {
        check_hifigan_lists(config)?;
        let mut conv_pre = load_conv(file, &format!("{base}.conv_pre"))?;
        conv_pre.padding = 3;
        let cond = if config.gin_channels != 0 {
            Some(PointwiseLinear::load(file, &format!("{base}.cond"))?)
        } else {
            None
        };
        let mut ups = Vec::new();
        for (i, u) in config.upsample_rates.iter().enumerate() {
            let k = config.upsample_kernel_sizes[i];
            let mut up = load_conv_transpose(file, &format!("{base}.ups.{i}"))?;
            up.stride = *u;
            up.padding = (k - u) / 2;
            ups.push(up);
        }
        let mut resblocks = Vec::new();
        for i in 0..ups.len() {
            for (j, k) in config.resblock_kernel_sizes.iter().enumerate() {
                let dils = &config.resblock_dilation_sizes[j];
                resblocks.push(ResBlock1::load(
                    file,
                    &format!(
                        "{base}.resblocks.{}",
                        i * config.resblock_kernel_sizes.len() + j
                    ),
                    *k,
                    dils,
                )?);
            }
        }
        let mut conv_post = load_conv(file, &format!("{base}.conv_post"))?;
        conv_post.padding = 3;
        Ok(HifiGan {
            conv_pre,
            cond,
            ups,
            resblocks,
            conv_post,
            num_kernels: config.resblock_kernel_sizes.len(),
        })
    }

    /// `z` channel-major `[inter_channels, t]`, `g` the speaker
    /// `[gin, 1]` -> waveform `[1, samples]`.
    fn forward(&self, z: &[f32], g: Option<&[f32]>) -> Vec<f32> {
        let mut x = self.conv_pre.forward(z);
        if let (Some(cond), Some(g)) = (&self.cond, g) {
            let bias = cond.forward(g);
            let t = x.len() / self.conv_pre.out_ch;
            for c in 0..self.conv_pre.out_ch {
                let b = bias[c];
                for ti in 0..t {
                    x[c * t + ti] += b;
                }
            }
        }
        for (i, up) in self.ups.iter().enumerate() {
            for v in x.iter_mut() {
                *v = leaky_relu(*v, 0.1);
            }
            x = up.forward(&x);
            let mut acc = vec![0.0f32; x.len()];
            for j in 0..self.num_kernels {
                let out = self.resblocks[i * self.num_kernels + j].forward(&x);
                for (a, b) in acc.iter_mut().zip(&out) {
                    *a += b;
                }
            }
            for a in acc.iter_mut() {
                *a /= self.num_kernels as f32;
            }
            x = acc;
        }
        for v in x.iter_mut() {
            *v = leaky_relu(*v, 0.1);
        }
        x = self.conv_post.forward(&x);
        for v in x.iter_mut() {
            *v = v.tanh();
        }
        x
    }
}

/// The `Flip` flow step: reverse the channel order of a channel-major
/// `[channels, t]` buffer (the reference flips in both directions).
fn flip_channels(x: &[f32], t: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * t];
    for (c, chunk) in out.chunks_mut(t).enumerate() {
        let src = channels - 1 - c;
        chunk.copy_from_slice(&x[src * t..(src + 1) * t]);
    }
    out
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// The MLX key-sequence RNG for the latent sample.
struct Rng(KeySequence);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(KeySequence::new(seed))
    }

    fn normal(&mut self, count: usize) -> Vec<f32> {
        rng::normal(self.0.next(), count)
    }
}

/// The loaded MeloTTS model.
pub struct Model {
    config: ModelConfig,
    enc_p: TextEncoder,
    dec: HifiGan,
    flow_layers: Vec<TransformerCouplingLayer>,
    dp: DurationPredictor,
    emb_g: Vec<f32>, // [n_speakers, gin]
}

impl Model {
    /// Opens a model directory: `config.json` and `model.safetensors`.
    /// The training-only `enc_q.*` / `sdp.*` tensors in the checkpoint
    /// are ignored.
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
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        let mut flow_layers = Vec::new();
        // The reference hard-codes four coupling+flip pairs; the coupling
        // layers sit at the even tree indices (the flips carry no
        // parameters). The config's n_layers_trans_flow is the coupling
        // Encoder depth.
        for i in 0..4 {
            flow_layers.push(TransformerCouplingLayer::load(
                &file,
                &format!("flow_layers.{}", i * 2),
                &config,
            )?);
        }
        Ok(Model {
            dec: HifiGan::load(&file, "dec", &config)?,
            enc_p: TextEncoder::load(&file, "enc_p", &config)?,
            flow_layers,
            dp: DurationPredictor::load(&file, "dp", &config)?,
            emb_g: file.load_as_f32("emb_g.weight")?,
            config,
        })
    }

    /// The reference voice resolution: the requested voice, then the lang
    /// code, then "EN-Default", then id 0.
    pub fn resolve_speaker(&self, voice: Option<&str>, lang_code: &str) -> i32 {
        let spk2id = &self.config.spk2id;
        if let Some(voice) = voice {
            if let Some(id) = spk2id.get(voice) {
                return *id;
            }
        }
        if let Some(id) = spk2id.get(lang_code) {
            return *id;
        }
        spk2id.get("EN-Default").copied().unwrap_or(0)
    }

    /// The reference `infer`: `(x, x_lengths, sid, tone, language, bert,
    /// ja_bert)` -> waveform `[1, samples]`. `length_scale` is the speed
    /// divisor (`1 / speed`). `sdp_ratio` must be zero: the stochastic
    /// duration predictor is training/sampling-only in this port.
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature mirrors the reference infer"
    )]
    pub fn infer(
        &self,
        phone_ids: &[u32],
        tone_ids: &[u32],
        lang_ids: &[u32],
        sid: i32,
        bert: &[f32],
        ja_bert: &[f32],
        noise_scale: f32,
        length_scale: f32,
        seed: u64,
        sdp_ratio: f32,
    ) -> Result<Vec<f32>> {
        if sdp_ratio != 0.0 {
            return Err(SpeechError::Unsupported {
                why: "sdp_ratio > 0 needs the stochastic duration predictor, which is ".to_string()
                    + "not ported; use sdp_ratio 0 (the reference default)",
            });
        }
        let t = phone_ids.len();
        if tone_ids.len() != t || lang_ids.len() != t {
            return Err(SpeechError::Input {
                why: "tone and language ids must match the phone count".to_string(),
            });
        }
        let gin = self.config.gin_channels;
        let sid_index = sid;
        if sid_index < 0 || sid_index as usize * gin >= self.emb_g.len() {
            return Err(SpeechError::Input {
                why: "speaker id does not fit the speaker embedding".to_string(),
            });
        }
        let g = self.emb_g[sid_index as usize * gin..(sid_index as usize + 1) * gin].to_vec();
        let (x, m_p, logs_p, t_used) =
            self.enc_p
                .forward(phone_ids, tone_ids, lang_ids, bert, ja_bert, t, Some(&g))?;
        let logw = self.dp.forward(&x, t_used, Some(&g));
        // w = exp(logw) * x_mask * length_scale; the alignment works on
        // the ceil of that.
        let mut w_ceil = vec![0.0f32; t_used];
        let mut y_frames = 0usize;
        for (i, v) in logw.iter().enumerate() {
            let w = v.exp() * length_scale;
            let ceil = w.ceil().max(0.0);
            w_ceil[i] = ceil;
            y_frames += ceil as usize;
        }
        let y_frames = y_frames.max(1);
        // _generate_path: attn[b, 1, t_x, t_y]; then m_p @ attn.squeeze(1).
        let mut cum = vec![0.0f32; t_used];
        let mut running = 0.0f32;
        for (i, v) in w_ceil.iter().enumerate() {
            running += v;
            cum[i] = running;
        }
        let mut attn = vec![0.0f32; t_used * y_frames];
        for (i, window) in attn.chunks_mut(y_frames).enumerate() {
            let start = if i == 0 { 0.0 } else { cum[i - 1] };
            let end = cum[i];
            for (y, slot) in window.iter_mut().enumerate() {
                if (y as f32) >= start && (y as f32) < end {
                    *slot = 1.0;
                }
            }
        }
        let align = |features: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; features.len() / t_used * y_frames];
            let channels = features.len() / t_used;
            for c in 0..channels {
                for (x_frame, window) in attn.chunks(y_frames).enumerate() {
                    for (y_frame, w) in window.iter().enumerate() {
                        out[c * y_frames + y_frame] += features[c * t_used + x_frame] * w;
                    }
                }
            }
            out
        };
        let m_aligned = align(&m_p);
        let logs_aligned = align(&logs_p);
        // z_p = m + normal * exp(logs) * noise_scale (one MLX draw in
        // channel-major order).
        let mut rng = Rng::new(seed);
        let noise = rng.normal(m_aligned.len());
        let mut z = vec![0.0f32; m_aligned.len()];
        for ((z, m), (n, l)) in z
            .iter_mut()
            .zip(&m_aligned)
            .zip(noise.iter().zip(&logs_aligned))
        {
            *z = m + n * l.exp() * noise_scale;
        }
        // The flow: the reference stack is [coupling, flip] x 4, so the
        // reverse pass runs flip, coupling 3, flip, coupling 2, flip,
        // coupling 1, flip, coupling 0 (no trailing flip).
        let latent = self.flow_reverse(&z, y_frames, Some(&g));
        Ok(self.dec.forward(&latent, Some(&g)))
    }

    /// The reference reverse flow over `z` (`[channels, t]` channel
    /// major): a channel flip before every reversed coupling layer.
    fn flow_reverse(&self, z: &[f32], t: usize, g: Option<&[f32]>) -> Vec<f32> {
        let channels = self.config.inter_channels;
        let mut latent = z.to_vec();
        for layer in self.flow_layers.iter().rev() {
            latent = flip_channels(&latent, t, channels);
            latent = layer.reverse(&latent, t, g);
        }
        latent
    }

    /// The config this model loaded.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
