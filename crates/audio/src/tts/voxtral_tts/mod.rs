//! Voxtral TTS 4B (repo `mlx-community/Voxtral-4B-TTS-2603`).
//!
//! Port of `mlx_audio/tts/models/voxtral_tts/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxtral_tts)
//! (MLX 0.32.3). A Mistral-geometry LM backbone (the shared [lm engine]
//! over llama math with traditional/interleaved rope, tied embeddings,
//! and no biases) consumes caller tokenized speech-request prompt ids
//! whose `[AUDIO]` positions are replaced by per-voice embedding rows.
//! Every step the last hidden state predicts one 80 ms frame: the
//! semantic code by masked argmax over a padded 8320-way head, the 36
//! acoustic FSQ codes by an 8-step Euler flow matching head with
//! classifier-free guidance (alpha 1.2) over a 3-token bidirectional
//! transformer. The frame's 37 codes sum their per-codebook embedding
//! rows (one shared table, per-codebook offsets) and feed the next LM
//! step. At the end the code grid is decoded to 24 kHz audio by the
//! codec decoder: weight-normed causal convolutions (replicate/reflect
//! pads, transposed upsampling) alternating with ALiBi sliding-window
//! transformer stages, an EMA semantic codebook, and a finite-scalar
//! acoustic quantizer.
//!
//! Scope, matching the reference single-prompt path: single batch,
//! non-streaming (the reference streaming overlap-add chunking is not
//! ported), caller-owned text tokenization (`_encode_text` needs
//! mistral-common's tekken speech tokenizer; hand over the prompt ids).
//! The reference accepts `temperature`/`top_k`/`top_p` but never uses
//! them: generation is deterministic given the per-frame flow noise,
//! which comes from a [`FlowNoise`] provider (recorded-draw replay for
//! fixtures, the crate-local threefry RNG for standalone runs). See
//! [README.md](README.md) for the verification evidence and open items.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::lm::engine::{Backbone, DenseConfig, Input};
use crate::lm::{Bias, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{quant, Result, SpeechError};

#[cfg(test)]
mod tests;

/// The flow noise source: one standard-normal draw of
/// `n_acoustic_codebook` values per generated frame (the initial Euler
/// state, scaled by `sigma_max`).
pub trait FlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// Seeded crate-local threefry noise for standalone runs.
pub struct SeededFlowNoise {
    keys: rng::KeySequence,
}

impl SeededFlowNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            keys: rng::KeySequence::new(seed),
        }
    }
}

impl FlowNoise for SeededFlowNoise {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        rng::normal(self.keys.next(), n)
    }
}

/// Round to nearest with ties to even (`mx.round` follows numpy; Rust's
/// `f32::round` rounds ties away from zero, which differs on the exact
/// halves the FSQ boundaries can produce).
fn round_ties_even(value: f32) -> f32 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 && rounded.rem_euclid(2.0) != 0.0 {
        rounded - value.signum()
    } else {
        rounded
    }
}

fn bad_config(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn usize_field(value: &Value, key: &str, default: usize) -> Result<usize> {
    match value.get(key).and_then(Value::as_u64) {
        Some(number) => usize::try_from(number).map_err(|_| bad_config(key, "does not fit usize")),
        None => Ok(default),
    }
}

fn f32_field(value: &Value, key: &str, default: f32) -> f32 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn bool_field(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn i32_field(value: &Value, key: &str, default: i32) -> i32 {
    value
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok())
        .unwrap_or(default)
}

fn lengths_field(value: &Value, key: &str, default: &str) -> Vec<usize> {
    let text = value.get(key).and_then(Value::as_str).unwrap_or(default);
    text.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

/// The params.json key the flat reader must not inherit from the top
/// level (its nested block owns every field the params path reads).
const MULTIMODAL_KEYS: &[&str] = &["multimodal"];

/// The resolved model config (`ModelConfig.from_dict`). Parses both the
/// direct voxtral format and the Mistral `params.json` layout with the
/// nested `multimodal` block; every field falls back to the reference
/// dataclass default when absent.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    // LM backbone (Mistral-like).
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub tied_embeddings: bool,
    pub use_biases: bool,
    pub max_position_embeddings: usize,
    pub sample_rate: u32,
    // Audio model.
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub frame_rate: f32,
    pub num_codebooks: usize,
    // Acoustic transformer (flow matching head).
    pub acoustic_dim: usize,
    pub acoustic_n_layers: usize,
    pub acoustic_head_dim: usize,
    pub acoustic_hidden_dim: usize,
    pub acoustic_n_heads: usize,
    pub acoustic_n_kv_heads: usize,
    pub acoustic_rope_theta: f32,
    pub acoustic_sigma: f32,
    pub acoustic_sigma_max: f32,
    // Audio tokenizer (codec decoder).
    pub tokenizer_dim: usize,
    pub tokenizer_hidden_dim: usize,
    pub tokenizer_n_heads: usize,
    pub tokenizer_n_kv_heads: usize,
    pub tokenizer_head_dim: usize,
    pub tokenizer_patch_size: usize,
    pub tokenizer_patch_proj_kernel_size: usize,
    pub tokenizer_semantic_dim: usize,
    pub tokenizer_acoustic_dim: usize,
    pub tokenizer_norm_eps: f32,
    pub tokenizer_decoder_transformer_lengths: Vec<usize>,
    pub tokenizer_decoder_convs_kernels: Vec<usize>,
    pub tokenizer_decoder_convs_strides: Vec<usize>,
    // Special tokens.
    pub bos_token_id: i32,
    pub audio_token_id: i32,
    pub begin_audio_token_id: i32,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let has_multimodal = value
            .get("multimodal")
            .map(|m| !m.is_null())
            .unwrap_or(false);
        let direct = value.get("model_type").and_then(Value::as_str) == Some("voxtral_tts")
            && !has_multimodal;
        if direct {
            return Self::from_flat(value);
        }
        // Mistral params.json layout: rename the nested blocks into the
        // flat direct-format field names (a missing block contributes no
        // fields, matching the reference `.get(..., {})` defaults).
        let mut flat = serde_json::Map::new();
        if let Some(object) = value.as_object() {
            for (key, field) in object {
                if !MULTIMODAL_KEYS.contains(&key.as_str()) {
                    flat.insert(key.clone(), field.clone());
                }
            }
        }
        let mm = value.get("multimodal").unwrap_or(&Value::Null);
        let audio = mm.get("audio_model_args").unwrap_or(&Value::Null);
        let encoding = audio.get("audio_encoding_args").unwrap_or(&Value::Null);
        let acoustic = audio
            .get("acoustic_transformer_args")
            .unwrap_or(&Value::Null);
        let tokenizer = mm.get("audio_tokenizer_args").unwrap_or(&Value::Null);
        let copy = |target: &mut serde_json::Map<String, Value>,
                    source: &Value,
                    pairs: &[(&str, &str)]| {
            for (from, to) in pairs {
                if let Some(field) = source.get(*from) {
                    if !field.is_null() {
                        target.insert((*to).to_owned(), field.clone());
                    }
                }
            }
        };
        copy(
            &mut flat,
            audio,
            &[
                ("semantic_codebook_size", "semantic_codebook_size"),
                ("acoustic_codebook_size", "acoustic_codebook_size"),
                ("n_acoustic_codebook", "n_acoustic_codebook"),
                ("audio_token_id", "audio_token_id"),
                ("begin_audio_token_id", "begin_audio_token_id"),
            ],
        );
        copy(
            &mut flat,
            encoding,
            &[
                ("sampling_rate", "sample_rate"),
                ("frame_rate", "frame_rate"),
                ("num_codebooks", "num_codebooks"),
            ],
        );
        copy(
            &mut flat,
            acoustic,
            &[
                ("dim", "acoustic_dim"),
                ("n_layers", "acoustic_n_layers"),
                ("head_dim", "acoustic_head_dim"),
                ("hidden_dim", "acoustic_hidden_dim"),
                ("n_heads", "acoustic_n_heads"),
                ("n_kv_heads", "acoustic_n_kv_heads"),
                ("rope_theta", "acoustic_rope_theta"),
                ("sigma", "acoustic_sigma"),
                ("sigma_max", "acoustic_sigma_max"),
            ],
        );
        copy(
            &mut flat,
            tokenizer,
            &[
                ("dim", "tokenizer_dim"),
                ("hidden_dim", "tokenizer_hidden_dim"),
                ("n_heads", "tokenizer_n_heads"),
                ("n_kv_heads", "tokenizer_n_kv_heads"),
                ("head_dim", "tokenizer_head_dim"),
                ("pretransform_patch_size", "tokenizer_patch_size"),
                ("patch_proj_kernel_size", "tokenizer_patch_proj_kernel_size"),
                ("semantic_dim", "tokenizer_semantic_dim"),
                ("acoustic_dim", "tokenizer_acoustic_dim"),
                ("norm_eps", "tokenizer_norm_eps"),
                (
                    "decoder_transformer_lengths_str",
                    "tokenizer_decoder_transformer_lengths_str",
                ),
                (
                    "decoder_convs_kernels_str",
                    "tokenizer_decoder_convs_kernels_str",
                ),
                (
                    "decoder_convs_strides_str",
                    "tokenizer_decoder_convs_strides_str",
                ),
            ],
        );
        copy(&mut flat, mm, &[("bos_token_id", "bos_token_id")]);
        Self::from_flat(&Value::Object(flat))
    }

    /// The flat reader: every key is the reference dataclass field name
    /// and every miss takes the dataclass default.
    fn from_flat(value: &Value) -> Result<Self> {
        Ok(Self {
            model_type: "voxtral_tts".to_owned(),
            dim: usize_field(value, "dim", 3072)?,
            n_layers: usize_field(value, "n_layers", 26)?,
            head_dim: usize_field(value, "head_dim", 128)?,
            hidden_dim: usize_field(value, "hidden_dim", 9216)?,
            n_heads: usize_field(value, "n_heads", 32)?,
            n_kv_heads: usize_field(value, "n_kv_heads", 8)?,
            vocab_size: usize_field(value, "vocab_size", 131072)?,
            rope_theta: f32_field(value, "rope_theta", 1_000_000.0),
            norm_eps: f32_field(value, "norm_eps", 1e-5),
            tied_embeddings: bool_field(value, "tied_embeddings", true),
            use_biases: bool_field(value, "use_biases", false),
            max_position_embeddings: usize_field(value, "max_position_embeddings", 128_000)?,
            sample_rate: usize_field(value, "sample_rate", 24_000)? as u32,
            semantic_codebook_size: usize_field(value, "semantic_codebook_size", 8192)?,
            acoustic_codebook_size: usize_field(value, "acoustic_codebook_size", 21)?,
            n_acoustic_codebook: usize_field(value, "n_acoustic_codebook", 36)?,
            frame_rate: f32_field(value, "frame_rate", 12.5),
            num_codebooks: usize_field(value, "num_codebooks", 37)?,
            acoustic_dim: usize_field(value, "acoustic_dim", 3072)?,
            acoustic_n_layers: usize_field(value, "acoustic_n_layers", 3)?,
            acoustic_head_dim: usize_field(value, "acoustic_head_dim", 128)?,
            acoustic_hidden_dim: usize_field(value, "acoustic_hidden_dim", 9216)?,
            acoustic_n_heads: usize_field(value, "acoustic_n_heads", 32)?,
            acoustic_n_kv_heads: usize_field(value, "acoustic_n_kv_heads", 8)?,
            acoustic_rope_theta: f32_field(value, "acoustic_rope_theta", 10_000.0),
            acoustic_sigma: f32_field(value, "acoustic_sigma", 1e-5),
            acoustic_sigma_max: f32_field(value, "acoustic_sigma_max", 1.0),
            tokenizer_dim: usize_field(value, "tokenizer_dim", 1024)?,
            tokenizer_hidden_dim: usize_field(value, "tokenizer_hidden_dim", 4096)?,
            tokenizer_n_heads: usize_field(value, "tokenizer_n_heads", 8)?,
            tokenizer_n_kv_heads: usize_field(value, "tokenizer_n_kv_heads", 8)?,
            tokenizer_head_dim: usize_field(value, "tokenizer_head_dim", 128)?,
            tokenizer_patch_size: usize_field(value, "tokenizer_patch_size", 240)?,
            tokenizer_patch_proj_kernel_size: usize_field(
                value,
                "tokenizer_patch_proj_kernel_size",
                7,
            )?,
            tokenizer_semantic_dim: usize_field(value, "tokenizer_semantic_dim", 256)?,
            tokenizer_acoustic_dim: usize_field(value, "tokenizer_acoustic_dim", 36)?,
            tokenizer_norm_eps: f32_field(value, "tokenizer_norm_eps", 0.01),
            tokenizer_decoder_transformer_lengths: lengths_field(
                value,
                "tokenizer_decoder_transformer_lengths_str",
                "2,2,2,2",
            ),
            tokenizer_decoder_convs_kernels: lengths_field(
                value,
                "tokenizer_decoder_convs_kernels_str",
                "3,4,4,4",
            ),
            tokenizer_decoder_convs_strides: lengths_field(
                value,
                "tokenizer_decoder_convs_strides_str",
                "1,2,2,2",
            ),
            bos_token_id: i32_field(value, "bos_token_id", 1),
            audio_token_id: i32_field(value, "audio_token_id", 24),
            begin_audio_token_id: i32_field(value, "begin_audio_token_id", 25),
        })
    }

    /// Semantic output width: the codebook padded up to the next 128
    /// boundary, `(size // 128 + 1) * 128` in the reference.
    fn semantic_padded(&self) -> usize {
        (self.semantic_codebook_size / 128 + 1) * 128
    }

    /// Acoustic block of the shared embedding table, padded to 128.
    fn acoustic_padded(&self) -> usize {
        let entries = self.acoustic_codebook_size * self.n_acoustic_codebook;
        entries.div_ceil(128) * 128
    }

    /// The shared audio embedding table rows: padded semantic codebook
    /// plus padded acoustic block.
    fn audio_embedding_size(&self) -> usize {
        self.semantic_padded() + self.acoustic_padded()
    }

    fn num_codebooks(&self) -> usize {
        1 + self.n_acoustic_codebook
    }

    /// The LM backbone geometry for the shared engine: llama math with
    /// interleaved (traditional) rope, no q/k norms, biases per config,
    /// tied head.
    fn dense_config(&self) -> DenseConfig {
        DenseConfig {
            model_type: "llama".to_owned(),
            hidden_size: self.dim,
            num_hidden_layers: self.n_layers,
            intermediate_size: self.hidden_dim,
            num_attention_heads: self.n_heads,
            num_key_value_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.norm_eps,
            vocab_size: self.vocab_size,
            qk_norms: false,
            attention_bias: self.use_biases,
            mlp_bias: self.use_biases,
            rope_theta: self.rope_theta,
            rope_traditional: true,
            rope_scaling: None,
            max_position_embeddings: Some(self.max_position_embeddings),
            tie_word_embeddings: self.tied_embeddings,
            sliding_window: None,
        }
    }
}

/// Translates the engine's HF-style backbone tensor names into the raw
/// consolidated Mistral layout of the voxtral checkpoints
/// (`tok_embeddings`, `layers.N.attention.wq`, `feed_forward.w1`,
/// `attention_norm`). Non-backbone names pass through unchanged.
fn mistral_name_map() -> Arc<dyn Fn(&str) -> String + Send + Sync> {
    Arc::new(|name: &str| {
        if let Some(rest) = name.strip_prefix("layers.") {
            if let Some((index, suffix)) = rest.split_once('.') {
                let mapped = match suffix {
                    "self_attn.q_proj" => Some("attention.wq"),
                    "self_attn.k_proj" => Some("attention.wk"),
                    "self_attn.v_proj" => Some("attention.wv"),
                    "self_attn.o_proj" => Some("attention.wo"),
                    "mlp.gate_proj" => Some("feed_forward.w1"),
                    "mlp.down_proj" => Some("feed_forward.w2"),
                    "mlp.up_proj" => Some("feed_forward.w3"),
                    "input_layernorm" => Some("attention_norm"),
                    "post_attention_layernorm" => Some("ffn_norm"),
                    _ => None,
                };
                if let Some(mapped) = mapped {
                    return format!("layers.{index}.{mapped}");
                }
            }
            return name.to_owned();
        }
        match name {
            "embed_tokens" => "tok_embeddings".to_owned(),
            "lm_head" => "output".to_owned(),
            _ => name.to_owned(),
        }
    })
}

/// One bidirectional transformer block of the flow matching head.
struct AcousticBlock {
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl AcousticBlock {
    fn load(weights: &Weights, config: &ModelConfig, index: usize) -> Result<Self> {
        let block = format!("acoustic_transformer.layers.{index}.");
        let attention = format!("{block}attention.");
        let feed_forward = format!("{block}feed_forward.");
        let bias = if config.use_biases {
            Bias::Required
        } else {
            Bias::None
        };
        let dim = config.acoustic_dim;
        let head_dim = config.acoustic_head_dim;
        let q_width = config.acoustic_n_heads * head_dim;
        let kv_width = config.acoustic_n_kv_heads * head_dim;
        Ok(Self {
            attention_norm: RmsNorm::new(
                weights.tensor(&format!("{block}attention_norm"), &[dim])?,
                config.norm_eps,
            ),
            ffn_norm: RmsNorm::new(
                weights.tensor(&format!("{block}ffn_norm"), &[dim])?,
                config.norm_eps,
            ),
            q: weights.linear(&format!("{attention}wq"), dim, q_width, bias)?,
            k: weights.linear(&format!("{attention}wk"), dim, kv_width, bias)?,
            v: weights.linear(&format!("{attention}wv"), dim, kv_width, bias)?,
            o: weights.linear(&format!("{attention}wo"), q_width, dim, bias)?,
            w1: weights.linear(
                &format!("{feed_forward}w1"),
                dim,
                config.acoustic_hidden_dim,
                bias,
            )?,
            w2: weights.linear(
                &format!("{feed_forward}w2"),
                config.acoustic_hidden_dim,
                dim,
                bias,
            )?,
            w3: weights.linear(
                &format!("{feed_forward}w3"),
                dim,
                config.acoustic_hidden_dim,
                bias,
            )?,
            n_heads: config.acoustic_n_heads,
            n_kv_heads: config.acoustic_n_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5) as f32,
        })
    }

    /// Bidirectional GQA attention without positional encoding over one
    /// batch row of `tokens` positions.
    fn attention(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let width = self.n_heads * self.head_dim;
        let q = self.q.forward(x, tokens);
        let k = self.k.forward(x, tokens);
        let v = self.v.forward(x, tokens);
        let rep = self.n_heads / self.n_kv_heads;
        let mut out = vec![0.0f32; tokens * width];
        let mut scores;
        for h in 0..self.n_heads {
            let kv = h / rep;
            for i in 0..tokens {
                let q_row = &q[i * width + h * self.head_dim..i * width + (h + 1) * self.head_dim];
                scores = Vec::with_capacity(tokens);
                let mut max = f32::NEG_INFINITY;
                for j in 0..tokens {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d]
                            * k[j * self.n_kv_heads * self.head_dim + kv * self.head_dim + d];
                    }
                    let score = dot * self.scale;
                    scores.push(score);
                    max = max.max(score);
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum;
                for j in 0..tokens {
                    let weight = scores[j] * inv;
                    let out_row = &mut out
                        [i * width + h * self.head_dim..i * width + (h + 1) * self.head_dim];
                    for d in 0..self.head_dim {
                        out_row[d] += weight
                            * v[j * self.n_kv_heads * self.head_dim + kv * self.head_dim + d];
                    }
                }
            }
        }
        Ok(self.o.forward(&out, tokens))
    }

    fn forward(&self, x: &mut [f32], tokens: usize) -> Result<()> {
        let mut attended = x.to_vec();
        self.attention_norm.apply(&mut attended, tokens);
        let attention = self.attention(&attended, tokens)?;
        for (value, add) in x.iter_mut().zip(attention) {
            *value += add;
        }
        let mut fed = x.to_vec();
        self.ffn_norm.apply(&mut fed, tokens);
        let gate = self.w1.forward(&fed, tokens);
        let up = self.w3.forward(&fed, tokens);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        let down = self.w2.forward(&fused, tokens);
        for (value, add) in x.iter_mut().zip(down) {
            *value += add;
        }
        Ok(())
    }
}

/// The flow matching acoustic head: semantic codes by masked argmax,
/// acoustic FSQ codes by an 8-step Euler flow with classifier-free
/// guidance over a 3-token (acoustic, time, llm) bidirectional
/// transformer.
struct AcousticHead {
    input_projection: Linear,
    llm_projection: Linear,
    time_projection: Linear,
    /// Sinusoidal time table, `(cos, sin)` halves, derived from
    /// `acoustic_rope_theta` exactly like the reference constructor.
    inv_freq: Vec<f32>,
    blocks: Vec<AcousticBlock>,
    semantic_codebook_output: Linear,
    acoustic_codebook_output: Linear,
    norm: RmsNorm,
    dim: usize,
    n_acoustic_codebook: usize,
    semantic_codebook_size: usize,
    acoustic_codebook_size: usize,
    n_denoising_steps: usize,
    cfg_alpha: f32,
    sigma_max: f32,
}

impl AcousticHead {
    fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        let dim = config.acoustic_dim;
        let input_dim = config.dim;
        let blocks = (0..config.acoustic_n_layers)
            .map(|index| AcousticBlock::load(weights, config, index))
            .collect::<Result<Vec<_>>>()?;
        // TimeEmbedding.inv_freq: exp(-ln(theta) * i / half).
        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (-config.acoustic_rope_theta.ln() * i as f32 / half as f32).exp())
            .collect();
        Ok(Self {
            input_projection: weights.linear(
                "acoustic_transformer.input_projection",
                config.n_acoustic_codebook,
                dim,
                Bias::None,
            )?,
            llm_projection: weights.linear(
                "acoustic_transformer.llm_projection",
                input_dim,
                dim,
                Bias::None,
            )?,
            time_projection: weights.linear(
                "acoustic_transformer.time_projection",
                dim,
                dim,
                Bias::None,
            )?,
            inv_freq,
            blocks,
            semantic_codebook_output: weights.linear(
                "acoustic_transformer.semantic_codebook_output",
                dim,
                config.semantic_padded(),
                Bias::None,
            )?,
            acoustic_codebook_output: weights.linear(
                "acoustic_transformer.acoustic_codebook_output",
                dim,
                config.n_acoustic_codebook,
                Bias::None,
            )?,
            norm: RmsNorm::new(
                weights.tensor("acoustic_transformer.norm", &[dim])?,
                config.norm_eps,
            ),
            dim,
            n_acoustic_codebook: config.n_acoustic_codebook,
            semantic_codebook_size: config.semantic_codebook_size,
            acoustic_codebook_size: config.acoustic_codebook_size,
            n_denoising_steps: 8,
            cfg_alpha: 1.2,
            sigma_max: config.acoustic_sigma_max,
        })
    }

    /// `(cos(t * inv_freq), sin(t * inv_freq))`, the vllm-omni convention.
    fn time_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.inv_freq.len();
        let mut out = vec![0.0f32; 2 * half];
        for (i, &freq) in self.inv_freq.iter().enumerate() {
            let angle = t * freq;
            out[i] = angle.cos();
            out[half + i] = angle.sin();
        }
        out
    }

    /// The bidirectional stack plus final norm over one batch row.
    fn run_transformer(&self, x: &mut [f32], tokens: usize) -> Result<()> {
        for block in &self.blocks {
            block.forward(x, tokens)?;
        }
        self.norm.apply(x, tokens);
        Ok(())
    }

    /// One velocity prediction: the acoustic, time, and llm embeddings
    /// stack into a 3-token sequence; token 0 projects to the velocity.
    /// The reference batches the conditional and unconditional rows into
    /// one forward pass; the rows are independent, so the port runs them
    /// as two single-row passes.
    fn predict_velocity(&self, x_t: &[f32], t: f32, llm: &[f32]) -> Result<Vec<f32>> {
        let acoustic_emb = self.input_projection.forward(x_t, 1);
        let time_emb = self.time_projection.forward(&self.time_embedding(t), 1);
        let llm_emb = self.llm_projection.forward(llm, 1);
        let mut x = Vec::with_capacity(3 * self.dim);
        x.extend_from_slice(&acoustic_emb);
        x.extend_from_slice(&time_emb);
        x.extend_from_slice(&llm_emb);
        self.run_transformer(&mut x, 3)?;
        Ok(self.acoustic_codebook_output.forward(&x[..self.dim], 1))
    }

    /// Masked argmax over the padded semantic head: the two special
    /// tokens (0 empty, 1 end) stay in range but every entry at or past
    /// `semantic_codebook_size + 2` and index 0 take a -1e9 mask.
    fn predict_semantic(&self, llm: &[f32]) -> usize {
        let mut logits = self.semantic_codebook_output.forward(llm, 1);
        let keep = self.semantic_codebook_size + 2;
        for (index, logit) in logits.iter_mut().enumerate() {
            if index == 0 || index >= keep {
                *logit += -1e9;
            }
        }
        crate::tts::sampling::argmax(&logits)
    }

    /// One frame: the semantic code plus the quantized acoustic codes,
    /// both carrying the +2 special-token offset.
    fn decode_one_frame(
        &self,
        llm: &[f32],
        llm_uncond: &[f32],
        noise: &mut dyn FlowNoise,
    ) -> Result<Vec<i32>> {
        let n = self.n_acoustic_codebook;
        let steps = self.n_denoising_steps;
        let denom = (steps - 1) as f32;
        let mut x_t: Vec<f32> = noise
            .normal(n)
            .into_iter()
            .map(|v| v * self.sigma_max)
            .collect();
        // Euler integration over linspace(0, 1, steps); the loop runs
        // steps - 1 iterations.
        for step in 0..steps.saturating_sub(1) {
            let t_val = step as f32 / denom;
            let dt = (step + 1) as f32 / denom - t_val;
            let v_cond = self.predict_velocity(&x_t, t_val, llm)?;
            let v_uncond = self.predict_velocity(&x_t, t_val, llm_uncond)?;
            for (index, value) in x_t.iter_mut().enumerate() {
                let v = self.cfg_alpha * v_cond[index] + (1.0 - self.cfg_alpha) * v_uncond[index];
                *value += v * dt;
            }
        }
        let mut codes = Vec::with_capacity(1 + n);
        codes.push(self.predict_semantic(llm) as i32);
        for value in &x_t {
            let clamped = value.clamp(-1.0, 1.0);
            let scaled =
                round_ties_even((clamped + 1.0) * (self.acoustic_codebook_size - 1) as f32 / 2.0);
            let quantized = scaled.clamp(0.0, (self.acoustic_codebook_size - 1) as f32);
            codes.push(quantized as i32 + 2);
        }
        Ok(codes)
    }
}

/// Causal weight-normalized Conv1d with the pad modes the reference
/// decoder stages use.
struct WeightNormConv {
    weight: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    pad_mode: PadMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PadMode {
    /// `mx.pad(..., mode="edge")`: repeat the boundary samples.
    Replicate,
    /// The manual numpy-style reflect (edge not repeated), with the
    /// reference's short-input extra-pad guard.
    Reflect,
}

impl WeightNormConv {
    /// Loads the parametrization pair and folds the weight exactly like
    /// the reference forward: `g * v / ||v||` per output channel.
    fn load(
        weights: &Weights,
        base: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        pad_mode: PadMode,
    ) -> Result<Self> {
        let g = weights.raw_tensor(
            &format!("{base}.conv.parametrizations.weight.original0"),
            &[out_ch, 1, 1],
        )?;
        let v = weights.raw_tensor(
            &format!("{base}.conv.parametrizations.weight.original1"),
            &[out_ch, in_ch, kernel],
        )?;
        let mut weight = vec![0.0f32; v.len()];
        for oc in 0..out_ch {
            let mut norm = 0.0f32;
            for value in &v[oc * in_ch * kernel..(oc + 1) * in_ch * kernel] {
                norm += value * value;
            }
            let norm = (norm + 1e-12).sqrt();
            for (index, value) in v[oc * in_ch * kernel..(oc + 1) * in_ch * kernel]
                .iter()
                .enumerate()
            {
                weight[oc * in_ch * kernel + index] = g[oc] * value / norm;
            }
        }
        Ok(Self {
            weight,
            out_ch,
            in_ch,
            kernel,
            pad_mode,
        })
    }

    /// Causal convolution over channels-last rows `[seq, in_ch]`.
    fn conv(&self, x: &[f32], seq: usize, stride: usize) -> Result<Vec<f32>> {
        let padding_total = self.kernel - stride;
        let n_frames =
            (seq as f32 - self.kernel as f32 + padding_total as f32) / stride as f32 + 1.0;
        let target_length = (n_frames.ceil() as usize - 1) * stride + (self.kernel - padding_total);
        let extra = target_length - seq;
        let plane = to_channels_first(x, seq, self.in_ch);
        let padded = self.pad(&plane, seq, padding_total, extra)?;
        let out = ops::conv1d(
            &padded,
            &self.weight,
            None,
            self.in_ch,
            self.out_ch,
            self.kernel,
            stride,
            0,
            1,
            1,
        );
        let out_seq = out.len() / self.out_ch;
        Ok(to_channels_last(&out, self.out_ch, out_seq))
    }

    /// Causal transposed convolution: upsamples by `stride` and trims the
    /// output to `seq * stride` samples.
    ///
    /// For the transposed stages the checkpoint's first axis is the
    /// CONVOLUTION INPUT (`C_in`) and its second axis the output, so the
    /// folded tensor already matches the ops PyTorch layout
    /// `w [in, out, k]` with the axis roles swapped relative to the
    /// regular convolution.
    fn conv_transpose(&self, x: &[f32], seq: usize, stride: usize) -> Result<Vec<f32>> {
        let plane = to_channels_first(x, seq, self.out_ch);
        let out = ops::conv_transpose1d(
            &plane,
            &self.weight,
            None,
            self.out_ch,
            self.in_ch,
            self.kernel,
            stride,
            0,
            0,
            1,
        );
        let out_seq = out.len() / self.in_ch;
        let trimmed = seq * stride;
        let mut cropped = vec![0.0f32; self.in_ch * trimmed];
        for oc in 0..self.in_ch {
            let source = oc * out_seq..oc * out_seq + trimmed;
            cropped[oc * trimmed..(oc + 1) * trimmed].copy_from_slice(&out[source]);
        }
        Ok(to_channels_last(&cropped, self.in_ch, trimmed))
    }

    fn pad(&self, plane: &[f32], seq: usize, left: usize, right: usize) -> Result<Vec<f32>> {
        if left == 0 && right == 0 {
            return Ok(plane.to_vec());
        }
        let channels = self.in_ch;
        match self.pad_mode {
            PadMode::Replicate => {
                let padded_len = seq + left + right;
                let mut out = vec![0.0f32; channels * padded_len];
                for c in 0..channels {
                    let source = &plane[c * seq..(c + 1) * seq];
                    let dst = &mut out[c * padded_len..(c + 1) * padded_len];
                    for value in dst.iter_mut().take(left) {
                        *value = source[0];
                    }
                    dst[left..left + seq].copy_from_slice(source);
                    for t in 0..right {
                        dst[left + seq + t] = source[seq - 1];
                    }
                }
                Ok(out)
            }
            PadMode::Reflect => {
                // numpy-style reflect (the edge sample is not repeated),
                // with the reference's short-input extra-pad guard.
                let max_pad = left.max(right);
                let mut values = plane.to_vec();
                let mut length = seq;
                let mut extra_pad = 0usize;
                if length <= max_pad {
                    extra_pad = max_pad - length + 1;
                    let extended_len = length + extra_pad;
                    let mut extended = vec![0.0f32; channels * extended_len];
                    for c in 0..channels {
                        extended[c * extended_len..c * extended_len + length]
                            .copy_from_slice(&values[c * length..(c + 1) * length]);
                    }
                    values = extended;
                    length += extra_pad;
                }
                let total = length + left + right;
                let mut out = vec![0.0f32; channels * total];
                for c in 0..channels {
                    let source = &values[c * length..(c + 1) * length];
                    let dst = &mut out[c * total..(c + 1) * total];
                    for t in 0..left {
                        dst[t] = source[left - t];
                    }
                    dst[left..left + length].copy_from_slice(source);
                    for k in 0..right {
                        dst[left + length + k] = source[length - 2 - k];
                    }
                }
                if extra_pad > 0 {
                    let kept = total - extra_pad;
                    let mut trimmed = vec![0.0f32; channels * kept];
                    for c in 0..channels {
                        trimmed[c * kept..(c + 1) * kept]
                            .copy_from_slice(&out[c * total..c * total + kept]);
                    }
                    return Ok(trimmed);
                }
                Ok(out)
            }
        }
    }
}

fn to_channels_first(x: &[f32], seq: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..seq {
        for c in 0..channels {
            out[c * seq + t] = x[t * channels + c];
        }
    }
    out
}

fn to_channels_last(x: &[f32], channels: usize, seq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for c in 0..channels {
        for t in 0..seq {
            out[t * channels + c] = x[c * seq + t];
        }
    }
    out
}

/// ALiBi slopes, the reference `_get_alibi_slopes` (power-of-two head
/// counts direct; others interleave the next power's slopes).
fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    fn slopes_power_of_2(n: usize) -> Vec<f32> {
        let start = 2f64.powf(-(2f64.powf(-(n as f64).log2() + 3.0)));
        (0..n)
            .map(|i| (start * start.powi(i as i32)) as f32)
            .collect()
    }
    if (n_heads as f64).log2().fract() == 0.0 {
        slopes_power_of_2(n_heads)
    } else {
        let closest = 2f64.powf((n_heads as f64).log2().floor()) as usize;
        let mut slopes = slopes_power_of_2(closest);
        let extra = slopes_power_of_2(2 * closest);
        slopes.extend(extra.into_iter().step_by(2).take(n_heads - closest));
        slopes
    }
}

/// One ALiBi transformer layer of the codec decoder, with layer scale.
struct TokLayer {
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention: TokAttention,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    attention_scale: Option<Vec<f32>>,
    ffn_scale: Option<Vec<f32>>,
}

struct TokAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// RMSNorm over the FULL projected width, not per head.
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl TokAttention {
    fn forward(&self, x: &[f32], tokens: usize, slopes: &[f32], window: usize) -> Result<Vec<f32>> {
        let width = self.n_heads * self.head_dim;
        let mut q = self.q.forward(x, tokens);
        let mut k = self.k.forward(x, tokens);
        let v = self.v.forward(x, tokens);
        if let Some(norm) = &self.q_norm {
            norm.apply(&mut q, tokens);
        }
        if let Some(norm) = &self.k_norm {
            norm.apply(&mut k, tokens);
        }
        let rep = self.n_heads / self.n_kv_heads;
        let mut out = vec![0.0f32; tokens * width];
        for h in 0..self.n_heads {
            let kv = h / rep;
            let slope = slopes[h];
            for i in 0..tokens {
                let q_row = &q[i * width + h * self.head_dim..i * width + (h + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; tokens];
                for j in 0..tokens {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d]
                            * k[j * self.n_kv_heads * self.head_dim + kv * self.head_dim + d];
                    }
                    let dist = j as f32 - i as f32;
                    let mut score = dot * self.scale + slope * dist;
                    // Causal mask, then the sliding window.
                    if j > i {
                        score += -1e9;
                    }
                    if window > 0 && (j as i64) < i as i64 - window as i64 {
                        score += -1e9;
                    }
                    scores[j] = score;
                    max = max.max(score);
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum;
                for j in 0..tokens {
                    let weight = scores[j] * inv;
                    let out_row = &mut out
                        [i * width + h * self.head_dim..i * width + (h + 1) * self.head_dim];
                    for d in 0..self.head_dim {
                        out_row[d] += weight
                            * v[j * self.n_kv_heads * self.head_dim + kv * self.head_dim + d];
                    }
                }
            }
        }
        Ok(self.o.forward(&out, tokens))
    }
}

impl TokLayer {
    fn load(weights: &Weights, config: &ModelConfig, stage: usize, index: usize) -> Result<Self> {
        let prefix = format!("audio_tokenizer.decoder_blocks.{stage}.layers.{index}.");
        let dim = config.tokenizer_dim;
        let head_dim = config.tokenizer_head_dim;
        let q_width = config.tokenizer_n_heads * head_dim;
        let kv_width = config.tokenizer_n_kv_heads * head_dim;
        let (q_norm, k_norm) = (
            Some(RmsNorm::new(
                weights.tensor(&format!("{prefix}attention.q_norm"), &[q_width])?,
                1e-6,
            )),
            Some(RmsNorm::new(
                weights.tensor(&format!("{prefix}attention.k_norm"), &[kv_width])?,
                1e-6,
            )),
        );
        Ok(Self {
            attention_norm: RmsNorm::new(
                weights.tensor(&format!("{prefix}attention_norm"), &[dim])?,
                config.tokenizer_norm_eps,
            ),
            ffn_norm: RmsNorm::new(
                weights.tensor(&format!("{prefix}ffn_norm"), &[dim])?,
                config.tokenizer_norm_eps,
            ),
            attention: TokAttention {
                q: weights.linear(&format!("{prefix}attention.wq"), dim, q_width, Bias::None)?,
                k: weights.linear(&format!("{prefix}attention.wk"), dim, kv_width, Bias::None)?,
                v: weights.linear(&format!("{prefix}attention.wv"), dim, kv_width, Bias::None)?,
                o: weights.linear(&format!("{prefix}attention.wo"), q_width, dim, Bias::None)?,
                q_norm,
                k_norm,
                n_heads: config.tokenizer_n_heads,
                n_kv_heads: config.tokenizer_n_kv_heads,
                head_dim,
                scale: (head_dim as f64).powf(-0.5) as f32,
            },
            w1: weights.linear(
                &format!("{prefix}feed_forward.w1"),
                dim,
                config.tokenizer_hidden_dim,
                Bias::None,
            )?,
            w2: weights.linear(
                &format!("{prefix}feed_forward.w2"),
                config.tokenizer_hidden_dim,
                dim,
                Bias::None,
            )?,
            w3: weights.linear(
                &format!("{prefix}feed_forward.w3"),
                dim,
                config.tokenizer_hidden_dim,
                Bias::None,
            )?,
            attention_scale: Some(weights.raw_tensor(&format!("{prefix}attention_scale"), &[dim])?),
            ffn_scale: Some(weights.raw_tensor(&format!("{prefix}ffn_scale"), &[dim])?),
        })
    }

    fn forward(&self, x: &mut [f32], tokens: usize, slopes: &[f32], window: usize) -> Result<()> {
        let dim = self.w2.output;
        let mut attended = x.to_vec();
        self.attention_norm.apply(&mut attended, tokens);
        let mut attention = self.attention.forward(&attended, tokens, slopes, window)?;
        if let Some(scale) = &self.attention_scale {
            apply_scale(&mut attention, scale, tokens, dim);
        }
        for (value, add) in x.iter_mut().zip(&attention) {
            *value += *add;
        }
        let mut fed = x.to_vec();
        self.ffn_norm.apply(&mut fed, tokens);
        let gate = self.w1.forward(&fed, tokens);
        let up = self.w3.forward(&fed, tokens);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        let mut down = self.w2.forward(&fused, tokens);
        if let Some(scale) = &self.ffn_scale {
            apply_scale(&mut down, scale, tokens, dim);
        }
        for (value, add) in x.iter_mut().zip(&down) {
            *value += *add;
        }
        Ok(())
    }
}

fn apply_scale(values: &mut [f32], scale: &[f32], tokens: usize, dim: usize) {
    for t in 0..tokens {
        for d in 0..dim {
            values[t * dim + d] *= scale[d];
        }
    }
}

/// One per-stage decode trace entry: `(stage name, values, rows, width)`.
pub(crate) type StageTrace = (&'static str, Vec<f32>, usize, usize);

/// One decoder stage: a weight-normed conv (transposed when the stride
/// upsamples) followed by a sliding-window ALiBi transformer block.
struct DecoderStage {
    conv: WeightNormConv,
    stride: usize,
    layers: Vec<TokLayer>,
    window: usize,
}

/// The codec decoder: EMA semantic codebook + FSQ acoustic quantizer,
/// alternating conv/transformer stages, and the patch output projection.
/// Strides `[1, 2, 2, 2]` upsample 8x; each frame then maps to
/// `8 * patch_size` samples (1920 at the real 240-sample patch).
struct TokenizerDecoder {
    semantic_usage: Vec<f32>,
    semantic_sum: Vec<f32>,
    semantic_codebook_size: usize,
    semantic_dim: usize,
    acoustic_codebook_size: usize,
    acoustic_dim: usize,
    alibi: Vec<f32>,
    stages: Vec<DecoderStage>,
    output_proj: WeightNormConv,
    dim: usize,
}

impl TokenizerDecoder {
    fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        let codebook_dim = config.tokenizer_semantic_dim + config.tokenizer_acoustic_dim;
        let strides = &config.tokenizer_decoder_convs_strides;
        let kernels = &config.tokenizer_decoder_convs_kernels;
        let lengths = &config.tokenizer_decoder_transformer_lengths;
        if strides.len() != kernels.len() || strides.len() != lengths.len() {
            return Err(bad_config(
                "decoder_convs_strides_str",
                "kernels, strides, and transformer lengths must have the same stage count",
            ));
        }
        if strides.is_empty() {
            return Err(bad_config(
                "decoder_convs_strides_str",
                "must list at least one decoder stage",
            ));
        }
        // The decoder reverses the encoder windows: [2, 4, 8, 16].
        let windows = [2usize, 4, 8, 16];
        let mut stages = Vec::with_capacity(strides.len());
        for (stage, (&stride, &kernel)) in strides.iter().zip(kernels).enumerate() {
            let in_ch = if stage == 0 {
                codebook_dim
            } else {
                config.tokenizer_dim
            };
            let conv = WeightNormConv::load(
                weights,
                // The reference interleaves convs and transformer blocks
                // in one flat list: conv stage s sits at 2s, its
                // transformer block at 2s + 1.
                &format!("audio_tokenizer.decoder_blocks.{}", 2 * stage),
                config.tokenizer_dim,
                in_ch,
                kernel,
                PadMode::Replicate,
            )?;
            let layers = (0..lengths[stage])
                .map(|index| TokLayer::load(weights, config, 2 * stage + 1, index))
                .collect::<Result<Vec<_>>>()?;
            stages.push(DecoderStage {
                conv,
                stride,
                layers,
                window: windows.get(stage).copied().unwrap_or(16),
            });
        }
        let output_proj = WeightNormConv::load(
            weights,
            "audio_tokenizer.output_proj",
            config.tokenizer_patch_size,
            config.tokenizer_dim,
            config.tokenizer_patch_proj_kernel_size,
            PadMode::Reflect,
        )?;
        Ok(Self {
            semantic_usage: weights.raw_tensor(
                "audio_tokenizer.quantizer.semantic_codebook.cluster_usage",
                &[config.semantic_codebook_size],
            )?,
            semantic_sum: weights.raw_tensor(
                "audio_tokenizer.quantizer.semantic_codebook.embedding_sum",
                &[config.semantic_codebook_size, config.tokenizer_semantic_dim],
            )?,
            semantic_codebook_size: config.semantic_codebook_size,
            semantic_dim: config.tokenizer_semantic_dim,
            acoustic_codebook_size: config.acoustic_codebook_size,
            acoustic_dim: config.tokenizer_acoustic_dim,
            alibi: alibi_slopes(config.tokenizer_n_heads),
            stages,
            output_proj,
            dim: config.tokenizer_dim,
        })
    }

    fn codebook_dim(&self) -> usize {
        self.semantic_dim + self.acoustic_dim
    }

    /// Decodes a `[frames * num_codebooks]` code grid (with the +2
    /// special-token offsets) to the waveform samples. Returns the
    /// per-stage hidden states alongside when `trace` requests it.
    fn decode(
        &self,
        codes: &[i32],
        frames: usize,
        num_codebooks: usize,
        mut trace: Option<&mut Vec<StageTrace>>,
    ) -> Result<Vec<f32>> {
        let codebook_dim = self.codebook_dim();
        if codes.len() != frames * num_codebooks {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {frames} x {num_codebooks} codes, got {}",
                    codes.len()
                ),
            });
        }
        // Quantizer decode, channels-last [frames, codebook_dim].
        let mut x = vec![0.0f32; frames * codebook_dim];
        for t in 0..frames {
            let semantic = codes[t * num_codebooks] - 2;
            if semantic < 0 || semantic as usize >= self.semantic_codebook_size {
                return Err(SpeechError::Input {
                    why: format!("semantic code {semantic} outside the codebook"),
                });
            }
            let semantic = semantic as usize;
            let usage = self.semantic_usage[semantic].max(1e-8);
            for d in 0..self.semantic_dim {
                x[t * codebook_dim + d] =
                    self.semantic_sum[semantic * self.semantic_dim + d] / usage;
            }
            for c in 0..self.acoustic_dim {
                let code = codes[t * num_codebooks + 1 + c] - 2;
                if code < 0 || code as usize >= self.acoustic_codebook_size {
                    return Err(SpeechError::Input {
                        why: format!("acoustic code {code} outside the FSQ range"),
                    });
                }
                x[t * codebook_dim + self.semantic_dim + c] =
                    2.0 * code as f32 / (self.acoustic_codebook_size - 1) as f32 - 1.0;
            }
        }
        if let Some(sink) = trace.as_deref_mut() {
            sink.push(("quantizer", x.clone(), frames, codebook_dim));
        }
        for stage in &self.stages {
            let seq = x.len() / stage.conv.in_ch;
            x = if stage.stride > 1 {
                stage.conv.conv_transpose(&x, seq, stage.stride)?
            } else {
                stage.conv.conv(&x, seq, stage.stride)?
            };
            let tokens = x.len() / self.dim;
            if let Some(sink) = trace.as_deref_mut() {
                sink.push(("conv", x.clone(), tokens, self.dim));
            }
            for layer in &stage.layers {
                layer.forward(&mut x, tokens, &self.alibi, stage.window)?;
            }
            if let Some(sink) = trace.as_deref_mut() {
                sink.push(("transformer", x.clone(), tokens, self.dim));
            }
        }
        let tokens = x.len() / self.dim;
        // (B, T_up, patch) reshaped to (B, -1): the row-major flatten.
        self.output_proj.conv(&x, tokens, 1)
    }
}

/// One loaded voice conditioning table (`voice_embedding/<name>.
/// safetensors`, key `embedding`).
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    pub rows: usize,
    pub values: Vec<f32>,
}

/// One generation result.
#[derive(Debug, Clone)]
pub struct Generated {
    /// Generated frame count.
    pub frames: usize,
    /// The per-frame codes with the +2 special-token offsets,
    /// `[frames * num_codebooks]` (semantic code first).
    pub codes: Vec<i32>,
    /// True when the semantic end token (1) ended the run instead of
    /// `max_tokens`.
    pub stopped: bool,
    /// The decoded 24 kHz waveform samples.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// `generate` parameters. The reference accepts `temperature`, `top_k`,
/// and `top_p` but never uses them (generation is deterministic given
/// the flow noise), so none appear here.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum generated frames.
    pub max_tokens: usize,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self { max_tokens: 4096 }
    }
}

/// The loaded Voxtral TTS model.
pub struct Model {
    pub config: ModelConfig,
    language_model: Backbone,
    /// The shared audio codebook embedding table.
    audio_embeddings: Vec<f32>,
    acoustic: AcousticHead,
    tokenizer: TokenizerDecoder,
    voice_files: Vec<(String, PathBuf)>,
    loaded_voices: Vec<Option<VoiceEmbedding>>,
    caches: crate::lm::LayerCaches,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors, the
    /// optional `voice_embedding/` folder beside them).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let config = ModelConfig::from_json(&config_value)?;
        let weights = Weights::open(dir)?;
        // The raw consolidated Mistral layout needs the engine's names
        // remapped; the sanitized HF layout loads as-is.
        let weights = if weights.contains("embed_tokens") {
            weights
        } else if weights.contains_absolute("tok_embeddings.weight") {
            weights.with_name_map(mistral_name_map())
        } else {
            return Err(SpeechError::Unsupported {
                why: "checkpoint has neither embed_tokens nor tok_embeddings tensors; not a voxtral_tts layout".into(),
            });
        };
        let language_model = Backbone::load(&weights, config.dense_config())?;
        let audio_rows = config.audio_embedding_size();
        let dim = config.dim;
        let audio_embeddings = if weights.contains("audio_codebook_embeddings.embeddings") {
            weights.embedding("audio_codebook_embeddings.embeddings", audio_rows, dim)?
        } else if weights
            .contains_absolute("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight")
        {
            weights.raw_tensor(
                "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
                &[audio_rows, dim],
            )?
        } else {
            return Err(SpeechError::Tensor {
                name: "audio_codebook_embeddings.embeddings.weight".into(),
                why: "tensor is missing".into(),
            });
        };
        let acoustic = AcousticHead::load(&weights, &config)?;
        let tokenizer = TokenizerDecoder::load(&weights, &config)?;
        let mut voice_files = Vec::new();
        let voice_dir = dir.join("voice_embedding");
        if voice_dir.is_dir() {
            let mut entries: Vec<PathBuf> = fs_entries(&voice_dir)?;
            entries.sort();
            for path in entries {
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_owned();
                    voice_files.push((name, path));
                }
            }
        }
        let loaded_voices = vec![None; voice_files.len()];
        let caches = crate::lm::make_prompt_cache(config.n_layers);
        Ok(Self {
            config,
            language_model,
            audio_embeddings,
            acoustic,
            tokenizer,
            voice_files,
            loaded_voices,
            caches,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn num_codebooks(&self) -> usize {
        self.config.num_codebooks()
    }

    /// Lists the voice names discovered under `voice_embedding/`.
    pub fn voices(&self) -> Vec<&str> {
        self.voice_files
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    fn voice(&mut self, name: &str) -> Result<VoiceEmbedding> {
        let index = self
            .voice_files
            .iter()
            .position(|(candidate, _)| candidate == name)
            .ok_or_else(|| SpeechError::Input {
                why: format!(
                    "unknown voice {name:?}; available: {}",
                    self.voices().join(", ")
                ),
            })?;
        if let Some(loaded) = &self.loaded_voices[index] {
            return Ok(loaded.clone());
        }
        let file = SafetensorsFile::open(&self.voice_files[index].1).map_err(SpeechError::from)?;
        let (rows, values) = if file.contains_tensor("embedding") {
            let descriptor = file
                .descriptor("embedding")
                .ok_or_else(|| SpeechError::Tensor {
                    name: "embedding".into(),
                    why: "descriptor is missing".into(),
                })?;
            let values = file.load_as_f32("embedding")?;
            let width = *descriptor.shape.last().ok_or_else(|| SpeechError::Tensor {
                name: "embedding".into(),
                why: "empty shape".into(),
            })?;
            let rows = values.len() / width;
            (rows, values)
        } else {
            return Err(SpeechError::Tensor {
                name: "embedding".into(),
                why: "voice safetensors must carry an 'embedding' tensor; the reference's first-tensor fallback is not ported".into(),
            });
        };
        let voice = VoiceEmbedding { rows, values };
        self.loaded_voices[index] = Some(voice.clone());
        Ok(voice)
    }

    /// Builds the prompt embeddings: the text table rows with the
    /// `[AUDIO]`-token positions replaced by the voice conditioning rows
    /// (`_build_input_embeddings`).
    pub fn build_prompt_embeddings(
        &self,
        prompt_ids: &[i32],
        voice: Option<&VoiceEmbedding>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.dim;
        let mut embeddings = crate::lm::engine::lookup_embeddings(
            self.language_model.embed_table(),
            self.config.vocab_size,
            dim,
            prompt_ids,
        )?;
        let Some(voice) = voice else {
            return Ok(embeddings);
        };
        if voice.values.len() != voice.rows * dim {
            return Err(SpeechError::Input {
                why: format!(
                    "voice embedding holds {} values, not a multiple of {dim}",
                    voice.values.len()
                ),
            });
        }
        let mut count = 0usize;
        for (position, &id) in prompt_ids.iter().enumerate() {
            if id != self.config.audio_token_id {
                continue;
            }
            // cumsum(mask) - 1, clipped to the voice table.
            let index = count.min(voice.rows.saturating_sub(1));
            count += 1;
            let dst = &mut embeddings[position * dim..(position + 1) * dim];
            dst.copy_from_slice(&voice.values[index * dim..(index + 1) * dim]);
        }
        Ok(embeddings)
    }

    /// The reference `generate` for one speech request. `prompt_ids` is
    /// the caller-built speech-request token sequence (text ids, the
    /// voice's `[AUDIO]` run, and the trailing `[BEGIN_AUDIO]` in the
    /// reference layout); `voice` selects a `voice_embedding/` table, and
    /// `None` runs without voice conditioning (the reference silently
    /// degrades when the file is missing; the port refuses).
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        voice: Option<&str>,
        params: &GenerateParams,
        noise: &mut dyn FlowNoise,
    ) -> Result<Generated> {
        if params.max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "max_tokens must be at least 1".into(),
            });
        }
        if prompt_ids.is_empty() {
            return Err(SpeechError::Input {
                why: "prompt_ids is empty".into(),
            });
        }
        let dim = self.config.dim;
        let voice_embedding = match voice {
            Some(name) => Some(self.voice(name)?),
            None => None,
        };
        let embeddings = self.build_prompt_embeddings(prompt_ids, voice_embedding.as_ref())?;
        for cache in &mut self.caches {
            cache.reset();
        }
        // Prefill: the prompt ends with [BEGIN_AUDIO], so its last hidden
        // state already predicts the first audio frame.
        let mut hidden = self
            .language_model
            .forward_hidden(Input::Embeddings(&embeddings), Some(&mut self.caches))?;
        let llm_uncond = vec![0.0f32; dim];
        let mut codes: Vec<i32> = Vec::new();
        let mut frames = 0usize;
        let mut stopped = false;
        for _ in 0..params.max_tokens {
            let h = &hidden[hidden.len() - dim..];
            let frame = self.acoustic.decode_one_frame(h, &llm_uncond, noise)?;
            if frame[0] <= 1 {
                stopped = true;
                break;
            }
            codes.extend_from_slice(&frame);
            frames += 1;
            // Sum the per-codebook embedding rows into the next LM input.
            // Each codebook has its own offset in the shared table.
            let mut next = vec![0.0f32; dim];
            for (codebook, &code) in frame.iter().enumerate() {
                let code = usize::try_from(code).map_err(|_| SpeechError::Input {
                    why: format!("negative code {code}"),
                })?;
                let offset = match codebook {
                    0 => 0,
                    _ => {
                        self.config.semantic_codebook_size
                            + 2
                            + (codebook - 1) * (self.config.acoustic_codebook_size + 2)
                    }
                };
                let global = offset + code;
                if code >= self.codebook_row_width(codebook)
                    || global >= self.audio_embeddings.len() / dim
                {
                    return Err(SpeechError::Input {
                        why: format!("code {code} outside codebook {codebook} range"),
                    });
                }
                let row = &self.audio_embeddings[global * dim..(global + 1) * dim];
                for (value, add) in next.iter_mut().zip(row) {
                    *value += *add;
                }
            }
            hidden = self
                .language_model
                .forward_hidden(Input::Embeddings(&next), Some(&mut self.caches))?;
        }
        if codes.is_empty() {
            return Err(SpeechError::Input {
                why: "no audio frames generated".into(),
            });
        }
        let samples = self
            .tokenizer
            .decode(&codes, frames, self.num_codebooks(), None)?;
        Ok(Generated {
            frames,
            codes,
            stopped,
            samples,
            sample_rate: self.sample_rate(),
        })
    }

    fn codebook_row_width(&self, codebook: usize) -> usize {
        if codebook == 0 {
            self.config.semantic_codebook_size + 2
        } else {
            self.config.acoustic_codebook_size + 2
        }
    }

    /// Decodes a code grid through the codec decoder without running the
    /// LM (the standalone `audio_tokenizer.decode` path).
    pub fn decode_codes(&self, codes: &[i32], frames: usize) -> Result<Vec<f32>> {
        self.tokenizer
            .decode(codes, frames, self.num_codebooks(), None)
    }

    /// [`Model::decode_codes`] with a per-stage trace: `(stage, values,
    /// rows, width)` in decode order (`quantizer`, then `conv` and
    /// `transformer` per stage).
    pub fn decode_codes_traced(
        &self,
        codes: &[i32],
        frames: usize,
    ) -> Result<(Vec<f32>, Vec<StageTrace>)> {
        let mut sink = Vec::new();
        let samples =
            self.tokenizer
                .decode(codes, frames, self.num_codebooks(), Some(&mut sink))?;
        Ok((samples, sink))
    }
}

fn fs_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|error| SpeechError::Input {
        why: format!("cannot read {}: {error}", dir.display()),
    })? {
        let entry = entry.map_err(|error| SpeechError::Input {
            why: format!("cannot read directory entry: {error}"),
        })?;
        entries.push(entry.path());
    }
    Ok(entries)
}
