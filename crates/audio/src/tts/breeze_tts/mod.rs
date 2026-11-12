//! Breeze TTS 2 family (BreezeBlue).
//!
//! Port of `mlx_audio/tts/models/breeze_tts/` (breeze_tts.py, config.py)
//! at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/breeze_tts):
//! a qwen3-style autoregressive backbone over a summed per-codebook audio
//! embedding predicts codebook zero of the Qwen3-TTS codec vocabulary, a
//! llama depth decoder with per-codebook output projections fills
//! codebooks 1..K-1, and a bidirectional T5Gemma2 text encoder (through a
//! projection) embeds the prompt text. Voice design, cloning, and
//! direction ride one prompt layout; classifier-free guidance combines a
//! conditional and an instruct-free pass. See [README.md](README.md) for
//! the verification evidence and open items.
//!
//! Scope of this port: batch size 1, non-streaming, and prompt
//! construction from pre-tokenized ids (the reference loads the bundled
//! fast tokenizer; the exact `{speaker}` / `<ins_bos>..` string layouts
//! are caller-owned). Generation ends at the codebook frames: the
//! waveform decode runs through the Qwen3-TTS speech tokenizer, which has
//! no Rust port yet (see the family README).

use serde_json::Value;

use crate::lm;
use crate::lm::cache::KvCache;
use crate::lm::engine::{Attention, Mlp};
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Default speaker tag when no voice is given (`_speaker`).
pub const DEFAULT_SPEAKER: &str = "S0";
/// Default sample rate (`ModelConfig.sample_rate` dataclass default).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// The wrapper `ModelConfig` (breeze config.py), fully resolved.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Qwen3 backbone configuration. The nested `backbone_config` wins
    /// over the wrapper-level fields (`backbone_value`).
    pub backbone: lm::DenseConfig,
    /// Wrapper audio geometry: codebooks and per-codebook vocabulary.
    pub num_codebooks: usize,
    /// Wrapper audio vocabulary per codebook, excluding the backbone EOS
    /// class (the wrapper lm head carries `vocab_size + 1`).
    pub vocab_size: usize,
    /// Embedding width of the audio wrapper; the backbone hidden size
    /// when the config omits it (`wrapper_audio_embed_size`).
    pub audio_embed_size: usize,
    pub text_vocab_size: usize,
    /// T5Gemma2 text encoder configuration (normalized).
    pub text_encoder: TextEncoderConfig,
    /// Depth decoder configuration (resolved llama flavors).
    pub depth_decoder: DepthDecoderConfig,
    /// First codec id outside the codec codebook; ids in
    /// `[codec_vocab_size, vocab_size)` are reserved and masked.
    pub codec_vocab_size: usize,
    pub codebook_pad_token_id: i32,
    pub codebook_eos_token_id: i32,
    pub sample_rate: u32,
}

/// Grouped-query attention needs `heads` to be an exact positive multiple
/// of `kv_heads`. The attention loop divides by the ratio and indexes the
/// KV table with it, so zero or a non-divisor panics or reads the wrong head.
fn check_gqa(stack: &str, heads: usize, kv_heads: usize) -> Result<()> {
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return Err(SpeechError::BadConfig {
            field: format!("{stack}.num_key_value_heads"),
            why: format!(
                "num_attention_heads {heads} must be a positive multiple of \
                 num_key_value_heads {kv_heads}"
            ),
        });
    }
    Ok(())
}

impl ModelConfig {
    /// `ModelConfig.from_dict` over a checkpoint `config.json` dict.
    pub fn from_json(value: &Value) -> Result<Self> {
        // Alias normalization: explicit audio_* names are the unambiguous
        // wrapper fields and win over num_codebooks/vocab_size.
        let audio_num_codebooks = value.get("audio_num_codebooks").and_then(Value::as_u64);
        let audio_vocab_size = value.get("audio_vocab_size").and_then(Value::as_u64);
        let num_codebooks = audio_num_codebooks
            .or_else(|| value.get("num_codebooks").and_then(Value::as_u64))
            .unwrap_or(16) as usize;
        let vocab_size = audio_vocab_size
            .or_else(|| value.get("vocab_size").and_then(Value::as_u64))
            .unwrap_or(2051) as usize;

        let codec = value.get("codec_config").cloned().unwrap_or(Value::Null);
        let sample_rate = codec
            .get("sampling_rate")
            .and_then(Value::as_u64)
            .or_else(|| codec.get("output_sample_rate").and_then(Value::as_u64))
            .or_else(|| value.get("sample_rate").and_then(Value::as_u64))
            .unwrap_or(24000) as u32;

        let codec_vocab_size = match codec.get("codebook_size").and_then(Value::as_u64) {
            Some(size) => {
                let size = size as usize;
                if size == 0 || size > vocab_size {
                    return Err(SpeechError::BadConfig {
                        field: "codec codebook_size".into(),
                        why: format!(
                            "must be between 1 and the wrapper vocab_size ({vocab_size}), got {size}"
                        ),
                    });
                }
                size
            }
            None => vocab_size
                .checked_sub(3)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "codec codebook_size".into(),
                    why: format!("wrapper vocab_size {vocab_size} leaves no codec entries"),
                })?,
        };

        let backbone_hidden = backbone_field(value, "hidden_size")
            .and_then(Value::as_u64)
            .unwrap_or(2048) as usize;
        let audio_embed_size = match value.get("audio_embed_size").and_then(Value::as_u64) {
            Some(size) => size as usize,
            None => backbone_hidden,
        };

        let config = ModelConfig {
            backbone: breeze_backbone_config(value, backbone_hidden)?,
            num_codebooks,
            vocab_size,
            audio_embed_size,
            text_vocab_size: value
                .get("text_vocab_size")
                .and_then(Value::as_u64)
                .unwrap_or(262158) as usize,
            text_encoder: TextEncoderConfig::normalize(
                value.get("text_encoder_config").unwrap_or(&Value::Null),
            )?,
            depth_decoder: DepthDecoderConfig::resolve(
                value.get("depth_decoder_config").unwrap_or(&Value::Null),
                num_codebooks,
                vocab_size,
                audio_embed_size,
                backbone_hidden,
            )?,
            codec_vocab_size,
            codebook_pad_token_id: value
                .get("codebook_pad_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(2050) as i32,
            codebook_eos_token_id: value
                .get("codebook_eos_token_id")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32,
            sample_rate,
        };
        check_gqa(
            "backbone",
            config.backbone.num_attention_heads,
            config.backbone.num_key_value_heads,
        )?;
        check_gqa(
            "text_encoder_config",
            config.text_encoder.num_attention_heads,
            config.text_encoder.num_key_value_heads,
        )?;
        check_gqa(
            "depth_decoder_config",
            config.depth_decoder.num_attention_heads,
            config.depth_decoder.num_key_value_heads,
        )?;
        // depth_tokens runs `num_codebooks - 1` head steps, so 0 would
        // underflow there.
        if config.num_codebooks == 0 {
            return Err(SpeechError::BadConfig {
                field: "num_codebooks".into(),
                why: "num_codebooks must be > 0".into(),
            });
        }
        Ok(config)
    }
}

/// `ModelConfig.backbone_value` without the default: the nested
/// `backbone_config` entry when present, else the wrapper-level field.
fn backbone_field<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    if let Some(backbone) = value.get("backbone_config") {
        if let Some(found) = backbone.get(name) {
            if !found.is_null() {
                return Some(found);
            }
        }
    }
    value.get(name).filter(|found| !found.is_null())
}

/// Builds the qwen3 `DenseConfig` for the backbone the way `_Backbone`
/// resolves its qwen3.ModelArgs: every field through `backbone_value`
/// with the wrapper field (or dataclass default) as fallback.
fn breeze_backbone_config(value: &Value, hidden: usize) -> Result<lm::DenseConfig> {
    let uint = |name: &str, default: u64| -> usize {
        backbone_field(value, name)
            .and_then(Value::as_u64)
            .unwrap_or(default) as usize
    };
    let f = |name: &str, default: f64| -> f64 {
        backbone_field(value, name)
            .and_then(Value::as_f64)
            .unwrap_or(default)
    };
    let rope_scaling = backbone_field(value, "rope_scaling")
        .filter(|found| !found.is_null())
        .cloned();
    let rms_norm_eps = f("rms_norm_eps", 1e-5);
    if rms_norm_eps <= 0.0 {
        return Err(SpeechError::BadConfig {
            field: "rms_norm_eps".into(),
            why: "must be a positive number".into(),
        });
    }
    Ok(lm::DenseConfig {
        model_type: "qwen3".into(),
        hidden_size: hidden,
        num_hidden_layers: uint("num_hidden_layers", 28),
        intermediate_size: uint("intermediate_size", 6144),
        num_attention_heads: uint("num_attention_heads", 16),
        num_key_value_heads: uint("num_key_value_heads", 8),
        head_dim: uint("head_dim", 128),
        rms_norm_eps: rms_norm_eps as f32,
        // Metadata only: the backbone embedding is the audio wrapper
        // table, and no backbone head exists (the wrapper lm_head runs).
        vocab_size: uint("vocab_size", 2051),
        qk_norms: true,
        attention_bias: false,
        mlp_bias: false,
        rope_theta: f("rope_theta", 500000.0) as f32,
        rope_traditional: false,
        rope_scaling,
        max_position_embeddings: Some(uint("max_position_embeddings", 2048)),
        tie_word_embeddings: false,
        sliding_window: None,
    })
}

/// The normalized T5Gemma2 text encoder configuration
/// (`_TextEncoder._normalize_config`).
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    /// Attention scale input (`query_pre_attn_scalar ** -0.5`).
    pub query_pre_attn_scalar: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub eoi_token_index: usize,
    /// One entry per layer: true for the bidirectional full-attention
    /// layers, false for the windowed ones.
    pub layer_types: Vec<bool>,
    /// RoPE base per layer class (sliding, full); 10000 / 1000000 on the
    /// upstream defaults, overridable through `rope_parameters`.
    pub rope_theta_sliding: f64,
    pub rope_theta_full: f64,
    /// Linear-scaling configs per class when a class's rope dict carries
    /// `rope_type: linear`; otherwise unscaled default rope.
    pub rope_scaling_sliding: Option<Value>,
    pub rope_scaling_full: Option<Value>,
}

impl TextEncoderConfig {
    fn normalize(value: &Value) -> Result<Self> {
        let u = |key: &str, default: u64| -> usize {
            value.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
        };
        let config = TextEncoderConfig {
            vocab_size: u("vocab_size", 262208),
            hidden_size: u("hidden_size", 2304),
            intermediate_size: u("intermediate_size", 9216),
            num_hidden_layers: u("num_hidden_layers", 26),
            num_attention_heads: u("num_attention_heads", 8),
            num_key_value_heads: u("num_key_value_heads", 4),
            head_dim: u("head_dim", 256),
            rms_norm_eps: value
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            query_pre_attn_scalar: value
                .get("query_pre_attn_scalar")
                .and_then(Value::as_f64)
                .unwrap_or(256.0),
            max_position_embeddings: u("max_position_embeddings", 131072),
            sliding_window: u("sliding_window", 4096),
            eoi_token_index: u("eoi_token_index", 256000),
            layer_types: Vec::new(),
            rope_theta_sliding: 10_000.0,
            rope_theta_full: 1_000_000.0,
            rope_scaling_sliding: None,
            rope_scaling_full: None,
        };
        Self::finish(config, value)
    }

    /// Applies the layer-type pattern and the per-class RoPE dicts
    /// (`_sliding_window_pattern` derivation plus the `rope_parameters`
    /// / `rope_scaling` merge) to a normalized config.
    fn finish(mut config: TextEncoderConfig, value: &Value) -> Result<Self> {
        let provided = value.get("layer_types").and_then(Value::as_array);
        let mut explicit = false;
        if let Some(entries) = provided {
            if !entries.is_empty() {
                explicit = true;
                config.layer_types = entries
                    .iter()
                    .map(|entry| match entry.as_str() {
                        Some("full_attention") => Ok(true),
                        Some("sliding_attention") => Ok(false),
                        other => Err(SpeechError::BadConfig {
                            field: "layer_types".into(),
                            why: format!("unsupported T5Gemma2 attention layer {other:?}"),
                        }),
                    })
                    .collect::<Result<Vec<bool>>>()?;
                if config.layer_types.len() != config.num_hidden_layers {
                    return Err(SpeechError::BadConfig {
                        field: "layer_types".into(),
                        why: format!(
                            "has {} entries for {} layers",
                            config.layer_types.len(),
                            config.num_hidden_layers
                        ),
                    });
                }
            }
        }
        if !explicit {
            // The pattern key precedence mirrors the reference double pop:
            // `_sliding_window_pattern` wins over `sliding_window_pattern`,
            // which wins over 6.
            let pattern = value
                .get("_sliding_window_pattern")
                .and_then(Value::as_u64)
                .or_else(|| value.get("sliding_window_pattern").and_then(Value::as_u64))
                .unwrap_or(6);
            if pattern == 0 {
                return Err(SpeechError::BadConfig {
                    field: "sliding_window_pattern".into(),
                    why: "must be a positive integer".into(),
                });
            }
            config.layer_types = (0..config.num_hidden_layers)
                .map(|index| (index + 1) % pattern as usize == 0)
                .collect();
        }

        // Per-class rope dicts: the upstream defaults, overridden by the
        // per-class `rope_parameters` mapping, with the generic
        // `rope_scaling` folding into the full-attention stream when that
        // class has no explicit entry (the reference dict update).
        let rope_parameters = value.get("rope_parameters");
        let has_full_entry = rope_parameters
            .and_then(|r| r.get("full_attention"))
            .is_some_and(|entry| !entry.is_null());
        let mut full = serde_json::json!({
            "rope_type": "default",
            "rope_theta": config.rope_theta_full,
        });
        let mut sliding = serde_json::json!({
            "rope_type": "default",
            "rope_theta": config.rope_theta_sliding,
        });
        if let Some(provided_scaling) = value.get("rope_scaling").filter(|s| !s.is_null()) {
            if !has_full_entry {
                if let Some(map) = provided_scaling.as_object() {
                    if let Some(target) = full.as_object_mut() {
                        for (key, entry) in map {
                            target.insert(key.clone(), entry.clone());
                        }
                    }
                }
            }
        }
        for class in ["sliding_attention", "full_attention"] {
            if let Some(entry) = rope_parameters.and_then(|r| r.get(class)) {
                if entry.is_null() {
                    continue;
                }
                let target = if class == "sliding_attention" {
                    &mut sliding
                } else {
                    &mut full
                };
                if let Some(map) = entry.as_object() {
                    if let Some(target) = target.as_object_mut() {
                        for (key, value) in map {
                            target.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }
        for (dict, theta, scaling) in [
            (
                &sliding,
                &mut config.rope_theta_sliding,
                &mut config.rope_scaling_sliding,
            ),
            (
                &full,
                &mut config.rope_theta_full,
                &mut config.rope_scaling_full,
            ),
        ] {
            if let Some(value) = dict.get("rope_theta").and_then(Value::as_f64) {
                *theta = value;
            }
            let rope_type = dict
                .get("rope_type")
                .and_then(Value::as_str)
                .unwrap_or("default");
            if rope_type == "linear" {
                *scaling = Some(dict.clone());
            } else if rope_type != "default" {
                return Err(SpeechError::Unsupported {
                    why: format!("T5Gemma2 rope_type {rope_type} is not ported"),
                });
            }
        }
        Ok(config)
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.layer_types[layer]
    }
}

/// The resolved depth decoder configuration (`_DepthModel` fields).
#[derive(Debug, Clone)]
pub struct DepthDecoderConfig {
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub audio_embed_size: usize,
    pub backbone_hidden_size: usize,
    /// The depth decoder's own llama hidden size.
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl DepthDecoderConfig {
    fn resolve(
        value: &Value,
        codebooks: usize,
        vocab_size: usize,
        audio_embed_size: usize,
        backbone_hidden: usize,
    ) -> Result<Self> {
        let uint = |key: &str, default: u64| -> usize {
            value.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
        };
        Ok(DepthDecoderConfig {
            vocab_size: uint("vocab_size", vocab_size as u64),
            num_codebooks: uint("num_codebooks", codebooks as u64),
            audio_embed_size: uint("audio_embed_size", audio_embed_size as u64),
            backbone_hidden_size: uint("backbone_hidden_size", backbone_hidden as u64),
            hidden_size: uint("hidden_size", 1024),
            num_hidden_layers: uint("num_hidden_layers", 12),
            intermediate_size: uint("intermediate_size", 8192),
            num_attention_heads: uint("num_attention_heads", 8),
            num_key_value_heads: uint("num_key_value_heads", 2),
            head_dim: uint("head_dim", 128),
            rms_norm_eps: value
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            rope_theta: value
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(500000.0) as f32,
        })
    }
}

/// Sampling and generation options with the reference `generate`
/// defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    /// `0` disables after the reference's vocabulary clamp.
    pub top_k: i32,
    /// 1.0 (or 0) disables; applied to the first codebook over every
    /// generated frame's first code, deduplicated, sign-aware, no window.
    pub repetition_penalty: f32,
    /// Classifier-free guidance scale; `None` behaves as 1.0.
    pub cfg_scale: Option<f32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_tokens: 750,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            cfg_scale: None,
        }
    }
}

impl GenerateParams {
    /// The `generate` entry-point validation (`max_tokens must be
    /// positive`, `top_k must be non-negative`, and so on).
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "max_tokens must be positive".into(),
            });
        }
        if self.top_k < 0 {
            return Err(SpeechError::Input {
                why: "top_k must be non-negative".into(),
            });
        }
        if self.temperature < 0.0 {
            return Err(SpeechError::Input {
                why: "temperature must be non-negative".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(SpeechError::Input {
                why: "top_p must be between 0 and 1".into(),
            });
        }
        if self.repetition_penalty <= 0.0 {
            return Err(SpeechError::Input {
                why: "repetition_penalty must be positive".into(),
            });
        }
        if let Some(scale) = self.cfg_scale {
            if !scale.is_finite() {
                return Err(SpeechError::Input {
                    why: "cfg_scale must be finite".into(),
                });
            }
        }
        Ok(())
    }

    /// `use_cfg`: guidance only runs with a conditioning-free pass and a
    /// scale different from 1.0 (`None` behaves as 1.0).
    pub fn use_cfg(&self) -> bool {
        self.cfg_scale.is_some() && self.cfg_scale != Some(1.0)
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The first codebook sampled the wrapper EOS class (`vocab_size`).
    EndOfText,
    /// The frame budget ran out.
    MaxTokens,
}

/// One generated segment: the sampled codebook frames (time-major, one
/// `num_codebooks` entry per frame) and why the loop stopped.
#[derive(Debug, Clone)]
pub struct Generated {
    pub frames: Vec<Vec<i32>>,
    pub stopped_by: StopReason,
}

/// A pre-encoded reference prompt (`_encode_reference` output): the
/// transcript ids and the codec codes `[K][T]` in book-major order. The
/// codec encode step needs the Qwen3-TTS speech tokenizer encoder, which
/// has no Rust port, so codes arrive precomputed.
#[derive(Debug, Clone)]
pub struct ReferencePrompt {
    pub text_ids: Vec<i32>,
    pub codes: Vec<Vec<i32>>,
}

/// One bidirectional T5Gemma2 attention layer.
struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Zero-initialized offset norms (`1.0 + weight`).
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: lm::rope::Rope,
    /// `None` for the bidirectional full-attention layers, `Some(width)`
    /// for the T5Gemma2 centred window.
    window: Option<usize>,
}

impl TextAttention {
    fn forward(&self, x: &[f32], rows: usize, mask: &TextMask) -> Result<Vec<f32>> {
        let heads = self.n_heads;
        let kv_heads = self.n_kv_heads;
        let dim = self.head_dim;
        let q = self.q_proj.forward(x, rows);
        let k = self.k_proj.forward(x, rows);
        let v = self.v_proj.forward(x, rows);
        let mut qh = head_major(&q, rows, heads, dim);
        let mut kh = head_major(&k, rows, kv_heads, dim);
        let vh = head_major(&v, rows, kv_heads, dim);
        self.q_norm.apply(&mut qh, heads * rows);
        self.k_norm.apply(&mut kh, kv_heads * rows);
        self.rope.apply(&mut qh, heads, rows, 0);
        self.rope.apply(&mut kh, kv_heads, rows, 0);

        let rep = heads / kv_heads;
        let mut out = vec![0.0f32; heads * rows * dim];
        for h in 0..heads {
            let kv = h / rep;
            let q_head = &qh[h * rows * dim..(h + 1) * rows * dim];
            let k_head = &kh[kv * rows * dim..(kv + 1) * rows * dim];
            let v_head = &vh[kv * rows * dim..(kv + 1) * rows * dim];
            for i in 0..rows {
                let visible: Vec<usize> = match mask {
                    TextMask::None => (0..rows).collect(),
                    TextMask::Window { left, right } => {
                        let distance = i as isize;
                        (0..rows)
                            .filter(|&j| {
                                let d = distance - j as isize;
                                (0 <= d && d < *left as isize) || (-(*right as isize) < d && d < 0)
                            })
                            .collect()
                    }
                };
                let mut scores = vec![0.0f32; visible.len()];
                let mut max = f32::NEG_INFINITY;
                for (slot, &j) in visible.iter().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_head[i * dim + d] * k_head[j * dim + d];
                    }
                    let score = dot * self.scale;
                    scores[slot] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row = &mut out[(h * rows + i) * dim..(h * rows + i + 1) * dim];
                for (slot, &j) in visible.iter().enumerate() {
                    let weight = scores[slot] * inv;
                    for d in 0..dim {
                        out_row[d] += weight * v_head[j * dim + d];
                    }
                }
            }
        }
        let mut joined = vec![0.0f32; rows * heads * dim];
        for t in 0..rows {
            for h in 0..heads {
                let src = (h * rows + t) * dim;
                let dst = t * heads * dim + h * dim;
                joined[dst..dst + dim].copy_from_slice(&out[src..src + dim]);
            }
        }
        Ok(self.o_proj.forward(&joined, rows))
    }
}

/// The T5Gemma2 attention pattern for one layer over a batch-of-one
/// unpadded input: unconstrained bidirectional, or the centred window
/// `(w + 1) // 2` left and `w // 2 + 1` right (both include the query
/// position).
enum TextMask {
    None,
    Window { left: usize, right: usize },
}

struct TextLayer {
    attention: TextAttention,
    pre_self_attn_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    mlp: TextMlp,
}

struct TextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMlp {
    /// `down(gelu_approx(gate(x)) * up(x))`; `nn.gelu_approx` is the
    /// tanh approximation.
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let gate = self.gate_proj.forward(x, rows);
        let up = self.up_proj.forward(x, rows);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let inner = std::f32::consts::SQRT_2 / std::f32::consts::PI.sqrt();
                let t = (inner * (g + 0.044715 * g * g * g)).tanh();
                0.5 * g * (1.0 + t) * u
            })
            .collect();
        self.down_proj.forward(&fused, rows)
    }
}

/// The bidirectional T5Gemma2 text encoder (`_TextEncoder`).
pub struct TextEncoder {
    config: TextEncoderConfig,
    /// Token table; the EOI id is REPLACED by `eoi_embedding` after the
    /// square-root scale (`_TextEmbedding`).
    embed_tokens: Vec<f32>,
    eoi_embedding: Vec<f32>,
    layers: Vec<TextLayer>,
    norm: RmsNorm,
}

impl TextEncoder {
    fn load(weights: &lm::Weights, config: &TextEncoderConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let prefix = "text_encoder.";
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_width = config.num_attention_heads * config.head_dim;
        let scale = (config.query_pre_attn_scalar).powf(-0.5) as f32;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}.");
            let full = config.layer_types[index];
            let rope = lm::rope::Rope::initialize(
                config.head_dim,
                (if full {
                    config.rope_theta_full
                } else {
                    config.rope_theta_sliding
                }) as f32,
                false,
                if full {
                    config.rope_scaling_full.as_ref()
                } else {
                    config.rope_scaling_sliding.as_ref()
                },
                Some(config.max_position_embeddings),
            )?;
            layers.push(TextLayer {
                attention: TextAttention {
                    q_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.q_proj"),
                        hidden,
                        q_width,
                        lm::Bias::None,
                    )?,
                    k_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.k_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    v_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.v_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    o_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        hidden,
                        lm::Bias::None,
                    )?,
                    q_norm: RmsNorm::new(
                        offset_scale(weights.tensor_absolute(
                            &format!("{layer}self_attn.q_norm.weight"),
                            &[config.head_dim],
                        )?),
                        config.rms_norm_eps,
                    ),
                    k_norm: RmsNorm::new(
                        offset_scale(weights.tensor_absolute(
                            &format!("{layer}self_attn.k_norm.weight"),
                            &[config.head_dim],
                        )?),
                        config.rms_norm_eps,
                    ),
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale,
                    rope,
                    window: if config.layer_types[index] {
                        None
                    } else {
                        Some(config.sliding_window)
                    },
                },
                pre_self_attn_layernorm: RmsNorm::new(
                    offset_scale(weights.tensor_absolute(
                        &format!("{layer}pre_self_attn_layernorm.weight"),
                        &[hidden],
                    )?),
                    config.rms_norm_eps,
                ),
                post_self_attn_layernorm: RmsNorm::new(
                    offset_scale(weights.tensor_absolute(
                        &format!("{layer}post_self_attn_layernorm.weight"),
                        &[hidden],
                    )?),
                    config.rms_norm_eps,
                ),
                pre_feedforward_layernorm: RmsNorm::new(
                    offset_scale(weights.tensor_absolute(
                        &format!("{layer}pre_feedforward_layernorm.weight"),
                        &[hidden],
                    )?),
                    config.rms_norm_eps,
                ),
                post_feedforward_layernorm: RmsNorm::new(
                    offset_scale(weights.tensor_absolute(
                        &format!("{layer}post_feedforward_layernorm.weight"),
                        &[hidden],
                    )?),
                    config.rms_norm_eps,
                ),
                mlp: TextMlp {
                    gate_proj: weights.linear_absolute(
                        &format!("{layer}mlp.gate_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    up_proj: weights.linear_absolute(
                        &format!("{layer}mlp.up_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    down_proj: weights.linear_absolute(
                        &format!("{layer}mlp.down_proj"),
                        config.intermediate_size,
                        hidden,
                        lm::Bias::None,
                    )?,
                },
            });
        }
        Ok(TextEncoder {
            config: config.clone(),
            embed_tokens: weights.tensor_absolute(
                &format!("{prefix}embed_tokens.weight"),
                &[config.vocab_size, hidden],
            )?,
            eoi_embedding: weights
                .tensor_absolute(&format!("{prefix}embed_tokens.eoi_embedding"), &[hidden])?,
            layers,
            norm: RmsNorm::new(
                offset_scale(weights.tensor_absolute(&format!("{prefix}norm.weight"), &[hidden])?),
                config.rms_norm_eps,
            ),
        })
    }

    /// The encoded text `[rows, hidden]`: scaled token table with the EOI
    /// replacement, per-layer bidirectional blocks with the two offset
    /// residual norms, final offset norm.
    pub fn forward(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        if tokens.is_empty() {
            return Err(SpeechError::Input {
                why: "empty text encoder input".into(),
            });
        }
        let scale = (hidden as f64).sqrt() as f32;
        let mut h = vec![0.0f32; tokens.len() * hidden];
        for (row, &token) in tokens.iter().enumerate() {
            let index = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative text token {token}"),
            })?;
            let base = row * hidden;
            if index == self.config.eoi_token_index {
                h[base..base + hidden].copy_from_slice(&self.eoi_embedding);
            } else {
                let source = embed_slice(&self.embed_tokens, index, hidden)?;
                for (col, value) in h[base..base + hidden].iter_mut().enumerate() {
                    *value = source[col] * scale;
                }
            }
        }
        let mask = |layer: &TextLayer| -> TextMask {
            match layer.attention.window {
                Some(width) => TextMask::Window {
                    left: width.div_ceil(2),
                    right: (width / 2) + 1,
                },
                None => TextMask::None,
            }
        };
        for layer in &self.layers {
            // The T5Gemma2 residual: h = h + post_norm(attn(pre_norm(h))).
            let mut normed = h.clone();
            layer
                .pre_self_attn_layernorm
                .apply(&mut normed, tokens.len());
            let attended = layer
                .attention
                .forward(&normed, tokens.len(), &mask(layer))?;
            let mut post = attended;
            layer
                .post_self_attn_layernorm
                .apply(&mut post, tokens.len());
            for (index, value) in post.iter().enumerate() {
                h[index] += *value;
            }
            // h = h + post_ffn_norm(mlp(pre_ffn_norm(h))).
            let mut fed = h.clone();
            layer
                .pre_feedforward_layernorm
                .apply(&mut fed, tokens.len());
            let fed_forward = layer.mlp.forward(&fed, tokens.len());
            let mut post = fed_forward;
            layer
                .post_feedforward_layernorm
                .apply(&mut post, tokens.len());
            for (index, value) in post.iter().enumerate() {
                h[index] += *value;
            }
        }
        let mut h_final = h;
        self.norm.apply(&mut h_final, tokens.len());
        Ok(h_final)
    }
}

fn embed_slice(table: &[f32], row: usize, width: usize) -> Result<&[f32]> {
    let start = row
        .checked_mul(width)
        .filter(|&s| s + width <= table.len())
        .ok_or_else(|| SpeechError::Input {
            why: format!("token id {row} outside the text table"),
        })?;
    Ok(&table[start..start + width])
}

/// `_T5Gemma2RMSNorm` runs `mx.fast.rms_norm(x, 1.0 + weight, eps)`, so
/// the offset folds into the stored scale vector.
fn offset_scale(weight: Vec<f32>) -> Vec<f32> {
    weight.into_iter().map(|w| 1.0 + w).collect()
}

/// One llama depth-decoder layer (shared engine attention and MLP).
struct DepthLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// The codebook-autoregressive depth decoder (`_DepthDecoder` over
/// `_DepthModel`).
pub struct DepthDecoder {
    config: DepthDecoderConfig,
    embed_tokens: Vec<f32>,
    layers: Vec<DepthLayer>,
    norm: RmsNorm,
    inputs_embeds_projector: Linear,
    /// Present when the backbone hidden width differs from the audio
    /// embedding width.
    backbone_projector: Option<Linear>,
    /// `[(K - 1), hidden, vocab_size]` per-position output projections.
    codebooks_head: Vec<f32>,
}

impl DepthDecoder {
    fn load(weights: &lm::Weights, config: &DepthDecoderConfig) -> Result<Self> {
        let prefix = "depth_decoder.model.";
        let hidden = config.hidden_size;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_width = config.num_attention_heads * config.head_dim;
        let rope =
            lm::rope::Rope::initialize(config.head_dim, config.rope_theta, false, None, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}.");
            layers.push(DepthLayer {
                attention: Attention {
                    q_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.q_proj"),
                        hidden,
                        q_width,
                        lm::Bias::None,
                    )?,
                    k_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.k_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    v_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.v_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    o_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        hidden,
                        lm::Bias::None,
                    )?,
                    q_norm: None,
                    k_norm: None,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale: (config.head_dim as f64).powf(-0.5) as f32,
                    rope: Some(rope.clone()),
                    window: None,
                    bidirectional: false,
                    exec: None,
                },
                mlp: Mlp {
                    gate_proj: weights.linear_absolute(
                        &format!("{layer}mlp.gate_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    up_proj: weights.linear_absolute(
                        &format!("{layer}mlp.up_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    down_proj: weights.linear_absolute(
                        &format!("{layer}mlp.down_proj"),
                        config.intermediate_size,
                        hidden,
                        lm::Bias::None,
                    )?,
                    exec: None,
                },
                input_layernorm: RmsNorm::new(
                    weights
                        .tensor_absolute(&format!("{layer}input_layernorm.weight"), &[hidden])?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{layer}post_attention_layernorm.weight"),
                        &[hidden],
                    )?,
                    config.rms_norm_eps,
                ),
            });
        }
        let backbone_projector = if config.backbone_hidden_size != config.audio_embed_size {
            Some(weights.linear_absolute(
                &format!("{prefix}backbone_hidden_state_projector"),
                config.backbone_hidden_size,
                config.audio_embed_size,
                lm::Bias::None,
            )?)
        } else {
            None
        };
        Ok(DepthDecoder {
            embed_tokens: weights.tensor_absolute(
                // The official checkpoint ties this table with the
                // backbone audio embedding and stores the depth tensor;
                // the reference derive order prefers it.
                &format!("{prefix}embed_tokens.weight"),
                &[
                    config.num_codebooks * config.vocab_size,
                    config.audio_embed_size,
                ],
            )?,
            layers,
            norm: RmsNorm::new(
                weights.tensor_absolute(&format!("{prefix}norm.weight"), &[hidden])?,
                config.rms_norm_eps,
            ),
            inputs_embeds_projector: weights.linear_absolute(
                &format!("{prefix}inputs_embeds_projector"),
                config.audio_embed_size,
                hidden,
                lm::Bias::None,
            )?,
            backbone_projector,
            codebooks_head: weights.tensor_absolute(
                "depth_decoder.codebooks_head.weight",
                &[
                    config.num_codebooks.saturating_sub(1),
                    hidden,
                    config.vocab_size,
                ],
            )?,
            config: config.clone(),
        })
    }

    /// `next_logits`: the vocab logits for the next codebook. `token_ids`
    /// starts with the placeholder and grows by one sampled code per
    /// step; position zero is the (projected) backbone state, later
    /// positions carry codebook-specific embedding offsets, and the head
    /// slice is `token_ids.len() - 2`.
    pub fn next_logits(&self, token_ids: &[i32], backbone_hidden: &[f32]) -> Result<Vec<f32>> {
        let config = &self.config;
        let heads = config.num_codebooks.saturating_sub(1);
        let head_idx = token_ids.len().saturating_sub(2);
        if token_ids.len() < 2 || head_idx >= heads {
            return Err(SpeechError::Input {
                why: format!(
                    "depth decoder input must contain the placeholder and at least one \
                     codebook token with a head to predict; got length {} for {heads} heads",
                    token_ids.len()
                ),
            });
        }
        let hidden = config.hidden_size;
        let width = config.audio_embed_size;
        // Position zero keeps the backbone state; later positions carry
        // the codebook offset (position i uses offset i - 1).
        let mut embeds = vec![0.0f32; token_ids.len() * width];
        for (position, &token) in token_ids.iter().enumerate() {
            if position == 0 {
                continue;
            }
            // Range-check against the per-codebook vocab. An id at or above
            // it would land in the next codebook's rows and embed silently.
            let code = usize::try_from(token)
                .ok()
                .filter(|&c| c < config.vocab_size)
                .ok_or_else(|| SpeechError::Input {
                    why: format!(
                        "depth token {token} outside codebook vocab {}",
                        config.vocab_size
                    ),
                })?;
            let index = code + (position - 1) * config.vocab_size;
            let source = embed_slice(&self.embed_tokens, index, width)?;
            embeds[position * width..(position + 1) * width].copy_from_slice(source);
        }
        let mut conditioned = backbone_hidden.to_vec();
        if let Some(projector) = &self.backbone_projector {
            conditioned = projector.forward(backbone_hidden, 1);
        }
        embeds[..width].copy_from_slice(&conditioned);
        let projected = self
            .inputs_embeds_projector
            .forward(&embeds, token_ids.len());
        // Causal cacheless pass over the whole prefix (the reference
        // rebuilds the depth state every step).
        let mut h = projected;
        let rows = token_ids.len();
        for layer in &self.layers {
            let mut attended = h.clone();
            layer.input_layernorm.apply(&mut attended, rows);
            let attn = layer.attention.forward(&attended, rows, 0, None)?;
            for (row, out) in attn.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            layer.post_attention_layernorm.apply(&mut fed, rows);
            let fed_forward = layer.mlp.forward(&fed, rows);
            for (row, out) in fed_forward.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, rows);
        let last = &h[(rows - 1) * hidden..rows * hidden];
        let head = &self.codebooks_head
            [head_idx * hidden * config.vocab_size..(head_idx + 1) * hidden * config.vocab_size];
        let mut logits = vec![0.0f32; config.vocab_size];
        for (d, &value) in last.iter().enumerate() {
            let base = d * config.vocab_size;
            for (v, logit) in logits.iter_mut().enumerate() {
                *logit += value * head[base + v];
            }
        }
        Ok(logits)
    }
}

/// The qwen3 backbone stack over the summed audio embeddings
/// (`_Backbone`), no embeddings or head of its own.
struct AudioBackbone {
    config: lm::DenseConfig,
    layers: Vec<BackboneLayer>,
    norm: RmsNorm,
}

struct BackboneLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl AudioBackbone {
    fn load(weights: &lm::Weights, config: &lm::DenseConfig) -> Result<Self> {
        let prefix = "backbone_model.";
        let hidden = config.hidden_size;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_width = config.num_attention_heads * config.head_dim;
        let rope = lm::rope::Rope::initialize(
            config.head_dim,
            config.rope_theta,
            false,
            config.rope_scaling.as_ref(),
            config.max_position_embeddings,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}.");
            let (q_norm, k_norm) = if config.qk_norms {
                (
                    Some(RmsNorm::new(
                        weights.tensor_absolute(
                            &format!("{layer}self_attn.q_norm.weight"),
                            &[config.head_dim],
                        )?,
                        config.rms_norm_eps,
                    )),
                    Some(RmsNorm::new(
                        weights.tensor_absolute(
                            &format!("{layer}self_attn.k_norm.weight"),
                            &[config.head_dim],
                        )?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None)
            };
            layers.push(BackboneLayer {
                attention: Attention {
                    q_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.q_proj"),
                        hidden,
                        q_width,
                        lm::Bias::None,
                    )?,
                    k_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.k_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    v_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.v_proj"),
                        hidden,
                        kv_width,
                        lm::Bias::None,
                    )?,
                    o_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        hidden,
                        lm::Bias::None,
                    )?,
                    q_norm,
                    k_norm,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale: (config.head_dim as f64).powf(-0.5) as f32,
                    rope: Some(rope.clone()),
                    window: None,
                    bidirectional: false,
                    exec: None,
                },
                mlp: Mlp {
                    gate_proj: weights.linear_absolute(
                        &format!("{layer}mlp.gate_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    up_proj: weights.linear_absolute(
                        &format!("{layer}mlp.up_proj"),
                        hidden,
                        config.intermediate_size,
                        lm::Bias::None,
                    )?,
                    down_proj: weights.linear_absolute(
                        &format!("{layer}mlp.down_proj"),
                        config.intermediate_size,
                        hidden,
                        lm::Bias::None,
                    )?,
                    exec: None,
                },
                input_layernorm: RmsNorm::new(
                    weights
                        .tensor_absolute(&format!("{layer}input_layernorm.weight"), &[hidden])?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor_absolute(
                        &format!("{layer}post_attention_layernorm.weight"),
                        &[hidden],
                    )?,
                    config.rms_norm_eps,
                ),
            });
        }
        Ok(AudioBackbone {
            config: config.clone(),
            layers,
            norm: RmsNorm::new(
                weights.tensor_absolute(&format!("{prefix}norm.weight"), &[hidden])?,
                config.rms_norm_eps,
            ),
        })
    }

    /// Runs the stack over row-major embeddings with the caches
    /// advancing, returning the post-norm hidden states.
    fn forward(&mut self, h: &[f32], caches: &mut [KvCache]) -> Result<Vec<f32>> {
        let rows = h.len() / self.config.hidden_size;
        let hidden = self.config.hidden_size;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty backbone input".into(),
            });
        }
        if caches.len() != self.layers.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} layer caches for {} layers",
                    caches.len(),
                    self.layers.len()
                ),
            });
        }
        let offset = caches.first().map(|c| c.offset).unwrap_or(0);
        let mut h = h.to_vec();
        for (index, block) in self.layers.iter_mut().enumerate() {
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, rows);
            let attn = block.attention.forward(
                &attended,
                rows,
                offset,
                Some(&mut caches[index] as &mut dyn lm::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.post_attention_layernorm.apply(&mut fed, rows);
            let mlp = block.mlp.forward(&fed, rows);
            for (row, out) in mlp.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, rows);
        Ok(h)
    }
}

/// The loaded family model (`Model`).
pub struct Model {
    pub config: ModelConfig,
    /// The audio wrapper embedding (`backbone_model.embed_tokens`): rows
    /// `num_codebooks * vocab_size`, width `audio_embed_size`, summed
    /// over the codebooks, optional width projector.
    audio_embed_tokens: Vec<f32>,
    audio_embeds_projector: Option<Linear>,
    backbone: AudioBackbone,
    /// The wrapper head: `[backbone_hidden, vocab_size + 1]`, the extra
    /// class is the backbone EOS.
    lm_head: Linear,
    /// The text side (`embed_text_tokens`), carried for loader parity;
    /// prompt text runs through the T5Gemma2 encoder instead.
    pub embed_text_tokens: Vec<f32>,
    text_encoder: TextEncoder,
    text_encoder_proj: Linear,
    depth_decoder: DepthDecoder,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let audio_rows = config.num_codebooks * config.vocab_size;
        // The official checkpoint ties the backbone wrapper table with
        // the depth table and stores only the depth tensor; the
        // reference sanitize derives the backbone copy from it. Read the
        // backbone name first (module-tree saves carry both) and fall
        // back to the depth tensor for tied exports.
        let audio_embed_tokens =
            if weights.contains_absolute("backbone_model.embed_tokens.embed_audio_tokens") {
                weights.tensor_absolute(
                    "backbone_model.embed_tokens.embed_audio_tokens.weight",
                    &[audio_rows, config.audio_embed_size],
                )?
            } else {
                weights.tensor_absolute(
                    "depth_decoder.model.embed_tokens.weight",
                    &[audio_rows, config.audio_embed_size],
                )?
            };
        let audio_embeds_projector =
            if weights.contains_absolute("backbone_model.embed_tokens.audio_embeds_projector") {
                Some(weights.linear_absolute(
                    "backbone_model.embed_tokens.audio_embeds_projector",
                    config.audio_embed_size,
                    config.backbone.hidden_size,
                    lm::Bias::None,
                )?)
            } else {
                None
            };
        Ok(Model {
            backbone: AudioBackbone::load(weights, &config.backbone)?,
            audio_embed_tokens,
            audio_embeds_projector,
            lm_head: weights.linear_absolute(
                "lm_head",
                config.backbone.hidden_size,
                config.vocab_size + 1,
                lm::Bias::None,
            )?,
            embed_text_tokens: weights.tensor_absolute(
                "embed_text_tokens.weight",
                &[config.text_vocab_size, config.backbone.hidden_size],
            )?,
            text_encoder: TextEncoder::load(weights, &config.text_encoder)?,
            text_encoder_proj: weights.linear_absolute(
                "text_encoder_proj",
                config.text_encoder.hidden_size,
                config.backbone.hidden_size,
                lm::Bias::None,
            )?,
            depth_decoder: DepthDecoder::load(weights, &config.depth_decoder)?,
            config,
        })
    }

    pub fn num_codebooks(&self) -> usize {
        self.config.num_codebooks
    }

    /// `_AudioEmbedding`: the summed per-codebook embeddings for one
    /// `[K]` frame, through the width projector when present.
    pub fn embed_audio_frame(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let width = self.config.audio_embed_size;
        if codes.len() != self.config.num_codebooks {
            return Err(SpeechError::Input {
                why: format!(
                    "audio codebooks must carry {} codes, got {}",
                    self.config.num_codebooks,
                    codes.len()
                ),
            });
        }
        let mut summed = vec![0.0f32; width];
        for (codebook, &code) in codes.iter().enumerate() {
            // Same bound as the depth path: codes come from user reference
            // prompts too, and an out-of-range id reads the next codebook.
            let within = usize::try_from(code)
                .ok()
                .filter(|&c| c < self.config.vocab_size)
                .ok_or_else(|| SpeechError::Input {
                    why: format!(
                        "audio code {code} outside codebook vocab {}",
                        self.config.vocab_size
                    ),
                })?;
            let index = within + codebook * self.config.vocab_size;
            let source = embed_slice(&self.audio_embed_tokens, index, width)?;
            for (col, value) in summed.iter_mut().enumerate() {
                *value += source[col];
            }
        }
        match &self.audio_embeds_projector {
            Some(projector) => Ok(projector.forward(&summed, 1)),
            None => Ok(summed),
        }
    }

    /// `_prompt_embeddings` for one segment: the optional reference text
    /// and audio turn, then the target text, all through the T5Gemma2
    /// encoder and projection (text) or the audio embedding (codes),
    /// with a `codebook_eos_token_id` frame after the reference audio.
    /// Returns the row-major `[T, backbone_hidden]` embedding.
    pub fn prompt_embeddings(
        &self,
        target_ids: &[i32],
        reference: Option<&ReferencePrompt>,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.backbone.hidden_size;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        if let Some(reference) = reference {
            rows.push(self.text_segment(&reference.text_ids)?);
            if reference.codes.len() != self.config.num_codebooks {
                return Err(SpeechError::Input {
                    why: format!(
                        "reference audio carries {} codebooks, expected {}",
                        reference.codes.len(),
                        self.config.num_codebooks
                    ),
                });
            }
            let steps = reference.codes.first().map_or(0, |book| book.len());
            for book in &reference.codes {
                if book.len() != steps {
                    return Err(SpeechError::Input {
                        why: "ragged reference codebooks".into(),
                    });
                }
            }
            for step in 0..steps {
                let frame: Vec<i32> = reference.codes.iter().map(|book| book[step]).collect();
                rows.push(self.embed_audio_frame(&frame)?);
            }
            rows.push(self.embed_audio_frame(&vec![
                self.config.codebook_eos_token_id;
                self.config.num_codebooks
            ])?);
        }
        rows.push(self.text_segment(target_ids)?);
        let mut out = Vec::with_capacity(rows.len() * hidden);
        for row in &rows {
            out.extend_from_slice(row);
        }
        Ok(out)
    }

    /// The text segment path: T5Gemma2 encode then the projection.
    fn text_segment(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let encoded = self.text_encoder.forward(ids)?;
        Ok(self.text_encoder_proj.forward(&encoded, ids.len()))
    }

    /// `_mask_reserved_codec_logits`: ids in
    /// `[codec_vocab_size, vocab_size)` drop to negative infinity. The
    /// vector may carry the extra EOS class (`vocab_size + 1` wide).
    pub fn mask_reserved_codec_logits(&self, logits: &mut [f32]) {
        let start = self.config.codec_vocab_size;
        if start >= self.config.vocab_size {
            return;
        }
        for logit in logits
            .iter_mut()
            .skip(start)
            .take(self.config.vocab_size - start)
        {
            *logit = f32::NEG_INFINITY;
        }
    }

    /// `_apply_repetition_penalty` over the first code of every generated
    /// frame: deduplicated, sign-aware, no context window.
    pub fn apply_repetition_penalty(
        &self,
        logits: &mut [f32],
        generated_tokens: &[i32],
        penalty: f32,
    ) -> Result<()> {
        if penalty == 1.0 || generated_tokens.is_empty() {
            return Ok(());
        }
        let mut seen: Vec<usize> = Vec::new();
        for &token in generated_tokens {
            let index = usize::try_from(token)
                .ok()
                .filter(|&index| index < logits.len());
            if let Some(index) = index {
                if !seen.contains(&index) {
                    seen.push(index);
                }
            }
        }
        for index in seen {
            logits[index] = if logits[index] < 0.0 {
                logits[index] * penalty
            } else {
                logits[index] / penalty
            };
        }
        Ok(())
    }

    /// `_sample` on one codebook distribution: the reserved-id mask, the
    /// slice to `vocab_size (+ 1 when allow_eos)`, the log-softmax, then
    /// the `make_sampler` chain (argmax at temperature 0, else top-p then
    /// top-k and the categorical draw). `key` feeds the crate-local
    /// threefry draw; greedy never draws.
    pub fn sample(
        &self,
        logits: &[f32],
        params: &GenerateParams,
        allow_eos: bool,
        key: rng::Key,
    ) -> Result<usize> {
        let valid = self.config.vocab_size + usize::from(allow_eos);
        if logits.len() < valid {
            return Err(SpeechError::Tensor {
                name: "logits".into(),
                why: format!(
                    "Breeze logits do not contain the expected codebook vocabulary: \
                     {} < {valid}",
                    logits.len()
                ),
            });
        }
        let mut masked = logits[..valid].to_vec();
        self.mask_reserved_codec_logits(&mut masked);
        // The reference clamps top-k to this distribution and disables it
        // when the clamp reaches the whole width.
        let effective_top_k = if params.top_k > 0 {
            let clamped = (params.top_k as usize).min(valid);
            if clamped == valid {
                0
            } else {
                clamped
            }
        } else {
            0
        };
        let logprobs = sampling::log_softmax(&masked);
        if params.temperature == 0.0 {
            return Ok(sampling::argmax(&logprobs));
        }
        let mut filtered = logprobs;
        if params.top_p > 0.0 && params.top_p < 1.0 {
            sampling::apply_top_p(&mut filtered, params.top_p);
        }
        if effective_top_k > 0 {
            sampling::apply_top_k(&mut filtered, effective_top_k)?;
        }
        Ok(sampling::sample_categorical(
            &filtered,
            params.temperature,
            key,
        ))
    }

    /// The depth-token fill for one frame (`_depth_tokens`): the
    /// placeholder plus the sampled first code, one greedy-or-sampled
    /// step per remaining codebook, with the CFG combine when guided.
    fn depth_tokens(
        &self,
        first_codebook: i32,
        conditional_hidden: &[f32],
        unconditional_hidden: Option<&[f32]>,
        params: &GenerateParams,
        keys: &mut rng::KeySequence,
    ) -> Result<Vec<i32>> {
        let scale = params.cfg_scale.unwrap_or(1.0);
        let mut tokens = vec![0i32, first_codebook];
        for _ in 0..self.config.num_codebooks - 1 {
            let mut logits = self
                .depth_decoder
                .next_logits(&tokens, conditional_hidden)?;
            if let Some(unconditional_hidden) = unconditional_hidden {
                let unconditional_logits = self
                    .depth_decoder
                    .next_logits(&tokens, unconditional_hidden)?;
                logits = combine_cfg(&logits, &unconditional_logits, scale)?;
            }
            self.mask_reserved_codec_logits(&mut logits);
            let next = self.sample(&logits, params, false, keys.next())?;
            tokens.push(next as i32);
        }
        tokens.remove(0);
        Ok(tokens)
    }

    /// Runs one segment end to end from prompt ids (`generate` for a
    /// single non-streaming segment): prompt cache, CFG prefill when
    /// guided, up to `max_tokens` frames, EOS or budget stop. `seed` only
    /// feeds the sampled draws; temperature 0 never draws and is fully
    /// deterministic.
    ///
    /// Generation ends at the codebook frames: the waveform decode needs
    /// the Qwen3-TTS speech tokenizer (no Rust port yet; see the README).
    pub fn generate(
        &mut self,
        target_ids: &[i32],
        uncond_ids: Option<&[i32]>,
        reference: Option<&ReferencePrompt>,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        params.validate()?;
        let cond = self.prompt_embeddings(target_ids, reference)?;
        let use_cfg = params.use_cfg() && uncond_ids.is_some();
        let scale = params.cfg_scale.unwrap_or(1.0);
        let uncond = match (use_cfg, uncond_ids) {
            (true, Some(ids)) => Some(self.prompt_embeddings(ids, reference)?),
            _ => None,
        };

        let hidden = self.config.backbone.hidden_size;
        let mut cond_cache = lm::make_prompt_cache(self.config.backbone.num_hidden_layers);
        let full = self.backbone.forward(&cond, &mut cond_cache)?;
        let rows = full.len() / hidden;
        let mut cond_hidden = full[(rows - 1) * hidden..].to_vec();
        let mut uncond_hidden: Option<Vec<f32>> = None;
        let mut uncond_cache = lm::make_prompt_cache(self.config.backbone.num_hidden_layers);
        if let Some(uncond_embeds) = &uncond {
            let full = self.backbone.forward(uncond_embeds, &mut uncond_cache)?;
            let rows = full.len() / hidden;
            uncond_hidden = Some(full[(rows - 1) * hidden..].to_vec());
        }

        let mut keys = rng::KeySequence::new(seed);
        let mut frames: Vec<Vec<i32>> = Vec::new();
        let mut first_codes: Vec<i32> = Vec::new();
        let mut stopped_by = StopReason::MaxTokens;
        for _ in 0..params.max_tokens {
            let mut logits = self.lm_head.forward(&cond_hidden, 1);
            if let Some(uncond_hidden) = &uncond_hidden {
                let uncond_logits = self.lm_head.forward(uncond_hidden, 1);
                logits = combine_cfg(&logits, &uncond_logits, scale)?;
            }
            self.apply_repetition_penalty(&mut logits, &first_codes, params.repetition_penalty)?;
            self.mask_reserved_codec_logits(&mut logits);
            let first = self.sample(&logits, params, true, keys.next())? as i32;
            if first == self.config.vocab_size as i32 {
                stopped_by = StopReason::EndOfText;
                break;
            }
            let frame = self.depth_tokens(
                first,
                &cond_hidden,
                uncond_hidden.as_deref(),
                params,
                &mut keys,
            )?;
            first_codes.push(first);
            frames.push(frame.clone());
            // The sampled frame feeds the next backbone step through the
            // shared audio embedding.
            let embed = self.embed_audio_frame(&frame)?;
            let full = self.backbone.forward(&embed, &mut cond_cache)?;
            cond_hidden = full[full.len() - hidden..].to_vec();
            if use_cfg {
                let full = self.backbone.forward(&embed, &mut uncond_cache)?;
                uncond_hidden = Some(full[full.len() - hidden..].to_vec());
            }
        }
        Ok(Generated { frames, stopped_by })
    }
}

/// The classifier-free guidance combine
/// `uncond + scale * (cond - uncond)`.
fn combine_cfg(conditional: &[f32], unconditional: &[f32], scale: f32) -> Result<Vec<f32>> {
    if conditional.len() != unconditional.len() {
        return Err(SpeechError::Input {
            why: format!(
                "CFG logit widths differ: {} vs {}",
                conditional.len(),
                unconditional.len()
            ),
        });
    }
    Ok(conditional
        .iter()
        .zip(unconditional)
        .map(|(&c, &u)| u + scale * (c - u))
        .collect())
}

/// Reinterprets `[rows, heads * dim]` as `[heads, rows, dim]`.
fn head_major(x: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * rows * dim];
    for t in 0..rows {
        for h in 0..heads {
            let src = t * heads * dim + h * dim;
            let dst = (h * rows + t) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}

#[cfg(test)]
mod tests;
