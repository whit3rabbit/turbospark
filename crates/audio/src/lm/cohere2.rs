//! The cohere2 backbone with sliding-window attention.
//!
//! Port of `mlx_audio/lm/models/cohere2.py` (vendored from mlx-lm 0.31.3)
//! at the pinned mlx-audio commit. Structure: pre-LayerNorm blocks where
//! one shared normalized input feeds attention and MLP in parallel
//! (`attn(ln(x)) + mlp(ln(x)) + x`), LayerNorm without a learned shift by
//! default, SwiGLU MLPs, and a tied head scaled by `logit_scale`.
//!
//! Attention: every `sliding_window_pattern`-th layer (0-indexed
//! `i % pattern == pattern - 1`) is global; it skips rope entirely and uses
//! the dense causal mask. All other layers are sliding-window: traditional
//! (interleaved) rope at the pre-update cache offset, a
//! `RotatingKvCache(max_size = sliding_window, keep = 0)` ring, and the
//! window mask the engine applies from `Attention::window`. The reference
//! always builds its ring with `max_size == window_size`, so the decode
//! path needs no extra mask (a window strictly smaller than the ring would
//! need the reference's rolled decode mask, which this configuration
//! cannot produce and which is refused here).
//!
//! Cacheless calls behave like the reference: rope at offset 0, causal
//! mask for global layers, and a window mask with offset 0 for sliding
//! layers.

use serde_json::Value;

use crate::nn::{self, LayerNorm};
use crate::ops;
use crate::{Result, SpeechError};

use super::cache::PromptCache;
use super::rope::Rope;
use super::{engine, Bias, Weights};

use super::engine::{Attention, Input, Mlp};

/// Parses a cohere2 `config.json`. `model_type` is required; everything
/// else carries the reference dataclass default.
pub fn from_json(value: &Value) -> Result<Cohere2Config> {
    Ok(Cohere2Config {
        model_type: nn::json::text_field(value, "model_type")?,
        hidden_size: value
            .get("hidden_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(4096),
        head_dim: value
            .get("head_dim")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(128),
        num_hidden_layers: value
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(32),
        intermediate_size: value
            .get("intermediate_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(14336),
        num_attention_heads: value
            .get("num_attention_heads")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(32),
        num_key_value_heads: value
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(8),
        rope_theta: value
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(50000.0) as f32,
        vocab_size: value
            .get("vocab_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(256000),
        layer_norm_eps: value
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5) as f32,
        logit_scale: value
            .get("logit_scale")
            .and_then(Value::as_f64)
            .unwrap_or(0.0625) as f32,
        attention_bias: value
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        layer_norm_bias: value
            .get("layer_norm_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        sliding_window: value
            .get("sliding_window")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(4096),
        sliding_window_pattern: value
            .get("sliding_window_pattern")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(4),
    })
}

/// Loads a cohere2 backbone from `weights` (prefix `model.` via the
/// `embed_tokens` autodetect).
pub fn load(weights: &Weights, value: &Value) -> Result<Cohere2Model> {
    Cohere2Model::load(weights, value)
}

/// The parsed cohere2 configuration (`ModelArgs`).
#[derive(Debug, Clone)]
pub struct Cohere2Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    pub logit_scale: f32,
    pub attention_bias: bool,
    pub layer_norm_bias: bool,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
}

struct Block {
    attn: Attention,
    mlp: Mlp,
    input_layernorm: LayerNorm,
}

/// A loaded cohere2 backbone ready to run.
pub struct Cohere2Model {
    pub config: Cohere2Config,
    embed_tokens: Vec<f32>,
    layers: Vec<Block>,
    norm: LayerNorm,
}

impl Cohere2Model {
    /// Loads every weight the configuration names (prefix `model.` for a
    /// full-model save).
    pub fn load(weights: &Weights, value: &Value) -> Result<Self> {
        let config = from_json(value)?;
        Self::from_config(weights, config)
    }

    /// Loads from an already parsed config.
    pub fn from_config(weights: &Weights, config: Cohere2Config) -> Result<Self> {
        if config.head_dim * config.num_attention_heads != config.hidden_size {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!(
                    "must be divisible by num_heads (got hidden_size {} and num_heads {})",
                    config.hidden_size, config.num_attention_heads
                ),
            });
        }
        if config.sliding_window == 0 {
            return Err(SpeechError::BadConfig {
                field: "sliding_window".into(),
                why: "must be positive".into(),
            });
        }
        if config.sliding_window_pattern == 0 {
            return Err(SpeechError::BadConfig {
                field: "sliding_window_pattern".into(),
                why: "must be positive".into(),
            });
        }
        // The ring is always built with max_size == sliding_window (the
        // reference make_cache); a window smaller than the ring would need
        // the reference's rolled decode mask, which this configuration
        // cannot produce, so no such mask exists here.
        let dim = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let bias = if config.attention_bias {
            Bias::Required
        } else {
            Bias::None
        };
        let norm_bias = |name: String| -> Result<_> {
            Ok(LayerNorm::new(
                weights.tensor(&name, &[dim])?,
                if config.layer_norm_bias {
                    Some(weights.tensor(&format!("{name}.bias"), &[dim])?)
                } else {
                    None
                },
                config.layer_norm_eps,
            ))
        };
        let sliding_rope = Rope::initialize(config.head_dim, config.rope_theta, true, None, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("layers.{index}.");
            let use_sliding_window = (index + 1) % config.sliding_window_pattern != 0;
            layers.push(Block {
                attn: Attention {
                    q_proj: weights.linear(
                        &format!("{layer}self_attn.q_proj"),
                        dim,
                        q_width,
                        bias,
                    )?,
                    k_proj: weights.linear(
                        &format!("{layer}self_attn.k_proj"),
                        dim,
                        kv_width,
                        bias,
                    )?,
                    v_proj: weights.linear(
                        &format!("{layer}self_attn.v_proj"),
                        dim,
                        kv_width,
                        bias,
                    )?,
                    o_proj: weights.linear(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        dim,
                        bias,
                    )?,
                    q_norm: None,
                    k_norm: None,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale: (config.head_dim as f64).powf(-0.5) as f32,
                    rope: if use_sliding_window {
                        Some(sliding_rope.clone())
                    } else {
                        None
                    },
                    window: if use_sliding_window {
                        Some(config.sliding_window)
                    } else {
                        None
                    },
                    bidirectional: false,
                },
                mlp: Mlp {
                    gate_proj: weights.linear(
                        &format!("{layer}mlp.gate_proj"),
                        dim,
                        config.intermediate_size,
                        Bias::None,
                    )?,
                    up_proj: weights.linear(
                        &format!("{layer}mlp.up_proj"),
                        dim,
                        config.intermediate_size,
                        Bias::None,
                    )?,
                    down_proj: weights.linear(
                        &format!("{layer}mlp.down_proj"),
                        config.intermediate_size,
                        dim,
                        Bias::None,
                    )?,
                },
                input_layernorm: norm_bias(format!("{layer}input_layernorm"))?,
            });
        }
        let embed_tokens = weights.embedding("embed_tokens", config.vocab_size, dim)?;
        let norm = norm_bias("norm".to_owned())?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Port of `Model.make_cache`: a `RotatingKvCache(max_size =
    /// sliding_window, keep = 0)` ring per sliding layer and a dense
    /// `KvCache` per global layer.
    pub fn make_cache(&self) -> Vec<PromptCache> {
        (0..self.config.num_hidden_layers)
            .map(|index| {
                if index % self.config.sliding_window_pattern
                    == self.config.sliding_window_pattern - 1
                {
                    PromptCache::dense()
                } else {
                    PromptCache::rotating(self.config.sliding_window, 0)
                }
            })
            .collect()
    }

    /// Runs the backbone and returns logits
    /// `[positions, vocab_size]`, already scaled by `logit_scale`.
    /// `caches` may be `None` for a cacheless call; when present it must
    /// hold one cache per layer in [`Cohere2Model::make_cache`] order and
    /// advances with every call.
    pub fn forward(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut Vec<PromptCache>>,
    ) -> Result<Vec<f32>> {
        Ok(self.forward_hidden_and_logits(input, caches)?.1)
    }

    /// Runs the backbone once and returns both the post-norm hidden states
    /// `[positions, hidden_size]` and the logit-scaled logits taken from
    /// them. The rumik_oss TTS family reads the last hidden row with its
    /// stop predictor while sampling the logits of the same pass; a
    /// forward plus a forward_hidden pair would run the transformer twice.
    pub fn forward_hidden_and_logits(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut Vec<PromptCache>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let hidden = self.forward_hidden(input, caches)?;
        let len = hidden.len() / self.config.hidden_size;
        let mut logits = ops::linear(
            &hidden,
            &self.embed_tokens,
            None,
            len,
            self.config.hidden_size,
            self.config.vocab_size,
        );
        for value in &mut logits {
            *value *= self.config.logit_scale;
        }
        Ok((hidden, logits))
    }

    /// Runs the backbone up to the final norm and returns the hidden state
    /// `[positions, hidden_size]`.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        mut caches: Option<&mut Vec<PromptCache>>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.hidden_size;
        let len = input.len(dim)?;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        let mut h = match input {
            Input::Tokens(tokens) => {
                engine::lookup_embeddings(&self.embed_tokens, self.config.vocab_size, dim, tokens)?
            }
            Input::Embeddings(values) => values.to_vec(),
        };
        if let Some(caches) = caches.as_deref() {
            if caches.len() != self.layers.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "{} layer caches provided for {} layers",
                        caches.len(),
                        self.layers.len()
                    ),
                });
            }
        }
        for (index, block) in self.layers.iter_mut().enumerate() {
            // One shared normalized input feeds attention and MLP.
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, len);
            let offset = caches
                .as_deref()
                .and_then(|c| c.get(index))
                .map(super::cache::LayerKv::offset)
                .unwrap_or(0);
            let attn = block.attn.forward(
                &attended,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(index))
                    .map(|c| c as &mut dyn super::cache::LayerKv),
            )?;
            let mlp = block.mlp.forward(&attended, len);
            // attn_h + ff_h + x
            for (row, values) in attn.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in values.iter().enumerate() {
                    h[base + col] += value + mlp[base + col];
                }
            }
        }
        self.norm.apply(&mut h, len);
        Ok(h)
    }
}
