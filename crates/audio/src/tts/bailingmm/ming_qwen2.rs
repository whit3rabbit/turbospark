//! The Ming Qwen2 dense backbone the bailingmm audio stacks and the
//! dense LLM variant build on.
//!
//! Port of the `MingQwen2Model` / `MingQwen2Attention` classes inside
//! `mlx_audio/tts/models/bailingmm/bailingmm.py` (subclassing
//! `mlx_audio/lm/models/qwen2.py`) at the pinned mlx-audio commit.
//! Structure: the llama-shaped qwen2 block (biased q/k/v, bias-free
//! output and MLP projections, non-traditional rope on the full head
//! dim, RMSNorm pre-norm) where every layer MAY carry a sliding window:
//! `MingQwen2Model.__init__` marks layers from `layer_types` (or those
//! at or beyond `max_window_layers`) when `use_sliding_window` is set,
//! and the attention combines the causal bound with
//! `k_pos > q_pos - window` through an additive minimum, over a DENSE
//! cache. That mask differs from the engine's rotating-ring window
//! (which assumes the ring was trimmed to the window), so this module
//! implements the forward itself instead of reusing
//! [`crate::lm::engine::Attention`].
//!
//! All tensor names are loaded prefix-free through the absolute accessors;
//! the family passes the full checkpoint prefix (`model.model.` for the
//! LLM, `audio.encoder.encoder.` / `audio.decoder.decoder.` for the
//! audio stacks). Inputs are precomputed embeddings (the reference
//! `input_embeddings` path; the token ids are always zeros) and the
//! backbone returns post-norm hidden states; no LM head exists here
//! because the bailingmm generate loop reads hidden states only.

use serde_json::Value;

use crate::lm::cache::LayerKv;
use crate::lm::rope::Rope;
use crate::lm::{self, Bias, Weights};
use crate::nn::{Linear, RmsNorm};
use crate::{Result, SpeechError};

use crate::lm::engine::head_major;

/// The Ming sliding-window configuration read from the raw backbone
/// dict (`MingQwen2Model.__init__`).
#[derive(Debug, Clone, Default)]
pub struct SlidingConfig {
    /// Window width; 0 disables sliding windows entirely.
    pub sliding_window: usize,
    /// Layers at or beyond this index slide when `layer_types` is absent.
    pub max_window_layers: Option<usize>,
    /// Per-layer `"sliding_attention"` marks when present.
    pub layer_types: Option<Vec<String>>,
}

impl SlidingConfig {
    pub fn from_dict(value: &Value) -> Self {
        let use_sliding = value
            .get("use_sliding_window")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !use_sliding {
            return Self::default();
        }
        let sliding_window = value
            .get("sliding_window")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(0);
        let max_window_layers = value
            .get("max_window_layers")
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        let layer_types = value
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            });
        Self {
            sliding_window,
            max_window_layers,
            layer_types,
        }
    }

    /// The per-layer window list (`per_layer_sliding`): `Some(window)`
    /// for sliding layers, `None` otherwise.
    fn windows(&self, num_layers: usize) -> Vec<Option<usize>> {
        (0..num_layers)
            .map(|index| {
                if self.sliding_window == 0 {
                    return None;
                }
                match &self.layer_types {
                    Some(types) if index < types.len() => {
                        if types[index] == "sliding_attention" {
                            Some(self.sliding_window)
                        } else {
                            None
                        }
                    }
                    _ => {
                        let threshold = self.max_window_layers.unwrap_or(num_layers);
                        if index >= threshold {
                            Some(self.sliding_window)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: Rope,
    window: Option<usize>,
}

impl Attention {
    /// Projects, applies rope at the cache offset, updates the cache,
    /// and runs the masked attention. Query `i` (absolute position
    /// `offset + i`) sees key slots `j <= offset + i` and, when a
    /// window is set, `j > offset + i - window` (the reference additive
    /// `minimum(causal, sw_allowed)` over the dense cache). Returns
    /// `[len, n_heads * head_dim]`.
    fn forward(
        &self,
        x: &[f32],
        len: usize,
        offset: usize,
        mut cache: Option<&mut dyn LayerKv>,
    ) -> Result<Vec<f32>> {
        let heads = self.n_heads;
        let kv_heads = self.n_kv_heads;
        let dim = self.head_dim;
        let q = self.q_proj.forward(x, len);
        let k = self.k_proj.forward(x, len);
        let v = self.v_proj.forward(x, len);
        let mut qh = head_major(&q, len, heads, dim);
        let mut kh = head_major(&k, len, kv_heads, dim);
        let vh = head_major(&v, len, kv_heads, dim);
        self.rope.apply(&mut qh, heads, len, offset);
        self.rope.apply(&mut kh, kv_heads, len, offset);
        let (keys, values) = match &mut cache {
            Some(cache) => cache.update_and_fetch(&kh, &vh, kv_heads, len, dim),
            None => (kh.as_slice(), vh.as_slice()),
        };
        let total = keys.len() / (kv_heads * dim);
        let rep = heads / kv_heads;
        let mut out = vec![0.0f32; heads * len * dim];
        let mut scores = vec![0.0f32; total];
        for h in 0..heads {
            let kv = h / rep;
            let q_head = &qh[h * len * dim..(h + 1) * len * dim];
            let k_head = &keys[kv * total * dim..(kv + 1) * total * dim];
            let v_head = &values[kv * total * dim..(kv + 1) * total * dim];
            for i in 0..len {
                let g = offset + i;
                let low = match self.window {
                    None => 0,
                    Some(w) => (g + 1).saturating_sub(w),
                };
                let high = g.min(total - 1);
                if low > high {
                    return Err(SpeechError::Input {
                        why: format!(
                            "sliding window {} leaves no visible keys at position {g}",
                            self.window.unwrap_or(0)
                        ),
                    });
                }
                let q_row = &q_head[i * dim..(i + 1) * dim];
                let mut max = f32::NEG_INFINITY;
                for j in low..=high {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_row[d] * k_head[j * dim + d];
                    }
                    let score = dot * self.scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores[low..=high] {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row = &mut out[(h * len + i) * dim..(h * len + i + 1) * dim];
                for j in low..=high {
                    let weight = scores[j] * inv;
                    for d in 0..dim {
                        out_row[d] += weight * v_head[j * dim + d];
                    }
                }
            }
        }
        let mut joined = vec![0.0f32; len * heads * dim];
        for t in 0..len {
            for h in 0..heads {
                let src = (h * len + t) * dim;
                let dst = t * heads * dim + h * dim;
                joined[dst..dst + dim].copy_from_slice(&out[src..src + dim]);
            }
        }
        Ok(self.o_proj.forward(&joined, len))
    }
}

struct Block {
    attention: Attention,
    mlp: crate::lm::engine::Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// A loaded Ming Qwen2 backbone (the audio encoder/decoder stacks and
/// the dense LLM variant).
pub struct MingQwen2 {
    pub config: lm::DenseConfig,
    /// The token embedding table; the dense LLM variant embeds prompt
    /// ids through it and the audio stacks bypass it entirely.
    pub embed_tokens: Vec<f32>,
    layers: Vec<Block>,
    norm: RmsNorm,
}

impl MingQwen2 {
    /// Loads the backbone. `prefix` is the full checkpoint prefix up to
    /// `layers.` (for example `model.model.` or `audio.decoder.decoder.`);
    /// tensors load prefix-free so the checkpoint prefix must be part of
    /// it. The qwen2 flavor is the shared [`lm::qwen2::from_json`]
    /// configuration.
    pub fn load_absolute(
        weights: &Weights,
        prefix: &str,
        config: lm::DenseConfig,
        sliding: &SlidingConfig,
    ) -> Result<Self> {
        if config.head_dim * config.num_attention_heads != config.hidden_size {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!(
                    "must be divisible by num_heads (got hidden_size {} and num_heads {})",
                    config.hidden_size, config.num_attention_heads
                ),
            });
        }
        let rope = Rope::initialize(
            config.head_dim,
            config.rope_theta,
            config.rope_traditional,
            config.rope_scaling.as_ref(),
            config.max_position_embeddings,
        )?;
        let hidden = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let windows = sliding.windows(config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, window) in windows.into_iter().enumerate() {
            let layer = format!("{prefix}layers.{index}.");
            layers.push(Block {
                attention: Attention {
                    q_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.q_proj"),
                        hidden,
                        q_width,
                        Bias::Required,
                    )?,
                    k_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.k_proj"),
                        hidden,
                        kv_width,
                        Bias::Required,
                    )?,
                    v_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.v_proj"),
                        hidden,
                        kv_width,
                        Bias::Required,
                    )?,
                    o_proj: weights.linear_absolute(
                        &format!("{layer}self_attn.o_proj"),
                        q_width,
                        hidden,
                        Bias::None,
                    )?,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale: (config.head_dim as f64).powf(-0.5) as f32,
                    rope: rope.clone(),
                    window,
                },
                mlp: crate::lm::engine::Mlp {
                    gate_proj: weights.linear_absolute(
                        &format!("{layer}mlp.gate_proj"),
                        hidden,
                        config.intermediate_size,
                        Bias::None,
                    )?,
                    up_proj: weights.linear_absolute(
                        &format!("{layer}mlp.up_proj"),
                        hidden,
                        config.intermediate_size,
                        Bias::None,
                    )?,
                    down_proj: weights.linear_absolute(
                        &format!("{layer}mlp.down_proj"),
                        config.intermediate_size,
                        hidden,
                        Bias::None,
                    )?,
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
        let embed_tokens = weights.tensor_absolute(
            &format!("{prefix}embed_tokens.weight"),
            &[config.vocab_size, hidden],
        )?;
        let norm = RmsNorm::new(
            weights.tensor_absolute(&format!("{prefix}norm.weight"), &[hidden])?,
            config.rms_norm_eps,
        );
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Embeds token ids through the table (the dense LLM prompt path).
    pub fn embed(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        crate::lm::engine::lookup_embeddings(
            &self.embed_tokens,
            self.config.vocab_size,
            self.config.hidden_size,
            tokens,
        )
    }

    /// Runs the backbone over precomputed embeddings
    /// `[positions, hidden_size]` and returns the post-norm hidden
    /// states. `caches` may be `None` for a cacheless call; when present
    /// it must hold one cache per layer and advances with every call.
    pub fn forward_hidden(
        &mut self,
        embeddings: &[f32],
        mut caches: Option<&mut lm::LayerCaches>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.hidden_size;
        if embeddings.is_empty() || embeddings.len() % dim != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "embedding input length {} is not a nonempty multiple of hidden size {dim}",
                    embeddings.len()
                ),
            });
        }
        let len = embeddings.len() / dim;
        let mut h = embeddings.to_vec();
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
            let offset = caches
                .as_deref()
                .and_then(|c| c.get(index))
                .map(|c| c.offset)
                .unwrap_or(0);
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, len);
            let attn = block.attention.forward(
                &attended,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(index))
                    .map(|c| c as &mut dyn LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.post_attention_layernorm.apply(&mut fed, len);
            let mlp = block.mlp.forward(&fed, len);
            for (row, out) in mlp.chunks_exact(dim).enumerate() {
                let base = row * dim;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, len);
        Ok(h)
    }
}
