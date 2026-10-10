//! Shared dense transformer engine behind the llama, qwen2, and qwen3
//! backbones.
//!
//! The three reference architectures are the same decoder-only transformer
//! with per-flavor configuration differences (q/k RMSNorm, attention bias,
//! head_dim derivation), which live in the per-backbone config modules. The
//! math here is the shared part: RMSNorm pre-norm blocks, GQA attention with
//! rope and a KV cache, SwiGLU MLPs, and a tied or untied LM head. Single
//! batch, matching the reference generation loops the TTS families drive.

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};

use super::cache::{LayerCaches, LayerKv};
use super::rope::Rope;
use super::{Bias, Weights};

/// Fully resolved backbone configuration; the per-backbone modules derive
/// their flavors' defaults before this reaches the engine.
#[derive(Debug, Clone)]
pub struct DenseConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    /// Qwen3-style per-head q/k RMSNorm before rope.
    pub qk_norms: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub rope_theta: f32,
    pub rope_traditional: bool,
    pub rope_scaling: Option<serde_json::Value>,
    pub max_position_embeddings: Option<usize>,
    pub tie_word_embeddings: bool,
    /// The dense llama-family backbones have no sliding-window layer types
    /// in this workspace: a declared window is refused by name. The
    /// rotating cache itself is ported (`cache::RotatingKvCache`) and
    /// cohere2 uses it; llama/qwen2/qwen3 simply never configure one.
    pub sliding_window: Option<usize>,
}

/// The forward input: token ids or precomputed embeddings (the reference
/// `input_embeddings` path the audio families use to inject codec
/// features).
#[derive(Debug, Clone, Copy)]
pub enum Input<'a> {
    Tokens(&'a [i32]),
    Embeddings(&'a [f32]),
}

impl<'a> Input<'a> {
    pub(crate) fn len(&self, hidden_size: usize) -> Result<usize> {
        match self {
            Input::Tokens(tokens) => Ok(tokens.len()),
            Input::Embeddings(values) => {
                if values.len() % hidden_size != 0 {
                    return Err(SpeechError::Input {
                        why: format!(
                            "embedding input length {} is not a multiple of hidden size {hidden_size}",
                            values.len()
                        ),
                    });
                }
                Ok(values.len() / hidden_size)
            }
        }
    }
}

/// GQA attention projections shared by the dense backbones and the stage-1
/// additions (gpt2, cohere2, bailing_moe). The fields are open to the
/// sibling backbone modules so they can load their own tensor names into
/// the same math.
pub(crate) struct Attention {
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: Option<RmsNorm>,
    pub(crate) k_norm: Option<RmsNorm>,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f32,
    /// Rotary construction applied before the cache update; `None` for the
    /// no-rope backbones (gpt2) and for cohere2's global layers.
    pub(crate) rope: Option<Rope>,
    /// Sliding-window width. `Some` mirrors the reference rotating-cache
    /// mask: a multi-position query `i` sees buffer slots
    /// `(min(w - 1, offset) + i - w, min(w - 1, offset) + i]`, and a
    /// single-position query sees the whole buffer (the reference returns
    /// no mask: the ring only holds positions inside the window). `None`
    /// is plain causal.
    pub(crate) window: Option<usize>,
}

/// SwiGLU feed-forward shared with the sibling backbone modules.
pub(crate) struct Mlp {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
}

struct Block {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// A loaded dense backbone ready to run.
pub struct Backbone {
    pub config: DenseConfig,
    embed_tokens: Vec<f32>,
    layers: Vec<Block>,
    norm: RmsNorm,
    /// Untied head; a tied head reuses `embed_tokens` as the weight.
    lm_head: Option<Linear>,
}

impl Backbone {
    /// Loads every weight the configuration names. The prefix on `weights`
    /// must cover the backbone tensors (typically `model.` for a full-model
    /// save; the LM head is then prefix-free, so untied models must set an
    /// empty prefix and name tensors explicitly).
    pub fn load(weights: &Weights, config: DenseConfig) -> Result<Self> {
        Self::load_with_embed_table(weights, config, "embed_tokens")
    }

    /// [`Backbone::load`] with an alternative embed-table tensor base name
    /// (the prefix still applies). The higgs_audio_v3 raw-checkpoint layout
    /// keeps the text embedding at `tied.embedding.text_embedding` while
    /// the layers and final norm carry the `body.` prefix, which no single
    /// prefix can express; such a family passes the raw name here. The
    /// default [`Backbone::load`] reads `embed_tokens`.
    pub fn load_with_embed_table(
        weights: &Weights,
        config: DenseConfig,
        embed_table: &str,
    ) -> Result<Self> {
        if let Some(window) = config.sliding_window {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "sliding_attention layers (window {window}) need layer-type-aware caches, which the dense engine does not configure; cohere2 owns the rotating cache path"
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
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_width = config.num_attention_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("layers.{index}.");
            let attention_bias = if config.attention_bias {
                Bias::Required
            } else {
                Bias::None
            };
            let mlp_bias = if config.mlp_bias {
                Bias::Required
            } else {
                Bias::None
            };
            let q_proj = weights.linear(
                &format!("{layer}self_attn.q_proj"),
                hidden,
                q_width,
                attention_bias,
            )?;
            let k_proj = weights.linear(
                &format!("{layer}self_attn.k_proj"),
                hidden,
                kv_width,
                attention_bias,
            )?;
            let v_proj = weights.linear(
                &format!("{layer}self_attn.v_proj"),
                hidden,
                kv_width,
                attention_bias,
            )?;
            let o_proj = weights.linear(
                &format!("{layer}self_attn.o_proj"),
                q_width,
                hidden,
                Bias::None,
            )?;
            let (q_norm, k_norm) = if config.qk_norms {
                (
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{layer}self_attn.q_norm"), &[config.head_dim])?,
                        config.rms_norm_eps,
                    )),
                    Some(RmsNorm::new(
                        weights.tensor(&format!("{layer}self_attn.k_norm"), &[config.head_dim])?,
                        config.rms_norm_eps,
                    )),
                )
            } else {
                (None, None)
            };
            let gate_proj = weights.linear(
                &format!("{layer}mlp.gate_proj"),
                hidden,
                config.intermediate_size,
                mlp_bias,
            )?;
            let up_proj = weights.linear(
                &format!("{layer}mlp.up_proj"),
                hidden,
                config.intermediate_size,
                mlp_bias,
            )?;
            let down_proj = weights.linear(
                &format!("{layer}mlp.down_proj"),
                config.intermediate_size,
                hidden,
                mlp_bias,
            )?;
            let input_layernorm = RmsNorm::new(
                weights.tensor(&format!("{layer}input_layernorm"), &[hidden])?,
                config.rms_norm_eps,
            );
            let post_attention_layernorm = RmsNorm::new(
                weights.tensor(&format!("{layer}post_attention_layernorm"), &[hidden])?,
                config.rms_norm_eps,
            );
            layers.push(Block {
                attention: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    // The reference computes the scale in Python f64 and
                    // hands it to the f32 attention kernel.
                    scale: (config.head_dim as f64).powf(-0.5) as f32,
                    rope: Some(rope.clone()),
                    window: None,
                },
                mlp: Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                input_layernorm,
                post_attention_layernorm,
            });
        }
        let embed_tokens = weights.embedding(embed_table, config.vocab_size, hidden)?;
        let norm = RmsNorm::new(weights.tensor("norm", &[hidden])?, config.rms_norm_eps);
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            // The head sits outside the model prefix in full-model saves.
            Some(weights.linear_absolute("lm_head", hidden, config.vocab_size, Bias::None)?)
        };
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// The loaded text embedding table. Read-only access for families
    /// that build prompt embeddings outside `forward` (the
    /// higgs_audio_v3 prompt splices text rows with fused audio rows).
    pub fn embed_table(&self) -> &[f32] {
        &self.embed_tokens
    }

    /// Runs the backbone and returns logits `[positions, vocab_size]`.
    /// `caches` may be `None` for a cacheless call (the reference passes
    /// `cache=None`); when present it must hold one cache per layer, and it
    /// advances with every call.
    pub fn forward(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let projected = self.forward_hidden(input, caches)?;
        Ok(self.apply_head(&projected))
    }

    /// Runs the backbone once and returns both the post-norm hidden states
    /// `[positions, hidden_size]` and the logits `[positions, vocab_size]`
    /// taken from them. The soprano TTS family needs both rows of the same
    /// pass (hidden states feed its vocoder while the last-position logits
    /// feed the sampler); a forward plus a forward_hidden pair would run
    /// the transformer twice.
    pub fn forward_hidden_and_logits(
        &mut self,
        input: Input<'_>,
        caches: Option<&mut LayerCaches>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let hidden = self.forward_hidden(input, caches)?;
        let logits = self.apply_head(&hidden);
        Ok((logits, hidden))
    }

    /// The tied or untied LM head over post-norm hidden states
    /// `[positions, hidden_size]`, yielding `[positions, vocab_size]`.
    fn apply_head(&self, projected: &[f32]) -> Vec<f32> {
        let len = projected.len() / self.config.hidden_size;
        match &self.lm_head {
            Some(head) => head.forward(projected, len),
            None => ops::linear(
                projected,
                &self.embed_tokens,
                None,
                len,
                self.config.hidden_size,
                self.config.vocab_size,
            ),
        }
    }

    /// Runs the backbone up to the final norm and returns the hidden state
    /// `[positions, hidden_size]`. The audio families use this to feed
    /// backbone features into their audio adapters instead of logits.
    pub fn forward_hidden(
        &mut self,
        input: Input<'_>,
        mut caches: Option<&mut LayerCaches>,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let len = input.len(hidden)?;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty forward input".into(),
            });
        }
        let mut h = match input {
            Input::Tokens(tokens) => {
                lookup_embeddings(&self.embed_tokens, self.config.vocab_size, hidden, tokens)?
            }
            Input::Embeddings(values) => values.to_vec(),
        };
        let offset = caches
            .as_deref()
            .and_then(|c| c.first())
            .map(|c| c.offset)
            .unwrap_or(0);
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
            let mut attended = h.clone();
            block.input_layernorm.apply(&mut attended, len);
            let attn = block.attention.forward(
                &attended,
                len,
                offset,
                caches
                    .as_deref_mut()
                    .map(|c| &mut c[index] as &mut dyn LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.post_attention_layernorm.apply(&mut fed, len);
            let mlp = block.mlp.forward(&fed, len);
            for (row, out) in mlp.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.norm.apply(&mut h, len);
        Ok(h)
    }
}

impl Attention {
    /// Projects, applies per-head q/k norms and rope, updates the cache,
    /// and runs masked GQA attention. Returns `[len, n_heads * head_dim]`.
    pub(crate) fn forward(
        &self,
        x: &[f32],
        len: usize,
        offset: usize,
        mut cache: Option<&mut dyn LayerKv>,
    ) -> Result<Vec<f32>> {
        let heads = self.n_heads;
        let kv_heads = self.n_kv_heads;
        let dim = self.head_dim;
        if heads % kv_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "num_attention_heads".into(),
                why: format!("{heads} query heads not a multiple of {kv_heads} kv heads"),
            });
        }
        let q = self.q_proj.forward(x, len);
        let k = self.k_proj.forward(x, len);
        let v = self.v_proj.forward(x, len);

        // [heads, len, dim] head-major views for rope and attention.
        let mut qh = head_major(&q, len, heads, dim);
        let mut kh = head_major(&k, len, kv_heads, dim);
        let vh = head_major(&v, len, kv_heads, dim);
        if let Some(q_norm) = &self.q_norm {
            q_norm.apply(&mut qh, heads * len);
        }
        if let Some(k_norm) = &self.k_norm {
            k_norm.apply(&mut kh, kv_heads * len);
        }
        if let Some(rope) = &self.rope {
            rope.apply(&mut qh, heads, len, offset);
            rope.apply(&mut kh, kv_heads, len, offset);
        }

        let (keys, values) = match &mut cache {
            Some(cache) => cache.update_and_fetch(&kh, &vh, kv_heads, len, dim),
            None => (kh.as_slice(), vh.as_slice()),
        };
        let total = keys.len() / (kv_heads * dim);

        // Masked softmax attention per query head and position. The causal
        // bound is `j <= g` for the dense appending cache; the sliding
        // window narrows it to the reference rotating-cache mask below.
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
                // Visible slot range (inclusive). `None` mirrors the
                // reference "causal" mask. `Some(w)` mirrors the rotating
                // cache masks: for multi-position calls
                // `create_causal_mask(N, min(w - 1, offset), w)` lets
                // query i see slots `(min(w - 1, offset) + i - w,
                // min(w - 1, offset) + i]` (the trimmed ring holds exactly
                // those rows), and a single-position call passes no mask
                // because the ring only ever retains the last
                // `min(offset, w)` positions.
                let (low, high) = match self.window {
                    None => (0, g),
                    Some(_) if len == 1 => (0, total - 1),
                    Some(w) => {
                        let end = offset.min(w - 1) + i;
                        (end.saturating_sub(w - 1), end)
                    }
                };
                let high = high.min(total - 1);
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

        // Back to [len, heads * dim] for the output projection.
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

impl Mlp {
    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
    pub(crate) fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let gate = self.gate_proj.forward(x, len);
        let up = self.up_proj.forward(x, len);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        self.down_proj.forward(&fused, len)
    }
}

/// Reinterprets `[len, heads * dim]` as `[heads, len, dim]`.
pub(crate) fn head_major(x: &[f32], len: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * len * dim];
    for t in 0..len {
        for h in 0..heads {
            let src = t * heads * dim + h * dim;
            let dst = (h * len + t) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}

/// Splits a fused projection (`gpt2` `c_attn`, bailing `query_key_value`)
/// into the q, k, and v linears over consecutive output ranges of widths
/// `q_width` and `k_width` (v takes the remainder). The reference computes
/// `mx.split(fused, ..., axis=-1)`, which cuts the `[out, in]` weight (and
/// the bias) into those same row ranges.
pub(crate) fn split_fused_linear(
    fused: &Linear,
    q_width: usize,
    k_width: usize,
) -> (Linear, Linear, Linear) {
    let v_width = fused.output - q_width - k_width;
    let split = |range: std::ops::Range<usize>| Linear {
        weight: fused.weight[range.start * fused.input..range.end * fused.input].to_vec(),
        bias: fused
            .bias
            .as_ref()
            .map(|bias| bias[range.start..range.end].to_vec()),
        input: fused.input,
        output: range.end - range.start,
    };
    (
        split(0..q_width),
        split(q_width..q_width + k_width),
        split(q_width + k_width..q_width + k_width + v_width),
    )
}

/// Embedding-table lookup shared by the backbones: `tokens` -> row-major
/// `[len, width]`, refusing negative and out-of-vocab ids.
pub(crate) fn lookup_embeddings(
    table: &[f32],
    vocab_size: usize,
    width: usize,
    tokens: &[i32],
) -> Result<Vec<f32>> {
    tokens
        .iter()
        .map(|&token| {
            let row = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative token id {token}"),
            })?;
            let start = row
                .checked_mul(width)
                .filter(|&s| s + width <= table.len())
                .ok_or_else(|| SpeechError::Input {
                    why: format!("token id {token} outside vocab {vocab_size}"),
                })?;
            Ok(table[start..start + width].to_vec())
        })
        .collect::<Result<Vec<Vec<f32>>>>()
        .map(|rows| rows.concat())
}
