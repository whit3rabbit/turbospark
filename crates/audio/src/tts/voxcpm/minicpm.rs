//! The family-local MiniCPM backbone shared by the voxcpm base LM,
//! residual LM, patch encoder, and DiT denoiser.
//!
//! Port of `mlx_audio/tts/models/voxcpm/minicpm.py` at the pinned
//! mlx-audio commit. The layer math is the llama recipe (RMSNorm
//! pre-norm, no-bias GQA attention, SwiGLU), so this looks like the
//! shared [lm engine](../../lm/README.md), but four reference details
//! keep it family-local: the MiniCPM LongRoPE multiplies the cos/sin
//! tables by a `scaling_factor` and swaps whole factor tables on the
//! sequence-length threshold, `use_mup` rescales both residual adds by
//! `scale_depth / sqrt(layers)`, the vocabulary-free variants carry no
//! embed table at all, and the patch encoder needs batch-of-rows
//! bidirectional attention, which this port expresses as a blocked
//! mask over one folded sequence with per-block rope positions.
//!
//! Single batch, f32, cache layout head-major `[kv_heads, total, dim]`
//! following the crate convention.

use crate::lm::cache::KvCache;
use crate::lm::{Bias, Weights};
use crate::nn::{Linear, RmsNorm};
use crate::{Result, SpeechError};

/// Attention shape for one forward pass.
#[derive(Debug, Clone, Copy)]
pub enum Mask {
    /// Lower-triangular over the prefix (LM prefill and decode).
    Causal,
    /// Every query attends the whole buffer (DiT denoiser).
    Full,
    /// The sequence is `total / len` independent rows of `len`
    /// positions; attention stays inside the containing row and rope
    /// positions are the local row offsets. This is the reference
    /// `(B*T, P+1, H)` batch flattened into one sequence.
    Blocks { len: usize },
}

/// The MiniCPM configuration (`LMConfig`). Only the fields the
/// backbone reads are kept; `dim_model_base` never reaches the forward
/// path in the reference either.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    /// Zero means no embed table (the residual LM, encoder, and DiT
    /// decoder variants).
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_long_factor: Vec<f32>,
    pub rope_short_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
    pub max_position_embeddings: usize,
    /// The embedding multiplier (`embed_tokens(ids) * scale_emb`).
    pub scale_emb: i32,
    pub scale_depth: f32,
    pub use_mup: bool,
    /// Explicit head dim override (the voxcpm2 `kv_channels`).
    pub kv_channels: Option<usize>,
    /// Skip rotary entirely (the voxcpm2 `no_rope` residual LM).
    pub no_rope: bool,
}

impl Config {
    /// The reference dataclass defaults.
    pub fn from_json(value: &serde_json::Value) -> Self {
        let i = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let f = |key: &str, default: f64| {
            value
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(default) as f32
        };
        let factors = |key: &str| -> Vec<f32> {
            value
                .get(key)
                .and_then(serde_json::Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|v| v.as_f64().map(|x| x as f32))
                        .collect()
                })
                .unwrap_or_default()
        };
        Self {
            hidden_size: i("hidden_size", 1024),
            num_hidden_layers: i("num_hidden_layers", 24),
            num_attention_heads: i("num_attention_heads", 16),
            num_key_value_heads: i("num_key_value_heads", 2),
            intermediate_size: i("intermediate_size", 4096),
            vocab_size: i("vocab_size", 73448),
            rms_norm_eps: f("rms_norm_eps", 1e-5),
            rope_theta: f("rope_theta", 10000.0),
            rope_long_factor: factors("rope_long_factor"),
            rope_short_factor: factors("rope_short_factor"),
            original_max_position_embeddings: i("original_max_position_embeddings", 32768),
            max_position_embeddings: i("max_position_embeddings", 32768),
            scale_emb: value
                .get("scale_emb")
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(12),
            scale_depth: f("scale_depth", 1.4),
            use_mup: value
                .get("use_mup")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            kv_channels: value
                .get("kv_channels")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize),
            no_rope: value
                .get("no_rope")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        }
    }

    /// A vocabulary-free copy of this config (the constructor pattern
    /// `res_config.vocab_size = 0` and the encoder/dit overrides).
    pub fn with_overrides(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_hidden_layers: usize,
    ) -> Self {
        let mut config = self.clone();
        config.hidden_size = hidden_size;
        config.intermediate_size = intermediate_size;
        config.num_attention_heads = num_attention_heads;
        config.num_hidden_layers = num_hidden_layers;
        config.vocab_size = 0;
        config
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct Layer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// The loaded backbone. The embed table is empty for the
/// vocabulary-free variants.
pub struct MiniCpmModel {
    pub config: Config,
    embed_tokens: Vec<f32>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// `rope.inv_freq` when the checkpoint carries it; recomputed from
    /// the config otherwise (the reference computes it in `__init__`
    /// and the sanitize adds the missing buffers back).
    inv_freq: Option<Vec<f32>>,
    long_factor: Option<Vec<f32>>,
    short_factor: Option<Vec<f32>>,
    /// `sqrt(1 + log(max(scale, 1)) / log(original_max_position))`.
    scaling_factor: f32,
}

impl MiniCpmModel {
    /// Loads one backbone from `prefix` (for example `base_lm.`). The
    /// embed table loads only when `config.vocab_size > 0`.
    pub fn load(weights: &Weights, prefix: &str, config: Config) -> Result<Self> {
        let head_dim = config
            .kv_channels
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let kv_width = config.num_key_value_heads * head_dim;
        let q_width = config.num_attention_heads * head_dim;
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "hidden_size".into(),
                why: format!(
                    "{} does not divide by {} attention heads",
                    config.hidden_size, config.num_attention_heads
                ),
            });
        }
        if config.num_key_value_heads == 0
            || config.num_attention_heads % config.num_key_value_heads != 0
        {
            return Err(SpeechError::BadConfig {
                field: "num_key_value_heads".into(),
                why: format!(
                    "must be non-zero and divide num_attention_heads {} (got {})",
                    config.num_attention_heads, config.num_key_value_heads
                ),
            });
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}.");
            let linear = |name: &str, input: usize, output: usize| {
                weights.linear(&format!("{layer}{name}"), input, output, Bias::None)
            };
            let self_attn = Attention {
                q_proj: linear("self_attn.q_proj", config.hidden_size, q_width)?,
                k_proj: linear("self_attn.k_proj", config.hidden_size, kv_width)?,
                v_proj: linear("self_attn.v_proj", config.hidden_size, kv_width)?,
                o_proj: linear("self_attn.o_proj", q_width, config.hidden_size)?,
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim,
            };
            let mlp = Mlp {
                gate_proj: linear(
                    "mlp.gate_proj",
                    config.hidden_size,
                    config.intermediate_size,
                )?,
                up_proj: linear("mlp.up_proj", config.hidden_size, config.intermediate_size)?,
                down_proj: linear(
                    "mlp.down_proj",
                    config.intermediate_size,
                    config.hidden_size,
                )?,
            };
            layers.push(Layer {
                self_attn,
                mlp,
                input_layernorm: RmsNorm::new(
                    weights.tensor(&format!("{layer}input_layernorm"), &[config.hidden_size])?,
                    config.rms_norm_eps,
                ),
                post_attention_layernorm: RmsNorm::new(
                    weights.tensor(
                        &format!("{layer}post_attention_layernorm"),
                        &[config.hidden_size],
                    )?,
                    config.rms_norm_eps,
                ),
            });
        }
        let rope_prefix = format!("{prefix}rope.");
        let half = head_dim / 2;
        let saved = |name: &str| -> Result<Option<Vec<f32>>> {
            if config.no_rope {
                return Ok(None);
            }
            if weights.contains_absolute(&format!("{rope_prefix}{name}")) {
                Ok(Some(weights.tensor_absolute(
                    &format!("{rope_prefix}{name}"),
                    &[half],
                )?))
            } else {
                Ok(None)
            }
        };
        let inv_freq = saved("inv_freq")?;
        let long_factor = saved("long_factor")?;
        let short_factor = saved("short_factor")?;
        let scale =
            config.max_position_embeddings as f64 / config.original_max_position_embeddings as f64;
        let scaling_factor = (1.0
            + (scale.max(1.0)).ln() / (config.original_max_position_embeddings as f64).ln())
        .sqrt() as f32;
        let embed_tokens = if config.vocab_size > 0 {
            weights.embedding(
                &format!("{prefix}embed_tokens"),
                config.vocab_size,
                config.hidden_size,
            )?
        } else {
            Vec::new()
        };
        Ok(Self {
            norm: RmsNorm::new(
                weights.tensor(&format!("{prefix}norm"), &[config.hidden_size])?,
                config.rms_norm_eps,
            ),
            config,
            embed_tokens,
            layers,
            inv_freq,
            long_factor,
            short_factor,
            scaling_factor,
        })
    }

    /// Embeds token ids and multiplies by `scale_emb` (the reference
    /// `embed_tokens(ids) * scale_emb`), returning
    /// `[ids.len(), hidden_size]`.
    pub fn embed(&self, tokens: &[i32], scale_emb: f32) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        for &token in tokens {
            let row = usize::try_from(token).map_err(|_| SpeechError::Input {
                why: format!("negative token id {token}"),
            })?;
            let start = row
                .checked_mul(hidden)
                .filter(|&s| s + hidden <= self.embed_tokens.len())
                .ok_or_else(|| SpeechError::Input {
                    why: format!("token id {token} outside vocab {}", self.config.vocab_size),
                })?;
            out.extend(
                self.embed_tokens[start..start + hidden]
                    .iter()
                    .map(|v| v * scale_emb),
            );
        }
        Ok(out)
    }

    fn inv_freq(&self) -> Vec<f32> {
        if let Some(saved) = &self.inv_freq {
            return saved.clone();
        }
        let half = self.config.hidden_size / self.config.num_attention_heads / 2;
        (0..half)
            .map(|i| 1.0 / self.config.rope_theta.powf(i as f32 / half as f32))
            .collect()
    }

    fn factors(&self, long: bool) -> Vec<f32> {
        let from_config = if long {
            &self.config.rope_long_factor
        } else {
            &self.config.rope_short_factor
        };
        let saved = if long {
            self.long_factor.as_ref()
        } else {
            self.short_factor.as_ref()
        };
        saved
            .or(if from_config.is_empty() {
                None
            } else {
                Some(from_config)
            })
            .cloned()
            .unwrap_or_default()
    }

    /// cos/sin pairs for local positions `0..seq_len`, scaled by the
    /// LongRoPE factor; `[(cos, sin); seq_len * half]`.
    fn cos_sin(&self, seq_len: usize) -> Vec<(f32, f32)> {
        let dim = self.config.hidden_size / self.config.num_attention_heads;
        let half = dim / 2;
        let long = seq_len > self.config.original_max_position_embeddings;
        let factors = self.factors(long);
        let inv_freq = self.inv_freq();
        let mut out = Vec::with_capacity(seq_len * half);
        for position in 0..seq_len {
            for d in 0..half {
                let angle = position as f32 * (1.0 / factors[d]) * inv_freq[d];
                out.push((
                    angle.cos() * self.scaling_factor,
                    angle.sin() * self.scaling_factor,
                ));
            }
        }
        out
    }

    /// Runs the backbone over one sequence of `len` positions and
    /// returns the post-norm hidden states `[len, hidden_size]`.
    /// `embeds` is `[len, hidden_size]` row-major. When `cache` is
    /// present it must hold one entry per layer and each entry advances
    /// with the call.
    pub fn forward(
        &mut self,
        embeds: &[f32],
        len: usize,
        mut cache: Option<&mut Vec<KvCache>>,
        mask: Mask,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        if embeds.len() != len * hidden {
            return Err(SpeechError::Input {
                why: format!("embed length {} is not {len} x {hidden}", embeds.len()),
            });
        }
        if let Some(cache) = cache.as_deref() {
            if cache.len() != self.layers.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "{} layer caches provided for {} layers",
                        cache.len(),
                        self.layers.len()
                    ),
                });
            }
        }
        let offset = match &cache {
            Some(cache) => cache.first().map(|kv| kv.offset).unwrap_or(0),
            None => 0,
        };
        // The reference builds the cos/sin tables for `seq_len` =
        // max position + 1 and selects the factor table on that
        // threshold; under the blocked mask the per-row max position is
        // the block length, not the folded total. `no_rope` skips the
        // tables and the rotation entirely.
        let seq_len = match mask {
            Mask::Blocks { len: block } => block,
            _ => offset + len,
        };
        let cos_sin = if self.config.no_rope {
            Vec::new()
        } else {
            self.cos_sin(seq_len)
        };
        let mup_scale = if self.config.use_mup {
            self.config.scale_depth / (self.config.num_hidden_layers as f32).sqrt()
        } else {
            1.0
        };

        let mut h = embeds.to_vec();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_deref_mut().map(|c| &mut c[index]);
            let mut normed = h.clone();
            layer.input_layernorm.apply(&mut normed, len);
            let attn =
                layer
                    .self_attn
                    .forward(&normed, len, offset, mask, &cos_sin, layer_cache)?;
            for (row, out) in attn.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value * mup_scale;
                }
            }
            let mut fed = h.clone();
            layer.post_attention_layernorm.apply(&mut fed, len);
            let mlp = layer.mlp.forward(&fed, len);
            for (row, out) in mlp.chunks_exact(hidden).enumerate() {
                let base = row * hidden;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value * mup_scale;
                }
            }
        }
        self.norm.apply(&mut h, len);
        Ok(h)
    }
}

impl Attention {
    /// Projects, applies LongRoPE, appends to the cache, and runs
    /// masked GQA attention. Returns `[len, heads * head_dim]`. The
    /// cos/sin table arrives precomputed (one table per forward, shared
    /// by every layer and head).
    fn forward(
        &self,
        x: &[f32],
        len: usize,
        offset: usize,
        mask: Mask,
        cos_sin: &[(f32, f32)],
        cache: Option<&mut KvCache>,
    ) -> Result<Vec<f32>> {
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let dim = self.head_dim;
        let half = dim / 2;
        let q = self.q_proj.forward(x, len);
        let k = self.k_proj.forward(x, len);
        let v = self.v_proj.forward(x, len);

        // Rope with the reference rotate-half convention; the local
        // position is the folded offset for causal/full and the row
        // offset for the blocked mask. With `no_rope` the cos/sin table
        // is empty and the values pass through.
        if cos_sin.is_empty() {
            let qh = head_major(&q, len, heads, dim);
            let kh = head_major(&k, len, kv_heads, dim);
            let vh = head_major(&v, len, kv_heads, dim);
            return attend(
                &qh,
                &kh,
                &vh,
                len,
                offset,
                mask,
                heads,
                kv_heads,
                dim,
                cache,
                &self.o_proj,
            );
        }
        let local = |i: usize| match mask {
            Mask::Blocks { len: block } => i % block,
            _ => offset + i,
        };
        // The keys carry `kv_heads * dim` per row, the queries
        // `heads * dim`; every head block holds its own rotate-half
        // pairs at (d, d + half).
        let rotate = |values: &[f32], rows: usize, block_heads: usize| {
            let width = block_heads * dim;
            let mut out = values.to_vec();
            for row in 0..rows {
                let position = local(row);
                for head in 0..block_heads {
                    let base = row * width + head * dim;
                    for d in 0..half {
                        let (cos, sin) = cos_sin[position * half + d];
                        let (va, vb) = (values[base + d], values[base + half + d]);
                        out[base + d] = va * cos - vb * sin;
                        out[base + half + d] = vb * cos + va * sin;
                    }
                }
            }
            out
        };
        let qh = head_major(&rotate(&q, len, heads), len, heads, dim);
        let kh = head_major(&rotate(&k, len, kv_heads), len, kv_heads, dim);
        let vh = head_major(&v, len, kv_heads, dim);
        attend(
            &qh,
            &kh,
            &vh,
            len,
            offset,
            mask,
            heads,
            kv_heads,
            dim,
            cache,
            &self.o_proj,
        )
    }
}

/// The masked GQA attention tail shared by the rope and no-rope paths:
/// append per head to the cache, softmax-attend, join heads, and apply
/// the output projection.
#[allow(clippy::too_many_arguments)]
fn attend(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    len: usize,
    offset: usize,
    mask: Mask,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    mut cache: Option<&mut KvCache>,
    o_proj: &Linear,
) -> Result<Vec<f32>> {
    let hidden = heads * dim;
    // Append per head; the shared KvCache owns the head-blocked insert
    // (a flat append would land the new rows inside the wrong head's
    // region).
    let (keys, values) = match &mut cache {
        Some(cache) => {
            let (keys, values) = cache.update_and_fetch(kh, vh, kv_heads, len, dim);
            (keys, values)
        }
        None => (kh, vh),
    };
    let total = keys.len() / (kv_heads * dim);

    // Visible slot range for query row i.
    let range = |i: usize| -> (usize, usize) {
        match mask {
            Mask::Full => (0, total - 1),
            Mask::Blocks { len: block } => {
                let start = (i / block) * block;
                (start, (start + block - 1).min(total - 1))
            }
            Mask::Causal => (0, (offset + i).min(total - 1)),
        }
    };

    let rep = heads / kv_heads;
    let scale = (dim as f64).powf(-0.5) as f32;
    let mut out = vec![0.0f32; heads * len * dim];
    let mut scores = vec![0.0f32; total];
    for head in 0..heads {
        let kv = head / rep;
        let q_head = &qh[head * len * dim..(head + 1) * len * dim];
        let k_head = &keys[kv * total * dim..(kv + 1) * total * dim];
        let v_head = &values[kv * total * dim..(kv + 1) * total * dim];
        for i in 0..len {
            let (low, high) = range(i);
            let q_row = &q_head[i * dim..(i + 1) * dim];
            let mut max = f32::NEG_INFINITY;
            for j in low..=high {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_row[d] * k_head[j * dim + d];
                }
                let score = dot * scale;
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
            let out_row = &mut out[(head * len + i) * dim..(head * len + i + 1) * dim];
            for j in low..=high {
                let weight = scores[j] * inv;
                for d in 0..dim {
                    out_row[d] += weight * v_head[j * dim + d];
                }
            }
        }
    }

    // Back to [len, heads * dim] for the output projection.
    let mut joined = vec![0.0f32; len * hidden];
    for t in 0..len {
        for head in 0..heads {
            let src = (head * len + t) * dim;
            let dst = t * hidden + head * dim;
            joined[dst..dst + dim].copy_from_slice(&out[src..src + dim]);
        }
    }
    Ok(o_proj.forward(&joined, len))
}

impl Mlp {
    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
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

/// Reinterprets `[rows, heads * dim]` as `[heads, rows, dim]`.
fn head_major(x: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * rows * dim];
    for row in 0..rows {
        for head in 0..heads {
            let src = row * heads * dim + head * dim;
            let dst = (head * rows + row) * dim;
            out[dst..dst + dim].copy_from_slice(&x[src..src + dim]);
        }
    }
    out
}
