//! arktts: the Audio8 TTS preview models (DualAR semantic + codec heads).
//!
//! Port of `mlx_audio/tts/models/arktts/arktts.py` at the pinned mlx-audio
//! commit. A slow AR stack predicts one semantic token per audio frame; a
//! fast AR stack predicts that frame's residual codec codebooks; the
//! bundled 44.1 kHz codec ([`codec`]) turns the 10-book code grid into
//! audio. Two checkpoints share the `arktts` model_type and are selected by
//! `slow_backbone`: the default pure-attention 0.6b stack (full-vocabulary
//! semantic logits tied to the text embedding) and the `falcon_h1` 0.1b
//! hybrid stack consumed from [`crate::lm::falcon_h1`] through its
//! `forward_stack` embedding seam with a compact `codebook_size + 1`
//! semantic head.
//!
//! Established family conventions: batch 1, non-streaming, text
//! tokenization caller-owned (the reference loads a Mistral BPE tokenizer;
//! its exact prompt strings are exposed through [`prompt_segments`]), greedy
//! mode deterministic and fixture-verified, sampled draws replay recorded
//! uniforms through the reference's exponential-race rule (mlx RNG is not
//! reproduced).

pub mod codec;

use serde_json::Value;

use crate::lm::falcon_h1::{self, FalconH1Model};
use crate::nn::Linear;
use crate::ops;
use crate::{Result, SpeechError};

use crate::lm::{Bias, Weights};
use crate::nn::argmax;

/// f32 value rounded through bf16 (nearest even), the reference's rope
/// table dtype round trip.
pub(super) fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// `[len, heads * dim]` -> `[heads, len, dim]`.
pub(super) fn head_major(x: &[f32], len: usize, heads: usize, dim: usize) -> Vec<f32> {
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

/// GPT-J-style interleaved rope with per-position (cos, sin) tables
/// `[len, dim / 2]`, in place over `[heads, len, dim]`.
pub(super) fn apply_interleaved_rope(
    x: &mut [f32],
    heads: usize,
    len: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    ops::rope_interleaved(x, heads, len, dim, cos, sin);
}

/// `[frames, channels]` -> `[channels, frames]`.
pub(super) fn to_channel_major(x: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * channels];
    for t in 0..frames {
        for c in 0..channels {
            out[c * frames + t] = x[t * channels + c];
        }
    }
    out
}

/// `[channels, frames]` -> `[frames, channels]`.
pub(super) fn to_row_major(x: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * channels];
    for c in 0..channels {
        for t in 0..frames {
            out[t * channels + c] = x[c * frames + t];
        }
    }
    out
}

/// Weightless RMS norm over the last axis (`ArkttsRMSNorm`): the scale is a
/// learned parameter loaded from the checkpoint.
#[derive(Debug, Clone)]
struct Norm {
    weight: Vec<f32>,
    epsilon: f32,
}

impl Norm {
    fn load(weights: &Weights, name: &str, width: usize, epsilon: f32) -> Result<Self> {
        // `tensor` loads `{name}.weight` itself; callers pass the base.
        Ok(Self {
            weight: weights.tensor(name, &[width])?,
            epsilon,
        })
    }

    /// In place over `[rows, width]`.
    fn apply(&self, x: &mut [f32], rows: usize) {
        let width = self.weight.len();
        for row in 0..rows {
            let base = row * width;
            let mut sum = 0.0f32;
            for value in &x[base..base + width] {
                sum += value * value;
            }
            let inv = 1.0 / (sum / width as f32 + self.epsilon).sqrt();
            for (value, weight) in x[base..base + width].iter_mut().zip(&self.weight) {
                *value = *value * inv * weight;
            }
        }
    }
}

/// Full model configuration (`ModelConfig` in the reference).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub slow_backbone: String,
    pub embedding_multiplier: f32,
    pub mlp_multipliers: Vec<f32>,
    pub ssm_multipliers: Vec<f32>,
    pub mamba_fields: Value,
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub attention_qkv_bias: bool,
    pub attention_qk_norm: bool,
    pub attention_o_bias: bool,
    pub tie_word_embeddings: bool,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_begin_id: usize,
    pub semantic_end_id: usize,
    pub n_fast_layer: usize,
    pub fast_dim: usize,
    pub fast_n_head: usize,
    pub fast_n_local_heads: usize,
    pub fast_head_dim: usize,
    pub fast_intermediate_size: usize,
    pub fast_attention_qkv_bias: bool,
    pub fast_attention_qk_norm: bool,
    pub fast_attention_o_bias: bool,
    pub norm_fastlayer_input: bool,
    pub ras_window_size: usize,
    pub ras_temperature: f32,
    pub ras_top_p: f32,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub sample_rate: usize,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let text = |key: &str, default: &str| -> String {
            value
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_owned()
        };
        let u = |key: &str, default: u64| -> usize {
            value.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
        };
        let f = |key: &str, default: f64| -> f32 {
            value.get(key).and_then(Value::as_f64).unwrap_or(default) as f32
        };
        let b = |key: &str, default: bool| -> bool {
            value.get(key).and_then(Value::as_bool).unwrap_or(default)
        };
        let mamba_fields = value.clone();
        Ok(Self {
            slow_backbone: text("slow_backbone", "arktts"),
            embedding_multiplier: f("embedding_multiplier", 1.0),
            mlp_multipliers: value
                .get("mlp_multipliers")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_f64)
                        .map(|v| v as f32)
                        .collect()
                })
                .unwrap_or_else(|| vec![1.0, 1.0]),
            ssm_multipliers: value
                .get("ssm_multipliers")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_f64)
                        .map(|v| v as f32)
                        .collect()
                })
                .unwrap_or_else(|| vec![1.0; 5]),
            mamba_fields,
            vocab_size: u("vocab_size", 155_776),
            dim: u("dim", 896),
            n_layer: u("n_layer", 24),
            n_head: u("n_head", 14),
            n_local_heads: u("n_local_heads", 2),
            head_dim: u("head_dim", 64),
            intermediate_size: u("intermediate_size", 4864),
            max_seq_len: u("max_seq_len", 2048),
            rope_base: f("rope_base", 1_000_000.0),
            norm_eps: f("norm_eps", 1e-6),
            attention_qkv_bias: b("attention_qkv_bias", true),
            attention_qk_norm: b("attention_qk_norm", false),
            attention_o_bias: b("attention_o_bias", false),
            tie_word_embeddings: b("tie_word_embeddings", true),
            codebook_size: u("codebook_size", 4096),
            num_codebooks: u("num_codebooks", 10),
            semantic_begin_id: u("semantic_begin_id", 151_678),
            semantic_end_id: u("semantic_end_id", 155_773),
            n_fast_layer: u("n_fast_layer", 4),
            fast_dim: u("fast_dim", 896),
            fast_n_head: u("fast_n_head", 14),
            fast_n_local_heads: u("fast_n_local_heads", 2),
            fast_head_dim: u("fast_head_dim", 64),
            fast_intermediate_size: u("fast_intermediate_size", 4864),
            fast_attention_qkv_bias: b("fast_attention_qkv_bias", false),
            fast_attention_qk_norm: b("fast_attention_qk_norm", false),
            fast_attention_o_bias: b("fast_attention_o_bias", false),
            norm_fastlayer_input: b("norm_fastlayer_input", true),
            ras_window_size: u("ras_window_size", 10),
            ras_temperature: f("ras_temperature", 1.0),
            ras_top_p: f("ras_top_p", 0.9),
            eos_token_id: u("eos_token_id", 151_645),
            pad_token_id: u("pad_token_id", 151_643),
            sample_rate: u("sample_rate", 44_100),
        })
    }

    pub fn uses_falcon_slow(&self) -> bool {
        self.slow_backbone == "falcon_h1"
    }
}

/// One transformer block of the pure-attention stacks (slow and fast share
/// the class).
struct Block {
    wqkv: Linear,
    wo: Linear,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    attention_norm: Norm,
    ffn_norm: Norm,
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
    /// The model rope base; the reference builds both the slow and the fast
    /// table with `config.rope_base`.
    rope_base: f32,
    /// Static full-buffer KV cache `[local_heads, max_len, head_dim]`,
    /// `None` outside generation.
    cache: Option<(Vec<f32>, Vec<f32>)>,
}

impl Block {
    fn attention(
        &mut self,
        x: &[f32],
        frames: usize,
        positions: &[usize],
        key_mask: &[bool],
        cache_start: usize,
        total_len: usize,
    ) -> Result<Vec<f32>> {
        let q_width = self.n_head * self.head_dim;
        let kv_width = self.n_local_heads * self.head_dim;
        let qkv = self.wqkv.forward(x, frames);
        // Split row-wise into PACKED [frames, width] buffers: the flat qkv
        // rows keep the full fused stride, and head_major indexes a packed
        // `[frames, heads, dim]` layout.
        let total_row = q_width + 2 * kv_width;
        let mut q = Vec::with_capacity(frames * q_width);
        let mut k = Vec::with_capacity(frames * kv_width);
        let mut v = Vec::with_capacity(frames * kv_width);
        for row in 0..frames {
            let row_values = &qkv[row * total_row..(row + 1) * total_row];
            q.extend_from_slice(&row_values[..q_width]);
            k.extend_from_slice(&row_values[q_width..q_width + kv_width]);
            v.extend_from_slice(&row_values[q_width + kv_width..]);
        }
        let mut qh = head_major(&q, frames, self.n_head, self.head_dim);
        let mut kh = head_major(&k, frames, self.n_local_heads, self.head_dim);
        let vh = head_major(&v, frames, self.n_local_heads, self.head_dim);
        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q_norm.apply(&mut qh, self.n_head * frames);
            k_norm.apply(&mut kh, self.n_local_heads * frames);
        }
        // Rope uses the POSITION ids (cumsum of the mask), not raw offsets.
        let half = self.head_dim / 2;
        let mut cos = vec![0.0f32; frames * half];
        let mut sin = vec![0.0f32; frames * half];
        for (row, &position) in positions.iter().enumerate() {
            for d in 0..half {
                let freq = 1.0 / (self.rope_base.powf(2.0 * d as f32 / self.head_dim as f32));
                let angle = round_bf16((position as f32 * freq).cos());
                cos[row * half + d] = angle;
                sin[row * half + d] = round_bf16((position as f32 * freq).sin());
            }
        }
        apply_interleaved_rope(&mut qh, self.n_head, frames, self.head_dim, &cos, &sin);
        apply_interleaved_rope(
            &mut kh,
            self.n_local_heads,
            frames,
            self.head_dim,
            &cos,
            &sin,
        );

        let (keys, values): (&[f32], &[f32]) = match &mut self.cache {
            Some((keys, values)) => {
                let stride = self.head_dim;
                for head in 0..self.n_local_heads {
                    let dst = head * total_len * stride + cache_start * stride;
                    let src = head * frames * stride;
                    keys[dst..dst + frames * stride]
                        .copy_from_slice(&kh[src..src + frames * stride]);
                    values[dst..dst + frames * stride]
                        .copy_from_slice(&vh[src..src + frames * stride]);
                }
                (keys, values)
            }
            None => (&kh, &vh),
        };
        let total = total_len.max(frames);
        let rep = self.n_head / self.n_local_heads;
        let scale = (self.head_dim as f64).powf(-0.5) as f32;
        let mut out = vec![0.0f32; frames * q_width];
        let mut scores = vec![0.0f32; total];
        for head in 0..self.n_head {
            let kv_head = head / rep;
            for t in 0..frames {
                let q_row = &qh
                    [(head * frames + t) * self.head_dim..(head * frames + t + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                for j in 0..total {
                    // Causal in ABSOLUTE positions: the query sits at
                    // cache_start + t, so the multi-frame prefill stays
                    // causal and a decode step attends every written slot.
                    let causal = j <= cache_start + t;
                    let allowed = causal && key_mask.get(j).copied().unwrap_or(false);
                    if !allowed {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_row = &keys[(kv_head * total + j) * self.head_dim
                        ..(kv_head * total + j + 1) * self.head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d] * k_row[d];
                    }
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    let weight = if *score == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (*score - max).exp()
                    };
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                for (j, weight) in scores.iter_mut().enumerate() {
                    let w = *weight * inv;
                    if w == 0.0 {
                        continue;
                    }
                    let v_row = &values[(kv_head * total + j) * self.head_dim
                        ..(kv_head * total + j + 1) * self.head_dim];
                    let out_row = &mut out[t * q_width + head * self.head_dim
                        ..t * q_width + (head + 1) * self.head_dim];
                    for (d, &v) in v_row.iter().enumerate() {
                        out_row[d] += w * v;
                    }
                }
            }
        }
        Ok(self.wo.forward(&out, frames))
    }

    fn forward(
        &mut self,
        x: &[f32],
        frames: usize,
        positions: &[usize],
        key_mask: &[bool],
        cache_start: usize,
        total_len: usize,
    ) -> Result<Vec<f32>> {
        let mut attended = x.to_vec();
        self.attention_norm.apply(&mut attended, frames);
        let attn = self.attention(
            &attended,
            frames,
            positions,
            key_mask,
            cache_start,
            total_len,
        )?;
        let mut hidden = x.to_vec();
        for (index, value) in attn.iter().enumerate() {
            hidden[index] += value;
        }
        let mut fed = hidden.clone();
        self.ffn_norm.apply(&mut fed, frames);
        let gate = self.w1.forward(&fed, frames);
        let up = self.w3.forward(&fed, frames);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        let ff = self.w2.forward(&fused, frames);
        for (index, value) in ff.iter().enumerate() {
            hidden[index] += value;
        }
        Ok(hidden)
    }
}

/// The loaded arktts model (both flavors) plus the fast AR stack.
pub struct ArkttsModel {
    pub config: ModelConfig,
    codebook_embeddings: Vec<f32>,
    falcon: Option<FalconH1Model>,
    /// The text embedding table: `model.embeddings` for the pure-attention
    /// flavor, `model.slow.embed_tokens` for the falcon flavor.
    embeddings: Option<Vec<f32>>,
    layers: Vec<Block>,
    norm: Option<Norm>,
    /// The falcon flavor's compact semantic head.
    semantic_output: Option<Linear>,
    fast_project_in: Option<Linear>,
    fast_embeddings: Vec<f32>,
    fast_layers: Vec<Block>,
    fast_norm: Norm,
    fast_output: Linear,
}

impl ArkttsModel {
    pub fn load(weights: &Weights, config: &ModelConfig) -> Result<Self> {
        let codebook_embeddings = weights.embedding(
            "model.codebook_embeddings",
            config.codebook_size * config.num_codebooks,
            config.dim,
        )?;
        let (falcon, embeddings, layers, norm, semantic_output) = if config.uses_falcon_slow() {
            // Unity muP multipliers only; the embedding multiplier stays
            // runtime-applied to the composite.
            check_unity(config, "attention_in_multiplier")?;
            check_unity(config, "attention_out_multiplier")?;
            check_unity(config, "key_multiplier")?;
            check_unity(config, "ssm_in_multiplier")?;
            check_unity(config, "ssm_out_multiplier")?;
            for (name, values) in [
                ("mlp_multipliers", &config.mlp_multipliers),
                ("ssm_multipliers", &config.ssm_multipliers),
            ] {
                if values.iter().any(|&v| v != 1.0) {
                    return Err(SpeechError::Unsupported {
                        why: format!(
                            "arktts's falcon_h1 backbone supports unity muP multipliers only, but {name} leaves 1.0"
                        ),
                    });
                }
            }
            let model = FalconH1Model::load_without_embed_table(
                weights,
                falcon_h1::from_json(&falcon_json(config)?)?,
                // The prefix covers the layer tensors and carries the
                // trailing dot (`load_impl` joins it directly).
                "model.slow.",
            )?;
            // The slow stack's text table lives INSIDE FalconSlowStack
            // (`model.slow.embed_tokens`): the composite embedding sums the
            // codebook rows into it before the stack runs, and the
            // reference sanitize deliberately does not fold the embedding
            // multiplier into it.
            let embeddings =
                weights.embedding("model.slow.embed_tokens", config.vocab_size, config.dim)?;
            let head = weights.linear(
                "model.semantic_output",
                config.dim,
                config.codebook_size + 1,
                Bias::None,
            )?;
            (Some(model), Some(embeddings), Vec::new(), None, Some(head))
        } else {
            let embeddings =
                weights.embedding("model.embeddings", config.vocab_size, config.dim)?;
            let mut layers = Vec::with_capacity(config.n_layer);
            for index in 0..config.n_layer {
                layers.push(load_block(
                    weights,
                    &format!("model.layers.{index}"),
                    config.dim,
                    config.intermediate_size,
                    config.n_head,
                    config.n_local_heads,
                    config.head_dim,
                    config.attention_qkv_bias,
                    config.attention_o_bias,
                    config.attention_qk_norm,
                    config.norm_eps,
                    config.rope_base,
                )?);
            }
            let norm = Norm::load(weights, "model.norm", config.dim, config.norm_eps)?;
            (None, Some(embeddings), layers, Some(norm), None)
        };

        let fast_project_in = if config.fast_dim != config.dim {
            Some(weights.linear(
                "model.fast_project_in",
                config.dim,
                config.fast_dim,
                Bias::Required,
            )?)
        } else {
            None
        };
        let fast_embeddings = weights.embedding(
            "model.fast_embeddings",
            config.codebook_size,
            config.fast_dim,
        )?;
        let mut fast_layers = Vec::with_capacity(config.n_fast_layer);
        for index in 0..config.n_fast_layer {
            fast_layers.push(load_block(
                weights,
                &format!("model.fast_layers.{index}"),
                config.fast_dim,
                config.fast_intermediate_size,
                config.fast_n_head,
                config.fast_n_local_heads,
                config.fast_head_dim,
                config.fast_attention_qkv_bias,
                config.fast_attention_o_bias,
                config.fast_attention_qk_norm,
                config.norm_eps,
                config.rope_base,
            )?);
        }
        let fast_norm = Norm::load(weights, "model.fast_norm", config.fast_dim, config.norm_eps)?;
        let fast_output = weights.linear(
            "model.fast_output",
            config.fast_dim,
            config.codebook_size,
            Bias::None,
        )?;
        Ok(Self {
            config: config.clone(),
            codebook_embeddings,
            falcon,
            embeddings,
            layers,
            norm,
            semantic_output,
            fast_project_in,
            fast_embeddings,
            fast_layers,
            fast_norm,
            fast_output,
        })
    }

    /// The UNSCALED composite embedding: text embedding of column 0 plus
    /// the (semantic-gated) sum of the ten codebook embeddings.
    fn embed(&self, input_ids: &[i32], books: &[i32]) -> Vec<f32> {
        let width = self.config.dim;
        let semantic = input_ids[0] >= self.config.semantic_begin_id as i32
            && input_ids[0] <= self.config.semantic_end_id as i32;
        let text_base = input_ids[0] as usize * width;
        let mut out = self.text_embeddings()[text_base..text_base + width].to_vec();
        if semantic {
            for (index, &code) in books.iter().enumerate() {
                let row = (code as usize + index * self.config.codebook_size) * width;
                for (d, &v) in self.codebook_embeddings[row..row + width]
                    .iter()
                    .enumerate()
                {
                    out[d] += v;
                }
            }
        }
        out
    }

    fn text_embeddings(&self) -> &[f32] {
        self.embeddings
            .as_deref()
            .expect("both flavors own a text embedding table")
    }

    /// Prefill over the full prompt column stack; returns the last-position
    /// semantic logits and the last hidden state (fast-AR input).
    fn slow_prefill(
        &mut self,
        prompt: &[Vec<i32>],
        frames: usize,
        caches: Option<&mut Vec<falcon_h1::FalconLayerCache>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows: Vec<Vec<i32>> = (0..frames)
            .map(|t| {
                let mut row = Vec::with_capacity(prompt.len());
                for column in prompt {
                    row.push(column[t]);
                }
                row
            })
            .collect();
        if self.falcon.is_some() {
            let mut hidden = Vec::with_capacity(frames * self.config.dim);
            for row in &rows {
                hidden.extend(self.embed(&row[0..1], &row[1..]));
            }
            // The multiplier applies to the composite before the stack.
            for value in hidden.iter_mut() {
                *value *= self.config.embedding_multiplier;
            }
            // The reference prefill runs WITH the fresh cache like every
            // other step (`_setup_generation_caches` runs before the first
            // `_slow_step`).
            let normalized = match self.falcon.as_mut() {
                Some(falcon) => falcon.forward_stack(&hidden, caches)?,
                None => unreachable!("the falcon branch runs on the falcon flavor"),
            };
            let last = &normalized[(frames - 1) * self.config.dim..frames * self.config.dim];
            let logits = self.semantic_output.as_ref().unwrap().forward(last, 1);
            return Ok((logits, last.to_vec()));
        }
        // Pure-attention path.
        let width = self.config.dim;
        let mut hidden = Vec::with_capacity(frames * width);
        for row in &rows {
            hidden.extend(self.embed(&row[0..1], &row[1..]));
        }
        let positions: Vec<usize> = (0..frames).collect();
        let key_mask: Vec<bool> = vec![true; frames];
        let mut current = hidden;
        for layer in &mut self.layers {
            // total_len must be the FULL cache stride (max_seq_len), or the
            // per-head cache rows are written compacted and the decode
            // steps read head 1 from an all-zero region.
            current = layer.forward(
                &current,
                frames,
                &positions,
                &key_mask,
                0,
                self.config.max_seq_len,
            )?;
        }
        // The generation prefill runs through the reference `_slow_step`,
        // which feeds the NORMALIZED last hidden to the fast AR under
        // `norm_fastlayer_input` (the standalone parity `__call__` returns
        // the raw hidden, but the generation loop never uses it).
        let raw_last = current[(frames - 1) * width..frames * width].to_vec();
        let norm = self.norm.as_ref().unwrap();
        norm.apply(&mut current, frames);
        let last_normalized = &current[(frames - 1) * width..frames * width];
        // Tied head: logits = normalized @ embeddings.T
        let logits = ops::linear(
            last_normalized,
            self.embeddings.as_ref().unwrap(),
            None,
            1,
            width,
            self.config.vocab_size,
        );
        let fast_hidden = if self.config.norm_fastlayer_input {
            last_normalized.to_vec()
        } else {
            raw_last
        };
        Ok((logits, fast_hidden))
    }

    /// One cached slow step over the new column.
    fn slow_step(
        &mut self,
        semantic: i32,
        books: &[i32],
        cache_start: usize,
        position: usize,
        falcon_step: Option<&mut Vec<falcon_h1::FalconLayerCache>>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let width = self.config.dim;
        let hidden_unscaled = self.embed(&[semantic], books);
        if let Some(falcon) = self.falcon.as_mut() {
            let mut hidden = hidden_unscaled;
            for value in hidden.iter_mut() {
                *value *= self.config.embedding_multiplier;
            }
            let normalized = falcon.forward_stack(&hidden, falcon_step.map(|c| &mut *c))?;
            let logits = self
                .semantic_output
                .as_ref()
                .expect("the falcon flavor owns the semantic head")
                .forward(&normalized, 1);
            return Ok((logits, normalized));
        }
        let _ = falcon_step;
        let key_mask: Vec<bool> = vec![true; cache_start + 1];
        let mut current = hidden_unscaled;
        for layer in &mut self.layers {
            current = layer.forward(
                &current,
                1,
                &[position],
                &key_mask,
                cache_start,
                self.config.max_seq_len,
            )?;
        }
        let norm = self.norm.as_ref().unwrap();
        norm.apply(&mut current, 1);
        let logits = ops::linear(
            &current,
            self.embeddings.as_ref().unwrap(),
            None,
            1,
            width,
            self.config.vocab_size,
        );
        Ok((logits, current))
    }

    /// The fast AR pass: codebook 0 is pinned from the semantic token, the
    /// remaining books decode one at a time through the cached fast stack.
    fn generate_codebooks(
        &mut self,
        slow_hidden: &[f32],
        semantic: i32,
        params: &SamplingParams,
        uniforms: &mut Uniforms,
    ) -> Vec<i32> {
        let config = &self.config;
        let mut hidden = slow_hidden.to_vec();
        if let Some(project) = &self.fast_project_in {
            hidden = project.forward(&hidden, 1);
        }
        // Position 0 runs the fast stack on the projected slow hidden,
        // warming the cache the later positions attend.
        {
            let key_mask: Vec<bool> = vec![true; config.num_codebooks];
            let mut h = hidden;
            for layer in &mut self.fast_layers {
                h = layer
                    .forward(&h, 1, &[0], &key_mask, 0, config.num_codebooks)
                    .expect("fast warmup");
            }
        }
        let mut codes = vec![0i32; config.num_codebooks];
        codes[0] = if config.uses_falcon_slow() {
            semantic.clamp(0, config.codebook_size as i32 - 1)
        } else {
            (semantic - config.semantic_begin_id as i32).clamp(0, config.codebook_size as i32 - 1)
        };
        let mut current = codes[0];
        for (position, slot) in codes.iter_mut().enumerate().skip(1) {
            let h = self.fast_embeddings
                [current as usize * config.fast_dim..(current as usize + 1) * config.fast_dim]
                .to_vec();
            // Run the fast stack with the static cache over num_codebooks.
            let key_mask: Vec<bool> = vec![true; config.num_codebooks];
            let mut h = h;
            for layer in &mut self.fast_layers {
                h = layer
                    .forward(
                        &h,
                        1,
                        &[position],
                        &key_mask,
                        position,
                        config.num_codebooks,
                    )
                    .expect("fast layer");
            }
            self.fast_norm.apply(&mut h, 1);
            let scores = self.fast_output.forward(&h, 1);
            let processed = legacy_top_k_top_p(&scores, params.top_k, params.top_p);
            let processed: Vec<f32> = processed
                .iter()
                .map(|&s| s / params.temperature.max(1e-5))
                .collect();
            current = if params.do_sample {
                exponential_race(&processed, &uniforms.next_n(processed.len()))
            } else {
                argmax(&processed) as i32
            };
            *slot = current;
        }
        codes
    }
}

#[allow(clippy::too_many_arguments)]
fn load_block(
    weights: &Weights,
    base: &str,
    dim: usize,
    intermediate: usize,
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
    qkv_bias: bool,
    o_bias: bool,
    qk_norm: bool,
    norm_eps: f32,
    rope_base: f32,
) -> Result<Block> {
    let total = (n_head + 2 * n_local_heads) * head_dim;
    let wqkv = weights.linear(
        &format!("{base}.attention.wqkv"),
        dim,
        total,
        if qkv_bias { Bias::Required } else { Bias::None },
    )?;
    let wo = weights.linear(
        &format!("{base}.attention.wo"),
        n_head * head_dim,
        dim,
        if o_bias { Bias::Required } else { Bias::None },
    )?;
    let w1 = weights.linear(
        &format!("{base}.feed_forward.w1"),
        dim,
        intermediate,
        Bias::None,
    )?;
    let w2 = weights.linear(
        &format!("{base}.feed_forward.w2"),
        intermediate,
        dim,
        Bias::None,
    )?;
    let w3 = weights.linear(
        &format!("{base}.feed_forward.w3"),
        dim,
        intermediate,
        Bias::None,
    )?;
    let (q_norm, k_norm) = if qk_norm {
        (
            Some(Norm::load(
                weights,
                &format!("{base}.attention.q_norm"),
                head_dim,
                norm_eps,
            )?),
            Some(Norm::load(
                weights,
                &format!("{base}.attention.k_norm"),
                head_dim,
                norm_eps,
            )?),
        )
    } else {
        (None, None)
    };
    Ok(Block {
        wqkv,
        wo,
        w1,
        w2,
        w3,
        attention_norm: Norm::load(weights, &format!("{base}.attention_norm"), dim, norm_eps)?,
        ffn_norm: Norm::load(weights, &format!("{base}.ffn_norm"), dim, norm_eps)?,
        q_norm,
        k_norm,
        n_head,
        n_local_heads,
        head_dim,
        rope_base,
        cache: None,
    })
}

/// The falcon backbone config derived from the arktts fields
/// (`_falcon_args` in the reference; a rename-free pass-through).
fn check_unity(config: &ModelConfig, name: &str) -> Result<()> {
    let value = config
        .mamba_fields
        .get(name)
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    if value != 1.0 {
        return Err(SpeechError::Unsupported {
            why: format!(
                "arktts's falcon_h1 backbone supports unity muP multipliers only, but {name} is {value}"
            ),
        });
    }
    Ok(())
}

fn falcon_json(config: &ModelConfig) -> Result<Value> {
    // A rename-free pass-through of the falcon_* fields the arktts config
    // carries (the reference _falcon_args), on top of the derived geometry.
    let mut out = serde_json::json!({
        "model_type": "falcon_h1",
        "vocab_size": config.vocab_size,
        "hidden_size": config.dim,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "num_key_value_heads": config.n_local_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_base,
        "max_position_embeddings": config.max_seq_len,
        "attention_bias": false,
        "tie_word_embeddings": config.tie_word_embeddings,
        "embedding_multiplier": config.embedding_multiplier,
    });
    if let (Some(target), Some(source)) = (out.as_object_mut(), config.mamba_fields.as_object()) {
        for (key, value) in source {
            if key.starts_with("mamba_")
                || key.ends_with("_multiplier")
                || key == "mlp_multipliers"
                || key == "ssm_multipliers"
                || key == "mlp_bias"
                || key == "projectors_bias"
                || key == "attention_bias"
                || key == "hidden_act"
            {
                target.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(out)
}

/// The sampling parameter block (`generate_codes` arguments).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub do_sample: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            do_sample: true,
        }
    }
}

/// Uniform draws in generation order (recorded streams in fixtures; the
/// crate RNG in production). Each exponential race consumes one uniform per
/// vocabulary entry, so a frame takes two full-width semantic streams
/// (normal + RAS-high) plus one codebook-width stream per remaining
/// codebook position.
pub struct Uniforms {
    draws: std::collections::VecDeque<f32>,
    fallback: music_rng::Rng,
}

impl Uniforms {
    pub fn new(seed: u64) -> Self {
        Self {
            draws: std::collections::VecDeque::new(),
            fallback: music_rng::Rng::new(seed),
        }
    }

    pub fn push(&mut self, values: &[f32]) {
        self.draws.extend(values.iter().copied());
    }

    fn next_n(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| match self.draws.pop_front() {
                Some(value) => value,
                None => self.fallback.next_f32(),
            })
            .collect()
    }
}

mod music_rng {
    /// Minimal threefry-style counter RNG matching the crate's established
    /// pattern (music3 rng) closely enough for replay-free greedy runs.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(
                seed.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407),
            )
        }

        pub fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
        }
    }
}

/// The reference `_legacy_top_k_top_p`: filter BEFORE temperature. The
/// removal test is an INCLUSIVE cumulative softmax (`mx.cumsum` counts the
/// current entry), and rank 0 always survives.
fn legacy_top_k_top_p(scores: &[f32], top_k: usize, top_p: f32) -> Vec<f32> {
    let vocab = scores.len();
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Softmax over the sorted scores.
    let sorted: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    let max = sorted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = sorted.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let mut cumulative = 0.0f32;
    let mut keep_sorted = vec![false; vocab];
    for (rank, &p) in probs.iter().enumerate() {
        cumulative += p;
        let remove = (cumulative > top_p) || (rank >= top_k);
        keep_sorted[rank] = !remove || rank == 0;
    }
    let mut out = vec![f32::NEG_INFINITY; vocab];
    for (rank, &index) in order.iter().enumerate() {
        if keep_sorted[rank] {
            out[index] = scores[index];
        }
    }
    out
}

/// The reference `_sample`: exponential race, `argmax(softmax / -ln u)` with
/// ONE uniform per vocabulary entry (`mx.random.uniform(shape=probabilities.shape)`),
/// so the draw order is the vocabulary index order. Ties keep the first
/// index like `mx.argmax`.
fn exponential_race(scores: &[f32], uniforms: &[f32]) -> i32 {
    assert_eq!(scores.len(), uniforms.len(), "one uniform per entry");
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, (&e, &uniform)) in exps.iter().zip(uniforms).enumerate() {
        // The reference applies -ln(u) unguarded; u == 0 maps to an
        // infinite noise and a zero ratio exactly like mx.log would.
        let noise = -uniform.ln();
        let ratio = (e / sum) / noise;
        if ratio > best_score {
            best_score = ratio;
            best = index;
        }
    }
    best as i32
}

/// Prompt assembly (`_prepare_prompt`): semantic row = prefix [+ reference
/// semantic codes] + suffix; codebook column `book + 1` carries reference
/// code row `book` (row 0 is the semantic book, expressed raw here and
/// offset in the semantic row), aligned after the prefix.
pub fn prepare_prompt(
    config: &ModelConfig,
    prefix: &[i32],
    suffix: &[i32],
    reference_codes: Option<&[Vec<i32>]>,
    reference_length: usize,
) -> (Vec<Vec<i32>>, Vec<bool>) {
    let mut semantic_row: Vec<i32> = prefix.to_vec();
    if let Some(codes) = reference_codes {
        let length = reference_length.min(codes[0].len());
        for &code in codes[0].iter().take(length) {
            semantic_row.push(code + config.semantic_begin_id as i32);
        }
    }
    semantic_row.extend_from_slice(suffix);
    let width = semantic_row.len();
    let mut prompt = vec![vec![0i32; width]; config.num_codebooks + 1];
    prompt[0] = semantic_row;
    if let Some(codes) = reference_codes {
        let length = reference_length.min(codes[0].len());
        for (book, row) in prompt.iter_mut().skip(1).enumerate() {
            for (offset, slot) in row[prefix.len()..].iter_mut().enumerate().take(length) {
                *slot = codes[book][offset];
            }
        }
    }
    let mask = vec![true; width];
    (prompt, mask)
}

/// True when the text already carries a `<|speaker:N|>` tag (the reference
/// checks the regex `<\|speaker:\d+\|>`; only the prepend decision depends
/// on it).
fn has_speaker_tag(text: &str) -> bool {
    let mut rest = text;
    while let Some(position) = rest.find("<|speaker:") {
        let after = &rest[position + "<|speaker:".len()..];
        let digits = after.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && after[digits..].starts_with("|>") {
            return true;
        }
        rest = &rest[position + "<|speaker:".len()..];
    }
    false
}

/// The exact tokenizer-independent prompt strings the reference builds
/// (`_prompt_segments`): the PREFIX and SUFFIX segment lists (empty suffix
/// without a reference voice). The caller encodes each segment with the
/// checkpoint's Mistral BPE tokenizer (`add_special_tokens=False`) and
/// concatenates the ids per side.
pub fn prompt_segments(
    text: &str,
    reference_text: Option<&str>,
) -> (Vec<String>, Vec<String>, String) {
    let clean = |raw: &str| raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let target = clean(text);
    let prefix = vec![
        "<|im_start|>system\n".to_owned(),
        "convert the provided text to speech".to_owned(),
        "<|im_end|>\n".to_owned(),
        "<|im_start|>user\n".to_owned(),
        target.clone(),
        "<|im_end|>\n".to_owned(),
        "<|im_start|>assistant\n<|voice|>".to_owned(),
    ];
    let Some(reference) = reference_text else {
        return (prefix, Vec::new(), target);
    };
    let formatted = if has_speaker_tag(&clean(reference)) {
        clean(reference)
    } else {
        format!("<|speaker:0|>{}", clean(reference))
    };
    (
        vec![
            "<|im_start|>system\n".to_owned(),
            "convert the provided text to speech reference to the following:\n\nText:\n".to_owned(),
            formatted,
            "\n\nSpeech:\n".to_owned(),
        ],
        vec![
            "<|im_end|>\n".to_owned(),
            "<|im_start|>user\n".to_owned(),
            target.clone(),
            "<|im_end|>\n".to_owned(),
            "<|im_start|>assistant\n<|voice|>".to_owned(),
        ],
        target,
    )
}

/// Generation output: the 10-book code grid `[num_codebooks][T]`.
pub struct Generation {
    pub codes: Vec<Vec<i32>>,
}

impl ArkttsModel {
    /// `generate_codes` (batch 1). Returns the code grid for the codec.
    /// Sampled draws come from the crate RNG seeded with `seed`; fixture
    /// replays pass recorded streams through
    /// [`ArkttsModel::generate_codes_with_uniforms`].
    pub fn generate_codes(
        &mut self,
        prefix: &[i32],
        suffix: &[i32],
        reference_codes: Option<&[Vec<i32>]>,
        reference_length: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Result<Generation> {
        let mut uniforms = Uniforms::new(seed);
        self.generate_codes_with_uniforms(
            prefix,
            suffix,
            reference_codes,
            reference_length,
            params,
            &mut uniforms,
        )
    }

    /// [`ArkttsModel::generate_codes`] with an explicit uniform stream: each
    /// exponential race consumes one uniform per vocabulary entry.
    pub fn generate_codes_with_uniforms(
        &mut self,
        prefix: &[i32],
        suffix: &[i32],
        reference_codes: Option<&[Vec<i32>]>,
        reference_length: usize,
        params: &SamplingParams,
        uniforms: &mut Uniforms,
    ) -> Result<Generation> {
        let config = self.config.clone();
        let (prompt, _mask) =
            prepare_prompt(&config, prefix, suffix, reference_codes, reference_length);
        let prompt_width = prompt[0].len();
        if prompt_width >= config.max_seq_len {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt length {prompt_width} must be smaller than {}",
                    config.max_seq_len
                ),
            });
        }
        let max_new_tokens = params.max_new_tokens.min(config.max_seq_len - prompt_width);
        let eos_index = if config.uses_falcon_slow() {
            config.codebook_size
        } else {
            config.eos_token_id
        };

        // Set up caches.
        let mut falcon_caches = self.falcon.as_mut().map(|f| f.make_cache());
        for layer in &mut self.layers {
            layer.cache = Some((
                vec![0.0; config.n_local_heads * config.max_seq_len * config.head_dim],
                vec![0.0; config.n_local_heads * config.max_seq_len * config.head_dim],
            ));
        }
        for layer in &mut self.fast_layers {
            layer.cache = Some((
                vec![0.0; config.fast_n_local_heads * config.num_codebooks * config.fast_head_dim],
                vec![0.0; config.fast_n_local_heads * config.num_codebooks * config.fast_head_dim],
            ));
        }

        let (mut logits, mut slow_hidden) =
            self.slow_prefill(&prompt, prompt_width, falcon_caches.as_mut())?;
        let mut previous: Option<Vec<i32>> = None;
        let mut frames: Vec<Vec<i32>> = Vec::new();

        for step in 0..max_new_tokens {
            let semantic = sample_semantic(&config, &logits, params, previous.as_deref(), uniforms);
            let codebooks = {
                let hidden = slow_hidden.clone();
                self.generate_codebooks(&hidden, semantic, params, uniforms)
            };
            if semantic != eos_index as i32 {
                frames.push(codebooks.clone());
            }

            previous = match previous.take() {
                None => Some(vec![0i32; config.ras_window_size]),
                Some(mut window) => {
                    window.remove(0);
                    window.push(semantic);
                    Some(window)
                }
            };
            if semantic == eos_index as i32 {
                break;
            }

            // The slow stack always consumes full-vocabulary ids.
            let semantic_in = if config.uses_falcon_slow() && semantic == eos_index as i32 {
                config.eos_token_id as i32
            } else if config.uses_falcon_slow() {
                semantic + config.semantic_begin_id as i32
            } else {
                semantic
            };
            let cache_start = prompt_width + step;
            let position = prompt_width + step;
            let books = codebooks.clone();
            let (next_logits, next_hidden) = self.slow_step(
                semantic_in,
                &books,
                cache_start,
                position,
                falcon_caches.as_mut(),
            )?;
            logits = next_logits;
            slow_hidden = next_hidden;
        }

        // Tear down caches.
        for layer in &mut self.layers {
            layer.cache = None;
        }
        for layer in &mut self.fast_layers {
            layer.cache = None;
        }
        Ok(Generation { codes: frames })
    }
}

/// The reference `_sample_semantic`: semantic-range filter, legacy top-k /
/// top-p, THEN temperature; greedy argmax or the exponential race with the
/// RAS repetition rescue (a repeated semantic inside the window switches to
/// the ras_top_p / ras_temperature draw).
fn sample_semantic(
    config: &ModelConfig,
    logits: &[f32],
    params: &SamplingParams,
    previous: Option<&[i32]>,
    uniforms: &mut Uniforms,
) -> i32 {
    let filtered = if config.uses_falcon_slow() {
        logits.to_vec()
    } else {
        let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
        for (id, value) in filtered.iter_mut().enumerate() {
            let semantic = (config.semantic_begin_id..=config.semantic_end_id).contains(&id);
            let eos = id == config.eos_token_id;
            if semantic || eos {
                *value = logits[id];
            }
        }
        filtered
    };
    let regular = legacy_top_k_top_p(&filtered, params.top_k, params.top_p);
    let regular: Vec<f32> = regular
        .iter()
        .map(|&s| s / params.temperature.max(1e-5))
        .collect();
    if !params.do_sample {
        return argmax(&regular) as i32;
    }
    let normal = exponential_race(&regular, &uniforms.next_n(regular.len()));
    let high_scores = legacy_top_k_top_p(&filtered, params.top_k, config.ras_top_p);
    let high_scores: Vec<f32> = high_scores
        .iter()
        .map(|&s| s / config.ras_temperature.max(1e-5))
        .collect();
    let high = exponential_race(&high_scores, &uniforms.next_n(high_scores.len()));
    let previous = match previous {
        Some(window) => window,
        None => return normal,
    };
    let repeated = previous.contains(&normal);
    let semantic_hit = if config.uses_falcon_slow() {
        normal < config.codebook_size as i32
    } else {
        normal >= config.semantic_begin_id as i32 && normal <= config.semantic_end_id as i32
    };
    if repeated && semantic_hit {
        high
    } else {
        normal
    }
}

/// Opens the model and its codec from a checkpoint directory.
pub struct Model {
    pub config: ModelConfig,
    pub model: ArkttsModel,
    pub codec: codec::ArkttsCodec,
}

impl Model {
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let weights = Weights::open(dir)?;
        let config_value = weights.config().clone();
        let config = ModelConfig::from_json(&config_value)?;
        let model = ArkttsModel::load(&weights, &config)?;
        let codec = codec::ArkttsCodec::load(
            &weights,
            &codec::CodecConfig::from_tts_config(&config_value),
        )?;
        Ok(Self {
            config,
            model,
            codec,
        })
    }

    /// Text -> codes -> waveform. `prefix`/`suffix` are the encoded prompt
    /// segments from [`prompt_segments`].
    pub fn generate(
        &mut self,
        prefix: &[i32],
        suffix: &[i32],
        reference_codes: Option<&[Vec<i32>]>,
        reference_length: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let generation = self.model.generate_codes(
            prefix,
            suffix,
            reference_codes,
            reference_length,
            params,
            seed,
        )?;
        if generation.codes.is_empty() {
            return Err(SpeechError::Input {
                why: "no audio generated (immediate end of speech)".into(),
            });
        }
        // The codec consumes book-major rows; the generation grid is
        // frame-major.
        let frames = generation.codes.len();
        let mut book_major = vec![Vec::with_capacity(frames); self.config.num_codebooks];
        for row in &generation.codes {
            for (book, code) in row.iter().enumerate() {
                book_major[book].push(*code);
            }
        }
        self.codec.decode(&book_major)
    }
}

#[cfg(test)]
mod tests;
