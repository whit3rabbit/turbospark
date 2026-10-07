//! Qwen2 decoder backbone of VibeVoice-ASR.
//!
//! Reference: `mlx_audio/lm/models/qwen2.py` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Standard Qwen2: grouped
//! query attention with biased q/k/v projections, bias-free o_proj and
//! MLP, full-width RoPE at `rope_theta` in the half-split ("neoX")
//! convention, RMSNorm pre-norm, and logits from the token embedding
//! matrix as a linear map (`tie_word_embeddings` is pinned true; the
//! stored `lm_head` tensor is ignored, matching mlx_lm's sanitize).
//!
//! Weights read the raw checkpoint names
//! `model.language_model.layers.N.*`, `model.language_model.norm.weight`,
//! and `model.language_model.embed_tokens.weight` directly.

use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::{Result, SpeechError};

use super::tokenizer_encoder::{load_linear_sharded, load_rms_norm_sharded, load_sharded};

/// Per-layer cached rotated keys and values, one vector per KV head.
struct LayerCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
}

impl LayerCache {
    fn new(key_value_heads: usize) -> Self {
        Self {
            keys: vec![Vec::new(); key_value_heads],
            values: vec![Vec::new(); key_value_heads],
            len: 0,
        }
    }
}

struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    theta: f32,
    scale: f32,
}

impl Qwen2Attention {
    fn load(
        files: &[turbospark_model_io::safetensors::SafetensorsFile],
        prefix: &str,
        query_heads: usize,
        key_value_heads: usize,
        hidden: usize,
        theta: f32,
    ) -> Result<Self> {
        let head_dim = hidden / query_heads;
        let q_width = query_heads * head_dim;
        let kv_width = key_value_heads * head_dim;
        Ok(Self {
            q_proj: load_linear_sharded(files, &format!("{prefix}.q_proj"), hidden, q_width, true)?,
            k_proj: load_linear_sharded(
                files,
                &format!("{prefix}.k_proj"),
                hidden,
                kv_width,
                true,
            )?,
            v_proj: load_linear_sharded(
                files,
                &format!("{prefix}.v_proj"),
                hidden,
                kv_width,
                true,
            )?,
            o_proj: load_linear_sharded(
                files,
                &format!("{prefix}.o_proj"),
                q_width,
                hidden,
                false,
            )?,
            query_heads,
            key_value_heads,
            head_dim,
            theta,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Causal attention over `[rows, hidden]` with an optional cache to
    /// append the rotated keys and values to.
    fn forward(&self, x: &[f32], rows: usize, cache: Option<&mut LayerCache>) -> Vec<f32> {
        let query_width = self.query_heads * self.head_dim;
        let query = self.q_proj.forward(x, rows);
        let key = self.k_proj.forward(x, rows);
        let value = self.v_proj.forward(x, rows);
        let mut query_heads = ops::split_heads(&query, rows, self.query_heads, self.head_dim);
        let mut key_heads = ops::split_heads(&key, rows, self.key_value_heads, self.head_dim);
        let value_heads = ops::split_heads(&value, rows, self.key_value_heads, self.head_dim);
        let (cos, sin) = ops::rope_tables(rows, self.head_dim, self.theta);
        ops::rope_neox(
            &mut query_heads,
            self.query_heads,
            rows,
            self.head_dim,
            &cos,
            &sin,
        );
        ops::rope_neox(
            &mut key_heads,
            self.key_value_heads,
            rows,
            self.head_dim,
            &cos,
            &sin,
        );
        if let Some(cache) = cache {
            for head in 0..self.key_value_heads {
                let start = head * rows * self.head_dim;
                let end = start + rows * self.head_dim;
                cache.keys[head].extend_from_slice(&key_heads[start..end]);
                cache.values[head].extend_from_slice(&value_heads[start..end]);
            }
            cache.len = rows;
        }

        // Grouped heads share one KV head; index it directly instead of
        // materializing repeats.
        debug_assert_eq!(self.query_heads % self.key_value_heads, 0);
        let group = self.query_heads / self.key_value_heads;
        let mut attended = vec![0.0f32; query_width * rows];
        for head in 0..self.query_heads {
            let kv_head = head / group;
            let head_start = head * rows * self.head_dim;
            let kv_start = kv_head * rows * self.head_dim;
            let query_head = &query_heads[head_start..head_start + rows * self.head_dim];
            let key_head = &key_heads[kv_start..kv_start + rows * self.head_dim];
            let value_head = &value_heads[kv_start..kv_start + rows * self.head_dim];
            for position in 0..rows {
                let query_row =
                    &query_head[position * self.head_dim..(position + 1) * self.head_dim];
                let output = ops::sdpa(
                    query_row,
                    &key_head[..(position + 1) * self.head_dim],
                    &value_head[..(position + 1) * self.head_dim],
                    None,
                    1,
                    position + 1,
                    self.head_dim,
                    self.head_dim,
                    self.scale,
                );
                let target = position * query_width + head * self.head_dim;
                attended[target..target + self.head_dim].copy_from_slice(&output);
            }
        }
        self.o_proj.forward(&attended, rows)
    }

    /// One decode step; `cos`/`sin` are the single-position RoPE rows the
    /// caller builds once per token.
    fn step(&self, x: &[f32], cache: &mut LayerCache, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let query_width = self.query_heads * self.head_dim;
        let mut query = self.q_proj.forward(x, 1);
        let mut key = self.k_proj.forward(x, 1);
        let value = self.v_proj.forward(x, 1);
        ops::rope_neox(&mut query, self.query_heads, 1, self.head_dim, cos, sin);
        ops::rope_neox(&mut key, self.key_value_heads, 1, self.head_dim, cos, sin);
        for head in 0..self.key_value_heads {
            let start = head * self.head_dim;
            cache.keys[head].extend_from_slice(&key[start..start + self.head_dim]);
            cache.values[head].extend_from_slice(&value[start..start + self.head_dim]);
        }
        cache.len += 1;
        debug_assert_eq!(self.query_heads % self.key_value_heads, 0);
        let group = self.query_heads / self.key_value_heads;
        let mut attended = vec![0.0f32; query_width];
        for head in 0..self.query_heads {
            let kv_head = head / group;
            let start = head * self.head_dim;
            let output = ops::sdpa(
                &query[start..start + self.head_dim],
                &cache.keys[kv_head],
                &cache.values[kv_head],
                None,
                1,
                cache.len,
                self.head_dim,
                self.head_dim,
                self.scale,
            );
            attended[start..start + self.head_dim].copy_from_slice(&output);
        }
        self.o_proj.forward(&attended, 1)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(
        files: &[turbospark_model_io::safetensors::SafetensorsFile],
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: load_linear_sharded(
                files,
                &format!("{prefix}.gate_proj"),
                hidden,
                intermediate,
                false,
            )?,
            up: load_linear_sharded(
                files,
                &format!("{prefix}.up_proj"),
                hidden,
                intermediate,
                false,
            )?,
            down: load_linear_sharded(
                files,
                &format!("{prefix}.down_proj"),
                intermediate,
                hidden,
                false,
            )?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut gate = self.gate.forward(x, rows);
        ops::silu(&mut gate);
        let up = self.up.forward(x, rows);
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate *= up;
        }
        self.down.forward(&gate, rows)
    }
}

struct DecoderLayer {
    attention: Qwen2Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    key_value_heads: usize,
}

impl DecoderLayer {
    fn forward(&self, x: &[f32], rows: usize) -> (Vec<f32>, LayerCache) {
        let mut normed = x.to_vec();
        self.input_norm.apply(&mut normed, rows);
        let mut cache = LayerCache::new(self.key_value_heads);
        let attention = self.attention.forward(&normed, rows, Some(&mut cache));
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }
        let mut normed = residual.clone();
        self.post_attention_norm.apply(&mut normed, rows);
        let mlp = self.mlp.forward(&normed, rows);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        (residual, cache)
    }

    fn step(&self, x: &[f32], cache: &mut LayerCache, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let mut normed = x.to_vec();
        self.input_norm.apply(&mut normed, 1);
        let attention = self.attention.step(&normed, cache, cos, sin);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }
        let mut normed = residual.clone();
        self.post_attention_norm.apply(&mut normed, 1);
        let mlp = self.mlp.forward(&normed, 1);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        residual
    }
}

/// Cached decode state shared by the greedy loop.
pub(crate) struct DecodeCache {
    layers: Vec<LayerCache>,
    theta: f32,
    head_dim: usize,
}

/// The loaded Qwen2 decoder with its tied embedding head.
pub(crate) struct Qwen2 {
    embeddings: Vec<f32>,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    hidden: usize,
    vocab: usize,
    theta: f32,
}

impl Qwen2 {
    /// Loads the decoder. `prefix` is the raw checkpoint base, for the
    /// pinned checkpoint `model.language_model`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        files: &[turbospark_model_io::safetensors::SafetensorsFile],
        prefix: &str,
        hidden: usize,
        intermediate: usize,
        layer_count: usize,
        query_heads: usize,
        key_value_heads: usize,
        vocab: usize,
        eps: f32,
        theta: f32,
    ) -> Result<Self> {
        let layers = (0..layer_count)
            .map(|index| {
                let layer_prefix = format!("{prefix}.layers.{index}");
                Ok(DecoderLayer {
                    attention: Qwen2Attention::load(
                        files,
                        &format!("{layer_prefix}.self_attn"),
                        query_heads,
                        key_value_heads,
                        hidden,
                        theta,
                    )?,
                    mlp: Mlp::load(files, &format!("{layer_prefix}.mlp"), hidden, intermediate)?,
                    input_norm: load_rms_norm_sharded(
                        files,
                        &format!("{layer_prefix}.input_layernorm"),
                        hidden,
                        eps,
                    )?,
                    post_attention_norm: load_rms_norm_sharded(
                        files,
                        &format!("{layer_prefix}.post_attention_layernorm"),
                        hidden,
                        eps,
                    )?,
                    key_value_heads,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings: load_sharded(files, &format!("{prefix}.embed_tokens"), &[vocab, hidden])?,
            layers,
            final_norm: load_rms_norm_sharded(files, &format!("{prefix}.norm"), hidden, eps)?,
            hidden,
            vocab,
            theta,
        })
    }

    /// Token embedding rows `[ids.len(), hidden]`; ids are signed 32-bit
    /// per the crate boundary contract.
    pub(crate) fn embed(&self, ids: &[i32]) -> Result<Vec<f32>> {
        if ids.iter().any(|&id| id < 0 || id as usize >= self.vocab) {
            return Err(SpeechError::Input {
                why: "token id outside the checkpoint vocabulary".into(),
            });
        }
        Ok(ops::embedding(&self.embeddings, self.hidden, ids))
    }

    /// Embedding matrix applied as a linear map (the tied head).
    pub(crate) fn tied_logits(&self, hidden: &[f32]) -> Vec<f32> {
        ops::linear(hidden, &self.embeddings, None, 1, self.hidden, self.vocab)
    }

    /// Evaluates the prompt and returns the final post-norm row plus the
    /// populated cache for the greedy loop.
    pub(crate) fn prefill(&self, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut hidden = input.to_vec();
        for layer in &self.layers {
            let (out, cache) = layer.forward(&hidden, rows);
            hidden = out;
            caches.push(cache);
        }
        self.final_norm.apply(&mut hidden, rows);
        let last = hidden[(rows - 1) * self.hidden..].to_vec();
        (
            last,
            DecodeCache {
                layers: caches,
                theta: self.theta,
                head_dim: self.hidden / self.layers[0].attention.query_heads,
            },
        )
    }

    /// One decode step against the cache; returns the post-norm row.
    pub(crate) fn step(&self, input: &[f32], cache: &mut DecodeCache) -> Vec<f32> {
        let (cos, sin) =
            ops::rope_tables_range(cache.layers[0].len, 1, cache.head_dim, cache.theta);
        let mut hidden = input.to_vec();
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.step(&hidden, layer_cache, &cos, &sin);
        }
        self.final_norm.apply(&mut hidden, 1);
        hidden
    }
}
#[cfg(test)]
mod tests {
    use super::{DecoderLayer, Mlp, Qwen2, Qwen2Attention};

    fn deterministic(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32 * 0.05
            })
            .collect()
    }

    fn linear(input: usize, output: usize, seed: u64) -> crate::nn::Linear {
        crate::nn::Linear::new(deterministic(input * output, seed), None, input, output)
    }

    fn norm(width: usize) -> crate::nn::RmsNorm {
        crate::nn::RmsNorm::new(vec![1.0; width], 1e-6)
    }

    /// The cache-free reference path, written locally so the production
    /// surface stays lean.
    fn full_forward(decoder: &Qwen2, input: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for layer in &decoder.layers {
            let (out, _) = layer.forward(&hidden, rows);
            hidden = out;
        }
        decoder.final_norm.apply(&mut hidden, rows);
        hidden
    }

    /// A tiny two-layer Qwen2 whose cached prefill plus decode steps must
    /// reproduce the full forward hidden states: the fixture's hidden rows
    /// come from the cache-free path while the gated run decodes through
    /// the cache, so the two must agree.
    #[test]
    fn cached_prefill_and_steps_match_the_full_forward() {
        let hidden = 8usize;
        let query_heads = 4usize;
        let key_value_heads = 2usize;
        let head_dim = hidden / query_heads;
        let attention = |seed: u64| Qwen2Attention {
            q_proj: linear(hidden, hidden, seed),
            k_proj: linear(hidden, key_value_heads * head_dim, seed + 1),
            v_proj: linear(hidden, key_value_heads * head_dim, seed + 2),
            o_proj: linear(hidden, hidden, seed + 3),
            query_heads,
            key_value_heads,
            head_dim,
            theta: 10_000.0,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let mlp = |seed: u64| Mlp {
            gate: linear(hidden, 12, seed),
            up: linear(hidden, 12, seed + 1),
            down: linear(12, hidden, seed + 2),
        };
        let decoder = Qwen2 {
            embeddings: deterministic(31 * hidden, 7),
            layers: (0..2)
                .map(|index| DecoderLayer {
                    attention: attention(100 + index),
                    mlp: mlp(200 + index),
                    input_norm: norm(hidden),
                    post_attention_norm: norm(hidden),
                    key_value_heads,
                })
                .collect(),
            final_norm: norm(hidden),
            hidden,
            vocab: 31,
            theta: 10_000.0,
        };

        let rows = 5usize;
        let mut input = deterministic(rows * hidden, 42);
        let full = full_forward(&decoder, &input, rows);
        let (last, mut cache) = decoder.prefill(&input, rows);
        for (cached, reference) in last.iter().zip(&full[(rows - 1) * hidden..]) {
            assert_eq!(cached.to_bits(), reference.to_bits());
        }
        for row in rows..rows + 3 {
            let token = deterministic(hidden, 1_000 + row as u64);
            input.extend_from_slice(&token);
            let stepped = decoder.step(&token, &mut cache);
            let reference = full_forward(&decoder, &input, row + 1);
            for (cached, full_row) in stepped.iter().zip(&reference[row * hidden..]) {
                assert_eq!(cached.to_bits(), full_row.to_bits());
            }
            assert_eq!(cache.layers[0].len, row + 1);
        }
    }
}
