//! Granite Speech 1B text backbone: the llama-style Granite decoder with
//! its four multipliers.
//!
//! Reference: `mlx_audio/lm/models/granite.py` (`Model`, `GraniteModel`,
//! `TransformerBlock`, `Attention`, `MLP`) plus the speech wrapper
//! `Model.__call__` in `mlx_audio/stt/models/granite_speech/granite_speech.py`
//! at mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! The decoder stack itself (grouped-query attention with KV cache, SwiGLU
//! MLP, RMSNorm) is the shared Qwen3 one in `qwen3_asr::decoder`; this
//! module only loads the checkpoint into it and carries the Granite scalars.
//!
//! Granite differs from a plain llama decoder in four load-bearing scalars
//! from `text_config`: `embedding_multiplier` scales the (possibly
//! injected audio) embeddings before layer 0, `attention_multiplier` is
//! the softmax scale in place of `head_dim^-0.5`, `residual_multiplier`
//! scales both branch outputs inside every block, and `logits_scaling`
//! divides the final logits. Attention is grouped-query with no q/k
//! post-norms and no projection biases; RoPE is the default non-
//! traditional (half-split) layout.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{Linear, RmsNorm};
use crate::stt::qwen3_asr::decoder::{
    DecodeCache, Decoder, DecoderLayer, Mlp, TextAttention, TokenDecoder,
};
use crate::{Result, SpeechError};

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: "missing from the checkpoint shards".to_owned(),
    }
}

fn shard_of<'a>(files: &'a [SafetensorsFile], name: &str) -> Result<&'a SafetensorsFile> {
    files
        .iter()
        .find(|file| file.contains_tensor(name))
        .ok_or_else(|| missing(name))
}

fn load_vector(files: &[SafetensorsFile], name: &str, len: usize) -> Result<Vec<f32>> {
    let values = shard_of(files, name)?.load_as_f32(name)?;
    if values.len() != len {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {len} values, got {}", values.len()),
        });
    }
    Ok(values)
}

fn load_linear(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let bias_name = format!("{base}.bias");
    if files.iter().any(|file| file.contains_tensor(&bias_name)) {
        return Err(SpeechError::Unsupported {
            why: format!(
                "{base} must stay bias-free; the config declares no attention or MLP \
                        biases"
            ),
        });
    }
    Ok(Linear::new(
        load_vector(files, &format!("{base}.weight"), input * output)?,
        None,
        input,
        output,
    ))
}

fn load_rms_norm(
    files: &[SafetensorsFile],
    base: &str,
    width: usize,
    epsilon: f32,
) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        load_vector(files, &format!("{base}.weight"), width)?,
        epsilon,
    ))
}

fn load_attention(
    files: &[SafetensorsFile],
    prefix: &str,
    config: &super::config::TextConfig,
) -> Result<TextAttention> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let query_width = config.num_attention_heads * head_dim;
    let kv_width = config.num_key_value_heads * head_dim;
    Ok(TextAttention {
        q_proj: load_linear(
            files,
            &format!("{prefix}.q_proj"),
            config.hidden_size,
            query_width,
        )?,
        k_proj: load_linear(
            files,
            &format!("{prefix}.k_proj"),
            config.hidden_size,
            kv_width,
        )?,
        v_proj: load_linear(
            files,
            &format!("{prefix}.v_proj"),
            config.hidden_size,
            kv_width,
        )?,
        o_proj: load_linear(
            files,
            &format!("{prefix}.o_proj"),
            query_width,
            config.hidden_size,
        )?,
        // No q/k post-norms in the Granite attention.
        q_norm: None,
        k_norm: None,
        query_heads: config.num_attention_heads,
        key_value_heads: config.num_key_value_heads,
        head_dim,
        rotary_dim: head_dim,
        theta: config.rope_theta,
        // Granite's attention_multiplier replaces head_dim^-0.5.
        scale: config.attention_multiplier,
    })
}

fn load_mlp(
    files: &[SafetensorsFile],
    prefix: &str,
    config: &super::config::TextConfig,
) -> Result<Mlp> {
    Ok(Mlp {
        gate: load_linear(
            files,
            &format!("{prefix}.gate_proj"),
            config.hidden_size,
            config.intermediate_size,
        )?,
        up: load_linear(
            files,
            &format!("{prefix}.up_proj"),
            config.hidden_size,
            config.intermediate_size,
        )?,
        down: load_linear(
            files,
            &format!("{prefix}.down_proj"),
            config.intermediate_size,
            config.hidden_size,
        )?,
        intermediate: config.intermediate_size,
    })
}

fn load_block(
    files: &[SafetensorsFile],
    index: usize,
    config: &super::config::TextConfig,
) -> Result<DecoderLayer> {
    let prefix = format!("language_model.model.layers.{index}");
    Ok(DecoderLayer {
        attention: load_attention(files, &format!("{prefix}.self_attn"), config)?,
        mlp: load_mlp(files, &format!("{prefix}.mlp"), config)?,
        input_norm: load_rms_norm(
            files,
            &format!("{prefix}.input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?,
        post_attention_norm: load_rms_norm(
            files,
            &format!("{prefix}.post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?,
        residual_multiplier: config.residual_multiplier,
        hidden: config.hidden_size,
    })
}

/// The Granite decoder: the shared [`Decoder`] with its embedding
/// multiplier, per-block residual multiplier, untied LM head, and the
/// logits scaling applied at [`GraniteLlm::logits`].
pub struct GraniteLlm {
    decoder: Decoder,
    pub(crate) embedding_multiplier: f32,
    pub(crate) logits_scaling: f32,
}

impl GraniteLlm {
    pub(crate) fn load_sharded(
        files: &[SafetensorsFile],
        config: &super::config::TextConfig,
    ) -> Result<Self> {
        let embeddings = load_vector(
            files,
            "language_model.model.embed_tokens.weight",
            config.vocab_size * config.hidden_size,
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| load_block(files, index, config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = load_rms_norm(
            files,
            "language_model.model.norm",
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let lm_head = load_linear(
            files,
            "language_model.lm_head",
            config.hidden_size,
            config.vocab_size,
        )?;
        Ok(Self {
            decoder: Decoder {
                embeddings,
                lm_head: Some(lm_head),
                layers,
                final_norm,
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
            },
            embedding_multiplier: config.embedding_multiplier,
            logits_scaling: config.logits_scaling,
        })
    }

    /// Raw token embeddings `[ids, hidden]`; the embedding multiplier is
    /// applied by the model wrapper, exactly as the reference does.
    pub(crate) fn embed(&self, token_ids: &[i32]) -> Result<Vec<f32>> {
        self.decoder.embed(token_ids)
    }

    /// Evaluate the prompt once and retain its rotated K/V projections;
    /// returns the last row's post-norm hidden state.
    pub(crate) fn prefill(&self, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
        self.decoder.prefill(input, rows)
    }

    /// LM head logits divided by `logits_scaling` (the reference
    /// `Model.__call__` tail).
    pub(crate) fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = self.decoder.logits(hidden);
        for value in &mut logits {
            *value /= self.logits_scaling;
        }
        logits
    }
}

impl TokenDecoder for GraniteLlm {
    fn next_hidden(&self, token: i32, cache: &mut DecodeCache) -> Result<Vec<f32>> {
        let mut input = self.embed(&[token])?;
        for value in &mut input {
            *value *= self.embedding_multiplier;
        }
        Ok(self.decoder.step(&input, cache))
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        GraniteLlm::logits(self, hidden)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn head_transpose_and_neox_rope_match_known_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            crate::ops::split_heads(&x, 2, 2, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        let mut q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (cos, sin) = crate::ops::rope_tables(2, 4, 1.0);
        crate::ops::rope_neox(&mut q, 1, 2, 4, &cos, &sin);
        assert_eq!(q[0], 1.0);
        assert_eq!(q[1], 0.0);
        assert!((q[4] - 0.5403023).abs() < 1e-6);
        assert!((q[6] - 0.84147096).abs() < 1e-6);
    }

    #[test]
    fn single_position_rope_matches_a_slice_of_the_full_table() {
        let (cos, sin) = crate::ops::rope_tables_range(7, 1, 4, 100.0);
        let mut single = vec![0.5, -0.25, 2.0, 1.0];
        crate::ops::rope_neox(&mut single, 1, 1, 4, &cos, &sin);
        let (full_cos, full_sin) = crate::ops::rope_tables(8, 4, 100.0);
        let mut full = vec![0.5, -0.25, 2.0, 1.0];
        for feature in 0..2 {
            let a = full[feature];
            let b = full[feature + 2];
            full[feature] = a * full_cos[7 * 2 + feature] - b * full_sin[7 * 2 + feature];
            full[feature + 2] = b * full_cos[7 * 2 + feature] + a * full_sin[7 * 2 + feature];
        }
        for (a, b) in single.iter().zip(full) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

/// Synthetic-weight parity against the pre-merge Granite stack, retained
/// verbatim below. The golden Granite tests need the checkpoint and are
/// ignored by default, so this is what proves the move onto the shared
/// Qwen3 decoder changed no bits.
#[cfg(test)]
mod parity {
    #![allow(dead_code)]

    use super::GraniteLlm as SharedLlm;
    use crate::nn::{Linear, RmsNorm};
    use crate::ops;
    use crate::stt::qwen3_asr::decoder::{
        greedy_generate, DecodeCache as SharedCache, Decoder, DecoderLayer, Mlp as SharedMlp,
        TextAttention,
    };
    use crate::{Result, SpeechError};

    // ---- retained pre-merge implementation (verbatim, loaders dropped) ----
    struct AttentionCache {
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        len: usize,
    }

    impl AttentionCache {
        fn new(heads: usize) -> Self {
            Self {
                keys: vec![Vec::new(); heads],
                values: vec![Vec::new(); heads],
                len: 0,
            }
        }
    }

    /// Per-layer KV cache handed back from [`GraniteLlm::prefill`].
    pub struct DecodeCache {
        layers: Vec<AttentionCache>,
    }

    struct GraniteAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        scale: f32,
    }

    impl GraniteAttention {
        /// Causal attention over `x [rows, hidden]`, appending to the cache
        /// when present.
        fn forward(&self, x: &[f32], rows: usize, cache: Option<&mut AttentionCache>) -> Vec<f32> {
            let query_width = self.query_heads * self.head_dim;
            let query = self.q_proj.forward(x, rows);
            let key = self.k_proj.forward(x, rows);
            let value = self.v_proj.forward(x, rows);

            let mut query_heads = ops::split_heads(&query, rows, self.query_heads, self.head_dim);
            let mut key_heads = ops::split_heads(&key, rows, self.key_value_heads, self.head_dim);
            let value_heads = ops::split_heads(&value, rows, self.key_value_heads, self.head_dim);
            let (cos, sin) = ops::rope_tables(rows, self.rotary_dim, self.theta);
            ops::rope_neox(
                &mut query_heads,
                self.query_heads,
                rows,
                self.rotary_dim,
                &cos,
                &sin,
            );
            ops::rope_neox(
                &mut key_heads,
                self.key_value_heads,
                rows,
                self.rotary_dim,
                &cos,
                &sin,
            );

            let Some(cache) = cache else {
                return self.attend(
                    &query_heads,
                    &key_heads,
                    &value_heads,
                    rows,
                    rows,
                    query_width,
                );
            };
            for head in 0..self.key_value_heads {
                let start = head * rows * self.head_dim;
                let end = start + rows * self.head_dim;
                cache.keys[head].extend_from_slice(&key_heads[start..end]);
                cache.values[head].extend_from_slice(&value_heads[start..end]);
            }
            cache.len += rows;
            // Prefill-only path: the cache held exactly the rows added above,
            // so query row p attends over the causal prefix p + 1.
            let mut attended = vec![0.0f32; query_width * rows];
            for head in 0..self.query_heads {
                let kv_head = head / (self.query_heads / self.key_value_heads);
                let head_start = head * rows * self.head_dim;
                let query_head = &query_heads[head_start..head_start + rows * self.head_dim];
                let keys = &cache.keys[kv_head];
                let values = &cache.values[kv_head];
                for position in 0..rows {
                    let query_row =
                        &query_head[position * self.head_dim..(position + 1) * self.head_dim];
                    let output = ops::sdpa(
                        query_row,
                        &keys[..(position + 1) * self.head_dim],
                        &values[..(position + 1) * self.head_dim],
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

        /// Single-token step against the cache.
        fn step(&self, x: &[f32], cache: &mut AttentionCache) -> Vec<f32> {
            let mut query = self.q_proj.forward(x, 1);
            let mut key = self.k_proj.forward(x, 1);
            let value = self.v_proj.forward(x, 1);
            let position = cache.len;
            let (cos, sin) = ops::rope_tables_range(position, 1, self.rotary_dim, self.theta);
            // One position per head: `[heads * head_dim]` is already head-major
            // with a single row, so one call rotates every head.
            ops::rope_neox(&mut query, self.query_heads, 1, self.rotary_dim, &cos, &sin);
            ops::rope_neox(
                &mut key,
                self.key_value_heads,
                1,
                self.rotary_dim,
                &cos,
                &sin,
            );
            for head in 0..self.key_value_heads {
                let start = head * self.head_dim;
                cache.keys[head].extend_from_slice(&key[start..start + self.head_dim]);
                cache.values[head].extend_from_slice(&value[start..start + self.head_dim]);
            }
            cache.len += 1;
            let mut attended = vec![0.0f32; self.query_heads * self.head_dim];
            for head in 0..self.query_heads {
                let kv_head = head / (self.query_heads / self.key_value_heads);
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

        /// Full causal attention of `rows` query rows against `cached` key and
        /// value rows (head-major planes).
        fn attend(
            &self,
            query_heads: &[f32],
            key_heads: &[f32],
            value_heads: &[f32],
            cached: usize,
            rows: usize,
            query_width: usize,
        ) -> Vec<f32> {
            let mut attended = vec![0.0f32; query_width * rows];
            for head in 0..self.query_heads {
                let kv_head = head / (self.query_heads / self.key_value_heads);
                let head_start = head * rows * self.head_dim;
                let query_head = &query_heads[head_start..head_start + rows * self.head_dim];
                let key_head = &key_heads
                    [kv_head * cached * self.head_dim..(kv_head + 1) * cached * self.head_dim];
                let value_head = &value_heads
                    [kv_head * cached * self.head_dim..(kv_head + 1) * cached * self.head_dim];
                for position in 0..rows {
                    let query_row =
                        &query_head[position * self.head_dim..(position + 1) * self.head_dim];
                    let key_prefix = &key_head[..(position + 1) * self.head_dim];
                    let value_prefix = &value_head[..(position + 1) * self.head_dim];
                    let output = ops::sdpa(
                        query_row,
                        key_prefix,
                        value_prefix,
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
    }

    struct Mlp {
        gate: Linear,
        up: Linear,
        down: Linear,
        intermediate: usize,
    }

    impl Mlp {
        fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
            let mut gate = self.gate.forward(x, rows);
            ops::silu(&mut gate);
            let up = self.up.forward(x, rows);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate *= up;
            }
            debug_assert_eq!(gate.len(), rows * self.intermediate);
            self.down.forward(&gate, rows)
        }
    }
    struct TransformerBlock {
        attention: GraniteAttention,
        mlp: Mlp,
        input_norm: RmsNorm,
        post_attention_norm: RmsNorm,
        residual_multiplier: f32,
        hidden: usize,
    }

    impl TransformerBlock {
        fn forward(&self, x: &[f32], rows: usize, cache: Option<&mut AttentionCache>) -> Vec<f32> {
            let mut normalized = x.to_vec();
            self.input_norm.apply(&mut normalized, rows);
            let attention = self.attention.forward(&normalized, rows, cache);
            // h = x + attn * residual_multiplier
            let mut h = x
                .iter()
                .zip(attention)
                .map(|(r, a)| r + a * self.residual_multiplier)
                .collect::<Vec<f32>>();
            let mut normalized = h.clone();
            self.post_attention_norm.apply(&mut normalized, rows);
            let mlp = self.mlp.forward(&normalized, rows);
            for (value, add) in h.iter_mut().zip(mlp) {
                *value += add * self.residual_multiplier;
            }
            debug_assert_eq!(h.len(), rows * self.hidden);
            h
        }

        fn step(&self, x: &[f32], cache: &mut AttentionCache) -> Vec<f32> {
            let mut normalized = x.to_vec();
            self.input_norm.apply(&mut normalized, 1);
            let attention = self.attention.step(&normalized, cache);
            let mut h = x
                .iter()
                .zip(attention)
                .map(|(r, a)| r + a * self.residual_multiplier)
                .collect::<Vec<f32>>();
            let mut normalized = h.clone();
            self.post_attention_norm.apply(&mut normalized, 1);
            let mlp = self.mlp.forward(&normalized, 1);
            for (value, add) in h.iter_mut().zip(mlp) {
                *value += add * self.residual_multiplier;
            }
            h
        }
    }
    /// The Granite decoder: token embeddings, `num_hidden_layers` blocks with
    /// the residual multiplier, final RMS norm, untied LM head, and the
    /// logits scaling applied at [`GraniteLlm::logits`].
    struct GraniteLlm {
        embeddings: Vec<f32>,
        layers: Vec<TransformerBlock>,
        final_norm: RmsNorm,
        lm_head: Linear,
        embedding_multiplier: f32,
        logits_scaling: f32,
        vocab_size: usize,
        hidden_size: usize,
    }

    impl GraniteLlm {
        /// Raw token embeddings `[ids, hidden]`; the embedding multiplier is
        /// applied by the model wrapper, exactly as the reference does.
        fn embed(&self, token_ids: &[i32]) -> Result<Vec<f32>> {
            if token_ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.vocab_size)
            {
                return Err(SpeechError::Input {
                    why: "text prompt contains a token outside the checkpoint vocabulary".into(),
                });
            }
            Ok(ops::embedding(
                &self.embeddings,
                self.hidden_size,
                token_ids,
            ))
        }

        /// Evaluate the prompt once and retain its rotated K/V projections;
        /// returns the last row's post-norm hidden state.
        fn prefill(&self, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
            let mut hidden = input.to_vec();
            let mut cache = DecodeCache {
                layers: self
                    .layers
                    .iter()
                    .map(|layer| AttentionCache::new(layer.attention.key_value_heads))
                    .collect(),
            };
            for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
                hidden = layer.forward(&hidden, rows, Some(layer_cache));
            }
            self.final_norm.apply(&mut hidden, rows);
            (hidden[(rows - 1) * self.hidden_size..].to_vec(), cache)
        }

        /// Evaluate one new embedding row against the growing cache.
        fn step(&self, input: &[f32], cache: &mut DecodeCache) -> Vec<f32> {
            let mut hidden = input.to_vec();
            for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
                hidden = layer.step(&hidden, layer_cache);
            }
            self.final_norm.apply(&mut hidden, 1);
            hidden
        }

        /// LM head logits divided by `logits_scaling` (the reference
        /// `Model.__call__` tail).
        fn logits(&self, hidden: &[f32]) -> Vec<f32> {
            let mut logits = self.lm_head.forward(hidden, 1);
            for value in &mut logits {
                *value /= self.logits_scaling;
            }
            logits
        }
    }

    // ---- test fixture ----

    const HIDDEN: usize = 16;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const INTERMEDIATE: usize = 24;
    const LAYERS: usize = 2;
    const VOCAB: usize = 29;
    const ATTENTION_MULTIPLIER: f32 = 0.37;
    const RESIDUAL_MULTIPLIER: f32 = 0.22;
    const EMBEDDING_MULTIPLIER: f32 = 12.0;
    const LOGITS_SCALING: f32 = 8.0;
    const EPS: f32 = 1e-5;
    const THETA: f32 = 10_000.0;

    struct Rng(u32);

    impl Rng {
        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        }

        fn linear(&mut self, input: usize, output: usize) -> Linear {
            Linear::new(self.vec(input * output, 0.6), None, input, output)
        }

        fn norm(&mut self, width: usize) -> RmsNorm {
            RmsNorm::new(self.vec(width, 1.0).iter().map(|v| v + 1.0).collect(), EPS)
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Builds the same random weights into both stacks.
    fn build(seed: u32) -> (GraniteLlm, SharedLlm) {
        let mut rng = Rng(seed);
        let embeddings = rng.vec(VOCAB * HIDDEN, 1.0);
        let mut old_layers = Vec::new();
        let mut new_layers = Vec::new();
        for _ in 0..LAYERS {
            let (q, k, v, o) = (
                rng.linear(HIDDEN, HEADS * HEAD_DIM),
                rng.linear(HIDDEN, KV_HEADS * HEAD_DIM),
                rng.linear(HIDDEN, KV_HEADS * HEAD_DIM),
                rng.linear(HEADS * HEAD_DIM, HIDDEN),
            );
            let (gate, up, down) = (
                rng.linear(HIDDEN, INTERMEDIATE),
                rng.linear(HIDDEN, INTERMEDIATE),
                rng.linear(INTERMEDIATE, HIDDEN),
            );
            let (input_norm, post_norm) = (rng.norm(HIDDEN), rng.norm(HIDDEN));
            old_layers.push(TransformerBlock {
                attention: GraniteAttention {
                    q_proj: q.clone(),
                    k_proj: k.clone(),
                    v_proj: v.clone(),
                    o_proj: o.clone(),
                    query_heads: HEADS,
                    key_value_heads: KV_HEADS,
                    head_dim: HEAD_DIM,
                    rotary_dim: HEAD_DIM,
                    theta: THETA,
                    scale: ATTENTION_MULTIPLIER,
                },
                mlp: Mlp {
                    gate: gate.clone(),
                    up: up.clone(),
                    down: down.clone(),
                    intermediate: INTERMEDIATE,
                },
                input_norm: input_norm.clone(),
                post_attention_norm: post_norm.clone(),
                residual_multiplier: RESIDUAL_MULTIPLIER,
                hidden: HIDDEN,
            });
            new_layers.push(DecoderLayer {
                attention: TextAttention {
                    q_proj: q,
                    k_proj: k,
                    v_proj: v,
                    o_proj: o,
                    q_norm: None,
                    k_norm: None,
                    query_heads: HEADS,
                    key_value_heads: KV_HEADS,
                    head_dim: HEAD_DIM,
                    rotary_dim: HEAD_DIM,
                    theta: THETA,
                    scale: ATTENTION_MULTIPLIER,
                },
                mlp: SharedMlp {
                    gate,
                    up,
                    down,
                    intermediate: INTERMEDIATE,
                },
                input_norm,
                post_attention_norm: post_norm,
                residual_multiplier: RESIDUAL_MULTIPLIER,
                hidden: HIDDEN,
            });
        }
        let final_norm = rng.norm(HIDDEN);
        let lm_head = rng.linear(HIDDEN, VOCAB);
        let old = GraniteLlm {
            embeddings: embeddings.clone(),
            layers: old_layers,
            final_norm: final_norm.clone(),
            lm_head: lm_head.clone(),
            embedding_multiplier: EMBEDDING_MULTIPLIER,
            logits_scaling: LOGITS_SCALING,
            vocab_size: VOCAB,
            hidden_size: HIDDEN,
        };
        let new = SharedLlm {
            decoder: Decoder {
                embeddings,
                lm_head: Some(lm_head),
                layers: new_layers,
                final_norm,
                vocab_size: VOCAB,
                hidden_size: HIDDEN,
            },
            embedding_multiplier: EMBEDDING_MULTIPLIER,
            logits_scaling: LOGITS_SCALING,
        };
        (old, new)
    }

    fn prompt_embeddings(old: &GraniteLlm, ids: &[i32]) -> Vec<f32> {
        // The wrapper's job: embed, then scale once by the multiplier.
        let mut embeddings = old.embed(ids).unwrap();
        for value in &mut embeddings {
            *value *= old.embedding_multiplier;
        }
        embeddings
    }

    #[test]
    fn prefill_step_and_logits_match_the_retired_stack_bitwise() {
        let (old, new) = build(7);
        let ids = [3, 17, 0, 28, 5, 11, 9];
        let embeddings = prompt_embeddings(&old, &ids);
        let new_embeddings = {
            let mut e = new.embed(&ids).unwrap();
            for value in &mut e {
                *value *= new.embedding_multiplier;
            }
            e
        };
        assert_eq!(bits(&embeddings), bits(&new_embeddings));

        let (old_hidden, mut old_cache) = old.prefill(&embeddings, ids.len());
        let (new_hidden, mut new_cache) = new.prefill(&embeddings, ids.len());
        assert_eq!(bits(&old_hidden), bits(&new_hidden));
        assert_eq!(
            bits(&old.logits(&old_hidden)),
            bits(&new.logits(&new_hidden))
        );

        for token in [4, 22, 1, 13, 27] {
            let mut input = old.embed(&[token]).unwrap();
            for value in &mut input {
                *value *= old.embedding_multiplier;
            }
            let old_next = old.step(&input, &mut old_cache);
            let new_next = new.decoder.step(&input, &mut new_cache);
            assert_eq!(bits(&old_next), bits(&new_next), "token {token}");
            assert_eq!(bits(&old.logits(&old_next)), bits(&new.logits(&new_next)));
        }
    }

    /// The retired Granite greedy loop from `granite_speech/mod.rs`, with the
    /// stage recording reduced to the retained decision logits.
    fn retired_granite_loop(
        old: &GraniteLlm,
        embeddings: &[f32],
        max_tokens: usize,
        eos: i32,
    ) -> (Vec<u32>, Vec<f32>) {
        let rows = embeddings.len() / HIDDEN;
        let (mut last_hidden, mut cache) = old.prefill(embeddings, rows);
        let logits = old.logits(&last_hidden);
        let mut generated: Vec<u32> = Vec::new();
        let mut next = crate::nn::argmax(&logits) as u32;
        let mut decision: Vec<f32> = logits;
        for _ in 0..max_tokens {
            if next as i32 == eos {
                break;
            }
            generated.push(next);
            let token_id = i32::try_from(next).unwrap();
            let mut input = old.embed(&[token_id]).unwrap();
            for value in &mut input {
                *value *= old.embedding_multiplier;
            }
            last_hidden = old.step(&input, &mut cache);
            let new_decision = old.logits(&last_hidden);
            next = crate::nn::argmax(&new_decision) as u32;
            if next as i32 != eos {
                decision = new_decision;
            }
        }
        (generated, decision)
    }

    #[test]
    fn greedy_generate_matches_the_retired_granite_loop() {
        let (old, new) = build(11);
        let ids = [2, 8, 19, 4, 4, 25];
        let embeddings = prompt_embeddings(&old, &ids);
        // An unreachable eos runs to max_tokens; then pick an eos that the
        // run emits mid-way so the stop path is exercised too.
        let (free_run, _) = retired_granite_loop(&old, &embeddings, 12, -1);
        assert_eq!(free_run.len(), 12);
        for (max_tokens, eos) in [
            (0usize, -1i32),
            (1, -1),
            (12, -1),
            (12, free_run[3] as i32),
            (12, free_run[0] as i32),
        ] {
            let (want, want_decision) = retired_granite_loop(&old, &embeddings, max_tokens, eos);

            let rows = ids.len();
            let (last_hidden, cache) = new.prefill(&embeddings, rows);
            let logits = new.logits(&last_hidden);
            let mut decision: Vec<f32> = logits.clone();
            let got = greedy_generate(
                &new,
                &logits,
                cache,
                max_tokens,
                |next| next as i32 == eos,
                "Granite Speech",
                |step_logits, next| {
                    if next as i32 != eos {
                        decision = step_logits.to_vec();
                    }
                },
            )
            .unwrap();
            assert_eq!(got, want, "max_tokens {max_tokens} eos {eos}");
            assert_eq!(bits(&decision), bits(&want_decision));
        }
    }

    // Keep the shared cache type nameable here so a signature drift in
    // `prefill` fails to compile instead of silently diverging.
    fn _cache_type_check(cache: SharedCache) -> SharedCache {
        cache
    }
}
