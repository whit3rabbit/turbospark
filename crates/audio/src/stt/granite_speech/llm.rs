//! Granite Speech 1B text backbone: the llama-style Granite decoder with
//! its four multipliers.
//!
//! Reference: `mlx_audio/lm/models/granite.py` (`Model`, `GraniteModel`,
//! `TransformerBlock`, `Attention`, `MLP`) plus the speech wrapper
//! `Model.__call__` in `mlx_audio/stt/models/granite_speech/granite_speech.py`
//! at mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
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

use crate::ops;
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
    Ok(Linear {
        weight: load_vector(files, &format!("{base}.weight"), input * output)?,
        input,
        output,
    })
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(x, &self.weight, None, rows, self.input, self.output)
    }
}

struct RmsNorm {
    weight: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl RmsNorm {
    fn load(files: &[SafetensorsFile], base: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: load_vector(files, &format!("{base}.weight"), width)?,
            width,
            epsilon,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::rmsnorm(values, rows, self.width, &self.weight, self.epsilon);
    }
}

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
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        config: &super::config::TextConfig,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let query_width = config.num_attention_heads * head_dim;
        let kv_width = config.num_key_value_heads * head_dim;
        Ok(Self {
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
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim,
            rotary_dim: head_dim,
            theta: config.rope_theta,
            // Granite's attention_multiplier replaces head_dim^-0.5.
            scale: config.attention_multiplier,
        })
    }

    /// Causal attention over `x [rows, hidden]`, appending to the cache
    /// when present.
    fn forward(&self, x: &[f32], rows: usize, cache: Option<&mut AttentionCache>) -> Vec<f32> {
        let query_width = self.query_heads * self.head_dim;
        let query = self.q_proj.forward(x, rows);
        let key = self.k_proj.forward(x, rows);
        let value = self.v_proj.forward(x, rows);

        let mut query_heads = transpose_heads(&query, rows, self.query_heads, self.head_dim);
        let mut key_heads = transpose_heads(&key, rows, self.key_value_heads, self.head_dim);
        let value_heads = transpose_heads(&value, rows, self.key_value_heads, self.head_dim);
        let (cos, sin) = ops::rope_tables(rows, self.rotary_dim, self.theta);
        apply_rope_neox(
            &mut query_heads,
            self.query_heads,
            rows,
            self.head_dim,
            self.rotary_dim,
            &cos,
            &sin,
        );
        apply_rope_neox(
            &mut key_heads,
            self.key_value_heads,
            rows,
            self.head_dim,
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
        for head in 0..self.query_heads {
            let start = head * self.head_dim;
            apply_rope_neox_position(
                &mut query[start..start + self.head_dim],
                self.head_dim,
                self.rotary_dim,
                &cos,
                &sin,
            );
        }
        for head in 0..self.key_value_heads {
            let start = head * self.head_dim;
            apply_rope_neox_position(
                &mut key[start..start + self.head_dim],
                self.head_dim,
                self.rotary_dim,
                &cos,
                &sin,
            );
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
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        config: &super::config::TextConfig,
    ) -> Result<Self> {
        Ok(Self {
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
    fn load(
        files: &[SafetensorsFile],
        index: usize,
        config: &super::config::TextConfig,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{index}");
        Ok(Self {
            attention: GraniteAttention::load(files, &format!("{prefix}.self_attn"), config)?,
            mlp: Mlp::load(files, &format!("{prefix}.mlp"), config)?,
            input_norm: RmsNorm::load(
                files,
                &format!("{prefix}.input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            post_attention_norm: RmsNorm::load(
                files,
                &format!("{prefix}.post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            residual_multiplier: config.residual_multiplier,
            hidden: config.hidden_size,
        })
    }

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
pub struct GraniteLlm {
    embeddings: Vec<f32>,
    layers: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    lm_head: Linear,
    pub(crate) embedding_multiplier: f32,
    pub(crate) logits_scaling: f32,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
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
            .map(|index| TransformerBlock::load(files, index, config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::load(
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
            embeddings,
            layers,
            final_norm,
            lm_head,
            embedding_multiplier: config.embedding_multiplier,
            logits_scaling: config.logits_scaling,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
        })
    }

    /// Raw token embeddings `[ids, hidden]`; the embedding multiplier is
    /// applied by the model wrapper, exactly as the reference does.
    pub(crate) fn embed(&self, token_ids: &[i32]) -> Result<Vec<f32>> {
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
    pub(crate) fn prefill(&self, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
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
    pub(crate) fn step(&self, input: &[f32], cache: &mut DecodeCache) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.step(&hidden, layer_cache);
        }
        self.final_norm.apply(&mut hidden, 1);
        hidden
    }

    /// LM head logits divided by `logits_scaling` (the reference
    /// `Model.__call__` tail).
    pub(crate) fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = self.lm_head.forward(hidden, 1);
        for value in &mut logits {
            *value /= self.logits_scaling;
        }
        logits
    }
}

fn transpose_heads(input: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let source = (row * heads + head) * dim;
            let target = (head * rows + row) * dim;
            output[target..target + dim].copy_from_slice(&input[source..source + dim]);
        }
    }
    output
}

fn apply_rope_neox(
    values: &mut [f32],
    heads: usize,
    rows: usize,
    dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    debug_assert!(rotary_dim <= dim && rotary_dim % 2 == 0);
    let half = rotary_dim / 2;
    for head in 0..heads {
        for row in 0..rows {
            let base = (head * rows + row) * dim;
            for feature in 0..half {
                let a = values[base + feature];
                let b = values[base + half + feature];
                let table = row * half + feature;
                values[base + feature] = a * cos[table] - b * sin[table];
                values[base + half + feature] = b * cos[table] + a * sin[table];
            }
        }
    }
}

fn apply_rope_neox_position(
    values: &mut [f32],
    dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    debug_assert!(rotary_dim <= dim && rotary_dim % 2 == 0);
    let half = rotary_dim / 2;
    for feature in 0..half {
        let a = values[feature];
        let b = values[half + feature];
        values[feature] = a * cos[feature] - b * sin[feature];
        values[feature + half] = b * cos[feature] + a * sin[feature];
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_rope_neox, apply_rope_neox_position, transpose_heads};

    #[test]
    fn head_transpose_and_neox_rope_match_known_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            transpose_heads(&x, 2, 2, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        let mut q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (cos, sin) = crate::ops::rope_tables(2, 4, 1.0);
        apply_rope_neox(&mut q, 1, 2, 4, 4, &cos, &sin);
        assert_eq!(q[0], 1.0);
        assert_eq!(q[1], 0.0);
        assert!((q[4] - 0.5403023).abs() < 1e-6);
        assert!((q[6] - 0.84147096).abs() < 1e-6);
    }

    #[test]
    fn single_position_rope_matches_a_slice_of_the_full_table() {
        let (cos, sin) = crate::ops::rope_tables_range(7, 1, 4, 100.0);
        let mut single = vec![0.5, -0.25, 2.0, 1.0];
        apply_rope_neox_position(&mut single, 4, 4, &cos, &sin);
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
