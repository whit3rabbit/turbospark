//! Qwen3 text decoder and affine-weight loader used by Qwen3-ASR.
//!
//! The prompt is evaluated once, then each greedy token appends to a
//! per-layer KV cache. The full-prefix path remains the parity reference.
//!
//! Every loader accepts a shard list: single-file callers pass one file
//! through [`Decoder::load`] and behavior is unchanged. The shard-aware
//! resolution exists for Higgs Audio v3, whose checkpoint stores the Qwen3
//! backbone under the raw HF names (`layers.*`, `embed_tokens.weight`,
//! `norm.weight`) across `model.safetensors` shards, with the untied LM head
//! under `audio_decoder_proj.text_lm_head.weight`.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::qwen3_asr::config::TextConfig;
use crate::ops;
use crate::quant::{load_quantized, QuantScheme};
use crate::{Result, SpeechError};

/// Resolves `base` against the shards, returning the carrying file plus the
/// actual key prefix. Candidate order preserves the historical single-file
/// behavior exactly; the last two candidates are the Higgs Audio v3
/// additions and can only match where every earlier candidate misses.
fn resolve_weight<'a>(files: &'a [SafetensorsFile], base: &str) -> (&'a SafetensorsFile, String) {
    let mut candidates = vec![
        format!("{base}.weight"),
        format!("thinker.{base}.weight"),
        format!("llm.{base}.weight"),
    ];
    if let Some(rest) = base.strip_prefix("model.") {
        candidates.push(format!("model.language_model.{rest}.weight"));
        // Higgs Audio v3 stores the shared qwen3 decoder under the raw HF
        // names without any "model." prefix.
        candidates.push(format!("{rest}.weight"));
    }
    if base == "lm_head" {
        // Higgs Audio v3 keeps the untied text LM head under its audio
        // decoder projection name; the reference sanitize renames it to
        // "lm_head.weight" at load.
        candidates.push("audio_decoder_proj.text_lm_head.weight".to_owned());
    }
    for name in &candidates {
        if let Some(file) = files.iter().find(|f| f.contains_tensor(name)) {
            return (file, name[..name.len() - ".weight".len()].to_owned());
        }
    }
    // Nothing resolved: hand back the first shard and the requested base so
    // the downstream load fails with the familiar missing-tensor error.
    (&files[0], base.to_owned())
}

#[derive(Clone)]
pub(crate) struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    pub(crate) fn load(
        file: &SafetensorsFile,
        base: &str,
        input: usize,
        output: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        Self::load_sharded(std::slice::from_ref(file), base, input, output, scheme)
    }

    pub(crate) fn load_sharded(
        files: &[SafetensorsFile],
        base: &str,
        input: usize,
        output: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let (file, base) = resolve_weight(files, base);
        let (weight, bias) = load_quantized(file, &base, scheme)?;
        if weight.len() != input * output {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: format!(
                    "expected dequantized shape [{output}, {input}], got {} values",
                    weight.len()
                ),
            });
        }
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }
}

struct RmsNorm {
    weight: Vec<f32>,
    width: usize,
    epsilon: f32,
}

impl RmsNorm {
    fn load_sharded(
        files: &[SafetensorsFile],
        base: &str,
        width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        let (file, base) = resolve_weight(files, base);
        let name = format!("{base}.weight");
        let values = file.load_as_f32(&name)?;
        if values.len() != width {
            return Err(SpeechError::Tensor {
                name,
                why: format!("expected {width} RMS norm weights, got {}", values.len()),
            });
        }
        Ok(Self {
            weight: values,
            width,
            epsilon,
        })
    }

    fn apply(&self, values: &mut [f32], rows: usize) {
        ops::rmsnorm(values, rows, self.width, &self.weight, self.epsilon);
    }
}

struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
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

pub(crate) struct DecodeCache {
    layers: Vec<AttentionCache>,
}

impl TextAttention {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            q_proj: Linear::load_sharded(
                files,
                &format!("{prefix}.q_proj"),
                config.hidden_size,
                q_width,
                scheme,
            )?,
            k_proj: Linear::load_sharded(
                files,
                &format!("{prefix}.k_proj"),
                config.hidden_size,
                kv_width,
                scheme,
            )?,
            v_proj: Linear::load_sharded(
                files,
                &format!("{prefix}.v_proj"),
                config.hidden_size,
                kv_width,
                scheme,
            )?,
            o_proj: Linear::load_sharded(
                files,
                &format!("{prefix}.o_proj"),
                q_width,
                config.hidden_size,
                scheme,
            )?,
            q_norm: config
                .qk_norm
                .then(|| {
                    RmsNorm::load_sharded(
                        files,
                        &format!("{prefix}.q_norm"),
                        config.head_dim,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            k_norm: config
                .qk_norm
                .then(|| {
                    RmsNorm::load_sharded(
                        files,
                        &format!("{prefix}.k_norm"),
                        config.head_dim,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            theta: config.rope_theta,
        })
    }

    fn forward_with_cache(
        &self,
        x: &[f32],
        rows: usize,
        cache: Option<&mut AttentionCache>,
    ) -> Vec<f32> {
        let query_width = self.query_heads * self.head_dim;
        let kv_width = self.key_value_heads * self.head_dim;
        let mut query = self.q_proj.forward(x, rows);
        let mut key = self.k_proj.forward(x, rows);
        let value = self.v_proj.forward(x, rows);
        if let Some(norm) = &self.q_norm {
            norm.apply(&mut query, rows * self.query_heads);
        }
        if let Some(norm) = &self.k_norm {
            norm.apply(&mut key, rows * self.key_value_heads);
        }

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

        if let Some(cache) = cache {
            for head in 0..self.key_value_heads {
                let start = head * rows * self.head_dim;
                let end = start + rows * self.head_dim;
                cache.keys[head].extend_from_slice(&key_heads[start..end]);
                cache.values[head].extend_from_slice(&value_heads[start..end]);
            }
            cache.len = rows;
        }

        let key_heads = ops::repeat_kv(
            &key_heads,
            self.key_value_heads,
            rows,
            self.head_dim,
            self.query_heads,
        );
        let value_heads = ops::repeat_kv(
            &value_heads,
            self.key_value_heads,
            rows,
            self.head_dim,
            self.query_heads,
        );
        let mut attended_heads = vec![0.0f32; query_width * rows];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for head in 0..self.query_heads {
            let head_start = head * rows * self.head_dim;
            let query_head = &query_heads[head_start..head_start + rows * self.head_dim];
            let key_head = &key_heads[head_start..head_start + rows * self.head_dim];
            let value_head = &value_heads[head_start..head_start + rows * self.head_dim];
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
                    scale,
                );
                let target = position * query_width + head * self.head_dim;
                attended_heads[target..target + self.head_dim].copy_from_slice(&output);
            }
        }
        debug_assert_eq!(kv_width, self.key_value_heads * self.head_dim);
        self.o_proj.forward(&attended_heads, rows)
    }

    fn step(&self, x: &[f32], cache: &mut AttentionCache) -> Vec<f32> {
        let query_width = self.query_heads * self.head_dim;
        let mut query = self.q_proj.forward(x, 1);
        let mut key = self.k_proj.forward(x, 1);
        let value = self.v_proj.forward(x, 1);
        if let Some(norm) = &self.q_norm {
            norm.apply(&mut query, self.query_heads);
        }
        if let Some(norm) = &self.k_norm {
            norm.apply(&mut key, self.key_value_heads);
        }
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
        let mut attended = vec![0.0f32; query_width];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
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
                scale,
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
    intermediate: usize,
}

impl Mlp {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        Ok(Self {
            gate: Linear::load_sharded(
                files,
                &format!("{prefix}.gate_proj"),
                config.hidden_size,
                config.intermediate_size,
                scheme,
            )?,
            up: Linear::load_sharded(
                files,
                &format!("{prefix}.up_proj"),
                config.hidden_size,
                config.intermediate_size,
                scheme,
            )?,
            down: Linear::load_sharded(
                files,
                &format!("{prefix}.down_proj"),
                config.intermediate_size,
                config.hidden_size,
                scheme,
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

struct DecoderLayer {
    attention: TextAttention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    hidden: usize,
}

impl DecoderLayer {
    fn load(
        files: &[SafetensorsFile],
        index: usize,
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{index}");
        Ok(Self {
            attention: TextAttention::load(files, &format!("{prefix}.self_attn"), config, scheme)?,
            mlp: Mlp::load(files, &format!("{prefix}.mlp"), config, scheme)?,
            input_norm: RmsNorm::load_sharded(
                files,
                &format!("{prefix}.input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            post_attention_norm: RmsNorm::load_sharded(
                files,
                &format!("{prefix}.post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            hidden: config.hidden_size,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.forward_with_cache(x, rows, None)
    }

    fn forward_with_cache(
        &self,
        x: &[f32],
        rows: usize,
        cache: Option<&mut AttentionCache>,
    ) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.input_norm.apply(&mut normalized, rows);
        let attention = self.attention.forward_with_cache(&normalized, rows, cache);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }

        let mut normalized = residual.clone();
        self.post_attention_norm.apply(&mut normalized, rows);
        let mlp = self.mlp.forward(&normalized, rows);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        debug_assert_eq!(residual.len(), rows * self.hidden);
        residual
    }

    fn step(&self, x: &[f32], cache: &mut AttentionCache) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.input_norm.apply(&mut normalized, 1);
        let attention = self.attention.step(&normalized, cache);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }
        let mut normalized = residual.clone();
        self.post_attention_norm.apply(&mut normalized, 1);
        let mlp = self.mlp.forward(&normalized, 1);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        residual
    }
}

pub(crate) struct Decoder {
    pub(crate) embeddings: Vec<f32>,
    lm_head: Option<Linear>,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
}

impl Decoder {
    pub(crate) fn load(
        file: &SafetensorsFile,
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        Self::load_sharded(std::slice::from_ref(file), config, scheme)
    }

    /// Loads the decoder from one or more safetensors shards. Weights are
    /// resolved per tensor, so a family may keep its backbone and LM head in
    /// different shards (Higgs Audio v3 does).
    pub(crate) fn load_sharded(
        files: &[SafetensorsFile],
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        let (embedding_file, embedding_base) = resolve_weight(files, "model.embed_tokens");
        let (embeddings, _) = load_quantized(embedding_file, &embedding_base, scheme)?;
        let expected = config.vocab_size * config.hidden_size;
        if embeddings.len() != expected {
            return Err(SpeechError::Tensor {
                name: format!("{embedding_base}.weight"),
                why: format!(
                    "expected {expected} embedding values, got {}",
                    embeddings.len()
                ),
            });
        }
        let layers = (0..config.num_hidden_layers)
            .map(|index| DecoderLayer::load(files, index, config, scheme))
            .collect::<Result<Vec<_>>>()?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Linear::load_sharded(
                files,
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                scheme,
            )?)
        };
        let final_norm =
            RmsNorm::load_sharded(files, "model.norm", config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            embeddings,
            lm_head,
            layers,
            final_norm,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
        })
    }

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

    pub(crate) fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, rows);
        }
        self.final_norm.apply(&mut hidden, rows);
        hidden
    }

    /// Evaluate the prompt once and retain its rotated K/V projections.
    pub(crate) fn prefill(&self, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
        let mut hidden = input.to_vec();
        let mut cache = DecodeCache {
            layers: self
                .layers
                .iter()
                .map(|layer| AttentionCache::new(layer.attention.key_value_heads))
                .collect(),
        };
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward_with_cache(&hidden, rows, Some(layer_cache));
        }
        self.final_norm.apply(&mut hidden, rows);
        (hidden[(rows - 1) * self.hidden_size..].to_vec(), cache)
    }

    /// Evaluate one new token against the growing cache.
    pub(crate) fn step(&self, input: &[f32], cache: &mut DecodeCache) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.step(&hidden, layer_cache);
        }
        self.final_norm.apply(&mut hidden, 1);
        hidden
    }

    pub(crate) fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(hidden, 1)
        } else {
            ops::linear(
                hidden,
                &self.embeddings,
                None,
                1,
                self.hidden_size,
                self.vocab_size,
            )
        }
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
        values[half + feature] = b * cos[feature] + a * sin[feature];
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_rope_neox, apply_rope_neox_position, transpose_heads, Decoder, DecoderLayer, Linear,
        Mlp, RmsNorm, TextAttention,
    };
    use crate::ops;

    #[test]
    fn head_transpose_and_neox_rope_match_known_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            transpose_heads(&x, 2, 2, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        let mut q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (cos, sin) = ops::rope_tables(2, 4, 1.0);
        apply_rope_neox(&mut q, 1, 2, 4, 4, &cos, &sin);
        assert_eq!(q[0], 1.0);
        assert_eq!(q[1], 0.0);
        assert!((q[4] - 0.5403023).abs() < 1e-6);
        assert!((q[6] - 0.84147096).abs() < 1e-6);
    }

    #[test]
    fn partial_rope_leaves_the_head_suffix_untouched() {
        let (cos, sin) = ops::rope_tables_range(1, 1, 4, 10_000.0);
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let suffix = values[4..].to_vec();
        apply_rope_neox_position(&mut values, 8, 4, &cos, &sin);
        assert_eq!(&values[4..], suffix);
    }

    #[test]
    fn incremental_cache_matches_full_prefix_with_grouped_query_heads() {
        fn linear(input: usize, output: usize, seed: usize) -> Linear {
            Linear {
                weight: (0..input * output)
                    .map(|index| ((index * 7 + seed) % 19) as f32 * 0.013 - 0.11)
                    .collect(),
                bias: None,
                input,
                output,
            }
        }
        fn norm(width: usize) -> RmsNorm {
            RmsNorm {
                weight: vec![1.0; width],
                width,
                epsilon: 1e-5,
            }
        }
        let hidden = 8;
        let attention = TextAttention {
            q_proj: linear(hidden, hidden, 1),
            k_proj: linear(hidden, 4, 2),
            v_proj: linear(hidden, 4, 3),
            o_proj: linear(hidden, hidden, 4),
            q_norm: Some(norm(2)),
            k_norm: Some(norm(2)),
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 2,
            rotary_dim: 2,
            theta: 10_000.0,
        };
        let layer = DecoderLayer {
            attention,
            mlp: Mlp {
                gate: linear(hidden, hidden, 5),
                up: linear(hidden, hidden, 6),
                down: linear(hidden, hidden, 7),
                intermediate: hidden,
            },
            input_norm: norm(hidden),
            post_attention_norm: norm(hidden),
            hidden,
        };
        let decoder = Decoder {
            embeddings: Vec::new(),
            lm_head: None,
            layers: vec![layer],
            final_norm: norm(hidden),
            vocab_size: 0,
            hidden_size: hidden,
        };
        let input: Vec<f32> = (0..5 * hidden)
            .map(|index| ((index * 11 + 3) % 23) as f32 * 0.041 - 0.32)
            .collect();
        let (prefill_last, mut cache) = decoder.prefill(&input[..3 * hidden], 3);
        let full = decoder.forward(&input[..3 * hidden], 3);
        for (cached, reference) in prefill_last.iter().zip(&full[2 * hidden..]) {
            assert!((cached - reference).abs() < 1e-6);
        }
        for rows in 4..=5 {
            let start = (rows - 1) * hidden;
            let cached = decoder.step(&input[start..start + hidden], &mut cache);
            let full = decoder.forward(&input[..rows * hidden], rows);
            for (index, (value, reference)) in cached.iter().zip(&full[start..]).enumerate() {
                assert!(
                    (value - reference).abs() < 1e-5,
                    "row {rows} column {index}: cached {value}, full {reference}"
                );
            }
            assert_eq!(cache.layers[0].len, rows);
            assert_eq!(cache.layers[0].keys[0].len(), rows * 2);
            assert_eq!(cache.layers[0].keys[1].len(), rows * 2);
        }
    }
}
