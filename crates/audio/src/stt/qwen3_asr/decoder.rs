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
pub(crate) use crate::nn::Linear;
use crate::nn::RmsNorm;
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

/// Quantization-aware, shard-aware loader for the shared [`Linear`]. The
/// plain f32 loader in `nn` cannot resolve Higgs/MOSS tensor names or
/// dequantize, so this family keeps its own loading and only builds the layer
/// through [`Linear::new`].
pub(crate) fn load_linear(
    file: &SafetensorsFile,
    base: &str,
    input: usize,
    output: usize,
    scheme: QuantScheme,
) -> Result<Linear> {
    load_linear_sharded(std::slice::from_ref(file), base, input, output, scheme)
}

pub(crate) fn load_linear_sharded(
    files: &[SafetensorsFile],
    base: &str,
    input: usize,
    output: usize,
    scheme: QuantScheme,
) -> Result<Linear> {
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
    Ok(Linear::new(weight, bias, input, output))
}

fn load_rms_norm_sharded(
    files: &[SafetensorsFile],
    base: &str,
    width: usize,
    epsilon: f32,
) -> Result<RmsNorm> {
    let (file, base) = resolve_weight(files, base);
    let name = format!("{base}.weight");
    let values = file.load_as_f32(&name)?;
    if values.len() != width {
        return Err(SpeechError::Tensor {
            name,
            why: format!("expected {width} RMS norm weights, got {}", values.len()),
        });
    }
    Ok(RmsNorm::new(values, epsilon))
}

/// Grouped-query attention with a KV cache. Granite's backbone reuses it
/// (see `granite_speech::llm`), which is why `scale` is a field: Qwen3 uses
/// `head_dim^-0.5`, Granite its `attention_multiplier`.
pub(crate) struct TextAttention {
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: Option<RmsNorm>,
    pub(crate) k_norm: Option<RmsNorm>,
    pub(crate) query_heads: usize,
    pub(crate) key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
    pub(crate) scale: f32,
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
            q_proj: load_linear_sharded(
                files,
                &format!("{prefix}.q_proj"),
                config.hidden_size,
                q_width,
                scheme,
            )?,
            k_proj: load_linear_sharded(
                files,
                &format!("{prefix}.k_proj"),
                config.hidden_size,
                kv_width,
                scheme,
            )?,
            v_proj: load_linear_sharded(
                files,
                &format!("{prefix}.v_proj"),
                config.hidden_size,
                kv_width,
                scheme,
            )?,
            o_proj: load_linear_sharded(
                files,
                &format!("{prefix}.o_proj"),
                q_width,
                config.hidden_size,
                scheme,
            )?,
            q_norm: config
                .qk_norm
                .then(|| {
                    load_rms_norm_sharded(
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
                    load_rms_norm_sharded(
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
            scale: 1.0 / (config.head_dim as f32).sqrt(),
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

        if let Some(cache) = cache {
            for head in 0..self.key_value_heads {
                let start = head * rows * self.head_dim;
                let end = start + rows * self.head_dim;
                cache.keys[head].extend_from_slice(&key_heads[start..end]);
                cache.values[head].extend_from_slice(&value_heads[start..end]);
            }
            cache.len = rows;
        }

        let mut attended_heads = vec![0.0f32; query_width * rows];
        let scale = self.scale;
        // Grouped-query heads share one KV head; index it directly instead
        // of materializing `ops::repeat_kv` copies of every KV head.
        debug_assert_eq!(self.query_heads % self.key_value_heads, 0);
        let group = self.query_heads / self.key_value_heads;
        for head in 0..self.query_heads {
            let head_start = head * rows * self.head_dim;
            let kv_start = (head / group) * rows * self.head_dim;
            let query_head = &query_heads[head_start..head_start + rows * self.head_dim];
            let key_head = &key_heads[kv_start..kv_start + rows * self.head_dim];
            let value_head = &value_heads[kv_start..kv_start + rows * self.head_dim];
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

    /// `cos`/`sin` are the one-row rope tables for the cache's next
    /// position, built once per token by the caller (every layer shares the
    /// rotary geometry).
    fn step(&self, x: &[f32], cache: &mut AttentionCache, cos: &[f32], sin: &[f32]) -> Vec<f32> {
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
        // One position per head: `[heads * head_dim]` is already head-major
        // with a single row, so one call rotates every head.
        ops::rope_neox(&mut query, self.query_heads, 1, self.rotary_dim, cos, sin);
        ops::rope_neox(&mut key, self.key_value_heads, 1, self.rotary_dim, cos, sin);
        for head in 0..self.key_value_heads {
            let start = head * self.head_dim;
            cache.keys[head].extend_from_slice(&key[start..start + self.head_dim]);
            cache.values[head].extend_from_slice(&value[start..start + self.head_dim]);
        }
        cache.len += 1;
        let mut attended = vec![0.0f32; query_width];
        let scale = self.scale;
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

pub(crate) struct Mlp {
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
    pub(crate) intermediate: usize,
}

impl Mlp {
    fn load(
        files: &[SafetensorsFile],
        prefix: &str,
        config: &TextConfig,
        scheme: QuantScheme,
    ) -> Result<Self> {
        Ok(Self {
            gate: load_linear_sharded(
                files,
                &format!("{prefix}.gate_proj"),
                config.hidden_size,
                config.intermediate_size,
                scheme,
            )?,
            up: load_linear_sharded(
                files,
                &format!("{prefix}.up_proj"),
                config.hidden_size,
                config.intermediate_size,
                scheme,
            )?,
            down: load_linear_sharded(
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

pub(crate) struct DecoderLayer {
    pub(crate) attention: TextAttention,
    pub(crate) mlp: Mlp,
    pub(crate) input_norm: RmsNorm,
    pub(crate) post_attention_norm: RmsNorm,
    /// Scales both branch outputs before the residual add. Exactly 1.0 for
    /// Qwen3 (`x * 1.0 == x`, so the add is unchanged); Granite's
    /// `residual_multiplier` otherwise.
    pub(crate) residual_multiplier: f32,
    pub(crate) hidden: usize,
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
            input_norm: load_rms_norm_sharded(
                files,
                &format!("{prefix}.input_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            post_attention_norm: load_rms_norm_sharded(
                files,
                &format!("{prefix}.post_attention_layernorm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            residual_multiplier: 1.0,
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
            *value += add * self.residual_multiplier;
        }

        let mut normalized = residual.clone();
        self.post_attention_norm.apply(&mut normalized, rows);
        let mlp = self.mlp.forward(&normalized, rows);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add * self.residual_multiplier;
        }
        debug_assert_eq!(residual.len(), rows * self.hidden);
        residual
    }

    fn step(&self, x: &[f32], cache: &mut AttentionCache, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.input_norm.apply(&mut normalized, 1);
        let attention = self.attention.step(&normalized, cache, cos, sin);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add * self.residual_multiplier;
        }
        let mut normalized = residual.clone();
        self.post_attention_norm.apply(&mut normalized, 1);
        let mlp = self.mlp.forward(&normalized, 1);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add * self.residual_multiplier;
        }
        residual
    }
}

pub(crate) struct Decoder {
    pub(crate) embeddings: Vec<f32>,
    pub(crate) lm_head: Option<Linear>,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RmsNorm,
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
            Some(load_linear_sharded(
                files,
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                scheme,
            )?)
        };
        let final_norm =
            load_rms_norm_sharded(files, "model.norm", config.hidden_size, config.rms_norm_eps)?;
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
        if let Some(first) = self.layers.first() {
            let attention = &first.attention;
            let (cos, sin) = ops::rope_tables_range(
                cache.layers[0].len,
                1,
                attention.rotary_dim,
                attention.theta,
            );
            for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
                hidden = layer.step(&hidden, layer_cache, &cos, &sin);
            }
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

/// The two operations a greedy loop needs beyond [`Decoder::step`]. Granite
/// wraps a [`Decoder`] with an embedding multiplier and a logits scale, so
/// the loop is generic over this instead of over the struct.
pub(crate) trait TokenDecoder {
    /// Embeds one generated token and evaluates it against the cache,
    /// returning the new post-norm hidden row.
    fn next_hidden(&self, token: i32, cache: &mut DecodeCache) -> Result<Vec<f32>>;
    fn logits(&self, hidden: &[f32]) -> Vec<f32>;
}

impl TokenDecoder for Decoder {
    fn next_hidden(&self, token: i32, cache: &mut DecodeCache) -> Result<Vec<f32>> {
        let embedding = self.embed(&[token])?;
        Ok(self.step(&embedding, cache))
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        Decoder::logits(self, hidden)
    }
}

/// Token ids of `<|im_end|>` and `<|endoftext|>` that exist in `tokenizer`.
pub(crate) fn chat_stop_ids(tokenizer: &turbospark_tokenizer::Tokenizer) -> Vec<u32> {
    ["<|im_end|>", "<|endoftext|>"]
        .into_iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect()
}

/// Stop test for [`chat_stop_ids`]. A tokenizer without either token falls
/// back to the Qwen3 ids so generation still terminates.
pub(crate) fn is_chat_stop(stop_ids: &[u32], token: u32) -> bool {
    stop_ids.contains(&token) || (stop_ids.is_empty() && matches!(token, 151_645 | 151_643))
}

/// Greedy decode shared by every Qwen3-backbone family.
///
/// `first_logits` are the logits of the prefill hidden row and `cache` is
/// the cache from [`Decoder::prefill`]. Each emitted
/// token is embedded and stepped, then the next decision is the argmax of
/// the new logits; `observe(logits, next)` sees every one of those
/// post-step logits (not the first). The step after the final emitted token
/// still runs its logits so observers see the decision that follows the
/// last token, as the per-family loops always did. `overflow_label` names
/// the family in the signed 32-bit overflow error.
pub(crate) fn greedy_generate<D: TokenDecoder>(
    decoder: &D,
    first_logits: &[f32],
    mut cache: DecodeCache,
    max_tokens: usize,
    is_stop: impl Fn(u32) -> bool,
    overflow_label: &str,
    mut observe: impl FnMut(&[f32], u32),
) -> Result<Vec<u32>> {
    let mut generated: Vec<u32> = Vec::new();
    let mut next = crate::nn::argmax(first_logits) as u32;
    for _ in 0..max_tokens {
        if is_stop(next) {
            break;
        }
        generated.push(next);
        let token_id = i32::try_from(next).map_err(|_| SpeechError::Input {
            why: format!("{overflow_label} generated token id exceeds signed 32-bit range"),
        })?;
        let hidden = decoder.next_hidden(token_id, &mut cache)?;
        let step_logits = decoder.logits(&hidden);
        next = crate::nn::argmax(&step_logits) as u32;
        observe(&step_logits, next);
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::{
        greedy_generate, is_chat_stop, Decoder, DecoderLayer, Linear, Mlp, RmsNorm, TextAttention,
    };
    use crate::ops;

    #[test]
    fn head_transpose_and_neox_rope_match_known_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            ops::split_heads(&x, 2, 2, 2),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        let mut q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (cos, sin) = ops::rope_tables(2, 4, 1.0);
        ops::rope_neox(&mut q, 1, 2, 4, &cos, &sin);
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
        ops::rope_neox(&mut values, 1, 1, 4, &cos, &sin);
        assert_eq!(&values[4..], suffix);
    }

    #[test]
    fn incremental_cache_matches_full_prefix_with_grouped_query_heads() {
        fn linear(input: usize, output: usize, seed: usize) -> Linear {
            Linear::new(
                (0..input * output)
                    .map(|index| ((index * 7 + seed) % 19) as f32 * 0.013 - 0.11)
                    .collect(),
                None,
                input,
                output,
            )
        }
        fn norm(width: usize) -> RmsNorm {
            RmsNorm::new(vec![1.0; width], 1e-5)
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
            scale: 1.0 / 2.0f32.sqrt(),
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
            residual_multiplier: 1.0,
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
    // ---- greedy_generate parity against the per-family loops it replaced ----

    const VOCAB: usize = 31;
    const HIDDEN: usize = 8;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn random_decoder(seed: u32, tied: bool) -> Decoder {
        let mut state = seed;
        let mut rand = move |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        let mut linear = |input: usize, output: usize| {
            Linear::new(rand(input * output, 0.8), None, input, output)
        };
        let layers = (0..2)
            .map(|_| DecoderLayer {
                attention: TextAttention {
                    q_proj: linear(HIDDEN, HIDDEN),
                    k_proj: linear(HIDDEN, 4),
                    v_proj: linear(HIDDEN, 4),
                    o_proj: linear(HIDDEN, HIDDEN),
                    q_norm: Some(RmsNorm::new(vec![1.0; 2], 1e-5)),
                    k_norm: Some(RmsNorm::new(vec![1.0; 2], 1e-5)),
                    query_heads: 4,
                    key_value_heads: 2,
                    head_dim: 2,
                    rotary_dim: 2,
                    theta: 10_000.0,
                    scale: 1.0 / 2.0f32.sqrt(),
                },
                mlp: Mlp {
                    gate: linear(HIDDEN, 12),
                    up: linear(HIDDEN, 12),
                    down: linear(12, HIDDEN),
                    intermediate: 12,
                },
                input_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
                post_attention_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
                residual_multiplier: 1.0,
                hidden: HIDDEN,
            })
            .collect();
        let lm_head = (!tied).then(|| linear(HIDDEN, VOCAB));
        let mut embeddings = Vec::new();
        for value in (0..VOCAB * HIDDEN).map(|i| ((i * 13 + 5) % 29) as f32 * 0.07 - 1.0) {
            embeddings.push(value);
        }
        Decoder {
            embeddings,
            lm_head,
            layers,
            final_norm: RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
            vocab_size: VOCAB,
            hidden_size: HIDDEN,
        }
    }

    /// Verbatim `argmax` fold the loops used.
    fn old_argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (id, &value)| {
                if value > best.1 {
                    (id, value)
                } else {
                    best
                }
            })
            .0 as u32
    }

    /// Retired `qwen3_asr` loop (lazy logits, fallback stop ids).
    fn old_lazy_loop(
        decoder: &Decoder,
        embeddings: &[f32],
        stop_ids: &[u32],
        max_tokens: usize,
    ) -> Vec<u32> {
        let rows = embeddings.len() / decoder.hidden_size;
        let (mut last_hidden, mut cache) = decoder.prefill(embeddings, rows);
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let logits = decoder.logits(&last_hidden);
            let next = logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (id, &value)| {
                    if value > best.1 {
                        (id, value)
                    } else {
                        best
                    }
                })
                .0;
            let next = u32::try_from(next).unwrap();
            if stop_ids.contains(&next)
                || (stop_ids.is_empty() && matches!(next, 151_645 | 151_643))
            {
                break;
            }
            generated.push(next);
            let token_id = i32::try_from(next).unwrap();
            let next_embedding = decoder.embed(&[token_id]).unwrap();
            last_hidden = decoder.step(&next_embedding, &mut cache);
        }
        generated
    }

    /// Retired Mega-ASR/Phonon loop (eager logits, first logits recorded only
    /// when the loop is entered) plus the Higgs per-step logits record.
    fn old_eager_loop(
        decoder: &Decoder,
        embeddings: &[f32],
        stop_ids: &[u32],
        max_tokens: usize,
    ) -> (Vec<u32>, Option<Vec<f32>>, Vec<Vec<f32>>) {
        let rows = embeddings.len() / decoder.hidden_size;
        let (mut last_hidden, mut cache) = decoder.prefill(embeddings, rows);
        let logits = decoder.logits(&last_hidden);
        let mut first_logits = None;
        let mut step_record = Vec::new();
        let mut generated: Vec<u32> = Vec::new();
        let mut next = old_argmax(&logits);
        for _ in 0..max_tokens {
            if first_logits.is_none() {
                first_logits = Some(logits.clone());
            }
            if stop_ids.contains(&next)
                || (stop_ids.is_empty() && matches!(next, 151_645 | 151_643))
            {
                break;
            }
            generated.push(next);
            let token_id = i32::try_from(next).unwrap();
            let next_embedding = decoder.embed(&[token_id]).unwrap();
            last_hidden = decoder.step(&next_embedding, &mut cache);
            let step_logits = decoder.logits(&last_hidden);
            next = old_argmax(&step_logits);
            step_record.push(step_logits);
        }
        (generated, first_logits, step_record)
    }

    #[test]
    fn greedy_generate_matches_the_retired_family_loops() {
        for tied in [false, true] {
            let decoder = random_decoder(5, tied);
            let ids = [1, 9, 30, 4, 4, 17];
            let embeddings = decoder.embed(&ids).unwrap();
            // A free run (no reachable stop) names tokens the run emits, so
            // the stop paths below are actually hit.
            let free = old_lazy_loop(&decoder, &embeddings, &[], 14);
            assert_eq!(free.len(), 14);
            let stop_sets: [Vec<u32>; 4] = [
                vec![],
                vec![free[3]],
                vec![free[0], 30],
                vec![VOCAB as u32 + 5],
            ];
            for stops in &stop_sets {
                for max_tokens in [0usize, 1, 2, 5, 14] {
                    let lazy = old_lazy_loop(&decoder, &embeddings, stops, max_tokens);
                    let (eager, first, steps) =
                        old_eager_loop(&decoder, &embeddings, stops, max_tokens);
                    assert_eq!(lazy, eager);

                    let rows = ids.len();
                    let (last_hidden, cache) = decoder.prefill(&embeddings, rows);
                    let logits = decoder.logits(&last_hidden);
                    let mut observed = Vec::new();
                    let got = greedy_generate(
                        &decoder,
                        &logits,
                        cache,
                        max_tokens,
                        |next| is_chat_stop(stops, next),
                        "Test",
                        |step_logits, next| {
                            assert_eq!(next, old_argmax(step_logits));
                            observed.push(step_logits.to_vec());
                        },
                    )
                    .unwrap();
                    assert_eq!(got, lazy, "tied {tied} stops {stops:?} max {max_tokens}");
                    // The caller-side gate Mega-ASR and Phonon keep.
                    let got_first = (max_tokens > 0).then(|| logits.clone());
                    assert_eq!(got_first.as_deref().map(bits), first.as_deref().map(bits));
                    assert_eq!(observed.len(), steps.len());
                    for (a, b) in observed.iter().zip(&steps) {
                        assert_eq!(bits(a), bits(b));
                    }
                }
            }
        }
    }

    #[test]
    fn chat_stop_fallback_only_applies_without_tokenizer_ids() {
        assert!(is_chat_stop(&[], 151_645));
        assert!(is_chat_stop(&[], 151_643));
        assert!(!is_chat_stop(&[], 5));
        assert!(!is_chat_stop(&[7], 151_645));
        assert!(is_chat_stop(&[7], 7));
    }

    #[test]
    fn unit_residual_multiplier_leaves_the_qwen3_layer_unchanged() {
        // The retired layer added the branch outputs without a multiply.
        let decoder = random_decoder(3, false);
        let layer = &decoder.layers[0];
        let x: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 - 3.0) * 0.31).collect();
        let mut cache_new = super::AttentionCache::new(2);
        let mut cache_old = super::AttentionCache::new(2);
        let (cos, sin) =
            crate::ops::rope_tables_range(0, 1, layer.attention.rotary_dim, layer.attention.theta);
        let new = layer.step(&x, &mut cache_new, &cos, &sin);

        let mut normalized = x.to_vec();
        layer.input_norm.apply(&mut normalized, 1);
        let attention = layer
            .attention
            .step(&normalized, &mut cache_old, &cos, &sin);
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }
        let mut normalized = residual.clone();
        layer.post_attention_norm.apply(&mut normalized, 1);
        let mlp = layer.mlp.forward(&normalized, 1);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        assert_eq!(bits(&new), bits(&residual));
    }
}

#[cfg(test)]
mod reference_parity {
    //! The `reference_*` functions are the previous prefill (materialized
    //! `repeat_kv` copies) and decode step (rope table rebuilt in every
    //! layer), kept verbatim. The shared-KV-head prefill and the
    //! once-per-token rope tables must reproduce them bitwise.
    use super::{
        AttentionCache, DecodeCache, Decoder, DecoderLayer, Linear, Mlp, RmsNorm, TextAttention,
    };
    use crate::ops;

    struct Rng(u64);

    impl Rng {
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 8) % 20001) as f32 / 10000.0 * scale - scale
                })
                .collect()
        }

        fn linear(&mut self, input: usize, output: usize, bias: bool) -> Linear {
            let weight = self.vec(input * output, 1.2 / (input as f32).sqrt());
            let bias = bias.then(|| self.vec(output, 0.2));
            Linear::new(weight, bias, input, output)
        }

        fn norm(&mut self, width: usize) -> RmsNorm {
            let weight = self.vec(width, 0.3).iter().map(|w| w + 1.0).collect();
            RmsNorm::new(weight, 1e-6)
        }
    }

    fn decoder(
        rng: &mut Rng,
        layers: usize,
        (query_heads, key_value_heads, head_dim, rotary_dim): (usize, usize, usize, usize),
    ) -> Decoder {
        let hidden = 8;
        let intermediate = 12;
        let layers = (0..layers)
            .map(|_| DecoderLayer {
                attention: TextAttention {
                    q_proj: rng.linear(hidden, query_heads * head_dim, true),
                    k_proj: rng.linear(hidden, key_value_heads * head_dim, true),
                    v_proj: rng.linear(hidden, key_value_heads * head_dim, true),
                    o_proj: rng.linear(query_heads * head_dim, hidden, false),
                    q_norm: Some(rng.norm(head_dim)),
                    k_norm: Some(rng.norm(head_dim)),
                    query_heads,
                    key_value_heads,
                    head_dim,
                    rotary_dim,
                    theta: 10_000.0,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                },
                mlp: Mlp {
                    gate: rng.linear(hidden, intermediate, false),
                    up: rng.linear(hidden, intermediate, false),
                    down: rng.linear(intermediate, hidden, false),
                    intermediate,
                },
                input_norm: rng.norm(hidden),
                post_attention_norm: rng.norm(hidden),
                hidden,
                residual_multiplier: 1.0,
            })
            .collect();
        Decoder {
            embeddings: Vec::new(),
            lm_head: None,
            layers,
            final_norm: rng.norm(hidden),
            vocab_size: 0,
            hidden_size: hidden,
        }
    }

    fn reference_attention_forward(
        this: &TextAttention,
        x: &[f32],
        rows: usize,
        cache: Option<&mut AttentionCache>,
    ) -> Vec<f32> {
        let query_width = this.query_heads * this.head_dim;
        let mut query = this.q_proj.forward(x, rows);
        let mut key = this.k_proj.forward(x, rows);
        let value = this.v_proj.forward(x, rows);
        if let Some(norm) = &this.q_norm {
            norm.apply(&mut query, rows * this.query_heads);
        }
        if let Some(norm) = &this.k_norm {
            norm.apply(&mut key, rows * this.key_value_heads);
        }

        let mut query_heads = ops::split_heads(&query, rows, this.query_heads, this.head_dim);
        let mut key_heads = ops::split_heads(&key, rows, this.key_value_heads, this.head_dim);
        let value_heads = ops::split_heads(&value, rows, this.key_value_heads, this.head_dim);
        let (cos, sin) = ops::rope_tables(rows, this.rotary_dim, this.theta);
        ops::rope_neox(
            &mut query_heads,
            this.query_heads,
            rows,
            this.rotary_dim,
            &cos,
            &sin,
        );
        ops::rope_neox(
            &mut key_heads,
            this.key_value_heads,
            rows,
            this.rotary_dim,
            &cos,
            &sin,
        );

        if let Some(cache) = cache {
            for head in 0..this.key_value_heads {
                let start = head * rows * this.head_dim;
                let end = start + rows * this.head_dim;
                cache.keys[head].extend_from_slice(&key_heads[start..end]);
                cache.values[head].extend_from_slice(&value_heads[start..end]);
            }
            cache.len = rows;
        }

        let key_heads = ops::repeat_kv(
            &key_heads,
            this.key_value_heads,
            rows,
            this.head_dim,
            this.query_heads,
        );
        let value_heads = ops::repeat_kv(
            &value_heads,
            this.key_value_heads,
            rows,
            this.head_dim,
            this.query_heads,
        );
        let mut attended_heads = vec![0.0f32; query_width * rows];
        let scale = 1.0 / (this.head_dim as f32).sqrt();
        for head in 0..this.query_heads {
            let head_start = head * rows * this.head_dim;
            let query_head = &query_heads[head_start..head_start + rows * this.head_dim];
            let key_head = &key_heads[head_start..head_start + rows * this.head_dim];
            let value_head = &value_heads[head_start..head_start + rows * this.head_dim];
            for position in 0..rows {
                let query_row =
                    &query_head[position * this.head_dim..(position + 1) * this.head_dim];
                let key_prefix = &key_head[..(position + 1) * this.head_dim];
                let value_prefix = &value_head[..(position + 1) * this.head_dim];
                let output = ops::sdpa(
                    query_row,
                    key_prefix,
                    value_prefix,
                    None,
                    1,
                    position + 1,
                    this.head_dim,
                    this.head_dim,
                    scale,
                );
                let target = position * query_width + head * this.head_dim;
                attended_heads[target..target + this.head_dim].copy_from_slice(&output);
            }
        }
        this.o_proj.forward(&attended_heads, rows)
    }

    fn reference_attention_step(
        this: &TextAttention,
        x: &[f32],
        cache: &mut AttentionCache,
    ) -> Vec<f32> {
        let query_width = this.query_heads * this.head_dim;
        let mut query = this.q_proj.forward(x, 1);
        let mut key = this.k_proj.forward(x, 1);
        let value = this.v_proj.forward(x, 1);
        if let Some(norm) = &this.q_norm {
            norm.apply(&mut query, this.query_heads);
        }
        if let Some(norm) = &this.k_norm {
            norm.apply(&mut key, this.key_value_heads);
        }
        let position = cache.len;
        let (cos, sin) = ops::rope_tables_range(position, 1, this.rotary_dim, this.theta);
        ops::rope_neox(&mut query, this.query_heads, 1, this.rotary_dim, &cos, &sin);
        ops::rope_neox(
            &mut key,
            this.key_value_heads,
            1,
            this.rotary_dim,
            &cos,
            &sin,
        );
        for head in 0..this.key_value_heads {
            let start = head * this.head_dim;
            cache.keys[head].extend_from_slice(&key[start..start + this.head_dim]);
            cache.values[head].extend_from_slice(&value[start..start + this.head_dim]);
        }
        cache.len += 1;
        let mut attended = vec![0.0f32; query_width];
        let scale = 1.0 / (this.head_dim as f32).sqrt();
        for head in 0..this.query_heads {
            let kv_head = head / (this.query_heads / this.key_value_heads);
            let start = head * this.head_dim;
            let output = ops::sdpa(
                &query[start..start + this.head_dim],
                &cache.keys[kv_head],
                &cache.values[kv_head],
                None,
                1,
                cache.len,
                this.head_dim,
                this.head_dim,
                scale,
            );
            attended[start..start + this.head_dim].copy_from_slice(&output);
        }
        this.o_proj.forward(&attended, 1)
    }

    fn reference_layer(
        this: &DecoderLayer,
        x: &[f32],
        rows: usize,
        attention: Vec<f32>,
    ) -> Vec<f32> {
        let mut residual = x.to_vec();
        for (value, add) in residual.iter_mut().zip(attention) {
            *value += add;
        }
        let mut normalized = residual.clone();
        this.post_attention_norm.apply(&mut normalized, rows);
        let mlp = this.mlp.forward(&normalized, rows);
        for (value, add) in residual.iter_mut().zip(mlp) {
            *value += add;
        }
        residual
    }

    fn reference_prefill(decoder: &Decoder, input: &[f32], rows: usize) -> (Vec<f32>, DecodeCache) {
        let mut hidden = input.to_vec();
        let mut cache = DecodeCache {
            layers: decoder
                .layers
                .iter()
                .map(|layer| AttentionCache::new(layer.attention.key_value_heads))
                .collect(),
        };
        for (layer, layer_cache) in decoder.layers.iter().zip(&mut cache.layers) {
            let mut normalized = hidden.clone();
            layer.input_norm.apply(&mut normalized, rows);
            let attention =
                reference_attention_forward(&layer.attention, &normalized, rows, Some(layer_cache));
            hidden = reference_layer(layer, &hidden, rows, attention);
        }
        decoder.final_norm.apply(&mut hidden, rows);
        (hidden[(rows - 1) * decoder.hidden_size..].to_vec(), cache)
    }

    fn reference_forward(decoder: &Decoder, input: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for layer in &decoder.layers {
            let mut normalized = hidden.clone();
            layer.input_norm.apply(&mut normalized, rows);
            let attention = reference_attention_forward(&layer.attention, &normalized, rows, None);
            hidden = reference_layer(layer, &hidden, rows, attention);
        }
        decoder.final_norm.apply(&mut hidden, rows);
        hidden
    }

    fn reference_step(decoder: &Decoder, input: &[f32], cache: &mut DecodeCache) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for (layer, layer_cache) in decoder.layers.iter().zip(&mut cache.layers) {
            let mut normalized = hidden.clone();
            layer.input_norm.apply(&mut normalized, 1);
            let attention = reference_attention_step(&layer.attention, &normalized, layer_cache);
            hidden = reference_layer(layer, &hidden, 1, attention);
        }
        decoder.final_norm.apply(&mut hidden, 1);
        hidden
    }

    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: got {g} want {w}");
        }
    }

    fn assert_caches_equal(got: &DecodeCache, want: &DecodeCache) {
        for (g, w) in got.layers.iter().zip(&want.layers) {
            assert_eq!(g.len, w.len);
            for (gk, wk) in g.keys.iter().zip(&w.keys) {
                assert_bits("keys", gk, wk);
            }
            for (gv, wv) in g.values.iter().zip(&w.values) {
                assert_bits("values", gv, wv);
            }
        }
    }

    #[test]
    fn shared_kv_head_prefill_and_hoisted_rope_match_the_originals_bitwise() {
        // (query heads, kv heads, head_dim, rotary_dim): grouped, multi-query,
        // plain MHA, and partial rotary.
        for (seed, shape) in [
            (1u64, (4, 2, 2, 2)),
            (2, (4, 1, 4, 4)),
            (3, (2, 2, 4, 2)),
            (4, (6, 2, 4, 4)),
        ] {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let decoder = decoder(&mut rng, 3, shape);
            let hidden = decoder.hidden_size;
            let rows = 7;
            let input = rng.vec(rows * hidden, 1.0);

            assert_bits(
                "full forward",
                &decoder.forward(&input, rows),
                &reference_forward(&decoder, &input, rows),
            );
            let (got_last, mut got_cache) = decoder.prefill(&input, rows);
            let (want_last, mut want_cache) = reference_prefill(&decoder, &input, rows);
            assert_bits("prefill hidden", &got_last, &want_last);
            assert_caches_equal(&got_cache, &want_cache);

            for step in 0..6 {
                let token = rng.vec(hidden, 1.0);
                let got = decoder.step(&token, &mut got_cache);
                let want = reference_step(&decoder, &token, &mut want_cache);
                assert_bits(&format!("seed {seed} step {step}"), &got, &want);
                assert_caches_equal(&got_cache, &want_cache);
            }
        }
    }
}
