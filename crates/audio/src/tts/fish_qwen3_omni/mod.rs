//! Fish Qwen3 Omni TTS family (fish_speech DualAR).
//!
//! Port of `mlx_audio/tts/models/fish_qwen3_omni/` (fish_speech.py,
//! config.py, prompt.py, tokenizer.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/fish_qwen3_omni):
//! a DualAR transformer whose slow stack predicts one semantic token per
//! speech step and whose fast stack fills the remaining codebooks from
//! the projected slow hidden state, over the Fish S1 DAC codec. See
//! [README.md](README.md) for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-sequence path
//! (`_generate_codes_for_batch`): batch size 1, non-streaming. The
//! batched `batch_generate` path (padded prompts and the attention-mask
//! arithmetic) is out of scope. Text tokenization is a caller concern
//! (the real checkpoints ship a Hugging Face tokenizer the portable
//! crate does not embed): the caller tokenizes every text piece,
//! including the `<|im_start|>`/`<|im_end|>` header strings, and hands
//! the ids to [`PromptMessage`].

use serde_json::Value;

use crate::codec::fish_s1_dac::FishS1Dac;
use crate::lm;
use crate::lm::cache::KvCache;
use crate::music::minimax_music3::rng;
use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// RAS window: repeats inside the last 10 semantic tokens trigger the
/// high-temperature resample (`RAS_WIN_SIZE`).
pub const RAS_WIN_SIZE: usize = 10;
/// RAS resample temperature (`RAS_HIGH_TEMP`).
pub const RAS_HIGH_TEMP: f32 = 1.0;
/// RAS resample top-p (`RAS_HIGH_TOP_P`).
pub const RAS_HIGH_TOP_P: f32 = 0.9;

/// One side of the DualAR transformer (text or audio decoder).
#[derive(Debug, Clone)]
pub struct StackConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub dim: usize,
    pub intermediate_size: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub attention_qkv_bias: bool,
    pub attention_o_bias: bool,
    pub attention_qk_norm: bool,
}

impl StackConfig {
    fn from_json(value: &Value, model_type: &str) -> Result<Self> {
        let u = |field: &str| -> Result<usize> {
            value
                .get(field)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: field.to_string(),
                    why: format!("{model_type} config is missing {field}"),
                })
        };
        let f = |field: &str| -> Result<f32> {
            value
                .get(field)
                .and_then(Value::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: field.to_string(),
                    why: format!("{model_type} config is missing {field}"),
                })
        };
        let b = |field: &str, default: bool| -> bool {
            value.get(field).and_then(Value::as_bool).unwrap_or(default)
        };
        Ok(Self {
            vocab_size: u("vocab_size")?,
            n_layer: u("n_layer")?,
            n_head: u("n_head")?,
            dim: u("dim")?,
            intermediate_size: u("intermediate_size")?,
            n_local_heads: u("n_local_heads")?,
            head_dim: u("head_dim")?,
            rope_base: f("rope_base")?,
            norm_eps: f("norm_eps")?,
            max_seq_len: u("max_seq_len")?,
            attention_qkv_bias: b("attention_qkv_bias", false),
            attention_o_bias: b("attention_o_bias", false),
            attention_qk_norm: b("attention_qk_norm", false),
        })
    }
}

/// The family `ModelConfig`: the top-level token boundaries plus the two
/// stack configs (`text_config`, `audio_decoder_config`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub pad_token_id: i32,
    pub eos_token_id: i32,
    pub audio_pad_token_id: i32,
    pub semantic_start_token_id: i32,
    pub semantic_end_token_id: i32,
    pub sample_rate: u32,
    pub text_config: StackConfig,
    pub audio_decoder_config: StackConfig,
    /// `audio_decoder_config.num_codebooks`: the books the fast stack
    /// fills per semantic step and the codec's book count.
    pub num_codebooks: usize,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let i = |field: &str, default: i64| -> i32 {
            value.get(field).and_then(Value::as_i64).unwrap_or(default) as i32
        };
        let text_value = value.get("text_config").cloned().unwrap_or(Value::Null);
        let audio_value = value
            .get("audio_decoder_config")
            .cloned()
            .unwrap_or(Value::Null);
        let audio_decoder_config = StackConfig::from_json(&audio_value, "audio_decoder")?;
        let num_codebooks = audio_value
            .get("num_codebooks")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "num_codebooks".to_string(),
                why: "audio_decoder_config is missing num_codebooks".to_string(),
            })?;
        // Decoding loops over `num_codebooks - 1` residual books, so zero
        // would underflow there. Refuse it at load.
        if num_codebooks == 0 {
            return Err(SpeechError::BadConfig {
                field: "num_codebooks".to_string(),
                why: "must be positive".to_string(),
            });
        }
        Ok(Self {
            pad_token_id: i("pad_token_id", 151669),
            eos_token_id: i("eos_token_id", 151645),
            audio_pad_token_id: i("audio_pad_token_id", 151677),
            semantic_start_token_id: i("semantic_start_token_id", 151678),
            semantic_end_token_id: i("semantic_end_token_id", 155773),
            sample_rate: i("sample_rate", 44100) as u32,
            text_config: StackConfig::from_json(&text_value, "text")?,
            audio_decoder_config,
            num_codebooks,
        })
    }
}

/// Rounds to bf16 precision (round-to-nearest-even) and back. The
/// reference `FishRotaryEmbedding` caches its cos/sin phases in bf16
/// ("match the reference implementation"), so the f32 port applies the
/// same rounding to the tables before rotating.
fn round_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits(rounded & 0xFFFF_0000)
}

/// The reference `FishRotaryEmbedding`: interleaved pairs rotated with
/// bf16-rounded phase tables `cos/sin(position * freq)`,
/// `freq = base^(-2 d / head_dim)`.
#[derive(Debug, Clone)]
struct FishRotary {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
}

impl FishRotary {
    fn new(head_dim: usize, rope_base: f32, max_position_embeddings: usize) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_position_embeddings * half];
        let mut sin = vec![0.0f32; max_position_embeddings * half];
        for position in 0..max_position_embeddings {
            for (pair, cos_out) in cos[position * half..(position + 1) * half]
                .iter_mut()
                .enumerate()
            {
                let freq = (rope_base).powf(-2.0 * pair as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                *cos_out = round_bf16(angle.cos());
                sin[position * half + pair] = round_bf16(angle.sin());
            }
        }
        Self { cos, sin, head_dim }
    }

    /// Rotates `x` head-major `[heads, rows, head_dim]` in place at
    /// table row `offset + position`. Interleaved pairs: even elements
    /// are the real part, odd elements the imaginary part.
    fn apply(&self, x: &mut [f32], heads: usize, rows: usize, offset: usize) {
        let half = self.head_dim / 2;
        for head in 0..heads {
            for row in 0..rows {
                let position = offset + row;
                let base = (head * rows + row) * self.head_dim;
                for pair in 0..half {
                    let cos_v = self.cos[position * half + pair];
                    let sin_v = self.sin[position * half + pair];
                    let even = x[base + 2 * pair];
                    let odd = x[base + 2 * pair + 1];
                    x[base + 2 * pair] = even * cos_v - odd * sin_v;
                    x[base + 2 * pair + 1] = odd * cos_v + even * sin_v;
                }
            }
        }
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rope: FishRotary,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    /// `x [rows, dim]` with a per-layer cache; returns
    /// `[rows, n_heads * head_dim]` before the output projection.
    fn forward(&self, x: &[f32], rows: usize, cache: Option<&mut KvCache>) -> Result<Vec<f32>> {
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;
        let projected = self.wqkv.forward(x, rows);
        let mut q_pm = vec![0.0f32; rows * q_size];
        let mut k_pm = vec![0.0f32; rows * kv_size];
        let mut v_pm = vec![0.0f32; rows * kv_size];
        for row in 0..rows {
            let src = &projected[row * (q_size + 2 * kv_size)..];
            q_pm[row * q_size..(row + 1) * q_size].copy_from_slice(&src[..q_size]);
            k_pm[row * kv_size..(row + 1) * kv_size]
                .copy_from_slice(&src[q_size..q_size + kv_size]);
            v_pm[row * kv_size..(row + 1) * kv_size]
                .copy_from_slice(&src[q_size + kv_size..q_size + 2 * kv_size]);
        }
        // The rope, cache, and attention math below are head-major
        // [heads, positions, head_dim].
        let mut qh = lm::engine::head_major(&q_pm, rows, self.n_heads, self.head_dim);
        let mut kh = lm::engine::head_major(&k_pm, rows, self.n_kv_heads, self.head_dim);
        let vh = lm::engine::head_major(&v_pm, rows, self.n_kv_heads, self.head_dim);
        if let Some(q_norm) = &self.q_norm {
            q_norm.apply(&mut qh, rows * self.n_heads);
        }
        if let Some(k_norm) = &self.k_norm {
            k_norm.apply(&mut kh, rows * self.n_kv_heads);
        }
        let offset = cache.as_ref().map_or(0, |c| c.offset);
        self.rope.apply(&mut qh, self.n_heads, rows, offset);
        self.rope.apply(&mut kh, self.n_kv_heads, rows, offset);
        let (keys, values) = match cache {
            Some(cache) => {
                cache.update_and_fetch(&kh, &vh, self.n_kv_heads, rows, self.head_dim);
                (&cache.keys, &cache.values)
            }
            None => (&kh, &vh),
        };
        let total = keys.len() / (self.n_kv_heads * self.head_dim);
        // Causal attention over the cache; multi-row calls mask the
        // future, single-row calls see everything.
        let rep = self.n_heads / self.n_kv_heads;
        let mut out = vec![0.0f32; rows * q_size];
        let mut scores = vec![0.0f32; total];
        for head in 0..self.n_heads {
            let kv = head / rep;
            let q_head = &qh[head * rows * self.head_dim..(head + 1) * rows * self.head_dim];
            let k_head = &keys[kv * total * self.head_dim..(kv + 1) * total * self.head_dim];
            let v_head = &values[kv * total * self.head_dim..(kv + 1) * total * self.head_dim];
            for row in 0..rows {
                let high = (offset + row).min(total - 1);
                let q_row = &q_head[row * self.head_dim..(row + 1) * self.head_dim];
                let mut max = f32::NEG_INFINITY;
                for j in 0..=high {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_row[d] * k_head[j * self.head_dim + d];
                    }
                    let score = dot * self.scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores[..=high] {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_row = &mut out
                    [(head * rows + row) * self.head_dim..(head * rows + row + 1) * self.head_dim];
                for j in 0..=high {
                    let weight = scores[j] * inv;
                    for d in 0..self.head_dim {
                        out_row[d] += weight * v_head[j * self.head_dim + d];
                    }
                }
            }
        }
        // Back to [rows, heads * dim].
        let mut joined = vec![0.0f32; rows * q_size];
        for row in 0..rows {
            for head in 0..self.n_heads {
                let src = (head * rows + row) * self.head_dim;
                let dst = row * q_size + head * self.head_dim;
                joined[dst..dst + self.head_dim].copy_from_slice(&out[src..src + self.head_dim]);
            }
        }
        Ok(self.wo.forward(&joined, rows))
    }
}

struct TransformerBlock {
    attention: Attention,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl TransformerBlock {
    fn forward(&self, x: &mut [f32], rows: usize, cache: Option<&mut KvCache>) -> Result<()> {
        let mut attended = x.to_vec();
        self.attention_norm.apply(&mut attended, rows);
        let attn = self.attention.forward(&attended, rows, cache)?;
        for (value, add) in x.iter_mut().zip(attn) {
            *value += add;
        }
        let mut fed = x.to_vec();
        self.ffn_norm.apply(&mut fed, rows);
        // SwiGLU: w2(silu(w1(x)) * w3(x)).
        let gate = self.w1.forward(&fed, rows);
        let up = self.w3.forward(&fed, rows);
        let fused: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
            .collect();
        let down = self.w2.forward(&fused, rows);
        for (value, add) in x.iter_mut().zip(down) {
            *value += add;
        }
        Ok(())
    }
}

/// The DualAR transformer: the slow text stack over
/// `[semantic + codebook sum]` embeddings and the fast audio stack over
/// the projected hidden state plus codebook embeddings.
pub struct DualARTransformer {
    pub config: ModelConfig,
    embeddings: Vec<f32>,
    codebook_embeddings: Vec<f32>,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    fast_project_in: Option<Linear>,
    fast_embeddings: Vec<f32>,
    fast_layers: Vec<TransformerBlock>,
    fast_norm: RmsNorm,
    fast_output: Linear,
}

impl DualARTransformer {
    /// Loads the sanitized checkpoint (all tensors under one prefix,
    /// `model.` for full-model saves; the `load_weights` remap already
    /// stripped the wrapper).
    pub fn load(weights: &lm::Weights, config: ModelConfig) -> Result<Self> {
        let tc = &config.text_config;
        let ac = &config.audio_decoder_config;
        let embeddings = weights.embedding("embeddings", tc.vocab_size, tc.dim)?;
        let codebook_embeddings = weights.embedding(
            "codebook_embeddings",
            ac.vocab_size * config.num_codebooks,
            tc.dim,
        )?;
        let mut layers = Vec::with_capacity(tc.n_layer);
        for index in 0..tc.n_layer {
            layers.push(Self::load_block(
                weights,
                &format!("layers.{index}."),
                tc,
                tc.max_seq_len,
            )?);
        }
        let norm = RmsNorm::new(weights.tensor("norm", &[tc.dim])?, tc.norm_eps);
        let fast_project_in = if tc.dim != ac.dim {
            Some(weights.linear("fast_project_in", tc.dim, ac.dim, lm::Bias::None)?)
        } else {
            None
        };
        let fast_embeddings = weights.embedding("fast_embeddings", ac.vocab_size, ac.dim)?;
        let mut fast_layers = Vec::with_capacity(ac.n_layer);
        for index in 0..ac.n_layer {
            fast_layers.push(Self::load_block(
                weights,
                &format!("fast_layers.{index}."),
                ac,
                config.num_codebooks,
            )?);
        }
        let fast_norm = RmsNorm::new(weights.tensor("fast_norm", &[ac.dim])?, ac.norm_eps);
        let fast_output = weights.linear("fast_output", ac.dim, ac.vocab_size, lm::Bias::None)?;
        Ok(Self {
            config,
            embeddings,
            codebook_embeddings,
            layers,
            norm,
            fast_project_in,
            fast_embeddings,
            fast_layers,
            fast_norm,
            fast_output,
        })
    }

    fn load_block(
        weights: &lm::Weights,
        prefix: &str,
        stack: &StackConfig,
        rope_positions: usize,
    ) -> Result<TransformerBlock> {
        let dim = stack.dim;
        let q_size = stack.n_head * stack.head_dim;
        let kv_size = stack.n_local_heads * stack.head_dim;
        let qkv_bias = if stack.attention_qkv_bias {
            lm::Bias::Required
        } else {
            lm::Bias::None
        };
        let o_bias = if stack.attention_o_bias {
            lm::Bias::Required
        } else {
            lm::Bias::None
        };
        let wqkv = weights.linear(
            &format!("{prefix}attention.wqkv"),
            dim,
            q_size + 2 * kv_size,
            qkv_bias,
        )?;
        let wo = weights.linear(&format!("{prefix}attention.wo"), q_size, dim, o_bias)?;
        let (q_norm, k_norm) = if stack.attention_qk_norm {
            (
                Some(RmsNorm::new(
                    weights.tensor(&format!("{prefix}attention.q_norm"), &[stack.head_dim])?,
                    stack.norm_eps,
                )),
                Some(RmsNorm::new(
                    weights.tensor(&format!("{prefix}attention.k_norm"), &[stack.head_dim])?,
                    stack.norm_eps,
                )),
            )
        } else {
            (None, None)
        };
        let attention = Attention {
            wqkv,
            wo,
            q_norm,
            k_norm,
            rope: FishRotary::new(stack.head_dim, stack.rope_base, rope_positions),
            n_heads: stack.n_head,
            n_kv_heads: stack.n_local_heads,
            head_dim: stack.head_dim,
            scale: (stack.head_dim as f64).powf(-0.5) as f32,
        };
        let w1 = weights.linear(
            &format!("{prefix}feed_forward.w1"),
            dim,
            stack.intermediate_size,
            lm::Bias::None,
        )?;
        let w2 = weights.linear(
            &format!("{prefix}feed_forward.w2"),
            stack.intermediate_size,
            dim,
            lm::Bias::None,
        )?;
        let w3 = weights.linear(
            &format!("{prefix}feed_forward.w3"),
            dim,
            stack.intermediate_size,
            lm::Bias::None,
        )?;
        Ok(TransformerBlock {
            attention,
            w1,
            w2,
            w3,
            attention_norm: RmsNorm::new(
                weights.tensor(&format!("{prefix}attention_norm"), &[dim])?,
                stack.norm_eps,
            ),
            ffn_norm: RmsNorm::new(
                weights.tensor(&format!("{prefix}ffn_norm"), &[dim])?,
                stack.norm_eps,
            ),
        })
    }

    /// The reference `_embed` over the book-major input planes
    /// `[num_codebooks + 1][len]`: plane 0 is the semantic token ids,
    /// planes 1.. the codebook codes. Semantic positions mix in the
    /// summed codebook embeddings and divide by
    /// `sqrt(num_codebooks + 1)`.
    fn embed(&self, inp: &[i32], rows: usize) -> Result<Vec<f32>> {
        let config = &self.config;
        let tc = &config.text_config;
        let ac_vocab = config.audio_decoder_config.vocab_size;
        let books = config.num_codebooks;
        if inp.len() != (books + 1) * rows {
            return Err(SpeechError::Input {
                why: format!(
                    "input length {} does not match {} planes of {rows}",
                    inp.len(),
                    books + 1
                ),
            });
        }
        let dim = tc.dim;
        let mut x = vec![0.0f32; rows * dim];
        let mut vq_sum = vec![0.0f32; rows * dim];
        for position in 0..rows {
            let semantic = inp[position];
            let semantic_index = usize::try_from(semantic).map_err(|_| SpeechError::Input {
                why: format!("negative semantic token {semantic}"),
            })?;
            if semantic_index >= tc.vocab_size {
                return Err(SpeechError::Input {
                    why: format!("semantic token {semantic} outside vocab {}", tc.vocab_size),
                });
            }
            let dst = &mut x[position * dim..(position + 1) * dim];
            dst.copy_from_slice(&self.embeddings[semantic_index * dim..(semantic_index + 1) * dim]);
            let semantic_mask = semantic >= config.semantic_start_token_id
                && semantic <= config.semantic_end_token_id;
            if semantic_mask {
                for book in 0..books {
                    let code = inp[(1 + book) * rows + position];
                    let code_index = usize::try_from(code).map_err(|_| SpeechError::Input {
                        why: format!("negative codebook code {code}"),
                    })?;
                    if code_index >= ac_vocab {
                        return Err(SpeechError::Input {
                            why: format!("codebook code {code} outside vocab {ac_vocab}"),
                        });
                    }
                    let table_row = code_index + book * ac_vocab;
                    let src = &self.codebook_embeddings[table_row * dim..(table_row + 1) * dim];
                    for (value, add) in vq_sum[position * dim..(position + 1) * dim]
                        .iter_mut()
                        .zip(src)
                    {
                        *value += add;
                    }
                }
            }
        }
        // Semantic rows carry the summed codebook embeddings and the
        // whole mix divides by sqrt(num_codebooks + 1); non-semantic
        // rows keep the plain embedding (their vq_sum stays zero).
        let scale = (books as f32 + 1.0).sqrt();
        for position in 0..rows {
            let semantic = inp[position];
            let semantic_mask = semantic >= config.semantic_start_token_id
                && semantic <= config.semantic_end_token_id;
            let row_x = &mut x[position * dim..(position + 1) * dim];
            let row_vq = &vq_sum[position * dim..(position + 1) * dim];
            for (value, add) in row_x.iter_mut().zip(row_vq) {
                *value += add;
            }
            if semantic_mask {
                for value in row_x.iter_mut() {
                    *value /= scale;
                }
            }
        }
        Ok(x)
    }

    /// The slow forward: `inp [rows, num_codebooks + 1]` through the
    /// layers with causal masks and per-layer caches, then the final
    /// norm. Returns `(logits [rows, vocab], hidden [rows, ac.dim])`
    /// where the hidden states are the fast projector input.
    pub fn forward(
        &mut self,
        inp: &[i32],
        rows: usize,
        caches: Option<&mut [KvCache]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let tc = &self.config.text_config;
        let mut caches = match caches {
            Some(caches) => {
                if caches.len() != self.layers.len() {
                    return Err(SpeechError::Input {
                        why: format!(
                            "{} layer caches provided for {} layers",
                            caches.len(),
                            self.layers.len()
                        ),
                    });
                }
                Some(caches)
            }
            None => None,
        };
        let _ = &mut caches;
        let mut x = self.embed(inp, rows)?;
        for (index, block) in self.layers.iter_mut().enumerate() {
            let cache = caches.as_deref_mut().map(|c| &mut c[index]);
            block.forward(&mut x, rows, cache)?;
        }
        self.norm.apply(&mut x, rows);
        let logits = ops::linear(&x, &self.embeddings, None, rows, tc.dim, tc.vocab_size);
        let hidden = match &self.fast_project_in {
            Some(project) => project.forward(&x, rows),
            None => x,
        };
        Ok((logits, hidden))
    }

    /// The reference `fast_forward_cached` over one position: appends
    /// `x [ac.dim]` to the fast caches and returns the fast head logits.
    pub fn fast_forward_cached(&mut self, x: &[f32], caches: &mut [KvCache]) -> Result<Vec<f32>> {
        if caches.len() != self.fast_layers.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} fast layer caches provided for {} layers",
                    caches.len(),
                    self.fast_layers.len()
                ),
            });
        }
        let mut h = x.to_vec();
        for (index, block) in self.fast_layers.iter_mut().enumerate() {
            block.forward(&mut h, 1, Some(&mut caches[index]))?;
        }
        self.fast_norm.apply(&mut h, 1);
        Ok(self.fast_output.forward(&h, 1))
    }

    /// `make_cache` / `make_fast_cache`: one empty cache per layer.
    pub fn make_cache(&self) -> Vec<KvCache> {
        vec![KvCache::new(); self.layers.len()]
    }

    pub fn make_fast_cache(&self) -> Vec<KvCache> {
        vec![KvCache::new(); self.fast_layers.len()]
    }
}

/// One prompt part: caller-tokenized text ids or VQ codes
/// `[num_codebooks][frames]` (row 0 is the semantic book).
#[derive(Debug, Clone)]
pub enum PromptPart {
    Text(Vec<i32>),
    Vq(Vec<Vec<i32>>),
}

/// One conversation turn: the caller-tokenized
/// `<|im_start|>{role}\n{modality}` header ids, the parts, and the
/// caller-tokenized `<|im_end|>\n` ids when the turn closes.
#[derive(Debug, Clone)]
pub struct PromptMessage {
    pub header_ids: Vec<i32>,
    pub parts: Vec<PromptPart>,
    pub im_end_ids: Vec<i32>,
}

/// The encoded conversation: the text-token row plus one codebook row
/// per book, all the same length (zeros where no VQ codes ride).
#[derive(Debug, Clone)]
pub struct PromptMatrix {
    pub tokens: Vec<i32>,
    pub codebooks: Vec<Vec<i32>>,
}

impl PromptMatrix {
    /// Flattens to the model input layout: book-major
    /// `[num_codebooks + 1][len]` planes, the semantic ids first and
    /// then the codebooks, exactly the reference `[B, C + 1, T]` tensor.
    pub fn to_model_input(&self, num_codebooks: usize) -> Vec<i32> {
        let len = self.tokens.len();
        let mut inp = Vec::with_capacity(len * (num_codebooks + 1));
        inp.extend_from_slice(&self.tokens);
        for book in self.codebooks.iter().take(num_codebooks) {
            inp.extend_from_slice(book);
        }
        inp
    }
}

/// The reference `Conversation.encode_for_inference`: header ids, part
/// ids, and im_end ids concatenate into the text row; VQ parts ride the
/// codebook rows at the same positions, with the semantic book offset
/// by `semantic_begin_id` inside the text row.
pub fn encode_for_inference(
    messages: &[PromptMessage],
    num_codebooks: usize,
    semantic_begin_id: i32,
) -> Result<PromptMatrix> {
    /// One encoded segment: the text ids plus the VQ codes riding the
    /// codebook rows when the segment carries audio.
    type Segment = (Vec<i32>, Option<Vec<Vec<i32>>>);
    let mut segments: Vec<Segment> = Vec::new();
    for message in messages {
        if !message.header_ids.is_empty() {
            segments.push((message.header_ids.clone(), None));
        }
        for part in &message.parts {
            match part {
                PromptPart::Text(ids) => segments.push((ids.clone(), None)),
                PromptPart::Vq(codes) => {
                    if codes.len() != num_codebooks {
                        return Err(SpeechError::Input {
                            why: format!(
                                "VQ part carries {} books, expected {num_codebooks}",
                                codes.len()
                            ),
                        });
                    }
                    let frames = codes.first().map_or(0, |book| book.len());
                    if codes.iter().any(|book| book.len() != frames) {
                        return Err(SpeechError::Input {
                            why: "VQ part books have ragged lengths".to_string(),
                        });
                    }
                    let semantic: Vec<i32> = codes[0]
                        .iter()
                        .map(|&code| code + semantic_begin_id)
                        .collect();
                    segments.push((semantic, Some(codes.clone())));
                }
            }
        }
        if !message.im_end_ids.is_empty() {
            segments.push((message.im_end_ids.clone(), None));
        }
    }
    if segments.is_empty() {
        return Err(SpeechError::Input {
            why: "conversation produced an empty prompt".to_string(),
        });
    }
    let len: usize = segments.iter().map(|(ids, _)| ids.len()).sum();
    let mut tokens = Vec::with_capacity(len);
    let mut codebooks = vec![vec![0i32; len]; num_codebooks];
    let mut cursor = 0;
    for (ids, vq) in &segments {
        tokens.extend_from_slice(ids);
        if let Some(vq) = vq {
            for (book, row) in vq.iter().enumerate() {
                codebooks[book][cursor..cursor + row.len()].copy_from_slice(row);
            }
        }
        cursor += ids.len();
    }
    Ok(PromptMatrix { tokens, codebooks })
}

/// The reference `_sample_logits` for one row. Temperature 0 (or less)
/// takes the argmax exactly like the reference. Otherwise: top-p plus
/// top-k masking on the sorted cumulative softmax, then the Gumbel-style
/// draw `argmax(probs / -log(uniform))`. `uniform` feeds ONE vector of
/// `logits.len()` draws per sampled call (the reference draws
/// `mx.random.uniform(shape=probs.shape, low=1e-6, high=1.0)` upfront),
/// indexed by vocabulary entry.
pub fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: i32,
    uniform: &mut dyn FnMut(usize) -> Vec<f32>,
) -> Result<usize> {
    if temperature <= 0.0 {
        return Ok(sampling::argmax(logits));
    }
    let vocab_size = logits.len();
    let top_k = if top_k <= 0 || top_k as usize > vocab_size {
        vocab_size
    } else {
        top_k as usize
    };
    let mut order: Vec<usize> = (0..vocab_size).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let mut cumulative = 0.0f32;
    let mut remove = vec![true; vocab_size];
    for (rank, &index) in order.iter().enumerate() {
        cumulative += (logits[index] - max).exp();
        remove[index] = (cumulative / total > top_p) || (rank >= top_k);
    }
    if let Some(&first) = order.first() {
        remove[first] = false;
    }
    // Filtered softmax at the temperature, then the noise draw.
    let mut filtered = logits.to_vec();
    for (value, &masked) in filtered.iter_mut().zip(&remove) {
        if masked {
            *value = f32::NEG_INFINITY;
        }
    }
    let logprobs = sampling::log_softmax(&filtered);
    let draws = uniform(vocab_size);
    // A short draw vector used to fall back to argmax silently, which
    // hides a caller bug and changes the sampled distribution.
    if draws.len() != vocab_size {
        return Err(SpeechError::Input {
            why: format!(
                "uniform draw width {} does not match vocabulary {vocab_size}",
                draws.len()
            ),
        });
    }
    let scale = 1.0 / temperature.max(1e-5);
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, &logprob) in logprobs.iter().enumerate() {
        if !logprob.is_finite() {
            continue;
        }
        // -log(u) with u in [1e-6, 1); the clamp keeps ln finite for a
        // degenerate 0 draw.
        let noise = -draws[index].max(1e-6).ln();
        let score = (logprob * scale).exp() / noise;
        if score > best_score {
            best = index;
            best_score = score;
        }
    }
    Ok(best)
}

/// Sampling and generation options for one batch, with the reference
/// `generate` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// `<= 0` (or past the vocabulary) samples the whole vocabulary,
    /// matching the reference guard.
    pub top_k: i32,
    /// 1.0 leaves the waveform untouched; other values resample it
    /// linearly (`_adjust_speed`).
    pub speed: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.7,
            top_k: 30,
            speed: 1.0,
        }
    }
}

/// One generated batch: the `[num_codebooks][steps]` codes and the
/// decoded mono waveform (empty for a codes-only run).
#[derive(Debug, Clone)]
pub struct Generated {
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    pub model: DualARTransformer,
    /// The additive semantic mask from `post_load_hook`: `-1e9` outside
    /// the semantic range and the im_end token, 0 inside. Built by
    /// [`Model::build_semantic_logit_bias`]; generate refuses to run
    /// without it.
    pub semantic_logit_bias: Option<Vec<f32>>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let mut weights = lm::Weights::open(dir)?;
        if weights.contains("embeddings") {
            // Backbone-only save without the wrapper prefix.
        } else {
            weights = weights.with_prefix("model.");
        }
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let model = DualARTransformer::load(weights, config.clone())?;
        Ok(Self {
            config,
            model,
            semantic_logit_bias: None,
        })
    }

    /// The `post_load_hook` bias: a `-1e9` vector of
    /// `max(tokenizer_vocab_size, text vocab)` entries, zeroed over the
    /// semantic range and the im_end id.
    pub fn build_semantic_logit_bias(
        &self,
        tokenizer_vocab_size: usize,
        im_end_id: i32,
    ) -> Vec<f32> {
        let vocab_size = tokenizer_vocab_size.max(self.config.text_config.vocab_size);
        let mut bias = vec![-1e9f32; vocab_size];
        let start = self.config.semantic_start_token_id.max(0) as usize;
        let end = (self.config.semantic_end_token_id.max(0) as usize).min(vocab_size - 1);
        if start < vocab_size {
            for value in &mut bias[start..=end] {
                *value = 0.0;
            }
        }
        if let Ok(index) = usize::try_from(im_end_id) {
            if index < vocab_size {
                bias[index] = 0.0;
            }
        }
        bias
    }

    /// Runs the full reference `generate` over the caller-split text
    /// batches: per batch, appends the user turn (header + text ids +
    /// im_end), generates the code matrix, decodes the waveform, and
    /// appends the assistant voice turn with the codes so the next batch
    /// sees the reply. `uniforms` feeds the sampling noise in reference
    /// consumption order; temperature 0 with no RAS trigger draws
    /// nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        codec: &FishS1Dac,
        system: &[PromptMessage],
        batches: &[Vec<i32>],
        batch_token_counts: &[usize],
        special: &SpecialIds,
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Vec<Generated>> {
        if self.semantic_logit_bias.is_none() {
            return Err(SpeechError::Input {
                why: "semantic logit bias is not initialized".to_string(),
            });
        }
        let mut conversation = system.to_vec();
        let mut results = Vec::with_capacity(batches.len());
        let mut keys = rng::KeySequence::new(seed);
        for (batch_ids, batch_count) in batches.iter().zip(batch_token_counts) {
            conversation.push(PromptMessage {
                header_ids: special.user_header_ids.clone(),
                parts: vec![PromptPart::Text(batch_ids.clone())],
                im_end_ids: special.im_end_ids.clone(),
            });
            let codes = self.generate_codes_for_batch(
                &conversation,
                special,
                *batch_count,
                params,
                &mut |width| rng::uniform01(keys.next(), width),
            )?;
            let samples = decode_codes(codec, &codes, params.speed)?;
            conversation.push(PromptMessage {
                header_ids: special.assistant_voice_header_ids.clone(),
                parts: vec![PromptPart::Vq(codes.clone())],
                im_end_ids: special.im_end_ids.clone(),
            });
            results.push(Generated { codes, samples });
        }
        Ok(results)
    }

    /// The reference `_generate_codes_for_batch`: prefill, the semantic
    /// loop with the RAS resample, the fast codebook fill, and the im_end
    /// / budget stops. Returns `[num_codebooks][steps]`. `uniforms`
    /// draws one full-vocabulary array per sampled call in reference
    /// consumption order (greedy calls draw nothing); the crate RNG
    /// stream feeds [`Model::generate`], and the fixture tests replay
    /// the recorded arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_codes_for_batch(
        &mut self,
        conversation: &[PromptMessage],
        special: &SpecialIds,
        text_token_count: usize,
        params: &GenerateParams,
        uniforms: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Result<Vec<Vec<i32>>> {
        let bias = self
            .semantic_logit_bias
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: "semantic logit bias is not initialized".to_string(),
            })?
            .clone();
        let im_end_id = special.im_end_token_id;
        let books = self.config.num_codebooks;
        let ac = &self.config.audio_decoder_config;

        let mut prompt_conversation = conversation.to_vec();
        prompt_conversation.push(PromptMessage {
            header_ids: special.assistant_voice_header_ids.clone(),
            parts: Vec::new(),
            im_end_ids: Vec::new(),
        });
        let prompt = encode_for_inference(
            &prompt_conversation,
            books,
            self.config.semantic_start_token_id,
        )?;
        let len = prompt.tokens.len();
        let max_seq_len = self.config.text_config.max_seq_len;
        if len > max_seq_len {
            return Err(SpeechError::Input {
                why: format!("prompt length {len} exceeds max_seq_len {max_seq_len}"),
            });
        }
        let semantic_token_budget = params
            .max_tokens
            .min(32usize.max(text_token_count.saturating_mul(12)));
        check_decode_fits(len, semantic_token_budget, max_seq_len)?;
        let mut caches = self.model.make_cache();
        let prompt_input = prompt.to_model_input(books);
        let (prompt_logits, prompt_hidden) =
            self.model.forward(&prompt_input, len, Some(&mut caches))?;
        let mut logits = last_row(&prompt_logits, self.config.text_config.vocab_size);
        let mut hidden = last_row(&prompt_hidden, ac.dim);

        let mut previous_semantic_tokens: Vec<i32> = Vec::new();
        let mut generated_steps: Vec<Vec<i32>> = Vec::new();
        for _step in 0..semantic_token_budget {
            let semantic_token =
                self.sample_semantic(&logits, &bias, &previous_semantic_tokens, params, uniforms)?;
            if semantic_token == im_end_id {
                break;
            }

            previous_semantic_tokens.push(semantic_token);
            let keep = previous_semantic_tokens.len().saturating_sub(RAS_WIN_SIZE);
            previous_semantic_tokens.drain(..keep);

            let semantic_code = (semantic_token - self.config.semantic_start_token_id)
                .clamp(0, ac.vocab_size as i32 - 1);
            let mut previous_codebooks = vec![semantic_code];
            let mut fast_cache = self.model.make_fast_cache();
            // Prefill the projected slow hidden state, then fill the
            // remaining books from the fast head.
            let fast_prefill = self.model.fast_forward_cached(&hidden, &mut fast_cache)?;
            let _ = fast_prefill;
            let mut fast_hidden =
                embedding_row(&self.model.fast_embeddings, ac.dim, semantic_code)?;

            for _ in 0..books - 1 {
                let residual_logits = self
                    .model
                    .fast_forward_cached(&fast_hidden, &mut fast_cache)?;
                let residual_token = sample_logits(
                    &residual_logits,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    &mut |width| uniforms(width),
                )?;
                previous_codebooks.push(residual_token as i32);
                fast_hidden =
                    embedding_row(&self.model.fast_embeddings, ac.dim, residual_token as i32)?;
            }

            generated_steps.push(previous_codebooks.clone());

            // The next slow step feeds [semantic, codebooks...] alone;
            // the cache carries the context.
            let mut next_row = vec![0i32; books + 1];
            next_row[0] = semantic_token;
            next_row[1..].copy_from_slice(&previous_codebooks);
            let (next_logits, next_hidden) = self.model.forward(&next_row, 1, Some(&mut caches))?;
            logits = last_row(&next_logits, self.config.text_config.vocab_size);
            hidden = last_row(&next_hidden, ac.dim);
        }

        if generated_steps.is_empty() {
            return Err(SpeechError::Input {
                why: "no audio tokens were generated for the batch text".to_string(),
            });
        }

        let mut codes = vec![vec![0i32; generated_steps.len()]; books];
        for (step, row) in generated_steps.iter().enumerate() {
            for (book, &code) in row.iter().enumerate() {
                codes[book][step] = code;
            }
        }
        Ok(codes)
    }

    /// The reference `_sample_semantic`: bias the logits, sample at the
    /// call temperature, and resample at the RAS high temperature when
    /// the draw repeats a recent semantic token.
    fn sample_semantic(
        &self,
        logits: &[f32],
        bias: &[f32],
        previous: &[i32],
        params: &GenerateParams,
        uniforms: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Result<i32> {
        let mut biased = logits.to_vec();
        if biased.len() != bias.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "semantic logit bias width {} does not match logits width {}",
                    bias.len(),
                    biased.len()
                ),
            });
        }
        for (value, add) in biased.iter_mut().zip(bias) {
            *value += add;
        }
        let normal = sample_logits(
            &biased,
            params.temperature,
            params.top_p,
            params.top_k,
            &mut |width| uniforms(width),
        )? as i32;
        let in_range = self.config.semantic_start_token_id <= normal
            && normal <= self.config.semantic_end_token_id;
        let should_use_high = in_range && previous.contains(&normal);
        if !should_use_high {
            return Ok(normal);
        }
        let high = sample_logits(
            &biased,
            RAS_HIGH_TEMP,
            RAS_HIGH_TOP_P,
            params.top_k,
            &mut |width| uniforms(width),
        )?;
        Ok(high as i32)
    }
}

/// The caller-resolved special token ids: the tokenized
/// `<|im_start|>user\n`, `<|im_start|>assistant\n<|voice|>`, and
/// `<|im_end|>\n` strings plus the raw im_end id the loop stops on.
#[derive(Debug, Clone)]
pub struct SpecialIds {
    pub user_header_ids: Vec<i32>,
    pub assistant_voice_header_ids: Vec<i32>,
    pub im_end_ids: Vec<i32>,
    pub im_end_token_id: i32,
}

/// Each decode step writes one slow-stack position, and the RoPE table
/// and KV cache are sized `max_seq_len`. Checking prompt plus budget up
/// front turns an out-of-range panic into an Input error. The check also
/// rejects runs that would have stopped early on EOS.
fn check_decode_fits(prompt_len: usize, decode_steps: usize, max_seq_len: usize) -> Result<()> {
    if prompt_len.saturating_add(decode_steps) > max_seq_len {
        return Err(SpeechError::Input {
            why: format!(
                "prompt length {prompt_len} plus {decode_steps} decode steps exceeds max_seq_len {max_seq_len}"
            ),
        });
    }
    Ok(())
}

fn embedding_row(table: &[f32], cols: usize, row: i32) -> Result<Vec<f32>> {
    let index = usize::try_from(row).map_err(|_| SpeechError::Input {
        why: format!("negative embedding row {row}"),
    })?;
    if (index + 1) * cols > table.len() {
        return Err(SpeechError::Input {
            why: format!(
                "embedding row {row} outside a {}-row table",
                table.len() / cols
            ),
        });
    }
    Ok(table[index * cols..(index + 1) * cols].to_vec())
}

fn last_row(rows: &[f32], width: usize) -> Vec<f32> {
    let len = rows.len();
    rows[len - width..].to_vec()
}

/// The reference `_decode_codes` plus `_adjust_speed`: `[books][steps]`
/// codes through the codec, cropped to the reported length, then the
/// linear speed resample.
pub fn decode_codes(codec: &FishS1Dac, codes: &[Vec<i32>], speed: f32) -> Result<Vec<f32>> {
    let samples = codec.decode(codes)?;
    adjust_speed(samples, speed)
}

/// The reference `_adjust_speed`: linear interpolation resample;
/// `speed` within 1e-6 of 1.0 is a no-op.
pub fn adjust_speed(samples: Vec<f32>, speed: f32) -> Result<Vec<f32>> {
    if (speed - 1.0).abs() < 1e-6 {
        return Ok(samples);
    }
    if speed <= 0.0 {
        return Err(SpeechError::Input {
            why: format!("speed must be positive, got {speed}"),
        });
    }
    let old_length = samples.len();
    if old_length == 0 {
        return Ok(samples);
    }
    let new_length = (((old_length as f32) / speed) as usize).max(1);
    let mut out = Vec::with_capacity(new_length);
    let denominator = (new_length - 1).max(1) as f32;
    for index in 0..new_length {
        let position = index as f32 * (old_length - 1) as f32 / denominator;
        let left = position.floor() as usize;
        let right = (left + 1).min(old_length - 1);
        let right_weight = position - left as f32;
        let left_weight = 1.0 - right_weight;
        out.push(left_weight * samples[left] + right_weight * samples[right]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;

/// Test access to a fast-embedding row (`embedding_row` is private).
#[cfg(test)]
pub(crate) fn embedding_row_for_tests(model: &Model, code: i32) -> Result<Vec<f32>> {
    embedding_row(
        &model.model.fast_embeddings,
        model.config.audio_decoder_config.dim,
        code,
    )
}
