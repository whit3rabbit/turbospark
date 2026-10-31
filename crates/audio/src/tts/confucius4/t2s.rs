//! T2S (Text2Semantic) GPT-2 decoder (`t2s.py`): semantic embedding +
//! learned positions over the shared GPT-2 block math, causal KV-cache
//! generation with BOS/EOS 8192/8193, the final norm + semantic head,
//! and the reference sampler (sign-aware repetition penalty over the
//! generated set, temperature, kth-value top-k, descending-cumulative
//! top-p). The categorical draw uses the crate-local threefry RNG (the
//! reference draws from `numpy.random`); `greedy` mode takes the exact
//! argmax and is the fixture-verified path.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::music::minimax_music3::rng;
use crate::nn::load_tensor;
use crate::{ops, Result, SpeechError};

pub const BOS: i32 = 8192;
pub const EOS: i32 = 8193;

#[derive(Debug, Clone)]
pub struct T2SConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub semantic_vocab: usize,
    pub max_positions: usize,
}

impl Default for T2SConfig {
    fn default() -> Self {
        T2SConfig {
            d_model: 1280,
            n_heads: 20,
            n_layers: 24,
            semantic_vocab: 8194,
            max_positions: 4096,
        }
    }
}

struct Block {
    ln_1: crate::nn::LayerNorm,
    c_attn: (Vec<f32>, Option<Vec<f32>>),
    c_proj: (Vec<f32>, Option<Vec<f32>>),
    ln_2: crate::nn::LayerNorm,
    c_fc: (Vec<f32>, Option<Vec<f32>>),
    mlp_proj: (Vec<f32>, Option<Vec<f32>>),
}

/// The loaded T2S stack plus the prefix encoders that share its
/// checkpoint (`t2s_model.safetensors`).
pub struct T2S {
    pub config: T2SConfig,
    semantic_embedding: Vec<f32>,
    semantic_position: Vec<f32>,
    blocks: Vec<Block>,
    ln_f: crate::nn::LayerNorm,
    final_norm: crate::nn::LayerNorm,
    head: (Vec<f32>, Vec<f32>),
}

impl T2S {
    pub fn open(path: &Path, config: T2SConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        Self::load(&file, config)
    }

    pub fn load(file: &SafetensorsFile, config: T2SConfig) -> Result<Self> {
        let d = config.d_model;
        let eps = 1e-5f32;
        let ln = |name: &str| crate::nn::LayerNorm::load(file, name, d, eps);
        let lin =
            |name: &str, input: usize, output: usize| -> Result<(Vec<f32>, Option<Vec<f32>>)> {
                // GPT-2 Conv1D layout: weight stored [in, out].
                Ok((
                    load_tensor(file, &format!("{name}.weight"), &[input, output])?,
                    Some(load_tensor(file, &format!("{name}.bias"), &[output])?),
                ))
            };
        let semantic_embedding = load_tensor(
            file,
            "semantic_embedding.weight",
            &[config.semantic_vocab, d],
        )?;
        let semantic_position = load_tensor(
            file,
            "semantic_position_embedding.embedding.weight",
            &[config.max_positions, d],
        )?;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            let p = format!("transformer.h.{index}");
            blocks.push(Block {
                ln_1: ln(&format!("{p}.ln_1"))?,
                c_attn: lin(&format!("{p}.attn.c_attn"), d, d * 3)?,
                c_proj: lin(&format!("{p}.attn.c_proj"), d, d)?,
                ln_2: ln(&format!("{p}.ln_2"))?,
                c_fc: lin(&format!("{p}.mlp.c_fc"), d, d * 4)?,
                mlp_proj: lin(&format!("{p}.mlp.c_proj"), d * 4, d)?,
            });
        }
        let head = (
            load_tensor(file, "semantic_head.weight", &[config.semantic_vocab, d])?,
            load_tensor(file, "semantic_head.bias", &[config.semantic_vocab])?,
        );
        Ok(T2S {
            config,
            semantic_embedding,
            semantic_position,
            blocks,
            ln_f: ln("transformer.ln_f")?,
            final_norm: ln("final_norm")?,
            head,
        })
    }

    fn semantic_row(&self, token: i32, position: usize) -> Result<Vec<f32>> {
        let d = self.config.d_model;
        let row = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: format!("negative semantic token {token}"),
        })?;
        if row >= self.config.semantic_vocab || position >= self.config.max_positions {
            return Err(SpeechError::Input {
                why: format!("semantic token {token} or position {position} out of range"),
            });
        }
        let mut out = self.semantic_embedding[row * d..(row + 1) * d].to_vec();
        for (col, value) in out.iter_mut().enumerate() {
            *value += self.semantic_position[position * d + col];
        }
        Ok(out)
    }

    /// One GPT-2 block over `[rows, d]` with explicit keys/values
    /// (causal over the concatenated cache).
    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        block: &Block,
        x: &[f32],
        rows: usize,
        total_rows: usize,
        cache: &mut LayerCache,
    ) -> Vec<f32> {
        let d = self.config.d_model;
        let heads = self.config.n_heads;
        let head_dim = d / heads;
        let mut h = x.to_vec();
        block.ln_1.apply(&mut h, rows);
        // Fused c_attn: x @ W ([in, out]) + b.
        let mut qkv = ops::matmul(&h, &block.c_attn.0, rows, d, d * 3);
        if let Some(bias) = &block.c_attn.1 {
            for row in 0..rows {
                for (col, b) in bias.iter().enumerate() {
                    qkv[row * d * 3 + col] += b;
                }
            }
        }
        let base = cache.keys.len() / d;
        // Append k, v to the cache; q over the new rows.
        for t in 0..rows {
            let src = t * d * 3;
            cache.keys.extend_from_slice(&qkv[src + d..src + 2 * d]);
            cache
                .values
                .extend_from_slice(&qkv[src + 2 * d..src + 3 * d]);
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attended = vec![0.0f32; rows * d];
        for t in 0..rows {
            let q_row = &qkv[t * d * 3..t * d * 3 + d];
            // Causal: row t of this pass sees the cache plus itself and
            // the earlier new rows (the reference adds the triu mask on
            // the prefill pass).
            let visible = base + t + 1;
            for head in 0..heads {
                let mut weighted = vec![0.0f32; head_dim];
                let mut scores = vec![0.0f32; visible];
                let mut max = f32::NEG_INFINITY;
                // The index also feeds offsets or math beyond this buffer, so an iterator is not simpler.
                #[allow(clippy::needless_range_loop)]
                for j in 0..visible {
                    let mut dot = 0.0f32;
                    for e in 0..head_dim {
                        dot += q_row[head * head_dim + e] * cache.keys[j * d + head * head_dim + e];
                    }
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                // The index also feeds offsets or math beyond this buffer, so an iterator is not simpler.
                #[allow(clippy::needless_range_loop)]
                for j in 0..visible {
                    let weight = scores[j] * inv;
                    for e in 0..head_dim {
                        weighted[e] += weight * cache.values[j * d + head * head_dim + e];
                    }
                }
                for e in 0..head_dim {
                    attended[t * d + head * head_dim + e] = weighted[e];
                }
            }
        }
        let mut projected = ops::matmul(&attended, &block.c_proj.0, rows, d, d);
        if let Some(bias) = &block.c_proj.1 {
            for row in 0..rows {
                for (col, b) in bias.iter().enumerate() {
                    projected[row * d + col] += b;
                }
            }
        }
        let mut out = x.to_vec();
        for (o, v) in out.iter_mut().zip(&projected) {
            *o += v;
        }
        let mut h2 = out.clone();
        block.ln_2.apply(&mut h2, rows);
        let mut fc = ops::matmul(&h2, &block.c_fc.0, rows, d, d * 4);
        if let Some(bias) = &block.c_fc.1 {
            for row in 0..rows {
                for (col, b) in bias.iter().enumerate() {
                    fc[row * d * 4 + col] += b;
                }
            }
        }
        for value in &mut fc {
            let v = *value;
            // gelu_new: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
            *value = 0.5
                * v
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh());
        }
        let mut mlp = ops::matmul(&fc, &block.mlp_proj.0, rows, d * 4, d);
        if let Some(bias) = &block.mlp_proj.1 {
            for row in 0..rows {
                for (col, b) in bias.iter().enumerate() {
                    mlp[row * d + col] += b;
                }
            }
        }
        for (o, v) in out.iter_mut().zip(&mlp) {
            *o += v;
        }
        let _ = total_rows;
        out
    }

    /// Runs the stack over `[rows, d]` embeddings, appending to the
    /// caches, and returns the post-`ln_f` states.
    fn transformer(&self, x: &[f32], rows: usize, caches: &mut [LayerCache]) -> Vec<f32> {
        let mut out = x.to_vec();
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            out = self.block_forward(block, &out, rows, rows, cache);
        }
        self.ln_f.apply(&mut out, rows);
        out
    }

    /// The semantic head over post-`ln_f` states.
    pub fn head(&self, h: &[f32], rows: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let mut out = h.to_vec();
        self.final_norm.apply(&mut out, rows);
        ops::linear(
            &out,
            &self.head.0,
            Some(&self.head.1),
            rows,
            d,
            self.config.semantic_vocab,
        )
    }

    /// Teacher-forced step logits over `[cond, text, BOS, codes...]`:
    /// one row per greedy step (the prefill head plus each fed code's
    /// head), matching the recorded fixture sequence.
    pub fn step_logits(
        &self,
        cond_emb: &[f32],
        text_emb: &[f32],
        codes: &[i32],
    ) -> Result<Vec<Vec<f32>>> {
        let mut caches: Vec<LayerCache> = (0..self.config.n_layers)
            .map(|_| LayerCache::default())
            .collect();
        let mut prefill = cond_emb.to_vec();
        prefill.extend_from_slice(text_emb);
        prefill.extend_from_slice(&self.semantic_row(BOS, 0)?);
        let rows = prefill.len() / self.config.d_model;
        let h = self.transformer(&prefill, rows, &mut caches);
        let mut logits = vec![self.head(
            &h[(rows - 1) * self.config.d_model..rows * self.config.d_model],
            1,
        )];
        for (position, &code) in codes.iter().enumerate() {
            let embed = self.semantic_row(code, position + 1)?;
            let h = self.transformer(&embed, 1, &mut caches);
            logits.push(self.head(&h, 1));
        }
        Ok(logits)
    }

    /// `cond_emb` runs in the prefix module (`prefix.py`); the speaker
    /// ECAPA lives in [`super::prefix`].
    ///
    /// `generate`: KV-cached sampling over `[cond_emb, text_emb, BOS]`.
    /// `greedy` takes the argmax (the fixture-verified path); otherwise
    /// the reference filter chain runs with crate-local threefry draws.
    /// Returns the semantic codes (BOS/EOS stripped) and the LM latent
    /// `[T, d]` gathered from the full re-pass (`hful[:, 1 + Tt:-2]`).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        cond_emb: &[f32],
        text_emb: &[f32],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rep_pen: f32,
        greedy: bool,
        key: rng::Key,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        if !greedy {
            validate_temperature(temperature)?;
        }
        let d = self.config.d_model;
        let text_len = text_emb.len() / d;
        let mut caches: Vec<LayerCache> = (0..self.config.n_layers)
            .map(|_| LayerCache::default())
            .collect();
        // Prefill [cond, text, BOS].
        let mut prefill = cond_emb.to_vec();
        prefill.extend_from_slice(text_emb);
        prefill.extend_from_slice(&self.semantic_row(BOS, 0)?);
        let prefill_rows = 1 + text_len + 1;
        let h = self.transformer(&prefill, prefill_rows, &mut caches);
        let mut logits = self.head(&h[(prefill_rows - 1) * d..prefill_rows * d], 1);
        let mut cur: Vec<i32> = vec![BOS];
        // Row 0 is BOS, so the first sampled token sits at position 1.
        for (position, _) in (1usize..).zip(0..max_new) {
            let token = if greedy {
                crate::tts::sampling::argmax(&logits)
            } else {
                sample_token(&logits, &cur[1..], temperature, top_k, top_p, rep_pen, key)
            } as i32;
            cur.push(token);
            if token == EOS {
                break;
            }
            let embed = self.semantic_row(token, position)?;
            let h = self.transformer(&embed, 1, &mut caches);
            logits = self.head(&h, 1);
        }
        // Full causal re-pass over [prefix, scodes]; the latent is
        // hful[:, 1 + Tt:-2] (drop the cond row, the text rows, the BOS
        // row, and the final EOS row).
        let scodes = &cur;
        let mut full = cond_emb.to_vec();
        full.extend_from_slice(text_emb);
        for (position, &token) in scodes.iter().enumerate() {
            full.extend_from_slice(&self.semantic_row(token, position)?);
        }
        let rows = full.len() / d;
        let caches_full: Vec<LayerCache> = (0..self.config.n_layers)
            .map(|_| LayerCache::default())
            .collect();
        let mut caches_full = caches_full;
        let hful = self.transformer(&full, rows, &mut caches_full);
        let latent_start = (1 + text_len) * d;
        let latent_end = (rows - 2) * d;
        let codes: Vec<i32> = scodes[1..scodes.len().saturating_sub(1)].to_vec();
        Ok((codes, hful[latent_start..latent_end].to_vec()))
    }
}

/// The sampling path divides logits by temperature. Zero gives inf logits
/// and NaN probabilities, and a non-finite value does the same, so both
/// are refused before any draw.
pub(crate) fn validate_temperature(temperature: f32) -> Result<()> {
    if temperature.is_finite() && temperature > 0.0 {
        return Ok(());
    }
    Err(SpeechError::Input {
        why: format!("temperature must be finite and > 0, got {temperature}"),
    })
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// The reference `_sample` filter chain on raw logits, returning the
/// final probability vector (exposed for the fixture parity gate).
pub fn sample_distribution(
    logits: &[f32],
    generated: &[i32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rep_pen: f32,
) -> Vec<f32> {
    let vocab = logits.len();
    let mut logits = logits.to_vec();
    // Repetition penalty over the SET of generated tokens.
    if rep_pen != 1.0 {
        let mut seen: Vec<usize> = Vec::new();
        for &token in generated {
            if let Ok(index) = usize::try_from(token) {
                if index < vocab && !seen.contains(&index) {
                    seen.push(index);
                }
            }
        }
        for index in seen {
            logits[index] = if logits[index] > 0.0 {
                logits[index] / rep_pen
            } else {
                logits[index] * rep_pen
            };
        }
    }
    for value in &mut logits {
        *value /= temperature;
    }
    if top_k > 0 && top_k < vocab {
        let kth = crate::music::minimax_music3::rng::kth_largest(&logits, top_k);
        for value in &mut logits {
            if *value < kth {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    // Descending order, cumulative probability, keep[0] always.
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
    let max = sorted.first().copied().unwrap_or(0.0);
    let mut probs: Vec<f32> = sorted.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    for value in &mut probs {
        *value /= total;
    }
    let mut cumulative = 0.0f32;
    let mut kept = vec![false; vocab];
    for (index, value) in probs.iter_mut().enumerate() {
        cumulative += *value;
        // np.cumsum includes the current entry; keep[0] always survives.
        let keep = cumulative <= top_p || index == 0;
        if keep {
            kept[index] = true;
        } else {
            *value = 0.0;
        }
    }
    // The reference rebuilds the probability vector in vocabulary
    // order: exp(full - max(full)) with -inf at the dropped entries,
    // renormalized over the survivors.
    let mut out = vec![0.0f32; vocab];
    let mut kept_mass = 0.0f32;
    for (index, &slot) in order.iter().enumerate() {
        if kept[index] {
            let p = (logits[slot] - max).exp();
            out[slot] = p;
            kept_mass += p;
        }
    }
    if kept_mass > 0.0 {
        for value in &mut out {
            *value /= kept_mass;
        }
    }
    out
}

/// The reference `_sample` filter chain plus the categorical draw.
#[allow(clippy::too_many_arguments)]
fn sample_token(
    logits: &[f32],
    generated: &[i32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rep_pen: f32,
    key: rng::Key,
) -> usize {
    let probs = sample_distribution(logits, generated, temperature, top_k, top_p, rep_pen);
    let log_probs: Vec<f32> = probs.iter().map(|&p| p.max(1e-45).ln()).collect();
    rng::categorical(&log_probs, key)
}
