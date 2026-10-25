//! The Echo-TTS denoiser transformer (`EchoDiT`).
//!
//! Port of `mlx_audio/tts/models/echo_tts/model.py` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/echo_tts),
//! single batch, portable f32. A flow-matching transformer over codec
//! latents conditioned on a bidirectional byte-level text encoder, a
//! causal patched speaker encoder, and a timestep embedding: joint
//! attention blocks whose queries attend their own keys plus an
//! optional latent-prefix KV, the text KV, and the speaker KV, with
//! low-rank AdaLN conditioning and sigmoid gates.
//!
//! Reference quirks kept exactly:
//! - the joint attention applies rotary embeddings to the FIRST HALF of
//!   the query/self-key sequence positions only (`_apply_rotary_half`);
//! - the per-head q/k norms carry a stored `[heads, dim]` weight (MLX
//!   registers the ones array as a parameter, so checkpoints carry
//!   trained values) and the vector norms a `[dim]` weight;
//! - the additive mask value is `-1e9`, not `-inf`;
//! - the speaker mask is subsampled by the patch size inside the DiT
//!   call (`speaker_mask[..., ::patch]`).

use crate::nn::Linear;
use crate::{Result, SpeechError};

use super::config::EchoDiTConfig;

/// The additive mask value of the reference `_bool_to_additive_mask`.
const MASK_VALUE: f32 = -1e9;

/// Precomputed rotary tables `[positions, dim / 2]`
/// (`precompute_freqs_cis`).
#[derive(Debug, Clone)]
pub struct RotaryCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub positions: usize,
}

impl RotaryCache {
    fn half(&self) -> usize {
        if self.positions == 0 {
            0
        } else {
            self.cos.len() / self.positions
        }
    }

    /// `cos/sin[start_pos .. start_pos + rows]` gathered into a fresh
    /// table (the reference slices the table before `_apply_rotary_half`).
    fn slice(&self, start_pos: usize, rows: usize) -> RotaryCache {
        let half = self.half();
        let start = start_pos.min(self.positions);
        let end = (start_pos + rows).min(self.positions);
        RotaryCache {
            cos: self.cos[start * half..end * half].to_vec(),
            sin: self.sin[start * half..end * half].to_vec(),
            positions: end - start,
        }
    }

    /// Gathers the rows `[p * stride]` for `p in 0..rows` (the latent
    /// prefix's patch-strided positions).
    fn gather_strided(&self, stride: usize, rows: usize) -> RotaryCache {
        let half = self.half();
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for p in 0..rows {
            let row = (p * stride).min(self.positions.saturating_sub(1));
            cos.extend_from_slice(&self.cos[row * half..(row + 1) * half]);
            sin.extend_from_slice(&self.sin[row * half..(row + 1) * half]);
        }
        RotaryCache {
            cos,
            sin,
            positions: rows,
        }
    }
}

/// `precompute_freqs_cis(dim, end, theta)`: `cos/sin(t * theta**-(2j/dim))`.
pub fn precompute_freqs_cis(dim: usize, end: usize, theta: f32) -> RotaryCache {
    let half = dim / 2;
    let mut cos = vec![0.0f32; end * half];
    let mut sin = vec![0.0f32; end * half];
    for t in 0..end {
        for j in 0..half {
            // mx.arange(0, dim, 2)[:dim // 2] / dim = 2j / dim.
            let freq = theta.powf(-((2.0 * j as f32) / dim as f32));
            let angle = t as f32 * freq;
            cos[t * half + j] = angle.cos();
            sin[t * half + j] = angle.sin();
        }
    }
    RotaryCache {
        cos,
        sin,
        positions: end,
    }
}

/// `get_timestep_embedding`: `[cos(t * f), sin(t * f)]` with
/// `f = 1000 * exp(-ln(10000) * j / half)`.
pub fn timestep_embedding(t: f32, embed_size: usize) -> Vec<f32> {
    let half = embed_size / 2;
    let mut out = vec![0.0f32; embed_size];
    let base = 10000.0f32.ln();
    for j in 0..half {
        let freq = 1000.0 * (-base * (j as f32) / half as f32).exp();
        let angle = t * freq;
        out[j] = angle.cos();
        out[half + j] = angle.sin();
    }
    out
}

/// Interleaved-pair rotary over `[batch, seq, heads, dim]` flat data:
/// pair `(2p, 2p + 1)` of every head rotates by position
/// `start_pos + t`'s angle.
pub fn apply_rotary_emb(
    x: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
    freqs: &RotaryCache,
    start_pos: usize,
) {
    let half = dim / 2;
    for b in 0..batch {
        for t in 0..seq {
            let position = start_pos + t;
            if position >= freqs.positions {
                continue;
            }
            for h in 0..heads {
                let row = ((b * seq + t) * heads + h) * dim;
                for p in 0..half {
                    let cos = freqs.cos[position * half + p];
                    let sin = freqs.sin[position * half + p];
                    let even = x[row + 2 * p];
                    let odd = x[row + 2 * p + 1];
                    x[row + 2 * p] = even * cos - odd * sin;
                    x[row + 2 * p + 1] = odd * cos + even * sin;
                }
            }
        }
    }
}

/// `_apply_rotary_half`: rotary over the first `heads / 2` heads only
/// (the reference slices `y.shape[-2]`, the head axis of the
/// `[batch, seq, heads, dim]` layout); the remaining heads pass through
/// untouched.
fn apply_rotary_half(
    x: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
    freqs: &RotaryCache,
    start_pos: usize,
) {
    let half_heads = heads / 2;
    if half_heads == 0 {
        return;
    }
    let sliced = freqs.slice(start_pos, seq);
    let half = dim / 2;
    for b in 0..batch {
        for t in 0..seq {
            if t >= sliced.positions {
                continue;
            }
            for h in 0..half_heads {
                let row = ((b * seq + t) * heads + h) * dim;
                for p in 0..half {
                    let cos = sliced.cos[t * half + p];
                    let sin = sliced.sin[t * half + p];
                    let even = x[row + 2 * p];
                    let odd = x[row + 2 * p + 1];
                    x[row + 2 * p] = even * cos - odd * sin;
                    x[row + 2 * p + 1] = odd * cos + even * sin;
                }
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// RMS norm with the reference's stored weight: over `dim`-wide rows of
/// the flat data, with the weight index of element `(row, col)` equal to
/// `(row % heads) * dim + col`. A plain vector weight is `heads = 1`
/// (then the index is just `col`); the per-head q/k norms use
/// `heads = num_heads`, `dim = head_dim` over `[.., heads, dim]` data.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Vec<f32>,
    heads: usize,
    dim: usize,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, heads: usize, dim: usize, eps: f32) -> Self {
        RmsNorm {
            weight,
            heads: heads.max(1),
            dim,
            eps,
        }
    }

    /// Applies the norm over the whole flat buffer
    /// (`[rows, heads * dim]` semantics).
    pub fn apply(&self, x: &mut [f32]) {
        let dim = self.dim;
        for (row, chunk) in x.chunks_exact_mut(dim).enumerate() {
            let mut mean_sq = 0.0f32;
            for value in chunk.iter() {
                mean_sq += value * value;
            }
            let inv = 1.0 / (mean_sq / dim as f32 + self.eps).sqrt();
            let head = row % self.heads;
            for (col, value) in chunk.iter_mut().enumerate() {
                *value *= inv * self.weight[head * dim + col];
            }
        }
    }
}

/// One per-layer KV block: keys and values `[batch, keys, heads, dim]`
/// flat (the reference `(k, v)` tuple).
#[derive(Debug, Clone)]
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub keys: usize,
    pub batch: usize,
}

impl KvCache {
    pub fn empty(batch: usize, _heads: usize, _dim: usize) -> Self {
        KvCache {
            k: Vec::new(),
            v: Vec::new(),
            keys: 0,
            batch,
        }
    }
}

/// Concatenates same-shape KV blocks along the BATCH axis (the
/// reference `_concat_kv_caches` runs `mx.concatenate(..., axis=0)`).
pub(crate) fn concat_kv(parts: &[&KvCache]) -> KvCache {
    let keys = parts.first().map_or(0, |p| p.keys);
    let mut k = Vec::new();
    let mut v = Vec::new();
    for part in parts {
        debug_assert_eq!(part.keys, keys, "mixed key counts in concat");
        k.extend_from_slice(&part.k);
        v.extend_from_slice(&part.v);
    }
    KvCache {
        k,
        v,
        keys,
        batch: parts.len(),
    }
}

/// Scales the first `max_layers` caches by `scale`
/// (`_multiply_kv_cache`).
pub(crate) fn multiply_kv(
    cache: &[KvCache],
    scale: f32,
    max_layers: Option<usize>,
) -> Vec<KvCache> {
    let limit = max_layers.unwrap_or(cache.len()).min(cache.len());
    cache
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            if index >= limit {
                return layer.clone();
            }
            KvCache {
                k: layer.k.iter().map(|v| v * scale).collect(),
                v: layer.v.iter().map(|v| v * scale).collect(),
                keys: layer.keys,
                batch: layer.batch,
            }
        })
        .collect()
}

/// Softmax attention over explicit key/value blocks with the reference
/// additive masking: masked scores sit at `-1e9` inside the softmax.
#[allow(clippy::too_many_arguments)]
fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    total_keys: usize,
    heads: usize,
    dim: usize,
    mask: &[bool],
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; batch * seq * heads * dim];
    let mut scores = vec![0.0f32; total_keys];
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq {
                let q_base = ((b * seq + i) * heads + h) * dim;
                let mut max = f32::NEG_INFINITY;
                for j in 0..total_keys {
                    let k_base = ((b * total_keys + j) * heads + h) * dim;
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let score = if mask[(b * seq + i) * total_keys + j] {
                        dot * scale
                    } else {
                        MASK_VALUE
                    };
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores[..] {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                let out_base = ((b * seq + i) * heads + h) * dim;
                for j in 0..total_keys {
                    let weight = scores[j] * inv;
                    if weight == 0.0 {
                        continue;
                    }
                    let v_base = ((b * total_keys + j) * heads + h) * dim;
                    for d in 0..dim {
                        out[out_base + d] += weight * v[v_base + d];
                    }
                }
            }
        }
    }
    out
}

pub(crate) struct Mlp {
    pub(crate) w1: Linear,
    pub(crate) w2: Linear,
    pub(crate) w3: Linear,
}

impl Mlp {
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let gate = self.w1.forward(x, rows);
        let up = self.w3.forward(x, rows);
        let mut fused = vec![0.0f32; gate.len()];
        for (f, (g, u)) in fused.iter_mut().zip(gate.iter().zip(&up)) {
            *f = g * sigmoid(*g) * u;
        }
        self.w2.forward(&fused, rows)
    }
}

impl LowRankAdaLN {}

pub(crate) struct LowRankAdaLN {
    pub(crate) shift_down: Linear,
    pub(crate) shift_up: Linear,
    pub(crate) scale_down: Linear,
    pub(crate) scale_up: Linear,
    pub(crate) gate_down: Linear,
    pub(crate) gate_up: Linear,
    pub(crate) model_size: usize,
    pub(crate) eps: f32,
}

impl LowRankAdaLN {
    /// Returns the conditioned norm output plus the tanh gate, both
    /// `[rows, model_size]`; `cond` is one `[3 * model_size]` row
    /// broadcast over every position (the reference keeps a
    /// `[batch, 1, 3m]` embedding).
    fn apply(&self, x: &[f32], rows: usize, cond: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let m = self.model_size;
        let resolve = |down: &Linear, up: &Linear, base: &[f32]| -> Vec<f32> {
            // self.shift_up(self.shift_down(silu(base))) + base: the SiLU
            // sits on the input, not between the projections.
            let activated: Vec<f32> = base.iter().map(|&v| v * sigmoid(v)).collect();
            let low = down.forward(&activated, 1);
            let mut out = up.forward(&low, 1);
            for (o, b) in out.iter_mut().zip(base) {
                *o += b;
            }
            out
        };
        let shift = resolve(&self.shift_down, &self.shift_up, &cond[..m]);
        let scale = resolve(&self.scale_down, &self.scale_up, &cond[m..2 * m]);
        let gate = resolve(&self.gate_down, &self.gate_up, &cond[2 * m..]);
        let mut normed = vec![0.0f32; x.len()];
        let mut gates = vec![0.0f32; rows * m];
        for row in 0..rows {
            let base = row * m;
            let mut mean_sq = 0.0f32;
            for value in &x[base..base + m] {
                mean_sq += value * value;
            }
            let inv = 1.0 / (mean_sq / m as f32 + self.eps).sqrt();
            for col in 0..m {
                normed[base + col] = x[base + col] * inv * (scale[col] + 1.0) + shift[col];
                gates[base + col] = gate[col].tanh();
            }
        }
        (normed, gates)
    }
}

pub(crate) struct SelfAttention {
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) wo: Linear,
    pub(crate) gate: Linear,
    pub(crate) q_norm: RmsNorm,
    pub(crate) k_norm: RmsNorm,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) is_causal: bool,
}

impl SelfAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        batch: usize,
        seq: usize,
        key_mask: Option<&[bool]>,
        freqs: &RotaryCache,
    ) -> Vec<f32> {
        let rows = batch * seq;
        let model = self.heads * self.head_dim;
        let mut xq = self.wq.forward(x, rows);
        let mut xk = self.wk.forward(x, rows);
        let xv = self.wv.forward(x, rows);
        let gate = self.gate.forward(x, rows);
        self.q_norm.apply(&mut xq);
        self.k_norm.apply(&mut xk);
        apply_rotary_emb(&mut xq, batch, seq, self.heads, self.head_dim, freqs, 0);
        apply_rotary_emb(&mut xk, batch, seq, self.heads, self.head_dim, freqs, 0);
        // Broadcast the key mask over the query axis; intersect with the
        // causal bound when configured.
        let mut mask = vec![true; batch * seq * seq];
        for b in 0..batch {
            for i in 0..seq {
                for j in 0..seq {
                    let mut visible = true;
                    if let Some(keys) = key_mask {
                        visible = keys[b * seq + j];
                    }
                    if self.is_causal {
                        visible &= j <= i;
                    }
                    mask[(b * seq + i) * seq + j] = visible;
                }
            }
        }
        let attended = masked_attention(
            &xq,
            &xk,
            &xv,
            batch,
            seq,
            seq,
            self.heads,
            self.head_dim,
            &mask,
        );
        let mut joined = vec![0.0f32; rows * model];
        for row in 0..rows {
            for h in 0..self.heads {
                let src = (row * self.heads + h) * self.head_dim;
                let dst = row * model + h * self.head_dim;
                joined[dst..dst + self.head_dim]
                    .copy_from_slice(&attended[src..src + self.head_dim]);
            }
            for (value, g) in joined[row * model..(row + 1) * model]
                .iter_mut()
                .zip(&gate[row * model..(row + 1) * model])
            {
                *value *= sigmoid(*g);
            }
        }
        self.wo.forward(&joined, rows)
    }
}

pub(crate) struct JointAttention {
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) gate: Linear,
    pub(crate) wo: Linear,
    pub(crate) wk_text: Linear,
    pub(crate) wv_text: Linear,
    pub(crate) wk_speaker: Linear,
    pub(crate) wv_speaker: Linear,
    pub(crate) wk_latent: Option<Linear>,
    pub(crate) wv_latent: Option<Linear>,
    pub(crate) q_norm: RmsNorm,
    pub(crate) k_norm: RmsNorm,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) speaker_patch_size: usize,
}

impl JointAttention {
    /// Builds the text KV block: `wk_text/wv_text` over the encoded text
    /// state with the shared k norm (`get_kv_cache_text`).
    pub fn kv_text(&self, text_state: &[f32], rows: usize, batch: usize) -> KvCache {
        let mut k = self.wk_text.forward(text_state, rows);
        let v = self.wv_text.forward(text_state, rows);
        self.k_norm.apply(&mut k);
        KvCache {
            k,
            v,
            keys: rows / batch.max(1),
            batch,
        }
    }

    /// `get_kv_cache_speaker`.
    pub fn kv_speaker(&self, speaker_state: &[f32], rows: usize, batch: usize) -> KvCache {
        let mut k = self.wk_speaker.forward(speaker_state, rows);
        let v = self.wv_speaker.forward(speaker_state, rows);
        self.k_norm.apply(&mut k);
        KvCache {
            k,
            v,
            keys: rows / batch.max(1),
            batch,
        }
    }

    /// `get_kv_cache_latent`: the shared k norm plus the rotary angles
    /// at the patch-strided positions.
    pub fn kv_latent(
        &self,
        latent_state: &[f32],
        batch: usize,
        seq: usize,
        freqs: &RotaryCache,
    ) -> Result<KvCache> {
        let wk_latent = self
            .wk_latent
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "latent-prefix KV modules are disabled in this checkpoint".into(),
            })?;
        let wv_latent = self
            .wv_latent
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "latent-prefix KV modules are disabled in this checkpoint".into(),
            })?;
        let rows = batch * seq;
        let mut k = wk_latent.forward(latent_state, rows);
        let v = wv_latent.forward(latent_state, rows);
        self.k_norm.apply(&mut k);
        let gathered = freqs.gather_strided(self.speaker_patch_size, seq);
        apply_rotary_half(&mut k, batch, seq, self.heads, self.head_dim, &gathered, 0);
        Ok(KvCache {
            k,
            v,
            keys: seq,
            batch,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        batch: usize,
        seq: usize,
        text_mask: &[bool],
        speaker_mask: &[bool],
        freqs: &RotaryCache,
        kv_text: &KvCache,
        kv_speaker: &KvCache,
        start_pos: usize,
        kv_latent: Option<&KvCache>,
    ) -> Vec<f32> {
        let rows = batch * seq;
        let heads = self.heads;
        let dim = self.head_dim;
        let heads_dim = heads * dim;
        let mut xq = self.wq.forward(x, rows);
        let mut xk = self.wk.forward(x, rows);
        let xv = self.wv.forward(x, rows);
        let gate = self.gate.forward(x, rows);
        self.q_norm.apply(&mut xq);
        self.k_norm.apply(&mut xk);
        apply_rotary_half(&mut xq, batch, seq, heads, dim, freqs, start_pos);
        apply_rotary_half(&mut xk, batch, seq, heads, dim, freqs, start_pos);

        // The latent mask: latent patch j is visible when its strided
        // position is below start_pos; one row per batch.
        let latent_keys = kv_latent.map_or(0, |c| c.keys);
        let latent_mask: Vec<bool> = (0..batch * latent_keys)
            .map(|index| index % latent_keys.max(1) * self.speaker_patch_size < start_pos)
            .collect();

        let text_keys = kv_text.keys;
        let speaker_keys = kv_speaker.keys;
        let total_keys = seq + latent_keys + text_keys + speaker_keys;

        // Concatenate keys, values, and the visibility mask per batch.
        let mut k_all = vec![0.0f32; batch * total_keys * heads_dim];
        let mut v_all = vec![0.0f32; batch * total_keys * heads_dim];
        let mut k_blocks: Vec<(&[f32], usize)> = Vec::with_capacity(4);
        let mut v_blocks: Vec<(&[f32], usize)> = Vec::with_capacity(4);
        k_blocks.push((&xk, 0));
        v_blocks.push((&xv, 0));
        if let Some(latent) = kv_latent {
            if latent_keys > 0 {
                k_blocks.push((&latent.k, seq));
                v_blocks.push((&latent.v, seq));
            }
        }
        k_blocks.push((&kv_text.k, seq + latent_keys));
        v_blocks.push((&kv_text.v, seq + latent_keys));
        k_blocks.push((&kv_speaker.k, seq + latent_keys + text_keys));
        v_blocks.push((&kv_speaker.v, seq + latent_keys + text_keys));
        for (src, offset) in k_blocks {
            if src.is_empty() {
                continue;
            }
            let span = src.len() / batch.max(1);
            for b in 0..batch {
                let dst = (b * total_keys + offset) * heads_dim;
                k_all[dst..dst + span].copy_from_slice(&src[b * span..(b + 1) * span]);
            }
        }
        for (src, offset) in v_blocks {
            if src.is_empty() {
                continue;
            }
            let span = src.len() / batch.max(1);
            for b in 0..batch {
                let dst = (b * total_keys + offset) * heads_dim;
                v_all[dst..dst + span].copy_from_slice(&src[b * span..(b + 1) * span]);
            }
        }
        // Per-key visibility, broadcast over the query axis for the
        // shared attention kernel.
        let mut key_mask = vec![false; batch * total_keys];
        for b in 0..batch {
            let base = b * total_keys;
            for j in 0..seq {
                key_mask[base + j] = true;
            }
            for j in 0..latent_keys {
                key_mask[base + seq + j] = latent_mask[b * latent_keys + j];
            }
            for j in 0..text_keys {
                key_mask[base + seq + latent_keys + j] = text_mask[b * text_keys + j];
            }
            for j in 0..speaker_keys {
                key_mask[base + seq + latent_keys + text_keys + j] =
                    speaker_mask[b * speaker_keys + j];
            }
        }
        let mut mask = vec![false; batch * seq * total_keys];
        for b in 0..batch {
            for i in 0..seq {
                let src = b * total_keys;
                let dst = (b * seq + i) * total_keys;
                mask[dst..dst + total_keys].copy_from_slice(&key_mask[src..src + total_keys]);
            }
        }

        let attended = masked_attention(
            &xq, &k_all, &v_all, batch, seq, total_keys, heads, dim, &mask,
        );
        let model = heads * dim;
        let mut joined = vec![0.0f32; rows * model];
        for row in 0..rows {
            for h in 0..heads {
                let src = (row * heads + h) * dim;
                let dst = row * model + h * dim;
                joined[dst..dst + dim].copy_from_slice(&attended[src..src + dim]);
            }
            for (value, g) in joined[row * model..(row + 1) * model]
                .iter_mut()
                .zip(&gate[row * model..(row + 1) * model])
            {
                *value *= sigmoid(*g);
            }
        }
        self.wo.forward(&joined, rows)
    }
}

pub(crate) struct EncoderBlock {
    pub(crate) attention: SelfAttention,
    pub(crate) mlp: Mlp,
    pub(crate) attention_norm: RmsNorm,
    pub(crate) mlp_norm: RmsNorm,
}

impl EncoderBlock {
    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq: usize,
        key_mask: Option<&[bool]>,
        freqs: &RotaryCache,
    ) {
        let rows = batch * seq;
        let mut normed = x.to_vec();
        self.attention_norm.apply(&mut normed);
        let attended = self.attention.forward(&normed, batch, seq, key_mask, freqs);
        for (value, add) in x.iter_mut().zip(attended) {
            *value += add;
        }
        let mut normed = x.to_vec();
        self.mlp_norm.apply(&mut normed);
        let fed = self.mlp.forward(&normed, rows);
        for (value, add) in x.iter_mut().zip(fed) {
            *value += add;
        }
    }
}

pub(crate) struct DecoderBlock {
    pub(crate) attention: JointAttention,
    pub(crate) mlp: Mlp,
    pub(crate) attention_adaln: LowRankAdaLN,
    pub(crate) mlp_adaln: LowRankAdaLN,
}

/// The text encoder: bidirectional byte-level transformer
/// (`TextEncoder`).
pub struct TextEncoder {
    pub(crate) embedding: Vec<f32>,
    pub(crate) vocab_size: usize,
    pub(crate) model_size: usize,
    pub(crate) blocks: Vec<EncoderBlock>,
    pub(crate) head_dim: usize,
}

impl TextEncoder {
    /// Encodes `[batch, seq]` ids into `[batch, seq, model]` states with
    /// a per-batch key mask.
    pub fn forward(
        &self,
        input_ids: &[i32],
        batch: usize,
        seq: usize,
        key_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let model = self.model_size;
        let mut x = vec![0.0f32; batch * seq * model];
        for (index, &id) in input_ids.iter().enumerate() {
            let row = usize::try_from(id).map_err(|_| SpeechError::Input {
                why: format!("negative text token id {id}"),
            })?;
            if row >= self.vocab_size {
                return Err(SpeechError::Input {
                    why: format!("text token id {id} outside vocab {}", self.vocab_size),
                });
            }
            x[index * model..(index + 1) * model]
                .copy_from_slice(&self.embedding[row * model..(row + 1) * model]);
        }
        let head_dim = self.head_dim;
        let freqs = precompute_freqs_cis(head_dim, seq, 10000.0);
        for block in &self.blocks {
            block.forward(&mut x, batch, seq, key_mask, &freqs);
        }
        Ok(x)
    }
}

/// The speaker (and latent-prefix) encoder: patched-latent causal
/// transformer (`SpeakerEncoder`).
pub struct SpeakerEncoder {
    pub(crate) in_proj: Linear,
    pub(crate) blocks: Vec<EncoderBlock>,
    pub(crate) patch_size: usize,
    pub(crate) head_dim: usize,
}

impl SpeakerEncoder {
    /// Encodes `[batch, rows, latent_dim]` flat data into patched
    /// causal states `[batch, rows / patch, model]` (the reference
    /// drops the unpatchable tail before the patch reshape and divides
    /// the projection by 6.0).
    pub fn forward(
        &self,
        latent: &[f32],
        batch: usize,
        rows: usize,
        latent_dim: usize,
    ) -> Vec<f32> {
        let patched = (rows / self.patch_size) * self.patch_size;
        if patched == 0 {
            return Vec::new();
        }
        let seq = patched / self.patch_size;
        let input_width = self.in_proj.input;
        let mut input = vec![0.0f32; batch * seq * input_width];
        for b in 0..batch {
            for p in 0..seq {
                for r in 0..self.patch_size {
                    let src = (b * rows + p * self.patch_size + r) * latent_dim;
                    let dst = (b * seq + p) * input_width + r * latent_dim;
                    input[dst..dst + latent_dim].copy_from_slice(&latent[src..src + latent_dim]);
                }
            }
        }
        let mut out = self.in_proj.forward(&input, batch * seq);
        for value in &mut out {
            *value /= 6.0;
        }
        let freqs = precompute_freqs_cis(self.head_dim, seq, 10000.0);
        for block in &self.blocks {
            block.forward(&mut out, batch, seq, None, &freqs);
        }
        out
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }
}

/// The Echo-TTS denoiser (`EchoDiT`).
pub struct EchoDiT {
    pub config: EchoDiTConfig,
    pub(crate) text_encoder: TextEncoder,
    speaker_encoder: SpeakerEncoder,
    latent_encoder: Option<SpeakerEncoder>,
    latent_norm: Option<RmsNorm>,
    text_norm: RmsNorm,
    speaker_norm: RmsNorm,
    cond_module: [Linear; 3],
    in_proj: Linear,
    blocks: Vec<DecoderBlock>,
    out_norm: RmsNorm,
    out_proj: Linear,
    head_dim: usize,
}

impl EchoDiT {
    pub(crate) fn from_parts(
        config: EchoDiTConfig,
        text_encoder: TextEncoder,
        speaker_encoder: SpeakerEncoder,
        latent_encoder: Option<SpeakerEncoder>,
        latent_norm: Option<RmsNorm>,
        text_norm: RmsNorm,
        speaker_norm: RmsNorm,
        cond_module: [Linear; 3],
        in_proj: Linear,
        blocks: Vec<DecoderBlock>,
        out_norm: RmsNorm,
        out_proj: Linear,
    ) -> Self {
        let head_dim = config.model_size / config.num_heads;
        EchoDiT {
            config,
            text_encoder,
            speaker_encoder,
            latent_encoder,
            latent_norm,
            text_norm,
            speaker_norm,
            cond_module,
            in_proj,
            blocks,
            out_norm,
            out_proj,
            head_dim,
        }
    }

    pub(crate) fn block_joint(&self, index: usize) -> &JointAttention {
        &self.blocks[index].attention
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn latent_stack(&self) -> Option<(&SpeakerEncoder, &RmsNorm)> {
        self.latent_encoder.as_ref().zip(self.latent_norm.as_ref())
    }

    /// The encoded text state (embedding + bidirectional encoder +
    /// norm): the input of the per-block text KV projections.
    pub fn text_state(&self, text_ids: &[i32], text_mask: &[bool]) -> Result<Vec<f32>> {
        let batch = 1usize;
        let seq = text_mask.len();
        let mut state = self
            .text_encoder
            .forward(text_ids, batch, seq, Some(text_mask))?;
        self.text_norm.apply(&mut state);
        Ok(state)
    }

    /// Builds the text KV caches: embeds the ids, runs the text encoder
    /// plus its norm, then projects the state per block
    /// (`get_kv_cache_text`).
    pub fn get_kv_cache_text(&self, text_ids: &[i32], text_mask: &[bool]) -> Result<Vec<KvCache>> {
        let batch = 1usize;
        let state = self.text_state(text_ids, text_mask)?;
        let seq = text_mask.len();
        Ok((0..self.num_blocks())
            .map(|index| self.block_joint(index).kv_text(&state, seq, batch))
            .collect())
    }

    /// Builds the speaker KV caches from the patched speaker latent
    /// `[rows, latent]` (`get_kv_cache_speaker`).
    pub fn get_kv_cache_speaker(&self, speaker_latent: &[f32], rows: usize) -> Vec<KvCache> {
        let latent_dim = self.config.latent_size;
        let state = self
            .speaker_encoder
            .forward(speaker_latent, 1, rows, latent_dim);
        let batch = 1usize;
        let seq = if rows == 0 {
            0
        } else {
            state.len() / self.config.speaker_model_size
        };
        let mut state = state;
        self.speaker_norm.apply(&mut state);
        (0..self.num_blocks())
            .map(|index| {
                self.block_joint(index)
                    .kv_speaker(&state, seq * batch, batch)
            })
            .collect()
    }

    /// Builds the latent-prefix KV caches from the prefix latent
    /// `[rows, latent]` (`get_kv_cache_latent`); the caller passes the
    /// tripled batch.
    pub fn get_kv_cache_latent(
        &self,
        prefix_latent: &[f32],
        batch: usize,
        rows: usize,
    ) -> Result<Vec<KvCache>> {
        let (encoder, norm) = self
            .latent_stack()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "latent-prefix modules are disabled in this checkpoint".into(),
            })?;
        if rows == 0 {
            let heads = self.config.speaker_num_heads;
            let dim = self.config.speaker_model_size / heads;
            return Ok((0..self.num_blocks())
                .map(|_| KvCache::empty(batch, heads, dim))
                .collect());
        }
        let latent_dim = self.config.latent_size;
        let mut state = encoder.forward(prefix_latent, batch, rows, latent_dim);
        norm.apply(&mut state);
        let seq = state.len() / (batch * self.config.speaker_model_size);
        let max_pos = seq * self.config.speaker_patch_size;
        let freqs = precompute_freqs_cis(self.head_dim, max_pos, 10000.0);
        (0..self.num_blocks())
            .map(|index| {
                self.block_joint(index)
                    .kv_latent(&state, batch, seq, &freqs)
            })
            .collect()
    }

    /// The conditional denoiser pass. `x` is `[batch, seq, latent]`
    /// flat; `t` carries one timestep per batch row; the KV caches are
    /// per decoder block.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        t: &[f32],
        text_mask: &[bool],
        speaker_mask: &[bool],
        kv_text: &[KvCache],
        kv_speaker: &[KvCache],
        start_pos: usize,
        kv_latent: Option<&[KvCache]>,
    ) -> Result<Vec<f32>> {
        let batch = t.len();
        let seq = x.len() / (batch * self.config.latent_size);
        // Subsample the per-sample speaker mask by the patch size
        // (reference `speaker_mask[..., ::patch]`).
        let patch = self.config.speaker_patch_size;
        let speaker_mask_patched: Vec<bool> = speaker_mask.iter().step_by(patch).copied().collect();
        let speaker_mask_patched = if speaker_mask_patched.len() * patch == speaker_mask.len() {
            speaker_mask_patched
        } else {
            // The mask already carries the patched granularity (the
            // zero-latent default path builds one flag per patch).
            speaker_mask.to_vec()
        };
        let mut cond = timestep_embedding(t[0], self.config.timestep_embed_size);
        cond = self.cond_module[0].forward(&cond, 1);
        for value in &mut cond {
            *value *= sigmoid(*value);
        }
        cond = self.cond_module[1].forward(&cond, 1);
        for value in &mut cond {
            *value *= sigmoid(*value);
        }
        let cond = self.cond_module[2].forward(&cond, 1);

        let mut h = self.in_proj.forward(x, batch * seq);
        let freqs = precompute_freqs_cis(self.head_dim, start_pos + seq, 10000.0);
        for (index, block) in self.blocks.iter().enumerate() {
            let (normed, gates) = block.attention_adaln.apply(&h, batch * seq, &cond);
            let attended = block.attention.forward(
                &normed,
                batch,
                seq,
                text_mask,
                &speaker_mask_patched,
                &freqs,
                &kv_text[index],
                &kv_speaker[index],
                start_pos,
                kv_latent.map(|c| &c[index]),
            );
            for (value, (add, gate)) in h.iter_mut().zip(attended.iter().zip(&gates)) {
                *value += add * gate;
            }
            let (normed, gates) = block.mlp_adaln.apply(&h, batch * seq, &cond);
            let fed = block.mlp.forward(&normed, batch * seq);
            for (value, (add, gate)) in h.iter_mut().zip(fed.iter().zip(&gates)) {
                *value += add * gate;
            }
        }
        self.out_norm.apply(&mut h);
        Ok(self.out_proj.forward(&h, batch * seq))
    }

    pub fn latent_size(&self) -> usize {
        self.config.latent_size
    }

    pub fn model_size(&self) -> usize {
        self.config.model_size
    }

    pub fn patch_size(&self) -> usize {
        self.config.speaker_patch_size
    }

    /// Whether the latent-prefix modules are present.
    pub fn blockwise_enabled(&self) -> bool {
        self.latent_encoder.is_some()
    }
}
