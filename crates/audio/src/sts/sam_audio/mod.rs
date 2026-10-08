//! SAM-Audio text-guided source separation.
//!
//! Port of `mlx_audio/sts/models/sam_audio` at reference commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. A DACVAE codec lifts the
//! waveform to codebook-space means; a T5 encoder embeds the text
//! prompt; a DiT with adaLN modulation, QK-normed attention
//! (SAM-Audio's head-dim-major reshape), adjacent-pair RoPE, and
//! cross-attention to the projected text memory predicts a velocity
//! field; a midpoint ODE integrates from noise to the separated
//! target/residual features, which the codec decodes back to audio.
//!
//! Layering: the T5 tokenizer stays outside this crate. The encode API
//! takes token ids and an attention mask, matching the reference where
//! the checkpoint ships no T5 weights and the tokenizer comes from
//! HuggingFace at runtime.

use std::path::Path;

use crate::codec::dacvae::{Dacvae, DacvaeConfig};
use crate::error::SpeechError;
use crate::ops;
use turbospark_model_io::safetensors::SafetensorsFile;

type Result<T> = std::result::Result<T, SpeechError>;

const ROPE_THETA_BASE: f32 = 10000.0;

/// Nested configuration, matching `SAMAudioConfig`.
#[derive(Debug, Clone)]
pub struct SamAudioConfig {
    pub in_channels: usize,
    pub audio_codec: serde_json::Value,
    pub text_encoder: T5EncoderConfig,
    pub transformer: TransformerConfig,
    pub num_anchors: usize,
    pub anchor_embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct T5EncoderConfig {
    pub dim: usize,
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub is_gated_act: bool,
    pub layer_norm_epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub norm_eps: f32,
    pub qk_norm: bool,
    pub fc_bias: bool,
    pub ffn_exp: usize,
    pub ffn_dim_multiplier: usize,
    pub multiple_of: usize,
    pub use_rope: bool,
    pub max_positions: usize,
    pub frequency_embedding_dim: usize,
    pub t_block_bias: bool,
    pub context_dim: usize,
    pub context_norm: bool,
    pub out_channels: usize,
}

impl SamAudioConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| SpeechError::BadConfig {
            field: "config".to_string(),
            why: "expected a JSON object".to_string(),
        })?;
        let num = |o: &serde_json::Map<String, serde_json::Value>, name: &str, default: usize| {
            o.get(name)
                .and_then(|v| v.as_u64())
                .unwrap_or(default as u64) as usize
        };
        let flo = |o: &serde_json::Map<String, serde_json::Value>, name: &str, default: f32| {
            o.get(name)
                .and_then(|v| v.as_f64())
                .unwrap_or(default as f64) as f32
        };
        let boolean =
            |o: &serde_json::Map<String, serde_json::Value>, name: &str, default: bool| {
                o.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
            };
        let t5 = obj
            .get("t5_arch")
            .or_else(|| obj.get("text_encoder"))
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let tr = obj
            .get("transformer")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let text_encoder = T5EncoderConfig {
            dim: num(
                &t5,
                "dim",
                obj.get("text_encoder")
                    .and_then(|v| v.get("dim"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(768) as usize,
            ),
            vocab_size: num(&t5, "vocab_size", 32128),
            d_model: num(&t5, "d_model", 768),
            d_kv: num(&t5, "d_kv", 64),
            d_ff: num(&t5, "d_ff", 3072),
            num_layers: num(&t5, "num_layers", 12),
            num_heads: num(&t5, "num_heads", 12),
            relative_attention_num_buckets: num(&t5, "relative_attention_num_buckets", 32),
            relative_attention_max_distance: num(&t5, "relative_attention_max_distance", 128),
            is_gated_act: boolean(&t5, "is_gated_act", true),
            layer_norm_epsilon: flo(&t5, "layer_norm_epsilon", 1e-6),
        };
        let transformer = TransformerConfig {
            dim: num(&tr, "dim", 2816),
            n_heads: num(&tr, "n_heads", 22),
            n_layers: num(&tr, "n_layers", 22),
            norm_eps: flo(&tr, "norm_eps", 1e-5),
            qk_norm: boolean(&tr, "qk_norm", true),
            fc_bias: boolean(&tr, "fc_bias", false),
            ffn_exp: num(&tr, "ffn_exp", 4),
            ffn_dim_multiplier: num(&tr, "ffn_dim_multiplier", 1),
            multiple_of: num(&tr, "multiple_of", 64),
            use_rope: boolean(&tr, "use_rope", true),
            max_positions: num(&tr, "max_positions", 10000),
            frequency_embedding_dim: num(&tr, "frequency_embedding_dim", 256),
            t_block_bias: boolean(&tr, "t_block_bias", true),
            context_dim: num(&tr, "context_dim", 2816),
            context_norm: boolean(&tr, "context_norm", false),
            out_channels: num(&tr, "out_channels", 256),
        };
        Ok(Self {
            in_channels: num(obj, "in_channels", 768),
            audio_codec: obj
                .get("audio_codec")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
            text_encoder,
            transformer,
            num_anchors: num(obj, "num_anchors", 3),
            anchor_embedding_dim: num(obj, "anchor_embedding_dim", 128),
        })
    }
}

/// Root-mean-square norm with additive eps inside rsqrt.
struct Rms {
    weight: Vec<f32>,
    eps: f32,
}

impl Rms {
    fn run(&self, x: &mut [f32]) {
        for row in x.chunks_exact_mut(self.weight.len()) {
            let mut mean_sq = 0.0f32;
            for value in row.iter() {
                mean_sq += value * value;
            }
            mean_sq /= self.weight.len() as f32;
            let f = (mean_sq + self.eps).sqrt().recip();
            for (value, &weight) in row.iter_mut().zip(&self.weight) {
                *value *= f * weight;
            }
        }
    }
}

struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    fn run(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.in_dim,
            self.out_dim,
        )
    }
}

/// SwiGLU or SiLU projection triple (`w1`, optional `w3`, `w2`).
struct ProjectionLayer {
    w1: Linear,
    w2: Linear,
    w3: Option<Linear>,
}

impl ProjectionLayer {
    fn run(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut hidden = self.w1.run(x, rows);
        match &self.w3 {
            Some(w3) => {
                let gate = w3.run(x, rows);
                ops::silu(&mut hidden);
                for (h, g) in hidden.iter_mut().zip(gate) {
                    *h *= g;
                }
            }
            None => ops::silu(&mut hidden),
        }
        self.w2.run(&hidden, rows)
    }
}

/// The DiT attention: head-dim-major reshape (SAM-Audio order),
/// optional QK RMSNorm, adjacent-pair RoPE, optional key padding mask.
struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: Option<Rms>,
    k_norm: Option<Rms>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// `x` is `[T, dim]` (batch 1). `cross` supplies K/V sources.
    /// `mask` is `[kv_len]` (1 = attend). Returns `[T, dim]`.
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        cross: Option<(&[f32], usize)>,
        mask: Option<&[f32]>,
        cos: Option<&[f32]>,
        sin: Option<&[f32]>,
    ) -> Vec<f32> {
        let dim = self.wq.in_dim;
        let (kv, kv_len) = match cross {
            Some((cx, len)) => (cx, len),
            None => (x, seq),
        };
        let xq = self.wq.run(x, seq);
        let xk = self.wk.run(kv, kv_len);
        let xv = self.wv.run(kv, kv_len);

        // Non-standard reshape: projection output (T, n_heads * head_dim)
        // is read as (T, head_dim, n_heads), so head h takes strided
        // features h, h + n_heads, h + 2 n_heads, ...
        let mut q = vec![0.0f32; seq * self.n_heads * self.head_dim];
        let mut k = vec![0.0f32; kv_len * self.n_kv_heads * self.head_dim];
        let mut v = vec![0.0f32; kv_len * self.n_kv_heads * self.head_dim];
        for t in 0..seq {
            for d in 0..self.head_dim {
                for h in 0..self.n_heads {
                    q[(t * self.n_heads + h) * self.head_dim + d] =
                        xq[t * dim + d * self.n_heads + h];
                }
            }
        }
        for t in 0..kv_len {
            for d in 0..self.head_dim {
                for h in 0..self.n_kv_heads {
                    k[(t * self.n_kv_heads + h) * self.head_dim + d] =
                        xk[t * dim + d * self.n_kv_heads + h];
                    v[(t * self.n_kv_heads + h) * self.head_dim + d] =
                        xv[t * dim + d * self.n_kv_heads + h];
                }
            }
        }

        if let Some(qn) = &self.q_norm {
            qn.run(&mut q);
        }
        if let Some(kn) = &self.k_norm {
            kn.run(&mut k);
        }

        // Adjacent-pair RoPE over (t, h, head_dim) rows.
        if let (Some(cos), Some(sin)) = (cos, sin) {
            let half = self.head_dim / 2;
            for t in 0..seq {
                for h in 0..self.n_heads {
                    let base = (t * self.n_heads + h) * self.head_dim;
                    for i in 0..half {
                        let c = cos[t * half + i];
                        let s = sin[t * half + i];
                        let x0 = q[base + 2 * i];
                        let x1 = q[base + 2 * i + 1];
                        q[base + 2 * i] = x0 * c - x1 * s;
                        q[base + 2 * i + 1] = x1 * c + x0 * s;
                    }
                }
            }
            for t in 0..kv_len {
                for h in 0..self.n_kv_heads {
                    let base = (t * self.n_kv_heads + h) * self.head_dim;
                    for i in 0..half {
                        let c = cos[t * half + i];
                        let s = sin[t * half + i];
                        let x0 = k[base + 2 * i];
                        let x1 = k[base + 2 * i + 1];
                        k[base + 2 * i] = x0 * c - x1 * s;
                        k[base + 2 * i + 1] = x1 * c + x0 * s;
                    }
                }
            }
        }

        let scale = (self.head_dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; seq * self.n_heads * self.head_dim];
        for h in 0..self.n_heads {
            let kv_h = h.min(self.n_kv_heads - 1);
            for t in 0..seq {
                let qbase = (t * self.n_heads + h) * self.head_dim;
                let mut scores = vec![0.0f32; kv_len];
                let mut max_score = f32::NEG_INFINITY;
                for u in 0..kv_len {
                    let kbase = (u * self.n_kv_heads + kv_h) * self.head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[qbase + d] * k[kbase + d];
                    }
                    let mut score = dot * scale;
                    if let Some(m) = mask {
                        if m[u] == 0.0 {
                            score = f32::NEG_INFINITY;
                        }
                    }
                    scores[u] = score;
                    max_score = max_score.max(score);
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max_score).exp();
                    sum += *score;
                }
                for score in scores.iter_mut() {
                    *score /= sum;
                }
                let obase = qbase;
                for (u, &weight) in scores.iter().enumerate() {
                    let vbase = (u * self.n_kv_heads + kv_h) * self.head_dim;
                    for d in 0..self.head_dim {
                        out[obase + d] += weight * v[vbase + d];
                    }
                }
            }
        }

        // Merge heads back with the same strided layout.
        let mut merged = vec![0.0f32; seq * dim];
        for t in 0..seq {
            for d in 0..self.head_dim {
                for h in 0..self.n_heads {
                    merged[t * dim + d * self.n_heads + h] =
                        out[(t * self.n_heads + h) * self.head_dim + d];
                }
            }
        }
        self.wo.run(&merged, seq)
    }
}

fn swiglu_hidden_dim(dim: usize, ffn_exp: usize, multiplier: usize, multiple_of: usize) -> usize {
    let hidden = 2 * dim * ffn_exp / 3;
    let hidden = multiplier * hidden;
    multiple_of * hidden.div_ceil(multiple_of)
}

struct DiTBlock {
    attention: Attention,
    cross_attention: Attention,
    feed_forward: ProjectionLayer,
    attention_norm: Rms,
    ffn_norm: Rms,
    scale_shift_table: Vec<f32>,
    dim: usize,
}

impl DiTBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        cross_x: Option<(&[f32], usize)>,
        t0: &[f32],
        padding_mask: Option<&[f32]>,
        memory_padding_mask: Option<&[f32]>,
        cos: Option<&[f32]>,
        sin: Option<&[f32]>,
    ) -> Vec<f32> {
        // Modulation: scale_shift_table [6, dim] + t0 [6, dim].
        let mut biases = vec![0.0f32; 6 * self.dim];
        for i in 0..6 * self.dim {
            biases[i] = self.scale_shift_table[i] + t0[i];
        }
        let seg = |idx: usize| &biases[idx * self.dim..(idx + 1) * self.dim];
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            (seg(0), seg(1), seg(2), seg(3), seg(4), seg(5));

        let mut h_normed = x.to_vec();
        self.attention_norm.run(&mut h_normed);
        for i in 0..seq * self.dim {
            h_normed[i] = h_normed[i] * (1.0 + scale_msa[i % self.dim]) + shift_msa[i % self.dim];
        }
        let h_attn = self
            .attention
            .forward(&h_normed, seq, None, padding_mask, cos, sin);
        let mut h = x.to_vec();
        for (hv, (av, g)) in h.iter_mut().zip(h_attn.iter().zip(gate_msa.iter().cycle())) {
            *hv += av * g;
        }

        if let Some((cx, kv_len)) = cross_x {
            let h_cross = self.cross_attention.forward(
                &h,
                seq,
                Some((cx, kv_len)),
                memory_padding_mask,
                None,
                None,
            );
            for (hv, cv) in h.iter_mut().zip(h_cross) {
                *hv += cv;
            }
        }

        let mut h_normed = h.clone();
        self.ffn_norm.run(&mut h_normed);
        for i in 0..seq * self.dim {
            h_normed[i] = h_normed[i] * (1.0 + scale_mlp[i % self.dim]) + shift_mlp[i % self.dim];
        }
        let h_ff = self.feed_forward.run(&h_normed, seq);
        for (hv, (fv, g)) in h.iter_mut().zip(h_ff.iter().zip(gate_mlp.iter().cycle())) {
            *hv += fv * g;
        }
        h
    }
}

/// GroupNorm with one group over (channels, length) jointly and
/// per-channel affine.
struct GroupNorm1 {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl GroupNorm1 {
    fn run(&self, x: &mut [f32], channels: usize, length: usize) {
        let count = (channels * length) as f32;
        let mut mean = 0.0f32;
        for value in x.iter() {
            mean += value;
        }
        mean /= count;
        let mut var = 0.0f32;
        for value in x.iter() {
            let d = value - mean;
            var += d * d;
        }
        var /= count;
        let denom = (var + 1e-5).sqrt();
        for value in x.iter_mut() {
            *value = (*value - mean) / denom;
        }
        for c in 0..channels {
            for t in 0..length {
                x[c * length + t] = x[c * length + t] * self.weight[c] + self.bias[c];
            }
        }
    }
}

struct PatcherResnet {
    /// GroupNorm(1 group, affine), SiLU, then Conv1d k=3 pad 1.
    blocks: [(GroupNorm1, Linear); 2],
    to_out: Option<Linear>,
    channels: usize,
}

impl PatcherResnet {
    /// `x` is `[channels, length]`. Returns `[channels, length]`.
    fn forward(&self, x: &[f32], length: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        for (norm, conv) in &self.blocks {
            let out_ch = conv.out_dim;
            norm.run(&mut h, self.channels, length);
            ops::silu(&mut h);
            // Conv1d k=3, stride 1, symmetric pad 1 over channels-first
            // [in, length]; weight rows are [out][k][in].
            let in_ch = self.channels;
            let mut next = vec![0.0f32; out_ch * length];
            for o in 0..out_ch {
                for t in 0..length {
                    let mut acc = conv.bias.as_ref().map(|b| b[o]).unwrap_or(0.0);
                    for k in 0..3usize {
                        let src = t as isize + k as isize - 1;
                        if src < 0 || src as usize >= length {
                            continue;
                        }
                        let w_row = o * in_ch * 3 + k * in_ch;
                        for c in 0..in_ch {
                            acc += h[c * length + src as usize] * conv.weight[w_row + c];
                        }
                    }
                    next[o * length + t] = acc;
                }
            }
            h = next;
        }
        match &self.to_out {
            Some(proj) => {
                for o in 0..self.channels {
                    for t in 0..length {
                        let mut acc = proj.bias.as_ref().map(|b| b[o]).unwrap_or(0.0);
                        for c in 0..self.channels {
                            acc += x[c * length + t] * proj.weight[o * self.channels + c];
                        }
                        h[o * length + t] += acc;
                    }
                }
            }
            None => {
                for (hv, xv) in h.iter_mut().zip(x) {
                    *hv += xv;
                }
            }
        }
        h
    }
}

/// The diffusion transformer.
pub struct DiT {
    x_embedder: PatcherResnet,
    layers: Vec<DiTBlock>,
    norm: Rms,
    output: Linear,
    y_embedder: ProjectionLayer,
    t_embedder: ProjectionLayer,
    t_block: Linear,
    final_layer_scale_shift_table: Vec<f32>,
    dim: usize,
    head_dim: usize,
    freq_embedding_dim: usize,
    /// exp(-ln(10000) * i / half) per frequency slot.
    t_freqs: Vec<f32>,
}

impl DiT {
    fn timestep_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.freq_embedding_dim / 2;
        let mut emb = vec![0.0f32; self.freq_embedding_dim];
        for i in 0..half {
            let arg = t * self.t_freqs[i];
            emb[i] = arg.cos();
            emb[half + i] = arg.sin();
        }
        emb
    }

    fn rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim / 2;
        let mut cos = vec![0.0f32; seq * half];
        let mut sin = vec![0.0f32; seq * half];
        let theta = ROPE_THETA_BASE.max(2.0 * 1024.0);
        for t in 0..seq {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / self.head_dim as f32);
                let ang = t as f32 * freq;
                cos[t * half + i] = ang.cos();
                sin[t * half + i] = ang.sin();
            }
        }
        (cos, sin)
    }

    /// `x` is `[seq, in_dim]`; returns `[seq, out_channels]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        time: f32,
        padding_mask: Option<&[f32]>,
        memory: Option<(&[f32], usize)>,
        memory_padding_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        // Patcher over channels-first layout.
        let mut cf = vec![0.0f32; self.dim * seq];
        for t in 0..seq {
            for c in 0..self.dim {
                cf[c * seq + t] = x[t * self.dim + c];
            }
        }
        let patched = self.x_embedder.forward(&cf, seq);
        let mut h = vec![0.0f32; self.dim * seq];
        for t in 0..seq {
            for c in 0..self.dim {
                h[t * self.dim + c] = patched[c * seq + t];
            }
        }

        // Timestep embedding path.
        let t_emb = self.timestep_embedding(time);
        let t = self.t_embedder.run(&t_emb, 1);
        let mut t0 = t.clone();
        ops::silu(&mut t0);
        let t0 = self.t_block.run(&t0, 1);

        // Context embedding.
        let y = memory.map(|(m, len)| {
            let projected = self.y_embedder.run(m, len);
            (projected, len)
        });

        let (cos, sin) = self.rope_tables(seq);
        for layer in &self.layers {
            h = layer.forward(
                &h,
                seq,
                y.as_ref().map(|(m, len)| (m.as_slice(), *len)),
                &t0,
                padding_mask,
                memory_padding_mask,
                Some(&cos),
                Some(&sin),
            );
        }

        // Final modulation: table [2, dim] + projected timestep t.
        let mut shift = vec![0.0f32; self.dim];
        let mut scale = vec![0.0f32; self.dim];
        for d in 0..self.dim {
            shift[d] = self.final_layer_scale_shift_table[d] + t[d];
            scale[d] = self.final_layer_scale_shift_table[self.dim + d] + t[d];
        }
        self.norm.run(&mut h);
        for i in 0..h.len() {
            h[i] = h[i] * (1.0 + scale[i % self.dim]) + shift[i % self.dim];
        }
        self.output.run(&h, seq)
    }
}

/// The T5 encoder (encoder stack only).
pub struct T5Encoder {
    shared: Vec<f32>,
    d_model: usize,
    vocab: usize,
    blocks: Vec<T5Block>,
    final_norm: Vec<f32>,
    eps: f32,
    num_buckets: usize,
    max_distance: usize,
    n_heads: usize,
    d_kv: usize,
}

struct T5Block {
    attn: T5Attention,
    attn_norm: Vec<f32>,
    ffn: T5Ffn,
    ffn_norm: Vec<f32>,
}

struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// Bucket -> head bias table (block 0 only).
    relative_bias: Option<Linear>,
}

struct T5Ffn {
    wi_0: Linear,
    wi_1: Option<Linear>,
    wo: Linear,
}

impl T5Encoder {
    /// Relative position bucket ids, T5 bidirectional formula.
    fn relative_bucket(&self, relative: isize) -> usize {
        let num_buckets = self.num_buckets;
        let mut bucket = 0usize;
        let rp = if relative > 0 {
            bucket = num_buckets / 2;
            relative
        } else {
            relative
        };
        let rp = rp.abs();
        let max_exact = num_buckets / 4;
        if rp < max_exact as isize {
            bucket + rp as usize
        } else {
            let large = max_exact
                + ((rp as f32 / max_exact as f32).ln()
                    / (self.max_distance as f32 / max_exact as f32).ln()
                    * (num_buckets / 2 - max_exact) as f32) as usize;
            bucket + large.min(num_buckets / 2 - 1)
        }
    }

    /// Encodes `[seq]` token ids with a `[seq]` mask (1 = attend).
    /// Returns `[seq, d_model]`.
    pub fn forward(&self, input_ids: &[i32], mask: &[f32]) -> Vec<f32> {
        let seq = input_ids.len();
        let mut h = vec![0.0f32; seq * self.d_model];
        for (t, &id) in input_ids.iter().enumerate() {
            let id = id.clamp(0, self.vocab as i32 - 1) as usize;
            h[t * self.d_model..(t + 1) * self.d_model]
                .copy_from_slice(&self.shared[id * self.d_model..(id + 1) * self.d_model]);
        }
        // Precompute position bias once (block 0 carries the table).
        let bias = self.blocks[0].attn.relative_bias.as_ref().map(|table| {
            let mut bias = vec![0.0f32; seq * seq * self.n_heads];
            for qi in 0..seq {
                for ki in 0..seq {
                    let bucket = self.relative_bucket(ki as isize - qi as isize);
                    for head in 0..self.n_heads {
                        bias[(qi * seq + ki) * self.n_heads + head] =
                            table.weight[bucket * self.n_heads + head];
                    }
                }
            }
            bias
        });
        for block in &self.blocks {
            // Self-attention with pre-norm (T5 RMS, no mean).
            let mut normed = h.clone();
            t5_rms(&mut normed, &block.attn_norm, self.eps);
            let attn_out = block
                .attn
                .forward(&normed, seq, bias.as_deref(), mask, self);
            for (hv, av) in h.iter_mut().zip(attn_out) {
                *hv += av;
            }
            // Gated feed-forward with pre-norm.
            let mut normed = h.clone();
            t5_rms(&mut normed, &block.ffn_norm, self.eps);
            let mut inner = block.ffn.wi_0.run(&normed, seq);
            ops::gelu_erf(&mut inner);
            if let Some(wi_1) = &block.ffn.wi_1 {
                let linear = wi_1.run(&normed, seq);
                for (gv, lv) in inner.iter_mut().zip(linear) {
                    *gv *= lv;
                }
            }
            let out = block.ffn.wo.run(&inner, seq);
            for (hv, ov) in h.iter_mut().zip(out) {
                *hv += ov;
            }
        }
        t5_rms(&mut h, &self.final_norm, self.eps);
        h
    }
}

/// T5 layer norm: RMS scale with weight, eps inside rsqrt.
fn t5_rms(x: &mut [f32], weight: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(weight.len()) {
        let mut var = 0.0f32;
        for value in row.iter() {
            var += value * value;
        }
        var /= weight.len() as f32;
        let f = (var + eps).sqrt().recip();
        for (value, &w) in row.iter_mut().zip(weight) {
            *value *= f * w;
        }
    }
}

impl T5Attention {
    fn forward(
        &self,
        h: &[f32],
        seq: usize,
        bias: Option<&[f32]>,
        mask: &[f32],
        enc: &T5Encoder,
    ) -> Vec<f32> {
        let n_heads = enc.n_heads;
        let d_kv = enc.d_kv;
        let q = self.q.run(h, seq);
        let k = self.k.run(h, seq);
        let v = self.v.run(h, seq);
        let mut out = vec![0.0f32; seq * n_heads * d_kv];
        for head in 0..n_heads {
            for t in 0..seq {
                let qbase = (t * n_heads + head) * d_kv;
                let mut scores = vec![0.0f32; seq];
                let mut max_score = f32::NEG_INFINITY;
                for u in 0..seq {
                    let kbase = (u * n_heads + head) * d_kv;
                    let mut dot = 0.0f32;
                    for d in 0..d_kv {
                        dot += q[qbase + d] * k[kbase + d];
                    }
                    let mut score = dot;
                    if let Some(bias) = bias {
                        score += bias[(t * seq + u) * n_heads + head];
                    }
                    if mask[u] == 0.0 {
                        score = f32::NEG_INFINITY;
                    }
                    scores[u] = score;
                    max_score = max_score.max(score);
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max_score).exp();
                    sum += *score;
                }
                for score in scores.iter_mut() {
                    *score /= sum;
                }
                for (u, &weight) in scores.iter().enumerate() {
                    let vbase = (u * n_heads + head) * d_kv;
                    for d in 0..d_kv {
                        out[qbase + d] += weight * v[vbase + d];
                    }
                }
            }
        }
        self.o.run(&out, seq)
    }
}

/// Anchor embedding: `proj(embed(alignment_ids)) * gate` added to the
/// aligned inputs.
struct EmbedAnchors {
    embed: Vec<f32>,
    embedding_dim: usize,
    proj: Linear,
    gate: f32,
}

impl EmbedAnchors {
    /// `anchor_ids` is `[num_anchors]`, `alignment` `[seq]` (0 = no
    /// anchor when the id slot is the padding entry).
    fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        anchor_ids: Option<&[i32]>,
        alignment: Option<&[i32]>,
    ) {
        let (Some(ids), Some(alignment)) = (anchor_ids, alignment) else {
            return;
        };
        let out_dim = self.proj.out_dim;
        for t in 0..seq {
            let slot = alignment[t];
            if slot < 0 || slot as usize >= ids.len() {
                continue;
            }
            let id = ids[slot as usize]
                .clamp(0, self.embed.len() as i32 / self.embedding_dim as i32 - 1)
                as usize;
            let projected = self.proj.run(
                &self.embed[id * self.embedding_dim..(id + 1) * self.embedding_dim],
                1,
            );
            for (d, pv) in projected.into_iter().enumerate() {
                x[t * out_dim + d] += pv * self.gate;
            }
        }
    }
}

/// The loaded SAM-Audio model.
pub struct SamAudio {
    pub config: SamAudioConfig,
    codec: Dacvae,
    text_encoder: T5Encoder,
    transformer: DiT,
    proj: Linear,
    embed_anchors: EmbedAnchors,
    memory_proj: Linear,
    /// exp(-ln(10000) * i / half).
    t_freqs: Vec<f32>,
}

/// ODE solver options (method, step size).
#[derive(Debug, Clone, Copy)]
pub struct OdeOptions {
    pub midpoint: bool,
    pub step_size: f32,
}

impl Default for OdeOptions {
    fn default() -> Self {
        Self {
            midpoint: true,
            step_size: 2.0 / 32.0,
        }
    }
}

impl SamAudio {
    /// Opens a checkpoint directory: `config.json`, the main weights
    /// (`model.safetensors` / `tiny_weights.safetensors`), and the
    /// codec subtree (`tiny_codec.safetensors`, bare DACVAE keys; the
    /// reference ships the codec under an `audio_codec.` prefix in a
    /// single file, which the runtime layer splits out).
    pub fn open(dir: &Path) -> Result<Self> {
        let config =
            SamAudioConfig::from_json(&crate::quant::read_json(&dir.join("config.json"))?)?;
        let weights = if dir.join("model.safetensors").exists() {
            dir.join("model.safetensors")
        } else {
            dir.join("tiny_weights.safetensors")
        };
        let codec_path = dir.join("tiny_codec.safetensors");
        let file = SafetensorsFile::open(&weights).map_err(|e| SpeechError::BadConfig {
            field: weights.display().to_string(),
            why: e.to_string(),
        })?;
        let codec_file =
            SafetensorsFile::open(&codec_path).map_err(|e| SpeechError::BadConfig {
                field: codec_path.display().to_string(),
                why: e.to_string(),
            })?;
        Self::load(config, &file, &codec_file)
    }

    /// Loads from an open main weights file and a bare-key codec file.
    pub fn load(
        config: SamAudioConfig,
        file: &SafetensorsFile,
        codec_file: &SafetensorsFile,
    ) -> Result<Self> {
        let f32_tensor = |name: &str| -> Result<Vec<f32>> {
            file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
                name: name.to_string(),
                why: e.to_string(),
            })
        };
        let linear = |name: &str, bias: bool| -> Result<Linear> {
            let weight = f32_tensor(&format!("{name}.weight"))?;
            let bias = if bias && file.contains_tensor(&format!("{name}.bias")) {
                Some(f32_tensor(&format!("{name}.bias"))?)
            } else {
                None
            };
            let desc =
                file.descriptor(&format!("{name}.weight"))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: name.to_string(),
                        why: "weight descriptor missing".to_string(),
                    })?;
            if desc.shape.len() != 2 {
                return Err(SpeechError::Tensor {
                    name: name.to_string(),
                    why: format!("expected 2-D weight, got {:?}", desc.shape),
                });
            }
            Ok(Linear {
                weight,
                bias,
                out_dim: desc.shape[0],
                in_dim: desc.shape[1],
            })
        };
        let rms = |name: &str, _dim: usize, eps: f32| -> Result<Rms> {
            Ok(Rms {
                weight: f32_tensor(&format!("{name}.weight"))?,
                eps,
            })
        };
        let projection = |name: &str, swiglu: bool, fc_bias: bool| -> Result<ProjectionLayer> {
            let w1 = linear(&format!("{name}.w1"), fc_bias)?;
            let w2 = linear(&format!("{name}.w2"), fc_bias)?;
            let w3 = if swiglu {
                Some(linear(&format!("{name}.w3"), fc_bias)?)
            } else {
                None
            };
            Ok(ProjectionLayer { w1, w2, w3 })
        };

        let tc = &config.transformer;
        let dim = tc.dim;
        let head_dim = dim / tc.n_heads;

        // Codec.
        let codec_config = match &config.audio_codec {
            serde_json::Value::Null => DacvaeConfig::from_json(&serde_json::Value::Null)?,
            v => DacvaeConfig::from_json(v)?,
        };
        let codec = Dacvae::load(codec_config, codec_file)?;

        // T5 encoder.
        let t5c = &config.text_encoder;
        let t5_linear = |name: &str| -> Result<Linear> {
            let weight = f32_tensor(&format!("{name}.weight"))?;
            let desc =
                file.descriptor(&format!("{name}.weight"))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: name.to_string(),
                        why: "weight descriptor missing".to_string(),
                    })?;
            Ok(Linear {
                weight,
                bias: None,
                out_dim: desc.shape[0],
                in_dim: desc.shape[1],
            })
        };
        let t5_norm = |name: &str| -> Result<Vec<f32>> { f32_tensor(&format!("{name}.weight")) };
        let mut t5_blocks = Vec::with_capacity(t5c.num_layers);
        for i in 0..t5c.num_layers {
            let attn_prefix = format!("text_encoder.model.encoder.block.{i}.layer.0.SelfAttention");
            let ffn_prefix = format!("text_encoder.model.encoder.block.{i}.layer.1.DenseReluDense");
            let relative_bias = if i == 0 {
                Some(t5_linear(&format!(
                    "{attn_prefix}.relative_attention_bias"
                ))?)
            } else {
                None
            };
            t5_blocks.push(T5Block {
                attn_norm: t5_norm(&format!(
                    "text_encoder.model.encoder.block.{i}.layer.0.layer_norm"
                ))?,
                ffn_norm: t5_norm(&format!(
                    "text_encoder.model.encoder.block.{i}.layer.1.layer_norm"
                ))?,
                attn: T5Attention {
                    q: t5_linear(&format!("{attn_prefix}.q"))?,
                    k: t5_linear(&format!("{attn_prefix}.k"))?,
                    v: t5_linear(&format!("{attn_prefix}.v"))?,
                    o: t5_linear(&format!("{attn_prefix}.o"))?,
                    relative_bias,
                },
                ffn: T5Ffn {
                    wi_0: t5_linear(&format!("{ffn_prefix}.wi_0"))?,
                    wi_1: if t5c.is_gated_act {
                        Some(t5_linear(&format!("{ffn_prefix}.wi_1"))?)
                    } else {
                        None
                    },
                    wo: t5_linear(&format!("{ffn_prefix}.wo"))?,
                },
            });
        }
        let text_encoder = T5Encoder {
            shared: f32_tensor("text_encoder.model.shared.weight")?,
            d_model: t5c.d_model,
            vocab: t5c.vocab_size,
            blocks: t5_blocks,
            final_norm: f32_tensor("text_encoder.model.encoder.final_layer_norm.weight")?,
            eps: t5c.layer_norm_epsilon,
            num_buckets: t5c.relative_attention_num_buckets,
            max_distance: t5c.relative_attention_max_distance,
            n_heads: t5c.num_heads,
            d_kv: t5c.d_kv,
        };

        // DiT.
        let swiglu = true;
        let mut layers = Vec::with_capacity(tc.n_layers);
        for i in 0..tc.n_layers {
            let lp = format!("transformer.layers.{i}");
            let attention_norm = rms(&format!("{lp}.attention_norm"), dim, tc.norm_eps)?;
            let ffn_norm = rms(&format!("{lp}.ffn_norm"), dim, tc.norm_eps)?;
            let attention = Attention {
                wq: linear(&format!("{lp}.attention.wq"), tc.fc_bias)?,
                wk: linear(&format!("{lp}.attention.wk"), tc.fc_bias)?,
                wv: linear(&format!("{lp}.attention.wv"), tc.fc_bias)?,
                wo: linear(&format!("{lp}.attention.wo"), tc.fc_bias)?,
                q_norm: if tc.qk_norm {
                    Some(rms(
                        &format!("{lp}.attention.q_norm"),
                        head_dim,
                        tc.norm_eps,
                    )?)
                } else {
                    None
                },
                k_norm: if tc.qk_norm {
                    Some(rms(
                        &format!("{lp}.attention.k_norm"),
                        head_dim,
                        tc.norm_eps,
                    )?)
                } else {
                    None
                },
                n_heads: tc.n_heads,
                n_kv_heads: tc.n_heads,
                head_dim,
            };
            let cross_attention = Attention {
                wq: linear(&format!("{lp}.cross_attention.wq"), tc.fc_bias)?,
                wk: linear(&format!("{lp}.cross_attention.wk"), tc.fc_bias)?,
                wv: linear(&format!("{lp}.cross_attention.wv"), tc.fc_bias)?,
                wo: linear(&format!("{lp}.cross_attention.wo"), tc.fc_bias)?,
                q_norm: if tc.qk_norm {
                    Some(rms(
                        &format!("{lp}.cross_attention.q_norm"),
                        head_dim,
                        tc.norm_eps,
                    )?)
                } else {
                    None
                },
                k_norm: if tc.qk_norm {
                    Some(rms(
                        &format!("{lp}.cross_attention.k_norm"),
                        head_dim,
                        tc.norm_eps,
                    )?)
                } else {
                    None
                },
                n_heads: tc.n_heads,
                n_kv_heads: tc.n_heads,
                head_dim,
            };
            let hidden = swiglu_hidden_dim(dim, tc.ffn_exp, tc.ffn_dim_multiplier, tc.multiple_of);
            let feed_forward = ProjectionLayer {
                w1: linear(&format!("{lp}.feed_forward.w1"), tc.fc_bias)?,
                w2: linear(&format!("{lp}.feed_forward.w2"), tc.fc_bias)?,
                w3: if swiglu {
                    Some(linear(&format!("{lp}.feed_forward.w3"), tc.fc_bias)?)
                } else {
                    None
                },
            };
            if feed_forward.w1.out_dim != hidden {
                return Err(SpeechError::Tensor {
                    name: format!("{lp}.feed_forward.w1"),
                    why: format!(
                        "hidden dim {} != computed {hidden}",
                        feed_forward.w1.out_dim
                    ),
                });
            }
            layers.push(DiTBlock {
                attention,
                cross_attention,
                feed_forward,
                attention_norm,
                ffn_norm,
                scale_shift_table: f32_tensor(&format!("{lp}.scale_shift_table"))?,
                dim,
            });
        }

        // Patcher ResnetBlock1d(dim -> dim, num_groups=1).
        let group = |name: &str| -> Result<GroupNorm1> {
            Ok(GroupNorm1 {
                weight: f32_tensor(&format!("{name}.weight"))?,
                bias: f32_tensor(&format!("{name}.bias"))?,
            })
        };
        let conv = |name: &str| -> Result<Linear> {
            // Patcher Conv1d weight is (out, k, in); flatten to the
            // [out][k][in] layout the block kernel walks.
            let weight = f32_tensor(&format!("{name}.weight"))?;
            let bias = if file.contains_tensor(&format!("{name}.bias")) {
                Some(f32_tensor(&format!("{name}.bias"))?)
            } else {
                None
            };
            let desc =
                file.descriptor(&format!("{name}.weight"))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: name.to_string(),
                        why: "weight descriptor missing".to_string(),
                    })?;
            if desc.shape.len() != 3 {
                return Err(SpeechError::Tensor {
                    name: name.to_string(),
                    why: format!("expected 3-D conv weight, got {:?}", desc.shape),
                });
            }
            let (out, k, input) = (desc.shape[0], desc.shape[1], desc.shape[2]);
            Ok(Linear {
                weight,
                bias,
                out_dim: out,
                in_dim: input * k,
            })
        };
        let block1_norm = group("transformer.x_embedder.block.block1.groupnorm")?;
        let block1_conv = conv("transformer.x_embedder.block.block1.project")?;
        let block2_norm = group("transformer.x_embedder.block.block2.groupnorm")?;
        let block2_conv = conv("transformer.x_embedder.block.block2.project")?;
        let to_out = if file.contains_tensor("transformer.x_embedder.block.to_out.weight") {
            Some(conv("transformer.x_embedder.block.to_out")?)
        } else {
            None
        };
        let x_embedder = PatcherResnet {
            blocks: [(block1_norm, block1_conv), (block2_norm, block2_conv)],
            to_out,
            channels: dim,
        };

        let t_embedder = projection("transformer.t_embedder.projection", swiglu, tc.fc_bias)?;
        if t_embedder.w1.in_dim != tc.frequency_embedding_dim {
            return Err(SpeechError::BadConfig {
                field: "frequency_embedding_dim".to_string(),
                why: format!("t embedder expects {}", t_embedder.w1.in_dim),
            });
        }
        let transformer = DiT {
            x_embedder,
            layers,
            norm: rms("transformer.norm", dim, tc.norm_eps)?,
            output: linear("transformer.output", tc.fc_bias)?,
            y_embedder: projection("transformer.y_embedder.projection", swiglu, tc.fc_bias)?,
            t_embedder,
            t_block: linear("transformer.t_block", tc.t_block_bias)?,
            final_layer_scale_shift_table: f32_tensor("transformer.final_layer_scale_shift_table")?,
            dim,
            head_dim,
            freq_embedding_dim: tc.frequency_embedding_dim,
            t_freqs: (0..tc.frequency_embedding_dim / 2)
                .map(|i| {
                    (-ROPE_THETA_BASE.ln() * i as f32 / (tc.frequency_embedding_dim / 2) as f32)
                        .exp()
                })
                .collect(),
        };

        // Top-level projections.
        let half_freq = tc.frequency_embedding_dim / 2;
        let _ = half_freq;
        let sam_proj = linear("proj", true)?;
        if sam_proj.in_dim != config.in_channels {
            return Err(SpeechError::BadConfig {
                field: "in_channels".to_string(),
                why: format!("proj expects {}", sam_proj.in_dim),
            });
        }
        let embed_anchors = EmbedAnchors {
            embed: f32_tensor("embed_anchors.embed.weight")?,
            embedding_dim: config.anchor_embedding_dim,
            proj: linear("embed_anchors.proj", false)?,
            gate: f32_tensor("embed_anchors.gate")?[0],
        };
        let memory_proj = linear("memory_proj", true)?;
        if memory_proj.in_dim != config.text_encoder.dim {
            return Err(SpeechError::BadConfig {
                field: "text_encoder.dim".to_string(),
                why: format!("memory_proj expects {}", memory_proj.in_dim),
            });
        }
        // Model-level timestep sinusoid spans transformer.dim
        // (SinusoidalEmbedding(config.transformer.dim)).
        let t_freqs = (0..dim / 2)
            .map(|i| (-ROPE_THETA_BASE.ln() * i as f32 / (dim / 2) as f32).exp())
            .collect();
        Ok(Self {
            config,
            codec,
            text_encoder,
            transformer,
            proj: sam_proj,
            embed_anchors,
            memory_proj,
            t_freqs,
        })
    }

    /// Encodes text: token ids + mask through the T5 encoder.
    pub fn encode_text(&self, input_ids: &[i32], mask: &[f32]) -> Vec<f32> {
        self.text_encoder.forward(input_ids, mask)
    }

    /// Codec codebook means `[frames, codebook_dim]` doubled for
    /// target/residual: `[frames, 2 * codebook_dim]`.
    pub fn audio_features(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let feats = self.codec.encode(samples)?;
        // encode returns channel-major [codebook_dim, frames].
        let dim = self.codec.config.codebook_dim;
        let frames = feats.len() / dim;
        let mut out = vec![0.0f32; frames * 2 * dim];
        for f in 0..frames {
            for c in 0..dim {
                let value = feats[c * frames + f];
                out[(f * 2) * dim + c] = value;
                out[(f * 2 + 1) * dim + c] = value;
            }
        }
        Ok(out)
    }

    /// One velocity evaluation. `noisy` and `features` are
    /// `[frames, 2 * codebook_dim]`; `text_features` is
    /// `[text_len, text_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(
        &self,
        noisy: &[f32],
        features: &[f32],
        text_features: &[f32],
        text_len: usize,
        time: f32,
        text_mask: Option<&[f32]>,
        anchor_ids: Option<&[i32]>,
        anchor_alignment: Option<&[i32]>,
    ) -> Vec<f32> {
        let channels = self.codec.config.codebook_dim;
        let frames = features.len() / (2 * channels);
        // Concatenate [noisy, zeros, features] and project.
        let mut concat = vec![0.0f32; frames * self.config.in_channels];
        for f in 0..frames {
            concat[f * self.config.in_channels..f * self.config.in_channels + 2 * channels]
                .copy_from_slice(&noisy[f * 2 * channels..(f + 1) * 2 * channels]);
            concat[(f + 1) * self.config.in_channels - 2 * channels
                ..(f + 1) * self.config.in_channels]
                .copy_from_slice(&features[f * 2 * channels..(f + 1) * 2 * channels]);
        }
        let mut aligned = self.proj.run(&concat, frames);
        self.embed_anchors
            .forward(&mut aligned, frames, anchor_ids, anchor_alignment);

        // Model-level timestep embedding over transformer.dim.
        let dim = self.transformer.dim;
        let half = self.t_freqs.len();
        let mut t_emb = vec![0.0f32; dim];
        for (i, &freq) in self.t_freqs.iter().enumerate() {
            let arg = time * freq;
            t_emb[i] = arg.cos();
            t_emb[half + i] = arg.sin();
        }
        let memory_projected = self.memory_proj.run(text_features, text_len);
        let mut memory = vec![0.0f32; text_len * dim];
        for t in 0..text_len {
            for d in 0..dim {
                memory[t * dim + d] = memory_projected[t * dim + d] + t_emb[d];
            }
        }
        self.transformer.forward(
            &aligned,
            frames,
            time,
            None,
            Some((&memory, text_len)),
            text_mask,
        )
    }

    /// One midpoint ODE step.
    fn ode_step_midpoint(
        &self,
        t: f32,
        dt: f32,
        noisy: &[f32],
        features: &[f32],
        text_features: &[f32],
        text_len: usize,
        text_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let v_t = self.velocity(
            noisy,
            features,
            text_features,
            text_len,
            t,
            text_mask,
            None,
            None,
        );
        let mut midpoint = noisy.to_vec();
        for (m, v) in midpoint.iter_mut().zip(&v_t) {
            *m += 0.5 * dt * v;
        }
        self.velocity(
            &midpoint,
            features,
            text_features,
            text_len,
            t + 0.5 * dt,
            text_mask,
            None,
            None,
        )
    }

    /// Separates mono audio into (target, residual) waveforms. `noise`
    /// is the initial state in feature space
    /// (`[frames, 2 * codebook_dim]`), matching the reference contract
    /// where callers pass seeded noise for determinism.
    #[allow(clippy::too_many_arguments)]
    pub fn separate(
        &self,
        samples: &[f32],
        input_ids: &[i32],
        text_mask: &[f32],
        noise: &[f32],
        ode: OdeOptions,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if ode.step_size <= 0.0 || ode.step_size >= 1.0 {
            return Err(SpeechError::Input {
                why: format!(
                    "step size {} must be between 0 and 1 (exclusive)",
                    ode.step_size
                ),
            });
        }
        let features = self.audio_features(samples)?;
        let text_len = input_ids.len();
        let text_features = self.encode_text(input_ids, text_mask);
        let channels = self.codec.config.codebook_dim;
        let frames = features.len() / (2 * channels);
        if noise.len() != frames * 2 * channels {
            return Err(SpeechError::Input {
                why: format!(
                    "noise length {} != {} feature frames x {}",
                    noise.len(),
                    frames,
                    2 * channels
                ),
            });
        }
        let mut noisy = noise.to_vec();
        let steps = (1.0 / ode.step_size) as usize;
        for i in 0..steps {
            let t = i as f32 * ode.step_size;
            let v_mid = self.ode_step_midpoint(
                t,
                ode.step_size,
                &noisy,
                &features,
                &text_features,
                text_len,
                Some(text_mask),
            );
            for (nv, v) in noisy.iter_mut().zip(v_mid) {
                *nv += ode.step_size * v;
            }
        }
        // Split target/residual and decode.
        let target: Vec<f32> = noisy[..frames * channels].to_vec();
        let residual: Vec<f32> = noisy[frames * channels..].to_vec();
        let target_wav = self.codec.decode(&target)?;
        let residual_wav = self.codec.decode(&residual)?;
        Ok((target_wav, residual_wav))
    }
}

#[cfg(test)]
mod tests;
