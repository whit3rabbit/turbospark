//! DialogueSidon two-speaker separation and speech restoration.
//!
//! Port of `mlx_audio/sts/models/dialogue_sidon` at reference commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. A w2v-BERT-style encoder
//! with relative-position attention and a causal GLU convolution module
//! embeds Kaldi-style 80-mel features; two projection heads form a
//! conditioning vector; an adaLN diffusion transformer with a
//! second-order DPM-Solver++ samples two latent tracks that a DAC
//! decoder turns into 24 kHz speaker waveforms.
//!
//! The DAC decoder rides the existing `codec::descript` loader; the
//! checkpoint directory supplies a DAC `config.json` plus weights whose
//! `decoder.*` subtree carries the separation decoder.

use std::path::Path;

use crate::codec::descript::{Dac, DacConfig};
use crate::error::SpeechError;
use crate::fft::RealFftPlan;
use crate::ops;
use turbospark_model_io::safetensors::SafetensorsFile;

type Result<T> = std::result::Result<T, SpeechError>;

/// Torchaudio Kaldi mel bank `[80, 258]` (spectrum padded by one bin)
/// and the povey analysis window `[400]`, matching the reference
/// `_filters` (float64 bounds, float32 tensor math).
fn sidon_filters() -> (Vec<f32>, Vec<f32>) {
    let low = 1127.0f64 * (1.0f64 + 20.0 / 700.0).ln();
    let high = 1127.0f64 * (1.0f64 + 8000.0 / 700.0).ln();
    let delta = (high - low) / 81.0;
    let mut bank = vec![0.0f32; 80 * 258];
    for b in 0..80usize {
        let left = low + b as f64 * delta;
        let center = low + (b as f64 + 1.0) * delta;
        let right = low + (b as f64 + 2.0) * delta;
        for m in 0..256usize {
            let mel = 1127.0f64 * (1.0 + (m as f32 * 31.25) as f64 / 700.0).ln();
            let up = ((mel - left) / (center - left)) as f32;
            let down = ((right - mel) / (right - center)) as f32;
            bank[b * 258 + m] = up.min(down).max(0.0);
        }
    }
    let window: Vec<f32> = (0..400usize)
        .map(|n| {
            let base = 0.5 - 0.5 * (n as f32 * (2.0 * std::f32::consts::PI / 399.0)).cos();
            base.powf(0.85)
        })
        .collect();
    (bank, window)
}

const LN_EPS: f32 = 1e-5;
const FEAT_EPS: f32 = 1e-5;
/// 2^-23, the reference log floor.
const LOG_FLOOR: f32 = 1.1920929e-7;

#[derive(Debug, Clone)]
pub struct SidonEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub feature_projection_input_dim: usize,
    pub conv_depthwise_kernel_size: usize,
    pub left_max_position_embeddings: usize,
    pub right_max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct SidonDiffusionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_ratio: f32,
    pub frequency_embedding_size: usize,
    pub num_train_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub prediction_type: String,
}

#[derive(Debug, Clone)]
pub struct SidonConfig {
    pub sample_rate: usize,
    pub latent_dim: usize,
    pub encoder: SidonEncoderConfig,
    pub diffusion: SidonDiffusionConfig,
    pub decoder_channels: usize,
    pub decoder_rates: Vec<usize>,
    pub latent_norm_initialized: bool,
    pub latent_norm_mean: Vec<f32>,
    pub latent_norm_std: Vec<f32>,
}

impl SidonConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| SpeechError::BadConfig {
            field: "config".to_string(),
            why: "expected a JSON object".to_string(),
        })?;
        let sub = |name: &str| -> serde_json::Map<String, serde_json::Value> {
            obj.get(name)
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default()
        };
        let num = |o: &serde_json::Map<String, serde_json::Value>, k: &str, d: usize| {
            o.get(k).and_then(|v| v.as_u64()).unwrap_or(d as u64) as usize
        };
        let enc = sub("encoder");
        let diff = sub("diffusion");
        let f32v = |o: &serde_json::Map<String, serde_json::Value>, k: &str, d: f32| {
            o.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32
        };
        let encoder = SidonEncoderConfig {
            hidden_size: num(&enc, "hidden_size", 1024),
            intermediate_size: num(&enc, "intermediate_size", 4096),
            num_hidden_layers: num(&enc, "num_hidden_layers", 13),
            num_attention_heads: num(&enc, "num_attention_heads", 16),
            feature_projection_input_dim: num(&enc, "feature_projection_input_dim", 160),
            conv_depthwise_kernel_size: num(&enc, "conv_depthwise_kernel_size", 31),
            left_max_position_embeddings: num(&enc, "left_max_position_embeddings", 64),
            right_max_position_embeddings: num(&enc, "right_max_position_embeddings", 8),
        };
        let diffusion = SidonDiffusionConfig {
            hidden_size: num(&diff, "hidden_size", 768),
            num_layers: num(&diff, "num_layers", 8),
            num_heads: num(&diff, "num_heads", 12),
            ffn_ratio: f32v(&diff, "ffn_ratio", 4.0),
            frequency_embedding_size: num(&diff, "frequency_embedding_size", 256),
            num_train_timesteps: num(&diff, "num_train_timesteps", 1000),
            beta_start: f32v(&diff, "beta_start", 0.0001),
            beta_end: f32v(&diff, "beta_end", 0.02),
            prediction_type: diff
                .get("prediction_type")
                .and_then(|v| v.as_str())
                .unwrap_or("v_prediction")
                .to_string(),
        };
        let latent_dim = num(obj, "latent_dim", 32);
        let latent_norm_initialized = obj
            .get("latent_norm_initialized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let vec_f32 = |name: &str| -> Vec<f32> {
            obj.get(name)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let config = Self {
            sample_rate: num(obj, "sample_rate", 24000),
            latent_dim,
            encoder,
            diffusion,
            decoder_channels: num(obj, "decoder_channels", 1536),
            decoder_rates: obj
                .get("decoder_rates")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as usize)
                        .collect()
                })
                .unwrap_or_else(|| vec![8, 5, 4, 3]),
            latent_norm_initialized,
            latent_norm_mean: vec_f32("latent_norm_mean"),
            latent_norm_std: vec_f32("latent_norm_std"),
        };
        if config.latent_norm_initialized
            && (config.latent_norm_mean.len() != latent_dim * 2
                || config.latent_norm_std.len() != latent_dim * 2)
        {
            return Err(SpeechError::BadConfig {
                field: "latent_norm_mean".to_string(),
                why: "expected statistics for both speakers".to_string(),
            });
        }
        Ok(config)
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

fn layernorm(x: &mut [f32], rows: usize, cols: usize, weight: &[f32], bias: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(cols).take(rows) {
        let mut mean = 0.0f32;
        for value in row.iter() {
            mean += value;
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for value in row.iter() {
            let d = value - mean;
            var += d * d;
        }
        var /= cols as f32;
        let denom = (var + eps).sqrt();
        for ((value, &w), &b) in row.iter_mut().zip(weight).zip(bias) {
            *value = (*value - mean) / denom * w + b;
        }
    }
}

/// The w2v-BERT encoder layer stack.
pub struct SidonEncoder {
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    proj: Linear,
    layers: Vec<EncoderLayer>,
    dim: usize,
    heads: usize,
    head_dim: usize,
    left: usize,
    right: usize,
    conv_kernel: usize,
    eps: f32,
}

struct EncoderLayer {
    ffn1_norm: (Vec<f32>, Vec<f32>),
    ffn1: (Linear, Linear),
    attn_norm: (Vec<f32>, Vec<f32>),
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    distance: Vec<f32>,
    conv_norm: (Vec<f32>, Vec<f32>),
    pw1: Linear,
    dw: Vec<f32>,
    dw_norm: (Vec<f32>, Vec<f32>),
    pw2: Linear,
    ffn2_norm: (Vec<f32>, Vec<f32>),
    ffn2: (Linear, Linear),
    final_norm: (Vec<f32>, Vec<f32>),
}

impl SidonEncoder {
    /// `features` is `[frames, 160]`; `mask` is `[paired_frames]`
    /// (1 = attend). Returns `[paired_frames, hidden]`.
    pub fn forward(&self, features: &[f32], frames: usize, mask: Option<&[f32]>) -> Vec<f32> {
        let dim = self.dim;
        let mut feats = features.to_vec();
        layernorm(&mut feats, frames, 160, &self.ln_w, &self.ln_b, self.eps);
        let mut x = self.proj.run(&feats, frames);
        if let Some(mask) = mask {
            for (t, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    for value in &mut x[t * dim..(t + 1) * dim] {
                        *value = 0.0;
                    }
                }
            }
        }
        for layer in &self.layers {
            // FFN with half-weight residual.
            let mut normed = x.clone();
            layernorm(
                &mut normed,
                frames,
                dim,
                &layer.ffn1_norm.0,
                &layer.ffn1_norm.1,
                self.eps,
            );
            let mut hidden = layer.ffn1.0.run(&normed, frames);
            ops::silu(&mut hidden);
            let out = layer.ffn1.1.run(&hidden, frames);
            for (hv, ov) in x.iter_mut().zip(out) {
                *hv += 0.5 * ov;
            }
            // Relative-position attention.
            let mut normed = x.clone();
            layernorm(
                &mut normed,
                frames,
                dim,
                &layer.attn_norm.0,
                &layer.attn_norm.1,
                self.eps,
            );
            let attn = self.attention(layer, &normed, frames, mask);
            for (hv, av) in x.iter_mut().zip(attn) {
                *hv += av;
            }
            // Convolution module.
            let mut normed = x.clone();
            layernorm(
                &mut normed,
                frames,
                dim,
                &layer.conv_norm.0,
                &layer.conv_norm.1,
                self.eps,
            );
            let conv = self.conv_module(layer, &normed, frames, mask);
            for (hv, cv) in x.iter_mut().zip(conv) {
                *hv += cv;
            }
            // Second FFN with half-weight residual, then the layer-final
            // normalization (w2v-BERT carries it per layer here).
            let mut normed = x.clone();
            layernorm(
                &mut normed,
                frames,
                dim,
                &layer.ffn2_norm.0,
                &layer.ffn2_norm.1,
                self.eps,
            );
            let mut hidden = layer.ffn2.0.run(&normed, frames);
            ops::silu(&mut hidden);
            let out = layer.ffn2.1.run(&hidden, frames);
            for (hv, ov) in x.iter_mut().zip(out) {
                *hv += 0.5 * ov;
            }
            layernorm(
                &mut x,
                frames,
                dim,
                &layer.final_norm.0,
                &layer.final_norm.1,
                self.eps,
            );
        }
        x
    }

    fn attention(
        &self,
        layer: &EncoderLayer,
        x: &[f32],
        pairs: usize,
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let dim = self.dim;
        let heads = self.heads;
        let head_dim = self.head_dim;
        let q = layer.q.run(x, pairs);
        let k = layer.k.run(x, pairs);
        let v = layer.v.run(x, pairs);
        // Relative-position bias: q projected onto the distance table,
        // gathered per clamped pair distance.
        let table = &layer.distance;
        let rel_width = self.left + self.right + 1;
        let mut bias = vec![0.0f32; pairs * pairs * heads];
        let scale = (head_dim as f32).powf(-0.5);
        for qi in 0..pairs {
            for ki in 0..pairs {
                let distance = (ki as isize - qi as isize)
                    .clamp(-(self.left as isize), self.right as isize)
                    + self.left as isize;
                for h in 0..heads {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(qi * heads + h) * head_dim + d]
                            * table[distance as usize * head_dim + d];
                    }
                    let mut value = dot * scale;
                    if let Some(mask) = mask {
                        if mask[ki] == 0.0 {
                            value = f32::NEG_INFINITY;
                        }
                    }
                    bias[(qi * pairs + ki) * heads + h] = value;
                }
            }
        }
        let _ = rel_width;
        // Standard scaled dot-product attention with the additive bias.
        let mut out = vec![0.0f32; pairs * heads * head_dim];
        for h in 0..heads {
            for t in 0..pairs {
                let qbase = (t * heads + h) * head_dim;
                let mut scores = vec![0.0f32; pairs];
                let mut max_score = f32::NEG_INFINITY;
                for u in 0..pairs {
                    let kbase = (u * heads + h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qbase + d] * k[kbase + d];
                    }
                    let score =
                        dot * (head_dim as f32).powf(-0.5) + bias[(t * pairs + u) * heads + h];
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
                    let vbase = (u * heads + h) * head_dim;
                    for d in 0..head_dim {
                        out[qbase + d] += weight * v[vbase + d];
                    }
                }
            }
        }
        // Merge heads: (t, h, d) -> (t, h*d).
        let mut merged = vec![0.0f32; pairs * dim];
        for t in 0..pairs {
            for h in 0..heads {
                merged[t * dim + h * head_dim..t * dim + (h + 1) * head_dim].copy_from_slice(
                    &out[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim],
                );
            }
        }
        layer.o.run(&merged, pairs)
    }

    fn conv_module(
        &self,
        layer: &EncoderLayer,
        x: &[f32],
        pairs: usize,
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let dim = self.dim;
        let mut h = x.to_vec();
        if let Some(mask) = mask {
            for (t, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    for value in &mut h[t * dim..(t + 1) * dim] {
                        *value = 0.0;
                    }
                }
            }
        }
        // Pointwise conv 1x1 to 2*dim (channels-last rows), then GLU.
        let gated = layer.pw1.run(&h, pairs);
        let mut a = vec![0.0f32; pairs * dim];
        for t in 0..pairs {
            for c in 0..dim {
                a[t * dim + c] =
                    gated[t * 2 * dim + c] * ops::sigmoid(gated[t * 2 * dim + dim + c]);
            }
        }
        // Causal depthwise conv over time (channels-last), kernel
        // [k][1][C] stored [C][k] row-major.
        let kern = self.conv_kernel;
        let mut conv = vec![0.0f32; pairs * dim];
        for t in 0..pairs {
            for c in 0..dim {
                let mut acc = 0.0f32;
                for k in 0..kern {
                    let src = t as isize + k as isize - (kern as isize - 1);
                    if src < 0 {
                        continue;
                    }
                    acc += a[src as usize * dim + c] * layer.dw[c * kern + k];
                }
                conv[t * dim + c] = acc;
            }
        }
        layernorm(
            &mut conv,
            pairs,
            dim,
            &layer.dw_norm.0,
            &layer.dw_norm.1,
            self.eps,
        );
        ops::silu(&mut conv);
        layer.pw2.run(&conv, pairs)
    }
}

/// AdaLN diffusion transformer (half-split RoPE, exact-erf GELU).
pub struct DiffusionHead {
    t_mlp: (Linear, Linear),
    freq_embedding_size: usize,
    latent_proj: Linear,
    cond_proj: Linear,
    blocks: Vec<DiTBlock>,
    final_ada: Linear,
    final_linear: Linear,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

struct DiTBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    mlp: (Linear, Linear),
    ada: Linear,
}

impl DiffusionHead {
    fn timestep_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.freq_embedding_size / 2;
        let mut emb = vec![0.0f32; self.freq_embedding_size];
        for i in 0..half {
            let freq = (-10000f32.ln() * i as f32 / half as f32).exp();
            let arg = t * freq;
            emb[i] = arg.cos();
            emb[half + i] = arg.sin();
        }
        emb
    }

    /// Half-split RoPE over `[frames, heads, head_dim]` buffers; the
    /// position advances along frames and every head rotates.
    fn rope(buf: &mut [f32], frames: usize, heads: usize, head_dim: usize) {
        let half = head_dim / 2;
        for t in 0..frames {
            for h in 0..heads {
                let base = (t * heads + h) * head_dim;
                for i in 0..half {
                    let freq = 10000f32.powf(-((2 * i) as f32) / head_dim as f32);
                    let (c, s) = ((t as f32 * freq).cos(), (t as f32 * freq).sin());
                    let x0 = buf[base + i];
                    let x1 = buf[base + half + i];
                    buf[base + i] = x0 * c - x1 * s;
                    buf[base + half + i] = x0 * s + x1 * c;
                }
            }
        }
    }

    /// `noisy` is `[frames, 2 * latent_dim]`, `conditioning` is
    /// `[frames, cond_dim]`; `timesteps` is a scalar batch. Returns
    /// the predicted velocity `[frames, 2 * latent_dim]`.
    pub fn forward(&self, noisy: &[f32], frames: usize, t: f32, conditioning: &[f32]) -> Vec<f32> {
        let dim = self.dim;
        let mut x = self.latent_proj.run(noisy, frames);
        let freq = self.timestep_embedding(t);
        let hidden = self.t_mlp.0.run(&freq, 1);
        let mut hidden = hidden;
        ops::silu(&mut hidden);
        let t_emb = self.t_mlp.1.run(&hidden, 1);
        let cond_projected = self.cond_proj.run(conditioning, frames);
        let mut c = vec![0.0f32; frames * dim];
        for f in 0..frames {
            for d in 0..dim {
                c[f * dim + d] = cond_projected[f * dim + d] + t_emb[d];
            }
        }
        for block in &self.blocks {
            // adaLN_modulation is SiLU then Linear.
            let mut c_silu = c.clone();
            ops::silu(&mut c_silu);
            let mods = block.ada.run(&c_silu, frames);
            let seg = |i: usize| &mods[i * dim..(i + 1) * dim];
            let (sa, ca, ga, sm, cm, gm) = (seg(0), seg(1), seg(2), seg(3), seg(4), seg(5));
            let mut h = x.clone();
            dit_norm(&mut h, frames, dim, self.eps);
            for i in 0..frames * dim {
                h[i] = h[i] * (1.0 + ca[i % dim]) + sa[i % dim];
            }
            let mut q = block.q.run(&h, frames);
            let mut k = block.k.run(&h, frames);
            let v = block.v.run(&h, frames);
            Self::rope(&mut q, frames, self.heads, self.head_dim);
            Self::rope(&mut k, frames, self.heads, self.head_dim);
            // Standard attention over (frames, heads, head_dim).
            let mut attended = vec![0.0f32; frames * dim];
            for head in 0..self.heads {
                for t_i in 0..frames {
                    let qbase = (t_i * self.heads + head) * self.head_dim;
                    let mut scores = vec![0.0f32; frames];
                    let mut max_score = f32::NEG_INFINITY;
                    for u in 0..frames {
                        let kbase = (u * self.heads + head) * self.head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q[qbase + d] * k[kbase + d];
                        }
                        scores[u] = dot * (self.head_dim as f32).powf(-0.5);
                        max_score = max_score.max(scores[u]);
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
                        let vbase = (u * self.heads + head) * self.head_dim;
                        for d in 0..self.head_dim {
                            attended[qbase + d] += weight * v[vbase + d];
                        }
                    }
                }
            }
            let mut merged = vec![0.0f32; frames * dim];
            for t_i in 0..frames {
                for head in 0..self.heads {
                    merged
                        [t_i * dim + head * self.head_dim..t_i * dim + (head + 1) * self.head_dim]
                        .copy_from_slice(
                            &attended[(t_i * self.heads + head) * self.head_dim
                                ..(t_i * self.heads + head + 1) * self.head_dim],
                        );
                }
            }
            let out = block.o.run(&merged, frames);
            for (xv, (ov, g)) in x.iter_mut().zip(out.iter().zip(ga.iter().cycle())) {
                *xv += ov * g;
            }
            let mut h = x.clone();
            dit_norm(&mut h, frames, dim, self.eps);
            for i in 0..frames * dim {
                h[i] = h[i] * (1.0 + cm[i % dim]) + sm[i % dim];
            }
            let mut hidden = block.mlp.0.run(&h, frames);
            ops::gelu_erf(&mut hidden);
            let out = block.mlp.1.run(&hidden, frames);
            for (xv, (ov, g)) in x.iter_mut().zip(out.iter().zip(gm.iter().cycle())) {
                *xv += ov * g;
            }
        }
        // Final layer.
        let mut c_silu = c.clone();
        ops::silu(&mut c_silu);
        let mods = self.final_ada.run(&c_silu, frames);
        let mut shift = vec![0.0f32; dim];
        let mut scale = vec![0.0f32; dim];
        // The batch is one row; shift/scale come from frame 0.
        shift
            .iter_mut()
            .zip(&mods[..dim])
            .for_each(|(s, &m)| *s = m);
        scale
            .iter_mut()
            .zip(&mods[dim..2 * dim])
            .for_each(|(s, &m)| *s = m);
        let _ = (&shift, &scale);
        let mut h = x.clone();
        dit_norm(&mut h, frames, dim, self.eps);
        for i in 0..frames * dim {
            h[i] = h[i] * (1.0 + scale[i % dim]) + shift[i % dim];
        }
        self.final_linear.run(&h, frames)
    }
}

/// Affine-free per-row layer norm written back into the buffer.
fn dit_norm(x: &mut [f32], rows: usize, dim: usize, eps: f32) {
    for row in x.chunks_exact_mut(dim).take(rows) {
        let mut mean = 0.0f32;
        for value in row.iter() {
            mean += value;
        }
        mean /= dim as f32;
        let mut var = 0.0f32;
        for value in row.iter() {
            let d = value - mean;
            var += d * d;
        }
        var /= dim as f32;
        let denom = (var + eps).sqrt();
        for value in row.iter_mut() {
            *value = (*value - mean) / denom;
        }
    }
}

/// Linear-beta linspace midpoint DPM-Solver++ (v-prediction).
pub struct DpmSolver {
    pub timesteps: Vec<i32>,
    alpha: Vec<f64>,
    sigma: Vec<f64>,
    lambdas: Vec<f64>,
    prediction_type: String,
    previous: Option<Vec<f32>>,
    index: usize,
}

impl DpmSolver {
    pub fn new(config: &SidonDiffusionConfig, num_steps: usize) -> Result<Self> {
        if num_steps == 0 || num_steps > config.num_train_timesteps {
            return Err(SpeechError::Input {
                why: format!(
                    "num_steps must be between 1 and {}",
                    config.num_train_timesteps
                ),
            });
        }
        // Linear beta schedule; cumulative products in f64 like the
        // reference.
        let n = config.num_train_timesteps;
        let betas: Vec<f64> = (0..n)
            .map(|i| {
                config.beta_start as f64
                    + (config.beta_end as f64 - config.beta_start as f64) * i as f64
                        / (n - 1) as f64
            })
            .collect();
        let mut cumulative = 1.0f64;
        let cumulative: Vec<f64> = betas
            .iter()
            .map(|&beta| {
                cumulative *= 1.0 - beta;
                cumulative
            })
            .collect();
        let timesteps: Vec<i32> = (0..num_steps + 1)
            .map(|i| {
                (0.0 + (config.num_train_timesteps - 1) as f64 * i as f64 / num_steps as f64)
                    .round() as i32
            })
            .rev()
            .take(num_steps)
            .collect();
        let mut alpha = Vec::with_capacity(num_steps + 1);
        let mut sigma = Vec::with_capacity(num_steps + 1);
        for &t in &timesteps {
            let c = cumulative[t as usize];
            alpha.push(c.sqrt());
            sigma.push((1.0 - c).sqrt());
        }
        alpha.push(1.0);
        sigma.push(0.0);
        let lambdas = alpha
            .iter()
            .zip(&sigma)
            .map(|(&a, &s)| {
                if s != 0.0 {
                    a.ln() - s.ln()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        Ok(Self {
            timesteps,
            alpha,
            sigma,
            lambdas,
            prediction_type: config.prediction_type.clone(),
            previous: None,
            index: 0,
        })
    }

    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>> {
        let i = self.index;
        let x0: Vec<f32> = match self.prediction_type.as_str() {
            "v_prediction" => sample
                .iter()
                .zip(model_output)
                .map(|(&s, &m)| (self.alpha[i] * s as f64 - self.sigma[i] * m as f64) as f32)
                .collect(),
            "epsilon" => sample
                .iter()
                .zip(model_output)
                .map(|(&s, &m)| ((s as f64 - self.sigma[i] * m as f64) / self.alpha[i]) as f32)
                .collect(),
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("prediction type {other}"),
                })
            }
        };
        let h = self.lambdas[i + 1] - self.lambdas[i];
        let coefficient = self.alpha[i + 1] * (-h).exp_m1();
        let mut result: Vec<f32> = sample
            .iter()
            .zip(&x0)
            .map(|(&s, &x)| {
                ((self.sigma[i + 1] / self.sigma[i]) * s as f64 - coefficient * x as f64) as f32
            })
            .collect();
        if let Some(prev) = &self.previous {
            if i < self.timesteps.len() - 1 {
                let r = (self.lambdas[i] - self.lambdas[i - 1]) / h;
                for (rv, (&x, &p)) in result.iter_mut().zip(x0.iter().zip(prev)) {
                    *rv -= 0.5 * coefficient as f32 * (x - p) / r as f32;
                }
            }
        }
        self.previous = Some(x0);
        self.index += 1;
        Ok(result)
    }
}

/// The loaded DialogueSidon model.
pub struct DialogueSidon {
    pub config: SidonConfig,
    encoder: SidonEncoder,
    linear1: Linear,
    linear2: Linear,
    diffusion_head: DiffusionHead,
    decoder: Dac,
}

impl DialogueSidon {
    /// Opens a checkpoint directory: `config.json`, the model weights
    /// (`model.safetensors`), and the standalone DAC tree the crate
    /// loader consumes (`dac_config.json` + `dac.safetensors`).
    pub fn open(dir: &Path) -> Result<Self> {
        let config = SidonConfig::from_json(&crate::quant::read_json(&dir.join("config.json"))?)?;
        let weights = dir.join("model.safetensors");
        let file = SafetensorsFile::open(&weights).map_err(|e| SpeechError::BadConfig {
            field: weights.display().to_string(),
            why: e.to_string(),
        })?;
        let dac_config =
            DacConfig::from_json(&crate::quant::read_json(&dir.join("dac_config.json"))?)?;
        let dac_file = SafetensorsFile::open(&dir.join("dac.safetensors")).map_err(|e| {
            SpeechError::BadConfig {
                field: dir.join("dac.safetensors").display().to_string(),
                why: e.to_string(),
            }
        })?;
        Self::load(config, &file, dac_config, &dac_file)
    }

    pub fn load(
        config: SidonConfig,
        file: &SafetensorsFile,
        dac_config: DacConfig,
        dac_file: &SafetensorsFile,
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
            // MLX kernel-1 convs store [out, 1, in]; unwrap to a linear.
            let (out_dim, in_dim) = match desc.shape.len() {
                2 => (desc.shape[0], desc.shape[1]),
                3 if desc.shape[1] == 1 => (desc.shape[0], desc.shape[2]),
                other => {
                    return Err(SpeechError::Tensor {
                        name: name.to_string(),
                        why: format!("unsupported weight rank {other}"),
                    })
                }
            };
            Ok(Linear {
                weight,
                bias,
                out_dim,
                in_dim,
            })
        };
        let norm = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                f32_tensor(&format!("{name}.weight"))?,
                f32_tensor(&format!("{name}.bias"))?,
            ))
        };

        let ec = &config.encoder;
        // Encoder.
        let mut layers = Vec::with_capacity(ec.num_hidden_layers);
        for i in 0..ec.num_hidden_layers {
            let p = format!("encoder.layers.{i}");
            let ffn = |name: &str| -> Result<(Linear, Linear)> {
                Ok((
                    linear(&format!("{p}.{name}.intermediate_dense"), true)?,
                    linear(&format!("{p}.{name}.output_dense"), true)?,
                ))
            };
            let depthwise = f32_tensor(&format!("{p}.conv_module.depthwise_conv.weight"))?;
            layers.push(EncoderLayer {
                ffn1_norm: norm(&format!("{p}.ffn1_layer_norm"))?,
                ffn1: ffn("ffn1")?,
                attn_norm: norm(&format!("{p}.self_attn_layer_norm"))?,
                q: linear(&format!("{p}.self_attn.linear_q"), true)?,
                k: linear(&format!("{p}.self_attn.linear_k"), true)?,
                v: linear(&format!("{p}.self_attn.linear_v"), true)?,
                o: linear(&format!("{p}.self_attn.linear_out"), true)?,
                distance: f32_tensor(&format!("{p}.self_attn.distance_embedding.weight"))?,
                conv_norm: norm(&format!("{p}.conv_module.layer_norm"))?,
                pw1: linear(&format!("{p}.conv_module.pointwise_conv1"), false)?,
                dw: depthwise,
                dw_norm: norm(&format!("{p}.conv_module.depthwise_layer_norm"))?,
                pw2: linear(&format!("{p}.conv_module.pointwise_conv2"), false)?,
                ffn2_norm: norm(&format!("{p}.ffn2_layer_norm"))?,
                ffn2: ffn("ffn2")?,
                final_norm: norm(&format!("{p}.final_layer_norm"))?,
            });
        }
        let encoder = SidonEncoder {
            ln_w: f32_tensor("encoder.feature_projection.layer_norm.weight")?,
            ln_b: f32_tensor("encoder.feature_projection.layer_norm.bias")?,
            proj: linear("encoder.feature_projection.projection", true)?,
            layers,
            dim: ec.hidden_size,
            heads: ec.num_attention_heads,
            head_dim: ec.hidden_size / ec.num_attention_heads,
            left: ec.left_max_position_embeddings,
            right: ec.right_max_position_embeddings,
            conv_kernel: ec.conv_depthwise_kernel_size,
            eps: LN_EPS,
        };

        // Diffusion head.
        let dc = &config.diffusion;
        let dim = dc.hidden_size;
        let mut blocks = Vec::with_capacity(dc.num_layers);
        for i in 0..dc.num_layers {
            let p = format!("diffusion_head.blocks.{i}");
            let ffn_dim = (dim as f32 * dc.ffn_ratio) as usize;
            blocks.push(DiTBlock {
                q: linear(&format!("{p}.q_proj"), true)?,
                k: linear(&format!("{p}.k_proj"), true)?,
                v: linear(&format!("{p}.v_proj"), true)?,
                o: linear(&format!("{p}.out_proj"), true)?,
                mlp: (
                    linear(&format!("{p}.mlp.layers.0"), true)?,
                    linear(&format!("{p}.mlp.layers.2"), true)?,
                ),
                ada: linear(&format!("{p}.adaLN_modulation.layers.1"), true)?,
            });
            if blocks.last().unwrap().mlp.0.out_dim != ffn_dim {
                return Err(SpeechError::Tensor {
                    name: format!("{p}.mlp.layers.0"),
                    why: format!("hidden dim mismatch, expected {ffn_dim}"),
                });
            }
        }
        let diffusion_head = DiffusionHead {
            t_mlp: (
                linear("diffusion_head.t_embedder.mlp.layers.0", false)?,
                linear("diffusion_head.t_embedder.mlp.layers.2", false)?,
            ),
            freq_embedding_size: dc.frequency_embedding_size,
            latent_proj: linear("diffusion_head.latent_proj", false)?,
            cond_proj: linear("diffusion_head.cond_proj", false)?,
            blocks,
            final_ada: linear("diffusion_head.final_layer.adaLN_modulation.layers.1", true)?,
            final_linear: linear("diffusion_head.final_layer.linear", false)?,
            dim,
            heads: dc.num_heads,
            head_dim: dim / dc.num_heads,
            eps: 1e-6,
        };

        // Decoder (standalone DAC).
        let decoder = Dac::load(dac_config, dac_file)?;

        Ok(Self {
            config,
            encoder,
            linear1: linear("linear1", true)?,
            linear2: linear("linear2", true)?,
            diffusion_head,
            decoder,
        })
    }

    /// Kaldi-style 80-mel features with per-bin variance normalization
    /// and the paired-frame mask. Input is the peak-normalized, padded
    /// mono waveform; returns `[paired, 160]` features and the mask.
    pub fn extract_features(&self, waveform: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if waveform.len() < 560 {
            return Err(SpeechError::Input {
                why: "feature extraction requires at least 560 mono samples".to_string(),
            });
        }
        let count = 1 + (waveform.len() - 400) / 160;
        // Mel bank: torchaudio Kaldi bounds, 257 spectrum bins padded to 258.
        let (bank, window) = sidon_filters();
        let plan = RealFftPlan::cached(512).map_err(|e| SpeechError::Input {
            why: format!("fbank plan: {e}"),
        })?;
        let mut feats = vec![0.0f32; count * 80];
        let mut frame = vec![0.0f32; 400];
        let mut padded = vec![0.0f32; 512];
        for f in 0..count {
            let start = f * 160;
            frame.copy_from_slice(&waveform[start..start + 400]);
            let mut mean = 0.0f32;
            for value in &frame {
                mean += value;
            }
            mean /= 400.0;
            // Pre-emphasis keeps 0.03 * first sample.
            let first = frame[0] - mean;
            frame[0] = first * 0.03;
            let mut prev = first;
            for value in frame.iter_mut().skip(1) {
                let cur = *value - mean;
                *value = cur - 0.97 * prev;
                prev = cur;
            }
            for (value, &window) in frame.iter_mut().zip(&window) {
                *value *= window;
            }
            padded[..400].copy_from_slice(&frame);
            padded[400..].fill(0.0);
            let spectrum = plan.forward(&padded).map_err(|e| SpeechError::Input {
                why: format!("fbank fft: {e}"),
            })?;
            for (mel, row) in feats[f * 80..(f + 1) * 80].iter_mut().enumerate() {
                let mut energy = 0.0f32;
                for (bin, &weight) in bank[mel * 258..(mel + 1) * 258].iter().enumerate() {
                    if weight != 0.0 && bin < spectrum.len() {
                        let re = spectrum[bin].re;
                        let im = spectrum[bin].im;
                        energy += weight * (re * re + im * im);
                    }
                }
                *row = energy.max(LOG_FLOOR).ln();
            }
        }
        // Per-bin variance normalization across frames (axis 0).
        let mut means = vec![0.0f32; 80];
        for f in 0..count {
            for (mel, value) in feats[f * 80..(f + 1) * 80].iter().enumerate() {
                means[mel] += value;
            }
        }
        for mean in means.iter_mut() {
            *mean /= count as f32;
        }
        for f in 0..count {
            for mel in 0..80 {
                feats[f * 80 + mel] -= means[mel];
            }
        }
        for mel in 0..80 {
            let mut var = 0.0f32;
            for f in 0..count {
                let d = feats[f * 80 + mel];
                var += d * d;
            }
            var /= (count - 1) as f32;
            let f = (var + FEAT_EPS).sqrt().recip();
            for fr in 0..count {
                feats[fr * 80 + mel] *= f;
            }
        }
        // Pad an odd frame count; adjacent 80-dim frames pair into
        // 160-wide encoder tokens and the mask keeps the even frames.
        let mut mask = vec![1.0f32; count];
        if count % 2 == 1 {
            feats.extend(std::iter::repeat_n(0.0, 80));
            mask.push(0.0);
        }
        let tokens = feats.len() / 160;
        let mut paired = vec![0.0f32; tokens * 160];
        for (t, token) in paired.chunks_exact_mut(160).enumerate() {
            token[..80].copy_from_slice(&feats[t * 2 * 80..(t * 2 + 1) * 80]);
            token[80..].copy_from_slice(&feats[(t * 2 + 1) * 80..(t * 2 + 2) * 80]);
        }
        let pair_mask: Vec<f32> = mask.iter().skip(1).step_by(2).copied().collect();
        Ok((paired, pair_mask))
    }

    /// Peak-normalizes to 0.9 and pads 160 samples on both sides.
    pub fn normalize_chunk(&self, waveform: &[f32]) -> Vec<f32> {
        let peak = waveform.iter().fold(1e-6f32, |a, &v| a.max(v.abs()));
        let mut out = Vec::with_capacity(waveform.len() + 320);
        out.extend(std::iter::repeat_n(0.0, 160));
        out.extend(waveform.iter().map(|&v| 0.9 * v / peak));
        out.extend(std::iter::repeat_n(0.0, 160));
        while out.len() < 560 {
            out.push(0.0);
        }
        out
    }

    /// Encodes features and builds the conditioning vector
    /// `[paired, hidden + 2 * latent_dim]`.
    pub fn conditioning(&self, features: &[f32], frames: usize, mask: &[f32]) -> Vec<f32> {
        let dim = self.config.encoder.hidden_size;
        let latent = self.config.latent_dim;
        let encoded = self.encoder.forward(features, frames, Some(mask));
        let first = self.linear1.run(&encoded, frames);
        let second = self.linear2.run(&encoded, frames);
        let mut predicted = vec![0.0f32; frames * 2 * latent];
        for f in 0..frames {
            predicted[f * 2 * latent..f * 2 * latent + latent]
                .copy_from_slice(&first[f * latent..(f + 1) * latent]);
            predicted[f * 2 * latent + latent..(f + 1) * 2 * latent]
                .copy_from_slice(&second[f * latent..(f + 1) * latent]);
        }
        let normed = self.normalize_latents(&predicted);
        let mut cond = vec![0.0f32; frames * (dim + 2 * latent)];
        for f in 0..frames {
            cond[f * (dim + 2 * latent)..f * (dim + 2 * latent) + 2 * latent]
                .copy_from_slice(&normed[f * 2 * latent..(f + 1) * 2 * latent]);
            cond[f * (dim + 2 * latent) + 2 * latent..(f + 1) * (dim + 2 * latent)]
                .copy_from_slice(&encoded[f * dim..(f + 1) * dim]);
        }
        cond
    }

    fn normalize_latents(&self, latents: &[f32]) -> Vec<f32> {
        if !self.config.latent_norm_initialized {
            return latents.to_vec();
        }
        latents
            .iter()
            .zip(&self.config.latent_norm_mean)
            .zip(&self.config.latent_norm_std)
            .map(|((&v, &m), &s)| (v - m) / s)
            .collect()
    }

    fn denormalize_latents(&self, latents: &[f32]) -> Vec<f32> {
        if !self.config.latent_norm_initialized {
            return latents.to_vec();
        }
        latents
            .iter()
            .zip(&self.config.latent_norm_mean)
            .zip(&self.config.latent_norm_std)
            .map(|((&v, &m), &s)| v * s + m)
            .collect()
    }

    /// Samples latents with DPM-Solver++ from `noise`
    /// (`[paired, 2 * latent_dim]`).
    pub fn sample_latents(
        &self,
        conditioning: &[f32],
        frames: usize,
        noise: &[f32],
        num_steps: usize,
    ) -> Result<Vec<f32>> {
        let mut solver = DpmSolver::new(&self.config.diffusion, num_steps)?;
        let mut latents = noise.to_vec();
        for (step, &t) in solver.timesteps.clone().iter().enumerate() {
            let _ = step;
            let prediction = self
                .diffusion_head
                .forward(&latents, frames, t as f32, conditioning);
            latents = solver.step(&prediction, &latents)?;
        }
        Ok(latents)
    }

    /// Decodes normalized latents `[paired, 2 * latent_dim]` into two
    /// mono waveforms.
    pub fn decode_latents(&self, latents: &[f32], frames: usize) -> Result<Vec<f32>> {
        let latent = self.config.latent_dim;
        let denorm = self.denormalize_latents(latents);
        let mut out = Vec::new();
        for speaker in 0..2 {
            // Transpose to channel-major [latent, frames] for the DAC.
            let mut track = vec![0.0f32; latent * frames];
            for f in 0..frames {
                for c in 0..latent {
                    track[c * frames + f] = denorm[f * 2 * latent + speaker * latent + c];
                }
            }
            let decoded = self.decoder.decode_latents(&track);
            out.extend(decoded);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
