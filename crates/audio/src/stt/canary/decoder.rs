//! Canary transformer decoder: pre-norm blocks with self and cross
//! attention, a fixed sinusoidal positional table, and a greedy KV-cache
//! decode loop.
//!
//! Reference: `mlx_audio/stt/models/canary/decoder.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Batch size one is
//! assumed throughout. The checkpoint names (`transf_decoder.layers.N
//! .first_sub_layer.linear_q` and friends) are the converter's MLX-native
//! layout that the reference sanitize maps to the module tree; the port
//! reads them directly and dequantizes every linear through the groupwise
//! affine scheme.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::quant::{load_quantized, QuantScheme};
use crate::{Result, SpeechError};

use super::config::DecoderConfig;

/// One quantized (or plain) linear: HF `[out, in]` weight plus bias.
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Linear {
    fn load(file: &SafetensorsFile, base: &str, scheme: QuantScheme) -> Result<Self> {
        let (w, b) = load_quantized(file, base, scheme)?;
        Ok(Self {
            w,
            b: b.ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.bias"),
                why: "missing from the Canary checkpoint".into(),
            })?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize, in_dim: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.w,
            Some(&self.b),
            rows,
            in_dim,
            self.w.len() / in_dim,
        )
    }
}

/// One biased multi-head attention (`linear_q/k/v/out`).
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Attention {
    fn load(file: &SafetensorsFile, base: &str, scheme: QuantScheme) -> Result<Self> {
        Ok(Self {
            q: Linear::load(file, &format!("{base}.linear_q"), scheme)?,
            k: Linear::load(file, &format!("{base}.linear_k"), scheme)?,
            v: Linear::load(file, &format!("{base}.linear_v"), scheme)?,
            out: Linear::load(file, &format!("{base}.linear_out"), scheme)?,
        })
    }
}

/// One pre-norm decoder block: self attention, cross attention over encoder
/// states, then a relu feed-forward, each wrapped by a LayerNorm residual.
struct DecoderBlock {
    self_attn_norm: (Vec<f32>, Vec<f32>),
    self_attn: Attention,
    cross_attn_norm: (Vec<f32>, Vec<f32>),
    cross_attn: Attention,
    ff_norm: (Vec<f32>, Vec<f32>),
    ff1: Linear,
    ff2: Linear,
}

fn load_norm(file: &SafetensorsFile, base: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        file.load_as_f32(&format!("{base}.weight"))?,
        file.load_as_f32(&format!("{base}.bias"))?,
    ))
}

impl DecoderBlock {
    fn load(file: &SafetensorsFile, prefix: &str, scheme: QuantScheme) -> Result<Self> {
        Ok(Self {
            self_attn_norm: load_norm(file, &format!("{prefix}.layer_norm_1"))?,
            self_attn: Attention::load(file, &format!("{prefix}.first_sub_layer"), scheme)?,
            cross_attn_norm: load_norm(file, &format!("{prefix}.layer_norm_2"))?,
            cross_attn: Attention::load(file, &format!("{prefix}.second_sub_layer"), scheme)?,
            ff_norm: load_norm(file, &format!("{prefix}.layer_norm_3"))?,
            ff1: Linear::load(file, &format!("{prefix}.third_sub_layer.linear1"), scheme)?,
            ff2: Linear::load(file, &format!("{prefix}.third_sub_layer.linear2"), scheme)?,
        })
    }
}

/// Per-layer KV cache: the self-attention keys and values grow with every
/// generated token, the cross-attention keys and values over the encoder
/// states are fixed after the first forward.
pub struct LayerCache {
    self_k: Vec<f32>,
    self_v: Vec<f32>,
    self_len: usize,
    cross_k: Vec<f32>,
    cross_v: Vec<f32>,
}

/// The decoder KV cache for one transcription.
pub struct Cache {
    layers: Vec<LayerCache>,
    /// Number of valid encoder frames the cross attention may attend to.
    pub encoder_len: usize,
}

/// The loaded Canary decoder.
pub struct CanaryDecoder {
    embedding: Vec<f32>,
    embedding_layer_norm: (Vec<f32>, Vec<f32>),
    blocks: Vec<DecoderBlock>,
    final_norm: (Vec<f32>, Vec<f32>),
    output_proj: Linear,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_sequence_length: usize,
}

impl CanaryDecoder {
    /// Loads the decoder weights from the checkpoint.
    pub fn load(
        file: &SafetensorsFile,
        cfg: &DecoderConfig,
        vocab_size: usize,
        scheme: QuantScheme,
    ) -> Result<Self> {
        if !file.contains_tensor("transf_decoder.token_embedding.weight") {
            return Err(SpeechError::Tensor {
                name: "transf_decoder.token_embedding.weight".into(),
                why: "missing from safetensors".into(),
            });
        }
        let (embedding, _) = load_quantized(file, "transf_decoder.token_embedding", scheme)?;
        let blocks = (0..cfg.num_layers)
            .map(|i| DecoderBlock::load(file, &format!("transf_decoder.layers.{i}"), scheme))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding,
            embedding_layer_norm: load_norm(file, "transf_decoder.embedding_layer_norm")?,
            blocks,
            final_norm: load_norm(file, "transf_decoder.final_layer_norm")?,
            output_proj: Linear::load(file, "head.classifier", scheme)?,
            d_model: cfg.hidden_size,
            heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            vocab_size,
            max_sequence_length: cfg.max_sequence_length,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embeds one token row.
    fn embed(&self, token: i32) -> Result<Vec<f32>> {
        let token = usize::try_from(token).map_err(|_| SpeechError::Input {
            why: "Canary token ids must be non-negative".into(),
        })?;
        if token >= self.vocab_size {
            return Err(SpeechError::Input {
                why: format!(
                    "token id {token} exceeds the vocabulary of {}",
                    self.vocab_size
                ),
            });
        }
        Ok(self.embedding[token * self.d_model..(token + 1) * self.d_model].to_vec())
    }

    /// Computes the fixed cross-attention keys and values over the encoder
    /// output (`[frames, d_model]`, `frames` rows valid) for every layer.
    /// This mirrors the reference's lazy first-call cross cache.
    pub fn new_cache(&self, encoder_output: &[f32], encoder_len: usize) -> Result<Cache> {
        if encoder_len == 0 || encoder_output.len() != encoder_len * self.d_model {
            return Err(SpeechError::Input {
                why: "Canary encoder output must be [frames, d_model] with frames >= 1".into(),
            });
        }
        let layers = self
            .blocks
            .iter()
            .map(|block| {
                let k = block
                    .cross_attn
                    .k
                    .forward(encoder_output, encoder_len, self.d_model);
                let v = block
                    .cross_attn
                    .v
                    .forward(encoder_output, encoder_len, self.d_model);
                LayerCache {
                    self_k: Vec::new(),
                    self_v: Vec::new(),
                    self_len: 0,
                    cross_k: k,
                    cross_v: v,
                }
            })
            .collect();
        Ok(Cache {
            layers,
            encoder_len,
        })
    }

    /// Runs the decoder over `token_ids` starting at absolute position
    /// `start_pos`, appending to `cache`, and returns the logits
    /// `[token_ids.len(), vocab_size]` plus the post-`final_norm` hidden
    /// rows `[token_ids.len(), d_model]`. `positional` is the fixed
    /// sinusoidal table (`[position, d_model]`).
    pub fn forward(
        &self,
        token_ids: &[i32],
        start_pos: usize,
        cache: &mut Cache,
        positional: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = token_ids.len();
        if t == 0 {
            return Err(SpeechError::Input {
                why: "Canary decoder needs at least one token".into(),
            });
        }
        if start_pos + t > self.max_sequence_length {
            return Err(SpeechError::Input {
                why: format!(
                    "Canary decode position {} exceeds the positional table of {} entries",
                    start_pos + t,
                    self.max_sequence_length
                ),
            });
        }
        let d = self.d_model;
        let mut x = Vec::with_capacity(t * d);
        for (i, &token) in token_ids.iter().enumerate() {
            x.extend_from_slice(&self.embed(token)?);
            let row = (start_pos + i) * d;
            let pos = &positional[row..row + d];
            for (value, p) in x[(i * d)..((i + 1) * d)].iter_mut().zip(pos) {
                *value += p;
            }
        }
        ops::layernorm(
            &mut x,
            t,
            d,
            &self.embedding_layer_norm.0,
            Some(&self.embedding_layer_norm.1),
            1e-5,
        );

        for (block, layer) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            // Self attention with a growing causal KV cache.
            let mut normed = x.clone();
            ops::layernorm(
                &mut normed,
                t,
                d,
                &block.self_attn_norm.0,
                Some(&block.self_attn_norm.1),
                1e-5,
            );
            let q = block.self_attn.q.forward(&normed, t, d);
            let k = block.self_attn.k.forward(&normed, t, d);
            let v = block.self_attn.v.forward(&normed, t, d);
            layer.self_k.extend_from_slice(&k);
            layer.self_v.extend_from_slice(&v);
            layer.self_len += t;
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let attended = self_attention(
                &q,
                &layer.self_k,
                &layer.self_v,
                t,
                layer.self_len,
                d,
                self.heads,
                self.head_dim,
                scale,
            );
            let attended = block.self_attn.out.forward(&attended, t, d);
            for (residual, value) in x.iter_mut().zip(&attended) {
                *residual += value;
            }
            // Cross attention over the fixed encoder states.
            let mut normed = x.clone();
            ops::layernorm(
                &mut normed,
                t,
                d,
                &block.cross_attn_norm.0,
                Some(&block.cross_attn_norm.1),
                1e-5,
            );
            let q = block.cross_attn.q.forward(&normed, t, d);
            let attended = cross_attention(
                &q,
                &layer.cross_k,
                &layer.cross_v,
                cache.encoder_len,
                t,
                d,
                self.heads,
                self.head_dim,
                scale,
            );
            let attended = block.cross_attn.out.forward(&attended, t, d);
            for (residual, value) in x.iter_mut().zip(&attended) {
                *residual += value;
            }
            // ReLU feed-forward.
            let mut normed = x.clone();
            ops::layernorm(
                &mut normed,
                t,
                d,
                &block.ff_norm.0,
                Some(&block.ff_norm.1),
                1e-5,
            );
            let hidden = block.ff1.forward(&normed, t, d);
            let hidden = {
                let mut relu = hidden;
                for value in relu.iter_mut() {
                    *value = value.max(0.0);
                }
                relu
            };
            let ff = block.ff2.forward(&hidden, t, block.ff1.w.len() / d);
            for (residual, value) in x.iter_mut().zip(&ff) {
                *residual += value;
            }
        }

        ops::layernorm(
            &mut x,
            t,
            d,
            &self.final_norm.0,
            Some(&self.final_norm.1),
            1e-5,
        );
        let logits = self.output_proj.forward(&x, t, d);
        Ok((logits, x))
    }
}

/// The additive mask standing in for masked attention scores. The reference
/// uses -1e9, whose exponent underflows to exactly zero in the f32 softmax.
const MASK: f32 = -1.0e9;

/// Scaled dot-product self attention over a causal KV cache. `q` is
/// `[t, d]`, `k`/`v` are `[total, d]` with `total >= t`; query row i may
/// attend to keys up to its absolute position.
#[allow(clippy::too_many_arguments)]
fn self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    total: usize,
    d: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    attention(
        q,
        k,
        v,
        t,
        total,
        d,
        heads,
        head_dim,
        scale,
        &|query_index, key_index| {
            // Queries from the current step occupy the last `t` cache
            // slots; causal masking applies within the prefill block.
            let absolute = total - t + query_index;
            if key_index > absolute {
                MASK
            } else {
                0.0
            }
        },
    )
}

/// Scaled dot-product cross attention over the encoder states; keys at or
/// past the valid frame count receive the additive mask.
#[allow(clippy::too_many_arguments)]
fn cross_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    valid: usize,
    t: usize,
    d: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let total = k.len() / d;
    attention(
        q,
        k,
        v,
        t,
        total,
        d,
        heads,
        head_dim,
        scale,
        &|_, key_index| {
            if key_index >= valid {
                MASK
            } else {
                0.0
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    total: usize,
    d: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    mask: &dyn Fn(usize, usize) -> f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; t * d];
    let mut scores = vec![0.0f32; total];
    for hh in 0..heads {
        for tt in 0..t {
            let q_row = &q[tt * d + hh * head_dim..tt * d + (hh + 1) * head_dim];
            for (ss, score) in scores.iter_mut().enumerate() {
                let k_row = &k[ss * d + hh * head_dim..ss * d + (hh + 1) * head_dim];
                let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                *score = dot * scale + mask(tt, ss);
            }
            ops::softmax_row(&mut scores);
            for (ss, &weight) in scores.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let v_row = &v[ss * d + hh * head_dim..ss * d + (hh + 1) * head_dim];
                for (j, out_value) in out[tt * d + hh * head_dim..tt * d + (hh + 1) * head_dim]
                    .iter_mut()
                    .enumerate()
                {
                    *out_value += weight * v_row[j];
                }
            }
        }
    }
    out
}

/// Builds the fixed sinusoidal positional table of
/// `FixedPositionalEncoding`: interleaved sin/cos divided by
/// `sqrt(d_model)`, `[max_len, d_model]` row-major.
pub fn fixed_positional_encoding(d_model: usize, max_len: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; max_len * d_model];
    let divisor = (-(10_000f64.ln()) / d_model as f64) as f32;
    let div_term: Vec<f32> = (0..d_model / 2)
        .map(|j| ((2 * j) as f32 * divisor).exp())
        .collect();
    // The reference divides the f32 table by sqrt(d_model); for every
    // supported width that scale is a power of two, so the f32 multiply by
    // its reciprocal is the same rounding as the reference's division.
    let scale = 1.0f32 / (d_model as f32).sqrt();
    for position in 0..max_len {
        for (j, div) in div_term.iter().enumerate() {
            let angle = position as f32 * div;
            table[position * d_model + 2 * j] = angle.sin() * scale;
            table[position * d_model + 2 * j + 1] = angle.cos() * scale;
        }
    }
    table
}
