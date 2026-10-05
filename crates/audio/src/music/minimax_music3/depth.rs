//! RVQ depth decoder: the residual-codebook transformer that expands
//! each semantic code into `num_codebooks - 1` residual codes.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/depth.py`. The
//! sequence mixes projected hidden states and codebook embeddings;
//! attention is causal over the growing sequence (at most 2 +
//! residual codebooks positions, bounded by
//! `depth_max_position_embeddings`).

use crate::ops;
use crate::Result;
use crate::SpeechError;

use super::weights::WeightStore;

pub(crate) struct DepthDims {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub audio_vocab: usize,
    pub num_codebooks: usize,
    pub max_positions: usize,
}

struct DepthAttention {
    to_q: Vec<f32>,
    to_k: Vec<f32>,
    to_v: Vec<f32>,
    to_out: Vec<f32>,
}

struct DepthBlock {
    input_ln_w: Vec<f32>,
    post_ln_w: Vec<f32>,
    attn: DepthAttention,
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
}

pub(crate) struct DepthDecoder {
    /// `[audio_vocab * residual, hidden]` shared residual embeddings.
    pub audio_embeddings: Vec<f32>,
    /// `[hidden, hidden]`, no bias.
    pub projection: Vec<f32>,
    pos_embedding: Vec<f32>,
    layers: Vec<DepthBlock>,
    norm_w: Vec<f32>,
    /// One `[audio_vocab, hidden]` head per residual codebook.
    pub audio_heads: Vec<Vec<f32>>,
    dims: DepthDims,
}

impl DepthDecoder {
    pub(crate) fn load(
        store: &mut WeightStore,
        base: &str,
        dims: DepthDims,
    ) -> Result<DepthDecoder> {
        let hidden = dims.hidden;
        let residual = dims.num_codebooks - 1;
        let embed = store.tensor(&format!("{base}.audio_embeddings.weight"))?;
        if embed.data.len() != dims.audio_vocab * residual * hidden {
            return Err(SpeechError::Tensor {
                name: format!("{base}.audio_embeddings.weight"),
                why: format!(
                    "expected {}x{}x{}, got shape {:?}",
                    dims.audio_vocab, residual, hidden, embed.shape
                ),
            });
        }
        let (projection, _) = store.linear(&format!("{base}.projection"))?;
        let pos = store.tensor(&format!("{base}.pos_embedding.weight"))?;
        if pos.data.len() != dims.max_positions * hidden {
            return Err(SpeechError::Tensor {
                name: format!("{base}.pos_embedding.weight"),
                why: format!("expected {}x{}", dims.max_positions, hidden),
            });
        }
        let norm_w = store.tensor(&format!("{base}.norm.weight"))?;
        let mut layers = Vec::with_capacity(dims.layers);
        for index in 0..dims.layers {
            let layer = format!("{base}.layers.{index}");
            let (to_q, _) = store.linear(&format!("{layer}.attn.to_q"))?;
            let (to_k, _) = store.linear(&format!("{layer}.attn.to_k"))?;
            let (to_v, _) = store.linear(&format!("{layer}.attn.to_v"))?;
            let (to_out, _) = store.linear(&format!("{layer}.attn.to_out"))?;
            for (name, w) in [
                ("attn.to_q", &to_q),
                ("attn.to_k", &to_k),
                ("attn.to_v", &to_v),
                ("attn.to_out", &to_out),
            ] {
                if w.len() != hidden * hidden {
                    return Err(SpeechError::Tensor {
                        name: format!("{layer}.{name}"),
                        why: format!("expected {hidden}x{hidden}"),
                    });
                }
            }
            let (gate_w, _) = store.linear(&format!("{layer}.gate_proj"))?;
            let (up_w, _) = store.linear(&format!("{layer}.up_proj"))?;
            let (down_w, _) = store.linear(&format!("{layer}.down_proj"))?;
            let input_ln_w = store.tensor(&format!("{layer}.input_layernorm.weight"))?;
            let post_ln_w = store.tensor(&format!("{layer}.post_attention_layernorm.weight"))?;
            layers.push(DepthBlock {
                input_ln_w: input_ln_w.data,
                post_ln_w: post_ln_w.data,
                attn: DepthAttention {
                    to_q,
                    to_k,
                    to_v,
                    to_out,
                },
                gate_w,
                up_w,
                down_w,
            });
        }
        let mut audio_heads = Vec::with_capacity(residual);
        for index in 0..residual {
            let (head, _) = store.linear(&format!("{base}.audio_heads.{index}"))?;
            if head.len() != dims.audio_vocab * hidden {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.audio_heads.{index}"),
                    why: format!("expected {}x{}", dims.audio_vocab, hidden),
                });
            }
            audio_heads.push(head);
        }
        Ok(DepthDecoder {
            audio_embeddings: embed.data,
            projection,
            pos_embedding: pos.data,
            layers,
            norm_w: norm_w.data,
            audio_heads,
            dims,
        })
    }

    /// Embedding lookup in the flat residual table: row
    /// `code + codebook * audio_vocab`.
    pub(crate) fn embed_code(&self, code: usize, codebook: usize) -> &[f32] {
        let hidden = self.dims.hidden;
        let row = code + codebook * self.dims.audio_vocab;
        &self.audio_embeddings[row * hidden..(row + 1) * hidden]
    }

    /// Forward over `[batch, len, hidden]` embeddings; returns the same
    /// layout after the final norm.
    pub(crate) fn forward(&self, embeddings: &[f32], batch: usize, len: usize) -> Result<Vec<f32>> {
        let dims = &self.dims;
        let hidden = dims.hidden;
        if embeddings.len() != batch * len * hidden {
            return Err(SpeechError::Input {
                why: format!(
                    "depth input {} does not match {batch}x{len}x{hidden}",
                    embeddings.len()
                ),
            });
        }
        if len > dims.max_positions {
            return Err(SpeechError::Input {
                why: format!(
                    "depth sequence {len} exceeds max positions {}",
                    dims.max_positions
                ),
            });
        }
        let mut h = vec![0.0f32; embeddings.len()];
        for b in 0..batch {
            for t in 0..len {
                let row = (b * len + t) * hidden;
                for d in 0..hidden {
                    h[row + d] = embeddings[row + d] + self.pos_embedding[t * hidden + d];
                }
            }
        }
        for layer in &self.layers {
            h = self.block_forward(layer, &h, batch, len);
        }
        ops::rmsnorm(&mut h, batch * len, hidden, &self.norm_w, dims.eps);
        Ok(h)
    }

    fn block_forward(&self, block: &DepthBlock, x: &[f32], batch: usize, len: usize) -> Vec<f32> {
        let dims = &self.dims;
        let (hidden, heads) = (dims.hidden, dims.heads);
        let head_dim = hidden / heads;
        let eps = dims.eps;
        let mut normed = x.to_vec();
        ops::rmsnorm(&mut normed, batch * len, hidden, &block.input_ln_w, eps);
        let q = ops::linear(&normed, &block.attn.to_q, None, batch * len, hidden, hidden);
        let k = ops::linear(&normed, &block.attn.to_k, None, batch * len, hidden, hidden);
        let v = ops::linear(&normed, &block.attn.to_v, None, batch * len, hidden, hidden);
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Causal additive mask over the full sequence.
        let mask: Vec<f32> = (0..len)
            .flat_map(|qi| (0..len).map(move |ki| if ki <= qi { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let mut attn_out = Vec::with_capacity(x.len());
        for b in 0..batch {
            let mut row_out = vec![0.0f32; len * hidden];
            for h in 0..heads {
                let q_head = plane(&q, b, len, hidden, h, head_dim);
                let k_head = plane(&k, b, len, hidden, h, head_dim);
                let v_head = plane(&v, b, len, hidden, h, head_dim);
                let out = ops::sdpa(
                    &q_head,
                    &k_head,
                    &v_head,
                    Some(&mask),
                    len,
                    len,
                    head_dim,
                    head_dim,
                    scale,
                );
                for t in 0..len {
                    let dst = t * hidden + h * head_dim;
                    row_out[dst..dst + head_dim]
                        .copy_from_slice(&out[t * head_dim..(t + 1) * head_dim]);
                }
            }
            attn_out.extend(ops::linear(
                &row_out,
                &block.attn.to_out,
                None,
                len,
                hidden,
                hidden,
            ));
        }
        let mut residual = x.to_vec();
        for (r, a) in residual.iter_mut().zip(&attn_out) {
            *r += *a;
        }
        let mut normed = residual.clone();
        ops::rmsnorm(&mut normed, batch * len, hidden, &block.post_ln_w, eps);
        let gate = ops::linear(
            &normed,
            &block.gate_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
        );
        let up = ops::linear(
            &normed,
            &block.up_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
        );
        let mut fused = gate;
        for (g, u) in fused.iter_mut().zip(up) {
            let silu = *g / (1.0 + (-*g).exp());
            *g = silu * u;
        }
        let down = ops::linear(
            &fused,
            &block.down_w,
            None,
            batch * len,
            dims.intermediate,
            hidden,
        );
        for (r, d) in residual.iter_mut().zip(down) {
            *r += d;
        }
        residual
    }
}

/// Copy head `h` out of a `[batch, len, hidden]` buffer with
/// `hidden / head_dim` heads, as a contiguous `[len, head_dim]`.
fn plane(x: &[f32], b: usize, len: usize, hidden: usize, h: usize, head_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * head_dim);
    for t in 0..len {
        let base = (b * len + t) * hidden + h * head_dim;
        out.extend_from_slice(&x[base..base + head_dim]);
    }
    out
}
