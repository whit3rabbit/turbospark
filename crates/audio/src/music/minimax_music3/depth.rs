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

use super::backend::{self, AttentionShape, ComputeBackend, Weight};
use super::precision::DType;
use super::weights::{Tensor, WeightStore};
use std::rc::Rc;

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
    to_q: Weight,
    to_k: Weight,
    to_v: Weight,
    to_out: Weight,
}

struct DepthBlock {
    input_ln_w: Tensor,
    post_ln_w: Tensor,
    attn: DepthAttention,
    gate_w: Weight,
    up_w: Weight,
    down_w: Weight,
}

impl DepthBlock {
    fn output_dtype(&self, input: DType) -> DType {
        let norm = input.promote(self.input_ln_w.dtype);
        let attention = self
            .attn
            .to_q
            .output_dtype(norm)
            .promote(self.attn.to_k.output_dtype(norm))
            .promote(self.attn.to_v.output_dtype(norm));
        let residual = input.promote(self.attn.to_out.output_dtype(attention));
        let norm = residual.promote(self.post_ln_w.dtype);
        let fused = self
            .gate_w
            .output_dtype(norm)
            .promote(self.up_w.output_dtype(norm));
        residual.promote(self.down_w.output_dtype(fused))
    }
}

pub(crate) struct DepthDecoder {
    /// `[audio_vocab * residual, hidden]` shared residual embeddings.
    pub audio_embeddings: Tensor,
    /// `[hidden, hidden]`, no bias.
    pub projection: Weight,
    pos_embedding: Tensor,
    layers: Vec<DepthBlock>,
    norm_w: Tensor,
    /// One `[audio_vocab, hidden]` head per residual codebook.
    pub audio_heads: Vec<Weight>,
    backend: Option<Rc<dyn ComputeBackend>>,
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
                input_ln_w,
                post_ln_w,
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
            audio_embeddings: embed,
            projection,
            pos_embedding: pos,
            layers,
            norm_w,
            audio_heads,
            backend: store.backend(),
            dims,
        })
    }

    pub(crate) fn dtype(&self) -> DType {
        self.audio_embeddings.dtype
    }
    pub(crate) fn output_dtype(&self, input: DType) -> DType {
        self.layers
            .iter()
            .fold(input.promote(self.pos_embedding.dtype), |d, b| {
                b.output_dtype(d)
            })
            .promote(self.norm_w.dtype)
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
    #[cfg(test)]
    pub(crate) fn forward(&self, embeddings: &[f32], batch: usize, len: usize) -> Result<Vec<f32>> {
        self.forward_typed(
            embeddings,
            batch,
            len,
            self.projection.output_dtype(self.dtype()),
        )
    }
    pub(crate) fn forward_typed(
        &self,
        embeddings: &[f32],
        batch: usize,
        len: usize,
        input_dtype: DType,
    ) -> Result<Vec<f32>> {
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
        let mut dtype = input_dtype.promote(self.pos_embedding.dtype);
        let mut h = vec![0.0f32; embeddings.len()];
        for b in 0..batch {
            for t in 0..len {
                let row = (b * len + t) * hidden;
                for d in 0..hidden {
                    h[row + d] =
                        dtype.round(embeddings[row + d] + self.pos_embedding[t * hidden + d]);
                }
            }
        }
        backend::trace(
            &self.backend,
            "depth.position",
            &h,
            dtype,
            &[batch, len, hidden],
        );
        for (layer_index, layer) in self.layers.iter().enumerate() {
            h = self.block_forward(layer, &h, batch, len, dtype, layer_index)?;
            dtype = layer.output_dtype(dtype);
        }
        h = backend::rms_norm(
            &self.backend,
            &h,
            &self.norm_w,
            batch * len,
            hidden,
            dims.eps,
            dtype.promote(self.norm_w.dtype),
        )?;
        backend::trace(
            &self.backend,
            "depth.final_norm",
            &h,
            dtype.promote(self.norm_w.dtype),
            &[batch, len, hidden],
        );
        Ok(h)
    }

    fn block_forward(
        &self,
        block: &DepthBlock,
        x: &[f32],
        batch: usize,
        len: usize,
        input_dtype: DType,
        layer_index: usize,
    ) -> Result<Vec<f32>> {
        let trace = |stage: &str, data: &[f32], dtype: DType, shape: &[usize]| {
            backend::trace(
                &self.backend,
                &format!("depth.{layer_index}.{stage}"),
                data,
                dtype,
                shape,
            )
        };
        let dims = &self.dims;
        let (hidden, heads) = (dims.hidden, dims.heads);
        let head_dim = hidden / heads;
        let eps = dims.eps;
        trace("input", x, input_dtype, &[batch, len, hidden]);
        let mut dtype = input_dtype.promote(block.input_ln_w.dtype);
        let normed = backend::rms_norm(
            &self.backend,
            x,
            &block.input_ln_w,
            batch * len,
            hidden,
            eps,
            dtype,
        )?;
        trace("input_norm", &normed, dtype, &[batch, len, hidden]);
        let q = backend::linear(
            &normed,
            &block.attn.to_q,
            None,
            batch * len,
            hidden,
            hidden,
            dtype,
        )?;
        trace(
            "q",
            &q,
            block.attn.to_q.output_dtype(dtype),
            &[batch, len, hidden],
        );
        let k = backend::linear(
            &normed,
            &block.attn.to_k,
            None,
            batch * len,
            hidden,
            hidden,
            dtype,
        )?;
        trace(
            "k",
            &k,
            block.attn.to_k.output_dtype(dtype),
            &[batch, len, hidden],
        );
        let v = backend::linear(
            &normed,
            &block.attn.to_v,
            None,
            batch * len,
            hidden,
            hidden,
            dtype,
        )?;
        trace(
            "v",
            &v,
            block.attn.to_v.output_dtype(dtype),
            &[batch, len, hidden],
        );
        dtype = block
            .attn
            .to_q
            .output_dtype(dtype)
            .promote(block.attn.to_k.output_dtype(dtype))
            .promote(block.attn.to_v.output_dtype(dtype));
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Causal additive mask over the full sequence.
        let mask: Vec<f32> = (0..len)
            .flat_map(|qi| (0..len).map(move |ki| if ki <= qi { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let attn_out = if self.backend.is_some() || dtype != DType::F32 {
            let rows = backend::attention(
                &self.backend,
                &q,
                &k,
                &v,
                AttentionShape {
                    batch,
                    queries: len,
                    keys: len,
                    heads,
                    kv_heads: heads,
                    dim: head_dim,
                    kv_time_major: false,
                    causal: true,
                    offset: 0,
                },
                dtype,
            )?;
            trace("attention", &rows, dtype, &[batch, len, hidden]);
            backend::linear(
                &rows,
                &block.attn.to_out,
                None,
                batch * len,
                hidden,
                hidden,
                dtype,
            )?
        } else {
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
                attn_out.extend(backend::linear(
                    &row_out,
                    &block.attn.to_out,
                    None,
                    len,
                    hidden,
                    hidden,
                    dtype,
                )?);
            }
            attn_out
        };

        let mut residual = x.to_vec();
        for (r, a) in residual.iter_mut().zip(&attn_out) {
            *r = input_dtype
                .promote(block.attn.to_out.output_dtype(dtype))
                .round(*r + *a);
        }
        trace(
            "residual",
            &residual,
            input_dtype.promote(block.attn.to_out.output_dtype(dtype)),
            &[batch, len, hidden],
        );
        dtype = input_dtype
            .promote(block.attn.to_out.output_dtype(dtype))
            .promote(block.post_ln_w.dtype);
        let normed = backend::rms_norm(
            &self.backend,
            &residual,
            &block.post_ln_w,
            batch * len,
            hidden,
            eps,
            dtype,
        )?;
        trace("post_norm", &normed, dtype, &[batch, len, hidden]);
        let gate = backend::linear(
            &normed,
            &block.gate_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
            dtype,
        )?;
        trace("gate", &gate, dtype, &[batch, len, dims.intermediate]);
        let up = backend::linear(
            &normed,
            &block.up_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
            dtype,
        )?;
        trace("up", &up, dtype, &[batch, len, dims.intermediate]);
        let mut fused = gate;
        for (g, u) in fused.iter_mut().zip(up) {
            let gate_dtype = block.gate_w.output_dtype(dtype);
            *g = gate_dtype
                .promote(block.up_w.output_dtype(dtype))
                .round(gate_dtype.silu(*g) * u);
        }
        dtype = block
            .gate_w
            .output_dtype(dtype)
            .promote(block.up_w.output_dtype(dtype));
        trace("swiglu", &fused, dtype, &[batch, len, dims.intermediate]);
        let down = backend::linear(
            &fused,
            &block.down_w,
            None,
            batch * len,
            dims.intermediate,
            hidden,
            dtype,
        )?;
        trace("down", &down, dtype, &[batch, len, hidden]);
        for (r, d) in residual.iter_mut().zip(down) {
            *r = block.output_dtype(input_dtype).round(*r + d);
        }
        trace(
            "output",
            &residual,
            block.output_dtype(input_dtype),
            &[batch, len, hidden],
        );
        Ok(residual)
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
