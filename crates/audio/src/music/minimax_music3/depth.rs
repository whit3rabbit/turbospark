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

/// Logical dtype of every stage of one block, shared by the per-operation
/// path and the device sequence so both round identically.
#[derive(Clone, Copy)]
struct BlockDtypes {
    /// Input norm, and the q/k/v projection input.
    norm: DType,
    q: DType,
    k: DType,
    v: DType,
    /// Attention, and the output projection input.
    attention: DType,
    residual: DType,
    /// Post-attention norm, and the gate/up projection input.
    post_norm: DType,
    gate: DType,
    /// SwiGLU, and the down projection input.
    fused: DType,
    output: DType,
}

impl BlockDtypes {
    fn all(&self, dtype: DType) -> bool {
        [
            self.norm,
            self.q,
            self.k,
            self.v,
            self.attention,
            self.residual,
            self.post_norm,
            self.gate,
            self.fused,
            self.output,
        ]
        .iter()
        .all(|d| *d == dtype)
    }
}

impl DepthBlock {
    fn dtypes(&self, input: DType) -> BlockDtypes {
        let norm = input.promote(self.input_ln_w.dtype);
        let (q, k, v) = (
            self.attn.to_q.output_dtype(norm),
            self.attn.to_k.output_dtype(norm),
            self.attn.to_v.output_dtype(norm),
        );
        let attention = q.promote(k).promote(v);
        let residual = input.promote(self.attn.to_out.output_dtype(attention));
        let post_norm = residual.promote(self.post_ln_w.dtype);
        let gate = self.gate_w.output_dtype(post_norm);
        let fused = gate.promote(self.up_w.output_dtype(post_norm));
        BlockDtypes {
            norm,
            q,
            k,
            v,
            attention,
            residual,
            post_norm,
            gate,
            fused,
            output: residual.promote(self.down_w.output_dtype(fused)),
        }
    }
    fn output_dtype(&self, input: DType) -> DType {
        self.dtypes(input).output
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
    #[cfg(test)]
    pub(crate) fn forward_typed(
        &self,
        embeddings: &[f32],
        batch: usize,
        len: usize,
        input_dtype: DType,
    ) -> Result<Vec<f32>> {
        let (h, dtype) = self.position(embeddings, batch, len, input_dtype)?;
        self.layers(h, batch, len, dtype)
    }

    /// Validate `[batch, len, hidden]` embeddings and add the positional
    /// embedding on the host. Returns the sum and its dtype.
    fn position(
        &self,
        embeddings: &[f32],
        batch: usize,
        len: usize,
        input_dtype: DType,
    ) -> Result<(Vec<f32>, DType)> {
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
        let dtype = input_dtype.promote(self.pos_embedding.dtype);
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
        Ok((h, dtype))
    }

    /// One depth step: the forward over `[batch, len, hidden]` embeddings,
    /// then audio head `head` on each row's last position, one row at a time
    /// (a rows=1 projection, as the reference does per CFG row). Returns the
    /// last-position hiddens `[batch, hidden]` and logits `[batch, vocab]`.
    pub(crate) fn step(
        &self,
        embeddings: &[f32],
        batch: usize,
        len: usize,
        input_dtype: DType,
        head: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (h, dtype) = self.position(embeddings, batch, len, input_dtype)?;
        if let Some(step) = self.step_sequence(&h, batch, len, dtype, head)? {
            return Ok(step);
        }
        let hidden = self.dims.hidden;
        let out = self.layers(h, batch, len, dtype)?;
        let head_dtype = self.output_dtype(input_dtype);
        let mut last_rows = Vec::with_capacity(batch * hidden);
        let mut logits = Vec::with_capacity(batch * self.dims.audio_vocab);
        for b in 0..batch {
            let base = (b * len + len - 1) * hidden;
            last_rows.extend_from_slice(&out[base..base + hidden]);
        }
        for row in last_rows.chunks_exact(hidden) {
            logits.extend(backend::linear(
                row,
                &self.audio_heads[head],
                None,
                1,
                hidden,
                self.dims.audio_vocab,
                head_dtype,
            )?);
        }
        Ok((last_rows, logits))
    }

    /// [`Self::step`] after the positional add as one device submission.
    /// `None` when the backend offers no sequence, a stage is not BF16 (the
    /// device elementwise steps reproduce the host BF16 rounding only), or
    /// the device met a case it cannot round like the host; the caller then
    /// runs the per-operation path, which gives the same values.
    fn step_sequence(
        &self,
        h: &[f32],
        batch: usize,
        len: usize,
        dtype: DType,
        head: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let Some(device) = &self.backend else {
            return Ok(None);
        };
        let mut stages = Vec::with_capacity(self.layers.len());
        let mut stage = dtype;
        for layer in &self.layers {
            let t = layer.dtypes(stage);
            stages.push(t);
            stage = t.output;
        }
        let final_dtype = stage.promote(self.norm_w.dtype);
        if dtype != DType::Bf16
            || final_dtype != DType::Bf16
            || !stages.iter().all(|t| t.all(DType::Bf16))
        {
            return Ok(None);
        }
        let Some(mut seq) = device.sequence("depth_step", &[batch, len]) else {
            return Ok(None);
        };
        let dims = &self.dims;
        let (hidden, heads, eps) = (dims.hidden, dims.heads, dims.eps);
        let rows = batch * len;
        let mut x = seq.upload(h)?;
        for (layer, t) in self.layers.iter().zip(&stages) {
            let normed = seq.rms_norm(x, &layer.input_ln_w, rows, hidden, eps, t.norm)?;
            let project = |seq: &mut dyn backend::Sequence, x, w: &Weight, inn, out, dtype| {
                backend::sequence_linear(seq, x, w, rows, inn, out, dtype)
            };
            let q = project(
                seq.as_mut(),
                normed,
                &layer.attn.to_q,
                hidden,
                hidden,
                t.norm,
            )?;
            let k = project(
                seq.as_mut(),
                normed,
                &layer.attn.to_k,
                hidden,
                hidden,
                t.norm,
            )?;
            let v = project(
                seq.as_mut(),
                normed,
                &layer.attn.to_v,
                hidden,
                hidden,
                t.norm,
            )?;
            let attention = seq.attention(
                q,
                k,
                v,
                AttentionShape {
                    batch,
                    queries: len,
                    keys: len,
                    heads,
                    kv_heads: heads,
                    dim: hidden / heads,
                    kv_time_major: false,
                    causal: true,
                    offset: 0,
                },
                t.attention,
            )?;
            let projected = project(
                seq.as_mut(),
                attention,
                &layer.attn.to_out,
                hidden,
                hidden,
                t.attention,
            )?;
            let residual = seq.add_round(x, projected, t.residual)?;
            let normed =
                seq.rms_norm(residual, &layer.post_ln_w, rows, hidden, eps, t.post_norm)?;
            let gate = project(
                seq.as_mut(),
                normed,
                &layer.gate_w,
                hidden,
                dims.intermediate,
                t.post_norm,
            )?;
            let up = project(
                seq.as_mut(),
                normed,
                &layer.up_w,
                hidden,
                dims.intermediate,
                t.post_norm,
            )?;
            let fused = seq.swiglu(gate, up, t.gate, t.fused)?;
            let down = project(
                seq.as_mut(),
                fused,
                &layer.down_w,
                dims.intermediate,
                hidden,
                t.fused,
            )?;
            x = seq.add_round(residual, down, t.output)?;
        }
        let normed = seq.rms_norm(x, &self.norm_w, rows, hidden, eps, final_dtype)?;
        let mut read = Vec::with_capacity(2 * batch);
        for b in 0..batch {
            read.push(seq.view(normed, (b * len + len - 1) * hidden, hidden)?);
        }
        for b in 0..batch {
            read.push(backend::sequence_linear(
                seq.as_mut(),
                read[b],
                &self.audio_heads[head],
                1,
                hidden,
                dims.audio_vocab,
                final_dtype,
            )?);
        }
        Ok(seq
            .finish(&read)?
            .map(|values| (values[..batch].concat(), values[batch..].concat())))
    }

    /// The blocks and the final norm, one device call per operation.
    fn layers(
        &self,
        mut h: Vec<f32>,
        batch: usize,
        len: usize,
        mut dtype: DType,
    ) -> Result<Vec<f32>> {
        let dims = &self.dims;
        let hidden = dims.hidden;
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
        let t = block.dtypes(input_dtype);
        let normed = backend::rms_norm(
            &self.backend,
            x,
            &block.input_ln_w,
            batch * len,
            hidden,
            eps,
            t.norm,
        )?;
        trace("input_norm", &normed, t.norm, &[batch, len, hidden]);
        let q = backend::linear(
            &normed,
            &block.attn.to_q,
            None,
            batch * len,
            hidden,
            hidden,
            t.norm,
        )?;
        trace("q", &q, t.q, &[batch, len, hidden]);
        let k = backend::linear(
            &normed,
            &block.attn.to_k,
            None,
            batch * len,
            hidden,
            hidden,
            t.norm,
        )?;
        trace("k", &k, t.k, &[batch, len, hidden]);
        let v = backend::linear(
            &normed,
            &block.attn.to_v,
            None,
            batch * len,
            hidden,
            hidden,
            t.norm,
        )?;
        trace("v", &v, t.v, &[batch, len, hidden]);
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Causal additive mask over the full sequence.
        let mask: Vec<f32> = (0..len)
            .flat_map(|qi| (0..len).map(move |ki| if ki <= qi { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let attn_out = if self.backend.is_some() || t.attention != DType::F32 {
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
                t.attention,
            )?;
            trace("attention", &rows, t.attention, &[batch, len, hidden]);
            backend::linear(
                &rows,
                &block.attn.to_out,
                None,
                batch * len,
                hidden,
                hidden,
                t.attention,
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
                    t.attention,
                )?);
            }
            attn_out
        };

        let mut residual = x.to_vec();
        for (r, a) in residual.iter_mut().zip(&attn_out) {
            *r = t.residual.round(*r + *a);
        }
        trace("residual", &residual, t.residual, &[batch, len, hidden]);
        let normed = backend::rms_norm(
            &self.backend,
            &residual,
            &block.post_ln_w,
            batch * len,
            hidden,
            eps,
            t.post_norm,
        )?;
        trace("post_norm", &normed, t.post_norm, &[batch, len, hidden]);
        let gate = backend::linear(
            &normed,
            &block.gate_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
            t.post_norm,
        )?;
        trace("gate", &gate, t.post_norm, &[batch, len, dims.intermediate]);
        let up = backend::linear(
            &normed,
            &block.up_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
            t.post_norm,
        )?;
        trace("up", &up, t.post_norm, &[batch, len, dims.intermediate]);
        let mut fused = gate;
        for (g, u) in fused.iter_mut().zip(up) {
            *g = t.fused.round(t.gate.silu(*g) * u);
        }
        trace("swiglu", &fused, t.fused, &[batch, len, dims.intermediate]);
        let down = backend::linear(
            &fused,
            &block.down_w,
            None,
            batch * len,
            dims.intermediate,
            hidden,
            t.fused,
        )?;
        trace("down", &down, t.fused, &[batch, len, hidden]);
        for (r, d) in residual.iter_mut().zip(down) {
            *r = t.output.round(*r + d);
        }
        trace("output", &residual, t.output, &[batch, len, hidden]);
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
