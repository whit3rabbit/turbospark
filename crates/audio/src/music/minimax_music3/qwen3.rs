//! Qwen3 dense transformer used as the MiniMax Music 3 AR backbone.
//!
//! Reference: `mlx_audio/lm/models/qwen3.py` at the pinned commit,
//! driven the way `minimax_music3/ar.py` drives it: embedding prefill,
//! per-frame feedback steps through a shared KV cache, QK RMSNorm
//! before NeoX rope with the cache offset, and a plain lm head (the
//! checkpoint does not tie embeddings).
//!
//! All tensors are row-major f32. Projections are HF `[out, in]`
//! linears; the KV cache is time-major `[t, batch, kv_heads, head_dim]`
//! and only grows by tail append, which is what the batch-2 conditional
//! / unconditional pair requires.

use crate::ops;
use crate::Result;
use crate::SpeechError;

use super::backend::{self, AttentionShape, ComputeBackend, DeviceBuffer, Weight};
use super::precision::DType;
use super::weights::{Tensor, WeightStore};
use std::rc::Rc;

pub(crate) struct Qwen3Dims {
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
    /// LM head rows a device keeps resident (see `window_logits`).
    pub logit_rows: std::ops::Range<usize>,
}

struct Attention {
    q_w: Weight,
    k_w: Weight,
    v_w: Weight,
    o_w: Weight,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
}

struct Mlp {
    gate_w: Weight,
    up_w: Weight,
    down_w: Weight,
}

struct Block {
    attn: Attention,
    mlp: Mlp,
    ln1_w: Tensor,
    ln2_w: Tensor,
}

/// Logical dtype of every stage of one block, shared by the per-operation
/// path and the device sequence so both round identically.
#[derive(Clone, Copy)]
struct BlockDtypes {
    /// Input norm, and the q/k/v projection input.
    norm: DType,
    q_proj: DType,
    k_proj: DType,
    v: DType,
    /// After the q norm (and its rope).
    q: DType,
    /// After the k norm (and its rope).
    k: DType,
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
            self.q_proj,
            self.k_proj,
            self.v,
            self.q,
            self.k,
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

impl Block {
    fn dtypes(&self, input: DType) -> BlockDtypes {
        let norm = input.promote(self.ln1_w.dtype);
        let (q_proj, k_proj, v) = (
            self.attn.q_w.output_dtype(norm),
            self.attn.k_w.output_dtype(norm),
            self.attn.v_w.output_dtype(norm),
        );
        let q = q_proj.promote(self.attn.q_norm_w.dtype);
        let k = k_proj.promote(self.attn.k_norm_w.dtype);
        let attention = q.promote(k).promote(v);
        let residual = input.promote(self.attn.o_w.output_dtype(attention));
        let post_norm = residual.promote(self.ln2_w.dtype);
        let gate = self.mlp.gate_w.output_dtype(post_norm);
        let fused = gate.promote(self.mlp.up_w.output_dtype(post_norm));
        BlockDtypes {
            norm,
            q_proj,
            k_proj,
            v,
            q,
            k,
            attention,
            residual,
            post_norm,
            gate,
            fused,
            output: residual.promote(self.mlp.down_w.output_dtype(fused)),
        }
    }
    fn output_dtype(&self, input: DType) -> DType {
        self.dtypes(input).output
    }
}

/// One layer's key/value history, time-major `[t, batch, kv_heads, head_dim]`.
pub(crate) struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    offset: usize,
    /// Positions a device copy is sized for: the prompt plus every decode
    /// step of the request. Zero keeps the cache on the host.
    capacity: usize,
    /// The cache after a decode step moved it to the device. The host
    /// vectors are empty until a per-operation step brings it back.
    device: Option<DeviceKv>,
}

struct DeviceKv {
    keys: Rc<dyn DeviceBuffer>,
    values: Rc<dyn DeviceBuffer>,
    /// Elements per position (`batch * kv_heads * head_dim`).
    row: usize,
}

impl KvCache {
    #[cfg(test)]
    pub(crate) fn new() -> KvCache {
        Self::with_capacity(0)
    }
    pub(crate) fn with_capacity(positions: usize) -> KvCache {
        KvCache {
            keys: Vec::new(),
            values: Vec::new(),
            offset: 0,
            capacity: positions,
            device: None,
        }
    }
    /// Bring a device cache back to the host before a per-operation step.
    /// Only the first `offset` positions are valid; a failed submission may
    /// have written the next one.
    fn return_to_host(&mut self) -> Result<()> {
        if let Some(device) = self.device.take() {
            let len = self.offset * device.row;
            self.keys = device.keys.read(0, len)?;
            self.values = device.values.read(0, len)?;
        }
        Ok(())
    }
}

pub(crate) struct Qwen3 {
    embed: Weight, // [vocab, hidden]
    /// The full `[vocab, hidden]` head. Absent when a device holds only
    /// `lm_window`.
    lm_head: Option<Weight>,
    /// `(first row, [rows, hidden])`: the head rows sampling can read.
    lm_window: Option<(usize, Weight)>,
    norm_w: Tensor,
    blocks: Vec<Block>,
    backend: Option<Rc<dyn ComputeBackend>>,
    dims: Qwen3Dims,
}

fn expect_len(name: &str, actual: usize, want: usize) -> Result<()> {
    if actual != want {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("element count {actual} does not match {want}"),
        });
    }
    Ok(())
}

impl Qwen3 {
    pub(crate) fn load(store: &mut WeightStore, base: &str, dims: Qwen3Dims) -> Result<Qwen3> {
        let hidden = dims.hidden;
        let q_kv = dims.kv_heads * dims.head_dim;
        let q_out = dims.heads * dims.head_dim;
        let embed = store.embedding(&format!("{base}.model.embed_tokens.weight"))?;
        expect_len("embed_tokens", embed.len(), dims.vocab * hidden)?;
        let window = if dims.tie_embeddings {
            None
        } else {
            store.device_linear_rows(
                &format!("{base}.lm_head"),
                dims.logit_rows.clone(),
                [dims.vocab, hidden],
            )?
        };
        let (lm_head, lm_window) = match window {
            Some(window) => (None, Some((dims.logit_rows.start, window))),
            None if dims.tie_embeddings => (Some(embed.clone()), None),
            None => {
                let (head, _) = store.linear(&format!("{base}.lm_head"))?;
                expect_len("lm_head", head.len(), dims.vocab * hidden)?;
                (Some(head), None)
            }
        };
        let norm_w = store.tensor(&format!("{base}.model.norm.weight"))?;
        expect_len("model.norm", norm_w.data.len(), hidden)?;
        let mut blocks = Vec::with_capacity(dims.layers);
        for index in 0..dims.layers {
            let layer = format!("{base}.model.layers.{index}");
            let (q_w, _) = store.linear(&format!("{layer}.self_attn.q_proj"))?;
            let (k_w, _) = store.linear(&format!("{layer}.self_attn.k_proj"))?;
            let (v_w, _) = store.linear(&format!("{layer}.self_attn.v_proj"))?;
            let (o_w, _) = store.linear(&format!("{layer}.self_attn.o_proj"))?;
            expect_len("q_proj", q_w.len(), q_out * hidden)?;
            expect_len("k_proj", k_w.len(), q_kv * hidden)?;
            expect_len("v_proj", v_w.len(), q_kv * hidden)?;
            expect_len("o_proj", o_w.len(), hidden * q_out)?;
            let q_norm_w = store.tensor(&format!("{layer}.self_attn.q_norm.weight"))?;
            let k_norm_w = store.tensor(&format!("{layer}.self_attn.k_norm.weight"))?;
            let (gate_w, _) = store.linear(&format!("{layer}.mlp.gate_proj"))?;
            let (up_w, _) = store.linear(&format!("{layer}.mlp.up_proj"))?;
            let (down_w, _) = store.linear(&format!("{layer}.mlp.down_proj"))?;
            expect_len("gate_proj", gate_w.len(), dims.intermediate * hidden)?;
            expect_len("up_proj", up_w.len(), dims.intermediate * hidden)?;
            expect_len("down_proj", down_w.len(), hidden * dims.intermediate)?;
            let ln1_w = store.tensor(&format!("{layer}.input_layernorm.weight"))?;
            let ln2_w = store.tensor(&format!("{layer}.post_attention_layernorm.weight"))?;
            expect_len("input_layernorm", ln1_w.data.len(), hidden)?;
            expect_len("post_attention_layernorm", ln2_w.data.len(), hidden)?;
            blocks.push(Block {
                attn: Attention {
                    q_w,
                    k_w,
                    v_w,
                    o_w,
                    q_norm_w,
                    k_norm_w,
                },
                mlp: Mlp {
                    gate_w,
                    up_w,
                    down_w,
                },
                ln1_w,
                ln2_w,
            });
        }
        Ok(Qwen3 {
            embed,
            lm_head,
            lm_window,
            norm_w,
            blocks,
            backend: store.backend(),
            dims,
        })
    }

    pub(crate) fn dtype(&self) -> DType {
        self.embed.dtype()
    }

    pub(crate) fn output_dtype(&self) -> DType {
        self.blocks
            .iter()
            .fold(self.dtype(), |d, b| b.output_dtype(d))
            .promote(self.norm_w.dtype)
    }

    /// Token embedding lookup for `[batch, len]` ids.
    pub(crate) fn embed_ids(&self, ids: &[i32], batch: usize, len: usize) -> Result<Vec<f32>> {
        if ids.len() != batch * len {
            return Err(SpeechError::Input {
                why: format!("ids len {} does not match {batch}x{len}", ids.len()),
            });
        }
        self.embed.embedding(ids, self.dims.hidden)
    }

    /// The transformer body on precomputed embeddings; returns
    /// `[batch, len, hidden]` after the final norm, updating `cache`.
    pub(crate) fn hidden_forward(
        &self,
        embeddings: &[f32],
        batch: usize,
        len: usize,
        cache: &mut [KvCache],
    ) -> Result<Vec<f32>> {
        let dims = &self.dims;
        let hidden = dims.hidden;
        if embeddings.len() != batch * len * hidden {
            return Err(SpeechError::Input {
                why: format!(
                    "embeddings {} do not match {batch}x{len}x{hidden}",
                    embeddings.len()
                ),
            });
        }
        if cache.len() != self.blocks.len() {
            return Err(SpeechError::Input {
                why: "cache length does not match layer count".to_string(),
            });
        }
        // Prefill (offset 0, len > 1) is causal; decode steps are not
        // masked, matching create_attention_mask in the reference.
        let causal = cache.first().is_some_and(|c| c.offset == 0) && len > 1;
        if len == 1 {
            if let Some(h) = self.decode_sequence(embeddings, batch, cache)? {
                return Ok(h);
            }
        }
        for layer_cache in cache.iter_mut() {
            layer_cache.return_to_host()?;
        }
        let mut dtype = self.dtype();
        let mut h = embeddings.to_vec();
        for (layer_index, (block, layer_cache)) in
            self.blocks.iter().zip(cache.iter_mut()).enumerate()
        {
            h = self.block_forward(
                block,
                layer_cache,
                &h,
                batch,
                len,
                causal,
                dtype,
                layer_index,
            )?;
            dtype = block.output_dtype(dtype);
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
            "ar.final_norm",
            &h,
            dtype.promote(self.norm_w.dtype),
            &[batch, len, hidden],
        );
        Ok(h)
    }

    /// One decode step (`len` 1) through every layer and the final norm as
    /// a single device submission, with the KV cache resident on the device.
    /// It runs the per-operation path's device kernels in the same order;
    /// only the residual adds and SwiGLU move from the host, and the device
    /// forms reproduce host BF16 rounding. `None` when the backend has no
    /// sequences, a stage is not BF16, the cache is full, or the device met a
    /// case it cannot round like the host. Offsets advance only after a
    /// successful submission, so the caller can rerun the step per operation.
    fn decode_sequence(
        &self,
        embeddings: &[f32],
        batch: usize,
        cache: &mut [KvCache],
    ) -> Result<Option<Vec<f32>>> {
        let Some(device) = &self.backend else {
            return Ok(None);
        };
        let dims = &self.dims;
        let (hidden, heads, kv_heads, head_dim, eps) = (
            dims.hidden,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            dims.eps,
        );
        let row = batch * kv_heads * head_dim;
        let mut stages = Vec::with_capacity(self.blocks.len());
        let mut stage = self.dtype();
        for block in &self.blocks {
            let t = block.dtypes(stage);
            stages.push(t);
            stage = t.output;
        }
        let final_dtype = stage.promote(self.norm_w.dtype);
        let offset = cache.first().map_or(0, |c| c.offset);
        if self.dtype() != DType::Bf16
            || final_dtype != DType::Bf16
            || !stages.iter().all(|t| t.all(DType::Bf16))
            || cache
                .iter()
                .any(|c| c.offset != offset || offset >= c.capacity)
            || cache
                .iter()
                .any(|c| c.device.as_ref().is_some_and(|d| d.row != row))
        {
            return Ok(None);
        }
        let Some(mut seq) = device.sequence("lm_decode_step", &[batch, offset]) else {
            return Ok(None);
        };
        // Move each layer's history to the device once per request.
        for layer_cache in cache.iter_mut() {
            if layer_cache.device.is_some() {
                continue;
            }
            let (Some(keys), Some(values)) = (
                device.buffer(layer_cache.capacity * row),
                device.buffer(layer_cache.capacity * row),
            ) else {
                return Ok(None);
            };
            let (keys, values) = (keys?, values?);
            keys.write(0, &layer_cache.keys)?;
            values.write(0, &layer_cache.values)?;
            layer_cache.keys = Vec::new();
            layer_cache.values = Vec::new();
            layer_cache.device = Some(DeviceKv { keys, values, row });
        }
        let total = offset + 1;
        let rope = |heads| backend::RopeShape {
            batch,
            seq: 1,
            heads,
            dim: head_dim,
            offset,
            theta: dims.rope_theta,
        };
        let mut x = seq.upload(embeddings)?;
        for ((block, t), layer_cache) in self.blocks.iter().zip(&stages).zip(cache.iter()) {
            let Some(kv) = &layer_cache.device else {
                return Ok(None);
            };
            let project = |seq: &mut dyn backend::Sequence, x, w: &Weight, inn, out, dtype| {
                backend::sequence_linear(seq, x, w, batch, inn, out, dtype)
            };
            let normed = seq.rms_norm(x, &block.ln1_w, batch, hidden, eps, t.norm)?;
            let q = project(
                seq.as_mut(),
                normed,
                &block.attn.q_w,
                hidden,
                heads * head_dim,
                t.norm,
            )?;
            let k = project(
                seq.as_mut(),
                normed,
                &block.attn.k_w,
                hidden,
                kv_heads * head_dim,
                t.norm,
            )?;
            let v = project(
                seq.as_mut(),
                normed,
                &block.attn.v_w,
                hidden,
                kv_heads * head_dim,
                t.norm,
            )?;
            let q = seq.rms_norm(q, &block.attn.q_norm_w, batch * heads, head_dim, eps, t.q)?;
            let k = seq.rms_norm(
                k,
                &block.attn.k_norm_w,
                batch * kv_heads,
                head_dim,
                eps,
                t.k,
            )?;
            let q = seq.rope(q, rope(heads), t.q)?;
            let k = seq.rope(k, rope(kv_heads), t.k)?;
            // A one-position step is already in the time-major row layout.
            let keys = seq.bind(kv.keys.as_ref())?;
            let values = seq.bind(kv.values.as_ref())?;
            let key_row = seq.view(keys, offset * row, row)?;
            let value_row = seq.view(values, offset * row, row)?;
            seq.copy(k, key_row)?;
            seq.copy(v, value_row)?;
            let keys = seq.view(keys, 0, total * row)?;
            let values = seq.view(values, 0, total * row)?;
            let attention = seq.attention(
                q,
                keys,
                values,
                AttentionShape {
                    batch,
                    queries: 1,
                    keys: total,
                    heads,
                    kv_heads,
                    dim: head_dim,
                    kv_time_major: true,
                    causal: false,
                    offset,
                },
                t.attention,
            )?;
            let projected = project(
                seq.as_mut(),
                attention,
                &block.attn.o_w,
                heads * head_dim,
                hidden,
                t.attention,
            )?;
            let residual = seq.add_round(x, projected, t.residual)?;
            let normed = seq.rms_norm(residual, &block.ln2_w, batch, hidden, eps, t.post_norm)?;
            let gate = project(
                seq.as_mut(),
                normed,
                &block.mlp.gate_w,
                hidden,
                dims.intermediate,
                t.post_norm,
            )?;
            let up = project(
                seq.as_mut(),
                normed,
                &block.mlp.up_w,
                hidden,
                dims.intermediate,
                t.post_norm,
            )?;
            let fused = seq.swiglu(gate, up, t.gate, t.fused)?;
            let down = project(
                seq.as_mut(),
                fused,
                &block.mlp.down_w,
                dims.intermediate,
                hidden,
                t.fused,
            )?;
            x = seq.add_round(residual, down, t.output)?;
        }
        let normed = seq.rms_norm(x, &self.norm_w, batch, hidden, eps, final_dtype)?;
        let Some(mut out) = seq.finish(&[normed])? else {
            return Ok(None);
        };
        for layer_cache in cache.iter_mut() {
            layer_cache.offset += 1;
        }
        Ok(Some(out.remove(0)))
    }

    /// Logits for `[rows, hidden]` hidden states.
    pub(crate) fn logits(&self, hidden_rows: &[f32]) -> Result<Vec<f32>> {
        let head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "the full LM head is not resident; use window_logits".to_string(),
            })?;
        self.head_logits(hidden_rows, head, self.dims.vocab)
    }

    /// `(first vocab id, width, [rows, width] logits)`. A device keeps only
    /// the window of head rows that sampling reads; otherwise this is the
    /// full head, `(0, vocab, ..)`.
    pub(crate) fn window_logits(&self, hidden_rows: &[f32]) -> Result<(usize, usize, Vec<f32>)> {
        match &self.lm_window {
            Some((start, head)) => {
                let width = self.dims.logit_rows.len();
                Ok((*start, width, self.head_logits(hidden_rows, head, width)?))
            }
            None => Ok((0, self.dims.vocab, self.logits(hidden_rows)?)),
        }
    }

    fn head_logits(&self, hidden_rows: &[f32], head: &Weight, width: usize) -> Result<Vec<f32>> {
        let dtype = self.output_dtype();
        let rows = hidden_rows.len() / self.dims.hidden;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty logits input".to_string(),
            });
        }
        backend::linear(
            hidden_rows,
            head,
            None,
            rows,
            self.dims.hidden,
            width,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        block: &Block,
        layer_cache: &mut KvCache,
        x: &[f32],
        batch: usize,
        len: usize,
        causal: bool,
        input_dtype: DType,
        layer_index: usize,
    ) -> Result<Vec<f32>> {
        let trace = |stage: &str, data: &[f32], dtype: DType, shape: &[usize]| {
            backend::trace(
                &self.backend,
                &format!("ar.{layer_index}.{stage}"),
                data,
                dtype,
                shape,
            )
        };
        let dims = &self.dims;
        let hidden = dims.hidden;
        let (heads, kv_heads, head_dim) = (dims.heads, dims.kv_heads, dims.head_dim);
        let eps = dims.eps;
        trace("input", x, input_dtype, &[batch, len, hidden]);
        let offset = layer_cache.offset;

        let t = block.dtypes(input_dtype);
        let normed = backend::rms_norm(
            &self.backend,
            x,
            &block.ln1_w,
            batch * len,
            hidden,
            eps,
            t.norm,
        )?;
        trace("input_norm", &normed, t.norm, &[batch, len, hidden]);

        let mut q = backend::linear(
            &normed,
            &block.attn.q_w,
            None,
            batch * len,
            hidden,
            heads * head_dim,
            t.norm,
        )?;
        trace("q", &q, t.q_proj, &[batch, len, heads * head_dim]);
        let mut k = backend::linear(
            &normed,
            &block.attn.k_w,
            None,
            batch * len,
            hidden,
            kv_heads * head_dim,
            t.norm,
        )?;
        trace("k", &k, t.k_proj, &[batch, len, kv_heads * head_dim]);
        let v = backend::linear(
            &normed,
            &block.attn.v_w,
            None,
            batch * len,
            hidden,
            kv_heads * head_dim,
            t.norm,
        )?;
        trace("v", &v, t.v, &[batch, len, kv_heads * head_dim]);

        let (q_dtype, k_dtype) = (t.q, t.k);
        q = backend::rms_norm(
            &self.backend,
            &q,
            &block.attn.q_norm_w,
            batch * len * heads,
            head_dim,
            eps,
            q_dtype,
        )?;
        trace("q_norm", &q, q_dtype, &[batch, len, heads, head_dim]);
        k = backend::rms_norm(
            &self.backend,
            &k,
            &block.attn.k_norm_w,
            batch * len * kv_heads,
            head_dim,
            eps,
            k_dtype,
        )?;
        trace("k_norm", &k, k_dtype, &[batch, len, kv_heads, head_dim]);
        if t.norm == DType::F32 {
            let (cos, sin) = ops::rope_tables_range(offset, len, head_dim, dims.rope_theta);
            rope_strided(&mut q, batch, len, heads, head_dim, &cos, &sin);
            rope_strided(&mut k, batch, len, kv_heads, head_dim, &cos, &sin);
        } else {
            let shape = |heads| backend::RopeShape {
                batch,
                seq: len,
                heads,
                dim: head_dim,
                offset,
                theta: dims.rope_theta,
            };
            q = match &self.backend {
                Some(b) => b.rope(&q, shape(heads), q_dtype)?,
                None => backend::rope(&q, shape(heads), q_dtype),
            };
            k = match &self.backend {
                Some(b) => b.rope(&k, shape(kv_heads), k_dtype)?,
                None => backend::rope(&k, shape(kv_heads), k_dtype),
            };
        }

        trace("q_rope", &q, q_dtype, &[batch, len, heads, head_dim]);
        trace("k_rope", &k, k_dtype, &[batch, len, kv_heads, head_dim]);
        append_kv(&mut layer_cache.keys, &k, batch, len, kv_heads, head_dim);
        append_kv(&mut layer_cache.values, &v, batch, len, kv_heads, head_dim);
        layer_cache.offset += len;
        let total = layer_cache.offset;

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Additive causal bias over the query window aligned to the end
        // of the key window; only built for the offset-0 prefill.
        let mask: Vec<f32> = if causal {
            (0..len)
                .flat_map(|qi| {
                    (0..total).map(move |ki| if ki <= qi { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect()
        } else {
            Vec::new()
        };
        let mask_ref = (!mask.is_empty()).then_some(mask.as_slice());

        let attn_out = if self.backend.is_some() || t.attention != DType::F32 {
            let rows = backend::attention(
                &self.backend,
                &q,
                &layer_cache.keys,
                &layer_cache.values,
                AttentionShape {
                    batch,
                    queries: len,
                    keys: total,
                    heads,
                    kv_heads,
                    dim: head_dim,
                    kv_time_major: true,
                    causal,
                    offset,
                },
                t.attention,
            )?;
            trace(
                "attention",
                &rows,
                t.attention,
                &[batch, len, heads * head_dim],
            );
            let projection = backend::linear(
                &rows,
                &block.attn.o_w,
                None,
                batch * len,
                heads * head_dim,
                hidden,
                t.attention,
            )?;
            x.iter()
                .zip(projection)
                .map(|(x, p)| t.residual.round(x + p))
                .collect::<Vec<_>>()
        } else {
            let mut attn_out = vec![0.0f32; batch * len * hidden];
            for b in 0..batch {
                // Attention reads the time-major cache in place through
                // strides: each (batch, kv_head) plane's rows are
                // `batch * kv_heads * dim` apart, shared by the query
                // heads mapped to it. Materializing contiguous planes per
                // query head re-copied each plane heads/kv_heads times
                // (measured 7.3 ms per layer per frame at the real KV
                // shapes and the 9000-frame ceiling).
                let kv_stride = batch * kv_heads * head_dim;
                let mut block_out = vec![0.0f32; len * heads * head_dim];
                for h in 0..heads {
                    let kv_h = h / (heads / kv_heads);
                    let base = (b * kv_heads + kv_h) * head_dim;
                    let mut q_head = vec![0.0f32; len * head_dim];
                    for t in 0..len {
                        let src = ((b * len + t) * heads + h) * head_dim;
                        q_head[t * head_dim..(t + 1) * head_dim]
                            .copy_from_slice(&q[src..src + head_dim]);
                    }
                    let out = ops::sdpa_strided(
                        &q_head,
                        &layer_cache.keys,
                        base,
                        kv_stride,
                        &layer_cache.values,
                        base,
                        kv_stride,
                        mask_ref,
                        len,
                        total,
                        head_dim,
                        head_dim,
                        scale,
                    );
                    for t in 0..len {
                        let dst = (t * heads + h) * head_dim;
                        block_out[dst..dst + head_dim]
                            .copy_from_slice(&out[t * head_dim..(t + 1) * head_dim]);
                    }
                }
                let proj = backend::linear(
                    &block_out,
                    &block.attn.o_w,
                    None,
                    len,
                    heads * head_dim,
                    hidden,
                    t.attention,
                )?;
                let base = b * len * hidden;
                for (res, (old, p)) in attn_out[base..base + len * hidden]
                    .iter_mut()
                    .zip(x[base..base + len * hidden].iter().zip(proj))
                {
                    *res = t.residual.round(old + p);
                }
            }

            attn_out
        };

        trace("residual", &attn_out, t.residual, &[batch, len, hidden]);
        let normed = backend::rms_norm(
            &self.backend,
            &attn_out,
            &block.ln2_w,
            batch * len,
            hidden,
            eps,
            t.post_norm,
        )?;
        trace("post_norm", &normed, t.post_norm, &[batch, len, hidden]);
        let gate = backend::linear(
            &normed,
            &block.mlp.gate_w,
            None,
            batch * len,
            hidden,
            dims.intermediate,
            t.post_norm,
        )?;
        trace("gate", &gate, t.post_norm, &[batch, len, dims.intermediate]);
        let up = backend::linear(
            &normed,
            &block.mlp.up_w,
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
            &block.mlp.down_w,
            None,
            batch * len,
            dims.intermediate,
            hidden,
            t.fused,
        )?;
        trace("down", &down, t.fused, &[batch, len, hidden]);
        let mut out = attn_out;
        for (o, d) in out.iter_mut().zip(down) {
            *o = t.output.round(*o + d);
        }
        trace("output", &out, t.output, &[batch, len, hidden]);
        Ok(out)
    }
}

/// NeoX rope on the leading `head_dim` features of every head in a
/// `[batch, len, heads, head_dim]` buffer, using per-position tables
/// `[seq, head_dim / 2]`. Mirrors `initialize_rope(traditional=False)`
/// with a cache offset baked into the tables.
fn rope_strided(
    x: &mut [f32],
    batch: usize,
    len: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for b in 0..batch {
        for t in 0..len {
            for h in 0..heads {
                let base = ((b * len + t) * heads + h) * head_dim;
                for d in 0..half {
                    let a = x[base + d];
                    let bb = x[base + half + d];
                    let c = cos[t * half + d];
                    let s = sin[t * half + d];
                    x[base + d] = a * c - bb * s;
                    x[base + half + d] = bb * c + a * s;
                }
            }
        }
    }
}

/// Append one step's keys/values into the time-major
/// `[t, batch, kv_heads, dim]` cache. The step arrives as
/// `[batch, len, kv_heads, dim]`; appending interleaves it per
/// position at the tail, so every append is O(len) and no existing
/// element ever moves. (The earlier plane-major `[batch, kv_heads, t,
/// dim]` layout re-laid the whole buffer per step: the growing `t`
/// stride meant O(t) copied bytes per append, O(t^2) per generation,
/// which measured 7.1 ms per layer per frame at the real checkpoint's
/// KV shapes and the 9000-frame ceiling.)
fn append_kv(
    cache: &mut Vec<f32>,
    step: &[f32],
    batch: usize,
    len: usize,
    kv_heads: usize,
    dim: usize,
) {
    let plane = kv_heads * dim;
    debug_assert_eq!(step.len(), batch * len * plane);
    cache.reserve(len * batch * plane);
    for pos in 0..len {
        for b in 0..batch {
            let src = (b * len + pos) * plane;
            cache.extend_from_slice(&step[src..src + plane]);
        }
    }
}

#[cfg(test)]
mod bench {
    //! Attribution micro-bench for the AR decode hot path at the tiny
    //! fixture shapes and the real checkpoint shapes: the per-step KV
    //! append (O(1) in the time-major layout), one strided attention
    //! read per query head over the full cache plane, and the per-step
    //! rope table row. Multiply the strided-sdpa line by the printed
    //! head and batch factors for the per-layer attention cost.
    //! Reports only; run with:
    //!
    //! ```sh
    //! cargo test -p turbospark-audio --release minimax_music3::qwen3::bench \
    //!   -- --ignored --nocapture --test-threads=1
    //! ```

    use std::time::Instant;

    use super::{append_kv, KvCache};
    use crate::ops;

    #[test]
    #[ignore = "timing report, not a gate"]
    fn rope_and_kv_terms_per_layer_per_frame() {
        for (label, kv_heads, head_dim, heads) in
            [("tiny", 2usize, 16usize, 4usize), ("real", 8, 128, 32)]
        {
            for t in [1000usize, 4000, 9000] {
                let batch = 2usize;
                let mut cache = KvCache {
                    keys: vec![0.5f32; batch * kv_heads * t * head_dim],
                    values: vec![0.25f32; batch * kv_heads * t * head_dim],
                    offset: t,
                    capacity: 0,
                    device: None,
                };
                let step = vec![0.5f32; batch * kv_heads * head_dim];
                let start = Instant::now();
                append_kv(&mut cache.keys, &step, batch, 1, kv_heads, head_dim);
                let append = start.elapsed();
                // One strided attention read: one q head over the full
                // cache plane, K pass and V pass.
                let q_head = vec![0.5f32; head_dim];
                let base = head_dim;
                let stride = batch * kv_heads * head_dim;
                let start = Instant::now();
                drop(ops::sdpa_strided(
                    &q_head,
                    &cache.keys,
                    base,
                    stride,
                    &cache.values,
                    base,
                    stride,
                    None,
                    1,
                    t,
                    head_dim,
                    head_dim,
                    0.1,
                ));
                let sdpa_one_head = start.elapsed();
                let start = Instant::now();
                let tables = ops::rope_tables_range(t, 1, head_dim, 1_000_000.0);
                let rope = start.elapsed();
                drop(tables);
                println!(
                    "  {label} t={t:>5}: append {append:?} + strided sdpa per q head {sdpa_one_head:?}                      x {heads} heads x {batch} + rope range {rope:?} per layer per frame"
                );
            }
        }
    }
}
