//! Flow-matching transformer: the velocity field of the MiniMax Music 3
//! latent ODE.
//!
//! Reference: `mlx_audio/music/models/minimax_music3/dit.py`. Input
//! latents, a zero placeholder, and the condition are concatenated on
//! the channel axis, passed through a kernel-1 conv residual, mixed
//! with a Fourier timestep token prepended to the sequence, and run
//! through partial-rotary transformer blocks. LayerNorm eps is the
//! nn.LayerNorm default (1e-5), not the model's RMS eps.

use crate::ops;
use crate::Result;
use crate::SpeechError;

use super::backend::{self, AttentionShape, ComputeBackend, Weight};
use super::conv::{ConvSpec, MlxConv1d};
use super::precision::DType;
use super::weights::{Tensor, WeightStore};
use std::rc::Rc;

const LN_EPS: f32 = 1e-5;

pub(crate) struct DitDims {
    pub in_channels: usize,
    pub num_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff_inner: usize,
    pub rotary_dim: usize,
    pub fourier: usize,
    pub condition_dim: usize,
}

struct DiTAttention {
    to_q: Weight,
    to_k: Weight,
    to_v: Weight,
    to_out: Weight,
}

struct DiTBlock {
    ln1_w: Tensor,
    ln1_b: Tensor,
    attn: DiTAttention,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ff_in_w: Weight,
    ff_in_b: Tensor,
    ff_out_w: Weight,
    ff_out_b: Tensor,
}

impl DiTBlock {
    fn output_dtype(&self, input: DType) -> DType {
        let norm = input.promote(self.ln1_w.dtype).promote(self.ln1_b.dtype);
        let attention = self
            .attn
            .to_q
            .output_dtype(norm)
            .promote(self.attn.to_k.output_dtype(norm))
            .promote(self.attn.to_v.output_dtype(norm));
        let residual = input.promote(self.attn.to_out.output_dtype(attention));
        let norm = residual.promote(self.ln2_w.dtype).promote(self.ln2_b.dtype);
        let fused = self.ff_in_b.promote(self.ff_in_w.output_dtype(norm));
        residual.promote(self.ff_out_b.promote(self.ff_out_w.output_dtype(fused)))
    }
}

pub(crate) struct FlowMatchingTransformer {
    time_proj_w: Tensor, // [fourier / 2, 1]
    time_l1_w: Weight,
    time_l1_b: Tensor,
    time_l2_w: Weight,
    time_l2_b: Tensor,
    pre_conv: MlxConv1d,
    proj_in_w: Weight,
    blocks: Vec<DiTBlock>,
    proj_out_w: Weight,
    post_conv: MlxConv1d,
    backend: Option<Rc<dyn ComputeBackend>>,
    dims: DitDims,
}

impl FlowMatchingTransformer {
    pub(crate) fn load(
        store: &mut WeightStore,
        base: &str,
        dims: DitDims,
    ) -> Result<FlowMatchingTransformer> {
        let inner = dims.heads * dims.head_dim;
        let concat = 2 * dims.in_channels + dims.condition_dim;
        let time_proj_w = store.tensor(&format!("{base}.time_proj.weight"))?;
        if time_proj_w.data.len() != dims.fourier / 2 {
            return Err(SpeechError::Tensor {
                name: format!("{base}.time_proj.weight"),
                why: format!("expected {} rows", dims.fourier / 2),
            });
        }
        let (time_l1_w, time_l1_b) = store.linear(&format!("{base}.time_embed.linear_1"))?;
        let (time_l2_w, time_l2_b) = store.linear(&format!("{base}.time_embed.linear_2"))?;
        if time_l1_w.len() != inner * dims.fourier || time_l2_w.len() != inner * inner {
            return Err(SpeechError::Tensor {
                name: format!("{base}.time_embed"),
                why: "timestep MLP shapes do not match the config".to_string(),
            });
        }
        let pre_conv = MlxConv1d::load(
            store,
            &format!("{base}.preprocess_conv"),
            ConvSpec {
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
            },
        )?;
        let (proj_in_w, _) = store.linear(&format!("{base}.proj_in"))?;
        if proj_in_w.len() != inner * concat {
            return Err(SpeechError::Tensor {
                name: format!("{base}.proj_in"),
                why: format!("expected {inner}x{concat}"),
            });
        }
        let mut blocks = Vec::with_capacity(dims.num_layers);
        for index in 0..dims.num_layers {
            let block = format!("{base}.transformer_blocks.{index}");
            let ln1_w = store.tensor(&format!("{block}.norm1.weight"))?;
            let ln1_b = store.tensor(&format!("{block}.norm1.bias"))?;
            let (to_q, _) = store.linear(&format!("{block}.attn.to_q"))?;
            let (to_k, _) = store.linear(&format!("{block}.attn.to_k"))?;
            let (to_v, _) = store.linear(&format!("{block}.attn.to_v"))?;
            let (to_out, _) = store.linear(&format!("{block}.attn.to_out.0"))?;
            let ln2_w = store.tensor(&format!("{block}.norm2.weight"))?;
            let ln2_b = store.tensor(&format!("{block}.norm2.bias"))?;
            let (ff_in_w, ff_in_b) = store.linear(&format!("{block}.ff_in"))?;
            let (ff_out_w, ff_out_b) = store.linear(&format!("{block}.ff_out"))?;
            for (name, w) in [
                ("attn.to_q", &to_q),
                ("attn.to_k", &to_k),
                ("attn.to_v", &to_v),
                ("attn.to_out", &to_out),
            ] {
                if w.len() != inner * inner {
                    return Err(SpeechError::Tensor {
                        name: format!("{block}.{name}"),
                        why: format!("expected {inner}x{inner}"),
                    });
                }
            }
            if ff_in_w.len() != 2 * dims.ff_inner * inner || ff_out_w.len() != inner * dims.ff_inner
            {
                return Err(SpeechError::Tensor {
                    name: format!("{block}.ff_in"),
                    why: "feed-forward shapes do not match the config".to_string(),
                });
            }
            blocks.push(DiTBlock {
                ln1_w,
                ln1_b,
                attn: DiTAttention {
                    to_q,
                    to_k,
                    to_v,
                    to_out,
                },
                ln2_w,
                ln2_b,
                ff_in_w,
                ff_in_b: ff_in_b.unwrap_or_default(),
                ff_out_w,
                ff_out_b: ff_out_b.unwrap_or_default(),
            });
        }
        let (proj_out_w, _) = store.linear(&format!("{base}.proj_out"))?;
        if proj_out_w.len() != dims.in_channels * inner {
            return Err(SpeechError::Tensor {
                name: format!("{base}.proj_out"),
                why: format!("expected {}x{inner}", dims.in_channels),
            });
        }
        let post_conv = MlxConv1d::load(
            store,
            &format!("{base}.postprocess_conv"),
            ConvSpec {
                kernel: 1,
                stride: 1,
                padding: 0,
                dilation: 1,
            },
        )?;
        Ok(FlowMatchingTransformer {
            time_proj_w,
            time_l1_w,
            time_l1_b: time_l1_b.unwrap_or_default(),
            time_l2_w,
            time_l2_b: time_l2_b.unwrap_or_default(),
            pre_conv,
            proj_in_w,
            blocks,
            proj_out_w,
            post_conv,
            backend: store.backend(),
            dims,
        })
    }

    #[cfg(test)]
    pub(crate) fn dtype(&self) -> DType {
        self.pre_conv.dtype(self.time_proj_w.dtype)
    }

    pub(crate) fn output_dtype(&self, input: DType, condition: DType) -> DType {
        let x = input.promote(condition);
        let x = x.promote(self.pre_conv.dtype(x));
        let time = input.promote(self.time_proj_w.dtype);
        let time = self.time_l1_b.promote(self.time_l1_w.output_dtype(time));
        let time = self.time_l2_b.promote(self.time_l2_w.output_dtype(time));
        let hidden = self.proj_in_w.output_dtype(x).promote(time);
        let hidden = self.blocks.iter().fold(hidden, |d, b| b.output_dtype(d));
        let out = self.proj_out_w.output_dtype(hidden);
        out.promote(self.post_conv.dtype(out))
    }

    pub(crate) fn trace(&self, stage: &str, data: &[f32], dtype: DType, shape: &[usize]) {
        backend::trace(&self.backend, stage, data, dtype, shape);
    }

    /// Velocity for latents `[1, in_channels, T]` at scalar `timestep`
    /// with condition `[1, condition_dim, T]`; returns `[1, in_channels,
    /// T]`.
    #[cfg(test)]
    pub(crate) fn forward(
        &self,
        latents: &[f32],
        timestep: f32,
        condition: &[f32],
        seq: usize,
    ) -> Result<Vec<f32>> {
        self.forward_typed(
            latents,
            timestep,
            condition,
            seq,
            self.dtype(),
            self.dtype(),
        )
    }

    pub(crate) fn forward_typed(
        &self,
        latents: &[f32],
        timestep: f32,
        condition: &[f32],
        seq: usize,
        input_dtype: DType,
        condition_dtype: DType,
    ) -> Result<Vec<f32>> {
        let dims = &self.dims;
        let inner = dims.heads * dims.head_dim;
        let concat = 2 * dims.in_channels + dims.condition_dim;
        let c_in = dims.in_channels;
        if latents.len() != c_in * seq || condition.len() != dims.condition_dim * seq {
            return Err(SpeechError::Input {
                why: format!(
                    "DiT inputs {} / {} do not match {c_in}+{} channels x {seq}",
                    latents.len(),
                    condition.len(),
                    dims.condition_dim
                ),
            });
        }
        // Channel concat [latents, zeros, condition] with a kernel-1
        // conv residual, in channel-major layout.
        let mut x = Vec::with_capacity(concat * seq);
        x.extend_from_slice(latents);
        x.extend(vec![0.0f32; c_in * seq]);
        x.extend_from_slice(condition);
        let x_dtype = input_dtype.promote(condition_dtype);
        let conv = self.pre_conv.forward_typed(&x, seq, x_dtype)?;
        let x_dtype = x_dtype.promote(self.pre_conv.dtype(x_dtype));
        for (v, r) in x.iter_mut().zip(conv) {
            *v = x_dtype.round(*v + r);
        }

        self.trace("dit.pre_conv_residual", &x, x_dtype, &[concat, seq]);
        // Timestep token: Fourier features then the two-layer MLP. The
        // reference concatenates cos(angles) then sin(angles).
        let mut dtype = input_dtype.promote(self.time_proj_w.dtype);
        let fourier = super::precision::fourier(
            &[timestep],
            &self.time_proj_w,
            input_dtype,
            self.time_proj_w.dtype,
        );
        self.trace("dit.time_fourier", &fourier, dtype, &[1, dims.fourier]);
        let time_rows = backend::linear(
            &fourier,
            &self.time_l1_w,
            Some(&self.time_l1_b),
            1,
            dims.fourier,
            inner,
            dtype,
        )?;
        dtype = self.time_l1_b.promote(self.time_l1_w.output_dtype(dtype));
        self.trace("dit.time_linear1", &time_rows, dtype, &[1, inner]);
        let mut time_rows = time_rows;
        for v in &mut time_rows {
            *v = dtype.silu(*v);
        }
        self.trace("dit.time_silu", &time_rows, dtype, &[1, inner]);
        let time_token = backend::linear(
            &time_rows,
            &self.time_l2_w,
            Some(&self.time_l2_b),
            1,
            inner,
            inner,
            dtype,
        )?;
        dtype = self.time_l2_b.promote(self.time_l2_w.output_dtype(dtype));
        self.trace("dit.time_token", &time_token, dtype, &[1, inner]);

        // Per-position input projection, then prepend the time token.
        // The conv residual lives channel-major `[concat, seq]`; the
        // projection consumes per-position rows, so transpose first.
        let mut position_rows = vec![0.0f32; concat * seq];
        for t in 0..seq {
            for c in 0..concat {
                position_rows[t * concat + c] = x[c * seq + t];
            }
        }
        let time_dtype = dtype;
        dtype = x_dtype;
        let mut rows = backend::linear(
            &position_rows,
            &self.proj_in_w,
            None,
            seq,
            concat,
            inner,
            dtype,
        )?;
        let mut sequence = Vec::with_capacity((seq + 1) * inner);
        self.trace(
            "dit.proj_in",
            &rows,
            self.proj_in_w.output_dtype(dtype),
            &[seq, inner],
        );
        sequence.extend_from_slice(&time_token);
        sequence.extend_from_slice(&rows);
        dtype = self.proj_in_w.output_dtype(dtype).promote(time_dtype);
        let len = seq + 1;

        let (cos, sin) = if dtype == DType::F32 {
            ops::rope_tables(len, dims.rotary_dim, 10_000.0)
        } else if let Some(compute) = &self.backend {
            compute.rotary_tables(len, dims.rotary_dim, 10_000.0)?
        } else {
            backend::rotary_tables(len, dims.rotary_dim, 10_000.0)
        };
        self.trace(
            "dit.rotary.cos",
            &cos,
            DType::F32,
            &[len, dims.rotary_dim / 2],
        );
        self.trace(
            "dit.rotary.sin",
            &sin,
            DType::F32,
            &[len, dims.rotary_dim / 2],
        );
        for (layer_index, block) in self.blocks.iter().enumerate() {
            sequence = self.block_forward(block, &sequence, len, &cos, &sin, dtype, layer_index)?;
            dtype = block.output_dtype(dtype);
        }

        // Drop the time token, project back to channels, conv residual.
        rows = sequence[inner..].to_vec();
        let out_rows = backend::linear(&rows, &self.proj_out_w, None, seq, inner, c_in, dtype)?;
        self.trace(
            "dit.proj_out",
            &out_rows,
            self.proj_out_w.output_dtype(dtype),
            &[seq, c_in],
        );
        // Transpose to channel-major for the post conv.
        let mut channel_major = vec![0.0f32; c_in * seq];
        for t in 0..seq {
            for c in 0..c_in {
                channel_major[c * seq + t] = out_rows[t * c_in + c];
            }
        }
        dtype = self.proj_out_w.output_dtype(dtype);
        let conv = self.post_conv.forward_typed(&channel_major, seq, dtype)?;
        dtype = dtype.promote(self.post_conv.dtype(dtype));
        for (v, r) in channel_major.iter_mut().zip(conv) {
            *v = dtype.round(*v + r);
        }
        self.trace(
            "dit.post_conv_residual",
            &channel_major,
            dtype,
            &[c_in, seq],
        );
        Ok(channel_major)
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        &self,
        block: &DiTBlock,
        x: &[f32],
        len: usize,
        cos: &[f32],
        sin: &[f32],
        input_dtype: DType,
        layer_index: usize,
    ) -> Result<Vec<f32>> {
        let trace = |stage: &str, data: &[f32], dtype: DType, shape: &[usize]| {
            backend::trace(
                &self.backend,
                &format!("dit.{layer_index}.{stage}"),
                data,
                dtype,
                shape,
            )
        };
        let dims = &self.dims;
        let inner = dims.heads * dims.head_dim;
        let head_dim = dims.head_dim;
        trace("input", x, input_dtype, &[len, inner]);
        let mut dtype = input_dtype
            .promote(block.ln1_w.dtype)
            .promote(block.ln1_b.dtype);
        let normed = backend::layer_norm(
            &self.backend,
            x,
            &block.ln1_w,
            Some(&block.ln1_b),
            len,
            inner,
            LN_EPS,
            dtype,
        )?;
        trace("input_norm", &normed, dtype, &[len, inner]);
        let q = backend::linear(&normed, &block.attn.to_q, None, len, inner, inner, dtype)?;
        trace("q", &q, dtype, &[len, inner]);
        let k = backend::linear(&normed, &block.attn.to_k, None, len, inner, inner, dtype)?;
        trace("k", &k, dtype, &[len, inner]);
        let v = backend::linear(&normed, &block.attn.to_v, None, len, inner, inner, dtype)?;
        trace("v", &v, dtype, &[len, inner]);
        dtype = block
            .attn
            .to_q
            .output_dtype(dtype)
            .promote(block.attn.to_k.output_dtype(dtype))
            .promote(block.attn.to_v.output_dtype(dtype));
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_rows = if self.backend.is_some() || dtype != DType::F32 {
            // Rotary touches only the leading features of each head. Apply
            // the same tables before the device's batched attention dispatch.
            let mut q = q;
            let mut k = k;
            for h in 0..dims.heads {
                let mut qp = head_plane(&q, len, inner, h, head_dim);
                let mut kp = head_plane(&k, len, inner, h, head_dim);
                partial_rope_typed(&mut qp, head_dim, dims.rotary_dim, cos, sin, dtype);
                partial_rope_typed(&mut kp, head_dim, dims.rotary_dim, cos, sin, dtype);
                for t in 0..len {
                    let dst = t * inner + h * head_dim;
                    q[dst..dst + head_dim].copy_from_slice(&qp[t * head_dim..(t + 1) * head_dim]);
                    k[dst..dst + head_dim].copy_from_slice(&kp[t * head_dim..(t + 1) * head_dim]);
                }
            }
            trace("q_rope", &q, dtype, &[1, len, dims.heads, head_dim]);
            trace("k_rope", &k, dtype, &[1, len, dims.heads, head_dim]);
            backend::attention(
                &self.backend,
                &q,
                &k,
                &v,
                AttentionShape {
                    batch: 1,
                    queries: len,
                    keys: len,
                    heads: dims.heads,
                    kv_heads: dims.heads,
                    dim: head_dim,
                    kv_time_major: false,
                    causal: false,
                    offset: 0,
                },
                dtype,
            )?
        } else {
            let mut attn_out = Vec::with_capacity(x.len());
            for h in 0..dims.heads {
                let mut q_head = head_plane(&q, len, inner, h, head_dim);
                let mut k_head = head_plane(&k, len, inner, h, head_dim);
                let v_head = head_plane(&v, len, inner, h, head_dim);
                partial_rope(&mut q_head, head_dim, dims.rotary_dim, cos, sin);
                partial_rope(&mut k_head, head_dim, dims.rotary_dim, cos, sin);
                let out = ops::sdpa(
                    &q_head, &k_head, &v_head, None, len, len, head_dim, head_dim, scale,
                );
                attn_out.extend(out);
            }
            // head_plane strips the batch dim (always 1 here); heads are
            // concatenated back per position by the scatter below.
            let mut attn_rows = vec![0.0f32; len * inner];
            for h in 0..dims.heads {
                for t in 0..len {
                    let src = (h * len + t) * head_dim;
                    let dst = (t * inner) + h * head_dim;
                    attn_rows[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
                }
            }
            attn_rows
        };

        trace("attention", &attn_rows, dtype, &[len, inner]);
        let proj = backend::linear(
            &attn_rows,
            &block.attn.to_out,
            None,
            len,
            inner,
            inner,
            dtype,
        )?;
        let mut residual = x.to_vec();
        for (r, p) in residual.iter_mut().zip(proj) {
            *r = input_dtype
                .promote(block.attn.to_out.output_dtype(dtype))
                .round(*r + p);
        }

        trace(
            "residual",
            &residual,
            input_dtype.promote(block.attn.to_out.output_dtype(dtype)),
            &[len, inner],
        );
        dtype = input_dtype
            .promote(block.attn.to_out.output_dtype(dtype))
            .promote(block.ln2_w.dtype)
            .promote(block.ln2_b.dtype);
        let normed = backend::layer_norm(
            &self.backend,
            &residual,
            &block.ln2_w,
            Some(&block.ln2_b),
            len,
            inner,
            LN_EPS,
            dtype,
        )?;
        trace("post_norm", &normed, dtype, &[len, inner]);
        let ff = backend::linear(
            &normed,
            &block.ff_in_w,
            Some(&block.ff_in_b),
            len,
            inner,
            2 * dims.ff_inner,
            dtype,
        )?;
        dtype = block.ff_in_b.promote(block.ff_in_w.output_dtype(dtype));
        trace("ff", &ff, dtype, &[len, 2 * dims.ff_inner]);
        let mut gated = vec![0.0f32; len * dims.ff_inner];
        for t in 0..len {
            for f in 0..dims.ff_inner {
                let state = ff[t * 2 * dims.ff_inner + f];
                let gate = ff[t * 2 * dims.ff_inner + dims.ff_inner + f];
                gated[t * dims.ff_inner + f] = dtype.round(state * dtype.silu(gate));
            }
        }
        trace("swiglu", &gated, dtype, &[len, dims.ff_inner]);
        let out = backend::linear(
            &gated,
            &block.ff_out_w,
            Some(&block.ff_out_b),
            len,
            dims.ff_inner,
            inner,
            dtype,
        )?;
        dtype = block.ff_out_b.promote(block.ff_out_w.output_dtype(dtype));
        trace("ff_out", &out, dtype, &[len, inner]);
        for (r, o) in residual.iter_mut().zip(out) {
            *r = block.output_dtype(input_dtype).round(*r + o);
        }
        trace(
            "output",
            &residual,
            block.output_dtype(input_dtype),
            &[len, inner],
        );
        Ok(residual)
    }
}

/// Copy head `h` out of row-major `[len, inner]` as `[len, head_dim]`.
fn head_plane(x: &[f32], len: usize, inner: usize, h: usize, head_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * head_dim);
    for t in 0..len {
        let base = t * inner + h * head_dim;
        out.extend_from_slice(&x[base..base + head_dim]);
    }
    out
}

/// Partial NeoX rotary over the leading `rotary` features of every
/// position in `[len, head_dim]`, tables `[len, rotary / 2]`.
fn partial_rope(x: &mut [f32], head_dim: usize, rotary: usize, cos: &[f32], sin: &[f32]) {
    let half = rotary / 2;
    let len = x.len() / head_dim;
    for t in 0..len {
        let base = t * head_dim;
        for d in 0..half {
            let a = x[base + d];
            let b = x[base + half + d];
            let c = cos[t * half + d];
            let s = sin[t * half + d];
            x[base + d] = a * c - b * s;
            x[base + half + d] = b * c + a * s;
        }
    }
}

pub(crate) fn partial_rope_typed(
    x: &mut [f32],
    head_dim: usize,
    rotary: usize,
    cos: &[f32],
    sin: &[f32],
    dtype: DType,
) {
    if dtype == DType::F32 {
        partial_rope(x, head_dim, rotary, cos, sin);
        return;
    }
    let half = rotary / 2;
    for t in 0..x.len() / head_dim {
        for d in 0..half {
            let base = t * head_dim;
            let a = x[base + d];
            let b = x[base + half + d];
            let c = dtype.round(cos[t * half + d]);
            let s = dtype.round(sin[t * half + d]);
            x[base + d] = dtype.round(dtype.round(a * c) - dtype.round(b * s));
            x[base + half + d] = dtype.round(dtype.round(b * c) + dtype.round(a * s));
        }
    }
}

#[cfg(test)]
mod precision_tests {
    use super::super::backend::{ConvShape, DeviceWeight, WeightData};
    use super::super::{Model, Music3Precision};
    use super::*;
    use std::cell::RefCell;

    struct FixedProjection;
    impl DeviceWeight for FixedProjection {
        fn linear(
            &self,
            _input: &[f32],
            bias: Option<&[f32]>,
            rows: usize,
            _input_dim: usize,
            output_dim: usize,
            dtype: DType,
        ) -> Result<Vec<f32>> {
            // Pre-cast projections from the independent pinned MLX mixed-bias
            // probe. The backend double isolates dtype routing from matmul.
            let raw = [0.6078644, -2.63517, 12.064575, 2.0074806];
            Ok((0..rows * output_dim)
                .map(|i| {
                    dtype.round(dtype.round(raw[i % 4]) + bias.map_or(0.0, |b| b[i % output_dim]))
                })
                .collect())
        }
        fn embedding(&self, _ids: &[i32], _width: usize) -> Result<Vec<f32>> {
            unreachable!("projection test does not use embeddings")
        }
        fn convolution(
            &self,
            _input: &[f32],
            _bias: Option<&[f32]>,
            _shape: ConvShape,
            _dtype: DType,
        ) -> Result<Vec<f32>> {
            unreachable!("projection test does not use convolution")
        }
    }

    struct TraceBackend {
        stage: &'static str,
        captured: RefCell<Option<(Vec<f32>, DType)>>,
    }
    impl ComputeBackend for TraceBackend {
        fn load_weight(&self, _data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
            unreachable!("test installs backend after loading")
        }
        fn attention(
            &self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            shape: AttentionShape,
            dtype: DType,
        ) -> Result<Vec<f32>> {
            backend::attention(&None, q, k, v, shape, dtype)
        }
        fn trace(&self, stage: &str, data: &[f32], dtype: DType, _shape: &[usize]) {
            if stage == self.stage {
                *self.captured.borrow_mut() = Some((data.to_vec(), dtype));
            }
        }
    }

    #[test]
    fn dit_mixed_bias_preserves_projection_precision_and_dense_addmm() {
        let mut failures = Vec::new();
        for packed in [true, false] {
            for stage in [
                "dit.time_linear1",
                "dit.time_token",
                "dit.0.ff",
                "dit.0.ff_out",
            ] {
                let model = Model::load_converted_with_precision(
                    &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("testdata/minimax_music3/precision/mxfp8"),
                    Music3Precision::Checkpoint,
                )
                .unwrap();
                let mut transformer = model.transformer;
                let inner = transformer.dims.heads * transformer.dims.head_dim;
                let block = &mut transformer.blocks[0];
                let (weight, bias, in_dim, out_dim) = match stage {
                    "dit.time_linear1" => (
                        &mut transformer.time_l1_w,
                        &mut transformer.time_l1_b,
                        transformer.dims.fourier,
                        inner,
                    ),
                    "dit.time_token" => (
                        &mut transformer.time_l2_w,
                        &mut transformer.time_l2_b,
                        inner,
                        inner,
                    ),
                    "dit.0.ff" => (
                        &mut block.ff_in_w,
                        &mut block.ff_in_b,
                        inner,
                        2 * transformer.dims.ff_inner,
                    ),
                    _ => (
                        &mut block.ff_out_w,
                        &mut block.ff_out_b,
                        transformer.dims.ff_inner,
                        inner,
                    ),
                };
                *weight = Weight::Device {
                    elements: in_dim * out_dim,
                    dtype: DType::Bf16,
                    packed,
                    dynamic: packed,
                    weight: Rc::new(FixedProjection),
                };
                *bias = Tensor {
                    data: vec![0.01; out_dim],
                    shape: vec![out_dim],
                    dtype: DType::F32,
                };
                let trace = Rc::new(TraceBackend {
                    stage,
                    captured: RefCell::new(None),
                });
                transformer.backend = Some(trace.clone());
                transformer
                    .forward_typed(
                        &vec![0.0; transformer.dims.in_channels],
                        0.25,
                        &vec![0.0; transformer.dims.condition_dim],
                        1,
                        DType::Bf16,
                        DType::Bf16,
                    )
                    .unwrap();
                let captured = trace.captured.borrow();
                let (actual, dtype) = captured.as_ref().expect("requested projection trace");
                // BF16 QuantizedLinear projects before the separate F32 bias;
                // dense addmm promotes its operands and keeps the fused result.
                let expected = if packed {
                    [0.619375, -2.630625, 12.0725, 2.01]
                } else {
                    [0.6178644, -2.62517, 12.074575, 2.0174806]
                };
                if *dtype != DType::F32
                    || actual
                        .iter()
                        .enumerate()
                        .any(|(i, v)| *v != expected[i % 4])
                {
                    failures.push(format!(
                        "{stage} packed={packed}: dtype={dtype:?}, first={:?}",
                        &actual[..4]
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
