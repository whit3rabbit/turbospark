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

use super::conv::{ConvSpec, MlxConv1d};
use super::weights::WeightStore;

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
    to_q: Vec<f32>,
    to_k: Vec<f32>,
    to_v: Vec<f32>,
    to_out: Vec<f32>,
}

struct DiTBlock {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn: DiTAttention,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ff_in_w: Vec<f32>,
    ff_in_b: Vec<f32>,
    ff_out_w: Vec<f32>,
    ff_out_b: Vec<f32>,
}

pub(crate) struct FlowMatchingTransformer {
    time_proj_w: Vec<f32>, // [fourier / 2, 1]
    time_l1_w: Vec<f32>,
    time_l1_b: Vec<f32>,
    time_l2_w: Vec<f32>,
    time_l2_b: Vec<f32>,
    pre_conv: MlxConv1d,
    proj_in_w: Vec<f32>,
    blocks: Vec<DiTBlock>,
    proj_out_w: Vec<f32>,
    post_conv: MlxConv1d,
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
                ln1_w: ln1_w.data,
                ln1_b: ln1_b.data,
                attn: DiTAttention {
                    to_q,
                    to_k,
                    to_v,
                    to_out,
                },
                ln2_w: ln2_w.data,
                ln2_b: ln2_b.data,
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
            time_proj_w: time_proj_w.data,
            time_l1_w,
            time_l1_b: time_l1_b.unwrap_or_default(),
            time_l2_w,
            time_l2_b: time_l2_b.unwrap_or_default(),
            pre_conv,
            proj_in_w,
            blocks,
            proj_out_w,
            post_conv,
            dims,
        })
    }

    /// Velocity for latents `[1, in_channels, T]` at scalar `timestep`
    /// with condition `[1, condition_dim, T]`; returns `[1, in_channels,
    /// T]`.
    pub(crate) fn forward(
        &self,
        latents: &[f32],
        timestep: f32,
        condition: &[f32],
        seq: usize,
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
        let conv = self.pre_conv.forward(&x, seq);
        for (v, r) in x.iter_mut().zip(conv) {
            *v += r;
        }

        // Timestep token: Fourier features then the two-layer MLP. The
        // reference concatenates cos(angles) then sin(angles).
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        let angle = two_pi * timestep;
        let mut fourier = vec![0.0f32; dims.fourier];
        for f in 0..dims.fourier / 2 {
            let a = angle * self.time_proj_w[f];
            fourier[f] = a.cos();
            fourier[dims.fourier / 2 + f] = a.sin();
        }
        let time_rows = ops::linear(
            &fourier,
            &self.time_l1_w,
            Some(&self.time_l1_b),
            1,
            dims.fourier,
            inner,
        );
        let mut time_rows = time_rows;
        for v in &mut time_rows {
            *v = *v / (1.0 + (-*v).exp());
        }
        let time_token = ops::linear(
            &time_rows,
            &self.time_l2_w,
            Some(&self.time_l2_b),
            1,
            inner,
            inner,
        );

        // Per-position input projection, then prepend the time token.
        // The conv residual lives channel-major `[concat, seq]`; the
        // projection consumes per-position rows, so transpose first.
        let mut position_rows = vec![0.0f32; concat * seq];
        for t in 0..seq {
            for c in 0..concat {
                position_rows[t * concat + c] = x[c * seq + t];
            }
        }
        let mut rows = ops::linear(&position_rows, &self.proj_in_w, None, seq, concat, inner);
        let mut sequence = Vec::with_capacity((seq + 1) * inner);
        sequence.extend_from_slice(&time_token);
        sequence.extend_from_slice(&rows);
        let len = seq + 1;

        let (cos, sin) = ops::rope_tables(len, dims.rotary_dim, 10_000.0);
        for block in &self.blocks {
            sequence = self.block_forward(block, &sequence, len, &cos, &sin);
        }

        // Drop the time token, project back to channels, conv residual.
        rows = sequence[inner..].to_vec();
        let out_rows = ops::linear(&rows, &self.proj_out_w, None, seq, inner, c_in);
        // Transpose to channel-major for the post conv.
        let mut channel_major = vec![0.0f32; c_in * seq];
        for t in 0..seq {
            for c in 0..c_in {
                channel_major[c * seq + t] = out_rows[t * c_in + c];
            }
        }
        let conv = self.post_conv.forward(&channel_major, seq);
        for (v, r) in channel_major.iter_mut().zip(conv) {
            *v += r;
        }
        Ok(channel_major)
    }

    fn block_forward(
        &self,
        block: &DiTBlock,
        x: &[f32],
        len: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let dims = &self.dims;
        let inner = dims.heads * dims.head_dim;
        let head_dim = dims.head_dim;
        let mut normed = x.to_vec();
        ops::layernorm(
            &mut normed,
            len,
            inner,
            &block.ln1_w,
            Some(&block.ln1_b),
            LN_EPS,
        );
        let q = ops::linear(&normed, &block.attn.to_q, None, len, inner, inner);
        let k = ops::linear(&normed, &block.attn.to_k, None, len, inner, inner);
        let v = ops::linear(&normed, &block.attn.to_v, None, len, inner, inner);
        let scale = 1.0 / (head_dim as f32).sqrt();
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
        let proj = ops::linear(&attn_rows, &block.attn.to_out, None, len, inner, inner);
        let mut residual = x.to_vec();
        for (r, p) in residual.iter_mut().zip(proj) {
            *r += p;
        }

        let mut normed = residual.clone();
        ops::layernorm(
            &mut normed,
            len,
            inner,
            &block.ln2_w,
            Some(&block.ln2_b),
            LN_EPS,
        );
        let ff = ops::linear(
            &normed,
            &block.ff_in_w,
            Some(&block.ff_in_b),
            len,
            inner,
            2 * dims.ff_inner,
        );
        let mut gated = vec![0.0f32; len * dims.ff_inner];
        for t in 0..len {
            for f in 0..dims.ff_inner {
                let state = ff[t * 2 * dims.ff_inner + f];
                let gate = ff[t * 2 * dims.ff_inner + dims.ff_inner + f];
                gated[t * dims.ff_inner + f] = state * (gate / (1.0 + (-gate).exp()));
            }
        }
        let out = ops::linear(
            &gated,
            &block.ff_out_w,
            Some(&block.ff_out_b),
            len,
            dims.ff_inner,
            inner,
        );
        for (r, o) in residual.iter_mut().zip(out) {
            *r += o;
        }
        residual
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
