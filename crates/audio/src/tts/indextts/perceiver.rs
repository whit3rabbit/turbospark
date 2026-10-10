//! The indextts `PerceiverResampler`.
//!
//! Port of `mlx_audio/tts/models/indextts/perceiver.py` at the pinned
//! mlx-audio commit: a learned latent set cross-attends over the
//! projected Conformer context (context rows first, latents appended)
//! for `n_depth` layers of bias-free multi-head attention plus gated
//! exact-GELU feed-forwards, then the final RMSNorm. Single batch, full
//! context, no mask (the reference conditioning path passes none).

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, BiasMode};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::{Linear, RmsNorm};
use crate::ops;
use crate::Result;

/// `nn.GELU()` (exact erf gelu) gated feed-forward: `w_2(gelu(gate) * x)`
/// where the `w_1` output splits into value and gate halves.
struct GatedGeluFeedForward {
    w_1: Linear,
    w_2: Linear,
    d_ff: usize,
}

impl GatedGeluFeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, d_ff: usize) -> Result<Self> {
        Ok(GatedGeluFeedForward {
            w_1: Linear::new(
                load_f32_shaped(file, &format!("{prefix}w_1.weight"), &[2 * d_ff, dim])?,
                load_bias(
                    file,
                    &format!("{prefix}w_1.bias"),
                    2 * d_ff,
                    BiasMode::Required,
                )?,
                dim,
                2 * d_ff,
            ),
            w_2: Linear::new(
                load_f32_shaped(file, &format!("{prefix}w_2.weight"), &[dim, d_ff])?,
                load_bias(file, &format!("{prefix}w_2.bias"), dim, BiasMode::Required)?,
                d_ff,
                dim,
            ),
            d_ff,
        })
    }

    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let hidden = self.w_1.forward(x, len);
        let mut gated = vec![0.0f32; len * self.d_ff];
        for t in 0..len {
            let value = &hidden[t * 2 * self.d_ff..t * 2 * self.d_ff + self.d_ff];
            let gate = &hidden[t * 2 * self.d_ff + self.d_ff..(t + 1) * 2 * self.d_ff];
            let mut gate_activated = gate.to_vec();
            // nn.GELU() defaults to the exact erf activation.
            ops::gelu_erf(&mut gate_activated);
            for d in 0..self.d_ff {
                gated[t * self.d_ff + d] = value[d] * gate_activated[d];
            }
        }
        self.w_2.forward(&gated, len)
    }
}

/// One attention + feed-forward layer over the latent rows.
struct PerceiverLayer {
    attn_q: Linear,
    attn_k: Linear,
    attn_v: Linear,
    attn_out: Linear,
    n_heads: usize,
    head_dim: usize,
    ff: GatedGeluFeedForward,
}

impl PerceiverLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        d_ff: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let width = n_heads * head_dim;
        let load = |name: &str, input: usize, output: usize| -> Result<Linear> {
            Ok(Linear::new(
                load_f32_shaped(file, &format!("{prefix}{name}.weight"), &[output, input])?,
                load_bias(
                    file,
                    &format!("{prefix}{name}.bias"),
                    output,
                    BiasMode::Optional,
                )?,
                input,
                output,
            ))
        };
        Ok(PerceiverLayer {
            attn_q: load("0.linear_q", dim, width)?,
            attn_k: load("0.linear_k", dim, width)?,
            attn_v: load("0.linear_v", dim, width)?,
            attn_out: load("0.linear_out", width, dim)?,
            n_heads,
            head_dim,
            ff: GatedGeluFeedForward::load(file, &format!("{prefix}1."), dim, d_ff)?,
        })
    }

    /// `latents [Tl, dim]`, `kv [Tk, dim]` -> updated latents.
    fn forward(&self, latents: &[f32], t_latents: usize, kv: &[f32], t_kv: usize) -> Vec<f32> {
        let heads = self.n_heads;
        let dim = self.head_dim;
        let width = heads * dim;
        let q = self.attn_q.forward(latents, t_latents);
        let k = self.attn_k.forward(kv, t_kv);
        let v = self.attn_v.forward(kv, t_kv);
        let mut out = vec![0.0f32; t_latents * width];
        let mut scores = vec![0.0f32; t_kv];
        let scale = (dim as f32).powf(-0.5);
        for h in 0..heads {
            for t in 0..t_latents {
                let mut max = f32::NEG_INFINITY;
                for j in 0..t_kv {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q[t * width + h * dim + d] * k[j * width + h * dim + d];
                    }
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                let inv = 1.0 / sum;
                for j in 0..t_kv {
                    let weight = scores[j] * inv;
                    for d in 0..dim {
                        out[t * width + h * dim + d] += weight * v[j * width + h * dim + d];
                    }
                }
            }
        }
        let attended = self.attn_out.forward(&out, t_latents);
        let mut fed = latents.to_vec();
        for (value, add) in fed.iter_mut().zip(&attended) {
            *value += add;
        }
        let ff = self.ff.forward(&fed, t_latents);
        for (value, add) in fed.iter_mut().zip(&ff) {
            *value += add;
        }
        fed
    }
}

/// The loaded `PerceiverResampler`.
pub struct PerceiverResampler {
    proj_context: Option<Linear>,
    /// `[n_latents, dim]` learned rows.
    latents: Vec<f32>,
    layers: Vec<PerceiverLayer>,
    norm: RmsNorm,
    n_latents: usize,
    dim: usize,
}

impl PerceiverResampler {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        n_dim: usize,
        n_dim_context: usize,
        n_latents: usize,
        n_heads: usize,
        n_dim_head: usize,
        n_ff_mult: usize,
    ) -> Result<Self> {
        let proj_context = if n_dim_context != n_dim {
            Some(Linear::new(
                load_f32_shaped(
                    file,
                    &format!("{prefix}proj_context.weight"),
                    &[n_dim, n_dim_context],
                )?,
                load_bias(
                    file,
                    &format!("{prefix}proj_context.bias"),
                    n_dim,
                    BiasMode::Required,
                )?,
                n_dim_context,
                n_dim,
            ))
        } else {
            None
        };
        let d_ff = (n_dim * n_ff_mult * 2) / 3;
        let layers = (0..2usize)
            .map(|index| {
                PerceiverLayer::load(
                    file,
                    &format!("{prefix}layers.{index}."),
                    n_dim,
                    d_ff,
                    n_heads,
                    n_dim_head,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PerceiverResampler {
            proj_context,
            latents: load_f32_shaped(file, &format!("{prefix}latents"), &[n_latents, n_dim])?,
            layers,
            norm: RmsNorm::new(
                load_f32_shaped(file, &format!("{prefix}norm.weight"), &[n_dim])?,
                1e-5,
            ),
            n_latents,
            dim: n_dim,
        })
    }

    /// The context rows `[T, context_dim]` to the latent rows
    /// `[n_latents, dim]`.
    pub fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let projected = match &self.proj_context {
            Some(proj) => proj.forward(x, frames),
            None => x.to_vec(),
        };
        let mut latents = self.latents.clone();
        for layer in &self.layers {
            // kv = concat([x, latents], axis=-2) is rebuilt INSIDE the
            // loop: each layer attends over the context plus its own
            // already-updated latents.
            let mut kv = projected.clone();
            kv.extend_from_slice(&latents);
            let t_kv = frames + self.n_latents;
            latents = layer.forward(&latents, self.n_latents, &kv, t_kv);
        }
        self.norm.apply(&mut latents, self.n_latents);
        debug_assert_eq!(latents.len(), self.n_latents * self.dim);
        Ok(latents)
    }
}
