//! NaFlex vision tower (`Siglip2VisionTransformer`).
//!
//! Patch embedding is a plain linear over the flattened `(row, col, chan)`
//! patch. The learned position table is a `g x g` grid (`g = sqrt(
//! num_patches)`) resized per image to its `(h, w)` patch grid with
//! `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`
//! in f32, then added. After the encoder and `post_layernorm`, the MAP
//! head (`Siglip2MultiheadAttentionPoolingHead`) pools: one learned probe
//! attends over the patches through `nn.MultiheadAttention`, then
//! `x + mlp(layernorm(x))`. That head output is the image embedding before
//! L2 normalization; SigLIP2 has no separate vision projection.
//!
//! Padded patches are not materialized. In the reference they get
//! `resized_pos[0]` and are masked as keys in every attention, so their
//! softmax weight is exactly zero and they cannot reach the pooled output.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::config::VisionConfig;
use super::encoder::{layer_norm, load_layers, Dims, EncoderLayer, Mlp};
use super::processing::{aa_bilinear_weights, NaflexInput};
use crate::error::{input_error, tensor_error};
use crate::weights::Weights;
use crate::Result;

const PREFIX: &str = "vision_model";

/// Intermediate tensors of one forward, for parity tests and debugging.
#[derive(Debug, Clone)]
pub struct VisionTrace {
    /// `[n, hidden]` position embeddings resized to the image grid.
    pub pos_resized: Vec<f32>,
    /// `[n, hidden]` patch embedding + position embedding.
    pub embeddings: Vec<f32>,
    /// `[n, hidden]` after the encoder and `post_layernorm`.
    pub last_hidden: Vec<f32>,
    /// `[hidden]` MAP head output (unnormalized image embedding).
    pub pooled: Vec<f32>,
}

/// The attention-pooling head.
#[derive(Debug, Clone)]
struct MapHead {
    probe: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    layernorm: LayerNorm,
    mlp: Mlp,
    heads: usize,
}

impl MapHead {
    fn load(w: &Weights, d: Dims) -> Result<Self> {
        let base = format!("{PREFIX}.head");
        let h = d.hidden;
        // nn.MultiheadAttention packs q, k, v row-wise into in_proj.
        let in_w = w.load(&format!("{base}.attention.in_proj_weight"), &[3 * h, h])?;
        let in_b = w.load(&format!("{base}.attention.in_proj_bias"), &[3 * h])?;
        let part = |i: usize| {
            Linear::new(
                in_w[i * h * h..(i + 1) * h * h].to_vec(),
                Some(in_b[i * h..(i + 1) * h].to_vec()),
                h,
                h,
            )
        };
        Ok(Self {
            probe: w.load(&format!("{base}.probe"), &[1, 1, h])?,
            q: part(0),
            k: part(1),
            v: part(2),
            out: w.linear(&format!("{base}.attention.out_proj"), h, h, true, None)?,
            layernorm: layer_norm(w, &format!("{base}.layernorm"), h, d.eps)?,
            mlp: Mlp::load(w, &format!("{base}.mlp"), d)?,
            heads: d.heads,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let h = self.probe.len();
        let hd = h / self.heads;
        let q = self.q.forward(&self.probe, 1);
        let k = ops::split_heads(&self.k.forward(x, rows), rows, self.heads, hd);
        let v = ops::split_heads(&self.v.forward(x, rows), rows, self.heads, hd);
        // A single query row is already head-major after a reshape.
        let scale = (hd as f32).powf(-0.5);
        let o = ops::mha(&q, &k, &v, None, self.heads, 1, rows, hd, scale);
        let attn = self.out.forward(&o, 1);
        let mut normed = attn.clone();
        self.layernorm.apply(&mut normed, 1);
        let m = self.mlp.forward(&normed, 1);
        attn.iter().zip(&m).map(|(a, b)| a + b).collect()
    }
}

/// Resizes a row-major `[in_h, in_w, ch]` f32 grid to `[out_h, out_w, ch]`
/// with PyTorch's separable antialiased bilinear: width pass first, then
/// height, f32 accumulation, an unchanged axis skipped.
pub fn resize_grid_aa_bilinear(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    ch: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    debug_assert_eq!(src.len(), in_h * in_w * ch);
    let mut cur = src.to_vec();
    let mut w = in_w;
    if out_w != in_w {
        let axis = aa_bilinear_weights(in_w, out_w);
        let mut next = vec![0.0f32; in_h * out_w * ch];
        for y in 0..in_h {
            for (ox, a) in axis.iter().enumerate() {
                let dst = &mut next[(y * out_w + ox) * ch..(y * out_w + ox + 1) * ch];
                for (k, &wt) in a.weights.iter().enumerate() {
                    let s = (y * in_w + a.start + k) * ch;
                    let wt = wt as f32;
                    for (d, &v) in dst.iter_mut().zip(&cur[s..s + ch]) {
                        *d += v * wt;
                    }
                }
            }
        }
        cur = next;
        w = out_w;
    }
    if out_h != in_h {
        let axis = aa_bilinear_weights(in_h, out_h);
        let mut next = vec![0.0f32; out_h * w * ch];
        for (oy, a) in axis.iter().enumerate() {
            for x in 0..w {
                let dst = &mut next[(oy * w + x) * ch..(oy * w + x + 1) * ch];
                for (k, &wt) in a.weights.iter().enumerate() {
                    let s = ((a.start + k) * w + x) * ch;
                    let wt = wt as f32;
                    for (d, &v) in dst.iter_mut().zip(&cur[s..s + ch]) {
                        *d += v * wt;
                    }
                }
            }
        }
        cur = next;
    }
    cur
}

/// The vision tower with its pooling head.
#[derive(Debug, Clone)]
pub struct VisionTower {
    cfg: VisionConfig,
    patch_embed: Linear,
    /// `[grid, grid, hidden]`.
    position: Vec<f32>,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    head: MapHead,
}

impl VisionTower {
    pub fn load(w: &Weights, cfg: &VisionConfig) -> Result<Self> {
        let d = Dims {
            hidden: cfg.hidden_size,
            intermediate: cfg.intermediate_size,
            heads: cfg.num_attention_heads,
            eps: cfg.layer_norm_eps,
        };
        let pd = cfg.patch_size * cfg.patch_size * cfg.num_channels;
        Ok(Self {
            patch_embed: w.linear(
                &format!("{PREFIX}.embeddings.patch_embedding"),
                pd,
                d.hidden,
                true,
                None,
            )?,
            position: w.load(
                &format!("{PREFIX}.embeddings.position_embedding.weight"),
                &[cfg.num_patches, d.hidden],
            )?,
            layers: load_layers(w, PREFIX, cfg.num_hidden_layers, d)?,
            post_layernorm: layer_norm(w, &format!("{PREFIX}.post_layernorm"), d.hidden, d.eps)?,
            head: MapHead::load(w, d)?,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Position embeddings for an `h x w` patch grid, `[h * w, hidden]`.
    pub fn resized_position_embeddings(&self, h: usize, w: usize) -> Vec<f32> {
        let g = self.cfg.position_grid();
        resize_grid_aa_bilinear(&self.position, g, g, self.cfg.hidden_size, h, w)
    }

    /// Runs the tower and keeps every stage.
    pub fn forward_trace(&self, input: &NaflexInput) -> Result<VisionTrace> {
        let (gh, gw) = input.spatial_shape;
        let n = gh * gw;
        if input.patch_size != self.cfg.patch_size {
            return Err(input_error(format!(
                "patch size {} does not match the tower's {}",
                input.patch_size, self.cfg.patch_size
            )));
        }
        if n == 0 || input.patches.len() != n * input.patch_dim() {
            return Err(input_error(format!(
                "{} patch values for a {gh}x{gw} grid of dim {}",
                input.patches.len(),
                input.patch_dim()
            )));
        }
        let hidden = self.cfg.hidden_size;
        let pos_resized = self.resized_position_embeddings(gh, gw);
        if pos_resized.len() != n * hidden {
            return Err(tensor_error(
                "position_embedding",
                "resize produced a wrong shape",
            ));
        }
        let mut x = self.patch_embed.forward(&input.patches, n);
        for (xi, pi) in x.iter_mut().zip(&pos_resized) {
            *xi += pi;
        }
        let embeddings = x.clone();
        for layer in &self.layers {
            layer.forward(&mut x, n);
        }
        self.post_layernorm.apply(&mut x, n);
        let pooled = self.head.forward(&x, n);
        Ok(VisionTrace {
            pos_resized,
            embeddings,
            last_hidden: x,
            pooled,
        })
    }

    /// The unnormalized pooled image embedding.
    pub fn forward(&self, input: &NaflexInput) -> Result<Vec<f32>> {
        Ok(self.forward_trace(input)?.pooled)
    }
}
