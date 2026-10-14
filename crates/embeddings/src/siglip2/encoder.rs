//! The pre-norm encoder block both towers share (`Siglip2EncoderLayer`):
//! `x += attn(ln1(x)); x += mlp(ln2(x))`, bidirectional, biased q/k/v/out
//! projections, `gelu_pytorch_tanh` MLP.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use crate::weights::Weights;
use crate::Result;

/// Shapes shared by every layer of one tower.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dims {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub eps: f32,
}

pub(crate) fn layer_norm(w: &Weights, base: &str, width: usize, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.load(&format!("{base}.weight"), &[width])?,
        Some(w.load(&format!("{base}.bias"), &[width])?),
        eps,
    ))
}

/// `fc1 -> gelu_tanh -> fc2` (`Siglip2MLP`).
#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn load(w: &Weights, base: &str, d: Dims) -> Result<Self> {
        Ok(Self {
            fc1: w.linear(&format!("{base}.fc1"), d.hidden, d.intermediate, true, None)?,
            fc2: w.linear(&format!("{base}.fc2"), d.intermediate, d.hidden, true, None)?,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.fc1.forward(x, rows);
        ops::gelu_tanh(&mut h);
        self.fc2.forward(&h, rows)
    }
}

/// Unmasked multi-head attention over `[rows, hidden]`. Masking is never
/// needed: the text tower attends to every position (pads included, as the
/// reference does without an attention mask) and the vision tower drops
/// padded patches before they get here.
pub(crate) fn self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let qh = ops::split_heads(q, rows, heads, head_dim);
    let kh = ops::split_heads(k, rows, heads, head_dim);
    let vh = ops::split_heads(v, rows, heads, head_dim);
    let scale = (head_dim as f32).powf(-0.5);
    let o = ops::mha(&qh, &kh, &vh, None, heads, rows, rows, head_dim, scale);
    ops::merge_heads(&o, rows, heads, head_dim)
}

#[derive(Debug, Clone)]
pub(crate) struct EncoderLayer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    mlp: Mlp,
    heads: usize,
    hidden: usize,
}

impl EncoderLayer {
    pub fn load(w: &Weights, base: &str, d: Dims) -> Result<Self> {
        let attn = format!("{base}.self_attn");
        let lin = |name: &str| w.linear(&format!("{attn}.{name}"), d.hidden, d.hidden, true, None);
        Ok(Self {
            ln1: layer_norm(w, &format!("{base}.layer_norm1"), d.hidden, d.eps)?,
            q: lin("q_proj")?,
            k: lin("k_proj")?,
            v: lin("v_proj")?,
            out: lin("out_proj")?,
            ln2: layer_norm(w, &format!("{base}.layer_norm2"), d.hidden, d.eps)?,
            mlp: Mlp::load(w, &format!("{base}.mlp"), d)?,
            heads: d.heads,
            hidden: d.hidden,
        })
    }

    pub fn forward(&self, x: &mut [f32], rows: usize) {
        let mut h = x.to_vec();
        self.ln1.apply(&mut h, rows);
        let q = self.q.forward(&h, rows);
        let k = self.k.forward(&h, rows);
        let v = self.v.forward(&h, rows);
        let a = self_attention(&q, &k, &v, rows, self.heads, self.hidden / self.heads);
        let a = self.out.forward(&a, rows);
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }
        let mut h = x.to_vec();
        self.ln2.apply(&mut h, rows);
        let m = self.mlp.forward(&h, rows);
        for (xi, mi) in x.iter_mut().zip(&m) {
            *xi += mi;
        }
    }
}

pub(crate) fn load_layers(
    w: &Weights,
    prefix: &str,
    count: usize,
    d: Dims,
) -> Result<Vec<EncoderLayer>> {
    // A checkpoint deeper than its config would otherwise run truncated.
    let extra = format!("{prefix}.encoder.layers.{count}.layer_norm1.weight");
    if w.contains(&extra) {
        return Err(crate::error::bad_config(
            "num_hidden_layers",
            format!("{count} but the checkpoint has {extra}"),
        ));
    }
    (0..count)
        .map(|i| EncoderLayer::load(w, &format!("{prefix}.encoder.layers.{i}"), d))
        .collect()
}
