//! The Nomic BERT encoder (`NomicBertModel` in the pinned
//! `modeling_hf_nomic_bert.py`), one unpadded sequence per call.
//!
//! ```text
//! x = emb_ln(word[id] + token_type[0])           (rotary: no position table)
//! per layer (post-LN, prenorm = false):
//!   q, k, v = split(Wqkv(x))                     ("(three h d)": q, k, v thirds)
//!   q, k    = rope_neox(q), rope_neox(k)         (non-interleaved, NTK-scaled past
//!                                                 max_trained_positions)
//!   x = norm1(x + out_proj(softmax(q k^T / sqrt(d)) v))
//!   x = norm2(x + fc2(fc11(x) * silu(fc12(x))))  (fc11 is the value, fc12 the gate)
//! ```
//!
//! Dynamic NTK (modeling:624-691): for a sequence of `n` tokens with
//! `n > max_trained_positions` (`M`) the rotary base becomes
//! `base * (f * n / M - (f - 1)) ^ (d / (d - 2))` with `f` the scaling
//! factor and `d` the rotary dim, and the cos/sin table is built from
//! that base. `n <= M` uses the plain base. The reference keeps the
//! scaled table cached across calls; this port is stateless and computes
//! what a freshly loaded reference computes for the one sequence (see
//! README).

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::config::NomicBertConfig;
use crate::error::{input_error, tensor_error};
use crate::weights::{EmbeddingTable, Weights};
use crate::Result;

/// Rotary base for an `n`-token sequence (modeling:639-642). The
/// reference evaluates this in Python floats (f64).
pub fn ntk_base(cfg: &NomicBertConfig, n: usize) -> f64 {
    let base = cfg.rotary_emb_base;
    let m = cfg.max_trained_positions;
    match cfg.rotary_scaling_factor {
        Some(f) if n > m => {
            let d = cfg.head_dim() as f64;
            base * ((f * n as f64) / m as f64 - (f - 1.0)).powf(d / (d - 2.0))
        }
        _ => base,
    }
}

/// `1 / base^(2i / dim)` (modeling:630-633): the exponent is an f32
/// division, base is cast to f32, and the power is evaluated in f64 and
/// rounded once. torch's f32 pow is within 1 ulp of that rounding, so a
/// non-representable scaled base can differ from the reference buffer
/// by up to 2 ulps (measured; see README).
pub fn inv_freq(base: f64, dim: usize) -> Vec<f32> {
    let base = f64::from(base as f32);
    (0..dim / 2)
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0f32 / base.powf(f64::from(e)) as f32
        })
        .collect()
}

/// cos/sin tables `[seq, dim/2]`: angle `t * inv_freq` as one f32
/// product (`torch.outer`), cos and sin of that f32 angle.
pub fn rope_tables(seq: usize, inv: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv.len();
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        for (i, &f) in inv.iter().enumerate() {
            let a = f64::from(t as f32 * f);
            cos[t * half + i] = a.cos() as f32;
            sin[t * half + i] = a.sin() as f32;
        }
    }
    (cos, sin)
}

#[derive(Debug, Clone)]
struct Layer {
    wqkv: Linear,
    out_proj: Linear,
    norm1: LayerNorm,
    fc11: Linear,
    fc12: Linear,
    fc2: Linear,
    norm2: LayerNorm,
}

/// The loaded encoder.
#[derive(Debug, Clone)]
pub struct Encoder {
    cfg: NomicBertConfig,
    word: EmbeddingTable,
    /// Row 0 of `token_type_embeddings`: `NomicBertModel.forward` uses
    /// zeros when no ids are passed (modeling:1063-1064) and a
    /// single-segment BERT tokenization yields all zeros.
    token_type: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<Layer>,
}

fn layer_norm(w: &Weights, base: &str, width: usize, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.load(&format!("{base}.weight"), &[width])?,
        Some(w.load(&format!("{base}.bias"), &[width])?),
        eps,
    ))
}

impl Encoder {
    /// Loads every tensor, shape-checked against `cfg`. The 30528 x 768
    /// table (94 MB) is copied to the heap; it is small enough that the
    /// mapped-row path the larger vocabularies use is not worth it.
    pub fn load(w: &Weights, cfg: &NomicBertConfig) -> Result<Self> {
        let (h, inner) = (cfg.n_embd, cfg.mlp_hidden());
        let eps = cfg.layer_norm_epsilon;
        let word = w.embedding(
            "embeddings.word_embeddings",
            cfg.padded_vocab_size(),
            h,
            None,
        )?;
        let types = w.load(
            "embeddings.token_type_embeddings.weight",
            &[cfg.type_vocab_size, h],
        )?;
        let emb_ln = layer_norm(w, "emb_ln", h, eps)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = format!("encoder.layers.{l}.");
            layers.push(Layer {
                wqkv: w.linear(&format!("{p}attn.Wqkv"), h, 3 * h, false, None)?,
                out_proj: w.linear(&format!("{p}attn.out_proj"), h, h, false, None)?,
                norm1: layer_norm(w, &format!("{p}norm1"), h, eps)?,
                fc11: w.linear(&format!("{p}mlp.fc11"), h, inner, false, None)?,
                fc12: w.linear(&format!("{p}mlp.fc12"), h, inner, false, None)?,
                fc2: w.linear(&format!("{p}mlp.fc2"), inner, h, false, None)?,
                norm2: layer_norm(w, &format!("{p}norm2"), h, eps)?,
            });
        }
        if w.contains("embeddings.position_embeddings.weight") {
            return Err(tensor_error(
                "embeddings.position_embeddings.weight",
                "absolute position table in a rotary checkpoint",
            ));
        }
        Ok(Self {
            cfg: cfg.clone(),
            word,
            token_type: types[..h].to_vec(),
            emb_ln,
            layers,
        })
    }

    pub fn config(&self) -> &NomicBertConfig {
        &self.cfg
    }

    /// The rotary `inv_freq` an `n`-token forward uses.
    pub fn inv_freq_for(&self, n: usize) -> Vec<f32> {
        inv_freq(ntk_base(&self.cfg, n), self.cfg.head_dim())
    }

    /// `last_hidden_state [ids.len(), n_embd]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_traced(ids, |_, _| {})
    }

    /// [`forward`](Self::forward), handing `trace` state 0 (`emb_ln`
    /// output) and state `l + 1` (output of layer `l`).
    pub fn forward_traced(
        &self,
        ids: &[u32],
        mut trace: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let seq = ids.len();
        if seq == 0 {
            return Err(input_error("empty token sequence"));
        }
        let h = c.n_embd;
        let mut x = self.word.gather(ids)?;
        for row in x.chunks_exact_mut(h) {
            for (v, t) in row.iter_mut().zip(&self.token_type) {
                *v += t;
            }
        }
        self.emb_ln.apply(&mut x, seq);
        trace(0, &x);

        let rope = rope_tables(seq, &self.inv_freq_for(seq));
        let inner = c.mlp_hidden();
        for (l, layer) in self.layers.iter().enumerate() {
            let a = self.attention(layer, &x, seq, &rope);
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }
            layer.norm1.apply(&mut x, seq);

            let mut y = layer.fc11.forward(&x, seq);
            let mut gate = layer.fc12.forward(&x, seq);
            debug_assert_eq!(gate.len(), seq * inner);
            ops::silu(&mut gate);
            for (yv, g) in y.iter_mut().zip(&gate) {
                *yv *= g;
            }
            let m = layer.fc2.forward(&y, seq);
            for (xv, mv) in x.iter_mut().zip(&m) {
                *xv += mv;
            }
            layer.norm2.apply(&mut x, seq);
            trace(l + 1, &x);
        }
        Ok(x)
    }

    fn attention(
        &self,
        layer: &Layer,
        x: &[f32],
        seq: usize,
        rope: &(Vec<f32>, Vec<f32>),
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, hd) = (c.n_embd, c.n_head, c.head_dim());
        let qkv = layer.wqkv.forward(x, seq);
        let mut q = ops::split_heads_strided(&qkv, 3 * h, 0, seq, nh, hd);
        let mut k = ops::split_heads_strided(&qkv, 3 * h, h, seq, nh, hd);
        let v = ops::split_heads_strided(&qkv, 3 * h, 2 * h, seq, nh, hd);
        ops::rope_neox(&mut q, nh, seq, hd, &rope.0, &rope.1);
        ops::rope_neox(&mut k, nh, seq, hd, &rope.0, &rope.1);
        let out = ops::mha(&q, &k, &v, None, nh, seq, seq, hd, c.attn_scale());
        layer
            .out_proj
            .forward(&ops::merge_heads(&out, seq, nh, hd), seq)
    }
}
