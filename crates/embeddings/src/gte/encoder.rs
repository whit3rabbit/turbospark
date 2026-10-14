//! The GTE encoder (`GteModel` in the pinned
//! `modeling_hf_alibaba_nlp_gte.py`), one unpadded sequence per call.
//!
//! ```text
//! x   = LayerNorm(word[id] + token_type[0])            (no absolute positions)
//! per layer (post-LN):
//!   q, k, v = split(qkv_proj(x))                       (thirds, in q, k, v order)
//!   q, k    = rope_neox(q), rope_neox(k)               (rotate_half layout)
//!   x = attn_ln(x + o_proj(softmax(q k^T / sqrt(d)) v))
//!   up, gate = split(up_gate_proj(x))                  (first half up, second gate)
//!   x = mlp_ln(x + down_proj(gelu_erf(gate) * up))
//! ```
//!
//! There is no final norm: the last layer's `mlp_ln` output is the
//! `last_hidden_state`. The pooler is not built (`add_pooling_layer`
//! defaults to false, modeling:832) and the checkpoint has no pooler
//! tensors.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::config::GteConfig;
use crate::error::{input_error, tensor_error};
use crate::weights::{EmbeddingTable, Weights};
use crate::Result;

const WORD_TABLE: &str = "embeddings.word_embeddings.weight";

/// `1 / theta^(2i / dim)` for `i in 0..dim/2`, as `RotaryEmbedding`
/// computes it (modeling:189): the exponent is an f32 division, theta is
/// cast to f32, and the power is evaluated in f64 and rounded once.
/// torch's f32 pow is within 1 ulp of that rounding; the fixture test
/// measures the result against the reference buffer.
pub fn inv_freq(theta: f64, dim: usize) -> Vec<f32> {
    let base = f64::from(theta as f32);
    (0..dim / 2)
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0f32 / base.powf(f64::from(e)) as f32
        })
        .collect()
}

/// cos/sin tables `[seq, dim/2]`: angle `t * inv_freq` as one f32
/// product (the reference einsum), cos and sin of that f32 angle.
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

/// Word table. The pinned vocabulary is 250048 x 768 f32 (768 MB) and a
/// forward reads one row per token, so an F32 table is read row by row
/// from the mapped file instead of copied to the heap.
enum TokenTable {
    Mapped {
        weights: Weights,
        rows: usize,
        cols: usize,
    },
    Dense(EmbeddingTable),
}

impl std::fmt::Debug for TokenTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenTable::Mapped { rows, cols, .. } => write!(f, "Mapped({rows}x{cols})"),
            TokenTable::Dense(t) => write!(f, "Dense({}x{})", t.rows(), t.cols()),
        }
    }
}

impl TokenTable {
    fn gather(&self, ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            TokenTable::Dense(t) => t.gather(ids),
            TokenTable::Mapped {
                weights,
                rows,
                cols,
            } => {
                let bytes = weights.file(WORD_TABLE)?.raw_bytes(WORD_TABLE)?;
                let mut out = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    let r = id as usize;
                    if r >= *rows {
                        return Err(input_error(format!(
                            "token id {id} out of range for {rows} embedding rows"
                        )));
                    }
                    let row = &bytes[r * cols * 4..(r + 1) * cols * 4];
                    out.extend(
                        row.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                }
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Layer {
    qkv: Linear,
    o: Linear,
    attn_ln: LayerNorm,
    up_gate: Linear,
    down: Linear,
    mlp_ln: LayerNorm,
}

/// The loaded encoder.
#[derive(Debug)]
pub struct Encoder {
    cfg: GteConfig,
    tokens: TokenTable,
    /// Row 0 of `token_type_embeddings`: the reference zeroes every
    /// token-type id when type_vocab_size < 2 and defaults them to 0
    /// otherwise (modeling:397-407), and XLM-R tokenizers emit none.
    token_type: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<Layer>,
    inv_freq: Vec<f32>,
}

fn layer_norm(w: &Weights, base: &str, width: usize, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.load(&format!("{base}.weight"), &[width])?,
        Some(w.load(&format!("{base}.bias"), &[width])?),
        eps,
    ))
}

impl Encoder {
    /// Loads every tensor, shape-checked against `cfg`. Takes `w` so an
    /// F32 word table can stay mapped.
    pub fn load(w: Weights, cfg: &GteConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let eps = cfg.layer_norm_eps;
        let shape = w.shape(WORD_TABLE)?;
        if shape != [cfg.vocab_size, h] {
            return Err(tensor_error(
                WORD_TABLE,
                format!("expected shape [{}, {h}], got {shape:?}", cfg.vocab_size),
            ));
        }
        let types = w.load(
            "embeddings.token_type_embeddings.weight",
            &[cfg.type_vocab_size, h],
        )?;
        let token_type = types[..h].to_vec();
        let emb_ln = layer_norm(&w, "embeddings.LayerNorm", h, eps)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("encoder.layer.{l}.");
            layers.push(Layer {
                qkv: w.linear(&format!("{p}attention.qkv_proj"), h, 3 * h, true, None)?,
                o: w.linear(&format!("{p}attention.o_proj"), h, h, true, None)?,
                attn_ln: layer_norm(&w, &format!("{p}attn_ln"), h, eps)?,
                up_gate: w.linear(&format!("{p}mlp.up_gate_proj"), h, 2 * i, false, None)?,
                down: w.linear(&format!("{p}mlp.down_proj"), i, h, true, None)?,
                mlp_ln: layer_norm(&w, &format!("{p}mlp_ln"), h, eps)?,
            });
        }
        let dtype = w
            .file(WORD_TABLE)?
            .descriptor(WORD_TABLE)
            .map(|d| d.dtype.to_uppercase())
            .unwrap_or_default();
        let tokens = if dtype == "F32" {
            let len = w.file(WORD_TABLE)?.raw_bytes(WORD_TABLE)?.len();
            if len != cfg.vocab_size * h * 4 {
                return Err(tensor_error(
                    WORD_TABLE,
                    format!("{len} bytes for an F32 table"),
                ));
            }
            TokenTable::Mapped {
                weights: w,
                rows: cfg.vocab_size,
                cols: h,
            }
        } else {
            TokenTable::Dense(w.embedding("embeddings.word_embeddings", cfg.vocab_size, h, None)?)
        };
        Ok(Self {
            inv_freq: inv_freq(cfg.rope_theta, cfg.head_dim()),
            cfg: cfg.clone(),
            tokens,
            token_type,
            emb_ln,
            layers,
        })
    }

    pub fn config(&self) -> &GteConfig {
        &self.cfg
    }

    /// The rotary `inv_freq` buffer `[head_dim / 2]`.
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// `last_hidden_state [ids.len(), hidden]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_traced(ids, |_, _| {})
    }

    /// [`forward`](Self::forward), handing `trace` state 0 (embedding
    /// LayerNorm output) and state `l + 1` (output of layer `l`).
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
        let h = c.hidden_size;
        let mut x = self.tokens.gather(ids)?;
        for row in x.chunks_exact_mut(h) {
            for (v, t) in row.iter_mut().zip(&self.token_type) {
                *v += t;
            }
        }
        self.emb_ln.apply(&mut x, seq);
        trace(0, &x);

        let rope = rope_tables(seq, &self.inv_freq);
        for (l, layer) in self.layers.iter().enumerate() {
            let a = self.attention(layer, &x, seq, &rope);
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }
            layer.attn_ln.apply(&mut x, seq);

            let ug = layer.up_gate.forward(&x, seq);
            let inter = c.intermediate_size;
            let mut act = vec![0.0f32; seq * inter];
            for (dst, row) in act.chunks_exact_mut(inter).zip(ug.chunks_exact(2 * inter)) {
                let (up, gate) = row.split_at(inter);
                dst.copy_from_slice(gate);
                ops::gelu_erf(dst);
                for (g, u) in dst.iter_mut().zip(up) {
                    *g *= u;
                }
            }
            let m = layer.down.forward(&act, seq);
            for (xv, mv) in x.iter_mut().zip(&m) {
                *xv += mv;
            }
            layer.mlp_ln.apply(&mut x, seq);
            trace(l + 1, &x);
        }
        Ok(x)
    }

    /// Self-attention for one layer: `x [seq, hidden]` to the `o_proj`
    /// output.
    fn attention(
        &self,
        layer: &Layer,
        x: &[f32],
        seq: usize,
        rope: &(Vec<f32>, Vec<f32>),
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (h, nh, hd) = (c.hidden_size, c.num_attention_heads, c.head_dim());
        let qkv = layer.qkv.forward(x, seq);
        let mut q = ops::split_heads_strided(&qkv, 3 * h, 0, seq, nh, hd);
        let mut k = ops::split_heads_strided(&qkv, 3 * h, h, seq, nh, hd);
        let v = ops::split_heads_strided(&qkv, 3 * h, 2 * h, seq, nh, hd);
        ops::rope_neox(&mut q, nh, seq, hd, &rope.0, &rope.1);
        ops::rope_neox(&mut k, nh, seq, hd, &rope.0, &rope.1);
        let out = ops::mha(&q, &k, &v, None, nh, seq, seq, hd, c.attn_scale());
        layer.o.forward(&ops::merge_heads(&out, seq, nh, hd), seq)
    }
}
