//! Gemma3 text backbone driven as a bidirectional encoder.
//!
//! Reference: transformers 4.57.1 `models/gemma3/modeling_gemma3.py`
//! (`Gemma3TextModel` with `use_bidirectional_attention = true`) and
//! `masking_utils.py`. One unpadded sequence per call, so the padding mask
//! is all ones and only the per-layer span matters:
//!
//! - full layers: every token sees every token;
//! - sliding layers: the causal sliding mask OR'd with
//!   `_bidirectional_window_overlay`, which reduces to `|q - k| < W`
//!   (strict), where `W = sliding_window // 2 + 1` because
//!   `Gemma3TextConfig` halves the configured window for bidirectional
//!   models (see [`GemmaConfig::band_width`]). Shorter sequences skip the
//!   band, which is exact.
//!
//! Block (four-norm sandwich):
//!
//! ```text
//! h   = x + post_attention_layernorm(attn(input_layernorm(x)))
//! out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
//! ```
//!
//! Every Gemma RMSNorm is `x * rsqrt(mean(x^2) + eps) * (1 + w)`; the
//! `1 + w` is folded in at load. The token embedding is multiplied by
//! `sqrt(hidden_size)` after the gather.

use turbospark_audio::nn::Linear;
use turbospark_audio::ops;

use super::config::{GemmaConfig, LayerType};
use crate::error::{input_error, tensor_error, EmbeddingError};
use crate::weights::{EmbeddingTable, Weights};

/// Gemma RMSNorm with the `1 + w` scale precomputed.
#[derive(Debug, Clone)]
pub(crate) struct GemmaNorm {
    scale: Vec<f32>,
    eps: f32,
}

impl GemmaNorm {
    fn load(w: &Weights, name: &str, width: usize, eps: f32) -> Result<Self, EmbeddingError> {
        let raw = w.load(name, &[width])?;
        Ok(Self::from_raw(&raw, eps))
    }

    /// `raw` is the checkpoint weight; the reference adds 1 in f32.
    pub(crate) fn from_raw(raw: &[f32], eps: f32) -> Self {
        Self {
            scale: raw.iter().map(|w| 1.0 + w).collect(),
            eps,
        }
    }

    /// Normalizes each `width`-row of `x` in place. The mean of squares
    /// accumulates in f64 so a 768- or 1152-wide row does not lose bits
    /// to a sequential f32 sum (torch reduces pairwise).
    pub(crate) fn apply(&self, x: &mut [f32]) {
        let width = self.scale.len();
        debug_assert_eq!(x.len() % width, 0);
        for row in x.chunks_exact_mut(width) {
            let ms = row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / width as f64;
            let inv = 1.0 / (ms as f32 + self.eps).sqrt();
            for (v, s) in row.iter_mut().zip(&self.scale) {
                *v = (*v * inv) * s;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Layer {
    kind: LayerType,
    input_norm: GemmaNorm,
    post_attn_norm: GemmaNorm,
    pre_ff_norm: GemmaNorm,
    post_ff_norm: GemmaNorm,
    q_norm: GemmaNorm,
    k_norm: GemmaNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// RoPE tables for one base, matching `Gemma3RotaryEmbedding` (default
/// rope type): `inv_freq_i = 1 / base^(2i / dim)` in f32, angle
/// `pos * inv_freq_i` in f32, then f32 cos/sin.
fn rope_tables(seq: usize, dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        for (i, &f) in inv.iter().enumerate() {
            let a = t as f32 * f;
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// The Gemma3 text transformer: embedding, layers, final norm.
#[derive(Debug, Clone)]
pub struct Backbone {
    config: GemmaConfig,
    embed: EmbeddingTable,
    layers: Vec<Layer>,
    final_norm: GemmaNorm,
}

impl Backbone {
    /// Loads from `weights` with tensor names `{prefix}embed_tokens.weight`,
    /// `{prefix}layers.{i}.*`, `{prefix}norm.weight` (`prefix` is `""` for
    /// the HF checkpoint and `"model."` for the MLX conversions).
    pub fn load(
        weights: &Weights,
        prefix: &str,
        config: &GemmaConfig,
    ) -> Result<Self, EmbeddingError> {
        let c = config;
        let q_scheme = c.quantization;
        let eps = c.rms_norm_eps;
        let (h, hd) = (c.hidden_size, c.head_dim);
        let q_out = c.num_attention_heads * hd;
        let kv_out = c.num_key_value_heads * hd;
        let embed =
            weights.embedding(&format!("{prefix}embed_tokens"), c.vocab_size, h, q_scheme)?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers());
        for (i, &kind) in c.layer_types.iter().enumerate() {
            let p = format!("{prefix}layers.{i}.");
            let norm = |n: &str, width: usize| {
                GemmaNorm::load(weights, &format!("{p}{n}.weight"), width, eps)
            };
            let lin = |n: &str, input: usize, output: usize| {
                weights.linear(&format!("{p}{n}"), input, output, false, q_scheme)
            };
            layers.push(Layer {
                kind,
                input_norm: norm("input_layernorm", h)?,
                post_attn_norm: norm("post_attention_layernorm", h)?,
                pre_ff_norm: norm("pre_feedforward_layernorm", h)?,
                post_ff_norm: norm("post_feedforward_layernorm", h)?,
                q_norm: norm("self_attn.q_norm", hd)?,
                k_norm: norm("self_attn.k_norm", hd)?,
                q: lin("self_attn.q_proj", h, q_out)?,
                k: lin("self_attn.k_proj", h, kv_out)?,
                v: lin("self_attn.v_proj", h, kv_out)?,
                o: lin("self_attn.o_proj", q_out, h)?,
                gate: lin("mlp.gate_proj", h, c.intermediate_size)?,
                up: lin("mlp.up_proj", h, c.intermediate_size)?,
                down: lin("mlp.down_proj", c.intermediate_size, h)?,
            });
        }
        let final_norm = GemmaNorm::load(weights, &format!("{prefix}norm.weight"), h, eps)?;
        if embed.cols() != h {
            return Err(tensor_error(
                &format!("{prefix}embed_tokens.weight"),
                "width != hidden_size",
            ));
        }
        Ok(Self {
            config: config.clone(),
            embed,
            layers,
            final_norm,
        })
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.config
    }

    /// Last hidden state `[ids.len(), hidden_size]` (after the final norm).
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        self.forward_traced(ids, |_, _| {})
    }

    /// [`forward`](Self::forward), handing `trace` the same states HF
    /// returns as `hidden_states`: index 0 is the scaled embedding, index
    /// `i` the output of layer `i - 1` for `i < num_layers`, and index
    /// `num_layers` the final-norm output (it replaces the last layer's
    /// raw output, as in the reference).
    pub fn forward_traced(
        &self,
        ids: &[u32],
        mut trace: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>, EmbeddingError> {
        let c = &self.config;
        let seq = ids.len();
        if seq == 0 {
            return Err(input_error("empty token sequence"));
        }
        let h = c.hidden_size;
        let mut x = self.embed.gather(ids)?;
        let scale = c.embed_scale();
        for v in x.iter_mut() {
            *v *= scale;
        }
        trace(0, &x);

        let rope_full = rope_tables(seq, c.head_dim, c.rope_theta);
        let rope_local = rope_tables(seq, c.head_dim, c.rope_local_base_freq);
        // Built once per call and only when the band actually cuts:
        // every distance is below the window otherwise.
        let w = c.band_width();
        let band = (seq > w).then(|| band_mask(seq, w));

        let n_layers = self.layers.len();
        for (li, layer) in self.layers.iter().enumerate() {
            let (rope, mask) = match layer.kind {
                LayerType::Full => (&rope_full, None),
                LayerType::Sliding => (&rope_local, band.as_deref()),
            };
            let mut a = x.clone();
            layer.input_norm.apply(&mut a);
            let mut a = self.attention(layer, &a, seq, rope, mask);
            layer.post_attn_norm.apply(&mut a);
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }

            let mut m = x.clone();
            layer.pre_ff_norm.apply(&mut m);
            let mut gate = layer.gate.forward(&m, seq);
            let up = layer.up.forward(&m, seq);
            ops::gelu_tanh(&mut gate);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g *= u;
            }
            let mut m = layer.down.forward(&gate, seq);
            layer.post_ff_norm.apply(&mut m);
            for (xv, mv) in x.iter_mut().zip(&m) {
                *xv += mv;
            }
            if li + 1 < n_layers {
                trace(li + 1, &x);
            }
        }
        self.final_norm.apply(&mut x);
        trace(n_layers, &x);
        debug_assert_eq!(x.len(), seq * h);
        Ok(x)
    }

    /// Grouped-query attention for one layer: `x [seq, hidden]` (already
    /// input-normed) to the `o_proj` output `[seq, hidden]`.
    fn attention(
        &self,
        layer: &Layer,
        x: &[f32],
        seq: usize,
        rope: &(Vec<f32>, Vec<f32>),
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let c = &self.config;
        let (nh, nkv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let mut q = ops::split_heads(&layer.q.forward(x, seq), seq, nh, hd);
        let mut k = ops::split_heads(&layer.k.forward(x, seq), seq, nkv, hd);
        let v = ops::split_heads(&layer.v.forward(x, seq), seq, nkv, hd);
        layer.q_norm.apply(&mut q);
        layer.k_norm.apply(&mut k);
        ops::rope_neox(&mut q, nh, seq, hd, &rope.0, &rope.1);
        ops::rope_neox(&mut k, nkv, seq, hd, &rope.0, &rope.1);

        // Query head h reads kv head h / group (HF repeat_kv order).
        let group = nh / nkv;
        let scale = c.attn_scale();
        let mut out = vec![0.0f32; nh * seq * hd];
        for (hq, out_plane) in out.chunks_exact_mut(seq * hd).enumerate() {
            let kv_base = (hq / group) * seq * hd;
            ops::sdpa_strided_into(
                out_plane,
                &q[hq * seq * hd..(hq + 1) * seq * hd],
                &k,
                kv_base,
                hd,
                &v,
                kv_base,
                hd,
                mask,
                seq,
                seq,
                hd,
                hd,
                scale,
            );
        }
        layer.o.forward(&ops::merge_heads(&out, seq, nh, hd), seq)
    }
}

/// Additive `[seq, seq]` mask: 0 where `|q - k| < window`, `-inf`
/// elsewhere. A masked score becomes `exp(-inf) = 0` in the softmax, so
/// the result equals attending over the band only.
pub(crate) fn band_mask(seq: usize, window: usize) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; seq * seq];
    for q in 0..seq {
        let lo = q.saturating_sub(window - 1);
        let hi = (q + window).min(seq);
        m[q * seq + lo..q * seq + hi].fill(0.0);
    }
    m
}
