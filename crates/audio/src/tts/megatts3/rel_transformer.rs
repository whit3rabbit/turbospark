//! `RelTransformerEncoder` from MegaTTS3's `ar_dur/commons/rel_transformer.py`.
//!
//! Shared by the DiT phone encoder and the duration LM text encoder. It is a
//! conv prenet (three k5 conv + LayerNorm + ReLU layers with a residual 1x1
//! projection), then pre-LN transformer layers whose attention adds a shared
//! relative-position term (window 4, keys and values), a ReLU conv FFN, and a
//! final LayerNorm.
//!
//! Activations are time-major `[T, C]`; only the k>1 convs transpose to the
//! channel-major layout `ops::conv1d` takes. Inference never pads a batch,
//! so the padding mask is all ones and is not materialized; zero token ids
//! (the pad id) are rejected instead of silently masked.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, load_mlx_conv_weight, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::{ops, Result, SpeechError};

/// The reference hardcodes the prenet: three k5 layers.
const PRE_LAYERS: usize = 3;
const PRE_KERNEL: usize = 5;
/// LayerNorm epsilon of this module's own `LayerNorm` (not `nn.LayerNorm`).
const LN_EPS: f32 = 1e-4;
const MAX_LAYERS: usize = 64;
const MAX_CHANNELS: usize = 8192;

#[derive(Debug, Clone, PartialEq)]
pub struct RelTransformerConfig {
    pub n_vocab: usize,
    pub hidden: usize,
    pub filter: usize,
    pub heads: usize,
    pub layers: usize,
    /// FFN conv kernel (3 in both MegaTTS3 users).
    pub kernel: usize,
}

impl RelTransformerConfig {
    pub fn validate(&self) -> Result<()> {
        let bad = |f: &str, why: &str| SpeechError::BadConfig {
            field: format!("rel_transformer.{f}"),
            why: why.to_string(),
        };
        if self.n_vocab == 0 || self.n_vocab > 1 << 20 {
            return Err(bad("n_vocab", "must be in 1..=2^20"));
        }
        if self.hidden == 0
            || self.hidden > MAX_CHANNELS
            || self.filter == 0
            || self.filter > MAX_CHANNELS
        {
            return Err(bad("hidden/filter", "must be in 1..=8192"));
        }
        if self.heads == 0 || self.hidden % self.heads != 0 {
            return Err(bad("heads", "must divide hidden"));
        }
        if self.layers == 0 || self.layers > MAX_LAYERS {
            return Err(bad("layers", "must be in 1..=64"));
        }
        if self.kernel == 0 || self.kernel % 2 == 0 || self.kernel > 15 {
            return Err(bad("kernel", "must be odd and at most 15"));
        }
        Ok(())
    }
}

/// The reference `LayerNorm`: normalize over channels with `gamma`/`beta`.
struct ChannelNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl ChannelNorm {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            gamma: load_f32_shaped(file, &format!("{prefix}.gamma"), &[c])?,
            beta: load_f32_shaped(file, &format!("{prefix}.beta"), &[c])?,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(
            x,
            rows,
            self.gamma.len(),
            &self.gamma,
            Some(&self.beta),
            LN_EPS,
        );
    }
}

fn conv_k(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) -> Result<Conv1d> {
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride: 1,
        padding: kernel / 2,
        dilation: 1,
        groups: 1,
        weight: load_mlx_conv_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)?,
        bias: load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Required)?,
    })
}

/// A kernel-1 conv is a `Linear`: the converter stores `[out, 1, in]`, whose
/// flat layout is exactly `[out, in]`.
fn conv1x1(file: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Linear> {
    let w = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, 1, in_ch])?;
    let b = load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?;
    Ok(Linear::new(w, Some(b), in_ch, out_ch))
}

/// The reference `PosEmb(dim)`: `[sin(p f_i) | cos(p f_i)]` for positions
/// `0..n`, with `f_i = exp(-ln(10000) * i / (dim / 2 - 1))`.
pub(crate) fn sinusoid_positions(n: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = (10_000.0f32).ln() / (half as f32 - 1.0);
    let freqs: Vec<f32> = (0..half).map(|i| (-(i as f32) * step).exp()).collect();
    let mut out = vec![0.0f32; n * dim];
    for p in 0..n {
        for (i, f) in freqs.iter().enumerate() {
            let a = p as f32 * f;
            out[p * dim + i] = a.sin();
            out[p * dim + half + i] = a.cos();
        }
    }
    out
}

/// `[T, C]` -> `[C, T]` and back.
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

fn conv_tm(conv: &Conv1d, x: &[f32], frames: usize) -> Vec<f32> {
    let cm = transpose(x, frames, conv.in_ch);
    let out = conv.forward(&cm);
    transpose(&out, conv.out_ch, frames)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// Shared across heads: `[2 * window + 1, head_dim]`.
    emb_k: Vec<f32>,
    emb_v: Vec<f32>,
    heads: usize,
    window: usize,
}

impl Attention {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize, heads: usize) -> Result<Self> {
        let dk = c / heads;
        let rel = file
            .descriptor(&format!("{prefix}.emb_rel_k"))
            .ok_or_else(|| SpeechError::Tensor {
                name: format!("{prefix}.emb_rel_k"),
                why: "missing from checkpoint".to_string(),
            })?;
        // [1, 2w + 1, dk]; the window is whatever the checkpoint was trained with.
        let slots = match rel.shape.as_slice() {
            [1, n, d] if *d == dk && n % 2 == 1 && *n <= 129 => *n,
            other => {
                return Err(SpeechError::Tensor {
                    name: format!("{prefix}.emb_rel_k"),
                    why: format!("expected [1, odd, {dk}], got {other:?}"),
                })
            }
        };
        Ok(Self {
            q: conv1x1(file, &format!("{prefix}.conv_q"), c, c)?,
            k: conv1x1(file, &format!("{prefix}.conv_k"), c, c)?,
            v: conv1x1(file, &format!("{prefix}.conv_v"), c, c)?,
            o: conv1x1(file, &format!("{prefix}.conv_o"), c, c)?,
            emb_k: load_f32_shaped(file, &format!("{prefix}.emb_rel_k"), &[1, slots, dk])?,
            emb_v: load_f32_shaped(file, &format!("{prefix}.emb_rel_v"), &[1, slots, dk])?,
            heads,
            window: (slots - 1) / 2,
        })
    }

    /// `x [T, C] -> [T, C]`, full self-attention, no padding mask.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let c = self.q.input;
        let h = self.heads;
        let dk = c / h;
        let w = self.window;
        let q = self.q.forward(x, t);
        let k = self.k.forward(x, t);
        let v = self.v.forward(x, t);
        let inv = 1.0 / (dk as f32).sqrt();
        let mut out = vec![0.0f32; t * c];
        let mut scores = vec![0.0f32; t];
        for head in 0..h {
            let col = head * dk;
            for i in 0..t {
                let qi = &q[i * c + col..i * c + col + dk];
                for j in 0..t {
                    let kj = &k[j * c + col..j * c + col + dk];
                    let mut s: f32 = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
                    // Relative keys: only |j - i| <= window has an embedding;
                    // the reference zero-pads the rest, which adds exactly 0.
                    if j + w >= i && j <= i + w {
                        let e = &self.emb_k[(j + w - i) * dk..(j + w - i + 1) * dk];
                        s += qi.iter().zip(e).map(|(a, b)| a * b).sum::<f32>();
                    }
                    scores[j] = s * inv;
                }
                ops::softmax_row(&mut scores);
                let oi = &mut out[i * c + col..i * c + col + dk];
                for (j, &p) in scores.iter().enumerate() {
                    let vj = &v[j * c + col..j * c + col + dk];
                    for (o, &vv) in oi.iter_mut().zip(vj) {
                        *o += p * vv;
                    }
                    if j + w >= i && j <= i + w {
                        let e = &self.emb_v[(j + w - i) * dk..(j + w - i + 1) * dk];
                        for (o, &ev) in oi.iter_mut().zip(e) {
                            *o += p * ev;
                        }
                    }
                }
            }
        }
        self.o.forward(&out, t)
    }
}

struct Layer {
    attn: Attention,
    norm1: ChannelNorm,
    conv1: Conv1d,
    conv2: Linear,
    norm2: ChannelNorm,
}

pub struct RelTransformer {
    cfg: RelTransformerConfig,
    emb: Vec<f32>,
    pre_convs: Vec<(Conv1d, ChannelNorm)>,
    pre_proj: Linear,
    layers: Vec<Layer>,
    last_ln: ChannelNorm,
}

impl RelTransformer {
    /// Loads `{prefix}.emb`, `{prefix}.pre.*`, and `{prefix}.encoder.*`.
    pub fn load(file: &SafetensorsFile, prefix: &str, cfg: RelTransformerConfig) -> Result<Self> {
        cfg.validate()?;
        let c = cfg.hidden;
        let emb = load_f32_shaped(file, &format!("{prefix}.emb.weight"), &[cfg.n_vocab, c])?;
        let mut pre_convs = Vec::new();
        for i in 0..PRE_LAYERS {
            pre_convs.push((
                conv_k(
                    file,
                    &format!("{prefix}.pre.conv_layers.{i}"),
                    c,
                    c,
                    PRE_KERNEL,
                )?,
                ChannelNorm::load(file, &format!("{prefix}.pre.norm_layers.{i}"), c)?,
            ));
        }
        let pre_proj = conv1x1(file, &format!("{prefix}.pre.proj"), c, c)?;
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let e = format!("{prefix}.encoder");
            layers.push(Layer {
                attn: Attention::load(file, &format!("{e}.attn_layers.{l}"), c, cfg.heads)?,
                norm1: ChannelNorm::load(file, &format!("{e}.norm_layers_1.{l}"), c)?,
                conv1: conv_k(
                    file,
                    &format!("{e}.ffn_layers.{l}.conv_1"),
                    c,
                    cfg.filter,
                    cfg.kernel,
                )?,
                conv2: conv1x1(file, &format!("{e}.ffn_layers.{l}.conv_2"), cfg.filter, c)?,
                norm2: ChannelNorm::load(file, &format!("{e}.norm_layers_2.{l}"), c)?,
            });
        }
        let last_ln = ChannelNorm::load(file, &format!("{prefix}.encoder.last_ln"), c)?;
        Ok(Self {
            cfg,
            emb,
            pre_convs,
            pre_proj,
            layers,
            last_ln,
        })
    }

    pub fn config(&self) -> &RelTransformerConfig {
        &self.cfg
    }

    /// `tokens [T]`, `other [T, C]` (added after the `sqrt(C)`-scaled
    /// embedding) -> `[T, C]`.
    pub fn forward(&self, tokens: &[i32], other: &[f32]) -> Result<Vec<f32>> {
        let t = tokens.len();
        let c = self.cfg.hidden;
        if t == 0 {
            return Err(SpeechError::Input {
                why: "empty token sequence".to_string(),
            });
        }
        if other.len() != t * c {
            return Err(SpeechError::Input {
                why: format!(
                    "other_embeds has {} values, expected {}",
                    other.len(),
                    t * c
                ),
            });
        }
        if let Some(bad) = tokens
            .iter()
            .find(|&&id| id <= 0 || id as usize >= self.cfg.n_vocab)
        {
            return Err(SpeechError::Input {
                why: format!("token id {bad} outside 1..{}", self.cfg.n_vocab),
            });
        }
        let scale = (c as f32).sqrt();
        let mut x = ops::embedding(&self.emb, c, tokens);
        for (v, o) in x.iter_mut().zip(other) {
            *v = *v * scale + o;
        }

        // Prenet (ConvReluNorm): the conv chain starts from x, the residual
        // adds the projection of its result to the original x.
        let x_org = x.clone();
        for (conv, norm) in &self.pre_convs {
            x = conv_tm(conv, &x, t);
            norm.apply(&mut x, t);
            for v in x.iter_mut() {
                *v = v.max(0.0);
            }
        }
        let proj = self.pre_proj.forward(&x, t);
        x = x_org;
        for (v, p) in x.iter_mut().zip(&proj) {
            *v += p;
        }

        for layer in &self.layers {
            let mut xn = x.clone();
            layer.norm1.apply(&mut xn, t);
            let y = layer.attn.forward(&xn, t);
            for (v, a) in x.iter_mut().zip(&y) {
                *v += a;
            }
            let mut xn = x.clone();
            layer.norm2.apply(&mut xn, t);
            let mut h = conv_tm(&layer.conv1, &xn, t);
            for v in h.iter_mut() {
                *v = v.max(0.0);
            }
            let y = layer.conv2.forward(&h, t);
            for (v, a) in x.iter_mut().zip(&y) {
                *v += a;
            }
        }
        self.last_ln.apply(&mut x, t);
        Ok(x)
    }
}
