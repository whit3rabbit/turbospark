//! The AudioDiT transformer (`dit.py`): Qwen2-style rotary embeddings
//! (base 100000, half-split rotation), global AdaLN conditioning with
//! per-block scale/shift residuals, self attention plus cross attention
//! to the text embedding, ConvNeXtV2 text processing, long skip
//! connection with the reference's mid-stack add at the repa layer, and
//! the final AdaLayerNormZero head.

use crate::nn::Linear;
use crate::{ops, Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {shape:?} elements, got {:?}", descriptor.shape),
        });
    }
    crate::nn::load_tensor(file, name, &descriptor.shape)
}

fn linear(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    let weight = tensor(file, &format!("{name}.weight"), &[input * output])?;
    let bias_value = if bias {
        Some(tensor(file, &format!("{name}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias_value, input, output))
}

/// Affine-free layer norm over the last axis (`_layer_norm`).
pub(crate) fn layer_norm_in_place(x: &mut [f32], cols: usize, eps: f32) {
    for row in x.chunks_exact_mut(cols) {
        let mean: f64 = row.iter().map(|v| *v as f64).sum::<f64>() / cols as f64;
        let var: f64 = row
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / cols as f64;
        let inv = ((var + eps as f64).sqrt()).recip() as f32;
        for v in row.iter_mut() {
            *v = (*v - mean as f32) * inv;
        }
    }
}

/// `_modulate`: layer norm (no affine) then `x * (1 + scale) + shift`
/// with the per-batch scale/shift broadcast over the time axis.
fn modulate(x: &mut [f32], cols: usize, eps: f32, scale: &[f32], shift: &[f32]) {
    layer_norm_in_place(x, cols, eps);
    for row in x.chunks_exact_mut(cols) {
        for (c, v) in row.iter_mut().enumerate() {
            *v = *v * (1.0 + scale[c]) + shift[c];
        }
    }
}

/// The custom `RMSNorm` (weight applied after the float32 norm).
struct Rms {
    weight: Vec<f32>,
    eps: f32,
}

impl Rms {
    fn load(file: &SafetensorsFile, name: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: tensor(file, &format!("{name}.weight"), &[dim])?,
            eps,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        let dim = self.weight.len();
        for row in x.chunks_exact_mut(dim).take(rows) {
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for (v, w) in row.iter_mut().zip(&self.weight) {
                *v = *v * inv * w;
            }
        }
    }
}

/// The GRN inside ConvNeXtV2 (`gamma * (x * nx) + beta + x`); the norm
/// axis is time, so with B = 1 every token shares one `gx` row.
struct Grn {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    dim: usize,
}

impl Grn {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: tensor(file, &format!("{prefix}.gamma"), &[dim])?,
            beta: tensor(file, &format!("{prefix}.beta"), &[dim])?,
            dim,
        })
    }

    fn apply(&self, x: &mut [f32], seq: usize) {
        let dim = self.dim;
        let mut gx = vec![0.0f32; dim];
        // gx[c] = sqrt(sum over tokens of x[t, c]^2).
        for t in 0..seq {
            for (c, v) in x[t * dim..(t + 1) * dim].iter().enumerate() {
                gx[c] += v * v;
            }
        }
        for g in &mut gx {
            *g = g.sqrt();
        }
        let mean: f32 = gx.iter().sum::<f32>() / dim as f32;
        for g in &mut gx {
            *g /= mean + 1e-6;
        }
        for t in 0..seq {
            for (c, v) in x[t * dim..(t + 1) * dim].iter_mut().enumerate() {
                let original = *v;
                *v = self.gamma[c] * (original * gx[c]) + self.beta[c] + original;
            }
        }
    }
}

/// The ConvNeXtV2 text-processing block (depthwise conv channels-last,
/// then the pointwise bottleneck with the GRN).
struct ConvNextV2Block {
    dwconv_weight: Vec<f32>,
    dwconv_bias: Vec<f32>,
    kernel: usize,
    padding: usize,
    norm: crate::nn::LayerNorm,
    pwconv1: Linear,
    grn: Grn,
    pwconv2: Linear,
    dim: usize,
}

impl ConvNextV2Block {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        inter: usize,
        eps: f32,
    ) -> Result<Self> {
        let kernel = 7usize;
        Ok(Self {
            dwconv_weight: tensor(file, &format!("{prefix}.dwconv_weight"), &[dim * kernel])?,
            dwconv_bias: tensor(file, &format!("{prefix}.dwconv_bias"), &[dim])?,
            kernel,
            padding: (kernel - 1) / 2,
            norm: crate::nn::LayerNorm::load(file, &format!("{prefix}.norm"), dim, eps)?,
            pwconv1: linear(file, &format!("{prefix}.pwconv1"), dim, inter, true)?,
            grn: Grn::load(file, &format!("{prefix}.grn"), inter)?,
            pwconv2: linear(file, &format!("{prefix}.pwconv2"), inter, dim, true)?,
            dim,
        })
    }

    /// `x [seq, dim] -> [seq, dim]` (residual).
    fn forward(&self, x: &mut Vec<f32>, seq: usize) -> Result<()> {
        let dim = self.dim;
        let residual = std::mem::take(x);
        let mut h = vec![0.0f32; seq * dim];
        for t in 0..seq {
            for c in 0..dim {
                let mut acc = self.dwconv_bias[c];
                for k in 0..self.kernel {
                    let src = t as isize + k as isize - self.padding as isize;
                    if src < 0 || src as usize >= seq {
                        continue;
                    }
                    acc +=
                        residual[src as usize * dim + c] * self.dwconv_weight[c * self.kernel + k];
                }
                h[t * dim + c] = acc;
            }
        }
        self.norm.apply(&mut h, seq);
        let mut wide = self.pwconv1.forward(&h, seq);
        ops::silu(&mut wide);
        self.grn.apply(&mut wide, seq);
        let out = self.pwconv2.forward(&wide, seq);
        let mut next = residual;
        for (dst, o) in next.iter_mut().zip(out) {
            *dst += o;
        }
        *x = next;
        Ok(())
    }
}

/// The `Embedder`: linear -> SiLU -> linear with mask zeroing applied
/// to the input and the output rows.
struct Embedder {
    proj0: Linear,
    proj1: Linear,
    input: usize,
    output: usize,
}

impl Embedder {
    fn load(file: &SafetensorsFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            proj0: linear(file, &format!("{prefix}.proj.0"), input, output, true)?,
            proj1: linear(file, &format!("{prefix}.proj.1"), output, output, true)?,
            input,
            output,
        })
    }

    /// `x [seq, input] -> [seq, output]`; `mask` (true = keep).
    fn forward(&self, x: &[f32], seq: usize, mask: Option<&[bool]>) -> Vec<f32> {
        let mut h = x.to_vec();
        if let Some(mask) = mask {
            for (r, keep) in mask.iter().enumerate().take(seq) {
                if !keep {
                    for v in &mut h[r * self.input..(r + 1) * self.input] {
                        *v = 0.0;
                    }
                }
            }
        }
        let projected = self.proj0.forward(&h, seq);
        let mut wide = projected;
        ops::silu(&mut wide);
        let mut out = self.proj1.forward(&wide, seq);
        if let Some(mask) = mask {
            for (r, keep) in mask.iter().enumerate().take(seq) {
                if !keep {
                    for v in &mut out[r * self.output..(r + 1) * self.output] {
                        *v = 0.0;
                    }
                }
            }
        }
        out
    }
}

/// `TimestepEmbedding`: the sinusoidal table (sin half, then cos half)
/// through a two-layer MLP.
struct TimestepEmbedding {
    mlp0: Linear,
    mlp1: Linear,
    freq_dim: usize,
}

impl TimestepEmbedding {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        let freq_dim = 256usize;
        Ok(Self {
            mlp0: linear(file, &format!("{prefix}.time_mlp.0"), freq_dim, dim, true)?,
            mlp1: linear(file, &format!("{prefix}.time_mlp.1"), dim, dim, true)?,
            freq_dim,
        })
    }

    fn forward(&self, timestep: f32) -> Vec<f32> {
        let half = self.freq_dim / 2;
        let mut emb = vec![0.0f32; self.freq_dim];
        for d in 0..half {
            let rate = (-(d as f64) * 10000f64.ln() / (half - 1) as f64).exp();
            let scaled = 1000.0f32 * timestep * rate as f32;
            emb[d] = scaled.sin();
            emb[half + d] = scaled.cos();
        }
        let mut h = self.mlp0.forward(&emb, 1);
        ops::silu(&mut h);
        self.mlp1.forward(&h, 1)
    }
}

/// One self or cross attention block with optional qk RMS norms and
/// rotary embeddings.
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: Option<Rms>,
    k_norm: Option<Rms>,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input: usize,
        output: usize,
        heads: usize,
        head_dim: usize,
        qk_norm: bool,
        eps: f32,
        bias: bool,
    ) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(Self {
            to_q: linear(file, &format!("{prefix}.to_q"), input, inner, bias)?,
            to_k: linear(file, &format!("{prefix}.to_k"), output, inner, bias)?,
            to_v: linear(file, &format!("{prefix}.to_v"), output, inner, bias)?,
            to_out: linear(file, &format!("{prefix}.to_out"), inner, input, bias)?,
            q_norm: if qk_norm {
                Some(Rms::load(file, &format!("{prefix}.q_norm"), inner, eps)?)
            } else {
                None
            },
            k_norm: if qk_norm {
                Some(Rms::load(file, &format!("{prefix}.k_norm"), inner, eps)?)
            } else {
                None
            },
            heads,
            head_dim,
        })
    }

    /// `x [seq, input]`, `cond [cond_seq, output]`; `rope`/`cond_rope`
    /// carry the `(cos, sin)` tables `[seq, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        cond: &[f32],
        cond_seq: usize,
        cond_mask: Option<&[bool]>,
        rope: Option<(&[f32], &[f32])>,
        cond_rope: Option<(&[f32], &[f32])>,
    ) -> Vec<f32> {
        let mut q = self.to_q.forward(x, seq);
        let mut k = self.to_k.forward(cond, cond_seq);
        let v = self.to_v.forward(cond, cond_seq);
        if let Some(norm) = &self.q_norm {
            norm.apply(&mut q, seq);
        }
        if let Some(norm) = &self.k_norm {
            norm.apply(&mut k, cond_seq);
        }
        let mut q_heads = ops::split_heads(&q, seq, self.heads, self.head_dim);
        let mut k_heads = ops::split_heads(&k, cond_seq, self.heads, self.head_dim);
        let v_heads = ops::split_heads(&v, cond_seq, self.heads, self.head_dim);
        if let Some((cos, sin)) = rope {
            ops::rope_neox(&mut q_heads, self.heads, seq, self.head_dim, cos, sin);
        }
        if let Some((cos, sin)) = cond_rope {
            ops::rope_neox(&mut k_heads, self.heads, cond_seq, self.head_dim, cos, sin);
        }

        // The additive key mask: denied keys attend as -inf.
        let additive: Option<Vec<f32>> = cond_mask.map(|m| {
            let mut out = vec![0.0f32; seq * cond_seq];
            for q in 0..seq {
                for (k, keep) in m.iter().take(cond_seq).enumerate() {
                    out[q * cond_seq + k] = if *keep { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            out
        });

        let mut out = vec![0.0f32; self.heads * seq * self.head_dim];
        for h in 0..self.heads {
            let q_head = &q_heads[h * seq * self.head_dim..(h + 1) * seq * self.head_dim];
            let k_head = &k_heads[h * cond_seq * self.head_dim..(h + 1) * cond_seq * self.head_dim];
            let v_head = &v_heads[h * cond_seq * self.head_dim..(h + 1) * cond_seq * self.head_dim];
            let attended = ops::sdpa(
                q_head,
                k_head,
                v_head,
                additive.as_deref(),
                seq,
                cond_seq,
                self.head_dim,
                self.head_dim,
                1.0 / (self.head_dim as f32).sqrt(),
            );
            out[h * seq * self.head_dim..(h + 1) * seq * self.head_dim].copy_from_slice(&attended);
        }
        let merged = ops::merge_heads(&out, seq, self.heads, self.head_dim);
        self.to_out.forward(&merged, seq)
    }
}

/// The gated feed forward (`gelu_approx` is the tanh approximation).
struct FeedForward {
    ff0: Linear,
    ff1: Linear,
}

impl FeedForward {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        inner: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            ff0: linear(file, &format!("{prefix}.ff.0"), dim, inner, bias)?,
            ff1: linear(file, &format!("{prefix}.ff.1"), inner, dim, bias)?,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let mut h = self.ff0.forward(x, seq);
        ops::gelu_tanh(&mut h);
        self.ff1.forward(&h, seq)
    }
}

/// One DiT block: global AdaLN adds the block's scale/shift residual,
/// local AdaLN replaces the conditioning with its own MLP over `t`.
struct DiTBlock {
    adaln_scale_shift: Vec<f32>,
    adaln_mlp: Option<Linear>,
    self_attn: Attention,
    cross_attn: Attention,
    ffn: FeedForward,
    eps: f32,
    dim: usize,
}

impl DiTBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Vec<f32>,
        seq: usize,
        cond: &[f32],
        cond_seq: usize,
        mask: Option<&[bool]>,
        cond_mask: Option<&[bool]>,
        rope: Option<(&[f32], &[f32])>,
        cond_rope: Option<(&[f32], &[f32])>,
        adaln_global_out: Option<&[f32]>,
        t: &[f32],
    ) -> Vec<f32> {
        let dim = self.dim;
        // The six-way split order: gate_sa, scale_sa, shift_sa,
        // gate_ffn, scale_ffn, shift_ffn.
        let mut adaln = match (adaln_global_out, &self.adaln_mlp) {
            (Some(global), _) => {
                let mut a = global.to_vec();
                for (v, s) in a.iter_mut().zip(&self.adaln_scale_shift) {
                    *v += s;
                }
                a
            }
            (None, Some(mlp)) => {
                // AdaLNMLP: silu then the linear.
                let mut cond = t.to_vec();
                ops::silu(&mut cond);
                mlp.forward(&cond, 1)
            }
            (None, None) => unreachable!("a block needs an AdaLN source"),
        };
        let (gate_sa, rest) = adaln.split_at_mut(dim);
        let (scale_sa, rest) = rest.split_at_mut(dim);
        let (shift_sa, rest) = rest.split_at_mut(dim);
        let (gate_ffn, rest) = rest.split_at_mut(dim);
        let (scale_ffn, shift_ffn) = rest.split_at_mut(dim);

        // Self attention with the modulated norm.
        let mut normed = x.clone();
        modulate(&mut normed, dim, self.eps, scale_sa, shift_sa);
        let attn = self
            .self_attn
            .forward(&normed, seq, &normed, seq, mask, rope, rope);
        // gate_sa broadcasts over the time axis (it is one [dim] row).
        let mut next = x;
        for (r, row) in next.chunks_exact_mut(dim).enumerate() {
            for (c, dst) in row.iter_mut().enumerate() {
                *dst += gate_sa[c] * attn[r * dim + c];
            }
        }

        // Cross attention to the text condition (residual, no gate).
        let cross = self
            .cross_attn
            .forward(&next, seq, cond, cond_seq, cond_mask, rope, cond_rope);
        for (dst, c) in next.iter_mut().zip(&cross) {
            *dst += c;
        }

        // Feed forward with the modulated norm.
        let mut normed = next.clone();
        modulate(&mut normed, dim, self.eps, scale_ffn, shift_ffn);
        let feed = self.ffn.forward(&normed, seq);
        for (r, row) in next.chunks_exact_mut(dim).enumerate() {
            for (c, dst) in row.iter_mut().enumerate() {
                *dst += gate_ffn[c] * feed[r * dim + c];
            }
        }
        next
    }
}

/// The `AudioDiTTransformer`.
pub(crate) struct AudioDiTTransformer {
    time_embed: TimestepEmbedding,
    input_embed: Embedder,
    text_embed: Embedder,
    rotary_head_dim: usize,
    rotary_base: f32,
    blocks: Vec<DiTBlock>,
    norm_out_linear: Linear,
    proj_out: Linear,
    adaln_global_mlp: Option<Linear>,
    local_adaln: bool,
    text_conv: Vec<ConvNextV2Block>,
    latent_embed: Option<Embedder>,
    latent_cond_embedder: Option<Embedder>,
    long_skip: bool,
    eps: f32,
    dim: usize,
}

impl AudioDiTTransformer {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &super::ModelConfig,
    ) -> Result<Self> {
        let dim = config.dit_dim;
        let head_dim = dim / config.dit_heads;
        let inner = (dim as f64 * config.dit_ff_mult) as usize;
        let bias = config.dit_bias;
        let eps = config.dit_eps as f32;
        let local_adaln = config.dit_adaln_type == "local";
        let mut text_conv = Vec::new();
        if config.dit_text_conv {
            for i in 0..4 {
                text_conv.push(ConvNextV2Block::load(
                    file,
                    &format!("{prefix}.text_conv_layer.{i}"),
                    dim,
                    dim * 2,
                    eps,
                )?);
            }
        }
        let mut blocks = Vec::with_capacity(config.dit_depth);
        for i in 0..config.dit_depth {
            let block_prefix = &format!("{prefix}.blocks.{i}");
            let (adaln_scale_shift, adaln_mlp) = if local_adaln {
                (
                    Vec::new(),
                    Some(linear(
                        file,
                        &format!("{block_prefix}.adaln_mlp.mlp.0"),
                        dim,
                        dim * 6,
                        true,
                    )?),
                )
            } else {
                (
                    tensor(
                        file,
                        &format!("{block_prefix}.adaln_scale_shift"),
                        &[dim * 6],
                    )?,
                    None,
                )
            };
            blocks.push(DiTBlock {
                adaln_scale_shift,
                adaln_mlp,
                self_attn: Attention::load(
                    file,
                    &format!("{block_prefix}.self_attn"),
                    dim,
                    dim,
                    config.dit_heads,
                    head_dim,
                    config.dit_qk_norm,
                    eps,
                    bias,
                )?,
                cross_attn: Attention::load(
                    file,
                    &format!("{block_prefix}.cross_attn"),
                    dim,
                    dim,
                    config.dit_heads,
                    head_dim,
                    config.dit_qk_norm,
                    eps,
                    bias,
                )?,
                ffn: FeedForward::load(file, &format!("{block_prefix}.ffn"), dim, inner, bias)?,
                eps,
                dim,
            });
        }
        Ok(Self {
            time_embed: TimestepEmbedding::load(file, &format!("{prefix}.time_embed"), dim)?,
            input_embed: Embedder::load(
                file,
                &format!("{prefix}.input_embed"),
                config.latent_dim,
                dim,
            )?,
            text_embed: Embedder::load(
                file,
                &format!("{prefix}.text_embed"),
                config.dit_text_dim,
                dim,
            )?,
            rotary_head_dim: head_dim,
            rotary_base: 100000.0,
            blocks,
            norm_out_linear: linear(
                file,
                &format!("{prefix}.norm_out.linear"),
                dim,
                dim * 2,
                true,
            )?,
            proj_out: linear(
                file,
                &format!("{prefix}.proj_out"),
                dim,
                config.latent_dim,
                true,
            )?,
            adaln_global_mlp: if local_adaln {
                None
            } else {
                Some(linear(
                    file,
                    &format!("{prefix}.adaln_global_mlp.mlp.0"),
                    dim,
                    dim * 6,
                    true,
                )?)
            },
            local_adaln,
            text_conv,
            latent_embed: if config.dit_use_latent_condition {
                Some(Embedder::load(
                    file,
                    &format!("{prefix}.latent_embed"),
                    config.latent_dim,
                    dim,
                )?)
            } else {
                None
            },
            latent_cond_embedder: if config.dit_use_latent_condition {
                Some(Embedder::load(
                    file,
                    &format!("{prefix}.latent_cond_embedder"),
                    dim * 2,
                    dim,
                )?)
            } else {
                None
            },
            long_skip: config.dit_long_skip,
            eps,
            dim,
        })
    }

    /// The transformer forward (`last_hidden_state` only; the repa
    /// hidden state the reference records is never read by generate,
    /// but the mid-stack long-skip add it triggers is reproduced).
    /// `mask`/`cond_mask` are per-position keep flags.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        seq: usize,
        text: &[f32],
        cond_seq: usize,
        time: f32,
        mask: Option<&[bool]>,
        cond_mask: Option<&[bool]>,
        repa_layer: Option<usize>,
        latent_cond: &[f32],
    ) -> Vec<f32> {
        let dim = self.dim;
        let t = self.time_embed.forward(time);
        let mut text_emb = self.text_embed.forward(text, cond_seq, cond_mask);
        for block in &self.text_conv {
            block.forward(&mut text_emb, cond_seq).expect("text conv");
        }
        if let Some(cm) = cond_mask {
            for (r, keep) in cm.iter().enumerate().take(cond_seq) {
                if !keep {
                    for v in &mut text_emb[r * dim..(r + 1) * dim] {
                        *v = 0.0;
                    }
                }
            }
        }
        let mut h = self.input_embed.forward(x, seq, mask);
        if let (Some(latent_embed), Some(cond_embedder)) =
            (&self.latent_embed, &self.latent_cond_embedder)
        {
            let latent = latent_embed.forward(latent_cond, seq, mask);
            let mut concat = vec![0.0f32; seq * dim * 2];
            for r in 0..seq {
                concat[r * dim * 2..r * dim * 2 + dim].copy_from_slice(&h[r * dim..(r + 1) * dim]);
                concat[r * dim * 2 + dim..(r + 1) * dim * 2]
                    .copy_from_slice(&latent[r * dim..(r + 1) * dim]);
            }
            h = cond_embedder.forward(&concat, seq, mask);
        }

        let x_clone = if self.long_skip {
            Some(h.clone())
        } else {
            None
        };

        let (cos, sin) = ops::rope_tables(seq, self.rotary_head_dim, self.rotary_base);
        let (cond_cos, cond_sin) =
            ops::rope_tables(cond_seq, self.rotary_head_dim, self.rotary_base);

        // The AdaLN condition: t + masked text mean (global) or just t
        // (local; the block MLPs it).
        let text_mean = {
            let mut mean = vec![0.0f32; dim];
            let count = match cond_mask {
                Some(cm) => cm.iter().take(cond_seq).filter(|k| **k).count(),
                None => cond_seq,
            };
            match cond_mask {
                Some(cm) => {
                    for (r, keep) in cm.iter().enumerate().take(cond_seq) {
                        if *keep {
                            for (m, v) in mean.iter_mut().zip(&text_emb[r * dim..(r + 1) * dim]) {
                                *m += v;
                            }
                        }
                    }
                }
                None => {
                    for r in 0..cond_seq {
                        for (m, v) in mean.iter_mut().zip(&text_emb[r * dim..(r + 1) * dim]) {
                            *m += v;
                        }
                    }
                }
            }
            for m in &mut mean {
                *m /= count.max(1) as f32;
            }
            mean
        };
        let mut norm_cond = t.clone();
        if !self.local_adaln {
            for (n, m) in norm_cond.iter_mut().zip(&text_mean) {
                *n += m;
            }
        }
        let adaln_global_out = self.adaln_global_mlp.as_ref().map(|mlp| {
            // AdaLNMLP: silu then the linear.
            let mut cond = norm_cond.clone();
            ops::silu(&mut cond);
            mlp.forward(&cond, 1)
        });

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(
                h,
                seq,
                &text_emb,
                cond_seq,
                mask,
                cond_mask,
                Some((&cos, &sin)),
                Some((&cond_cos, &cond_sin)),
                adaln_global_out.as_deref(),
                &t,
            );
            if repa_layer == Some(i + 1) && self.long_skip {
                if let Some(clone) = &x_clone {
                    for (dst, src) in h.iter_mut().zip(clone) {
                        *dst += src;
                    }
                }
            }
        }
        if let Some(clone) = &x_clone {
            for (dst, src) in h.iter_mut().zip(clone) {
                *dst += src;
            }
        }

        // AdaLayerNormZeroFinal: SiLU, linear, split, modulate.
        let mut cond_for_out = norm_cond.clone();
        ops::silu(&mut cond_for_out);
        let emb = self.norm_out_linear.forward(&cond_for_out, 1);
        let (scale, shift) = emb.split_at(dim);
        let mut normed = h;
        layer_norm_in_place(&mut normed, dim, self.eps);
        for (r, row) in normed.chunks_exact_mut(dim).enumerate() {
            let _ = r;
            for (c, v) in row.iter_mut().enumerate() {
                *v = *v * (1.0 + scale[c]) + shift[c];
            }
        }
        self.proj_out.forward(&normed, seq)
    }
}
