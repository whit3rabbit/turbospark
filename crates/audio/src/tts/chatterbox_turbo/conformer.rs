//! The chatterbox_turbo S3Gen conformer encoder.
//!
//! Reference: `chatterbox_turbo/models/s3gen/encoder.py`
//! (`EspnetRelPositionalEncoding`, `LinearInput`,
//! `RelPositionMultiHeadedAttention`, `PositionwiseFeedForward`,
//! `ConformerEncoderLayer`, `PreLookaheadLayer`, `Upsample1DEncoder`,
//! `UpsampleConformerEncoder`) at the pinned commit recorded in the
//! parent module. The file consolidates the same modules the sibling
//! chatterbox family ports from `transformer/`; the math is identical
//! (this fork exists because the two conversions disagree on key
//! naming: turbo keeps the MLX module-list indices dotted,
//! `encoders.0.`, where the sibling checkpoint renames to
//! `encoders_0.`, and the `Conv1dPT` wrappers put every conv one
//! `.conv` deeper).
//!
//! Single batch, exact lengths: the reference threads padding masks
//! through every stage, but a batch-of-one forward with
//! `token_len == sequence length` makes every mask all-valid, so the
//! attention mask `mx.where(mask > 0, scores, -inf)` and the mask
//! multiplies reduce to identity and are not threaded (the encoder
//! never builds a chunk mask at all).

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

use crate::tts::chatterbox::s3gen::{leaky_relu, SaConv1d, SaLayerNorm, SaLinear};

/// Initial size the reference positional encodings are built for
/// (`max_len = 5000`); longer inputs regenerate the table.
const PE_MAX_LEN: usize = 5000;

/// Espnet relative positional encoding
/// (`encoder.py::EspnetRelPositionalEncoding`). The table holds
/// positions `L-1 .. 0, -1 .. -(L-1)` with interleaved `sin`/`cos`;
/// `position_encoding(T)` slices the centered `2T-1` rows. The table is
/// derived pure math at init (real checkpoints store it; this port
/// recomputes it).
#[derive(Debug, Clone)]
struct EspnetPosEnc {
    d_model: usize,
    /// Row-major `[2L-1, d_model]`, `L >= PE_MAX_LEN`.
    pe: Vec<f32>,
    pe_len: usize,
}

impl EspnetPosEnc {
    fn new(d_model: usize) -> Self {
        let mut enc = EspnetPosEnc {
            d_model,
            pe: Vec::new(),
            pe_len: 0,
        };
        enc.extend(PE_MAX_LEN);
        enc
    }

    fn extend(&mut self, size: usize) {
        let l = size.max(PE_MAX_LEN);
        if self.pe_len >= 2 * l - 1 {
            return;
        }
        let half = self.d_model / 2;
        let div: Vec<f32> = (0..half)
            .map(|i| (-(10_000.0f32.ln() / self.d_model as f32) * (2 * i) as f32).exp())
            .collect();
        let mut pe = vec![0.0f32; (2 * l - 1) * self.d_model];
        for m in 0..2 * l - 1 {
            let pos: f32 = if m < l {
                (l - 1 - m) as f32
            } else {
                -((m - (l - 1)) as f32)
            };
            for (i, &d) in div.iter().enumerate() {
                let arg = pos * d;
                pe[m * self.d_model + 2 * i] = arg.sin();
                pe[m * self.d_model + 2 * i + 1] = arg.cos();
            }
        }
        self.pe = pe;
        self.pe_len = 2 * l - 1;
    }

    /// Returns `x` scaled by `sqrt(d_model)` and the `[2T-1, d_model]`
    /// positional slice.
    fn forward(&mut self, x: &mut [f32], t: usize) -> Vec<f32> {
        self.extend(t);
        let xscale = (self.d_model as f32).sqrt();
        for v in x.iter_mut() {
            *v *= xscale;
        }
        let center = self.pe_len / 2;
        let start = center - (t - 1);
        self.pe[start * self.d_model..(start + 2 * t - 1) * self.d_model].to_vec()
    }
}

/// Rel-position multi-head attention
/// (`encoder.py::RelPositionMultiHeadedAttention`).
#[derive(Debug, Clone)]
struct RelPosAttention {
    n_head: usize,
    d_model: usize,
    linear_q: SaLinear,
    linear_k: SaLinear,
    linear_v: SaLinear,
    linear_out: SaLinear,
    linear_pos: SaLinear,
    pos_bias_u: Vec<f32>,
    pos_bias_v: Vec<f32>,
}

impl RelPosAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        d_model: usize,
        n_head: usize,
        key_bias: bool,
    ) -> Result<Self> {
        let linear =
            |name: &str| SaLinear::load(file, &format!("{prefix}.{name}"), d_model, d_model);
        let pos = SaLinear::load_no_bias(file, &format!("{prefix}.linear_pos"), d_model, d_model)?;
        let k = linear("linear_k")?;
        if key_bias && k.bias.is_none() {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.linear_k.bias"),
                why: "required by the reference layer (key_bias=True)".to_string(),
            });
        }
        let head_dim = d_model / n_head;
        Ok(RelPosAttention {
            n_head,
            d_model,
            linear_q: linear("linear_q")?,
            linear_k: k,
            linear_v: linear("linear_v")?,
            linear_out: linear("linear_out")?,
            linear_pos: pos,
            pos_bias_u: load_f32_shaped(
                file,
                &format!("{prefix}.pos_bias_u"),
                &[n_head, head_dim],
            )?,
            pos_bias_v: load_f32_shaped(
                file,
                &format!("{prefix}.pos_bias_v"),
                &[n_head, head_dim],
            )?,
        })
    }

    /// `x` and `pos_emb` are row-major `[T, D]` and `[2T-1, D]`. The
    /// reference `_rel_shift` pad/reshape/slice reduces to table index
    /// `(T - 1) + j - i` for query `i`, key `j`. The all-valid padding
    /// mask reduces to identity; the NaN guard after the softmax is the
    /// reference's `mx.where(isnan(attn), 0, attn)` and never fires on
    /// all-valid rows.
    fn forward(&self, x: &[f32], pos_emb: &[f32], t: usize) -> Vec<f32> {
        let d_head = self.d_model / self.n_head;
        let q = self.linear_q.forward(x, t);
        let k = self.linear_k.forward(x, t);
        let v = self.linear_v.forward(x, t);
        let p = self.linear_pos.forward(pos_emb, 2 * t - 1);
        let scale = (d_head as f32).sqrt();
        let mut out = vec![0.0f32; t * self.d_model];
        let mut scores = vec![0.0f32; t];
        for h in 0..self.n_head {
            let off = h * d_head;
            for i in 0..t {
                for j in 0..t {
                    let mut ac = 0.0f32;
                    let mut bd = 0.0f32;
                    for d in 0..d_head {
                        let qv = q[i * self.d_model + off + d];
                        ac +=
                            (qv + self.pos_bias_u[h * d_head + d]) * k[j * self.d_model + off + d];
                        let prow = (t - 1 + j - i) * self.d_model + off + d;
                        bd += (qv + self.pos_bias_v[h * d_head + d]) * p[prow];
                    }
                    scores[j] = (ac + bd) / scale;
                }
                ops::softmax_row(&mut scores);
                for j in 0..t {
                    let s = scores[j];
                    for d in 0..d_head {
                        out[i * self.d_model + off + d] += s * v[j * self.d_model + off + d];
                    }
                }
            }
        }
        self.linear_out.forward(&out, t)
    }
}

/// Conformer encoder layer: attention and feed-forward only (the turbo
/// constructor never builds a macaron feed-forward or a conv module),
/// pre-norm with 1e-12 LayerNorms, SiLU feed-forward.
#[derive(Debug, Clone)]
struct ConformerLayer {
    attn: RelPosAttention,
    norm_mha: SaLayerNorm,
    norm_ff: SaLayerNorm,
    w_1: SaLinear,
    w_2: SaLinear,
}

impl ConformerLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        d_model: usize,
        n_head: usize,
        linear_units: usize,
        key_bias: bool,
    ) -> Result<Self> {
        Ok(ConformerLayer {
            attn: RelPosAttention::load(
                file,
                &format!("{prefix}.self_attn"),
                d_model,
                n_head,
                key_bias,
            )?,
            norm_mha: SaLayerNorm::load(file, &format!("{prefix}.norm_mha"), d_model)?,
            norm_ff: SaLayerNorm::load(file, &format!("{prefix}.norm_ff"), d_model)?,
            w_1: SaLinear::load(
                file,
                &format!("{prefix}.feed_forward.w_1"),
                d_model,
                linear_units,
            )?,
            w_2: SaLinear::load(
                file,
                &format!("{prefix}.feed_forward.w_2"),
                linear_units,
                d_model,
            )?,
        })
    }

    fn forward(&self, x: &mut [f32], pos_emb: &[f32], t: usize) {
        let residual = x.to_vec();
        self.norm_mha.forward(x, t, 1e-12);
        let attended = self.attn.forward(x, pos_emb, t);
        for (v, (r, a)) in x.iter_mut().zip(residual.iter().zip(&attended)) {
            *v = r + a;
        }
        let residual = x.to_vec();
        self.norm_ff.forward(x, t, 1e-12);
        let mut hidden = self.w_1.forward(x, t);
        ops::silu(&mut hidden);
        let projected = self.w_2.forward(&hidden, t);
        for (v, (r, p)) in x.iter_mut().zip(residual.iter().zip(&projected)) {
            *v = r + p;
        }
    }
}

/// `LinearInput`: linear -> LayerNorm(1e-5) -> espnet rel-pos scaling.
/// Owns its positional encoding.
#[derive(Debug, Clone)]
struct LinearInput {
    linear: SaLinear,
    norm: SaLayerNorm,
    pos_enc: EspnetPosEnc,
}

impl LinearInput {
    fn load(file: &SafetensorsFile, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(LinearInput {
            linear: SaLinear::load(file, &format!("{prefix}.linear"), in_dim, out_dim)?,
            norm: SaLayerNorm::load(file, &format!("{prefix}.norm"), out_dim)?,
            pos_enc: EspnetPosEnc::new(out_dim),
        })
    }

    fn forward(&mut self, x: &[f32], t: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut h = self.linear.forward(x, t);
        self.norm.forward(&mut h, t, 1e-5);
        let pos_emb = self.pos_enc.forward(&mut h, t);
        Ok((h, pos_emb))
    }
}

/// Right/left zero pad along the sequence axis of channel-major
/// `[ch, seq]`.
fn pad_right_cm(x: &[f32], ch: usize, seq: usize, pad: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ch * (seq + pad)];
    for c in 0..ch {
        out[c * (seq + pad)..c * (seq + pad) + seq].copy_from_slice(&x[c * seq..(c + 1) * seq]);
    }
    out
}

fn pad_left_cm(x: &[f32], ch: usize, seq: usize, pad: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ch * (seq + pad)];
    for c in 0..ch {
        out[c * (seq + pad) + pad..(c + 1) * (seq + pad)]
            .copy_from_slice(&x[c * seq..(c + 1) * seq]);
    }
    out
}

/// `PreLookaheadLayer`: right-pad `pre_lookahead_len` frames, conv
/// (kernel len+1), leaky ReLU (the `nn.leaky_relu` default slope 0.01),
/// left-pad 2, conv (kernel 3), residual. Row-major `[T, C]` in and out.
#[derive(Debug, Clone)]
struct PreLookaheadLayer {
    channels: usize,
    pre_lookahead_len: usize,
    conv1: SaConv1d,
    conv2: SaConv1d,
}

impl PreLookaheadLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        channels: usize,
        lookahead: usize,
    ) -> Result<Self> {
        Ok(PreLookaheadLayer {
            channels,
            pre_lookahead_len: lookahead,
            conv1: SaConv1d::load(
                file,
                &format!("{prefix}.conv1"),
                channels,
                channels,
                lookahead + 1,
                1,
                0,
                1,
                1,
            )?,
            conv2: SaConv1d::load(
                file,
                &format!("{prefix}.conv2"),
                channels,
                channels,
                3,
                1,
                0,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let cm = crate::tts::chatterbox::s3gen::to_cm(x, t, self.channels);
        let cm = pad_right_cm(&cm, self.channels, t, self.pre_lookahead_len);
        let mut h = self.conv1.forward(&cm);
        leaky_relu(&mut h, 0.01);
        let h = pad_left_cm(&h, self.channels, t, 2);
        let h = self.conv2.forward(&h);
        let mut out = crate::tts::chatterbox::s3gen::to_rows(&h, self.channels, t);
        for (v, r) in out.iter_mut().zip(x) {
            *v += r;
        }
        out
    }
}

/// The encoder `Upsample1DEncoder`: repeat each frame `stride` times,
/// left-pad `2 * stride` zeros, conv (kernel `2 * stride + 1`).
#[derive(Debug, Clone)]
struct Upsample1D {
    channels: usize,
    stride: usize,
    conv: SaConv1d,
}

impl Upsample1D {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize, stride: usize) -> Result<Self> {
        Ok(Upsample1D {
            channels,
            stride,
            conv: SaConv1d::load(
                file,
                &format!("{prefix}.conv"),
                channels,
                channels,
                stride * 2 + 1,
                1,
                0,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let cm = crate::tts::chatterbox::s3gen::to_cm(x, t, self.channels);
        let up = self.stride;
        let mut repeated = vec![0.0f32; self.channels * t * up];
        for c in 0..self.channels {
            for f in 0..t {
                for r in 0..up {
                    repeated[c * t * up + f * up + r] = cm[c * t + f];
                }
            }
        }
        let padded = pad_left_cm(&repeated, self.channels, t * up, up * 2);
        let conv = self.conv.forward(&padded);
        crate::tts::chatterbox::s3gen::to_rows(&conv, self.channels, t * up)
    }
}

/// `UpsampleConformerEncoder`: embed + pre-lookahead + conformer stack +
/// upsample + second embed + conformer stack + final norm. Row-major
/// `[T, input_size]` -> `[T * stride, output_size]`.
#[derive(Debug, Clone)]
pub(crate) struct UpsampleConformerEncoder {
    embed: LinearInput,
    pre_lookahead_layer: PreLookaheadLayer,
    encoders: Vec<ConformerLayer>,
    up_layer: Upsample1D,
    up_embed: LinearInput,
    up_encoders: Vec<ConformerLayer>,
    after_norm: SaLayerNorm,
    stride: usize,
}

impl UpsampleConformerEncoder {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input_size: usize,
        output_size: usize,
        n_head: usize,
        linear_units: usize,
        num_blocks: usize,
        up_stride: usize,
        pre_lookahead_len: usize,
    ) -> Result<Self> {
        // The turbo checkpoint keeps the block lists dotted
        // (`encoders.0.`): the index is part of the name.
        let layer = |i: usize, name: &str| {
            ConformerLayer::load(
                file,
                &format!("{prefix}.{name}.{i}"),
                output_size,
                n_head,
                linear_units,
                true,
            )
        };
        Ok(UpsampleConformerEncoder {
            embed: LinearInput::load(file, &format!("{prefix}.embed"), input_size, output_size)?,
            pre_lookahead_layer: PreLookaheadLayer::load(
                file,
                &format!("{prefix}.pre_lookahead_layer"),
                output_size,
                pre_lookahead_len,
            )?,
            encoders: (0..num_blocks)
                .map(|i| layer(i, "encoders"))
                .collect::<Result<_>>()?,
            up_layer: Upsample1D::load(
                file,
                &format!("{prefix}.up_layer"),
                output_size,
                up_stride,
            )?,
            up_embed: LinearInput::load(
                file,
                &format!("{prefix}.up_embed"),
                input_size,
                output_size,
            )?,
            // The turbo constructor hardcodes the post-upsample stack
            // at four layers (`for _ in range(4)`); unlike the sibling
            // family there is no `num_up_blocks` knob.
            up_encoders: (0..4)
                .map(|i| layer(i, "up_encoders"))
                .collect::<Result<_>>()?,
            after_norm: SaLayerNorm::load(file, &format!("{prefix}.after_norm"), output_size)?,
            stride: up_stride,
        })
    }

    pub(crate) fn up_rate(&self) -> usize {
        self.stride
    }

    pub(crate) fn forward(&mut self, xs: &[f32], t: usize) -> Result<Vec<f32>> {
        let (x, pos_emb) = self.embed.forward(xs, t)?;
        let mut x = self.pre_lookahead_layer.forward(&x, t);
        for layer in &self.encoders {
            layer.forward(&mut x, &pos_emb, t);
        }
        let up = self.up_layer.forward(&x, t);
        let up_t = t * self.stride;
        let (mut x, pos_emb) = self.up_embed.forward(&up, up_t)?;
        for layer in &self.up_encoders {
            layer.forward(&mut x, &pos_emb, up_t);
        }
        self.after_norm.forward(&mut x, up_t, 1e-5);
        Ok(x)
    }
}
