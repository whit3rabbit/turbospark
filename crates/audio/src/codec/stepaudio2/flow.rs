//! The StepAudio2 flow model: tokens + speaker prompt -> mel.
//!
//! Reference: `mlx_audio/codec/models/stepaudio2/` (flow.py,
//! upsample_encoder_v2.py, decoder_dit.py, flow_matching.py) and the
//! chatterbox s3gen transformer modules the encoder imports, all at the
//! pinned commit recorded in the parent module.
//!
//! Structure: tokens (6561-vocab, clipped) embed into a linear-input
//! conformer encoder with an espnet relative-position attention stack
//! and a pre-lookahead conv, upsample by `up_stride` through a second
//! conformer stack, project to 80 mel dims, and enter a conditional
//! flow-matching DiT. The DiT concatenates x, mu, the speaker embedding,
//! and the prompt-conditioning mel into 320 channels and runs
//! adaLN-zero blocks (attention, causal conv, MLP in that order) with
//! qk-norm and a cos/sin timestep embedding. The CFM solver takes 10
//! Euler steps over the cosine-spaced schedule `1 - cos(t*pi/2)` with
//! classifier-free guidance batch doubling (`rand_noise` is a
//! checkpoint tensor, so the solve is deterministic).
//!
//! The espnet relative-attention `rel_shift` is applied here in closed
//! form: the literal pad/reshape/slice dance produces, for query `i` and
//! key `j`, the dot product of `q + pos_bias_v` with the positional
//! projection at relative position `i - j`, whose table row is
//! `(T - 1) - (i - j)`. The test fixture pins this against the
//! reference.

use turbospark_model_io::safetensors::SafetensorsFile;

use super::{leaky_relu, mish, to_cm, to_rows, SaConv1d, SaLayerNorm, SaLinear, StepAudio2Config};
use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// Initial size the reference positional encodings are built for
/// (`max_len = 5000`); longer inputs regenerate the table.
const PE_MAX_LEN: usize = 5000;

/// Espnet relative positional encoding
/// (`s3gen/transformer/embedding.py::EspnetRelPositionalEncoding`).
/// The table holds positions `L-1 .. 0, -1 .. -(L-1)` with interleaved
/// `sin`/`cos`; `position_encoding(T)` slices the centered `2T-1` rows
/// (row `k` of the slice is relative position `T-1-k`).
#[derive(Debug, Clone)]
struct EspnetPosEnc {
    d_model: usize,
    /// Row-major `[2L-1, d_model]`, `L >= PE_MAX_LEN`.
    pe: Vec<f32>,
    /// `L` of the current table.
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
        // div_term[i] = exp(-(ln(10000) / d_model) * 2i), i over D/2.
        let half = self.d_model / 2;
        let div: Vec<f32> = (0..half)
            .map(|i| (-(10.0f32.ln() / self.d_model as f32) * (2 * i) as f32).exp())
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
    /// positional slice (the `RelPositionalEncoding` call contract).
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
/// (`s3gen/transformer/attention.py::RelPositionMultiHeadedAttention`).
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
        let u = load_f32_shaped(
            file,
            &format!("{prefix}.pos_bias_u"),
            &[n_head, d_model / n_head],
        )?;
        let v = load_f32_shaped(
            file,
            &format!("{prefix}.pos_bias_v"),
            &[n_head, d_model / n_head],
        )?;
        Ok(RelPosAttention {
            n_head,
            d_model,
            linear_q: linear("linear_q")?,
            linear_k: k,
            linear_v: linear("linear_v")?,
            linear_out: linear("linear_out")?,
            linear_pos: pos,
            pos_bias_u: u,
            pos_bias_v: v,
        })
    }

    /// `x` and `pos_emb` are row-major `[T, D]` and `[2T-1, D]`.
    fn forward(&self, x: &[f32], pos_emb: &[f32], t: usize) -> Vec<f32> {
        let d_head = self.d_model / self.n_head;
        let q = self.linear_q.forward(x, t);
        let k = self.linear_k.forward(x, t);
        let v = self.linear_v.forward(x, t);
        let p = self.linear_pos.forward(pos_emb, 2 * t - 1);
        let scale = (d_head as f32).sqrt();
        let mut out = vec![0.0f32; t * self.d_model];
        let mut scores = vec![0.0f32; t * t];
        for h in 0..self.n_head {
            let q_off = h * d_head;
            let k_off = h * d_head;
            let p_off = h * d_head;
            for i in 0..t {
                // Content term with pos_bias_u and position term with
                // pos_bias_v, the latter indexed at relative position
                // i - j (the closed form of rel_shift).
                for j in 0..t {
                    let mut ac = 0.0f32;
                    let mut bd = 0.0f32;
                    for d in 0..d_head {
                        let qv = q[i * self.d_model + q_off + d];
                        ac += (qv + self.pos_bias_u[h * d_head + d])
                            * k[j * self.d_model + k_off + d];
                        let prow = (t - 1 + i - j) * self.d_model + p_off + d;
                        bd += (qv + self.pos_bias_v[h * d_head + d]) * p[prow];
                    }
                    scores[i * t + j] = (ac + bd) / scale;
                }
            }
            for i in 0..t {
                ops::softmax_row(&mut scores[i * t..(i + 1) * t]);
                for j in 0..t {
                    let s = scores[i * t + j];
                    for d in 0..d_head {
                        out[i * self.d_model + q_off + d] += s * v[j * self.d_model + k_off + d];
                    }
                }
            }
        }
        self.linear_out.forward(&out, t)
    }
}

/// Conformer encoder layer as the s3gen stack builds it here: attention
/// and feed-forward only (no macaron, no conv module), pre-norm with
/// 1e-12 LayerNorms.
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

    fn forward(&self, x: &mut Vec<f32>, pos_emb: &[f32], t: usize) {
        let residual = x.clone();
        self.norm_mha.forward(x, t, 1e-12);
        let attended = self.attn.forward(x, pos_emb, t);
        for (v, (r, a)) in x.iter_mut().zip(residual.iter().zip(&attended)) {
            *v = r + a;
        }
        let residual = x.clone();
        self.norm_ff.forward(x, t, 1e-12);
        let mut hidden = self.w_1.forward(x, t);
        ops::silu(&mut hidden);
        let projected = self.w_2.forward(&hidden, t);
        for (v, (r, p)) in x.iter_mut().zip(residual.iter().zip(&projected)) {
            *v = r + p;
        }
    }
}

/// `LinearNoSubsampling`: linear -> LayerNorm(1e-5) -> espnet rel-pos
/// scaling. Owns its positional encoding.
#[derive(Debug, Clone)]
struct LinearNoSub {
    linear: SaLinear,
    norm: SaLayerNorm,
    pos_enc: EspnetPosEnc,
}

impl LinearNoSub {
    fn load(file: &SafetensorsFile, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(LinearNoSub {
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

/// `PreLookaheadLayer`: right-pad `pre_lookahead_len` frames, conv
/// (kernel len+1), leaky ReLU, left-pad 2, conv (kernel 3), residual.
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

    /// Row-major `[T, C]` in and out.
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let cm = to_cm(x, t, self.channels);
        let cm = pad_right_cm(&cm, self.channels, t, self.pre_lookahead_len);
        let mut h = self.conv1.forward(&cm);
        leaky_relu(&mut h, 0.01);
        let h = pad_left_cm(&h, self.channels, t, 2);
        let h = self.conv2.forward(&h);
        let mut out = to_rows(&h, self.channels, t);
        for (v, r) in out.iter_mut().zip(x) {
            *v += r;
        }
        out
    }
}

/// Right/left zero pad along the sequence axis of `[ch, seq]`.
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

/// `Upsample1D` with the reference stride-2 conv: repeat, left-pad
/// `2 * stride` zeros, conv (kernel `2 * stride + 1`).
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
        let cm = to_cm(x, t, self.channels);
        // Repeat each frame `stride` times, then left-pad.
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
        to_rows(&conv, self.channels, t * up)
    }
}

/// `UpsampleConformerEncoderV2`: embed + pre-lookahead + conformer
/// stack + upsample + second embed + conformer stack + final norm.
#[derive(Debug, Clone)]
struct UpsampleConformerEncoder {
    embed: LinearNoSub,
    pre_lookahead_layer: PreLookaheadLayer,
    encoders: Vec<ConformerLayer>,
    up_layer: Upsample1D,
    up_embed: LinearNoSub,
    up_encoders: Vec<ConformerLayer>,
    after_norm: SaLayerNorm,
    normalize_before: bool,
}

impl UpsampleConformerEncoder {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input_size: usize,
        output_size: usize,
        n_head: usize,
        linear_units: usize,
        num_blocks: usize,
        num_up_blocks: usize,
        up_stride: usize,
        pre_lookahead_len: usize,
        key_bias: bool,
    ) -> Result<Self> {
        let layer = |i: usize, name: &str| {
            ConformerLayer::load(
                file,
                &format!("{prefix}.{name}.{i}"),
                output_size,
                n_head,
                linear_units,
                key_bias,
            )
        };
        Ok(UpsampleConformerEncoder {
            embed: LinearNoSub::load(file, &format!("{prefix}.embed"), input_size, output_size)?,
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
            up_embed: LinearNoSub::load(
                file,
                &format!("{prefix}.up_embed"),
                input_size,
                output_size,
            )?,
            up_encoders: (0..num_up_blocks)
                .map(|i| layer(i, "up_encoders"))
                .collect::<Result<_>>()?,
            after_norm: SaLayerNorm::load(file, &format!("{prefix}.after_norm"), output_size)?,
            normalize_before: true,
        })
    }

    /// Row-major `[T, input_size]` -> `[T * stride, output_size]`.
    fn forward(&mut self, xs: &[f32], t: usize) -> Result<Vec<f32>> {
        let (x, pos_emb) = self.embed.forward(xs, t)?;
        let mut x = self.pre_lookahead_layer.forward(&x, t);
        for layer in &self.encoders {
            layer.forward(&mut x, &pos_emb, t);
        }
        let up_t = t * self.up_layer.stride;
        let up = self.up_layer.forward(&x, t);
        let (mut x, pos_emb) = self.up_embed.forward(&up, up_t)?;
        for layer in &self.up_encoders {
            layer.forward(&mut x, &pos_emb, up_t);
        }
        if self.normalize_before {
            self.after_norm.forward(&mut x, up_t, 1e-5);
        }
        Ok(x)
    }
}

/// Tanh-approximated GELU used by the DiT MLP (`decoder_dit.py`).
fn approx_gelu(x: &mut [f32]) {
    ops::gelu_tanh(x);
}

/// The DiT MLP: fc1 -> approx GELU -> fc2.
#[derive(Debug, Clone)]
struct DitMlp {
    fc1: SaLinear,
    fc2: SaLinear,
}

impl DitMlp {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        Ok(DitMlp {
            fc1: SaLinear::load(file, &format!("{prefix}.fc1"), dim, hidden)?,
            fc2: SaLinear::load(file, &format!("{prefix}.fc2"), hidden, dim)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.fc1.forward(x, rows);
        approx_gelu(&mut h);
        self.fc2.forward(&h, rows)
    }
}

/// DiT attention with qk-norm (`decoder_dit.py::Attention`): shared
/// LayerNorm over the head dim, bias-true projections.
#[derive(Debug, Clone)]
struct DitAttention {
    n_head: usize,
    dim: usize,
    head_dim: usize,
    to_q: SaLinear,
    to_k: SaLinear,
    to_v: SaLinear,
    q_norm: SaLayerNorm,
    k_norm: SaLayerNorm,
    proj: SaLinear,
}

impl DitAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let linear =
            |name: &str| SaLinear::load(file, &format!("{prefix}.{name}"), dim, n_head * head_dim);
        Ok(DitAttention {
            n_head,
            dim,
            head_dim,
            to_q: linear("to_q")?,
            to_k: linear("to_k")?,
            to_v: linear("to_v")?,
            q_norm: SaLayerNorm::load(file, &format!("{prefix}.q_norm"), head_dim)?,
            k_norm: SaLayerNorm::load(file, &format!("{prefix}.k_norm"), head_dim)?,
            proj: SaLinear::load(file, &format!("{prefix}.proj"), n_head * head_dim, dim)?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let heads = self.n_head;
        let dh = self.head_dim;
        let q = self.to_q.forward(x, t);
        let k = self.to_k.forward(x, t);
        let v = self.to_v.forward(x, t);
        // qk-norm runs over the head dim, per head and frame.
        let mut qh = q;
        let mut kh = k;
        for i in 0..t {
            self.q_norm
                .forward(&mut qh[i * self.dim..(i + 1) * self.dim], heads, 1e-5);
            self.k_norm
                .forward(&mut kh[i * self.dim..(i + 1) * self.dim], heads, 1e-5);
        }
        let scale = (dh as f32).powf(-0.5);
        let mut out = vec![0.0f32; t * heads * dh];
        let mut scores = vec![0.0f32; t];
        for h in 0..heads {
            for i in 0..t {
                for (j, s) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..dh {
                        acc += qh[i * self.dim + h * dh + d] * kh[j * self.dim + h * dh + d];
                    }
                    *s = acc * scale;
                }
                ops::softmax_row(&mut scores);
                for (j, &s) in scores.iter().enumerate() {
                    for d in 0..dh {
                        out[(i * heads + h) * dh + d] += s * v[j * self.dim + h * dh + d];
                    }
                }
            }
        }
        self.proj.forward(&out, t)
    }
}

/// Causal ConvBlock (`decoder_dit.py::CausalConvBlock`): causal conv ->
/// LayerNorm -> Mish -> causal conv. Operates on row-major `[T, C]`
/// (the LayerNorm is over channels, matching the channels-last MLX
/// reference).
#[derive(Debug, Clone)]
struct CausalConvBlock {
    channels: usize,
    kernel: usize,
    conv1: SaConv1d,
    norm: SaLayerNorm,
    conv2: SaConv1d,
}

impl CausalConvBlock {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Self> {
        Ok(CausalConvBlock {
            channels,
            kernel: 3,
            conv1: SaConv1d::load(
                file,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                3,
                1,
                0,
                1,
                1,
            )?,
            norm: SaLayerNorm::load(file, &format!("{prefix}.block.3"), channels)?,
            conv2: SaConv1d::load(
                file,
                &format!("{prefix}.block.6"),
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
        let mut cm = to_cm(x, t, self.channels);
        let padded = pad_left_cm(&cm, self.channels, t, self.kernel - 1);
        let mut h = self.conv1.forward(&padded);
        let mut rows = to_rows(&h, self.channels, t);
        self.norm.forward(&mut rows, t, 1e-5);
        mish(&mut rows);
        cm = to_cm(&rows, t, self.channels);
        let padded = pad_left_cm(&cm, self.channels, t, self.kernel - 1);
        h = self.conv2.forward(&padded);
        to_rows(&h, self.channels, t)
    }
}

/// DiT adaLN-zero block. The modulation vector comes from the
/// (single-row) timestep conditioning; the 9-way split order is
/// shift/scale/gate for msa, mlp, conv, and the block order is
/// attention, conv, MLP.
#[derive(Debug, Clone)]
struct DitBlock {
    norm1_dim: usize,
    attn: DitAttention,
    mlp: DitMlp,
    conv: CausalConvBlock,
    adaln: SaLinear,
    hidden: usize,
}

impl DitBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        hidden: usize,
        n_head: usize,
        head_dim: usize,
        mlp_hidden: usize,
    ) -> Result<Self> {
        Ok(DitBlock {
            norm1_dim: hidden,
            attn: DitAttention::load(file, &format!("{prefix}.attn"), hidden, n_head, head_dim)?,
            mlp: DitMlp::load(file, &format!("{prefix}.mlp"), hidden, mlp_hidden)?,
            conv: CausalConvBlock::load(file, &format!("{prefix}.conv"), hidden)?,
            adaln: SaLinear::load(
                file,
                &format!("{prefix}.adaLN_modulation.1"),
                hidden,
                9 * hidden,
            )?,
            hidden,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, t_emb: &[f32], t: usize) {
        let h = self.hidden;
        let mut mod_in = t_emb.to_vec();
        ops::silu(&mut mod_in);
        let mod_v = self.adaln.forward(&mod_in, 1);
        let chunk = |i: usize| &mod_v[i * h..(i + 1) * h];
        let (shift_msa, scale_msa, gate_msa) = (chunk(0), chunk(1), chunk(2));
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3), chunk(4), chunk(5));
        let (shift_conv, scale_conv, gate_conv) = (chunk(6), chunk(7), chunk(8));

        // Attention branch.
        let mut normed = x.clone();
        layernorm_noaffine(&mut normed, t, h, 1e-6);
        for (v, (s, sc)) in normed
            .iter_mut()
            .zip(shift_msa.iter().cycle().zip(scale_msa.iter().cycle()))
        {
            *v = *v * (1.0 + sc) + s;
        }
        let attended = self.attn.forward(&normed, t);
        for (v, (a, g)) in x
            .iter_mut()
            .zip(attended.iter().zip(gate_msa.iter().cycle()))
        {
            *v += g * a;
        }

        // Conv branch.
        let mut normed = x.clone();
        layernorm_noaffine(&mut normed, t, h, 1e-6);
        for (v, (s, sc)) in normed
            .iter_mut()
            .zip(shift_conv.iter().cycle().zip(scale_conv.iter().cycle()))
        {
            *v = *v * (1.0 + sc) + s;
        }
        let conv_out = self.conv.forward(&normed, t);
        for (v, (a, g)) in x
            .iter_mut()
            .zip(conv_out.iter().zip(gate_conv.iter().cycle()))
        {
            *v += g * a;
        }

        // MLP branch.
        let mut normed = x.clone();
        layernorm_noaffine(&mut normed, t, h, 1e-6);
        for (v, (s, sc)) in normed
            .iter_mut()
            .zip(shift_mlp.iter().cycle().zip(scale_mlp.iter().cycle()))
        {
            *v = *v * (1.0 + sc) + s;
        }
        let mlp_out = self.mlp.forward(&normed, t);
        for (v, (a, g)) in x
            .iter_mut()
            .zip(mlp_out.iter().zip(gate_mlp.iter().cycle()))
        {
            *v += g * a;
        }
    }
}

/// Affine-free LayerNorm over row-major `[rows, dim]`.
fn layernorm_noaffine(x: &mut [f32], rows: usize, dim: usize, eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv;
        }
    }
}

/// DiT final layer: adaLN (shift/scale), affine-free norm, linear.
#[derive(Debug, Clone)]
struct FinalLayer {
    adaln: SaLinear,
    linear: SaLinear,
    hidden: usize,
    out_channels: usize,
}

impl FinalLayer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        hidden: usize,
        out_channels: usize,
    ) -> Result<Self> {
        Ok(FinalLayer {
            adaln: SaLinear::load(
                file,
                &format!("{prefix}.adaLN_modulation.1"),
                hidden,
                2 * hidden,
            )?,
            linear: SaLinear::load(file, &format!("{prefix}.linear"), hidden, out_channels)?,
            hidden,
            out_channels,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, t_emb: &[f32], t: usize) {
        let h = self.hidden;
        let mut mod_in = t_emb.to_vec();
        ops::silu(&mut mod_in);
        let mod_v = self.adaln.forward(&mod_in, 1);
        let (shift, scale) = mod_v.split_at(h);
        for (v, (s, sc)) in x
            .iter_mut()
            .zip(shift.iter().cycle().zip(scale.iter().cycle()))
        {
            *v = *v * (1.0 + sc) + s;
        }
        layernorm_noaffine(x, t, h, 1e-6);
        *x = self.linear.forward(x, t);
    }
}

/// Timestep embedder: cos/sin features of `t * 1000` through an
/// SiLU MLP (`decoder_dit.py::TimestepEmbedder`).
#[derive(Debug, Clone)]
struct TimestepEmbedder {
    mlp0: SaLinear,
    mlp2: SaLinear,
    freq_embedding_size: usize,
}

impl TimestepEmbedder {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize) -> Result<Self> {
        let size = 256;
        Ok(TimestepEmbedder {
            mlp0: SaLinear::load(file, &format!("{prefix}.mlp.0"), size, hidden)?,
            mlp2: SaLinear::load(file, &format!("{prefix}.mlp.2"), hidden, hidden)?,
            freq_embedding_size: size,
        })
    }

    fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
        let half = dim / 2;
        let mut emb = vec![0.0f32; dim];
        for d in 0..half {
            let freq = (-(10.0f32.ln()) * d as f32 / half as f32).exp();
            let arg = t * freq;
            emb[d] = arg.cos();
            emb[half + d] = arg.sin();
        }
        emb
    }

    /// Returns the `[hidden]` embedding of one timestep.
    fn forward(&self, t: f32) -> Result<Vec<f32>> {
        let emb = Self::timestep_embedding(t * 1000.0, self.freq_embedding_size);
        let mut h = self.mlp0.forward(&emb, 1);
        ops::silu(&mut h);
        Ok(self.mlp2.forward(&h, 1))
    }
}

/// The DiT estimator (`decoder_dit.py::DiT`). Input assembly runs in
/// channel-major `[4 * mel, T]` (x, mu, speaker, cond), the transformer
/// in rows.
#[derive(Debug, Clone)]
struct DiT {
    t_embedder: TimestepEmbedder,
    in_proj: SaLinear,
    blocks: Vec<DitBlock>,
    final_layer: FinalLayer,
    in_channels: usize,
    out_channels: usize,
    hidden: usize,
}

impl DiT {
    fn load(file: &SafetensorsFile, prefix: &str, config: &StepAudio2Config) -> Result<Self> {
        let in_channels = 4 * config.output_size;
        let hidden = config.dit_hidden;
        let mlp_hidden = (hidden as f32 * config.dit_mlp_ratio) as usize;
        let blocks = (0..config.dit_depth)
            .map(|i| {
                DitBlock::load(
                    file,
                    &format!("{prefix}.blocks.{i}"),
                    hidden,
                    config.dit_heads,
                    config.dit_head_dim,
                    mlp_hidden,
                )
            })
            .collect::<Result<_>>()?;
        Ok(DiT {
            t_embedder: TimestepEmbedder::load(file, &format!("{prefix}.t_embedder"), hidden)?,
            in_proj: SaLinear::load(file, &format!("{prefix}.in_proj"), in_channels, hidden)?,
            blocks,
            final_layer: FinalLayer::load(
                file,
                &format!("{prefix}.final_layer"),
                hidden,
                config.output_size,
            )?,
            in_channels,
            out_channels: config.output_size,
            hidden,
        })
    }

    /// Channel-major inputs: `x`, `mu`, `cond` are `[mel, T]`, `spks`
    /// is `[mel]` (broadcast over time). Returns `[mel, T]`.
    fn forward(
        &self,
        x: &[f32],
        mu: &[f32],
        spks: Option<&[f32]>,
        cond: Option<&[f32]>,
        t: f32,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let mel = self.out_channels;
        // Assemble [4 * mel, T]: x, mu, spks, cond.
        let mut packed = vec![0.0f32; self.in_channels * frames];
        packed[..mel * frames].copy_from_slice(x);
        packed[mel * frames..2 * mel * frames].copy_from_slice(mu);
        if let Some(s) = spks {
            for (i, v) in packed[2 * mel * frames..3 * mel * frames]
                .iter_mut()
                .enumerate()
            {
                *v = s[i % mel];
            }
        }
        if let Some(c) = cond {
            packed[3 * mel * frames..4 * mel * frames].copy_from_slice(c);
        }
        let mut rows = to_rows(&packed, self.in_channels, frames);
        rows = self.in_proj.forward(&rows, frames);
        let t_emb = self.t_embedder.forward(t)?;
        for block in &self.blocks {
            block.forward(&mut rows, &t_emb, frames);
        }
        self.final_layer.forward(&mut rows, &t_emb, frames);
        Ok(to_cm(&rows, frames, mel))
    }
}

/// Causal conditional flow matching (`flow_matching.py
/// ::CausalConditionalCFM`): 10-step Euler over the cosine schedule
/// with classifier-free guidance.
#[derive(Debug, Clone)]
struct CausalConditionalCfm {
    estimator: DiT,
    cfg_rate: f32,
    /// Checkpoint noise buffer, channel-major `[mel, 50 * 600]`.
    rand_noise: Vec<f32>,
}

impl CausalConditionalCfm {
    fn load(file: &SafetensorsFile, prefix: &str, config: &StepAudio2Config) -> Result<Self> {
        let rand_noise = load_f32_shaped(
            file,
            &format!("{prefix}.rand_noise"),
            &[config.output_size, 50 * 600],
        )?;
        Ok(CausalConditionalCfm {
            estimator: DiT::load(file, &format!("{prefix}.estimator"), config)?,
            cfg_rate: config.inference_cfg_rate,
            rand_noise,
        })
    }

    /// `mu` and `cond` are channel-major `[mel, T]`, `spks` is `[mel]`.
    /// Returns the solved `[mel, T]`.
    fn forward(
        &self,
        mu: &[f32],
        spks: &[f32],
        cond: &[f32],
        n_timesteps: usize,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let mel = self.estimator.out_channels;
        let zeros = vec![0.0f32; mel * frames];
        let zeros_spk = vec![0.0f32; mel];
        // t_span = 1 - cos(linspace(0, 1, n + 1) * pi / 2).
        let t_span: Vec<f32> = (0..=n_timesteps)
            .map(|i| 1.0 - ((i as f32 / n_timesteps as f32) * 0.5 * std::f32::consts::PI).cos())
            .collect();
        let mut x = self.rand_noise[..mel * frames].to_vec();
        let mut t = t_span[0];
        let mut dt = t_span[1] - t_span[0];
        for step in 1..t_span.len() {
            let dphi = self
                .estimator
                .forward(&x, mu, Some(spks), Some(cond), t, frames)?;
            let cfg_dphi =
                self.estimator
                    .forward(&x, &zeros, Some(&zeros_spk), Some(&zeros), t, frames)?;
            for i in 0..x.len() {
                x[i] += dt * ((1.0 + self.cfg_rate) * dphi[i] - self.cfg_rate * cfg_dphi[i]);
            }
            t += dt;
            if step < t_span.len() - 1 {
                dt = t_span[step + 1] - t;
            }
        }
        Ok(x)
    }
}

/// The assembled flow model (`flow.py::CausalMaskedDiffWithXvec`).
pub struct StepAudio2Flow {
    config: StepAudio2Config,
    input_embedding: Vec<f32>,
    spk_embed_affine: SaLinear,
    encoder: UpsampleConformerEncoder,
    encoder_proj: SaLinear,
    decoder: CausalConditionalCfm,
}

impl StepAudio2Flow {
    pub fn load(config: &StepAudio2Config, file: &SafetensorsFile) -> Result<Self> {
        let prefix = "flow";
        let embedding = load_f32_shaped(
            file,
            &format!("{prefix}.input_embedding.weight"),
            &[config.vocab_size, config.input_size],
        )?;
        Ok(StepAudio2Flow {
            config: config.clone(),
            input_embedding: embedding,
            spk_embed_affine: SaLinear::load(
                file,
                &format!("{prefix}.spk_embed_affine_layer"),
                config.spk_embed_dim,
                config.output_size,
            )?,
            encoder: UpsampleConformerEncoder::load(
                file,
                &format!("{prefix}.encoder"),
                config.input_size,
                config.input_size,
                config.attention_heads,
                config.linear_units,
                config.num_blocks,
                config.num_up_blocks,
                config.up_stride,
                config.pre_lookahead_len,
                true,
            )?,
            encoder_proj: SaLinear::load(
                file,
                &format!("{prefix}.encoder_proj"),
                config.input_size,
                config.output_size,
            )?,
            decoder: CausalConditionalCfm::load(file, &format!("{prefix}.decoder"), config)?,
        })
    }

    /// Mel frames per token (the encoder upsample factor).
    pub fn up_rate(&self) -> usize {
        self.config.up_stride
    }

    /// Batch-of-one inference: prompt + generated tokens and the
    /// length-matched prompt mel in, generated mel frames out as rows
    /// `[mel_len2, num_mels]`.
    pub fn infer(
        &mut self,
        speech_tokens: &[i32],
        prompt_token: &[i32],
        prompt_feat: &[f32],
        embedding: &[f32],
        n_timesteps: usize,
    ) -> Result<Vec<f32>> {
        let mel = self.config.output_size;
        if embedding.len() != self.config.spk_embed_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    embedding.len(),
                    self.config.spk_embed_dim
                ),
            });
        }
        if prompt_feat.len() % mel != 0 {
            return Err(SpeechError::Input {
                why: "prompt_feat must be row-major mel frames".to_string(),
            });
        }
        // L2-normalize the speaker embedding (+1e-8) and project.
        let mut spk = embedding.to_vec();
        let norm = spk.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        for v in &mut spk {
            *v /= norm;
        }
        let spk = self.spk_embed_affine.forward(&spk, 1);

        // Embed the clipped concatenated tokens (the all-valid mask
        // reduces to an identity multiply).
        let mut tokens = prompt_token.to_vec();
        tokens.extend_from_slice(speech_tokens);
        let mut token_rows = vec![0.0f32; tokens.len() * self.config.input_size];
        for (i, &tok) in tokens.iter().enumerate() {
            let idx = tok.clamp(0, self.config.vocab_size as i32 - 1) as usize;
            token_rows[i * self.config.input_size..(i + 1) * self.config.input_size]
                .copy_from_slice(
                    &self.input_embedding
                        [idx * self.config.input_size..(idx + 1) * self.config.input_size],
                );
        }
        let t = tokens.len();
        let h = self.encoder.forward(&token_rows, t)?;
        let mu_rows = self.encoder_proj.forward(&h, t * self.config.up_stride);
        let mu = to_cm(&mu_rows, t * self.config.up_stride, mel);

        // conds = [prompt_feat, zeros] over the full length, channel-major.
        let prompt_len = prompt_feat.len() / mel;
        let total = t * self.config.up_stride;
        let mut cond_cm = vec![0.0f32; mel * total];
        for (i, &v) in prompt_feat.iter().enumerate() {
            let frame = i / mel;
            let bin = i % mel;
            cond_cm[bin * total + frame] = v;
        }
        let feat = self
            .decoder
            .forward(&mu, &spk, &cond_cm, n_timesteps, total)?;
        // Slice off the prompt span, back to rows.
        let gen_len = total - prompt_len;
        let mut out = vec![0.0f32; gen_len * mel];
        for bin in 0..mel {
            for f in 0..gen_len {
                out[f * mel + bin] = feat[bin * total + prompt_len + f];
            }
        }
        Ok(out)
    }
}
