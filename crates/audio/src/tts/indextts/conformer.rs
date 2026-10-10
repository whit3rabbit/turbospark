//! The indextts conditioning Conformer.
//!
//! Port of `mlx_audio/tts/models/indextts/conformer.py` and its
//! `attention.py` pieces (RelPositionalEncoding, MultiHeadAttention,
//! RelPositionMultiHeadAttention) at the pinned mlx-audio commit. The
//! encoder maps the 100-bin log-mel reference to the GPT conditioning
//! context: conv2d subsampling, relative-position attention (content
//! plus position bias through `pos_bias_u` / `pos_bias_v`), the
//! gated-GLU depthwise convolution module, and per-block SiLU
//! feed-forwards.
//!
//! The reference consumes `NLC` activations; this port keeps row-major
//! `[T, C]` and transposes inside the conv modules only. The streaming
//! cache path is refused: the indextts `get_conditioning` always runs
//! the encoder cacheless over the full subsampled sequence.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{load_bias, load_mlx_conv_weight, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::ops;
use crate::{Result, SpeechError};

/// `ConformerArgs`, with the reference dataclass defaults.
#[derive(Debug, Clone)]
pub struct ConformerArgs {
    pub input_size: usize,
    pub output_size: usize,
    pub num_blocks: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub pos_enc_layer_type: String,
    pub input_layer: String,
    pub cnn_module_kernel: usize,
    pub pos_emb_max_len: usize,
    pub use_bias: bool,
    pub xscaling: bool,
    pub macaron_style: bool,
    pub perceiver_mult: usize,
}

impl ConformerArgs {
    /// `ConformerArgs` from the `condition_module` config dict; missing
    /// keys take the dataclass defaults. The invalid `input_layer`
    /// default (`"conv2d"` is not in the reference `CONV_LAYERS` table
    /// and would KeyError there) is refused here.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let get = |key: &str| value.get(key);
        let u = |key: &str, fallback: usize| -> usize {
            get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(fallback as u64) as usize
        };
        let args = ConformerArgs {
            input_size: u("input_size", 100),
            output_size: u("output_size", 256),
            num_blocks: u("num_blocks", 6),
            linear_units: u("linear_units", 2048),
            attention_heads: u("attention_heads", 4),
            pos_enc_layer_type: get("pos_enc_layer_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("rel_pos")
                .to_owned(),
            input_layer: get("input_layer")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("conv2d")
                .to_owned(),
            cnn_module_kernel: u("cnn_module_kernel", 15),
            pos_emb_max_len: u("pos_emb_max_len", 2048),
            use_bias: get("use_bias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            xscaling: get("xscaling")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            macaron_style: get("macaron_style")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            perceiver_mult: u("perceiver_mult", 2),
        };
        if subsampling_of(&args.input_layer).is_none() {
            return Err(SpeechError::BadConfig {
                field: "condition_module.input_layer".into(),
                why: format!(
                    "{:?} is not one of conv2d2/conv2d3/conv2d4/conv2d6/conv2d8 \
                     (the reference CONV_LAYERS lookup fails on it)",
                    args.input_layer
                ),
            });
        }
        if args.pos_enc_layer_type != "rel_pos" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "condition_module.pos_enc_layer_type {:?} is refused: only the \
                     rel_pos attention is ported (the reference falls back to plain \
                     attention without position embeddings)",
                    args.pos_enc_layer_type
                ),
            });
        }
        if (args.cnn_module_kernel - 1) % 2 != 0 {
            return Err(SpeechError::BadConfig {
                field: "condition_module.cnn_module_kernel".into(),
                why: "the reference asserts (kernel - 1) % 2 == 0".into(),
            });
        }
        Ok(args)
    }
}

/// The reference `CONV_LAYERS` table: per input layer, its
/// `(kernel, stride)` stack. The mask slices are out of contract (the
/// conditioning pass runs unmasked).
fn subsampling_of(input_layer: &str) -> Option<Vec<(usize, usize)>> {
    match input_layer {
        "conv2d2" => Some(vec![(3, 2)]),
        "conv2d3" => Some(vec![(5, 3)]),
        "conv2d4" => Some(vec![(3, 2), (3, 2)]),
        "conv2d6" => Some(vec![(3, 2), (5, 3)]),
        "conv2d8" => Some(vec![(3, 2), (3, 2), (3, 2)]),
        _ => None,
    }
}

/// Loads a Linear whose stored weight is `[out, in]` with an optional
/// `.bias`.
fn load_linear(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
    bias: BiasMode,
) -> Result<Linear> {
    let weight = load_f32_shaped(file, &format!("{name}.weight"), &[output, input])?;
    let bias = load_bias(file, &format!("{name}.bias"), output, bias)?;
    Ok(Linear::new(weight, bias, input, output))
}

/// Loads a LayerNorm (`weight` plus `bias`) with the reference eps 1e-5.
fn load_layer_norm(file: &SafetensorsFile, name: &str, dim: usize) -> Result<crate::nn::LayerNorm> {
    Ok(crate::nn::LayerNorm::new(
        load_f32_shaped(file, &format!("{name}.weight"), &[dim])?,
        Some(load_f32_shaped(file, &format!("{name}.bias"), &[dim])?),
        1e-5,
    ))
}

/// `RelPositionalEncoding`: the fixed sinusoidal table plus the
/// `sqrt(d_model)` input scaling (`xscaling`).
struct RelPositionalEncoding {
    /// `[max_len, dim]` row-major.
    pe: Vec<f32>,
    dim: usize,
    scale: f32,
}

impl RelPositionalEncoding {
    fn new(d_model: usize, max_len: usize, scale_input: bool) -> Self {
        // calculate_pe: pe[:, 0::2] = sin(pos * div_term), 1::2 = cos.
        let mut pe = vec![0.0f32; max_len * d_model];
        for position in 0..max_len {
            for i in 0..d_model / 2 {
                let angle =
                    position as f32 * ((2 * i) as f32 * -(10000.0f32.ln() / d_model as f32)).exp();
                pe[position * d_model + 2 * i] = angle.sin();
                pe[position * d_model + 2 * i + 1] = angle.cos();
            }
        }
        RelPositionalEncoding {
            pe,
            dim: d_model,
            scale: if scale_input {
                (d_model as f32).sqrt()
            } else {
                1.0
            },
        }
    }

    /// Scales `x` in place and returns the position embedding rows
    /// `[T, dim]` for `offset..offset + T`.
    fn apply(&self, x: &mut [f32], len: usize, offset: usize) -> Vec<f32> {
        for value in x.iter_mut() {
            *value *= self.scale;
        }
        let start = offset * self.dim;
        self.pe[start..start + len * self.dim].to_vec()
    }
}

/// One `MultiHeadAttention` bundle over `[T, C]` rows: q/k/v/out
/// projections. The conditioning pass never streams, so only the
/// relative-position flavor runs a forward here (the perceiver family
/// module carries its own attention over [context | latents]).
struct MultiHeadAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        n_feat: usize,
        n_heads: usize,
        head_dim: usize,
        bias: bool,
    ) -> Result<Self> {
        let width = head_dim * n_heads;
        let mode = if bias {
            BiasMode::Required
        } else {
            BiasMode::Optional
        };
        Ok(MultiHeadAttention {
            linear_q: load_linear(file, &format!("{prefix}.linear_q"), n_feat, width, mode)?,
            linear_k: load_linear(file, &format!("{prefix}.linear_k"), n_feat, width, mode)?,
            linear_v: load_linear(file, &format!("{prefix}.linear_v"), n_feat, width, mode)?,
            linear_out: load_linear(file, &format!("{prefix}.linear_out"), width, n_feat, mode)?,
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }
}

/// `RelPositionMultiHeadAttention`: the content path through
/// `pos_bias_u`, the relative position bias `q_v @ p^T * scale` from
/// `pos_bias_v`, both added before the softmax.
struct RelPositionMultiHeadAttention {
    base: MultiHeadAttention,
    linear_pos: Linear,
    /// `[heads, head_dim]`.
    pos_bias_u: Vec<f32>,
    /// `[heads, head_dim]`.
    pos_bias_v: Vec<f32>,
}

impl RelPositionMultiHeadAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        n_feat: usize,
        n_heads: usize,
        head_dim: usize,
        bias: bool,
    ) -> Result<Self> {
        let base = MultiHeadAttention::load(file, prefix, n_feat, n_heads, head_dim, bias)?;
        let width = head_dim * n_heads;
        let pos_bias_u =
            load_f32_shaped(file, &format!("{prefix}.pos_bias_u"), &[n_heads, head_dim])?;
        let pos_bias_v =
            load_f32_shaped(file, &format!("{prefix}.pos_bias_v"), &[n_heads, head_dim])?;
        Ok(RelPositionMultiHeadAttention {
            base,
            linear_pos: load_linear(
                file,
                &format!("{prefix}.linear_pos"),
                n_feat,
                width,
                BiasMode::Optional,
            )?,
            pos_bias_u,
            pos_bias_v,
        })
    }

    /// Self-attention over `[T, C]` with the position embedding rows
    /// `[T, C]` (full context, no mask, no cache).
    fn forward(&self, x: &[f32], len: usize, pos_emb: &[f32]) -> Vec<f32> {
        let heads = self.base.n_heads;
        let dim = self.base.head_dim;
        let width = heads * dim;
        let q = self.base.linear_q.forward(x, len);
        let k = self.base.linear_k.forward(x, len);
        let v = self.base.linear_v.forward(x, len);
        let p = self.linear_pos.forward(pos_emb, len);

        // q_u = q + pos_bias_u, q_v = q + pos_bias_v, per head.
        let mut q_u = q;
        let mut q_v = q_u.clone();
        for t in 0..len {
            for h in 0..heads {
                for d in 0..dim {
                    let index = t * width + h * dim + d;
                    q_u[index] += self.pos_bias_u[h * dim + d];
                    q_v[index] += self.pos_bias_v[h * dim + d];
                }
            }
        }

        // matrix_bd[h, t, j] = sum_d q_v[t, h, d] * p[j, h, d], scaled;
        // the reference adds it to the attention scores through the
        // float mask argument, PER HEAD.
        let mut biased = vec![0.0f32; heads * len * len];
        for h in 0..heads {
            for t in 0..len {
                for j in 0..len {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_v[t * width + h * dim + d] * p[j * width + h * dim + d];
                    }
                    biased[h * len * len + t * len + j] = dot * self.base.scale;
                }
            }
        }

        let mut out = vec![0.0f32; len * width];
        let mut scores = vec![0.0f32; len];
        for h in 0..heads {
            for t in 0..len {
                let mut max = f32::NEG_INFINITY;
                for j in 0..len {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_u[t * width + h * dim + d] * k[j * width + h * dim + d];
                    }
                    let score = dot * self.base.scale + biased[h * len * len + t * len + j];
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
                for j in 0..len {
                    let weight = scores[j] * inv;
                    for d in 0..dim {
                        out[t * width + h * dim + d] += weight * v[j * width + h * dim + d];
                    }
                }
            }
        }
        self.base.linear_out.forward(&out, len)
    }
}

/// One conv layer of the subsampling stem.
struct SubsamplingConv {
    /// ops layout `[out, in_g, kh, kw]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    kernel: usize,
    stride: usize,
}

/// The conv2d subsampling stem (`Conv2dSubsampling`): ReLU conv stack
/// over `(time, freq)`, then the flatten + linear projection.
struct Conv2dSubsampling {
    convs: Vec<SubsamplingConv>,
    out: Linear,
    output_size: usize,
    input_size: usize,
}

impl Conv2dSubsampling {
    fn load(file: &SafetensorsFile, prefix: &str, args: &ConformerArgs) -> Result<Self> {
        let layers = subsampling_of(&args.input_layer).ok_or_else(|| SpeechError::BadConfig {
            field: "input_layer".into(),
            why: format!("unknown input layer {:?}", args.input_layer),
        })?;
        let mut convs = Vec::with_capacity(layers.len());
        let mut in_channels = 1usize;
        let mut out_freq = args.input_size;
        for (index, &(kernel, stride)) in layers.iter().enumerate() {
            let name = format!("{prefix}conv.{index}");
            let stored = load_f32_shaped(
                file,
                &format!("{name}.weight"),
                &[args.output_size, kernel, kernel, in_channels],
            )?;
            // MLX [out, kh, kw, in_g] -> ops [out, in_g, kh, kw].
            let out_ch = args.output_size;
            let mut weight = vec![0.0f32; stored.len()];
            for o in 0..out_ch {
                for kh in 0..kernel {
                    for kw in 0..kernel {
                        for ic in 0..in_channels {
                            weight[o * in_channels * kernel * kernel
                                + ic * kernel * kernel
                                + kh * kernel
                                + kw] =
                                stored[((o * kernel + kh) * kernel + kw) * in_channels + ic];
                        }
                    }
                }
            }
            let bias = load_bias(file, &format!("{name}.bias"), out_ch, BiasMode::Required)?;
            convs.push(SubsamplingConv {
                weight,
                bias,
                kernel,
                stride,
            });
            in_channels = out_ch;
            // The reference out_freq recurrence: (f - k + s) // s.
            out_freq = (out_freq + stride - kernel) / stride;
        }
        let out = load_linear(
            file,
            &format!("{prefix}out.0"),
            args.output_size * out_freq,
            args.output_size,
            BiasMode::Required,
        )?;
        Ok(Conv2dSubsampling {
            convs,
            out,
            output_size: args.output_size,
            input_size: args.input_size,
        })
    }

    /// `[T, input_size]` mel rows to `[T', output_size]`.
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        // Channel-major [1, T, freq] for the conv stack.
        let mut h = x.to_vec();
        let mut height = frames;
        let mut width = self.input_size;
        for conv in &self.convs {
            h = ops::conv2d(
                &h,
                &conv.weight,
                conv.bias.as_deref(),
                1,
                self.output_size,
                height,
                width,
                conv.kernel,
                conv.kernel,
                conv.stride,
                0,
                1,
            );
            for value in h.iter_mut() {
                *value = value.max(0.0);
            }
            height = (height - conv.kernel) / conv.stride + 1;
            width = (width - conv.kernel) / conv.stride + 1;
        }
        // [out, T', W'] -> per frame, concat (channel, freq): the
        // reference swapaxes(2, 3).reshape folds [B, T', C, W'] into
        // [B, T', C * W'].
        let mut flattened = vec![0.0f32; height * self.output_size * width];
        for t in 0..height {
            for c in 0..self.output_size {
                for w in 0..width {
                    flattened[t * self.output_size * width + c * width + w] =
                        h[c * height * width + t * width + w];
                }
            }
        }
        self.out.forward(&flattened, height)
    }
}

/// The gated-GLU depthwise convolution module (`Convolution`).
struct Convolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: crate::nn::LayerNorm,
    pointwise_conv2: Conv1d,
    output_size: usize,
}

impl Convolution {
    fn load(file: &SafetensorsFile, prefix: &str, args: &ConformerArgs) -> Result<Self> {
        let c = args.output_size;
        let k = args.cnn_module_kernel;
        let bias_mode = BiasMode::required(args.use_bias);
        let load = |name: &str,
                    out_ch: usize,
                    in_ch: usize,
                    kernel: usize,
                    groups: usize,
                    pad: usize|
         -> Result<Conv1d> {
            Ok(Conv1d {
                weight: load_mlx_conv_weight(
                    file,
                    &format!("{prefix}{name}.weight"),
                    out_ch,
                    kernel,
                    in_ch / groups,
                )?,
                bias: load_bias(file, &format!("{prefix}{name}.bias"), out_ch, bias_mode)?,
                in_ch,
                out_ch,
                kernel,
                stride: 1,
                padding: pad,
                dilation: 1,
                groups,
            })
        };
        Ok(Convolution {
            pointwise_conv1: load("pointwise_conv1", 2 * c, c, 1, 1, 0)?,
            depthwise_conv: load("depthwise_conv", c, c, k, c, (k - 1) / 2)?,
            norm: load_layer_norm(file, &format!("{prefix}norm"), c)?,
            pointwise_conv2: load("pointwise_conv2", c, c, 1, 1, 0)?,
            output_size: c,
        })
    }

    /// `[T, C]` -> `[T, C]`.
    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let c = self.output_size;
        let to_channels = |x: &[f32]| {
            let mut out = vec![0.0f32; x.len()];
            for t in 0..len {
                for ch in 0..c {
                    out[ch * len + t] = x[t * c + ch];
                }
            }
            out
        };
        let from_channels = |x: &[f32]| {
            let mut out = vec![0.0f32; x.len()];
            for t in 0..len {
                for ch in 0..c {
                    out[t * c + ch] = x[ch * len + t];
                }
            }
            out
        };
        let h = self.pointwise_conv1.forward(&to_channels(x));
        // GLU over the channel axis: first half * sigmoid(second half).
        let frames = h.len() / (2 * c);
        let mut gated = vec![0.0f32; c * frames];
        for ch in 0..c {
            for t in 0..frames {
                gated[ch * frames + t] =
                    h[ch * frames + t] * ops::sigmoid(h[(c + ch) * frames + t]);
            }
        }
        // Symmetric zero-free edge replicate? No: the reference pads with
        // zeros here (nn.Conv1d padding), which ops::conv1d applies
        // through the `padding` argument.
        let convolved = self.depthwise_conv.forward(&gated);
        let mut normed = from_channels(&convolved);
        self.norm.apply(&mut normed, frames);
        ops::silu(&mut normed);
        let out = self.pointwise_conv2.forward(&to_channels(&normed));
        from_channels(&out)
    }
}

/// `FeedForward`: `w_2(silu(w_1(x)))` with biases.
struct FeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl FeedForward {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        d_ff: usize,
        use_bias: bool,
    ) -> Result<Self> {
        let mode = BiasMode::required(use_bias);
        Ok(FeedForward {
            w_1: load_linear(file, &format!("{prefix}w_1"), dim, d_ff, mode)?,
            w_2: load_linear(file, &format!("{prefix}w_2"), d_ff, dim, mode)?,
        })
    }

    fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut hidden = self.w_1.forward(x, len);
        ops::silu(&mut hidden);
        self.w_2.forward(&hidden, len)
    }
}

struct ConformerBlock {
    norm_mha: crate::nn::LayerNorm,
    self_attn: RelPositionMultiHeadAttention,
    norm_conv: crate::nn::LayerNorm,
    conv_module: Convolution,
    norm_ff: crate::nn::LayerNorm,
    feed_forward: FeedForward,
    norm_final: crate::nn::LayerNorm,
}

impl ConformerBlock {
    fn forward(&self, x: &[f32], len: usize, pos_emb: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        let mut normed = h.clone();
        self.norm_mha.apply(&mut normed, len);
        let attended = self.self_attn.forward(&normed, len, pos_emb);
        for (value, add) in h.iter_mut().zip(&attended) {
            *value += add;
        }
        let mut normed = h.clone();
        self.norm_conv.apply(&mut normed, len);
        let conv = self.conv_module.forward(&normed, len);
        for (value, add) in h.iter_mut().zip(&conv) {
            *value += add;
        }
        let mut normed = h.clone();
        self.norm_ff.apply(&mut normed, len);
        let fed = self.feed_forward.forward(&normed, len);
        for (value, add) in h.iter_mut().zip(&fed) {
            *value += add;
        }
        self.norm_final.apply(&mut h, len);
        Ok(h)
    }
}

/// The loaded Conformer conditioning encoder.
pub struct Conformer {
    pos_enc: RelPositionalEncoding,
    embed: Conv2dSubsampling,
    encoders: Vec<ConformerBlock>,
    after_norm: crate::nn::LayerNorm,
    output_size: usize,
}

impl Conformer {
    pub fn load(file: &SafetensorsFile, args: &ConformerArgs, prefix: &str) -> Result<Self> {
        if args.macaron_style {
            return Err(SpeechError::Unsupported {
                why: "condition_module.macaron_style=true needs the macaron \
                      feed-forward the pinned checkpoints never enable"
                    .into(),
            });
        }
        let pos_enc =
            RelPositionalEncoding::new(args.output_size, args.pos_emb_max_len, args.xscaling);
        let embed = Conv2dSubsampling::load(file, &format!("{prefix}embed."), args)?;
        let mut encoders = Vec::with_capacity(args.num_blocks);
        for block in 0..args.num_blocks {
            let p = format!("{prefix}encoders.{block}.");
            let c = args.output_size;
            encoders.push(ConformerBlock {
                norm_mha: load_layer_norm(file, &format!("{p}norm_mha"), c)?,
                self_attn: RelPositionMultiHeadAttention::load(
                    file,
                    &format!("{p}self_attn"),
                    c,
                    args.attention_heads,
                    c / args.attention_heads,
                    args.use_bias,
                )?,
                norm_conv: load_layer_norm(file, &format!("{p}norm_conv"), c)?,
                conv_module: Convolution::load(file, &format!("{p}conv_module."), args)?,
                norm_ff: load_layer_norm(file, &format!("{p}norm_ff"), c)?,
                feed_forward: FeedForward::load(
                    file,
                    &format!("{p}feed_forward."),
                    c,
                    args.linear_units,
                    args.use_bias,
                )?,
                norm_final: load_layer_norm(file, &format!("{p}norm_final"), c)?,
            });
        }
        let after_norm = load_layer_norm(file, &format!("{prefix}after_norm"), args.output_size)?;
        Ok(Conformer {
            pos_enc,
            embed,
            encoders,
            after_norm,
            output_size: args.output_size,
        })
    }

    /// `[T, input_size]` mel rows to the `[T', output_size]` context.
    pub fn forward(&self, x: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut h = self.embed.forward(x, frames);
        let len = h.len() / self.output_size;
        if len == 0 {
            return Err(SpeechError::Input {
                why: "empty conformer input".into(),
            });
        }
        let pos_emb = self.pos_enc.apply(&mut h, len, 0);
        for block in &self.encoders {
            h = block.forward(&h, len, &pos_emb)?;
        }
        self.after_norm.apply(&mut h, len);
        Ok(h)
    }
}
