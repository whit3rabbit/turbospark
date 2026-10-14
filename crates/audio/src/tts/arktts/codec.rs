//! The bundled Audio8 44.1 kHz codec behind arktts.
//!
//! Port of `mlx_audio/tts/models/arktts/codec.py` (modeling_arktts_codec)
//! at the pinned mlx-audio commit, decode direction only: residual VQ
//! `from_codes`, the post window transformer, the transposed-conv upsampler,
//! and the Snake decoder. The encode direction (encoder, downsample,
//! pre-transformer, quantize) is a documented refusal; voice-clone callers
//! pass precomputed reference codes, exactly like the higgs_audio family.
//!
//! Layout: the Python is channels-last `(B, T, C)`; this port keeps
//! channel-major `[C, T]` buffers for the conv stages (the crate kernel
//! convention) and row-major `[T, C]` for the transformer blocks,
//! transposing at the section boundaries. Fixture weights are saved
//! post-sanitize: weight norm folded, conv weights in the PyTorch layouts
//! the crate kernels take (`[out, in/groups, K]`, transposed
//! `[in, out/groups, K]`), linear weights `[out, in]`.

use serde_json::Value;

use crate::codec::conv::{permute_okc_to_cok, permute_okc_to_ock, Conv1d, ConvTranspose1d};
use crate::nn::{self, Linear};
use crate::{Result, SpeechError};

use super::Weights;

/// Codec geometry. The reference hardcodes every field except the post
/// transformer sizes (which ride the TTS ModelConfig); the defaults are the
/// published checkpoint's, and the fixture generator overrides them with a
/// tiny pipeline that exercises the same block structure.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub hidden: usize,
    pub quantizer_input_dim: usize,
    pub decoder_channels: usize,
    /// Upsample strides, outermost first.
    pub upsample_strides: Vec<usize>,
    pub semantic_codebook_size: usize,
    pub residual_books: usize,
    pub residual_codebook_size: usize,
    pub codebook_dim: usize,
    pub post_n_layer: usize,
    pub post_n_head: usize,
    pub post_n_local_heads: usize,
    pub post_intermediate_size: usize,
}

impl CodecConfig {
    /// The published checkpoint's geometry, with the post-transformer
    /// sizes from the TTS config (`codec_post_*` keys).
    pub fn from_tts_config(value: &Value) -> Self {
        let u = |key: &str, default: u64| -> usize {
            value.get(key).and_then(Value::as_u64).unwrap_or(default) as usize
        };
        Self {
            hidden: 1024,
            quantizer_input_dim: 1024,
            decoder_channels: 1536,
            upsample_strides: vec![8, 8, 4, 2],
            semantic_codebook_size: 4096,
            residual_books: 9,
            residual_codebook_size: 1024,
            codebook_dim: 8,
            post_n_layer: u("codec_post_n_layer", 8),
            post_n_head: u("codec_post_n_head", 16),
            post_n_local_heads: u("codec_post_n_local_heads", 8),
            post_intermediate_size: u("codec_post_intermediate_size", 1216),
        }
    }
}

struct WindowTransformerConfig {
    n_layer: usize,
    n_head: usize,
    n_local_heads: usize,
    dim: usize,
    intermediate_size: usize,
    head_dim: usize,
    rope_base: f32,
    norm_eps: f32,
}

/// x + (1 / (alpha + 1e-9)) * sin(alpha x)^2, the codec-local Snake with the
/// guard the shared `wnconv::snake1d` does not carry. Channel-major.
fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    for c in 0..channels {
        let a = alpha[c];
        for t in 0..frames {
            let v = x[c * frames + t];
            let s = (a * v).sin();
            x[c * frames + t] = v + s * s / (a + 1e-9);
        }
    }
}

/// The codec-local exact-GELU (MLX `nn.gelu`).
fn gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + crate::ops::erf(v / std::f32::consts::SQRT_2)))
        .collect()
}

/// Causal conv: left pad `effective_kernel - stride`, right pad to the
/// ideal length, then an unpadded strided conv.
struct CausalConv {
    conv: Conv1d,
    left_pad: usize,
    effective_kernel: usize,
    stride: usize,
    in_ch: usize,
}

impl CausalConv {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let frames = x.len() / self.in_ch;
        let out_frames_ideal = ((frames - self.effective_kernel + self.left_pad) as f64
            / self.stride as f64)
            .ceil() as usize;
        let ideal_len =
            (out_frames_ideal.max(1) - 1) * self.stride + self.effective_kernel - self.left_pad;
        let right = ideal_len.saturating_sub(frames);
        let total = frames + self.left_pad + right;
        let mut padded = vec![0.0f32; total * self.in_ch];
        for c in 0..self.in_ch {
            padded[c * total + self.left_pad..c * total + self.left_pad + frames]
                .copy_from_slice(&x[c * frames..(c + 1) * frames]);
        }
        self.conv.forward(&padded)
    }
}

struct CausalConvTranspose {
    conv: ConvTranspose1d,
    crop: usize,
}

impl CausalConvTranspose {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.conv.forward(x);
        if self.crop > 0 {
            let frames = out.len() / self.conv.out_ch;
            out.truncate((frames - self.crop) * self.conv.out_ch);
        }
        out
    }
}

struct ResidualUnit {
    snake1: Vec<f32>,
    conv_dilated: CausalConv,
    snake2: Vec<f32>,
    conv_point: CausalConv,
    channels: usize,
}

impl ResidualUnit {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let frames = x.len() / self.channels;
        let mut h = x.to_vec();
        snake(&mut h, &self.snake1, self.channels, frames);
        let h = self.conv_dilated.forward(&h);
        let frames = h.len() / self.channels;
        let mut h2 = h;
        snake(&mut h2, &self.snake2, self.channels, frames);
        let h2 = self.conv_point.forward(&h2);
        // The reference crops x's tail when the residual ran longer.
        let x_frames = x.len() / self.channels;
        let keep = x_frames.min(frames);
        let mut out = h2;
        for c in 0..self.channels {
            for t in 0..keep {
                out[c * frames + t] += x[c * x_frames + t];
            }
        }
        out
    }
}

struct ConvNeXtBlock {
    dwconv: CausalConv,
    norm: nn::LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Vec<f32>,
    channels: usize,
}

impl ConvNeXtBlock {
    /// `x` is `[frames, channels]`.
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let major = super::to_channel_major(x, frames, self.channels);
        let convolved = self.dwconv.forward(&major);
        let mut rows = super::to_row_major(&convolved, frames, self.channels);
        self.norm.apply(&mut rows, frames);
        let activated = gelu(&self.pwconv1.forward(&rows, frames));
        let mut projected = self.pwconv2.forward(&activated, frames);
        for row in projected.chunks_mut(self.channels) {
            for (value, gamma) in row.iter_mut().zip(&self.gamma) {
                *value *= gamma;
            }
        }
        for (value, resid) in projected.iter_mut().zip(x) {
            *value += resid;
        }
        projected
    }
}

struct DecoderBlock {
    snake: Vec<f32>,
    upsample: CausalConvTranspose,
    units: Vec<ResidualUnit>,
    input_dim: usize,
}

struct WindowBlock {
    wqkv: Linear,
    wo: Linear,
    w1: Linear,
    w3: Linear,
    w2: Linear,
    attn_norm: nn::RmsNorm,
    ffn_norm: nn::RmsNorm,
    attn_scale: Vec<f32>,
    ffn_scale: Vec<f32>,
    heads: usize,
    local_heads: usize,
    head_dim: usize,
}

struct WindowTransformer {
    layers: Vec<WindowBlock>,
    norm: nn::RmsNorm,
    window: usize,
    rope_base: f32,
}

impl WindowTransformer {
    /// Full-length attention with the causal-window mask; `x` is
    /// `[frames, dim]`.
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let dim = self.layers[0].wqkv.input;
        let half = self.layers[0].head_dim / 2;
        let mut cos = vec![0.0f32; frames * half];
        let mut sin = vec![0.0f32; frames * half];
        for t in 0..frames {
            for d in 0..half {
                let freq = self.rope_base.powf(-(2.0 * d as f32) / (half * 2) as f32);
                let angle = t as f32 * freq;
                cos[t * half + d] = super::round_bf16(angle.cos());
                sin[t * half + d] = super::round_bf16(angle.sin());
            }
        }
        let mut h = x.to_vec();
        for layer in &self.layers {
            let heads = layer.heads;
            let local = layer.local_heads;
            let head_dim = layer.head_dim;
            let rep = heads / local;
            let scale = (head_dim as f64).powf(-0.5) as f32;

            let mut attended = h.clone();
            layer.attn_norm.apply(&mut attended, frames);
            let qkv = layer.wqkv.forward(&attended, frames);
            let q_width = heads * head_dim;
            let kv_width = local * head_dim;
            let q = &qkv[..frames * q_width];
            let k = &qkv[frames * q_width..frames * (q_width + kv_width)];
            let v = &qkv[frames * (q_width + kv_width)..];
            let mut qh = super::head_major(q, frames, heads, head_dim);
            let mut kh = super::head_major(k, frames, local, head_dim);
            let vh = super::head_major(v, frames, local, head_dim);
            super::apply_interleaved_rope(&mut qh, heads, frames, head_dim, &cos, &sin);
            super::apply_interleaved_rope(&mut kh, local, frames, head_dim, &cos, &sin);

            let mut out = vec![0.0f32; frames * q_width];
            let mut scores = vec![0.0f32; frames];
            for head in 0..heads {
                let kv_head = head / rep;
                for t in 0..frames {
                    let q_row =
                        &qh[(head * frames + t) * head_dim..(head * frames + t + 1) * head_dim];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..frames {
                        if j > t || (self.window > 0 && j + self.window <= t) {
                            scores[j] = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_row = &kh[(kv_head * frames + j) * head_dim
                            ..(kv_head * frames + j + 1) * head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_row[d] * k_row[d];
                        }
                        let score = dot * scale;
                        scores[j] = score;
                        if score > max {
                            max = score;
                        }
                    }
                    let mut sum = 0.0f32;
                    for score in scores.iter_mut() {
                        let weight = if *score == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (*score - max).exp()
                        };
                        *score = weight;
                        sum += weight;
                    }
                    let inv = 1.0 / sum;
                    for (j, weight) in scores.iter_mut().enumerate() {
                        let w = *weight * inv;
                        if w == 0.0 {
                            continue;
                        }
                        let v_row = &vh[(kv_head * frames + j) * head_dim
                            ..(kv_head * frames + j + 1) * head_dim];
                        let out_row = &mut out
                            [t * q_width + head * head_dim..t * q_width + (head + 1) * head_dim];
                        for (d, &v) in v_row.iter().enumerate() {
                            out_row[d] += w * v;
                        }
                    }
                }
            }
            let mut attn_out = layer.wo.forward(&out, frames);
            for (value, gamma) in attn_out.iter_mut().zip(layer.attn_scale.iter().cycle()) {
                *value *= gamma;
            }
            for (index, value) in attn_out.iter_mut().enumerate() {
                h[index] += *value;
            }

            let mut fed = h.clone();
            layer.ffn_norm.apply(&mut fed, frames);
            let gate = layer.w1.forward(&fed, frames);
            let up = layer.w3.forward(&fed, frames);
            let fused: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
                .collect();
            let mut ff = layer.w2.forward(&fused, frames);
            for (value, gamma) in ff.iter_mut().zip(layer.ffn_scale.iter().cycle()) {
                *value *= gamma;
            }
            for (index, value) in ff.iter_mut().enumerate() {
                h[index] += *value;
            }
        }
        self.norm.apply(&mut h, frames);
        let _ = dim;
        h
    }
}

struct Quantizers {
    semantic_codebook: Vec<f32>,
    semantic_out: Linear,
    residual_codebooks: Vec<Vec<f32>>,
    residual_out: Vec<Linear>,
    codebook_dim: usize,
    semantic_codebook_size: usize,
    residual_codebook_size: usize,
}

impl Quantizers {
    /// `codes[0]` is the semantic book, the rest are the residual books.
    /// Returns `[frames, input_dim]`.
    fn from_codes(&self, codes: &[Vec<i32>], frames: usize) -> Vec<f32> {
        let mut embedded = vec![0.0f32; frames * self.codebook_dim];
        for t in 0..frames {
            let index = codes[0][t].clamp(0, self.semantic_codebook_size as i32 - 1) as usize;
            embedded[t * self.codebook_dim..(t + 1) * self.codebook_dim].copy_from_slice(
                &self.semantic_codebook[index * self.codebook_dim..(index + 1) * self.codebook_dim],
            );
        }
        let mut sum = self.semantic_out.forward(&embedded, frames);
        for (book, codes) in codes.iter().skip(1).enumerate() {
            let mut rows = vec![0.0f32; frames * self.codebook_dim];
            for t in 0..frames {
                let index = codes[t].clamp(0, self.residual_codebook_size as i32 - 1) as usize;
                rows[t * self.codebook_dim..(t + 1) * self.codebook_dim].copy_from_slice(
                    &self.residual_codebooks[book]
                        [index * self.codebook_dim..(index + 1) * self.codebook_dim],
                );
            }
            let projected = self.residual_out[book].forward(&rows, frames);
            for (sum_v, v) in sum.iter_mut().zip(&projected) {
                *sum_v += v;
            }
        }
        sum
    }
}

/// The loaded decode path.
pub struct ArkttsCodec {
    quantizer: Quantizers,
    post_module: WindowTransformer,
    upsample: Vec<(CausalConvTranspose, ConvNeXtBlock)>,
    hidden: usize,
    decoder: Decoder,
}

struct Decoder {
    first: CausalConv,
    blocks: Vec<DecoderBlock>,
    last_snake: Vec<f32>,
    last_conv: CausalConv,
}

impl Decoder {
    /// Channel-major `[channels, T]` -> waveform `[samples]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.first.forward(x);
        for block in &self.blocks {
            let frames = h.len() / block.input_dim;
            snake(&mut h, &block.snake, block.input_dim, frames);
            let mut next = block.upsample.forward(&h);
            for unit in &block.units {
                next = unit.forward(&next);
            }
            h = next;
        }
        let channels = self.last_conv.conv.in_ch;
        let frames = h.len() / channels;
        snake(&mut h, &self.last_snake, channels, frames);
        let out = self.last_conv.forward(&h);
        out.iter().map(|v| v.tanh()).collect()
    }
}

impl ArkttsCodec {
    /// Loads the decode path from the sanitized checkpoint.
    pub fn load(weights: &Weights, config: &CodecConfig) -> Result<Self> {
        let tf = WindowTransformerConfig {
            n_layer: config.post_n_layer,
            n_head: config.post_n_head,
            n_local_heads: config.post_n_local_heads,
            dim: config.hidden,
            intermediate_size: config.post_intermediate_size,
            head_dim: 64,
            rope_base: 10000.0,
            norm_eps: 1e-5,
        };
        let post_module = load_window_transformer(weights, "quantizer.post_module", &tf)?;
        let quantizer = load_quantizers(weights, config)?;

        let mut upsample = Vec::new();
        for (index, &stride) in config.upsample_strides.iter().enumerate() {
            let kernel = 2 * stride;
            let conv_t = load_convt(
                weights,
                &format!("quantizer.upsample.{index}.0.conv"),
                config.hidden,
                config.hidden,
                kernel,
                stride,
            )?;
            let block = load_convnext(
                weights,
                &format!("quantizer.upsample.{index}.1"),
                config.hidden,
            )?;
            upsample.push((
                CausalConvTranspose {
                    conv: conv_t,
                    crop: kernel - stride,
                },
                block,
            ));
        }
        let decoder = load_decoder(weights, config)?;
        Ok(Self {
            quantizer,
            post_module,
            upsample,
            hidden: config.hidden,
            decoder,
        })
    }

    /// `codes`: one row per book (book 0 semantic, then the residual books),
    /// each `[frames]`. Returns the mono waveform.
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let expected = 1 + self.quantizer.residual_codebooks.len();
        if codes.len() != expected {
            return Err(SpeechError::Input {
                why: format!("expected {expected} codebook rows, got {}", codes.len()),
            });
        }
        let frames = codes[0].len();
        if codes.iter().any(|book| book.len() != frames) {
            return Err(SpeechError::Input {
                why: "codebook rows must share one frame count".into(),
            });
        }
        let z = self.quantizer.from_codes(codes, frames);
        let z = self.post_module.forward(&z, frames);
        let mut major = super::to_channel_major(&z, frames, self.hidden);
        for (upsample, convnext) in &self.upsample {
            major = upsample.forward(&major);
            let frames = major.len() / self.hidden;
            let rows = convnext.forward(&super::to_row_major(&major, frames, self.hidden), frames);
            major = super::to_channel_major(&rows, frames, self.hidden);
        }
        Ok(self.decoder.forward(&major))
    }
}

// Loading helpers ------------------------------------------------------------

fn load_conv_weight(
    weights: &Weights,
    name: &str,
    out_ch: usize,
    in_g: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let stored = weights.tensor(name, &[out_ch, kernel, in_g])?;
    Ok(permute_okc_to_ock(&stored, out_ch, kernel, in_g))
}

fn load_conv(
    weights: &Weights,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Conv1d> {
    let weight = load_conv_weight(
        weights,
        &format!("{name}.weight"),
        out_ch,
        in_ch / groups,
        kernel,
    )?;
    let bias = weights.tensor_optional(&format!("{name}.bias"), &[out_ch])?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: 0,
        dilation,
        groups,
        weight,
        bias,
    })
}

fn load_causal_conv(
    weights: &Weights,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    stride: usize,
    groups: usize,
) -> Result<CausalConv> {
    let effective = (kernel - 1) * dilation + 1;
    let conv = load_conv(
        weights,
        &format!("{name}.conv"),
        in_ch,
        out_ch,
        kernel,
        stride,
        dilation,
        groups,
    )?;
    Ok(CausalConv {
        conv,
        left_pad: effective - stride,
        effective_kernel: effective,
        stride,
        in_ch,
    })
}

fn load_convt(
    weights: &Weights,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
) -> Result<ConvTranspose1d> {
    let stored = weights.tensor(name, &[out_ch, kernel, in_ch])?;
    let weight = permute_okc_to_cok(&stored, out_ch, kernel, in_ch);
    let bias = weights.tensor_optional(
        &format!("{}.bias", name.strip_suffix(".conv").unwrap_or(name)),
        &[out_ch],
    )?;
    Ok(ConvTranspose1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding: 0,
        output_padding: 0,
        groups: 1,
        weight,
        bias,
    })
}

fn load_convnext(weights: &Weights, name: &str, channels: usize) -> Result<ConvNeXtBlock> {
    let dwconv = load_causal_conv(
        weights,
        &format!("{name}.dwconv"),
        channels,
        channels,
        7,
        1,
        1,
        channels,
    )?;
    let norm = nn::LayerNorm {
        weight: weights.tensor(&format!("{name}.norm.weight"), &[channels])?,
        bias: Some(weights.tensor(&format!("{name}.norm.bias"), &[channels])?),
        epsilon: 1e-6,
    };
    let pwconv1 = weights.linear(
        &format!("{name}.pwconv1"),
        channels,
        4 * channels,
        super::Bias::Required,
    )?;
    let pwconv2 = weights.linear(
        &format!("{name}.pwconv2"),
        4 * channels,
        channels,
        super::Bias::Required,
    )?;
    let gamma = weights.tensor(&format!("{name}.gamma"), &[channels])?;
    Ok(ConvNeXtBlock {
        dwconv,
        norm,
        pwconv1,
        pwconv2,
        gamma,
        channels,
    })
}

fn load_window_transformer(
    weights: &Weights,
    name: &str,
    config: &WindowTransformerConfig,
) -> Result<WindowTransformer> {
    let mut layers = Vec::with_capacity(config.n_layer);
    for index in 0..config.n_layer {
        let base = format!("{name}.layers.{index}");
        let total = (config.n_head + 2 * config.n_local_heads) * config.head_dim;
        layers.push(WindowBlock {
            wqkv: weights.linear(
                &format!("{base}.attention.wqkv"),
                config.dim,
                total,
                super::Bias::None,
            )?,
            wo: weights.linear(
                &format!("{base}.attention.wo"),
                config.n_head * config.head_dim,
                config.dim,
                super::Bias::None,
            )?,
            w1: weights.linear(
                &format!("{base}.feed_forward.w1"),
                config.dim,
                config.intermediate_size,
                super::Bias::None,
            )?,
            w3: weights.linear(
                &format!("{base}.feed_forward.w3"),
                config.dim,
                config.intermediate_size,
                super::Bias::None,
            )?,
            w2: weights.linear(
                &format!("{base}.feed_forward.w2"),
                config.intermediate_size,
                config.dim,
                super::Bias::None,
            )?,
            attn_norm: nn::RmsNorm::new(
                weights.tensor(&format!("{base}.attention_norm.weight"), &[config.dim])?,
                config.norm_eps,
            ),
            ffn_norm: nn::RmsNorm::new(
                weights.tensor(&format!("{base}.ffn_norm.weight"), &[config.dim])?,
                config.norm_eps,
            ),
            attn_scale: weights.tensor(
                &format!("{base}.attention_layer_scale.gamma"),
                &[config.dim],
            )?,
            ffn_scale: weights.tensor(&format!("{base}.ffn_layer_scale.gamma"), &[config.dim])?,
            heads: config.n_head,
            local_heads: config.n_local_heads,
            head_dim: config.head_dim,
        });
    }
    let norm = nn::RmsNorm::new(
        weights.tensor(&format!("{name}.norm.weight"), &[config.dim])?,
        config.norm_eps,
    );
    Ok(WindowTransformer {
        layers,
        norm,
        window: 128,
        rope_base: config.rope_base,
    })
}

fn load_quantizers(weights: &Weights, config: &CodecConfig) -> Result<Quantizers> {
    let semantic_codebook = weights.tensor(
        "quantizer.semantic_quantizer.quantizers.0.codebook.weight",
        &[config.semantic_codebook_size, config.codebook_dim],
    )?;
    let semantic_out = weights.linear(
        "quantizer.semantic_quantizer.quantizers.0.out_proj",
        config.codebook_dim,
        config.quantizer_input_dim,
        super::Bias::Required,
    )?;
    let mut residual_codebooks = Vec::with_capacity(config.residual_books);
    let mut residual_out = Vec::with_capacity(config.residual_books);
    for book in 0..config.residual_books {
        residual_codebooks.push(weights.tensor(
            &format!("quantizer.quantizer.quantizers.{book}.codebook.weight"),
            &[config.residual_codebook_size, config.codebook_dim],
        )?);
        residual_out.push(weights.linear(
            &format!("quantizer.quantizer.quantizers.{book}.out_proj"),
            config.codebook_dim,
            config.quantizer_input_dim,
            super::Bias::Required,
        )?);
    }
    Ok(Quantizers {
        semantic_codebook,
        semantic_out,
        residual_codebooks,
        residual_out,
        codebook_dim: config.codebook_dim,
        semantic_codebook_size: config.semantic_codebook_size,
        residual_codebook_size: config.residual_codebook_size,
    })
}

fn load_decoder(weights: &Weights, config: &CodecConfig) -> Result<Decoder> {
    let channels = config.decoder_channels;
    let first = load_causal_conv(
        weights,
        "decoder.model.0",
        config.quantizer_input_dim,
        channels,
        7,
        1,
        1,
        1,
    )?;
    let mut blocks = Vec::new();
    let mut input_dim = channels;
    for (index, &stride) in config.upsample_strides.iter().enumerate() {
        let output_dim = channels / (2usize.pow(index as u32 + 1));
        let base = format!("decoder.model.{}", index + 1);
        let snake = weights
            .tensor(&format!("{base}.block.0.alpha"), &[1, 1, input_dim])
            .or_else(|_| weights.tensor(&format!("{base}.block.0.alpha"), &[input_dim]))?;
        let upsample = load_convt(
            weights,
            &format!("{base}.block.1.conv"),
            input_dim,
            output_dim,
            2 * stride,
            stride,
        )?;
        let mut units = Vec::new();
        for (unit_index, dilation) in [1usize, 3, 9].into_iter().enumerate() {
            let unit_base = format!("{base}.block.{}", unit_index + 2);
            units.push(ResidualUnit {
                snake1: weights
                    .tensor(&format!("{unit_base}.block.0.alpha"), &[1, 1, output_dim])
                    .or_else(|_| {
                        weights.tensor(&format!("{unit_base}.block.0.alpha"), &[output_dim])
                    })?,
                conv_dilated: load_causal_conv(
                    weights,
                    &format!("{unit_base}.block.1"),
                    output_dim,
                    output_dim,
                    7,
                    dilation,
                    1,
                    1,
                )?,
                snake2: weights
                    .tensor(&format!("{unit_base}.block.2.alpha"), &[1, 1, output_dim])
                    .or_else(|_| {
                        weights.tensor(&format!("{unit_base}.block.2.alpha"), &[output_dim])
                    })?,
                conv_point: load_causal_conv(
                    weights,
                    &format!("{unit_base}.block.3"),
                    output_dim,
                    output_dim,
                    1,
                    1,
                    1,
                    1,
                )?,
                channels: output_dim,
            });
        }
        blocks.push(DecoderBlock {
            snake,
            upsample: CausalConvTranspose {
                conv: upsample,
                crop: stride,
            },
            units,
            input_dim,
        });
        input_dim = output_dim;
    }
    let last_book = config.upsample_strides.len() + 1;
    let last_snake = weights
        .tensor(
            &format!("decoder.model.{last_book}.block.0.alpha"),
            &[1, 1, input_dim],
        )
        .or_else(|_| {
            weights.tensor(
                &format!("decoder.model.{last_book}.block.0.alpha"),
                &[input_dim],
            )
        })?;
    let last_conv = load_causal_conv(
        weights,
        &format!("decoder.model.{}", last_book + 1),
        input_dim,
        1,
        7,
        1,
        1,
        1,
    )?;
    Ok(Decoder {
        first,
        blocks,
        last_snake,
        last_conv,
    })
}
