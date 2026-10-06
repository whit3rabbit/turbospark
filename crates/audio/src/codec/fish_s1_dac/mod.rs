//! Fish S1 DAC (fishaudio/fish-speech S1 acoustic tokenizer).
//!
//! Reference: `mlx_audio/codec/models/fish_s1_dac/fish_s1_dac.py` at
//! mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/fish_s1_dac)
//! (the `build_ae()` factory is the reference geometry). A causal DAC
//! whose last encoder block carries a window-limited transformer, a
//! downsampled dual residual quantizer (one semantic book at 4096
//! codes plus nine acoustic books at 1024, both 8-dim, 2x2 downsample
//! with pre/post window-limited transformers at dim 1024), and a
//! mirrored transposed-conv decoder with tanh output. 44.1 kHz, hop
//! 512, frame length 2048.
//!
//! Scope: the causal (`build_ae`) geometry only. Convolution weights
//! are stored in the PyTorch layout (`(out, in, K)` / `(in, out, K)`)
//! with `weight_g` of shape `(dim0, 1, 1)` normalized over every axis
//! but the first. The transformers' `freqs_cis` tables and causal
//! masks are derived state (recomputed from the config, never loaded).

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::{load_f32_shaped, snake1d};
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// WindowLimitedTransformer geometry (`ModelArgs`).
#[derive(Debug, Clone)]
pub struct FishTransformerConfig {
    pub window_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub dim: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl FishTransformerConfig {
    /// The quantizer pre/post module config from `build_ae`'s
    /// `q_config` (block_size 4096, 8 layers, 16 heads, dim 1024).
    pub fn quantizer_default() -> Self {
        FishTransformerConfig {
            window_size: 128,
            n_layer: 8,
            n_head: 16,
            dim: 1024,
            intermediate_size: 3072,
            head_dim: 64,
            rope_base: 10000.0,
            norm_eps: 1e-5,
        }
    }

    /// The encoder block transformer config (window 512, head count
    /// `dim / 64`, intermediate `3 * dim`).
    pub fn encoder_default(dim: usize, n_layer: usize) -> Self {
        FishTransformerConfig {
            window_size: 512,
            n_layer,
            n_head: dim / 64,
            dim,
            intermediate_size: dim * 3,
            head_dim: 64,
            rope_base: 10000.0,
            norm_eps: 1e-5,
        }
    }
}

/// Downsampled dual RVQ geometry.
#[derive(Debug, Clone)]
pub struct FishQuantizerConfig {
    pub input_dim: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub semantic_codebook_size: usize,
    pub codebook_dim: usize,
    pub downsample_factor: Vec<usize>,
    pub pre: FishTransformerConfig,
    pub post: FishTransformerConfig,
}

/// Codec geometry, one-to-one with `build_ae()`.
#[derive(Debug, Clone)]
pub struct FishS1DacConfig {
    pub sample_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub encoder_transformer_layers: Vec<usize>,
    pub encoder_transformer: FishTransformerConfig,
    pub quantizer: FishQuantizerConfig,
}

impl FishS1DacConfig {
    /// `build_ae()` defaults.
    pub fn build_ae() -> Self {
        FishS1DacConfig {
            sample_rate: 44_100,
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 8, 8],
            latent_dim: 1024,
            decoder_dim: 1536,
            decoder_rates: vec![8, 8, 4, 2],
            encoder_transformer_layers: vec![0, 0, 0, 4],
            encoder_transformer: FishTransformerConfig::encoder_default(1024, 4),
            quantizer: FishQuantizerConfig {
                input_dim: 1024,
                n_codebooks: 9,
                codebook_size: 1024,
                semantic_codebook_size: 4096,
                codebook_dim: 8,
                downsample_factor: vec![2, 2],
                pre: FishTransformerConfig::quantizer_default(),
                post: FishTransformerConfig::quantizer_default(),
            },
        }
    }

    /// Parses a `config.json`-style object over the `build_ae`
    /// defaults (the checkpoint config carries the flat DAC fields;
    /// quantizer/transformer geometry stays at the defaults unless
    /// present).
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let mut config = Self::build_ae();
        let u = |field: &str| value.get(field).and_then(|v| v.as_u64());
        let vec_u = |field: &str| -> Option<Vec<usize>> {
            value.get(field).and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
        };
        if let Some(sr) = u("sample_rate") {
            config.sample_rate = sr as u32;
        }
        if let Some(d) = u("encoder_dim") {
            config.encoder_dim = d as usize;
        }
        if let Some(v) = vec_u("encoder_rates") {
            config.encoder_rates = v;
        }
        if let Some(d) = u("latent_dim") {
            config.latent_dim = d as usize;
        }
        if let Some(d) = u("decoder_dim") {
            config.decoder_dim = d as usize;
        }
        if let Some(v) = vec_u("decoder_rates") {
            config.decoder_rates = v;
        }
        if let Some(v) = vec_u("encoder_transformer_layers") {
            config.encoder_transformer_layers = v;
        }
        let transformer =
            |value: &serde_json::Value, default: FishTransformerConfig| -> FishTransformerConfig {
                let u = |field: &str| value.get(field).and_then(|v| v.as_u64());
                let f = |field: &str| value.get(field).and_then(|v| v.as_f64());
                FishTransformerConfig {
                    window_size: u("window_size").unwrap_or(default.window_size as u64) as usize,
                    n_layer: u("n_layer").unwrap_or(default.n_layer as u64) as usize,
                    n_head: u("n_head").unwrap_or(default.n_head as u64) as usize,
                    dim: u("dim").unwrap_or(default.dim as u64) as usize,
                    intermediate_size: u("intermediate_size")
                        .unwrap_or(default.intermediate_size as u64)
                        as usize,
                    head_dim: u("head_dim").unwrap_or(default.head_dim as u64) as usize,
                    rope_base: f("rope_base").unwrap_or(default.rope_base as f64) as f32,
                    norm_eps: f("norm_eps").unwrap_or(default.norm_eps as f64) as f32,
                }
            };
        if let Some(t) = value.get("encoder_transformer") {
            config.encoder_transformer = transformer(t, config.encoder_transformer.clone());
        }
        if let Some(q) = value.get("quantizer") {
            let qu = |field: &str| q.get(field).and_then(|v| v.as_u64());
            let vec_u_q = |field: &str| -> Option<Vec<usize>> {
                q.get(field).and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
            };
            if let Some(d) = qu("input_dim") {
                config.quantizer.input_dim = d as usize;
            }
            if let Some(d) = qu("n_codebooks") {
                config.quantizer.n_codebooks = d as usize;
            }
            if let Some(d) = qu("codebook_size") {
                config.quantizer.codebook_size = d as usize;
            }
            if let Some(d) = qu("semantic_codebook_size") {
                config.quantizer.semantic_codebook_size = d as usize;
            }
            if let Some(d) = qu("codebook_dim") {
                config.quantizer.codebook_dim = d as usize;
            }
            if let Some(v) = vec_u_q("downsample_factor") {
                config.quantizer.downsample_factor = v;
            }
            if let Some(t) = q.get("pre") {
                config.quantizer.pre = transformer(t, config.quantizer.pre.clone());
            }
            if let Some(t) = q.get("post") {
                config.quantizer.post = transformer(t, config.quantizer.post.clone());
            }
        }
        Ok(config)
    }

    /// Encoder downsampling product (the codec hop).
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    /// Samples per emitted token: hop times the quantizer's 2x2
    /// downsample (the reference `frame_length`).
    pub fn frame_length(&self) -> usize {
        self.hop_length() * self.quantizer.downsample_factor.iter().product::<usize>()
    }

    /// Total books: one semantic + `n_codebooks` acoustic.
    pub fn total_books(&self) -> usize {
        1 + self.quantizer.n_codebooks
    }
}

/// Weight-norm causal conv1d: torch-layout weight `(out, in, K)`,
/// `weight_g (out, 1, 1)` normalized over `(in, K)`, left pad
/// `kernel_eff - stride` plus the reference's right `extra` pad.
#[derive(Debug, Clone)]
struct FishWnConv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    /// Folded weight, PyTorch layout `[out, in/groups, K]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl FishWnConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let in_g = in_ch / groups;
        let v = load_f32_shaped(file, &format!("{prefix}.weight_v"), &[out_ch, in_g, kernel])?;
        let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[out_ch, 1, 1])?;
        let bias = if file.contains_tensor(&format!("{prefix}.bias")) {
            Some(load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?)
        } else {
            if with_bias {
                return Err(SpeechError::Tensor {
                    name: format!("{prefix}.bias"),
                    why: "required by the reference layer".to_string(),
                });
            }
            None
        };
        // normalize except dim 0: per out channel over (in, K).
        let mut weight = vec![0.0f32; v.len()];
        for (oc, &gain) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..in_g {
                for k in 0..kernel {
                    let value = v[oc * in_g * kernel + i * kernel + k];
                    acc += value * value;
                }
            }
            let norm = acc.sqrt();
            for i in 0..in_g {
                for k in 0..kernel {
                    weight[oc * in_g * kernel + i * kernel + k] =
                        gain * v[oc * in_g * kernel + i * kernel + k] / norm;
                }
            }
        }
        Ok(FishWnConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            dilation,
            groups,
            weight,
            bias,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
    }
}

/// Causal padding wrapper (`CausalConvNet` / `CausalWNConv1d`): left
/// pad `kernel_eff - stride`, right pad to a stride multiple.
#[derive(Debug, Clone)]
struct FishCausalConv {
    conv: FishWnConv1d,
    pad: usize,
}

impl FishCausalConv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        wn: bool,
    ) -> Result<Self> {
        let kernel_eff = (kernel - 1) * dilation + 1;
        let conv = if wn {
            FishWnConv1d::load(
                file, prefix, in_ch, out_ch, kernel, stride, dilation, groups, true,
            )?
        } else {
            // Plain torch-layout weight under `{prefix}.conv`.
            FishWnConv1d::load_plain(
                file,
                &format!("{prefix}.conv"),
                in_ch,
                out_ch,
                kernel,
                stride,
                dilation,
                groups,
                true,
            )?
        };
        Ok(FishCausalConv {
            conv,
            pad: kernel_eff - stride,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let extra = seq.div_ceil(self.conv.stride) * self.conv.stride - seq;
        let padded_seq = seq + self.pad + extra;
        let mut padded = vec![0.0f32; self.conv.in_ch * padded_seq];
        for c in 0..self.conv.in_ch {
            padded[c * padded_seq + self.pad..c * padded_seq + self.pad + seq]
                .copy_from_slice(&x[c * seq..(c + 1) * seq]);
        }
        let _ = padded_seq;
        self.conv.forward(&padded)
    }
}

impl FishWnConv1d {
    /// Plain (non-weight-normed) torch-layout conv under `{prefix}`,
    /// stored `(out, in, K)` + bias.
    #[allow(clippy::too_many_arguments)]
    fn load_plain(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let in_g = in_ch / groups;
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, in_g, kernel])?;
        let mut weight = vec![0.0f32; stored.len()];
        for oc in 0..out_ch {
            for i in 0..in_g {
                for k in 0..kernel {
                    weight[oc * in_g * kernel + i * kernel + k] =
                        stored[oc * in_g * kernel + i * kernel + k];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            if with_bias {
                return Err(SpeechError::Tensor {
                    name: bias_name,
                    why: "required by the reference layer".to_string(),
                });
            }
            None
        };
        Ok(FishWnConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            dilation,
            groups,
            weight,
            bias,
        })
    }
}

/// Weight-norm causal transposed conv: stored `(in, out, K)` with
/// `weight_g (in, 1, 1)` normalized over `(out, K)`; convtr at pad 0
/// then trim `(K - stride)` from the right.
#[derive(Debug, Clone)]
struct FishCausalConvTr {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl FishCausalConvTr {
    /// The quantizer's upsample uses a plain `ConvTranspose1dTorch`
    /// (torch-layout `(in, out, K)` under `.conv`); the decoder's
    /// `CausalWNConvTranspose1d` is weight-normed with `weight_g
    /// (in, 1, 1)` normalized over `(out, K)` directly on the prefix.
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        wn: bool,
    ) -> Result<Self> {
        let (weight, bias) = if wn {
            let v = load_f32_shaped(
                file,
                &format!("{prefix}.weight_v"),
                &[in_ch, out_ch, kernel],
            )?;
            let g = load_f32_shaped(file, &format!("{prefix}.weight_g"), &[in_ch, 1, 1])?;
            let bias = load_f32_shaped(file, &format!("{prefix}.bias"), &[out_ch])?;
            let mut weight = vec![0.0f32; v.len()];
            for (ic, &gain) in g.iter().enumerate() {
                let mut acc = 0.0f32;
                for o in 0..out_ch {
                    for k in 0..kernel {
                        let value = v[ic * out_ch * kernel + o * kernel + k];
                        acc += value * value;
                    }
                }
                let norm = acc.sqrt();
                for o in 0..out_ch {
                    for k in 0..kernel {
                        weight[ic * out_ch * kernel + o * kernel + k] =
                            gain * v[ic * out_ch * kernel + o * kernel + k] / norm;
                    }
                }
            }
            (weight, Some(bias))
        } else {
            let stored = load_f32_shaped(
                file,
                &format!("{prefix}.conv.weight"),
                &[in_ch, out_ch, kernel],
            )?;
            let bias_name = format!("{prefix}.conv.bias");
            let bias = if file.contains_tensor(&bias_name) {
                Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
            } else {
                None
            };
            (stored, bias)
        };
        Ok(FishCausalConvTr {
            in_ch,
            out_ch,
            kernel,
            stride,
            weight,
            bias,
        })
    }

    /// Channel-major in -> channel-major out, right-trimmed.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let y = ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            0,
            0,
            1,
        );
        let out_seq = y.len() / self.out_ch;
        let trim_right = self.kernel - self.stride;
        let keep = out_seq - trim_right;
        let mut trimmed = vec![0.0f32; keep * self.out_ch];
        for c in 0..self.out_ch {
            trimmed[c * keep..(c + 1) * keep].copy_from_slice(&y[c * out_seq..c * out_seq + keep]);
        }
        trimmed
    }
}

/// Fish Snake1d: alpha stored `(1, C, 1)`.
fn load_fish_alpha(file: &SafetensorsFile, prefix: &str, channels: usize) -> Result<Vec<f32>> {
    let stored = load_f32_shaped(file, &format!("{prefix}.alpha"), &[1, channels, 1])?;
    Ok(stored)
}

/// `ConvNeXtBlock`: causal depthwise conv, LayerNorm (eps 1e-6),
/// pwconv1, exact GELU, pwconv2, learned gamma, residual. The norm and
/// pointwise layers run channels-last.
#[derive(Debug, Clone)]
struct FishConvNeXtBlock {
    dwconv: FishCausalConv,
    norm: (Vec<f32>, Vec<f32>),
    pwconv1: (usize, usize, Vec<f32>, Vec<f32>),
    pwconv2: (usize, usize, Vec<f32>, Vec<f32>),
    gamma: Vec<f32>,
    dim: usize,
}

impl FishConvNeXtBlock {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, kernel: usize) -> Result<Self> {
        let linear = |name: &str,
                      in_dim: usize,
                      out_dim: usize|
         -> Result<(usize, usize, Vec<f32>, Vec<f32>)> {
            Ok((
                in_dim,
                out_dim,
                load_f32_shaped(file, &format!("{prefix}.{name}.weight"), &[out_dim, in_dim])?,
                load_f32_shaped(file, &format!("{prefix}.{name}.bias"), &[out_dim])?,
            ))
        };
        Ok(FishConvNeXtBlock {
            dwconv: FishCausalConv::load(
                file,
                &format!("{prefix}.dwconv"),
                dim,
                dim,
                kernel,
                1,
                1,
                dim,
                false,
            )?,
            norm: (
                load_f32_shaped(file, &format!("{prefix}.norm.weight"), &[dim])?,
                load_f32_shaped(file, &format!("{prefix}.norm.bias"), &[dim])?,
            ),
            pwconv1: linear("pwconv1", dim, dim * 4)?,
            pwconv2: linear("pwconv2", dim * 4, dim)?,
            gamma: load_f32_shaped(file, &format!("{prefix}.gamma"), &[dim])?,
            dim,
        })
    }

    fn forward(&self, x: &mut Vec<f32>, seq: usize) {
        let dim = self.dim;
        let residual = x.clone();
        // Causal depthwise conv on channel-major.
        let padded_input = x.clone();
        let mut h = self.dwconv.forward(&padded_input, seq);
        // Norm + FFN channels-last.
        let mut rows = vec![0.0f32; h.len()];
        for t in 0..seq {
            for d in 0..dim {
                rows[t * dim + d] = h[d * seq + t];
            }
        }
        ops::layernorm(&mut rows, seq, dim, &self.norm.0, Some(&self.norm.1), 1e-6);
        let (in1, out1, w1, b1) = &self.pwconv1;
        let mut f = ops::linear(&rows, w1, Some(b1), seq, *in1, *out1);
        ops::gelu_erf(&mut f);
        let (in2, out2, w2, b2) = &self.pwconv2;
        let mut f = ops::linear(&f, w2, Some(b2), seq, *in2, *out2);
        for chunk in f.chunks_exact_mut(dim) {
            for (d, v) in chunk.iter_mut().enumerate() {
                *v *= self.gamma[d];
            }
        }
        // Back to channel-major and residual.
        for t in 0..seq {
            for d in 0..dim {
                h[d * seq + t] = residual[d * seq + t] + f[t * dim + d];
            }
        }
        *x = h;
    }
}

/// RMSNorm (weight applied after, eps from config).
#[derive(Debug, Clone)]
struct FishRms {
    weight: Vec<f32>,
    eps: f32,
}

impl FishRms {
    fn apply(&self, x: &mut [f32], rows: usize, dim: usize) {
        ops::rmsnorm(x, rows, dim, &self.weight, self.eps);
    }
}

/// Pre-norm transformer block: RMSNorm attention with learned
/// LayerScale, RMSNorm SwiGLU FFN with learned LayerScale.
#[derive(Debug, Clone)]
struct FishBlock {
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    dim: usize,
    n_head: usize,
    n_local: usize,
    head_dim: usize,
    attention_norm: FishRms,
    ffn_norm: FishRms,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    ffn_dim: usize,
    attention_gamma: Vec<f32>,
    ffn_gamma: Vec<f32>,
}

impl FishBlock {
    fn load(file: &SafetensorsFile, prefix: &str, config: &FishTransformerConfig) -> Result<Self> {
        let total = (config.n_head + 2 * config.n_head) * config.head_dim;
        Ok(FishBlock {
            wqkv: load_f32_shaped(
                file,
                &format!("{prefix}.attention.wqkv.weight"),
                &[total, config.dim],
            )?,
            wo: load_f32_shaped(
                file,
                &format!("{prefix}.attention.wo.weight"),
                &[config.dim, config.n_head * config.head_dim],
            )?,
            dim: config.dim,
            n_head: config.n_head,
            n_local: config.n_head,
            head_dim: config.head_dim,
            attention_norm: FishRms {
                weight: load_f32_shaped(
                    file,
                    &format!("{prefix}.attention_norm.weight"),
                    &[config.dim],
                )?,
                eps: config.norm_eps,
            },
            ffn_norm: FishRms {
                weight: load_f32_shaped(file, &format!("{prefix}.ffn_norm.weight"), &[config.dim])?,
                eps: config.norm_eps,
            },
            w1: load_f32_shaped(
                file,
                &format!("{prefix}.feed_forward.w1.weight"),
                &[config.intermediate_size, config.dim],
            )?,
            w2: load_f32_shaped(
                file,
                &format!("{prefix}.feed_forward.w2.weight"),
                &[config.dim, config.intermediate_size],
            )?,
            w3: load_f32_shaped(
                file,
                &format!("{prefix}.feed_forward.w3.weight"),
                &[config.intermediate_size, config.dim],
            )?,
            ffn_dim: config.intermediate_size,
            attention_gamma: load_f32_shaped(
                file,
                &format!("{prefix}.attention_layer_scale.gamma"),
                &[config.dim],
            )?,
            ffn_gamma: load_f32_shaped(
                file,
                &format!("{prefix}.ffn_layer_scale.gamma"),
                &[config.dim],
            )?,
        })
    }

    fn forward(&self, x: &mut [f32], seq: usize, mask: &[f32], rope_cos: &[f32], rope_sin: &[f32]) {
        let dim = self.dim;
        let mut normed = x.to_vec();
        self.attention_norm.apply(&mut normed, seq, dim);
        let qkv = ops::linear(&normed, &self.wqkv, None, seq, dim, 3 * dim);
        let kv_size = self.n_local * self.head_dim;
        let mut qh = vec![0.0f32; self.n_head * seq * self.head_dim];
        let mut kh = vec![0.0f32; self.n_local * seq * self.head_dim];
        let mut vh = kh.clone();
        for t in 0..seq {
            for h in 0..self.n_head {
                for d in 0..self.head_dim {
                    qh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + h * self.head_dim + d];
                }
            }
            for h in 0..self.n_local {
                for d in 0..self.head_dim {
                    kh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + kv_size + h * self.head_dim + d];
                    vh[(h * seq + t) * self.head_dim + d] =
                        qkv[t * 3 * dim + 2 * kv_size + h * self.head_dim + d];
                }
            }
        }
        // Interleaved pairs (x[2i], x[2i+1]) per apply_rotary_emb's
        // (..., dim/2, 2) reshape.
        let half = self.head_dim / 2;
        for plane in [&mut qh, &mut kh] {
            for h in 0..plane.len() / (seq * self.head_dim) {
                for t in 0..seq {
                    let base = (h * seq + t) * self.head_dim;
                    for d in 0..half {
                        let a = plane[base + 2 * d];
                        let b = plane[base + 2 * d + 1];
                        let c = rope_cos[t * half + d];
                        let s = rope_sin[t * half + d];
                        plane[base + 2 * d] = a * c - b * s;
                        plane[base + 2 * d + 1] = b * c + a * s;
                    }
                }
            }
        }
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; qh.len()];
        for h in 0..self.n_head {
            let plane = h * seq * self.head_dim;
            let o = ops::sdpa(
                &qh[plane..plane + seq * self.head_dim],
                &kh[plane..plane + seq * self.head_dim],
                &vh[plane..plane + seq * self.head_dim],
                Some(mask),
                seq,
                seq,
                self.head_dim,
                self.head_dim,
                scale,
            );
            out[plane..plane + o.len()].copy_from_slice(&o);
        }
        let mut merged = vec![0.0f32; seq * dim];
        for h in 0..self.n_head {
            for t in 0..seq {
                let src = (h * seq + t) * self.head_dim;
                merged[t * dim + h * self.head_dim..t * dim + (h + 1) * self.head_dim]
                    .copy_from_slice(&out[src..src + self.head_dim]);
            }
        }
        let attn = ops::linear(&merged, &self.wo, None, seq, dim, dim);
        for (x_chunk, a_chunk) in x.chunks_exact_mut(dim).zip(attn.chunks_exact(dim)) {
            for (d, v) in x_chunk.iter_mut().enumerate() {
                *v += a_chunk[d] * self.attention_gamma[d];
            }
        }
        let mut normed = x.to_vec();
        self.ffn_norm.apply(&mut normed, seq, dim);
        let h1 = ops::linear(&normed, &self.w1, None, seq, dim, self.ffn_dim);
        let h3 = ops::linear(&normed, &self.w3, None, seq, dim, self.ffn_dim);
        let mut gated = vec![0.0f32; h1.len()];
        for ((g, &a), &b) in gated.iter_mut().zip(&h1).zip(&h3) {
            // silu(a) * b, elementwise (crate silu works in place).
            let mut tmp = [a];
            ops::silu(&mut tmp);
            *g = tmp[0] * b;
        }
        let ffn = ops::linear(&gated, &self.w2, None, seq, self.ffn_dim, dim);
        for (x_chunk, f_chunk) in x.chunks_exact_mut(dim).zip(ffn.chunks_exact(dim)) {
            for (d, v) in x_chunk.iter_mut().enumerate() {
                *v += f_chunk[d] * self.ffn_gamma[d];
            }
        }
    }
}

/// WindowLimitedTransformer over rows `[seq, input_dim]`.
#[derive(Debug, Clone)]
struct FishTransformer {
    blocks: Vec<FishBlock>,
    final_norm: FishRms,
    input_proj: Option<(usize, usize, Vec<f32>, Vec<f32>)>,
    output_proj: Option<(usize, usize, Vec<f32>, Vec<f32>)>,
    window_size: Option<usize>,
    dim: usize,
    head_dim: usize,
    rope_base: f32,
}

impl FishTransformer {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        config: &FishTransformerConfig,
        input_dim: usize,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(FishBlock::load(
                file,
                &format!("{prefix}.layers.{i}"),
                config,
            )?);
        }
        let input_proj = if input_dim != config.dim {
            Some((
                input_dim,
                config.dim,
                load_f32_shaped(
                    file,
                    &format!("{prefix}.input_proj.weight"),
                    &[config.dim, input_dim],
                )?,
                load_f32_shaped(file, &format!("{prefix}.input_proj.bias"), &[config.dim])?,
            ))
        } else {
            None
        };
        let output_proj = if input_dim != config.dim {
            Some((
                config.dim,
                input_dim,
                load_f32_shaped(
                    file,
                    &format!("{prefix}.output_proj.weight"),
                    &[input_dim, config.dim],
                )?,
                load_f32_shaped(file, &format!("{prefix}.output_proj.bias"), &[input_dim])?,
            ))
        } else {
            None
        };
        Ok(FishTransformer {
            blocks,
            final_norm: FishRms {
                weight: load_f32_shaped(file, &format!("{prefix}.norm.weight"), &[config.dim])?,
                eps: config.norm_eps,
            },
            input_proj,
            output_proj,
            window_size: Some(config.window_size),
            dim: config.dim,
            head_dim: config.head_dim,
            rope_base: config.rope_base,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        // Input projection (rows already).
        let mut h = match &self.input_proj {
            Some((i, o, w, b)) => ops::linear(x, w, Some(b), seq, *i, *o),
            None => x.to_vec(),
        };
        let (rope_cos, rope_sin) = ops::rope_tables(seq, self.head_dim, self.rope_base);
        // Window-limited causal additive mask: the window reaches back
        // `window` frames, so j is valid when
        // max(i - window + 1, 0) <= j <= i.
        let window = self.window_size.unwrap_or(seq);
        let mut mask = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let valid = j <= i && i < j + window;
                if !valid {
                    mask[i * seq + j] = -1.0e9;
                }
            }
        }
        for block in &self.blocks {
            block.forward(&mut h, seq, &mask, &rope_cos, &rope_sin);
        }
        self.final_norm.apply(&mut h, seq, self.dim);
        match &self.output_proj {
            Some((i, o, w, b)) => ops::linear(&h, w, Some(b), seq, *i, *o),
            None => h,
        }
    }
}

/// One RVQ book: k=1 WN convs in/out plus the Euclidean codebook over
/// L2-normalized vectors (floor 1e-12), emitting raw rows.
#[derive(Debug, Clone)]
struct FishVq {
    /// Folded k=1 conv weight `[codebook_dim, input_dim]` (row-major
    /// per output channel) and bias.
    in_proj_w: Vec<f32>,
    in_proj_b: Vec<f32>,
    out_proj_w: Vec<f32>,
    out_proj_b: Vec<f32>,
    codebook: Vec<f32>,
    codebook_norm: Vec<f32>,
    codebook_size: usize,
    codebook_dim: usize,
}

impl FishVq {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input_dim: usize,
        size: usize,
        dim: usize,
    ) -> Result<Self> {
        // in_proj/out_proj are WNConv1d k=1 with torch-layout weights.
        let v_in = load_f32_shaped(
            file,
            &format!("{prefix}.in_proj.weight_v"),
            &[dim, input_dim, 1],
        )?;
        let g_in = load_f32_shaped(file, &format!("{prefix}.in_proj.weight_g"), &[dim, 1, 1])?;
        let b_in = load_f32_shaped(file, &format!("{prefix}.in_proj.bias"), &[dim])?;
        let v_out = load_f32_shaped(
            file,
            &format!("{prefix}.out_proj.weight_v"),
            &[input_dim, dim, 1],
        )?;
        let g_out = load_f32_shaped(
            file,
            &format!("{prefix}.out_proj.weight_g"),
            &[input_dim, 1, 1],
        )?;
        let b_out = load_f32_shaped(file, &format!("{prefix}.out_proj.bias"), &[input_dim])?;
        let fold = |v: &[f32], g: &[f32]| -> Vec<f32> {
            let out_ch = g.len();
            let mut w = vec![0.0f32; v.len()];
            for (oc, &gain) in g.iter().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..v.len() / out_ch {
                    acc += v[oc * (v.len() / out_ch) + i] * v[oc * (v.len() / out_ch) + i];
                }
                let norm = acc.sqrt();
                for i in 0..v.len() / out_ch {
                    w[oc * (v.len() / out_ch) + i] = gain * v[oc * (v.len() / out_ch) + i] / norm;
                }
            }
            w
        };
        let in_w = fold(&v_in, &g_in);
        let out_w = fold(&v_out, &g_out);
        let codebook = load_f32_shaped(file, &format!("{prefix}.codebook.weight"), &[size, dim])?;
        let mut codebook_norm = vec![0.0f32; size];
        for (r, norm) in codebook_norm.iter_mut().enumerate() {
            let row = &codebook[r * dim..(r + 1) * dim];
            *norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        }
        Ok(FishVq {
            in_proj_w: in_w,
            in_proj_b: b_in,
            out_proj_w: out_w,
            out_proj_b: b_out,
            codebook,
            codebook_norm,
            codebook_size: size,
            codebook_dim: dim,
        })
    }

    /// Channel-major `[in_dim, T]` -> projected `[out_dim, T]` (k=1
    /// conv over channel-major input).
    fn proj(
        &self,
        w: &[f32],
        b: &[f32],
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        frames: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim * frames];
        for t in 0..frames {
            for o in 0..out_dim {
                let mut acc = b[o];
                let row = &w[o * in_dim..(o + 1) * in_dim];
                for i in 0..in_dim {
                    acc += x[i * frames + t] * row[i];
                }
                out[o * frames + t] = acc;
            }
        }
        out
    }

    /// Latents `[codebook_dim, T]` -> raw rows + indices.
    fn decode_latents(&self, latents: &[f32], frames: usize) -> (Vec<f32>, Vec<i32>) {
        let mut indices = vec![0i32; frames];
        let mut zq = vec![0.0f32; self.codebook_dim * frames];
        for t in 0..frames {
            let mut col = vec![0.0f32; self.codebook_dim];
            for (d, slot) in col.iter_mut().enumerate() {
                *slot = latents[d * frames + t];
            }
            let norm = col.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for c in 0..self.codebook_size {
                let row = &self.codebook[c * self.codebook_dim..(c + 1) * self.codebook_dim];
                let mut dot = 0.0f32;
                for (a, &b) in col.iter().zip(row) {
                    dot += (a / norm) * (b / self.codebook_norm[c]);
                }
                if dot > best {
                    best = dot;
                    best_idx = c;
                }
            }
            indices[t] = best_idx as i32;
            for d in 0..self.codebook_dim {
                zq[d * frames + t] = self.codebook[best_idx * self.codebook_dim + d];
            }
        }
        (zq, indices)
    }

    /// Codes -> raw rows channel-major.
    fn decode_code(&self, codes: &[i32], frames: usize) -> Result<Vec<f32>> {
        let mut zq = vec![0.0f32; self.codebook_dim * frames];
        for (t, &c) in codes.iter().enumerate() {
            let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                why: format!("negative code {c}"),
            })?;
            if idx >= self.codebook_size {
                return Err(SpeechError::Input {
                    why: format!("code {c} outside the codebook of {}", self.codebook_size),
                });
            }
            for d in 0..self.codebook_dim {
                zq[d * frames + t] = self.codebook[idx * self.codebook_dim + d];
            }
        }
        Ok(zq)
    }
}

/// Greedy residual stack of books.
#[derive(Debug, Clone)]
struct FishRvq {
    quantizers: Vec<FishVq>,
    input_dim: usize,
}

impl FishRvq {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        n_codebooks: usize,
        input_dim: usize,
        size: usize,
        dim: usize,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(n_codebooks);
        for i in 0..n_codebooks {
            quantizers.push(FishVq::load(
                file,
                &format!("{prefix}.quantizers.{i}"),
                input_dim,
                size,
                dim,
            )?);
        }
        Ok(FishRvq {
            quantizers,
            input_dim,
        })
    }

    /// Codes per book -> quantized channel-major (sum of out_proj of
    /// the raw rows).
    #[allow(clippy::wrong_self_convention)]
    fn from_codes(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        let mut z_q = vec![0.0f32; frames * self.input_dim];
        for (q, book_codes) in codes.iter().enumerate() {
            let book = &self.quantizers[q];
            let raw = book.decode_code(book_codes, frames)?;
            let projected = book.proj(
                &book.out_proj_w,
                &book.out_proj_b,
                book.codebook_dim,
                self.input_dim,
                &raw,
                frames,
            );
            for (v, p) in z_q.iter_mut().zip(projected) {
                *v += p;
            }
        }
        Ok(z_q)
    }
}

/// The downsampled dual quantizer with pre/post transformers.
#[derive(Debug, Clone)]
struct FishDownsampleRvq {
    downsample: Vec<(FishCausalConv, FishConvNeXtBlock)>,
    upsample: Vec<(FishCausalConvTr, FishConvNeXtBlock)>,
    pre: FishTransformer,
    post: FishTransformer,
    semantic: FishRvq,
    residual: FishRvq,
}

impl FishDownsampleRvq {
    fn load(config: &FishS1DacConfig, file: &SafetensorsFile) -> Result<Self> {
        let q = &config.quantizer;
        let mut dims = vec![q.input_dim];
        for _ in q.downsample_factor.iter() {
            dims.push(q.input_dim);
        }
        let mut downsample = Vec::new();
        for (idx, &factor) in q.downsample_factor.iter().enumerate() {
            let conv = FishCausalConv::load(
                file,
                &format!("quantizer.downsample.{idx}.0"),
                dims[idx],
                dims[idx + 1],
                factor,
                factor,
                1,
                1,
                false,
            )?;
            let block = FishConvNeXtBlock::load(
                file,
                &format!("quantizer.downsample.{idx}.1"),
                dims[idx + 1],
                7,
            )?;
            downsample.push((conv, block));
        }
        // Upsample mirrors the downsample list in reverse.
        let mut upsample = Vec::new();
        for (u_idx, (rev_idx, &factor)) in q.downsample_factor.iter().enumerate().rev().enumerate()
        {
            let conv = FishCausalConvTr::load(
                file,
                &format!("quantizer.upsample.{u_idx}.0"),
                dims[rev_idx + 1],
                dims[rev_idx],
                factor,
                factor,
                false,
            )?;
            let block = FishConvNeXtBlock::load(
                file,
                &format!("quantizer.upsample.{u_idx}.1"),
                dims[rev_idx],
                7,
            )?;
            upsample.push((conv, block));
        }
        Ok(FishDownsampleRvq {
            downsample,
            upsample,
            pre: FishTransformer::load(file, "quantizer.pre_module", &q.pre, q.input_dim)?,
            post: FishTransformer::load(file, "quantizer.post_module", &q.post, q.input_dim)?,
            semantic: FishRvq::load(
                file,
                "quantizer.semantic_quantizer",
                1,
                q.input_dim,
                q.semantic_codebook_size,
                q.codebook_dim,
            )?,
            residual: FishRvq::load(
                file,
                "quantizer.quantizer",
                q.n_codebooks,
                q.input_dim,
                q.codebook_size,
                q.codebook_dim,
            )?,
        })
    }

    /// Codes per book (semantic first) -> channel-major audio latents
    /// at the encoder frame rate, upsampled by the downsample factor.
    fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<(Vec<f32>, usize)> {
        if codes.is_empty() {
            return Err(SpeechError::Input {
                why: "codes must carry at least the semantic book".to_string(),
            });
        }
        let semantic = self.semantic.from_codes(&codes[..1], frames)?;
        let z_q = if codes.len() > 1 {
            self.residual.from_codes(&codes[1..], frames)?
        } else {
            vec![0.0f32; semantic.len()]
        };
        let mut z = vec![0.0f32; z_q.len()];
        for (v, pair) in z.iter_mut().zip(semantic.iter().zip(z_q.iter())) {
            *v = pair.0 + pair.1;
        }
        let in_dim = self.semantic.input_dim;
        let mut seq = frames;
        let mut rows = vec![0.0f32; z.len()];
        for t in 0..seq {
            for d in 0..in_dim {
                rows[t * in_dim + d] = z[d * seq + t];
            }
        }
        let rows = self.post.forward(&rows, seq);
        for t in 0..seq {
            for d in 0..in_dim {
                z[d * seq + t] = rows[t * in_dim + d];
            }
        }
        for (conv, block) in &self.upsample {
            z = conv.forward(&z);
            seq = z.len() / conv.out_ch;
            block.forward(&mut z, seq);
        }
        Ok((z, seq))
    }
}

/// Loaded Fish S1 DAC, batch-of-one.
pub struct FishS1Dac {
    pub config: FishS1DacConfig,
    encoder_conv_in: FishCausalConv,
    encoder_blocks: Vec<FishEncoderBlock>,
    encoder_alpha: Vec<f32>,
    encoder_conv_out: FishCausalConv,
    encoder_out_ch: usize,
    quantizer: FishDownsampleRvq,
    decoder_conv_in: FishCausalConv,
    decoder_blocks: Vec<FishDecoderBlock>,
    decoder_alpha: Vec<f32>,
    decoder_conv_out: FishCausalConv,
}

#[derive(Debug, Clone)]
struct FishEncoderBlock {
    res_units: [FishResidualUnit; 3],
    alpha: Vec<f32>,
    conv: FishCausalConv,
    transformer: Option<FishTransformer>,
    in_ch: usize,
}

#[derive(Debug, Clone)]
struct FishResidualUnit {
    alpha1: Vec<f32>,
    conv1: FishCausalConv,
    alpha2: Vec<f32>,
    conv2: FishCausalConv,
    channels: usize,
    causal: bool,
}

impl FishResidualUnit {
    #[allow(clippy::too_many_arguments)]
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, dilation: usize) -> Result<Self> {
        Ok(FishResidualUnit {
            alpha1: load_fish_alpha(file, &format!("{prefix}.block.0"), dim)?,
            conv1: FishCausalConv::load(
                file,
                &format!("{prefix}.block.1"),
                dim,
                dim,
                7,
                1,
                dilation,
                1,
                true,
            )?,
            alpha2: load_fish_alpha(file, &format!("{prefix}.block.2"), dim)?,
            conv2: FishCausalConv::load(
                file,
                &format!("{prefix}.block.3"),
                dim,
                dim,
                1,
                1,
                1,
                1,
                true,
            )?,
            channels: dim,
            causal: true,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let mut y = x.to_vec();
        snake1d(&mut y, &self.alpha1, self.channels, seq);
        let y = self.conv1.forward(&y, seq);
        let seq2 = y.len() / self.channels;
        let mut y = y;
        snake1d(&mut y, &self.alpha2, self.channels, seq2);
        let y = self.conv2.forward(&y, seq2);
        let out_seq = y.len() / self.channels;
        // causal: crop the residual's tail; symmetric otherwise.
        let pad = seq.saturating_sub(out_seq);
        let mut out = vec![0.0f32; out_seq * self.channels];
        for c in 0..self.channels {
            for t in 0..out_seq {
                let r = if self.causal {
                    if t < seq {
                        x[c * seq + t]
                    } else {
                        0.0
                    }
                } else {
                    let start = pad / 2;
                    if t + start < seq {
                        x[c * seq + t + start]
                    } else {
                        0.0
                    }
                };
                out[c * out_seq + t] = r + y[c * out_seq + t];
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct FishDecoderBlock {
    alpha: Vec<f32>,
    conv_tr: FishCausalConvTr,
    res_units: [FishResidualUnit; 3],
    in_ch: usize,
}

impl FishDecoderBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(FishDecoderBlock {
            alpha: load_fish_alpha(file, &format!("{prefix}.block.0"), in_dim)?,
            conv_tr: FishCausalConvTr::load(
                file,
                &format!("{prefix}.block.1"),
                in_dim,
                out_dim,
                2 * stride,
                stride,
                true,
            )?,
            res_units: [
                FishResidualUnit::load(file, &format!("{prefix}.block.2"), out_dim, 1)?,
                FishResidualUnit::load(file, &format!("{prefix}.block.3"), out_dim, 3)?,
                FishResidualUnit::load(file, &format!("{prefix}.block.4"), out_dim, 9)?,
            ],
            in_ch: in_dim,
        })
    }
}

impl FishS1Dac {
    /// Opens a checkpoint directory (`config.json` optional; the
    /// geometry comes from `build_ae()` unless the config overrides
    /// the flat fields).
    pub fn open(dir: &Path) -> Result<Self> {
        let file =
            SafetensorsFile::open(&dir.join("model.safetensors")).map_err(SpeechError::from)?;
        let mut config = FishS1DacConfig::build_ae();
        if let Ok(text) = std::fs::read_to_string(dir.join("config.json")) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                config = FishS1DacConfig::from_json(&value)?;
            }
        }
        Self::load(config, &file)
    }

    pub fn load(config: FishS1DacConfig, file: &SafetensorsFile) -> Result<Self> {
        if config.encoder_rates.len() != config.encoder_transformer_layers.len() {
            return Err(SpeechError::BadConfig {
                field: "encoder_transformer_layers".to_string(),
                why: "must match encoder_rates length".to_string(),
            });
        }
        let d = config.encoder_dim;
        let encoder_conv_in =
            FishCausalConv::load(file, "encoder.block.0", 1, d, 7, 1, 1, 1, true)?;
        let mut encoder_blocks = Vec::new();
        let mut channels = d;
        for (i, &stride) in config.encoder_rates.iter().enumerate() {
            channels *= 2;
            let half = channels / 2;
            let base = format!("encoder.block.{}", i + 1);
            let transformer = if config.encoder_transformer_layers[i] > 0 {
                let t_config = FishTransformerConfig {
                    n_layer: config.encoder_transformer_layers[i],
                    ..config.encoder_transformer.clone()
                };
                Some(FishTransformer::load(
                    file,
                    &format!("{base}.block.5"),
                    &t_config,
                    channels,
                )?)
            } else {
                None
            };
            encoder_blocks.push(FishEncoderBlock {
                res_units: [
                    FishResidualUnit::load(file, &format!("{base}.block.0"), half, 1)?,
                    FishResidualUnit::load(file, &format!("{base}.block.1"), half, 3)?,
                    FishResidualUnit::load(file, &format!("{base}.block.2"), half, 9)?,
                ],
                alpha: load_fish_alpha(file, &format!("{base}.block.3"), half)?,
                conv: FishCausalConv::load(
                    file,
                    &format!("{base}.block.4"),
                    half,
                    channels,
                    2 * stride,
                    stride,
                    1,
                    1,
                    true,
                )?,
                transformer,
                in_ch: half,
            });
        }
        let encoder_alpha = load_fish_alpha(
            file,
            &format!("encoder.block.{}", config.encoder_rates.len() + 1),
            channels,
        )?;
        let encoder_conv_out = FishCausalConv::load(
            file,
            &format!("encoder.block.{}", config.encoder_rates.len() + 2),
            channels,
            config.latent_dim,
            3,
            1,
            1,
            1,
            true,
        )?;
        let encoder_out_ch = config.latent_dim;

        let quantizer = FishDownsampleRvq::load(&config, file)?;

        let dec = config.decoder_dim;
        let decoder_conv_in = FishCausalConv::load(
            file,
            "decoder.model.0",
            config.latent_dim,
            dec,
            7,
            1,
            1,
            1,
            true,
        )?;
        let mut decoder_blocks = Vec::new();
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let in_dim = dec >> i;
            let out_dim = dec >> (i + 1);
            decoder_blocks.push(FishDecoderBlock::load(
                file,
                &format!("decoder.model.{}", i + 1),
                in_dim,
                out_dim,
                stride,
            )?);
        }
        let final_dim = dec >> config.decoder_rates.len();
        // Tail: model.{n+1} Snake, model.{n+2} conv, model.{n+3} tanh.
        let n = config.decoder_rates.len();
        let decoder_alpha = load_fish_alpha(file, &format!("decoder.model.{}", n + 1), final_dim)?;
        let decoder_conv_out = FishCausalConv::load(
            file,
            &format!("decoder.model.{}", n + 2),
            final_dim,
            1,
            7,
            1,
            1,
            1,
            true,
        )?;
        Ok(FishS1Dac {
            config,
            encoder_conv_in,
            encoder_blocks,
            encoder_alpha,
            encoder_conv_out,
            encoder_out_ch,
            quantizer,
            decoder_conv_in,
            decoder_blocks,
            decoder_alpha,
            decoder_conv_out,
        })
    }

    /// Mono samples -> codes per book plus the code length. Pads to a
    /// frame-length multiple exactly like the reference `encode`.
    pub fn encode(&self, samples: &[f32]) -> Result<(Vec<Vec<i32>>, usize)> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        let frame_length = self.config.frame_length();
        let padded = samples.len().div_ceil(frame_length) * frame_length;
        let mut x = vec![0.0f32; padded];
        x[..samples.len()].copy_from_slice(samples);
        let seq = padded;
        let mut h = self.encoder_conv_in.forward(&x, seq);
        for block in &self.encoder_blocks {
            let half = block.in_ch;
            for ru in &block.res_units {
                let seq_in = h.len() / half;
                h = ru.forward(&h, seq_in);
            }
            let seq_in = h.len() / half;
            let mut h2 = h.clone();
            snake1d(&mut h2, &block.alpha, half, seq_in);
            h = block.conv.forward(&h2, seq_in);
            let seq_t = h.len() / block.conv.conv.out_ch;
            if let Some(transformer) = &block.transformer {
                let channels = block.conv.conv.out_ch;
                let mut rows = vec![0.0f32; h.len()];
                for t in 0..seq_t {
                    for c in 0..channels {
                        rows[t * channels + c] = h[c * seq_t + t];
                    }
                }
                let out = transformer.forward(&rows, seq_t);
                for t in 0..seq_t {
                    for c in 0..channels {
                        h[c * seq_t + t] = out[t * channels + c];
                    }
                }
            }
        }
        let seq = h.len() / self.encoder_out_ch;
        snake1d(&mut h, &self.encoder_alpha, self.encoder_out_ch, seq);
        let z = self.encoder_conv_out.forward(&h, seq);
        let frames = z.len() / self.config.quantizer.input_dim;
        let (_quantized, codes) = self.quantizer_residual_encode(&z, frames);
        let code_len = codes.first().map_or(0, |c| c.len());
        Ok((codes, code_len))
    }

    fn quantizer_residual_encode(&self, z: &[f32], frames: usize) -> (Vec<f32>, Vec<Vec<i32>>) {
        // Downsample stack.
        let mut z = z.to_vec();
        let mut seq = frames;
        let mut dims = vec![self.config.quantizer.input_dim];
        for _ in &self.config.quantizer.downsample_factor {
            dims.push(self.config.quantizer.input_dim);
        }
        for (idx, (conv, block)) in self.quantizer.downsample.iter().enumerate() {
            let in_dim = dims[idx];
            seq = z.len() / in_dim.max(1);
            z = conv.forward(&z, seq);
            let out_dim = dims[idx + 1];
            seq = z.len() / out_dim;
            block.forward(&mut z, seq);
        }
        // Pre transformer (rows round trip).
        let in_dim = self.config.quantizer.input_dim;
        let to_rows = |cm: &[f32], seq: usize, dim: usize| {
            let mut rows = vec![0.0f32; cm.len()];
            for t in 0..seq {
                for d in 0..dim {
                    rows[t * dim + d] = cm[d * seq + t];
                }
            }
            rows
        };
        let to_cm = |rows: &[f32], seq: usize, dim: usize| {
            let mut cm = vec![0.0f32; rows.len()];
            for t in 0..seq {
                for d in 0..dim {
                    cm[d * seq + t] = rows[t * dim + d];
                }
            }
            cm
        };
        let rows = to_rows(&z, seq, in_dim);
        let rows = self.quantizer.pre.forward(&rows, seq);
        let z = to_cm(&rows, seq, in_dim);
        // Semantic book then greedy residual books.
        let semantic = &self.quantizer.semantic.quantizers[0];
        let z_e = semantic.proj(
            &semantic.in_proj_w,
            &semantic.in_proj_b,
            in_dim,
            semantic.codebook_dim,
            &z,
            seq,
        );
        let (zq_sem, sem_idx) = semantic.decode_latents(&z_e, seq);
        let zq_sem = semantic.proj(
            &semantic.out_proj_w,
            &semantic.out_proj_b,
            semantic.codebook_dim,
            in_dim,
            &zq_sem,
            seq,
        );
        let mut residual = vec![0.0f32; z.len()];
        for (r, pair) in residual.iter_mut().zip(z.iter().zip(zq_sem.iter())) {
            *r = pair.0 - pair.1;
        }
        let mut all_codes = vec![sem_idx];
        for book in &self.quantizer.residual.quantizers {
            let z_e = book.proj(
                &book.in_proj_w,
                &book.in_proj_b,
                in_dim,
                book.codebook_dim,
                &residual,
                seq,
            );
            let (zq_raw, idx) = book.decode_latents(&z_e, seq);
            let z_q = book.proj(
                &book.out_proj_w,
                &book.out_proj_b,
                book.codebook_dim,
                in_dim,
                &zq_raw,
                seq,
            );
            for d in 0..in_dim {
                for t in 0..seq {
                    residual[d * seq + t] -= z_q[d * seq + t];
                }
            }
            all_codes.push(idx);
        }
        (z, all_codes)
    }

    /// Codes per book (semantic first) -> mono samples.
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        if codes.len() != self.config.total_books() {
            return Err(SpeechError::Input {
                why: format!(
                    "expected {} books (semantic + acoustic), got {}",
                    self.config.total_books(),
                    codes.len()
                ),
            });
        }
        let frames = codes[0].len();
        let (z, seq) = self.quantizer.decode(codes, frames)?;
        if seq
            != frames
                * self
                    .config
                    .quantizer
                    .downsample_factor
                    .iter()
                    .product::<usize>()
        {
            return Err(SpeechError::Input {
                why: format!("upsampled frames {seq} do not match the code length"),
            });
        }
        let mut h = self.decoder_conv_in.forward(&z, seq);
        for block in &self.decoder_blocks {
            let seq_in = h.len() / block.in_ch;
            let mut h2 = h.clone();
            snake1d(&mut h2, &block.alpha, block.in_ch, seq_in);
            let h3 = block.conv_tr.forward(&h2);
            let out_ch = block.conv_tr.out_ch;
            let mut h3 = h3;
            for ru in &block.res_units {
                let s = h3.len() / out_ch;
                h3 = ru.forward(&h3, s);
            }
            h = h3;
        }
        let final_dim = self.config.decoder_dim >> self.config.decoder_rates.len();
        let seq = h.len() / final_dim;
        snake1d(&mut h, &self.decoder_alpha, final_dim, seq);
        let mut audio = self.decoder_conv_out.forward(&h, seq);
        for v in &mut audio {
            *v = v.tanh();
        }
        Ok(audio)
    }
}

#[cfg(test)]
mod tests;
