//! Mimi: Kyutai's neural audio codec (Moshi's semantic/acoustic
//! tokenizer) as ported by mlx-audio.
//!
//! Reference: `mlx_audio/codec/models/mimi/` (mimi.py, modules/) at
//! mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/mimi).
//! A causal SEANet encoder/decoder with constant padding, split
//! residual vector quantization (one semantic codebook plus a 31-level
//! acoustic stack, Euclidean half-norm lookup over
//! `embedding_sum / max(cluster_usage, 1e-5)`), a learned
//! downsample/upsample pair around the frame-rate gap, and a
//! norm-first RoPE transformer on each side with a bounded context
//! window.
//!
//! Scope: the batch `encode` / `decode` paths, which the reference
//! defines as the streaming paths with fully reset state. The
//! per-frame `encode_step` / `decode_step` state machines (conv ring
//! buffers, KV caches, the streaming add) are a follow-up; batch
//! parity is the reset-state streaming contract. The model-level
//! tensor contract is `(B, C, T)` like the reference.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// SEANet geometry (`SeanetConfig`).
#[derive(Debug, Clone)]
pub struct MimiSeanetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub causal: bool,
    pub nfilters: usize,
    pub nresidual_layers: usize,
    pub ratios: Vec<usize>,
    pub ksize: usize,
    pub residual_ksize: usize,
    pub last_ksize: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub true_skip: bool,
    pub compress: usize,
}

/// Transformer geometry (`TransformerConfig`).
#[derive(Debug, Clone)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f32>,
    pub positional_embedding: String,
    pub gating: bool,
    pub norm: String,
    pub context: usize,
    pub max_period: f32,
    pub max_seq_len: usize,
    pub kv_repeat: usize,
    pub dim_feedforward: usize,
    pub conv_layout: bool,
    pub use_conv_block: bool,
    pub cross_attention: bool,
    pub conv_kernel_size: usize,
    pub use_conv_bias: bool,
}

/// Full Mimi geometry (`MimiConfig`).
#[derive(Debug, Clone)]
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: f32,
    pub frame_rate: f32,
    pub renormalize: bool,
    pub seanet: MimiSeanetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer_nq: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl MimiConfig {
    /// The `mimi_202407(32)` factory defaults, parameterized by
    /// codebook count.
    pub fn mimi_202407(num_codebooks: usize) -> MimiConfig {
        MimiConfig {
            channels: 1,
            sample_rate: 24000.0,
            frame_rate: 12.5,
            renormalize: true,
            seanet: MimiSeanetConfig {
                dimension: 512,
                channels: 1,
                causal: true,
                nfilters: 64,
                nresidual_layers: 1,
                ratios: vec![8, 6, 5, 4],
                ksize: 7,
                residual_ksize: 3,
                last_ksize: 3,
                dilation_base: 2,
                pad_mode: "constant".to_string(),
                true_skip: true,
                compress: 2,
            },
            transformer: MimiTransformerConfig {
                d_model: 512,
                num_heads: 8,
                num_layers: 8,
                causal: true,
                norm_first: true,
                bias_ff: false,
                bias_attn: false,
                layer_scale: Some(0.01),
                positional_embedding: "rope".to_string(),
                use_conv_bias: true,
                gating: false,
                norm: "layer_norm".to_string(),
                context: 250,
                max_period: 10000.0,
                max_seq_len: 8192,
                kv_repeat: 1,
                dim_feedforward: 2048,
                conv_layout: true,
                use_conv_block: false,
                cross_attention: false,
                conv_kernel_size: 3,
            },
            quantizer_nq: num_codebooks,
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }

    /// Parses the fixture-style config JSON (nested `seanet` and
    /// `transformer` objects).
    pub fn from_json(value: &serde_json::Value) -> Result<MimiConfig> {
        let mut config = Self::mimi_202407(32);
        let u = |v: &serde_json::Value, field: &str| v.get(field).and_then(|x| x.as_u64());
        let f = |v: &serde_json::Value, field: &str| v.get(field).and_then(|x| x.as_f64());
        let b = |v: &serde_json::Value, field: &str| v.get(field).and_then(|x| x.as_bool());
        let s = |v: &serde_json::Value, field: &str| {
            v.get(field).and_then(|x| x.as_str()).map(|x| x.to_string())
        };
        config.channels = u(value, "channels").unwrap_or(1) as usize;
        config.sample_rate = f(value, "sample_rate").unwrap_or(24000.0) as f32;
        config.frame_rate = f(value, "frame_rate").unwrap_or(12.5) as f32;
        config.renormalize = b(value, "renormalize").unwrap_or(true);
        config.quantizer_nq = u(value, "quantizer_nq").unwrap_or(32) as usize;
        config.quantizer_bins = u(value, "quantizer_bins").unwrap_or(2048) as usize;
        config.quantizer_dim = u(value, "quantizer_dim").unwrap_or(256) as usize;
        if let Some(sn) = value.get("seanet") {
            let seanet = &mut config.seanet;
            if let Some(d) = u(sn, "dimension") {
                seanet.dimension = d as usize;
            }
            if let Some(v) = u(sn, "channels") {
                seanet.channels = v as usize;
            }
            if let Some(v) = b(sn, "causal") {
                seanet.causal = v;
            }
            if let Some(v) = u(sn, "nfilters") {
                seanet.nfilters = v as usize;
            }
            if let Some(v) = u(sn, "nresidual_layers") {
                seanet.nresidual_layers = v as usize;
            }
            if let Some(arr) = sn.get("ratios").and_then(|x| x.as_array()) {
                seanet.ratios = arr
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect();
            }
            if let Some(v) = u(sn, "ksize") {
                seanet.ksize = v as usize;
            }
            if let Some(v) = u(sn, "residual_ksize") {
                seanet.residual_ksize = v as usize;
            }
            if let Some(v) = u(sn, "last_ksize") {
                seanet.last_ksize = v as usize;
            }
            if let Some(v) = u(sn, "dilation_base") {
                seanet.dilation_base = v as usize;
            }
            if let Some(v) = s(sn, "pad_mode") {
                seanet.pad_mode = v;
            }
            if let Some(v) = b(sn, "true_skip") {
                seanet.true_skip = v;
            }
            if let Some(v) = u(sn, "compress") {
                seanet.compress = v as usize;
            }
        }
        if let Some(tf) = value.get("transformer") {
            let tfc = &mut config.transformer;
            if let Some(v) = u(tf, "d_model") {
                tfc.d_model = v as usize;
            }
            if let Some(v) = u(tf, "num_heads") {
                tfc.num_heads = v as usize;
            }
            if let Some(v) = u(tf, "num_layers") {
                tfc.num_layers = v as usize;
            }
            if let Some(v) = b(tf, "causal") {
                tfc.causal = v;
            }
            if let Some(v) = b(tf, "norm_first") {
                tfc.norm_first = v;
            }
            if let Some(v) = b(tf, "bias_ff") {
                tfc.bias_ff = v;
            }
            if let Some(v) = b(tf, "bias_attn") {
                tfc.bias_attn = v;
            }
            tfc.layer_scale = f(tf, "layer_scale").map(|v| v as f32);
            if let Some(v) = s(tf, "positional_embedding") {
                tfc.positional_embedding = v;
            }
            if let Some(v) = b(tf, "gating") {
                tfc.gating = v;
            }
            if let Some(v) = s(tf, "norm") {
                tfc.norm = v;
            }
            if let Some(v) = u(tf, "context") {
                tfc.context = v as usize;
            }
            if let Some(v) = f(tf, "max_period") {
                tfc.max_period = v as f32;
            }
            if let Some(v) = u(tf, "max_seq_len") {
                tfc.max_seq_len = v as usize;
            }
            if let Some(v) = u(tf, "kv_repeat") {
                tfc.kv_repeat = v as usize;
            }
            if let Some(v) = u(tf, "dim_feedforward") {
                tfc.dim_feedforward = v as usize;
            }
            if let Some(v) = b(tf, "conv_layout") {
                tfc.conv_layout = v;
            }
            if let Some(v) = b(tf, "use_conv_block") {
                tfc.use_conv_block = v;
            }
            if let Some(v) = b(tf, "cross_attention") {
                tfc.cross_attention = v;
            }
            if let Some(v) = u(tf, "conv_kernel_size") {
                tfc.conv_kernel_size = v as usize;
            }
        }
        Ok(config)
    }
}

/// ELU with alpha 1 (`nn.elu`).
fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v <= 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// The reference `get_extra_padding_for_conv1d`.
fn extra_padding(len: usize, ksize: usize, stride: usize, padding_total: usize) -> usize {
    let nframes = (len + padding_total).saturating_sub(ksize) as f64 / stride as f64 + 1.0;
    let ideal_len = (nframes.ceil() as usize - 1) * stride + ksize - padding_total;
    ideal_len.saturating_sub(len)
}

/// Right-pad applied by the causal conv path: `(padding_total, extra)`.
fn causal_pad_amounts(len: usize, k_eff: usize, stride: usize) -> (usize, usize) {
    let padding_total = k_eff - stride;
    let extra = extra_padding(len, k_eff, stride, padding_total);
    (padding_total, extra)
}

/// A plain conv1d over channel-major `[ch, seq]` with the stored MLX
/// weight `[out, K, in]` (converted at load).
struct MimiConv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    /// PyTorch layout `[out, in, K]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl MimiConv1d {
    fn load(file: &SafetensorsFile, name: &str) -> Result<Self> {
        load_conv_named(file, name, 1)
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
            1,
        )
    }
}

fn load_conv_named(file: &SafetensorsFile, name: &str, groups: usize) -> Result<MimiConv1d> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing conv weight".to_string(),
    })?;
    if desc.shape.len() != 3 {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected 3-D conv weight, got {:?}", desc.shape),
        });
    }
    let (out_ch, kernel, in_g) = (desc.shape[0], desc.shape[1], desc.shape[2]);
    let in_ch = in_g * groups;
    let v = load_f32_shaped(file, name, &[out_ch, kernel, in_g])?;
    let mut weight = vec![0.0f32; out_ch * in_ch * kernel];
    for o in 0..out_ch {
        for k in 0..kernel {
            for i in 0..in_g {
                weight[o * in_ch * kernel + i * kernel + k] = v[o * kernel * in_g + k * in_g + i];
            }
        }
    }
    let bias_name = format!("{}.bias", name.strip_suffix(".weight").unwrap_or(name));
    let bias = if file.contains_tensor(&bias_name) {
        Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
    } else {
        None
    };
    Ok(MimiConv1d {
        in_ch,
        out_ch,
        kernel,
        stride: 1,
        dilation: 1,
        weight,
        bias,
    })
}

/// Causal conv with the reference extra-padding formula and pad mode.
struct StreamableConv1d {
    conv: MimiConv1d,
    causal: bool,
    reflect_edge: bool,
}

impl StreamableConv1d {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let ch = self.conv.in_ch;
        let len = x.len() / ch;
        let k_eff = (self.conv.kernel - 1) * self.conv.dilation + 1;
        let (padding_total, extra) = causal_pad_amounts(len, k_eff, self.conv.stride);
        let (left, right) = if self.causal {
            (padding_total, extra)
        } else {
            let pr = padding_total / 2;
            (padding_total - pr, pr + extra)
        };
        let padded_len = len + left + right;
        let mut padded = vec![0.0f32; ch * padded_len];
        match (self.reflect_edge, left, right) {
            (false, _, _) => {
                // constant zero pad
                for c in 0..ch {
                    padded[c * padded_len + left..c * padded_len + left + len]
                        .copy_from_slice(&x[c * len..(c + 1) * len]);
                }
            }
            (true, l, r) => {
                // edge (replicate) pad; the reference "edge" mode
                for c in 0..ch {
                    for p in 0..l {
                        padded[c * padded_len + p] = x[c * len];
                    }
                    padded[c * padded_len + l..c * padded_len + l + len]
                        .copy_from_slice(&x[c * len..(c + 1) * len]);
                    for p in 0..r {
                        padded[c * padded_len + l + len + p] = x[c * len + len - 1];
                    }
                }
            }
        }
        Ok(self.conv.forward(&padded))
    }
}

/// Causal transposed conv; the batch path runs the conv then trims
/// `padding_total` samples from the right.
struct StreamableConvTranspose1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
    causal: bool,
    /// PyTorch layout `[in, out/groups, K]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl StreamableConvTranspose1d {
    fn load(file: &SafetensorsFile, name: &str, groups: usize) -> Result<Self> {
        let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
            name: name.to_string(),
            why: "missing convtr weight".to_string(),
        })?;
        if desc.shape.len() != 3 {
            return Err(SpeechError::Tensor {
                name: name.to_string(),
                why: format!("expected 3-D convtr weight, got {:?}", desc.shape),
            });
        }
        // MLX post-mapping layout for convtr: (out, K, in/groups) where
        // the out axis enumerates (group, out-in-group) pairs.
        let (out_ch, kernel, in_g) = (desc.shape[0], desc.shape[1], desc.shape[2]);
        let in_ch = in_g * groups;
        let v = load_f32_shaped(file, name, &[out_ch, kernel, in_g])?;
        let out_g = out_ch / groups;
        let mut weight = vec![0.0f32; in_ch * out_g * kernel];
        for g in 0..groups {
            for i in 0..in_g {
                for og in 0..out_g {
                    for k in 0..kernel {
                        // PyTorch [in, out/groups, K] from stored
                        // [out = g * out_g + og, K, i].
                        weight[(g * in_g + i) * out_g * kernel + og * kernel + k] =
                            v[(g * out_g + og) * kernel * in_g + k * in_g + i];
                    }
                }
            }
        }
        let bias_name = format!("{}.bias", name.strip_suffix(".weight").unwrap_or(name));
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(StreamableConvTranspose1d {
            in_ch,
            out_ch,
            kernel,
            stride: 1,
            groups,
            causal: true,
            weight,
            bias,
        })
    }

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
            self.groups,
        );
        let padding_total = self.kernel.saturating_sub(self.stride);
        let ch = self.out_ch;
        let out_len = y.len() / ch;
        let kept = out_len.saturating_sub(padding_total);
        let mut out = vec![0.0f32; ch * kept];
        for c in 0..ch {
            out[c * kept..(c + 1) * kept].copy_from_slice(&y[c * out_len..c * out_len + kept]);
        }
        out
    }
}

/// One SEANet resnet block (two convs through the compress bottleneck,
/// true_skip residual). The convs keep the causal stream padding.
struct SeanetResnetBlock {
    block: Vec<StreamableConv1d>,
}

impl SeanetResnetBlock {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        for conv in &self.block {
            let mut a = h;
            elu(&mut a);
            h = conv.forward(&a)?;
        }
        Ok(h.iter().zip(x).map(|(a, b)| a + b).collect())
    }
}

struct EncoderLayer {
    residuals: Vec<SeanetResnetBlock>,
    downsample: StreamableConv1d,
}

impl EncoderLayer {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        for r in &self.residuals {
            h = r.forward(&h)?;
        }
        elu(&mut h);
        self.downsample.forward(&h)
    }
}

struct SeanetEncoder {
    init_conv1d: StreamableConv1d,
    layers: Vec<EncoderLayer>,
    final_conv1d: StreamableConv1d,
}

impl SeanetEncoder {
    fn forward(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut h = self.init_conv1d.forward(samples)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        elu(&mut h);
        self.final_conv1d.forward(&h)
    }
}

struct DecoderLayer {
    upsample: StreamableConvTranspose1d,
    residuals: Vec<SeanetResnetBlock>,
}

impl DecoderLayer {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = x.to_vec();
        elu(&mut h);
        let h = self.upsample.forward(&h);
        let mut h = h;
        for r in &self.residuals {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

struct SeanetDecoder {
    init_conv1d: StreamableConv1d,
    layers: Vec<DecoderLayer>,
    final_conv1d: StreamableConv1d,
}

impl SeanetDecoder {
    fn forward(&self, z: &[f32]) -> Result<Vec<f32>> {
        let mut h = self.init_conv1d.forward(z)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        elu(&mut h);
        self.final_conv1d.forward(&h)
    }
}

/// Euclidean codebook over `embedding_sum / max(cluster_usage, eps)`.
struct EuclideanCodebook {
    /// `[bins, dim]` raw rows.
    embedding: Vec<f32>,
    /// `||emb||^2 / 2` per row.
    c2: Vec<f32>,
    dim: usize,
}

impl EuclideanCodebook {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        let desc = file
            .descriptor(&format!("{prefix}.embedding_sum"))
            .ok_or_else(|| SpeechError::Tensor {
                name: format!("{prefix}.embedding_sum"),
                why: "missing".to_string(),
            })?;
        if desc.shape.len() != 2 || desc.shape[1] != dim {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.embedding_sum"),
                why: format!("expected [bins, {dim}], got {:?}", desc.shape),
            });
        }
        let bins = desc.shape[0];
        let sum = load_f32_shaped(file, &format!("{prefix}.embedding_sum"), &[bins, dim])?;
        let usage = load_f32_shaped(file, &format!("{prefix}.cluster_usage"), &[bins])?;
        if sum.len() != bins * dim || usage.len() != bins {
            return Err(SpeechError::Tensor {
                name: format!("{prefix}.embedding_sum"),
                why: format!("expected [{bins}, {dim}] sum with {bins} usage entries"),
            });
        }
        let eps = 1e-5f32;
        let mut embedding = vec![0.0f32; bins * dim];
        let mut c2 = vec![0.0f32; bins];
        for r in 0..bins {
            let denom = usage[r].max(eps);
            let mut sq = 0.0f32;
            for d in 0..dim {
                let e = sum[r * dim + d] / denom;
                embedding[r * dim + d] = e;
                sq += e * e;
            }
            c2[r] = sq / 2.0;
        }
        Ok(EuclideanCodebook { embedding, c2, dim })
    }

    fn row(&self, idx: usize) -> &[f32] {
        &self.embedding[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Nearest row: argmin of `c2 - dot`, first-wins ties.
    fn nearest(&self, col: &[f32]) -> usize {
        let mut best = f32::INFINITY;
        let mut best_idx = 0usize;
        for (idx, c2) in self.c2.iter().enumerate() {
            let row = self.row(idx);
            let mut dot = 0.0f32;
            for (a, &b) in col.iter().zip(row) {
                dot += a * b;
            }
            let dist = c2 - dot;
            if dist < best {
                best = dist;
                best_idx = idx;
            }
        }
        best_idx
    }
}

/// One vector-quantization level: linear projections (dim != codebook
/// dim) plus the codebook. All in channel-major.
struct MimiVq {
    project_in: Option<Vec<f32>>,
    project_out: Option<Vec<f32>>,
    dim: usize,
    codebook_dim: usize,
    codebook: EuclideanCodebook,
}

impl MimiVq {
    /// Projects `[dim, frames]` into codebook space `[cbd, frames]`.
    fn project_in(&self, x: &[f32], frames: usize) -> Vec<f32> {
        match &self.project_in {
            None => x.to_vec(),
            Some(w) => {
                // Linear over columns: w [cbd, dim] HF layout.
                let mut out = vec![0.0f32; self.codebook_dim * frames];
                for o in 0..self.codebook_dim {
                    let wr = &w[o * self.dim..(o + 1) * self.dim];
                    for f in 0..frames {
                        let mut acc = 0.0f32;
                        for (c, &v) in wr.iter().enumerate() {
                            acc += x[c * frames + f] * v;
                        }
                        out[o * frames + f] = acc;
                    }
                }
                out
            }
        }
    }

    fn project_out(&self, x: &[f32], frames: usize) -> Vec<f32> {
        match &self.project_out {
            None => x.to_vec(),
            Some(w) => {
                let mut out = vec![0.0f32; self.dim * frames];
                for o in 0..self.dim {
                    let wr = &w[o * self.codebook_dim..(o + 1) * self.codebook_dim];
                    for f in 0..frames {
                        let mut acc = 0.0f32;
                        for (c, &v) in wr.iter().enumerate() {
                            acc += x[c * frames + f] * v;
                        }
                        out[o * frames + f] = acc;
                    }
                }
                out
            }
        }
    }

    /// Codes for `[dim, frames]` latents.
    fn encode(&self, x: &[f32], frames: usize) -> Vec<i32> {
        let projected = self.project_in(x, frames);
        (0..frames)
            .map(|f| {
                let col: Vec<f32> = (0..self.codebook_dim)
                    .map(|d| projected[d * frames + f])
                    .collect();
                self.codebook.nearest(&col) as i32
            })
            .collect()
    }

    /// `[dim, frames]` quantized latents for the given codes.
    fn decode(&self, codes: &[i32], frames: usize) -> Vec<f32> {
        let mut quantized = vec![0.0f32; self.codebook_dim * frames];
        for (f, &code) in codes.iter().enumerate() {
            for (d, &v) in self.codebook.row(code as usize).iter().enumerate() {
                quantized[d * frames + f] = v;
            }
        }
        self.project_out(&quantized, frames)
    }
}

/// Residual stack over one codebook set.
struct MimiRvq {
    layers: Vec<MimiVq>,
    dim: usize,
}

impl MimiRvq {
    /// Codes `[nq][frames]` plus nothing else; residual across levels.
    fn encode(&self, x: &[f32], frames: usize) -> Vec<Vec<i32>> {
        let mut residual = x.to_vec();
        let mut codes = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let indices = layer.encode(&residual, frames);
            let quantized = layer.decode(&indices, frames);
            for (r, q) in residual.iter_mut().zip(&quantized) {
                *r -= q;
            }
            codes.push(indices);
        }
        codes
    }

    /// Sums the level decodes.
    fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.len() > self.layers.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "{} levels exceed the {} available",
                    codes.len(),
                    self.layers.len()
                ),
            });
        }
        let mut out = vec![0.0f32; self.dim * frames];
        for (layer, level) in self.layers.iter().zip(codes) {
            let part = layer.decode(level, frames);
            for (o, v) in out.iter_mut().zip(&part) {
                *o += v;
            }
        }
        Ok(out)
    }
}

/// The split quantizer: one semantic codebook plus a residual stack,
/// both applied to the same projected input (transcribed as-is from
/// the reference; the rest does not consume the first's residual).
struct SplitRvq {
    first_proj_in: Vec<f32>,
    first_proj_out: Vec<f32>,
    first: MimiVq,
    rest_proj_in: Vec<f32>,
    rest_proj_out: Vec<f32>,
    rest: MimiRvq,
    dim: usize,
    input_dim: usize,
    nq: usize,
}

impl SplitRvq {
    /// `codes [nq][frames]` for `[input_dim, frames]` latents.
    fn encode(&self, x: &[f32], frames: usize) -> Vec<Vec<i32>> {
        let mut codes = Vec::with_capacity(self.nq);
        let projected_first = matvec(&self.first_proj_in, x, self.input_dim, self.dim, frames);
        codes.push(self.first.encode(&projected_first, frames));
        let projected_rest = matvec(&self.rest_proj_in, x, self.input_dim, self.dim, frames);
        codes.extend(self.rest.encode(&projected_rest, frames));
        codes
    }

    /// `[input_dim, frames]` latents from codes.
    fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.len() != self.nq {
            return Err(SpeechError::Input {
                why: format!("expected {} code levels, got {}", self.nq, codes.len()),
            });
        }
        let first_q = self.first.decode(&codes[0], frames);
        let first_out = matvec(
            &self.first_proj_out,
            &first_q,
            self.dim,
            self.input_dim,
            frames,
        );
        let rest_q = self.rest.decode(&codes[1..], frames)?;
        let rest_out = matvec(
            &self.rest_proj_out,
            &rest_q,
            self.dim,
            self.input_dim,
            frames,
        );
        Ok(first_out.iter().zip(rest_out).map(|(a, b)| a + b).collect())
    }
}

/// Kernel-1 conv as a matrix multiply over channels: `w [out, in]`.
fn matvec(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim * frames];
    for o in 0..out_dim {
        let wr = &w[o * in_dim..(o + 1) * in_dim];
        for f in 0..frames {
            let mut acc = 0.0f32;
            for (c, &v) in wr.iter().enumerate() {
                acc += x[c * frames + f] * v;
            }
            out[o * frames + f] = acc;
        }
    }
    out
}

/// One transformer layer: norm-first attention with RoPE and a
/// bounded causal context, then a gated-off MLP with tanh-GELU.
struct TransformerLayer {
    in_proj_w: Vec<f32>,
    out_proj_w: Vec<f32>,
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    linear1_w: Vec<f32>,
    linear2_w: Vec<f32>,
    layer_scale_1: Vec<f32>,
    layer_scale_2: Vec<f32>,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    context: usize,
    rope_base: f32,
}

impl TransformerLayer {
    fn forward(&self, x: &mut [f32], frames: usize, pos_offset: usize) {
        let d = self.d_model;
        // norm1 -> attention
        let mut n1 = x.to_vec();
        layernorm_rows(&mut n1, frames, d, &self.norm1_w, &self.norm1_b, 1e-5);
        let attn = self.attention(&n1, frames, pos_offset);
        for (t, a) in attn.iter().enumerate() {
            for c in 0..d {
                x[c * frames + t] += a[c] * self.layer_scale_1[c];
            }
        }
        // norm2 -> mlp
        let mut n2 = x.to_vec();
        layernorm_rows(&mut n2, frames, d, &self.norm2_w, &self.norm2_b, 1e-5);
        let mlp = self.mlp(&n2, frames);
        for (t, a) in mlp.iter().enumerate() {
            for c in 0..d {
                x[c * frames + t] += a[c] * self.layer_scale_2[c];
            }
        }
    }

    fn attention(&self, x: &[f32], frames: usize, pos_offset: usize) -> Vec<Vec<f32>> {
        let d = self.d_model;
        let h = self.heads;
        let hd = self.head_dim;
        // Packed qkv over frames: rows [t][3d].
        let mut qkv = vec![0.0f32; frames * 3 * d];
        for t in 0..frames {
            for o in 0..3 * d {
                let wr = &self.in_proj_w[o * d..(o + 1) * d];
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += x[c * frames + t] * wr[c];
                }
                qkv[t * 3 * d + o] = acc;
            }
        }
        // Split into per-head q/k/v [head, frame, hd].
        let mut q = vec![0.0f32; h * frames * hd];
        let mut k = vec![0.0f32; h * frames * hd];
        let mut v = vec![0.0f32; h * frames * hd];
        for t in 0..frames {
            for head in 0..h {
                for dd in 0..hd {
                    let o = (head * hd) + dd;
                    q[(head * frames + t) * hd + dd] = qkv[t * 3 * d + o];
                    k[(head * frames + t) * hd + dd] = qkv[t * 3 * d + d + o];
                    v[(head * frames + t) * hd + dd] = qkv[t * 3 * d + 2 * d + o];
                }
            }
        }
        // Traditional (interleaved) RoPE at the absolute positions:
        // pair (x[2i], x[2i+1]) rotated by position * base^(-2i/hd)
        // (the nn.RoPE(traditional=True) contract).
        apply_traditional_rope(&mut q, h, frames, hd, pos_offset, self.rope_base);
        apply_traditional_rope(&mut k, h, frames, hd, pos_offset, self.rope_base);
        // Causal attention with the context bound.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; h * frames * hd];
        for head in 0..h {
            for t in 0..frames {
                let mut scores = vec![0.0f32; t + 1];
                for s in 0..=t {
                    let delta = pos_offset + t - (pos_offset + s);
                    let allowed = delta < self.context;
                    if !allowed {
                        scores[s] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut acc = 0.0f32;
                    for dd in 0..hd {
                        acc += q[(head * frames + t) * hd + dd] * k[(head * frames + s) * hd + dd];
                    }
                    scores[s] = acc * scale;
                }
                ops::softmax_row(&mut scores);
                for dd in 0..hd {
                    let mut acc = 0.0f32;
                    for s in 0..=t {
                        acc += scores[s] * v[(head * frames + s) * hd + dd];
                    }
                    out[(head * frames + t) * hd + dd] = acc;
                }
            }
        }
        // Concat heads and apply out_proj per frame.
        let mut concat = vec![0.0f32; frames * d];
        for t in 0..frames {
            for head in 0..h {
                for dd in 0..hd {
                    concat[t * d + head * hd + dd] = out[(head * frames + t) * hd + dd];
                }
            }
        }
        let mut outs = Vec::with_capacity(frames);
        for t in 0..frames {
            let mut o = vec![0.0f32; d];
            for (oi, ow) in o.iter_mut().enumerate() {
                let wr = &self.out_proj_w[oi * d..(oi + 1) * d];
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += concat[t * d + c] * wr[c];
                }
                *ow = acc;
            }
            outs.push(o);
        }
        outs
    }

    fn mlp(&self, x: &[f32], frames: usize) -> Vec<Vec<f32>> {
        let d = self.d_model;
        let ff = self.linear1_w.len() / d;
        let mut outs = Vec::with_capacity(frames);
        for t in 0..frames {
            let mut hidden = vec![0.0f32; ff];
            for (oi, h) in hidden.iter_mut().enumerate() {
                let wr = &self.linear1_w[oi * d..(oi + 1) * d];
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += x[c * frames + t] * wr[c];
                }
                // gelu tanh approximation.
                *h = 0.5
                    * acc
                    * (1.0 + (0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)).tanh());
            }
            let mut o = vec![0.0f32; d];
            for (oi, ov) in o.iter_mut().enumerate() {
                let wr = &self.linear2_w[oi * ff..(oi + 1) * ff];
                let mut acc = 0.0f32;
                for (c, &hv) in hidden.iter().enumerate() {
                    acc += hv * wr[c];
                }
                *ov = acc;
            }
            outs.push(o);
        }
        outs
    }
}

/// MLX `nn.RoPE(traditional=True)`: interleaved pairs rotated with
/// per-pair frequencies `base^(-2i / dim)` at absolute positions
/// `offset..offset + seq`. `x` is `[heads, seq, dim]`.
fn apply_traditional_rope(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    dim: usize,
    offset: usize,
    base: f32,
) {
    let half = dim / 2;
    for head in 0..heads {
        for t in 0..seq {
            let position = (offset + t) as f32;
            let ro = (head * seq + t) * dim;
            for d in 0..half {
                let freq = base.powf(-2.0 * d as f32 / dim as f32);
                let angle = position * freq;
                let (c, s) = (angle.cos(), angle.sin());
                let a = x[ro + 2 * d];
                let b = x[ro + 2 * d + 1];
                x[ro + 2 * d] = a * c - b * s;
                x[ro + 2 * d + 1] = b * c + a * s;
            }
        }
    }
}

fn layernorm_rows(x: &mut [f32], frames: usize, dim: usize, w: &[f32], b: &[f32], eps: f32) {
    for t in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..dim {
            mean += x[c * frames + t];
        }
        mean /= dim as f32;
        let mut var = 0.0f32;
        for c in 0..dim {
            let dd = x[c * frames + t] - mean;
            var += dd * dd;
        }
        var /= dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            x[c * frames + t] = (x[c * frames + t] - mean) * inv * w[c] + b[c];
        }
    }
}

struct ProjectedTransformer {
    layers: Vec<TransformerLayer>,
    d_model: usize,
}

impl ProjectedTransformer {
    /// `x [d_model, frames]` channel-major in and out (the reference's
    /// conv_layout swap is implicit in this layout choice).
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            layer.forward(&mut h, frames, 0);
        }
        h
    }
}

/// Loaded Mimi codec, batch-of-one.
pub struct Mimi {
    pub config: MimiConfig,
    encoder: SeanetEncoder,
    decoder: SeanetDecoder,
    quantizer: SplitRvq,
    encoder_transformer: ProjectedTransformer,
    decoder_transformer: ProjectedTransformer,
    downsample: StreamableConv1d,
    upsample: StreamableConvTranspose1d,
}

impl Mimi {
    /// Loads from a parsed config and a safetensors file whose keys
    /// use the post-mapping MLX names (what the fixture generator
    /// saves) or the raw PyTorch checkpoint names (the rename rules
    /// of `load_pytorch_weights` are applied first).
    pub fn load(config: MimiConfig, file: &SafetensorsFile) -> Result<Mimi> {
        if config.seanet.channels != 1 || config.channels != 1 {
            return Err(SpeechError::Unsupported {
                why: "mimi port is mono-only".to_string(),
            });
        }
        if !config.seanet.true_skip || !config.seanet.causal {
            return Err(SpeechError::Unsupported {
                why: "only the causal true_skip SEANet of mimi_202407 is ported".to_string(),
            });
        }
        if config.transformer.positional_embedding != "rope"
            || config.transformer.gating
            || config.transformer.kv_repeat != 1
            || config.transformer.norm != "layer_norm"
        {
            return Err(SpeechError::Unsupported {
                why: "transformer must be rope + layer_norm + no gating + kv_repeat 1".to_string(),
            });
        }
        let dim = config.seanet.dimension;
        let encoder = Self::load_encoder(&config, file)?;
        let decoder = Self::load_decoder(&config, file)?;
        let quantizer = Self::load_quantizer(&config, file)?;
        let encoder_transformer = Self::load_transformer(&config, file, "encoder_transformer")?;
        let decoder_transformer = Self::load_transformer(&config, file, "decoder_transformer")?;
        // downsample.conv.conv.conv.weight (Streamable -> Norm -> Conv1d).
        let ds_stride = (config.sample_rate
            / (config.seanet.ratios.iter().product::<usize>() as f32)
            / config.frame_rate) as usize;
        let mut ds_conv = load_conv_named(file, "downsample.conv.conv.conv.weight", 1)?;
        ds_conv.stride = ds_stride;
        let downsample = StreamableConv1d {
            conv: ds_conv,
            causal: true,
            reflect_edge: true,
        };
        let upsample = StreamableConvTranspose1d::load(
            file,
            &format!("upsample.convtr.convtr.convtr.weight"),
            dim,
        )?;
        Ok(Mimi {
            config,
            encoder,
            decoder,
            quantizer,
            encoder_transformer,
            decoder_transformer,
            downsample,
            upsample: StreamableConvTranspose1d {
                stride: ds_stride,
                ..upsample
            },
        })
    }

    fn load_encoder(config: &MimiConfig, file: &SafetensorsFile) -> Result<SeanetEncoder> {
        let sn = &config.seanet;
        let pad_edge = sn.pad_mode == "edge";
        let stream_conv =
            |name: String, stride: usize, dilation: usize| -> Result<StreamableConv1d> {
                let mut conv = load_conv_named(file, &name, 1)?;
                conv.stride = stride;
                conv.dilation = dilation;
                Ok(StreamableConv1d {
                    conv,
                    causal: sn.causal,
                    reflect_edge: pad_edge,
                })
            };
        let init_conv1d = stream_conv("encoder.init_conv1d.conv.conv.weight".to_string(), 1, 1)?;
        let mut layers = Vec::new();
        let mut mult = 1usize;
        for (li, &ratio) in sn.ratios.iter().rev().enumerate() {
            let mut residuals = Vec::new();
            let mut dilation = 1usize;
            for ri in 0..sn.nresidual_layers {
                let block = vec![
                    {
                        let mut c = load_conv_named(
                            file,
                            &format!("encoder.layers.{li}.residuals.{ri}.block.0.conv.conv.weight"),
                            1,
                        )?;
                        c.dilation = dilation;
                        StreamableConv1d {
                            conv: c,
                            causal: sn.causal,
                            reflect_edge: pad_edge,
                        }
                    },
                    StreamableConv1d {
                        conv: load_conv_named(
                            file,
                            &format!("encoder.layers.{li}.residuals.{ri}.block.1.conv.conv.weight"),
                            1,
                        )?,
                        causal: sn.causal,
                        reflect_edge: pad_edge,
                    },
                ];
                residuals.push(SeanetResnetBlock { block });
                dilation *= sn.dilation_base;
            }
            let downsample = stream_conv(
                format!("encoder.layers.{li}.downsample.conv.conv.weight"),
                ratio,
                1,
            )?;
            layers.push(EncoderLayer {
                residuals,
                downsample,
            });
            mult *= 2;
        }
        let final_conv1d = stream_conv("encoder.final_conv1d.conv.conv.weight".to_string(), 1, 1)?;
        Ok(SeanetEncoder {
            init_conv1d,
            layers,
            final_conv1d,
        })
    }

    fn load_decoder(config: &MimiConfig, file: &SafetensorsFile) -> Result<SeanetDecoder> {
        let sn = &config.seanet;
        let pad_edge = sn.pad_mode == "edge";
        let stream_conv =
            |name: String, stride: usize, dilation: usize| -> Result<StreamableConv1d> {
                let mut conv = load_conv_named(file, &name, 1)?;
                conv.stride = stride;
                conv.dilation = dilation;
                Ok(StreamableConv1d {
                    conv,
                    causal: sn.causal,
                    reflect_edge: pad_edge,
                })
            };
        let init_conv1d = stream_conv("decoder.init_conv1d.conv.conv.weight".to_string(), 1, 1)?;
        let mut layers = Vec::new();
        let mut mult = 1usize << sn.ratios.len();
        for (li, &ratio) in sn.ratios.iter().enumerate() {
            let upsample = StreamableConvTranspose1d {
                stride: ratio,
                ..StreamableConvTranspose1d::load(
                    file,
                    &format!("decoder.layers.{li}.upsample.convtr.convtr.weight"),
                    1,
                )?
            };
            let mut residuals = Vec::new();
            let mut dilation = 1usize;
            for ri in 0..sn.nresidual_layers {
                let block = vec![
                    {
                        let mut c = load_conv_named(
                            file,
                            &format!("decoder.layers.{li}.residuals.{ri}.block.0.conv.conv.weight"),
                            1,
                        )?;
                        c.dilation = dilation;
                        StreamableConv1d {
                            conv: c,
                            causal: sn.causal,
                            reflect_edge: pad_edge,
                        }
                    },
                    StreamableConv1d {
                        conv: load_conv_named(
                            file,
                            &format!("decoder.layers.{li}.residuals.{ri}.block.1.conv.conv.weight"),
                            1,
                        )?,
                        causal: sn.causal,
                        reflect_edge: pad_edge,
                    },
                ];
                residuals.push(SeanetResnetBlock { block });
                dilation *= sn.dilation_base;
            }
            layers.push(DecoderLayer {
                upsample,
                residuals,
            });
            mult /= 2;
        }
        let final_conv1d = stream_conv("decoder.final_conv1d.conv.conv.weight".to_string(), 1, 1)?;
        Ok(SeanetDecoder {
            init_conv1d,
            layers,
            final_conv1d,
        })
    }

    fn load_quantizer(config: &MimiConfig, file: &SafetensorsFile) -> Result<SplitRvq> {
        let dim = config.quantizer_dim;
        let input_dim = config.seanet.dimension;
        // The projections are kernel-1 convs stored [out, 1, in].
        let load_proj = |name: &str| -> Result<Vec<f32>> {
            let v = load_f32_shaped(file, name, &[dim.max(input_dim), 1, dim.min(input_dim)])
                .or_else(|_| load_f32_shaped(file, name, &[dim, 1, input_dim]))
                .or_else(|_| load_f32_shaped(file, name, &[input_dim, 1, dim]))?;
            Ok(v)
        };
        let first_proj_in = load_proj("quantizer.rvq_first.input_proj.weight")?;
        let first_proj_out = load_proj("quantizer.rvq_first.output_proj.weight")?;
        let rest_proj_in = load_proj("quantizer.rvq_rest.input_proj.weight")?;
        let rest_proj_out = load_proj("quantizer.rvq_rest.output_proj.weight")?;
        let first = MimiVq {
            project_in: None,
            project_out: None,
            dim,
            codebook_dim: dim,
            codebook: EuclideanCodebook::load(
                file,
                "quantizer.rvq_first.vq.layers.0.codebook",
                dim,
            )?,
        };
        let mut rest_layers = Vec::new();
        for i in 0..(config.quantizer_nq - 1) {
            rest_layers.push(MimiVq {
                project_in: None,
                project_out: None,
                dim,
                codebook_dim: dim,
                codebook: EuclideanCodebook::load(
                    file,
                    &format!("quantizer.rvq_rest.vq.layers.{i}.codebook"),
                    dim,
                )?,
            });
        }
        Ok(SplitRvq {
            first_proj_in,
            first_proj_out,
            first,
            rest_proj_in,
            rest_proj_out,
            rest: MimiRvq {
                layers: rest_layers,
                dim,
            },
            dim,
            input_dim,
            nq: config.quantizer_nq,
        })
    }

    fn load_transformer(
        config: &MimiConfig,
        file: &SafetensorsFile,
        prefix: &str,
    ) -> Result<ProjectedTransformer> {
        let tf = &config.transformer;
        let d = tf.d_model;
        let mut layers = Vec::new();
        for li in 0..tf.num_layers {
            let base = format!("{prefix}.transformer.layers.{li}");
            let layer = TransformerLayer {
                in_proj_w: load_f32_shaped(
                    file,
                    &format!("{base}.self_attn.in_proj.weight"),
                    &[3 * d, d],
                )?,
                out_proj_w: load_f32_shaped(
                    file,
                    &format!("{base}.self_attn.out_proj.weight"),
                    &[d, d],
                )?,
                norm1_w: load_f32_shaped(file, &format!("{base}.norm1.weight"), &[d])?,
                norm1_b: load_f32_shaped(file, &format!("{base}.norm1.bias"), &[d])?,
                norm2_w: load_f32_shaped(file, &format!("{base}.norm2.weight"), &[d])?,
                norm2_b: load_f32_shaped(file, &format!("{base}.norm2.bias"), &[d])?,
                linear1_w: load_f32_shaped(
                    file,
                    &format!("{base}.gating.linear1.weight"),
                    &[tf.dim_feedforward, d],
                )?,
                linear2_w: load_f32_shaped(
                    file,
                    &format!("{base}.gating.linear2.weight"),
                    &[d, tf.dim_feedforward],
                )?,
                layer_scale_1: load_f32_shaped(file, &format!("{base}.layer_scale_1.scale"), &[d])?,
                layer_scale_2: load_f32_shaped(file, &format!("{base}.layer_scale_2.scale"), &[d])?,
                d_model: d,
                heads: tf.num_heads,
                head_dim: d / tf.num_heads,
                context: tf.context,
                rope_base: tf.max_period,
            };
            layers.push(layer);
        }
        Ok(ProjectedTransformer { layers, d_model: d })
    }

    /// Encodes mono samples into `[nq][frames]` code levels.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<i32>>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "empty waveform".to_string(),
            });
        }
        // Model contract is (B, C, T) = channel-major [1, T] here.
        let z = self.encoder.forward(samples)?;
        let dim = self.config.seanet.dimension;
        let frames = z.len() / dim;
        let z = self.encoder_transformer.forward(&z, frames);
        let z = self.downsample.forward(&z)?;
        let frames = z.len() / dim;
        Ok(self.quantizer.encode(&z, frames))
    }

    /// Decodes code levels to mono samples.
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let dim = self.config.seanet.dimension;
        let frames = codes.first().map(|c| c.len()).unwrap_or(0);
        for level in codes {
            if level.len() != frames {
                return Err(SpeechError::Input {
                    why: "ragged code levels".to_string(),
                });
            }
        }
        let z = self.quantizer.decode(codes, frames)?;
        let z = self.upsample.forward(&z);
        let frames = z.len() / dim;
        let z = self.decoder_transformer.forward(&z, frames);
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod tests;
