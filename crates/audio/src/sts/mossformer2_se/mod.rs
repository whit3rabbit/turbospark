//! MossFormer2 SE 48 kHz speech enhancement.
//!
//! Port of `mlx_audio/sts/models/mossformer2_se` at reference commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The network is a
//! MossFormer MaskNet: a global-layer-norm frontend, a 1x1 encoder, a
//! scaled sinusoidal position embedding, a stack of FLASH
//! (ReLU^2 group attention + linear attention) blocks interleaved with
//! gated FSMN blocks, and a gated tanh/sigmoid decoder that predicts a
//! complex STFT mask. Enhancement multiplies the Hamming-windowed STFT
//! of the input by the mask and overlap-adds the inverse STFT.
//!
//! Divergences from the reference, all documented in the family README:
//! - Kaldi fbank dither defaults to 0 here; the reference default is
//!   1.0, which draws fresh Gaussian noise on every call and makes the
//!   reference itself nondeterministic. The dither amount is a
//!   parameter and nonzero values are refused (no MLX RNG parity).
//! - Only plain F32/F16 linear weights and 8-bit group-64 affine
//!   quantization are accepted; other schemes are refused.
//! - Segmented (>20 s) and chunked (>60 s) reassembly follow the
//!   reference index math but are pinned by invariant tests only; the
//!   full path carries fixture and checkpoint parity.

use std::path::Path;

use crate::error::SpeechError;
use crate::fft::RealFftPlan;
use crate::ops;
use crate::quant::{self, QuantScheme};

type Result<T> = std::result::Result<T, SpeechError>;

const MAX_WAV_VALUE: f32 = 32768.0;
/// Attention group size. The reference MaskNet never forwards its
/// `group_size` argument into the computation block, so the MossFormerM
/// default of 256 is the only reachable geometry.
const GROUP_SIZE: usize = 256;
/// Shared query/key width, fixed by the MossFormerM default.
const QUERY_KEY_DIM: usize = 128;
/// Rotary dimensions inside the 128-wide query/key (`min(32, qk_dim)`).
const ROPE_DIMS: usize = 32;
/// FSMN kernel span (2 * lorder - 1 taps, symmetric pad lorder - 1).
const FSMN_LORDER: usize = 20;
/// Gated FSMN block width, fixed by the MossFormerBlock_GFSMN default.
const INNER_CHANNELS: usize = 256;
const LN_EPS: f32 = 1e-8;

/// Processing and architecture configuration, matching
/// `MossFormer2SEConfig` in the reference.
#[derive(Debug, Clone)]
pub struct MossFormer2SeConfig {
    pub sample_rate: usize,
    pub win_len: usize,
    pub win_inc: usize,
    pub fft_len: usize,
    pub win_type: String,
    pub num_mels: usize,
    pub preemphasis: f32,
    pub one_time_decode_length: usize,
    pub decode_window: usize,
    pub chunk_seconds: f32,
    pub chunk_overlap: f32,
    pub auto_chunk_threshold: f32,
    pub in_channels: usize,
    pub out_channels: usize,
    pub out_channels_final: usize,
    pub num_blocks: usize,
}

impl Default for MossFormer2SeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            win_len: 1920,
            win_inc: 384,
            fft_len: 1920,
            win_type: "hamming".to_string(),
            num_mels: 60,
            preemphasis: 0.97,
            one_time_decode_length: 20,
            decode_window: 4,
            chunk_seconds: 4.0,
            chunk_overlap: 0.25,
            auto_chunk_threshold: 60.0,
            in_channels: 180,
            out_channels: 512,
            out_channels_final: 961,
            num_blocks: 24,
        }
    }
}

impl MossFormer2SeConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let mut config = Self::default();
        let obj = value.as_object().ok_or_else(|| SpeechError::BadConfig {
            field: "config".to_string(),
            why: "expected a JSON object".to_string(),
        })?;
        let _field = |name: &str| -> Result<serde_json::Value> {
            obj.get(name)
                .cloned()
                .ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "missing".to_string(),
                })
        };
        let get_u64 = |name: &str, default: usize| -> Result<usize> {
            Ok(obj
                .get(name)
                .and_then(|v| v.as_u64())
                .unwrap_or(default as u64) as usize)
        };
        let get_f64 = |name: &str, default: f32| -> Result<f32> {
            Ok(obj
                .get(name)
                .and_then(|v| v.as_f64())
                .unwrap_or(default as f64) as f32)
        };
        let get_str = |name: &str, default: &str| -> Result<String> {
            Ok(obj
                .get(name)
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string())
        };
        config.sample_rate = get_u64("sample_rate", config.sample_rate)?;
        config.win_len = get_u64("win_len", config.win_len)?;
        config.win_inc = get_u64("win_inc", config.win_inc)?;
        config.fft_len = get_u64("fft_len", config.fft_len)?;
        config.win_type = get_str("win_type", &config.win_type)?;
        config.num_mels = get_u64("num_mels", config.num_mels)?;
        config.preemphasis = get_f64("preemphasis", config.preemphasis)?;
        config.one_time_decode_length =
            get_u64("one_time_decode_length", config.one_time_decode_length)?;
        config.decode_window = get_u64("decode_window", config.decode_window)?;
        config.chunk_seconds = get_f64("chunk_seconds", config.chunk_seconds)?;
        config.chunk_overlap = get_f64("chunk_overlap", config.chunk_overlap)?;
        config.auto_chunk_threshold = get_f64("auto_chunk_threshold", config.auto_chunk_threshold)?;
        config.in_channels = get_u64("in_channels", config.in_channels)?;
        config.out_channels = get_u64("out_channels", config.out_channels)?;
        config.out_channels_final = get_u64("out_channels_final", config.out_channels_final)?;
        config.num_blocks = get_u64("num_blocks", config.num_blocks)?;
        if config.out_channels_final != config.fft_len / 2 + 1 {
            return Err(SpeechError::BadConfig {
                field: "out_channels_final".to_string(),
                why: format!(
                    "must equal fft_len / 2 + 1 ({}), got {}",
                    config.fft_len / 2 + 1,
                    config.out_channels_final
                ),
            });
        }
        if config.in_channels != 3 * config.num_mels {
            return Err(SpeechError::BadConfig {
                field: "in_channels".to_string(),
                why: format!(
                    "must equal 3 * num_mels ({}), got {}",
                    3 * config.num_mels,
                    config.in_channels
                ),
            });
        }
        Ok(config)
    }
}

/// A linear layer: weight `[out, in]`, optional bias.
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    fn run(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.in_dim,
            self.out_dim,
        )
    }
}

/// The FFConvM norm is either a LayerNorm (weight + bias) or a ScaleNorm.
enum Norm {
    LayerNorm { weight: Vec<f32>, bias: Vec<f32> },
    ScaleNorm { g: f32 },
}

/// Depthwise convolution over time plus residual, shared by ConvModule
/// (17 taps, pad 8) and the inner FSMN conv (39 taps, pad 19). The
/// stored weight is `[kernel][channel]`.
struct TimeDepthwiseConv {
    weight: Vec<f32>,
    kernel: usize,
    channels: usize,
}

impl TimeDepthwiseConv {
    /// Convolution output only (no residual).
    fn conv(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let pad = (self.kernel - 1) / 2;
        let mut out = vec![0.0f32; frames * self.channels];
        for t in 0..frames {
            for k in 0..self.kernel {
                let src = t as isize + k as isize - pad as isize;
                if src < 0 || src as usize >= frames {
                    continue;
                }
                let src_row = (src as usize) * self.channels;
                let w_row = k * self.channels;
                for c in 0..self.channels {
                    out[t * self.channels + c] += x[src_row + c] * self.weight[w_row + c];
                }
            }
        }
        out
    }

    fn conv_residual(&self, x: &mut [f32], frames: usize) {
        let out = self.conv(x, frames);
        for (xv, ov) in x.iter_mut().zip(out) {
            *xv += ov;
        }
    }
}

struct FfConvM {
    norm: Norm,
    linear: Linear,
    conv: TimeDepthwiseConv,
}

impl FfConvM {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        match &self.norm {
            Norm::LayerNorm { weight, bias } => ops::layernorm(
                &mut out,
                frames,
                self.linear.in_dim,
                weight,
                Some(bias),
                LN_EPS,
            ),
            Norm::ScaleNorm { g } => scale_norm(&mut out, frames, self.linear.in_dim, *g),
        }
        let mut projected = self.linear.run(&out, frames);
        ops::silu(&mut projected);
        self.conv.conv_residual(&mut projected, frames);
        projected
    }
}

/// scale_norm helper: x * (g / max(||x|| * dim^-0.5, eps)) per row.
fn scale_norm(x: &mut [f32], _rows: usize, cols: usize, g: f32) {
    let scale = (cols as f32).powf(-0.5);
    for row in x.chunks_exact_mut(cols) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let norm = (sum_sq.sqrt() * scale).max(1e-8);
        let factor = g / norm;
        for value in row.iter_mut() {
            *value *= factor;
        }
    }
}

/// Global layer norm over (channels, time) jointly with per-channel
/// affine; the stored weights are `[dim, 1]` in the checkpoint.
struct GlobalLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    channels: usize,
}

impl GlobalLayerNorm {
    fn run(&self, x: &mut [f32], frames: usize) {
        let count = (frames * self.channels) as f32;
        let mut mean = 0.0f32;
        for value in x.iter() {
            mean += value;
        }
        mean /= count;
        let mut var = 0.0f32;
        for value in x.iter() {
            let d = value - mean;
            var += d * d;
        }
        var /= count;
        let denom = (var + LN_EPS).sqrt();
        for t in 0..frames {
            for c in 0..self.channels {
                let i = t * self.channels + c;
                x[i] = self.weight[c] * (x[i] - mean) / denom + self.bias[c];
            }
        }
    }
}

struct OffsetScale {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    heads: usize,
    dim: usize,
}

impl OffsetScale {
    /// Returns `heads` rows-major `[frames, dim]` tensors.
    fn run(&self, x: &[f32], frames: usize) -> Vec<Vec<f32>> {
        let mut outs = Vec::with_capacity(self.heads);
        for h in 0..self.heads {
            let mut out = vec![0.0f32; frames * self.dim];
            for t in 0..frames {
                for d in 0..self.dim {
                    out[t * self.dim + d] = x[t * self.dim + d] * self.gamma[h * self.dim + d]
                        + self.beta[h * self.dim + d];
                }
            }
            outs.push(out);
        }
        outs
    }
}

/// UniDeepFsmn: linear -> relu -> project -> depthwise time conv with
/// symmetric pad, plus both residuals.
struct UniDeepFsmn {
    linear: Linear,
    project: Linear,
    conv: TimeDepthwiseConv,
}

impl UniDeepFsmn {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let f1 = self.linear.run(x, frames);
        let mut f1relu = f1;
        for value in f1relu.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        let p1 = self.project.run(&f1relu, frames);
        let conv = self.conv.conv(&p1, frames);
        let mut out = vec![0.0f32; frames * self.conv.channels];
        for i in 0..out.len() {
            out[i] = x[i] + p1[i] + conv[i];
        }
        out
    }
}

struct GatedFsmn {
    to_u: FfConvM,
    to_v: FfConvM,
    fsmn: UniDeepFsmn,
}

impl GatedFsmn {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let x_u = self.to_u.forward(x, frames);
        let x_v = self.to_v.forward(x, frames);
        let x_u = self.fsmn.forward(&x_u, frames);
        let mut out = vec![0.0f32; x.len()];
        for i in 0..out.len() {
            out[i] = x_v[i] * x_u[i] + x[i];
        }
        out
    }
}

struct GatedFsmnBlock {
    conv1: Linear,
    prelu: f32,
    norm1: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    gated_fsmn: GatedFsmn,
    conv2: Linear,
}

impl GatedFsmnBlock {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let residual = x;
        let mut h = self.conv1.run(x, frames);
        for value in h.iter_mut() {
            if *value < 0.0 {
                *value *= self.prelu;
            }
        }
        ops::layernorm(
            &mut h,
            frames,
            INNER_CHANNELS,
            &self.norm1.0,
            Some(&self.norm1.1),
            LN_EPS,
        );
        let h = self.gated_fsmn.forward(&h, frames);
        let mut h = h;
        ops::layernorm(
            &mut h,
            frames,
            INNER_CHANNELS,
            &self.norm2.0,
            Some(&self.norm2.1),
            LN_EPS,
        );
        let mut h = self.conv2.run(&h, frames);
        for (hv, rv) in h.iter_mut().zip(residual) {
            *hv += rv;
        }
        h
    }
}

struct FlashLayer {
    to_hidden: FfConvM,
    to_qk: FfConvM,
    to_out: FfConvM,
    offset_scale: OffsetScale,
}

impl FlashLayer {
    /// Half-split NeoX rotary over the first ROPE_DIMS features; the
    /// position advances along time. `buf` is `[frames, QUERY_KEY_DIM]`.
    fn rope(buf: &mut [f32], frames: usize) {
        let half = ROPE_DIMS / 2;
        let mut cos = vec![0.0f32; frames * half];
        let mut sin = vec![0.0f32; frames * half];
        for t in 0..frames {
            for i in 0..half {
                let freq = 10000f32.powf(-((2 * i) as f32) / ROPE_DIMS as f32);
                let ang = t as f32 * freq;
                cos[t * half + i] = ang.cos();
                sin[t * half + i] = ang.sin();
            }
        }
        for t in 0..frames {
            let row = &mut buf[t * QUERY_KEY_DIM..(t + 1) * QUERY_KEY_DIM];
            for i in 0..half {
                let (c, s) = (cos[t * half + i], sin[t * half + i]);
                let x0 = row[i];
                let x1 = row[half + i];
                row[i] = x0 * c - x1 * s;
                row[half + i] = x0 * s + x1 * c;
            }
        }
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let dim = self.to_hidden.linear.in_dim;
        let hidden_dim = self.to_hidden.linear.out_dim;
        let half_dim = dim / 2;
        let value_dim = hidden_dim / 2;

        // Token shifting: the first feature half moves one step forward
        // in time (zeros at frame 0); the second half passes through.
        let mut normed = vec![0.0f32; frames * dim];
        for t in 0..frames {
            for c in 0..half_dim {
                normed[t * dim + c] = if t == 0 { 0.0 } else { x[(t - 1) * dim + c] };
            }
            for c in half_dim..dim {
                normed[t * dim + c] = x[t * dim + c];
            }
        }

        let hidden = self.to_hidden.forward(&normed, frames);
        // Split into v (first half) and u (second half).
        let mut v = vec![0.0f32; frames * value_dim];
        let mut u = vec![0.0f32; frames * value_dim];
        for t in 0..frames {
            v[t * value_dim..(t + 1) * value_dim]
                .copy_from_slice(&hidden[t * hidden_dim..t * hidden_dim + value_dim]);
            u[t * value_dim..(t + 1) * value_dim]
                .copy_from_slice(&hidden[t * hidden_dim + value_dim..(t + 1) * hidden_dim]);
        }
        let qk = self.to_qk.forward(&normed, frames);
        self.forward_inner(x, frames, value_dim, v, u, qk)
    }

    /// The attention body after the projections: rotary, group pad,
    /// ReLU^2 + linear attention, gating, output projection, residual.
    fn forward_inner(
        &self,
        x: &[f32],
        frames: usize,
        value_dim: usize,
        v: Vec<f32>,
        u: Vec<f32>,
        qk: Vec<f32>,
    ) -> Vec<f32> {
        let mut heads = self.offset_scale.run(&qk, frames);
        let mut quad_q = std::mem::take(&mut heads[0]);
        let mut lin_q = std::mem::take(&mut heads[1]);
        let mut quad_k = std::mem::take(&mut heads[2]);
        let mut lin_k = std::mem::take(&mut heads[3]);
        Self::rope(&mut quad_q, frames);
        Self::rope(&mut lin_q, frames);
        Self::rope(&mut quad_k, frames);
        Self::rope(&mut lin_k, frames);

        // Pad to a multiple of GROUP_SIZE and process groupwise.
        let padded = frames.div_ceil(GROUP_SIZE) * GROUP_SIZE;
        let groups = padded / GROUP_SIZE;
        let zero_row_q = vec![0.0f32; QUERY_KEY_DIM];
        let zero_row_v = vec![0.0f32; value_dim];
        let mut pq_q = Vec::with_capacity(padded * QUERY_KEY_DIM);
        let mut pq_k = pq_q.clone();
        let mut pl_q = pq_q.clone();
        let mut pl_k = pq_q.clone();
        let mut pv = Vec::with_capacity(padded * value_dim);
        let mut pu = pv.clone();
        for g in 0..groups {
            for r in 0..GROUP_SIZE {
                let idx = g * GROUP_SIZE + r;
                if idx < frames {
                    let base = idx * QUERY_KEY_DIM;
                    pq_q.extend_from_slice(&quad_q[base..base + QUERY_KEY_DIM]);
                    pq_k.extend_from_slice(&quad_k[base..base + QUERY_KEY_DIM]);
                    pl_q.extend_from_slice(&lin_q[base..base + QUERY_KEY_DIM]);
                    pl_k.extend_from_slice(&lin_k[base..base + QUERY_KEY_DIM]);
                    let vbase = idx * value_dim;
                    pv.extend_from_slice(&v[vbase..vbase + value_dim]);
                    pu.extend_from_slice(&u[vbase..vbase + value_dim]);
                } else {
                    pq_q.extend_from_slice(&zero_row_q);
                    pq_k.extend_from_slice(&zero_row_q);
                    pl_q.extend_from_slice(&zero_row_q);
                    pl_k.extend_from_slice(&zero_row_q);
                    pv.extend_from_slice(&zero_row_v);
                    pu.extend_from_slice(&zero_row_v);
                }
            }
        }

        let att_v =
            quad_plus_linear_attention(&pq_q, &pq_k, &pl_q, &pl_k, &pv, groups, value_dim, frames);
        let att_u =
            quad_plus_linear_attention(&pq_q, &pq_k, &pl_q, &pl_k, &pu, groups, value_dim, frames);

        // Gating: (att_u * v) * sigmoid(att_v * u), unpadded frames.
        let mut gated = vec![0.0f32; frames * value_dim];
        for i in 0..frames * value_dim {
            let gate = ops::sigmoid(att_v[i] * u[i]);
            gated[i] = att_u[i] * v[i] * gate;
        }
        let mut out = self.to_out.forward(&gated, frames);
        for (ov, xv) in out.iter_mut().zip(x) {
            *ov += xv;
        }
        out
    }
}

/// ReLU^2 quadratic group attention plus non-causal linear attention,
/// both matching `FlashAttentionImplementations` / `cal_attention`.
/// `q*`/`k*` are `[groups * GROUP_SIZE, QUERY_KEY_DIM]`, `v` is
/// `[groups * GROUP_SIZE, value_dim]`; `frames` is the unpadded length
/// used for the linear-attention normalizer.
fn quad_plus_linear_attention(
    quad_q: &[f32],
    quad_k: &[f32],
    lin_q: &[f32],
    lin_k: &[f32],
    v: &[f32],
    groups: usize,
    value_dim: usize,
    frames: usize,
) -> Vec<f32> {
    let scale = 1.0 / GROUP_SIZE as f32;
    let mut out = vec![0.0f32; groups * GROUP_SIZE * value_dim];
    // Linear attention context: sum over all padded rows of k^T v / n.
    let mut lin_kv = vec![0.0f32; QUERY_KEY_DIM * value_dim];
    let n = frames as f32;
    for r in 0..groups * GROUP_SIZE {
        let k_row = &lin_k[r * QUERY_KEY_DIM..(r + 1) * QUERY_KEY_DIM];
        let v_row = &v[r * value_dim..(r + 1) * value_dim];
        for d in 0..QUERY_KEY_DIM {
            let kd = k_row[d];
            if kd == 0.0 {
                continue;
            }
            let kv_row = &mut lin_kv[d * value_dim..(d + 1) * value_dim];
            for (e, kv) in kv_row.iter_mut().enumerate() {
                *kv += kd * v_row[e];
            }
        }
    }
    for value in lin_kv.iter_mut() {
        *value /= n;
    }

    for g in 0..groups {
        let base = g * GROUP_SIZE;
        for i in 0..GROUP_SIZE {
            let q_row = &quad_q[(base + i) * QUERY_KEY_DIM..(base + i + 1) * QUERY_KEY_DIM];
            let lq_row = &lin_q[(base + i) * QUERY_KEY_DIM..(base + i + 1) * QUERY_KEY_DIM];
            let out_row = &mut out[(base + i) * value_dim..(base + i + 1) * value_dim];
            for j in 0..GROUP_SIZE {
                let k_row = &quad_k[(base + j) * QUERY_KEY_DIM..(base + j + 1) * QUERY_KEY_DIM];
                let mut sim = 0.0f32;
                for d in 0..QUERY_KEY_DIM {
                    sim += q_row[d] * k_row[d];
                }
                sim *= scale;
                if sim > 0.0 {
                    let attn = sim * sim;
                    let v_row = &v[(base + j) * value_dim..(base + j + 1) * value_dim];
                    for (e, ov) in out_row.iter_mut().enumerate() {
                        *ov += attn * v_row[e];
                    }
                }
            }
            for d in 0..QUERY_KEY_DIM {
                let lq = lq_row[d];
                if lq == 0.0 {
                    continue;
                }
                let kv_row = &lin_kv[d * value_dim..(d + 1) * value_dim];
                for (e, ov) in out_row.iter_mut().enumerate() {
                    *ov += lq * kv_row[e];
                }
            }
        }
    }
    out
}

/// One MossFormer GFSMN stack: alternating FLASH and gated FSMN blocks
/// followed by the MossFormerM layer norm.
struct MossFormerM {
    layers: Vec<FlashLayer>,
    fsmns: Vec<GatedFsmnBlock>,
    norm: (Vec<f32>, Vec<f32>),
}

impl MossFormerM {
    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        for (layer, fsmn) in self.layers.iter().zip(&self.fsmns) {
            h = layer.forward(&h, frames);
            h = fsmn.forward(&h, frames);
        }
        let dim = self.norm.0.len();
        ops::layernorm(
            &mut h,
            frames,
            dim,
            &self.norm.0,
            Some(&self.norm.1),
            LN_EPS,
        );
        h
    }
}

/// The full mask network: frontend norm, encoder conv, positional
/// embedding, the MossFormer stack, and the gated decoder.
pub struct MossFormerMaskNet {
    gln: GlobalLayerNorm,
    encoder: Linear,
    pos_scale: f32,
    pos_inv_freq: Vec<f32>,
    mdl: MossFormerM,
    intra_norm: (Vec<f32>, Vec<f32>),
    prelu: f32,
    spk_conv: Linear,
    output: Linear,
    output_gate: Linear,
    decoder: Linear,
    num_spks: usize,
    out_channels: usize,
}

impl MossFormerMaskNet {
    /// Predicts the complex mask `[frames, out_channels_final]` from
    /// the concatenated fbank features `[frames, in_channels]`.
    pub fn forward(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let mut x = features.to_vec();
        self.gln.run(&mut x, frames);
        let mut x = self.encoder.run(&x, frames);

        // Scaled sinusoidal position embedding added over channels.
        let dim = self.out_channels;
        let base = x.clone();
        for t in 0..frames {
            for (j, &freq) in self.pos_inv_freq.iter().enumerate() {
                let ang = t as f32 * freq;
                x[t * dim + j] = base[t * dim + j] + ang.sin() * self.pos_scale;
                x[t * dim + dim / 2 + j] = base[t * dim + dim / 2 + j] + ang.cos() * self.pos_scale;
            }
        }

        let intra = self.mdl.forward(&x, frames);
        // Computation_Block: GroupNorm(1, C, eps 1e-8) over (T, C) then
        // the skip around the intra stack.
        let mut intra = intra;
        {
            let count = (frames * dim) as f32;
            let mut mean = 0.0f32;
            for value in intra.iter() {
                mean += value;
            }
            mean /= count;
            let mut var = 0.0f32;
            for value in intra.iter() {
                let d = value - mean;
                var += d * d;
            }
            var /= count;
            let denom = (var + LN_EPS).sqrt();
            let (w, b) = &self.intra_norm;
            for value in intra.iter_mut() {
                *value = (*value - mean) / denom;
            }
            for t in 0..frames {
                for c in 0..dim {
                    let i = t * dim + c;
                    intra[i] = intra[i] * w[c] + b[c];
                }
            }
        }
        for (iv, xv) in intra.iter_mut().zip(x.iter()) {
            *iv += xv;
        }

        // PReLU, speaker expansion, gated decoder; first speaker wins.
        let mut h = intra;
        for value in h.iter_mut() {
            if *value < 0.0 {
                *value *= self.prelu;
            }
        }
        let spk_hidden = self.spk_conv.run(&h, frames); // [frames, out * spks]
        let mut result = vec![0.0f32; frames * self.decoder.out_dim];
        for spk in 0..self.num_spks {
            let mut xs = vec![0.0f32; frames * self.out_channels];
            for t in 0..frames {
                xs[t * self.out_channels..(t + 1) * self.out_channels].copy_from_slice(
                    &spk_hidden[t * self.out_channels * self.num_spks + spk * self.out_channels
                        ..t * self.out_channels * self.num_spks + (spk + 1) * self.out_channels],
                );
            }
            let gate_val: Vec<f32> = self
                .output_gate
                .run(&xs, frames)
                .into_iter()
                .map(ops::sigmoid)
                .collect();
            let output_val: Vec<f32> = self
                .output
                .run(&xs, frames)
                .into_iter()
                .map(|v| v.tanh())
                .collect();
            let mut gated = vec![0.0f32; frames * self.out_channels];
            for i in 0..gated.len() {
                gated[i] = output_val[i] * gate_val[i];
            }
            let dec = self.decoder.run(&gated, frames);
            if spk == 0 {
                for (rv, dv) in result.iter_mut().zip(dec) {
                    *rv = dv.max(0.0);
                }
            }
        }
        result
    }
}

/// The loaded MossFormer2 SE model: network plus the DSP front end.
pub struct MossFormer2Se {
    pub config: MossFormer2SeConfig,
    masknet: MossFormerMaskNet,
    window: Vec<f32>,
}

impl MossFormer2Se {
    /// Loads from a checkpoint directory holding `model.safetensors`
    /// and an optional `config.json` (defaults apply when absent, which
    /// matches the reference repository that ships no config file).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config = if config_path.exists() {
            let value = crate::quant::read_json(&config_path)?;
            MossFormer2SeConfig::from_json(&value)?
        } else {
            MossFormer2SeConfig::default()
        };
        let file =
            turbospark_model_io::safetensors::SafetensorsFile::open(&dir.join("model.safetensors"))
                .map_err(|e| SpeechError::BadConfig {
                    field: "model.safetensors".to_string(),
                    why: e.to_string(),
                })?;
        Self::load(config, &file)
    }

    /// Loads weights from an open safetensors file. Keys follow the
    /// reference checkpoint tree with an optional leading `model.`
    /// component; both spellings resolve.
    pub fn load(
        config: MossFormer2SeConfig,
        file: &turbospark_model_io::safetensors::SafetensorsFile,
    ) -> Result<Self> {
        let dim = config.out_channels;
        let in_ch = config.in_channels;
        let blocks = config.num_blocks;

        // Checkpoints store the network under an optional prefix
        // ("", "mossformer.", "model.mossformer."); linears append
        // ".weight" themselves via the quantized loader, while direct
        // tensors (norms, convs) are stored with the ".weight" suffix.
        let prefixes = ["", "mossformer.", "model."];
        let resolve_base = |name: &str| -> String {
            for prefix in prefixes {
                let candidate = format!("{prefix}{name}");
                // The quantized loader appends ".weight" (and looks for
                // ".scales"), so match through those suffixes.
                if file.contains_tensor(&candidate)
                    || file.contains_tensor(&format!("{candidate}.weight"))
                    || file.contains_tensor(&format!("{candidate}.scales"))
                {
                    return candidate;
                }
            }
            name.to_string()
        };
        let resolve_weight = |name: &str| -> String {
            for prefix in prefixes {
                let candidate = format!("{prefix}{name}.weight");
                if file.contains_tensor(&candidate) {
                    return candidate;
                }
            }
            for prefix in prefixes {
                let candidate = format!("{prefix}{name}");
                if file.contains_tensor(&candidate) {
                    return candidate;
                }
            }
            format!("{name}.weight")
        };
        let f32_tensor = |name: &str| -> Result<Vec<f32>> {
            let resolved = resolve_weight(name);
            file.load_as_f32(&resolved)
                .map_err(|e| SpeechError::Tensor {
                    name: resolved,
                    why: e.to_string(),
                })
        };
        let linear = |name: &str| -> Result<Linear> {
            let (weight, bias) = quant::load_quantized(
                file,
                &resolve_base(name),
                QuantScheme {
                    bits: 8,
                    group_size: 64,
                },
            )
            .map_err(|e| SpeechError::Tensor {
                name: resolve_base(name),
                why: e.to_string(),
            })?;
            let desc =
                file.descriptor(&resolve_weight(name))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: resolve_base(name),
                        why: "weight descriptor missing".to_string(),
                    })?;
            // Plain weights are [out, 1, in] (MLX 1-D conv layout) or
            // [out, in]; quantized weights dequantize to [out, in].
            let (out_dim, in_dim) = match desc.shape.len() {
                2 => (desc.shape[0], desc.shape[1]),
                3 if desc.shape[1] == 1 => (desc.shape[0], desc.shape[2]),
                _ => {
                    return Err(SpeechError::Tensor {
                        name: resolve_base(name),
                        why: format!("unsupported weight shape {:?}", desc.shape),
                    })
                }
            };
            if weight.len() != out_dim * in_dim {
                return Err(SpeechError::Tensor {
                    name: resolve_base(name),
                    why: format!("weight length {} != {}x{}", weight.len(), out_dim, in_dim),
                });
            }
            Ok(Linear {
                weight,
                bias,
                out_dim,
                in_dim,
            })
        };
        // Conv1d/Conv2d weights over time convert from [C, K, ...] to
        // [K, C] for the depthwise kernels.
        let time_conv = |name: &str, kernel: usize, channels: usize| -> Result<TimeDepthwiseConv> {
            let raw = f32_tensor(&format!("{name}.weight"))?;
            if raw.len() != kernel * channels {
                return Err(SpeechError::Tensor {
                    name: resolve_base(name),
                    why: format!(
                        "depthwise weight length {} != {}x{}",
                        raw.len(),
                        kernel,
                        channels
                    ),
                });
            }
            // Stored [C, K] row-major (trailing 1s squeezed).
            let mut weight = vec![0.0f32; kernel * channels];
            for c in 0..channels {
                for k in 0..kernel {
                    weight[k * channels + c] = raw[c * kernel + k];
                }
            }
            Ok(TimeDepthwiseConv {
                weight,
                kernel,
                channels,
            })
        };
        let ffc = |prefix: &str, scale_norm: bool| -> Result<FfConvM> {
            let linear = linear(&format!("{prefix}.linear"))?;
            let norm = if scale_norm {
                let g = f32_tensor(&format!("{prefix}.norm.g"))?;
                if g.len() != 1 {
                    return Err(SpeechError::Tensor {
                        name: resolve_base(&format!("{prefix}.norm.g")),
                        why: "expected a scalar".to_string(),
                    });
                }
                Norm::ScaleNorm { g: g[0] }
            } else {
                let weight = f32_tensor(&format!("{prefix}.norm.weight"))?;
                let bias = f32_tensor(&format!("{prefix}.norm.bias"))?;
                Norm::LayerNorm { weight, bias }
            };
            let dim_in = linear.in_dim;
            let dim_out = linear.out_dim;
            let conv = time_conv(&format!("{prefix}.conv_module"), 17, dim_out)?;
            let _ = dim_in;
            Ok(FfConvM { norm, linear, conv })
        };

        let scalar = |name: &str| -> Result<f32> {
            let v = f32_tensor(name)?;
            if v.len() != 1 {
                return Err(SpeechError::Tensor {
                    name: resolve_base(name),
                    why: "expected a scalar".to_string(),
                });
            }
            Ok(v[0])
        };
        let vector = |name: &str, len: usize| -> Result<Vec<f32>> {
            let v = f32_tensor(name)?;
            if v.len() != len {
                return Err(SpeechError::Tensor {
                    name: resolve_base(name),
                    why: format!("expected {len} values, got {}", v.len()),
                });
            }
            Ok(v)
        };

        // Global layer norm weights are stored [C, 1]; squeeze.
        let gln_raw = f32_tensor("mossformer.norm.weight")?;
        let gln_bias_raw = f32_tensor("mossformer.norm.bias")?;
        if gln_raw.len() != in_ch || gln_bias_raw.len() != in_ch {
            return Err(SpeechError::Tensor {
                name: resolve_weight("mossformer.norm.weight"),
                why: format!("expected {in_ch} channels, got {}", gln_raw.len()),
            });
        }
        let gln = GlobalLayerNorm {
            weight: gln_raw,
            bias: gln_bias_raw,
            channels: in_ch,
        };
        let encoder = linear("mossformer.conv1d_encoder")?;
        if encoder.bias.is_some() {
            return Err(SpeechError::Tensor {
                name: resolve_base("mossformer.conv1d_encoder"),
                why: "the reference encoder conv has no bias".to_string(),
            });
        }
        let pos_scale = scalar("mossformer.pos_enc.scale")?;
        let pos_inv_freq: Vec<f32> = (0..dim / 2)
            .map(|j| 10000f32.powf(-((2 * j) as f32) / dim as f32))
            .collect();

        let mut layers = Vec::with_capacity(blocks);
        let mut fsmns = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let lp = format!("mossformer.mdl.intra_mdl.mossformerM.layers.{i}");
            let to_hidden = ffc(&format!("{lp}.to_hidden"), true)?;
            let to_qk = ffc(&format!("{lp}.to_qk"), true)?;
            let to_out = ffc(&format!("{lp}.to_out"), true)?;
            let gamma = vector(&format!("{lp}.qk_offset_scale.gamma"), 4 * QUERY_KEY_DIM)?;
            let beta = vector(&format!("{lp}.qk_offset_scale.beta"), 4 * QUERY_KEY_DIM)?;
            layers.push(FlashLayer {
                to_hidden,
                to_qk,
                to_out,
                offset_scale: OffsetScale {
                    gamma,
                    beta,
                    heads: 4,
                    dim: QUERY_KEY_DIM,
                },
            });

            let fp = format!("mossformer.mdl.intra_mdl.mossformerM.fsmn.{i}");
            let conv1 = linear(&format!("{fp}.conv1"))?;
            let conv2 = linear(&format!("{fp}.conv2"))?;
            let gated_to_u = ffc(&format!("{fp}.gated_fsmn.to_u"), false)?;
            let gated_to_v = ffc(&format!("{fp}.gated_fsmn.to_v"), false)?;
            let fsmn_linear = linear(&format!("{fp}.gated_fsmn.fsmn.linear"))?;
            let fsmn_project = linear(&format!("{fp}.gated_fsmn.fsmn.project"))?;
            let fsmn_conv = time_conv(
                &format!("{fp}.gated_fsmn.fsmn.conv1"),
                2 * FSMN_LORDER - 1,
                INNER_CHANNELS,
            )?;
            fsmns.push(GatedFsmnBlock {
                prelu: scalar(&format!("{fp}.prelu.weight"))?,
                norm1: (
                    vector(&format!("{fp}.norm1.weight"), INNER_CHANNELS)?,
                    vector(&format!("{fp}.norm1.bias"), INNER_CHANNELS)?,
                ),
                norm2: (
                    vector(&format!("{fp}.norm2.weight"), INNER_CHANNELS)?,
                    vector(&format!("{fp}.norm2.bias"), INNER_CHANNELS)?,
                ),
                gated_fsmn: GatedFsmn {
                    to_u: gated_to_u,
                    to_v: gated_to_v,
                    fsmn: UniDeepFsmn {
                        linear: fsmn_linear,
                        project: fsmn_project,
                        conv: fsmn_conv,
                    },
                },
                conv1,
                conv2,
            });
        }
        let mdl_norm = (
            vector("mossformer.mdl.intra_mdl.norm.weight", dim)?,
            vector("mossformer.mdl.intra_mdl.norm.bias", dim)?,
        );
        let intra_norm = (
            vector("mossformer.mdl.intra_norm.weight", dim)?,
            vector("mossformer.mdl.intra_norm.bias", dim)?,
        );
        let spk_conv = linear("mossformer.conv1d_out")?;
        let decoder = linear("mossformer.conv1_decoder")?;
        if decoder.bias.is_some() {
            return Err(SpeechError::Tensor {
                name: resolve_base("mossformer.conv1_decoder"),
                why: "the reference decoder conv has no bias".to_string(),
            });
        }
        let num_spks = spk_conv.out_dim / dim;
        let masknet = MossFormerMaskNet {
            gln,
            encoder,
            pos_scale,
            pos_inv_freq,
            mdl: MossFormerM {
                layers,
                fsmns,
                norm: mdl_norm,
            },
            intra_norm,
            prelu: scalar("mossformer.prelu.weight")?,
            spk_conv,
            output: linear("mossformer.output")?,
            output_gate: linear("mossformer.output_gate")?,
            decoder,
            num_spks,
            out_channels: dim,
        };

        let window = hamming_window(config.win_len);
        Ok(Self {
            config,
            masknet,
            window,
        })
    }

    /// Kaldi-compatible log mel-filterbank features `[frames, mels]`
    /// with snip_edges framing and the given dither. Only dither = 0 is
    /// supported (the reference default of 1.0 is stochastic).
    pub fn kaldi_fbank(&self, audio: &[f32], dither: f32) -> Result<Vec<f32>> {
        if dither != 0.0 {
            return Err(SpeechError::Unsupported {
                why: "nonzero Kaldi fbank dither requires MLX RNG parity and is refused"
                    .to_string(),
            });
        }
        let sample_rate = self.config.sample_rate;
        let window_size = self.config.win_len;
        let window_shift = self.config.win_inc;
        let padded = window_size.next_power_of_two();
        if audio.len() < window_size {
            return Ok(Vec::new());
        }
        let frames = 1 + (audio.len() - window_size) / window_shift;

        // Mel filterbank over the first padded/2 FFT bins (Kaldi drops
        // the Nyquist bin; a zero column is appended to match rfft).
        let fft_bin_width = sample_rate as f32 / padded as f32;
        let num_fft_bins = padded / 2;
        let nyquist = 0.5 * sample_rate as f32;
        let high_freq = nyquist; // high_freq <= 0 resolves to Nyquist
        let mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
        let _inv_mel = |m: f32| 700.0 * ((m / 1127.0).exp() - 1.0);
        let mel_low = mel(20.0);
        let mel_high = mel(high_freq);
        let mel_delta = (mel_high - mel_low) / (self.config.num_mels + 1) as f32;
        let mut banks = vec![0.0f32; self.config.num_mels * num_fft_bins];
        for bin in 0..self.config.num_mels {
            let left = mel_low + bin as f32 * mel_delta;
            let center = mel_low + (bin as f32 + 1.0) * mel_delta;
            let right = mel_low + (bin as f32 + 2.0) * mel_delta;
            for m in 0..num_fft_bins {
                let mel_m = mel(fft_bin_width * m as f32);
                let up = (mel_m - left) / (center - left);
                let down = (right - mel_m) / (right - center);
                banks[bin * num_fft_bins + m] = up.min(down).max(0.0);
            }
        }

        let plan = RealFftPlan::cached(padded).map_err(|e| SpeechError::Input {
            why: format!("fbank fft plan: {e}"),
        })?;
        let mut features = vec![0.0f32; frames * self.config.num_mels];
        let mut frame = vec![0.0f32; padded];
        for f in 0..frames {
            let start = f * window_shift;
            // DC removal.
            let mut mean = 0.0f32;
            for value in &audio[start..start + window_size] {
                mean += value;
            }
            mean /= window_size as f32;
            // Pre-emphasis keeps the first sample, subtracts 0.97*x[i-1].
            let mut prev = audio[start] - mean;
            frame[0] = prev;
            for (i, &value) in audio[start + 1..start + window_size].iter().enumerate() {
                let cur = value - mean;
                frame[i + 1] = cur - self.config.preemphasis * prev;
                prev = cur;
            }
            // Hamming analysis window.
            for (i, value) in frame.iter_mut().enumerate().take(window_size) {
                *value *= self.window[i];
            }
            for value in frame.iter_mut().skip(window_size) {
                *value = 0.0;
            }
            let spec = plan.forward(&frame).map_err(|e| SpeechError::Input {
                why: format!("fbank fft: {e}"),
            })?;
            let out_row = &mut features[f * self.config.num_mels..(f + 1) * self.config.num_mels];
            for bin in 0..self.config.num_mels {
                let mut energy = 0.0f32;
                let bank_row = &banks[bin * num_fft_bins..(bin + 1) * num_fft_bins];
                for (m, &weight) in bank_row.iter().enumerate() {
                    if weight != 0.0 {
                        let re = spec[m].re;
                        let im = spec[m].im;
                        energy += weight * (re * re + im * im);
                    }
                }
                out_row[bin] = energy.max(1e-8).ln();
            }
        }
        Ok(features)
    }

    /// Kaldi delta features along time with edge padding; input is
    /// `[features, frames]` transposed by the caller as in the
    /// reference (the deltas run across time per feature row).
    pub fn kaldi_deltas(
        specgram: &[f32],
        rows: usize,
        frames: usize,
        win_length: usize,
    ) -> Vec<f32> {
        let n = (win_length - 1) / 2;
        let denom = (n * (n + 1) * (2 * n + 1)) as f32 / 3.0;
        let kernel: Vec<f32> = (-(n as isize)..=(n as isize)).map(|k| k as f32).collect();
        let mut out = vec![0.0f32; rows * frames];
        for r in 0..rows {
            for f in 0..frames {
                let mut acc = 0.0f32;
                for (k, &weight) in kernel.iter().enumerate() {
                    let src = f as isize + k as isize - n as isize;
                    let src = src.clamp(0, frames as isize - 1) as usize;
                    acc += weight * specgram[r * frames + src];
                }
                out[r * frames + f] = acc / denom;
            }
        }
        out
    }

    /// Forward STFT (center=False, unnormalized rfft per frame) of the
    /// scaled audio; returns per-frame spectra.
    pub fn stft_frames(&self, audio: &[f32]) -> Result<Vec<Vec<crate::fft::ComplexF32>>> {
        let win = self.config.win_len;
        let hop = self.config.win_inc;
        if audio.len() < win {
            return Err(SpeechError::Input {
                why: format!("audio length {} shorter than window {}", audio.len(), win),
            });
        }
        let frames = 1 + (audio.len() - win) / hop;
        let plan = RealFftPlan::cached(win).map_err(|e| SpeechError::Input {
            why: format!("stft fft plan: {e}"),
        })?;
        let mut out = Vec::with_capacity(frames);
        let mut frame = vec![0.0f32; win];
        for f in 0..frames {
            let start = f * hop;
            for (i, value) in frame.iter_mut().enumerate() {
                *value = audio[start + i] * self.window[i];
            }
            let spec = plan.forward(&frame).map_err(|e| SpeechError::Input {
                why: format!("stft fft: {e}"),
            })?;
            out.push(spec);
        }
        Ok(out)
    }

    /// Cached-style iSTFT: irfft per frame, synthesis window, overlap
    /// add normalized by the window-squared sum floored at 1e-10, then
    /// trimmed to `audio_length`.
    pub fn istft(
        &self,
        spectra: &[Vec<crate::fft::ComplexF32>],
        audio_length: usize,
    ) -> Result<Vec<f32>> {
        let n_fft = self.config.fft_len;
        let hop = self.config.win_inc;
        let plan = RealFftPlan::cached(n_fft).map_err(|e| SpeechError::Input {
            why: format!("istft fft plan: {e}"),
        })?;
        let total = (spectra.len() - 1) * hop + n_fft;
        let mut acc = vec![0.0f32; total];
        let mut norm = vec![0.0f32; total];
        for (f, spectrum) in spectra.iter().enumerate() {
            let samples = plan.inverse(spectrum).map_err(|e| SpeechError::Input {
                why: format!("istft fft: {e}"),
            })?;
            let offset = f * hop;
            for (i, &sample) in samples.iter().enumerate() {
                acc[offset + i] += sample * self.window[i];
                norm[offset + i] += self.window[i] * self.window[i];
            }
        }
        let mut out = vec![0.0f32; audio_length.min(total)];
        for (i, value) in out.iter_mut().enumerate() {
            *value = acc[i] / norm[i].max(1e-10);
        }
        Ok(out)
    }

    /// Core chunk processing: features, mask, masked STFT, iSTFT.
    /// `audio` is the scaled (x32768) mono chunk.
    pub fn process_chunk(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let fbank = self.kaldi_fbank(audio, 0.0)?;
        let frames = audio
            .len()
            .checked_sub(self.config.win_len)
            .map(|n| 1 + n / self.config.win_inc)
            .unwrap_or(0);
        if frames == 0 || fbank.is_empty() {
            return Err(SpeechError::Input {
                why: format!(
                    "audio length {} too short for win_len {}",
                    audio.len(),
                    self.config.win_len
                ),
            });
        }
        let mels = self.config.num_mels;
        // Deltas across time: transpose to [mels, frames].
        let mut fbank_t = vec![0.0f32; frames * mels];
        for f in 0..frames {
            for c in 0..mels {
                fbank_t[c * frames + f] = fbank[f * mels + c];
            }
        }
        let delta_t = Self::kaldi_deltas(&fbank_t, mels, frames, 5);
        let ddelta_t = Self::kaldi_deltas(&delta_t, mels, frames, 5);
        let mut features = vec![0.0f32; frames * 3 * mels];
        for f in 0..frames {
            for c in 0..mels {
                features[f * 3 * mels + c] = fbank[f * mels + c];
                features[f * 3 * mels + mels + c] = delta_t[c * frames + f];
                features[f * 3 * mels + 2 * mels + c] = ddelta_t[c * frames + f];
            }
        }
        let mask = self.masknet.forward(&features, frames); // [frames, 961]

        let spectra = self.stft_frames(audio)?;
        let freqs = self.config.fft_len / 2 + 1;
        let mut masked: Vec<Vec<crate::fft::ComplexF32>> = Vec::with_capacity(spectra.len());
        for (f, spectrum) in spectra.iter().enumerate() {
            let row: Vec<crate::fft::ComplexF32> = spectrum
                .iter()
                .enumerate()
                .map(|(b, value)| {
                    let m = mask[f * freqs + b];
                    crate::fft::ComplexF32::new(value.re * m, value.im * m)
                })
                .collect();
            masked.push(row);
        }
        self.istft(&masked, audio.len())
    }

    /// Enhances mono audio. Durations above `one_time_decode_length`
    /// (20 s) use the reference segmented overlap-discarding path.
    pub fn enhance(&self, audio: &[f32]) -> Result<Vec<f32>> {
        self.enhance_with_mode(audio, None)
    }

    /// As [`enhance`](Self::enhance) with an explicit chunked override
    /// (`Some(true)` forces the >60 s chunked path regardless of
    /// duration, `Some(false)` disables it).
    pub fn enhance_with_mode(&self, audio: &[f32], chunked: Option<bool>) -> Result<Vec<f32>> {
        let original_len = audio.len();
        let scaled: Vec<f32> = audio.iter().map(|&v| v * MAX_WAV_VALUE).collect();
        let use_chunked = chunked.unwrap_or_else(|| {
            (original_len as f32 / self.config.sample_rate as f32)
                >= self.config.auto_chunk_threshold
        });
        let mut out = if use_chunked {
            self.decode_chunked(&scaled)?
        } else {
            self.decode_one_audio(&scaled)?
        };
        out.truncate(original_len);
        for value in out.iter_mut() {
            *value /= MAX_WAV_VALUE;
        }
        Ok(out)
    }

    fn decode_one_audio(&self, scaled: &[f32]) -> Result<Vec<f32>> {
        let limit = self.config.sample_rate * self.config.one_time_decode_length;
        if scaled.len() > limit {
            return self.decode_segmented(scaled);
        }
        self.process_chunk(scaled)
    }

    /// Reference segmented path: 4 s windows at 75 % stride, discarding
    /// `(window - stride) / 2` samples at each segment edge.
    fn decode_segmented(&self, scaled: &[f32]) -> Result<Vec<f32>> {
        let window_size = self.config.sample_rate * self.config.decode_window;
        let stride = window_size * 3 / 4;
        let mut audio = scaled.to_vec();
        let t = audio.len();
        if t < window_size + stride {
            audio.resize(window_size + stride, 0.0);
        } else if (t - window_size) % stride != 0 {
            let padding = t - (t - window_size) / stride * stride;
            audio.resize(t + padding, 0.0);
        }
        let total = audio.len();
        let give_up = (window_size - stride) / 2;
        let mut output = vec![0.0f32; total];
        let mut current = 0usize;
        while current + window_size <= total {
            let segment = self.process_chunk(&audio[current..current + window_size])?;
            let (start, end) = if current == 0 {
                (current, current + window_size - give_up)
            } else {
                (current + give_up, current + window_size - give_up)
            };
            output[start..end].copy_from_slice(&segment[start - current..end - current]);
            current += stride;
        }
        output.truncate(scaled.len());
        Ok(output)
    }

    /// Reference chunked path: 4 s chunks at 25 % overlap with
    /// discard-edges reassembly.
    fn decode_chunked(&self, scaled: &[f32]) -> Result<Vec<f32>> {
        let chunk_samples = (self.config.sample_rate as f32 * self.config.chunk_seconds) as usize;
        let overlap_samples = (chunk_samples as f32 * self.config.chunk_overlap) as usize;
        let stride = chunk_samples - overlap_samples;
        let give_up = overlap_samples / 2;
        let original_len = scaled.len();
        if original_len <= chunk_samples {
            return self.process_chunk(scaled);
        }
        let mut output = vec![0.0f32; original_len];
        let mut current = 0usize;
        let mut is_first = true;
        while current < original_len {
            let end = (current + chunk_samples).min(original_len);
            let segment = self.process_chunk(&scaled[current..end])?;
            let chunk_len = segment.len();
            let is_last = current + chunk_samples >= original_len;
            let (keep_start, keep_end) = if is_last && chunk_len < chunk_samples {
                (if is_first { 0 } else { give_up }, chunk_len)
            } else {
                (if is_first { 0 } else { give_up }, chunk_len - give_up)
            };
            let output_start = current + keep_start;
            let output_end = (current + keep_end).min(original_len);
            if output_end > output_start {
                output[output_start..output_end].copy_from_slice(
                    &segment[keep_start..keep_start + (output_end - output_start)],
                );
            }
            is_first = false;
            current += stride;
        }
        Ok(output)
    }
}

/// Hamming window with the reference periodic=False convention
/// (`0.54 - 0.46 cos(2 pi n / (size - 1))`).
fn hamming_window(size: usize) -> Vec<f32> {
    let denom = (size - 1) as f32;
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / denom).cos())
        .collect()
}

#[cfg(test)]
mod tests;
