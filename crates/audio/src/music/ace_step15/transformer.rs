//! Shared transformer pieces of the ACE-Step 1.5 DiT checkpoint: the
//! config, Qwen3-style attention with per-head q/k RMSNorm and GQA, the
//! SwiGLU MLP, the bidirectional encoder layer, and HF default RoPE.
//!
//! Reference: `modeling_acestep_v15_turbo.py` at the pinned checkpoint
//! revision (`AceStepAttention`, `AceStepEncoderLayer`, `create_4d_mask`)
//! and the transformers Qwen3 helpers it imports.
//!
//! Layout: activations are time-major `[seq, width]`, one batch item,
//! as in the reference `[B, T, D]` with B = 1. Attention masks are never
//! padding masks here: callers refuse padded inputs, and every sequence
//! is either fully visible or a symmetric sliding window.
//!
//! Device seam: a [`Loader`] with a [`ComputeBackend`] keeps every large
//! projection resident on the device (`Proj::Device`) and runs attention
//! there whenever no window mask applies. Norms, RoPE, residuals and
//! windowed attention stay on the host. Windowed attention runs banded
//! (only the keys inside the window), which equals the dense masked
//! softmax exactly because masked keys contribute exact zeros.

use std::rc::Rc;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::backend::{
    AttentionShape, ComputeBackend, DType, DeviceWeight, WeightData, WeightEncoding,
};
use crate::nn::{load_tensor, Linear, RmsNorm};
use crate::{ops, Result, SpeechError};

/// Width and depth bounds for an untrusted `config.json`. The shipped
/// turbo config is hidden 2048, 24 layers, 16 heads.
const MAX_WIDTH: usize = 16384;
const MAX_LAYERS: usize = 256;
const MAX_HEADS: usize = 256;

/// Attention span of one layer, from the config `layer_types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Every key is visible.
    Full,
    /// Keys with `|i - j| <= window` are visible (bidirectional).
    Sliding,
}

/// The fields of `AceStepConfig` the ported modules read.
#[derive(Debug, Clone, PartialEq)]
pub struct AceConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_lyric_encoder_hidden_layers: usize,
    pub num_timbre_encoder_hidden_layers: usize,
    pub num_attention_pooler_hidden_layers: usize,
    pub text_hidden_dim: usize,
    pub timbre_hidden_dim: usize,
    pub audio_acoustic_hidden_dim: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub pool_window_size: usize,
    /// FSQ levels per code dimension (`fsq_input_levels`); one quantizer.
    pub fsq_levels: Vec<usize>,
    /// `None` when `use_sliding_window` is false (every layer is full).
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<LayerKind>,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub is_turbo: bool,
}

impl AceConfig {
    /// Parses the checkpoint `config.json`. Every geometry field is
    /// required: the reference defaults are the shipped model, and a
    /// silently defaulted field would load the wrong network.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let bad = |field: &str, why: String| SpeechError::BadConfig {
            field: field.to_string(),
            why,
        };
        let num = |field: &str, max: usize| -> Result<usize> {
            let n = v
                .get(field)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| bad(field, "missing or not an integer".into()))?
                as usize;
            if n == 0 || n > max {
                return Err(bad(field, format!("{n} outside 1..={max}")));
            }
            Ok(n)
        };
        let float = |field: &str| -> Result<f32> {
            let x = v
                .get(field)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| bad(field, "missing or not a number".into()))?;
            if !(x.is_finite() && x > 0.0) {
                return Err(bad(field, format!("{x} must be finite and positive")));
            }
            Ok(x as f32)
        };
        let use_sliding = v
            .get("use_sliding_window")
            .and_then(|x| x.as_bool())
            .unwrap_or(true);
        let sliding_window = if use_sliding {
            Some(num("sliding_window", 1 << 20)?)
        } else {
            None
        };
        let types = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| bad("layer_types", "missing or not a list".into()))?;
        let layer_types = types
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(LayerKind::Full),
                Some("sliding_attention") => Ok(LayerKind::Sliding),
                other => Err(bad("layer_types", format!("unknown entry {other:?}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope_theta = match v.get("rope_parameters").and_then(|p| p.get("rope_theta")) {
            Some(x) => x
                .as_f64()
                .filter(|t| t.is_finite() && *t > 0.0)
                .ok_or_else(|| bad("rope_parameters.rope_theta", "not a positive number".into()))?
                as f32,
            None => float("rope_theta")?,
        };
        let fsq_levels = v
            .get("fsq_input_levels")
            .and_then(|x| x.as_array())
            .ok_or_else(|| bad("fsq_input_levels", "missing or not a list".into()))?
            .iter()
            .map(|l| match l.as_u64() {
                // Levels of 2 need the symmetric bound, which the shipped
                // config does not use; 256 bounds the codebook product.
                Some(n) if (3..=256).contains(&n) => Ok(n as usize),
                _ => Err(bad("fsq_input_levels", "entries must be in 3..=256".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        if fsq_levels.is_empty() || fsq_levels.len() > 16 {
            return Err(bad("fsq_input_levels", "expected 1..=16 levels".into()));
        }
        let quantizers = num("fsq_input_num_quantizers", 64)?;
        if quantizers != 1 {
            return Err(bad(
                "fsq_input_num_quantizers",
                format!("{quantizers}: only the single-quantizer FSQ is ported"),
            ));
        }
        if let Some(scaling) = v.get("rope_scaling").filter(|s| !s.is_null()) {
            return Err(bad("rope_scaling", format!("unsupported {scaling}")));
        }
        let c = Self {
            hidden_size: num("hidden_size", MAX_WIDTH)?,
            intermediate_size: num("intermediate_size", 4 * MAX_WIDTH)?,
            num_hidden_layers: num("num_hidden_layers", MAX_LAYERS)?,
            num_attention_heads: num("num_attention_heads", MAX_HEADS)?,
            num_key_value_heads: num("num_key_value_heads", MAX_HEADS)?,
            head_dim: num("head_dim", 1024)?,
            num_lyric_encoder_hidden_layers: num("num_lyric_encoder_hidden_layers", MAX_LAYERS)?,
            num_timbre_encoder_hidden_layers: num("num_timbre_encoder_hidden_layers", MAX_LAYERS)?,
            num_attention_pooler_hidden_layers: num(
                "num_attention_pooler_hidden_layers",
                MAX_LAYERS,
            )?,
            text_hidden_dim: num("text_hidden_dim", MAX_WIDTH)?,
            timbre_hidden_dim: num("timbre_hidden_dim", MAX_WIDTH)?,
            audio_acoustic_hidden_dim: num("audio_acoustic_hidden_dim", MAX_WIDTH)?,
            in_channels: num("in_channels", MAX_WIDTH)?,
            patch_size: num("patch_size", 64)?,
            pool_window_size: num("pool_window_size", 64)?,
            fsq_levels,
            sliding_window,
            layer_types,
            rope_theta,
            rms_norm_eps: float("rms_norm_eps")?,
            is_turbo: v.get("is_turbo").and_then(|x| x.as_bool()).unwrap_or(false),
        };
        if c.num_attention_heads % c.num_key_value_heads != 0 {
            return Err(bad(
                "num_key_value_heads",
                "must divide num_attention_heads".into(),
            ));
        }
        // The quantizer projects the pooler output, which is hidden wide.
        if num("fsq_dim", MAX_WIDTH)? != c.hidden_size {
            return Err(bad("fsq_dim", "must equal hidden_size".into()));
        }
        if c.head_dim % 2 != 0 {
            return Err(bad(
                "head_dim",
                "rotary embedding needs an even width".into(),
            ));
        }
        // The DiT input is [context latents (source + chunk mask), noisy
        // latents], three acoustic widths.
        if c.in_channels != 3 * c.audio_acoustic_hidden_dim {
            return Err(bad(
                "in_channels",
                "must be 3 x audio_acoustic_hidden_dim".into(),
            ));
        }
        let deepest = c
            .num_hidden_layers
            .max(c.num_lyric_encoder_hidden_layers)
            .max(c.num_timbre_encoder_hidden_layers)
            .max(c.num_attention_pooler_hidden_layers);
        if c.layer_types.len() < deepest {
            return Err(bad(
                "layer_types",
                format!("{} entries, need {deepest}", c.layer_types.len()),
            ));
        }
        Ok(c)
    }

    /// The span a layer at `idx` uses: the reference indexes
    /// `config.layer_types[idx]` in every stack, and `use_sliding_window
    /// = false` makes the sliding mask `None`, which the encoders then
    /// treat as unmasked.
    pub fn span(&self, idx: usize) -> Option<usize> {
        match (self.layer_types[idx], self.sliding_window) {
            (LayerKind::Sliding, Some(w)) => Some(w),
            _ => None,
        }
    }
}

/// HF default rotary tables for positions `0..seq`: `inv_freq_i = 1 /
/// theta^(2i / dim)` in f32, angle = position * inv_freq, as
/// `Qwen3RotaryEmbedding` computes them. Returns `(cos, sin)`, each
/// `[seq, dim / 2]`.
pub fn rope_tables(seq: usize, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        for (i, f) in inv.iter().enumerate() {
            let a = t as f32 * f;
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// Additive `[seq, seq]` mask for a symmetric window: 0 where
/// `|i - j| <= window`, `f32::MIN` elsewhere (the reference fills with
/// `finfo(dtype).min`, not `-inf`).
pub fn window_mask(seq: usize, window: usize) -> Vec<f32> {
    let mut m = vec![f32::MIN; seq * seq];
    for i in 0..seq {
        let lo = i.saturating_sub(window);
        let hi = (i + window).min(seq - 1);
        m[i * seq + lo..=i * seq + hi].fill(0.0);
    }
    m
}

/// Loads checkpoint tensors, optionally as device-resident projections.
#[derive(Clone)]
pub struct Loader<'a> {
    pub file: &'a SafetensorsFile,
    pub backend: Option<Rc<dyn ComputeBackend>>,
}

impl<'a> Loader<'a> {
    /// Host-only loading: every projection is a CPU [`Linear`].
    pub fn cpu(file: &'a SafetensorsFile) -> Self {
        Self {
            file,
            backend: None,
        }
    }

    /// Projections become device weights on `backend`; attention without a
    /// window mask runs there too.
    pub fn with_backend(file: &'a SafetensorsFile, backend: Rc<dyn ComputeBackend>) -> Self {
        Self {
            file,
            backend: Some(backend),
        }
    }
}

/// A projection `y = x @ w.T + b`, on the host or resident on a device.
#[derive(Clone)]
pub enum Proj {
    Cpu(Linear),
    Device {
        weight: Rc<dyn DeviceWeight>,
        bias: Option<Vec<f32>>,
        input: usize,
        output: usize,
    },
}

impl std::fmt::Debug for Proj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Proj::Cpu(l) => write!(f, "Proj::Cpu({}x{})", l.output, l.input),
            Proj::Device { input, output, .. } => write!(f, "Proj::Device({output}x{input})"),
        }
    }
}

impl Proj {
    /// `x [rows, input] -> [rows, output]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        match self {
            Proj::Cpu(l) => Ok(l.forward(x, rows)),
            Proj::Device {
                weight,
                bias,
                input,
                output,
            } => weight.linear(x, bias.as_deref(), rows, *input, *output, DType::F32),
        }
    }
}

/// The checkpoint dtype of `name` as a device weight encoding.
fn encoding(file: &SafetensorsFile, name: &str) -> Result<WeightEncoding> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "tensor is missing".into(),
    })?;
    match desc.dtype.as_str() {
        "BF16" => Ok(WeightEncoding::Bf16),
        "F16" => Ok(WeightEncoding::F16),
        "F32" => Ok(WeightEncoding::F32),
        other => Err(SpeechError::Unsupported {
            why: format!("{name}: dtype {other} has no device encoding"),
        }),
    }
}

pub(crate) fn linear(
    l: &Loader<'_>,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Proj> {
    let Some(backend) = &l.backend else {
        return Ok(Proj::Cpu(Linear::load(l.file, name, input, output, bias)?));
    };
    let wname = format!("{name}.weight");
    let shape = [output, input];
    match l.file.descriptor(&wname) {
        Some(d) if d.shape == shape => {}
        Some(d) => {
            return Err(SpeechError::Tensor {
                name: wname,
                why: format!("expected shape {shape:?}, got {:?}", d.shape),
            })
        }
        None => {
            return Err(SpeechError::Tensor {
                name: wname,
                why: "tensor is missing".into(),
            })
        }
    }
    let weight = backend.load_weight(WeightData {
        name: &wname,
        shape: &shape,
        bytes: l.file.raw_bytes(&wname)?,
        encoding: encoding(l.file, &wname)?,
        dtype: DType::F32,
        scales_dtype: None,
        offsets_dtype: None,
        scales: &[],
        offsets: &[],
        block_scales: &[],
        planned_rows: None,
        // The reference is PyTorch's f32 chain on widened bf16, not an
        // MLX kernel sequence, so tiled float32 accumulation is allowed.
        tiled_f32: true,
    })?;
    let bias = if bias {
        Some(load_tensor(l.file, &format!("{name}.bias"), &[output])?)
    } else {
        None
    };
    Ok(Proj::Device {
        weight,
        bias,
        input,
        output,
    })
}

pub(crate) fn rms(l: &Loader<'_>, name: &str, width: usize, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        load_tensor(l.file, &format!("{name}.weight"), &[width])?,
        eps,
    ))
}

/// The window a layer must enforce at this sequence length: `None` when
/// the layer is full or the window already covers every key.
pub fn active_window(span: Option<usize>, seq: usize) -> Option<usize> {
    span.filter(|&w| w + 1 < seq)
}

/// NeoX rotary on time-major `[seq, heads, dim]` (half-split pairs), the
/// same arithmetic as `ops::rope_neox` on head-major data.
fn rope_time_major(x: &mut [f32], seq: usize, heads: usize, dim: usize, cos: &[f32], sin: &[f32]) {
    let half = dim / 2;
    for t in 0..seq {
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let base = (t * heads + h) * dim;
            let (lo, hi) = x[base..base + dim].split_at_mut(half);
            for d in 0..half {
                let a = lo[d];
                let b = hi[d];
                lo[d] = a * c[d] - b * s[d];
                hi[d] = b * c[d] + a * s[d];
            }
        }
    }
}

/// Query rows per block in banded attention. Keys per block are at most
/// `BAND_ROWS + 2 * window`.
const BAND_ROWS: usize = 32;

/// `AceStepAttention` for one sequence: per-head RMSNorm on q and k,
/// RoPE on self-attention only, GQA by repeating each KV head across
/// `heads / kv_heads` query heads.
#[derive(Clone)]
pub struct Attention {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    backend: Option<Rc<dyn ComputeBackend>>,
}

impl Attention {
    pub fn load(l: &Loader<'_>, prefix: &str, c: &AceConfig) -> Result<Self> {
        let (d, hd) = (c.hidden_size, c.head_dim);
        let (h, kv) = (c.num_attention_heads, c.num_key_value_heads);
        Ok(Self {
            q: linear(l, &format!("{prefix}.q_proj"), d, h * hd, false)?,
            k: linear(l, &format!("{prefix}.k_proj"), d, kv * hd, false)?,
            v: linear(l, &format!("{prefix}.v_proj"), d, kv * hd, false)?,
            o: linear(l, &format!("{prefix}.o_proj"), h * hd, d, false)?,
            q_norm: rms(l, &format!("{prefix}.q_norm"), hd, c.rms_norm_eps)?,
            k_norm: rms(l, &format!("{prefix}.k_norm"), hd, c.rms_norm_eps)?,
            heads: h,
            kv_heads: kv,
            head_dim: hd,
            backend: l.backend.clone(),
        })
    }

    /// Self-attention over `x [seq, hidden]` (already normalized), with
    /// the window `active_window` returned for this layer.
    pub fn forward_self(
        &self,
        x: &[f32],
        seq: usize,
        rope: (&[f32], &[f32]),
        window: Option<usize>,
    ) -> Result<Vec<f32>> {
        self.forward(x, seq, x, seq, Some(rope), window)
    }

    /// Cross-attention from `x [seq, hidden]` to `enc [enc_seq, hidden]`:
    /// no RoPE and no mask (the DiT drops its encoder mask).
    pub fn forward_cross(
        &self,
        x: &[f32],
        seq: usize,
        enc: &[f32],
        enc_seq: usize,
    ) -> Result<Vec<f32>> {
        self.forward(x, seq, enc, enc_seq, None, None)
    }

    fn forward(
        &self,
        x: &[f32],
        seq: usize,
        kv_src: &[f32],
        kv_seq: usize,
        rope: Option<(&[f32], &[f32])>,
        window: Option<usize>,
    ) -> Result<Vec<f32>> {
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let mut q = self.q.forward(x, seq)?;
        self.q_norm.apply(&mut q, seq * h);
        let mut k = self.k.forward(kv_src, kv_seq)?;
        self.k_norm.apply(&mut k, kv_seq * kvh);
        let v = self.v.forward(kv_src, kv_seq)?;
        if let Some((cos, sin)) = rope {
            rope_time_major(&mut q, seq, h, hd, cos, sin);
            rope_time_major(&mut k, kv_seq, kvh, hd, cos, sin);
        }
        let merged = match (&self.backend, window) {
            (Some(b), None) => b.attention(
                &q,
                &k,
                &v,
                AttentionShape {
                    batch: 1,
                    queries: seq,
                    keys: kv_seq,
                    heads: h,
                    kv_heads: kvh,
                    dim: hd,
                    kv_time_major: false,
                    causal: false,
                    offset: 0,
                },
                DType::F32,
            )?,
            _ => self.host_attention(&q, &k, &v, seq, kv_seq, window),
        };
        self.o.forward(&merged, seq)
    }

    /// Host attention over time-major q `[seq, h, hd]`, k/v `[kv_seq, kvh,
    /// hd]`; returns time-major `[seq, h * hd]`. With a window, each block
    /// of `BAND_ROWS` queries sees only keys within reach of the block.
    fn host_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        kv_seq: usize,
        window: Option<usize>,
    ) -> Vec<f32> {
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let group = h / kvh;
        let scale = (hd as f32).powf(-0.5);
        let kv_stride = kvh * hd;
        let mut merged = vec![0.0f32; seq * h * hd];
        let mut qh = vec![0.0f32; seq * hd];
        for head in 0..h {
            let kv = head / group;
            for t in 0..seq {
                qh[t * hd..(t + 1) * hd]
                    .copy_from_slice(&q[(t * h + head) * hd..(t * h + head + 1) * hd]);
            }
            let blocks: Vec<(usize, usize)> = match window {
                None => vec![(0, seq)],
                Some(_) => (0..seq)
                    .step_by(BAND_ROWS)
                    .map(|s| (s, (s + BAND_ROWS).min(seq)))
                    .collect(),
            };
            for (q0, q1) in blocks {
                let (k0, k1, mask) = match window {
                    None => (0, kv_seq, None),
                    Some(w) => {
                        let k0 = q0.saturating_sub(w);
                        let k1 = (q1 - 1 + w + 1).min(kv_seq);
                        let span = k1 - k0;
                        let mut m = vec![f32::MIN; (q1 - q0) * span];
                        for i in q0..q1 {
                            let lo = i.saturating_sub(w).max(k0) - k0;
                            let hi = (i + w).min(kv_seq - 1) - k0;
                            m[(i - q0) * span + lo..=(i - q0) * span + hi].fill(0.0);
                        }
                        (k0, k1, Some(m))
                    }
                };
                let rows = q1 - q0;
                let mut out = vec![0.0f32; rows * hd];
                ops::sdpa_strided_into(
                    &mut out,
                    &qh[q0 * hd..q1 * hd],
                    k,
                    k0 * kv_stride + kv * hd,
                    kv_stride,
                    v,
                    k0 * kv_stride + kv * hd,
                    kv_stride,
                    mask.as_deref(),
                    rows,
                    k1 - k0,
                    hd,
                    hd,
                    scale,
                );
                for (r, t) in (q0..q1).enumerate() {
                    merged[(t * h + head) * hd..(t * h + head + 1) * hd]
                        .copy_from_slice(&out[r * hd..(r + 1) * hd]);
                }
            }
        }
        merged
    }
}

/// `Qwen3MLP`: `down(silu(gate(x)) * up(x))`, no biases.
#[derive(Clone, Debug)]
pub struct Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Mlp {
    pub fn load(l: &Loader<'_>, prefix: &str, c: &AceConfig) -> Result<Self> {
        let (d, f) = (c.hidden_size, c.intermediate_size);
        Ok(Self {
            gate: linear(l, &format!("{prefix}.gate_proj"), d, f, false)?,
            up: linear(l, &format!("{prefix}.up_proj"), d, f, false)?,
            down: linear(l, &format!("{prefix}.down_proj"), f, d, false)?,
        })
    }

    pub fn forward(&self, x: &[f32], seq: usize) -> Result<Vec<f32>> {
        let mut g = self.gate.forward(x, seq)?;
        ops::silu(&mut g);
        let u = self.up.forward(x, seq)?;
        for (a, b) in g.iter_mut().zip(&u) {
            *a *= b;
        }
        self.down.forward(&g, seq)
    }
}

/// `AceStepEncoderLayer`: pre-norm bidirectional self-attention and MLP
/// with plain residuals.
#[derive(Clone)]
pub struct EncoderLayer {
    input_norm: RmsNorm,
    attn: Attention,
    post_norm: RmsNorm,
    mlp: Mlp,
    span: Option<usize>,
}

impl EncoderLayer {
    pub fn load(l: &Loader<'_>, prefix: &str, c: &AceConfig, idx: usize) -> Result<Self> {
        let (d, eps) = (c.hidden_size, c.rms_norm_eps);
        Ok(Self {
            input_norm: rms(l, &format!("{prefix}.input_layernorm"), d, eps)?,
            attn: Attention::load(l, &format!("{prefix}.self_attn"), c)?,
            post_norm: rms(l, &format!("{prefix}.post_attention_layernorm"), d, eps)?,
            mlp: Mlp::load(l, &format!("{prefix}.mlp"), c)?,
            span: c.span(idx),
        })
    }

    pub fn span(&self) -> Option<usize> {
        self.span
    }

    pub fn forward(&self, x: &mut [f32], seq: usize, rope: (&[f32], &[f32])) -> Result<()> {
        let d = self.input_norm.width();
        let mut h = x.to_vec();
        self.input_norm.apply(&mut h, seq);
        let a = self
            .attn
            .forward_self(&h, seq, rope, active_window(self.span, seq))?;
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }
        let mut h = x.to_vec();
        self.post_norm.apply(&mut h, seq);
        let m = self.mlp.forward(&h, seq)?;
        debug_assert_eq!(m.len(), seq * d);
        for (xi, mi) in x.iter_mut().zip(&m) {
            *xi += mi;
        }
        Ok(())
    }
}

/// A stack of encoder layers with an input projection and final norm
/// (`AceStepLyricEncoder`, `AceStepTimbreEncoder`).
#[derive(Clone)]
pub struct EncoderStack {
    embed: Proj,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    head_dim: usize,
    rope_theta: f32,
}

impl EncoderStack {
    pub fn load(
        l: &Loader<'_>,
        prefix: &str,
        c: &AceConfig,
        input: usize,
        depth: usize,
    ) -> Result<Self> {
        Ok(Self {
            embed: linear(
                l,
                &format!("{prefix}.embed_tokens"),
                input,
                c.hidden_size,
                true,
            )?,
            layers: (0..depth)
                .map(|i| EncoderLayer::load(l, &format!("{prefix}.layers.{i}"), c, i))
                .collect::<Result<_>>()?,
            norm: rms(l, &format!("{prefix}.norm"), c.hidden_size, c.rms_norm_eps)?,
            head_dim: c.head_dim,
            rope_theta: c.rope_theta,
        })
    }

    /// `x [seq, input]` -> `[seq, hidden]`.
    pub fn forward(&self, x: &[f32], seq: usize) -> Result<Vec<f32>> {
        let mut h = self.embed.forward(x, seq)?;
        let (cos, sin) = rope_tables(seq, self.head_dim, self.rope_theta);
        for layer in &self.layers {
            layer.forward(&mut h, seq, (&cos, &sin))?;
        }
        self.norm.apply(&mut h, seq);
        Ok(h)
    }
}
