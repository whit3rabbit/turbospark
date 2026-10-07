//! MiMo audio tokenizer (XiaomiMiMo/MiMo-Audio-Tokenizer).
//!
//! Reference: `mlx_audio/codec/models/mimo_audio_tokenizer/` (model.py,
//! config.py, quantization.py, audio.py) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/mimo_audio_tokenizer).
//! A whisper-shaped mel encoder with pre-norm transformer layers (RoPE
//! attention, optional causality and band masks), a 20-book Euclidean
//! RVQ whose books have per-book sizes, and a mirrored decoder with
//! causal transposed convs (GroupNorm-affined) feeding a Vocos-style
//! transformer and a magnitude/phase iSTFT head.
//!
//! Scope: the batch-of-one `encode_mels` / `decode` contract. The
//! upstream `segment_size` chunking (6000 mel frames on encode, 1500
//! codec frames on decode) is a memory optimization around the same
//! per-segment math and is not reproduced; callers pass whole
//! sequences. The `istft.window` tensor that leaks into the MLX
//! parameter tree is derived state (a periodic Hann) and is recomputed
//! from the config rather than loaded.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{
    load_bias, load_mlx_conv_weight, load_mlx_convt_weight, BiasMode, Conv1d, ConvTranspose1d,
};
use crate::codec::wnconv::load_f32_shaped;
use crate::fft::{ComplexF32, ComplexFftPlan};
use crate::ops;
use crate::{dsp, stft, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Tokenizer geometry, one-to-one with the reference `ModelConfig`.
#[derive(Debug, Clone)]
pub struct MimoConfig {
    pub d_model: usize,
    pub n_mels: usize,
    pub sampling_rate: u32,
    pub nfft: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub fmin: f32,
    pub fmax: Option<f32>,
    pub kernel_size: usize,
    pub stride_size: usize,
    pub avg_pooler: usize,
    pub encoder_layers: usize,
    pub encoder_skip_layer_id: Option<usize>,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_causal: bool,
    pub encoder_attn_window: (i64, i64),
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_kernel_size: usize,
    pub decoder_stride_size: usize,
    pub decoder_causal: bool,
    pub decoder_attn_window: (i64, i64),
    pub vocoder_dim: usize,
    pub vocoder_intermediate_dim: usize,
    pub vocoder_num_layers: usize,
    pub vocoder_attention_heads: usize,
    pub vocoder_attn_window: (i64, i64),
    pub num_quantizers: usize,
    pub codebook_size: Vec<usize>,
    pub rope_theta: f32,
    pub ln_type: LnormType,
}

/// `ln_type`: LayerNorm (eps 1e-5) or RMSNorm (eps 1e-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LnormType {
    LayerNorm,
    RmsNorm,
}

impl MimoConfig {
    /// Reference dataclass defaults.
    pub fn defaults() -> Self {
        MimoConfig {
            d_model: 1280,
            n_mels: 128,
            sampling_rate: 24_000,
            nfft: 960,
            hop_length: 240,
            window_size: 960,
            fmin: 0.0,
            fmax: None,
            kernel_size: 3,
            stride_size: 2,
            avg_pooler: 2,
            encoder_layers: 32,
            encoder_skip_layer_id: Some(3),
            encoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            encoder_causal: false,
            encoder_attn_window: (-1, -1),
            decoder_layers: 32,
            decoder_attention_heads: 20,
            decoder_ffn_dim: 5120,
            decoder_kernel_size: 3,
            decoder_stride_size: 2,
            decoder_causal: true,
            decoder_attn_window: (-1, -1),
            vocoder_dim: 256,
            vocoder_intermediate_dim: 1024,
            vocoder_num_layers: 16,
            vocoder_attention_heads: 16,
            vocoder_attn_window: (40, 10),
            num_quantizers: 20,
            codebook_size: {
                let mut sizes = vec![1024, 1024];
                sizes.resize(20, 128);
                sizes
            },
            rope_theta: 10000.0,
            ln_type: LnormType::LayerNorm,
        }
    }

    /// Parses a `config.json`-style object over the defaults, then
    /// applies the reference `__post_init__` validation.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let d = Self::defaults();
        let u =
            |field: &str| -> Result<Option<u64>> { Ok(value.get(field).and_then(|v| v.as_u64())) };
        let f =
            |field: &str| -> Result<Option<f64>> { Ok(value.get(field).and_then(|v| v.as_f64())) };
        let bool_ = |field: &str| -> Result<Option<bool>> {
            Ok(value.get(field).and_then(|v| v.as_bool()))
        };
        let vec_u = |field: &str| -> Result<Option<Vec<usize>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_u64().map(|n| n as usize))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of integers".to_string(),
                            })?,
                    ))
                }
            }
        };
        let vec_i = |field: &str| -> Result<Option<Vec<i64>>> {
            match value.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "expected a list".to_string(),
                    })?;
                    Ok(Some(
                        arr.iter()
                            .map(|x| x.as_i64())
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| SpeechError::BadConfig {
                                field: field.to_string(),
                                why: "expected a list of integers".to_string(),
                            })?,
                    ))
                }
            }
        };
        let window = |field: &str, fallback: (i64, i64)| -> Result<(i64, i64)> {
            Ok(vec_i(field)?.map_or(fallback, |v| (v[0], v[1])))
        };
        let skip = match value.get("encoder_skip_layer_id") {
            None | Some(serde_json::Value::Null) => d.encoder_skip_layer_id,
            Some(v) => v.as_u64().map(|n| n as usize),
        };
        let ln_type = match value.get("ln_type").and_then(|v| v.as_str()) {
            None => d.ln_type,
            Some("LayerNorm") => LnormType::LayerNorm,
            Some("RMSNorm") => LnormType::RmsNorm,
            Some(other) => {
                return Err(SpeechError::BadConfig {
                    field: "ln_type".to_string(),
                    why: format!("unsupported tokenizer normalization {other}"),
                })
            }
        };
        let config = MimoConfig {
            d_model: u("d_model")?.unwrap_or(d.d_model as u64) as usize,
            n_mels: u("n_mels")?.unwrap_or(d.n_mels as u64) as usize,
            sampling_rate: u("sampling_rate")?.unwrap_or(d.sampling_rate as u64) as u32,
            nfft: u("nfft")?.unwrap_or(d.nfft as u64) as usize,
            hop_length: u("hop_length")?.unwrap_or(d.hop_length as u64) as usize,
            window_size: u("window_size")?.unwrap_or(d.window_size as u64) as usize,
            fmin: f("fmin")?.unwrap_or(d.fmin as f64) as f32,
            fmax: f("fmax")?.map(|v| v as f32),
            kernel_size: u("kernel_size")?.unwrap_or(d.kernel_size as u64) as usize,
            stride_size: u("stride_size")?.unwrap_or(d.stride_size as u64) as usize,
            avg_pooler: u("avg_pooler")?.unwrap_or(d.avg_pooler as u64) as usize,
            encoder_layers: u("encoder_layers")?.unwrap_or(d.encoder_layers as u64) as usize,
            encoder_skip_layer_id: skip,
            encoder_attention_heads: u("encoder_attention_heads")?
                .unwrap_or(d.encoder_attention_heads as u64)
                as usize,
            encoder_ffn_dim: u("encoder_ffn_dim")?.unwrap_or(d.encoder_ffn_dim as u64) as usize,
            encoder_causal: bool_("encoder_causal")?.unwrap_or(d.encoder_causal),
            encoder_attn_window: window("encoder_attn_window_size", d.encoder_attn_window)?,
            decoder_layers: u("decoder_layers")?.unwrap_or(d.decoder_layers as u64) as usize,
            decoder_attention_heads: u("decoder_attention_heads")?
                .unwrap_or(d.decoder_attention_heads as u64)
                as usize,
            decoder_ffn_dim: u("decoder_ffn_dim")?.unwrap_or(d.decoder_ffn_dim as u64) as usize,
            decoder_kernel_size: u("decoder_kernel_size")?.unwrap_or(d.decoder_kernel_size as u64)
                as usize,
            decoder_stride_size: u("decoder_stride_size")?.unwrap_or(d.decoder_stride_size as u64)
                as usize,
            decoder_causal: bool_("decoder_causal")?.unwrap_or(d.decoder_causal),
            decoder_attn_window: window("decoder_attn_window_size", d.decoder_attn_window)?,
            vocoder_dim: u("vocoder_dim")?.unwrap_or(d.vocoder_dim as u64) as usize,
            vocoder_intermediate_dim: u("vocoder_intermediate_dim")?
                .unwrap_or(d.vocoder_intermediate_dim as u64)
                as usize,
            vocoder_num_layers: u("vocoder_num_layers")?.unwrap_or(d.vocoder_num_layers as u64)
                as usize,
            vocoder_attention_heads: u("vocoder_attention_heads")?
                .unwrap_or(d.vocoder_attention_heads as u64)
                as usize,
            vocoder_attn_window: window("vocoder_attn_window_size", d.vocoder_attn_window)?,
            num_quantizers: u("num_quantizers")?.unwrap_or(d.num_quantizers as u64) as usize,
            codebook_size: vec_u("codebook_size")?.unwrap_or(d.codebook_size),
            rope_theta: f("rope_theta")?.unwrap_or(d.rope_theta as f64) as f32,
            ln_type,
        };
        config.validate()?;
        Ok(config)
    }

    /// The reference `__post_init__` checks.
    fn validate(&self) -> Result<()> {
        if self.codebook_size.len() != self.num_quantizers {
            return Err(SpeechError::BadConfig {
                field: "codebook_size".to_string(),
                why: "codebook_size must specify every RVQ codebook".to_string(),
            });
        }
        for (dim, heads) in [
            (self.d_model, self.encoder_attention_heads),
            (self.d_model, self.decoder_attention_heads),
            (self.vocoder_dim, self.vocoder_attention_heads),
        ] {
            if heads == 0 || dim % heads != 0 || (dim / heads) % 2 != 0 {
                return Err(SpeechError::BadConfig {
                    field: "attention heads".to_string(),
                    why: "dimensions must divide into even head sizes".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Samples per codec token: hop x stride x pooler.
    pub fn downsample_rate(&self) -> usize {
        self.hop_length * self.stride_size * self.avg_pooler
    }
}

/// LayerNorm / RMSNorm by config (`ln_type`), stored `[dim]` weight and
/// optional bias.
#[derive(Debug, Clone)]
enum MimoNorm {
    LayerNorm { weight: Vec<f32>, bias: Vec<f32> },
    Rms { weight: Vec<f32> },
}

impl MimoNorm {
    fn load(file: &SafetensorsFile, prefix: &str, dim: usize, ln_type: LnormType) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[dim])?;
        match ln_type {
            LnormType::LayerNorm => {
                let bias = load_f32_shaped(file, &format!("{prefix}.bias"), &[dim])?;
                Ok(MimoNorm::LayerNorm { weight, bias })
            }
            LnormType::RmsNorm => Ok(MimoNorm::Rms { weight }),
        }
    }

    fn apply(&self, x: &mut [f32], rows: usize, dim: usize) {
        match self {
            MimoNorm::LayerNorm { weight, bias } => {
                ops::layernorm(x, rows, dim, weight, Some(bias), 1e-5);
            }
            MimoNorm::Rms { weight } => ops::rmsnorm(x, rows, dim, weight, 1e-6),
        }
    }
}

/// Standard linear (`[out, in]` weight, optional bias).
#[derive(Debug, Clone)]
struct MimoLinear {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl MimoLinear {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_dim])?)
        } else {
            if with_bias {
                return Err(SpeechError::Tensor {
                    name: bias_name,
                    why: "required by the reference layer".to_string(),
                });
            }
            None
        };
        Ok(MimoLinear {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
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

/// Conv1d loaded from the MLX layout `[out, K, in]`.
#[allow(clippy::too_many_arguments)]
fn load_mimo_conv(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    with_bias: bool,
) -> Result<Conv1d> {
    let weight = load_mlx_conv_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)?;
    let bias = load_bias(
        file,
        &format!("{prefix}.bias"),
        out_ch,
        BiasMode::required(with_bias),
    )?;
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel,
        stride,
        padding,
        dilation: 1,
        groups: 1,
        weight,
        bias,
    })
}

/// ConvTranspose1d loaded from the MLX layout `[out, K, in]`.
fn load_mimo_convtr(
    file: &SafetensorsFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
) -> Result<ConvTranspose1d> {
    let weight = load_mlx_convt_weight(file, &format!("{prefix}.weight"), out_ch, kernel, in_ch)?;
    let bias = load_bias(file, &format!("{prefix}.bias"), out_ch, BiasMode::Optional)?;
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

/// RoPE split-half attention with optional causal and band masks;
/// `sdpa` runs at `head_dim^-0.5`.
#[derive(Debug, Clone)]
struct MimoAttention {
    heads: usize,
    head_dim: usize,
    theta: f32,
    causal: bool,
    window: (i64, i64),
    q_proj: MimoLinear,
    k_proj: MimoLinear,
    v_proj: MimoLinear,
    out_proj: MimoLinear,
}

impl MimoAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        theta: f32,
        causal: bool,
        window: (i64, i64),
    ) -> Result<Self> {
        let proj = |name: &str, with_bias: bool| {
            MimoLinear::load(file, &format!("{prefix}.{name}"), dim, dim, with_bias)
        };
        Ok(MimoAttention {
            heads,
            head_dim: dim / heads,
            theta,
            causal,
            window,
            q_proj: proj("q_proj", true)?,
            k_proj: proj("k_proj", false)?,
            v_proj: proj("v_proj", true)?,
            out_proj: proj("out_proj", true)?,
        })
    }

    fn forward(&self, x: &[f32], seq: usize) -> Vec<f32> {
        let q = self.q_proj.forward(x, seq);
        let k = self.k_proj.forward(x, seq);
        let v = self.v_proj.forward(x, seq);
        let mut qh = ops::split_heads(&q, seq, self.heads, self.head_dim);
        let mut kh = ops::split_heads(&k, seq, self.heads, self.head_dim);
        let vh = ops::split_heads(&v, seq, self.heads, self.head_dim);
        let (cos, sin) = ops::rope_tables(seq, self.head_dim, self.theta);
        ops::rope_neox(&mut qh, self.heads, seq, self.head_dim, &cos, &sin);
        ops::rope_neox(&mut kh, self.heads, seq, self.head_dim, &cos, &sin);

        // Additive mask: rows are queries, columns are keys.
        let mut mask = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let difference = j as i64 - i as i64;
                let mut allowed = true;
                if self.window.0 >= 0 {
                    allowed &= difference >= -self.window.0;
                }
                if self.window.1 >= 0 {
                    allowed &= difference <= self.window.1;
                }
                if self.causal {
                    allowed &= difference <= 0;
                }
                if !allowed {
                    mask[i * seq + j] = -1.0e10;
                }
            }
        }
        let has_mask = self.causal || self.window.0 >= 0 || self.window.1 >= 0;

        let scale = (self.head_dim as f32).powf(-0.5);
        let out = ops::mha(
            &qh,
            &kh,
            &vh,
            if has_mask { Some(&mask) } else { None },
            self.heads,
            seq,
            seq,
            self.head_dim,
            scale,
        );
        // Back to row-major and project.
        let merged = ops::merge_heads(&out, seq, self.heads, self.head_dim);
        self.out_proj.forward(&merged, seq)
    }
}

/// Pre-norm transformer layer: attention residual then GELU MLP
/// residual, each behind its own norm.
#[derive(Debug, Clone)]
struct MimoLayer {
    self_attn: MimoAttention,
    attn_norm: MimoNorm,
    final_norm: MimoNorm,
    fc1: MimoLinear,
    fc2: MimoLinear,
    dim: usize,
}

impl MimoLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        ffn: usize,
        theta: f32,
        causal: bool,
        window: (i64, i64),
        ln_type: LnormType,
    ) -> Result<Self> {
        Ok(MimoLayer {
            self_attn: MimoAttention::load(
                file,
                &format!("{prefix}.self_attn"),
                dim,
                heads,
                theta,
                causal,
                window,
            )?,
            attn_norm: MimoNorm::load(
                file,
                &format!("{prefix}.self_attn_layer_norm"),
                dim,
                ln_type,
            )?,
            final_norm: MimoNorm::load(file, &format!("{prefix}.final_layer_norm"), dim, ln_type)?,
            fc1: MimoLinear::load(file, &format!("{prefix}.fc1"), dim, ffn, true)?,
            fc2: MimoLinear::load(file, &format!("{prefix}.fc2"), ffn, dim, true)?,
            dim,
        })
    }

    fn forward(&self, x: &mut [f32], seq: usize) {
        let mut normed = x.to_vec();
        self.attn_norm.apply(&mut normed, seq, self.dim);
        let attn = self.self_attn.forward(&normed, seq);
        for (v, a) in x.iter_mut().zip(attn) {
            *v += a;
        }
        let mut normed = x.to_vec();
        self.final_norm.apply(&mut normed, seq, self.dim);
        let mut h = self.fc1.forward(&normed, seq);
        ops::gelu_erf(&mut h);
        let h = self.fc2.forward(&h, seq);
        for (v, m) in x.iter_mut().zip(h) {
            *v += m;
        }
    }
}

/// Inference-only Euclidean RVQ; the distance omits the squared input
/// norm (constant per row), and residuals accumulate in f32.
struct MimoRvq {
    /// Per-book `[size, dim]` weights.
    codebooks: Vec<Vec<f32>>,
    dim: usize,
}

impl MimoRvq {
    fn load(file: &SafetensorsFile, sizes: &[usize], dim: usize) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(sizes.len());
        for (i, &size) in sizes.iter().enumerate() {
            codebooks.push(load_f32_shaped(
                file,
                &format!("encoder.quantizer.codebooks.{i}.weight"),
                &[size, dim],
            )?);
        }
        Ok(MimoRvq { codebooks, dim })
    }

    /// Rows `[T, dim]` -> codes per book (greedy residual, first-wins
    /// argmin).
    pub fn encode(
        &self,
        features: &[f32],
        frames: usize,
        count: Option<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        let count = count.unwrap_or(self.codebooks.len());
        if !(1..=self.codebooks.len()).contains(&count) {
            return Err(SpeechError::Input {
                why: "num_quantizers is outside the codec's codebook range".to_string(),
            });
        }
        let mut residual = features.to_vec();
        let mut codes = Vec::with_capacity(count);
        for book in &self.codebooks[..count] {
            let size = book.len() / self.dim;
            let mut book_sq = vec![0.0f32; size];
            for (c, sq) in book_sq.iter_mut().enumerate() {
                *sq = book[c * self.dim..(c + 1) * self.dim]
                    .iter()
                    .map(|v| v * v)
                    .sum();
            }
            let mut idx = vec![0i32; frames];
            for (t, code) in idx.iter_mut().enumerate() {
                let row = &residual[t * self.dim..(t + 1) * self.dim];
                let mut best = f32::INFINITY;
                let mut best_idx = 0usize;
                for c in 0..size {
                    let e = &book[c * self.dim..(c + 1) * self.dim];
                    let mut dot = 0.0f32;
                    for (a, &b) in row.iter().zip(e) {
                        dot += a * b;
                    }
                    let dist = book_sq[c] - 2.0 * dot;
                    if dist < best {
                        best = dist;
                        best_idx = c;
                    }
                }
                *code = best_idx as i32;
            }
            for (t, &c) in idx.iter().enumerate() {
                let base = c as usize * self.dim;
                for d in 0..self.dim {
                    residual[t * self.dim + d] -= book[base + d];
                }
            }
            codes.push(idx);
        }
        Ok(codes)
    }

    /// Codes per book -> rows `[T, dim]`.
    pub fn decode(&self, codes: &[Vec<i32>], frames: usize) -> Result<Vec<f32>> {
        if codes.is_empty() || codes.len() > self.codebooks.len() {
            return Err(SpeechError::Input {
                why: "codes must carry 1..=num_quantizers books".to_string(),
            });
        }
        let mut out = vec![0.0f32; frames * self.dim];
        for (q, book_codes) in codes.iter().enumerate() {
            let book = &self.codebooks[q];
            let size = book.len() / self.dim;
            for (t, &c) in book_codes.iter().enumerate() {
                let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                    why: format!("negative code {c}"),
                })?;
                if idx >= size {
                    return Err(SpeechError::Input {
                        why: format!("code {c} outside codebook {q} of {size}"),
                    });
                }
                for d in 0..self.dim {
                    out[t * self.dim + d] += book[idx * self.dim + d];
                }
            }
        }
        Ok(out)
    }
}

/// Causal transposed conv: plain ConvTranspose1d, GroupNorm(1, out)
/// affine over every (channel, frame) element, then a per-channel tail
/// trim of `max(0, kernel - stride)` frames.
struct CausalConvTr {
    conv: ConvTranspose1d,
    gn_weight: Vec<f32>,
    gn_bias: Vec<f32>,
    trim: usize,
    out_ch: usize,
}

impl CausalConvTr {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(CausalConvTr {
            conv: load_mimo_convtr(
                file,
                &format!("{prefix}.conv"),
                in_ch,
                out_ch,
                kernel,
                stride,
            )?,
            gn_weight: load_f32_shaped(file, &format!("{prefix}.norm.weight"), &[out_ch])?,
            gn_bias: load_f32_shaped(file, &format!("{prefix}.norm.bias"), &[out_ch])?,
            trim: kernel.saturating_sub(stride),
            out_ch,
        })
    }

    /// Channel-major in / out.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let y = self.conv.forward(x);
        let frames = y.len() / self.out_ch;
        // GroupNorm(1): normalize over all elements, per-channel affine.
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / y.len() as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let mut y = y;
        for c in 0..self.out_ch {
            for f in 0..frames {
                y[c * frames + f] =
                    (y[c * frames + f] - mean) * inv * self.gn_weight[c] + self.gn_bias[c];
            }
        }
        if self.trim > 0 && frames > self.trim {
            let keep = frames - self.trim;
            let mut trimmed = vec![0.0f32; keep * self.out_ch];
            for c in 0..self.out_ch {
                trimmed[c * keep..(c + 1) * keep]
                    .copy_from_slice(&y[c * frames..c * frames + keep]);
            }
            return trimmed;
        }
        y
    }
}

/// Vocos-style transformer head: mel embedding, windowed transformer
/// layers, then a magnitude/phase iSTFT.
struct Vocos {
    embeddings: MimoLinear,
    layers: Vec<MimoLayer>,
    layer_norm: MimoNorm,
    head_out: MimoLinear,
    dim: usize,
    hop: usize,
    plan: ComplexFftPlan,
}

impl Vocos {
    fn load(config: &MimoConfig, file: &SafetensorsFile) -> Result<Self> {
        let embeddings = MimoLinear::load(
            file,
            "decoder.vocoder.embeddings",
            config.n_mels,
            config.vocoder_dim,
            false,
        )?;
        let mut layers = Vec::with_capacity(config.vocoder_num_layers);
        for i in 0..config.vocoder_num_layers {
            layers.push(MimoLayer::load(
                file,
                &format!("decoder.vocoder.layers.{i}"),
                config.vocoder_dim,
                config.vocoder_attention_heads,
                config.vocoder_intermediate_dim,
                config.rope_theta,
                false,
                config.vocoder_attn_window,
                config.ln_type,
            )?);
        }
        Ok(Vocos {
            embeddings,
            layers,
            layer_norm: MimoNorm::load(
                file,
                "decoder.vocoder.layer_norm",
                config.vocoder_dim,
                config.ln_type,
            )?,
            head_out: MimoLinear::load(
                file,
                "decoder.vocoder.head.out",
                config.vocoder_dim,
                config.nfft + 2,
                true,
            )?,
            dim: config.vocoder_dim,
            hop: config.hop_length,
            plan: ComplexFftPlan::new(config.nfft)?,
        })
    }

    /// Rows `[frames, n_mels]` -> waveform samples (the iSTFT head
    /// included).
    fn forward(
        &self,
        x: &[f32],
        frames: usize,
        plan: &ComplexFftPlan,
        window: &[f32],
    ) -> Result<Vec<f32>> {
        let mut h = self.embeddings.forward(x, frames);
        for layer in &self.layers {
            layer.forward(&mut h, frames);
        }
        self.layer_norm.apply(&mut h, frames, self.dim);
        let spectrum = self.head_out.forward(&h, frames);
        Ok(istft_head(&spectrum, frames, plan, window, self.hop))
    }
}

/// ISTFTHead math: split the projection into magnitude and phase
/// halves, cap the magnitude at 100 after exp, overlap-add with the
/// squared-Hann denominator, and trim `(n_fft - hop) / 2` per side.
fn istft_head(
    spectrum: &[f32],
    frames: usize,
    cfft: &ComplexFftPlan,
    window: &[f32],
    hop: usize,
) -> Vec<f32> {
    let n_fft = window.len();
    let bins = n_fft / 2 + 1;
    let ola_len = (frames - 1) * hop + n_fft;
    let mut ola = vec![0.0f32; ola_len];
    let mut denom = vec![0.0f32; ola_len];
    for f in 0..frames {
        let start = f * hop;
        for (t, &w) in window.iter().enumerate() {
            denom[start + t] += w * w;
        }
    }
    let mut ext = vec![ComplexF32::new(0.0, 0.0); n_fft];
    for f in 0..frames {
        for b in 0..bins {
            let magnitude = spectrum[f * 2 * bins + b].exp().min(100.0);
            let phase = spectrum[f * 2 * bins + bins + b];
            ext[b] = ComplexF32::new(magnitude * phase.cos(), magnitude * phase.sin());
        }
        ext[0].im = 0.0;
        ext[bins - 1].im = 0.0;
        for k in bins..n_fft {
            ext[k] = ext[n_fft - k].conj();
        }
        let frame = cfft.inverse(&ext).expect("istft frame inverse");
        let start = f * hop;
        for (t, &v) in frame.iter().enumerate() {
            ola[start + t] += v.re * window[t];
        }
    }
    let trim = (n_fft - hop) / 2;
    let mut out = Vec::with_capacity(ola_len.saturating_sub(2 * trim));
    for (t, &v) in ola.iter().enumerate().skip(trim) {
        if t >= ola_len - trim {
            break;
        }
        let d = denom[t];
        out.push(v / if d > 1e-10 { d } else { 1.0 });
    }
    out
}

/// Loaded MiMo audio tokenizer, batch-of-one.
pub struct MiMoAudioTokenizer {
    pub config: MimoConfig,
    encoder_conv1: Conv1d,
    encoder_conv2: Conv1d,
    encoder_layers: Vec<MimoLayer>,
    encoder_layer_norm: MimoNorm,
    down_sample: Option<(Conv1d, MimoNorm)>,
    rvq: MimoRvq,
    dconv1: Option<CausalConvTr>,
    decoder_layers: Vec<MimoLayer>,
    decoder_layer_norm: MimoNorm,
    dconv2: CausalConvTr,
    vocoder: Vocos,
    /// The synthesis window: the checkpoint's `istft.window` parameter
    /// when present, else the derived periodic Hann.
    window: Vec<f32>,
}

impl MiMoAudioTokenizer {
    /// Opens a checkpoint directory with `config.json` plus
    /// `model.safetensors`.
    pub fn open(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = MimoConfig::from_json(&value)?;
        let file =
            SafetensorsFile::open(&dir.join("model.safetensors")).map_err(SpeechError::from)?;
        Self::load(config, &file)
    }

    /// Loads from a parsed config and a safetensors file.
    pub fn load(config: MimoConfig, file: &SafetensorsFile) -> Result<Self> {
        let dim = config.d_model;
        let encoder_conv1 = load_mimo_conv(
            file,
            "encoder.conv1",
            config.n_mels,
            dim,
            config.kernel_size,
            1,
            1,
            true,
        )?;
        let encoder_conv2 = load_mimo_conv(
            file,
            "encoder.conv2",
            dim,
            dim,
            config.kernel_size,
            config.stride_size,
            1,
            true,
        )?;
        let mut encoder_layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            encoder_layers.push(MimoLayer::load(
                file,
                &format!("encoder.layers.{i}"),
                dim,
                config.encoder_attention_heads,
                config.encoder_ffn_dim,
                config.rope_theta,
                config.encoder_causal,
                config.encoder_attn_window,
                config.ln_type,
            )?);
        }
        let encoder_layer_norm = MimoNorm::load(file, "encoder.layer_norm", dim, config.ln_type)?;
        let down_sample = if config.avg_pooler != 1 {
            Some((
                load_mimo_conv(
                    file,
                    "encoder.down_sample_layer.layers.0",
                    dim,
                    dim,
                    config.avg_pooler,
                    config.avg_pooler,
                    0,
                    false,
                )?,
                MimoNorm::load(file, "encoder.down_sample_norm", dim, config.ln_type)?,
            ))
        } else {
            None
        };
        let rvq = MimoRvq::load(file, &config.codebook_size, dim)?;

        let dconv1 = if config.avg_pooler != 1 {
            Some(CausalConvTr::load(
                file,
                "decoder.dconv1",
                dim,
                dim,
                config.avg_pooler,
                config.avg_pooler,
            )?)
        } else {
            None
        };
        let mut decoder_layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            decoder_layers.push(MimoLayer::load(
                file,
                &format!("decoder.layers.{i}"),
                dim,
                config.decoder_attention_heads,
                config.decoder_ffn_dim,
                config.rope_theta,
                config.decoder_causal,
                config.decoder_attn_window,
                config.ln_type,
            )?);
        }
        let decoder_layer_norm = MimoNorm::load(file, "decoder.layer_norm", dim, config.ln_type)?;
        let dconv2 = CausalConvTr::load(
            file,
            "decoder.dconv2",
            dim,
            config.n_mels,
            config.decoder_kernel_size,
            config.decoder_stride_size,
        )?;
        let vocoder = Vocos::load(&config, file)?;
        // The reference registers the synthesis window as a module
        // attribute, so checkpoints carry it; prefer the stored tensor
        // and fall back to the periodic Hann derivation.
        let window = if file.contains_tensor("decoder.vocoder.head.istft.window") {
            load_f32_shaped(
                file,
                "decoder.vocoder.head.istft.window",
                &[config.window_size],
            )?
        } else {
            dsp::hann_window(config.window_size)
        };
        Ok(MiMoAudioTokenizer {
            config,
            encoder_conv1,
            encoder_conv2,
            encoder_layers,
            encoder_layer_norm,
            down_sample,
            rvq,
            dconv1,
            decoder_layers,
            decoder_layer_norm,
            dconv2,
            vocoder,
            window,
        })
    }

    /// The reference `audio.log_mel_spectrogram`: periodic Hann STFT
    /// (center, reflect), magnitude mel (HTK, no norm), `log(max(.,
    /// 1e-7))`. Returns rows `[frames, n_mels]`. The analysis window is
    /// always the derived periodic Hann (audio.py builds it fresh).
    pub fn log_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "expected a nonempty mono waveform".to_string(),
            });
        }
        let mut padded: Vec<f32> = samples.to_vec();
        if padded.len() <= self.config.nfft / 2 {
            padded.resize(self.config.nfft / 2 + 1, 0.0);
        }
        let mut window = dsp::hann_window(self.config.window_size);
        window.resize(self.config.nfft, 0.0);
        let options = stft::StftOptions {
            fft_size: self.config.nfft,
            hop: self.config.hop_length,
            window,
            center: true,
        };
        let spectra = stft::stft_with_modes(
            &padded,
            &options,
            stft::StftPaddingMode::Reflect,
            stft::StftWindowPlacement::Left,
        )?;
        let projector = crate::mel::mel_projector_cached(
            self.config.n_mels,
            self.config.nfft,
            self.config.sampling_rate,
            self.config.fmin,
            self.config.fmax,
            crate::mel::MelScale::Htk,
        )?;
        let mut out = Vec::with_capacity(spectra.len());
        let mut mags: Vec<f32> = Vec::new();
        for frame in &spectra {
            mags.clear();
            mags.extend(frame.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()));
            let mut row = Vec::with_capacity(self.config.n_mels);
            projector.project_into(&mags, &mut row)?;
            for v in &mut row {
                *v = v.max(1e-7).ln();
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Mel rows `[frames, n_mels]` plus the valid frame count -> codes
    /// per book.
    pub fn encode_mels(&self, mels: &[Vec<f32>], length: usize) -> Result<Vec<Vec<i32>>> {
        if mels.is_empty() || mels[0].len() != self.config.n_mels {
            return Err(SpeechError::Input {
                why: "expected nonempty (frames, n_mels) features".to_string(),
            });
        }
        let features = self.encode_features(mels, length)?;
        self.rvq
            .encode(&features, features.len() / self.config.d_model, None)
    }

    /// Encoder hidden rows `[valid, d_model]`.
    fn encode_features(&self, mels: &[Vec<f32>], length: usize) -> Result<Vec<f32>> {
        let c = &self.config;
        let frames = mels.len();
        let mut mel_cm = vec![0.0f32; frames * c.n_mels];
        for (t, row) in mels.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                mel_cm[d * frames + t] = v;
            }
        }
        let mut x = self.encoder_conv1.forward(&mel_cm);
        ops::gelu_erf(&mut x);
        let mut x = self.encoder_conv2.forward(&x);
        ops::gelu_erf(&mut x);
        let padded_length = x.len() / c.d_model;
        let valid = (length + 3 - c.kernel_size + 2 - c.kernel_size) / c.stride_size + 1;
        let valid = valid.min(padded_length);
        let mut trimmed = vec![0.0f32; valid * c.d_model];
        for d in 0..c.d_model {
            trimmed[d * valid..(d + 1) * valid]
                .copy_from_slice(&x[d * padded_length..d * padded_length + valid]);
        }
        let mut rows = vec![0.0f32; trimmed.len()];
        for t in 0..valid {
            for d in 0..c.d_model {
                rows[t * c.d_model + d] = trimmed[d * valid + t];
            }
        }
        let mut skip: Option<Vec<f32>> = None;
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            layer.forward(&mut rows, valid);
            if Some(i + 1) == c.encoder_skip_layer_id {
                skip = Some(rows.clone());
            }
        }
        if let Some(skip) = skip {
            for (v, s) in rows.iter_mut().zip(skip) {
                *v += s;
            }
        }
        self.encoder_layer_norm.apply(&mut rows, valid, c.d_model);
        if c.avg_pooler != 1 {
            let padded_len = padded_length;
            let mut pooled_in = vec![0.0f32; rows.len()];
            for t in 0..valid {
                pooled_in[t * c.d_model..(t + 1) * c.d_model]
                    .copy_from_slice(&rows[t * c.d_model..(t + 1) * c.d_model]);
            }
            for t in valid..padded_len {
                pooled_in[t * c.d_model..(t + 1) * c.d_model]
                    .copy_from_slice(&rows[(valid - 1) * c.d_model..valid * c.d_model]);
            }
            let extra = padded_len % c.avg_pooler;
            let total = padded_len + extra;
            let mut cm = vec![0.0f32; total * c.d_model];
            for t in 0..padded_len {
                for d in 0..c.d_model {
                    cm[d * total + t] = pooled_in[t * c.d_model + d];
                }
            }
            let ds = self.down_sample.as_ref().unwrap();
            let mut y = ds.0.forward(&cm);
            ops::gelu_erf(&mut y);
            let y_frames = y.len() / c.d_model;
            let mut y_rows = vec![0.0f32; y.len()];
            for t in 0..y_frames {
                for d in 0..c.d_model {
                    y_rows[t * c.d_model + d] = y[d * y_frames + t];
                }
            }
            ds.1.apply(&mut y_rows, y_frames, c.d_model);
            let keep = valid.div_ceil(c.avg_pooler);
            y_rows.truncate(keep * c.d_model);
            return Ok(y_rows);
        }
        Ok(rows)
    }

    /// Codes per book -> mono waveform at `sampling_rate`.
    pub fn decode(&self, codes: &[Vec<i32>]) -> Result<Vec<f32>> {
        let c = &self.config;
        if codes.is_empty() || codes.len() > c.num_quantizers {
            return Err(SpeechError::Input {
                why: "codes must carry 1..=num_quantizers books".to_string(),
            });
        }
        let frames = codes[0].len();
        if codes.iter().any(|book| book.len() != frames) {
            return Err(SpeechError::Input {
                why: "codebooks must share one frame count".to_string(),
            });
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let features = self.rvq.decode(codes, frames)?;
        let to_cm = |rows: &[f32], frames: usize, dim: usize| -> Vec<f32> {
            let mut cm = vec![0.0f32; rows.len()];
            for t in 0..frames {
                for d in 0..dim {
                    cm[d * frames + t] = rows[t * dim + d];
                }
            }
            cm
        };
        let to_rows = |cm: &[f32], frames: usize, dim: usize| -> Vec<f32> {
            let mut rows = vec![0.0f32; cm.len()];
            for t in 0..frames {
                for d in 0..dim {
                    rows[t * dim + d] = cm[d * frames + t];
                }
            }
            rows
        };
        let mut x = features;
        if let Some(dconv1) = &self.dconv1 {
            let cm = to_cm(&x, frames, c.d_model);
            let out = dconv1.forward(&cm);
            let out_frames = out.len() / c.d_model;
            x = to_rows(&out, out_frames, c.d_model);
        }
        let seq = x.len() / c.d_model;
        for layer in &self.decoder_layers {
            layer.forward(&mut x, seq);
        }
        self.decoder_layer_norm.apply(&mut x, seq, c.d_model);
        let cm = to_cm(&x, seq, c.d_model);
        let mel_cm = self.dconv2.forward(&cm);
        let mel_frames = mel_cm.len() / c.n_mels;
        let mel_rows = to_rows(&mel_cm, mel_frames, c.n_mels);
        self.vocoder
            .forward(&mel_rows, mel_frames, &self.vocoder.plan, &self.window)
    }
}

#[cfg(test)]
mod tests;
