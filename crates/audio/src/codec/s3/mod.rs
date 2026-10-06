//! S3 speech tokenizers (CosyVoice / FunAudioLLM S3Tokenizer v1 and v2).
//!
//! Reference: `mlx_audio/codec/models/s3/model.py` (v1) and
//! `model_v2.py` (v2) at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/s3).
//! Both generations share the whisper-shaped front end (two strided
//! convs with frame masking) and a residual attention stack; v1 adds
//! sinusoidal positional embeddings and quantizes with an Euclidean
//! codebook over L2-normalized frames, v2 replaces the positional
//! embedding with NeoX-style RoPE plus an FSMN memory branch inside
//! attention and quantizes with finite scalar quantization (8 ternary
//! digits, 3^8 = 6561 codes).
//!
//! Scope: the batch-of-one `quantize` contract (mel in, codes out).
//! The >30 s sliding-window path of `S3TokenizerV2.quantize`
//! (3000-frame windows, 2600-frame stride, half-overlap merge) is
//! included; the batch>1 orchestration around it is not. `FSQCodebook.
//! decode` raises upstream ("no official up project component") and is
//! likewise not ported.
//!
//! Upstream defect, documented at the pinned commit: the v1 codebook
//! lives on `VectorQuantization._codebook`, an underscore-private
//! attribute MLX's parameter tree cannot traverse, so `load_weights`
//! and `tree_flatten` never see it (a checkpoint written from the MLX
//! tree carries no v1 codebook at all). The Rust loader therefore
//! reads the codebook from `quantizer.embed`, accepting the raw
//! PyTorch-converted names `quantizer._codebook.embed` and
//! `quantizer.codebook.embed` as well, and refuses to load without one.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::whisper::{WHISPER_FMAX, WHISPER_HOP, WHISPER_N_FFT, WHISPER_SAMPLE_RATE};
use crate::{mel, stft, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Maximum mel frames before v2 switches to the sliding-window path
/// (30 s at 100 mel frames/s).
const MAX_FRAMES: usize = 3000;
/// Sliding-window stride in mel frames: 30 s windows minus 4 s overlap.
const FRAMES_PER_STRIDE: usize = 2600;
/// Quantization rate of v2 codes, used by the overlap merge (25/s).
const V2_TOKEN_RATE: usize = 25;
/// The `tanh` scaling FSQ applies before rounding. The reference
/// literal 0.9990000128746033 is an f64 in Python source but the
/// multiply runs on f32 tensors, so the f32-rounded value is the
/// faithful one.
const FSQ_SCALE: f32 = 0.999_000_012_874_603_3_f64 as f32;
/// RoPE head dimension and context the v2 encoder hardcodes
/// (`precompute_freqs_cis(64, 1024 * 2)`).
const V2_ROPE_DIM: usize = 64;
const V2_ROPE_END: usize = 1024 * 2;

/// Geometry shared by both tokenizer generations, one-to-one with the
/// reference `ModelConfig`.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_codebook_size: usize,
}

impl S3Config {
    /// Reference dataclass defaults (the v1 values; v2 overrides
    /// `n_codebook_size` to 3^8 = 6561).
    pub fn v1() -> Self {
        S3Config {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 6,
            n_codebook_size: 4096,
        }
    }

    /// v2 defaults: same encoder geometry, FSQ codebook.
    pub fn v2() -> Self {
        let mut config = Self::v1();
        config.n_codebook_size = 6561;
        config
    }

    /// Parses a `config.json`-style object over the given generation
    /// defaults; zero/missing fields keep the default.
    pub fn from_json_with_defaults(value: &serde_json::Value, defaults: S3Config) -> Result<Self> {
        let field = |name: &str| -> Result<Option<usize>> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => {
                    v.as_u64()
                        .map(|n| Some(n as usize))
                        .ok_or_else(|| SpeechError::BadConfig {
                            field: name.to_string(),
                            why: "expected an integer".to_string(),
                        })
                }
            }
        };
        let pick =
            |name: &str, fallback: usize| -> Result<usize> { Ok(field(name)?.unwrap_or(fallback)) };
        Ok(S3Config {
            n_mels: pick("n_mels", defaults.n_mels)?,
            n_audio_ctx: pick("n_audio_ctx", defaults.n_audio_ctx)?,
            n_audio_state: pick("n_audio_state", defaults.n_audio_state)?,
            n_audio_head: pick("n_audio_head", defaults.n_audio_head)?,
            n_audio_layer: pick("n_audio_layer", defaults.n_audio_layer)?,
            n_codebook_size: pick("n_codebook_size", defaults.n_codebook_size)?,
        })
    }
}

/// Standard linear layer; HF and MLX share the `[out, in]` layout.
#[derive(Debug, Clone)]
struct S3Linear {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl S3Linear {
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
        Ok(S3Linear {
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

/// Conv1d loaded from the MLX checkpoint layout `[out, K, in]`, stored
/// PyTorch `[out, in, K]` for the crate kernel.
#[derive(Debug, Clone)]
struct S3Conv1d {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl S3Conv1d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_g = in_ch / groups;
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kernel, in_g])?;
        let mut weight = vec![0.0f32; stored.len()];
        for oc in 0..out_ch {
            for kk in 0..kernel {
                for ic in 0..in_g {
                    weight[oc * in_g * kernel + ic * kernel + kk] =
                        stored[oc * kernel * in_g + kk * in_g + ic];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(S3Conv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            groups,
            weight,
            bias,
        })
    }

    /// Channel-major `x [in_ch, seq]` -> `[out_ch, out_seq]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            1,
            self.groups,
        )
    }
}

/// Multi-head self-attention with the reference scaling contract: both
/// q and k are pre-scaled by `(d_head)^-0.25` and the score dot runs at
/// scale 1, matching the reference elementwise order. The additive mask
/// is key-only (the reference broadcasts a `(B, 1, 1, T)` bias across
/// queries and heads). v2 additionally runs a depthwise FSMN memory
/// branch on v and NeoX-style RoPE over the full 64-dim head.
#[derive(Debug, Clone)]
struct S3Attention {
    n_head: usize,
    state: usize,
    query: S3Linear,
    key: S3Linear,
    value: S3Linear,
    out: S3Linear,
    /// v2 only: depthwise conv (kernel 31, no bias, symmetric zero pad
    /// 15/15), residual on v, masked by the valid frames.
    fsmn: Option<S3Conv1d>,
}

impl S3Attention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        state: usize,
        n_head: usize,
        fsmn_kernel: Option<usize>,
    ) -> Result<Self> {
        let linear = |name: &str, with_bias: bool| {
            S3Linear::load(file, &format!("{prefix}.{name}"), state, state, with_bias)
        };
        let fsmn = match fsmn_kernel {
            None => None,
            Some(kernel) => Some(S3Conv1d::load(
                file,
                &format!("{prefix}.fsmn_block"),
                state,
                state,
                kernel,
                1,
                0,
                state,
            )?),
        };
        Ok(S3Attention {
            n_head,
            state,
            query: linear("query", true)?,
            key: linear("key", false)?,
            value: linear("value", true)?,
            out: linear("out", true)?,
            fsmn,
        })
    }

    /// Projects `x [seq, state]` rows to q, k, v, each `[seq, state]`.
    fn project(&self, x: &[f32], seq: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            self.query.forward(x, seq),
            self.key.forward(x, seq),
            self.value.forward(x, seq),
        )
    }

    /// `forward_fsmn`: channel-major v through the padded depthwise
    /// conv, residual add, frame mask. In / out `[state, seq]`.
    fn fsmn_forward(&self, v_cm: &[f32], seq: usize, mask_pad: &[f32]) -> Vec<f32> {
        let fsmn = self.fsmn.as_ref().expect("fsmn_forward needs fsmn_block");
        let left = (fsmn.kernel - 1) / 2;
        let right = fsmn.kernel - 1 - left;
        let padded_seq = seq + left + right;
        let mut padded = vec![0.0f32; self.state * padded_seq];
        for d in 0..self.state {
            for t in 0..seq {
                padded[d * padded_seq + left + t] = v_cm[d * seq + t] * mask_pad[t];
            }
        }
        let y = fsmn.forward(&padded);
        let mut out = vec![0.0f32; self.state * seq];
        for d in 0..self.state {
            for t in 0..seq {
                let v = v_cm[d * seq + t] * mask_pad[t];
                out[d * seq + t] = (y[d * seq + t] + v) * mask_pad[t];
            }
        }
        out
    }
}

/// Residual attention block. v1 LayerNorms use MLX's default eps 1e-5;
/// the v2 `attn_ln` is constructed with eps 1e-6.
#[derive(Debug, Clone)]
struct S3Block {
    attn: S3Attention,
    attn_ln: (Vec<f32>, Vec<f32>),
    attn_ln_eps: f32,
    mlp_in: S3Linear,
    mlp_out: S3Linear,
    mlp_ln: (Vec<f32>, Vec<f32>),
    state: usize,
}

impl S3Block {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        state: usize,
        n_head: usize,
        fsmn_kernel: Option<usize>,
        attn_ln_eps: f32,
    ) -> Result<Self> {
        let ln = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                load_f32_shaped(file, &format!("{prefix}.{name}.weight"), &[state])?,
                load_f32_shaped(file, &format!("{prefix}.{name}.bias"), &[state])?,
            ))
        };
        Ok(S3Block {
            attn: S3Attention::load(file, &format!("{prefix}.attn"), state, n_head, fsmn_kernel)?,
            attn_ln: ln("attn_ln")?,
            attn_ln_eps,
            mlp_in: S3Linear::load(
                file,
                &format!("{prefix}.mlp.layers.0"),
                state,
                state * 4,
                true,
            )?,
            mlp_out: S3Linear::load(
                file,
                &format!("{prefix}.mlp.layers.2"),
                state * 4,
                state,
                true,
            )?,
            mlp_ln: ln("mlp_ln")?,
            state,
        })
    }

    /// In-place residual block update on `x [seq, state]`.
    fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        key_bias: &[f32],
        mask_pad: Option<&[f32]>,
        rope: Option<(&[f32], &[f32])>,
    ) {
        let mut normed = x.to_vec();
        ops::layernorm(
            &mut normed,
            seq,
            self.state,
            &self.attn_ln.0,
            Some(&self.attn_ln.1),
            self.attn_ln_eps,
        );
        let (q, k, v) = self.attn.project(&normed, seq);
        let d_head = self.state / self.attn.n_head;
        let heads = self.attn.n_head;
        let scale = (d_head as f32).powf(-0.25);

        // Per-head planes [head, seq, d_head]; q and k pre-scaled.
        let mut qh = vec![0.0f32; heads * seq * d_head];
        let mut kh = vec![0.0f32; heads * seq * d_head];
        let mut vh = vec![0.0f32; heads * seq * d_head];
        for h in 0..heads {
            for t in 0..seq {
                let src = t * self.state + h * d_head;
                for d in 0..d_head {
                    qh[(h * seq + t) * d_head + d] = q[src + d] * scale;
                    kh[(h * seq + t) * d_head + d] = k[src + d] * scale;
                    vh[(h * seq + t) * d_head + d] = v[src + d];
                }
            }
        }
        if let Some((cos, sin)) = rope {
            apply_rope_neox(&mut qh, heads, seq, d_head, V2_ROPE_DIM, cos, sin);
            apply_rope_neox(&mut kh, heads, seq, d_head, V2_ROPE_DIM, cos, sin);
        }

        // FSMN memory branch on v (channel-major round trip).
        let fsmn_out = match (&self.attn.fsmn, mask_pad) {
            (Some(_), Some(mask_pad)) => {
                let mut v_cm = vec![0.0f32; v.len()];
                for t in 0..seq {
                    for d in 0..self.state {
                        v_cm[d * seq + t] = v[t * self.state + d];
                    }
                }
                let mem = self.attn.fsmn_forward(&v_cm, seq, mask_pad);
                let mut rows = vec![0.0f32; v.len()];
                for d in 0..self.state {
                    for t in 0..seq {
                        rows[t * self.state + d] = mem[d * seq + t];
                    }
                }
                rows
            }
            _ => Vec::new(),
        };

        // Additive key bias expanded per head: sdpa indexes
        // m[h * kv + j] with h spanning query frames here, so the
        // mask is [seq, seq] with every row equal to key_bias.
        let mask: Vec<f32> = if key_bias.is_empty() {
            Vec::new()
        } else {
            let mut m = Vec::with_capacity(seq * seq);
            for _ in 0..seq {
                m.extend_from_slice(key_bias);
            }
            m
        };

        let mut out = vec![0.0f32; heads * seq * d_head];
        for h in 0..heads {
            let o = ops::sdpa(
                &qh[h * seq * d_head..(h + 1) * seq * d_head],
                &kh[h * seq * d_head..(h + 1) * seq * d_head],
                &vh[h * seq * d_head..(h + 1) * seq * d_head],
                if mask.is_empty() { None } else { Some(&mask) },
                seq,
                seq,
                d_head,
                d_head,
                1.0,
            );
            out[h * seq * d_head..(h + 1) * seq * d_head].copy_from_slice(&o);
        }
        // Back to row-major, then the output projection + FSMN add.
        let mut merged = vec![0.0f32; seq * self.state];
        for h in 0..heads {
            for t in 0..seq {
                let src = (h * seq + t) * d_head;
                merged[t * self.state + h * d_head..t * self.state + (h + 1) * d_head]
                    .copy_from_slice(&out[src..src + d_head]);
            }
        }
        let projected = self.attn.out.forward(&merged, seq);
        for (v, a) in x.iter_mut().zip(projected) {
            *v += a;
        }
        if !fsmn_out.is_empty() {
            for (v, m) in x.iter_mut().zip(fsmn_out) {
                *v += m;
            }
        }

        let mut normed = x.to_vec();
        ops::layernorm(
            &mut normed,
            seq,
            self.state,
            &self.mlp_ln.0,
            Some(&self.mlp_ln.1),
            1e-5,
        );
        let mut h = self.mlp_in.forward(&normed, seq);
        ops::gelu_erf(&mut h);
        let h = self.mlp_out.forward(&h, seq);
        for (v, m) in x.iter_mut().zip(h) {
            *v += m;
        }
    }
}

/// Whisper sinusoids: `concat(sin, cos)` of `arange(L) * exp(-log(10^4)
/// * arange(C/2) / (C/2 - 1))`, stored `[L, C]`.
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    assert!(channels % 2 == 0);
    let half = channels / 2;
    let increment = 10000f32.ln() / (half - 1) as f32;
    let mut table = vec![0.0f32; length * channels];
    for t in 0..length {
        for d in 0..half {
            let scaled = t as f32 * (-increment * d as f32).exp();
            table[t * channels + d] = scaled.sin();
            table[t * channels + half + d] = scaled.cos();
        }
    }
    table
}

fn apply_rope_neox(
    values: &mut [f32],
    heads: usize,
    seq: usize,
    dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rotary_dim / 2;
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..half {
                let a = values[base + d];
                let b = values[base + half + d];
                let table = t * half + d;
                values[base + d] = a * cos[table] - b * sin[table];
                values[base + half + d] = b * cos[table] + a * sin[table];
            }
        }
    }
}

/// Shared conv front end: conv1 (k=3, pad 1) + gelu, conv2 (k=3,
/// stride 2, pad 1) + gelu, with the reference's frame masking between
/// stages. Length updates follow
/// `(x_len + 2 - (3 - 1) - 1) // stride + 1`.
struct ConvFrontend {
    conv1: S3Conv1d,
    conv2: S3Conv1d,
    stride: usize,
}

impl ConvFrontend {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        n_mels: usize,
        state: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(ConvFrontend {
            conv1: S3Conv1d::load(
                file,
                &format!("{prefix}.conv1"),
                n_mels,
                state,
                3,
                stride,
                1,
                1,
            )?,
            conv2: S3Conv1d::load(file, &format!("{prefix}.conv2"), state, state, 3, 2, 1, 1)?,
            stride,
        })
    }

    fn uninit(n_mels: usize, state: usize, stride: usize) -> Self {
        let conv = |in_ch: usize, out_ch: usize, k_stride: usize| S3Conv1d {
            in_ch,
            out_ch,
            kernel: 3,
            stride: k_stride,
            padding: 1,
            groups: 1,
            weight: Vec::new(),
            bias: None,
        };
        ConvFrontend {
            conv1: conv(n_mels, state, stride),
            conv2: conv(state, state, 2),
            stride,
        }
    }

    /// `mel [n_mels, T]` channel-major, `mel_len` valid frames ->
    /// `(hidden [seq, state], valid)`.
    fn forward(&self, mel: &[f32], mel_len: usize) -> (Vec<f32>, usize) {
        let total = mel.len() / self.conv1.in_ch;
        let mut x = mel.to_vec();
        for d in 0..self.conv1.in_ch {
            for t in mel_len..total {
                x[d * total + t] = 0.0;
            }
        }
        let mut x = self.conv1.forward(&x);
        ops::gelu_erf(&mut x);
        let seq1 = x.len() / self.conv1.out_ch;
        let len1 = (mel_len + 2 - 1 * (3 - 1) - 1) / self.stride + 1;
        for d in 0..self.conv2.in_ch {
            for t in len1.min(seq1)..seq1 {
                x[d * seq1 + t] = 0.0;
            }
        }
        let mut x = self.conv2.forward(&x);
        ops::gelu_erf(&mut x);
        let len2 = (len1 + 2 - 1 * (3 - 1) - 1) / 2 + 1;
        let seq2 = x.len() / self.conv2.out_ch;
        let mut row_major = vec![0.0f32; x.len()];
        for d in 0..self.conv2.out_ch {
            for t in 0..seq2 {
                row_major[t * self.conv2.out_ch + d] = x[d * seq2 + t];
            }
        }
        (row_major, len2)
    }
}

/// Euclidean nearest-code lookup over the v1 codebook `[size, dim]`:
/// `argmax(-(x^2 - 2 x.c + c^2))`, ties to the first index.
#[derive(Debug)]
struct EuclideanCodebook {
    embed: Vec<f32>,
    embed_sq: Vec<f32>,
    size: usize,
    dim: usize,
}

impl EuclideanCodebook {
    /// Loader name contract in the module docs: the Rust-canonical
    /// `quantizer.embed` plus the raw PyTorch-converted names.
    const CODEBOOK_NAMES: [&str; 3] = [
        "quantizer.embed",
        "quantizer._codebook.embed",
        "quantizer.codebook.embed",
    ];

    fn load(file: &SafetensorsFile) -> Result<Self> {
        let mut found: Option<(String, usize, usize)> = None;
        for name in Self::CODEBOOK_NAMES {
            if let Some(desc) = file.descriptor(name) {
                if desc.shape.len() != 2 {
                    return Err(SpeechError::Tensor {
                        name: name.to_string(),
                        why: format!("expected 2-D codebook, got {:?}", desc.shape),
                    });
                }
                found = Some((name.to_string(), desc.shape[0], desc.shape[1]));
                break;
            }
        }
        let (name, size, dim) = found.ok_or_else(|| SpeechError::Tensor {
            name: Self::CODEBOOK_NAMES[0].to_string(),
            why: "missing v1 codebook".to_string(),
        })?;
        let embed = load_f32_shaped(file, &name, &[size, dim])?;
        let mut embed_sq = vec![0.0f32; size];
        for (r, sq) in embed_sq.iter_mut().enumerate() {
            *sq = embed[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
        }
        Ok(EuclideanCodebook {
            embed,
            embed_sq,
            size,
            dim,
        })
    }

    /// Nearest code per row of `x [rows, dim]`; first-wins argmax like
    /// the reference.
    fn nearest(&self, x: &[f32], rows: usize) -> Vec<i32> {
        let mut codes = vec![0i32; rows];
        for (r, code) in codes.iter_mut().enumerate() {
            let row = &x[r * self.dim..(r + 1) * self.dim];
            let x2: f32 = row.iter().map(|v| v * v).sum();
            let mut best = f32::INFINITY;
            let mut best_idx = 0usize;
            for c in 0..self.size {
                let e = &self.embed[c * self.dim..(c + 1) * self.dim];
                let mut dot = 0.0f32;
                for (a, &b) in row.iter().zip(e) {
                    dot += a * b;
                }
                let dist = x2 - 2.0 * dot + self.embed_sq[c];
                if dist < best {
                    best = dist;
                    best_idx = c;
                }
            }
            *code = best_idx as i32;
        }
        codes
    }

    /// Rows `[size, dim]` gathered per code.
    fn dequantize(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(codes.len() * self.dim);
        for &c in codes {
            let idx = usize::try_from(c).map_err(|_| SpeechError::Input {
                why: format!("negative code {c}"),
            })?;
            if idx >= self.size {
                return Err(SpeechError::Input {
                    why: format!("code {c} out of range for {} codes", self.size),
                });
            }
            out.extend_from_slice(&self.embed[idx * self.dim..(idx + 1) * self.dim]);
        }
        Ok(out)
    }
}

/// S3Tokenizer v1: encoder + Euclidean codebook over L2-normalized
/// frames (normalize epsilon 1e-8).
pub struct S3TokenizerV1 {
    pub config: S3Config,
    frontend: ConvFrontend,
    positional_embedding: Vec<f32>,
    blocks: Vec<S3Block>,
    codebook: EuclideanCodebook,
}

/// The v1 tokenizer name that selects the 2x-strided (25 Hz) front end.
pub const V1_25HZ: &str = "speech_tokenizer_v1_25hz";

impl S3TokenizerV1 {
    pub fn new(name: &str, config: S3Config) -> Self {
        let stride = if name == V1_25HZ { 2 } else { 1 };
        let positional_embedding = sinusoids(config.n_audio_ctx, config.n_audio_state);
        let codebook = EuclideanCodebook {
            embed: Vec::new(),
            embed_sq: Vec::new(),
            size: config.n_codebook_size,
            dim: config.n_audio_state,
        };
        S3TokenizerV1 {
            frontend: ConvFrontend::uninit(config.n_mels, config.n_audio_state, stride),
            positional_embedding,
            blocks: Vec::new(),
            codebook,
            config,
        }
    }

    /// Loads the encoder and codebook; `name` selects the v1 stride as
    /// in [`Self::new`].
    pub fn load(name: &str, config: S3Config, file: &SafetensorsFile) -> Result<Self> {
        let stride = if name == V1_25HZ { 2 } else { 1 };
        let frontend =
            ConvFrontend::load(file, "encoder", config.n_mels, config.n_audio_state, stride)?;
        let positional_embedding = load_f32_shaped(
            file,
            "encoder.positional_embedding",
            &[config.n_audio_ctx, config.n_audio_state],
        )?;
        let mut blocks = Vec::new();
        for i in 0..config.n_audio_layer {
            blocks.push(S3Block::load(
                file,
                &format!("encoder.blocks.{i}"),
                config.n_audio_state,
                config.n_audio_head,
                None,
                1e-5,
            )?);
        }
        let codebook = EuclideanCodebook::load(file)?;
        if codebook.dim != config.n_audio_state {
            return Err(SpeechError::Tensor {
                name: EuclideanCodebook::CODEBOOK_NAMES[0].to_string(),
                why: format!(
                    "codebook dim {} does not match n_audio_state {}",
                    codebook.dim, config.n_audio_state
                ),
            });
        }
        if codebook.size != config.n_codebook_size {
            return Err(SpeechError::Tensor {
                name: EuclideanCodebook::CODEBOOK_NAMES[0].to_string(),
                why: format!(
                    "codebook size {} does not match n_codebook_size {}",
                    codebook.size, config.n_codebook_size
                ),
            });
        }
        Ok(S3TokenizerV1 {
            config,
            frontend,
            positional_embedding,
            blocks,
            codebook,
        })
    }

    /// Opens a checkpoint directory holding `config.json` plus a
    /// safetensors file.
    pub fn open(name: &str, dir: &Path) -> Result<Self> {
        let file = open_checkpoint_file(dir)?;
        let config = load_config_json(dir)?;
        let config = S3Config::from_json_with_defaults(&config, S3Config::v1())?;
        S3TokenizerV1::load(name, config, &file)
    }

    /// Mel frames `[n_mels, T]` plus their valid count -> codes and
    /// code length.
    pub fn quantize(&self, mel: &[f32], mel_len: usize) -> Result<(Vec<i32>, usize)> {
        let (mut hidden, code_len, seq) = self.encode(mel, mel_len)?;
        let state = self.config.n_audio_state;
        // L2-normalize the valid frames with the 1e-8 epsilon.
        for t in 0..code_len {
            let row = &mut hidden[t * state..(t + 1) * state];
            let norm = (row.iter().map(|v| v * v).sum::<f32>() + 1e-8).sqrt();
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
        let codes = self.codebook.nearest(&hidden[..code_len * state], code_len);
        let _ = seq;
        Ok((codes, code_len))
    }

    /// Dequantizes codes back to the channel-major embedding frames
    /// `[state, T]` (the reference `decode` path).
    pub fn dequantize(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let rows = self.codebook.dequantize(codes)?;
        let state = self.config.n_audio_state;
        let mut out = vec![0.0f32; rows.len()];
        for (r, chunk) in rows.chunks_exact(state).enumerate() {
            for (d, &v) in chunk.iter().enumerate() {
                out[d * codes.len() + r] = v;
            }
        }
        Ok(out)
    }

    fn encode(&self, mel: &[f32], mel_len: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (mut x, len) = self.frontend.forward(mel, mel_len);
        let state = self.config.n_audio_state;
        let seq = x.len() / state;
        if seq > self.config.n_audio_ctx {
            return Err(SpeechError::Input {
                why: format!(
                    "encoder frames {seq} exceed n_audio_ctx {}",
                    self.config.n_audio_ctx
                ),
            });
        }
        for t in 0..seq {
            for d in 0..state {
                x[t * state + d] += self.positional_embedding[t * state + d];
            }
        }
        let mut key_bias = vec![0.0f32; seq];
        for t in len..seq {
            key_bias[t] = -1.0e10;
        }
        for block in &self.blocks {
            block.forward(&mut x, seq, &key_bias, None, None);
        }
        Ok((x, len, seq))
    }
}

/// S3Tokenizer v2: RoPE + FSMN attention encoder and the ternary FSQ
/// quantizer (project to 8 scalars, `tanh * 0.999...`, round half to
/// even, +1, base-3 digits). Sliding-window quantization covers
/// inputs longer than 30 s.
pub struct S3TokenizerV2 {
    pub config: S3Config,
    frontend: ConvFrontend,
    blocks: Vec<S3Block>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    project_down: S3Linear,
}

impl S3TokenizerV2 {
    pub fn new(config: S3Config) -> Self {
        let (rope_cos, rope_sin) = ops::rope_tables(V2_ROPE_END, V2_ROPE_DIM, 10000.0);
        let project_down = S3Linear {
            in_dim: config.n_audio_state,
            out_dim: 8,
            weight: Vec::new(),
            bias: None,
        };
        S3TokenizerV2 {
            frontend: ConvFrontend::uninit(config.n_mels, config.n_audio_state, 2),
            blocks: Vec::new(),
            rope_cos,
            rope_sin,
            project_down,
            config,
        }
    }

    pub fn load(config: S3Config, file: &SafetensorsFile) -> Result<Self> {
        if config.n_codebook_size != 6561 {
            return Err(SpeechError::BadConfig {
                field: "n_codebook_size".to_string(),
                why: "v2 asserts 3^8 = 6561 codes".to_string(),
            });
        }
        if config.n_audio_state / config.n_audio_head != V2_ROPE_DIM {
            return Err(SpeechError::BadConfig {
                field: "n_audio_head".to_string(),
                why: format!("v2 rope covers the full {}-dim head", V2_ROPE_DIM),
            });
        }
        let frontend = ConvFrontend::load(file, "encoder", config.n_mels, config.n_audio_state, 2)?;
        let mut blocks = Vec::new();
        for i in 0..config.n_audio_layer {
            blocks.push(S3Block::load(
                file,
                &format!("encoder.blocks.{i}"),
                config.n_audio_state,
                config.n_audio_head,
                Some(31),
                1e-6,
            )?);
        }
        let project_down = S3Linear::load(
            file,
            "quantizer.fsq_codebook.project_down",
            config.n_audio_state,
            8,
            true,
        )?;
        let (rope_cos, rope_sin) = ops::rope_tables(V2_ROPE_END, V2_ROPE_DIM, 10000.0);
        Ok(S3TokenizerV2 {
            config,
            frontend,
            blocks,
            rope_cos,
            rope_sin,
            project_down,
        })
    }

    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &Path) -> Result<Self> {
        let file = open_checkpoint_file(dir)?;
        let config = load_config_json(dir)?;
        let config = S3Config::from_json_with_defaults(&config, S3Config::v2())?;
        S3TokenizerV2::load(config, &file)
    }

    /// Mel frames plus valid count -> codes. Mel longer than 3000
    /// frames takes the sliding-window path with the half-overlap
    /// merge.
    pub fn quantize(&self, mel: &[f32], mel_len: usize) -> Result<(Vec<i32>, usize)> {
        if mel_len > MAX_FRAMES {
            self.quantize_long(mel, mel_len)
        } else {
            let (hidden, code_len, _) = self.encode_window(mel, mel_len)?;
            let state = self.config.n_audio_state;
            let codes = self.fsq_codes(&hidden[..code_len * state], code_len)?;
            Ok((codes, code_len))
        }
    }

    /// 30 s windows with a 4 s overlap; `merge_tokenized_segments`
    /// trims half the overlap from each seam: the first window drops
    /// its last 50 codes, the last window drops its first 50, and a
    /// middle window would drop both.
    fn quantize_long(&self, mel: &[f32], mel_len: usize) -> Result<(Vec<i32>, usize)> {
        let n_mels = self.config.n_mels;
        let mut segments: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        while start < mel_len {
            let seg_len = (mel_len - start).min(MAX_FRAMES);
            segments.push((start, seg_len));
            start += FRAMES_PER_STRIDE;
        }
        let overlap_tokens = (4 / 2) * V2_TOKEN_RATE;
        let mut merged: Vec<i32> = Vec::new();
        let count = segments.len();
        for (i, &(seg_start, seg_len)) in segments.iter().enumerate() {
            let mut window = vec![0.0f32; n_mels * MAX_FRAMES];
            for d in 0..n_mels {
                window[d * MAX_FRAMES..d * MAX_FRAMES + seg_len].copy_from_slice(
                    &mel[d * mel_len + seg_start..d * mel_len + seg_start + seg_len],
                );
            }
            let (hidden, code_len, _) = self.encode_window(&window, seg_len)?;
            let state = self.config.n_audio_state;
            let codes = self.fsq_codes(&hidden[..code_len * state], code_len)?;
            let left = if i == 0 { 0 } else { overlap_tokens };
            let right = if i != count - 1 {
                codes.len() - overlap_tokens
            } else {
                codes.len()
            };
            merged.extend_from_slice(&codes[left..right]);
        }
        let len = merged.len();
        Ok((merged, len))
    }

    fn encode_window(&self, mel: &[f32], mel_len: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (mut x, len) = self.frontend.forward(mel, mel_len);
        let state = self.config.n_audio_state;
        let seq = x.len() / state;
        let mut key_bias = vec![0.0f32; seq];
        let mut mask_pad = vec![1.0f32; seq];
        for t in len..seq {
            key_bias[t] = -1.0e10;
            mask_pad[t] = 0.0;
        }
        let rope = (
            &self.rope_cos[..seq * (V2_ROPE_DIM / 2)],
            &self.rope_sin[..seq * (V2_ROPE_DIM / 2)],
        );
        for block in &self.blocks {
            block.forward(&mut x, seq, &key_bias, Some(&mask_pad), Some(rope));
        }
        Ok((x, len, seq))
    }

    fn fsq_codes(&self, hidden: &[f32], frames: usize) -> Result<Vec<i32>> {
        let h = self.project_down.forward(hidden, frames);
        let powers: [f32; 8] = [1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0];
        let mut codes = Vec::with_capacity(frames);
        for row in h.chunks_exact(8) {
            let mut mu = 0.0f32;
            for (i, &v) in row.iter().enumerate() {
                // h = round(tanh(v) * scale) + 1, then base-3 digits.
                let digit = (v.tanh() * FSQ_SCALE).round_ties_even() + 1.0;
                mu += digit * powers[i];
            }
            codes.push(mu as i32);
        }
        Ok(codes)
    }
}

fn open_checkpoint_file(dir: &Path) -> Result<SafetensorsFile> {
    let canonical = dir.join("model.safetensors");
    if canonical.exists() {
        return SafetensorsFile::open(&canonical).map_err(SpeechError::from);
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| SpeechError::BadConfig {
            field: dir.display().to_string(),
            why: e.to_string(),
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    match entries.len() {
        1 => SafetensorsFile::open(entries.remove(0).as_path()).map_err(SpeechError::from),
        0 => Err(SpeechError::BadConfig {
            field: dir.display().to_string(),
            why: "no safetensors checkpoint found".to_string(),
        }),
        _ => Err(SpeechError::BadConfig {
            field: dir.display().to_string(),
            why: "multiple safetensors files; name one model.safetensors".to_string(),
        }),
    }
}

fn load_config_json(dir: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(dir.join("config.json")).map_err(|e| SpeechError::BadConfig {
            field: "config.json".to_string(),
            why: e.to_string(),
        })?;
    serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
        field: "config.json".to_string(),
        why: e.to_string(),
    })
}

/// The reference `s3/utils.py::log_mel_spectrogram`: whisper's 400-FFT
/// / 160-hop front end with a periodic Hann, reflect padding, power
/// spectrum, Slaney mel, `log10` floored at 1e-10, peak-relative clamp
/// at `max - 8`, and `(x + 4) / 4`. Unlike whisper's own frontend the
/// final centered frame is kept.
pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Result<Vec<Vec<f32>>> {
    let options = mel::MelSpectrogramOptions {
        stft: stft::StftOptions {
            fft_size: WHISPER_N_FFT,
            hop: WHISPER_HOP,
            window: crate::dsp::hann_window(WHISPER_N_FFT),
            center: true,
        },
        num_mels: n_mels,
        sample_rate: WHISPER_SAMPLE_RATE,
        fmin: 0.0,
        fmax: Some(WHISPER_FMAX),
        scale: mel::MelScale::Slaney,
        power: 2.0,
    };
    let frames = mel::mel_spectrogram(samples, &options)?;
    let mut peak = f32::NEG_INFINITY;
    for frame in &frames {
        for &x in frame {
            peak = peak.max(x.max(1e-10).log10());
        }
    }
    Ok(frames
        .into_iter()
        .map(|frame| {
            frame
                .into_iter()
                .map(|x| ((x.max(1e-10).log10().max(peak - 8.0)) + 4.0) / 4.0)
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests;
