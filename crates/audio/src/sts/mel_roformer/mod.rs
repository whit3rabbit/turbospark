//! Mel-Band RoFormer music/vocal source separation.
//!
//! Port of `mlx_audio/sts/models/mel_roformer` at reference commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Pipeline: symmetric-Hann
//! STFT (center, reflect) -> channel-interleaved (CaC) spectrum ->
//! binarized Slaney mel band split with per-band RMSNorm+linear
//! projections -> depth x (time-axis, freq-axis) RoFormer transformers
//! (F-normalize RMSNorm, interleaved RoPE, per-head sigmoid gates) ->
//! per-band MLP mask estimation with GLU -> scatter-merge with overlap
//! averaging -> complex multiply -> overlap-add iSTFT.
//!
//! The reference deliberately ships no default config: callers must name
//! the checkpoint family preset. `open` keeps that posture and refuses a
//! directory without a config.json. Weights are expected in the MLX
//! (post-sanitize) key layout; packed `to_qkv` keys are refused.

use std::path::{Path, PathBuf};

use crate::error::SpeechError;
use crate::fft::{ComplexF32, RealFftPlan};
use crate::mel::{mel_filterbank_cached, MelScale};
use crate::ops;
use turbospark_model_io::safetensors::SafetensorsFile;

type Result<T> = std::result::Result<T, SpeechError>;

/// Architecture and STFT configuration, matching `MelRoFormerConfig`.
#[derive(Debug, Clone)]
pub struct MelRoFormerConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub num_bands: usize,
    pub num_stems: usize,
    pub ff_mult: usize,
    pub mlp_expansion_factor: usize,
    pub mask_estimator_depth: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub sample_rate: usize,
    pub chunk_size: usize,
    pub num_overlap: usize,
    pub checkpoint_family: Option<String>,
}

impl MelRoFormerConfig {
    /// The reference default field values (only reachable through an
    /// explicit config in the reference; used by fixture tests).
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let obj = value.as_object().ok_or_else(|| SpeechError::BadConfig {
            field: "config".to_string(),
            why: "expected a JSON object".to_string(),
        })?;
        let get = |name: &str, default: usize| -> Result<usize> {
            match obj.get(name) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.into(),
                        why: "expected a nonnegative integer".into(),
                    }),
            }
        };
        let mut config = Self {
            dim: get("dim", 384)?,
            depth: get("depth", 6)?,
            heads: get("heads", 8)?,
            dim_head: get("dim_head", 64)?,
            num_bands: get("num_bands", 60)?,
            num_stems: get("num_stems", 1)?,
            ff_mult: get("ff_mult", 4)?,
            mlp_expansion_factor: get("mlp_expansion_factor", 4)?,
            mask_estimator_depth: get("mask_estimator_depth", 2)?,
            n_fft: get("n_fft", 2048)?,
            hop_length: get("hop_length", 441)?,
            win_length: get("win_length", 2048)?,
            sample_rate: get("sample_rate", 44100)?,
            chunk_size: get("chunk_size", 352800)?,
            num_overlap: get("num_overlap", 2)?,
            checkpoint_family: obj
                .get("checkpoint_family")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        };
        if config.n_fft % 2 != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_fft".to_string(),
                why: "must be even".to_string(),
            });
        }
        if config.win_length == 0 {
            config.win_length = config.n_fft;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let invalid = |field: &str, why: &str| SpeechError::BadConfig {
            field: field.into(),
            why: why.into(),
        };
        for (field, value) in [
            ("dim", self.dim),
            ("depth", self.depth),
            ("heads", self.heads),
            ("dim_head", self.dim_head),
            ("num_bands", self.num_bands),
            ("ff_mult", self.ff_mult),
            ("mlp_expansion_factor", self.mlp_expansion_factor),
            ("mask_estimator_depth", self.mask_estimator_depth),
            ("n_fft", self.n_fft),
            ("hop_length", self.hop_length),
            ("sample_rate", self.sample_rate),
            ("chunk_size", self.chunk_size),
            ("num_overlap", self.num_overlap),
        ] {
            if value == 0 {
                return Err(invalid(field, "must be positive"));
            }
        }
        if self.n_fft % 2 != 0 || self.dim_head % 2 != 0 {
            return Err(invalid("n_fft/dim_head", "must be even"));
        }
        if self.num_stems != 1 {
            return Err(invalid(
                "num_stems",
                "this loader supports exactly one stem",
            ));
        }
        if self.win_length != self.n_fft || self.hop_length > self.n_fft {
            return Err(invalid(
                "win_length/hop_length",
                "requires win_length = n_fft and hop_length <= n_fft",
            ));
        }
        if self.num_bands > self.freq_bins() || self.sample_rate > u32::MAX as usize {
            return Err(invalid(
                "num_bands/sample_rate",
                "unsupported filterbank geometry",
            ));
        }
        for product in [
            self.heads.checked_mul(self.dim_head),
            self.dim.checked_mul(self.ff_mult),
            self.dim.checked_mul(self.mlp_expansion_factor),
            self.n_fft
                .checked_mul(self.num_bands)
                .and_then(|v| v.checked_mul(4)),
        ] {
            if product.is_none() {
                return Err(invalid("dimensions", "size overflow"));
            }
        }
        Ok(())
    }

    fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// ZFTurbo-style F-normalize RMSNorm:
/// `x / max(||x||, 1e-12) * sqrt(dim) * weight` per row.
struct RmsNormF {
    weight: Vec<f32>,
    dim: usize,
}

impl RmsNormF {
    fn run(&self, x: &mut [f32]) {
        let scale = (self.dim as f32).sqrt();
        for row in x.chunks_exact_mut(self.dim) {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            let f = scale / norm.max(1e-12);
            for (value, &weight) in row.iter_mut().zip(&self.weight) {
                *value = *value * f * weight;
            }
        }
    }
}

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

/// Per-band geometry: gather indices into the CaC spectrum (each stereo
/// frequency bin contributes entries `2b` and `2b + 1`), per-band input
/// dims, and the overlap count used to average scattered masks.
struct BandGeometry {
    freq_indices: Vec<Vec<usize>>,
    band_dims: Vec<usize>,
    num_bands_per_freq: Vec<f32>,
    freq_bins: usize,
}

impl BandGeometry {
    fn build(config: &MelRoFormerConfig) -> Self {
        let freq_bins = config.freq_bins();
        let bank = mel_filterbank_cached(
            config.num_bands,
            config.n_fft,
            config.sample_rate as u32,
            0.0,
            None,
            MelScale::Slaney,
        )
        .expect("slaney filterbank");
        // Binarize the support; force the DC and Nyquist bins into the
        // first and last band (the reference assigns them explicitly).
        let mut support: Vec<bool> = bank.weights.iter().map(|&w| w > 0.0).collect();
        support[0] = true;
        support[config.num_bands * freq_bins - 1] = true;

        let mut freq_indices = Vec::with_capacity(config.num_bands);
        let mut band_dims = Vec::with_capacity(config.num_bands);
        let mut num_bands_per_freq = vec![0.0f32; freq_bins * 2];
        for band in 0..config.num_bands {
            let mut cac = Vec::new();
            for b in 0..freq_bins {
                if support[band * freq_bins + b] {
                    cac.push(b * 2);
                    cac.push(b * 2 + 1);
                }
            }
            if cac.is_empty() {
                // Reference fallback for an empty band support.
                cac.push(band * 2);
                cac.push(band * 2 + 1);
            }
            for &idx in &cac {
                num_bands_per_freq[idx] += 1.0;
            }
            band_dims.push(cac.len() * 2);
            freq_indices.push(cac);
        }
        for value in num_bands_per_freq.iter_mut() {
            *value = (*value).max(1.0);
        }
        Self {
            freq_indices,
            band_dims,
            num_bands_per_freq,
            freq_bins,
        }
    }
}

struct BandSplit {
    geometry: BandGeometry,
    /// Per band: RMSNorm gain and the projection to `dim`.
    norms: Vec<RmsNormF>,
    projections: Vec<Linear>,
}

impl BandSplit {
    /// `stft_repr` is `[F2, T, 2]` (CaC interleaved channels, real/imag
    /// last). Returns `[T, num_bands, dim]`.
    fn split(&self, stft_repr: &[f32], frames: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * self.geometry.band_dims.len() * dim];
        for (band, indices) in self.geometry.freq_indices.iter().enumerate() {
            let mut band_input = vec![0.0f32; frames * indices.len() * 2];
            for (j, &idx) in indices.iter().enumerate() {
                for t in 0..frames {
                    band_input[t * indices.len() * 2 + j * 2] = stft_repr[(idx * frames + t) * 2];
                    band_input[t * indices.len() * 2 + j * 2 + 1] =
                        stft_repr[(idx * frames + t) * 2 + 1];
                }
            }
            self.norms[band].run(&mut band_input);
            let projected = self.projections[band].run(&band_input, frames);
            let base = band * dim;
            for t in 0..frames {
                out[t * self.geometry.band_dims.len() * dim + base
                    ..t * self.geometry.band_dims.len() * dim + base + dim]
                    .copy_from_slice(&projected[t * dim..(t + 1) * dim]);
            }
        }
        out
    }

    /// Scatters per-band masks back over the spectrum with overlap
    /// averaging. Returns `[F2, T, 2]`.
    fn merge(&self, band_masks: &[Vec<f32>], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.geometry.freq_bins * 2 * frames * 2];
        for (band, indices) in self.geometry.freq_indices.iter().enumerate() {
            let mask = &band_masks[band];
            let n_freqs = indices.len();
            for (j, &idx) in indices.iter().enumerate() {
                for t in 0..frames {
                    out[(idx * frames + t) * 2] += mask[t * n_freqs * 2 + j * 2];
                    out[(idx * frames + t) * 2 + 1] += mask[t * n_freqs * 2 + j * 2 + 1];
                }
            }
        }
        for (idx, &count) in self.geometry.num_bands_per_freq.iter().enumerate() {
            for t in 0..frames {
                out[(idx * frames + t) * 2] /= count;
                out[(idx * frames + t) * 2 + 1] /= count;
            }
        }
        out
    }
}

struct RoFormerAttention {
    norm: RmsNormF,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_gates: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    base: f32,
}

impl RoFormerAttention {
    /// `x` is `[rows, T, dim]`; returns `[rows, T, dim]`.
    fn forward(&self, x: &[f32], rows: usize, seq: usize) -> Vec<f32> {
        let inner = self.heads * self.dim_head;
        let mut h = x.to_vec();
        self.norm.run(&mut h);

        let q = self.to_q.run(&h, rows * seq);
        let k = self.to_k.run(&h, rows * seq);
        let v = self.to_v.run(&h, rows * seq);

        // Interleaved RoPE with per-head duplicated frequencies.
        let half = self.dim_head / 2;
        let mut cos_t = vec![0.0f32; seq * self.dim_head];
        let mut sin_t = vec![0.0f32; seq * self.dim_head];
        for t in 0..seq {
            for i in 0..half {
                let freq = 1.0 / self.base.powf(i as f32 / half as f32);
                let ang = t as f32 * freq;
                let (c, s) = (ang.cos(), ang.sin());
                cos_t[t * self.dim_head + 2 * i] = c;
                cos_t[t * self.dim_head + 2 * i + 1] = c;
                sin_t[t * self.dim_head + 2 * i] = s;
                sin_t[t * self.dim_head + 2 * i + 1] = s;
            }
        }
        let mut qr = q.clone();
        let mut kr = k.clone();
        for buf in [&mut qr, &mut kr] {
            for row in 0..rows * seq {
                let t = row % seq;
                let base = row * inner;
                for head in 0..self.heads {
                    let hbase = base + head * self.dim_head;
                    for i in 0..half {
                        let even = buf[hbase + 2 * i];
                        let odd = buf[hbase + 2 * i + 1];
                        let c = cos_t[t * self.dim_head + 2 * i];
                        let s = sin_t[t * self.dim_head + 2 * i];
                        buf[hbase + 2 * i] = even * c - odd * s;
                        buf[hbase + 2 * i + 1] = odd * c + even * s;
                    }
                }
            }
        }

        let scale = 1.0 / (self.dim_head as f32).sqrt();
        let mut attended = vec![0.0f32; rows * seq * inner];
        for row in 0..rows {
            for head in 0..self.heads {
                for t in 0..seq {
                    let qbase = (row * seq + t) * inner + head * self.dim_head;
                    let mut acc = vec![0.0f32; self.dim_head];
                    let mut max_score = f32::NEG_INFINITY;
                    let mut scores = vec![0.0f32; seq];
                    for u in 0..seq {
                        let kbase = (row * seq + u) * inner + head * self.dim_head;
                        let mut dot = 0.0f32;
                        for d in 0..self.dim_head {
                            dot += qr[qbase + d] * kr[kbase + d];
                        }
                        scores[u] = dot * scale;
                        max_score = max_score.max(scores[u]);
                    }
                    let mut sum = 0.0f32;
                    for score in scores.iter_mut() {
                        *score = (*score - max_score).exp();
                        sum += *score;
                    }
                    for score in scores.iter_mut() {
                        *score /= sum;
                    }
                    for (u, &weight) in scores.iter().enumerate() {
                        let vbase = (row * seq + u) * inner + head * self.dim_head;
                        for d in 0..self.dim_head {
                            acc[d] += weight * v[vbase + d];
                        }
                    }
                    attended[qbase..qbase + self.dim_head].copy_from_slice(&acc);
                }
            }
        }

        // Per-head sigmoid gates.
        let gates = self.to_gates.run(&h, rows * seq);
        let mut merged = vec![0.0f32; rows * seq * inner];
        for row in 0..rows {
            for t in 0..seq {
                let src = (row * seq + t) * inner;
                for head in 0..self.heads {
                    let gate = ops::sigmoid(gates[(row * seq + t) * self.heads + head]);
                    for d in 0..self.dim_head {
                        merged[src + head * self.dim_head + d] =
                            attended[src + head * self.dim_head + d] * gate;
                    }
                }
            }
        }
        self.to_out.run(&merged, rows * seq)
    }
}

struct RoFormerFfn {
    norm: RmsNormF,
    expand: Linear,
    compress: Linear,
}

impl RoFormerFfn {
    fn forward(&self, x: &[f32], rows: usize, seq: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        self.norm.run(&mut h);
        let mut h = self.expand.run(&h, rows * seq);
        ops::gelu_erf(&mut h);
        let mut out = self.compress.run(&h, rows * seq);
        for (o, xv) in out.iter_mut().zip(x) {
            *o += xv;
        }
        out
    }
}

/// Single-axis transformer (the reference constructs depth 1) with an
/// output RMSNorm.
struct AxisTransformer {
    attentions: Vec<RoFormerAttention>,
    ffns: Vec<RoFormerFfn>,
    norm: RmsNormF,
}

impl AxisTransformer {
    fn forward(&self, x: &[f32], rows: usize, seq: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        for (attn, ffn) in self.attentions.iter().zip(&self.ffns) {
            let attended = attn.forward(&h, rows, seq);
            for (hv, av) in h.iter_mut().zip(attended) {
                *hv += av;
            }
            let fed = ffn.forward(&h, rows, seq);
            h = fed;
        }
        self.norm.run(&mut h);
        h
    }
}

struct MaskEstimator {
    /// Per band: hidden linears (tanh between) then the GLU projection.
    layers: Vec<Vec<Linear>>,
}

impl MaskEstimator {
    /// `x` is `[T, num_bands, dim]`; returns one mask per band.
    fn forward(&self, x: &[f32], frames: usize, num_bands: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut masks = Vec::with_capacity(num_bands);
        for band in 0..num_bands {
            let mut h = vec![0.0f32; frames * dim];
            for t in 0..frames {
                h[t * dim..(t + 1) * dim].copy_from_slice(
                    &x[(t * num_bands + band) * dim..(t * num_bands + band + 1) * dim],
                );
            }
            for layer in self.layers[band].iter().take(self.layers[band].len() - 1) {
                let mut out = layer.run(&h, frames);
                for value in out.iter_mut() {
                    *value = value.tanh();
                }
                h = out;
            }
            let out = self.layers[band]
                .last()
                .expect("final layer")
                .run(&h, frames);
            let out_dim = out.len() / frames;
            let half = out_dim / 2;
            let mut gated = vec![0.0f32; frames * half];
            for t in 0..frames {
                for j in 0..half {
                    gated[t * half + j] =
                        out[t * out_dim + j] * ops::sigmoid(out[t * out_dim + half + j]);
                }
            }
            masks.push(gated);
        }
        masks
    }
}

/// The loaded Mel-Band RoFormer model.
pub struct MelRoFormer {
    pub config: MelRoFormerConfig,
    band_split: BandSplit,
    /// depth x [time transformer, freq transformer]
    layers: Vec<[AxisTransformer; 2]>,
    mask_estimator: MaskEstimator,
    window: Vec<f32>,
}

impl MelRoFormer {
    /// Opens a checkpoint directory: the weights file (any
    /// `.safetensors`, preferring the reference names) and a
    /// `config.json` (the reference has no default config on purpose).
    pub fn open(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: format!(
                    "no config.json in {}; the reference requires naming the checkpoint family preset",
                    dir.display()
                ),
            });
        }
        let config = MelRoFormerConfig::from_json(&crate::quant::read_json(&config_path)?)?;
        let mut candidates: Vec<PathBuf> = [
            "mel_roformer_vocals.safetensors",
            "weights.safetensors",
            "model.safetensors",
        ]
        .iter()
        .map(|name| dir.join(name))
        .collect();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut others: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            others.sort();
            candidates.extend(others);
        }
        let weights_path =
            candidates
                .iter()
                .find(|p| p.exists())
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "weights".to_string(),
                    why: format!("no .safetensors file in {}", dir.display()),
                })?;
        let file = SafetensorsFile::open(weights_path).map_err(|e| SpeechError::BadConfig {
            field: weights_path.display().to_string(),
            why: e.to_string(),
        })?;
        Self::load(config, &file)
    }

    /// Loads weights in the MLX (post-sanitize) key layout.
    pub fn load(config: MelRoFormerConfig, file: &SafetensorsFile) -> Result<Self> {
        config.validate()?;
        let geometry = BandGeometry::build(&config);
        let dim = config.dim;

        let f32_tensor = |name: &str| -> Result<Vec<f32>> {
            file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
                name: name.to_string(),
                why: e.to_string(),
            })
        };
        let linear = |name: &str, expected_out: usize, expected_in: usize| -> Result<Linear> {
            let weight = f32_tensor(&format!("{name}.weight"))?;
            let bias = if file.contains_tensor(&format!("{name}.bias")) {
                Some(f32_tensor(&format!("{name}.bias"))?)
            } else {
                None
            };
            let desc =
                file.descriptor(&format!("{name}.weight"))
                    .ok_or_else(|| SpeechError::Tensor {
                        name: name.to_string(),
                        why: "weight descriptor missing".to_string(),
                    })?;
            if desc.shape.len() != 2 {
                return Err(SpeechError::Tensor {
                    name: name.to_string(),
                    why: format!("expected 2-D weight, got {:?}", desc.shape),
                });
            }
            if desc.shape != [expected_out, expected_in]
                || bias.as_ref().is_some_and(|v| v.len() != expected_out)
            {
                return Err(SpeechError::Tensor { name: name.into(), why: format!("expected [{expected_out}, {expected_in}] weight and matching bias, got {:?}", desc.shape) });
            }
            if weight.len() != desc.shape[0] * desc.shape[1] {
                return Err(SpeechError::Tensor {
                    name: name.to_string(),
                    why: "weight length mismatch".to_string(),
                });
            }
            Ok(Linear {
                weight,
                bias,
                out_dim: desc.shape[0],
                in_dim: desc.shape[1],
            })
        };
        let rms = |name: &str, expected: usize| -> Result<RmsNormF> {
            let weight = f32_tensor(&format!("{name}.weight"))?;
            if weight.len() != expected {
                return Err(SpeechError::Tensor {
                    name: format!("{name}.weight"),
                    why: format!("expected {expected} values, got {}", weight.len()),
                });
            }
            Ok(RmsNormF {
                weight,
                dim: expected,
            })
        };
        if file.contains_tensor("band_split.to_qkv.weight")
            || (0..1)
                .any(|i| file.contains_tensor(&format!("layers.{i}.0.layers.0.0.to_qkv.weight")))
        {
            return Err(SpeechError::Unsupported {
                why: "checkpoint still carries packed to_qkv weights; run the reference \
                      sanitize (convert.py) before loading"
                    .to_string(),
            });
        }

        // Band split projections.
        let mut norms = Vec::with_capacity(geometry.band_dims.len());
        let mut projections = Vec::with_capacity(geometry.band_dims.len());
        for (band, &bd) in geometry.band_dims.iter().enumerate() {
            norms.push(rms(&format!("band_split.to_features.{band}.0"), bd)?);
            projections.push(linear(
                &format!("band_split.to_features.{band}.1"),
                dim,
                bd,
            )?);
        }
        let band_split = BandSplit {
            geometry,
            norms,
            projections,
        };

        // Dual-axis transformers.
        let mut layers = Vec::with_capacity(config.depth);
        for d in 0..config.depth {
            let mut axes: [AxisTransformer; 2] = [
                AxisTransformer {
                    attentions: Vec::new(),
                    ffns: Vec::new(),
                    norm: rms(&format!("layers.{d}.0.norm"), dim)?,
                },
                AxisTransformer {
                    attentions: Vec::new(),
                    ffns: Vec::new(),
                    norm: rms(&format!("layers.{d}.1.norm"), dim)?,
                },
            ];
            for (axis, transformer) in axes.iter_mut().enumerate() {
                let prefix = format!("layers.{d}.{axis}.layers.0");
                let attention = RoFormerAttention {
                    norm: rms(&format!("{prefix}.0.norm"), dim)?,
                    to_q: linear(
                        &format!("{prefix}.0.to_q"),
                        config.heads * config.dim_head,
                        dim,
                    )?,
                    to_k: linear(
                        &format!("{prefix}.0.to_k"),
                        config.heads * config.dim_head,
                        dim,
                    )?,
                    to_v: linear(
                        &format!("{prefix}.0.to_v"),
                        config.heads * config.dim_head,
                        dim,
                    )?,
                    to_gates: linear(&format!("{prefix}.0.to_gates"), config.heads, dim)?,
                    to_out: linear(
                        &format!("{prefix}.0.to_out"),
                        dim,
                        config.heads * config.dim_head,
                    )?,
                    heads: config.heads,
                    dim_head: config.dim_head,
                    base: 10000.0,
                };
                let ffn = RoFormerFfn {
                    norm: rms(&format!("{prefix}.1.net.0"), dim)?,
                    expand: linear(&format!("{prefix}.1.net.1"), dim * config.ff_mult, dim)?,
                    compress: linear(&format!("{prefix}.1.net.4"), dim, dim * config.ff_mult)?,
                };
                transformer.attentions.push(attention);
                transformer.ffns.push(ffn);
            }
            layers.push(axes);
        }

        // Mask estimator MLPs.
        let mut estimator_layers = Vec::with_capacity(band_split.geometry.band_dims.len());
        for (band, &bd) in band_split.geometry.band_dims.iter().enumerate() {
            let hidden = dim * config.mlp_expansion_factor;
            let mut mlp = vec![linear(
                &format!("mask_estimators.0.to_freqs.{band}.0.0"),
                hidden,
                dim,
            )?];
            for mid in 1..config.mask_estimator_depth {
                mlp.push(linear(
                    &format!("mask_estimators.0.to_freqs.{band}.{mid}.0"),
                    hidden,
                    hidden,
                )?);
            }
            mlp.push(linear(
                &format!(
                    "mask_estimators.0.to_freqs.{band}.{}.0",
                    config.mask_estimator_depth
                ),
                bd * 2,
                hidden,
            )?);
            let last = mlp.last().expect("final mask linear");
            if last.out_dim != bd * 2 {
                return Err(SpeechError::Tensor {
                    name: format!("mask_estimators.0.to_freqs.{band}"),
                    why: format!(
                        "final projection out {} != band dim {} x 2",
                        last.out_dim, bd
                    ),
                });
            }
            estimator_layers.push(mlp);
        }

        // Symmetric Hann window (np.hanning(n_fft + 1)[:-1]); the
        // crate's periodic Hann matches it exactly.
        let window = crate::dsp::hann_window(config.n_fft);
        Ok(Self {
            config,
            band_split,
            layers,
            mask_estimator: MaskEstimator {
                layers: estimator_layers,
            },
            window,
        })
    }

    /// Separates the stereo input `[2, samples]` into one stem
    /// `[2, samples]`.
    pub fn forward(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() || audio.len() % 2 != 0 || audio.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "expected nonempty finite planar stereo PCM with equal channel lengths".into(),
            });
        }
        let channels = 2;
        let original_length = audio.len() / channels;
        let config = &self.config;
        let freq_bins = config.freq_bins();

        // STFT per channel (center=True, reflect padding). Audio is the
        // reference [2, samples] planar layout.
        let mut spectra = Vec::with_capacity(channels);
        for c in 0..channels {
            let signal: Vec<f32> = audio[c * original_length..(c + 1) * original_length].to_vec();
            spectra.push(self.stft_reflect(&signal)?);
        }
        let frames = spectra[0].len();

        // CaC interleave: [channels, freq, T] -> [freq*2, T, 2] with
        // real/imag last. stft returns [T, freq] complex per channel.
        let mut stft_repr = vec![0.0f32; freq_bins * 2 * frames * 2];
        for b in 0..freq_bins {
            for t in 0..frames {
                // CaC interleave: entry 2b is the left channel, 2b + 1
                // the right; the last axis is (real, imag).
                let dst = ((b * 2) * frames + t) * 2;
                stft_repr[dst] = spectra[0][t][b].re;
                stft_repr[dst + 1] = spectra[0][t][b].im;
                let dst = ((b * 2 + 1) * frames + t) * 2;
                stft_repr[dst] = spectra[1][t][b].re;
                stft_repr[dst + 1] = spectra[1][t][b].im;
            }
        }

        // Band split -> [T, num_bands, dim].
        let num_bands = self.band_split.geometry.band_dims.len();
        let mut x = self.band_split.split(&stft_repr, frames, config.dim);

        // Dual-axis transformers.
        for pair in &self.layers {
            // Time axis: rows = num_bands sequences over frames.
            let mut time_in = vec![0.0f32; num_bands * frames * config.dim];
            for t in 0..frames {
                for b in 0..num_bands {
                    let src = (t * num_bands + b) * config.dim;
                    let dst = (b * frames + t) * config.dim;
                    time_in[dst..dst + config.dim].copy_from_slice(&x[src..src + config.dim]);
                }
            }
            let time_out = pair[0].forward(&time_in, num_bands, frames);
            for t in 0..frames {
                for b in 0..num_bands {
                    let src = (b * frames + t) * config.dim;
                    let dst = (t * num_bands + b) * config.dim;
                    x[dst..dst + config.dim].copy_from_slice(&time_out[src..src + config.dim]);
                }
            }
            // Freq axis: rows = frames sequences over bands.
            let freq_out = pair[1].forward(&x, frames, num_bands);
            x = freq_out;
        }

        // Masks and merge.
        let band_masks = self
            .mask_estimator
            .forward(&x, frames, num_bands, config.dim);
        let full_mask = self.band_split.merge(&band_masks, frames);

        // Complex multiply and de-interleave.
        let mut separated = Vec::with_capacity(channels * original_length);
        let mut out_channels: Vec<Vec<ComplexF32>> = Vec::with_capacity(channels);
        for _ in 0..channels {
            out_channels.push(vec![ComplexF32::new(0.0, 0.0); frames * freq_bins]);
        }
        for b in 0..freq_bins {
            for t in 0..frames {
                // Channel c lives at CaC entry 2b + c with (re, im) last.
                let in_r0 = stft_repr[((b * 2) * frames + t) * 2];
                let in_i0 = stft_repr[((b * 2) * frames + t) * 2 + 1];
                let in_r1 = stft_repr[((b * 2 + 1) * frames + t) * 2];
                let in_i1 = stft_repr[((b * 2 + 1) * frames + t) * 2 + 1];
                let m_r0 = full_mask[((b * 2) * frames + t) * 2];
                let m_i0 = full_mask[((b * 2) * frames + t) * 2 + 1];
                let m_r1 = full_mask[((b * 2 + 1) * frames + t) * 2];
                let m_i1 = full_mask[((b * 2 + 1) * frames + t) * 2 + 1];
                out_channels[0][t * freq_bins + b] =
                    ComplexF32::new(in_r0 * m_r0 - in_i0 * m_i0, in_r0 * m_i0 + in_i0 * m_r0);
                out_channels[1][t * freq_bins + b] =
                    ComplexF32::new(in_r1 * m_r1 - in_i1 * m_i1, in_r1 * m_i1 + in_i1 * m_r1);
            }
        }
        for channel in &out_channels {
            let wave = self.istft_normalized(channel, frames, original_length)?;
            separated.extend(wave);
        }
        Ok(separated)
    }

    /// Forward STFT with center=True reflect padding; returns per-frame
    /// spectra of `n_fft / 2 + 1` bins.
    fn stft_reflect(&self, signal: &[f32]) -> Result<Vec<Vec<ComplexF32>>> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let pad = n_fft / 2;
        if signal.len() + 2 * pad < n_fft {
            return Err(SpeechError::Input {
                why: format!(
                    "signal length {} too short for n_fft {}",
                    signal.len(),
                    n_fft
                ),
            });
        }
        // Reflect pad without repeating the edge sample: the prefix is
        // x[1..=pad] reversed and the suffix is x[-(pad+1)..-1] reversed.
        let mut padded = Vec::with_capacity(signal.len() + 2 * pad);
        let head_take = pad.min(signal.len().saturating_sub(1));
        for value in signal[1..=head_take].iter().rev() {
            padded.push(*value);
        }
        while padded.len() < pad {
            padded.push(signal[0]);
        }
        padded.extend_from_slice(signal);
        let tail_take = pad.min(signal.len().saturating_sub(1));
        for value in signal[signal.len() - 1 - tail_take..signal.len() - 1]
            .iter()
            .rev()
        {
            padded.push(*value);
        }
        while padded.len() < signal.len() + 2 * pad {
            padded.push(*signal.last().unwrap_or(&0.0));
        }

        let frames = 1 + (padded.len() - n_fft) / hop;
        let plan = RealFftPlan::cached(n_fft).map_err(|e| SpeechError::Input {
            why: format!("stft plan: {e}"),
        })?;
        let mut out = Vec::with_capacity(frames);
        let mut frame = vec![0.0f32; n_fft];
        for f in 0..frames {
            let start = f * hop;
            for (i, value) in frame.iter_mut().enumerate() {
                *value = padded[start + i] * self.window[i];
            }
            out.push(plan.forward(&frame).map_err(|e| SpeechError::Input {
                why: format!("stft: {e}"),
            })?);
        }
        Ok(out)
    }

    /// Inverse STFT with window-squared (COLA) normalization and the
    /// center strip removed, trimmed/padded to `length`.
    fn istft_normalized(
        &self,
        spectrum: &[ComplexF32],
        frames: usize,
        length: usize,
    ) -> Result<Vec<f32>> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let plan = RealFftPlan::cached(n_fft).map_err(|e| SpeechError::Input {
            why: format!("istft plan: {e}"),
        })?;
        let total = (frames - 1) * hop + n_fft;
        let mut acc = vec![0.0f32; total];
        let mut norm = vec![0.0f32; total];
        for f in 0..frames {
            let samples = plan
                .inverse(&spectrum[f * (n_fft / 2 + 1)..(f + 1) * (n_fft / 2 + 1)])
                .map_err(|e| SpeechError::Input {
                    why: format!("istft: {e}"),
                })?;
            let offset = f * hop;
            for (i, &sample) in samples.iter().enumerate() {
                acc[offset + i] += sample * self.window[i];
                norm[offset + i] += self.window[i] * self.window[i];
            }
        }
        let start = n_fft / 2;
        let end = total - start;
        let mut out = vec![0.0f32; length];
        for i in 0..length.min(end.saturating_sub(start)) {
            out[i] = acc[start + i] / norm[start + i].max(1e-10);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
