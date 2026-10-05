//! Nemotron 3 Diarization: streaming who-spoke-when probabilities for up
//! to eight speakers at 10 ms resolution.
//!
//! Reference: `mlx_audio/vad/models/nemotron_diarization/`
//! (`nemotron_diarization.py`, `config.py`) at the cloned v0.5.7
//! revision, itself ported from NVIDIA NeMo Speech. The architecture is a
//! 31-layer RoPE Transformer over linear feature-stacking subsampling
//! (8 x 10 ms mel frames per encoder frame), a speaker head with a
//! subpixel upsampling conv, and the shared Sortformer AOSC/FIFO
//! streaming machinery. Unlike Sortformer there is no conv frontend and
//! no per-feature normalization: the preprocessor is preemphasized,
//! center-padded log-mels using the checkpoint's own window and
//! filterbank buffers.
//!
//! The ported entry points mirror the reference surface:
//! - [`NemotronDiarization::forward_window`]: single-window forward
//!   (reference `Model.__call__`).
//! - [`NemotronDiarization::feed`]: incremental PCM chunks with an exact
//!   STFT-boundary buffer (reference `Model.feed`).
//! - [`NemotronDiarization::generate_stream`] and
//!   [`NemotronDiarization::generate`]: file-mode streaming and the
//!   offline wrapper that re-segments the concatenated probabilities
//!   (reference `Model.generate_stream` / `Model.generate`).
//!
//! The stage functions ([`mel_features`](Self::mel_features),
//! [`pre_encode`](Self::pre_encode),
//! [`encoder_forward`](Self::encoder_forward),
//! [`speaker_head`](Self::speaker_head)) are exposed so parity tooling
//! can bisect against the reference stage by stage.
//!
//! The AOSC speaker-cache compression is the same algorithm as
//! Sortformer v2.1 and reuses
//! [`crate::models::vad::sortformer`]'s implementation with this model's
//! parameters; the silence embedding is the static learned checkpoint
//! tensor (the reference passes `learnable_sil_emb` directly, not a
//! running silence profile).
//!
//! The HuggingFace conversion ships bfloat16 weights and is loaded here
//! with each tensor upcast to f32; the reference computes the encoder in
//! the checkpoint dtype, so f32 parity gates run against a locally
//! converted float32 checkpoint (see `tasks.md` task 5).

use std::path::Path;

use serde_json::Value;
use turbospark_audio::fft::real_fft_forward;
use turbospark_audio::mel::{mel_filterbank, MelFilterbank, MelScale};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::vad::sortformer::{
    compress_spkcache_aosc, preds_to_segments, DiarizationOutput, DiarizationSegment,
    GenerateOptions, ModulesConfig as AoscModulesConfig,
};
use crate::ops;
use crate::{Result, SpeechError};

/// The log guard the reference adds before the mel log (`2**-24`).
const LOG_GUARD: f32 = 5.960_464_5e-8;

// =============================================================================
// Configuration
// =============================================================================

/// RoPE Transformer encoder configuration (`encoder_config`).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub feat_in: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub subsampling_factor: usize,
    pub ff_expansion: f32,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub pre_block_norm: bool,
    pub xscaling: bool,
    pub rope_base: f32,
    pub rotary_fraction: f32,
}

/// Speaker head and streaming configuration (`modules_config`).
#[derive(Debug, Clone)]
pub struct SpeakerConfig {
    pub num_speakers: usize,
    pub fc_d_model: usize,
    pub tf_d_model: usize,
    pub subsampling_factor: usize,
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub spkcache_update_period: usize,
    pub chunk_left_context: usize,
    pub chunk_right_context: usize,
    pub spkcache_sil_frames_per_spk: usize,
    pub pred_score_threshold: f32,
    pub max_index: usize,
    pub scores_boost_latest: f32,
    pub sil_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
    pub use_aosc: bool,
    pub use_learnable_sil_emb: bool,
    pub use_activity_head: bool,
}

/// Mel preprocessor configuration (`processor_config`).
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub preemphasis: f32,
    pub pad_to: usize,
}

impl Default for MelConfig {
    fn default() -> Self {
        MelConfig {
            feature_size: 128,
            sampling_rate: 16_000,
            hop_length: 160,
            n_fft: 512,
            win_length: 400,
            preemphasis: 0.97,
            pad_to: 16,
        }
    }
}

/// Full model configuration as shipped in `config.json`.
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub num_speakers: usize,
    pub output_subsampling_factor: usize,
    pub encoder: EncoderConfig,
    pub modules: SpeakerConfig,
    pub processor: MelConfig,
}

fn field_u64(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn field_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn field_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

impl NemotronConfig {
    /// Parses `config.json`; every field falls back to the reference
    /// `config.py` default when absent.
    pub fn from_json(v: &Value) -> Result<Self> {
        let section = |name: &str| -> Result<&Value> {
            v.get(name).ok_or_else(|| SpeechError::BadConfig {
                field: name.to_string(),
                why: "missing".into(),
            })
        };
        let enc = section("encoder_config")?;
        let modules = section("modules_config")?;
        let proc_v = section("processor_config")?;
        let u = |x: &Value, k: &str, d: u64| field_u64(x, k, d) as usize;
        let num_speakers = u(v, "num_speakers", 8);
        Ok(NemotronConfig {
            num_speakers,
            output_subsampling_factor: u(v, "output_subsampling_factor", 1),
            encoder: EncoderConfig {
                feat_in: u(enc, "feat_in", 128),
                d_model: u(enc, "d_model", 512),
                n_layers: u(enc, "n_layers", 31),
                n_heads: u(enc, "n_heads", 8),
                subsampling_factor: u(enc, "subsampling_factor", 8),
                ff_expansion: field_f64(enc, "ff_expansion", 4.0) as f32,
                qkv_bias: field_bool(enc, "qkv_bias", false),
                qk_norm: field_bool(enc, "qk_norm", false),
                pre_block_norm: field_bool(enc, "pre_block_norm", true),
                xscaling: field_bool(enc, "xscaling", false),
                rope_base: field_f64(enc, "rope_base", 10_000.0) as f32,
                rotary_fraction: field_f64(enc, "rotary_fraction", 1.0) as f32,
            },
            modules: SpeakerConfig {
                num_speakers: u(modules, "num_speakers", num_speakers as u64),
                fc_d_model: u(modules, "fc_d_model", 512),
                tf_d_model: u(modules, "tf_d_model", 192),
                subsampling_factor: u(modules, "subsampling_factor", 8),
                chunk_len: u(modules, "chunk_len", 340),
                fifo_len: u(modules, "fifo_len", 40),
                spkcache_len: u(modules, "spkcache_len", 264),
                spkcache_update_period: u(modules, "spkcache_update_period", 300),
                chunk_left_context: u(modules, "chunk_left_context", 0),
                chunk_right_context: u(modules, "chunk_right_context", 40),
                spkcache_sil_frames_per_spk: u(modules, "spkcache_sil_frames_per_spk", 1),
                pred_score_threshold: field_f64(modules, "pred_score_threshold", 0.25) as f32,
                max_index: u(modules, "max_index", 99_999),
                scores_boost_latest: field_f64(modules, "scores_boost_latest", 0.05) as f32,
                sil_threshold: field_f64(modules, "sil_threshold", 0.2) as f32,
                strong_boost_rate: field_f64(modules, "strong_boost_rate", 0.75) as f32,
                weak_boost_rate: field_f64(modules, "weak_boost_rate", 1.5) as f32,
                min_pos_scores_rate: field_f64(modules, "min_pos_scores_rate", 0.5) as f32,
                use_aosc: field_bool(modules, "use_aosc", true),
                use_learnable_sil_emb: field_bool(modules, "use_learnable_sil_emb", true),
                use_activity_head: field_bool(modules, "use_activity_head", true),
            },
            processor: MelConfig {
                feature_size: u(proc_v, "feature_size", 128),
                sampling_rate: u(proc_v, "sampling_rate", 16_000) as u32,
                hop_length: u(proc_v, "hop_length", 160),
                n_fft: u(proc_v, "n_fft", 512),
                win_length: u(proc_v, "win_length", 400),
                preemphasis: field_f64(proc_v, "preemphasis", 0.97) as f32,
                pad_to: u(proc_v, "pad_to", 16),
            },
        })
    }
}

/// NVIDIA input-buffer latency presets (`Model.set_streaming_config`).
///
/// Presets describe input-buffer latency, excluding compute and the STFT
/// window: offline=30.4 s, low=1.04 s, very_low=0.64 s, ultra_low=0.32 s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyPreset {
    Offline,
    Low,
    VeryLow,
    UltraLow,
}

impl LatencyPreset {
    /// `(chunk_len, chunk_right_context, fifo_len, spkcache_update_period)`.
    fn windows(self) -> (usize, usize, usize, usize) {
        match self {
            LatencyPreset::Offline => (340, 40, 40, 300),
            LatencyPreset::Low => (9, 4, 264, 222),
            LatencyPreset::VeryLow => (6, 2, 264, 222),
            LatencyPreset::UltraLow => (3, 1, 264, 222),
        }
    }
}

// =============================================================================
// Weights
// =============================================================================

struct Attention {
    /// Fused qkv projection `[3 * d_model, d_model]` (no bias when
    /// `qkv_bias` is false, which is what the shipped checkpoint uses).
    w_qkv_w: Vec<f32>,
    w_qkv_b: Option<Vec<f32>>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

struct FeedForward {
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
}

struct TransformerLayer {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    attn: Attention,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    ffn: FeedForward,
}

struct SpeakerModules {
    encoder_proj_w: Vec<f32>,
    encoder_proj_b: Vec<f32>,
    hidden_to_hidden_w: Vec<f32>,
    hidden_to_hidden_b: Vec<f32>,
    single_to_spks_w: Vec<f32>,
    single_to_spks_b: Vec<f32>,
    /// Subpixel upsampling conv, PyTorch layout `[out, in, kernel]`
    /// (permuted on load from the checkpoint's MLX `[out, kernel, in]`).
    subpixel_w: Vec<f32>,
    subpixel_b: Vec<f32>,
    /// Static learned silence embedding fed to the AOSC compression.
    learnable_sil_emb: Vec<f32>,
    /// Loaded because the checkpoint ships them, but the reference only
    /// uses them in the training-time joint forward.
    #[allow(dead_code)]
    hidden_to_spks_w: Vec<f32>,
    #[allow(dead_code)]
    hidden_to_spks_b: Vec<f32>,
    #[allow(dead_code)]
    activity_head: Option<ActivityHead>,
}

/// The two-layer activity head (LayerNorm then Linear) the checkpoint
/// ships for training-time scoring; unused at inference.
type ActivityHead = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// A loaded Nemotron 3 Diarization model.
pub struct NemotronDiarization {
    config: NemotronConfig,
    /// Feature-stacking projection `[d_model, feat_in * factor]`.
    pre_encode_proj_w: Vec<f32>,
    embed_norm: Option<(Vec<f32>, Vec<f32>)>,
    layers: Vec<TransformerLayer>,
    final_norm_w: Vec<f32>,
    final_norm_b: Vec<f32>,
    modules: SpeakerModules,
    /// The preprocessor window zero-padded to `n_fft` (checkpoint buffer
    /// when shipped, otherwise the computed symmetric Hann).
    window: Vec<f32>,
    /// Mel filterbank `[n_mels, n_fft / 2 + 1]` (checkpoint buffer when
    /// shipped, otherwise the computed Slaney filterbank).
    fb: Vec<f32>,
    /// The AOSC parameters in the shape the shared sortformer
    /// compression consumes.
    aosc_cfg: AoscModulesConfig,
}

fn load_vec(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>> {
    file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
        name: name.to_string(),
        why: format!("load failed: {e}"),
    })
}

impl NemotronDiarization {
    /// Opens a model directory holding `config.json` and
    /// `model.safetensors` (the HuggingFace bfloat16 conversion or a
    /// locally converted float32 checkpoint).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| {
                SpeechError::BadConfig {
                    field: "config.json".into(),
                    why: format!("unreadable: {e}"),
                }
            })?,
        )
        .map_err(|e| SpeechError::BadConfig {
            field: "config.json".into(),
            why: format!("invalid JSON: {e}"),
        })?;
        let config = NemotronConfig::from_json(&config)?;
        let file = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;

        let enc = &config.encoder;
        let d = enc.d_model;
        if d % enc.n_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "d_model".into(),
                why: "must be divisible by n_heads".into(),
            });
        }
        if enc.feat_in != config.processor.feature_size {
            return Err(SpeechError::BadConfig {
                field: "feat_in".into(),
                why: "encoder and preprocessor feature dimensions must match".into(),
            });
        }
        if d != config.modules.fc_d_model || config.num_speakers != config.modules.num_speakers {
            return Err(SpeechError::BadConfig {
                field: "num_speakers".into(),
                why: "encoder and speaker head dimensions must match".into(),
            });
        }
        let pre_encode_proj_w = load_vec(&file, "encoder.pre_encode.proj.weight")?;
        let embed_norm = if enc.pre_block_norm {
            Some((
                load_vec(&file, "encoder.embed_norm.weight")?,
                load_vec(&file, "encoder.embed_norm.bias")?,
            ))
        } else {
            None
        };
        let layers = (0..enc.n_layers)
            .map(|i| {
                let p = format!("encoder.layers.{i}");
                Ok(TransformerLayer {
                    norm1_w: load_vec(&file, &format!("{p}.norm1.weight"))?,
                    norm1_b: load_vec(&file, &format!("{p}.norm1.bias"))?,
                    attn: Attention {
                        w_qkv_w: load_vec(&file, &format!("{p}.attn.w_qkv.weight"))?,
                        w_qkv_b: if enc.qkv_bias {
                            Some(load_vec(&file, &format!("{p}.attn.w_qkv.bias"))?)
                        } else {
                            None
                        },
                        out_w: load_vec(&file, &format!("{p}.attn.out_proj.weight"))?,
                        out_b: load_vec(&file, &format!("{p}.attn.out_proj.bias"))?,
                    },
                    norm2_w: load_vec(&file, &format!("{p}.norm2.weight"))?,
                    norm2_b: load_vec(&file, &format!("{p}.norm2.bias"))?,
                    ffn: FeedForward {
                        l1_w: load_vec(&file, &format!("{p}.ffn.linear1.weight"))?,
                        l1_b: load_vec(&file, &format!("{p}.ffn.linear1.bias"))?,
                        l2_w: load_vec(&file, &format!("{p}.ffn.linear2.weight"))?,
                        l2_b: load_vec(&file, &format!("{p}.ffn.linear2.bias"))?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let final_norm_w = load_vec(&file, "encoder.final_norm.weight")?;
        let final_norm_b = load_vec(&file, "encoder.final_norm.bias")?;

        let m = &config.modules;
        let tf = m.tf_d_model;
        let activity_head =
            if file.contains_tensor("sortformer_modules.activity_head.layers.0.weight") {
                Some((
                    load_vec(&file, "sortformer_modules.activity_head.layers.0.weight")?,
                    load_vec(&file, "sortformer_modules.activity_head.layers.0.bias")?,
                    load_vec(&file, "sortformer_modules.activity_head.layers.1.weight")?,
                    load_vec(&file, "sortformer_modules.activity_head.layers.1.bias")?,
                ))
            } else {
                None
            };
        let modules = SpeakerModules {
            encoder_proj_w: load_vec(&file, "sortformer_modules.encoder_proj.weight")?,
            encoder_proj_b: load_vec(&file, "sortformer_modules.encoder_proj.bias")?,
            hidden_to_hidden_w: load_vec(
                &file,
                "sortformer_modules.first_hidden_to_hidden.weight",
            )?,
            hidden_to_hidden_b: load_vec(&file, "sortformer_modules.first_hidden_to_hidden.bias")?,
            single_to_spks_w: load_vec(&file, "sortformer_modules.single_hidden_to_spks.weight")?,
            single_to_spks_b: load_vec(&file, "sortformer_modules.single_hidden_to_spks.bias")?,
            subpixel_w: load_subpixel_weight(&file, tf * m.subsampling_factor, tf)?,
            subpixel_b: load_vec(&file, "sortformer_modules.subpixel_upsample.bias")?,
            learnable_sil_emb: load_vec(&file, "sortformer_modules.learnable_sil_emb")?,
            hidden_to_spks_w: load_vec(&file, "sortformer_modules.hidden_to_spks.weight")?,
            hidden_to_spks_b: load_vec(&file, "sortformer_modules.hidden_to_spks.bias")?,
            activity_head,
        };

        // The checkpoint ships the exact window and filterbank buffers;
        // fall back to computing them for checkpoints without.
        let proc = &config.processor;
        let window = if file.contains_tensor("preprocessor.window") {
            pad_window(&load_vec(&file, "preprocessor.window")?, proc.n_fft)
        } else {
            pad_window(&symmetric_hann(proc.win_length), proc.n_fft)
        };
        let fb = if file.contains_tensor("preprocessor.fb") {
            load_vec(&file, "preprocessor.fb")?
        } else {
            let MelFilterbank { weights, .. } = mel_filterbank(
                proc.feature_size,
                proc.n_fft,
                proc.sampling_rate,
                0.0,
                None,
                MelScale::Slaney,
            )
            .map_err(|e| SpeechError::Audio(format!("mel filterbank: {e}")))?;
            weights
        };

        let aosc_cfg = AoscModulesConfig {
            num_speakers: m.num_speakers,
            fc_d_model: m.fc_d_model,
            tf_d_model: m.tf_d_model,
            subsampling_factor: m.subsampling_factor,
            chunk_len: m.chunk_len,
            fifo_len: m.fifo_len,
            spkcache_len: m.spkcache_len,
            spkcache_update_period: m.spkcache_update_period,
            chunk_left_context: m.chunk_left_context,
            chunk_right_context: m.chunk_right_context,
            spkcache_sil_frames_per_spk: m.spkcache_sil_frames_per_spk,
            pred_score_threshold: m.pred_score_threshold,
            max_index: m.max_index,
            scores_boost_latest: m.scores_boost_latest,
            sil_threshold: m.sil_threshold,
            strong_boost_rate: m.strong_boost_rate,
            weak_boost_rate: m.weak_boost_rate,
            min_pos_scores_rate: m.min_pos_scores_rate,
            use_aosc: m.use_aosc,
        };

        Ok(NemotronDiarization {
            config,
            pre_encode_proj_w,
            embed_norm,
            layers,
            final_norm_w,
            final_norm_b,
            modules,
            window,
            fb,
            aosc_cfg,
        })
    }

    pub fn config(&self) -> &NemotronConfig {
        &self.config
    }

    /// Selects an NVIDIA latency preset before starting a new recording
    /// (the reference `set_streaming_config`).
    pub fn set_latency_preset(&mut self, preset: LatencyPreset) {
        let (chunk, right, fifo, period) = preset.windows();
        let m = &mut self.config.modules;
        m.chunk_len = chunk;
        m.chunk_right_context = right;
        m.fifo_len = fifo;
        m.spkcache_update_period = period;
        self.aosc_cfg.chunk_len = chunk;
        self.aosc_cfg.chunk_right_context = right;
        self.aosc_cfg.fifo_len = fifo;
        self.aosc_cfg.spkcache_update_period = period;
    }
}

/// Permutes the checkpoint's MLX conv weight `[out, kernel, in]` into the
/// PyTorch `[out, in, kernel]` layout `ops::conv1d` consumes.
fn load_subpixel_weight(file: &SafetensorsFile, out: usize, in_ch: usize) -> Result<Vec<f32>> {
    let name = "sortformer_modules.subpixel_upsample.weight";
    let raw = load_vec(file, name)?;
    let k = raw.len() / (out * in_ch);
    if raw.len() != out * in_ch * k {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("unexpected size {} for {out}x{in_ch}x{k}", raw.len()),
        });
    }
    let mut w = vec![0.0f32; raw.len()];
    for o in 0..out {
        for i in 0..in_ch {
            for kk in 0..k {
                w[o * in_ch * k + i * k + kk] = raw[o * k * in_ch + kk * in_ch + i];
            }
        }
    }
    Ok(w)
}

/// Symmetric Hann window (`hanning(size, periodic=False)`).
fn symmetric_hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            let denom = (size - 1) as f64;
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / denom).cos())) as f32
        })
        .collect()
}

/// Zero-pads the `win_length` window to `n_fft`, split half left / half
/// right of the remainder.
fn pad_window(window: &[f32], n_fft: usize) -> Vec<f32> {
    if window.len() >= n_fft {
        return window[..n_fft].to_vec();
    }
    let pad = n_fft - window.len();
    let left = pad / 2;
    let mut out = vec![0.0f32; left];
    out.extend_from_slice(window);
    out.extend(std::iter::repeat_n(0.0f32, pad - left));
    out
}

// =============================================================================
// Stage-by-stage forward (exposed for parity tooling)
// =============================================================================

impl NemotronDiarization {
    /// Reference `MelFeatures.__call__`: global log-mel frames
    /// `[start, start + count)` computed from a PCM buffer that starts at
    /// `sample_offset`, where `total_samples` bounds the valid stream.
    /// Returns channel-major `[n_mels, count]`.
    pub fn mel_features(
        &self,
        buffer: &[f32],
        start: usize,
        count: usize,
        sample_offset: usize,
        total_samples: usize,
    ) -> Result<Vec<f32>> {
        let proc = &self.config.processor;
        if count == 0 {
            return Ok(Vec::new());
        }
        let valid_frames = total_samples / proc.hop_length;
        let mut frames = vec![0.0f32; count * proc.n_fft];
        if !buffer.is_empty() {
            let total = total_samples as i64;
            let offset = sample_offset as i64;
            let buf_len = buffer.len() as i64;
            for (fi, frame_id) in (start..start + count).enumerate() {
                let row = &mut frames[fi * proc.n_fft..(fi + 1) * proc.n_fft];
                for (k, slot) in row.iter_mut().enumerate() {
                    let pos = frame_id as i64 * proc.hop_length as i64 + k as i64
                        - (proc.n_fft / 2) as i64;
                    let gather = |idx: i64| -> f32 {
                        if idx < 0 || idx >= total {
                            0.0
                        } else {
                            let local = (idx - offset).clamp(0, buf_len - 1);
                            buffer[local as usize]
                        }
                    };
                    if pos >= 0 && pos < total {
                        *slot = gather(pos) - proc.preemphasis * gather(pos - 1);
                    }
                }
            }
        }

        let bins = proc.n_fft / 2 + 1;
        let mut feats = vec![0.0f32; proc.feature_size * count];
        for t in 0..count {
            let row = &frames[t * proc.n_fft..(t + 1) * proc.n_fft];
            let windowed: Vec<f32> = row.iter().zip(&self.window).map(|(v, w)| v * w).collect();
            let spec =
                real_fft_forward(&windowed).map_err(|e| SpeechError::Audio(format!("fft: {e}")))?;
            for m in 0..proc.feature_size {
                let fb_row = &self.fb[m * bins..(m + 1) * bins];
                let mut acc = 0.0f32;
                for (b, c) in spec.iter().enumerate() {
                    acc += (c.re * c.re + c.im * c.im) * fb_row[b];
                }
                feats[m * count + t] = (acc + LOG_GUARD).ln();
            }
        }
        // Whole-frame zeroing past the valid stream.
        for (fi, frame_id) in (start..start + count).enumerate() {
            if frame_id >= valid_frames {
                for m in 0..proc.feature_size {
                    feats[m * count + fi] = 0.0;
                }
            }
        }
        Ok(feats)
    }

    /// Reference `FeatureStacking.__call__`: stacks `factor` consecutive
    /// mel frames and projects. Returns the stacked embeddings
    /// `[ceil(mel_frames / factor), d_model]` and the encoder mask length
    /// `ceil(mel_length / factor)`; the mask length may be shorter than
    /// the padded frame count on the final flush.
    pub fn pre_encode(
        &self,
        features: &[f32],
        mel_frames: usize,
        mel_length: usize,
    ) -> (Vec<f32>, usize) {
        let enc = &self.config.encoder;
        let factor = enc.subsampling_factor;
        let n_mels = enc.feat_in;
        let rows = mel_frames.div_ceil(factor);
        let mut stacked = vec![0.0f32; rows * n_mels * factor];
        for t in 0..mel_frames {
            let base = (t / factor) * n_mels * factor + (t % factor) * n_mels;
            for m in 0..n_mels {
                stacked[base + m] = features[m * mel_frames + t];
            }
        }
        let emb = ops::linear(
            &stacked,
            &self.pre_encode_proj_w,
            None,
            rows,
            n_mels * factor,
            enc.d_model,
        );
        (emb, mel_length.div_ceil(factor))
    }

    /// Reference `Attention.__call__` for one layer: fused qkv, per-head
    /// NeoX rope, masked SDPA, output projection.
    fn attention_forward(
        &self,
        attn: &Attention,
        x: &[f32],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
        key_mask: &[f32],
    ) -> Vec<f32> {
        let enc = &self.config.encoder;
        let d = enc.d_model;
        let heads = enc.n_heads;
        let head_dim = d / heads;
        let rotary_dim = self.rotary_dim();
        let qkv = ops::linear(x, &attn.w_qkv_w, attn.w_qkv_b.as_deref(), rows, d, 3 * d);
        let scale = (head_dim as f32).powf(-0.5);
        let mut gathered = vec![0.0f32; rows * d];
        let mut q = vec![0.0f32; rows * head_dim];
        let mut k = vec![0.0f32; rows * head_dim];
        let mut v = vec![0.0f32; rows * head_dim];
        for h in 0..heads {
            for t in 0..rows {
                for dd in 0..head_dim {
                    q[t * head_dim + dd] = qkv[t * 3 * d + h * head_dim + dd];
                    k[t * head_dim + dd] = qkv[t * 3 * d + d + h * head_dim + dd];
                    v[t * head_dim + dd] = qkv[t * 3 * d + 2 * d + h * head_dim + dd];
                }
            }
            rope_neox(&mut q, rows, head_dim, rotary_dim, cos, sin);
            rope_neox(&mut k, rows, head_dim, rotary_dim, cos, sin);
            // ops::sdpa consumes one head's rows; the key mask repeats on
            // every query row: mask[t * rows + j] = key_mask[j].
            let head_mask: Vec<f32> = (0..rows).flat_map(|_| key_mask.iter().copied()).collect();
            let o = ops::sdpa(
                &q,
                &k,
                &v,
                Some(&head_mask),
                rows,
                rows,
                head_dim,
                head_dim,
                scale,
            );
            for t in 0..rows {
                gathered[t * d + h * head_dim..t * d + (h + 1) * head_dim]
                    .copy_from_slice(&o[t * head_dim..(t + 1) * head_dim]);
            }
        }
        ops::linear(&gathered, &attn.out_w, Some(&attn.out_b), rows, d, d)
    }

    /// The rotated head-dimension span: `int(head_dim * rotary_fraction)`
    /// rounded down to an even count (the reference `config.__post_init__`
    /// guarantees at least 2).
    fn rotary_dim(&self) -> usize {
        let enc = &self.config.encoder;
        let head_dim = enc.d_model / enc.n_heads;
        let rotary = (head_dim as f32 * enc.rotary_fraction).floor() as usize;
        rotary.min(head_dim) & !1
    }

    /// Reference `Encoder.__call__`: input scaling, embed norm, RoPE
    /// Transformer layers with a key-validity mask, final norm. `x` is
    /// `[rows, d_model]` t-major and updated in place.
    pub fn encoder_forward(&self, x: &mut [f32], rows: usize, lengths: usize) {
        let enc = &self.config.encoder;
        let d = enc.d_model;
        if enc.xscaling {
            let scale = (d as f64).sqrt() as f32;
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
        if let Some((w, b)) = &self.embed_norm {
            ops::layernorm(x, rows, d, w, Some(b), 1e-5);
        }
        let rotary_dim = self.rotary_dim();
        let (cos, sin) = ops::rope_tables(rows, rotary_dim, enc.rope_base);
        let key_mask: Vec<f32> = (0..rows)
            .map(|t| if t < lengths { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let hidden = (d as f32 * enc.ff_expansion) as usize;
        for layer in &self.layers {
            let mut h = x.to_vec();
            ops::layernorm(&mut h, rows, d, &layer.norm1_w, Some(&layer.norm1_b), 1e-5);
            let attn = self.attention_forward(&layer.attn, &h, rows, &cos, &sin, &key_mask);
            for (r, a) in x.iter_mut().zip(&attn) {
                *r += a;
            }
            let mut h2 = x.to_vec();
            ops::layernorm(&mut h2, rows, d, &layer.norm2_w, Some(&layer.norm2_b), 1e-5);
            let mut ff = ops::linear(&h2, &layer.ffn.l1_w, Some(&layer.ffn.l1_b), rows, d, hidden);
            ops::gelu_erf(&mut ff);
            let ff = ops::linear(&ff, &layer.ffn.l2_w, Some(&layer.ffn.l2_b), rows, hidden, d);
            for (r, a) in x.iter_mut().zip(&ff) {
                *r += a;
            }
        }
        ops::layernorm(
            x,
            rows,
            d,
            &self.final_norm_w,
            Some(&self.final_norm_b),
            1e-5,
        );
    }

    /// Reference `SpeakerModules.__call__`: encoder projection, subpixel
    /// upsampling conv, ReLU/ReLU/sigmoid head. Returns
    /// `[rows * factor, num_speakers]` probabilities.
    pub fn speaker_head(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let m = &self.config.modules;
        let tf = m.tf_d_model;
        let factor = m.subsampling_factor;
        let n_spk = self.config.num_speakers;
        let h = ops::linear(
            x,
            &self.modules.encoder_proj_w,
            Some(&self.modules.encoder_proj_b),
            rows,
            m.fc_d_model,
            tf,
        );
        // conv over time: channel-major [tf, rows] -> [tf * factor, rows]
        let mut cm = vec![0.0f32; tf * rows];
        for t in 0..rows {
            for c in 0..tf {
                cm[c * rows + t] = h[t * tf + c];
            }
        }
        let conv = ops::conv1d(
            &cm,
            &self.modules.subpixel_w,
            Some(&self.modules.subpixel_b),
            tf,
            tf * factor,
            3,
            1,
            1,
            1,
            1,
        );
        // Reshape to subpixel frames: output frame t * factor + c takes
        // conv output channels [c * tf, (c + 1) * tf) at time t.
        let mut up = vec![0.0f32; rows * factor * tf];
        for t in 0..rows {
            for c in 0..factor {
                let dst = (t * factor + c) * tf;
                for dd in 0..tf {
                    up[dst + dd] = conv[(c * tf + dd) * rows + t];
                }
            }
        }
        for v in up.iter_mut() {
            *v = v.max(0.0);
        }
        let mut hid = ops::linear(
            &up,
            &self.modules.hidden_to_hidden_w,
            Some(&self.modules.hidden_to_hidden_b),
            rows * factor,
            tf,
            tf,
        );
        for v in hid.iter_mut() {
            *v = v.max(0.0);
        }
        let mut logits = ops::linear(
            &hid,
            &self.modules.single_to_spks_w,
            Some(&self.modules.single_to_spks_b),
            rows * factor,
            tf,
            n_spk,
        );
        for v in logits.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        logits
    }

    /// Reference `Model.__call__`: single-window forward from
    /// channel-major mel features. `mel_length` may be shorter than
    /// `mel_frames`; outputs at or past it are zeroed. Returns
    /// `[ceil(mel_frames / factor) * factor, num_speakers]`.
    pub fn forward_window(
        &self,
        features: &[f32],
        mel_frames: usize,
        mel_length: usize,
    ) -> Vec<f32> {
        let (emb, lengths) = self.pre_encode(features, mel_frames, mel_length);
        let rows = emb.len() / self.config.encoder.d_model;
        let mut x = emb;
        self.encoder_forward(&mut x, rows, lengths);
        let mut probs = self.speaker_head(&x, rows);
        for (t, slot) in probs.chunks_exact_mut(self.config.num_speakers).enumerate() {
            if t >= mel_length {
                for s in slot.iter_mut() {
                    *s = 0.0;
                }
            }
        }
        probs
    }
}

/// NeoX-style (half-split) rotary embedding, the MLX
/// `nn.RoPE(traditional=False)` convention, applied in place to
/// `x [seq, dim]` rows. `cos`/`sin` are `[seq, rotary_dim / 2]` tables.
fn rope_neox(x: &mut [f32], seq: usize, dim: usize, rotary_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = rotary_dim / 2;
    for t in 0..seq {
        let base = t * dim;
        for d in 0..half {
            let c = cos[t * half + d];
            let s = sin[t * half + d];
            let a = x[base + d];
            let b = x[base + half + d];
            x[base + d] = a * c - b * s;
            x[base + half + d] = b * c + a * s;
        }
    }
}

// =============================================================================
// Streaming inference
// =============================================================================

/// State carried between streaming chunks: the long-term speaker cache
/// and the recent-context FIFO (both at encoder resolution, 80 ms per
/// frame), the PCM history needed for exact STFT boundaries, and the
/// bookkeeping counters. Mirrors the reference `StreamingState`.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// `[spkcache_frames, emb_dim]` pre-encoded embeddings.
    pub spkcache: Vec<f32>,
    /// `[spkcache_frames, num_speakers]`.
    pub spkcache_preds: Vec<f32>,
    /// `[fifo_frames, emb_dim]`.
    pub fifo: Vec<f32>,
    /// `[fifo_frames, num_speakers]`.
    pub fifo_preds: Vec<f32>,
    /// True once the cache has been AOSC-compressed (afterwards the
    /// cache predictions stop being refreshed, matching the reference).
    pub spkcache_compressed: bool,
    /// Native 10 ms frames processed, before output downsampling.
    pub frames_processed: usize,
    pub samples_received: usize,
    pub sample_offset: usize,
    pub audio_buffer: Vec<f32>,
    pub finished: bool,
}

impl NemotronDiarization {
    fn emb_dim(&self) -> usize {
        self.config.encoder.d_model
    }

    /// An empty streaming state (`init_streaming_state`).
    pub fn init_streaming_state(&self) -> StreamingState {
        StreamingState {
            spkcache: Vec::new(),
            spkcache_preds: Vec::new(),
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            spkcache_compressed: false,
            frames_processed: 0,
            samples_received: 0,
            sample_offset: 0,
            audio_buffer: Vec::new(),
            finished: false,
        }
    }

    /// Reference `streaming_step`: encodes `[spkcache | fifo | chunk]`,
    /// returns this step's central mel-frame predictions
    /// `[central_frames, num_speakers]`, and folds the chunk into the
    /// FIFO with cache overflow popped into the AOSC-compressed cache.
    /// `features` is channel-major `[n_mels, window_frames]`;
    /// `feature_length` is the valid mel length used for the encoder
    /// mask (shorter than `window_frames` on the final flush).
    fn streaming_step(
        &self,
        features: &[f32],
        window_frames: usize,
        feature_length: usize,
        state: &mut StreamingState,
        central_frames: usize,
    ) -> Vec<f32> {
        let m = self.config.modules.clone();
        let factor = m.subsampling_factor;
        let d = self.emb_dim();
        let n_spk = self.config.num_speakers;

        let (chunk, lengths) = self.pre_encode(features, window_frames, feature_length);
        let chunk_rows = chunk.len() / d;
        let cache_len = state.spkcache.len() / d;
        let fifo_len = state.fifo.len() / d;

        let mut combined = Vec::with_capacity((cache_len + fifo_len + chunk_rows) * d);
        combined.extend_from_slice(&state.spkcache);
        combined.extend_from_slice(&state.fifo);
        combined.extend_from_slice(&chunk);
        self.encoder_forward(
            &mut combined,
            cache_len + fifo_len + chunk_rows,
            lengths + cache_len + fifo_len,
        );
        let mut high = self.speaker_head(&combined, cache_len + fifo_len + chunk_rows);

        // Zero the subpixel padding frames past the valid span.
        let valid_mel = (lengths + cache_len + fifo_len) * factor;
        for (t, slot) in high.chunks_exact_mut(n_spk).enumerate() {
            if t >= valid_mel {
                for s in slot.iter_mut() {
                    *s = 0.0;
                }
            }
        }

        // Cache scoring always operates at encoder resolution, using
        // mean probabilities over the subsampled group.
        let enc_total = high.len() / n_spk / factor;
        let mut low = vec![0.0f32; enc_total * n_spk];
        for e in 0..enc_total {
            for k in 0..n_spk {
                let mut acc = 0.0f32;
                for f in 0..factor {
                    acc += high[(e * factor + f) * n_spk + k];
                }
                low[e * n_spk + k] = acc / factor as f32;
            }
        }

        let n_enc = central_frames.div_ceil(factor);
        let start = cache_len + fifo_len;
        let result =
            high[start * factor * n_spk..(start * factor + central_frames) * n_spk].to_vec();

        state.fifo.extend_from_slice(&chunk[..n_enc * d]);
        let mut fifo_preds = Vec::with_capacity((fifo_len + n_enc) * n_spk);
        fifo_preds.extend_from_slice(&low[cache_len * n_spk..start * n_spk]);
        fifo_preds.extend_from_slice(&low[start * n_spk..(start + n_enc) * n_spk]);
        state.fifo_preds = fifo_preds;

        if state.fifo.len() / d > m.fifo_len {
            let fifo_total = state.fifo.len() / d;
            let pop = fifo_total.min(m.spkcache_update_period.max(fifo_total - m.fifo_len));
            let mut cache = std::mem::take(&mut state.spkcache);
            cache.extend_from_slice(&state.fifo[..pop * d]);
            let mut cache_preds = if state.spkcache_compressed {
                std::mem::take(&mut state.spkcache_preds)
            } else {
                low[..cache_len * n_spk].to_vec()
            };
            cache_preds.extend_from_slice(&state.fifo_preds[..pop * n_spk]);
            state.fifo.drain(..pop * d);
            state.fifo_preds.drain(..pop * n_spk);
            state.spkcache = cache;
            state.spkcache_preds = cache_preds;
            if state.spkcache.len() / d > m.spkcache_len {
                let frames = state.spkcache.len() / d;
                let (c, p) = compress_spkcache_aosc(
                    &state.spkcache,
                    &state.spkcache_preds,
                    frames,
                    &self.modules.learnable_sil_emb,
                    &self.aosc_cfg,
                );
                state.spkcache = c;
                state.spkcache_preds = p;
                state.spkcache_compressed = true;
            }
        }
        state.frames_processed += central_frames;
        result
    }

    /// Feeds one chunk of raw mono PCM at the model sample rate (the
    /// reference `feed`). Call once with `final = true` to flush the
    /// lookahead. Empty output means more samples are needed for the
    /// configured chunk, right context and STFT window.
    pub fn feed(
        &self,
        chunk: &[f32],
        mut state: StreamingState,
        final_chunk: bool,
        opts: &GenerateOptions,
    ) -> Result<(DiarizationOutput, StreamingState)> {
        if state.finished {
            return Err(SpeechError::Input {
                why: "this stream is finished; initialize a new state".into(),
            });
        }
        let cfg = self.config.clone();
        let proc = &cfg.processor;
        let factor = cfg.encoder.subsampling_factor;
        let m = &cfg.modules;
        let central = m.chunk_len * factor;
        let right = m.chunk_right_context * factor;
        state.audio_buffer.extend_from_slice(chunk);
        state.samples_received += chunk.len();
        let offset =
            state.frames_processed as f64 * proc.hop_length as f64 / proc.sampling_rate as f64;
        let stride = proc.hop_length as f64 / proc.sampling_rate as f64;
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        loop {
            let available = state.samples_received / proc.hop_length - state.frames_processed;
            let needed = (state.frames_processed + central + right - 1) as u64
                * proc.hop_length as u64
                + (proc.n_fft / 2) as u64;
            if available == 0 || (!final_chunk && (state.samples_received as u64) < needed) {
                break;
            }
            let n = central.min(available);
            let window_frames = if final_chunk {
                // NeMo masks the extra centered STFT frame, then pads to
                // pad_to.
                let mut total_frames = state.samples_received / proc.hop_length + 1;
                if proc.pad_to > 0 {
                    let rem = total_frames % proc.pad_to;
                    if rem > 0 {
                        total_frames += proc.pad_to - rem;
                    }
                }
                (central + right).min(total_frames - state.frames_processed)
            } else {
                central + right
            };
            let features = self.mel_features(
                &state.audio_buffer,
                state.frames_processed,
                window_frames,
                state.sample_offset,
                state.samples_received,
            )?;
            let feature_length = window_frames.min(available);
            let result =
                self.streaming_step(&features, window_frames, feature_length, &mut state, n);
            outputs.push(result);
            let keep_from =
                (state.frames_processed * proc.hop_length).saturating_sub(proc.n_fft / 2 + 1);
            let drop = keep_from - state.sample_offset;
            state.audio_buffer.drain(..drop);
            state.sample_offset = keep_from;
        }
        state.finished = final_chunk;
        if final_chunk {
            state.audio_buffer.clear();
        }
        let probs: Vec<f32> = outputs.concat();
        let segments = self.output_segments(
            &probs,
            offset,
            opts,
            Some(state.frames_processed as f64 * stride),
        );
        let num_speakers = segments
            .iter()
            .map(|s| s.speaker)
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok((
            DiarizationOutput {
                segments,
                speaker_probs: probs,
                num_speakers,
            },
            state,
        ))
    }

    /// Reference `_output`: optional output downsampling, segment
    /// conversion, offset shift and end clamping.
    fn output_segments(
        &self,
        probs: &[f32],
        offset: f64,
        opts: &GenerateOptions,
        clamp_end: Option<f64>,
    ) -> Vec<DiarizationSegment> {
        let cfg = &self.config;
        let proc = &cfg.processor;
        let factor = cfg.output_subsampling_factor;
        let n_spk = cfg.num_speakers;
        let mut probs = probs.to_vec();
        if factor > 1 && !probs.is_empty() {
            let length = probs.len() / n_spk;
            let padded = length.div_ceil(factor) * factor;
            probs.resize(padded * n_spk, 0.0);
            let mut down = Vec::with_capacity((padded / factor) * n_spk);
            for g in 0..padded / factor {
                let count = factor.min(length - g * factor) as f32;
                for k in 0..n_spk {
                    let mut acc = 0.0f32;
                    for f in 0..factor {
                        acc += probs[(g * factor + f) * n_spk + k];
                    }
                    down.push(acc / count);
                }
            }
            probs = down;
        }
        let stride = proc.hop_length as f64 / proc.sampling_rate as f64;
        let mut segments = preds_to_segments(
            &probs,
            n_spk,
            stride * factor as f64,
            opts.threshold,
            opts.min_duration,
            opts.merge_gap,
        );
        for seg in &mut segments {
            seg.start += offset;
            seg.end += offset;
            if let Some(end) = clamp_end {
                seg.end = seg.end.min(end);
            }
        }
        segments
    }

    /// File-mode streaming (the reference `generate_stream` over a full
    /// waveform): feeds one configured window at a time. Returns every
    /// non-empty chunk result in order plus the final state.
    pub fn generate_stream(
        &self,
        samples: &[f32],
        opts: &GenerateOptions,
    ) -> Result<(Vec<DiarizationOutput>, StreamingState)> {
        let mut state = self.init_streaming_state();
        let step = self.config.modules.chunk_len
            * self.config.encoder.subsampling_factor
            * self.config.processor.hop_length;
        let mut results = Vec::new();
        let mut i = 0;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let (r, s) = self.feed(&samples[i..end], state, false, opts)?;
            state = s;
            if !r.speaker_probs.is_empty() {
                results.push(r);
            }
            i = end;
        }
        let (r, s) = self.feed(&[], state, true, opts)?;
        state = s;
        if !r.speaker_probs.is_empty() {
            results.push(r);
        }
        Ok((results, state))
    }

    /// Offline diarization with bounded AOSC context, including long
    /// recordings (the reference `generate`): streaming concat, then one
    /// segmentation pass over the concatenated probabilities with ends
    /// clamped to the processed span.
    pub fn generate(&self, samples: &[f32], opts: &GenerateOptions) -> Result<DiarizationOutput> {
        let (results, state) = self.generate_stream(samples, opts)?;
        let probs: Vec<f32> = results
            .iter()
            .flat_map(|r| r.speaker_probs.iter().copied())
            .collect();
        let stride = self.config.processor.hop_length as f64
            * self.config.output_subsampling_factor as f64
            / self.config.processor.sampling_rate as f64;
        let mut segments = preds_to_segments(
            &probs,
            self.config.num_speakers,
            stride,
            opts.threshold,
            opts.min_duration,
            opts.merge_gap,
        );
        let end = state.frames_processed as f64 * self.config.processor.hop_length as f64
            / self.config.processor.sampling_rate as f64;
        for seg in &mut segments {
            seg.end = seg.end.min(end);
        }
        let num_speakers = segments
            .iter()
            .map(|s| s.speaker)
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok(DiarizationOutput {
            segments,
            speaker_probs: probs,
            num_speakers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect(path);
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        let shape: Vec<usize> = {
            let start = header.find('(').unwrap() + 1;
            let end = header[start..].find(')').unwrap() + start;
            header[start..end]
                .split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .collect()
        };
        let data_start = 10 + header_len;
        let data = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (shape, data)
    }

    fn testdata(name: &str) -> String {
        format!(
            "{}/testdata/nemotron_diarization/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn default_config() -> NemotronConfig {
        NemotronConfig::from_json(&serde_json::json!({
            "num_speakers": 8,
            "encoder_config": {},
            "modules_config": {},
            "processor_config": {},
        }))
        .unwrap()
    }

    /// Symmetric Hann endpoint values and the win-to-nfft zero padding
    /// split (half left, remainder right).
    #[test]
    fn hann_window_and_padding_known_answer() {
        let w = symmetric_hann(5);
        assert_eq!(w.len(), 5);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[4], 0.0);
        assert!((w[2] - 1.0).abs() < 1e-6);
        assert!((w[1] - 0.5).abs() < 1e-6);
        assert!((w[1] - w[3]).abs() < 1e-6);
        let padded = pad_window(&w, 8);
        assert_eq!(padded.len(), 8);
        assert_eq!(padded[..1], [0.0]);
        assert_eq!(&padded[1..6], &w[..]);
        assert_eq!(padded[6..], [0.0, 0.0]);
    }

    /// Feature stacking order: encoder frame g takes mel frames
    /// [g*factor, (g+1)*factor) concatenated mel-major within each
    /// frame, matching the reference transpose + reshape.
    #[test]
    fn feature_stacking_order_known_answer() {
        let mut config = default_config();
        config.encoder.feat_in = 2;
        config.encoder.d_model = 4;
        config.encoder.subsampling_factor = 2;
        config.modules.subsampling_factor = 2;
        // Identity projection: d_model = feat_in * factor.
        let model = NemotronDiarization {
            aosc_cfg: AoscModulesConfig {
                num_speakers: 8,
                fc_d_model: 4,
                tf_d_model: 4,
                subsampling_factor: 2,
                chunk_len: 1,
                fifo_len: 0,
                spkcache_len: 1,
                spkcache_update_period: 1,
                chunk_left_context: 0,
                chunk_right_context: 0,
                spkcache_sil_frames_per_spk: 1,
                pred_score_threshold: 0.25,
                max_index: 99_999,
                scores_boost_latest: 0.05,
                sil_threshold: 0.2,
                strong_boost_rate: 0.75,
                weak_boost_rate: 1.5,
                min_pos_scores_rate: 0.5,
                use_aosc: true,
            },
            config: config.clone(),
            pre_encode_proj_w: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            embed_norm: None,
            layers: Vec::new(),
            final_norm_w: vec![],
            final_norm_b: vec![],
            modules: SpeakerModules {
                encoder_proj_w: vec![],
                encoder_proj_b: vec![],
                hidden_to_hidden_w: vec![],
                hidden_to_hidden_b: vec![],
                single_to_spks_w: vec![],
                single_to_spks_b: vec![],
                subpixel_w: vec![],
                subpixel_b: vec![],
                learnable_sil_emb: vec![],
                hidden_to_spks_w: vec![],
                hidden_to_spks_b: vec![],
                activity_head: None,
            },
            window: vec![],
            fb: vec![],
        };
        // Channel-major features [n_mels=2, frames=3]: mel 0 = 1,2,3 and
        // mel 1 = 10,20,30; the final frame is padding.
        let feats = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        // Identity projection output rows = stacked input rows:
        // frame 0 = [m0(t0), m1(t0), m0(t1), m1(t1)].
        let (emb, lengths) = model.pre_encode(&feats, 3, 3);
        assert_eq!(emb.len(), 8);
        assert_eq!(&emb[..4], &[1.0, 10.0, 2.0, 20.0]);
        assert_eq!(&emb[4..], &[3.0, 30.0, 0.0, 0.0]);
        assert_eq!(lengths, 2);
    }

    /// Subpixel head reshape: output frame t * factor + c consumes conv
    /// channels [c * tf, (c + 1) * tf) at input time t (row-major split
    /// of the 1536-channel conv output), with the ReLU/sigmoid head
    /// applied after.
    #[test]
    fn subpixel_head_reshape_known_answer() {
        let mut config = default_config();
        config.encoder.d_model = 2;
        config.encoder.n_heads = 1;
        config.modules.fc_d_model = 2;
        config.modules.tf_d_model = 2;
        config.modules.subsampling_factor = 2;
        config.num_speakers = 2;
        config.modules.num_speakers = 2;
        // One time step, encoder output [1, 2] = [a, b].
        // encoder_proj: 2 -> 2 identity.
        // subpixel conv: out 4 channels, kernel 3, diagonal weight so
        // channel o at time t passes input channel o at time t through
        // (padding 1, zero neighbors).
        let mut conv_w = vec![0.0f32; 4 * 2 * 3];
        for o in 0..4 {
            conv_w[o * 2 * 3 + (o % 2) * 3 + 1] = 1.0;
        }
        // hidden_to_hidden identity, single_to_spks: out = 2 * in.
        let model = NemotronDiarization {
            aosc_cfg: AoscModulesConfig {
                num_speakers: 2,
                fc_d_model: 2,
                tf_d_model: 2,
                subsampling_factor: 2,
                chunk_len: 1,
                fifo_len: 0,
                spkcache_len: 1,
                spkcache_update_period: 1,
                chunk_left_context: 0,
                chunk_right_context: 0,
                spkcache_sil_frames_per_spk: 1,
                pred_score_threshold: 0.25,
                max_index: 99_999,
                scores_boost_latest: 0.05,
                sil_threshold: 0.2,
                strong_boost_rate: 0.75,
                weak_boost_rate: 1.5,
                min_pos_scores_rate: 0.5,
                use_aosc: true,
            },
            config,
            pre_encode_proj_w: vec![],
            embed_norm: None,
            layers: Vec::new(),
            final_norm_w: vec![],
            final_norm_b: vec![],
            modules: SpeakerModules {
                encoder_proj_w: vec![1.0, 0.0, 0.0, 1.0],
                encoder_proj_b: vec![0.0; 2],
                hidden_to_hidden_w: vec![1.0, 0.0, 0.0, 1.0],
                hidden_to_hidden_b: vec![0.0; 2],
                single_to_spks_w: vec![2.0, 0.0, 0.0, 2.0],
                single_to_spks_b: vec![0.0; 2],
                subpixel_w: conv_w,
                subpixel_b: vec![0.0; 4],
                learnable_sil_emb: vec![],
                hidden_to_spks_w: vec![],
                hidden_to_spks_b: vec![],
                activity_head: None,
            },
            window: vec![],
            fb: vec![],
        };
        // Encoder input [a, b] = [0.1, -0.2]: conv channels pass [0.1,
        // -0.2, 0.1, -0.2], reshape gives frames [0.1, -0.2] and [0.1,
        // -0.2] (relu keeps only the positive), head doubles and
        // sigmoids: each frame becomes [sigmoid(0.2), sigmoid(0)].
        let probs = model.speaker_head(&[0.1, -0.2], 1);
        let active = 1.0 / (1.0 + (-(0.2f32)).exp());
        let silent = 0.5f32;
        assert_eq!(probs.len(), 4);
        for frame in probs.chunks_exact(2) {
            assert!(close(frame[0], active, 1e-6), "{} vs {active}", frame[0]);
            assert!(close(frame[1], silent, 1e-6), "{} vs {silent}", frame[1]);
        }
    }

    /// NeoX rope: pairs are (d, d + half), counterclockwise rotation by
    /// position angle; position 0 is identity.
    #[test]
    fn rope_neox_known_answer() {
        // One position, rotary over the whole head: (1, 0) rotated by
        // 90 degrees becomes (0, 1).
        let mut x = vec![1.0f32, 0.0];
        let cos = vec![0.0f32];
        let sin = vec![1.0f32];
        rope_neox(&mut x, 1, 2, 2, &cos, &sin);
        assert!(close(x[0], 0.0, 1e-6) && close(x[1], 1.0, 1e-6));
        // Angle 0 is the identity.
        let mut x = vec![0.5f32, -0.25];
        let cos = vec![1.0f32];
        let sin = vec![0.0f32];
        rope_neox(&mut x, 1, 2, 2, &cos, &sin);
        assert!(close(x[0], 0.5, 1e-6) && close(x[1], -0.25, 1e-6));
    }

    /// The mel frontend and single-window forward against the Python
    /// reference on a deterministic synthetic clip (checkpoint gated;
    /// set TURBOSPEECH_NEMOTRON_MODEL to override the path).
    #[test]
    fn window_forward_matches_reference() {
        let dir = std::env::var("TURBOSPEECH_NEMOTRON_MODEL").unwrap_or_else(|_| {
            format!(
                "{}/models/Nemotron-3-Diarization-fp32",
                std::env::var("HOME").unwrap()
            )
        });
        if !std::path::Path::new(&dir).join("config.json").exists() {
            eprintln!("skipping: checkpoint not present at {dir}");
            return;
        }
        let model = NemotronDiarization::load(std::path::Path::new(&dir)).unwrap();
        let (samples_shape, samples) = read_npy(&testdata("synthetic_samples.npy"));
        assert_eq!(samples_shape.len(), 1);
        let (mel_shape, golden_mel) = read_npy(&testdata("synthetic_mel.npy"));
        assert_eq!(mel_shape, vec![1, 128, 64]);

        let total = samples.len();
        let count = total / model.config().processor.hop_length;
        let mel = model.mel_features(&samples, 0, count, 0, total).unwrap();
        assert_eq!(mel.len(), 128 * 64);
        for (a, b) in mel.iter().zip(&golden_mel) {
            assert!(close(*a, *b, 1e-3), "mel mismatch {a} vs {b}");
        }

        let preds = model.forward_window(&mel, count, count);
        let (preds_shape, golden_preds) = read_npy(&testdata("synthetic_window_preds.npy"));
        assert_eq!(preds_shape, vec![1, 64, 8]);
        assert_eq!(preds.len(), 64 * 8);
        for (a, b) in preds.iter().zip(&golden_preds) {
            assert!(close(*a, *b, 1e-5), "preds mismatch {a} vs {b}");
        }
    }

    /// AOSC compression against the Python reference with this model's
    /// parameters (8 speakers, sil_per_spk 1, 0.75/1.5 boost rates) on a
    /// seeded fixture (300 frames, 16-dim embeddings).
    #[test]
    fn aosc_compression_matches_reference_fixture() {
        let (_, in_embs) = read_npy(&testdata("aosc_fixture_in_embs.npy"));
        let (_, in_preds) = read_npy(&testdata("aosc_fixture_in_preds.npy"));
        let (_, in_sil) = read_npy(&testdata("aosc_fixture_in_sil.npy"));
        let (_, out_embs) = read_npy(&testdata("aosc_fixture_out_embs.npy"));
        let (_, out_preds) = read_npy(&testdata("aosc_fixture_out_preds.npy"));

        let mc = AoscModulesConfig {
            num_speakers: 8,
            fc_d_model: 512,
            tf_d_model: 192,
            subsampling_factor: 8,
            chunk_len: 340,
            fifo_len: 40,
            spkcache_len: 264,
            spkcache_update_period: 300,
            chunk_left_context: 0,
            chunk_right_context: 40,
            spkcache_sil_frames_per_spk: 1,
            pred_score_threshold: 0.25,
            max_index: 99_999,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            use_aosc: true,
        };
        let frames = in_embs.len() / 16;
        let (embs, preds) = compress_spkcache_aosc(&in_embs, &in_preds, frames, &in_sil, &mc);
        assert_eq!(embs.len(), out_embs.len());
        assert_eq!(preds.len(), out_preds.len());
        for (a, b) in embs.iter().zip(&out_embs) {
            assert!(close(*a, *b, 1e-4), "aosc emb mismatch {a} vs {b}");
        }
        for (a, b) in preds.iter().zip(&out_preds) {
            assert!(close(*a, *b, 1e-4), "aosc pred mismatch {a} vs {b}");
        }
    }
}
