//! StepAudio2 generative token-to-wav pipeline.
//!
//! Reference: `mlx_audio/codec/models/stepaudio2/` (token2wav.py,
//! flow.py, upsample_encoder_v2.py, decoder_dit.py, flow_matching.py,
//! hift.py, speaker.py, convert.py) plus the chatterbox s3gen modules
//! it imports (transformer/, hifigan.py, f0_predictor.py, xvector.py,
//! mel.py), all at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models/stepaudio2).
//!
//! Unlike the encode/decode tokenizers in this folder, StepAudio2 is a
//! conditional generative pipeline: an S3-token stream plus a speaker
//! prompt go through a flow-matching DiT (tokens -> mel), and an
//! NSF-iSTFT vocoder renders mel -> 24 kHz audio. The three submodels
//! live in [`flow`], [`hift`], and [`campplus`]; this module owns the
//! configuration, the checkpoint loaders (one file, one prefix per
//! submodel), and the [`StepAudio2Token2Wav`] orchestration.
//!
//! Scope: batch-of-one inference with exact lengths. The reference
//! asserts batch 1, and with `token_len == sequence length` every
//! padding mask in the flow reduces to all-valid, so the additive
//! attention masks are not threaded. File loading, resampling, and the
//! S3 speech tokenizer stay outside the port (the caller passes
//! pre-resampled waveforms and prompt tokens, as the runtime does).
//!
//! Randomness: the CFM `rand_noise` is a checkpoint tensor (the
//! reference materializes it at init and convert.py keeps it), so the
//! flow solve is deterministic. The HiFT sine generator draws an
//! initial phase and a noise bed at inference; the Rust port takes
//! those as explicit [`hift::SineDraws`] inputs, which is also how the
//! fixture records and replays one realization.

pub mod campplus;
pub mod flow;
pub mod hift;

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::mel::{self, MelScale};
use crate::stft;
use crate::{Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// Output sample rate of the HiFT vocoder.
pub const SAMPLE_RATE: u32 = 24_000;

/// Chatterbox s3gen prompt-mel geometry (chatterbox/s3gen/mel.py).
const MEL_N_FFT: usize = 1920;
const MEL_HOP: usize = 480;
const MEL_WIN: usize = 1920;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 8000.0;

/// Geometry of the flow model, one-to-one with the reference keyword
/// defaults (`flow.py::CausalMaskedDiffWithXvec` and the encoder / DiT
/// values it builds). The HiFT and CAMPPlus submodels keep the full
/// reference structure and carry no configuration.
#[derive(Debug, Clone)]
pub struct StepAudio2Config {
    /// Token embedding width and encoder width.
    pub input_size: usize,
    /// Mel bins (fixed by the vocoder at 80).
    pub output_size: usize,
    /// CAMPPlus speaker embedding width feeding `spk_embed_affine_layer`.
    pub spk_embed_dim: usize,
    /// FSQ token vocabulary (3^8 for the 25 Hz S3 tokenizer).
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub linear_units: usize,
    pub num_blocks: usize,
    pub num_up_blocks: usize,
    /// Mel-frames-per-token factor: the encoder upsamples by this.
    pub up_stride: usize,
    pub pre_lookahead_len: usize,
    /// DiT estimator geometry.
    pub dit_depth: usize,
    pub dit_hidden: usize,
    pub dit_heads: usize,
    pub dit_head_dim: usize,
    pub dit_mlp_ratio: f32,
    /// Classifier-free guidance rate of the Euler solver.
    pub inference_cfg_rate: f32,
}

impl StepAudio2Config {
    /// Reference keyword defaults.
    pub fn reference() -> Self {
        StepAudio2Config {
            input_size: 512,
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561,
            attention_heads: 8,
            linear_units: 2048,
            num_blocks: 6,
            num_up_blocks: 4,
            up_stride: 2,
            pre_lookahead_len: 3,
            dit_depth: 16,
            dit_hidden: 512,
            dit_heads: 8,
            dit_head_dim: 64,
            dit_mlp_ratio: 4.0,
            inference_cfg_rate: 0.7,
        }
    }

    /// Parses a JSON object over the reference defaults; missing or
    /// null fields keep the default.
    pub fn from_json_with_defaults(
        value: &serde_json::Value,
        defaults: StepAudio2Config,
    ) -> Result<Self> {
        let field = |name: &str| -> Result<Option<f64>> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => v.as_f64().map(Some).ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "expected a number".to_string(),
                }),
            }
        };
        let int = |name: &str, fallback: usize| -> Result<usize> {
            Ok(field(name)?.map(|v| v as usize).unwrap_or(fallback))
        };
        let float = |name: &str, fallback: f32| -> Result<f32> {
            Ok(field(name)?.map(|v| v as f32).unwrap_or(fallback))
        };
        Ok(StepAudio2Config {
            input_size: int("input_size", defaults.input_size)?,
            output_size: int("output_size", defaults.output_size)?,
            spk_embed_dim: int("spk_embed_dim", defaults.spk_embed_dim)?,
            vocab_size: int("vocab_size", defaults.vocab_size)?,
            attention_heads: int("attention_heads", defaults.attention_heads)?,
            linear_units: int("linear_units", defaults.linear_units)?,
            num_blocks: int("num_blocks", defaults.num_blocks)?,
            num_up_blocks: int("num_up_blocks", defaults.num_up_blocks)?,
            up_stride: int("up_stride", defaults.up_stride)?,
            pre_lookahead_len: int("pre_lookahead_len", defaults.pre_lookahead_len)?,
            dit_depth: int("dit_depth", defaults.dit_depth)?,
            dit_hidden: int("dit_hidden", defaults.dit_hidden)?,
            dit_heads: int("dit_heads", defaults.dit_heads)?,
            dit_head_dim: int("dit_head_dim", defaults.dit_head_dim)?,
            dit_mlp_ratio: float("dit_mlp_ratio", defaults.dit_mlp_ratio)?,
            inference_cfg_rate: float("inference_cfg_rate", defaults.inference_cfg_rate)?,
        })
    }

    /// Parses a JSON object over the reference defaults.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        Self::from_json_with_defaults(value, Self::reference())
    }
}

/// Plain linear layer, `[out, in]` row-major weights, row-major
/// activations `[rows, in]` -> `[rows, out]`.
#[derive(Debug, Clone)]
pub(crate) struct SaLinear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaLinear {
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_dim])?)
        } else {
            None
        };
        Ok(SaLinear {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    pub fn load_no_bias(
        file: &SafetensorsFile,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let weight = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        Ok(SaLinear {
            in_dim,
            out_dim,
            weight,
            bias: None,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        crate::ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.in_dim,
            self.out_dim,
        )
    }
}

/// Plain Conv1d loaded from the MLX checkpoint layout `[out, K, in]`
/// and stored PyTorch `[out, in/groups, K]` for the crate kernel.
/// Channel-major `x [in_ch, seq]` -> `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub(crate) struct SaConv1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
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
        Ok(SaConv1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        crate::ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

/// ConvTranspose1d loaded from the MLX layout `[out, K, in]` (the
/// out-first transposed-conv layout, matching the reference sanitize
/// transpose `(1, 2, 0)` of the PyTorch `[in, out, K]` weight) and
/// stored PyTorch `[in, out, K]` for the crate kernel. Channel-major
/// `x [in_ch, seq]` -> `[out_ch, out_seq]`.
#[derive(Debug, Clone)]
pub(crate) struct SaConvTr1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl SaConvTr1d {
    pub fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let stored = load_f32_shaped(file, &format!("{prefix}.weight"), &[out_ch, kernel, in_ch])?;
        let mut weight = vec![0.0f32; stored.len()];
        for ic in 0..in_ch {
            for kk in 0..kernel {
                for oc in 0..out_ch {
                    weight[ic * out_ch * kernel + oc * kernel + kk] =
                        stored[oc * kernel * in_ch + kk * in_ch + ic];
                }
            }
        }
        let bias_name = format!("{prefix}.bias");
        let bias = if file.contains_tensor(&bias_name) {
            Some(load_f32_shaped(file, &bias_name, &[out_ch])?)
        } else {
            None
        };
        Ok(SaConvTr1d {
            in_ch,
            out_ch,
            kernel,
            stride,
            padding,
            weight,
            bias,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        crate::ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            0,
            1,
        )
    }
}

/// Affine LayerNorm state.
#[derive(Debug, Clone)]
pub(crate) struct SaLayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl SaLayerNorm {
    pub fn load(file: &SafetensorsFile, prefix: &str, dim: usize) -> Result<Self> {
        Ok(SaLayerNorm {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[dim])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[dim])?,
        })
    }

    /// In-place over row-major `[rows, dim]`.
    pub fn forward(&self, x: &mut [f32], rows: usize, eps: f32) {
        let dim = self.weight.len();
        crate::ops::layernorm(x, rows, dim, &self.weight, Some(&self.bias), eps);
    }
}

/// Channel-major <-> row-major conversions used across the submodels.
pub(crate) fn to_cm(rows: &[f32], rows_n: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows_n * cols];
    for r in 0..rows_n {
        for c in 0..cols {
            out[c * rows_n + r] = rows[r * cols + c];
        }
    }
    out
}

pub(crate) fn to_rows(cm: &[f32], ch: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * ch];
    for c in 0..ch {
        for f in 0..frames {
            out[f * ch + c] = cm[c * frames + f];
        }
    }
    out
}

pub(crate) fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

pub(crate) fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// Mish activation (`x * tanh(softplus(x))`), numerically stable.
pub(crate) fn mish(x: &mut [f32]) {
    for v in x.iter_mut() {
        let sp = v.max(0.0) + (1.0 + (-v.abs()).exp()).ln();
        *v *= sp.tanh();
    }
}

/// Symmetric Hann window (the `periodic=False` convention the
/// chatterbox mel frontend uses; `dsp::hann_window` is the periodic
/// variant the STFT reconstruction paths need).
pub(crate) fn hann_symmetric(size: usize) -> Vec<f32> {
    match size {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..size)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos()))
                    as f32
            })
            .collect(),
    }
}

/// Torch-style reflect padding along the sequence axis of a
/// channel-major `[ch, seq]` signal (edge sample not repeated).
pub(crate) fn reflect_pad_cm(x: &[f32], ch: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / ch;
    let mut out = vec![0.0f32; ch * (seq + 2 * pad)];
    for c in 0..ch {
        let src = &x[c * seq..(c + 1) * seq];
        let dst = &mut out[c * (seq + 2 * pad)..(c + 1) * (seq + 2 * pad)];
        for p in 0..pad {
            // torch reflect: descending prefix [x_pad, ..., x_1], the
            // edge sample is not repeated.
            dst[p] = src[pad - p];
            dst[pad + seq + p] = src[seq - 2 - p];
        }
        dst[pad..pad + seq].copy_from_slice(src);
    }
    out
}

/// The chatterbox s3gen 24 kHz prompt mel
/// (`chatterbox/s3gen/mel.py::mel_spectrogram`): reflect-pad by
/// `(n_fft - hop) / 2`, uncentered STFT with a symmetric Hann window,
/// magnitude spectrum, Slaney filterbank (scale and area norm), and
/// natural-log compression floored at 1e-5. Returns frames as rows
/// `[T', num_mels]`.
pub(crate) fn chatterbox_mel(samples: &[f32], num_mels: usize) -> Result<Vec<Vec<f32>>> {
    let pad = (MEL_N_FFT - MEL_HOP) / 2;
    if samples.len() < pad + 2 {
        return Err(SpeechError::Input {
            why: format!(
                "prompt waveform needs at least {} samples for the mel reflect pad",
                pad + 2
            ),
        });
    }
    let padded = reflect_pad_cm(samples, 1, pad);
    let options = stft::StftOptions {
        fft_size: MEL_N_FFT,
        hop: MEL_HOP,
        window: hann_symmetric(MEL_WIN),
        center: false,
    };
    let spectra = stft::stft(&padded, &options)?;
    let filterbank = mel::mel_filterbank(
        num_mels,
        MEL_N_FFT,
        SAMPLE_RATE,
        MEL_FMIN,
        Some(MEL_FMAX),
        MelScale::Slaney,
    )?;
    Ok(spectra
        .into_iter()
        .map(|spectrum| {
            let mag: Vec<f32> = spectrum.iter().map(|c| c.re.hypot(c.im)).collect();
            filterbank
                .project(&mag)
                .map(|row| row.into_iter().map(|v| v.max(1e-5).ln()).collect())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

/// A prepared speaker prompt: prompt tokens, the length-matched prompt
/// mel, and the speaker embedding.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub prompt_token: Vec<i32>,
    /// Prompt mel frames, row-major `[target_len, num_mels]`.
    pub prompt_feat: Vec<f32>,
    /// Speaker embedding `[spk_embed_dim]`.
    pub embedding: Vec<f32>,
}

/// The assembled StepAudio2 pipeline: flow (tokens -> mel), HiFT
/// (mel -> wav), and the optional CAMPPlus speaker encoder used by
/// [`Self::prepare_prompt`].
pub struct StepAudio2Token2Wav {
    pub config: StepAudio2Config,
    pub flow: flow::StepAudio2Flow,
    pub hift: hift::StepAudio2HiFT,
    pub campplus: Option<campplus::StepAudio2CampPlus>,
}

impl StepAudio2Token2Wav {
    /// Loads all submodels from one safetensors file with the
    /// `flow.` / `hift.` / `campplus.` prefixes the fixture and the
    /// converted checkpoints use. CAMPPlus is optional (the reference
    /// only requires it when the speaker embedding is not supplied).
    pub fn load(config: StepAudio2Config, file: &SafetensorsFile) -> Result<Self> {
        let flow = flow::StepAudio2Flow::load(&config, file)?;
        let hift = hift::StepAudio2HiFT::load(file)?;
        let campplus = if file.contains_tensor("campplus.head.conv1.weight") {
            Some(campplus::StepAudio2CampPlus::load(file)?)
        } else {
            None
        };
        Ok(StepAudio2Token2Wav {
            config,
            flow,
            hift,
            campplus,
        })
    }

    /// Opens a checkpoint directory holding the tiny fixture file
    /// (`tiny_weights.safetensors` + `tiny_config.json`) or the
    /// converted checkpoint layout (`flow.safetensors`, `hift.safetensors`,
    /// `campplus.safetensors`; the flow geometry is read from
    /// `flow.yaml`-style config.json when present).
    pub fn open(dir: &Path) -> Result<Self> {
        let tiny = dir.join("tiny_weights.safetensors");
        if tiny.exists() {
            let file = SafetensorsFile::open(&tiny).map_err(SpeechError::from)?;
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("tiny_config.json")).map_err(|e| {
                    SpeechError::BadConfig {
                        field: "tiny_config.json".to_string(),
                        why: e.to_string(),
                    }
                })?,
            )
            .map_err(|e| SpeechError::BadConfig {
                field: "tiny_config.json".to_string(),
                why: e.to_string(),
            })?;
            let config = StepAudio2Config::from_json(&config)?;
            return Self::load(config, &file);
        }
        Err(SpeechError::BadConfig {
            field: dir.display().to_string(),
            why: "no tiny_weights.safetensors found (converted-checkpoint loading \
                  requires the three per-submodel safetensors files; point the \
                  loader at a combined file instead)"
                .to_string(),
        })
    }

    /// Builds the speaker prompt from pre-resampled waveforms. When
    /// `speaker_embedding` is `None` the CAMPPlus encoder computes it
    /// from the 16 kHz waveform; when `prompt_tokens` is `None` the
    /// caller must have provided a speech tokenizer upstream, so this
    /// port requires them (the decode contract consumes tokens).
    pub fn prepare_prompt(
        &self,
        audio_16k: &[f32],
        audio_24k: &[f32],
        prompt_tokens: &[i32],
        speaker_embedding: Option<Vec<f32>>,
    ) -> Result<Prompt> {
        let embedding = match speaker_embedding {
            Some(e) => e,
            None => {
                let encoder = self.campplus.as_ref().ok_or_else(|| SpeechError::Input {
                    why: "speaker_embedding is required unless a CAMPPlus speaker \
                          encoder is loaded"
                        .to_string(),
                })?;
                encoder.inference(audio_16k)?
            }
        };
        if embedding.len() != self.config.spk_embed_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    embedding.len(),
                    self.config.spk_embed_dim
                ),
            });
        }
        let mel_frames = chatterbox_mel(audio_24k, self.config.output_size)?;
        let n_mels = self.config.output_size;
        let target = prompt_tokens.len() * self.flow.up_rate();
        let mut flat: Vec<f32> = Vec::with_capacity(target * n_mels);
        if mel_frames.len() < target {
            // Pad by repeating the last frame.
            for frame in &mel_frames {
                flat.extend_from_slice(frame);
            }
            if let Some(last) = mel_frames.last() {
                while flat.len() < target * n_mels {
                    flat.extend_from_slice(last);
                }
            }
        } else {
            for frame in &mel_frames[..target] {
                flat.extend_from_slice(frame);
            }
        }
        Ok(Prompt {
            prompt_token: prompt_tokens.to_vec(),
            prompt_feat: flat,
            embedding,
        })
    }

    /// Synthesizes the audio for `speech_tokens` conditioned on the
    /// prompt. Returns 24 kHz samples. `draws` injects the HiFT
    /// sine-generator randomness; `None` means unvoiced-neutral zeros
    /// (documented in the hift module).
    pub fn decode(
        &mut self,
        speech_tokens: &[i32],
        prompt: &Prompt,
        n_timesteps: usize,
        draws: Option<&hift::SineDraws>,
    ) -> Result<Vec<f32>> {
        let mel = self.flow.infer(
            speech_tokens,
            &prompt.prompt_token,
            &prompt.prompt_feat,
            &prompt.embedding,
            n_timesteps,
        )?;
        self.hift.infer(&mel, draws)
    }
}

#[cfg(test)]
mod tests;
