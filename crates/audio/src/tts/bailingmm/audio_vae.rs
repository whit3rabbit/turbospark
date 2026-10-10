//! The bailingmm audio tokenizer: the VAE encoder (voice-clone latents),
//! the VAE decoder with the streaming ISTFT head, and the patch
//! upsample.
//!
//! Port of the `AudioVAE` / `Encoder` / `Decoder` / `ISTFTHead` /
//! `_ISTFTWindowState` classes and the `_upsample` /
//! `_linear_upsample_1d` / `_streaming_linear_upsample` helpers in
//! `mlx_audio/tts/models/bailingmm/bailingmm.py` at the pinned
//! mlx-audio commit. All backbones are the Ming Qwen2 variant over
//! precomputed embeddings ([`super::ming_qwen2::MingQwen2`]).
//!
//! Carried reference behavior:
//! - `get_frames` slides `input_dim`-wide windows at `hop_size` with a
//!   zero pad on the ragged tail.
//! - The ISTFT head exponentiates and caps the magnitude at 1e2, forms
//!   the complex spectrum, and runs `np.fft.irfft` (1/n normalized) per
//!   frame over the periodic Hann window, overlap-adding the signal and
//!   the squared-window envelope. Non-streaming trims `pad` on both
//!   ends; streaming keeps the last `n_fft - hop` samples of each chunk
//!   in audio/window buffers, adds the saved buffer into the next
//!   chunk's head, trims the tail (or the pad on the final chunk), and
//!   divides by the envelope floored at 1e-11.
//! - The linear upsample interpolates at `(i + 0.5) / scale - 0.5`
//!   between floor/ceil neighbors with the indices clipped to range.
//! - The streaming upsample emits one patch behind: the first chunk is
//!   only buffered, each later chunk upsamples
//!   `[history_last, prev, lookahead]` and keeps the middle, and the
//!   last chunk upsamples `[history_last, prev]` from `scale` on.

use crate::fft::{ComplexF32, RealFftPlan};
use crate::lm::{self, Bias, Weights};
use crate::nn::Linear;
use crate::{Result, SpeechError};

use super::ming_qwen2::{MingQwen2, SlidingConfig};

fn periodic_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// The `ISTFTHead` plus its streaming buffers.
pub struct IstftHead {
    out: Linear,
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
}

/// The streaming state carried across decoder chunks
/// (`(audio_buffer, window_buffer)` in the reference).
#[derive(Default, Clone)]
pub struct IstftState {
    pub audio_buffer: Option<Vec<f32>>,
    pub window_buffer: Option<Vec<f32>>,
}

impl IstftHead {
    pub fn load_absolute(
        weights: &Weights,
        prefix: &str,
        dim: usize,
        n_fft: usize,
        hop: usize,
    ) -> Result<Self> {
        // The reference defaults the window to the periodic Hann but a
        // checkpoint may override it (head.istft.window is a saved
        // parameter on the MLX side).
        let window_name = format!("{prefix}istft.window");
        let window = if weights.contains_absolute(&window_name) {
            weights.tensor_absolute(&window_name, &[n_fft])?
        } else {
            periodic_hann(n_fft)
        };
        Ok(Self {
            out: weights.linear_absolute(
                &format!("{prefix}out"),
                dim,
                n_fft + 2,
                Bias::Required,
            )?,
            n_fft,
            hop,
            window,
        })
    }

    /// Spectrum to samples for one frame: `irfft(mag * e^{i phase})` with
    /// the magnitude exponentiated and capped at 1e2.
    fn frames_to_ifft(&self, rows: &[f32], frames: usize) -> Result<Vec<Vec<f32>>> {
        let bins = self.n_fft / 2 + 1;
        let plan =
            RealFftPlan::new(self.n_fft).map_err(|error| SpeechError::Audio(error.to_string()))?;
        let mut chunks = Vec::with_capacity(frames);
        let mut spectrum = Vec::with_capacity(bins);
        for t in 0..frames {
            spectrum.clear();
            for b in 0..bins {
                let mag = rows[t * (self.n_fft + 2) + b];
                let phase = rows[t * (self.n_fft + 2) + bins + b];
                let magnitude = mag.exp().min(1e2);
                // numpy irfft reads only the real parts of the DC and
                // Nyquist bins; zero the imaginary parts so the shared
                // real-IFFT plan agrees on arbitrary (non-Hermitian)
                // spectra.
                let (re, im) = if b == 0 || b == bins - 1 {
                    (magnitude * phase.cos(), 0.0)
                } else {
                    (magnitude * phase.cos(), magnitude * phase.sin())
                };
                spectrum.push(ComplexF32::new(re, im));
            }
            chunks.push(
                plan.inverse(&spectrum)
                    .map_err(|error| SpeechError::Audio(error.to_string()))?,
            );
        }
        Ok(chunks)
    }

    /// Window and overlap-add the ifft frames into the signal and the
    /// squared-window envelope.
    fn overlap_add(&self, frames: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let output_size = (frames.len() - 1) * self.hop + self.n_fft;
        let mut audio = vec![0.0f32; output_size];
        let mut env = vec![0.0f32; output_size];
        for (t, chunk) in frames.iter().enumerate() {
            let start = t * self.hop;
            for (offset, sample) in chunk.iter().enumerate() {
                let windowed = sample * self.window[offset];
                audio[start + offset] += windowed;
                env[start + offset] += self.window[offset] * self.window[offset];
            }
        }
        (audio, env)
    }

    /// The reference `_buffer_process`: streaming trim/merge on one
    /// signal-buffer pair. Returns the trimmed signal and the new buffer.
    fn buffer_process(
        &self,
        mut x: Vec<f32>,
        buffer: &mut Option<Vec<f32>>,
        last_chunk: bool,
    ) -> Vec<f32> {
        let buffer_len = self.n_fft - self.hop;
        let pad = (self.n_fft - self.hop) / 2;
        if buffer.is_none() {
            x.drain(..pad.min(x.len()));
        } else if let Some(saved) = buffer.as_ref() {
            for (sample, saved) in x.iter_mut().zip(saved) {
                *sample += saved;
            }
        }
        *buffer = Some(x[x.len().saturating_sub(buffer_len)..].to_vec());
        let keep = if last_chunk {
            x.len().saturating_sub(pad)
        } else {
            x.len().saturating_sub(buffer_len)
        };
        x.truncate(keep);
        x
    }

    fn normalize(&self, audio: &mut [f32], env: &mut [f32]) {
        for (sample, weight) in audio.iter_mut().zip(env) {
            *sample /= weight.max(1e-11);
        }
    }

    /// The cacheless decode (`low_level_reconstruct` with
    /// `use_cache=False`): trim `pad` on both ends.
    pub fn forward(&self, hidden: &[f32], hidden_dim: usize) -> Result<Vec<f32>> {
        let rows = self.out.forward(hidden, hidden.len() / hidden_dim);
        let frames = rows.len() / (self.n_fft + 2);
        let chunks = self.frames_to_ifft(&rows, frames)?;
        let (mut audio, mut env) = self.overlap_add(&chunks);
        let pad = (self.n_fft - self.hop) / 2;
        let keep = audio.len().saturating_sub(2 * pad);
        audio.drain(keep..);
        audio.drain(..pad.min(audio.len()));
        env.drain(keep..);
        env.drain(..pad.min(env.len()));
        self.normalize(&mut audio, &mut env);
        Ok(audio)
    }

    /// The streaming decode (`use_cache=True`): buffers in `state`
    /// advance per chunk; the final chunk trims the pad instead of the
    /// tail.
    pub fn forward_streaming(
        &self,
        hidden: &[f32],
        hidden_dim: usize,
        state: &mut IstftState,
        last_chunk: bool,
    ) -> Result<Vec<f32>> {
        let rows = self.out.forward(hidden, hidden.len() / hidden_dim);
        let frames = rows.len() / (self.n_fft + 2);
        let chunks = self.frames_to_ifft(&rows, frames)?;
        let (mut audio, mut env) = self.overlap_add(&chunks);
        audio = self.buffer_process(audio, &mut state.audio_buffer, last_chunk);
        env = self.buffer_process(env, &mut state.window_buffer, last_chunk);
        self.normalize(&mut audio, &mut env);
        Ok(audio)
    }
}

/// The streaming linear-upsample state (`prev_chunk`, `history_last`).
#[derive(Default, Clone)]
pub struct UpsampleState {
    prev_chunk: Option<Vec<f32>>,
    history_last: Option<Vec<f32>>,
}

/// The `Decoder`: patch upsample, Ming Qwen2 backbone over the cached
/// positions, and the streaming ISTFT head.
pub struct Decoder {
    backbone: MingQwen2,
    fc1: Linear,
    head: IstftHead,
    patch_size: usize,
}

#[allow(clippy::too_many_arguments)]
impl Decoder {
    pub fn load_absolute(
        weights: &Weights,
        stack_prefix: &str,
        backbone_prefix: &str,
        backbone_config: lm::DenseConfig,
        backbone_sliding: &SlidingConfig,
        latent_dim: usize,
        output_dim: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let hidden = backbone_config.hidden_size;
        Ok(Self {
            backbone: MingQwen2::load_absolute(
                weights,
                backbone_prefix,
                backbone_config,
                backbone_sliding,
            )?,
            fc1: weights.linear_absolute(
                &format!("{stack_prefix}fc1"),
                latent_dim,
                hidden,
                Bias::Required,
            )?,
            head: IstftHead::load_absolute(
                weights,
                &format!("{stack_prefix}head."),
                hidden,
                output_dim * 4,
                output_dim,
            )?,
            patch_size,
        })
    }

    /// The linear upsample over `[rows, channels]` with the reference
    /// `(i + 0.5) / scale - 0.5` index mapping.
    fn linear_upsample(&self, x: &[f32], rows: usize, channels: usize) -> Vec<f32> {
        let scale = self.patch_size;
        let out_rows = rows * scale;
        let mut out = vec![0.0f32; out_rows * channels];
        for out_row in 0..out_rows {
            let idx = (out_row as f32 + 0.5) / scale as f32 - 0.5;
            let left_f = idx.floor();
            let left = (left_f as i64).clamp(0, rows as i64 - 1) as usize;
            let right = (left_f as i64 + 1).clamp(0, rows as i64 - 1) as usize;
            let w_right = idx - left_f;
            for c in 0..channels {
                let left_value = x[left * channels + c];
                let right_value = x[right * channels + c];
                out[out_row * channels + c] = left_value * (1.0 - w_right) + right_value * w_right;
            }
        }
        out
    }

    /// The non-streaming upsample (`_upsample`).
    fn upsample(&self, x: &[f32], rows: usize, channels: usize) -> Vec<f32> {
        self.linear_upsample(x, rows, channels)
    }

    /// `_streaming_linear_upsample` over `[rows, channels]` chunks.
    /// `None` input with `is_last == false` keeps the state untouched.
    /// Returns the emitted chunk, if any.
    fn streaming_upsample(
        &self,
        x: Option<&[f32]>,
        rows: usize,
        channels: usize,
        state: &mut UpsampleState,
        is_last: bool,
    ) -> Option<Vec<f32>> {
        if x.is_none() && !is_last {
            return None;
        }
        let first_use = state.prev_chunk.is_none();
        if first_use && is_last {
            let out = self.upsample(x.expect("last chunk input"), rows, channels);
            *state = UpsampleState::default();
            return Some(out);
        }
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        if state.prev_chunk.is_none() {
            state.prev_chunk = Some(x.expect("first chunk input").to_vec());
            if !is_last {
                return None;
            }
        }
        if let Some(prev) = state.prev_chunk.clone() {
            let prev_rows = prev.len() / channels;
            let lookahead: Vec<f32> = match x {
                Some(chunk) => chunk[..channels].to_vec(),
                None => prev[prev.len() - channels..].to_vec(),
            };
            match &state.history_last {
                None => {
                    let mut input = prev.clone();
                    input.extend_from_slice(&lookahead);
                    let up = self.linear_upsample(&input, prev_rows + 1, channels);
                    let keep = prev_rows * self.patch_size * channels;
                    outputs.push(up[..keep].to_vec());
                }
                Some(history) => {
                    let mut input = history.clone();
                    input.extend_from_slice(&prev);
                    input.extend_from_slice(&lookahead);
                    let up = self.linear_upsample(&input, prev_rows + 2, channels);
                    let start = self.patch_size * channels;
                    let end = start + prev_rows * self.patch_size * channels;
                    outputs.push(up[start..end].to_vec());
                }
            }
            state.history_last = Some(prev[prev.len() - channels..].to_vec());
            state.prev_chunk = x.map(|chunk| chunk.to_vec());
        }
        if is_last {
            let prev = state.prev_chunk.clone().expect("previous chunk");
            let prev_rows = prev.len() / channels;
            let history = state
                .history_last
                .clone()
                .unwrap_or_else(|| prev[..channels].to_vec());
            let mut input = history;
            input.extend_from_slice(&prev);
            let up = self.linear_upsample(&input, prev_rows + 1, channels);
            outputs.push(up[self.patch_size * channels..].to_vec());
            *state = UpsampleState::default();
        }
        if outputs.is_empty() {
            return None;
        }
        Some(outputs.concat())
    }

    /// `low_level_reconstruct` with `use_cache=True`, the reference
    /// generate path. `latent` is `[patch, latent_dim]`; the emitted
    /// waveform chunk (possibly empty) advances the caches, the
    /// upsample state, and the ISTFT buffers.
    pub fn low_level_reconstruct(
        &mut self,
        latent: &[f32],
        latent_rows: usize,
        caches: &mut lm::LayerCaches,
        upsample_state: &mut UpsampleState,
        istft_state: &mut IstftState,
        last_chunk: bool,
    ) -> Result<Vec<f32>> {
        let channels = self.fc1.input;
        let _ = channels;
        let hidden = self.fc1.forward(latent, latent_rows);
        let hidden_dim = self.fc1.output;
        let upsampled = if self.patch_size != 0 && self.patch_size != usize::MAX {
            match self.streaming_upsample(
                Some(&hidden),
                latent_rows,
                hidden_dim,
                upsample_state,
                last_chunk,
            ) {
                Some(chunk) => chunk,
                None => return Ok(Vec::new()),
            }
        } else {
            hidden
        };
        let decoded = self.backbone.forward_hidden(&upsampled, Some(caches))?;
        self.head
            .forward_streaming(&decoded, hidden_dim, istft_state, last_chunk)
    }

    /// The cacheless decode (`use_cache=False`), exposed for tests.
    pub fn reconstruct_cacheless(
        &mut self,
        latent: &[f32],
        latent_rows: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.fc1.forward(latent, latent_rows);
        let hidden_dim = self.fc1.output;
        let upsampled = self.upsample(&hidden, latent_rows, hidden_dim);
        let decoded = self.backbone.forward_hidden(&upsampled, None)?;
        self.head.forward(&decoded, hidden_dim)
    }
}

/// The `Encoder`: frame windows, the Ming Qwen2 stack over the frame
/// embeddings, the optional patch aggregator, and the mean/logvar
/// projection.
pub struct Encoder {
    backbone: MingQwen2,
    aggregator: Option<MingQwen2>,
    cls_embed: Option<Vec<f32>>,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    input_dim: usize,
    hop_size: usize,
    latent_dim: usize,
    patch_size: usize,
}

#[allow(clippy::too_many_arguments)]
impl Encoder {
    pub fn load_absolute(
        weights: &Weights,
        stack_prefix: &str,
        backbone_prefix: &str,
        aggregator_prefix: &str,
        cls_name: &str,
        backbone_config: lm::DenseConfig,
        backbone_sliding: &SlidingConfig,
        input_dim: usize,
        hop_size: usize,
        latent_dim: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let hidden = backbone_config.hidden_size;
        // The patch aggregator is a second backbone with 4 layers over
        // the same geometry, present only when patch_size != -1. The
        // CLS embedding is a plain array in the reference (it only
        // reaches a checkpoint when the saver freezes it); the loader
        // accepts its absence and refuses the patch path without it.
        let (aggregator, cls_embed) = if patch_size != 0 {
            let mut agg_config = backbone_config.clone();
            agg_config.num_hidden_layers = 4;
            let agg = MingQwen2::load_absolute(
                weights,
                aggregator_prefix,
                agg_config,
                &SlidingConfig::default(),
            )?;
            let cls = if weights.contains_absolute(cls_name) {
                Some(weights.tensor_absolute(cls_name, &[1, hidden])?)
            } else {
                None
            };
            (Some(agg), cls)
        } else {
            (None, None)
        };
        Ok(Self {
            backbone: MingQwen2::load_absolute(
                weights,
                backbone_prefix,
                backbone_config,
                backbone_sliding,
            )?,
            aggregator,
            cls_embed,
            fc1: weights.linear_absolute(
                &format!("{stack_prefix}fc1"),
                input_dim,
                hidden,
                Bias::None,
            )?,
            fc2: weights.linear_absolute(
                &format!("{stack_prefix}fc2"),
                hidden,
                hidden,
                Bias::Required,
            )?,
            fc3: weights.linear_absolute(
                &format!("{stack_prefix}fc3"),
                hidden,
                latent_dim * 2,
                Bias::Required,
            )?,
            input_dim,
            hop_size,
            latent_dim,
            patch_size,
        })
    }

    /// `get_frames`: sliding `input_dim` windows at `hop_size` with a
    /// zero pad on the ragged tail. Returns `[frames, input_dim]`.
    pub fn get_frames(&self, samples: &[f32]) -> Vec<f32> {
        let num_frames = samples.len().div_ceil(self.hop_size);
        let expected_len = (num_frames - 1) * self.hop_size + self.input_dim;
        let mut padded = samples.to_vec();
        padded.resize(expected_len.max(samples.len()), 0.0);
        let mut frames = Vec::with_capacity(num_frames * self.input_dim);
        for frame in 0..num_frames {
            let start = frame * self.hop_size;
            frames.extend_from_slice(&padded[start..start + self.input_dim]);
        }
        frames
    }

    /// `pad_patch_insert_cls`: pad the row count to a multiple of the
    /// patch size, split into patches, and append the learned CLS row to
    /// each patch.
    fn pad_patch_insert_cls(&self, x: &[f32], rows: usize, channels: usize) -> Result<Vec<f32>> {
        let cls = self.cls_embed.as_ref().ok_or_else(|| SpeechError::Input {
            why: "patch aggregator requested without a cls embedding".into(),
        })?;
        let remainder = rows % self.patch_size;
        let pad = if remainder == 0 {
            0
        } else {
            self.patch_size - remainder
        };
        let padded_rows = rows + pad;
        let patches = padded_rows / self.patch_size;
        let mut out = vec![0.0f32; patches * (self.patch_size + 1) * channels];
        for patch in 0..patches {
            let dst = patch * (self.patch_size + 1) * channels;
            // The reference concatenates the learned CLS row AFTER each
            // patch's rows ([patch rows, CLS]), so the last position of
            // every group is the CLS.
            for row in 0..self.patch_size {
                let src_row = patch * self.patch_size + row;
                let dst_row = dst + row * channels;
                if src_row < rows {
                    out[dst_row..dst_row + channels]
                        .copy_from_slice(&x[src_row * channels..(src_row + 1) * channels]);
                }
            }
            let dst_row = dst + self.patch_size * channels;
            out[dst_row..dst_row + channels].copy_from_slice(cls);
        }
        Ok(out)
    }

    /// The encoder forward: `[frames, 2 * latent_dim]` mean/logvar rows.
    pub fn forward(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let frames_src = self.get_frames(samples);
        let frame_rows = frames_src.len() / self.input_dim;
        let hidden_dim = self.fc1.output;
        let x = self.fc1.forward(&frames_src, frame_rows);
        let x = self.fc2.forward(&x, frame_rows);
        let mut x = self.backbone.forward_hidden(&x, None)?;
        if self.aggregator.is_some() {
            let patched = self.pad_patch_insert_cls(&x, frame_rows, hidden_dim)?;
            let agg_rows = patched.len() / hidden_dim;
            if let Some(aggregator) = self.aggregator.as_mut() {
                let aggregated = aggregator.forward_hidden(&patched, None)?;
                // Reshape [-1, patch + 1, dim] and keep each patch's last
                // (CLS) row.
                let group = self.patch_size + 1;
                let patches = agg_rows / group;
                let mut kept = Vec::with_capacity(patches * hidden_dim);
                for patch in 0..patches {
                    let src = (patch * group + (group - 1)) * hidden_dim;
                    kept.extend_from_slice(&aggregated[src..src + hidden_dim]);
                }
                x = kept;
            }
        }
        Ok(self.fc3.forward(&x, x.len() / hidden_dim))
    }

    /// `encode_latent`: the reparameterization draw
    /// `mu + std * normal` with the noise supplied by the caller.
    /// Returns the latent `[rows, latent_dim]`.
    pub fn encode_latent(
        &mut self,
        samples: &[f32],
        noise: &mut dyn FnMut() -> f32,
    ) -> Result<(Vec<f32>, usize)> {
        let stats = self.forward(samples)?;
        let rows = stats.len() / (2 * self.latent_dim);
        let channels = self.latent_dim;
        // The reference draws the noise at mu's pre-transpose shape
        // [batch, latent, frames] and consumes it row-major, i.e.
        // channel-major over these [frames, latent] rows; keep that
        // order so a recorded draw replays 1:1.
        let mut mu = vec![0.0f32; rows * channels];
        let mut std = vec![0.0f32; rows * channels];
        for row in 0..rows {
            for c in 0..channels {
                mu[row * channels + c] = stats[row * 2 * channels + c];
                let logvar = stats[row * 2 * channels + channels + c];
                std[row * channels + c] = (0.5 * logvar.clamp(-30.0, 20.0)).exp();
            }
        }
        let mut latent = vec![0.0f32; rows * channels];
        for c in 0..channels {
            for row in 0..rows {
                latent[row * channels + c] =
                    mu[row * channels + c] + std[row * channels + c] * noise();
            }
        }
        Ok((latent, rows))
    }
}
