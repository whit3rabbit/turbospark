//! Echo-TTS audio bridge (`audio.py`): the PCA latent space over the
//! Fish S1 DAC continuous latents, the flattening-point crop, and the
//! speaker-latent extraction from a reference waveform.

use std::path::Path;

use crate::codec::fish_s1_dac::FishS1Dac;
use crate::{ops, Result, SpeechError};

/// `PCAState`: `pca_components [latent, codec_dim]` row-major,
/// `pca_mean [codec_dim]`, and the scalar `latent_scale`.
#[derive(Debug, Clone)]
pub struct PcaState {
    pub components: Vec<f32>,
    pub mean: Vec<f32>,
    pub latent_scale: f32,
    pub latent_dim: usize,
    pub codec_dim: usize,
}

impl PcaState {
    /// `load_pca_state`: reads `pca_components`, `pca_mean`, and
    /// `latent_scale` from one safetensors file (the stored shapes are
    /// whatever the checkpoint saved; only the element counts matter).
    pub fn open(path: &Path) -> Result<Self> {
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(path)
            .map_err(SpeechError::from)?;
        let tensor = |name: &str| -> Result<Vec<f32>> {
            let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
                name: name.to_owned(),
                why: "tensor is missing".into(),
            })?;
            let shape = descriptor.shape.clone();
            crate::nn::load_tensor(&file, name, &shape)
        };
        let components = tensor("pca_components")?;
        let mean = tensor("pca_mean")?;
        let scale = tensor("latent_scale")?;
        if mean.is_empty() || components.len() % mean.len() != 0 {
            return Err(SpeechError::Tensor {
                name: "pca_components".into(),
                why: "pca components are not a multiple of the mean width".into(),
            });
        }
        if scale.len() != 1 {
            return Err(SpeechError::Tensor {
                name: "latent_scale".into(),
                why: "expected a scalar".into(),
            });
        }
        Ok(PcaState {
            latent_dim: components.len() / mean.len(),
            codec_dim: mean.len(),
            components,
            mean,
            latent_scale: scale[0],
        })
    }
}

/// `ae_encode`: reference waveform chunk -> continuous codec latents
/// `[frames, codec_dim]` -> PCA rows `[frames, latent_dim]`. The codec
/// latents arrive channel-major and transpose here.
pub fn ae_encode(codec: &FishS1Dac, pca: &PcaState, audio: &[f32]) -> Result<(Vec<f32>, usize)> {
    let (zq, frames) = codec.encode_zq(audio)?;
    let codec_dim = pca.codec_dim;
    let mut rows = vec![0.0f32; frames * codec_dim];
    for t in 0..frames {
        for d in 0..codec_dim {
            rows[t * codec_dim + d] = zq[d * frames + t];
        }
    }
    // (z - mean) @ components.T * scale
    subtract_mean_and_project(&mut rows, frames, pca);
    Ok((rows, frames))
}

/// `ae_decode`: PCA rows `[frames, latent_dim]` back to codec space
/// `[frames, codec_dim]` and through the codec decoder.
pub fn ae_decode(
    codec: &FishS1Dac,
    pca: &PcaState,
    latent: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    let rows = latent.to_vec();
    if rows.len() != frames * pca.latent_dim {
        return Err(SpeechError::Input {
            why: format!(
                "latent length {} does not match {} frames x latent {}",
                rows.len(),
                frames,
                pca.latent_dim
            ),
        });
    }
    // z = (z / scale) @ components + mean, written channel-major
    // ([codec_dim, frames]) for the codec.
    let mut zq = vec![0.0f32; frames * pca.codec_dim];
    for t in 0..frames {
        let row = &rows[t * pca.latent_dim..(t + 1) * pca.latent_dim];
        for (d, value) in row.iter().enumerate() {
            let scaled = value / pca.latent_scale;
            for c in 0..pca.codec_dim {
                zq[c * frames + t] += scaled * pca.components[d * pca.codec_dim + c];
            }
        }
    }
    for c in 0..pca.codec_dim {
        for t in 0..frames {
            zq[c * frames + t] += pca.mean[c];
        }
    }
    codec.decode_zq(&zq, frames)
}

fn subtract_mean_and_project(rows: &mut [f32], frames: usize, pca: &PcaState) {
    let codec_dim = pca.codec_dim;
    let latent_dim = pca.latent_dim;
    let mut centered = rows.to_vec();
    for frame in 0..frames {
        for (value, mean) in centered[frame * codec_dim..(frame + 1) * codec_dim]
            .iter_mut()
            .zip(&pca.mean)
        {
            *value -= mean;
        }
    }
    // rows = centered @ components.T (components [latent, codec_dim]);
    // ops::linear fuses the transpose for exactly this layout.
    let projected = ops::linear(
        &centered,
        &pca.components,
        None,
        frames,
        codec_dim,
        latent_dim,
    );
    for (value, p) in rows.iter_mut().zip(&projected) {
        *value = p * pca.latent_scale;
    }
}

/// `find_flattening_point`: first `[window, cols]` window of the latent
/// whose overall std is below `std_threshold` and whose mean is within
/// 0.1 of `target_value`, scanning the window-padded latent; the latent
/// row count when none matches.
pub fn find_flattening_point(
    data: &[f32],
    rows: usize,
    cols: usize,
    target_value: f32,
    window_size: usize,
    std_threshold: f32,
) -> usize {
    let padded_rows = rows + window_size;
    let window_elements = window_size * cols;
    let limit = padded_rows.saturating_sub(window_size);
    for i in 0..limit {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for t in 0..window_size {
            let row = i + t;
            if row >= rows {
                // The zero window appended by the reference padding.
                continue;
            }
            let base = row * cols;
            for value in &data[base..base + cols] {
                sum += value;
                sum_sq += value * value;
            }
        }
        let n = window_elements as f32;
        let mean = sum / n;
        let var = sum_sq / n - mean * mean;
        if var.sqrt() < std_threshold && (mean - target_value).abs() < 0.1 {
            return i;
        }
    }
    rows
}

/// `crop_audio_to_flattening_point`: keep the first
/// `flattening_point * 2048` samples (the reference hardcodes the
/// 44.1 kHz codec's frame length; a shorter waveform passes through).
pub fn crop_audio_to_flattening_point(audio: &[f32], flattening_point: usize) -> Vec<f32> {
    let keep = (flattening_point * 2048).min(audio.len());
    audio[..keep].to_vec()
}

/// `get_speaker_latent_and_mask`: chunked PCA latents over the
/// reference waveform plus the per-sample validity mask. `frame_factor`
/// is the reference's hardcoded 2048-sample latent stride
/// (`ae_downsample_factor`).
#[allow(clippy::too_many_arguments)]
pub fn get_speaker_latent_and_mask(
    codec: &FishS1Dac,
    pca: &PcaState,
    audio: &[f32],
    max_speaker_latent_length: usize,
    frame_factor: usize,
    audio_chunk_size: usize,
    pad_to_max: bool,
    divis_by_patch_size: usize,
) -> Result<(Vec<f32>, usize, Vec<bool>)> {
    let max_audio_len = max_speaker_latent_length * frame_factor;
    let audio = &audio[..audio.len().min(max_audio_len)];
    let mut latent_rows = 0usize;
    let latent_dim = pca.latent_dim;
    let mut latent = Vec::new();
    let mut start = 0usize;
    while start < audio.len() {
        let end = (start + audio_chunk_size).min(audio.len());
        let mut chunk = audio[start..end].to_vec();
        if chunk.len() < audio_chunk_size {
            chunk.resize(audio_chunk_size, 0.0);
        }
        let (rows, _frames) = ae_encode(codec, pca, &chunk)?;
        latent.extend_from_slice(&rows);
        latent_rows += rows.len() / latent_dim;
        start = end;
    }
    let actual_latent_length = audio.len() / frame_factor;
    let mut mask: Vec<bool> = (0..latent_rows).map(|j| j < actual_latent_length).collect();
    if pad_to_max && latent_rows < max_speaker_latent_length {
        let pad_t = max_speaker_latent_length - latent_rows;
        latent.resize((latent_rows + pad_t) * latent_dim, 0.0);
        mask.resize(latent_rows + pad_t, false);
        latent_rows += pad_t;
    } else if !pad_to_max {
        let keep = latent_rows.min(actual_latent_length);
        latent.truncate(keep * latent_dim);
        mask.truncate(keep);
        latent_rows = keep;
    }
    if divis_by_patch_size > 0 && latent_rows > 0 {
        let limit = (latent_rows / divis_by_patch_size) * divis_by_patch_size;
        latent.truncate(limit * latent_dim);
        mask.truncate(limit);
        latent_rows = limit;
    }
    Ok((latent, latent_rows, mask))
}
