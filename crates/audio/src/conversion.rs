//! Channel-layout moves over interleaved f32 audio, plus the mono-resample
//! pipelines model frontends call.
//!
//! Port of upstream audio.cpp's `conversion.cpp`, which is float32-only: it
//! deliberately does not touch integer PCM, leaving that to the WAV reader.
//! Every function validates the interleaved length against the channel count
//! and reports disagreements with both values. OpenMP parallelism from the
//! upstream is not ported; these loops are memory-bound at app scale.

use crate::error::AudioError;
use crate::resample::{resample_mono_linear, resample_mono_sinc_hann, SincHannOptions};
use crate::waveform::Waveform;

/// Accumulation precision for the mixdown average. `Float64` adds the
/// channel contributions in double precision before a single downcast,
/// which keeps wide mixes (8+ channels) from losing low bits; `Float32`
/// matches the upstream default exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonoMixAccumulation {
    #[default]
    Float32,
    Float64,
}

/// Mixes interleaved audio down to mono by averaging channels per frame.
///
/// A mono input is returned unchanged (same values, new buffer), matching
/// the upstream short-circuit.
pub fn mixdown_interleaved_to_mono_average(
    samples: &[f32],
    channel_count: usize,
    accumulation: MonoMixAccumulation,
) -> Result<Vec<f32>, AudioError> {
    if channel_count == 0 {
        return Err(AudioError::InvalidParameter {
            name: "channel_count".to_string(),
            value: "0".to_string(),
            why: "channel count must be positive".to_string(),
        });
    }
    if samples.len() % channel_count != 0 {
        return Err(interleaved_shape_mismatch(samples.len(), channel_count));
    }
    if channel_count == 1 {
        return Ok(samples.to_vec());
    }
    let frames = samples.len() / channel_count;
    let mut out = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * channel_count;
        let avg = match accumulation {
            MonoMixAccumulation::Float32 => {
                let mut sum = 0.0f32;
                for &s in &samples[base..base + channel_count] {
                    sum += s;
                }
                sum / channel_count as f32
            }
            MonoMixAccumulation::Float64 => {
                let mut sum = 0.0f64;
                for &s in &samples[base..base + channel_count] {
                    sum += f64::from(s);
                }
                (sum / channel_count as f64) as f32
            }
        };
        out.push(avg);
    }
    Ok(out)
}

/// Replicates a mono stream into `channel_count` identical channels.
pub fn duplicate_mono_to_interleaved(
    mono: &[f32],
    channel_count: usize,
) -> Result<Vec<f32>, AudioError> {
    if channel_count == 0 {
        return Err(AudioError::InvalidParameter {
            name: "channel_count".to_string(),
            value: "0".to_string(),
            why: "channel count must be positive".to_string(),
        });
    }
    let mut out = Vec::with_capacity(mono.len() * channel_count);
    for &s in mono {
        for _ in 0..channel_count {
            out.push(s);
        }
    }
    Ok(out)
}

/// Splits interleaved audio into planar buffers, one per channel:
/// `planar[channel][frame]`.
pub fn deinterleave_to_planar(
    samples: &[f32],
    channel_count: usize,
) -> Result<Vec<Vec<f32>>, AudioError> {
    if channel_count == 0 {
        return Err(AudioError::InvalidParameter {
            name: "channel_count".to_string(),
            value: "0".to_string(),
            why: "channel count must be positive".to_string(),
        });
    }
    if samples.len() % channel_count != 0 {
        return Err(interleaved_shape_mismatch(samples.len(), channel_count));
    }
    let frames = samples.len() / channel_count;
    let mut planar = vec![Vec::with_capacity(frames); channel_count];
    for (i, &s) in samples.iter().enumerate() {
        planar[i % channel_count].push(s);
    }
    Ok(planar)
}

/// Joins planar buffers back into interleaved audio. All planes must carry
/// the same frame count.
pub fn interleave_planar(planar: &[Vec<f32>]) -> Result<Vec<f32>, AudioError> {
    if planar.is_empty() {
        return Err(AudioError::InvalidParameter {
            name: "planar".to_string(),
            value: "empty".to_string(),
            why: "at least one channel plane is required".to_string(),
        });
    }
    let frames = planar[0].len();
    for (ch, plane) in planar.iter().enumerate() {
        if plane.len() != frames {
            return Err(AudioError::InvalidParameter {
                name: "planar".to_string(),
                value: format!("channel {ch} has {} frames", plane.len()),
                why: format!("all planes must match channel 0's {frames} frames"),
            });
        }
    }
    let channels = planar.len();
    let mut out = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for plane in planar {
            out.push(plane[frame]);
        }
    }
    Ok(out)
}

/// Pulls one channel out of an interleaved stream by index.
pub fn extract_interleaved_channel(
    samples: &[f32],
    channel_count: usize,
    index: usize,
) -> Result<Vec<f32>, AudioError> {
    if channel_count == 0 {
        return Err(AudioError::InvalidParameter {
            name: "channel_count".to_string(),
            value: "0".to_string(),
            why: "channel count must be positive".to_string(),
        });
    }
    if index >= channel_count {
        return Err(AudioError::InvalidParameter {
            name: "index".to_string(),
            value: index.to_string(),
            why: format!("channel index must be below {channel_count}"),
        });
    }
    if samples.len() % channel_count != 0 {
        return Err(interleaved_shape_mismatch(samples.len(), channel_count));
    }
    Ok(samples
        .iter()
        .skip(index)
        .step_by(channel_count)
        .copied()
        .collect())
}

/// Resampling strategy for the mono convenience pipelines. `SincHann` is the
/// torchaudio-compatible polyphase filter; `Linear` is the upstream's
/// guaranteed fallback.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MonoResampleStrategy {
    #[default]
    Linear,
    SincHann(SincHannOptions),
}

/// Mixdown to mono followed by resampling to `target_rate`, the pipeline the
/// model frontends need (upstream `convert_wav_to_mono_linear_resampled` and
/// its sinc-Hann sibling).
pub fn to_mono_resampled(
    wave: &Waveform,
    target_rate: u32,
    strategy: &MonoResampleStrategy,
) -> Result<Vec<f32>, AudioError> {
    let mono = mixdown_interleaved_to_mono_average(
        &wave.samples,
        wave.channels as usize,
        MonoMixAccumulation::Float64,
    )?;
    if wave.sample_rate == target_rate {
        return Ok(mono);
    }
    match strategy {
        MonoResampleStrategy::Linear => resample_mono_linear(&mono, wave.sample_rate, target_rate),
        MonoResampleStrategy::SincHann(options) => {
            resample_mono_sinc_hann(&mono, wave.sample_rate, target_rate, options)
        }
    }
}

fn interleaved_shape_mismatch(len: usize, channels: usize) -> AudioError {
    AudioError::ShapeMismatch {
        what: "interleaved samples vs channel count",
        expected: len - len % channels,
        actual: len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixdown_averages_channels() {
        let out = mixdown_interleaved_to_mono_average(
            &[1.0, 3.0, 5.0, 7.0],
            2,
            MonoMixAccumulation::Float32,
        )
        .unwrap();
        assert_eq!(out, vec![2.0, 6.0]);
    }

    #[test]
    fn mono_mixdown_short_circuits() {
        let input = vec![0.1, 0.2, 0.3];
        let out =
            mixdown_interleaved_to_mono_average(&input, 1, MonoMixAccumulation::Float32).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn mixdown_rejects_partial_frame() {
        let err = mixdown_interleaved_to_mono_average(&[0.0; 5], 2, MonoMixAccumulation::Float32)
            .unwrap_err();
        assert!(err.to_string().contains("expected 4, got 5"), "{err}");
    }

    #[test]
    fn planar_roundtrip() {
        let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let planar = deinterleave_to_planar(&interleaved, 3).unwrap();
        assert_eq!(planar, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        let back = interleave_planar(&planar).unwrap();
        assert_eq!(back, interleaved);
    }

    #[test]
    fn interleave_rejects_unequal_planes() {
        let err = interleave_planar(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert!(err.to_string().contains("channel 1 has 1 frames"), "{err}");
    }

    #[test]
    fn extract_channel_picks_index() {
        let out = extract_interleaved_channel(&[1.0, 2.0, 3.0, 4.0], 2, 1).unwrap();
        assert_eq!(out, vec![2.0, 4.0]);
        let err = extract_interleaved_channel(&[1.0, 2.0], 2, 2).unwrap_err();
        assert!(err.to_string().contains("must be below 2"), "{err}");
    }

    #[test]
    fn to_mono_resampled_passthrough_at_same_rate() {
        let wave = Waveform::new(16_000, 2, vec![1.0, 3.0, -1.0, -3.0]).unwrap();
        let out = to_mono_resampled(&wave, 16_000, &MonoResampleStrategy::Linear).unwrap();
        assert_eq!(out, vec![2.0, -2.0]);
    }
}
