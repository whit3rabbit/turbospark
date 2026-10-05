//! The waveform value every other module consumes and produces.
//!
//! Mirrors upstream audio.cpp's `WavData`: interleaved f32 samples plus a
//! sample rate and channel count. Interleaved is the storage order because
//! it is what WAV bytes, the resampler, and the channel converters all speak;
//! planar forms exist only inside [`crate::conversion`], where they are
//! produced and consumed by named functions.

use crate::error::{check_channel_count, check_sample_rate, AudioError};

/// Interleaved f32 audio: `samples[frame * channels + channel]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Waveform {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Waveform {
    /// Builds a waveform, validating the layout and that the interleaved
    /// length is a whole number of frames.
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Self, AudioError> {
        check_sample_rate(sample_rate)?;
        let channels = channels as usize;
        check_channel_count(channels)?;
        if samples.len() % channels != 0 {
            return Err(AudioError::ShapeMismatch {
                what: "interleaved samples vs channel count",
                expected: samples.len() - samples.len() % channels,
                actual: samples.len(),
            });
        }
        Ok(Self {
            sample_rate,
            channels: channels as u16,
            samples,
        })
    }

    /// A silent waveform of `frames` frames and `channels` channels.
    pub fn silence(sample_rate: u32, channels: u16, frames: usize) -> Result<Self, AudioError> {
        check_sample_rate(sample_rate)?;
        check_channel_count(channels as usize)?;
        let samples = frames
            .checked_mul(channels as usize)
            .filter(|&count| count <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or(AudioError::BufferTooLarge {
                what: "silent waveform",
                samples: frames,
            })?;
        Self::new(sample_rate, channels, vec![0.0; samples])
    }

    /// Mono convenience constructor.
    pub fn mono(sample_rate: u32, samples: Vec<f32>) -> Result<Self, AudioError> {
        Self::new(sample_rate, 1, samples)
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds, computed in f64 so long files do not lose
    /// sub-second precision.
    pub fn duration_seconds(&self) -> f64 {
        self.frame_count() as f64 / f64::from(self.sample_rate)
    }

    /// The frame range `[start, end)` as a new waveform. `end` is clamped
    /// to the frame count; a start past the end yields an empty waveform
    /// with the same rate and channel count.
    pub fn slice_frames(&self, start: usize, end: usize) -> Result<Self, AudioError> {
        let frames = self.frame_count();
        let start = start.min(frames);
        let end = end.min(frames);
        if end < start {
            return Err(AudioError::ShapeMismatch {
                what: "frame range start vs end",
                expected: start,
                actual: end,
            });
        }
        let width = self.channels as usize;
        Ok(Self {
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples: self.samples[start * width..end * width].to_vec(),
        })
    }

    /// Appends another waveform with the same rate and channel count.
    pub fn extend(&mut self, other: &Waveform) -> Result<(), AudioError> {
        if other.sample_rate != self.sample_rate {
            return Err(AudioError::ShapeMismatch {
                what: "sample rate",
                expected: self.sample_rate as usize,
                actual: other.sample_rate as usize,
            });
        }
        if other.channels != self.channels {
            return Err(AudioError::ShapeMismatch {
                what: "channel count",
                expected: usize::from(self.channels),
                actual: usize::from(other.channels),
            });
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_non_multiple_of_channels() {
        let err = Waveform::new(16_000, 2, vec![0.0; 5]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "shape mismatch for interleaved samples vs channel count: expected 4, got 5"
        );
    }

    #[test]
    fn new_rejects_zero_rate_and_channels() {
        assert!(Waveform::new(0, 1, vec![0.0]).is_err());
        assert!(Waveform::new(16_000, 0, vec![0.0]).is_err());
    }

    #[test]
    fn frame_count_slice_and_duration() {
        let w = Waveform::new(8_000, 2, (0..16).map(|i| i as f32).collect()).unwrap();
        assert_eq!(w.frame_count(), 8);
        assert_eq!(w.duration_seconds(), 0.001);
        let mid = w.slice_frames(2, 4).unwrap();
        assert_eq!(mid.samples, vec![4.0, 5.0, 6.0, 7.0]);
        // A start past the end clamps to empty, same format.
        let empty = w.slice_frames(100, 200).unwrap();
        assert_eq!(empty.frame_count(), 0);
        assert_eq!(empty.sample_rate, 8_000);
        assert_eq!(empty.channels, 2);
    }

    #[test]
    fn extend_checks_format() {
        let mut a = Waveform::mono(16_000, vec![1.0]).unwrap();
        let b = Waveform::mono(8_000, vec![2.0]).unwrap();
        let err = a.extend(&b).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
        let c = Waveform::mono(16_000, vec![2.0, 3.0]).unwrap();
        a.extend(&c).unwrap();
        assert_eq!(a.samples, vec![1.0, 2.0, 3.0]);
    }
}

#[cfg(test)]
mod allocation_regression {
    use super::*;

    #[test]
    fn silence_validates_before_multiplication_or_allocation() {
        assert!(matches!(
            Waveform::silence(16_000, 2, usize::MAX),
            Err(AudioError::BufferTooLarge { .. })
        ));
        assert!(matches!(
            Waveform::silence(16_000, 0, usize::MAX),
            Err(AudioError::InvalidWavLayout { .. })
        ));
    }
}
