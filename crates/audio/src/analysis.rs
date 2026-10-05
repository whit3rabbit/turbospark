//! Waveform peaks and meter levels.

use std::path::Path;

use serde::Serialize;

use crate::decode::DecodedStream;
use crate::error::AudioError;

/// Display peaks for a whole file.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PeaksReport {
    /// Exactly `buckets` values in 0...1, loudest bucket = 1 (all zero for
    /// silence).
    pub peaks: Vec<f32>,
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Envelope resolution while streaming: 100 blocks per second keeps a
/// 30-minute clip at 180k floats regardless of how many buckets the view
/// asks for afterwards.
const ENVELOPE_BLOCKS_PER_SECOND: u32 = 100;

/// Upper bound on requested buckets; a waveform wider than this is not a
/// waveform anyone can read.
pub const MAX_BUCKETS: usize = 8_192;

/// Streams `path` once and returns `buckets` normalized peaks.
///
/// MAX magnitude per bucket rather than mean: a waveform is read for its
/// transients, and a mean flattens a plosive into the noise floor.
pub fn peaks(path: &Path, buckets: usize) -> Result<PeaksReport, AudioError> {
    if buckets == 0 || buckets > MAX_BUCKETS {
        return Err(AudioError::InvalidOption(format!(
            "buckets must be 1..={MAX_BUCKETS}, got {buckets}"
        )));
    }
    let mut stream = DecodedStream::open(path)?;
    let sample_rate = stream.sample_rate();
    let block = (sample_rate / ENVELOPE_BLOCKS_PER_SECOND).max(1) as usize;
    let mut envelope: Vec<f32> = Vec::new();
    let mut current = 0.0f32;
    let mut in_block = 0usize;
    let mut frames = 0u64;
    while let Some(chunk) = stream.next_chunk()? {
        let length = chunk.first().map_or(0, Vec::len);
        for i in 0..length {
            let magnitude = chunk.iter().map(|c| c[i].abs()).fold(0.0f32, f32::max);
            current = current.max(magnitude);
            in_block += 1;
            if in_block == block {
                envelope.push(current);
                current = 0.0;
                in_block = 0;
            }
        }
        frames += length as u64;
    }
    if in_block > 0 {
        envelope.push(current);
    }
    Ok(PeaksReport {
        peaks: normalize(&resample_max(&envelope, buckets)),
        duration_seconds: frames as f64 / f64::from(sample_rate.max(1)),
        sample_rate,
        channels: stream.channels() as u32,
    })
}

/// Max-reduces `values` into exactly `count` buckets. Fewer values than
/// buckets repeats values, so bar count never depends on clip length.
pub fn resample_max(values: &[f32], count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if values.is_empty() {
        return vec![0.0; count];
    }
    let ratio = values.len() as f64 / count as f64;
    (0..count)
        .map(|i| {
            let start = (i as f64 * ratio) as usize;
            let end = (((i + 1) as f64 * ratio) as usize)
                .min(values.len())
                .max(start + 1);
            values[start..end].iter().copied().fold(0.0f32, f32::max)
        })
        .collect()
}

/// Scales so the largest value is 1. All-zero input stays all zero instead
/// of dividing into NaN.
pub fn normalize(values: &[f32]) -> Vec<f32> {
    let peak = values.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v / peak).min(1.0)).collect()
}

/// Root mean square of `samples`; 0 for an empty slice.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Maps an RMS value to a 0...1 meter position over -50...0 dBFS.
///
/// Linear RMS of speech sits around 0.01-0.1, which draws as a flat line;
/// -50 dBFS to full scale is the range a voice actually moves through.
pub fn display_level(rms: f32) -> f32 {
    if rms.is_nan() || rms <= 0.0 {
        return 0.0;
    }
    let decibels = 20.0 * rms.log10();
    ((decibels + 50.0) / 50.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_max_keeps_transients() {
        let mut values = vec![0.1f32; 1_000];
        values[503] = 0.9;
        let buckets = resample_max(&values, 10);
        assert_eq!(buckets.len(), 10);
        assert_eq!(buckets[5], 0.9);
        assert_eq!(buckets[4], 0.1);
    }

    #[test]
    fn resample_max_pads_short_input_to_the_bucket_count() {
        assert_eq!(resample_max(&[0.2, 0.4], 4), vec![0.2, 0.2, 0.4, 0.4]);
        assert_eq!(resample_max(&[], 3), vec![0.0; 3]);
    }

    #[test]
    fn normalize_handles_silence() {
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert_eq!(normalize(&[0.25, 0.5]), vec![0.5, 1.0]);
    }

    #[test]
    fn display_level_spans_minus_fifty_to_zero_dbfs() {
        assert_eq!(display_level(0.0), 0.0);
        assert_eq!(display_level(f32::NAN), 0.0);
        assert!((display_level(1.0) - 1.0).abs() < 1e-6);
        // -25 dBFS is half way.
        assert!((display_level(10f32.powf(-25.0 / 20.0)) - 0.5).abs() < 1e-4);
        assert_eq!(display_level(1e-4), 0.0);
    }

    #[test]
    fn rms_of_a_full_scale_square_is_one() {
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert_eq!(rms(&[]), 0.0);
    }
}
