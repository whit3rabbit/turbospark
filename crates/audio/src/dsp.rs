//! Small waveform operations and window functions.
//!
//! Port of the pieces of upstream audio.cpp's `waveform_ops.cpp` the other
//! modules share, plus the periodic Hann window the STFT and mel frontend
//! consume. All math runs in f64 internally and stores f32, so cascading
//! these operations does not accumulate f32 rounding faster than necessary.

/// Periodic Hann window of `size` points: `0.5 * (1 - cos(2 pi i / N))`.
///
/// Periodic (not symmetric): the last point is not the mirror of the first,
/// which is what overlap-add reconstruction needs. N = 1 yields `[1.0]`.
pub fn hann_window(size: usize) -> Vec<f32> {
    match size {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..size)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos())) as f32
            })
            .collect(),
    }
}

/// The maximum absolute sample; 0.0 for an empty slice.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
}

/// Root mean square; 0.0 for an empty slice. Accumulated in f64.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Multiplies every sample by a linear gain.
pub fn scale_by_gain(samples: &[f32], gain: f32) -> Vec<f32> {
    samples.iter().map(|&s| s * gain).collect()
}

/// Scales the waveform so its peak reaches `target_peak`.
///
/// A silent input (peak 0) is returned unchanged rather than scaled by
/// infinity; the caller decides what silence should become.
pub fn normalize_peak(samples: &[f32], target_peak: f32) -> Vec<f32> {
    let current = peak(samples);
    if current == 0.0 || !current.is_finite() {
        return samples.to_vec();
    }
    scale_by_gain(samples, target_peak / current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_and_sum() {
        let w = hann_window(8);
        assert_eq!(w[0], 0.0);
        // Periodic Hann wraps: point 4 is the peak (1.0 at N/2 for even N).
        assert!((w[4] - 1.0).abs() < 1e-6);
        assert_eq!(hann_window(1), vec![1.0]);
        assert!(hann_window(0).is_empty());
    }

    #[test]
    fn peak_rms_and_gain() {
        assert_eq!(peak(&[-0.5, 0.25, 2.0, -3.0]), 3.0);
        assert_eq!(peak(&[]), 0.0);
        // rms of [-1, 1] is 1; of [3, 4] is sqrt(12.5).
        assert!((rms(&[-1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((rms(&[3.0, 4.0]) - 3.53553).abs() < 1e-4);
        assert_eq!(scale_by_gain(&[1.0, -2.0], 0.5), vec![0.5, -1.0]);
    }

    #[test]
    fn normalize_peak_scales_to_target() {
        let out = normalize_peak(&[0.25, -0.5, 0.1], 1.0);
        assert!((out[1] + 1.0).abs() < 1e-6);
        assert_eq!(normalize_peak(&[0.0, 0.0], 1.0), vec![0.0, 0.0]);
    }
}
