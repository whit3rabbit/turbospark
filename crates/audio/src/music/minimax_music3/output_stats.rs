//! Cheap objective checks on a generated stereo waveform.
//!
//! These catch broken output (NaN, silence, clipping, a dead or duplicated
//! channel). They say nothing about whether the music is good.

/// Sample magnitude at or above which a sample counts as clipped.
const CLIP_LEVEL: f32 = 0.999;
/// Sample magnitude below which a sample counts as silent.
const SILENCE_LEVEL: f32 = 1e-4;

#[derive(Debug, Clone, PartialEq)]
pub struct OutputStats {
    /// Stereo frames analysed.
    pub frames: usize,
    pub all_finite: bool,
    /// Largest absolute sample.
    pub peak: f32,
    /// Root mean square over both channels.
    pub rms: f64,
    /// Fraction of samples at or above the clip level.
    pub clip_ratio: f64,
    /// Fraction of samples below the silence level.
    pub silence_ratio: f64,
    /// Mean sample value per channel; a large value is a DC offset.
    pub dc_left: f64,
    pub dc_right: f64,
    /// Pearson correlation of left and right; `None` when either channel
    /// is constant. Exactly 1.0 means the channels are duplicates.
    pub channel_correlation: Option<f64>,
}

/// Analyse interleaved stereo samples (`[frame * 2 + channel]`).
/// A trailing unpaired sample is ignored.
pub fn output_stats(interleaved: &[f32]) -> OutputStats {
    let frames = interleaved.len() / 2;
    let samples = &interleaved[..frames * 2];
    let mut stats = OutputStats {
        frames,
        all_finite: true,
        peak: 0.0,
        rms: 0.0,
        clip_ratio: 0.0,
        silence_ratio: 0.0,
        dc_left: 0.0,
        dc_right: 0.0,
        channel_correlation: None,
    };
    if frames == 0 {
        return stats;
    }
    let (mut sum_l, mut sum_r) = (0.0f64, 0.0f64);
    let (mut sq_l, mut sq_r, mut cross) = (0.0f64, 0.0f64, 0.0f64);
    let (mut clipped, mut silent) = (0usize, 0usize);
    for frame in samples.chunks_exact(2) {
        let (l, r) = (frame[0], frame[1]);
        if !l.is_finite() || !r.is_finite() {
            stats.all_finite = false;
            continue;
        }
        for v in [l, r] {
            stats.peak = stats.peak.max(v.abs());
            clipped += usize::from(v.abs() >= CLIP_LEVEL);
            silent += usize::from(v.abs() < SILENCE_LEVEL);
        }
        let (l, r) = (f64::from(l), f64::from(r));
        sum_l += l;
        sum_r += r;
        sq_l += l * l;
        sq_r += r * r;
        cross += l * r;
    }
    let n = frames as f64;
    stats.rms = ((sq_l + sq_r) / (2.0 * n)).sqrt();
    stats.clip_ratio = clipped as f64 / (2.0 * n);
    stats.silence_ratio = silent as f64 / (2.0 * n);
    stats.dc_left = sum_l / n;
    stats.dc_right = sum_r / n;
    let var_l = sq_l / n - stats.dc_left * stats.dc_left;
    let var_r = sq_r / n - stats.dc_right * stats.dc_right;
    // Variance below this is rounding noise on a constant channel.
    if var_l > 1e-12 && var_r > 1e-12 {
        let covariance = cross / n - stats.dc_left * stats.dc_right;
        stats.channel_correlation = Some(covariance / (var_l.sqrt() * var_r.sqrt()));
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, left_phase: f32, right_phase: f32) -> Vec<f32> {
        (0..frames)
            .flat_map(|t| {
                let x = t as f32 * 0.05;
                [0.5 * (x + left_phase).sin(), 0.5 * (x + right_phase).sin()]
            })
            .collect()
    }

    #[test]
    fn reports_level_and_identical_channels() {
        let stats = output_stats(&sine(2000, 0.0, 0.0));
        assert!(stats.all_finite);
        assert!((stats.peak - 0.5).abs() < 1e-3);
        // RMS of a 0.5-amplitude sine is 0.5 / sqrt(2).
        assert!((stats.rms - 0.5 / 2f64.sqrt()).abs() < 5e-3);
        assert_eq!(stats.clip_ratio, 0.0);
        let correlation = stats.channel_correlation.unwrap();
        assert!((correlation - 1.0).abs() < 1e-9, "{correlation}");
    }

    #[test]
    fn distinct_channels_are_not_fully_correlated() {
        let stats = output_stats(&sine(2000, 0.0, 1.5));
        assert!(stats.channel_correlation.unwrap() < 0.9);
    }

    #[test]
    fn silence_and_dead_channel_are_visible() {
        let stats = output_stats(&vec![0.0f32; 400]);
        assert_eq!(stats.silence_ratio, 1.0);
        assert_eq!(stats.rms, 0.0);
        assert_eq!(stats.channel_correlation, None);
    }

    #[test]
    fn clipping_dc_and_non_finite_are_counted() {
        let mut samples: Vec<f32> = [1.0f32, 0.3].repeat(100);
        let clipped = output_stats(&samples);
        assert!((clipped.clip_ratio - 0.5).abs() < 1e-12);
        assert!((clipped.dc_right - 0.3).abs() < 1e-6);
        samples[10] = f32::NAN;
        assert!(!output_stats(&samples).all_finite);
    }

    #[test]
    fn empty_and_odd_lengths_do_not_panic() {
        assert_eq!(output_stats(&[]).frames, 0);
        assert_eq!(output_stats(&[0.5, 0.5, 0.25]).frames, 1);
    }
}
