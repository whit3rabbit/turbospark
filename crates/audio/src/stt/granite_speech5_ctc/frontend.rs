//! Granite Speech 5.0 TurboCTC features from mlx-audio 0.5.7.

use turbospark_audio::dsp::hann_window;
use turbospark_audio::stft::{stft_with_modes, StftOptions, StftPaddingMode, StftWindowPlacement};

use crate::{Result, SpeechError};

const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const FRAME_STACKING: usize = 2;
const LOGMEL_FLOOR_DB: f32 = 8.0;

/// Converts mono 16 kHz PCM into `[frames / 2, 4 * mel_bins]` features.
pub(crate) fn compute_features(samples: &[f32], mel_bins: usize) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "audio must contain at least one 10 ms frame".into(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "audio samples must be finite".into(),
        });
    }
    if !matches!(mel_bins, 80 | 128) {
        return Err(SpeechError::Unsupported {
            why: format!("Granite Speech 5 supports 80 or 128 mel bins, got {mel_bins}"),
        });
    }

    let mel_frames = samples.len() / HOP_LENGTH;
    let num_frames = mel_frames.div_ceil(FRAME_STACKING) * FRAME_STACKING;
    if num_frames == 0 {
        return Err(SpeechError::Input {
            why: "audio must contain at least one 10 ms frame".into(),
        });
    }
    let needed_samples = (num_frames - 1)
        .checked_mul(HOP_LENGTH)
        .and_then(|n| n.checked_add(1))
        .filter(|&n| n <= 1 << 27)
        .ok_or_else(|| SpeechError::Input {
            why: "audio is too long for the bounded feature buffer".into(),
        })?;
    let mut padded_samples = samples.to_vec();
    padded_samples.resize(padded_samples.len().max(needed_samples), 0.0);

    let options = StftOptions {
        fft_size: N_FFT,
        hop: HOP_LENGTH,
        window: hann_window(WIN_LENGTH),
        center: true,
    };
    let spectra = stft_with_modes(
        &padded_samples,
        &options,
        StftPaddingMode::Reflect,
        StftWindowPlacement::Center,
    )?;
    if spectra.len() < num_frames {
        return Err(SpeechError::Input {
            why: format!(
                "centered STFT produced {} frames, expected at least {num_frames}",
                spectra.len()
            ),
        });
    }

    let filters = precise_htk_filterbank(mel_bins);
    let mel_values = mel_bins * num_frames;
    let mut logmel = vec![0.0f32; mel_values];
    for frame in 0..num_frames {
        for mel in 0..mel_bins {
            let filter = &filters[mel * (N_FFT / 2 + 1)..(mel + 1) * (N_FFT / 2 + 1)];
            let power = spectra[frame]
                .iter()
                .zip(filter)
                .map(|(bin, &weight)| (bin.re * bin.re + bin.im * bin.im) * weight)
                .sum::<f32>();
            logmel[frame * mel_bins + mel] = power.max(1e-10).log10();
        }
    }
    let peak = logmel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = peak - LOGMEL_FLOOR_DB;
    for value in &mut logmel {
        *value = value.max(floor) / 4.0 + 1.0;
    }

    let deltas = compute_deltas(&logmel, num_frames, mel_bins);
    let mut stacked = vec![0.0f32; (num_frames / FRAME_STACKING) * (4 * mel_bins)];
    let row_width = 4 * mel_bins;
    for row in 0..num_frames / FRAME_STACKING {
        for within_pair in 0..FRAME_STACKING {
            let frame = row * FRAME_STACKING + within_pair;
            let start = frame * mel_bins;
            let output = row * row_width + within_pair * 2 * mel_bins;
            stacked[output..output + mel_bins].copy_from_slice(&logmel[start..start + mel_bins]);
            stacked[output + mel_bins..output + 2 * mel_bins]
                .copy_from_slice(&deltas[start..start + mel_bins]);
        }
    }
    Ok((stacked, num_frames / FRAME_STACKING))
}

fn compute_deltas(features: &[f32], frames: usize, bins: usize) -> Vec<f32> {
    let mut deltas = vec![0.0f32; frames * bins];
    if frames == 0 {
        return deltas;
    }
    for frame in 0..frames {
        let before = frame.saturating_sub(1);
        let after = (frame + 1).min(frames - 1);
        let before_start = before * bins;
        let after_start = after * bins;
        let output_start = frame * bins;
        for bin in 0..bins {
            deltas[output_start + bin] =
                (features[after_start + bin] - features[before_start + bin]) * 0.5;
        }
    }
    deltas
}

/// Mirrors mlx_audio.dsp.mel_filters(..., mel_scale="htk", precise=True).
fn precise_htk_filterbank(mel_bins: usize) -> Vec<f32> {
    let frequency_bins = N_FFT / 2 + 1;
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(f64::from(SAMPLE_RATE / 2));
    let edges: Vec<f64> = (0..mel_bins + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f64 / (mel_bins + 1) as f64)
        .map(mel_to_hz)
        .collect();

    let mut filters = vec![0.0f32; mel_bins * frequency_bins];
    for mel in 0..mel_bins {
        let low = edges[mel];
        let center = edges[mel + 1];
        let high = edges[mel + 2];
        for bin in 0..frequency_bins {
            let frequency = (bin as f64) * f64::from(SAMPLE_RATE / 2) / (frequency_bins - 1) as f64;
            let down = (frequency - low) / (center - low);
            let up = (high - frequency) / (high - center);
            filters[mel * frequency_bins + bin] = down.min(up).max(0.0) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::{compute_deltas, compute_features, precise_htk_filterbank};

    #[test]
    fn features_match_the_pinned_mlx_reference_fixture() {
        let samples: Vec<f32> = (0..1280)
            .map(|i| ((i * 7 % 31) as i32 - 15) as f32 / 32.0)
            .collect();
        let golden: Vec<Vec<f32>> =
            serde_json::from_str(include_str!("../../../testdata/granite5_features.json"))
                .expect("MLX feature fixture is valid JSON");
        let (actual, rows) = compute_features(&samples, 80).expect("features compute");
        assert_eq!(rows, 4);
        assert_eq!(golden.len(), rows);
        let mut max_error = 0.0f32;
        for (actual_row, golden_row) in actual.chunks_exact(320).zip(golden) {
            assert_eq!(golden_row.len(), 320);
            for (actual, expected) in actual_row.iter().zip(golden_row) {
                max_error = max_error.max((actual - expected).abs());
            }
        }
        assert!(
            max_error <= 2e-4,
            "feature fixture max abs error {max_error}"
        );
    }

    #[test]
    fn deltas_replicate_both_edges() {
        let deltas = compute_deltas(&[1.0, 2.0, 4.0], 3, 1);
        assert_eq!(deltas, vec![0.5, 1.5, 1.0]);
    }

    #[test]
    fn precise_filterbank_has_triangular_rows() {
        let filters = precise_htk_filterbank(80);
        assert_eq!(filters.len(), 80 * 257);
        assert!(filters
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert!(filters.iter().any(|&value| value == 0.0));
        assert!(filters.iter().any(|&value| value > 0.0));
    }
}
