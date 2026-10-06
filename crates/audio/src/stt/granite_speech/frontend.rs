//! Granite Speech 1B log-mel frontend.
//!
//! Reference: `mlx_audio/stt/models/granite_speech/granite_speech.py`
//! (`Model._extract_features`) plus `mlx_audio/dsp.py` (`hanning`,
//! `stft`, `mel_filters`) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//!
//! Pipeline: centered 512-point STFT (reflect padding, the 400-sample
//! periodic Hann window zero-padded and centered inside each frame), power
//! spectrum, an 80-band HTK mel filterbank computed in f32 exactly like the
//! reference default (`precise=False`), `log10` with a 1e-10 floor, a
//! single global peak-relative floor at `max - 8` with `/4 + 1`
//! normalization, then pair stacking: an odd trailing frame is dropped and
//! each output row concatenates two consecutive 80-band frames into the
//! 160-dim encoder input.
//!
//! This differs from the crate's whisper frontend in three load-bearing
//! ways: the filterbank is HTK-scaled without Slaney area normalization
//! over a zero-padded 512-point transform, the global peak is taken over
//! the whole spectrogram BEFORE any frame is dropped (the odd-frame drop
//! exists only to make pairing exact), and rows are stacked in pairs.

use crate::dsp::hann_window;
use crate::ops;
use crate::stft::{stft_with_modes, StftOptions, StftPaddingMode, StftWindowPlacement};
use crate::{Result, SpeechError};

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const N_FFT: usize = 512;
pub(crate) const HOP_LENGTH: usize = 160;
pub(crate) const WIN_LENGTH: usize = 400;
pub(crate) const MEL_BINS: usize = 80;
/// Two 80-band frames stacked per encoder row.
pub(crate) const PAIR_WIDTH: usize = 2 * MEL_BINS;
/// The reference clamps `mel_spec` at 1e-10 before `log10`.
const LOG_FLOOR: f32 = 1e-10;
/// The reference peak-relative floor (`mx.max(logmel) - 8.0`).
const PEAK_MARGIN: f32 = 8.0;

/// The reference 80-band HTK mel filterbank in f32, built in the exact
/// operation order of `mlx_audio.dsp.mel_filters` with its default
/// `precise=False`: a f32 `linspace` grid, f32 mel/hz round trip, and f32
/// triangle slopes.
fn htk_filterbank_f32() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let nyquist = (SAMPLE_RATE / 2) as f32;
    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| nyquist * i as f32 / (n_freqs - 1) as f32)
        .collect();
    let m_min = 0.0f32;
    let m_max = (2595.0f64 * (1.0 + 8000.0f64 / 700.0).log10()) as f32;
    let points = MEL_BINS + 2;
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
    let f_pts: Vec<f32> = (0..points)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f32 / (points - 1) as f32))
        .collect();

    let mut filterbank = vec![0.0f32; MEL_BINS * n_freqs];
    for mel in 0..MEL_BINS {
        let low = f_pts[mel];
        let center = f_pts[mel + 1];
        let high = f_pts[mel + 2];
        let down_diff = center - low;
        let up_diff = high - center;
        for (bin, frequency) in all_freqs.iter().enumerate() {
            let down = (*frequency - low) / down_diff;
            let up = (high - frequency) / up_diff;
            filterbank[mel * n_freqs + bin] = down.min(up).max(0.0);
        }
    }
    filterbank
}

/// The filterbank transposed to `[n_freqs, n_mels]` for the power matmul.
fn htk_filterbank_transposed() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let filterbank = htk_filterbank_f32();
    let mut transposed = vec![0.0f32; n_freqs * MEL_BINS];
    for mel in 0..MEL_BINS {
        for bin in 0..n_freqs {
            transposed[bin * MEL_BINS + mel] = filterbank[mel * n_freqs + bin];
        }
    }
    transposed
}

/// Converts mono 16 kHz PCM into the `[rows, 160]` row-major encoder input.
/// Returns the plane plus its row count (the paired frame count).
pub(crate) fn features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "granite speech audio must not be empty".into(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "granite speech audio samples must be finite".into(),
        });
    }
    let options = StftOptions {
        fft_size: N_FFT,
        hop: HOP_LENGTH,
        window: hann_window(WIN_LENGTH),
        center: true,
    };
    let spectra = stft_with_modes(
        samples,
        &options,
        StftPaddingMode::Reflect,
        StftWindowPlacement::Center,
    )?;
    if spectra.is_empty() {
        return Err(SpeechError::Input {
            why: "the centered STFT produced no frames".into(),
        });
    }
    let n_freqs = N_FFT / 2 + 1;
    let frames = spectra.len();
    let power: Vec<f32> = spectra
        .iter()
        .flat_map(|frame| frame.iter().map(|bin| bin.re * bin.re + bin.im * bin.im))
        .collect();
    let mel = ops::matmul(
        &power,
        &htk_filterbank_transposed(),
        frames,
        n_freqs,
        MEL_BINS,
    );

    // log10(clamp) over every entry, then one global peak, then the
    // peak-relative floor and /4 + 1 normalization. The odd trailing frame
    // is dropped only after normalization, exactly as the reference does.
    let mut peak = f32::NEG_INFINITY;
    for value in &mel {
        peak = peak.max(value.max(LOG_FLOOR).log10());
    }
    let floor = peak - PEAK_MARGIN;
    let kept = frames - (frames % 2);
    if kept == 0 {
        return Err(SpeechError::Input {
            why: "the log-mel frontend produced no retained frames".into(),
        });
    }
    let mut normalized = vec![0.0f32; mel.len()];
    for (out, value) in normalized.iter_mut().zip(&mel) {
        *out = value.max(LOG_FLOOR).log10().max(floor) / 4.0 + 1.0;
    }

    let rows = kept / 2;
    let mut stacked = vec![0.0f32; rows * PAIR_WIDTH];
    for row in 0..rows {
        for within_pair in 0..2 {
            let frame = row * 2 + within_pair;
            let source = frame * MEL_BINS;
            let target = row * PAIR_WIDTH + within_pair * MEL_BINS;
            stacked[target..target + MEL_BINS]
                .copy_from_slice(&normalized[source..source + MEL_BINS]);
        }
    }
    Ok((stacked, rows))
}

#[cfg(test)]
mod tests {
    use super::{features, htk_filterbank_f32};

    #[test]
    fn filterbank_rows_are_triangles_anchored_from_silence() {
        let filterbank = htk_filterbank_f32();
        assert_eq!(filterbank.len(), 80 * 257);
        let first = &filterbank[..257];
        // HTK mel(0) = 0, so the first band's triangle starts at bin 0.
        assert_eq!(first[0], 0.0);
        assert!(first.iter().any(|&v| v > 0.0));
        // Every row is non-negative and peaks near but not necessarily at
        // 1.0: the apex lands on a bin only for the dense high-frequency
        // bands, while the coarse low bands peak well below 1.
        let mut widest_peak = 0.0f32;
        for band in 0..80 {
            let row = &filterbank[band * 257..(band + 1) * 257];
            assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
            let peak = row.iter().cloned().fold(0.0f32, f32::max);
            assert!(peak > 0.3, "band {band} peak {peak}");
            assert!(peak <= 1.0 + 1e-5, "band {band} peak {peak}");
            widest_peak = widest_peak.max(peak);
        }
        assert!(widest_peak >= 0.95, "widest band peak {widest_peak}");
        // The highest band covers only the top of the spectrum.
        let last = &filterbank[79 * 257..80 * 257];
        assert_eq!(last[0], 0.0);
        assert!(last.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn frontend_pairs_frames_and_normalizes_into_a_narrow_band() {
        // 16 kHz of a deterministic chirp-ish signal: 16000 samples.
        let samples: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let (plane, rows) = features(&samples).unwrap();
        // 101 centered STFT frames for 16000 samples; the odd trailing
        // frame drops before pairing, leaving 50 rows.
        assert_eq!(1 + samples.len() / 160, 101);
        assert_eq!(rows, 50);
        assert_eq!(plane.len(), rows * 160);
        // The 8 dB peak-relative floor spans exactly 2.0 after the /4
        // normalization: max - min is the retained dynamic range, capped at
        // the floor and extended nowhere above the peak.
        let max = plane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = plane.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max - min <= 2.0 + 1e-4, "spread {}", max - min);
        assert!(max - min > 1.0, "degenerate spread {}", max - min);
        assert!(plane.iter().all(|v| v.is_finite()));
    }
}
