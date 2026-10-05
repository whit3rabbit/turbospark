//! Whisper's log-mel frontend, on top of the generic DSP modules.
//!
//! Parameterization is the reference's own: a 400-point FFT (400-sample
//! Hann window, no zero padding -- the FFT runs Bluestein to reach the
//! non-power-of-two size), 160 hop, 80 or 128 Slaney mel bands over
//! [0, 8 kHz], power spectrum, `log10` with a 1e-10 floor, a peak-relative
//! clamp at `max - 8`, and the `(x + 4) / 4` normalization. Everything is
//! computed per 30-second window, matching the reference's per-call
//! normalization.
//!
//! The golden fixture in the tests was produced with numpy's independent
//! FFT and OpenAI's exported Slaney filterbank, so a packing,
//! split-step, or filterbank bug in this crate cannot reproduce it.

use crate::error::AudioError;
use crate::mel::{mel_filterbank, mel_spectrogram, MelScale, MelSpectrogramOptions};
use crate::stft::StftOptions;

/// Sample rate every whisper model consumes.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// FFT size per frame: 400 samples, matching the reference exactly.
pub const WHISPER_N_FFT: usize = 400;
/// Hop between frames in samples.
pub const WHISPER_HOP: usize = 160;
/// One 30-second window of PCM at [`WHISPER_SAMPLE_RATE`].
pub const WHISPER_WINDOW_SAMPLES: usize = 480_000;
/// Mel frames per 30-second window: `1 + 480000/160` minus the frame the
/// reference drops off the tail.
pub const WHISPER_MEL_FRAMES: usize = 3000;
/// Highest frequency a mel band covers.
pub const WHISPER_FMAX: f32 = 8_000.0;

/// The Slaney (librosa) mel filterbank whisper uses, at 80 or 128 bands.
pub fn whisper_mel_filterbank(n_mels: usize) -> Result<crate::mel::MelFilterbank, AudioError> {
    if n_mels != 80 && n_mels != 128 {
        return Err(AudioError::InvalidParameter {
            name: "n_mels".to_string(),
            value: n_mels.to_string(),
            why: "whisper models use 80 or 128 mel bands".to_string(),
        });
    }
    mel_filterbank(
        n_mels,
        WHISPER_N_FFT,
        WHISPER_SAMPLE_RATE,
        0.0,
        Some(WHISPER_FMAX),
        MelScale::Slaney,
    )
}

/// Log-mel spectrogram for arbitrary-length PCM at 16 kHz, one row per
/// frame: the log10/peak-clamp/normalize pipeline over the power mel
/// spectrogram. For the fixed 30-second windows the encoder consumes, use
/// [`whisper_log_mel_window`].
pub fn whisper_log_mel(samples: &[f32], n_mels: usize) -> Result<Vec<Vec<f32>>, AudioError> {
    let options = MelSpectrogramOptions {
        stft: StftOptions {
            fft_size: WHISPER_N_FFT,
            hop: WHISPER_HOP,
            window: crate::dsp::hann_window(WHISPER_N_FFT),
            center: true,
        },
        num_mels: n_mels,
        sample_rate: WHISPER_SAMPLE_RATE,
        fmin: 0.0,
        fmax: Some(WHISPER_FMAX),
        scale: MelScale::Slaney,
        power: 2.0,
    };
    let mut mel = mel_spectrogram(samples, &options)?;
    // OpenAI drops the final centered STFT frame before taking the global
    // maximum. Including it can alter the clamp for every retained frame.
    mel.pop();
    normalize_log_mel(mel)
}

/// The encoder input for one 30-second window: exactly
/// [`WHISPER_MEL_FRAMES`] rows of `n_mels` values, band-major per frame.
///
/// Requires exactly [`WHISPER_WINDOW_SAMPLES`] samples at 16 kHz; the
/// reference drops the one extra centered frame off the tail
/// (`log_spec[:, :-1]`).
pub fn whisper_log_mel_window(samples: &[f32], n_mels: usize) -> Result<Vec<Vec<f32>>, AudioError> {
    if samples.len() != WHISPER_WINDOW_SAMPLES {
        return Err(AudioError::ShapeMismatch {
            what: "whisper window samples",
            expected: WHISPER_WINDOW_SAMPLES,
            actual: samples.len(),
        });
    }
    if samples.len() < WHISPER_SAMPLE_RATE as usize {
        return Err(AudioError::InvalidParameter {
            name: "samples".to_string(),
            value: samples.len().to_string(),
            why: "below the minimum 1 second".to_string(),
        });
    }
    let mut frames = whisper_log_mel(samples, n_mels)?;
    // 1 + 480000/160 = 3001 centered frames; the reference drops the last.
    if frames.len() == WHISPER_MEL_FRAMES + 1 {
        frames.truncate(WHISPER_MEL_FRAMES);
    }
    if frames.len() != WHISPER_MEL_FRAMES {
        return Err(AudioError::ShapeMismatch {
            what: "whisper mel frames",
            expected: WHISPER_MEL_FRAMES,
            actual: frames.len(),
        });
    }
    Ok(frames)
}

/// The reference's post-processing: `log10(max(x, 1e-10))`, clamp at the
/// global peak minus 8, then `(x + 4) / 4`.
fn normalize_log_mel(mel: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, AudioError> {
    let mut peak = f32::NEG_INFINITY;
    for frame in &mel {
        for &x in frame {
            let logged = x.max(1e-10).log10();
            peak = peak.max(logged);
        }
    }
    if !peak.is_finite() {
        return Err(AudioError::InvalidParameter {
            name: "mel".to_string(),
            value: "empty".to_string(),
            why: "spectrogram carries no frames".to_string(),
        });
    }
    let clamp_at = peak - 8.0;
    Ok(mel
        .into_iter()
        .map(|frame| {
            frame
                .into_iter()
                .map(|x| ((x.max(1e-10).log10()).max(clamp_at) + 4.0) / 4.0)
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed synthetic input both this test and the numpy reference
    /// script regenerate: one second at 16 kHz.
    fn synthetic_second() -> Vec<f32> {
        let n = WHISPER_SAMPLE_RATE as usize;
        (0..n)
            .map(|i| {
                let t = i as f32;
                0.5 * (t * 0.037).sin() + 0.25 * (t * 0.11).cos() + 0.1 * (t * 0.0017).sin()
            })
            .collect()
    }

    /// Recorded with NumPy rfft and OpenAI Whisper v20250625's official
    /// mel_filters.npz (SHA256 pinned in mel.rs), then the reference's
    /// frame-drop/log/clamp/normalize pipeline. The old locally generated
    /// filterbank fixture repeated the incorrect mel-width normalization.
    const GOLDEN_F0: [f32; 8] = [
        1.407_503_2,
        1.295_149_7,
        1.252_713_6,
        1.269_207_7,
        1.179_239_4,
        1.038_145_8,
        1.203_587_3,
        1.218_232_6,
    ];
    const GOLDEN_F50: [f32; 8] = [
        1.246_597_3,
        1.424_199_1,
        1.396_429_3,
        1.239_487_9,
        0.837_558_2,
        1.092_791_6,
        1.249_224_3,
        1.254_393_7,
    ];
    const GOLDEN_F0_B79: f32 = -0.414_880_6;

    #[test]
    fn matches_the_numpy_golden_fixture() {
        let got = whisper_log_mel(&synthetic_second(), 80).unwrap();
        assert_eq!(got.len(), 100);
        for (band, want) in GOLDEN_F0.iter().enumerate() {
            assert!(
                (got[0][band] - want).abs() < 5e-4,
                "frame 0 band {band}: {} vs {want}",
                got[0][band]
            );
        }
        for (band, want) in GOLDEN_F50.iter().enumerate() {
            assert!(
                (got[50][band] - want).abs() < 5e-4,
                "frame 50 band {band}: {} vs {want}",
                got[50][band]
            );
        }
        assert!(
            (got[0][79] - GOLDEN_F0_B79).abs() < 5e-4,
            "frame 0 band 79: {} vs {GOLDEN_F0_B79}",
            got[0][79]
        );
    }

    #[test]
    fn matches_the_numpy_128_golden_fixture() {
        // Generated by tests/reference/generate_whisper.py, which loads
        // official mel_128 weights and never constructs its own mel scale.
        // NumPy 2.5.3: float32 PCM/window/power/filterbank/projection/log;
        // rfft returns complex64. Values use the existing 5e-4 tolerance.
        // Include interior/high bands so low-band agreement alone cannot
        // hide a frequency-dependent normalization error.
        let got = whisper_log_mel(&synthetic_second(), 128).unwrap();
        assert_eq!(got.len(), 100);
        assert!(got.iter().all(|frame| frame.len() == 128));
        for (frame, band, want) in [
            (0, 0, 1.331_744_1),
            (0, 1, 1.429_308_4),
            (0, 2, 1.275_983),
            (0, 3, 1.241_528_2),
            (0, 4, 1.291_045_4),
            (0, 5, 1.187_713_1),
            (0, 6, 1.315_551_9),
            (0, 7, 1.030_345),
            (50, 0, 1.170_838_1),
            (50, 1, 1.268_402_6),
            (50, 2, 1.431_052_2),
            (50, 3, 1.396_597_4),
            (50, 4, 1.431_739_6),
            (50, 5, 1.229_586_6),
            (50, 6, 0.994_286_1),
            (50, 7, 0.552_494_8),
            (0, 64, 0.067_115_31),
            (50, 64, -0.567_001_2),
            (0, 127, -0.415_347_22),
        ] {
            assert!(
                (got[frame][band] - want).abs() < 5e-4,
                "frame {frame} band {band}: {} vs {want}",
                got[frame][band]
            );
        }
    }

    #[test]
    fn discarded_tail_energy_does_not_change_retained_frame_normalization() {
        let mut samples = vec![0.0; 3200];
        samples[3199] = 16.0;
        // The final impulse has nearly 100x more power in the discarded
        // centered frame than in any retained frame. Its amplitude keeps
        // the peak-relative clamp above the log floor in both orders.
        // NumPy plus the official asset gives these retained-only goldens;
        // normalizing before the drop raises silent frames by about 0.5
        // while still returning exactly the same 20-frame shape.
        for (n_mels, silent, last_low, last_high) in [
            (80, -1.289_807_3, 0.701_288_7, 0.701_562_4),
            (128, -1.242_611_4, 0.625_529_5, 0.701_609),
        ] {
            let got = whisper_log_mel(&samples, n_mels).unwrap();
            assert_eq!(got.len(), 20);
            assert!(got.iter().all(|frame| frame.len() == n_mels));
            for (frame, row) in got[..19].iter().enumerate() {
                for (band, &value) in row.iter().enumerate() {
                    assert!(
                        (value - silent).abs() < 5e-4,
                        "{n_mels} mels, silent frame {frame} band {band}: {value} vs {silent}"
                    );
                }
            }
            for (band, want) in [(0, last_low), (n_mels - 1, last_high)] {
                assert!(
                    (got[19][band] - want).abs() < 5e-4,
                    "{n_mels} mels, frame 19 band {band}: {} vs {want}",
                    got[19][band]
                );
            }
        }
    }

    #[test]
    fn window_path_truncates_the_tail_frame() {
        let mut samples = synthetic_second();
        samples.extend(synthetic_second());
        samples.extend(synthetic_second());
        // Pad to exactly one 30-second window with silence.
        samples.resize(WHISPER_WINDOW_SAMPLES, 0.0);
        // The window path must be exactly the full pipeline's first 3000
        // frames (same mel matrix, tail frame dropped), and its peak sits
        // near the official-filterbank reference's 1.4252 global max.
        let full = whisper_log_mel(&samples, 80).unwrap();
        let frames = whisper_log_mel_window(&samples, 80).unwrap();
        assert_eq!(frames.len(), WHISPER_MEL_FRAMES);
        assert_eq!(frames[0].len(), 80);
        assert_eq!(&frames[..], &full[..WHISPER_MEL_FRAMES]);
        let peak = frames.iter().flatten().cloned().fold(f32::MIN, f32::max);
        assert!((peak - 1.425_168_4).abs() < 2e-3, "peak {peak}");
    }

    #[test]
    fn rejects_wrong_n_mels_and_wrong_window_length() {
        let err = whisper_mel_filterbank(64).unwrap_err();
        assert!(err.to_string().contains("80 or 128"), "{err}");
        let err = whisper_log_mel_window(&[0.0; 1000], 80).unwrap_err();
        assert!(err.to_string().contains("whisper window samples"), "{err}");
    }
}
