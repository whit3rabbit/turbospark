//! NeMo-style log-mel frontend shared by speech model families.
//!
//! The numeric path follows mlx-audio 0.5.7 at
//! e1b19b9054bf163f5d812221a54fcc346f1890e9,
//! mlx_audio/stt/models/parakeet/audio.py and nemotron_asr/audio.py:
//! pre-emphasis, a symmetric Hann window, Slaney power mel bands, and natural
//! log. Parakeet uses constant signal padding plus normalization; Nemotron
//! uses reflect padding with no normalization. Inference stays in speech.

use crate::error::AudioError;
use crate::mel::{mel_projector_cached, MelScale};
use crate::nn::symmetric_hann;
use crate::stft::{stft_each, StftOptions, StftPaddingMode, StftWindowPlacement};

/// Statistics used by the NeMo frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoMelNormalization {
    /// Leave log-mel features unnormalized, matching NeMo's `normalize: NA`.
    None,
    /// Normalize each mel band independently with sample variance.
    PerFeature,
    /// Normalize all time and mel values together with population variance.
    Global,
}

/// NeMo audio-to-log-mel options.
///
/// The STFT window carries the exact analysis window chosen by the
/// checkpoint config. Centered placement and edge padding are selected by the
/// frontend call; the default options match Parakeet.
#[derive(Debug, Clone)]
pub struct NemoMelOptions {
    pub stft: StftOptions,
    pub sample_rate: u32,
    pub num_mels: usize,
    pub normalize: NemoMelNormalization,
    pub preemphasis: f32,
    pub log_zero_guard_value: f32,
    /// Pad the waveform to at least this many samples before pre-emphasis.
    pub pad_to: usize,
    pub pad_value: f32,
    /// Normalize from valid frames only and zero padded feature frames after.
    pub normalize_valid_frames: bool,
}

impl Default for NemoMelOptions {
    fn default() -> Self {
        Self {
            stft: StftOptions {
                fft_size: 512,
                hop: 160,
                window: symmetric_hann(400),
                center: true,
            },
            sample_rate: 16_000,
            num_mels: 128,
            normalize: NemoMelNormalization::PerFeature,
            preemphasis: 0.97,
            log_zero_guard_value: 2.0f32.powi(-24),
            pad_to: 0,
            pad_value: 0.0,
            normalize_valid_frames: false,
        }
    }
}

/// Converts mono PCM samples to [time, mel] log-mel features.
///
/// The output time axis includes the centered final frame. With
/// valid-frame normalization, only complete hop frames contribute to the
/// statistics, then frames past that valid extent are set to zero.
pub fn nemo_log_mel_spectrogram(
    samples: &[f32],
    options: &NemoMelOptions,
) -> Result<Vec<Vec<f32>>, AudioError> {
    nemo_log_mel_spectrogram_with_padding(samples, options, StftPaddingMode::Constant)
}

/// Converts mono PCM samples to log-mel features with explicit centered STFT
/// edge padding. Parakeet uses constant padding; Nemotron uses reflect.
pub fn nemo_log_mel_spectrogram_with_padding(
    samples: &[f32],
    options: &NemoMelOptions,
    padding_mode: StftPaddingMode,
) -> Result<Vec<Vec<f32>>, AudioError> {
    crate::error::check_sample_rate(options.sample_rate)?;
    if options.stft.hop == 0 {
        return Err(AudioError::InvalidParameter {
            name: "hop".to_string(),
            value: "0".to_string(),
            why: "must be positive".to_string(),
        });
    }
    if options.stft.window.is_empty() || options.stft.window.len() > options.stft.fft_size {
        return Err(AudioError::InvalidParameter {
            name: "window length".to_string(),
            value: options.stft.window.len().to_string(),
            why: format!(
                "must be between 1 and the FFT size {}",
                options.stft.fft_size
            ),
        });
    }
    if !options.stft.center {
        return Err(AudioError::InvalidParameter {
            name: "center".to_string(),
            value: "false".to_string(),
            why: "the NeMo frontend requires centered STFT frames".to_string(),
        });
    }
    if options.num_mels == 0 {
        return Err(AudioError::InvalidParameter {
            name: "num_mels".to_string(),
            value: "0".to_string(),
            why: "at least one mel band is required".to_string(),
        });
    }
    if !options.preemphasis.is_finite() || options.preemphasis < 0.0 {
        return Err(AudioError::InvalidParameter {
            name: "preemphasis".to_string(),
            value: options.preemphasis.to_string(),
            why: "must be finite and non-negative".to_string(),
        });
    }
    if !options.log_zero_guard_value.is_finite() || options.log_zero_guard_value <= 0.0 {
        return Err(AudioError::InvalidParameter {
            name: "log_zero_guard_value".to_string(),
            value: options.log_zero_guard_value.to_string(),
            why: "must be finite and positive".to_string(),
        });
    }
    if !options.pad_value.is_finite() {
        return Err(AudioError::InvalidParameter {
            name: "pad_value".to_string(),
            value: options.pad_value.to_string(),
            why: "must be finite".to_string(),
        });
    }

    let valid_frames = samples.len() / options.stft.hop;
    if options.normalize_valid_frames && valid_frames < 2 {
        return Err(AudioError::InvalidParameter {
            name: "valid_frames".to_string(),
            value: valid_frames.to_string(),
            why: "valid-frame normalization requires at least two frames".to_string(),
        });
    }

    let padded_len = samples.len().max(options.pad_to);
    if padded_len > 1 << 24 {
        return Err(AudioError::BufferTooLarge {
            what: "NeMo frontend waveform",
            samples: padded_len,
        });
    }
    let mut waveform = vec![options.pad_value; padded_len];
    waveform[..samples.len()].copy_from_slice(samples);
    if options.preemphasis > 0.0 {
        for i in (1..waveform.len()).rev() {
            waveform[i] -= options.preemphasis * waveform[i - 1];
        }
    }

    // Stream frames through reusable buffers instead of materializing every
    // complex spectrum. The filterbank result is checked only after the STFT
    // so an STFT error still takes precedence over a filterbank error, as
    // when the STFT ran to completion first.
    let projector = mel_projector_cached(
        options.num_mels,
        options.stft.fft_size,
        options.sample_rate,
        0.0,
        None,
        MelScale::Slaney,
    );
    let mut features: Vec<Vec<f32>> = Vec::new();
    let mut power: Vec<f32> = Vec::new();
    stft_each(
        &waveform,
        &options.stft,
        padding_mode,
        StftWindowPlacement::Center,
        |spectrum| {
            let Ok(projector) = &projector else {
                return Ok(());
            };
            power.clear();
            power.extend(
                spectrum
                    .iter()
                    .map(|value| value.re * value.re + value.im * value.im),
            );
            let mut row = Vec::with_capacity(options.num_mels);
            projector.project_into(&power, &mut row)?;
            for value in &mut row {
                *value = (*value + options.log_zero_guard_value).ln();
            }
            features.push(row);
            Ok(())
        },
    )?;
    projector?;

    if features.is_empty() {
        return Ok(features);
    }
    let statistics_frames = if options.normalize_valid_frames {
        valid_frames.min(features.len())
    } else {
        features.len()
    };
    let statistics_values = statistics_frames * options.num_mels;
    if statistics_values == 0 {
        return Err(AudioError::InvalidParameter {
            name: "normalization extent".to_string(),
            value: "0".to_string(),
            why: "must contain at least one mel value".to_string(),
        });
    }

    match options.normalize {
        NemoMelNormalization::None => {}
        NemoMelNormalization::PerFeature => {
            let denominator = statistics_frames.saturating_sub(1).max(1) as f32;
            for mel in 0..options.num_mels {
                let mean = features[..statistics_frames]
                    .iter()
                    .map(|row| row[mel])
                    .sum::<f32>()
                    / statistics_frames as f32;
                let variance = features[..statistics_frames]
                    .iter()
                    .map(|row| {
                        let delta = row[mel] - mean;
                        delta * delta
                    })
                    .sum::<f32>()
                    / denominator;
                let scale = variance.sqrt() + 1e-5;
                for row in &mut features {
                    row[mel] = (row[mel] - mean) / scale;
                }
            }
        }
        NemoMelNormalization::Global => {
            let mean = features[..statistics_frames]
                .iter()
                .flat_map(|row| row.iter())
                .sum::<f32>()
                / statistics_values as f32;
            let variance = features[..statistics_frames]
                .iter()
                .flat_map(|row| row.iter())
                .map(|&value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / statistics_values as f32;
            let scale = variance.sqrt() + 1e-5;
            for row in &mut features {
                for value in row {
                    *value = (*value - mean) / scale;
                }
            }
        }
    }

    if options.normalize_valid_frames {
        for row in features.iter_mut().skip(valid_frames) {
            row.fill(0.0);
        }
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_input() -> Vec<f32> {
        (0..512)
            .map(|i| (((i * 17) % 23) as f32 - 11.0) / 13.0)
            .collect()
    }

    fn small_options() -> NemoMelOptions {
        NemoMelOptions {
            stft: StftOptions {
                fft_size: 256,
                hop: 80,
                window: symmetric_hann(200),
                center: true,
            },
            sample_rate: 16_000,
            num_mels: 8,
            normalize: NemoMelNormalization::PerFeature,
            preemphasis: 0.97,
            log_zero_guard_value: 2.0f32.powi(-24),
            pad_to: 0,
            pad_value: 0.0,
            normalize_valid_frames: false,
        }
    }

    #[test]
    fn matches_pinned_mlx_audio_reference_values() {
        // Generated from mlx-audio 0.5.7, source commit
        // e1b19b9054bf163f5d812221a54fcc346f1890e9, using
        // parakeet.audio.log_mel_spectrogram. Input is 512 little-endian f32
        // values from reference_input(); SHA-256:
        // 84a2e27c871ecacb3421d2d242016c68456f221f75873d77f1387bcb41b244f8.
        // MLX output shape was [1, 7, 8], float32.
        let output = nemo_log_mel_spectrogram(&reference_input(), &small_options()).unwrap();
        assert_eq!((output.len(), output[0].len()), (7, 8));
        let indices = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 7),
            (1, 0),
            (2, 1),
            (3, 0),
            (3, 7),
            (6, 7),
        ];
        let expected = [
            -0.33181682,
            -2.2649765,
            -1.4148855,
            -2.2321172,
            -0.3646566,
            0.3568823,
            -0.39401114,
            0.4492494,
            -0.02202247,
        ];
        for ((time, mel), expected) in indices.into_iter().zip(expected) {
            assert!(
                (output[time][mel] - expected).abs() < 5e-4,
                "feature ({time}, {mel}): got {}, expected {expected}",
                output[time][mel]
            );
        }
    }

    #[test]
    fn valid_frame_mode_zeros_padded_feature_frames() {
        let samples = reference_input();
        let mut options = small_options();
        options.pad_to = 800;
        options.normalize_valid_frames = true;
        let output = nemo_log_mel_spectrogram(&samples, &options).unwrap();
        assert_eq!((output.len(), output[0].len()), (11, 8));
        assert!(output[6..].iter().flatten().all(|&value| value == 0.0));
    }

    #[test]
    fn valid_frame_mode_rejects_less_than_two_complete_hops() {
        let mut options = small_options();
        options.normalize_valid_frames = true;
        let error = nemo_log_mel_spectrogram(&[0.0; 159], &options).unwrap_err();
        assert!(error.to_string().contains("at least two frames"), "{error}");
    }

    #[test]
    fn nemotron_reflect_frontend_matches_pinned_mlx_reference() {
        // Generated by mlx-audio 0.5.7 nemotron_asr/audio.py from the same
        // deterministic 640-sample input (SHA-256:
        // f59b5f74de6087665855623471c6e0b4479c920dce9e77e4f3b2527327c86760).
        // This profile uses reflect centering and `normalize: NA`, unlike
        // Parakeet.
        let samples: Vec<f32> = (0..640)
            .map(|i| (((i * 29) % 113) as f32 / 113.0 - 0.5) * 0.2)
            .collect();
        let mut options = NemoMelOptions::default();
        options.num_mels = 4;
        options.normalize = NemoMelNormalization::None;
        let output =
            nemo_log_mel_spectrogram_with_padding(&samples, &options, StftPaddingMode::Reflect)
                .unwrap();
        let expected = [
            -8.06771088,
            -7.76390409,
            -4.18426085,
            -2.82281947,
            -8.06157589,
            -7.61776352,
            -4.23273420,
            -2.79012704,
            -8.06308270,
            -7.61670303,
            -4.23296833,
            -2.78991175,
            -8.05265331,
            -7.61036158,
            -4.23228121,
            -2.79016161,
            -7.37214661,
            -7.12100172,
            -4.13430691,
            -2.82784700,
        ];
        assert_eq!((output.len(), output[0].len()), (5, 4));
        for (row, expected_row) in output.iter().zip(expected.chunks_exact(4)) {
            for (&actual, &expected) in row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");
            }
        }
    }
}

/// Bitwise parity of the streaming, cached frontend against the previous
/// feature extraction, retained verbatim in `old`. Everything after the log
/// (normalization, valid-frame zeroing) is untouched, so comparing the
/// unnormalized features isolates the changed span.
#[cfg(test)]
mod parity_tests {
    use super::*;
    use crate::dsp::TestRng;
    use crate::mel::mel_filterbank;
    use crate::stft::stft_with_modes;

    mod old {
        use super::super::*;
        use super::{mel_filterbank, stft_with_modes};

        pub(super) fn features(
            samples: &[f32],
            options: &NemoMelOptions,
            padding_mode: StftPaddingMode,
        ) -> Result<Vec<Vec<f32>>, AudioError> {
            let padded_len = samples.len().max(options.pad_to);
            let mut waveform = vec![options.pad_value; padded_len];
            waveform[..samples.len()].copy_from_slice(samples);
            if options.preemphasis > 0.0 {
                for i in (1..waveform.len()).rev() {
                    waveform[i] -= options.preemphasis * waveform[i - 1];
                }
            }
            let spectra = stft_with_modes(
                &waveform,
                &options.stft,
                padding_mode,
                StftWindowPlacement::Center,
            )?;
            let filters = mel_filterbank(
                options.num_mels,
                options.stft.fft_size,
                options.sample_rate,
                0.0,
                None,
                MelScale::Slaney,
            )?;
            let mut features = Vec::with_capacity(spectra.len());
            for spectrum in spectra {
                let power: Vec<f32> = spectrum
                    .iter()
                    .map(|value| value.re * value.re + value.im * value.im)
                    .collect();
                let mut row = filters.project(&power)?;
                for value in &mut row {
                    *value = (*value + options.log_zero_guard_value).ln();
                }
                features.push(row);
            }
            Ok(features)
        }
    }

    #[test]
    fn unnormalized_features_match_old_bitwise() {
        let mut rng = TestRng(0x4E45_4D4F_4D45_4C31);
        let configs = [
            NemoMelOptions {
                normalize: NemoMelNormalization::None,
                ..NemoMelOptions::default()
            },
            NemoMelOptions {
                stft: StftOptions {
                    fft_size: 256,
                    hop: 80,
                    window: symmetric_hann(200),
                    center: true,
                },
                num_mels: 8,
                normalize: NemoMelNormalization::None,
                pad_to: 4_000,
                ..NemoMelOptions::default()
            },
            NemoMelOptions {
                stft: StftOptions {
                    fft_size: 512,
                    hop: 160,
                    window: symmetric_hann(400),
                    center: true,
                },
                num_mels: 128,
                normalize: NemoMelNormalization::None,
                preemphasis: 0.0,
                ..NemoMelOptions::default()
            },
        ];
        for (c, options) in configs.iter().enumerate() {
            for len in [1_000usize, 6_000, 16_000] {
                let samples = rng.vec(len);
                for padding in [StftPaddingMode::Constant, StftPaddingMode::Reflect] {
                    let got =
                        nemo_log_mel_spectrogram_with_padding(&samples, options, padding).unwrap();
                    let want = old::features(&samples, options, padding).unwrap();
                    assert_eq!(got.len(), want.len());
                    for (f, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(g.len(), w.len());
                        for (b, (a, c2)) in g.iter().zip(w).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                c2.to_bits(),
                                "config {c} len {len} {padding:?} [{f}][{b}]"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stft_error_still_precedes_filterbank_error() {
        // Both the STFT (reflect-pad needs 129 samples at fft 256) and the
        // filterbank (num_mels overflow) fail; the old order surfaced the
        // STFT error first.
        let options = NemoMelOptions {
            stft: StftOptions {
                fft_size: 256,
                hop: 80,
                window: symmetric_hann(200),
                center: true,
            },
            num_mels: usize::MAX,
            normalize: NemoMelNormalization::None,
            ..NemoMelOptions::default()
        };
        let samples = vec![0.1f32; 50];
        let want = old::features(&samples, &options, StftPaddingMode::Reflect)
            .unwrap_err()
            .to_string();
        let got =
            nemo_log_mel_spectrogram_with_padding(&samples, &options, StftPaddingMode::Reflect)
                .unwrap_err()
                .to_string();
        assert_eq!(got, want);
        assert!(want.contains("reflect-pad"), "{want}");
    }
}
