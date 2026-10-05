//! Qwen3-ASR WhisperFeatureExtractor-compatible log-mel frontend.
//!
//! The reference calls Transformers' `WhisperFeatureExtractor` with 128 mel
//! bands, a 400-sample Hann window, 160-sample hop, log10 power and the global
//! peak-minus-eight normalization. The shared audio crate owns that numeric
//! pipeline; this module adapts its time-major result to Qwen3's band-major
//! `[mel_bins, frames]` layout.

use turbospark_audio::whisper::whisper_log_mel;

use crate::{Result, SpeechError};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
const MAX_SAMPLES: usize = SAMPLE_RATE as usize * 1_200;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Band-major `[mel_bins, frames]`, matching `input_features` in MLX.
    pub values: Vec<f32>,
    pub frames: usize,
}

pub fn compute_features(samples: &[f32]) -> Result<AudioFeatures> {
    if samples.is_empty() {
        return Err(SpeechError::Input {
            why: "audio must contain at least one sample".into(),
        });
    }
    if samples.len() > MAX_SAMPLES {
        return Err(SpeechError::Input {
            why: format!("audio exceeds the 20-minute frontend limit ({MAX_SAMPLES} samples)"),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SpeechError::Input {
            why: "audio samples must be finite".into(),
        });
    }

    let mel = whisper_log_mel(samples, MEL_BINS)?;
    if mel.is_empty() || mel.iter().any(|frame| frame.len() != MEL_BINS) {
        return Err(SpeechError::Audio(
            "128-band Whisper frontend returned an invalid shape".into(),
        ));
    }
    let frames = mel.len();
    let mut values = vec![0.0f32; MEL_BINS * frames];
    for (frame, row) in mel.iter().enumerate() {
        for (bin, &value) in row.iter().enumerate() {
            values[bin * frames + frame] = value;
        }
    }
    Ok(AudioFeatures { values, frames })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct FeatureFixture {
        sample_count: usize,
        feature_shape: [usize; 2],
        selected_frames: Vec<usize>,
        selected_values: Vec<Vec<f32>>,
    }

    fn reference_audio() -> Vec<f32> {
        let tau = std::f64::consts::TAU;
        (0..44_720)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                (0.1 * (tau * 440.0 * t).sin() + 0.03 * (tau * 997.0 * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn features_match_pinned_transformers_reference_fixture() {
        let fixture: FeatureFixture =
            serde_json::from_str(include_str!("../../../testdata/qwen3_asr_features.json"))
                .unwrap();
        let samples = reference_audio();
        assert_eq!(samples.len(), fixture.sample_count);
        let output = compute_features(&samples).unwrap();
        assert_eq!(fixture.feature_shape, [MEL_BINS, output.frames]);
        assert_eq!(fixture.selected_frames.len(), fixture.selected_values.len());

        for (frame, expected) in fixture.selected_frames.iter().zip(&fixture.selected_values) {
            assert_eq!(expected.len(), MEL_BINS);
            for (bin, &want) in expected.iter().enumerate() {
                let got = output.values[bin * output.frames + frame];
                assert!(
                    (got - want).abs() <= 2.0e-4,
                    "feature mismatch at mel={bin}, frame={frame}: got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_and_overlong_audio_before_frontend_allocation() {
        assert!(matches!(
            compute_features(&[]),
            Err(SpeechError::Input { .. })
        ));
        assert!(matches!(
            compute_features(&[0.0, f32::NAN]),
            Err(SpeechError::Input { .. })
        ));
        assert!(matches!(
            compute_features(&vec![0.0; MAX_SAMPLES + 1]),
            Err(SpeechError::Input { .. })
        ));
    }
}
