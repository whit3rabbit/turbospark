use serde_json::Value;

use crate::stt::sensevoice::frontend::{compute_fbank, FrontendConfig};
use crate::{Result, SpeechError};

const SAMPLE_RATE: usize = 16_000;
const MEL_BINS: usize = 80;

pub(super) struct Frontend {
    means: Vec<f32>,
    inverse_std: Vec<f32>,
}

impl Frontend {
    pub(super) fn from_json(value: &Value) -> Result<Self> {
        let means = numbers(value, "means")?;
        let inverse_std = numbers(value, "istd")?;
        if means.len() != MEL_BINS || inverse_std.len() != MEL_BINS {
            return Err(SpeechError::BadConfig {
                field: "cmvn.json".into(),
                why: format!("expected {MEL_BINS} means and inverse standard deviations"),
            });
        }
        if means
            .iter()
            .chain(&inverse_std)
            .any(|value| !value.is_finite())
        {
            return Err(SpeechError::BadConfig {
                field: "cmvn.json".into(),
                why: "normalization values must be finite".into(),
            });
        }
        Ok(Self { means, inverse_std })
    }

    pub(super) fn extract(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "FireRedASR2 audio must not be empty".into(),
            });
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "FireRedASR2 audio samples must be finite".into(),
            });
        }

        // mlx-audio scales normalized float waveforms to int16 range before
        // calling Kaldi FBANK. The shared implementation applies that scale.
        let peak = samples
            .iter()
            .fold(0.0f32, |best, value| best.max(value.abs()));
        let normalized;
        let input = if peak > 1.0 {
            normalized = samples
                .iter()
                .map(|sample| sample / 32768.0)
                .collect::<Vec<_>>();
            normalized.as_slice()
        } else {
            samples
        };
        let config = FrontendConfig {
            sample_rate: SAMPLE_RATE,
            num_mels: MEL_BINS,
            frame_length_ms: 25,
            frame_shift_ms: 10,
            lfr_m: 1,
            lfr_n: 1,
        };
        let mut features = compute_fbank(input, config)?;
        let frames = features.len() / MEL_BINS;
        if frames == 0 {
            return Err(SpeechError::Input {
                why: "FireRedASR2 audio is shorter than one complete 25 ms frame".into(),
            });
        }
        for row in 0..frames {
            for mel in 0..MEL_BINS {
                let index = row * MEL_BINS + mel;
                features[index] = (features[index] - self.means[mel]) * self.inverse_std[mel];
            }
        }
        Ok((features, frames))
    }
}

fn numbers(value: &Value, field: &str) -> Result<Vec<f32>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| SpeechError::BadConfig {
            field: format!("cmvn.json.{field}"),
            why: "must be an array of numbers".into(),
        })?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: format!("cmvn.json.{field}"),
                    why: "array must contain only numbers".into(),
                })
        })
        .collect()
}
