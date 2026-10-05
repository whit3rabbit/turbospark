//! FireRedASR2-AED encoder-decoder transcription from mlx-audio 0.5.7.
//!
//! Reference: `mlx_audio/stt/models/fireredasr2/` at source commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

mod config;
mod decoder;
mod encoder;
mod frontend;

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

pub use config::FireRedAsr2Config;

/// Immutable Hugging Face model profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireRedAsr2Profile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const FIREREDASR2_AED: FireRedAsr2Profile = FireRedAsr2Profile {
    name: "FireRedASR2-AED",
    repository: "mlx-community/FireRedASR2-AED-mlx",
    revision: "f3212eacfa49b851130b97c63653c8e06ee09bdb",
};

#[derive(Debug, Clone, Copy)]
pub struct FireRedAsr2Options {
    pub beam_size: usize,
    /// Zero follows mlx-audio and decodes for the encoder output length.
    pub max_len: usize,
    pub softmax_smoothing: f32,
    pub length_penalty: f32,
    pub eos_penalty: f32,
}

impl Default for FireRedAsr2Options {
    fn default() -> Self {
        Self {
            beam_size: 3,
            max_len: 0,
            softmax_smoothing: 1.25,
            length_penalty: 0.6,
            eos_penalty: 1.0,
        }
    }
}

/// Loaded FireRedASR2-AED snapshot. The checkpoint and its CMVN/dictionary
/// sidecars are read locally; this type never downloads model files.
pub struct FireRedAsr2 {
    config: FireRedAsr2Config,
    frontend: frontend::Frontend,
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    dictionary: Vec<String>,
}

impl FireRedAsr2 {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_value: Value =
            serde_json::from_slice(&fs::read(&config_path).map_err(|error| {
                SpeechError::BadConfig {
                    field: config_path.display().to_string(),
                    why: error.to_string(),
                }
            })?)
            .map_err(|error| SpeechError::BadConfig {
                field: "config.json".into(),
                why: error.to_string(),
            })?;
        let config = FireRedAsr2Config::from_json(&config_value)?;

        let cmvn_path = model_dir.join("cmvn.json");
        let cmvn_value: Value = serde_json::from_slice(&fs::read(&cmvn_path).map_err(|error| {
            SpeechError::BadConfig {
                field: cmvn_path.display().to_string(),
                why: error.to_string(),
            }
        })?)
        .map_err(|error| SpeechError::BadConfig {
            field: "cmvn.json".into(),
            why: error.to_string(),
        })?;
        let frontend = frontend::Frontend::from_json(&cmvn_value)?;

        let dictionary_path = model_dir.join("dict.txt");
        let dictionary =
            parse_dictionary(&fs::read_to_string(&dictionary_path).map_err(|error| {
                SpeechError::BadConfig {
                    field: dictionary_path.display().to_string(),
                    why: error.to_string(),
                }
            })?);
        if dictionary.len() != config.vocab_size {
            return Err(SpeechError::BadConfig {
                field: "dict.txt".into(),
                why: format!(
                    "expected {} dictionary entries, got {}",
                    config.vocab_size,
                    dictionary.len()
                ),
            });
        }

        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = encoder::Encoder::load(&weights, config.clone())?;
        let decoder = decoder::Decoder::load(&weights, config.clone())?;
        Ok(Self {
            config,
            frontend,
            encoder,
            decoder,
            dictionary,
        })
    }

    pub fn profile(&self) -> FireRedAsr2Profile {
        FIREREDASR2_AED
    }

    /// Transcribe mono normalized f32 PCM at 16 kHz with upstream beam defaults.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, FireRedAsr2Options::default())
    }

    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        options: FireRedAsr2Options,
    ) -> Result<String> {
        let (features, frames) = self.frontend.extract(samples)?;
        let encoded = self.encoder.forward(&features, frames, false)?;
        let max_len = if options.max_len == 0 {
            encoded.rows
        } else {
            options.max_len
        };
        self.decoder.decode(
            &encoded.values,
            encoded.rows,
            &self.dictionary,
            options.beam_size,
            max_len,
            options.softmax_smoothing,
            options.length_penalty,
            options.eos_penalty,
        )
    }
}

fn parse_dictionary(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .map(str::to_owned)
                .unwrap_or_else(|| " ".into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::{config::FireRedAsr2Config, encoder::Snapshot, *};

    #[test]
    fn parses_only_the_pinned_profile_geometry() {
        let value: Value = serde_json::json!({
            "model_type":"fireredasr2", "idim":80, "odim":8667, "d_model":1280,
            "sos_id":3, "eos_id":4, "pad_id":2, "blank_id":0,
            "encoder":{"n_layers":16,"n_head":20,"d_model":1280,"kernel_size":33,"pe_maxlen":5000},
            "decoder":{"n_layers":16,"n_head":20,"d_model":1280,"pe_maxlen":5000}
        });
        let config = FireRedAsr2Config::from_json(&value).unwrap();
        assert_eq!(config.vocab_size, 8667);
        assert_eq!(config.encoder_layers, 16);
        assert!(
            FireRedAsr2Config::from_json(&serde_json::json!({"model_type":"whisper"})).is_err()
        );
        let wrong = serde_json::json!({
            "model_type":"fireredasr2", "idim":80, "odim":8667, "d_model":1024,
            "sos_id":3, "eos_id":4, "pad_id":2,
            "encoder":{"n_layers":16,"n_head":20,"d_model":1024,"kernel_size":33,"pe_maxlen":5000},
            "decoder":{"n_layers":16,"n_head":20,"d_model":1024,"pe_maxlen":5000}
        });
        assert!(FireRedAsr2Config::from_json(&wrong).is_err());
    }

    #[test]
    fn parses_dictionary_tokens_and_empty_entries_like_reference() {
        assert_eq!(
            parse_dictionary("a 10\n<space> 4\n\n"),
            vec!["a", "<space>", " "]
        );
    }

    #[test]
    #[ignore = "requires the reference WAV and the pinned FireRedASR2 sidecars"]
    fn pinned_frontend_matches_all_mlx_frames() {
        let model_dir = std::env::var_os("TURBOSPARK_FIREREDASR2_DIR")
            .map(std::path::PathBuf::from)
            .expect("set TURBOSPARK_FIREREDASR2_DIR to the pinned model snapshot");
        let cmvn: Value = serde_json::from_slice(
            &fs::read(model_dir.join("cmvn.json")).expect("pinned CMVN loads"),
        )
        .expect("pinned CMVN parses");
        let frontend = frontend::Frontend::from_json(&cmvn).expect("CMVN validates");
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = turbospark_audio::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let raw = crate::stt::sensevoice::frontend::compute_fbank(
            &audio.samples,
            crate::stt::sensevoice::frontend::FrontendConfig {
                sample_rate: 16_000,
                num_mels: 80,
                frame_length_ms: 25,
                frame_shift_ms: 10,
                lfr_m: 1,
                lfr_n: 1,
            },
        )
        .expect("raw FBANK runs");
        let (features, _) = frontend.extract(&audio.samples).expect("FBANK runs");
        let reference: Value =
            serde_json::from_str(include_str!("../../../testdata/fireredasr2_reference.json"))
                .expect("FireRedASR2 MLX fixture parses");
        let expected = reference["stages"]["features"]["all_values"]
            .as_array()
            .expect("complete MLX FBANK fixture exists");
        assert_eq!(features.len(), expected.len());
        let expected_raw = reference["stages"]["features"]["raw_values"]
            .as_array()
            .expect("complete MLX raw FBANK fixture exists");
        assert_eq!(raw.len(), expected_raw.len());
        let raw_max_diff = raw
            .iter()
            .zip(expected_raw)
            .map(|(&got, expected)| (got - expected.as_f64().unwrap() as f32).abs())
            .fold(0.0f32, f32::max);
        assert!(raw_max_diff <= 1e-2, "raw FBANK max diff {raw_max_diff}");
        let mut max_diff = (0.0f32, 0usize);
        let mut outliers = Vec::new();
        for (index, (&got, expected)) in features.iter().zip(expected).enumerate() {
            let expected = expected.as_f64().unwrap() as f32;
            let difference = (got - expected).abs();
            if difference > max_diff.0 {
                max_diff = (difference, index);
            }
            if difference > 1e-2 && outliers.len() < 12 {
                outliers.push((index, got, expected, difference));
            }
        }
        assert!(
            max_diff.0 <= 1e-2,
            "max FBANK diff {max_diff:?}; first outliers {outliers:?}"
        );
    }

    #[test]
    #[ignore = "requires the pinned FireRedASR2 snapshot in TURBOSPARK_FIREREDASR2_DIR"]
    fn pinned_first_conformer_block_stages_match_mlx() {
        let model_dir = std::env::var_os("TURBOSPARK_FIREREDASR2_DIR")
            .map(std::path::PathBuf::from)
            .expect("set TURBOSPARK_FIREREDASR2_DIR to the pinned model snapshot");
        let model = FireRedAsr2::open(&model_dir).expect("pinned FireRedASR2 checkpoint loads");
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = turbospark_audio::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let (features, frames) = model.frontend.extract(&audio.samples).expect("FBANK runs");
        let encoded = model
            .encoder
            .forward_first_block(&features, frames, true)
            .expect("first encoder block runs");
        let reference: Value =
            serde_json::from_str(include_str!("../../../testdata/fireredasr2_reference.json"))
                .expect("FireRedASR2 MLX fixture parses");
        let trace = encoded.trace.as_ref().expect("first-block trace requested");
        assert_snapshot(
            "subsampled",
            &trace.subsampled,
            &reference["stages"]["subsampled"],
        );
        for (name, actual) in [
            "ffn1_norm",
            "ffn1_expand",
            "ffn1_silu",
            "ffn1_project",
            "ffn1_residual",
            "ffn1",
            "mhsa",
            "conv",
            "ffn2",
            "layer_norm",
        ]
        .into_iter()
        .zip(&trace.first_block_stages)
        {
            let expected = reference["stages"]["first_block_components"][name]["values"]
                .as_array()
                .unwrap();
            let max_diff = actual
                .values
                .iter()
                .zip(expected)
                .map(|(&got, expected)| (got - expected.as_f64().unwrap() as f32).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff <= 0.03, "{name} max sampled diff {max_diff}");
        }
    }

    #[test]
    #[ignore = "requires the pinned FireRedASR2 snapshot in TURBOSPARK_FIREREDASR2_DIR"]
    fn pinned_checkpoint_matches_mlx_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_FIREREDASR2_DIR")
            .map(std::path::PathBuf::from)
            .expect("set TURBOSPARK_FIREREDASR2_DIR to the pinned model snapshot");
        let model = FireRedAsr2::open(&model_dir).expect("pinned FireRedASR2 checkpoint loads");
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = turbospark_audio::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let (features, frames) = model.frontend.extract(&audio.samples).expect("FBANK runs");
        let encoded = model
            .encoder
            .forward(&features, frames, true)
            .expect("encoder runs");
        let reference: Value =
            serde_json::from_str(include_str!("../../../testdata/fireredasr2_reference.json"))
                .expect("FireRedASR2 MLX fixture parses");
        assert_snapshot(
            "features",
            &Snapshot {
                shape: vec![frames, 80],
                indices: sample_indices(features.len()),
                values: sample_values(&features),
            },
            &reference["stages"]["features"],
        );
        let trace = encoded.trace.as_ref().expect("encoder trace requested");
        assert_snapshot(
            "subsampled",
            &trace.subsampled,
            &reference["stages"]["subsampled"],
        );
        assert_snapshot(
            "first_block",
            &trace.first_block,
            &reference["stages"]["first_block"],
        );
        assert_snapshot(
            "final_block",
            &trace.final_block,
            &reference["stages"]["final_block"],
        );
        let transcript = model
            .decoder
            .decode(
                &encoded.values,
                encoded.rows,
                &model.dictionary,
                3,
                encoded.rows,
                1.25,
                0.6,
                1.0,
            )
            .expect("FireRedASR2 beam decoder runs");
        assert_eq!(transcript, reference["transcript"].as_str().unwrap());
    }

    fn sample_indices(count: usize) -> Vec<usize> {
        let samples = count.min(16);
        if samples <= 1 {
            return vec![0];
        }
        (0..samples)
            .map(|index| index * (count - 1) / (samples - 1))
            .collect()
    }

    fn sample_values(values: &[f32]) -> Vec<f32> {
        sample_indices(values.len())
            .iter()
            .map(|&index| values[index])
            .collect()
    }

    fn assert_snapshot(name: &str, actual: &Snapshot, expected: &Value) {
        let expected_shape = expected["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert!(
            actual.shape == expected_shape
                || (expected_shape.first() == Some(&1) && actual.shape == expected_shape[1..]),
            "{name} shape: Rust {:?}, MLX {expected_shape:?}",
            actual.shape
        );
        let expected_indices = expected["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(actual.indices, expected_indices, "{name} indices");
        let expected_values = expected["values"].as_array().unwrap();
        let tolerance = if name == "features" { 1e-2 } else { 3e-2 };
        for (index, (&got, expected)) in actual.values.iter().zip(expected_values).enumerate() {
            let expected = expected.as_f64().unwrap() as f32;
            assert!(
                (got - expected).abs() <= tolerance,
                "{name}[{}]: Rust {got}, MLX {expected}",
                actual.indices[index]
            );
        }
    }
}
