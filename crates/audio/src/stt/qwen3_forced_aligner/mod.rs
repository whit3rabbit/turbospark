//! Qwen3 word-level forced alignment.
//!
//! Reference: `mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py` and
//! `qwen3_asr.py` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer;

use crate::models::stt::qwen3_asr::{config, decoder, encoder, frontend, load_tokenizer};
use crate::quant::QuantScheme;
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3ForcedAlignerProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const QWEN3_FORCED_ALIGNER_06B_8BIT: Qwen3ForcedAlignerProfile = Qwen3ForcedAlignerProfile {
    name: "Qwen3-ForcedAligner 0.6B 8-bit",
    repository: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    revision: "0e1a68e91d815300c7c9754b2a7639378b23db15",
};

/// Verified timestamp-classification subset of the pinned forced aligner.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3ForcedAlignerConfig {
    pub timestamp_token_id: i32,
    pub timestamp_segment_time_ms: f32,
    pub classify_num: usize,
    base: config::Qwen3Config,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.into(),
    }
}

fn positive_usize(root: &Value, nested: &Value, field: &str) -> Result<usize> {
    nested
        .get(field)
        .or_else(|| root.get(field))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| bad(field, "must be a positive integer"))
}

impl Qwen3ForcedAlignerConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        let thinker = root.get("thinker_config").unwrap_or(root);
        if thinker.get("model_type").and_then(Value::as_str) != Some("qwen3_forced_aligner") {
            return Err(bad(
                "thinker_config.model_type",
                "expected qwen3_forced_aligner",
            ));
        }
        let timestamp_token_id = thinker
            .get("timestamp_token_id")
            .or_else(|| root.get("timestamp_token_id"))
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .filter(|&value| value >= 0)
            .ok_or_else(|| {
                bad(
                    "timestamp_token_id",
                    "must be a nonnegative signed 32-bit id",
                )
            })?;
        let timestamp_segment_time_ms = thinker
            .get("timestamp_segment_time")
            .or_else(|| root.get("timestamp_segment_time"))
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= f32::MAX as f64)
            .map(|value| value as f32)
            .ok_or_else(|| bad("timestamp_segment_time", "must be finite and positive"))?;
        let classify_num = positive_usize(root, thinker, "classify_num")?;
        if classify_num != 5000 || timestamp_segment_time_ms != 80.0 {
            return Err(SpeechError::Unsupported {
                why: "only the pinned 5000-class, 80 ms Qwen3 aligner head is supported".into(),
            });
        }

        Ok(Self {
            timestamp_token_id,
            timestamp_segment_time_ms,
            classify_num,
            base: config::Qwen3Config::from_json(root)?,
        })
    }
}

/// One aligned word or character span. Timestamps are rounded to milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct ForcedAlignItem {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
}

/// Word- or character-level alignment result for one clip.
#[derive(Debug, Clone, PartialEq)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
}

impl ForcedAlignResult {
    pub fn text(&self) -> String {
        self.items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Loaded Qwen3-ForcedAligner. It shares the audio encoder and Qwen3 decoder
/// implementation with Qwen3-ASR but uses a 5000-class timestamp head.
pub struct Qwen3ForcedAligner {
    config: Qwen3ForcedAlignerConfig,
    encoder: encoder::AudioEncoder,
    decoder: decoder::Decoder,
    timestamp_head: decoder::Linear,
    tokenizer: Tokenizer,
}

impl Qwen3ForcedAligner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|error| bad("config.json", error.to_string()))?,
        )
        .map_err(|error| bad("config.json", error.to_string()))?;
        let config = Qwen3ForcedAlignerConfig::from_json(&root)?;
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let scheme = QuantScheme {
            bits: config.base.quant_bits,
            group_size: config.base.quant_group_size,
        };
        let encoder = encoder::AudioEncoder::load(&weights, &config.base.audio)?;
        let decoder = decoder::Decoder::load(&weights, &config.base.text, scheme)?;
        let timestamp_head = decoder::Linear::load(
            &weights,
            "lm_head",
            config.base.text.hidden_size,
            config.classify_num,
            scheme,
        )?;
        let tokenizer = load_tokenizer(model_dir)?;
        for (token, expected) in [
            ("<|audio_pad|>", config.base.audio_token_id),
            ("<|audio_start|>", config.base.audio_start_token_id),
            ("<|audio_end|>", config.base.audio_end_token_id),
            ("<timestamp>", config.timestamp_token_id),
        ] {
            if tokenizer.token_to_id(token).map(|id| id as i32) != Some(expected) {
                return Err(bad(
                    "tokenizer_config.json",
                    format!("{token} id does not match config.json"),
                ));
            }
        }

        Ok(Self {
            config,
            encoder,
            decoder,
            timestamp_head,
            tokenizer,
        })
    }

    pub fn config(&self) -> &Qwen3ForcedAlignerConfig {
        &self.config
    }

    /// Aligns space-delimited language text against mono 16 kHz PCM.
    pub fn align(&self, samples: &[f32], text: &str) -> Result<ForcedAlignResult> {
        self.align_with_language(samples, text, "English")
    }

    /// Aligns a transcript using the given language's word segmentation.
    /// Japanese and Korean segmentation require optional Python packages in
    /// mlx-audio and are explicitly unsupported in this portable Rust port.
    pub fn align_with_language(
        &self,
        samples: &[f32],
        text: &str,
        language: &str,
    ) -> Result<ForcedAlignResult> {
        let language = self
            .config
            .base
            .supported_languages
            .iter()
            .find(|known| known.eq_ignore_ascii_case(language))
            .map(String::as_str)
            .unwrap_or(language);
        if language.eq_ignore_ascii_case("japanese") || language.eq_ignore_ascii_case("korean") {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "Qwen3 forced alignment language segmentation is unavailable for {language}"
                ),
            });
        }

        let words = tokenize_alignment_text(text, language)?;
        if words.is_empty() {
            return Err(SpeechError::Input {
                why: "alignment transcript contains no letters or numbers".into(),
            });
        }
        let features = frontend::compute_features(samples)?;
        let audio_features = self.encoder.forward(&features)?;
        let hidden_size = self.decoder.hidden_size;
        if audio_features.len() % hidden_size != 0 {
            return Err(SpeechError::Tensor {
                name: "audio_tower.output".into(),
                why: "encoder output is not a whole number of decoder embeddings".into(),
            });
        }
        let audio_rows = audio_features.len() / hidden_size;
        let transcript = words
            .iter()
            .map(|word| format!("{word}<timestamp><timestamp>"))
            .collect::<String>();
        let prompt = format!(
            "<|audio_start|>{}<|audio_end|>{transcript}",
            "<|audio_pad|>".repeat(audio_rows)
        );
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| SpeechError::Input {
                why: format!("Qwen3 aligner prompt tokenization failed: {error}"),
            })?;
        let ids = encoded
            .get_ids()
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "Qwen3 aligner token id exceeds signed 32-bit range".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_positions = matching_positions(&ids, self.config.base.audio_token_id);
        let timestamp_positions = matching_positions(&ids, self.config.timestamp_token_id);
        if audio_positions.len() != audio_rows {
            return Err(SpeechError::Input {
                why: format!(
                    "Qwen3 aligner prompt has {} audio placeholders for {audio_rows} encoded frames",
                    audio_positions.len()
                ),
            });
        }
        if timestamp_positions.len() != words.len() * 2 {
            return Err(SpeechError::Input {
                why: format!(
                    "Qwen3 aligner prompt has {} timestamp markers for {} words",
                    timestamp_positions.len(),
                    words.len()
                ),
            });
        }

        let mut embeddings = self.decoder.embed(&ids)?;
        for (audio_row, &position) in audio_positions.iter().enumerate() {
            let target = position * hidden_size;
            let source = audio_row * hidden_size;
            embeddings[target..target + hidden_size]
                .copy_from_slice(&audio_features[source..source + hidden_size]);
        }
        let rows = ids.len();
        let hidden = self.decoder.forward(&embeddings, rows);
        let mut timestamp_hidden = Vec::with_capacity(timestamp_positions.len() * hidden_size);
        for position in &timestamp_positions {
            let start = position * hidden_size;
            timestamp_hidden.extend_from_slice(&hidden[start..start + hidden_size]);
        }
        let logits = self
            .timestamp_head
            .forward(&timestamp_hidden, timestamp_positions.len());
        let timestamp_ms = logits
            .chunks_exact(self.config.classify_num)
            .map(|row| {
                let class = argmax(row);
                class as f32 * self.config.timestamp_segment_time_ms
            })
            .collect::<Vec<_>>();
        let repaired = fix_timestamps(&timestamp_ms);

        let items = words
            .into_iter()
            .enumerate()
            .map(|(index, word)| ForcedAlignItem {
                text: word,
                start_time: round_millis(repaired[index * 2] / 1000.0),
                end_time: round_millis(repaired[index * 2 + 1] / 1000.0),
            })
            .collect();
        Ok(ForcedAlignResult { items })
    }
}

fn matching_positions(ids: &[i32], target: i32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(index, &id)| (id == target).then_some(index))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

fn round_millis(seconds: f32) -> f32 {
    (seconds * 1000.0).round() / 1000.0
}

fn tokenize_alignment_text(text: &str, language: &str) -> Result<Vec<String>> {
    if language.eq_ignore_ascii_case("japanese") || language.eq_ignore_ascii_case("korean") {
        return Err(SpeechError::Unsupported {
            why: format!(
                "Qwen3 forced alignment requires an external word tokenizer for {language}"
            ),
        });
    }
    let mut words = Vec::new();
    for segment in text.split_whitespace() {
        let mut latin = String::new();
        for ch in segment.chars() {
            if is_cjk_char(ch) {
                if !latin.is_empty() {
                    words.push(std::mem::take(&mut latin));
                }
                words.push(ch.to_string());
            } else if ch == '\'' || ch.is_alphanumeric() {
                latin.push(ch);
            } else if !latin.is_empty() {
                words.push(std::mem::take(&mut latin));
            }
        }
        if !latin.is_empty() {
            words.push(latin);
        }
    }
    Ok(words)
}

fn is_cjk_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
    )
}

/// Port of the reference longest-nondecreasing-subsequence timestamp repair.
fn fix_timestamps(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut tails = Vec::<f32>::new();
    let mut levels = vec![Vec::<usize>::new()];
    for (index, &value) in data.iter().enumerate() {
        if value.is_nan() {
            levels[0].push(index);
            continue;
        }
        let length = tails.partition_point(|tail| *tail <= value);
        if length == tails.len() {
            tails.push(value);
            if length == levels.len() {
                levels.push(Vec::new());
            }
        } else {
            tails[length] = value;
        }
        levels[length].push(index);
    }
    if tails.is_empty() {
        return data.to_vec();
    }

    let mut normal = vec![false; data.len()];
    let mut index = levels
        .last()
        .and_then(|items| items.first())
        .copied()
        .unwrap();
    normal[index] = true;
    for level in levels.iter().rev().skip(1) {
        index = level
            .iter()
            .copied()
            .find(|&candidate| data[candidate] <= data[index])
            .unwrap();
        normal[index] = true;
    }

    let mut result = data.to_vec();
    let mut start = 0;
    while start < data.len() {
        if normal[start] {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < data.len() && !normal[end] {
            end += 1;
        }
        let count = end - start;
        let left = (0..start).rev().find(|&candidate| normal[candidate]);
        let right = (end..data.len()).find(|&candidate| normal[candidate]);
        if count <= 2 {
            for candidate in start..end {
                result[candidate] = match (left, right) {
                    (None, None) => result[candidate],
                    (None, Some(r)) => result[r],
                    (Some(l), None) => result[l],
                    (Some(l), Some(r)) => {
                        let left_distance = candidate - (start - 1);
                        let right_distance = end - candidate;
                        if left_distance <= right_distance {
                            result[l]
                        } else {
                            result[r]
                        }
                    }
                };
            }
        } else {
            match (left, right) {
                (Some(l), Some(r)) => {
                    let step = (result[r] - result[l]) / (count + 1) as f32;
                    for candidate in start..end {
                        result[candidate] = result[l] + step * (candidate - start + 1) as f32;
                    }
                }
                (Some(l), None) => {
                    let value = result[l];
                    result[start..end].fill(value);
                }
                (None, Some(r)) => {
                    let value = result[r];
                    result[start..end].fill(value);
                }
                (None, None) => {}
            }
        }
        start = end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{
        fix_timestamps, is_cjk_char, tokenize_alignment_text, Qwen3ForcedAligner,
        QWEN3_FORCED_ALIGNER_06B_8BIT,
    };
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize)]
    struct Fixture {
        sample_rate: u32,
        sample_count: usize,
        transcript: String,
        items: Vec<ExpectedItem>,
    }

    #[derive(Deserialize)]
    struct ExpectedItem {
        text: String,
        start_time: f32,
        end_time: f32,
    }

    #[test]
    fn profile_and_timestamp_config_parse_the_pinned_architecture() {
        let root = json!({
            "model_type": "qwen3_asr",
            "quantization_config": {"bits": 8, "group_size": 64, "mode": "affine"},
            "timestamp_token_id": 151705,
            "timestamp_segment_time": 80,
            "thinker_config": {
                "model_type": "qwen3_forced_aligner",
                "audio_token_id": 151676,
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670,
                "classify_num": 5000,
                "audio_config": {
                    "num_mel_bins": 128, "encoder_layers": 24,
                    "encoder_attention_heads": 16, "encoder_ffn_dim": 4096,
                    "d_model": 1024, "max_source_positions": 1500,
                    "n_window": 50, "output_dim": 1024,
                    "n_window_infer": 800, "downsample_hidden_size": 480,
                    "scale_embedding": false
                },
                "text_config": {
                    "vocab_size": 152064, "hidden_size": 1024,
                    "intermediate_size": 3072, "num_hidden_layers": 28,
                    "num_attention_heads": 16, "num_key_value_heads": 8,
                    "head_dim": 128, "rms_norm_eps": 0.000001,
                    "rope_theta": 1000000, "tie_word_embeddings": false
                }
            }
        });
        let config = super::Qwen3ForcedAlignerConfig::from_json(&root).unwrap();
        assert_eq!(config.timestamp_token_id, 151705);
        assert_eq!(config.timestamp_segment_time_ms, 80.0);
        assert_eq!(config.classify_num, 5000);
        assert!(!config.base.text.tie_word_embeddings);
        assert_eq!(
            QWEN3_FORCED_ALIGNER_06B_8BIT.revision,
            "0e1a68e91d815300c7c9754b2a7639378b23db15"
        );
    }

    #[test]
    fn english_and_chinese_transcripts_match_reference_word_segmentation() {
        assert_eq!(
            tokenize_alignment_text("Hello, world! can't 42", "English").unwrap(),
            ["Hello", "world", "can't", "42"]
        );
        assert_eq!(
            tokenize_alignment_text("你好 world!", "Chinese").unwrap(),
            ["你", "好", "world"]
        );
        assert!(is_cjk_char('你'));
        assert!(!is_cjk_char('a'));
        assert!(tokenize_alignment_text("こんにちは", "Japanese").is_err());
    }

    #[test]
    fn timestamp_repair_matches_reference_for_short_and_long_anomalies() {
        assert_eq!(fix_timestamps(&[0.0, 80.0, 160.0]), [0.0, 80.0, 160.0]);
        assert_eq!(
            fix_timestamps(&[0.0, 160.0, 80.0, 240.0]),
            [0.0, 160.0, 160.0, 240.0]
        );
        assert_eq!(
            fix_timestamps(&[0.0, 500.0, 400.0, 300.0, 200.0, 100.0, 600.0]),
            [0.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0]
        );
        assert!(fix_timestamps(&[]).is_empty());
    }

    #[test]
    #[ignore = "requires the pinned Qwen3-ForcedAligner checkpoint"]
    fn pinned_checkpoint_matches_mlx_word_alignment_fixture() {
        let model_dir = std::env::var_os("TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR")
            .expect("set TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR to the pinned checkpoint");
        let model = Qwen3ForcedAligner::load(std::path::Path::new(&model_dir)).unwrap();
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../testdata/qwen3_forced_aligner.json"))
                .unwrap();
        let audio = turbospark_audio::wav::read_wav_f32_bytes(include_bytes!(
            "../../../testdata/qwen3_forced_aligner_reference.wav"
        ))
        .unwrap();
        assert_eq!(audio.sample_rate, fixture.sample_rate);
        assert_eq!(audio.samples.len(), fixture.sample_count);

        let result = model.align(&audio.samples, &fixture.transcript).unwrap();
        assert_eq!(result.items.len(), fixture.items.len());
        for (actual, expected) in result.items.iter().zip(fixture.items) {
            assert_eq!(actual.text, expected.text);
            assert!((actual.start_time - expected.start_time).abs() <= 0.001);
            assert!((actual.end_time - expected.end_time).abs() <= 0.001);
        }
    }
}
