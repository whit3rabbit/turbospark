//! Phonon-1: Fermion Research's English speech-to-text family.
//!
//! Reference: `mlx_audio/stt/models/phonon/` (phonon.py, packed.py,
//! transport.py, config.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The architecture is
//! Qwen3-ASR: the 128-band log-mel frontend, chunked audio tower, text
//! decoder, tokenizer loader, prompt construction, and greedy decode are
//! reused from `crate::stt::qwen3_asr`. This module adds the Phonon
//! detection and materialization contract, the packed five-value decoder
//! linears ([`packed`]), the strict manifest parsing ([`config`]), and the
//! temporary f32 checkpoint conversion ([`materialize`]). See README.md for
//! the pinned profile and verification evidence.

use std::path::Path;

use turbospark_tokenizer::Tokenizer;

use crate::stt::qwen3_asr::decoder::{chat_stop_ids, greedy_generate, is_chat_stop, Decoder};
use crate::stt::qwen3_asr::encoder::AudioEncoder;
use crate::stt::qwen3_asr::frontend::compute_features;
use crate::stt::qwen3_asr::{load_tokenizer, prompt_token_ids, transcript_from_tokens};
use crate::{Result, SpeechError};

pub mod config;
pub mod materialize;
pub mod packed;

pub use config::{PhononConfig, PhononManifest};

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhononProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned release profile. The archive carries
/// `phonon-audio6.bps.tar.zst` (415,077,202 bytes, SHA-256
/// `214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2`) and
/// must be materialized with the mlx-audio reference before loading; see
/// README.md for the full artifact table.
pub const PHONON_1: PhononProfile = PhononProfile {
    name: "Phonon-1",
    repository: "FermionResearch/Phonon-1",
    revision: "0428da04625c51b6f069a9829c7060e6b167b92a",
};

/// One sparse stage tensor witness: the row-major `[shape]` values at the
/// recorded row and column indices, matching the fixture generator's
/// `selected_rows`.
#[derive(Debug, Clone, PartialEq)]
pub struct StageWitness {
    pub name: String,
    pub shape: Vec<usize>,
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub values: Vec<Vec<f32>>,
}

/// The first decoded-step logits summary: the argmax plus the top-16 token
/// ids and values (the fixture records the reference top-8, which must
/// survive inside this port's top-16 under bf16-vs-f32 drift).
#[derive(Debug, Clone, PartialEq)]
pub struct FirstLogits {
    pub argmax: u32,
    pub argmax_value: f32,
    pub top16: Vec<(u32, f32)>,
}

/// Stages captured by a checkpoint-gated transcription.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscribeStages {
    pub input_features: Option<StageWitness>,
    pub audio_embeddings: Option<StageWitness>,
    pub prompt_token_ids: Vec<i32>,
    pub prefill_last_hidden: Option<Vec<f32>>,
    pub first_logits: Option<FirstLogits>,
    pub generated_token_ids: Vec<u32>,
}

/// Loaded Phonon-1 model: Qwen3-ASR backbone with packed five-value decoder
/// linears materialized to f32 at load.
pub struct Phonon {
    config: PhononConfig,
    encoder: AudioEncoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
}

impl Phonon {
    /// Loads one materialized local checkpoint directory. Tests never
    /// download; the un-materialized release archive is refused with
    /// materialization instructions (see `PhononConfig::load`).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = PhononConfig::load(model_dir)?;
        let shard = config
            .manifest
            .shards
            .first()
            .ok_or_else(|| SpeechError::BadConfig {
                field: "packed_manifest.shards".into(),
                why: "the manifest lists no weight shard".into(),
            })?
            .clone();
        if config.manifest.shards.len() != 1 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "this port verifies the single-shard layout only; the manifest lists {} shards",
                    config.manifest.shards.len()
                ),
            });
        }
        let weights = materialize::open_shard(model_dir, &shard)?;
        let converted = materialize::convert_checkpoint(&weights, &config)?;
        let converted_path = materialize::temporary_path()?;
        if let Err(error) = materialize::write_safetensors(&converted_path, &converted) {
            let _ = std::fs::remove_file(&converted_path);
            return Err(error);
        }
        drop(converted);
        let loaded = Self::open_with_config(&converted_path, model_dir, config);
        match loaded {
            Ok(model) => {
                std::fs::remove_file(&converted_path).map_err(|error| SpeechError::BadConfig {
                    field: "temporary converted checkpoint".into(),
                    why: format!("failed to remove {}: {error}", converted_path.display()),
                })?;
                Ok(model)
            }
            Err(error) => {
                let _ = std::fs::remove_file(&converted_path);
                Err(error)
            }
        }
    }

    fn open_with_config(
        converted_path: &Path,
        model_dir: &Path,
        config: PhononConfig,
    ) -> Result<Self> {
        let (encoder, decoder) = materialize::load_models(converted_path, &config)?;
        let tokenizer = load_tokenizer(model_dir)?;
        for (token, expected) in [
            ("<|audio_pad|>", config.backbone.audio_token_id),
            ("<|audio_start|>", config.backbone.audio_start_token_id),
            ("<|audio_end|>", config.backbone.audio_end_token_id),
        ] {
            if tokenizer.token_to_id(token).map(|id| id as i32) != Some(expected) {
                return Err(SpeechError::BadConfig {
                    field: "tokenizer.json".into(),
                    why: format!("{token} id does not match config.json"),
                });
            }
        }
        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn profile(&self) -> PhononProfile {
        PHONON_1
    }

    pub fn config(&self) -> &PhononConfig {
        &self.config
    }

    /// Transcribes mono PCM already sampled at 16 kHz with greedy decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, None, 512)
    }

    /// Transcribes one mono 16 kHz clip, optionally prompting the supported
    /// output language ("English"; the released checkpoints are English
    /// only). `max_tokens` bounds greedy decoding. The prompt and chat
    /// template match the reference exactly (see
    /// `qwen3_asr::prompt_token_ids`).
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        self.transcribe_impl(samples, language, max_tokens, None)
            .map(|(text, _)| text)
    }

    /// Transcription with optional stage capture for checkpoint-gated parity
    /// tests. The greedy loop mirrors the reference
    /// `PhononASRModel.stream_generate`: prefill once, take the first token
    /// from the last prompt position, then step token by token with an EOS
    /// check before each emission.
    pub(crate) fn transcribe_impl(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<(String, Vec<u32>)> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Phonon max_tokens must be positive".into(),
            });
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "Phonon audio must be non-empty and finite".into(),
            });
        }
        let language = language
            .map(|requested| {
                self.config
                    .backbone
                    .supported_languages
                    .iter()
                    .find(|known| known.eq_ignore_ascii_case(requested))
                    .map(String::as_str)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("unsupported Phonon language: {requested}"),
                    })
            })
            .transpose()?;

        let features = compute_features(samples)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.input_features = Some(witness(
                "input_features",
                &[
                    features.values.len() / features.frames.max(1),
                    features.frames,
                ],
                &features.values,
                &[],
                &[],
            ));
        }
        let audio_embeddings = self.encoder.forward(&features)?;
        if audio_embeddings.len() % self.decoder.hidden_size != 0 {
            return Err(SpeechError::Tensor {
                name: "audio_tower.output".into(),
                why: "encoder output is not a whole number of decoder embeddings".into(),
            });
        }
        let audio_rows = audio_embeddings.len() / self.decoder.hidden_size;
        if audio_rows == 0 {
            return Err(SpeechError::Input {
                why: "Phonon audio encoder produced no features".into(),
            });
        }
        if let Some(stages) = stages.as_deref_mut() {
            stages.audio_embeddings = Some(witness(
                "audio_embeddings",
                &[audio_rows, self.decoder.hidden_size],
                &audio_embeddings,
                &[],
                &[],
            ));
        }

        let (token_ids, audio_positions) =
            prompt_token_ids(&self.config.backbone, &self.tokenizer, audio_rows, language)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = token_ids.clone();
        }
        let mut embeddings = self.decoder.embed(&token_ids)?;
        for (audio_row, &position) in audio_positions.iter().enumerate() {
            let target = position * self.decoder.hidden_size;
            let source = audio_row * self.decoder.hidden_size;
            embeddings[target..target + self.decoder.hidden_size]
                .copy_from_slice(&audio_embeddings[source..source + self.decoder.hidden_size]);
        }

        let stop_ids = chat_stop_ids(&self.tokenizer);
        let rows = embeddings.len() / self.decoder.hidden_size;
        let (last_hidden, cache) = self.decoder.prefill(&embeddings, rows);
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden = Some(last_hidden.clone());
        }
        let logits = self.decoder.logits(&last_hidden);
        // Recorded only when the loop is entered, as before.
        let first_logits = (max_tokens > 0).then(|| top_logits(&logits));
        let generated = greedy_generate(
            &self.decoder,
            &logits,
            cache,
            max_tokens,
            |next| is_chat_stop(&stop_ids, next),
            "Phonon",
            |_, _| {},
        )?;
        if let Some(stages) = stages {
            stages.first_logits = first_logits;
            stages.generated_token_ids = generated.clone();
        }

        let text = transcript_from_tokens(&self.tokenizer, &generated, language)?;
        Ok((text, generated))
    }
}

fn top_logits(logits: &[f32]) -> FirstLogits {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    FirstLogits {
        argmax: order[0] as u32,
        argmax_value: logits[order[0]],
        top16: order
            .iter()
            .take(16)
            .map(|&index| (index as u32, logits[index]))
            .collect(),
    }
}

/// Builds the fixture's sparse row/column witness: rows and columns default
/// to the reference's selected set (first two, middle, last) when not given.
fn witness(
    name: &str,
    shape: &[usize],
    values: &[f32],
    rows: &[usize],
    columns: &[usize],
) -> StageWitness {
    let (rows_total, columns_total) = match shape.len() {
        1 => (1usize, shape[0]),
        2 => (shape[0], shape[1]),
        _ => (values.len(), 1),
    };
    let mut row_set: Vec<usize> = if rows.is_empty() {
        vec![
            0,
            1.min(rows_total.saturating_sub(1)),
            rows_total / 2,
            rows_total - 1,
        ]
    } else {
        rows.to_vec()
    };
    row_set.sort_unstable();
    row_set.dedup();
    let mut column_set: Vec<usize> = if columns.is_empty() {
        vec![
            0,
            1.min(columns_total.saturating_sub(1)),
            columns_total / 2,
            columns_total - 1,
        ]
    } else {
        columns.to_vec()
    };
    column_set.sort_unstable();
    column_set.dedup();
    let mut out = Vec::with_capacity(row_set.len());
    for &row in &row_set {
        let base = if shape.len() == 1 {
            0
        } else {
            row * columns_total
        };
        out.push(
            column_set
                .iter()
                .map(|&column| values[base + column])
                .collect(),
        );
    }
    StageWitness {
        name: name.to_owned(),
        shape: shape.to_vec(),
        rows: row_set,
        columns: column_set,
        values: out,
    }
}

#[cfg(test)]
mod tests {
    use super::{Phonon, StageWitness, TranscribeStages};
    use crate::stt::qwen3_asr::frontend::compute_features;
    use serde_json::Value;
    use std::path::Path;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../testdata/phonon_reference.json")).unwrap()
    }

    fn smoke_samples() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = crate::wav::read_wav_f32(&path).expect("reference WAV loads");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        audio.samples
    }

    fn decode_base64(raw: &str) -> Vec<u8> {
        fn value(byte: u8) -> Option<u32> {
            match byte {
                b'A'..=b'Z' => Some((byte - b'A') as u32),
                b'a'..=b'z' => Some((byte - b'a') as u32 + 26),
                b'0'..=b'9' => Some((byte - b'0') as u32 + 52),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        }
        let mut bytes = Vec::with_capacity(raw.len() / 4 * 3);
        let mut buffer = 0u32;
        let mut bits = 0u32;
        for byte in raw.bytes() {
            if let Some(entry) = value(byte) {
                buffer = (buffer << 6) | entry;
                bits += 6;
                if bits >= 8 {
                    bits -= 8;
                    bytes.push(((buffer >> bits) & 0xFF) as u8);
                }
            }
        }
        bytes
    }

    fn compare_witness(actual: &StageWitness, expected: &Value, tolerance: f32) -> f32 {
        let shape: Vec<usize> = serde_json::from_value(expected["shape"].clone()).unwrap();
        let rows: Vec<usize> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let columns: Vec<usize> = serde_json::from_value(expected["columns"].clone()).unwrap();
        let values: Vec<Vec<f32>> = serde_json::from_value(expected["values"].clone()).unwrap();
        assert_eq!(actual.shape, shape, "stage {} shape", actual.name);
        assert_eq!(actual.rows, rows, "stage {} rows", actual.name);
        assert_eq!(actual.columns, columns, "stage {} columns", actual.name);
        let mut max_abs = 0.0f32;
        for (actual_row, expected_row) in actual.values.iter().zip(&values) {
            for (a, e) in actual_row.iter().zip(expected_row) {
                max_abs = max_abs.max((a - e).abs());
            }
        }
        assert!(
            max_abs <= tolerance,
            "stage {} maximum absolute difference {max_abs} exceeds {tolerance}",
            actual.name
        );
        max_abs
    }

    /// Always-on fixture parity for the checkpoint-free stage: the 128-band
    /// log-mel frontend over the shared smoke clip.
    #[test]
    fn frontend_features_match_the_pinned_reference() {
        let fixture = fixture();
        let samples = smoke_samples();
        let features = compute_features(&samples).unwrap();
        let witness = super::witness(
            "input_features",
            &[128, features.frames],
            &features.values,
            &[],
            &[],
        );
        let max_abs = compare_witness(&witness, &fixture["backbone"]["input_features"], 2.0e-4);
        eprintln!("phonon frontend parity: input_features max abs diff {max_abs:.3e}");
    }

    /// Always-on exact goldens: the transport quint5 bytes must unpack to
    /// the reference 2-bit plane words, and the slim metadata must fuse into
    /// the exact reference weight spots under the fixture generator's f32
    /// arithmetic.
    #[test]
    fn quint5_unpack_and_fused_weight_match_the_reference_goldens() {
        let fixture = fixture();
        for module in fixture["quint5"]["selected"].as_array().unwrap() {
            let name = module["name"].as_str().unwrap();
            let in_features = module["in_features"].as_u64().unwrap() as usize;
            let bytes_per_row = module["packed_bytes_per_row"].as_u64().unwrap() as usize;
            let rows: Vec<usize> = serde_json::from_value(module["rows"].clone()).unwrap();
            let packed_rows: Vec<String> =
                serde_json::from_value(module["packed_rows_base64"].clone()).unwrap();
            let base_rows: Vec<Vec<u32>> =
                serde_json::from_value(module["base_words_rows"].clone()).unwrap();
            let residual_rows: Vec<Vec<u32>> =
                serde_json::from_value(module["residual_words_rows"].clone()).unwrap();
            let alpha: Vec<f32> =
                serde_json::from_value(module["base_alpha_values"].clone()).unwrap();
            let residual_scale: f32 =
                serde_json::from_value(module["residual_scale_value"].clone()).unwrap();

            for (index, (&row, packed)) in rows.iter().zip(&packed_rows).enumerate() {
                let bytes = decode_base64(packed);
                assert_eq!(bytes.len(), bytes_per_row, "{name} row {row} byte count");
                let (base, residual) = super::packed::unpack_codes(&bytes, 1, in_features).unwrap();
                assert_eq!(base, base_rows[index], "{name} row {row} base words");
                assert_eq!(
                    residual, residual_rows[index],
                    "{name} row {row} residual words"
                );
            }

            let columns: Vec<usize> =
                serde_json::from_value(module["weight_columns"].clone()).unwrap();
            let expected: Vec<Vec<f64>> =
                serde_json::from_value(module["weight_values"].clone()).unwrap();
            let mut spots = 0usize;
            for (index, &row) in rows.iter().enumerate() {
                let metadata = super::packed::slim_metadata(
                    &[alpha[index]],
                    &[residual_scale],
                    1,
                    in_features / super::packed::GROUP_SIZE,
                )
                .unwrap();
                let weight = super::packed::materialize_weight(
                    &base_rows[index],
                    &residual_rows[index],
                    &metadata,
                    1,
                    in_features,
                )
                .unwrap();
                assert_eq!(weight.len(), in_features);
                for (column, expected_value) in columns.iter().zip(&expected[index]) {
                    assert_eq!(
                        weight[*column] as f64, *expected_value,
                        "{name} row {row} column {column}"
                    );
                    spots += 1;
                }
            }
            eprintln!("phonon quint5 golden {name}: {spots} weight spots exact");
        }
    }

    /// Always-on structural check of the fixture's prompt contract: the
    /// audio placeholder span is uniform and the chat scaffolding matches
    /// the reference prompt construction.
    #[test]
    fn fixture_prompt_has_a_uniform_audio_placeholder_span() {
        let fixture = fixture();
        let prompt: Vec<i32> =
            serde_json::from_value(fixture["backbone"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(prompt[0], 151_644, "<|im_start|> opens the prompt");
        assert_eq!(prompt[3], 151_645, "<|im_end|> closes the system turn");
        assert_eq!(prompt[8], 151_669, "<|audio_start|> opens the audio span");
        let audio_id = 151_676;
        let audio_positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id == audio_id).then_some(position))
            .collect();
        assert_eq!(
            audio_positions.len(),
            fixture["backbone"]["audio_rows"].as_u64().unwrap() as usize
        );
        assert!(
            prompt[audio_positions[0]..audio_positions.last().unwrap() + 1]
                .iter()
                .all(|&id| id == audio_id),
            "the placeholder span is contiguous"
        );
    }

    #[test]
    #[ignore = "requires the materialized Phonon-1 checkpoint in TURBOSPARK_PHONON_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_PHONON_MODEL_DIR")
            .expect("set TURBOSPARK_PHONON_MODEL_DIR to the materialized checkpoint directory");
        let model = Phonon::load(Path::new(&model_dir)).expect("checkpoint loads");
        let samples = smoke_samples();
        let mut stages = TranscribeStages::default();
        let (text, _) = model
            .transcribe_impl(&samples, None, 96, Some(&mut stages))
            .expect("checkpoint transcribes");
        let fixture = fixture();
        assert_eq!(
            text,
            fixture["backbone"]["transcript"].as_str().unwrap(),
            "the pinned transcript must match the mlx-audio reference exactly"
        );

        let expected_tokens: Vec<u32> =
            serde_json::from_value(fixture["backbone"]["generated_token_ids"].clone()).unwrap();
        assert_eq!(stages.generated_token_ids, expected_tokens);
        let expected_prompt: Vec<i32> =
            serde_json::from_value(fixture["backbone"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);

        let features_max = compare_witness(
            stages.input_features.as_ref().unwrap(),
            &fixture["backbone"]["input_features"],
            2.0e-4,
        );
        let embeddings_max = compare_witness(
            stages.audio_embeddings.as_ref().unwrap(),
            &fixture["backbone"]["audio_embeddings"],
            2.0e-4,
        );
        let expected_hidden: Vec<f32> =
            serde_json::from_value(fixture["backbone"]["prefill_last_hidden"].clone()).unwrap();
        let hidden = stages.prefill_last_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), expected_hidden.len());
        let hidden_max = hidden
            .iter()
            .zip(&expected_hidden)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let expected_max = expected_hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // The reference decoder computes in bf16 end to end (the quantized
        // embedding emits bf16 and every packed linear is two bf16-scale
        // quantized matmuls summed at the output); this port computes the
        // materialized f32 pipeline. Drift of a few percent of the row scale
        // through 28 layers is the expected bf16-vs-f32 gap, while the
        // greedy decode stays token-identical.
        assert!(
            hidden_max <= 6.0e-2 * expected_max,
            "prefill hidden difference {hidden_max} exceeds 6% of the row scale {expected_max}"
        );
        let hidden_relative = hidden_max / expected_max;
        let first = stages.first_logits.as_ref().unwrap();
        let expected_argmax = fixture["backbone"]["first_logits"]["argmax"]
            .as_u64()
            .unwrap() as u32;
        assert_eq!(first.argmax, expected_argmax, "first greedy token id");
        let expected_top8: Vec<(u32, f64)> =
            serde_json::from_value(fixture["backbone"]["first_logits"]["top8"].clone()).unwrap();
        let expected_top8_max = expected_top8
            .iter()
            .map(|(_, v)| v.abs())
            .fold(0.0f64, f64::max);
        // The token ids of the reference top-8 must all survive the drift
        // (checked against this port's top-16, since bf16 rounding can
        // reorder near ties), and every top-8 logit value must stay within
        // 5% of the reference.
        let mut top_logit_max = 0.0f64;
        for (expected_id, expected_value) in &expected_top8 {
            let actual_value = first
                .top16
                .iter()
                .find(|(id, _)| id == expected_id)
                .unwrap_or_else(|| {
                    panic!("reference top-8 token {expected_id} left this port's top-16")
                })
                .1 as f64;
            top_logit_max = top_logit_max.max((actual_value - expected_value).abs());
            assert!(
                (actual_value - expected_value).abs() <= 5.0e-2 * expected_value.abs(),
                "top-8 logit for {expected_id}: {actual_value} vs {expected_value}"
            );
        }
        let top8_relative = top_logit_max / expected_top8_max;
        eprintln!(
            "phonon checkpoint parity: input_features max {features_max:.3e}, \
             audio_embeddings max {embeddings_max:.3e}, prefill hidden max {hidden_max:.3e} \
             (relative to row max {hidden_relative:.3e}), top-8 logit max {top_logit_max:.3e} \
             (relative {top8_relative:.3e})"
        );
    }
}
