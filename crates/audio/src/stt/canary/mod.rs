//! Canary-1B-v2 multilingual ASR and translation.
//!
//! Reference: `mlx_audio/stt/models/canary/` (canary.py, config.py,
//! decoder.py, tokenizer.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The encoder reuses the
//! Parakeet FastConformer (`crate::vad::sortformer::FastConformer`), which
//! is exactly what the reference does when it imports the Parakeet
//! `Conformer`; the NeMo log-mel frontend (per-feature normalization,
//! preemph 0.97) comes from `turbospark_audio::nemo_mel`. This module adds
//! the transformer decoder with its fixed positional table, the prompt
//! builder, the embedded SentencePiece vocabulary, and the greedy decode
//! loop. See README.md for the pinned profile and verification evidence.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nemo_mel::nemo_log_mel_spectrogram;
use crate::nn::argmax;
use crate::vad::sortformer::FastConformer;
use crate::{Result, SpeechError};

mod config;
mod decoder;
mod tokenizer;

pub use config::{CanaryConfig, DecoderConfig};
pub use decoder::Cache as CanaryCache;
pub use tokenizer::CanaryTokenizer;

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanaryProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned, smoke-verified profile: FastConformer encoder (32 layers,
/// d_model 1024) plus an 8-layer transformer decoder, 8-bit groupwise
/// affine quantization, tokenizer embedded in `config.json`.
pub const CANARY_1B_V2_Q8: CanaryProfile = CanaryProfile {
    name: "Canary-1B-v2 8-bit",
    repository: "Mediform/canary-1b-v2-mlx-q8",
    revision: "0b6b32ee10f30c89e3ead7249bb636445e3019ee",
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

/// The prefill logits summary: the argmax (the first greedy token) plus the
/// top-16 token ids and values (the fixture records the reference top-8,
/// which must survive inside this port's top-16 under bf16-vs-f32 drift).
#[derive(Debug, Clone, PartialEq)]
pub struct FirstLogits {
    pub argmax: u32,
    pub argmax_value: f32,
    pub top16: Vec<(u32, f32)>,
}

/// Stages captured by a checkpoint-gated transcription.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscribeStages {
    pub mel: Option<StageWitness>,
    pub encoder_output: Option<StageWitness>,
    pub prompt_token_ids: Vec<i32>,
    pub prefill_last_hidden: Option<Vec<f32>>,
    pub first_logits: Option<FirstLogits>,
    pub generated_token_ids: Vec<u32>,
}

/// Loaded Canary model: FastConformer encoder, transformer decoder, and the
/// vocabulary embedded in `config.json`.
pub struct Canary {
    config: CanaryConfig,
    encoder: FastConformer,
    decoder: decoder::CanaryDecoder,
    positional: Vec<f32>,
    tokenizer: CanaryTokenizer,
}

impl Canary {
    /// Loads one local checkpoint directory holding `config.json` and
    /// `model.safetensors`. Tests never download.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json: Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| {
                SpeechError::BadConfig {
                    field: config_path.display().to_string(),
                    why: format!("cannot read: {error}"),
                }
            })?)
            .map_err(|error| SpeechError::BadConfig {
                field: "config.json".into(),
                why: error.to_string(),
            })?;
        let config = CanaryConfig::from_json(&config_json)?;
        let tokenizer = CanaryTokenizer::from_config(&config_json)?;
        if tokenizer.vocab_size() != config.vocab_size {
            return Err(SpeechError::BadConfig {
                field: "tokenizer".into(),
                why: format!(
                    "embedded vocabulary holds {} pieces, config says {}",
                    tokenizer.vocab_size(),
                    config.vocab_size
                ),
            });
        }
        if config.vocab_size > i32::MAX as usize {
            return Err(SpeechError::BadConfig {
                field: "vocab_size".into(),
                why: "must fit signed 32-bit token ids".into(),
            });
        }
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let encoder = FastConformer::load_canary(&weights, &config.encoder, config.quant)?;
        let decoder = decoder::CanaryDecoder::load(
            &weights,
            &config.decoder,
            config.vocab_size,
            config.quant,
        )?;
        if decoder.vocab_size() != config.vocab_size {
            return Err(SpeechError::BadConfig {
                field: "head.classifier".into(),
                why: "output projection width disagrees with the vocabulary".into(),
            });
        }
        let positional = decoder::fixed_positional_encoding(
            config.decoder.hidden_size,
            config.decoder.max_sequence_length,
        );
        Ok(Self {
            config,
            encoder,
            decoder,
            positional,
            tokenizer,
        })
    }

    pub fn profile(&self) -> CanaryProfile {
        CANARY_1B_V2_Q8
    }

    pub fn config(&self) -> &CanaryConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &CanaryTokenizer {
        &self.tokenizer
    }

    /// Transcribes mono 16 kHz PCM with the reference defaults
    /// (source_lang "en", target_lang "en", punctuation and capitalization
    /// on, greedy, at most 200 tokens like `Model.generate`).
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, "en", "en", true, 200)
    }

    /// Transcribes or translates one mono 16 kHz clip. The prompt mirrors
    /// `CanaryTokenizer.build_prompt_tokens`; `max_tokens` bounds greedy
    /// decoding exactly like `Model.generate` (where `max_tokens = 0` still
    /// yields the first sampled token).
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        source_lang: &str,
        target_lang: &str,
        use_pnc: bool,
        max_tokens: usize,
    ) -> Result<String> {
        self.transcribe_impl(samples, source_lang, target_lang, use_pnc, max_tokens, None)
            .map(|(text, _)| text)
    }

    /// Transcription with optional stage capture for checkpoint-gated
    /// parity tests.
    pub fn transcribe_impl(
        &self,
        samples: &[f32],
        source_lang: &str,
        target_lang: &str,
        use_pnc: bool,
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<(String, Vec<u32>)> {
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "Canary audio must be non-empty and finite".into(),
            });
        }
        let prompt = self
            .tokenizer
            .build_prompt_tokens(source_lang, target_lang, use_pnc)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = prompt.clone();
        }

        // Log-mel frontend plus FastConformer encoder, batch one, full
        // length: the reference computes lengths from the unpadded mel and
        // the resulting mask covers every frame.
        let mel = nemo_log_mel_spectrogram(samples, &self.config.mel)?;
        let mel_frames = mel.len();
        let mel_width = self.config.encoder.num_mel_bins;
        let mut channel_major = vec![0.0f32; mel_width * mel_frames];
        for (t, row) in mel.iter().enumerate() {
            if row.len() != mel_width {
                return Err(SpeechError::Audio(
                    "unexpected NeMo mel feature width".into(),
                ));
            }
            for (m, &value) in row.iter().enumerate() {
                channel_major[m * mel_frames + t] = value;
            }
        }
        if let Some(stages) = stages.as_deref_mut() {
            stages.mel = Some(witness(
                "mel",
                &[mel_frames, mel_width],
                &{
                    let mut flat = Vec::with_capacity(mel_frames * mel_width);
                    for row in &mel {
                        flat.extend_from_slice(row);
                    }
                    flat
                },
                &[],
                &[],
            ));
        }
        let (encoder_output, encoder_len) = self.encoder.encode_mel(
            &channel_major,
            mel_frames,
            mel_frames,
            &self.config.encoder,
        )?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.encoder_output = Some(witness(
                "encoder_output",
                &[encoder_len, self.config.encoder.hidden_size],
                &encoder_output,
                &[],
                &[],
            ));
        }

        let mut cache = self.decoder.new_cache(&encoder_output, encoder_len)?;
        let (logits, hidden) = self
            .decoder
            .forward(&prompt, 0, &mut cache, &self.positional)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden =
                Some(hidden[(hidden.len() - self.config.decoder.hidden_size)..].to_vec());
        }
        let vocab = self.decoder.vocab_size();
        let last_row = logits.len() - vocab;
        let first_logits = top_logits(&logits[last_row..]);
        let mut generated: Vec<u32> = Vec::new();
        let eos = self.tokenizer.eos_id();
        let mut next = first_logits.argmax;
        if next as i32 != eos {
            generated.push(next);
            // The reference loops `range(max_tokens - 1)` after pushing the
            // first token, so `max_tokens = 0` still yields one token.
            for step in 0..max_tokens.saturating_sub(1) {
                let token = next as i32;
                let (step_logits, _) = self.decoder.forward(
                    &[token],
                    prompt.len() + step,
                    &mut cache,
                    &self.positional,
                )?;
                next = argmax(&step_logits[step_logits.len() - vocab..]) as u32;
                if next as i32 == eos {
                    break;
                }
                generated.push(next);
            }
        }
        if let Some(stages) = stages {
            stages.first_logits = Some(first_logits);
            stages.generated_token_ids = generated.clone();
        }
        let text = self.tokenizer.decode(&generated)?;
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
    use super::*;

    const TOLERANCE_MEL: f32 = 2.0e-4;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../testdata/canary_reference.json")).unwrap()
    }

    fn smoke_samples() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = crate::wav::read_wav_f32(&path).expect("reference WAV loads");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        audio.samples
    }

    fn witness_max_diff(actual: &StageWitness, expected: &serde_json::Value) -> f32 {
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
        max_abs
    }

    fn compare_witness(actual: &StageWitness, expected: &serde_json::Value, tolerance: f32) -> f32 {
        let max_abs = witness_max_diff(actual, expected);
        assert!(
            max_abs <= tolerance,
            "stage {} maximum absolute difference {max_abs} exceeds {tolerance}",
            actual.name
        );
        max_abs
    }

    #[test]
    fn profile_is_pinned() {
        assert_eq!(CANARY_1B_V2_Q8.repository, "Mediform/canary-1b-v2-mlx-q8");
        assert_eq!(
            CANARY_1B_V2_Q8.revision,
            "0b6b32ee10f30c89e3ead7249bb636445e3019ee"
        );
        assert_eq!(CANARY_1B_V2_Q8.revision.len(), 40);
    }

    #[test]
    fn pinned_config_fixture_parses_with_pinned_dimensions() {
        let fixture = fixture();
        let config = CanaryConfig::from_json(&fixture["config"]).unwrap();
        assert_eq!(config.encoder.num_hidden_layers, 32);
        assert_eq!(config.encoder.hidden_size, 1024);
        assert_eq!(config.encoder.num_attention_heads, 8);
        assert_eq!(config.encoder.intermediate_size, 4096);
        assert_eq!(config.encoder.conv_kernel_size, 9);
        assert_eq!(config.encoder.num_mel_bins, 128);
        assert!(config.encoder.attention_bias);
        assert!(!config.encoder.scale_input);
        assert_eq!(config.decoder.num_layers, 8);
        assert_eq!(config.decoder.hidden_size, 1024);
        assert_eq!(config.decoder.num_attention_heads, 8);
        assert_eq!(config.decoder.inner_size, 4096);
        assert_eq!(config.decoder.max_sequence_length, 1024);
        assert_eq!(config.vocab_size, 16384);
        assert_eq!(config.quant.bits, 8);
        assert_eq!(config.quant.group_size, 64);
        assert_eq!(config.mel.sample_rate, 16_000);
        assert_eq!(config.mel.num_mels, 128);
        assert_eq!(config.mel.preemphasis, 0.97);
    }

    #[test]
    fn config_refuses_unverified_graphs() {
        let base = fixture()["config"].clone();
        let mut config = base.clone();
        config["quantization"]["bits"] = serde_json::json!(4);
        assert!(CanaryConfig::from_json(&config).is_err(), "4-bit refuses");
        let mut config = base.clone();
        config["quantization"]["group_size"] = serde_json::json!(32);
        assert!(
            CanaryConfig::from_json(&config).is_err(),
            "group size 32 refuses"
        );
        let mut config = base.clone();
        config["encoder"]["causal_downsampling"] = serde_json::json!(true);
        assert!(
            CanaryConfig::from_json(&config).is_err(),
            "causal downsampling refuses"
        );
        let mut config = base.clone();
        config["encoder"]["self_attention_model"] = serde_json::json!("rel_pos_local");
        assert!(CanaryConfig::from_json(&config).is_err());
        let mut config = base.clone();
        config["transf_decoder"]["hidden_act"] = serde_json::json!("gelu");
        assert!(CanaryConfig::from_json(&config).is_err());
        let mut config = base.clone();
        config["transf_decoder"]["pre_ln"] = serde_json::json!(false);
        assert!(CanaryConfig::from_json(&config).is_err());
        let mut config = base.clone();
        config["enc_output_dim"] = serde_json::json!(512);
        assert!(
            CanaryConfig::from_json(&config).is_err(),
            "encoder projection geometry refuses"
        );
        let mut config = base.clone();
        config["head"]["num_classes"] = serde_json::json!(16000);
        assert!(CanaryConfig::from_json(&config).is_err());
        let mut config = base.clone();
        config["preprocessor"]["sample_rate"] = serde_json::json!(8000);
        assert!(CanaryConfig::from_json(&config).is_err());
        let mut config = base.clone();
        config["preprocessor"]["normalize"] = serde_json::json!("NA");
        assert!(CanaryConfig::from_json(&config).is_err());
    }

    #[test]
    fn tokenizer_matches_the_pinned_reference() {
        let fixture = fixture();
        let tokenizer = CanaryTokenizer::from_config(&fixture).unwrap();
        let expected: serde_json::Map<String, serde_json::Value> = fixture["tokenizer"]
            ["special_token_ids"]
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(tokenizer.vocab_size(), fixture["tokenizer"]["piece_count"]);
        for (token, id) in expected {
            assert_eq!(
                tokenizer.token_to_id(&token),
                Some(id.as_i64().unwrap() as i32),
                "special token {token}"
            );
        }
        let ids: Vec<usize> =
            serde_json::from_value(fixture["tokenizer"]["selected_piece_ids"].clone()).unwrap();
        let pieces: Vec<String> =
            serde_json::from_value(fixture["tokenizer"]["selected_pieces"].clone()).unwrap();
        for (id, piece) in ids.iter().zip(&pieces) {
            assert_eq!(tokenizer.piece(*id), piece);
        }
    }

    #[test]
    fn prompts_and_decode_match_the_pinned_reference() {
        let fixture = fixture();
        let tokenizer = CanaryTokenizer::from_config(&fixture).unwrap();
        let prompt: Vec<i32> =
            serde_json::from_value(fixture["decoder"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(
            tokenizer.build_prompt_tokens("en", "en", true).unwrap(),
            prompt
        );
        let second: Vec<i32> =
            serde_json::from_value(fixture["tokenizer"]["prompt_de_fr_nopnc"].clone()).unwrap();
        assert_eq!(
            tokenizer.build_prompt_tokens("de", "fr", false).unwrap(),
            second
        );
        assert!(tokenizer.build_prompt_tokens("xx", "en", true).is_err());
        let generated: Vec<u32> =
            serde_json::from_value(fixture["decoder"]["generated_token_ids"].clone()).unwrap();
        let pieces: Vec<String> =
            serde_json::from_value(fixture["tokenizer"]["generated_pieces"].clone()).unwrap();
        for (id, piece) in generated.iter().zip(&pieces) {
            assert_eq!(tokenizer.piece(*id as usize), piece);
        }
        assert_eq!(
            tokenizer.decode(&generated).unwrap(),
            fixture["decoder"]["transcript"].as_str().unwrap()
        );
        assert_eq!(tokenizer.eos_id(), 3);
    }

    #[test]
    fn fixed_positional_encoding_matches_the_pinned_reference() {
        let fixture = fixture();
        let config = CanaryConfig::from_json(&fixture["config"]).unwrap();
        let table = decoder::fixed_positional_encoding(
            config.decoder.hidden_size,
            config.decoder.max_sequence_length,
        );
        let d = config.decoder.hidden_size;
        let values: Vec<f64> =
            serde_json::from_value(fixture["decoder"]["decoder_pe_values"][0].clone()).unwrap();
        let spots = [
            (0usize, 0usize),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (1023, 0),
            (1023, 1),
        ];
        for ((row, column), expected) in spots.iter().zip(&values) {
            let actual = table[row * d + column];
            assert!(
                (actual - *expected as f32).abs() <= 1.0e-6,
                "decoder pe[{row},{column}]: {actual} vs {expected}"
            );
        }
        // The encoder relative-position slice is pinned by the fixture too;
        // the port computes the same table on the fly inside the shared
        // encoder, so mirror that computation here.
        let encoder_frames = fixture["encoding"]["encoder_frames"].as_u64().unwrap() as usize;
        let slice = relative_position_slice(encoder_frames, d);
        let rows: Vec<usize> =
            serde_json::from_value(fixture["decoder"]["encoder_pe_rows"].clone()).unwrap();
        let expected_rows: Vec<Vec<f64>> =
            serde_json::from_value(fixture["decoder"]["encoder_pe_values"].clone()).unwrap();
        for (&row, expected) in rows.iter().zip(&expected_rows) {
            for (column, expected) in expected.iter().enumerate() {
                let actual = slice[row * d + column];
                assert!(
                    (actual - *expected as f32).abs() <= 1.0e-5,
                    "encoder pe slice[{row},{column}]: {actual} vs {expected}"
                );
            }
        }
    }

    /// The `[2 * seq - 1, d_model]` relative-position table the shared
    /// encoder consumes, positions counting down from `seq - 1`.
    fn relative_position_slice(seq: usize, d_model: usize) -> Vec<f32> {
        let n = 2 * seq - 1;
        let scale = -(10_000f64.ln() / d_model as f64) as f32;
        let div_term: Vec<f32> = (0..d_model / 2)
            .map(|j| ((2 * j) as f32 * scale).exp())
            .collect();
        let mut pe = vec![0.0f32; n * d_model];
        for i in 0..n {
            let pos = (seq as isize - 1 - i as isize) as f32;
            for (j, div) in div_term.iter().enumerate() {
                let angle = pos * div;
                pe[i * d_model + 2 * j] = angle.sin();
                pe[i * d_model + 2 * j + 1] = angle.cos();
            }
        }
        pe
    }

    /// The always-on frontend gate: the Rust log-mel must reproduce the
    /// reference mel on the shared smoke clip before any weight is loaded.
    #[test]
    fn mel_frontend_matches_the_pinned_reference() {
        let fixture = fixture();
        let config = CanaryConfig::from_json(&fixture["config"]).unwrap();
        let mel = nemo_log_mel_spectrogram(&smoke_samples(), &config.mel).unwrap();
        let expected_frames = fixture["encoding"]["mel_frames"].as_u64().unwrap() as usize;
        assert_eq!(mel.len(), expected_frames);
        let mut flat = Vec::with_capacity(mel.len() * config.mel.num_mels);
        for row in &mel {
            flat.extend_from_slice(row);
        }
        let witness = witness("mel", &[mel.len(), config.mel.num_mels], &flat, &[], &[]);
        let max = compare_witness(&witness, &fixture["encoding"]["mel"], TOLERANCE_MEL);
        eprintln!("canary mel parity: max abs diff {max:.3e}");
    }

    #[test]
    #[ignore = "requires the pinned Canary checkpoint in TURBOSPARK_CANARY_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_CANARY_MODEL_DIR")
            .expect("set TURBOSPARK_CANARY_MODEL_DIR to the pinned local snapshot");
        let model = Canary::load(Path::new(&model_dir)).expect("checkpoint loads");
        assert_eq!(model.profile(), CANARY_1B_V2_Q8);
        let samples = smoke_samples();
        let mut stages = TranscribeStages::default();
        let started = std::time::Instant::now();
        let (text, _) = model
            .transcribe_impl(&samples, "en", "en", true, 96, Some(&mut stages))
            .expect("checkpoint transcribes");
        eprintln!(
            "canary checkpoint transcribe took {:?} (CPU f32)",
            started.elapsed()
        );
        let fixture = fixture();
        assert_eq!(
            text,
            fixture["decoder"]["transcript"].as_str().unwrap(),
            "the pinned transcript must match the mlx-audio reference exactly"
        );

        let expected_tokens: Vec<u32> =
            serde_json::from_value(fixture["decoder"]["generated_token_ids"].clone()).unwrap();
        assert_eq!(stages.generated_token_ids, expected_tokens);
        let expected_prompt: Vec<i32> =
            serde_json::from_value(fixture["decoder"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);

        let mel_max = compare_witness(
            stages.mel.as_ref().unwrap(),
            &fixture["encoding"]["mel"],
            TOLERANCE_MEL,
        );
        // The reference runs the encoder in bfloat16 (the mel is cast and
        // every quantized layer dequantizes to bf16); this port computes the
        // dequantized f32 pipeline, so the honest gate is the witness
        // difference relative to the witness scale.
        let encoder_max = witness_max_diff(
            stages.encoder_output.as_ref().unwrap(),
            &fixture["encoding"]["encoder_output"],
        );
        let encoder_scale: f64 = serde_json::from_value::<Vec<Vec<f64>>>(
            fixture["encoding"]["encoder_output"]["values"].clone(),
        )
        .unwrap()
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()));
        let encoder_relative = encoder_max as f64 / encoder_scale;
        assert!(
            encoder_relative <= 2.0e-2,
            "encoder output relative difference {encoder_relative:.3e} exceeds 2% of the \
             witness scale {encoder_scale}"
        );

        let expected_hidden: Vec<f32> =
            serde_json::from_value(fixture["decoder"]["prefill_last_hidden"].clone()).unwrap();
        let hidden = stages.prefill_last_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), expected_hidden.len());
        let hidden_max = hidden
            .iter()
            .zip(&expected_hidden)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let expected_max = expected_hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            hidden_max <= 2.0e-2 * expected_max,
            "prefill hidden difference {hidden_max} exceeds 2% of the row scale {expected_max}"
        );
        let hidden_relative = hidden_max / expected_max;

        let first = stages.first_logits.as_ref().unwrap();
        let expected_argmax = fixture["decoder"]["first_logits"]["argmax"]
            .as_u64()
            .unwrap() as u32;
        assert_eq!(first.argmax, expected_argmax, "first greedy token id");
        let expected_top8: Vec<(u32, f64)> =
            serde_json::from_value(fixture["decoder"]["first_logits"]["top8"].clone()).unwrap();
        let expected_top8_max = expected_top8
            .iter()
            .map(|(_, v)| v.abs())
            .fold(0.0f64, f64::max);
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
            "canary checkpoint parity: mel max {mel_max:.3e}, encoder output max \
             {encoder_max:.3e} (relative {encoder_relative:.3e}), prefill hidden max \
             {hidden_max:.3e} (relative {hidden_relative:.3e}), top-8 logit max \
             {top_logit_max:.3e} (relative {top8_relative:.3e})"
        );
    }
}
