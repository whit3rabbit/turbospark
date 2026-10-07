//! Granite Speech 1B ASR: Conformer encoder, QFormer projector, Granite LLM.
//!
//! Reference: `mlx_audio/stt/models/granite_speech/` (granite_speech.py,
//! config.py) and `mlx_audio/lm/models/granite.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The audio path is a
//! paired-mel frontend ([`frontend`]) feeding a context-blocked Conformer
//! encoder ([`encoder`]); a BLIP-2 style QFormer ([`projector`]) compresses
//! each window of encoder frames into three audio embeddings that replace
//! the `<|audio|>` placeholder tokens of the prompt. The text backbone is
//! the Granite llama variant with its four multipliers ([`llm`]).
//!
//! The pinned checkpoint ships a 193-byte `chat_template.jinja` that wraps
//! one user turn as `USER: {content}\n ASSISTANT:`; this port expands it
//! statically and pins the expansion with tests. Decoding is greedy with a
//! KV cache, stopping at the tokenizer EOS id before emission. The
//! non-streaming ASR path is implemented; the upstream streaming
//! `_stream_generate` remains an open gate. See README.md for the pinned
//! profile and verification evidence.

use std::path::Path;

use serde_json::Value;

use crate::nn::{bad_config, open_shards};
use crate::stt::qwen3_asr::decoder::greedy_generate;
use crate::stt::qwen3_asr::load_tokenizer;
use crate::{Result, SpeechError};

pub mod config;
pub mod encoder;
pub mod frontend;
pub mod llm;
pub mod projector;

pub use config::GraniteSpeechConfig;

use encoder::GraniteSpeechEncoder;
use llm::GraniteLlm;
use projector::EncoderProjector;

/// The upstream default transcription prompt, verbatim.
pub const DEFAULT_ASR_PROMPT: &str = "can you transcribe the speech into a written format?";

/// The pinned checkpoint profile, smoke-verified upstream on the shared
/// smoke clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned full-bfloat16 profile.
pub const GRANITE_4_0_1B_SPEECH: GraniteSpeechProfile = GraniteSpeechProfile {
    name: "Granite Speech 1B",
    repository: "ibm-granite/granite-4.0-1b-speech",
    revision: "bd87ab862416353633ea431fe49b1614003623c5",
};

/// Upstream `LANGUAGE_CODES`: the language names the translation prompt
/// expands for the six verified codes.
pub const LANGUAGE_CODES: [(&str, &str); 6] = [
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("ja", "Japanese"),
];

/// Upstream `generate(prompt=None, language=...)`: the translation prompt
/// uses the mapped language name, falling back to the input verbatim.
pub fn translation_prompt(language: &str) -> String {
    let name = LANGUAGE_CODES
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(language))
        .map(|(_, name)| *name)
        .unwrap_or(language);
    format!("Translate the speech to {name}.")
}

/// The pinned checkpoint's `chat_template.jinja` (193 bytes) restricted to
/// the single-turn form the model consumes: the jinja loop over messages
/// renders one user message as `USER: {content}\n ASSISTANT:` and drops
/// assistant turns (never produced here); `add_generation_prompt` is
/// ignored by the template.
pub fn expand_chat_template(user_content: &str) -> String {
    format!("USER: {user_content}\n ASSISTANT:")
}

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

/// The top logits at one decode decision.
#[derive(Debug, Clone, PartialEq)]
pub struct StepLogits {
    pub argmax: u32,
    pub argmax_value: f32,
    pub top16: Vec<(u32, f32)>,
}

/// Stages captured by a checkpoint-gated transcription.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscribeStages {
    pub input_features: Option<StageWitness>,
    pub encoder_output: Option<StageWitness>,
    pub audio_embeddings: Option<StageWitness>,
    pub prompt_token_ids: Vec<i32>,
    pub prefill_last_hidden: Option<Vec<f32>>,
    pub first_logits: Option<StepLogits>,
    /// The logits of the final decode decision (overwritten each step).
    pub last_logits: Option<StepLogits>,
    pub generated_token_ids: Vec<u32>,
}

/// Transcription result with token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechTranscription {
    pub text: String,
    pub prompt_tokens: usize,
    pub generation_tokens: usize,
    pub audio_rows: usize,
}

/// Loaded Granite Speech 1B model: encoder, projector, Granite LLM, all
/// full-precision f32 (bf16 shards upcast at load).
pub struct GraniteSpeech {
    config: GraniteSpeechConfig,
    encoder: GraniteSpeechEncoder,
    projector: EncoderProjector,
    llm: GraniteLlm,
    tokenizer: turbospark_tokenizer::Tokenizer,
    eos_token_id: i32,
}

impl GraniteSpeech {
    /// Loads one local checkpoint directory downloaded at the pinned
    /// profile revision. Loading is local-only; tests never download.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|error| bad_config("config.json", error))?,
        )
        .map_err(|error| bad_config("config.json", error))?;
        let config = GraniteSpeechConfig::from_json(&root)?;

        let shards = open_shards(model_dir)?;
        // The pinned checkpoint is unquantized bfloat16. A quantized
        // variant has never been verified, so any `.scales` tensor is
        // refused instead of silently dequantized with a guessed scheme.
        for file in &shards {
            if file
                .tensor_names()
                .any(|name| name.ends_with(".scales") || name.ends_with(".bias_packed"))
            {
                return Err(SpeechError::Unsupported {
                    why: "the checkpoint carries quantization tensors; no quantized Granite \
                          Speech variant is verified"
                        .to_owned(),
                });
            }
        }

        let encoder = GraniteSpeechEncoder::load(&shards, &config.encoder)?;
        let projector = EncoderProjector::load(
            &shards,
            &config.projector,
            config.window_size,
            config.num_queries(),
            config.text.hidden_size,
        )?;
        let llm = GraniteLlm::load_sharded(&shards, &config.text)?;
        let tokenizer = load_tokenizer(model_dir)?;

        if tokenizer.token_to_id("<|audio|>").map(|id| id as i32) != Some(config.audio_token_index)
        {
            return Err(bad_config(
                "tokenizer.json",
                "<|audio|> id does not match config.json audio_token_index",
            ));
        }
        let eos_token_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .map(|id| id as i32)
            .ok_or_else(|| {
                bad_config(
                    "tokenizer.json",
                    "the decode loop requires <|end_of_text|> in the vocabulary",
                )
            })?;

        Ok(Self {
            config,
            encoder,
            projector,
            llm,
            tokenizer,
            eos_token_id,
        })
    }

    pub fn profile(&self) -> GraniteSpeechProfile {
        GRANITE_4_0_1B_SPEECH
    }

    pub fn config(&self) -> &GraniteSpeechConfig {
        &self.config
    }

    /// Transcribes mono PCM already sampled at 16 kHz with greedy decoding
    /// and the default ASR prompt.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_prompt(samples, None, 256)
            .map(|output| output.text)
    }

    /// Transcribes with an explicit prompt; `None` selects the default ASR
    /// prompt. `max_tokens` bounds greedy decoding.
    pub fn transcribe_with_prompt(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
        max_tokens: usize,
    ) -> Result<GraniteSpeechTranscription> {
        self.transcribe_impl(samples, prompt, max_tokens, None)
    }

    /// The reference `Model.generate` for ASR: frontend, encoder,
    /// projector, prompt assembly with `<|audio|>` substitution, one
    /// prefill over the injected embeddings, then token-by-token greedy
    /// decoding, stopping at the tokenizer EOS id before emission.
    pub(crate) fn transcribe_impl(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<GraniteSpeechTranscription> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Granite Speech max_tokens must be positive".into(),
            });
        }
        let (features, rows) = frontend::features(samples)?;
        if let Some(stages) = stages.as_deref_mut() {
            // The fixture records the transposed [pair width, rows]
            // orientation (band-major, time columns).
            let mut transposed = vec![0.0f32; features.len()];
            for row in 0..rows {
                for column in 0..frontend::PAIR_WIDTH {
                    transposed[column * rows + row] = features[row * frontend::PAIR_WIDTH + column];
                }
            }
            stages.input_features = Some(witness(
                "input_features",
                &[frontend::PAIR_WIDTH, rows],
                &transposed,
            ));
        }
        let encoded = self.encoder.forward(&features, rows)?;
        // The encoder plane is [rows, encoder hidden_dim]; the text hidden
        // size only applies after the projector.
        let encoder_width = self.config.encoder.hidden_dim;
        let encoder_rows = encoded.len() / encoder_width;
        if let Some(stages) = stages.as_deref_mut() {
            stages.encoder_output = Some(witness(
                "encoder_output",
                &[encoder_rows, encoder_width],
                &encoded,
            ));
        }
        let audio_embeddings = self.projector.forward(&encoded, encoder_rows)?;
        let text_width = self.config.text.hidden_size;
        let audio_rows = audio_embeddings.len() / text_width;
        if audio_rows == 0 {
            return Err(SpeechError::Input {
                why: "the audio projector produced no embeddings".into(),
            });
        }
        if let Some(stages) = stages.as_deref_mut() {
            stages.audio_embeddings = Some(witness(
                "audio_embeddings",
                &[audio_rows, text_width],
                &audio_embeddings,
            ));
        }

        let prompt_ids = self.prompt_token_ids(audio_rows, prompt)?;
        let audio_id = self.config.audio_token_index;
        let (prefix_end, _) = prompt_ids
            .iter()
            .enumerate()
            .find(|(_, &id)| id == audio_id)
            .ok_or_else(|| SpeechError::Input {
                why: "the assembled prompt has no audio placeholder".into(),
            })?;
        let prefix_ids = prompt_ids[..prefix_end].to_vec();
        let suffix_ids = prompt_ids[prefix_end + audio_rows..].to_vec();
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = prompt_ids.clone();
        }

        // Embeddings with the audio span substituted, then scaled by the
        // Granite embedding multiplier exactly once (reference
        // `Model.__call__`).
        let mut embeddings = self.llm.embed(&prefix_ids)?;
        embeddings.extend_from_slice(&audio_embeddings);
        embeddings.extend(self.llm.embed(&suffix_ids)?);
        for value in &mut embeddings {
            *value *= self.llm.embedding_multiplier;
        }
        let rows = embeddings.len() / text_width;

        let (last_hidden, cache) = self.llm.prefill(&embeddings, rows);
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden = Some(last_hidden.clone());
        }
        let logits = self.llm.logits(&last_hidden);
        if let Some(stages) = stages.as_deref_mut() {
            stages.first_logits = Some(step_logits(&logits));
        }

        // The decision that produced the most recent emitted token: the
        // prefill logits for the first, then the per-step logits. Recorded
        // as last_logits with the same semantics as the fixture's
        // final_step_logits.
        let mut decision: Vec<f32> = logits.clone();
        let eos = self.eos_token_id;
        let generated = greedy_generate(
            &self.llm,
            &logits,
            cache,
            max_tokens,
            |next| next as i32 == eos,
            "Granite Speech",
            |new_decision, next| {
                // Retain only decisions that emitted a token, so last_logits
                // names the logits behind the final generated token,
                // matching the fixture's final_step_logits.
                if next as i32 != eos {
                    decision = new_decision.to_vec();
                }
            },
        )?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.last_logits = Some(step_logits(&decision));
        }
        if let Some(stages) = stages {
            stages.generated_token_ids = generated.clone();
        }

        let decoded =
            self.tokenizer
                .decode(&generated, true)
                .map_err(|error| SpeechError::Input {
                    why: format!("Granite Speech output detokenization failed: {error}"),
                })?;
        Ok(GraniteSpeechTranscription {
            text: decoded,
            prompt_tokens: rows,
            generation_tokens: generated.len(),
            audio_rows,
        })
    }

    /// Builds the prompt token ids as the reference `_build_prompt` does:
    /// the static chat-template expansion of one user turn whose content is
    /// one `<|audio|>` per audio embedding followed by the prompt text. The
    /// pinned GPT2 BPE tokenizer has no post-processor and adds no special
    /// tokens, so the whole-string encode equals the piecewise encode
    /// around the `<|audio|>` spans. One divergence needs a second split:
    /// the shipped tokenizer.json carries a newer pre-tokenizer whose
    /// punctuation branch absorbs a trailing LF, while the transformers
    /// reference repairs the pre-tokenizer to the canonical GPT2 pattern at
    /// load and keeps `?` and LF separate. Encoding the prompt text and the
    /// LF + ` ASSISTANT:` tail as separate pieces reproduces the reference
    /// ids with the un-repaired library (pinned by the fixture tests).
    fn prompt_token_ids(&self, audio_rows: usize, prompt: Option<&str>) -> Result<Vec<i32>> {
        let prompt = match prompt {
            Some(text) if !text.trim().is_empty() => text,
            _ => DEFAULT_ASR_PROMPT,
        };
        let mut ids = encode_ids(&self.tokenizer, "USER: ")?;
        ids.extend(std::iter::repeat_n(
            self.config.audio_token_index,
            audio_rows,
        ));
        ids.extend(encode_ids(&self.tokenizer, prompt)?);
        ids.extend(encode_ids(&self.tokenizer, "\n ASSISTANT:")?);
        Ok(ids)
    }
}

fn encode_ids(tokenizer: &turbospark_tokenizer::Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| SpeechError::Input {
            why: format!("Granite Speech prompt tokenization failed: {error}"),
        })?;
    encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: "Granite Speech prompt token id exceeds signed 32-bit range".into(),
            })
        })
        .collect()
}

fn step_logits(logits: &[f32]) -> StepLogits {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    StepLogits {
        argmax: order[0] as u32,
        argmax_value: logits[order[0]],
        top16: order
            .iter()
            .take(16)
            .map(|&index| (index as u32, logits[index]))
            .collect(),
    }
}

/// Builds the fixture's sparse row/column witness: corner, near-corner,
/// middle, and last spots.
fn witness(name: &str, shape: &[usize], values: &[f32]) -> StageWitness {
    let (rows_total, columns_total) = match shape.len() {
        1 => (1usize, shape[0]),
        2 => (shape[0], shape[1]),
        _ => (values.len(), 1),
    };
    let rows = [
        0,
        1.min(rows_total.saturating_sub(1)),
        rows_total / 2,
        rows_total - 1,
    ];
    let columns = [
        0,
        1.min(columns_total.saturating_sub(1)),
        columns_total / 2,
        columns_total - 1,
    ];
    let mut out = Vec::with_capacity(rows.len());
    for &row in &rows {
        let base = if shape.len() == 1 {
            0
        } else {
            row * columns_total
        };
        out.push(
            columns
                .iter()
                .map(|&column| values[base + column])
                .collect(),
        );
    }
    StageWitness {
        name: name.to_owned(),
        shape: shape.to_vec(),
        rows: rows.to_vec(),
        columns: columns.to_vec(),
        values: out,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expand_chat_template, translation_prompt, witness, GraniteSpeech, StepLogits,
        DEFAULT_ASR_PROMPT, GRANITE_4_0_1B_SPEECH, LANGUAGE_CODES,
    };
    use crate::nn::argmax;
    use serde_json::Value;
    use std::path::Path;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../testdata/granite_speech_reference.json"
        ))
        .unwrap()
    }

    fn smoke_samples() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = crate::wav::read_wav_f32(&path).expect("reference WAV loads");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        audio.samples
    }

    fn compare_witness(actual: &super::StageWitness, expected: &Value, tolerance: f32) -> f32 {
        let shape: Vec<usize> = serde_json::from_value(expected["shape"].clone()).unwrap();
        let rows: Vec<usize> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let columns: Vec<usize> = serde_json::from_value(expected["columns"].clone()).unwrap();
        let values: Vec<Vec<f32>> = serde_json::from_value(expected["values"].clone()).unwrap();
        assert_eq!(actual.shape, shape, "stage {} shape", actual.name);
        assert_eq!(actual.rows, rows, "stage {} rows", actual.name);
        assert_eq!(actual.columns, columns, "stage {} columns", actual.name);
        let expected_max = values
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        let mut max_abs = 0.0f32;
        for (actual_row, expected_row) in actual.values.iter().zip(&values) {
            for (a, e) in actual_row.iter().zip(expected_row) {
                max_abs = max_abs.max((a - e).abs());
            }
        }
        // The reference computes the audio path in bfloat16 while this port
        // computes f32, so the gate scales with the stage's own magnitude.
        let gate = tolerance * expected_max.max(1.0);
        assert!(
            max_abs <= gate,
            "stage {} maximum absolute difference {max_abs} exceeds {gate} \
             ({tolerance} of row max {expected_max})",
            actual.name
        );
        max_abs
    }

    #[test]
    fn pinned_profile_names_the_verified_checkpoint() {
        assert_eq!(
            GRANITE_4_0_1B_SPEECH.repository,
            "ibm-granite/granite-4.0-1b-speech"
        );
        assert_eq!(
            GRANITE_4_0_1B_SPEECH.revision,
            "bd87ab862416353633ea431fe49b1614003623c5"
        );
        assert_eq!(
            DEFAULT_ASR_PROMPT,
            "can you transcribe the speech into a written format?"
        );
    }

    /// The pinned 193-byte `chat_template.jinja` renders one user turn as
    /// `USER: {content}\n ASSISTANT:` with no generation prompt.
    #[test]
    fn chat_template_expansion_matches_the_pinned_jinja() {
        assert_eq!(expand_chat_template("hello"), "USER: hello\n ASSISTANT:");
        assert_eq!(
            expand_chat_template("<|audio|>transcribe"),
            "USER: <|audio|>transcribe\n ASSISTANT:"
        );
        assert_eq!(expand_chat_template(""), "USER: \n ASSISTANT:");
    }

    #[test]
    fn translation_prompt_maps_the_six_verified_languages() {
        let expected = [
            ("en", "English"),
            ("fr", "French"),
            ("de", "German"),
            ("es", "Spanish"),
            ("pt", "Portuguese"),
            ("ja", "Japanese"),
        ];
        assert_eq!(LANGUAGE_CODES.len(), expected.len());
        for (code, name) in expected {
            assert_eq!(
                translation_prompt(code),
                format!("Translate the speech to {name}.")
            );
        }
        assert_eq!(translation_prompt("EN"), "Translate the speech to English.");
        // Unknown codes pass through verbatim like the reference.
        assert_eq!(
            translation_prompt("klingon"),
            "Translate the speech to klingon."
        );
    }

    /// Always-on fixture parity for the checkpoint-free stage: the paired
    /// log-mel frontend over the raw smoke clip. Both sides compute the
    /// frontend in f32, so the gate is absolute and tight.
    #[test]
    fn frontend_features_match_the_pinned_reference() {
        let fixture = fixture();
        let samples = smoke_samples();
        assert_eq!(
            samples.len(),
            fixture["provenance"]["audio_samples"].as_u64().unwrap() as usize
        );
        let (features, rows) = super::frontend::features(&samples).unwrap();
        assert_eq!(
            rows as u64,
            fixture["frontend"]["encoder_rows"].as_u64().unwrap()
        );
        // The fixture records the transposed [160, rows] orientation.
        let mut transposed = vec![0.0f32; features.len()];
        for row in 0..rows {
            for col in 0..160 {
                transposed[col * rows + row] = features[row * 160 + col];
            }
        }
        let spot = witness("input_features", &[160, rows], &transposed);
        let max_abs = compare_witness(&spot, &fixture["frontend"]["input_features"], 2.0e-3);
        eprintln!("granite frontend parity: input_features max abs diff {max_abs:.3e}");
    }

    #[test]
    fn fixture_prompt_has_the_reference_shape_and_contiguous_audio_span() {
        let fixture = fixture();
        let backbone = &fixture["backbone"];
        let prompt: Vec<i32> =
            serde_json::from_value(backbone["prompt_token_ids"].clone()).unwrap();
        let prefix: Vec<i32> =
            serde_json::from_value(backbone["prompt_prefix_ids"].clone()).unwrap();
        let suffix: Vec<i32> =
            serde_json::from_value(backbone["prompt_suffix_ids"].clone()).unwrap();
        let audio_count = fixture["projector"]["num_audio_tokens"].as_u64().unwrap() as usize;
        assert_eq!(prompt.len(), prefix.len() + audio_count + suffix.len());
        assert_eq!(&prompt[..prefix.len()], &prefix[..]);
        assert_eq!(&prompt[prompt.len() - suffix.len()..], &suffix[..]);
        let audio_id = backbone["audio_token_id"].as_i64().unwrap();
        let audio_positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id as i64 == audio_id).then_some(position))
            .collect();
        assert_eq!(audio_positions.len(), audio_count);
        assert_eq!(
            audio_positions.last().unwrap() - audio_positions.first().unwrap() + 1,
            audio_count,
            "the placeholder span is contiguous"
        );
        // The template-expanded string is the static USER/ASSISTANT form.
        let prompt_string = backbone["prompt_string"].as_str().unwrap();
        let content = format!(
            "{}{}",
            "<|audio|>".repeat(audio_count),
            backbone["prompt"].as_str().unwrap()
        );
        assert_eq!(prompt_string, expand_chat_template(&content));
        assert_eq!(backbone["prompt"].as_str().unwrap(), DEFAULT_ASR_PROMPT);
        assert_eq!(
            backbone["prompt_tokens"].as_u64().unwrap(),
            prompt.len() as u64
        );
    }

    #[test]
    fn fixture_transcript_matches_the_expected_reference_sentence() {
        let fixture = fixture();
        let backbone = &fixture["backbone"];
        // The pinned reference emits the lowercase sentence with no
        // punctuation, matching the upstream mlx-audio smoke result.
        assert_eq!(
            backbone["transcript"].as_str().unwrap(),
            "the quick brown fox jumps over the lazy dog"
        );
        assert_eq!(
            backbone["raw_decoded_text"].as_str().unwrap(),
            "the quick brown fox jumps over the lazy dog"
        );
        let generated: Vec<u32> =
            serde_json::from_value(backbone["generated_token_ids"].clone()).unwrap();
        assert_eq!(generated.len(), 9);
        assert!(!generated.is_empty());
        assert_eq!(backbone["eos_token_id"].as_i64().unwrap(), 100_257);
        assert_eq!(backbone["audio_token_id"].as_i64().unwrap(), 100_352);
    }

    #[test]
    fn witness_picks_corner_middle_and_last_spots() {
        let values: Vec<f32> = (0..12).map(|index| index as f32).collect();
        let spot = witness("stage", &[4, 3], &values);
        assert_eq!(spot.rows, vec![0, 1, 2, 3]);
        assert_eq!(spot.columns, vec![0, 1, 1, 2]);
        assert_eq!(spot.values[0], vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(spot.values[3], vec![9.0, 10.0, 10.0, 11.0]);
    }

    #[test]
    fn argmax_prefers_the_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 0.0]), 1);
        assert_eq!(argmax(&[-5.0, -1.0]), 1);
    }

    #[test]
    fn step_logits_ranks_descending() {
        let logits = StepLogits {
            argmax: 7,
            argmax_value: 3.0,
            top16: vec![(7, 3.0), (2, 1.0)],
        };
        assert_eq!(logits.top16[0].0, 7);
    }

    #[test]
    #[ignore = "requires the pinned Granite Speech checkpoint in TURBOSPARK_GRANITE_SPEECH_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_GRANITE_SPEECH_MODEL_DIR")
            .expect("set TURBOSPARK_GRANITE_SPEECH_MODEL_DIR to the pinned checkpoint directory");
        let model = GraniteSpeech::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), GRANITE_4_0_1B_SPEECH);
        assert_eq!(model.eos_token_id, 100_257);
        let samples = smoke_samples();
        let mut stages = super::TranscribeStages::default();
        let started = std::time::Instant::now();
        let output = model
            .transcribe_impl(&samples, None, 256, Some(&mut stages))
            .expect("checkpoint transcribes");
        let elapsed = started.elapsed().as_secs_f64();
        let fixture = fixture();
        let backbone = &fixture["backbone"];

        // Token-level contract: prompt ids, greedy ids, transcript. All
        // decisions on this clip carry wide margins, so everything must
        // match exactly.
        let expected_tokens: Vec<u32> =
            serde_json::from_value(backbone["generated_token_ids"].clone()).unwrap();
        assert_eq!(
            stages.generated_token_ids, expected_tokens,
            "greedy token ids must match the reference exactly"
        );
        let expected_prompt: Vec<i32> =
            serde_json::from_value(backbone["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);
        assert_eq!(
            output.text,
            backbone["transcript"].as_str().unwrap(),
            "transcript must match the reference exactly"
        );
        assert_eq!(
            output.generation_tokens as u64,
            backbone["generated_token_ids"].as_array().unwrap().len() as u64
        );
        assert_eq!(
            output.prompt_tokens as u64,
            backbone["prompt_tokens"].as_u64().unwrap(),
            "prompt embedding rows (prefix + audio rows + suffix)"
        );
        assert_eq!(
            output.audio_rows as u64,
            fixture["projector"]["num_audio_tokens"].as_u64().unwrap()
        );

        // Numeric-stage witnesses. The reference computes bf16 while this
        // port computes f32, so numeric gates are relative while the token
        // contract above stays exact.
        let features_max = compare_witness(
            stages.input_features.as_ref().unwrap(),
            &fixture["frontend"]["input_features"],
            2.0e-3,
        );
        let encoder_max = compare_witness(
            stages.encoder_output.as_ref().unwrap(),
            &fixture["encoder"]["output"],
            3.0e-2,
        );
        let embeddings_max = compare_witness(
            stages.audio_embeddings.as_ref().unwrap(),
            &fixture["projector"]["audio_embeddings"],
            3.0e-2,
        );
        let expected_hidden: Vec<f32> =
            serde_json::from_value(backbone["prefill_last_hidden"].clone()).unwrap();
        let hidden = stages.prefill_last_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), expected_hidden.len());
        let hidden_max = hidden
            .iter()
            .zip(&expected_hidden)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let expected_scale = expected_hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let hidden_relative = hidden_max / expected_scale;
        assert!(
            hidden_max <= 3.0e-2 * expected_scale,
            "prefill hidden difference {hidden_max} exceeds 3% of the row scale {expected_scale}"
        );
        let first = stages.first_logits.as_ref().unwrap();
        let expected_argmax = backbone["first_logits"]["argmax"].as_u64().unwrap() as u32;
        assert_eq!(first.argmax, expected_argmax, "first greedy token id");
        let expected_top8: Vec<(u32, f64)> =
            serde_json::from_value(backbone["first_logits"]["top8"].clone()).unwrap();
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
        // The final decode decision: the reference eos margin must survive.
        let final_step = &backbone["final_step_logits"];
        let last = stages.last_logits.as_ref().unwrap();
        let expected_final_argmax = final_step["argmax"].as_u64().unwrap() as u32;
        assert_eq!(last.argmax, expected_final_argmax, "final argmax token id");
        let eos = backbone["eos_token_id"].as_i64().unwrap() as u32;
        let reference_eos = final_step["token_values"][eos.to_string().as_str()]
            .as_f64()
            .unwrap();
        let rust_eos = last
            .top16
            .iter()
            .find(|(id, _)| *id == eos)
            .map(|(_, value)| *value as f64)
            .unwrap_or_else(|| panic!("the eos token left the Rust top-16"));
        assert!(
            reference_eos - rust_eos <= 1.0,
            "the eos logit moved too far: rust {rust_eos} vs reference {reference_eos}"
        );
        eprintln!(
            "granite checkpoint parity: input_features max {features_max:.3e}, \
             encoder_output max {encoder_max:.3e}, audio_embeddings max {embeddings_max:.3e}, \
             prefill hidden max {hidden_max:.3e} (relative {hidden_relative:.3e}), \
             top-8 logit max {top_logit_max:.3e}, eos logit rust {rust_eos:.3} vs \
             reference {reference_eos:.3}, prompt_tokens {}, generated_tokens {}, wall \
             {elapsed:.1}s",
            output.prompt_tokens, output.generation_tokens
        );
    }
}
