//! Higgs Audio v3 STT: whisper-style encoder, MLP projector, Qwen3 decoder.
//!
//! Reference: `mlx_audio/stt/models/higgs_audio_3/` (higgs_audio_3.py,
//! audio.py, config.py, vad.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The architecture is a
//! 128-band whisper-style audio encoder ([`encoder`]) with per-layer
//! pre-norm blocks, pairwise time pooling, an MLP projector whose depthwise
//! stride-2 conv halves the time axis, and the shared `qwen3_asr` decoder
//! (Qwen3 1.7B geometry, 28 layers, untied LM head) as the text backbone.
//! The reference prompt wraps the audio in
//! `<|audio_bos|>`/`<|audio_eos|>` inside one user turn and decodes greedily
//! until the Qwen EOS ids. This port reuses the qwen3 decoder machinery and
//! tokenizer loader, and runs the reference VAD chunking ([`vad`]) with the
//! shared Silero VAD when a checkpoint directory is provided. See README.md
//! for the pinned profile and verification evidence.

use std::path::Path;

use serde_json::Value;

use crate::nn::{bad_config, open_shards};
use crate::quant::QuantScheme;
use crate::stt::qwen3_asr::decoder::{greedy_generate, Decoder};
use crate::stt::qwen3_asr::load_tokenizer;
use crate::{Result, SpeechError};

pub mod config;
pub mod encoder;
pub mod vad;

pub use config::HiggsConfig;

use encoder::{FeatureProjector, HiggsAudioEncoder};
use vad::SpeechRanges;

/// The upstream default transcription prompt, verbatim.
pub const DEFAULT_PROMPT: &str =
    "Transcribe the speech. Output only the spoken words in lowercase with no punctuation.";

/// The Qwen EOS ids the reference decode loop stops on
/// (`<|im_end|>` and `<|endoftext|>`).
const STOP_IDS: [u32; 2] = [151_645, 151_643];

/// The projector's fixed hidden width (`nn.Linear(audio_dim, 2048)`).
const PROJECTOR_WIDE: usize = 2048;

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiggsAudioV3Profile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned full-bfloat16 profile, smoke-verified against mlx-audio 0.5.7.
pub const HIGGS_AUDIO_V3_STT: HiggsAudioV3Profile = HiggsAudioV3Profile {
    name: "Higgs Audio v3 STT",
    repository: "bosonai/higgs-audio-v3-stt",
    revision: "2ffd1aa39f5a1266931e405cba12e404a9f994b2",
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
    pub encoder_output: Option<StageWitness>,
    pub audio_embeddings: Option<StageWitness>,
    pub prompt_token_ids: Vec<i32>,
    pub prefill_last_hidden: Option<Vec<f32>>,
    pub first_logits: Option<FirstLogits>,
    /// The logits of the final decode decision (overwritten each step), the
    /// evidence for the documented near-tie analysis.
    pub last_logits: Option<FirstLogits>,
    pub generated_token_ids: Vec<u32>,
    /// Silero speech ranges the VAD backend reported, empty when no backend
    /// ran or it failed (both fall back to uniform chunking).
    pub vad_cuts: Vec<(usize, usize)>,
    /// The chunk ranges the audio was split into.
    pub chunk_ranges: Vec<(usize, usize)>,
}

/// Transcription result with token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct HiggsTranscription {
    pub text: String,
    pub prompt_tokens: usize,
    pub generation_tokens: usize,
    pub audio_rows: usize,
}

/// Loaded Higgs Audio v3 STT model: audio tower, projector, and the shared
/// qwen3 decoder with an untied LM head, all full-precision.
pub struct HiggsAudioV3Stt {
    config: HiggsConfig,
    encoder: HiggsAudioEncoder,
    projector: FeatureProjector,
    decoder: Decoder,
    tokenizer: turbospark_tokenizer::Tokenizer,
    vad: Option<vad::SileroRanges>,
}

impl HiggsAudioV3Stt {
    /// Loads one local checkpoint directory downloaded at the pinned profile
    /// revision. Loading is local-only; tests never download.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_inner(model_dir, None)
    }

    /// Loads the model plus the Silero VAD checkpoint directory used for
    /// `vad_cut` chunking. Without a VAD the reference's backend-failure
    /// fallback applies: uniform `chunk_size_seconds` chunks.
    pub fn load_with_vad(model_dir: &Path, vad_dir: &Path) -> Result<Self> {
        Self::load_inner(model_dir, Some(vad::SileroRanges::load(vad_dir)?))
    }

    fn load_inner(model_dir: &Path, vad: Option<vad::SileroRanges>) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|error| bad_config("config.json", error))?,
        )
        .map_err(|error| bad_config("config.json", error))?;
        let config = HiggsConfig::from_json(&root)?;

        let shards = open_shards(model_dir)?;
        // The pinned checkpoint is unquantized bfloat16. A quantized
        // variant has never been verified, so any `.scales` tensor is
        // refused instead of silently dequantized with a guessed scheme.
        for file in &shards {
            if file.tensor_names().any(|name| name.ends_with(".scales")) {
                return Err(SpeechError::Unsupported {
                    why: "the checkpoint carries groupwise `.scales` tensors; no quantized \
                          Higgs Audio v3 variant is verified"
                        .to_owned(),
                });
            }
        }

        let audio = &config.audio;
        let encoder = HiggsAudioEncoder::load(
            &shards,
            audio.encoder_layers,
            audio.encoder_attention_heads,
            audio.encoder_ffn_dim,
            audio.d_model,
            audio.num_mel_bins,
            audio.max_source_positions,
        )?;
        let projector = FeatureProjector::load(
            &shards,
            audio.d_model,
            PROJECTOR_WIDE,
            config.text.hidden_size,
        )?;
        // The scheme never engages: quantization is refused above and the
        // tensors carry no `.scales`, so every load takes the plain path.
        let scheme = QuantScheme {
            bits: 4,
            group_size: 64,
        };
        let decoder = Decoder::load_sharded(&shards, &config.text, scheme)?;
        let tokenizer = load_tokenizer(model_dir)?;

        if tokenizer.token_to_id("<|AUDIO|>").map(|id| id as i32) != Some(config.audio_in_token_idx)
        {
            return Err(bad_config(
                "tokenizer.json",
                "<|AUDIO|> id does not match config.json audio_in_token_idx",
            ));
        }
        if tokenizer.token_to_id("<|audio_eos|>").map(|id| id as i32)
            != Some(config.audio_eos_token_id)
        {
            return Err(bad_config(
                "tokenizer.json",
                "<|audio_eos|> id does not match config.json audio_eos_token_id",
            ));
        }
        for token in [
            "<|im_start|>",
            "<|im_end|>",
            "<|audio_bos|>",
            "<|endoftext|>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(bad_config(
                    "tokenizer.json",
                    format!("the prompt tokens require {token} in the vocabulary"),
                ));
            }
        }

        Ok(Self {
            config,
            encoder,
            projector,
            decoder,
            tokenizer,
            vad,
        })
    }

    pub fn profile(&self) -> HiggsAudioV3Profile {
        HIGGS_AUDIO_V3_STT
    }

    pub fn config(&self) -> &HiggsConfig {
        &self.config
    }

    /// Transcribes mono PCM already sampled at 16 kHz with greedy decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, 1024)
            .map(|output| output.text)
    }

    /// Transcribes one mono 16 kHz clip with greedy decoding, returning the
    /// text and token counts. `max_tokens` bounds greedy decoding; the
    /// reference default is 1024.
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        max_tokens: usize,
    ) -> Result<HiggsTranscription> {
        self.transcribe_impl(samples, max_tokens, None)
            .map(|(output, _)| output)
    }

    /// Transcription with optional stage capture for checkpoint-gated parity
    /// tests. Mirrors the reference `Model.generate`: VAD chunking, per-chunk
    /// encode and project, prompt assembly, one prefill over the injected
    /// embeddings, then token-by-token greedy decoding, stopping at the Qwen
    /// EOS ids before emission.
    pub(crate) fn transcribe_impl(
        &self,
        samples: &[f32],
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<(HiggsTranscription, Vec<u32>)> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Higgs max_tokens must be positive".into(),
            });
        }
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "Higgs audio must be non-empty and finite".into(),
            });
        }

        // Reference `_chunk_waveform`: the Silero backend provides speech
        // ranges when `vad_cut` is on; a failed or missing backend degrades
        // to the reference's empty-cuts fallback (uniform chunking).
        let cuts = if self.config.vad_cut {
            match self.vad.as_ref() {
                Some(backend) => backend.speech_ranges(samples).unwrap_or_default(),
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let ranges = vad::chunk_ranges(
            samples.len(),
            self.config.chunk_samples(),
            &cuts,
            self.config.split_vads,
        );
        if let Some(stages) = stages.as_deref_mut() {
            stages.vad_cuts = cuts;
            stages.chunk_ranges = ranges.clone();
        }

        // Shorter chunks are zero-padded to the longest chunk before the
        // frontend (reference `np.pad(c, (0, max_frames - len(c)))`).
        let max_len = ranges
            .iter()
            .map(|(start, end)| end - start)
            .max()
            .unwrap_or(0);
        let hidden = self.decoder.hidden_size;
        let d_model = self.encoder_d_model();
        let mut audio_embeddings = Vec::new();
        let mut audio_rows = 0usize;
        for (index, (start, end)) in ranges.iter().enumerate() {
            let mut chunk = samples[*start..*end].to_vec();
            chunk.resize(max_len, 0.0);
            let (features, frames) = encoder::log_mel_128(&chunk)?;
            if index == 0 {
                if let Some(stages) = stages.as_deref_mut() {
                    stages.input_features = Some(witness(
                        "input_features",
                        &[frames, self.config.audio.num_mel_bins],
                        &features,
                    ));
                }
            }
            let encoded = self.encoder.forward(&features, frames)?;
            let encoded_rows = encoded.len() / d_model;
            if index == 0 {
                if let Some(stages) = stages.as_deref_mut() {
                    stages.encoder_output = Some(witness(
                        "encoder_output",
                        &[encoded_rows, d_model],
                        &encoded,
                    ));
                }
            }
            let projected = self.projector.forward(&encoded, encoded_rows);
            if index == 0 {
                if let Some(stages) = stages.as_deref_mut() {
                    stages.audio_embeddings = Some(witness(
                        "audio_embeddings",
                        &[projected.len() / hidden, hidden],
                        &projected,
                    ));
                }
            }
            audio_rows += projected.len() / hidden;
            audio_embeddings.extend_from_slice(&projected);
        }
        if audio_rows == 0 {
            return Err(SpeechError::Input {
                why: "Higgs audio encoder produced no features".into(),
            });
        }

        let prompt_ids = self.prompt_token_ids(ranges.len(), None)?;
        let (prefix_end, _) = prompt_ids
            .iter()
            .enumerate()
            .find(|(_, &id)| id == self.config.audio_in_token_idx)
            .ok_or_else(|| SpeechError::Input {
                why: "the assembled prompt has no audio placeholder".into(),
            })?;
        let prefix_ids = prompt_ids[..prefix_end].to_vec();
        let suffix_ids = prompt_ids[prefix_end + ranges.len()..].to_vec();
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = prompt_ids.clone();
        }

        let mut embeddings = self.decoder.embed(&prefix_ids)?;
        embeddings.extend_from_slice(&audio_embeddings);
        embeddings.extend(self.decoder.embed(&suffix_ids)?);
        let rows = embeddings.len() / hidden;

        let (last_hidden, cache) = self.decoder.prefill(&embeddings, rows);
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden = Some(last_hidden.clone());
        }
        let logits = self.decoder.logits(&last_hidden);
        if let Some(stages) = stages.as_deref_mut() {
            stages.first_logits = Some(top_logits(&logits));
        }

        let generated = greedy_generate(
            &self.decoder,
            &logits,
            cache,
            max_tokens,
            |next| STOP_IDS.contains(&next),
            "Higgs",
            |step_logits, _| {
                if let Some(stages) = stages.as_deref_mut() {
                    stages.last_logits = Some(top_logits(step_logits));
                }
            },
        )?;
        if let Some(stages) = stages {
            stages.generated_token_ids = generated.clone();
        }

        let decoded =
            self.tokenizer
                .decode(&generated, false)
                .map_err(|error| SpeechError::Input {
                    why: format!("Higgs output detokenization failed: {error}"),
                })?;
        Ok((
            HiggsTranscription {
                text: parse_output(&decoded),
                prompt_tokens: rows,
                generation_tokens: generated.len(),
                audio_rows,
            },
            generated,
        ))
    }

    fn encoder_d_model(&self) -> usize {
        self.config.audio.d_model
    }

    /// Builds the prompt token ids exactly as the reference
    /// `get_input_embeddings` does: one user turn, the prompt text, the
    /// audio bos marker, one audio placeholder per chunk, the audio eos
    /// marker, and the assistant generation header. All pieces encode
    /// without extra special tokens.
    fn prompt_token_ids(&self, chunk_count: usize, prompt: Option<&str>) -> Result<Vec<i32>> {
        let prompt = match prompt {
            Some(text) if !text.trim().is_empty() => text,
            _ => DEFAULT_PROMPT,
        };
        let mut ids = encode_ids(&self.tokenizer, "<|im_start|>user\n")?;
        ids.extend(encode_ids(&self.tokenizer, prompt)?);
        ids.extend(encode_ids(&self.tokenizer, "<|audio_bos|>")?);
        ids.extend(std::iter::repeat_n(
            self.config.audio_in_token_idx,
            chunk_count,
        ));
        ids.extend(encode_ids(&self.tokenizer, "<|audio_eos|>")?);
        ids.extend(encode_ids(&self.tokenizer, "<|im_end|>\n")?);
        ids.extend(encode_ids(&self.tokenizer, "<|im_start|>assistant\n")?);
        Ok(ids)
    }
}

fn encode_ids(tokenizer: &turbospark_tokenizer::Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| SpeechError::Input {
            why: format!("Higgs prompt tokenization failed: {error}"),
        })?;
    encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: "Higgs prompt token id exceeds signed 32-bit range".into(),
            })
        })
        .collect()
}

/// The reference `_parse_output`: drop complete `<think>...</think>` spans
/// (dot-matches-all, non-greedy), keep everything after an unclosed
/// `<think>`, drop every `<|...|>` special-token span on one line, and trim.
pub fn parse_output(text: &str) -> String {
    let mut kept = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => {
                kept.push_str(&rest[..start]);
                rest = &after[end + "</think>".len()..];
            }
            // No closing tag: the regex never matches; the whole remainder
            // survives for the unclosed-`<think>` cut below.
            None => {
                kept.push_str(rest);
                rest = "";
                break;
            }
        }
    }
    kept.push_str(rest);

    let mut text: &str = &kept;
    if let Some(position) = text.find("<think>") {
        text = &text[position + "<think>".len()..];
    }
    remove_special_spans(text).trim().to_owned()
}

/// `re.sub(r"<\|.*?\|>", "", text)`: removes each shortest `<|` ... `|>`
/// span that carries no newline (the pattern has no DOTALL flag); spans
/// without a closing `|>` on the same line stay untouched.
fn remove_special_spans(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'|') {
            let mut j = i + 2;
            let mut end = None;
            while j < bytes.len() && bytes[j] != b'\n' {
                if bytes[j] == b'|' && bytes.get(j + 1) == Some(&b'>') {
                    end = Some(j + 2);
                    break;
                }
                j += 1;
            }
            if let Some(end) = end {
                i = end;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
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

/// Builds the fixture's sparse row/column witness: the first two rows and
/// columns, the middle pair, and the last pair.
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
        parse_output, remove_special_spans, witness, HiggsAudioV3Stt, DEFAULT_PROMPT,
        HIGGS_AUDIO_V3_STT,
    };
    use crate::nn::argmax;
    use serde_json::Value;
    use std::path::Path;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../testdata/higgs_audio_3_reference.json"
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
        // The reference stores these stages in bfloat16 while this port
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
        assert_eq!(HIGGS_AUDIO_V3_STT.repository, "bosonai/higgs-audio-v3-stt");
        assert_eq!(
            HIGGS_AUDIO_V3_STT.revision,
            "2ffd1aa39f5a1266931e405cba12e404a9f994b2"
        );
        assert_eq!(
            DEFAULT_PROMPT,
            "Transcribe the speech. Output only the spoken words in lowercase with no \
             punctuation."
        );
    }

    /// Always-on fixture parity for the checkpoint-free stage: the 128-band
    /// log-mel frontend over the raw smoke clip. The reference stores the
    /// frontend output in bfloat16, so the gate is bf16-rounded.
    #[test]
    fn frontend_features_match_the_pinned_reference() {
        let fixture = fixture();
        let samples = smoke_samples();
        assert_eq!(
            samples.len(),
            fixture["provenance"]["audio_samples"].as_u64().unwrap() as usize
        );
        let (features, frames) = super::encoder::log_mel_128(&samples).unwrap();
        let spot = witness("input_features", &[frames, 128], &features);
        let max_abs = compare_witness(&spot, &fixture["backbone"]["input_features"], 3.0e-3);
        eprintln!("higgs frontend parity: input_features max abs diff {max_abs:.3e}");
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
        let audio_count = backbone["chunk_count"].as_u64().unwrap() as usize;
        assert_eq!(prompt.len(), prefix.len() + audio_count + suffix.len());
        assert_eq!(&prompt[..prefix.len()], &prefix[..]);
        assert_eq!(&prompt[prompt.len() - suffix.len()..], &suffix[..]);
        let audio_id = 151_672;
        let audio_positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id == audio_id).then_some(position))
            .collect();
        assert_eq!(audio_positions.len(), audio_count);
        assert_eq!(
            audio_positions.last().unwrap() - audio_positions.first().unwrap() + 1,
            audio_count,
            "the placeholder span is contiguous"
        );
        // The audio markers sit immediately around the placeholder span.
        assert_eq!(prompt[prefix.len() - 1], 151_669, "<|audio_bos|>");
        assert_eq!(prompt[prefix.len() + audio_count], 151_670, "<|audio_eos|>");
        assert_eq!(prompt[0], 151_644, "<|im_start|> opens the prompt");
        // The default prompt text is the pinned sentence.
        assert_eq!(backbone["prompt"].as_str().unwrap(), DEFAULT_PROMPT);
    }

    #[test]
    fn fixture_transcript_matches_the_expected_reference_sentence() {
        let fixture = fixture();
        // The pinned reference emits the lowercase sentence; the model adds
        // a final period despite the no-punctuation prompt, and the fixture
        // records that verbatim.
        let transcript = fixture["backbone"]["transcript"].as_str().unwrap();
        assert_eq!(transcript, "the quick brown fox jumps over the lazy dog.");
        let generated: Vec<u32> =
            serde_json::from_value(fixture["backbone"]["generated_token_ids"].clone()).unwrap();
        assert!(!generated.is_empty());
        assert_eq!(
            fixture["backbone"]["eos_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>(),
            vec![151_643, 151_645]
        );
        // The reference strips nothing else: the raw decode already equals
        // the parsed transcript for the pinned clip.
        assert_eq!(
            fixture["backbone"]["raw_decoded_text"].as_str().unwrap(),
            "the quick brown fox jumps over the lazy dog."
        );
    }

    #[test]
    fn parse_output_strips_think_blocks_and_special_spans() {
        // A complete think span is removed entirely.
        assert_eq!(
            parse_output("<think>reasoning here</think>the answer"),
            "the answer"
        );
        // An unclosed think span keeps everything after the marker.
        assert_eq!(parse_output("prefix <think>tail"), "tail");
        assert_eq!(parse_output("a<think>b</think>c<think>d"), "d");
        // Special-token spans vanish; spans without a closer stay.
        assert_eq!(parse_output("<|audio_bos|>hello<|im_end|>"), "hello");
        assert_eq!(parse_output("<|unterminated kept"), "<|unterminated kept");
        // The regex has no DOTALL flag: a newline blocks the match.
        assert_eq!(parse_output("<|a\nb|>kept"), "<|a\nb|>kept");
        // Plain output only trims.
        assert_eq!(
            parse_output("  the quick brown fox  "),
            "the quick brown fox"
        );
    }

    #[test]
    fn remove_special_spans_keeps_the_shortest_match_per_open() {
        // Non-greedy: the first "|>" closes the span; the scan resumes after
        // it, so the trailing "|b|>" survives untouched.
        assert_eq!(remove_special_spans("<|a|>|b|>"), "|b|>");
        assert_eq!(remove_special_spans("x<|y|>z"), "xz");
        assert_eq!(remove_special_spans("no markers"), "no markers");
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
    #[ignore = "requires the pinned Higgs Audio v3 checkpoint in TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR")
            .expect("set TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR to the pinned checkpoint directory");
        let model = HiggsAudioV3Stt::load(Path::new(&model_dir)).expect("pinned checkpoint loads");
        assert_eq!(model.profile(), HIGGS_AUDIO_V3_STT);
        let samples = smoke_samples();
        let mut stages = super::TranscribeStages::default();
        let started = std::time::Instant::now();
        let (output, _) = model
            .transcribe_impl(&samples, 256, Some(&mut stages))
            .expect("checkpoint transcribes");
        let elapsed = started.elapsed().as_secs_f64();
        let fixture = fixture();
        let backbone = &fixture["backbone"];

        // Token-level contract: prompt ids, greedy ids, transcript. The
        // reference's final decision on this clip is a documented one-ulp
        // near tie: its bf16 logits for the period (13) and <|im_end|>
        // (151645) sit 0.125 apart at magnitude 20.9, which is the bf16 ulp
        // there, so an f32 port may settle the tie the other way and stop
        // one token early. Everything before that step must match exactly.
        let expected_tokens: Vec<u32> =
            serde_json::from_value(backbone["generated_token_ids"].clone()).unwrap();
        let generated = &stages.generated_token_ids;
        if generated.len() == expected_tokens.len() {
            assert_eq!(
                generated, &expected_tokens,
                "greedy token ids must match the reference exactly"
            );
        } else {
            // The near-tie path: exact prefix, tie evidence in the logits.
            assert_eq!(
                generated.len(),
                expected_tokens.len() - 1,
                "the only permitted divergence is the documented final near tie"
            );
            assert_eq!(
                generated,
                &expected_tokens[..expected_tokens.len() - 1],
                "greedy token ids before the near tie must match the reference exactly"
            );
            let near_tie = &backbone["final_step_logits"];
            let reference_dot = near_tie["token_values"]["13"].as_f64().unwrap();
            let reference_im_end = near_tie["token_values"]["151645"].as_f64().unwrap();
            let reference_gap = reference_dot - reference_im_end;
            assert!(
                reference_gap > 0.0 && reference_gap < 0.5,
                "the fixture must document a genuine near tie, got gap {reference_gap}"
            );
            let last = stages.last_logits.as_ref().unwrap();
            let rust_dot = last
                .top16
                .iter()
                .find(|(id, _)| *id == 13)
                .unwrap_or_else(|| panic!("the period token left the Rust top-16"))
                .1 as f64;
            let rust_im_end = last
                .top16
                .iter()
                .find(|(id, _)| *id == 151_645)
                .unwrap_or_else(|| panic!("<|im_end|> left the Rust top-16"))
                .1 as f64;
            let rust_gap = rust_dot - rust_im_end;
            assert!(
                (rust_gap - reference_gap).abs() <= 0.5,
                "the near-tie gap moved too far: rust {rust_dot} vs {rust_im_end} \
                 (gap {rust_gap}), reference gap {reference_gap}"
            );
            eprintln!(
                "higgs near tie at the final step: reference period {reference_dot} vs \
                 <|im_end|> {reference_im_end} (gap {reference_gap}), rust f32 {rust_dot} vs \
                 {rust_im_end} (gap {rust_gap})"
            );
        }
        let expected_prompt: Vec<i32> =
            serde_json::from_value(backbone["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);
        let reference_transcript = backbone["transcript"].as_str().unwrap();
        let without_period = reference_transcript
            .strip_suffix('.')
            .unwrap_or(reference_transcript);
        assert!(
            output.text == reference_transcript || output.text == without_period,
            "transcript {:?} matches neither the reference {reference_transcript:?} nor its \
             near-tie prefix without the final period",
            output.text
        );
        assert_eq!(
            output.generation_tokens,
            if generated.len() == expected_tokens.len() {
                expected_tokens.len()
            } else {
                expected_tokens.len() - 1
            }
        );
        assert_eq!(
            output.prompt_tokens as u64,
            backbone["prompt_tokens"].as_u64().unwrap(),
            "prompt embedding rows (prefix + audio rows + suffix)"
        );
        assert_eq!(
            output.audio_rows as u64,
            backbone["audio_rows"].as_u64().unwrap()
        );

        // Numeric-stage witnesses. The reference computes the whole pipeline
        // in bfloat16 while this port computes f32, so numeric gates are
        // relative while the token contract above stays exact.
        let features_max = compare_witness(
            stages.input_features.as_ref().unwrap(),
            &backbone["input_features"],
            3.0e-3,
        );
        let encoder_max = compare_witness(
            stages.encoder_output.as_ref().unwrap(),
            &backbone["encoder_output"],
            6.0e-2,
        );
        let embeddings_max = compare_witness(
            stages.audio_embeddings.as_ref().unwrap(),
            &backbone["audio_embeddings"],
            6.0e-2,
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
            hidden_max <= 6.0e-2 * expected_scale,
            "prefill hidden difference {hidden_max} exceeds 6% of the row scale {expected_scale}"
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
        eprintln!(
            "higgs checkpoint parity: input_features max {features_max:.3e}, \
             encoder_output max {encoder_max:.3e}, audio_embeddings max {embeddings_max:.3e}, \
             prefill hidden max {hidden_max:.3e} (relative {hidden_relative:.3e}), \
             top-8 logit max {top_logit_max:.3e}, \
             prompt_tokens {}, generated_tokens {}, wall {elapsed:.1}s",
            output.prompt_tokens, output.generation_tokens
        );
    }
}
