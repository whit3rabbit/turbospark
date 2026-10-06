//! MOSS-Transcribe-Diarize: timestamped transcription with speaker labels.
//!
//! Reference: `mlx_audio/stt/models/moss_transcribe_diarize/`
//! (moss_transcribe_diarize.py, config.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The architecture is a whisper
//! audio encoder ([`encoder`]) built from glmasr's `WhisperEncoderLayer`
//! blocks, a Linear-SiLU-Linear-LayerNorm VQ adaptor ([`adaptor`]), and the
//! shared `qwen3_asr` text decoder as the MOSS backbone. The prompt is the
//! checkpoint's chat template with an audio placeholder span; time markers
//! inject digit tokens every few seconds of audio. This port reuses the
//! qwen3 decoder (prefill, per-layer KV cache, tied LM head), the qwen3
//! tokenizer loader, and the shared 80-band whisper log-mel frontend. See
//! README.md for the pinned profile and verification evidence.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::quant::QuantScheme;
use crate::stt::qwen3_asr::decoder::Decoder;
use crate::stt::qwen3_asr::load_tokenizer;
use crate::whisper::{whisper_log_mel, WHISPER_WINDOW_SAMPLES};
use crate::{Result, SpeechError};

pub mod adaptor;
pub mod config;
pub mod encoder;

pub use config::{MossConfig, ProcessorConfig};
pub use encoder::MossWhisperEncoder;

use adaptor::VqAdaptor;
use config::PINNED_CHAT_TEMPLATE_SHA256;

/// The upstream default transcription prompt, verbatim.
pub const DEFAULT_PROMPT: &str = "Transcribe the audio into text. Start each segment with the \
start timestamp and speaker label ([S01], [S02], [S03], ...), write the corresponding spoken \
content, and end each segment with the ending timestamp to clearly mark the segment range.";

/// The audio placeholder token the prompt is split on.
pub const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MossTranscribeDiarizeProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned 4-bit profile, smoke-verified against mlx-audio 0.5.7.
pub const MOSS_TRANSCRIBE_DIARIZE_4BIT: MossTranscribeDiarizeProfile =
    MossTranscribeDiarizeProfile {
        name: "MOSS-Transcribe-Diarize 4-bit",
        repository: "vanch007/mlx-MOSS-Transcribe-Diarize-4bit",
        revision: "d42a296ee807e933ddd7588e2041dbbc84aff85d",
    };

/// One parsed transcript segment: `[start][Sxx] text [end]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    pub start: f32,
    pub end: f32,
    /// `"[S01] spoken content"` when a speaker label was parsed, otherwise
    /// the raw text (the reference fallback shape).
    pub text: String,
    pub speaker_id: Option<String>,
}

/// Full transcription result with parsed speaker/timestamp segments.
#[derive(Debug, Clone, PartialEq)]
pub struct MossTranscription {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub prompt_tokens: usize,
    pub generation_tokens: usize,
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
    pub generated_token_ids: Vec<u32>,
    pub digit_token_ids: Vec<u32>,
}

/// Loaded MOSS-Transcribe-Diarize model: whisper encoder, VQ adaptor, and
/// the shared qwen3 decoder with an affine 4-bit backbone.
pub struct MossTranscribeDiarize {
    config: MossConfig,
    processor: ProcessorConfig,
    encoder: MossWhisperEncoder,
    adaptor: VqAdaptor,
    decoder: Decoder,
    tokenizer: turbospark_tokenizer::Tokenizer,
    stop_ids: Vec<u32>,
    digit_token_ids: Vec<u32>,
}

impl MossTranscribeDiarize {
    /// Loads one local checkpoint directory downloaded at the pinned profile
    /// revision. Loading is local-only; tests never download.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let root: Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|error| bad_config("config.json", error))?,
        )
        .map_err(|error| bad_config("config.json", error))?;
        let config = MossConfig::from_json(&root)?;

        let processor_path = model_dir.join("processor_config.json");
        let processor = if processor_path.is_file() {
            let value: Value = serde_json::from_slice(
                &std::fs::read(&processor_path)
                    .map_err(|error| bad_config("processor_config.json", error))?,
            )
            .map_err(|error| bad_config("processor_config.json", error))?;
            ProcessorConfig::from_value(&value)?
        } else {
            ProcessorConfig::from_value(&Value::Object(serde_json::Map::new()))?
        };

        // The prompt contract comes from this exact chat template; anything
        // else renders differently and would silently mis-prompt the model.
        let template_path = model_dir.join("chat_template.jinja");
        let digest = turbospark_model_io::hash_file(&template_path, 1 << 20)
            .map_err(|error| bad_config("chat_template.jinja", error))?;
        if digest != PINNED_CHAT_TEMPLATE_SHA256 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "chat_template.jinja digest {digest} does not match the pinned prompt \
                     template {PINNED_CHAT_TEMPLATE_SHA256}"
                ),
            });
        }

        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let scheme = QuantScheme {
            bits: config.quant_bits,
            group_size: config.quant_group_size,
        };
        let encoder = MossWhisperEncoder::load(&weights, &config.audio)?;
        let adaptor = VqAdaptor::load(
            &weights,
            config.adaptor_input_dim,
            config.text.hidden_size,
            config.text.rms_norm_eps,
        )?;
        let decoder = Decoder::load(&weights, &config.text, scheme)?;
        let tokenizer = load_tokenizer(model_dir)?;

        if tokenizer.token_to_id(AUDIO_PAD_TOKEN).map(|id| id as i32) != Some(config.audio_token_id)
        {
            return Err(bad_config(
                "tokenizer.json",
                format!("{AUDIO_PAD_TOKEN} id does not match config.json audio_token_id"),
            ));
        }

        // The reference requires every digit to be a single token; time
        // markers splice them straight into the audio span.
        let mut digit_token_ids = Vec::with_capacity(10);
        for digit in '0'..='9' {
            let text = digit.to_string();
            let encoded = tokenizer
                .encode(text.as_str(), false)
                .map_err(|error| bad_config("tokenizer.json", error))?;
            let ids = encoded.get_ids();
            if ids.len() != 1 {
                return Err(bad_config(
                    "tokenizer.json",
                    format!("digit {digit} encodes to {} tokens", ids.len()),
                ));
            }
            digit_token_ids.push(ids[0]);
        }

        let mut stop_ids: Vec<u32> = vec![151_643, 151_645];
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }

        Ok(Self {
            config,
            processor,
            encoder,
            adaptor,
            decoder,
            tokenizer,
            stop_ids,
            digit_token_ids,
        })
    }

    pub fn profile(&self) -> MossTranscribeDiarizeProfile {
        MOSS_TRANSCRIBE_DIARIZE_4BIT
    }

    pub fn config(&self) -> &MossConfig {
        &self.config
    }

    pub fn processor_config(&self) -> &ProcessorConfig {
        &self.processor
    }

    /// Transcribes mono PCM already sampled at 16 kHz with greedy decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, 512)
            .map(|output| output.text)
    }

    /// Transcribes one mono 16 kHz clip with greedy decoding, returning the
    /// text, parsed speaker/timestamp segments, and token counts.
    /// `max_tokens` bounds greedy decoding.
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        max_tokens: usize,
    ) -> Result<MossTranscription> {
        self.transcribe_impl(samples, max_tokens, None)
            .map(|(output, _)| output)
    }

    /// Transcription with optional stage capture for checkpoint-gated parity
    /// tests. The greedy loop mirrors the reference `Model.generate`:
    /// prefill once over the injected prompt embeddings, then step token by
    /// token, stopping at the Qwen EOS ids before emission.
    pub(crate) fn transcribe_impl(
        &self,
        samples: &[f32],
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<(MossTranscription, Vec<u32>)> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "MOSS max_tokens must be positive".into(),
            });
        }
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "MOSS audio must be non-empty and finite".into(),
            });
        }
        if samples.len() > WHISPER_WINDOW_SAMPLES {
            return Err(SpeechError::Unsupported {
                why: "MOSS currently accepts one audio chunk of at most 30 seconds; the \
                      reference multi-chunk path is not part of this port"
                    .to_owned(),
            });
        }

        let (features, frames) = compute_mel_features(samples)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.input_features = Some(witness(
                "input_features",
                &[frames, self.config.audio.num_mel_bins],
                &features,
            ));
        }

        let audio_token_length = (samples.len() - 1) / self.config.samples_per_audio_token() + 1;
        let (encoded, encoder_rows) = self.encoder.forward(&features, frames)?;
        if audio_token_length * self.config.audio_merge_size > encoder_rows {
            return Err(SpeechError::Input {
                why: format!(
                    "MOSS audio needs {} encoder frames but the encoder produced {encoder_rows}",
                    audio_token_length * self.config.audio_merge_size
                ),
            });
        }

        // Time merge: each adaptor row concatenates `audio_merge_size`
        // consecutive encoder frames, then the VQ adaptor projects it.
        let merge = self.config.audio_merge_size;
        let d_model = self.config.audio.d_model;
        let merged_width = d_model * merge;
        let mut merged = vec![0.0f32; audio_token_length * merged_width];
        for row in 0..audio_token_length {
            for frame in 0..merge {
                let source = (row * merge + frame) * d_model;
                let target = row * merged_width + frame * d_model;
                merged[target..target + d_model]
                    .copy_from_slice(&encoded[source..source + d_model]);
            }
        }
        let audio_embeddings = self.adaptor.forward(&merged, audio_token_length);
        if let Some(stages) = stages.as_deref_mut() {
            stages.encoder_output = Some(witness(
                "encoder_output",
                &[encoder_rows, d_model],
                &encoded,
            ));
            stages.audio_embeddings = Some(witness(
                "audio_embeddings",
                &[audio_token_length, self.decoder.hidden_size],
                &audio_embeddings,
            ));
        }

        let token_ids = self.prompt_token_ids(audio_token_length, None)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = token_ids.clone();
            stages.digit_token_ids = self.digit_token_ids.clone();
        }

        let mut embeddings = self.decoder.embed(&token_ids)?;
        let audio_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id == self.config.audio_token_id).then_some(position))
            .collect::<Vec<_>>();
        if audio_positions.len() != audio_token_length {
            return Err(SpeechError::Input {
                why: format!(
                    "MOSS prompt has {} audio placeholders for {audio_token_length} encoded \
                     frames",
                    audio_positions.len()
                ),
            });
        }
        for (audio_row, &position) in audio_positions.iter().enumerate() {
            let target = position * self.decoder.hidden_size;
            let source = audio_row * self.decoder.hidden_size;
            embeddings[target..target + self.decoder.hidden_size]
                .copy_from_slice(&audio_embeddings[source..source + self.decoder.hidden_size]);
        }

        let rows = embeddings.len() / self.decoder.hidden_size;
        let (mut last_hidden, mut cache) = self.decoder.prefill(&embeddings, rows);
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden = Some(last_hidden.clone());
        }
        let logits = self.decoder.logits(&last_hidden);
        if let Some(stages) = stages.as_deref_mut() {
            stages.first_logits = Some(top_logits(&logits));
        }

        let mut generated: Vec<u32> = Vec::new();
        let mut next = argmax(&logits);
        for _ in 0..max_tokens {
            if self.stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let token_id = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: "MOSS generated token id exceeds signed 32-bit range".into(),
            })?;
            let next_embedding = self.decoder.embed(&[token_id])?;
            last_hidden = self.decoder.step(&next_embedding, &mut cache);
            next = argmax(&self.decoder.logits(&last_hidden));
        }
        if let Some(stages) = stages {
            stages.generated_token_ids = generated.clone();
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| SpeechError::Input {
                why: format!("MOSS output detokenization failed: {error}"),
            })?
            .trim()
            .to_owned();
        let duration = samples.len() as f32 / self.config.sample_rate as f32;
        let segments = parse_segments(&text, duration);
        Ok((
            MossTranscription {
                text,
                segments,
                prompt_tokens: rows,
                generation_tokens: generated.len(),
            },
            generated,
        ))
    }

    /// Builds the prompt token ids: the rendered chat template split on the
    /// audio placeholder, with the audio span ids spliced in between.
    fn prompt_token_ids(&self, audio_token_count: usize, prompt: Option<&str>) -> Result<Vec<i32>> {
        let rendered = render_prompt(prompt)?;
        let (before, after) =
            rendered
                .split_once(AUDIO_PAD_TOKEN)
                .ok_or_else(|| SpeechError::Input {
                    why: format!("expected exactly one {AUDIO_PAD_TOKEN} token in the prompt"),
                })?;
        let mut ids = encode_ids(&self.tokenizer, before)?;
        ids.extend(self.audio_span_ids(audio_token_count));
        ids.extend(encode_ids(&self.tokenizer, after)?);
        Ok(ids)
    }

    /// The reference `_audio_span_ids`: plain audio placeholders, or audio
    /// placeholders with decimal time markers spliced in every
    /// `time_marker_every_seconds` seconds of audio.
    fn audio_span_ids(&self, audio_seq_len: usize) -> Vec<i32> {
        audio_span_ids_for(
            self.config.audio_token_id,
            &self.processor,
            &self.digit_token_ids,
            audio_seq_len,
        )
    }
}

fn bad_config(field: &str, error: impl std::fmt::Display) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: error.to_string(),
    }
}

/// Renders the pinned chat template for one user turn with audio. The
/// checkpoint's `chat_template.jinja` (verified by digest at load) renders
/// exactly this string for the no-tools, no-system-message shape upstream
/// uses, with `add_generation_prompt=True`.
fn render_prompt(prompt: Option<&str>) -> Result<String> {
    let prompt = match prompt {
        Some(text) if !text.trim().is_empty() => text,
        _ => DEFAULT_PROMPT,
    };
    if prompt.contains(AUDIO_PAD_TOKEN) {
        // The upstream contract requires exactly one placeholder; a caller
        // prompt that carries its own audio token cannot be split safely.
        return Err(SpeechError::Input {
            why: format!("the prompt must not embed {AUDIO_PAD_TOKEN} itself"),
        });
    }
    Ok(format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\
         <|audio_start|>{AUDIO_PAD_TOKEN}<|audio_end|>\n{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    ))
}

fn encode_ids(tokenizer: &turbospark_tokenizer::Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|error| SpeechError::Input {
            why: format!("MOSS prompt tokenization failed: {error}"),
        })?;
    encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: "MOSS prompt token id exceeds signed 32-bit range".into(),
            })
        })
        .collect()
}

/// The reference `_audio_span_ids` as a free function so the marker
/// arithmetic is unit-testable without weights.
fn audio_span_ids_for(
    audio_token_id: i32,
    processor: &ProcessorConfig,
    digit_token_ids: &[u32],
    audio_seq_len: usize,
) -> Vec<i32> {
    let every = processor.time_marker_every_seconds;
    if !processor.enable_time_marker || audio_seq_len == 0 || every == 0 {
        return vec![audio_token_id; audio_seq_len];
    }
    let tokens_per_marker = (processor.audio_tokens_per_second * every as f32) as usize;
    if tokens_per_marker == 0 {
        return vec![audio_token_id; audio_seq_len];
    }
    let duration = audio_seq_len as f32 / processor.audio_tokens_per_second;
    let mut output: Vec<i32> = Vec::with_capacity(audio_seq_len + 8);
    let mut consumed = 0usize;
    let mut second = every;
    while second <= duration as usize {
        let position = (second / every) * tokens_per_marker;
        let segment_len = position.saturating_sub(consumed);
        if segment_len > 0 {
            output.extend(std::iter::repeat_n(audio_token_id, segment_len));
            consumed += segment_len;
        }
        for character in second.to_string().chars() {
            output.push(digit_token_ids[character as usize - '0' as usize] as i32);
        }
        second += every;
    }
    if audio_seq_len > consumed {
        output.extend(std::iter::repeat_n(
            audio_token_id,
            audio_seq_len - consumed,
        ));
    }
    output
}

/// 80-band whisper log-mel over the zero-padded 30-second window the
/// reference feature extractor produces (`padding="max_length"`).
fn compute_mel_features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    let mut padded = samples.to_vec();
    padded.resize(WHISPER_WINDOW_SAMPLES, 0.0);
    let mel = whisper_log_mel(&padded, 80)?;
    if mel.is_empty() || mel.iter().any(|frame| frame.len() != 80) {
        return Err(SpeechError::Audio(
            "80-band whisper frontend returned an invalid shape".into(),
        ));
    }
    let frames = mel.len();
    let mut values = Vec::with_capacity(frames * 80);
    for frame in &mel {
        values.extend_from_slice(frame);
    }
    Ok((values, frames))
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (id, &value)| {
            if value > best.1 {
                (id, value)
            } else {
                best
            }
        })
        .0 as u32
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

/// Parses `[start][Sxx] text [end]` segments, the port of the reference
/// `TRANSCRIPT_SEGMENT_RE.finditer` loop: start and end are decimal numbers
/// in brackets, the speaker is `S` plus digits, the text is the lazy span
/// between (so the first following numbered bracket closes the segment),
/// segments with `end < start` or empty text are dropped, and a text with no
/// matches falls back to one unsegmented entry over the clip duration.
pub fn parse_segments(text: &str, fallback_end: f32) -> Vec<TranscriptSegment> {
    let bytes = text.as_bytes();
    let mut segments = Vec::new();
    let mut position = 0usize;
    while position < bytes.len() {
        if let Some((start, speaker, after_speaker)) = try_segment_start(bytes, position) {
            if let Some((end, text_end, after_end)) = find_segment_end(bytes, after_speaker) {
                if end >= start {
                    let segment_text = text[after_speaker..text_end].trim();
                    if !segment_text.is_empty() {
                        segments.push(TranscriptSegment {
                            start,
                            end,
                            text: format!("[{speaker}] {segment_text}"),
                            speaker_id: Some(speaker.to_owned()),
                        });
                    }
                }
                position = after_end;
                continue;
            }
        }
        position += 1;
    }
    if segments.is_empty() {
        return vec![TranscriptSegment {
            start: 0.0,
            end: fallback_end.max(0.0),
            text: text.to_owned(),
            speaker_id: None,
        }];
    }
    segments
}

/// Matches `[number][Sdigits]` at `position`, returning the start time, the
/// speaker label (`S` plus digits, as the reference captures it), and the
/// index just past the speaker bracket.
fn try_segment_start(bytes: &[u8], position: usize) -> Option<(f32, &str, usize)> {
    if bytes.get(position) != Some(&b'[') {
        return None;
    }
    let (start, mut index) = parse_number(bytes, position + 1)?;
    if bytes.get(index) != Some(&b']') {
        return None;
    }
    index += 1;
    if bytes.get(index) != Some(&b'[') || bytes.get(index + 1) != Some(&b'S') {
        return None;
    }
    let speaker_start = index + 1;
    let mut index = speaker_start + 1;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == speaker_start + 1 || bytes.get(index) != Some(&b']') {
        return None;
    }
    let speaker = std::str::from_utf8(&bytes[speaker_start..index]).ok()?;
    Some((start, speaker, index + 1))
}

/// Scans forward for the first `[number]` bracket, returning its value, the
/// index of the opening bracket, and the index just past the closing one.
fn find_segment_end(bytes: &[u8], from: usize) -> Option<(f32, usize, usize)> {
    let mut index = from;
    while index < bytes.len() {
        if bytes[index] == b'[' {
            if let Some((value, after)) = parse_number(bytes, index + 1) {
                if bytes.get(after) == Some(&b']') {
                    return Some((value, index, after + 1));
                }
            }
        }
        index += 1;
    }
    None
}

/// Matches `\d+(?:\.\d+)?` starting at `position`, returning the value and
/// the index just past the number.
fn parse_number(bytes: &[u8], position: usize) -> Option<(f32, usize)> {
    let mut end = position;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == position {
        return None;
    }
    if end + 1 < bytes.len() && bytes[end] == b'.' && bytes[end + 1].is_ascii_digit() {
        end += 2;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    let text = std::str::from_utf8(&bytes[position..end]).ok()?;
    let value = text.parse::<f32>().ok()?;
    Some((value, end))
}

#[cfg(test)]
mod tests {
    use super::{
        audio_span_ids_for, parse_segments, render_prompt, witness, MossTranscribeDiarize,
        TranscriptSegment, AUDIO_PAD_TOKEN, DEFAULT_PROMPT,
    };
    use crate::stt::moss_transcribe_diarize::config::ProcessorConfig;
    use serde_json::Value;
    use std::path::Path;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../testdata/moss_transcribe_diarize_reference.json"
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

    fn processor() -> ProcessorConfig {
        ProcessorConfig {
            audio_tokens_per_second: 12.5,
            time_marker_every_seconds: 5,
            enable_time_marker: true,
        }
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
    fn rendered_prompt_matches_the_pinned_chat_template() {
        let rendered = render_prompt(None).unwrap();
        assert_eq!(
            rendered,
            fixture()["backbone"]["rendered_prompt"].as_str().unwrap()
        );
        assert!(rendered.contains(DEFAULT_PROMPT));
        assert_eq!(rendered.matches(AUDIO_PAD_TOKEN).count(), 1);
        // Blank prompts fall back to the reference default; a prompt that
        // embeds the audio placeholder itself is refused.
        assert!(render_prompt(Some("  ")).is_ok());
        let embedded = format!("custom {AUDIO_PAD_TOKEN} prompt");
        assert!(render_prompt(Some(&embedded)).is_err());
    }

    #[test]
    fn fixture_prompt_has_a_contiguous_audio_placeholder_span() {
        let fixture = fixture();
        let prompt: Vec<i32> =
            serde_json::from_value(fixture["backbone"]["prompt_token_ids"].clone()).unwrap();
        let audio_rows = fixture["backbone"]["audio_rows"].as_u64().unwrap() as usize;
        assert_eq!(prompt[0], 151_644, "<|im_start|> opens the prompt");
        assert_eq!(prompt[14], 151_669, "<|audio_start|> opens the audio span");
        let audio_id = 151_671;
        let audio_positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter_map(|(position, &id)| (id == audio_id).then_some(position))
            .collect();
        assert_eq!(audio_positions.len(), audio_rows);
        assert_eq!(
            audio_positions.last().unwrap() - audio_positions.first().unwrap() + 1,
            audio_rows,
            "the placeholder span is contiguous"
        );
    }

    /// Always-on fixture parity for the checkpoint-free stage: the 80-band
    /// log-mel frontend over the zero-padded smoke clip. The reference
    /// stores these values in bfloat16, so the gate is bf16-rounded.
    #[test]
    fn frontend_features_match_the_pinned_reference() {
        let fixture = fixture();
        let samples = smoke_samples();
        assert_eq!(
            samples.len(),
            fixture["provenance"]["audio_samples"].as_u64().unwrap() as usize
        );
        let (features, frames) = super::compute_mel_features(&samples).unwrap();
        let witness = witness("input_features", &[frames, 80], &features);
        let max_abs = compare_witness(&witness, &fixture["backbone"]["input_features"], 3.0e-3);
        eprintln!("moss frontend parity: input_features max abs diff {max_abs:.3e}");
    }

    #[test]
    fn segment_parser_matches_the_reference_transcript_and_fallback() {
        let fixture = fixture();
        let transcript = fixture["backbone"]["transcript"].as_str().unwrap();
        let duration = fixture["backbone"]["duration_seconds"].as_f64().unwrap() as f32;
        let segments = parse_segments(transcript, duration);
        let expected: Vec<Value> =
            serde_json::from_value(fixture["backbone"]["segments"].clone()).unwrap();
        assert_eq!(segments.len(), expected.len());
        for (segment, expected) in segments.iter().zip(&expected) {
            assert_eq!(
                segment.speaker_id.as_deref(),
                expected["speaker_id"].as_str()
            );
            assert!((segment.start - expected["start"].as_f64().unwrap() as f32).abs() < 1e-6);
            assert!((segment.end - expected["end"].as_f64().unwrap() as f32).abs() < 1e-6);
            assert_eq!(segment.text, expected["text"].as_str().unwrap());
        }
        // The reference fallback keeps the raw text with the clip duration.
        let fallback = parse_segments("no markers here", 2.5);
        assert_eq!(
            fallback,
            vec![TranscriptSegment {
                start: 0.0,
                end: 2.5,
                text: "no markers here".to_owned(),
                speaker_id: None,
            }]
        );
    }

    #[test]
    fn segment_parser_skips_inverted_ranges_and_empty_text() {
        // The reference regex consumes "[2.00][S01] dropped [1.00]" (dropped:
        // end < start), "[1.00][S02] kept [3.00]" (kept), "[3.00][S01]  [3.50]"
        // (dropped: whitespace-only text), and "[3.50][S03] tail [4.00]"
        // (kept). The leading words keep every candidate reachable after the
        // previous match ends.
        let text = "a [2.00][S01] dropped [1.00] b [1.00][S02] kept [3.00] c \
                    [3.00][S01]  [3.50] d [3.50][S03] tail [4.00]";
        let segments = parse_segments(text, 9.0);
        assert_eq!(
            segments,
            vec![
                TranscriptSegment {
                    start: 1.0,
                    end: 3.0,
                    text: "[S02] kept".to_owned(),
                    speaker_id: Some("S02".to_owned()),
                },
                TranscriptSegment {
                    start: 3.5,
                    end: 4.0,
                    text: "[S03] tail".to_owned(),
                    speaker_id: Some("S03".to_owned()),
                },
            ]
        );
        // A text whose every candidate is dropped falls back unsegmented.
        let fallback = parse_segments("[2.00][S01] inverted [1.00]", 1.0);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].speaker_id, None);
        assert_eq!(fallback[0].text, "[2.00][S01] inverted [1.00]");
        // No closing bracket: the whole text falls back unsegmented.
        let fallback = parse_segments("[0.00][S01] unterminated", 1.0);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].speaker_id, None);
    }

    #[test]
    fn audio_span_ids_inject_decimal_time_markers() {
        let digits: Vec<u32> = (15..25).collect();
        let pad = 151_671;
        // 2.8 s of audio stays below the first marker boundary.
        assert_eq!(
            audio_span_ids_for(pad, &processor(), &digits, 35),
            vec![pad; 35]
        );
        // 5.04 s: 62 placeholders, the "5" marker token, then one trailing
        // placeholder (the reference counts only placeholders as consumed).
        let span = audio_span_ids_for(pad, &processor(), &digits, 63);
        assert_eq!(span.len(), 64);
        assert_eq!(span[62], 20, "digit 5 marker token");
        assert_eq!(span[63], pad);
        assert!(span[..62].iter().all(|&id| id == pad));
        // 10.08 s: two markers; "10" is the two-token decimal string, and
        // two trailing placeholders follow because consumed only tracks
        // placeholder positions (reference arithmetic, kept faithfully).
        let span = audio_span_ids_for(pad, &processor(), &digits, 126);
        assert_eq!(span[62], 20, "digit 5 marker token");
        assert_eq!(&span[125..127], &[16, 15], "digits 1 and 0");
        assert_eq!(span.len(), 129);
        assert_eq!(span[127], pad);
        assert_eq!(span[128], pad);
        // Disabled markers degrade to a plain placeholder span.
        let disabled = ProcessorConfig {
            audio_tokens_per_second: 12.5,
            time_marker_every_seconds: 5,
            enable_time_marker: false,
        };
        assert_eq!(
            audio_span_ids_for(pad, &disabled, &digits, 40),
            vec![pad; 40]
        );
    }

    #[test]
    fn witness_picks_corner_middle_and_last_spots() {
        let values: Vec<f32> = (0..12).map(|index| index as f32).collect();
        let stage = witness("stage", &[4, 3], &values);
        assert_eq!(stage.rows, vec![0, 1, 2, 3]);
        assert_eq!(stage.columns, vec![0, 1, 1, 2]);
        assert_eq!(stage.values[0], vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(stage.values[3], vec![9.0, 10.0, 10.0, 11.0]);
    }

    #[test]
    #[ignore = "requires the pinned MOSS checkpoint in TURBOSPARK_MOSS_TRANSCRIBE_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_MOSS_TRANSCRIBE_MODEL_DIR")
            .expect("set TURBOSPARK_MOSS_TRANSCRIBE_MODEL_DIR to the pinned checkpoint directory");
        let model = MossTranscribeDiarize::load(Path::new(&model_dir))
            .expect("pinned MOSS checkpoint loads");
        assert_eq!(model.profile(), super::MOSS_TRANSCRIBE_DIARIZE_4BIT);
        let samples = smoke_samples();
        let mut stages = super::TranscribeStages::default();
        let (output, _) = model
            .transcribe_impl(&samples, 256, Some(&mut stages))
            .expect("checkpoint transcribes");
        let fixture = fixture();

        // Token-level contract: prompt ids, greedy ids, transcript, digits.
        let expected_tokens: Vec<u32> =
            serde_json::from_value(fixture["backbone"]["generated_token_ids"].clone()).unwrap();
        assert_eq!(
            stages.generated_token_ids, expected_tokens,
            "greedy token ids must match the reference exactly"
        );
        let expected_prompt: Vec<i32> =
            serde_json::from_value(fixture["backbone"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);
        assert_eq!(
            output.text,
            fixture["backbone"]["transcript"].as_str().unwrap()
        );
        let expected_digits: Vec<u32> = (0..10)
            .map(|digit| {
                fixture["backbone"]["digit_token_ids"][&digit.to_string()]
                    .as_u64()
                    .unwrap() as u32
            })
            .collect();
        assert_eq!(stages.digit_token_ids, expected_digits);
        assert_eq!(
            output.prompt_tokens,
            fixture["backbone"]["prompt_token_ids"]
                .as_array()
                .unwrap()
                .len()
        );
        assert_eq!(output.generation_tokens, expected_tokens.len());

        // Numeric-stage witnesses. The reference stores the encoder, adaptor,
        // and decoder in bfloat16 and runs quantized backbone matmuls; this
        // port computes the same pipeline in f32, so numeric gates are
        // relative while the token contract above stays exact.
        let features_max = compare_witness(
            stages.input_features.as_ref().unwrap(),
            &fixture["backbone"]["input_features"],
            3.0e-3,
        );
        let encoder_max = compare_witness(
            stages.encoder_output.as_ref().unwrap(),
            &fixture["backbone"]["encoder_output"],
            6.0e-2,
        );
        let embeddings_max = compare_witness(
            stages.audio_embeddings.as_ref().unwrap(),
            &fixture["backbone"]["audio_embeddings"],
            6.0e-2,
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
        let hidden_relative = hidden_max / expected_max;
        assert!(
            hidden_max <= 6.0e-2 * expected_max,
            "prefill hidden difference {hidden_max} exceeds 6% of the row scale {expected_max}"
        );
        let first = stages.first_logits.as_ref().unwrap();
        let expected_argmax = fixture["backbone"]["first_logits"]["argmax"]
            .as_u64()
            .unwrap() as u32;
        assert_eq!(first.argmax, expected_argmax, "first greedy token id");
        let expected_top8: Vec<(u32, f64)> =
            serde_json::from_value(fixture["backbone"]["first_logits"]["top8"].clone()).unwrap();
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
            "moss checkpoint parity: input_features max {features_max:.3e}, \
             encoder_output max {encoder_max:.3e}, audio_embeddings max {embeddings_max:.3e}, \
             prefill hidden max {hidden_max:.3e} (relative {hidden_relative:.3e}), \
             top-8 logit max {top_logit_max:.3e}"
        );
    }
}
