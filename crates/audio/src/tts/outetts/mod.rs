//! OuteTTS family (Llama-OuteTTS-1.0 style two-codebook DAC).
//!
//! Port of `mlx_audio/tts/models/outetts/` (outetts.py,
//! prompt_processor.py, tokens.py, audio_processor.py, dac_interface.py)
//! at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/outetts):
//! a llama, qwen2, or qwen3 backbone (shared [`crate::lm`] ports,
//! selected by the config `model_type` exactly like the reference model
//! map) that emits interleaved `<|c1_i|>` / `<|c2_i|>` audio tokens, the
//! word-level prompt construction over a speaker JSON, and the two-book
//! DAC decode. See [README.md](README.md) for the verification evidence
//! and open items.
//!
//! Scope of this port, matching the reference single-segment path:
//! batch size 1, non-streaming decode (streaming intervals out of
//! scope), and prompt construction at the string level. Text
//! tokenization is a caller concern (the reference loads a Hugging Face
//! tokenizer in `Model.__init__` and encodes the completion prompt with
//! `add_special_tokens=False`); so is the `<|c1_i|>` / `<|c2_i|>`
//! vocabulary lookup, supplied here through [`AudioTokenMaps`]. Speaker
//! creation from reference audio needs a word-timestamped transcript
//! (whisper) and stays on the caller; speakers arrive as the reference
//! speaker JSON (the shipped `default_speaker.json` shape).

mod loudness;

pub use loudness::{integrated_loudness, process_audio_array};

use std::collections::HashMap;

use serde_json::Value;

use crate::codec::descript::Dac;
use crate::lm;
use crate::lm::engine::Input;
use crate::music::minimax_music3::rng;
use crate::tts::sampling;
use crate::{quant, Result, SpeechError};

/// Pinned mlx-audio commit this port was transcribed from.
pub const REFERENCE_COMMIT: &str = "e1b19b9054bf163f5d812221a54fcc346f1890e9";

/// The `SpecialTokens` dataclass strings (tokens.py). Formatted tokens
/// (c1/c2/time/features) carry the `{}` holes; use the helpers below.
pub struct SpecialTokens;

impl SpecialTokens {
    pub const BOS: &'static str = "<|im_start|>";
    pub const EOS: &'static str = "<|im_end|>";
    pub const C1_FMT: &'static str = "<|c1_{}|>";
    pub const C2_FMT: &'static str = "<|c2_{}|>";
    pub const TEXT_START: &'static str = "<|text_start|>";
    pub const TEXT_END: &'static str = "<|text_end|>";
    pub const VOICE_CHARACTERISTIC_START: &'static str = "<|voice_characteristic_start|>";
    pub const VOICE_CHARACTERISTIC_END: &'static str = "<|voice_characteristic_end|>";
    pub const EMOTION_START: &'static str = "<|emotion_start|>";
    pub const EMOTION_END: &'static str = "<|emotion_end|>";
    pub const AUDIO_START: &'static str = "<|audio_start|>";
    pub const AUDIO_END: &'static str = "<|audio_end|>";
    pub const TIME_FMT: &'static str = "<|t_{:.2}|>";
    pub const CODE: &'static str = "<|code|>";
    pub const ENERGY_FMT: &'static str = "<|energy_{}|>";
    pub const SPECTRAL_CENTROID_FMT: &'static str = "<|spectral_centroid_{}|>";
    pub const PITCH_FMT: &'static str = "<|pitch_{}|>";
    pub const WORD_START: &'static str = "<|word_start|>";
    pub const WORD_END: &'static str = "<|word_end|>";
    pub const FEATURES: &'static str = "<|features|>";
    pub const GLOBAL_FEATURES_START: &'static str = "<|global_features_start|>";
    pub const GLOBAL_FEATURES_END: &'static str = "<|global_features_end|>";

    /// `<|c1_{code}|>`.
    pub fn c1(code: i32) -> String {
        format!("<|c1_{code}|>")
    }

    /// `<|c2_{code}|>`.
    pub fn c2(code: i32) -> String {
        format!("<|c2_{code}|>")
    }

    /// `<|t_{duration:.2}|>` (the reference `{:.2f}` format).
    pub fn time(duration: f64) -> String {
        format!("<|t_{duration:.2}|>")
    }
}

/// Number of codebook entries the audio token maps cover (the reference
/// builds both maps over `range(1025)`).
pub const CODEBOOK_TOKENS: usize = 1025;

/// The `<|c1_i|>` / `<|c2_i|>` vocabulary lookup the reference builds
/// from its tokenizer (`get_audio_token_map`): token id for every code
/// 0..1025 on both books, plus the reverse maps `extract_audio_from_tokens`
/// reads.
#[derive(Debug, Clone, Default)]
pub struct AudioTokenMaps {
    /// Code index -> token id for `<|c1_i|>`.
    pub c1: Vec<Option<i32>>,
    /// Code index -> token id for `<|c2_i|>`.
    pub c2: Vec<Option<i32>>,
    c1_by_id: HashMap<i32, i32>,
    c2_by_id: HashMap<i32, i32>,
}

impl AudioTokenMaps {
    /// Builds both maps from a tokenizer-style lookup: `lookup(token)`
    /// returns the id the tokenizer assigns, or `None` when absent.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<i32>) -> Self {
        let mut maps = Self {
            c1: vec![None; CODEBOOK_TOKENS],
            c2: vec![None; CODEBOOK_TOKENS],
            ..Self::default()
        };
        for index in 0..CODEBOOK_TOKENS {
            let index = index as i32;
            if let Some(id) = lookup(&SpecialTokens::c1(index)) {
                maps.c1[index as usize] = Some(id);
                maps.c1_by_id.insert(id, index);
            }
            if let Some(id) = lookup(&SpecialTokens::c2(index)) {
                maps.c2[index as usize] = Some(id);
                maps.c2_by_id.insert(id, index);
            }
        }
        maps
    }

    /// Builds both maps from explicit id tables (fixture and offline
    /// construction; `ids[i]` is the token id of code `i`).
    pub fn from_ids(c1: &[Option<i32>], c2: &[Option<i32>]) -> Result<Self> {
        if c1.len() != CODEBOOK_TOKENS || c2.len() != CODEBOOK_TOKENS {
            return Err(SpeechError::Input {
                why: format!(
                    "audio token maps need {CODEBOOK_TOKENS} entries per book, got {} and {}",
                    c1.len(),
                    c2.len()
                ),
            });
        }
        let mut maps = Self {
            c1: c1.to_vec(),
            c2: c2.to_vec(),
            ..Self::default()
        };
        for index in 0..CODEBOOK_TOKENS {
            let index = index as i32;
            if let Some(id) = c1[index as usize] {
                maps.c1_by_id.insert(id, index);
            }
            if let Some(id) = c2[index as usize] {
                maps.c2_by_id.insert(id, index);
            }
        }
        Ok(maps)
    }

    /// Reference `extract_audio_from_tokens`: collect the codes of the
    /// c1 and c2 tokens in order and truncate the longer book to the
    /// shorter length. Unknown ids are skipped exactly like the
    /// reference list comprehensions.
    pub fn extract_audio_from_tokens(&self, tokens: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let codebook1: Vec<i32> = tokens
            .iter()
            .filter_map(|id| self.c1_by_id.get(id).copied())
            .collect();
        let codebook2: Vec<i32> = tokens
            .iter()
            .filter_map(|id| self.c2_by_id.get(id).copied())
            .collect();
        let keep = codebook1.len().min(codebook2.len());
        (codebook1[..keep].to_vec(), codebook2[..keep].to_vec())
    }
}

/// The per-word speaker entry the prompt codes consume.
#[derive(Debug, Clone)]
pub struct SpeakerWord {
    pub word: String,
    pub duration: f64,
    pub c1: Vec<i32>,
    pub c2: Vec<i32>,
    /// energy / spectral_centroid / pitch, each the 0-100 scaled integer.
    pub features: [i64; 3],
}

/// A loaded speaker JSON: the reference transcript, word codes, and the
/// optional global features triple.
#[derive(Debug, Clone, Default)]
pub struct Speaker {
    pub text: String,
    pub words: Vec<SpeakerWord>,
    /// energy / spectral_centroid / pitch for the global features block.
    pub global_features: Option<[i64; 3]>,
}

impl Speaker {
    /// Parses the reference speaker JSON (`load_speaker` shape:
    /// text, words with word/duration/c1/c2/features, and the optional
    /// global_features dict).
    pub fn from_json(value: &Value) -> Result<Self> {
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| SpeechError::Input {
                why: "speaker json is missing text".into(),
            })?
            .to_owned();
        let feature_triple = |obj: &Value| -> Result<[i64; 3]> {
            let pick = |name: &str| {
                obj.get(name)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("speaker features are missing {name}"),
                    })
            };
            Ok([pick("energy")?, pick("spectral_centroid")?, pick("pitch")?])
        };
        let mut words = Vec::new();
        for word in value
            .get("words")
            .and_then(Value::as_array)
            .ok_or_else(|| SpeechError::Input {
                why: "speaker json is missing words".into(),
            })?
        {
            let c1 = word
                .get("c1")
                .and_then(Value::as_array)
                .ok_or_else(|| SpeechError::Input {
                    why: "speaker word is missing c1".into(),
                })?
                .iter()
                .map(|v| v.as_i64().map(|x| x as i32))
                .collect::<Option<Vec<i32>>>()
                .ok_or_else(|| SpeechError::Input {
                    why: "speaker word c1 codes must be integers".into(),
                })?;
            let c2 = word
                .get("c2")
                .and_then(Value::as_array)
                .ok_or_else(|| SpeechError::Input {
                    why: "speaker word is missing c2".into(),
                })?
                .iter()
                .map(|v| v.as_i64().map(|x| x as i32))
                .collect::<Option<Vec<i32>>>()
                .ok_or_else(|| SpeechError::Input {
                    why: "speaker word c2 codes must be integers".into(),
                })?;
            words.push(SpeakerWord {
                word: word
                    .get("word")
                    .and_then(Value::as_str)
                    .ok_or_else(|| SpeechError::Input {
                        why: "speaker word is missing word".into(),
                    })?
                    .to_owned(),
                duration: word
                    .get("duration")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| SpeechError::Input {
                        why: "speaker word is missing duration".into(),
                    })?,
                c1,
                c2,
                features: feature_triple(word.get("features").ok_or_else(|| {
                    SpeechError::Input {
                        why: "speaker word is missing features".into(),
                    }
                })?)?,
            });
        }
        let global_features = match value.get("global_features") {
            Some(obj) if !obj.is_null() => Some(feature_triple(obj)?),
            _ => None,
        };
        Ok(Self {
            text,
            words,
            global_features,
        })
    }
}

/// Text normalizations the reference runs before prompt construction
/// (`PromptProcessor.text_normalizations`): collapse whitespace runs,
/// replace the ellipsis character, normalize curly quotes and dashes to
/// ASCII, strip, and drop control characters. Matches Python `re`
/// `\s` (which also matches the U+001C-U+001F file separators).
pub fn text_normalizations(text: &str) -> String {
    let is_space = |c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c);
    let mut collapsed = String::with_capacity(text.len());
    let mut last_space = false;
    for c in text.chars() {
        if is_space(c) {
            if !last_space {
                collapsed.push(' ');
            }
            last_space = true;
        } else {
            collapsed.push(c);
            last_space = false;
        }
    }
    let collapsed = collapsed.replace('\u{2026}', "...");
    let collapsed = collapsed.trim().to_owned();
    collapsed
        .chars()
        .filter_map(|c| match c {
            '\u{201c}' | '\u{201d}' => Some('"'),
            '\u{2018}' | '\u{2019}' => Some('\''),
            '\u{2013}' | '\u{2014}' => Some('-'),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

/// Reference `chunk_text`: split into sentences on runs of terminal
/// punctuation (including the CJK forms), then group whole words into
/// chunks of at most `max_words` (a single longer sentence overflows
/// into its own chunk, and the grouping can emit empty chunks exactly
/// like the reference join). `max_words` is the reference 30 default.
pub fn chunk_text(text: &str, max_words: usize) -> Vec<String> {
    let is_terminal = |c: char| {
        matches!(
            c,
            '.' | '!' | '?' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}' | '\u{fe55}' | '\u{fe56}'
        )
    };
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if is_terminal(c) {
            if !current.is_empty() {
                sentences.push(std::mem::take(&mut current));
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    let sentences: Vec<&str> = sentences
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk: Vec<&str> = Vec::new();
    let mut current_length = 0usize;
    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let word_count = words.len();
        if current_length + word_count > max_words {
            chunks.push(current_chunk.join(" "));
            current_chunk = Vec::new();
            current_length = 0;
        }
        current_chunk.extend(words);
        current_length += word_count;
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }
    chunks
}

/// Reference `_get_separator`: the sentence ender that fits the speaker
/// transcript script.
fn get_separator(text: &str) -> &'static str {
    let has = |range: std::ops::RangeInclusive<char>| text.chars().any(|c| range.contains(&c));
    if has('\u{3040}'..='\u{309f}') || has('\u{30a0}'..='\u{30ff}') || has('\u{4e00}'..='\u{9fff}')
    {
        "\u{3002}"
    } else {
        ". "
    }
}

/// Reference `merge_speaker_text`: splice the input text onto the
/// speaker transcript with the script-appropriate separator. Returns
/// the merged text and the appended suffix (the reference `rs.strip()`).
pub fn merge_speaker_text(input_text: &str, speaker_text: &str) -> (String, String) {
    let speaker_text = speaker_text.trim();
    let separator = get_separator(speaker_text);
    let allowed: &[char] = if separator == "\u{3002}" {
        &['\u{3002}', '\u{ff1f}', '\u{ff01}', '?', '!']
    } else {
        &['.', '?', '!']
    };
    let mut rs = String::new();
    if !speaker_text.is_empty() {
        let last_char = speaker_text.chars().last().unwrap();
        if !allowed.contains(&last_char) {
            rs = separator.to_owned();
        } else if separator != "\u{3002}" {
            rs = " ".to_owned();
        }
    }
    let output = format!("{}{}{}", speaker_text, rs, input_text.trim());
    (output, rs.trim().to_owned())
}

/// Reference `create_codes`: the `<|word_start|>` block per word with
/// the features header, the per-word time and features tokens, and the
/// interleaved c1/c2 pairs, joined by newlines. The reference indexes
/// `c2` alongside `c1` and crashes on a short book; this port refuses
/// instead.
pub fn create_codes(words: &[SpeakerWord]) -> Result<String> {
    let mut codes: Vec<String> = Vec::with_capacity(words.len());
    for word in words {
        if word.c2.len() < word.c1.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker word {:?} has {} c2 codes for {} c1 codes",
                    word.word,
                    word.c2.len(),
                    word.c1.len()
                ),
            });
        }
        let mut entry = String::new();
        entry.push_str(&word.word);
        entry.push_str(SpecialTokens::FEATURES);
        entry.push_str(&SpecialTokens::time(word.duration));
        for (name, value) in [
            ("energy", word.features[0]),
            ("spectral_centroid", word.features[1]),
            ("pitch", word.features[2]),
        ] {
            entry.push_str(&format!("<|{name}_{value}|>"));
        }
        entry.push_str(SpecialTokens::CODE);
        for (c1, c2) in word.c1.iter().zip(&word.c2) {
            entry.push_str(&SpecialTokens::c1(*c1));
            entry.push_str(&SpecialTokens::c2(*c2));
        }
        codes.push(format!(
            "{}{}{}",
            SpecialTokens::WORD_START,
            entry,
            SpecialTokens::WORD_END
        ));
    }
    Ok(codes.join("\n"))
}

/// Reference `get_global_features`: the `<|global_features_start|>` block.
pub fn get_global_features(features: &[i64; 3]) -> String {
    let mut out = String::from(SpecialTokens::GLOBAL_FEATURES_START);
    for (name, value) in [
        ("energy", features[0]),
        ("spectral_centroid", features[1]),
        ("pitch", features[2]),
    ] {
        out.push_str(&format!("<|{name}_{value}|>"));
    }
    out.push_str(SpecialTokens::GLOBAL_FEATURES_END);
    out.push('\n');
    out
}

/// The TTS `ModelConfig`: the llama/qwen2/qwen3 backbone args (selected
/// by `model_type`, the reference model map) plus the tokenizer name and
/// sample rate the reference dataclass adds.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub backbone: lm::DenseConfig,
    pub tokenizer_name: Option<String>,
    pub sample_rate: u32,
}

/// Reference sample rate (the deployed 24 kHz DAC).
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let model_type = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let backbone = match model_type.as_str() {
            "llama" => lm::llama::from_json(value)?,
            "qwen2" => lm::qwen2::from_json(value)?,
            "qwen3" => lm::qwen3::from_json(value)?,
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("Unsupported model type: {other}"),
                })
            }
        };
        let tokenizer_name = value
            .get("tokenizer_name")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let sample_rate = value
            .get("sample_rate")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_SAMPLE_RATE as u64) as u32;
        Ok(Self {
            model_type,
            backbone,
            tokenizer_name,
            sample_rate,
        })
    }
}

/// Sampling and generation options for one segment, with the reference
/// `generate` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 0.0 selects the exact greedy path (`make_sampler(0, ...)`).
    pub temperature: f32,
    pub top_p: f32,
    /// The reference default 0.05; `make_sampler` skips the filter at 0.
    pub min_p: f32,
    /// Minimum tokens the min-p filter cannot remove (reference 1).
    pub min_tokens_to_keep: usize,
    /// The reference default 40; disabled at 0 or negative.
    pub top_k: i32,
    pub logit_bias: Vec<(i32, f32)>,
    /// The reference default 1.1; disabled at 0.
    pub repetition_penalty: f32,
    /// The reference default 64.
    pub repetition_context_size: usize,
    pub max_tokens: usize,
    /// The tokenizer end-of-text ids `stream_generate` stops on. The
    /// sampled stop token is appended before the loop ends, exactly like
    /// the reference loop appending the final `stream_generate` response.
    pub stop_token_ids: Vec<i32>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.4,
            top_p: 0.9,
            min_p: 0.05,
            min_tokens_to_keep: 1,
            top_k: 40,
            logit_bias: Vec::new(),
            repetition_penalty: 1.1,
            repetition_context_size: 64,
            max_tokens: 1200,
            stop_token_ids: Vec::new(),
        }
    }
}

/// Why the generate loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    StopToken,
    MaxTokens,
}

/// One generated segment: sampled ids, the extracted two-book codes,
/// and the decoded mono waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<i32>,
    pub codebook1: Vec<i32>,
    pub codebook2: Vec<i32>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The loaded family model: backbone plus the family config.
pub struct Model {
    pub config: ModelConfig,
    pub backbone: lm::Backbone,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict.
    pub fn load(weights: &lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let backbone = lm::Backbone::load(weights, config.backbone.clone())?;
        Ok(Self { config, backbone })
    }

    /// Reference `get_completion_prompt` for one chunk: normalize the
    /// text, splice the speaker transcript in front (appending the
    /// separator to the speaker's last word, exactly like the reference
    /// mutating `speaker["words"][-1]`), and wrap the merged text in the
    /// audio-prompt frame followed by the word codes. Without a speaker
    /// the prompt is the bare text frame. The caller tokenizes the
    /// result with `add_special_tokens=False`. A speaker with no words
    /// is refused (the reference `words[-1]` index error).
    pub fn get_completion_prompt(text: &str, speaker: Option<&mut Speaker>) -> Result<String> {
        let normalized = text_normalizations(text);
        let (prompt_text, codes) = match speaker {
            Some(speaker) => {
                if speaker.words.is_empty() {
                    return Err(SpeechError::Input {
                        why: "speaker has no words to merge".into(),
                    });
                }
                let (merged, separator) = merge_speaker_text(&normalized, &speaker.text);
                if let Some(last) = speaker.words.last_mut() {
                    last.word.push_str(&separator);
                }
                (merged, create_codes(&speaker.words)?)
            }
            None => (normalized, String::new()),
        };
        let mut prompt = format!(
            "{}\n{}{}{}\n{}\n",
            SpecialTokens::BOS,
            SpecialTokens::TEXT_START,
            prompt_text,
            SpecialTokens::TEXT_END,
            SpecialTokens::AUDIO_START
        );
        if !codes.is_empty() {
            prompt.push_str(&codes);
            prompt.push('\n');
            prompt.push_str(SpecialTokens::WORD_START);
        }
        Ok(prompt)
    }

    /// Runs one segment end to end: prompt cache, greedy or sampled
    /// generate loop with the reference logits processors and the
    /// top-p/min-p/top-k sampler chain, stop at a stop token or
    /// `max_tokens`, then extract the two codebooks and decode all codes
    /// to one loudness-normalized waveform. `seed` only feeds the
    /// categorical draws; temperature 0 never draws and is fully
    /// deterministic.
    pub fn generate(
        &mut self,
        dac: &Dac,
        maps: &AudioTokenMaps,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (tokens, stopped_by) = self.generate_tokens_from_prompt(prompt_ids, params, seed)?;
        let (codebook1, codebook2) = maps.extract_audio_from_tokens(&tokens);
        let samples = if codebook1.is_empty() {
            Vec::new()
        } else {
            decode_codes_to_audio(dac, &codebook1, &codebook2)?
        };
        Ok(Generated {
            tokens,
            codebook1,
            codebook2,
            samples,
            stopped_by,
        })
    }

    /// The generate loop alone: prefill, greedy or sampled steps with the
    /// reference logits processors, and the stop conditions. The sampled
    /// stop token is included in the returned ids (the reference appends
    /// the final `stream_generate` response before its loop ends).
    /// Callers that stream decode intervals (out of scope here) drive
    /// this directly.
    pub fn generate_tokens_from_prompt(
        &mut self,
        prompt_ids: &[i32],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<i32>, StopReason)> {
        if prompt_ids.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let mut caches = lm::make_prompt_cache(self.backbone.config.num_hidden_layers);
        // The reference `generate_step` prefills everything except the
        // last token unconditionally (processors never see it), then
        // steps the last token through the processors.
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        if !prefill.is_empty() {
            self.backbone
                .forward(Input::Tokens(prefill), Some(&mut caches))?;
        }
        let mut logits = self
            .backbone
            .forward(Input::Tokens(last), Some(&mut caches))?;
        if logits.len() != self.backbone.config.vocab_size {
            return Err(SpeechError::Tensor {
                name: "logits".into(),
                why: format!(
                    "expected {} values for one position, got {}",
                    self.backbone.config.vocab_size,
                    logits.len()
                ),
            });
        }

        // The reference logits processors accumulate every token that
        // passed through a step call: the last prompt token first, then
        // each generated token.
        let mut context: Vec<i32> = Vec::with_capacity(params.max_tokens + 1);
        context.push(last[last.len() - 1]);
        let mut tokens = Vec::with_capacity(params.max_tokens);
        let mut stopped_by = StopReason::MaxTokens;
        let mut keys = rng::KeySequence::new(seed);

        for step in 0..params.max_tokens {
            let mut processed = logits;
            if !params.logit_bias.is_empty() {
                sampling::apply_logit_bias(&mut processed, &params.logit_bias)?;
            }
            if params.repetition_penalty != 0.0 {
                sampling::apply_repetition_penalty(
                    &mut processed,
                    &context,
                    params.repetition_penalty,
                    params.repetition_context_size,
                )?;
            }
            let logprobs = sampling::log_softmax(&processed);
            let next = if params.temperature == 0.0 {
                sampling::argmax(&logprobs)
            } else {
                // make_sampler chain order: top-p, then min-p, then
                // XTC (never enabled here), then top-k.
                let mut filtered = logprobs;
                if params.top_p > 0.0 && params.top_p < 1.0 {
                    sampling::apply_top_p(&mut filtered, params.top_p);
                }
                if params.min_p != 0.0 {
                    sampling::apply_min_p(&mut filtered, params.min_p, params.min_tokens_to_keep)?;
                }
                if params.top_k > 0 {
                    sampling::apply_top_k(&mut filtered, params.top_k as usize)?;
                }
                sampling::sample_categorical(&filtered, params.temperature, keys.next())
            };
            let next = i32::try_from(next).map_err(|_| SpeechError::Input {
                why: format!("sampled token {next} does not fit i32"),
            })?;
            context.push(next);
            tokens.push(next);
            if params.stop_token_ids.contains(&next) {
                stopped_by = StopReason::StopToken;
                break;
            }
            if step + 1 == params.max_tokens {
                stopped_by = StopReason::MaxTokens;
                break;
            }
            logits = self
                .backbone
                .forward(Input::Tokens(&[next]), Some(&mut caches))?;
        }
        Ok((tokens, stopped_by))
    }
}

/// Reference `DacInterface.decode` on the extracted two books: decode
/// the codes in 4096-frame chunks (the reference `chunk_length`), join
/// the chunks along samples, and run the loudness stage over the mono
/// waveform. Deviation: the reference joins its decode chunks along the
/// size-1 channel axis, which matches only while the decode is a single
/// chunk (a second `[1, samples, 1]` chunk mismatches on the samples
/// axis and `mx.concatenate` raises), so inputs beyond 4096 codes crash
/// upstream; this port concatenates along samples, the single-chunk
/// path the reference takes and the intent beyond it. See the
/// [loudness] module docs for the shape contract and the family README
/// for the tolerance evidence.
pub fn decode_codes_to_audio(dac: &Dac, codebook1: &[i32], codebook2: &[i32]) -> Result<Vec<f32>> {
    if codebook1.len() != codebook2.len() {
        return Err(SpeechError::Input {
            why: format!(
                "codebooks differ in length: {} vs {}",
                codebook1.len(),
                codebook2.len()
            ),
        });
    }
    const CHUNK_LENGTH: usize = 4096;
    let mut samples = Vec::new();
    for start in (0..codebook1.len()).step_by(CHUNK_LENGTH) {
        let end = (start + CHUNK_LENGTH).min(codebook1.len());
        samples.extend(dac.decode_codes(&[
            codebook1[start..end].to_vec(),
            codebook2[start..end].to_vec(),
        ])?);
    }
    Ok(process_audio_array(
        &samples,
        dac.config.sample_rate,
        -18.0,
        -1.0,
        0.4,
    ))
}

#[cfg(test)]
mod tests;
