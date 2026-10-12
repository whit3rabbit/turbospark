//! MOSS-TTS TTS family (OpenMOSS MOSS-TTS / MOSS-TTSD / MOSS-TTS-Local).
//!
//! Port of `mlx_audio/tts/models/moss_tts/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/moss_tts):
//! a Qwen3 backbone over `n_vq + 1` interleaved channels (column 0 is the
//! text token, columns 1.. carry one code per RVQ book) with the two
//! inference variants the reference config selects:
//!
//! - Delay pattern (`model_type` `moss_tts_delay`, the MOSS-TTS v1.0 /
//!   v1.5 / MOSS-TTSD checkpoints): the backbone heads are one text head
//!   (`lm_heads.0`) plus one head per book (`lm_heads.1..`, last pad slot
//!   masked), and per-step generation runs the delay bookkeeping
//!   (`generate_delay_pattern_ids`): text rows carry audio-start /
//!   assistant-slot / assistant-delay-slot / audio-end / im-end tokens
//!   while book `b` of frame `f` is written at row `f + b`; the decode
//!   undoes the delay (`apply_de_delay_pattern`), drops all-pad rows,
//!   splits contiguous segments, and decodes each through the MOSS audio
//!   tokenizer.
//! - v1.5 local transformer (`model_type` `moss_tts_local` or a
//!   `gpt2_config` block, the MOSS-TTS-Local-Transformer-v1.5
//!   checkpoint): the qwen3 global transformer plus separate
//!   `text_lm_head` / per-book `audio_lm_heads`, and a GPT-2 local
//!   transformer (the moss_tts_nano `gpt2.py`, re-declared in
//!   [gpt2.rs](gpt2.rs)) that turns the last global hidden state into the
//!   continue/stop choice (assistant slot vs audio end) and then one code
//!   per book (`generate_v15_local_ids`). The prompt is row-based like
//!   the nano family: all-audio-pad text rows and slot-token audio rows.
//!
//! The legacy local-transformer checkpoints (the four
//! `additional_mlp_ffn_hidden_size` / `local_ffn_hidden_size` /
//! `local_hidden_size` / `local_num_layers` fields with no `gpt2_config`
//! and `model_type != moss_tts_local`) are refused at load with a named
//! [`SpeechError::Unsupported`]; see the README open items.
//!
//! Scope of this port, matching the reference single-prompt paths: batch
//! size 1 (the reference refuses batches too), non-streaming decode (the
//! reference streams only the v1.5 local variant through a codec session;
//! not ported), and prompt construction from caller-encoded ids. Text
//! tokenization (the checkpoint tokenizer) and the `text.py`
//! `normalize_tts_text` cleanup are caller concerns; the delay chat
//! template, audio-placeholder expansion, unified-code assembly, delay
//! pattern, and the v1.5 row builders are ported here. See
//! [README.md](README.md) for the verification evidence and open items.

use serde_json::Value;

use self::gpt2::LocalGpt2;
use self::sampling as family_sampling;
use crate::codec::moss_audio_tokenizer::MossAudioTokenizer;
use crate::lm::engine::{Backbone, Input};
use crate::lm::{qwen3, Weights};
use crate::music::minimax_music3::rng;
use crate::nn::Linear;
use crate::tts::moss_tts_nano::Gpt2Config;
use crate::{quant, Result, SpeechError};

mod gpt2;
mod sampling;

#[cfg(test)]
mod tests;

/// `config.DEFAULT_AUDIO_TOKENIZER_REPO`.
pub const DEFAULT_AUDIO_TOKENIZER_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";
/// `config.DEFAULT_AUDIO_TOKENIZER_V2_REPO` (the v1.5 local transformer).
pub const DEFAULT_AUDIO_TOKENIZER_V2_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2";

/// `processor.AUDIO_PLACEHOLDER`; the caller splits message content on it.
pub const AUDIO_PLACEHOLDER: &str = "<|audio|>";

/// `_INT64_MAX`, the `delayed_lengths` sentinel (the "not engaged" state).
const INT64_MAX: i64 = 9223372036854775807;

/// `ModelConfig.from_dict` defaults.
const DEFAULT_N_VQ: usize = 32;
const DEFAULT_AUDIO_VOCAB_SIZE: usize = 1024;
const DEFAULT_AUDIO_USER_SLOT: i32 = 151654;
const DEFAULT_AUDIO_ASSISTANT_SLOT: i32 = 151656;
const DEFAULT_AUDIO_ASSISTANT_DELAY_SLOT: i32 = 151662;
const DEFAULT_AUDIO_START: i32 = 151652;
const DEFAULT_AUDIO_END: i32 = 151653;
const DEFAULT_AUDIO_PAD: i32 = 1024;
const DEFAULT_PAD: i32 = 151643;
const DEFAULT_IM_START: i32 = 151644;
const DEFAULT_IM_END: i32 = 151645;
const DEFAULT_SAMPLING_RATE: u32 = 24000;

/// Which inference stack the config selects
/// (`ModelConfig.is_v15_local_transformer` / `is_legacy_local_transformer`
/// / else the delay pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `generate_delay_pattern_ids` (the `moss_tts_delay` model type).
    Delay,
    /// The refused legacy local-transformer layout.
    LegacyLocal,
    /// `generate_v15_local_ids` (`moss_tts_local` or a `gpt2_config`).
    V15Local,
}

/// The family-local qwen3 language config: the shared dense
/// configuration the reference `Qwen3ModelConfig.from_dict` builds.
#[derive(Debug, Clone)]
pub struct LanguageConfig {
    pub dense: crate::lm::DenseConfig,
}

impl LanguageConfig {
    pub fn hidden_size(&self) -> usize {
        self.dense.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.dense.vocab_size
    }
}

/// The TTS `ModelConfig` (`config.ModelConfig.from_dict`); dropout and
/// initializer ranges are training-only and dropped.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub language_config: LanguageConfig,
    pub n_vq: usize,
    pub audio_vocab_size: usize,
    /// `None` falls back to `audio_vocab_size` per book at load time.
    pub audio_codebook_sizes: Option<Vec<usize>>,
    pub audio_user_slot_token_id: i32,
    pub audio_assistant_slot_token_id: i32,
    pub audio_assistant_gen_slot_token_id: i32,
    pub audio_assistant_delay_slot_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub audio_pad_token_id: i32,
    pub audio_pad_code: i32,
    pub pad_token_id: i32,
    pub im_start_token_id: i32,
    pub im_end_token_id: i32,
    pub sampling_rate: u32,
    /// The resolved audio tokenizer repo
    /// (`audio_tokenizer_pretrained_name_or_path` fallback chain).
    pub audio_tokenizer_repo: String,
    pub additional_mlp_ffn_hidden_size: Option<usize>,
    pub local_ffn_hidden_size: Option<usize>,
    pub local_hidden_size: Option<usize>,
    pub local_num_layers: Option<usize>,
    pub gpt2_config: Option<Gpt2Config>,
    pub local_transformer_layers: usize,
    pub local_text_head_mode: String,
    pub use_static_local_kv_cache: bool,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let language_value = value
            .get("language_config")
            .or_else(|| value.get("qwen3_config"))
            .ok_or_else(|| SpeechError::BadConfig {
                field: "language_config".into(),
                why: "MOSS-TTS requires language_config (or qwen3_config)".into(),
            })?;
        // from_dict setdefaults: model_type qwen3, untied head (the
        // upstream checkpoint keeps an explicit text head), and the
        // rope_parameters.rope_theta alias.
        let mut language_params = language_value.clone();
        if let Some(object) = language_params.as_object_mut() {
            object
                .entry("model_type".to_owned())
                .or_insert_with(|| Value::String("qwen3".into()));
            object
                .entry("tie_word_embeddings".to_owned())
                .or_insert(Value::Bool(false));
            if !object.contains_key("rope_theta") {
                let rope_theta = object
                    .get("rope_parameters")
                    .and_then(|v| v.get("rope_theta"))
                    .cloned();
                if let Some(rope_theta) = rope_theta {
                    object.insert("rope_theta".to_owned(), rope_theta);
                }
            }
        }
        let language_config = LanguageConfig {
            dense: qwen3::from_json(&language_params)?,
        };

        let model_type = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("moss_tts_delay")
            .to_owned();
        let n_vq = value
            .get("n_vq")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_N_VQ as u64) as usize;
        // Zero channels leaves no audio tables and makes the delay frame
        // width (`n_vq - 1` trims) degenerate. Refused before any sizing.
        if n_vq == 0 {
            return Err(SpeechError::BadConfig {
                field: "n_vq".into(),
                why: "must be at least 1".into(),
            });
        }
        let audio_vocab_size = value
            .get("audio_vocab_size")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_AUDIO_VOCAB_SIZE as u64) as usize;
        let audio_codebook_sizes = match value.get("audio_codebook_sizes").and_then(Value::as_array)
        {
            Some(list) => {
                let sizes: Vec<usize> = list
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<usize>>>()
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "audio_codebook_sizes".into(),
                        why: "must be a list of integers".into(),
                    })?;
                if sizes.len() != n_vq {
                    return Err(SpeechError::BadConfig {
                        field: "audio_codebook_sizes".into(),
                        why: format!(
                            "must have one entry per VQ channel (expected {n_vq}, got {})",
                            sizes.len()
                        ),
                    });
                }
                Some(sizes)
            }
            None => None,
        };
        let audio_pad_token_id = value
            .get("audio_pad_token_id")
            .and_then(Value::as_i64)
            .or_else(|| value.get("audio_pad_code").and_then(Value::as_i64))
            .map(|n| n as i32)
            .unwrap_or(DEFAULT_AUDIO_PAD);
        let audio_pad_code = value
            .get("audio_pad_code")
            .and_then(Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(audio_pad_token_id);
        // The assistant slot falls through audio_assistant_slot_token_id,
        // audio_assistant_gen_slot_token_id, then 151656; an explicit null
        // takes the default like the reference None check.
        let assistant_slot = value
            .get("audio_assistant_slot_token_id")
            .and_then(Value::as_i64)
            .or_else(|| {
                value
                    .get("audio_assistant_gen_slot_token_id")
                    .and_then(Value::as_i64)
            })
            .map(|n| n as i32)
            .unwrap_or(DEFAULT_AUDIO_ASSISTANT_SLOT);
        let i = |key: &str, default: i32| {
            value
                .get(key)
                .and_then(Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(default)
        };
        let tokenizer_repo = value
            .get("audio_tokenizer_pretrained_name_or_path")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .get("audio_tokenizer_name_or_path")
                    .and_then(Value::as_str)
            })
            .unwrap_or(if model_type == "moss_tts_local" {
                DEFAULT_AUDIO_TOKENIZER_V2_REPO
            } else {
                DEFAULT_AUDIO_TOKENIZER_REPO
            })
            .to_owned();
        let gpt2_config = match value.get("gpt2_config") {
            Some(v) if v.is_object() => Some(Gpt2Config::from_json(v)?),
            _ => None,
        };
        Ok(Self {
            model_type,
            language_config,
            n_vq,
            audio_vocab_size,
            audio_codebook_sizes,
            audio_user_slot_token_id: i("audio_user_slot_token_id", DEFAULT_AUDIO_USER_SLOT),
            audio_assistant_slot_token_id: assistant_slot,
            audio_assistant_gen_slot_token_id: assistant_slot,
            audio_assistant_delay_slot_token_id: i(
                "audio_assistant_delay_slot_token_id",
                DEFAULT_AUDIO_ASSISTANT_DELAY_SLOT,
            ),
            audio_start_token_id: i("audio_start_token_id", DEFAULT_AUDIO_START),
            audio_end_token_id: i("audio_end_token_id", DEFAULT_AUDIO_END),
            audio_pad_token_id,
            audio_pad_code,
            pad_token_id: i("pad_token_id", DEFAULT_PAD),
            im_start_token_id: i("im_start_token_id", DEFAULT_IM_START),
            im_end_token_id: i("im_end_token_id", DEFAULT_IM_END),
            sampling_rate: value
                .get("sampling_rate")
                .or_else(|| value.get("sample_rate"))
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_SAMPLING_RATE as u64) as u32,
            audio_tokenizer_repo: tokenizer_repo,
            additional_mlp_ffn_hidden_size: usize_option(value, "additional_mlp_ffn_hidden_size"),
            local_ffn_hidden_size: usize_option(value, "local_ffn_hidden_size"),
            local_hidden_size: usize_option(value, "local_hidden_size"),
            local_num_layers: usize_option(value, "local_num_layers"),
            gpt2_config,
            local_transformer_layers: value
                .get("local_transformer_layers")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
            local_text_head_mode: value
                .get("local_text_head_mode")
                .and_then(Value::as_str)
                .unwrap_or("full_vocab")
                .to_owned(),
            use_static_local_kv_cache: value
                .get("use_static_local_kv_cache")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        })
    }

    /// `is_v15_local_transformer`: the local model type or a gpt2_config.
    pub fn is_v15_local_transformer(&self) -> bool {
        self.model_type == "moss_tts_local" || self.gpt2_config.is_some()
    }

    /// `is_legacy_local_transformer`: all four legacy local fields set.
    pub fn is_legacy_local_transformer(&self) -> bool {
        self.additional_mlp_ffn_hidden_size.is_some()
            && self.local_ffn_hidden_size.is_some()
            && self.local_hidden_size.is_some()
            && self.local_num_layers.is_some()
    }

    /// The variant the reference `Model.__init__` dispatches on.
    pub fn variant(&self) -> Variant {
        if self.is_v15_local_transformer() {
            Variant::V15Local
        } else if self.is_legacy_local_transformer() {
            Variant::LegacyLocal
        } else {
            Variant::Delay
        }
    }

    /// The per-book head widths: `audio_codebook_sizes` or
    /// `audio_vocab_size` repeated `n_vq` times.
    pub fn codebook_sizes(&self) -> Vec<usize> {
        match &self.audio_codebook_sizes {
            Some(sizes) => sizes.clone(),
            None => vec![self.audio_vocab_size; self.n_vq],
        }
    }

    /// `local_gpt2_config`: the global gpt2 config with the local
    /// context (`n_vq + 1` positions) and depth.
    pub fn local_gpt2_config(&self) -> Result<Gpt2Config> {
        let mut config = self
            .gpt2_config
            .clone()
            .ok_or_else(|| SpeechError::BadConfig {
                field: "gpt2_config".into(),
                why: "the v1.5 local transformer requires a gpt2_config block".into(),
            })?;
        config.n_positions = self.n_vq + 1;
        config.n_layer = self.local_transformer_layers;
        Ok(config)
    }
}

fn usize_option(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

// ---------------------------------------------------------------------------
// Delay-pattern prompt building (processor.py in id space)
// ---------------------------------------------------------------------------

/// Message role of the delay chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// The caller-encoded chat template pieces
/// (`apply_chat_template(role, content, add_generation_prompt)` =
/// `<|im_start|>{role}\n{content}<|im_end|>\n` plus
/// `<|im_start|>assistant\n` for the generation prompt).
#[derive(Debug, Clone, Copy)]
pub struct ChatTemplateIds<'a> {
    /// `encode("user\n")`.
    pub user_role: &'a [i32],
    /// `encode("assistant\n")`.
    pub assistant_role: &'a [i32],
    /// `encode("\n")`, the newline after `<|im_end|>`.
    pub after_im_end: &'a [i32],
}

/// One piece of a rendered message content: caller-encoded text or one
/// `<|audio|>` placeholder with its reference codes.
#[derive(Debug, Clone, Copy)]
pub enum ContentPiece<'a> {
    Text(&'a [i32]),
    Audio(&'a [Vec<i32>]),
}

/// The composed content id stream and the normalized per-block codes
/// `build_delay_message_ids` returns.
pub type DelayMessageIds = (Vec<i32>, Vec<Vec<Vec<i32>>>);

/// One conversation message for the delay processor.
#[derive(Debug, Clone, Copy)]
pub struct DelayMessage<'a> {
    pub role: Role,
    pub pieces: &'a [ContentPiece<'a>],
}

/// Processor mode (`mode` in the reference `__call__`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayMode {
    Generation,
    Continuation,
}

/// `apply_delay_pattern`: book `b` of frame `f` moves to row `f + b`,
/// every unwritten slot filled with the pad code.
pub fn apply_delay_pattern(
    codes: &[Vec<i32>],
    pad_code: i32,
    n_vq: usize,
) -> Result<Vec<Vec<i32>>> {
    if codes.iter().any(|row| row.len() != n_vq) {
        return Err(SpeechError::Input {
            why: format!("Expected codes shape [frames, {n_vq}]"),
        });
    }
    let frames = codes.len();
    let mut delayed = vec![vec![pad_code; n_vq]; frames + n_vq.saturating_sub(1)];
    for book in 0..n_vq {
        for (frame, row) in codes.iter().enumerate() {
            delayed[frame + book][book] = row[book];
        }
    }
    Ok(delayed)
}

/// `apply_de_delay_pattern`: book `b` of row `t` moves back to frame
/// `t - b`; the trailing `n_vq - 1` partial rows are dropped.
pub fn apply_de_delay_pattern(delayed: &[Vec<i32>], n_vq: usize) -> Result<Vec<Vec<i32>>> {
    if delayed.iter().any(|row| row.len() != n_vq) {
        return Err(SpeechError::Input {
            why: format!("Expected delayed codes shape [frames, {n_vq}]"),
        });
    }
    let output_length = (delayed.len() as i64) - (n_vq as i64) + 1;
    if output_length <= 0 {
        return Ok(Vec::new());
    }
    let output_length = output_length as usize;
    let mut tokens = vec![vec![0i32; n_vq]; output_length];
    for book in 0..n_vq {
        for frame in 0..output_length {
            tokens[frame][book] = delayed[frame + book][book];
        }
    }
    Ok(tokens)
}

/// `_normalize_audio_codes_list` for one block: transposes codebook-major
/// input (fewer channels than frames with at least `n_vq` rows), refuses
/// short channel counts, and truncates to `n_vq`.
pub fn normalize_audio_codes(codes: &[Vec<i32>], n_vq: usize) -> Result<Vec<Vec<i32>>> {
    if codes.is_empty() {
        return Ok(Vec::new());
    }
    let rows = codes.len();
    let cols = codes[0].len();
    if codes.iter().any(|row| row.len() != cols) {
        return Err(SpeechError::Input {
            why: "audio codes rows must all have the same width".into(),
        });
    }
    let (frames, channels) = if cols < n_vq && rows >= n_vq {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if channels < n_vq {
        return Err(SpeechError::Input {
            why: format!("audio_codes channels ({channels}) < model n_vq ({n_vq})"),
        });
    }
    let source: Vec<Vec<i32>> = if channels != cols {
        // Transposed input: `source[frame][channel]` is the caller's
        // `codes[channel][frame]`.
        (0..frames)
            .map(|frame| (0..channels).map(|channel| codes[channel][frame]).collect())
            .collect()
    } else {
        codes.to_vec()
    };
    Ok(source.into_iter().map(|row| row[..n_vq].to_vec()).collect())
}

impl Model {
    /// `apply_chat_template` + `_replace_audio_placeholders` in id space:
    /// returns the composed content id stream (audio blocks expanded into
    /// audio-start / assistant-slot / assistant-delay-slot / audio-end id
    /// runs; zero-frame placeholders collapse to start + end) and the
    /// normalized per-block reference codes.
    pub fn build_delay_message_ids(
        &self,
        role: Role,
        pieces: &[ContentPiece],
        template: &ChatTemplateIds,
        add_generation_prompt: bool,
    ) -> Result<DelayMessageIds> {
        let use_delay_pattern = true;
        // The reference substitutes role-appropriate slot tokens into the
        // audio blocks: user messages carry the user slot in BOTH the
        // generation and delay positions; assistant messages carry the
        // assistant gen slot and the assistant delay slot.
        let (slot_token_id, delay_slot_token_id) = match role {
            Role::User => (
                self.config.audio_user_slot_token_id,
                self.config.audio_user_slot_token_id,
            ),
            Role::Assistant => (
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
            ),
        };
        let mut content_ids: Vec<i32> = Vec::new();
        let mut audio_codes_list: Vec<Vec<Vec<i32>>> = Vec::new();
        for piece in pieces {
            match piece {
                ContentPiece::Text(ids) => content_ids.extend_from_slice(ids),
                ContentPiece::Audio(codes) => {
                    let normalized = normalize_audio_codes(codes, self.config.n_vq)?;
                    let frames = normalized.len();
                    audio_codes_list.push(normalized);
                    content_ids.push(self.config.audio_start_token_id);
                    if frames > 0 {
                        for _ in 0..frames {
                            content_ids.push(slot_token_id);
                        }
                        if use_delay_pattern {
                            for _ in 0..self.config.n_vq.saturating_sub(1) {
                                content_ids.push(delay_slot_token_id);
                            }
                        }
                    }
                    content_ids.push(self.config.audio_end_token_id);
                }
            }
        }
        let role_ids = match role {
            Role::User => template.user_role,
            Role::Assistant => template.assistant_role,
        };
        let mut ids = Vec::with_capacity(
            2 + role_ids.len() + content_ids.len() + template.after_im_end.len(),
        );
        ids.push(self.config.im_start_token_id);
        ids.extend_from_slice(role_ids);
        ids.extend_from_slice(&content_ids);
        ids.push(self.config.im_end_token_id);
        ids.extend_from_slice(template.after_im_end);
        if add_generation_prompt {
            ids.push(self.config.im_start_token_id);
            ids.extend_from_slice(template.assistant_role);
        }
        Ok((ids, audio_codes_list))
    }

    /// `_get_unified_codes`: walks the composed content stream and builds
    /// the `[rows, n_vq + 1]` grid whose column 0 carries the text ids
    /// and whose audio columns carry the pad code with the (delayed)
    /// reference codes interleaved at the placeholder positions.
    /// `truncation` (continuation mode) drops the last `n_vq - 1` delayed
    /// rows of the final block and the trailing pad rows instead of
    /// appending them, which also truncates the text column.
    pub fn build_delay_unified_codes(
        &self,
        content_ids: &[i32],
        audio_codes_list: &[&[Vec<i32>]],
        truncation: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let n_vq = self.config.n_vq;
        let pad_code = self.config.audio_pad_code;
        let codes_list = audio_codes_list
            .iter()
            .map(|codes| normalize_audio_codes(codes, n_vq))
            .collect::<Result<Vec<_>>>()?;

        let audio_start_indices: Vec<usize> = content_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.config.audio_start_token_id)
            .map(|(index, _)| index)
            .collect();
        let audio_end_indices: Vec<usize> = content_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.config.audio_end_token_id)
            .map(|(index, _)| index)
            .collect();
        if audio_start_indices.len() != codes_list.len()
            || audio_end_indices.len() != codes_list.len()
        {
            return Err(SpeechError::Input {
                why: "Audio placeholders do not match the provided audio codes list".into(),
            });
        }

        // Sections are BLOCKS of rows like the reference `sections` list;
        // the truncation shortens the last block.
        let mut blocks: Vec<Vec<Vec<i32>>> = Vec::new();
        if codes_list.is_empty() {
            blocks.push(vec![vec![pad_code; n_vq]; content_ids.len()]);
        } else {
            let mut prefix_index = 0usize;
            for ((&start_index, &end_index), codes) in audio_start_indices
                .iter()
                .zip(&audio_end_indices)
                .zip(&codes_list)
            {
                let delayed = apply_delay_pattern(codes, pad_code, n_vq)?;
                // A start id before the previous end id means interleaved or
                // reordered placeholders; `-` would underflow (panic in debug).
                let gap =
                    start_index
                        .checked_sub(prefix_index)
                        .ok_or_else(|| SpeechError::Input {
                            why: format!(
                                "audio start at {start_index} precedes the previous audio end at \
                             {prefix_index}; placeholders must be ordered"
                            ),
                        })?;
                blocks.push(vec![vec![pad_code; n_vq]; gap + 1]);
                blocks.push(delayed);
                prefix_index = end_index;
            }
            if truncation {
                let last = blocks.last_mut().ok_or_else(|| SpeechError::Input {
                    why: "truncation requires an audio section".into(),
                })?;
                let keep = last.len().saturating_sub(n_vq.saturating_sub(1));
                last.truncate(keep);
            } else {
                let tail = content_ids.len() - audio_end_indices[audio_end_indices.len() - 1];
                blocks.push(vec![vec![pad_code; n_vq]; tail]);
            }
        }
        let sections: Vec<Vec<i32>> = blocks.into_iter().flatten().collect();

        if sections.len() > content_ids.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "audio grid has {} rows for {} text ids; the content stream is shorter \
                     than the audio blocks",
                    sections.len(),
                    content_ids.len()
                ),
            });
        }
        let mut rows = Vec::with_capacity(sections.len());
        for (section, &text_id) in sections.iter().zip(content_ids) {
            let mut row = Vec::with_capacity(n_vq + 1);
            row.push(text_id);
            row.extend_from_slice(section);
            rows.push(row);
        }
        Ok(rows)
    }

    /// The delay processor `__call__` for one conversation: chat
    /// template, audio-block expansion, and unified codes per message,
    /// concatenated. Generation mode takes an odd message count ending on
    /// the user (generation prompt appended); continuation mode takes an
    /// even count ending on the assistant and truncates the final block.
    pub fn build_delay_prompt(
        &self,
        messages: &[DelayMessage],
        template: &ChatTemplateIds,
        mode: DelayMode,
    ) -> Result<Vec<Vec<i32>>> {
        let generation = mode == DelayMode::Generation;
        if generation == (messages.len() % 2 == 0) {
            return Err(SpeechError::Input {
                why: "Invalid conversation length for mode".into(),
            });
        }
        let expected_last = if generation {
            Role::User
        } else {
            Role::Assistant
        };
        if messages
            .last()
            .map(|message| message.role != expected_last)
            .unwrap_or(true)
        {
            return Err(SpeechError::Input {
                why: "Invalid final role for mode".into(),
            });
        }
        let mut rows: Vec<Vec<i32>> = Vec::new();
        for (index, message) in messages.iter().enumerate() {
            let add_generation_prompt = generation && index + 1 == messages.len();
            let (content_ids, codes) = self.build_delay_message_ids(
                message.role,
                message.pieces,
                template,
                add_generation_prompt,
            )?;
            let code_refs: Vec<&[Vec<i32>]> = codes.iter().map(|block| block.as_slice()).collect();
            // Reference processor: truncation is continuation mode AND not a
            // user message. User audio keeps its trailing text pad rows.
            let truncation = !generation && message.role != Role::User;
            rows.extend(self.build_delay_unified_codes(&content_ids, &code_refs, truncation)?);
        }
        Ok(rows)
    }
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// The loaded family model; the variant picks the stack.
pub struct Model {
    pub config: ModelConfig,
    inner: Inner,
}

enum Inner {
    Delay(Box<DelayStack>),
    V15Local(Box<V15Stack>),
}

/// The delay-pattern stack: qwen3 backbone, one embedding table per book
/// (`emb_ext`, `audio_vocab_size + 1` rows), and the text + audio heads.
struct DelayStack {
    language_model: Backbone,
    emb_ext: Vec<Vec<f32>>,
    lm_heads: Vec<Linear>,
}

/// The v1.5 local stack: qwen3 global transformer, per-book audio
/// embeddings, the text and audio heads, the optional binary local text
/// head, and the GPT-2 local transformer.
struct V15Stack {
    transformer: Backbone,
    audio_embeddings: Vec<Vec<f32>>,
    text_lm_head: Linear,
    audio_lm_heads: Vec<Linear>,
    local_text_lm_head: Option<Linear>,
    local_transformer: LocalGpt2,
}

/// Loads the shared qwen3 backbone, flipping the untied head off when the
/// checkpoint carries no `lm_head` (the family never reads the backbone
/// head; only `forward_hidden` runs, so the tied fallback is inert).
fn load_language_model(weights: &Weights, dense: &crate::lm::DenseConfig) -> Result<Backbone> {
    let mut dense = dense.clone();
    if !dense.tie_word_embeddings
        && !weights.contains("lm_head")
        && !weights.contains_absolute("lm_head")
    {
        dense.tie_word_embeddings = true;
    }
    Backbone::load(weights, dense)
}

/// Opens the checkpoint with the backbone prefix the converted saves
/// use: `{name}.` (the mlx model tree), optionally under the raw
/// composite `model.` prefix. Falls back to the unprefixed view when
/// neither layout matches (an already-prefixed caller view).
fn open_backbone_weights(dir: &std::path::Path, backbone: &str) -> Result<Weights> {
    let probe = Weights::open(dir)?;
    if probe.contains_absolute(&format!("{backbone}.embed_tokens")) {
        let prefix = format!("{backbone}.");
        return Ok(probe.with_prefix(prefix));
    }
    if probe.contains_absolute(&format!("model.{backbone}.embed_tokens")) {
        let prefix = format!("model.{backbone}.");
        return Weights::open(dir).map(|w| w.with_prefix(prefix));
    }
    Ok(probe)
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors). The
    /// backbone prefix is probed from the converted (model-tree) layouts
    /// the mlx saves use: `language_model.` for the delay variant and
    /// `transformer.` for the v1.5 local variant, each optionally under
    /// the raw composite `model.` prefix.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let parsed = ModelConfig::from_json(&config_value)?;
        match parsed.variant() {
            Variant::V15Local => {
                let transformer = open_backbone_weights(dir, "transformer")?;
                let weights = Weights::open(dir)?;
                Self::load_v15_local(&transformer, &weights, parsed)
            }
            Variant::LegacyLocal => Err(SpeechError::Unsupported {
                why: "the legacy local-transformer MOSS-TTS checkpoints (four local size \
                      fields, no gpt2_config) are not ported; see the family README open items"
                    .into(),
            }),
            Variant::Delay => {
                let weights = open_backbone_weights(dir, "language_model")?;
                Self::load_delay(&weights, parsed)
            }
        }
    }

    /// Loads the delay-pattern variant from opened weights and the parsed
    /// config. Tensor names are the checkpoint's own: the backbone under
    /// the `Weights` prefix (`model.` for raw HF saves, empty for the
    /// sanitized mlx saves), `emb_ext.{channel}` and `lm_heads.{index}`
    /// prefix-free.
    pub fn load(weights: &Weights, config: &Value) -> Result<Self> {
        let parsed = ModelConfig::from_json(config)?;
        match parsed.variant() {
            Variant::V15Local => Err(SpeechError::Unsupported {
                why: "the v1.5 local transformer needs two weight views; use Model::open or \
                      Model::load_v15_local with a transformer-prefixed Weights"
                    .into(),
            }),
            Variant::LegacyLocal => Err(SpeechError::Unsupported {
                why: "the legacy local-transformer MOSS-TTS checkpoints are not ported".into(),
            }),
            Variant::Delay => Self::load_delay(weights, parsed),
        }
    }

    /// [`Model::open`] for the v1.5 local variant with explicit weight
    /// views: `transformer` must carry the `transformer.` prefix, the
    /// other weights stay prefix-free.
    pub fn load_v15_local(
        transformer: &Weights,
        weights: &Weights,
        config: impl Into<ModelConfig>,
    ) -> Result<Self> {
        Self::load_v15_local_impl(transformer, weights, config.into())
    }

    fn load_delay(weights: &Weights, config: ModelConfig) -> Result<Self> {
        let hidden = config.language_config.hidden_size();
        let mut emb_ext = Vec::with_capacity(config.n_vq);
        for channel in 0..config.n_vq {
            emb_ext.push(weights.tensor_absolute(
                &format!("emb_ext.{channel}.weight"),
                &[config.audio_vocab_size + 1, hidden],
            )?);
        }
        let mut lm_heads = Vec::with_capacity(config.n_vq + 1);
        lm_heads.push(weights.linear_absolute(
            "lm_heads.0",
            hidden,
            config.language_config.vocab_size(),
            crate::lm::Bias::None,
        )?);
        for channel in 0..config.n_vq {
            lm_heads.push(weights.linear_absolute(
                &format!("lm_heads.{}", channel + 1),
                hidden,
                config.audio_vocab_size + 1,
                crate::lm::Bias::None,
            )?);
        }
        let language_model = load_language_model(weights, &config.language_config.dense)?;
        Ok(Self {
            config,
            inner: Inner::Delay(Box::new(DelayStack {
                language_model,
                emb_ext,
                lm_heads,
            })),
        })
    }

    fn load_v15_local_impl(
        transformer: &Weights,
        weights: &Weights,
        config: ModelConfig,
    ) -> Result<Self> {
        let hidden = config.language_config.hidden_size();
        let local_config = config.local_gpt2_config()?;
        // The local stack consumes the global hidden states with no
        // projection, so its width must equal the backbone hidden size.
        // A mismatch otherwise surfaces as an opaque shape error mid-decode.
        if local_config.n_embd != hidden {
            return Err(SpeechError::BadConfig {
                field: "gpt2_config.n_embd".into(),
                why: format!(
                    "must equal the backbone hidden size {hidden} (got {})",
                    local_config.n_embd
                ),
            });
        }
        // The local transformer tensors are top level in the checkpoint,
        // so they load through the prefix-free view.
        let local_transformer = LocalGpt2::load(weights, "local_transformer", &local_config)?;
        let global = load_language_model(transformer, &config.language_config.dense)?;
        let mut audio_embeddings = Vec::with_capacity(config.n_vq);
        for (channel, size) in config.codebook_sizes().iter().enumerate() {
            audio_embeddings.push(weights.tensor_absolute(
                &format!("audio_embeddings.{channel}.weight"),
                &[*size, hidden],
            )?);
        }
        let text_lm_head = weights.linear_absolute(
            "text_lm_head",
            hidden,
            config.language_config.vocab_size(),
            crate::lm::Bias::None,
        )?;
        let mut audio_lm_heads = Vec::with_capacity(config.n_vq);
        for (channel, size) in config.codebook_sizes().iter().enumerate() {
            audio_lm_heads.push(weights.linear_absolute(
                &format!("audio_lm_heads.{channel}"),
                hidden,
                *size,
                crate::lm::Bias::None,
            )?);
        }
        let binary = config
            .local_text_head_mode
            .trim()
            .eq_ignore_ascii_case("binary");
        let local_text_lm_head = if binary {
            Some(weights.linear_absolute("local_text_lm_head", hidden, 2, crate::lm::Bias::None)?)
        } else {
            None
        };
        Ok(Self {
            config,
            inner: Inner::V15Local(Box::new(V15Stack {
                transformer: global,
                audio_embeddings,
                text_lm_head,
                audio_lm_heads,
                local_text_lm_head,
                local_transformer,
            })),
        })
    }

    pub fn variant(&self) -> Variant {
        self.config.variant()
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    /// Reference `_build_inputs_embeds` (delay): the text embedding of
    /// column 0 plus the per-book embeddings of columns 1..; the pad code
    /// embeds its own table row (the tables carry `audio_vocab_size + 1`
    /// rows).
    fn delay_inputs_embeds(&self, rows: &[Vec<i32>]) -> Result<Vec<f32>> {
        let stack = self.delay_stack()?;
        let hidden = self.config.language_config.hidden_size();
        let width = self.config.n_vq + 1;
        let vocab = self.config.language_config.vocab_size();
        let mut out = vec![0.0f32; rows.len() * hidden];
        for (index, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(SpeechError::Input {
                    why: format!("expected input_ids shape [batch, seq, {width}]"),
                });
            }
            let base = index * hidden;
            let text = lookup_row(stack.language_model.embed_table(), vocab, hidden, row[0])?;
            out[base..base + hidden].copy_from_slice(text);
            for channel in 0..self.config.n_vq {
                let id = row[channel + 1];
                let embed = lookup_row(
                    &stack.emb_ext[channel],
                    self.config.audio_vocab_size + 1,
                    hidden,
                    id,
                )?;
                for (col, value) in embed.iter().enumerate() {
                    out[base + col] += value;
                }
            }
        }
        Ok(out)
    }

    /// Reference `_build_v15_local_inputs_embeds`: the text embedding of
    /// column 0 plus the masked per-book embeddings (audio-pad columns
    /// contribute zero).
    fn v15_inputs_embeds(&self, rows: &[Vec<i32>]) -> Result<Vec<f32>> {
        let stack = self.v15_stack()?;
        let hidden = self.config.language_config.hidden_size();
        let width = self.config.n_vq + 1;
        let vocab = self.config.language_config.vocab_size();
        let codebook_sizes = self.config.codebook_sizes();
        let mut out = vec![0.0f32; rows.len() * hidden];
        for (index, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(SpeechError::Input {
                    why: format!("expected input_ids shape [batch, seq, {width}]"),
                });
            }
            let base = index * hidden;
            let text = lookup_row(stack.transformer.embed_table(), vocab, hidden, row[0])?;
            out[base..base + hidden].copy_from_slice(text);
            for channel in 0..self.config.n_vq {
                let id = row[channel + 1];
                if id == self.config.audio_pad_token_id {
                    continue;
                }
                let size = codebook_sizes[channel];
                if id < 0 || id as usize >= size {
                    return Err(SpeechError::Input {
                        why: format!("channel {channel} code {id} outside codebook size {size}"),
                    });
                }
                let embed = &stack.audio_embeddings[channel]
                    [id as usize * hidden..(id as usize + 1) * hidden];
                for (col, value) in embed.iter().enumerate() {
                    out[base + col] += value;
                }
            }
        }
        Ok(out)
    }

    fn v15_stack(&self) -> Result<&V15Stack> {
        match &self.inner {
            Inner::V15Local(stack) => Ok(stack),
            Inner::Delay(_) => Err(SpeechError::Input {
                why: "the v1.5 local path needs the v1.5 local variant".into(),
            }),
        }
    }

    fn delay_stack(&self) -> Result<&DelayStack> {
        match &self.inner {
            Inner::Delay(stack) => Ok(stack),
            Inner::V15Local(_) => Err(SpeechError::Input {
                why: "the delay path needs the delay variant".into(),
            }),
        }
    }

    /// Reference `generate_delay_pattern_ids` for batch 1: the backbone
    /// cache steps one row per frame while the delay bookkeeping picks
    /// the text token (forced delay slots, the keep/exclude sets inside
    /// and outside audio, the im_end warmup guard) and the per-book heads
    /// sample the codes with the sign-aware penalty over each book's full
    /// history. Greedy (`text_temperature = 0`, `audio_temperature = 0`)
    /// never draws. Returns the prompt tail row count and the tail rows
    /// from the last prompt `im_start + 3` position.
    pub fn generate_delay_pattern_ids(
        &mut self,
        prompt_rows: &[Vec<i32>],
        params: &DelayParams,
        seed: u64,
    ) -> Result<(usize, Vec<Vec<i32>>)> {
        let config = &self.config;
        let n_vq = config.n_vq;
        if prompt_rows.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let width = n_vq + 1;
        if prompt_rows.iter().any(|row| row.len() != width) {
            return Err(SpeechError::Input {
                why: format!("expected prompt rows of width {width}"),
            });
        }
        let text_do_sample = params.text_temperature > 0.0;
        let audio_do_sample = params.audio_temperature > 0.0;
        let text_temperature = if text_do_sample {
            params.text_temperature
        } else {
            1.0
        };
        let audio_temperature = if audio_do_sample {
            params.audio_temperature
        } else {
            1.0
        };
        let mut keys = rng::KeySequence::new(seed);

        let mut caches =
            crate::lm::make_prompt_cache(config.language_config.dense.num_hidden_layers);
        let seq_len = prompt_rows.len();
        let mut generation_rows: Vec<Vec<i32>> = prompt_rows.to_vec();

        let mut is_stopping = false;
        let mut audio_lengths: i64 = 0;
        let mut delayed_lengths: i64 = INT64_MAX;

        let last_text_token = prompt_rows
            .last()
            .and_then(|row| row.first())
            .copied()
            .ok_or_else(|| SpeechError::Input {
                why: "empty prompt rows".into(),
            })?;
        let is_continuation = last_text_token == config.audio_start_token_id
            || last_text_token == config.audio_assistant_gen_slot_token_id;
        let prompt_first_column: Vec<i32> = prompt_rows.iter().map(|row| row[0]).collect();
        let audio_start_idx = find_last_equal(&prompt_first_column, config.audio_start_token_id);
        let mut is_audio = is_continuation && audio_start_idx != -1;
        if is_audio {
            audio_lengths = seq_len as i64 - audio_start_idx;
        }

        let text_exclude_outside_audio = [
            config.pad_token_id,
            config.audio_assistant_gen_slot_token_id,
            config.audio_assistant_delay_slot_token_id,
            config.audio_end_token_id,
        ];
        let text_keep_inside_audio = [
            config.audio_assistant_gen_slot_token_id,
            config.audio_assistant_delay_slot_token_id,
        ];

        let hidden = config.language_config.hidden_size();
        for time_step in 0..params.max_new_tokens {
            // The reference feeds the whole prompt on step 0 and the new
            // row afterwards; the cache state is identical either way.
            let step_rows = step_input_rows(&generation_rows, seq_len);
            let embeds = self.delay_inputs_embeds(&step_rows)?;
            let hidden_states = {
                let Inner::Delay(stack) = &mut self.inner else {
                    return Err(SpeechError::Input {
                        why: "the delay path needs the delay variant".into(),
                    });
                };
                stack
                    .language_model
                    .forward_hidden(Input::Embeddings(&embeds), Some(&mut caches))?
            };
            let last_row = &hidden_states[hidden_states.len() - hidden..];

            // Temperature-scaled last-position logits per head.
            let stack = self.delay_stack()?;
            let text_head = &stack.lm_heads[0];
            let text_logits = scale(text_head.forward(last_row, 1), 1.0 / text_temperature);
            let audio_logits: Vec<Vec<f32>> = (1..=n_vq)
                .map(|head| {
                    scale(
                        stack.lm_heads[head].forward(last_row, 1),
                        1.0 / audio_temperature,
                    )
                })
                .collect();

            let mut next_text_token_value = config.pad_token_id;
            if !is_stopping && delayed_lengths < n_vq as i64 {
                next_text_token_value = config.audio_assistant_delay_slot_token_id;
            } else if !is_stopping && delayed_lengths == n_vq as i64 {
                next_text_token_value = config.audio_end_token_id;
                is_audio = false;
            } else if !is_stopping {
                let mut text_scores = text_logits;
                if is_audio {
                    keep_only_logits(&mut text_scores, &text_keep_inside_audio)?;
                } else {
                    set_logits_to_neg_inf(&mut text_scores, &text_exclude_outside_audio)?;
                }
                if time_step == 0 {
                    set_logits_to_neg_inf(
                        &mut text_scores,
                        &[config.audio_assistant_delay_slot_token_id],
                    )?;
                }
                if time_step <= n_vq {
                    set_logits_to_neg_inf(&mut text_scores, &[config.im_end_token_id])?;
                }
                next_text_token_value = family_sampling::sample_token(
                    &text_scores,
                    None,
                    1.0,
                    Some(params.text_top_p),
                    Some(params.text_top_k),
                    text_do_sample,
                    keys.next(),
                )? as i32;
            }

            if next_text_token_value == config.audio_start_token_id {
                is_audio = true;
            }
            if next_text_token_value == config.im_end_token_id {
                is_stopping = true;
            }

            let mut next_audio_tokens = vec![config.audio_pad_code; n_vq];
            for codebook_index in 0..n_vq {
                let pre_audio = audio_lengths > codebook_index as i64;
                let post_audio = if delayed_lengths == INT64_MAX {
                    true
                } else {
                    codebook_index as i64 > delayed_lengths - 1
                };
                if !(pre_audio && post_audio) {
                    continue;
                }
                let mut channel_scores = audio_logits[codebook_index].clone();
                set_logits_to_neg_inf(&mut channel_scores, &[config.audio_pad_code])?;
                let previous: Vec<i32> = generation_rows
                    .iter()
                    .map(|row| row[codebook_index + 1])
                    .collect();
                let channel_token = family_sampling::sample_token(
                    &channel_scores,
                    Some(&previous),
                    params.audio_repetition_penalty,
                    Some(params.audio_top_p),
                    Some(params.audio_top_k),
                    audio_do_sample,
                    keys.next(),
                )? as i32;
                next_audio_tokens[codebook_index] = channel_token;
            }

            if next_text_token_value == config.audio_start_token_id
                || next_text_token_value == config.audio_assistant_gen_slot_token_id
                || next_text_token_value == config.audio_assistant_delay_slot_token_id
            {
                audio_lengths += 1;
            }
            if next_text_token_value == config.audio_end_token_id {
                audio_lengths = 0;
            }
            if delayed_lengths == INT64_MAX
                && next_text_token_value == config.audio_assistant_delay_slot_token_id
            {
                delayed_lengths = 0;
            }
            if delayed_lengths != INT64_MAX {
                delayed_lengths += 1;
            }
            if delayed_lengths > n_vq as i64 {
                delayed_lengths = INT64_MAX;
            }

            let mut row = Vec::with_capacity(width);
            row.push(next_text_token_value);
            row.extend_from_slice(&next_audio_tokens);
            generation_rows.push(row);

            if is_stopping {
                break;
            }
        }

        let start = find_last_equal(&prompt_first_column, config.im_start_token_id);
        let start_idx = if start != -1 {
            (start + 3).clamp(0, generation_rows.len() as i64) as usize
        } else {
            seq_len
        };
        let start_length = if start != -1 {
            seq_len.saturating_sub(start_idx)
        } else {
            0
        };
        let tail = generation_rows.split_off(start_idx);
        Ok((start_length, tail))
    }

    /// Reference `_iter_v15_local_rows` + `generate_v15_local_ids` for
    /// batch 1: the global transformer proposes the continue/stop token
    /// through the GPT-2 local transformer and the candidate-gathered
    /// text head; while it is the assistant slot, the local transformer
    /// replays `[global hidden, previous book embedding, ...]` cacheless
    /// and each book head samples one code with the per-book generated
    /// history as the penalty context. Returns the prompt tail row count
    /// and the tail rows from the last prompt `audio_start` position.
    pub fn generate_v15_local_ids(
        &mut self,
        prompt_rows: &[Vec<i32>],
        params: &V15Params,
        seed: u64,
    ) -> Result<(usize, Vec<Vec<i32>>)> {
        let config = &self.config;
        let n_vq = config.n_vq;
        if prompt_rows.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        let width = n_vq + 1;
        if prompt_rows.iter().any(|row| row.len() != width) {
            return Err(SpeechError::Input {
                why: format!("expected prompt rows of width {width}"),
            });
        }
        let text_do_sample = params.do_sample && params.text_temperature > 0.0;
        let audio_do_sample = params.do_sample && params.audio_temperature > 0.0;
        let text_temperature = if text_do_sample {
            params.text_temperature
        } else {
            1.0
        };
        let audio_temperature = if audio_do_sample {
            params.audio_temperature
        } else {
            1.0
        };
        let mut keys = rng::KeySequence::new(seed);
        let hidden = config.language_config.hidden_size();
        let assistant_slot = config.audio_assistant_slot_token_id;
        let codebook_sizes = config.codebook_sizes();
        let binary_head_mode = config
            .local_text_head_mode
            .trim()
            .eq_ignore_ascii_case("binary");

        let mut caches = if params.use_kv_cache {
            Some(crate::lm::make_prompt_cache(
                config.language_config.dense.num_hidden_layers,
            ))
        } else {
            None
        };
        let mut generation_rows: Vec<Vec<i32>> = prompt_rows.to_vec();
        let mut current_model_rows: Vec<Vec<i32>> = prompt_rows.to_vec();
        let mut generated_frames: Vec<Vec<i32>> = Vec::new();

        for _ in 0..params.max_new_tokens {
            let embeds = self.v15_inputs_embeds(&current_model_rows)?;
            let hidden_states = {
                let Inner::V15Local(stack) = &mut self.inner else {
                    return Err(SpeechError::Input {
                        why: "the v1.5 local path needs the v1.5 local variant".into(),
                    });
                };
                match caches.as_mut() {
                    Some(caches) => stack
                        .transformer
                        .forward_hidden(Input::Embeddings(&embeds), Some(caches))?,
                    None => stack
                        .transformer
                        .forward_hidden(Input::Embeddings(&embeds), None)?,
                }
            };
            let global_last = hidden_states[hidden_states.len() - hidden..].to_vec();

            // Continue/stop choice over the two candidate head rows; the
            // local transformer replays [global hidden] first.
            let stack = self.v15_stack()?;
            let mut local_rows: Vec<f32> = global_last.clone();
            let mut local_hidden = stack.local_transformer.forward(&local_rows)?;
            let mut local_last = local_hidden[local_hidden.len() - hidden..].to_vec();
            let next_text_token = sample_v15_assistant_text_token(
                &stack.text_lm_head,
                stack.local_text_lm_head.as_ref(),
                binary_head_mode,
                &local_last,
                config.language_config.vocab_size(),
                config.audio_assistant_slot_token_id,
                config.audio_end_token_id,
                text_do_sample,
                text_temperature,
                params.text_top_k,
                params.text_top_p,
                keys.next(),
            )?;
            if next_text_token != assistant_slot {
                break;
            }

            let mut frame: Vec<i32> = Vec::with_capacity(n_vq);
            for channel in 0..n_vq {
                let head = &stack.audio_lm_heads[channel];
                let channel_logits = head.forward(&local_last, 1);
                let scaled = scale(channel_logits, 1.0 / audio_temperature);
                let previous: Vec<i32> = generated_frames.iter().map(|f| f[channel]).collect();
                let token = family_sampling::sample_token(
                    &scaled,
                    if generated_frames.is_empty() {
                        None
                    } else {
                        Some(&previous)
                    },
                    params.audio_repetition_penalty,
                    Some(params.audio_top_p),
                    Some(params.audio_top_k),
                    audio_do_sample,
                    keys.next(),
                )?;
                frame.push(token as i32);
                if channel + 1 < n_vq {
                    let size = codebook_sizes[channel];
                    if token >= size {
                        return Err(SpeechError::Input {
                            why: format!("channel {channel} code {token} outside codebook {size}"),
                        });
                    }
                    let embedding =
                        &stack.audio_embeddings[channel][token * hidden..(token + 1) * hidden];
                    local_rows.extend_from_slice(embedding);
                    local_hidden = stack.local_transformer.forward(&local_rows)?;
                    local_last = local_hidden[local_hidden.len() - hidden..].to_vec();
                }
            }

            generated_frames.push(frame.clone());
            let mut next_row = Vec::with_capacity(width);
            next_row.push(assistant_slot);
            next_row.extend_from_slice(&frame);
            generation_rows.push(next_row.clone());
            current_model_rows = if params.use_kv_cache {
                vec![next_row]
            } else {
                generation_rows.clone()
            };
        }

        let prompt_first_column: Vec<i32> = prompt_rows.iter().map(|row| row[0]).collect();
        let audio_start_idx = find_last_equal(&prompt_first_column, config.audio_start_token_id);
        let seq_len = prompt_rows.len();
        let start_idx = if audio_start_idx != -1 {
            audio_start_idx as usize
        } else {
            seq_len
        };
        let start_length = if audio_start_idx != -1 {
            seq_len.saturating_sub(start_idx + 1)
        } else {
            0
        };
        let tail = generation_rows.split_off(start_idx);
        Ok((start_length, tail))
    }

    /// Reference `_decode_generated_audio`: the tail rows' audio columns,
    /// de-delayed for the delay variant, all-pad rows dropped, contiguous
    /// segments decoded through the MOSS audio tokenizer, and the first
    /// segment trimmed proportionally to the prompt tail length.
    /// Returns the joined waveform and the decoded code row count.
    pub fn decode_generated_audio(
        &self,
        tokenizer: &MossAudioTokenizer,
        start_length: usize,
        rows: &[Vec<i32>],
    ) -> Result<(Vec<f32>, usize)> {
        let n_vq = self.config.n_vq;
        let mut audio_codes: Vec<Vec<i32>> = rows
            .iter()
            .map(|row| {
                if row.len() != n_vq + 1 {
                    Err(SpeechError::Input {
                        why: format!("expected rows of width {}", n_vq + 1),
                    })
                } else {
                    Ok(row[1..].to_vec())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if self.config.variant() == Variant::Delay {
            audio_codes = apply_de_delay_pattern(&audio_codes, n_vq)?;
        }
        let non_pad_indices: Vec<usize> = audio_codes
            .iter()
            .enumerate()
            .filter(|(_, row)| row.iter().any(|&code| code != self.config.audio_pad_code))
            .map(|(index, _)| index)
            .collect();
        if non_pad_indices.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let mut breaks: Vec<usize> = vec![0];
        for index in 1..non_pad_indices.len() {
            if non_pad_indices[index] != non_pad_indices[index - 1] + 1 {
                breaks.push(index);
            }
        }
        breaks.push(non_pad_indices.len());

        let mut audio_segments: Vec<Vec<f32>> = Vec::new();
        let mut token_count = 0usize;
        for window in breaks.windows(2) {
            let (break_start, break_end) = (window[0], window[1]);
            let segment_indices = &non_pad_indices[break_start..break_end];
            let first = segment_indices[0];
            let last = segment_indices[segment_indices.len() - 1];
            let codes = &audio_codes[first..last + 1];
            token_count += codes.len();
            let books: Vec<Vec<i32>> = (0..n_vq)
                .map(|book| codes.iter().map(|row| row[book]).collect())
                .collect();
            let mut audio = tokenizer.decode(&books)?;
            if start_length > 0 && audio_segments.is_empty() && !codes.is_empty() {
                let first_codes_length = codes.len();
                // f64 like the reference: f32 loses integer precision past
                // 2^24 samples, which long generations reach.
                let trim_ratio = (start_length as f64 / first_codes_length as f64).clamp(0.0, 1.0);
                let trim_samples = ((audio.len() as f64 * trim_ratio) as usize).min(audio.len());
                audio.drain(..trim_samples);
            }
            audio_segments.push(audio);
        }
        Ok((audio_segments.concat(), token_count))
    }

    /// Runs one prompt end to end for the loaded variant: the variant's
    /// generate loop then [`Model::decode_generated_audio`]. Sampling
    /// draws use the crate-local threefry RNG (`seed`); greedy runs are
    /// fully deterministic.
    pub fn generate(
        &mut self,
        tokenizer: &MossAudioTokenizer,
        prompt_rows: &[Vec<i32>],
        params: &GenerationParams,
        seed: u64,
    ) -> Result<Generated> {
        let (start_length, tail) = match (&self.inner, params) {
            (Inner::Delay(_), GenerationParams::Delay(params)) => {
                self.generate_delay_pattern_ids(prompt_rows, params, seed)?
            }
            (Inner::V15Local(_), GenerationParams::V15Local(params)) => {
                self.generate_v15_local_ids(prompt_rows, params, seed)?
            }
            (Inner::Delay(_), GenerationParams::V15Local(_)) => {
                return Err(SpeechError::Input {
                    why: "the delay variant takes GenerationParams::Delay".into(),
                })
            }
            (Inner::V15Local(_), GenerationParams::Delay(_)) => {
                return Err(SpeechError::Input {
                    why: "the v1.5 local variant takes GenerationParams::V15Local".into(),
                })
            }
        };
        let (samples, token_count) = self.decode_generated_audio(tokenizer, start_length, &tail)?;
        Ok(Generated {
            rows: tail,
            start_length,
            samples,
            token_count,
        })
    }
}

/// Reference `_sample_v15_assistant_text_token`: the two gathered
/// candidate logits (or the binary head), the temperature division, and
/// the capped top-k choice; returns the winning candidate token id.
#[allow(clippy::too_many_arguments)]
fn sample_v15_assistant_text_token(
    text_lm_head: &Linear,
    local_text_lm_head: Option<&Linear>,
    binary_head_mode: bool,
    local_hidden: &[f32],
    vocab: usize,
    assistant_slot: i32,
    audio_end: i32,
    do_sample: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    key: rng::Key,
) -> Result<i32> {
    let candidates = [assistant_slot, audio_end];
    let mut candidate_scores = Vec::with_capacity(2);
    if binary_head_mode {
        let binary = local_text_lm_head.ok_or_else(|| SpeechError::BadConfig {
            field: "local_text_head_mode".into(),
            why: "binary mode requires the local_text_lm_head weights".into(),
        })?;
        let logits = binary.forward(local_hidden, 1);
        if logits.len() != 2 {
            return Err(SpeechError::Input {
                why: "the binary local text head must produce 2 logits".into(),
            });
        }
        candidate_scores = logits;
    } else {
        let logits = text_lm_head.forward(local_hidden, 1);
        debug_assert_eq!(logits.len(), vocab);
        for &id in &candidates {
            let index = usize::try_from(id).map_err(|_| SpeechError::Input {
                why: format!("negative candidate token {id}"),
            })?;
            let score = logits
                .get(index)
                .copied()
                .ok_or_else(|| SpeechError::Input {
                    why: format!("candidate token {id} outside vocab {}", logits.len()),
                })?;
            candidate_scores.push(score);
        }
    }
    if do_sample {
        for score in &mut candidate_scores {
            *score /= temperature;
        }
    }
    let sampled = family_sampling::sample_token(
        &candidate_scores,
        None,
        1.0,
        Some(top_p),
        Some(top_k.min(2)),
        do_sample,
        key,
    )?;
    Ok(candidates[sampled])
}

/// The generation parameter set, variant-tagged.
#[derive(Debug, Clone)]
pub enum GenerationParams {
    Delay(DelayParams),
    V15Local(V15Params),
}

/// `generate_delay_pattern_ids` defaults; temperature 0 means greedy for
/// that stream (the reference derives `do_sample` from `temperature > 0`).
#[derive(Debug, Clone)]
pub struct DelayParams {
    pub max_new_tokens: usize,
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f32,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            text_temperature: 1.5,
            text_top_p: 1.0,
            text_top_k: 50,
            audio_temperature: 1.7,
            audio_top_p: 0.8,
            audio_top_k: 25,
            audio_repetition_penalty: 1.0,
        }
    }
}

/// `generate_v15_local_ids` defaults.
#[derive(Debug, Clone)]
pub struct V15Params {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f32,
    pub use_kv_cache: bool,
}

impl Default for V15Params {
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            do_sample: true,
            text_temperature: 1.0,
            text_top_p: 1.0,
            text_top_k: 50,
            audio_temperature: 1.7,
            audio_top_p: 0.8,
            audio_top_k: 25,
            audio_repetition_penalty: 1.0,
            use_kv_cache: true,
        }
    }
}

/// One generation result: the returned tail rows, their prompt row
/// count, the decoded waveform, and the decoded code row count.
#[derive(Debug, Clone)]
pub struct Generated {
    pub rows: Vec<Vec<i32>>,
    pub start_length: usize,
    pub samples: Vec<f32>,
    pub token_count: usize,
}

// ---------------------------------------------------------------------------
// v1.5 local prompt rows (MossTTSLocalV15Processor in id space)
// ---------------------------------------------------------------------------

/// `processor.LOCAL_V15_*` template strings; the caller encodes them with
/// the checkpoint tokenizer and hands the ids to [`V15TemplateIds`].
pub const LOCAL_V15_USER_ROLE_PREFIX: &str = "user\n";
pub const LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX: &str = "<user_inst>\n- Reference(s):\n";
pub const LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX: &str = "\n- Text:\n";
pub const LOCAL_V15_USER_TEMPLATE_SUFFIX: &str = "\n</user_inst>";
pub const LOCAL_V15_ASSISTANT_TURN_PREFIX: &str = "\n";
pub const LOCAL_V15_ASSISTANT_ROLE_PREFIX: &str = "assistant\n";

/// `_normalize_template_value`: missing or blank fields render as None.
pub fn normalize_template_value(value: Option<&str>) -> String {
    match value {
        None => "None".to_owned(),
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                "None".to_owned()
            } else {
                trimmed.to_owned()
            }
        }
    }
}

/// `_render_local_v15_user_prompt_after_reference`: the field block
/// between the reference section and the text.
pub fn render_local_v15_user_prompt_after_reference(
    instruction: Option<&str>,
    tokens: Option<&str>,
    quality: Option<&str>,
    sound_event: Option<&str>,
    ambient_sound: Option<&str>,
    language: Option<&str>,
) -> String {
    format!(
        "\n- Instruction:\n{}\n- Tokens:\n{}\n- Quality:\n{}\n- Sound Event:\n{}\n\
         - Ambient Sound:\n{}\n- Language:\n{}\n{}",
        normalize_template_value(instruction),
        normalize_template_value(tokens),
        normalize_template_value(quality),
        normalize_template_value(sound_event),
        normalize_template_value(ambient_sound),
        normalize_template_value(language),
        LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX
    )
}

/// The caller-encoded v1.5 template pieces (the strings above plus the
/// encoded rendered after-reference block and the `None` stand-in).
#[derive(Debug, Clone, Copy)]
pub struct V15TemplateIds<'a> {
    /// `encode(LOCAL_V15_USER_ROLE_PREFIX)`.
    pub user_role_prefix: &'a [i32],
    /// `encode(LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX)`.
    pub user_template_reference_prefix: &'a [i32],
    /// The encoded [`render_local_v15_user_prompt_after_reference`]
    /// output (or `LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX` with
    /// all-None fields, identical to the nano family template).
    pub user_template_after_reference: &'a [i32],
    /// `encode(LOCAL_V15_USER_TEMPLATE_SUFFIX)`.
    pub user_template_suffix: &'a [i32],
    /// `encode(LOCAL_V15_ASSISTANT_TURN_PREFIX)`.
    pub assistant_turn_prefix: &'a [i32],
    /// `encode(LOCAL_V15_ASSISTANT_ROLE_PREFIX)`.
    pub assistant_role_prefix: &'a [i32],
    /// `encode("None")`, the no-reference stand-in.
    pub none_reference: &'a [i32],
}

/// v1.5 prompt mode (`MossTTSLocalV15Processor.__call__` mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V15PromptMode {
    Generation,
    Continuation,
}

impl Model {
    /// `MossTTSLocalV15Processor` for batch 1: generation / voice-clone
    /// composes the user prompt rows (one audio block per reference,
    /// bracketed by audio-start / audio-end text rows) or the all-text
    /// `None`-reference prompt; continuation appends the assistant audio
    /// rows after the user prompt rows. The v1.5 continuation
    /// concatenates ref text + target text as ONE string on the caller
    /// side before encoding (the reference encodes the joined string).
    pub fn build_v15_inference_input_ids(
        &self,
        mode: V15PromptMode,
        template: &V15TemplateIds,
        text_ids: &[i32],
        prompt_codes: Option<&[Vec<Vec<i32>>]>,
    ) -> Result<Vec<Vec<i32>>> {
        let n_vq = self.config.n_vq;
        let pad = self.config.audio_pad_token_id;
        let mut prefix_ids: Vec<i32> = Vec::new();
        prefix_ids.push(self.config.im_start_token_id);
        prefix_ids.extend_from_slice(template.user_role_prefix);
        prefix_ids.extend_from_slice(template.user_template_reference_prefix);
        let mut tail_ids: Vec<i32> = Vec::new();
        tail_ids.extend_from_slice(template.user_template_after_reference);
        tail_ids.extend_from_slice(text_ids);
        tail_ids.extend_from_slice(template.user_template_suffix);
        tail_ids.push(self.config.im_end_token_id);
        tail_ids.extend_from_slice(template.assistant_turn_prefix);
        tail_ids.push(self.config.im_start_token_id);
        tail_ids.extend_from_slice(template.assistant_role_prefix);
        tail_ids.push(self.config.audio_start_token_id);

        let mut rows: Vec<Vec<i32>> = Vec::new();
        match mode {
            V15PromptMode::Generation => match prompt_codes {
                Some(codes_list) if !codes_list.is_empty() => {
                    rows.extend(v15_text_rows(&prefix_ids, n_vq, pad));
                    for codes in codes_list {
                        let normalized = normalize_audio_codes(codes, n_vq)?;
                        rows.extend(v15_text_rows(
                            &[self.config.audio_start_token_id],
                            n_vq,
                            pad,
                        ));
                        rows.extend(v15_audio_rows(
                            &normalized,
                            self.config.audio_user_slot_token_id,
                            n_vq,
                            pad,
                        ));
                        rows.extend(v15_text_rows(&[self.config.audio_end_token_id], n_vq, pad));
                    }
                    rows.extend(v15_text_rows(&tail_ids, n_vq, pad));
                }
                _ => {
                    let mut prompt_ids = prefix_ids;
                    prompt_ids.extend_from_slice(template.none_reference);
                    prompt_ids.extend_from_slice(&tail_ids);
                    rows.extend(v15_text_rows(&prompt_ids, n_vq, pad));
                }
            },
            V15PromptMode::Continuation => {
                let codes_list = prompt_codes.ok_or_else(|| SpeechError::Input {
                    why: "continuation mode requires prompt_codes".into(),
                })?;
                if codes_list.len() != 1 {
                    return Err(SpeechError::Input {
                        why: "the v1.5 continuation expects one prompt audio item".into(),
                    });
                }
                let normalized = normalize_audio_codes(&codes_list[0], n_vq)?;
                let mut prompt_ids = prefix_ids;
                prompt_ids.extend_from_slice(template.none_reference);
                prompt_ids.extend_from_slice(&tail_ids);
                rows.extend(v15_text_rows(&prompt_ids, n_vq, pad));
                rows.extend(v15_audio_rows(
                    &normalized,
                    self.config.audio_assistant_slot_token_id,
                    n_vq,
                    pad,
                ));
            }
        }
        Ok(rows)
    }
}

/// `_build_text_rows`: all-audio-pad rows with the text token in column 0.
fn v15_text_rows(token_ids: &[i32], n_vq: usize, audio_pad_token_id: i32) -> Vec<Vec<i32>> {
    token_ids
        .iter()
        .map(|&id| {
            let mut row = vec![audio_pad_token_id; n_vq + 1];
            row[0] = id;
            row
        })
        .collect()
}

/// `_build_audio_rows`: all-audio-pad rows with the slot token in column
/// 0 and up to `n_vq` codes in the audio columns.
fn v15_audio_rows(
    codes: &[Vec<i32>],
    slot_token_id: i32,
    n_vq: usize,
    audio_pad_token_id: i32,
) -> Vec<Vec<i32>> {
    codes
        .iter()
        .map(|frame| {
            let mut row = vec![audio_pad_token_id; n_vq + 1];
            row[0] = slot_token_id;
            for (channel, &code) in frame.iter().take(n_vq).enumerate() {
                row[channel + 1] = code;
            }
            row
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The rows the backbone consumes this step: the whole prompt on the
/// first call, the newest row afterwards (the reference feeds
/// `current_input_ids`, which holds the prompt once and the new row
/// after each step; the cache state is identical).
fn step_input_rows(generation_rows: &[Vec<i32>], seq_len: usize) -> Vec<Vec<i32>> {
    if generation_rows.len() > seq_len {
        vec![generation_rows[generation_rows.len() - 1].clone()]
    } else {
        generation_rows.to_vec()
    }
}

fn find_last_equal(values: &[i32], target: i32) -> i64 {
    values
        .iter()
        .rposition(|&value| value == target)
        .map(|index| index as i64)
        .unwrap_or(-1)
}

fn scale(mut logits: Vec<f32>, factor: f32) -> Vec<f32> {
    for value in &mut logits {
        *value *= factor;
    }
    logits
}

/// Index of `id` in a logits row of `len` entries. Out-of-range ids are an
/// error, not a silent no-op: a config id past the vocab would otherwise
/// drop a constraint and let the sampler pick a token the model never
/// allowed.
fn logit_index(id: i32, len: usize) -> Result<usize> {
    usize::try_from(id)
        .ok()
        .filter(|&index| index < len)
        .ok_or_else(|| SpeechError::Input {
            why: format!("token id {id} is outside the logits row of {len} entries"),
        })
}

/// `_set_logits_to_neg_inf`.
fn set_logits_to_neg_inf(logits: &mut [f32], token_ids: &[i32]) -> Result<()> {
    for &id in token_ids {
        logits[logit_index(id, logits.len())?] = f32::NEG_INFINITY;
    }
    Ok(())
}

/// `_keep_only_logits`: everything outside `token_ids` goes to negative
/// infinity.
fn keep_only_logits(logits: &mut [f32], token_ids: &[i32]) -> Result<()> {
    let mut keep = vec![false; logits.len()];
    for &id in token_ids {
        keep[logit_index(id, logits.len())?] = true;
    }
    for (index, value) in logits.iter_mut().enumerate() {
        if !keep[index] {
            *value = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Embedding row lookup refusing negative and out-of-range ids.
fn lookup_row(table: &[f32], vocab: usize, width: usize, token: i32) -> Result<&[f32]> {
    let row = usize::try_from(token).map_err(|_| SpeechError::Input {
        why: format!("negative token id {token}"),
    })?;
    if row >= vocab {
        return Err(SpeechError::Input {
            why: format!("token id {token} outside vocab {vocab}"),
        });
    }
    Ok(&table[row * width..(row + 1) * width])
}
