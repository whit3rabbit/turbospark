//! ACE-Step 1.5 DiT text inputs: the caption prompt, the lyric prompt,
//! and their tokenization.
//!
//! Reference: upstream at the pinned commit, `acestep/constants.py`
//! (`SFT_GEN_PROMPT`, `TASK_INSTRUCTIONS`), and the handler helpers
//! `MetadataMixin._build_metadata_dict` / `_dict_to_meta_string` and
//! `PromptMixin._format_instruction` / `_format_lyrics` /
//! `extract_caption_from_sft_format`. Tokenization is the checkpoint's
//! Qwen3-Embedding `tokenizer.json`, called as upstream does
//! (`truncation=True`, 256 caption and 2048 lyric tokens, no padding for
//! one request).

use std::path::Path;

use tokenizers::Tokenizer;

use super::text::{MAX_CAPTION_TOKENS, MAX_LYRIC_TOKENS};
use crate::{Result, SpeechError};

/// The DiT task, which picks the instruction line of the caption prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Text2Music,
    Cover,
    Repaint,
}

impl Task {
    /// `TASK_INSTRUCTIONS[task]`.
    pub fn instruction(self) -> &'static str {
        match self {
            Task::Text2Music => "Fill the audio semantic mask based on the given conditions:",
            Task::Repaint => "Repaint the mask area based on the given conditions:",
            Task::Cover => "Generate audio semantic tokens based on the given conditions:",
        }
    }
}

/// `DEFAULT_DIT_INSTRUCTION`: also the non-cover branch of a cover run.
pub const DEFAULT_DIT_INSTRUCTION: &str =
    "Fill the audio semantic mask based on the given conditions:";

/// Optional musical metadata. Empty or zero fields render as `N/A`, as
/// upstream's falsy checks do.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metas {
    pub bpm: Option<u32>,
    pub key_scale: String,
    pub time_signature: String,
    pub duration_seconds: Option<f64>,
}

impl Metas {
    /// `_dict_to_meta_string(_build_metadata_dict(...))`.
    pub fn render(&self) -> String {
        let bpm = match self.bpm {
            Some(b) if b != 0 => b.to_string(),
            _ => "N/A".to_string(),
        };
        // Upstream tests `.strip()` for emptiness but keeps the original.
        let key = if self.key_scale.trim().is_empty() {
            "N/A"
        } else {
            &self.key_scale
        };
        let ts = if !self.time_signature.trim().is_empty() && self.time_signature != "N/A" {
            &self.time_signature
        } else {
            "N/A"
        };
        // int(duration) truncates toward zero; absent means 30.
        let duration = match self.duration_seconds {
            Some(d) => format!("{} seconds", d.trunc() as i64),
            None => "30 seconds".to_string(),
        };
        format!("- bpm: {bpm}\n- timesignature: {ts}\n- keyscale: {key}\n- duration: {duration}\n")
    }
}

/// `_format_instruction`: a trailing colon is appended when missing.
pub fn format_instruction(instruction: &str) -> String {
    if instruction.ends_with(':') {
        instruction.to_string()
    } else {
        format!("{instruction}:")
    }
}

/// `SFT_GEN_PROMPT.format(instruction, caption, metas)`.
pub fn caption_prompt(instruction: &str, caption: &str, metas: &Metas) -> String {
    format!(
        "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n",
        format_instruction(instruction),
        extract_caption(caption),
        metas.render()
    )
}

/// `_format_lyrics(lyrics, language)`.
pub fn lyric_prompt(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
}

/// `extract_caption_from_sft_format`: when the caption is itself an SFT
/// prompt, take the text under `# Caption` up to `# Metas`. A hand port
/// of `#\s*Caption\s*\n(.*?)(?:\n\s*#\s*Metas|$)` (DOTALL), then strip.
pub fn extract_caption(caption: &str) -> String {
    if !(caption.contains("# Instruction") && caption.contains("# Caption")) {
        return caption.to_string();
    }
    let skip_ws = |s: &str, from: usize| -> usize {
        from + s[from..]
            .char_indices()
            .find(|(_, c)| !c.is_whitespace())
            .map_or(s.len() - from, |(i, _)| i)
    };
    for (hash, _) in caption.match_indices('#') {
        let at = skip_ws(caption, hash + 1);
        if !caption[at..].starts_with("Caption") {
            continue;
        }
        let after = at + "Caption".len();
        // `\s*\n`: the greedy run backtracks to its last newline.
        let run_end = skip_ws(caption, after);
        let Some(nl) = caption[after..run_end].rfind('\n') else {
            continue;
        };
        let start = after + nl + 1;
        // `(.*?)(?:\n\s*#\s*Metas|$)`: the earliest terminator wins.
        let mut end = caption.len();
        for (p, _) in caption[start..].match_indices('\n') {
            let q = skip_ws(caption, start + p + 1);
            if caption[q..].starts_with('#') {
                let r = skip_ws(caption, q + 1);
                if caption[r..].starts_with("Metas") {
                    end = start + p;
                    break;
                }
            }
        }
        return caption[start..end].trim().to_string();
    }
    caption.to_string()
}

/// Token ids for one prompt, and whether upstream's truncation cut it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    pub ids: Vec<i32>,
    pub truncated: bool,
}

/// The checkpoint's text tokenizer (`Qwen3-Embedding-0.6B/tokenizer.json`).
///
/// Truncation is the tokenizer's own (right side, applied before its
/// post-processor appends the end token), so an over-long prompt keeps
/// its final `<|endoftext|>` exactly as upstream's `truncation=True` does.
/// One instance per cap keeps `encode` immutable.
pub struct PromptTokenizer {
    caption: Tokenizer,
    lyrics: Tokenizer,
}

fn with_cap(path: &Path, max: usize) -> Result<Tokenizer> {
    let mut t = Tokenizer::from_file(path).map_err(|e| SpeechError::BadConfig {
        field: "tokenizer.json".into(),
        why: format!("{}: {e}", path.display()),
    })?;
    t.with_truncation(Some(tokenizers::TruncationParams {
        max_length: max,
        ..Default::default()
    }))
    .map_err(|e| SpeechError::BadConfig {
        field: "tokenizer.json".into(),
        why: format!("truncation: {e}"),
    })?;
    // One request is never padded (upstream pads to the longest, itself).
    t.with_padding(None);
    Ok(t)
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Encoded> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|e| SpeechError::Input {
            why: format!("tokenizer failed: {e}"),
        })?;
    let ids = encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: format!("token id {id} exceeds the signed 32-bit boundary"),
            })
        })
        .collect::<Result<Vec<i32>>>()?;
    Ok(Encoded {
        ids,
        truncated: !encoded.get_overflowing().is_empty(),
    })
}

impl PromptTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        Ok(Self {
            caption: with_cap(&path, MAX_CAPTION_TOKENS)?,
            lyrics: with_cap(&path, MAX_LYRIC_TOKENS)?,
        })
    }

    /// Caption prompt ids, truncated at 256 as upstream does.
    pub fn caption(&self, prompt: &str) -> Result<Encoded> {
        encode(&self.caption, prompt)
    }

    /// Lyric prompt ids, truncated at 2048 as upstream does.
    pub fn lyrics(&self, prompt: &str) -> Result<Encoded> {
        encode(&self.lyrics, prompt)
    }
}
