//! ACE-Step 1.5 text conditioning: Qwen3-Embedding-0.6B.
//!
//! Upstream (`conditioning_embed.py` at the pinned commit) feeds the
//! caption prompt through the full encoder and uses its
//! `last_hidden_state` (post final norm, causal attention), while lyrics
//! use only the token embedding table (`text_encoder.embed_tokens`).
//! Prompt formatting and tokenization are pipeline concerns and live
//! above this module; this takes token ids.

use std::path::Path;

use crate::lm::engine::{Backbone, Input};
use crate::lm::{qwen3, Weights};
use crate::{Result, SpeechError};

/// Caption and lyric token caps upstream applies (`max_length` with
/// truncation in `conditioning_text.py`).
pub const MAX_CAPTION_TOKENS: usize = 256;
pub const MAX_LYRIC_TOKENS: usize = 2048;

/// The Qwen3-Embedding text encoder.
pub struct TextEncoder {
    backbone: Backbone,
}

impl TextEncoder {
    /// Loads `config.json` and `model.safetensors` from the checkpoint's
    /// `Qwen3-Embedding-0.6B/` directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let weights = Weights::open(dir)?;
        let model_type = weights.config().get("model_type").and_then(|v| v.as_str());
        if model_type != Some("qwen3") {
            return Err(SpeechError::BadConfig {
                field: "model_type".into(),
                why: format!("expected qwen3, got {model_type:?}"),
            });
        }
        let backbone = qwen3::load(&weights, weights.config())?;
        Ok(Self { backbone })
    }

    pub fn hidden_size(&self) -> usize {
        self.backbone.config.hidden_size
    }

    /// Caption prompt tokens -> `[tokens, hidden]` last hidden state.
    pub fn encode_caption(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        check_len("caption", tokens, MAX_CAPTION_TOKENS)?;
        self.backbone.forward_hidden(Input::Tokens(tokens), None)
    }

    /// Lyric tokens -> `[tokens, hidden]` rows of the embedding table.
    pub fn embed_lyrics(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        check_len("lyrics", tokens, MAX_LYRIC_TOKENS)?;
        let c = &self.backbone.config;
        let width = c.hidden_size;
        let table = self.backbone.embed_table();
        tokens
            .iter()
            .map(|&t| {
                let row = usize::try_from(t)
                    .ok()
                    .filter(|&r| r < c.vocab_size && (r + 1) * width <= table.len())
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("lyric token id {t} outside vocab {}", c.vocab_size),
                    })?;
                Ok(&table[row * width..(row + 1) * width])
            })
            .collect::<Result<Vec<&[f32]>>>()
            .map(|rows| rows.concat())
    }
}

/// Upstream truncates silently; the port refuses instead so the caller
/// decides what to drop.
fn check_len(what: &str, tokens: &[i32], max: usize) -> Result<()> {
    if tokens.is_empty() || tokens.len() > max {
        return Err(SpeechError::Input {
            why: format!("{what}: {} tokens, expected 1..={max}", tokens.len()),
        });
    }
    Ok(())
}
