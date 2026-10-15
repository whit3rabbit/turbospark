//! The Drama 3 root `config.json`: token ids, prompt template, tower shape.
//!
//! Mirrors mlx-audio `fish_drama3/config.py` plus the checks its
//! `post_load_hook` makes against the tokenizer, at
//! [`super::REFERENCE_COMMIT`]. The text backbone's own fields are the
//! trunk intake's (`crates/repack`'s `fish_drama3`), not read here beyond the
//! hidden size the tower must agree with.

use tokenizers::Tokenizer;

use super::tower::TowerConfig;
use crate::{Result, SpeechError};

/// The root `model_type` of a Drama 3 checkpoint.
pub const MODEL_TYPE: &str = "fish_drama3";

/// The one prompt template this port builds.
pub const SPEECH_TEMPLATE: &str = "reference-bracket-v1";

/// The fields the TTS side reads from the root `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Drama3Config {
    pub tower: TowerConfig,
    pub semantic_start_token_id: i32,
    pub semantic_end_token_id: i32,
    pub eos_token_id: i32,
    /// `text_config.hidden_size`: the width of the trunk's hidden state.
    pub hidden_size: usize,
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.into(),
        why: why.into(),
    }
}

impl Drama3Config {
    /// Parses and checks a root `config.json`.
    pub fn from_json(root: &serde_json::Value) -> Result<Self> {
        let model_type = root.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some(MODEL_TYPE) {
            return Err(bad(
                "model_type",
                format!("{model_type:?} is not {MODEL_TYPE:?}"),
            ));
        }
        let template = root
            .get("speech_template_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(SPEECH_TEMPLATE);
        if template != SPEECH_TEMPLATE {
            return Err(bad(
                "speech_template_version",
                format!("{template:?}; only {SPEECH_TEMPLATE:?} is built"),
            ));
        }
        let id = |key: &str| -> Result<i32> {
            root.get(key)
                .and_then(serde_json::Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| bad(key, "missing or not an i32"))
        };
        let tower = TowerConfig::from_json(
            root.get("audio_decoder_config")
                .ok_or_else(|| bad("audio_decoder_config", "missing"))?,
        )?;
        let hidden_size =
            root.get("text_config")
                .and_then(|t| t.get("hidden_size"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| bad("text_config.hidden_size", "missing"))? as usize;
        let cfg = Drama3Config {
            semantic_start_token_id: id("semantic_start_token_id")?,
            semantic_end_token_id: id("semantic_end_token_id")?,
            eos_token_id: id("eos_token_id")?,
            hidden_size,
            tower,
        };
        let band = cfg.semantic_end_token_id - cfg.semantic_start_token_id + 1;
        if band != cfg.tower.vocab_size as i32 {
            return Err(bad(
                "semantic_end_token_id",
                format!(
                    "the semantic band holds {band} tokens; the codebooks hold {}",
                    cfg.tower.vocab_size
                ),
            ));
        }
        if cfg.semantic_band().contains(&cfg.eos_token_id) {
            return Err(bad("eos_token_id", "EOS lies inside the semantic band"));
        }
        if cfg.hidden_size != cfg.tower.dim {
            return Err(bad(
                "audio_decoder_config.dim",
                format!(
                    "the tower reads the trunk's {}-wide hidden state with a {}-wide input",
                    cfg.hidden_size, cfg.tower.dim
                ),
            ));
        }
        Ok(cfg)
    }

    /// `semantic_start_token_id..=semantic_end_token_id`.
    pub fn semantic_band(&self) -> std::ops::RangeInclusive<i32> {
        self.semantic_start_token_id..=self.semantic_end_token_id
    }

    /// The tokens a step may emit, in the semantic head's row order: the
    /// semantic band, then EOS.
    pub fn candidate_ids(&self) -> Vec<i32> {
        self.semantic_band()
            .chain(std::iter::once(self.eos_token_id))
            .collect()
    }

    /// Refuses a tokenizer that disagrees with the config about the band's
    /// ends or EOS (the reference's `post_load_hook` check).
    pub fn check_tokenizer(&self, tokenizer: &Tokenizer) -> Result<()> {
        let last = self.tower.vocab_size - 1;
        for (token, want) in [
            ("<|semantic:0|>".to_string(), self.semantic_start_token_id),
            (format!("<|semantic:{last}|>"), self.semantic_end_token_id),
            ("<|im_end|>".to_string(), self.eos_token_id),
        ] {
            let got = tokenizer.token_to_id(&token).map(|v| v as i64);
            if got != Some(i64::from(want)) {
                return Err(bad(
                    "tokenizer.json",
                    format!("{token} is {got:?} in the tokenizer, {want} in config.json"),
                ));
            }
        }
        Ok(())
    }
}
