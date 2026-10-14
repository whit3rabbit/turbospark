//! Thin wrapper over the Hugging Face `tokenizers` crate.
//!
//! Families own truncation, padding, and prompt policy; this only loads a
//! `tokenizer.json` and encodes. Whatever truncation or padding the file
//! itself configures is cleared at load so a family's explicit policy is
//! the only one that applies.

use std::path::Path;

use crate::error::EmbeddingError;

fn tok_err(why: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::Tokenizer {
        why: why.to_string(),
    }
}

/// A loaded `tokenizer.json`.
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for TextTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextTokenizer").finish_non_exhaustive()
    }
}

impl TextTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, EmbeddingError> {
        let mut inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| tok_err(format!("{}: {e}", path.display())))?;
        inner.with_padding(None);
        inner.with_truncation(None).map_err(tok_err)?;
        Ok(Self { inner })
    }

    /// Token ids for `text`, with the post-processor's special tokens when
    /// `add_special_tokens`.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, EmbeddingError> {
        Ok(self
            .inner
            .encode(text, add_special_tokens)
            .map_err(tok_err)?
            .get_ids()
            .to_vec())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
