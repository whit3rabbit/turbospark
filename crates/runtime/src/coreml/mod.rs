//! Core ML model pipelines. Each submodule turns one Core ML export (the
//! FluidInference ones, plus the eg2demo word-CTC experiment) into a
//! TurboSpark capability; `turbospark-coreml` does the Objective-C work.
//! See `docs/COREML.md`.

use std::path::Path;

pub mod embedding_gemma2;
pub mod embedding_gemma2_media;
pub mod embeddinggemma2_ctc;
pub mod parakeet;
pub mod unlimited_ocr;

pub use embedding_gemma2::{EmbeddingGemma2CoreMl, EmbeddingGemma2Options};
pub use embedding_gemma2_media::AudioWindowEmbedding;
pub use parakeet::{ParakeetCoreMl, ParakeetCoreMlOptions};
pub use unlimited_ocr::{OcrEvent, OcrOptions, OcrPage, OcrResult, OcrStats, UnlimitedOcrCoreMl};

/// Which Core ML pipeline a model directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlLayout {
    /// FluidInference Parakeet TDT export: Preprocessor, Encoder, Decoder,
    /// JointDecisionv3 plus `parakeet_vocab.json`.
    ParakeetTdt,
    /// FluidInference EmbeddingGemma 2 export: the multifunction text
    /// package, the bf16 token table, `tokenizer.json` and `config.json`
    /// (audio and vision packages optional).
    EmbeddingGemma2,
}

impl CoreMlLayout {
    pub fn family(self) -> &'static str {
        match self {
            Self::ParakeetTdt => "parakeet_tdt",
            Self::EmbeddingGemma2 => "embeddinggemma2",
        }
    }
}

/// Recognizes a Core ML model directory by its files. These exports ship no
/// `model_type`, so dispatch that reads `config.json` first must call this
/// before it. A directory that is not a known layout returns `None` and the
/// caller falls through to its usual checks.
pub fn detect_layout(dir: &Path) -> Option<CoreMlLayout> {
    if parakeet::REQUIRED_DIRS.iter().all(|d| dir.join(d).is_dir())
        && dir.join(parakeet::VOCAB_FILE).is_file()
    {
        return Some(CoreMlLayout::ParakeetTdt);
    }
    if dir.join(embedding_gemma2::TEXT_PACKAGE).is_dir()
        && [
            embedding_gemma2::TOKEN_TABLE,
            embedding_gemma2::TOKENIZER,
            embedding_gemma2::CONFIG,
        ]
        .iter()
        .all(|f| dir.join(f).is_file())
    {
        return Some(CoreMlLayout::EmbeddingGemma2);
    }
    None
}

/// Where compiled packages are cached: inside the install, so it moves and
/// is removed with it and no new store path is introduced.
pub(crate) fn cache_root(dir: &Path) -> std::path::PathBuf {
    dir.join(".compiled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_parakeet_only_with_every_part() {
        let d = std::env::temp_dir().join(format!(
            "ts-coreml-layout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for sub in parakeet::REQUIRED_DIRS {
            std::fs::create_dir_all(d.join(sub)).unwrap();
        }
        assert_eq!(detect_layout(&d), None, "vocab missing");
        std::fs::write(d.join(parakeet::VOCAB_FILE), b"{}").unwrap();
        assert_eq!(detect_layout(&d), Some(CoreMlLayout::ParakeetTdt));
        std::fs::remove_dir_all(d.join("Decoder.mlmodelc")).unwrap();
        assert_eq!(detect_layout(&d), None, "decoder missing");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn detects_embeddinggemma2_only_with_every_part() {
        let d = std::env::temp_dir().join(format!(
            "ts-coreml-layout-eg2-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(d.join(embedding_gemma2::TEXT_PACKAGE)).unwrap();
        for f in [embedding_gemma2::TOKEN_TABLE, embedding_gemma2::TOKENIZER] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        assert_eq!(detect_layout(&d), None, "config missing");
        std::fs::write(d.join(embedding_gemma2::CONFIG), b"{}").unwrap();
        assert_eq!(detect_layout(&d), Some(CoreMlLayout::EmbeddingGemma2));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
