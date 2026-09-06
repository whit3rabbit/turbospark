//! The real encoder embedding backend: a live `EncoderRunner` behind [`ChatModel`].

use std::path::Path;

use runtime::encoder::EncoderRunner;
use runtime::{GenerationConfig, LogitProducer, RawDecodeProgress, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;

use crate::model::ChatModel;

/// A live encoder model for embedding generation.
pub struct RealEncoderModel {
    tokenizer: MfTokenizer,
    runner: EncoderRunner,
    model_id: String,
}

impl RealEncoderModel {
    /// Opens an encoder model from a directory containing `config.json`,
    /// `model.safetensors`, and `tokenizer.json`.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let runner = EncoderRunner::open(model_dir)?;
        let model_id = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("encoder")
            .to_string();

        let tokenizer = MfTokenizer::load_from_dir(model_dir)
            .map_err(|e| format!("failed to load tokenizer from {}: {e}", model_dir.display()))?;

        Ok(Self {
            tokenizer,
            runner,
            model_id,
        })
    }

    pub fn with_model_id(mut self, id: String) -> Self {
        self.model_id = id;
        self
    }
}

impl ChatModel for RealEncoderModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }

    fn vocab_size(&self) -> usize {
        self.runner.config.vocab_size
    }

    fn max_context(&self) -> u32 {
        self.runner.config.max_position_embeddings as u32
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.runner.encode_batch_text(texts)
    }

    fn with_producer(
        &self,
        _f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        Err(RuntimeError::Producer(format!(
            "model '{}' is an embedding model and does not support generative completions",
            self.model_id
        )))
    }

    fn run_completion(
        &self,
        _prompt_ids: &[foundation::TokenId],
        _config: &GenerationConfig,
        _images: Option<&crate::vision::RequestImages>,
        _cancel: runtime::CancelFlag<'_>,
        _on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        Err(RuntimeError::Producer(format!(
            "model '{}' is an embedding model and does not support generative completions",
            self.model_id
        )))
    }
}
