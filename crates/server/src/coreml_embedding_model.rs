//! EmbeddingGemma 2 on Core ML behind [`ChatModel`], for `/v1/embeddings`
//! and Ollama `/api/embed`. Sibling of `encoder_model.rs`.
//!
//! Requests carry no task, so texts embed without a prompt prefix. The
//! handler's generic `dimensions` truncation plus re-normalization is the
//! model's Matryoshka rule, so full 768-d vectors are returned here.

use std::path::Path;
use std::sync::Arc;

use runtime::coreml::{EmbeddingGemma2CoreMl, EmbeddingGemma2Options};
use runtime::{GenerationConfig, LogitProducer, RawDecodeProgress, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;

use crate::model::ChatModel;

pub struct CoreMlEmbeddingModel {
    tokenizer: MfTokenizer,
    model: EmbeddingGemma2CoreMl,
    model_id: String,
}

impl CoreMlEmbeddingModel {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let model = EmbeddingGemma2CoreMl::open(model_dir, &EmbeddingGemma2Options::from_env()?)?;
        // Usage accounting counts tokens with the server's tokenizer type.
        let tokenizer = MfTokenizer::load_from_dir(model_dir)
            .map_err(|e| format!("failed to load tokenizer from {}: {e}", model_dir.display()))?;
        let model_id = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("embeddinggemma-2-coreml")
            .to_string();
        Ok(Self {
            tokenizer,
            model,
            model_id,
        })
    }

    pub fn with_model_id(mut self, id: String) -> Self {
        self.model_id = id;
        self
    }

    fn no_generation(&self) -> RuntimeError {
        RuntimeError::Producer(format!(
            "model '{}' is an embedding model and does not support generative completions",
            self.model_id
        ))
    }
}

impl ChatModel for CoreMlEmbeddingModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }

    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    fn max_context(&self) -> u32 {
        self.model.max_tokens() as u32
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_aliases(&self) -> Vec<String> {
        Vec::new()
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.model.embed_texts(texts, None, None)
    }

    fn with_producer(
        &self,
        _f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        Err(self.no_generation())
    }

    fn run_completion(
        &self,
        _prompt_ids: &[foundation::TokenId],
        _config: &GenerationConfig,
        _images: Option<&crate::vision::RequestImages>,
        _cancel: runtime::CancelFlag<'_>,
        _on_progress: &mut dyn FnMut(RawDecodeProgress),
    ) -> Result<RawDecodeResult, RuntimeError> {
        Err(self.no_generation())
    }
}

/// Opens an embedding model directory with the backend its layout needs:
/// the Core ML EmbeddingGemma 2 export, or a safetensors encoder.
pub fn open_embedding_model(dir: &Path, model_id: String) -> Result<Arc<dyn ChatModel>, String> {
    if runtime::coreml::detect_layout(dir) == Some(runtime::coreml::CoreMlLayout::EmbeddingGemma2) {
        return Ok(Arc::new(
            CoreMlEmbeddingModel::open(dir)?.with_model_id(model_id),
        ));
    }
    Ok(Arc::new(
        crate::RealEncoderModel::open(dir)?.with_model_id(model_id),
    ))
}
