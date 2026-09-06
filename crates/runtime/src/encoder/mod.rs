//! The BERT and XLM-RoBERTa encoder runner.
//!
//! Executes bidirectional encoder models to produce dense vector embeddings
//! for semantic search, retrieval, and similarity calculation.

pub mod weights;

use std::fs;
use std::path::Path;

pub use compute::cosine_similarity;
use compute::encoder::{
    cls_pool_and_normalize, encoder_block_forward, encoder_embeddings_lookup,
    EncoderReferenceConfig,
};
use model_io::encoder_config::EncoderConfig;
use model_io::safetensors::SafetensorsFile;
use tokenizer::Tokenizer;
use weights::EncoderWeights;

/// Runtime runner for BERT / XLM-RoBERTa encoder embedding models.
pub struct EncoderRunner {
    pub config: EncoderConfig,
    pub weights: EncoderWeights,
    pub tokenizer: Option<Tokenizer>,
}

impl EncoderRunner {
    /// Open an encoder model from a directory containing `config.json`,
    /// `model.safetensors`, and optionally `tokenizer.json`.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        let config = EncoderConfig::from_json_str(&config_str)
            .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;

        let safetensors_path = model_dir.join("model.safetensors");
        let file = SafetensorsFile::open(&safetensors_path)
            .map_err(|e| format!("failed to open {}: {e}", safetensors_path.display()))?;

        let weights = EncoderWeights::load_from_safetensors(&file, &config)
            .map_err(|e| format!("failed to load weights: {e}"))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(&tokenizer_path).ok()
        } else {
            None
        };

        Ok(Self {
            config,
            weights,
            tokenizer,
        })
    }

    /// Construct runner directly from config and weights (e.g. for testing).
    pub fn from_parts(config: EncoderConfig, weights: EncoderWeights) -> Self {
        Self {
            config,
            weights,
            tokenizer: None,
        }
    }

    /// Reference config used by compute kernels.
    fn reference_config(&self) -> EncoderReferenceConfig {
        EncoderReferenceConfig {
            hidden_size: self.config.hidden_size,
            num_attention_heads: self.config.num_attention_heads,
            intermediate_size: self.config.intermediate_size,
            layer_norm_eps: self.config.layer_norm_eps,
            position_offset: self.config.position_offset(),
            use_tanh_gelu: false,
        }
    }

    /// Compute unit-normalized embedding vector for a single sequence of token IDs.
    pub fn encode_tokens(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
    ) -> Result<Vec<f32>, String> {
        if input_ids.is_empty() {
            return Err("input_ids cannot be empty".to_string());
        }

        let ref_config = self.reference_config();
        let seq = input_ids.len();

        // 1. Embedding lookup + LayerNorm
        let mut x = encoder_embeddings_lookup(
            input_ids,
            token_type_ids,
            &self.weights.word_embeddings,
            &self.weights.position_embeddings,
            self.weights.token_type_embeddings.as_deref(),
            &self.weights.emb_ln_weight,
            &self.weights.emb_ln_bias,
            &ref_config,
        );

        // 2. Transformer layers
        for layer in &self.weights.layers {
            let borrowed = layer.as_borrowed();
            x = encoder_block_forward(&x, seq, &borrowed, &ref_config);
        }

        // 3. CLS pooling + Euclidean L2 normalization
        let pooled = cls_pool_and_normalize(&x, self.config.hidden_size);
        Ok(pooled)
    }

    /// Encode a batch of token ID sequences.
    pub fn encode_batch_tokens(&self, batch: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, String> {
        let mut results = Vec::with_capacity(batch.len());
        for tokens in batch {
            results.push(self.encode_tokens(tokens, None)?);
        }
        Ok(results)
    }

    /// Encode a text string using the loaded tokenizer.
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>, String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| "no tokenizer loaded in EncoderRunner".to_string())?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        self.encode_tokens(ids, Some(type_ids))
    }

    /// Encode a batch of text strings.
    pub fn encode_batch_text(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.encode_text(text)?);
        }
        Ok(out)
    }
}
