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

/// `config.json` `model_type` values this runner implements: post-LN BERT
/// and XLM-RoBERTa with CLS pooling. Other encoders (GTE, Nomic BERT, the
/// Gemma3 behind EmbeddingGemma) differ in attention, positions or pooling
/// and would load and return fluent, wrong vectors, so the family is checked
/// from `config.json` before any weights are read, never inferred from
/// tensor names.
pub const SUPPORTED_MODEL_TYPES: &[&str] = &["bert", "xlm-roberta"];

/// The supported `model_type` named by `config`, or why it is refused. A
/// missing `model_type` is refused rather than assumed to be BERT.
pub fn check_model_type(config: &serde_json::Value) -> Result<&'static str, String> {
    match config.get("model_type").and_then(|v| v.as_str()) {
        None => Err(
            "config.json names no model_type; the encoder runner needs bert or xlm-roberta"
                .into(),
        ),
        Some(t) => SUPPORTED_MODEL_TYPES
            .iter()
            .find(|s| **s == t)
            .copied()
            .ok_or_else(|| {
                format!("model_type {t:?} is not supported by the encoder runner (bert or xlm-roberta only)")
            }),
    }
}

/// sentence-transformers pooling modes other than CLS. Any of them set in
/// `1_Pooling/config.json` means the checkpoint was trained for a pooling
/// this runner does not compute.
const NON_CLS_POOLING: &[&str] = &[
    "pooling_mode_mean_tokens",
    "pooling_mode_max_tokens",
    "pooling_mode_mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens",
    "pooling_mode_lasttoken",
];

/// Refuses a sentence-transformers pooling config that is not CLS-only
/// (e5, all-MiniLM and other mean-pooled BERTs report `model_type: bert`).
/// A directory without `1_Pooling/` passes: the arctic-embed MLX
/// conversions ship none.
pub fn check_pooling(model_dir: &Path) -> Result<(), String> {
    let path = model_dir.join("1_Pooling").join("config.json");
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("failed to read {}: {e}", path.display())),
    };
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
    let set = |k: &str| v.get(k).and_then(|x| x.as_bool()) == Some(true);
    if set("pooling_mode_cls_token") && !NON_CLS_POOLING.iter().any(|k| set(k)) {
        Ok(())
    } else {
        Err("the model's sentence-transformers pooling is not CLS; the encoder runner pools the CLS token only".into())
    }
}

/// Family and pooling checks for an encoder directory, before any weights
/// are opened. Returns the `model_type`.
pub fn check_encoder_dir(model_dir: &Path) -> Result<&'static str, String> {
    let config_path = model_dir.join("config.json");
    let text = fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;
    let model_type = check_model_type(&v)?;
    check_pooling(model_dir)?;
    Ok(model_type)
}

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
        check_encoder_dir(model_dir)?;
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
        if input_ids.len() + self.config.position_offset() > self.config.max_position_embeddings {
            return Err(format!(
                "sequence length {} with position offset {} exceeds max_position_embeddings {}",
                input_ids.len(),
                self.config.position_offset(),
                self.config.max_position_embeddings
            ));
        }
        if let Some(&bad) = input_ids
            .iter()
            .find(|&&id| id as usize >= self.config.vocab_size)
        {
            return Err(format!(
                "token id {bad} exceeds vocab_size {}",
                self.config.vocab_size
            ));
        }
        if let Some(types) = token_type_ids {
            if types.len() != input_ids.len() {
                return Err(format!(
                    "token_type_ids length {} does not match input_ids length {}",
                    types.len(),
                    input_ids.len()
                ));
            }
            if let Some(&bad_type) = types
                .iter()
                .find(|&&t| t as usize >= self.config.type_vocab_size)
            {
                return Err(format!(
                    "token type id {bad_type} exceeds type_vocab_size {}",
                    self.config.type_vocab_size
                ));
            }
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
