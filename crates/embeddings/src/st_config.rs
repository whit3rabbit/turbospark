//! Sentence-transformers checkpoint layout: `modules.json`, the
//! `*_Pooling/config.json`, `*_Dense/config.json`, and the prompt table in
//! `config_sentence_transformers.json`.
//!
//! Only the module types this crate can execute are accepted. An unknown
//! module type is refused rather than skipped, because skipping a module
//! silently changes the embedding.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{bad_config, EmbeddingError};
use crate::pooling::PoolingMode;

fn read(path: &Path) -> Result<Vec<u8>, EmbeddingError> {
    std::fs::read(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })
}

fn parse<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, EmbeddingError> {
    serde_json::from_slice(&read(path)?).map_err(|e| bad_config(&path.display().to_string(), e))
}

/// `1_Pooling/config.json`. Field names follow the file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: usize,
    #[serde(rename = "pooling_mode_cls_token", default)]
    pub cls: bool,
    #[serde(rename = "pooling_mode_mean_tokens", default)]
    pub mean: bool,
    #[serde(rename = "pooling_mode_max_tokens", default)]
    pub max: bool,
    #[serde(rename = "pooling_mode_mean_sqrt_len_tokens", default)]
    pub mean_sqrt_len: bool,
    #[serde(rename = "pooling_mode_weightedmean_tokens", default)]
    pub weighted_mean: bool,
    #[serde(rename = "pooling_mode_lasttoken", default)]
    pub last_token: bool,
    #[serde(default = "default_true")]
    pub include_prompt: bool,
}

fn default_true() -> bool {
    true
}

impl PoolingConfig {
    /// Mean pooling only, prompt included: the common default.
    pub fn mean(dim: usize) -> Self {
        Self {
            word_embedding_dimension: dim,
            cls: false,
            mean: true,
            max: false,
            mean_sqrt_len: false,
            weighted_mean: false,
            last_token: false,
            include_prompt: true,
        }
    }

    pub fn from_path(path: &Path) -> Result<Self, EmbeddingError> {
        parse(path)
    }

    /// Enabled modes in the reference concatenation order.
    pub fn modes(&self) -> Vec<PoolingMode> {
        [
            (self.cls, PoolingMode::Cls),
            (self.max, PoolingMode::Max),
            (self.mean, PoolingMode::Mean),
            (self.mean_sqrt_len, PoolingMode::MeanSqrtLen),
            (self.weighted_mean, PoolingMode::WeightedMean),
            (self.last_token, PoolingMode::LastToken),
        ]
        .into_iter()
        .filter_map(|(on, mode)| on.then_some(mode))
        .collect()
    }

    /// Width of the pooled vector (`dim * enabled modes`).
    pub fn output_dimension(&self) -> usize {
        self.word_embedding_dimension * self.modes().len()
    }
}

/// `*_Dense/config.json`. Only the identity activation is executable here;
/// anything else is refused at load.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DenseConfig {
    pub in_features: usize,
    pub out_features: usize,
    #[serde(default = "default_true")]
    pub bias: bool,
    #[serde(default = "identity")]
    pub activation_function: String,
}

fn identity() -> String {
    IDENTITY_ACTIVATION.to_owned()
}

/// The activation path the sentence-transformers Dense config writes for a
/// plain linear layer.
pub const IDENTITY_ACTIVATION: &str = "torch.nn.modules.linear.Identity";

impl DenseConfig {
    pub fn from_path(path: &Path) -> Result<Self, EmbeddingError> {
        let config: Self = parse(path)?;
        if config.activation_function != IDENTITY_ACTIVATION {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "{}: Dense activation {} is not implemented",
                    path.display(),
                    config.activation_function
                ),
            });
        }
        Ok(config)
    }
}

/// One `modules.json` entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModuleEntry {
    pub idx: usize,
    pub name: String,
    pub path: String,
    #[serde(rename = "type")]
    pub kind: String,
}

/// The module kinds this crate executes, in pipeline order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StModule {
    /// The backbone (`sentence_transformers.models.Transformer`).
    Transformer,
    Pooling(PoolingConfig),
    Dense {
        path: String,
        config: DenseConfig,
    },
    Normalize,
}

/// Reads `modules.json` under `dir` and loads each module's config.
pub fn read_modules(dir: &Path) -> Result<Vec<StModule>, EmbeddingError> {
    let mut entries: Vec<ModuleEntry> = parse(&dir.join("modules.json"))?;
    entries.sort_by_key(|e| e.idx);
    entries
        .into_iter()
        .map(|entry| {
            let base = dir.join(&entry.path);
            match entry.kind.as_str() {
                "sentence_transformers.models.Transformer" => Ok(StModule::Transformer),
                "sentence_transformers.models.Pooling" => Ok(StModule::Pooling(
                    PoolingConfig::from_path(&base.join("config.json"))?,
                )),
                "sentence_transformers.models.Dense" => Ok(StModule::Dense {
                    path: entry.path.clone(),
                    config: DenseConfig::from_path(&base.join("config.json"))?,
                }),
                "sentence_transformers.models.Normalize" => Ok(StModule::Normalize),
                other => Err(EmbeddingError::Unsupported {
                    why: format!("modules.json entry {}: module type {other}", entry.name),
                }),
            }
        })
        .collect()
}

/// `config_sentence_transformers.json` prompt table.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct PromptConfig {
    #[serde(default)]
    pub prompts: BTreeMap<String, String>,
    #[serde(default)]
    pub default_prompt_name: Option<String>,
    #[serde(default)]
    pub similarity_fn_name: Option<String>,
}

impl PromptConfig {
    /// Reads the file when present; a missing file is an empty table.
    pub fn from_dir(dir: &Path) -> Result<Self, EmbeddingError> {
        let path = dir.join("config_sentence_transformers.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        parse(&path)
    }

    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.get(name).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_config_parses_reference_fields() {
        let json = br#"{"word_embedding_dimension": 768, "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_weightedmean_tokens": false,
            "pooling_mode_lasttoken": false, "include_prompt": true}"#;
        let c: PoolingConfig = serde_json::from_slice(json).unwrap();
        assert_eq!(c, PoolingConfig::mean(768));
        assert_eq!(c.output_dimension(), 768);
    }

    #[test]
    fn pooling_config_defaults_include_prompt() {
        let c: PoolingConfig = serde_json::from_slice(
            br#"{"word_embedding_dimension": 4, "pooling_mode_lasttoken": true}"#,
        )
        .unwrap();
        assert!(c.include_prompt);
        assert_eq!(c.modes(), vec![PoolingMode::LastToken]);
    }

    #[test]
    fn dense_refuses_nonidentity_activation() {
        let dir = std::env::temp_dir().join(format!("tse-dense-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            br#"{"in_features": 2, "out_features": 3, "bias": false,
                "activation_function": "torch.nn.modules.activation.Tanh"}"#,
        )
        .unwrap();
        let err = DenseConfig::from_path(&path).unwrap_err();
        assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
