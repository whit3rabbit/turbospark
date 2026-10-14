//! GTE (`Snowflake/snowflake-arctic-embed-m-v2.0`): the Alibaba GTE
//! ("new") encoder, a post-LN BERT with rotary positions, a packed qkv
//! projection, and a gated GELU MLP, followed by sentence-transformers
//! CLS pooling and L2 normalization.
//!
//! Numeric reference: the remote modeling code pinned with the checkpoint
//! (`modeling_hf_alibaba_nlp_gte.py`) under transformers 4.57.1, plus
//! sentence-transformers 5.1.2 (`SentenceTransformer.encode`). See
//! `README.md` in this directory for pins, citations, verified stages,
//! and divergences.
//!
//! Text handling follows `SentenceTransformer.encode`: the prompt string
//! is prepended, the result is stripped (Python `str.strip`), tokenized
//! with `<s>` and `</s>`, and cut to `max_seq_length` with the special
//! tokens kept. Each text runs unpadded, so a result does not depend on
//! its batch neighbors.

pub mod config;
pub mod encoder;

#[cfg(test)]
mod tests;

use std::path::Path;

use serde_json::Value;

use crate::error::{bad_config, input_error, EmbeddingError};
use crate::normalize::{l2_normalize, truncate_and_normalize};
use crate::pooling::pool;
use crate::st_config::{read_modules, PoolingConfig, PromptConfig, StModule};
use crate::tokenize::TextTokenizer;
use crate::weights::Weights;
use crate::Result;

pub use config::GteConfig;
pub use encoder::Encoder;

/// The tokenizer plus the special tokens its post-processor wraps around
/// the content, so truncation keeps them like HF `truncation=True`.
#[derive(Debug)]
struct WrappedTokenizer {
    inner: TextTokenizer,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

impl WrappedTokenizer {
    fn load(path: &Path) -> Result<Self> {
        let inner = TextTokenizer::from_file(path)?;
        let probe = "probe text";
        let with = inner.encode(probe, true)?;
        let content = inner.encode(probe, false)?;
        let start = (0..=with.len().saturating_sub(content.len()))
            .find(|&s| with[s..].starts_with(&content))
            .filter(|_| !content.is_empty())
            .ok_or_else(|| EmbeddingError::Tokenizer {
                why: "post-processor does not wrap content as prefix + text + suffix".into(),
            })?;
        Ok(Self {
            prefix: with[..start].to_vec(),
            suffix: with[start + content.len()..].to_vec(),
            inner,
        })
    }

    /// `prefix + content + suffix`, content cut from the end to fit.
    fn encode(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let content = self.inner.encode(text, false)?;
        let room = max_len.saturating_sub(self.prefix.len() + self.suffix.len());
        let keep = content.len().min(room);
        let mut ids = Vec::with_capacity(self.prefix.len() + keep + self.suffix.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend_from_slice(&content[..keep]);
        ids.extend_from_slice(&self.suffix);
        Ok(ids)
    }
}

/// Python `str.strip()` with no argument.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| bad_config(&path.display().to_string(), e))
}

/// A loaded GTE sentence encoder.
#[derive(Debug)]
pub struct Gte {
    encoder: Encoder,
    pooling: PoolingConfig,
    normalize: bool,
    tokenizer: Option<WrappedTokenizer>,
    prompts: PromptConfig,
    max_seq_length: usize,
    truncate_dim: Option<usize>,
}

impl Gte {
    /// Loads a sentence-transformers checkpoint directory. `tokenizer.json`
    /// is optional; without it only [`embed_ids`](Self::embed_ids) works.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = GteConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        let encoder = Encoder::load(Weights::open(dir)?, &config)?;
        let (pooling, normalize) = load_st_head(dir)?;
        if pooling.word_embedding_dimension != config.hidden_size {
            return Err(bad_config(
                "1_Pooling.word_embedding_dimension",
                format!(
                    "{} != hidden_size {}",
                    pooling.word_embedding_dimension, config.hidden_size
                ),
            ));
        }
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.is_file() {
            Some(WrappedTokenizer::load(&tok_path)?)
        } else {
            None
        };
        let max_seq_length = read_max_seq_length(dir, config.max_position_embeddings)?;
        if let Some(t) = &tokenizer {
            if max_seq_length <= t.prefix.len() + t.suffix.len() {
                return Err(bad_config(
                    "max_seq_length",
                    "leaves no room for content after special tokens",
                ));
            }
        }
        Ok(Self {
            encoder,
            pooling,
            normalize,
            tokenizer,
            prompts: PromptConfig::from_dir(dir)?,
            max_seq_length,
            truncate_dim: None,
        })
    }

    /// Matryoshka output: keep the first `dim` values and renormalize.
    /// arctic-embed-m-v2.0 was trained for 256 (`matryoshka_dimensions`).
    /// `None` restores the full width.
    pub fn with_truncate_dim(mut self, dim: Option<usize>) -> Result<Self> {
        if let Some(d) = dim {
            let full = self.full_dimension();
            if d == 0 || d > full {
                return Err(input_error(format!("truncate_dim {d} outside 1..={full}")));
            }
        }
        self.truncate_dim = dim;
        Ok(self)
    }

    pub fn config(&self) -> &GteConfig {
        self.encoder.config()
    }

    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Token budget per text, prompt and special tokens included.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Width of the pooled vector before any Matryoshka truncation.
    pub fn full_dimension(&self) -> usize {
        self.pooling.output_dimension()
    }

    /// Width of the vectors `embed*` return.
    pub fn dimension(&self) -> usize {
        self.truncate_dim.unwrap_or_else(|| self.full_dimension())
    }

    /// Prompt names from `config_sentence_transformers.json`.
    pub fn prompt_names(&self) -> impl Iterator<Item = &str> {
        self.prompts.prompts.keys().map(String::as_str)
    }

    /// The prompt string for `name`.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.prompt(name)
    }

    fn resolve_prompt(&self, name: Option<&str>) -> Result<&str> {
        match name.or(self.prompts.default_prompt_name.as_deref()) {
            None => Ok(""),
            Some(n) => self.prompts.prompt(n).ok_or_else(|| {
                let known: Vec<&str> = self.prompt_names().collect();
                input_error(format!("unknown prompt name {n:?}; known: {known:?}"))
            }),
        }
    }

    /// Token ids sentence-transformers would feed for `text` under
    /// `prompt_name` (or the default prompt when `None`).
    pub fn tokenize(&self, text: &str, prompt_name: Option<&str>) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EmbeddingError::Tokenizer {
                why: "checkpoint directory has no tokenizer.json".into(),
            })?;
        let full = format!("{}{text}", self.resolve_prompt(prompt_name)?);
        tok.encode(py_strip(&full), self.max_seq_length)
    }

    /// Embeds `text` with the default prompt (none for this checkpoint;
    /// documents are embedded without a prompt).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, None)?)
    }

    /// Embeds `text` with a named prompt (`"query"`).
    pub fn embed_with_prompt(&self, prompt_name: &str, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, Some(prompt_name))?)
    }

    /// Embeds each text independently (unpadded).
    pub fn embed_batch(&self, texts: &[&str], prompt_name: Option<&str>) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| self.embed_ids(&self.tokenize(t, prompt_name)?))
            .collect()
    }

    /// Runs the pipeline on already-tokenized ids (special tokens
    /// included): encoder, pooling, Normalize, Matryoshka.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let limit = self.config().max_position_embeddings;
        if ids.len() > limit {
            return Err(input_error(format!(
                "{} tokens exceed max_position_embeddings {limit}",
                ids.len()
            )));
        }
        let hidden = self.encoder.forward(ids)?;
        self.head(&hidden, ids.len())
    }

    /// Pooling, Normalize, and truncation over `hidden [seq, dim]`.
    pub fn head(&self, hidden: &[f32], seq: usize) -> Result<Vec<f32>> {
        let mut x = pool(&self.pooling, hidden, &vec![1u32; seq], 0)?;
        if self.normalize {
            l2_normalize(&mut x);
        }
        Ok(match self.truncate_dim {
            Some(dim) => truncate_and_normalize(&x, dim),
            None => x,
        })
    }
}

/// sentence-transformers 5.1.2 `Transformer.__init__`: the
/// `sentence_bert_config.json` value when set, else
/// `min(max_position_embeddings, tokenizer.model_max_length)`. The
/// pinned checkpoint has no sentence_bert_config.json, so this resolves
/// to min(8192, 32768) = 8192.
fn read_max_seq_length(dir: &Path, limit: usize) -> Result<usize> {
    let sbc = dir.join("sentence_bert_config.json");
    let mut explicit = None;
    if sbc.is_file() {
        let v = read_json(&sbc)?;
        if v.get("do_lower_case").and_then(Value::as_bool) == Some(true) {
            return Err(EmbeddingError::Unsupported {
                why: "sentence_bert_config do_lower_case = true is not implemented".into(),
            });
        }
        match v.get("max_seq_length") {
            None | Some(Value::Null) => {}
            Some(n) => {
                explicit = Some(
                    n.as_u64()
                        .ok_or_else(|| bad_config("max_seq_length", "not a non-negative integer"))?
                        as usize,
                )
            }
        }
    }
    let n = match explicit {
        Some(n) => n,
        None => {
            let tc = dir.join("tokenizer_config.json");
            let model_max = if tc.is_file() {
                read_json(&tc)?
                    .get("model_max_length")
                    .and_then(Value::as_f64)
            } else {
                None
            };
            match model_max {
                Some(m) if m >= 1.0 && m < limit as f64 => m as usize,
                _ => limit,
            }
        }
    };
    if n == 0 || n > limit {
        return Err(bad_config(
            "max_seq_length",
            format!("{n} outside 1..=max_position_embeddings {limit}"),
        ));
    }
    Ok(n)
}

/// `modules.json` must read Transformer, Pooling, then optionally
/// Normalize. A Dense head or an excluded prompt is not implemented.
fn load_st_head(dir: &Path) -> Result<(PoolingConfig, bool)> {
    let mut it = read_modules(dir)?.into_iter();
    if it.next() != Some(StModule::Transformer) {
        return Err(bad_config(
            "modules.json",
            "first module must be the Transformer",
        ));
    }
    let pooling = match it.next() {
        Some(StModule::Pooling(p)) => p,
        _ => return Err(bad_config("modules.json", "second module must be Pooling")),
    };
    if !pooling.include_prompt {
        return Err(EmbeddingError::Unsupported {
            why: "pooling include_prompt = false is not implemented".into(),
        });
    }
    let normalize = match it.next() {
        None => false,
        Some(StModule::Normalize) => true,
        Some(other) => {
            return Err(EmbeddingError::Unsupported {
                why: format!("modules.json module {other:?} after Pooling is not implemented"),
            })
        }
    };
    if let Some(other) = it.next() {
        return Err(EmbeddingError::Unsupported {
            why: format!("modules.json module {other:?} after Normalize is not implemented"),
        });
    }
    Ok((pooling, normalize))
}
