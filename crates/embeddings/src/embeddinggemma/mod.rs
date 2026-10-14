//! EmbeddingGemma (`google/embeddinggemma-300m`): a Gemma3 text backbone
//! run as a bidirectional encoder, mean pooling, a two-layer bias-free
//! Dense head (768 -> 3072 -> 768), and L2 normalization.
//!
//! Numeric reference: transformers 4.57.1 `Gemma3TextModel` plus
//! sentence-transformers 5.1.2 (`SentenceTransformer.encode`). See
//! `README.md` in this directory for pins, verified stages, and
//! divergences.
//!
//! [`EmbeddingGemma::load`] accepts both on-disk layouts:
//!
//! - sentence-transformers (the upstream checkpoint): backbone tensors
//!   without a prefix in `model.safetensors`, the head in `2_Dense/` and
//!   `3_Dense/`, pipeline order from `modules.json`;
//! - MLX (the mlx-community conversions): backbone under `model.`, the
//!   head as `dense.{i}` in the same file, optionally MLX-affine
//!   quantized. Those repos ship `modules.json` but none of its module
//!   directories, so pooling falls back to EmbeddingGemma's mean pooling
//!   when `1_Pooling/config.json` is absent.
//!
//! Text handling follows `SentenceTransformer.encode`: the prompt string
//! is prepended, the result is stripped of surrounding whitespace
//! (Python `str.strip`), tokenized with `<bos>` and `<eos>`, and cut to
//! `max_seq_length` with the special tokens kept.

pub mod backbone;
pub mod config;

#[cfg(test)]
mod tests;

use std::path::Path;

use serde_json::Value;
use turbospark_audio::nn::Linear;

use crate::error::{bad_config, input_error, EmbeddingError};
use crate::normalize::{l2_normalize, truncate_and_normalize};
use crate::pooling::pool;
use crate::st_config::{read_modules, ModuleEntry, PoolingConfig, PromptConfig, StModule};
use crate::tokenize::TextTokenizer;
use crate::weights::Weights;
use crate::Result;

pub use backbone::Backbone;
pub use config::{GemmaConfig, LayerType};

/// Which checkpoint layout [`EmbeddingGemma::load`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Upstream sentence-transformers directory.
    SentenceTransformers,
    /// mlx-community conversion (dense or affine-quantized).
    Mlx,
}

/// The tokenizer plus the special tokens its template wraps around the
/// content, so truncation can keep them like HF `truncation=True` does.
#[derive(Debug)]
struct GemmaTokenizer {
    inner: TextTokenizer,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

impl GemmaTokenizer {
    fn load(path: &Path) -> Result<Self> {
        let inner = TextTokenizer::from_file(path)?;
        // Probe the post-processor template: the content ids must appear
        // contiguously inside the special-token encoding.
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

    fn encode(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let content = self.inner.encode(text, false)?;
        Ok(assemble(&self.prefix, &content, &self.suffix, max_len))
    }
}

/// `prefix + content + suffix`, with `content` cut from the end so the
/// whole fits `max_len`. HF fast tokenizers truncate the sequence before
/// the post-processor adds special tokens, so `<eos>` survives.
pub(crate) fn assemble(
    prefix: &[u32],
    content: &[u32],
    suffix: &[u32],
    max_len: usize,
) -> Vec<u32> {
    let room = max_len.saturating_sub(prefix.len() + suffix.len());
    let keep = content.len().min(room);
    let mut ids = Vec::with_capacity(prefix.len() + keep + suffix.len());
    ids.extend_from_slice(prefix);
    ids.extend_from_slice(&content[..keep]);
    ids.extend_from_slice(suffix);
    ids
}

/// Python `str.strip()` with no argument: Rust's `White_Space` set plus
/// the four information separators U+001C..U+001F, which Python also
/// treats as whitespace.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// A loaded EmbeddingGemma model.
#[derive(Debug)]
pub struct EmbeddingGemma {
    backbone: Backbone,
    layout: Layout,
    pooling: PoolingConfig,
    dense: Vec<Linear>,
    normalize: bool,
    tokenizer: Option<GemmaTokenizer>,
    prompts: PromptConfig,
    max_seq_length: usize,
    truncate_dim: Option<usize>,
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| bad_config(&path.display().to_string(), e))
}

impl EmbeddingGemma {
    /// Loads a checkpoint directory in either layout. `tokenizer.json` is
    /// optional; without it only [`embed_ids`](Self::embed_ids) works.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = GemmaConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        let weights = Weights::open(dir)?;
        let (layout, prefix) = if weights.contains("model.embed_tokens.weight") {
            (Layout::Mlx, "model.")
        } else if weights.contains("embed_tokens.weight") {
            (Layout::SentenceTransformers, "")
        } else {
            return Err(EmbeddingError::Tensor {
                name: "embed_tokens.weight".into(),
                why: "neither embed_tokens.weight nor model.embed_tokens.weight is present".into(),
            });
        };
        let backbone = Backbone::load(&weights, prefix, &config)?;
        let hidden = config.hidden_size;
        let (pooling, dense, normalize) = match layout {
            Layout::SentenceTransformers => load_st_head(dir)?,
            Layout::Mlx => load_mlx_head(dir, &weights, &config)?,
        };
        if !pooling.include_prompt {
            // EmbeddingGemma pools over the prompt; excluding it needs
            // the reference prompt-length rule, which is unported.
            return Err(EmbeddingError::Unsupported {
                why: "pooling include_prompt = false is not implemented".into(),
            });
        }
        if pooling.word_embedding_dimension != hidden {
            return Err(bad_config(
                "1_Pooling.word_embedding_dimension",
                format!(
                    "{} != hidden_size {hidden}",
                    pooling.word_embedding_dimension
                ),
            ));
        }
        let mut width = pooling.output_dimension();
        for (i, d) in dense.iter().enumerate() {
            if d.input != width {
                return Err(bad_config(
                    &format!("dense[{i}].in_features"),
                    format!("{} != previous output {width}", d.input),
                ));
            }
            width = d.output;
        }

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.is_file() {
            Some(GemmaTokenizer::load(&tok_path)?)
        } else {
            None
        };
        let prompts = PromptConfig::from_dir(dir)?;
        let max_seq_length = read_max_seq_length(dir, &config)?;
        if let Some(t) = &tokenizer {
            if max_seq_length <= t.prefix.len() + t.suffix.len() {
                return Err(bad_config(
                    "max_seq_length",
                    "leaves no room for content after special tokens",
                ));
            }
        }
        Ok(Self {
            backbone,
            layout,
            pooling,
            dense,
            normalize,
            tokenizer,
            prompts,
            max_seq_length,
            truncate_dim: None,
        })
    }

    /// Matryoshka output: keep the first `dim` values and renormalize.
    /// EmbeddingGemma was trained for 768, 512, 256, and 128. `None`
    /// restores the full width.
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

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn config(&self) -> &GemmaConfig {
        self.backbone.config()
    }

    pub fn backbone(&self) -> &Backbone {
        &self.backbone
    }

    /// Token budget per text, prompt and special tokens included.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Width of the head output before any Matryoshka truncation.
    pub fn full_dimension(&self) -> usize {
        self.dense
            .last()
            .map_or(self.pooling.output_dimension(), |d| d.output)
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
        let name = match name {
            Some(n) => Some(n),
            None => self.prompts.default_prompt_name.as_deref(),
        };
        match name {
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
        let prompt = self.resolve_prompt(prompt_name)?;
        let full = format!("{prompt}{text}");
        tok.encode(py_strip(&full), self.max_seq_length)
    }

    /// Embeds `text` with the default prompt (none for EmbeddingGemma).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, None)?)
    }

    /// Embeds `text` with a named prompt (`"query"`, `"document"`, ...).
    pub fn embed_with_prompt(&self, prompt_name: &str, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, Some(prompt_name))?)
    }

    /// Embeds each text independently (unpadded), so a result does not
    /// depend on its batch neighbors.
    pub fn embed_batch(&self, texts: &[&str], prompt_name: Option<&str>) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| self.embed_ids(&self.tokenize(t, prompt_name)?))
            .collect()
    }

    /// Runs the pipeline on already-tokenized ids (special tokens
    /// included): backbone, pooling, Dense head, Normalize, Matryoshka.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let limit = self.config().max_position_embeddings;
        if ids.len() > limit {
            return Err(input_error(format!(
                "{} tokens exceed max_position_embeddings {limit}",
                ids.len()
            )));
        }
        let hidden = self.backbone.forward(ids)?;
        self.head(&hidden, ids.len())
    }

    /// Pooling, Dense, Normalize, and truncation over `hidden [seq, dim]`.
    pub fn head(&self, hidden: &[f32], seq: usize) -> Result<Vec<f32>> {
        let mask = vec![1u32; seq];
        let mut x = pool(&self.pooling, hidden, &mask, 0)?;
        for d in &self.dense {
            x = d.forward(&x, 1);
        }
        if self.normalize {
            l2_normalize(&mut x);
        }
        Ok(match self.truncate_dim {
            Some(dim) => truncate_and_normalize(&x, dim),
            None => x,
        })
    }
}

fn read_max_seq_length(dir: &Path, config: &GemmaConfig) -> Result<usize> {
    let path = dir.join("sentence_bert_config.json");
    let limit = config.max_position_embeddings;
    if !path.is_file() {
        return Ok(limit);
    }
    let v = read_json(&path)?;
    if v.get("do_lower_case").and_then(Value::as_bool) == Some(true) {
        return Err(EmbeddingError::Unsupported {
            why: "sentence_bert_config do_lower_case = true is not implemented".into(),
        });
    }
    let n = match v.get("max_seq_length") {
        None | Some(Value::Null) => return Ok(limit),
        Some(n) => n
            .as_u64()
            .ok_or_else(|| bad_config("max_seq_length", "not a non-negative integer"))?
            as usize,
    };
    if n == 0 || n > limit {
        return Err(bad_config(
            "max_seq_length",
            format!("{n} outside 1..=max_position_embeddings {limit}"),
        ));
    }
    Ok(n)
}

/// Pooling, Dense layers, and Normalize from `modules.json`, which must
/// read Transformer, Pooling, Dense*, then optionally Normalize.
fn load_st_head(dir: &Path) -> Result<(PoolingConfig, Vec<Linear>, bool)> {
    let modules = read_modules(dir)?;
    let mut it = modules.into_iter();
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
    let mut dense = Vec::new();
    let mut normalize = false;
    for m in it {
        match m {
            StModule::Dense { path, config } if !normalize => {
                let file = dir.join(&path).join("model.safetensors");
                if !file.is_file() {
                    return Err(EmbeddingError::Io {
                        path: file.display().to_string(),
                        why: "Dense weights must be safetensors".into(),
                    });
                }
                let w = Weights::open_file(&file)?;
                dense.push(w.linear(
                    "linear",
                    config.in_features,
                    config.out_features,
                    config.bias,
                    None,
                )?);
            }
            StModule::Normalize if !normalize => normalize = true,
            other => {
                return Err(bad_config(
                    "modules.json",
                    format!("unexpected module {other:?} after Pooling/Dense/Normalize"),
                ))
            }
        }
    }
    Ok((pooling, dense, normalize))
}

/// The MLX conversions keep the Dense head as `dense.{i}` in the main
/// file and ship `modules.json` without the module directories. The
/// module list still decides how many Dense layers and whether Normalize
/// runs; the tensors give the widths.
fn load_mlx_head(
    dir: &Path,
    weights: &Weights,
    config: &GemmaConfig,
) -> Result<(PoolingConfig, Vec<Linear>, bool)> {
    let hidden = config.hidden_size;
    let modules_path = dir.join("modules.json");
    let kinds: Option<Vec<String>> = if modules_path.is_file() {
        let mut entries: Vec<ModuleEntry> = serde_json::from_value(read_json(&modules_path)?)
            .map_err(|e| bad_config("modules.json", e))?;
        entries.sort_by_key(|e| e.idx);
        Some(entries.into_iter().map(|e| e.kind).collect())
    } else {
        None
    };
    let pool_path = dir.join("1_Pooling").join("config.json");
    let pooling = if pool_path.is_file() {
        PoolingConfig::from_path(&pool_path)?
    } else {
        PoolingConfig::mean(hidden)
    };

    let mut dense = Vec::new();
    let mut width = pooling.output_dimension();
    while weights.contains(&format!("dense.{}.weight", dense.len())) {
        let base = format!("dense.{}", dense.len());
        let out = *weights
            .shape(&format!("{base}.weight"))?
            .first()
            .ok_or_else(|| bad_config(&base, "scalar weight"))?;
        dense.push(weights.linear(&base, width, out, false, config.quantization)?);
        width = out;
    }

    let normalize = match &kinds {
        Some(kinds) => {
            let expect_dense = kinds
                .iter()
                .filter(|k| k.as_str() == "sentence_transformers.models.Dense")
                .count();
            if expect_dense != dense.len() {
                return Err(bad_config(
                    "modules.json",
                    format!(
                        "{expect_dense} Dense modules but {} dense.* tensors",
                        dense.len()
                    ),
                ));
            }
            for k in kinds {
                if !matches!(
                    k.as_str(),
                    "sentence_transformers.models.Transformer"
                        | "sentence_transformers.models.Pooling"
                        | "sentence_transformers.models.Dense"
                        | "sentence_transformers.models.Normalize"
                ) {
                    return Err(EmbeddingError::Unsupported {
                        why: format!("modules.json module type {k}"),
                    });
                }
            }
            kinds
                .iter()
                .any(|k| k == "sentence_transformers.models.Normalize")
        }
        // No module list: the EmbeddingGemma pipeline normalizes.
        None => true,
    };
    Ok((pooling, dense, normalize))
}
