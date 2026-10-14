//! BERT and XLM-RoBERTa sentence encoders (Snowflake arctic-embed): a
//! post-LN transformer encoder, sentence-transformers pooling (CLS for
//! every pinned checkpoint), and L2 normalization.
//!
//! Numeric reference: transformers 4.57.1 `BertModel` / `XLMRobertaModel`
//! plus sentence-transformers 5.1.2 (`SentenceTransformer.encode`). See
//! `README.md` in this directory for pins, verified stages, and
//! divergences.
//!
//! [`BertEmbedder::load`] reads `modules.json`, which must list the
//! Transformer, one Pooling module, and optionally Normalize. Pooling
//! comes from the module's `config.json`. The mlx-community conversions
//! ship `modules.json` without the module directories; for those the
//! pooling is taken from the pinned upstream the conversion was made
//! from, but only when the directory matches that upstream's config and
//! prompt table exactly (see [`known_conversion`]). Any other directory
//! without a pooling config is refused: guessing would change every
//! embedding. [`BertEmbedder::load_with_pooling`] takes the pooling from
//! the caller instead.
//!
//! Text handling follows `SentenceTransformer.encode`: the prompt string
//! is prepended, the result is stripped (Python `str.strip`), tokenized
//! with the post-processor's special tokens (`[CLS]`/`[SEP]` or
//! `<s>`/`</s>`), and the content is cut so the whole fits
//! `max_seq_length` with the special tokens kept.
//!
//! ```no_run
//! use turbospark_embeddings::bert::BertEmbedder;
//! use turbospark_embeddings::cosine_similarity;
//!
//! # fn main() -> turbospark_embeddings::Result<()> {
//! let model = BertEmbedder::load("/path/to/snowflake-arctic-embed-l-v2.0")?;
//! let query = model.embed_query("What is a cloud data warehouse?")?;
//! let doc = model.embed_document("Snowflake separates storage from compute.")?;
//! println!("{:.4}", cosine_similarity(&query, &doc));
//! // Matryoshka: l-v2.0 and m-v1.5 were trained for 256.
//! let small = model.with_truncate_dim(Some(256))?;
//! assert_eq!(small.embed_query("short")?.len(), 256);
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod encoder;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::error::{bad_config, input_error, EmbeddingError};
use crate::normalize::{l2_normalize, truncate_and_normalize};
use crate::pooling::pool;
use crate::st_config::{ModuleEntry, PoolingConfig, PromptConfig};
use crate::tokenize::TextTokenizer;
use crate::weights::Weights;
use crate::Result;

pub use config::{xlmr_position_ids, Arch, BertConfig};
pub use encoder::Encoder;

/// Which checkpoint layout [`BertEmbedder::load`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Upstream sentence-transformers directory: the Pooling module's
    /// directory is present.
    SentenceTransformers,
    /// mlx-community conversion: `modules.json` names a Pooling directory
    /// that is absent. Weights may be dense (BF16) or affine-quantized.
    Mlx,
}

/// Where the pooling configuration came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolingSource {
    /// The Pooling module's `config.json` (path relative to the directory).
    Module(String),
    /// The pinned upstream checkpoint a recognized MLX conversion was made
    /// from (repository id).
    ConvertedFrom(&'static str),
    /// [`BertEmbedder::load_with_pooling`].
    Caller,
}

/// sentence-transformers reads the first of these that exists
/// (`Transformer.load_config`, 5.1.2).
const ST_CONFIG_FILES: &[&str] = &[
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
];

const ST_TRANSFORMER: &str = "sentence_transformers.models.Transformer";
const ST_POOLING: &str = "sentence_transformers.models.Pooling";
const ST_NORMALIZE: &str = "sentence_transformers.models.Normalize";

/// An upstream checkpoint whose MLX conversions drop `1_Pooling/`.
struct KnownUpstream {
    repo: &'static str,
    arch: Arch,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    pad_token_id: u32,
    /// `config_sentence_transformers.json` prompts, exactly.
    prompts: &'static [(&'static str, &'static str)],
    /// The upstream `1_Pooling/config.json` mode (CLS for arctic-embed).
    cls: bool,
}

/// Conversions recognized by [`known_conversion`]. Every entry must be a
/// pinned upstream with at least one pinned conversion (unit-tested).
const KNOWN_UPSTREAMS: &[KnownUpstream] = &[KnownUpstream {
    repo: "Snowflake/snowflake-arctic-embed-l-v2.0",
    arch: Arch::XlmRoberta,
    vocab_size: 250002,
    hidden_size: 1024,
    num_hidden_layers: 24,
    num_attention_heads: 16,
    intermediate_size: 4096,
    max_position_embeddings: 8194,
    type_vocab_size: 1,
    pad_token_id: 1,
    prompts: &[("query", "query: ")],
    cls: true,
}];

/// The pinned upstream an MLX directory was converted from, with that
/// upstream's pooling, when the directory's config and prompt table match
/// it field for field. Matching is a fingerprint, not proof of origin:
/// another XLM-R-large checkpoint with the same shape and the same prompt
/// table would match. bge-m3, for example, has the same shape but no
/// prompts, so it does not.
pub fn known_conversion(
    config: &BertConfig,
    prompts: &PromptConfig,
) -> Option<(&'static str, PoolingConfig)> {
    KNOWN_UPSTREAMS.iter().find_map(|k| {
        let same = config.arch == k.arch
            && config.vocab_size == k.vocab_size
            && config.hidden_size == k.hidden_size
            && config.num_hidden_layers == k.num_hidden_layers
            && config.num_attention_heads == k.num_attention_heads
            && config.intermediate_size == k.intermediate_size
            && config.max_position_embeddings == k.max_position_embeddings
            && config.type_vocab_size == k.type_vocab_size
            && config.pad_token_id == k.pad_token_id
            && prompts.prompts.len() == k.prompts.len()
            && k.prompts.iter().all(|(n, p)| prompts.prompt(n) == Some(*p));
        same.then(|| {
            let mut pooling = PoolingConfig::mean(k.hidden_size);
            pooling.mean = !k.cls;
            pooling.cls = k.cls;
            (k.repo, pooling)
        })
    })
}

/// The tokenizer plus the special tokens its template wraps around the
/// content, so truncation can keep them like HF `truncation=True` does.
#[derive(Debug)]
struct WrappedTokenizer {
    inner: TextTokenizer,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

impl WrappedTokenizer {
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

/// `prefix + content + suffix` with `content` cut from the end so the
/// whole fits `max_len`. HF fast tokenizers truncate before the
/// post-processor adds special tokens, so `[SEP]` / `</s>` survives.
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

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|e| EmbeddingError::Io {
        path: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| bad_config(&path.display().to_string(), e))
}

/// A loaded BERT / XLM-RoBERTa sentence encoder.
#[derive(Debug)]
pub struct BertEmbedder {
    encoder: Encoder,
    layout: Layout,
    pooling: PoolingConfig,
    pooling_source: PoolingSource,
    normalize: bool,
    tokenizer: Option<WrappedTokenizer>,
    /// sentence-transformers' `{"query": "", "document": ""}` updated with
    /// the file's prompts.
    prompts: BTreeMap<String, String>,
    default_prompt_name: Option<String>,
    max_seq_length: usize,
    truncate_dim: Option<usize>,
}

/// The module list: Transformer, Pooling (its path), optional Normalize.
fn read_module_list(dir: &Path) -> Result<(String, bool)> {
    let path = dir.join("modules.json");
    if !path.is_file() {
        // sentence-transformers would invent mean pooling here; arctic
        // checkpoints are CLS, so a missing module list is an error.
        return Err(EmbeddingError::Io {
            path: path.display().to_string(),
            why: "modules.json is required (no implicit pooling)".into(),
        });
    }
    let mut entries: Vec<ModuleEntry> =
        serde_json::from_value(read_json(&path)?).map_err(|e| bad_config("modules.json", e))?;
    entries.sort_by_key(|e| e.idx);
    let kinds: Vec<&str> = entries.iter().map(|e| e.kind.as_str()).collect();
    let normalize = match kinds.as_slice() {
        [ST_TRANSFORMER, ST_POOLING] => false,
        [ST_TRANSFORMER, ST_POOLING, ST_NORMALIZE] => true,
        _ => {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "modules.json {kinds:?}; only Transformer, Pooling[, Normalize] is implemented"
                ),
            })
        }
    };
    if !entries[0].path.is_empty() {
        return Err(bad_config(
            "modules.json",
            "the Transformer module must live at the directory root",
        ));
    }
    Ok((entries[1].path.clone(), normalize))
}

/// `max_seq_length` as sentence-transformers 5.1.2 resolves it: the
/// first `sentence_*_config.json` that exists, else
/// `min(max_position_embeddings, tokenizer.model_max_length)` where an
/// absent `model_max_length` is transformers' effectively infinite
/// default.
fn read_max_seq_length(dir: &Path, config: &BertConfig) -> Result<usize> {
    let limit = config.max_position_embeddings;
    if let Some(path) = ST_CONFIG_FILES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
    {
        let v = read_json(&path)?;
        if v.get("do_lower_case").and_then(Value::as_bool) == Some(true) {
            return Err(EmbeddingError::Unsupported {
                why: format!(
                    "{}: do_lower_case = true is not implemented",
                    path.display()
                ),
            });
        }
        match v.get("max_seq_length") {
            None | Some(Value::Null) => {}
            Some(n) => {
                let n = n
                    .as_u64()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| bad_config("max_seq_length", "not a positive integer"))?;
                return Ok(n as usize);
            }
        }
    }
    let tok_config = dir.join("tokenizer_config.json");
    let model_max = if tok_config.is_file() {
        match read_json(&tok_config)?.get("model_max_length") {
            None | Some(Value::Null) => None,
            Some(n) => Some(
                n.as_f64()
                    .filter(|x| *x >= 1.0)
                    .ok_or_else(|| bad_config("model_max_length", "not a positive number"))?,
            ),
        }
    } else {
        None
    };
    Ok(match model_max {
        Some(m) if m < limit as f64 => m as usize,
        _ => limit,
    })
}

impl BertEmbedder {
    /// Loads a checkpoint directory (sentence-transformers or MLX layout).
    /// `tokenizer.json` is optional; without it only
    /// [`embed_ids`](Self::embed_ids) works.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_inner(dir.as_ref(), None)
    }

    /// [`load`](Self::load) with the pooling supplied by the caller, for a
    /// directory whose Pooling module config is missing and that
    /// [`known_conversion`] does not recognize.
    pub fn load_with_pooling(dir: impl AsRef<Path>, pooling: PoolingConfig) -> Result<Self> {
        Self::load_inner(dir.as_ref(), Some(pooling))
    }

    fn load_inner(dir: &Path, explicit: Option<PoolingConfig>) -> Result<Self> {
        let config = BertConfig::from_json(&read_json(&dir.join("config.json"))?)?;
        let (pool_path, normalize) = read_module_list(dir)?;
        let prompt_file = PromptConfig::from_dir(dir)?;
        let pool_config = dir.join(&pool_path).join("config.json");
        let layout = if pool_config.is_file() {
            Layout::SentenceTransformers
        } else {
            Layout::Mlx
        };
        let (pooling, pooling_source) = match explicit {
            Some(p) => (p, PoolingSource::Caller),
            None if layout == Layout::SentenceTransformers => (
                PoolingConfig::from_path(&pool_config)?,
                PoolingSource::Module(pool_path.clone()),
            ),
            None => match known_conversion(&config, &prompt_file) {
                Some((repo, p)) => (p, PoolingSource::ConvertedFrom(repo)),
                None => {
                    return Err(EmbeddingError::Unsupported {
                        why: format!(
                            "{} is missing and the directory is not a recognized conversion \
                             of a pinned upstream; pass the pooling with load_with_pooling",
                            pool_config.display()
                        ),
                    })
                }
            },
        };
        let hidden = config.hidden_size;
        if !pooling.include_prompt {
            // Needs the reference prompt-length rule, which is unported.
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
        if pooling.modes().is_empty() {
            return Err(EmbeddingError::Unsupported {
                why: "pooling config enables no pooling mode".into(),
            });
        }

        let encoder = Encoder::load(Weights::open(dir)?, &config)?;
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = if tok_path.is_file() {
            Some(WrappedTokenizer::load(&tok_path)?)
        } else {
            None
        };
        let max_seq_length = read_max_seq_length(dir, &config)?;
        if let Some(t) = &tokenizer {
            if max_seq_length <= t.prefix.len() + t.suffix.len() {
                return Err(bad_config(
                    "max_seq_length",
                    "leaves no room for content after special tokens",
                ));
            }
        }
        let mut prompts: BTreeMap<String, String> = [("query", ""), ("document", "")]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        prompts.extend(prompt_file.prompts.clone());
        if let Some(d) = &prompt_file.default_prompt_name {
            if !prompts.contains_key(d) {
                return Err(bad_config(
                    "default_prompt_name",
                    format!("{d:?} is not a prompt name"),
                ));
            }
        }
        Ok(Self {
            encoder,
            layout,
            pooling,
            pooling_source,
            normalize,
            tokenizer,
            prompts,
            default_prompt_name: prompt_file.default_prompt_name,
            max_seq_length,
            truncate_dim: None,
        })
    }

    /// Matryoshka output: keep the first `dim` values and renormalize
    /// (sentence-transformers `truncate_dim` with
    /// `normalize_embeddings=True`). arctic-embed-m-v1.5 and l-v2.0 were
    /// trained for 256 (see [`matryoshka_dimensions`](Self::matryoshka_dimensions));
    /// other widths are allowed but untrained. `None` restores the full width.
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

    pub fn pooling(&self) -> &PoolingConfig {
        &self.pooling
    }

    pub fn pooling_source(&self) -> &PoolingSource {
        &self.pooling_source
    }

    pub fn config(&self) -> &BertConfig {
        self.encoder.config()
    }

    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Token budget per text, prompt and special tokens included.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Matryoshka widths the checkpoint's `config.json` advertises.
    pub fn matryoshka_dimensions(&self) -> &[usize] {
        &self.config().matryoshka_dimensions
    }

    /// Width of the pooled vector before any Matryoshka truncation.
    pub fn full_dimension(&self) -> usize {
        self.pooling.output_dimension()
    }

    /// Width of the vectors `embed*` return.
    pub fn dimension(&self) -> usize {
        self.truncate_dim.unwrap_or_else(|| self.full_dimension())
    }

    /// Prompt names: sentence-transformers' `query` and `document`
    /// defaults plus the checkpoint's own.
    pub fn prompt_names(&self) -> impl Iterator<Item = &str> {
        self.prompts.keys().map(String::as_str)
    }

    /// The prompt string for `name`.
    pub fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts.get(name).map(String::as_str)
    }

    fn resolve_prompt(&self, name: Option<&str>) -> Result<&str> {
        match name.or(self.default_prompt_name.as_deref()) {
            None => Ok(""),
            Some(n) => self.prompt(n).ok_or_else(|| {
                let known: Vec<&str> = self.prompt_names().collect();
                input_error(format!("unknown prompt name {n:?}; known: {known:?}"))
            }),
        }
    }

    /// Token ids sentence-transformers would feed for `text` under
    /// `prompt_name` (or the default prompt, none for arctic-embed).
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

    /// Embeds `text` with the default prompt (none for arctic-embed), as
    /// `SentenceTransformer.encode(text)` does.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, None)?)
    }

    /// Embeds `text` with a named prompt (`"query"`, `"document"`).
    pub fn embed_with_prompt(&self, prompt_name: &str, text: &str) -> Result<Vec<f32>> {
        self.embed_ids(&self.tokenize(text, Some(prompt_name))?)
    }

    /// `encode_query`: the `query` prompt.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_with_prompt("query", text)
    }

    /// `encode_document`: the `document` prompt (empty for arctic-embed).
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_with_prompt("document", text)
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
    /// included), every token attended: encoder, pooling, Normalize,
    /// Matryoshka.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.encoder.forward(ids)?;
        self.head(&hidden, &vec![1u32; ids.len()])
    }

    /// Pooling, Normalize, and truncation over `hidden [seq, dim]` with
    /// the attention `mask [seq]`.
    pub fn head(&self, hidden: &[f32], mask: &[u32]) -> Result<Vec<f32>> {
        let mut x = pool(&self.pooling, hidden, mask, 0)?;
        if self.normalize {
            l2_normalize(&mut x);
        }
        Ok(match self.truncate_dim {
            Some(dim) => truncate_and_normalize(&x, dim),
            None => x,
        })
    }
}
