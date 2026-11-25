//! A Hugging Face `tokenizer.json` and `tokenizer_config.json` synthesized
//! from a GGUF's own `tokenizer.ggml.*` metadata.
//!
//! Without this a GGUF install needs a second repository (the checkpoint it
//! was converted from) just for its tokenizer, which no arbitrary local file
//! can name. The runtime keeps loading an HF `tokenizer.json`, so this
//! produces one at install time and nothing downstream changes.
//!
//! **Scope is deliberately narrow and everything else is refused by name.**
//! A tokenizer that splits text differently from the one the model was
//! trained with produces fluent, wrong output (the same failure class as a
//! wrong family), so:
//!
//! - only `tokenizer.ggml.model == "gpt2"` (byte-level BPE) is built. The
//!   SentencePiece models (`llama`, used by Llama 2, Mistral and Gemma) need
//!   a Unigram or byte-fallback BPE with a normalizer and are refused;
//! - only the pre-tokenizer names in [`PRE_TOKENIZER_REGEXES`] are built. The
//!   regexes are llama.cpp's, copied for those names only. An unknown
//!   `tokenizer.ggml.pre` is refused rather than guessed, because llama.cpp
//!   adds a new name for every model whose splitting differs.
//!
//! Encode equality against the HF `tokenizer.json` of the same model is the
//! gate that matters and is NOT run by the unit tests here (they check the
//! structure). See `tests/gguf_tokenizer_network.rs`.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::gguf_header::{GgufHeader, GgufValue};

/// llama.cpp `tokenizer.ggml.pre` name -> the split regex it applies before
/// byte-level BPE. A name appears only with its regex; several names share one.
const PRE_TOKENIZER_REGEXES: &[(&str, &str)] = &[
    // GPT-2's original splitter.
    (
        "gpt-2",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    ),
    // Llama 3's: digits grouped by up to three.
    (
        "llama-bpe",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
    // Qwen 2 / 2.5 / 3: digits split one at a time.
    (
        "qwen2",
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
];

/// ggml's `llama_token_type` values this builder distinguishes.
const TOKEN_TYPE_NORMAL: u64 = 1;
const TOKEN_TYPE_CONTROL: u64 = 3;
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufTokenizerError {
    /// A required `tokenizer.ggml.*` key is absent or has the wrong type.
    Missing(&'static str),
    /// A tokenizer model other than byte-level BPE.
    UnsupportedModel(String),
    /// A pre-tokenizer name with no reviewed regex in this port.
    UnsupportedPre(String),
    /// The file is malformed in a way that would change tokenization.
    Malformed(String),
}

impl std::fmt::Display for GgufTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(key) => write!(f, "GGUF metadata {key} is missing or malformed"),
            Self::UnsupportedModel(m) => write!(
                f,
                "GGUF tokenizer model {m:?} cannot be rebuilt as a tokenizer.json (only \
                 byte-level BPE, \"gpt2\", is); pass the HF tokenizer.json from the checkpoint \
                 this file was converted from"
            ),
            Self::UnsupportedPre(p) => write!(
                f,
                "GGUF pre-tokenizer {p:?} has no reviewed regex here (known: {}); guessing one \
                 would split text differently from training. Pass the HF tokenizer.json from \
                 the checkpoint this file was converted from",
                PRE_TOKENIZER_REGEXES
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Malformed(why) => write!(f, "GGUF tokenizer metadata is malformed: {why}"),
        }
    }
}

impl std::error::Error for GgufTokenizerError {}

/// The synthesized tokenizer files.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizerFiles {
    /// Contents of `tokenizer.json`.
    pub tokenizer_json: Value,
    /// Contents of `tokenizer_config.json`: the chat template and the
    /// bos/eos strings, which is what the runtime reads from it.
    pub tokenizer_config_json: Value,
}

fn string_array(
    meta: &BTreeMap<String, GgufValue>,
    key: &'static str,
) -> Result<Vec<String>, GgufTokenizerError> {
    let Some(GgufValue::Array(items)) = meta.get(key) else {
        return Err(GgufTokenizerError::Missing(key));
    };
    items
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or(GgufTokenizerError::Missing(key))
        })
        .collect()
}

fn token_id(meta: &BTreeMap<String, GgufValue>, key: &str, vocab: usize) -> Option<usize> {
    let id = usize::try_from(meta.get(key)?.as_u64()?).ok()?;
    (id < vocab).then_some(id)
}

fn flag(meta: &BTreeMap<String, GgufValue>, key: &str) -> Option<bool> {
    match meta.get(key)? {
        GgufValue::Bool(b) => Some(*b),
        _ => None,
    }
}

/// Builds `tokenizer.json` and `tokenizer_config.json` from a parsed header.
pub fn tokenizer_files_from_gguf(
    header: &GgufHeader,
) -> Result<GgufTokenizerFiles, GgufTokenizerError> {
    let meta = &header.metadata;
    let model = meta
        .get("tokenizer.ggml.model")
        .and_then(GgufValue::as_str)
        .ok_or(GgufTokenizerError::Missing("tokenizer.ggml.model"))?;
    if model != "gpt2" {
        return Err(GgufTokenizerError::UnsupportedModel(model.to_owned()));
    }
    // llama.cpp treats an absent `pre` as "default", which is not one regex
    // but a fallback it warns about. Refuse instead of picking one.
    let pre = meta
        .get("tokenizer.ggml.pre")
        .and_then(GgufValue::as_str)
        .ok_or(GgufTokenizerError::Missing("tokenizer.ggml.pre"))?;
    let regex = PRE_TOKENIZER_REGEXES
        .iter()
        .find(|(name, _)| *name == pre)
        .map(|(_, re)| *re)
        .ok_or_else(|| GgufTokenizerError::UnsupportedPre(pre.to_owned()))?;

    let tokens = string_array(meta, "tokenizer.ggml.tokens")?;
    let merges = string_array(meta, "tokenizer.ggml.merges")?;
    if tokens.is_empty() {
        return Err(GgufTokenizerError::Malformed("empty vocabulary".into()));
    }

    // Token types default to "normal" when the array is absent, as in ggml.
    let types: Vec<u64> = match meta.get("tokenizer.ggml.token_type") {
        Some(GgufValue::Array(items)) => {
            if items.len() != tokens.len() {
                return Err(GgufTokenizerError::Malformed(format!(
                    "{} token types for {} tokens",
                    items.len(),
                    tokens.len()
                )));
            }
            items
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or(GgufTokenizerError::Missing("tokenizer.ggml.token_type"))
                })
                .collect::<Result<_, _>>()?
        }
        Some(_) => return Err(GgufTokenizerError::Missing("tokenizer.ggml.token_type")),
        None => vec![TOKEN_TYPE_NORMAL; tokens.len()],
    };

    // A duplicate string would silently drop an id from the vocab map and
    // shift every id after the lost entry's position in nothing, but make that
    // id unreachable, so it is an error and not a last-write-wins.
    let mut vocab = Map::with_capacity(tokens.len());
    for (id, token) in tokens.iter().enumerate() {
        if vocab.insert(token.clone(), json!(id)).is_some() {
            return Err(GgufTokenizerError::Malformed(format!(
                "token {token:?} appears twice (second at id {id})"
            )));
        }
    }

    for (n, merge) in merges.iter().enumerate() {
        // HF reads "a b" as one pair split on the single space, so a merge
        // without exactly one separator would be misread or dropped.
        if merge.matches(' ').count() != 1 {
            return Err(GgufTokenizerError::Malformed(format!(
                "merge {n} {merge:?} is not exactly two pieces"
            )));
        }
    }

    let added_tokens: Vec<Value> = tokens
        .iter()
        .zip(&types)
        .enumerate()
        .filter(|(_, (_, ty))| **ty == TOKEN_TYPE_CONTROL || **ty == TOKEN_TYPE_USER_DEFINED)
        .map(|(id, (token, ty))| {
            json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": *ty == TOKEN_TYPE_CONTROL,
            })
        })
        .collect();

    let bos = token_id(meta, "tokenizer.ggml.bos_token_id", tokens.len());
    let eos = token_id(meta, "tokenizer.ggml.eos_token_id", tokens.len());
    // HF's llama3 tokenizer prepends BOS through a post-processor and Qwen
    // does not; the GGUF records which with `add_bos_token`.
    let post_processor = match (flag(meta, "tokenizer.ggml.add_bos_token"), bos) {
        (Some(true), Some(id)) => json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": tokens[id], "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": tokens[id], "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": tokens[id], "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                tokens[id].clone(): {
                    "id": tokens[id],
                    "ids": [id],
                    "tokens": [tokens[id]],
                }
            },
        }),
        _ => Value::Null,
    };

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": regex},
                    "behavior": "Isolated",
                    "invert": false,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false,
                },
            ],
        },
        "post_processor": post_processor,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        },
    });

    let mut config = Map::new();
    config.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    if let Some(template) = meta
        .get("tokenizer.chat_template")
        .and_then(GgufValue::as_str)
    {
        config.insert("chat_template".into(), json!(template));
    }
    if let Some(id) = bos {
        config.insert("bos_token".into(), json!(tokens[id]));
    }
    if let Some(id) = eos {
        config.insert("eos_token".into(), json!(tokens[id]));
    }
    if let Some(add) = flag(meta, "tokenizer.ggml.add_bos_token") {
        config.insert("add_bos_token".into(), json!(add));
    }

    Ok(GgufTokenizerFiles {
        tokenizer_json,
        tokenizer_config_json: Value::Object(config),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(entries: Vec<(&str, GgufValue)>) -> GgufHeader {
        GgufHeader {
            version: 3,
            metadata: entries
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
            tensors: BTreeMap::new(),
            alignment: 32,
            data_region_start: 0,
        }
    }

    fn strings(items: &[&str]) -> GgufValue {
        GgufValue::Array(
            items
                .iter()
                .map(|s| GgufValue::String((*s).into()))
                .collect(),
        )
    }

    fn base(pre: &str) -> Vec<(&'static str, GgufValue)> {
        vec![
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String(pre.into())),
            (
                "tokenizer.ggml.tokens",
                strings(&["a", "b", "ab", "<|bos|>", "<|x|>"]),
            ),
            ("tokenizer.ggml.merges", strings(&["a b"])),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(vec![
                    GgufValue::I32(1),
                    GgufValue::I32(1),
                    GgufValue::I32(1),
                    GgufValue::I32(3),
                    GgufValue::I32(4),
                ]),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(3)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(3)),
        ]
    }

    #[test]
    fn builds_vocab_merges_added_tokens_and_config() {
        let mut entries = base("qwen2");
        entries.push((
            "tokenizer.chat_template",
            GgufValue::String("{{ messages }}".into()),
        ));
        let files = tokenizer_files_from_gguf(&header(entries)).unwrap();
        let t = &files.tokenizer_json;
        assert_eq!(t["model"]["vocab"]["ab"], json!(2));
        assert_eq!(t["model"]["merges"], json!(["a b"]));
        // Control is special, user-defined is not.
        let added = t["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 2);
        assert_eq!(added[0]["content"], json!("<|bos|>"));
        assert_eq!(added[0]["special"], json!(true));
        assert_eq!(added[1]["special"], json!(false));
        assert_eq!(t["post_processor"], Value::Null);
        let c = &files.tokenizer_config_json;
        assert_eq!(c["chat_template"], json!("{{ messages }}"));
        assert_eq!(c["bos_token"], json!("<|bos|>"));
    }

    #[test]
    fn qwen2_and_llama_bpe_differ_only_in_digit_grouping() {
        let regex_of = |pre: &str| {
            let f = tokenizer_files_from_gguf(&header(base(pre))).unwrap();
            f.tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        let q = regex_of("qwen2");
        let l = regex_of("llama-bpe");
        assert!(q.contains(r"|\p{N}|") && !q.contains(r"\p{N}{1,3}"));
        assert!(l.contains(r"\p{N}{1,3}"));
        assert_eq!(
            q.replace(r"|\p{N}|", "|X|"),
            l.replace(r"|\p{N}{1,3}|", "|X|")
        );
    }

    #[test]
    fn add_bos_token_emits_a_template_post_processor() {
        let mut entries = base("llama-bpe");
        entries.push(("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)));
        let f = tokenizer_files_from_gguf(&header(entries)).unwrap();
        let pp = &f.tokenizer_json["post_processor"];
        assert_eq!(pp["type"], json!("TemplateProcessing"));
        assert_eq!(pp["special_tokens"]["<|bos|>"]["ids"], json!([3]));
    }

    #[test]
    fn refuses_sentencepiece_unknown_pre_and_missing_pre() {
        let mut spm = base("qwen2");
        spm[0].1 = GgufValue::String("llama".into());
        assert!(matches!(
            tokenizer_files_from_gguf(&header(spm)),
            Err(GgufTokenizerError::UnsupportedModel(_))
        ));
        assert!(matches!(
            tokenizer_files_from_gguf(&header(base("some-new-pre"))),
            Err(GgufTokenizerError::UnsupportedPre(_))
        ));
        let mut no_pre = base("qwen2");
        no_pre.retain(|(k, _)| *k != "tokenizer.ggml.pre");
        assert_eq!(
            tokenizer_files_from_gguf(&header(no_pre)),
            Err(GgufTokenizerError::Missing("tokenizer.ggml.pre"))
        );
    }

    #[test]
    fn refuses_duplicate_tokens_and_ragged_merges_and_type_length() {
        let mut dup = base("qwen2");
        dup[2].1 = strings(&["a", "a", "ab", "<|bos|>", "<|x|>"]);
        assert!(matches!(
            tokenizer_files_from_gguf(&header(dup)),
            Err(GgufTokenizerError::Malformed(_))
        ));
        let mut merge = base("qwen2");
        merge[3].1 = strings(&["a b c"]);
        assert!(matches!(
            tokenizer_files_from_gguf(&header(merge)),
            Err(GgufTokenizerError::Malformed(_))
        ));
        let mut types = base("qwen2");
        types[4].1 = GgufValue::Array(vec![GgufValue::I32(1)]);
        assert!(matches!(
            tokenizer_files_from_gguf(&header(types)),
            Err(GgufTokenizerError::Malformed(_))
        ));
    }
}
