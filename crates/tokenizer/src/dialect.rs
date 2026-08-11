//! Tokenizer wrapper and chat-dialect resolution. Ported from the loading
//! and special-token-resolution parts of `Tokenization/Tokenizer.swift`.
//!
//! The dialect resolved here decides SPECIAL TOKEN IDS and the stop set. It
//! does NOT decide chat framing: that is a property of the checkpoint, which
//! ships its own Jinja template (loaded below, in either of HF's two
//! conventions) and is preferred by `MfTokenizer::apply_chat_template`. See
//! `chat_template.rs` for why the two came apart.

use std::collections::BTreeSet;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::TokenizerError;

/// Chat framing dialect, resolved from the loaded tokenizer's special
/// tokens. `Deepseek` is detected by the presence of the `<|User|>` special
/// token (DeepSeek-V4), `ChatMl` by `<|im_end|>` (Qwen-style ChatML);
/// everything else uses the Gemma 4 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatDialect {
    Gemma,
    ChatMl,
    Deepseek,
    /// Mistral / Mixtral: `[INST] user [/INST] assistant</s>`, framed in
    /// PLAIN TEXT rather than special tokens (ROADMAP Phase M2). The only
    /// special tokens involved are `<s>` and `</s>`.
    Mistral,
}

/// Sentinel for token roles a dialect frames as plain text rather than a
/// single special token (DeepSeek's tool markers). Never a valid token ID.
pub const NO_SUCH_TOKEN_ID: i32 = -1;

const DEEPSEEK_USER_MARK: &str = "<\u{FF5C}User\u{FF5C}>";
const DEEPSEEK_ASSISTANT_MARK: &str = "<\u{FF5C}Assistant\u{FF5C}>";
pub(crate) const DEEPSEEK_BOS_MARK: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const DEEPSEEK_EOS_MARK: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
const IM_END_MARK: &str = "<|im_end|>";
/// Gemma's end-of-turn marker, used as the POSITIVE test for that dialect
/// now that it is no longer the fallback for everything unrecognized.
const GEMMA_TURN_MARK: &str = "<turn|>";
/// Mistral / Mixtral frame turns as PLAIN TEXT (`[INST] ... [/INST]`), so
/// there is no instruction marker in the special-token table to key on and
/// the only reliable witness is the sentence pair.
const MISTRAL_BOS_MARK: &str = "<s>";
const MISTRAL_EOS_MARK: &str = "</s>";
const IM_START_MARK: &str = "<|im_start|>";

pub struct MfTokenizer {
    pub dialect: ChatDialect,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub end_of_turn_id: i32,
    pub tool_call_start_id: i32,
    pub tool_call_end_id: i32,
    pub tool_response_id: i32,
    pub tool_response_end_id: i32,
    pub channel_start_id: i32,
    pub channel_end_id: i32,
    pub think_start_id: Option<i32>,
    pub think_end_id: Option<i32>,
    pub stop_token_ids: BTreeSet<i32>,
    pub vocab_size: usize,
    /// BOS actually prepended by `encode(_, add_bos: true)`; `None` for
    /// dialects that never use a BOS prefix (ChatML).
    bos_prefix_id: Option<i32>,
    tokenizer: Tokenizer,
    /// The checkpoint's own chat-template source, from whichever of HF's
    /// two conventions the directory shipped it in. Rendered by
    /// [`crate::jinja_chat_template`] for BOTH tool chat and plain text
    /// chat; `None` means fall back to the per-dialect renderer.
    pub(crate) chat_template_source: Option<String>,
}

struct Resolved {
    bos_id: i32,
    bos_prefix_id: Option<i32>,
    eos_id: i32,
    pad_id: i32,
    end_of_turn_id: i32,
    tool_call_start_id: i32,
    tool_call_end_id: i32,
    tool_response_id: i32,
    tool_response_end_id: i32,
    channel_start_id: i32,
    channel_end_id: i32,
    think_start_id: Option<i32>,
    think_end_id: Option<i32>,
    stop_token_ids: BTreeSet<i32>,
    vocab_size: usize,
}

impl MfTokenizer {
    /// Load a `tokenizer.json` (and, if present alongside it,
    /// `tokenizer_config.json` for the Gemma dialect's BOS/EOS token names,
    /// plus `generation_config.json` for the checkpoint's full EOS set)
    /// from a directory.
    pub fn load_from_dir(dir: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;
        let config_path = dir.join("tokenizer_config.json");
        let config: TokenizerConfig = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        // Two conventions, and which one a checkpoint uses is a matter of
        // its converter's vintage rather than of its family: HF moved the
        // template out of `tokenizer_config.json` into a standalone
        // `chat_template.jinja` partway through, and llama.cpp's GGUF
        // converter still writes the older embedded key. Reading only the
        // file made every pre-move checkpoint look template-less and fall
        // through to the dialect renderer, which is how TinyLlama-1.1B-Chat
        // (Zephyr framing) came to be fed Mistral's `[INST]`.
        let chat_template_source = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| config.chat_template_source());
        // `generation_config.json` is the authority for the checkpoint's
        // FULL end-of-sequence set: `tokenizer_config.json` only ever
        // carries one `eos_token` string, so a multi-stop checkpoint
        // looks single-stop without this. The real Gemma 4 26B-A4B
        // declares `eos_token_id: [1, 106, 50]` while its
        // tokenizer_config says only `<eos>`.
        let extra_eos = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<GenerationConfig>(&s).ok())
            .map(|g| g.eos_ids())
            .unwrap_or_default();
        Self::new(tokenizer, &config, chat_template_source, &extra_eos)
    }

    fn new(
        tokenizer: Tokenizer,
        config: &TokenizerConfig,
        chat_template_source: Option<String>,
        extra_eos: &[i32],
    ) -> Result<Self, TokenizerError> {
        // Gemma is still the FALLBACK, deliberately: every fixture and
        // install predating ROADMAP Phase M2 lands there, and moving the
        // default would change what an unrecognized tokenizer does. Mistral
        // is therefore tested for POSITIVELY, and only after Gemma's own
        // marker has been ruled out -- `</s>` is far too common a token to
        // decide a dialect on its own.
        let dialect = if special_token_id(&tokenizer, DEEPSEEK_USER_MARK).is_some() {
            ChatDialect::Deepseek
        } else if special_token_id(&tokenizer, IM_END_MARK).is_some() {
            ChatDialect::ChatMl
        } else if special_token_id(&tokenizer, GEMMA_TURN_MARK).is_none()
            && special_token_id(&tokenizer, MISTRAL_BOS_MARK).is_some()
            && special_token_id(&tokenizer, MISTRAL_EOS_MARK).is_some()
        {
            ChatDialect::Mistral
        } else {
            ChatDialect::Gemma
        };
        let mut resolved = match dialect {
            ChatDialect::Gemma => resolve_gemma(&tokenizer, config)?,
            ChatDialect::ChatMl => resolve_chatml(&tokenizer)?,
            ChatDialect::Deepseek => resolve_deepseek(&tokenizer)?,
            ChatDialect::Mistral => resolve_mistral(&tokenizer)?,
        };
        resolved
            .stop_token_ids
            .extend(extra_eos.iter().copied().filter(|&id| id >= 0));
        Ok(Self {
            dialect,
            bos_id: resolved.bos_id,
            bos_prefix_id: resolved.bos_prefix_id,
            eos_id: resolved.eos_id,
            pad_id: resolved.pad_id,
            end_of_turn_id: resolved.end_of_turn_id,
            tool_call_start_id: resolved.tool_call_start_id,
            tool_call_end_id: resolved.tool_call_end_id,
            tool_response_id: resolved.tool_response_id,
            tool_response_end_id: resolved.tool_response_end_id,
            channel_start_id: resolved.channel_start_id,
            channel_end_id: resolved.channel_end_id,
            think_start_id: resolved.think_start_id,
            think_end_id: resolved.think_end_id,
            stop_token_ids: resolved.stop_token_ids,
            vocab_size: resolved.vocab_size,
            tokenizer,
            chat_template_source,
        })
    }

    /// Encode UTF-8 text to token IDs. `add_bos = true` prepends the
    /// dialect's BOS token, when it has one.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|&id| id as i32).collect::<Vec<_>>())
            .unwrap_or_default();
        match (add_bos, self.bos_prefix_id) {
            (true, Some(bos)) => {
                let mut out = Vec::with_capacity(ids.len() + 1);
                out.push(bos);
                out.extend(ids);
                out
            }
            _ => ids,
        }
    }

    /// Decode token IDs to text. `skip_special_tokens` strips BOS/EOS/turn
    /// markers from the output.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&ids_u32, skip_special_tokens)
            .unwrap_or_default()
    }

    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.tokenizer.token_to_id(token).map(|id| id as i32)
    }

    pub fn id_to_token(&self, id: i32) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

#[derive(Default, serde::Deserialize)]
struct TokenizerConfig {
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// The pre-`chat_template.jinja` convention. HF allowed either one
    /// template string or a NAMED LIST of them (the `default` /
    /// `tool_use` split some checkpoints ship), so both shapes parse.
    #[serde(default)]
    chat_template: Option<EmbeddedChatTemplate>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum EmbeddedChatTemplate {
    One(String),
    Named(Vec<NamedChatTemplate>),
}

#[derive(serde::Deserialize)]
struct NamedChatTemplate {
    name: String,
    template: String,
}

impl TokenizerConfig {
    /// The embedded template's source, if any. From a named list this takes
    /// the entry called `default`, falling back to the first: the other
    /// names are tool-use variants, and the tool path here renders through
    /// [`crate::jinja_chat_template`] with `tools` in the context rather
    /// than by selecting a different template.
    fn chat_template_source(&self) -> Option<String> {
        match self.chat_template.as_ref()? {
            EmbeddedChatTemplate::One(source) => Some(source.clone()),
            EmbeddedChatTemplate::Named(entries) => entries
                .iter()
                .find(|entry| entry.name == "default")
                .or_else(|| entries.first())
                .map(|entry| entry.template.clone()),
        }
    }
}

/// The slice of `generation_config.json` this loader reads. HF writes
/// `eos_token_id` as either one integer or an array of them.
#[derive(Default, serde::Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<EosTokenIds>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    One(i64),
    Many(Vec<i64>),
}

impl GenerationConfig {
    fn eos_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenIds::One(id)) => vec![*id as i32],
            Some(EosTokenIds::Many(ids)) => ids.iter().map(|&id| id as i32).collect(),
            None => Vec::new(),
        }
    }
}

/// Resolves a token string to its ID, rejecting the unk-token fallback some
/// tokenizers substitute for out-of-vocabulary strings.
fn special_token_id(tokenizer: &Tokenizer, token: &str) -> Option<i32> {
    let id = tokenizer.token_to_id(token)?;
    if tokenizer.id_to_token(id).as_deref() == Some(token) {
        Some(id as i32)
    } else {
        None
    }
}

fn required_id(tokenizer: &Tokenizer, token: &str) -> Result<i32, TokenizerError> {
    special_token_id(tokenizer, token)
        .ok_or_else(|| TokenizerError::MissingSpecialToken(token.to_string()))
}

fn resolve_gemma(
    tokenizer: &Tokenizer,
    config: &TokenizerConfig,
) -> Result<Resolved, TokenizerError> {
    let bos_token = config
        .bos_token
        .as_deref()
        .ok_or_else(|| TokenizerError::MissingSpecialToken("<bos>".to_string()))?;
    let bos = required_id(tokenizer, bos_token)?;
    let eos_token = config
        .eos_token
        .as_deref()
        .ok_or_else(|| TokenizerError::MissingSpecialToken("<eos>".to_string()))?;
    let eos = required_id(tokenizer, eos_token)?;
    let pad = required_id(tokenizer, "<pad>")?;
    let eot = required_id(tokenizer, "<turn|>")?;
    let tool_response = required_id(tokenizer, "<|tool_response>")?;
    let tool_call_start = required_id(tokenizer, "<|tool_call>")?;
    let tool_call_end = required_id(tokenizer, "<tool_call|>")?;
    let tool_response_end = required_id(tokenizer, "<tool_response|>")?;
    let channel_start = required_id(tokenizer, "<|channel>")?;
    let channel_end = required_id(tokenizer, "<channel|>")?;
    Ok(Resolved {
        bos_id: bos,
        bos_prefix_id: Some(bos),
        eos_id: eos,
        pad_id: pad,
        end_of_turn_id: eot,
        tool_call_start_id: tool_call_start,
        tool_call_end_id: tool_call_end,
        tool_response_id: tool_response,
        tool_response_end_id: tool_response_end,
        channel_start_id: channel_start,
        channel_end_id: channel_end,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [eos, eot, tool_response].into_iter().collect(),
        vocab_size: 262_144,
    })
}

/// Mistral / Mixtral (ROADMAP Phase M2).
///
/// Every tool and channel id is [`NO_SUCH_TOKEN_ID`], the same sentinel the
/// DeepSeek resolver uses for roles its checkpoint frames as text: Mixtral
/// 8x7B-Instruct v0.1 has exactly three special tokens (`<unk>`, `<s>`,
/// `</s>`) and no tool-calling or thinking markup at all. Inventing ids here
/// would make `StructuredDecoder` look for markup the model cannot emit.
///
/// End of turn IS `</s>`, not a separate marker: the model closes an
/// assistant turn with the sentence end.
fn resolve_mistral(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, MISTRAL_BOS_MARK)?;
    let eos = required_id(tokenizer, MISTRAL_EOS_MARK)?;
    Ok(Resolved {
        bos_id: bos,
        bos_prefix_id: Some(bos),
        eos_id: eos,
        pad_id: eos,
        end_of_turn_id: eos,
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        channel_start_id: NO_SUCH_TOKEN_ID,
        channel_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [eos].into_iter().collect(),
        vocab_size: 32_000,
    })
}

fn resolve_chatml(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let _im_start = required_id(tokenizer, IM_START_MARK)?;
    let im_end = required_id(tokenizer, IM_END_MARK)?;
    let end_of_text = required_id(tokenizer, "<|endoftext|>")?;
    let tool_call_start = required_id(tokenizer, "<tool_call>")?;
    let tool_call_end = required_id(tokenizer, "</tool_call>")?;
    let tool_response = required_id(tokenizer, "<tool_response>")?;
    let tool_response_end = required_id(tokenizer, "</tool_response>")?;
    let think_start = required_id(tokenizer, "<think>")?;
    let think_end = required_id(tokenizer, "</think>")?;
    Ok(Resolved {
        bos_id: end_of_text,
        bos_prefix_id: None,
        eos_id: end_of_text,
        pad_id: end_of_text,
        end_of_turn_id: im_end,
        tool_call_start_id: tool_call_start,
        tool_call_end_id: tool_call_end,
        tool_response_id: tool_response,
        tool_response_end_id: tool_response_end,
        channel_start_id: think_start,
        channel_end_id: think_end,
        think_start_id: Some(think_start),
        think_end_id: Some(think_end),
        stop_token_ids: [im_end, end_of_text].into_iter().collect(),
        // The model's padded embedding/lm_head row count, not the
        // tokenizer's actual vocab; logits buffers use this.
        vocab_size: 248_320,
    })
}

fn resolve_deepseek(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, DEEPSEEK_BOS_MARK)?;
    let eos = required_id(tokenizer, DEEPSEEK_EOS_MARK)?;
    let _user = required_id(tokenizer, DEEPSEEK_USER_MARK)?;
    let _assistant = required_id(tokenizer, DEEPSEEK_ASSISTANT_MARK)?;
    let think_start = required_id(tokenizer, "<think>")?;
    let think_end = required_id(tokenizer, "</think>")?;
    Ok(Resolved {
        bos_id: bos,
        bos_prefix_id: Some(bos),
        eos_id: eos,
        pad_id: eos,
        end_of_turn_id: eos,
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        channel_start_id: think_start,
        channel_end_id: think_end,
        think_start_id: Some(think_start),
        think_end_id: Some(think_end),
        stop_token_ids: [eos].into_iter().collect(),
        vocab_size: 129_280,
    })
}

#[cfg(test)]
mod embedded_template_tests {
    use super::*;

    fn parse(json: &str) -> TokenizerConfig {
        serde_json::from_str(json).expect("config parses")
    }

    #[test]
    fn a_plain_string_template_is_read() {
        let config = parse(r#"{"chat_template": "<|user|>\n{{ x }}"}"#);
        assert_eq!(
            config.chat_template_source().as_deref(),
            Some("<|user|>\n{{ x }}")
        );
    }

    /// The other shape the pre-`chat_template.jinja` convention allowed. A
    /// loader that handles only the string form does not FAIL on this one,
    /// it silently reports no template and falls back to the dialect
    /// renderer, which is the failure mode this whole change exists to
    /// remove.
    #[test]
    fn a_named_list_resolves_to_the_default_entry() {
        let config = parse(
            r#"{"chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": "PLAIN"}
            ]}"#,
        );
        assert_eq!(config.chat_template_source().as_deref(), Some("PLAIN"));
    }

    #[test]
    fn a_named_list_without_a_default_takes_the_first_entry() {
        let config = parse(r#"{"chat_template": [{"name": "rag", "template": "R"}]}"#);
        assert_eq!(config.chat_template_source().as_deref(), Some("R"));
    }

    #[test]
    fn a_config_without_the_key_reports_no_template() {
        assert!(parse(r#"{"eos_token": "</s>"}"#)
            .chat_template_source()
            .is_none());
    }
}
