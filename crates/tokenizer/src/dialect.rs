//! Tokenizer wrapper and chat-dialect resolution. Ported from the loading
//! and special-token-resolution parts of `Tokenization/Tokenizer.swift`.
//!
//! Chat-template Jinja rendering (the generic tool-call path routed through
//! the checkpoint's `chat_template.jinja` for Gemma/ChatML) is out of scope:
//! this port implements the text-only chat templates and the DeepSeek native
//! (non-Jinja) tool chat, which cover the CLI's raw-completion and
//! instruction-chat paths. See `chat_template.rs`.

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
}

/// Sentinel for token roles a dialect frames as plain text rather than a
/// single special token (DeepSeek's tool markers). Never a valid token ID.
pub const NO_SUCH_TOKEN_ID: i32 = -1;

const DEEPSEEK_USER_MARK: &str = "<\u{FF5C}User\u{FF5C}>";
const DEEPSEEK_ASSISTANT_MARK: &str = "<\u{FF5C}Assistant\u{FF5C}>";
pub(crate) const DEEPSEEK_BOS_MARK: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const DEEPSEEK_EOS_MARK: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
const IM_END_MARK: &str = "<|im_end|>";
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
    /// The installed `chat_template.jinja` source, if the tokenizer
    /// directory shipped one. Used by [`crate::jinja_chat_template`] for
    /// the generic tool-chat path.
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
    /// `tokenizer_config.json` for the Gemma dialect's BOS/EOS token names)
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
        let chat_template_source = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();
        Self::new(tokenizer, &config, chat_template_source)
    }

    fn new(
        tokenizer: Tokenizer,
        config: &TokenizerConfig,
        chat_template_source: Option<String>,
    ) -> Result<Self, TokenizerError> {
        let dialect = if special_token_id(&tokenizer, DEEPSEEK_USER_MARK).is_some() {
            ChatDialect::Deepseek
        } else if special_token_id(&tokenizer, IM_END_MARK).is_some() {
            ChatDialect::ChatMl
        } else {
            ChatDialect::Gemma
        };
        let resolved = match dialect {
            ChatDialect::Gemma => resolve_gemma(&tokenizer, config)?,
            ChatDialect::ChatMl => resolve_chatml(&tokenizer)?,
            ChatDialect::Deepseek => resolve_deepseek(&tokenizer)?,
        };
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
