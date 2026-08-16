//! Tokenizer wrapper and chat-dialect resolution. Ported from the loading
//! and special-token-resolution parts of `Tokenization/Tokenizer.swift`.
//!
//! The dialect resolved here decides SPECIAL TOKEN IDS and the stop set. It
//! does NOT decide chat framing: that is a property of the checkpoint, which
//! ships its own Jinja template (loaded below, in either of HF's two
//! conventions) and is preferred by `MfTokenizer::apply_chat_template`. See
//! `chat_template` for why the two came apart.

mod config;
mod resolve;

use std::collections::BTreeSet;
use std::path::Path;

use tokenizers::Tokenizer;

use self::config::{GenerationConfig, TokenizerConfig};
pub(crate) use self::resolve::{
    DEEPSEEK_BOS_MARK, DEEPSEEK_EOS_MARK, HARMONY_END_MARK, HARMONY_MESSAGE_MARK,
    HARMONY_START_MARK, MUSE_EOT_MARK, MUSE_MESSAGE_MARK, MUSE_START_MARK,
};
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
    /// `gpt-oss`'s Harmony format (ROADMAP M5):
    /// `<|start|>role<|message|>content<|end|>`, with an explicit
    /// `<|channel|>` for the analysis / final split.
    ///
    /// **THE RENDERER IS THE CHECKPOINT'S OWN 17 KB TEMPLATE, NOT A CASE IN
    /// `chat_template`.** Harmony carries a system preamble, a reasoning
    /// effort knob and a tool namespace written in TypeScript syntax; a
    /// hand-rolled renderer for it would be a large second implementation of
    /// something the checkpoint already ships, and Gotcha 1 makes the
    /// checkpoint's version win anyway. This variant therefore exists for
    /// what a dialect IS evidence for -- the ids and the STOP SET.
    Harmony,
    /// `muse_glimmer`'s frame:
    /// `<|start|>role<|message|>content<|eot|>`.
    ///
    /// **SHARES `<|start|>` AND `<|message|>` WITH HARMONY AND NOTHING
    /// ELSE**, which is why `detect_dialect` tests Harmony first and why
    /// Harmony's probe requires `<|channel|>`. This family has no channels,
    /// no `<|return|>` and no `<|call|>`; it closes a turn with `<|eot|>` and
    /// a handoff with `<|eom|>`, and frames tool calls as
    /// `<atem:function_calls>` PLAIN TEXT rather than as special tokens.
    ///
    /// Like [`ChatDialect::Harmony`] it has NO fallback renderer: the
    /// checkpoint's own template carries an image/video content macro and
    /// the ATEM tool DSL, so this variant exists for what a dialect IS
    /// evidence for -- the ids and the STOP SET.
    MuseGlimmer,
}

/// Sentinel for token roles a dialect frames as plain text rather than a
/// single special token (DeepSeek's tool markers). Never a valid token ID.
pub const NO_SUCH_TOKEN_ID: i32 = -1;

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
    /// The one member of this dialect's STOP SET that means "the model is
    /// invoking a tool" rather than "the turn is over", or
    /// [`NO_SUCH_TOKEN_ID`] where the dialect has none.
    ///
    /// Read by `run_raw_completion`'s stop ladder, and by nothing else: it
    /// answers a question about how a generation ENDED, which is separate from
    /// the markup ids the structured decoder reads. Two dialects have one and
    /// they are not the same shape of token -- Gemma's is the tool-RESPONSE
    /// marker it hands over with, Harmony's is `<|call|>` -- which is exactly
    /// why the ladder cannot derive it from `tool_response_id` alone.
    pub tool_call_stop_id: i32,
    pub channel_start_id: i32,
    pub channel_end_id: i32,
    /// The token that ends a channel HEADER and opens its body, and the one
    /// that closes that body. Harmony's frame is a header/body pair rather
    /// than the bracketing token pair `channel_start_id`/`channel_end_id`
    /// describes, so those two cannot express it: `<|channel|>` opens a
    /// header and `<|end|>` closes a whole message, with `<|message|>`
    /// between them. Both are [`NO_SUCH_TOKEN_ID`] for every other dialect
    /// here, whose channels either bracket (Gemma) or are plain text
    /// (DeepSeek). Read by [`crate::structured_decoder`]"'"s Harmony arm.
    pub message_start_id: i32,
    pub message_end_id: i32,
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
        let dialect = resolve::detect_dialect(&tokenizer);
        let mut resolved = resolve::resolve_dialect(dialect, &tokenizer, config)?;
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
            tool_call_stop_id: resolved.tool_call_stop_id,
            channel_start_id: resolved.channel_start_id,
            channel_end_id: resolved.channel_end_id,
            message_start_id: resolved.message_start_id,
            message_end_id: resolved.message_end_id,
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
