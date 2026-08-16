//! Special token detection and per-dialect token ID resolution.

use std::collections::BTreeSet;
use tokenizers::Tokenizer;

use super::config::TokenizerConfig;
use super::{ChatDialect, NO_SUCH_TOKEN_ID};
use crate::error::TokenizerError;

pub(crate) const DEEPSEEK_USER_MARK: &str = "<\u{FF5C}User\u{FF5C}>";
pub(crate) const DEEPSEEK_ASSISTANT_MARK: &str = "<\u{FF5C}Assistant\u{FF5C}>";
pub(crate) const DEEPSEEK_BOS_MARK: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const DEEPSEEK_EOS_MARK: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
pub(crate) const IM_END_MARK: &str = "<|im_end|>";
/// Gemma's end-of-turn marker, used as the POSITIVE test for that dialect
/// now that it is no longer the fallback for everything unrecognized.
pub(crate) const GEMMA_TURN_MARK: &str = "<turn|>";
/// Mistral / Mixtral frame turns as PLAIN TEXT (`[INST] ... [/INST]`), so
/// there is no instruction marker in the special-token table to key on and
/// the only reliable witness is the sentence pair.
pub(crate) const MISTRAL_BOS_MARK: &str = "<s>";
pub(crate) const MISTRAL_EOS_MARK: &str = "</s>";
pub(crate) const IM_START_MARK: &str = "<|im_start|>";
/// Harmony's turn frame (ROADMAP M5). `<|start|>` is the witness because it
/// opens every turn and appears in no other family's table; `<|return|>` is
/// the end-of-turn marker and `<|call|>` the tool-call one, and BOTH end a
/// generation. All three names were read off `openai/gpt-oss-20b`'s
/// `tokenizer_config.json` rather than recalled -- their ids there are
/// 200006 / 200002 / 200012, and they are looked up BY NAME here for the
/// reason crate Gotcha 2 gives.
pub(crate) const HARMONY_START_MARK: &str = "<|start|>";
pub(crate) const HARMONY_MESSAGE_MARK: &str = "<|message|>";
pub(crate) const HARMONY_END_MARK: &str = "<|end|>";
pub(crate) const HARMONY_RETURN_MARK: &str = "<|return|>";
pub(crate) const HARMONY_CALL_MARK: &str = "<|call|>";
pub(crate) const HARMONY_CHANNEL_MARK: &str = "<|channel|>";
pub(crate) const HARMONY_BOS_MARK: &str = "<|startoftext|>";
pub(crate) const HARMONY_PAD_MARK: &str = "<|endoftext|>";

pub(crate) struct Resolved {
    pub(crate) bos_id: i32,
    pub(crate) bos_prefix_id: Option<i32>,
    pub(crate) eos_id: i32,
    pub(crate) pad_id: i32,
    pub(crate) end_of_turn_id: i32,
    pub(crate) tool_call_start_id: i32,
    pub(crate) tool_call_end_id: i32,
    pub(crate) tool_response_id: i32,
    pub(crate) tool_response_end_id: i32,
    pub(crate) channel_start_id: i32,
    pub(crate) channel_end_id: i32,
    pub(crate) message_start_id: i32,
    pub(crate) message_end_id: i32,
    pub(crate) think_start_id: Option<i32>,
    pub(crate) think_end_id: Option<i32>,
    pub(crate) stop_token_ids: BTreeSet<i32>,
    pub(crate) vocab_size: usize,
}

/// Resolves a token string to its ID, rejecting the unk-token fallback some
/// tokenizers substitute for out-of-vocabulary strings.
pub(crate) fn special_token_id(tokenizer: &Tokenizer, token: &str) -> Option<i32> {
    let id = tokenizer.token_to_id(token)?;
    if tokenizer.id_to_token(id).as_deref() == Some(token) {
        Some(id as i32)
    } else {
        None
    }
}

pub(crate) fn required_id(tokenizer: &Tokenizer, token: &str) -> Result<i32, TokenizerError> {
    special_token_id(tokenizer, token)
        .ok_or_else(|| TokenizerError::MissingSpecialToken(token.to_string()))
}

pub(crate) fn detect_dialect(tokenizer: &Tokenizer) -> ChatDialect {
    // Gemma is still the FALLBACK, deliberately: every fixture and
    // install predating ROADMAP Phase M2 lands there, and moving the
    // default would change what an unrecognized tokenizer does. Mistral
    // is therefore tested for POSITIVELY, and only after Gemma's own
    // marker has been ruled out -- `</s>` is far too common a token to
    // decide a dialect on its own.
    if special_token_id(tokenizer, DEEPSEEK_USER_MARK).is_some() {
        ChatDialect::Deepseek
    } else if special_token_id(tokenizer, HARMONY_START_MARK).is_some()
        && special_token_id(tokenizer, HARMONY_MESSAGE_MARK).is_some()
    {
        // TESTED BEFORE ChatML, and the order is load-bearing rather than
        // arbitrary: Harmony's table carries neither `<|im_end|>` nor
        // `<|im_start|>` today, so the two probes are disjoint on the real
        // checkpoints -- but `<|start|>` and `<|message|>` are a far more
        // specific pair than a single `<|im_end|>`, and Gotcha 41's lesson
        // is that a dialect probe stops being injective the moment a
        // second checkpoint arrives. Two markers rather than one for the
        // same reason.
        ChatDialect::Harmony
    } else if special_token_id(tokenizer, IM_END_MARK).is_some() {
        ChatDialect::ChatMl
    } else if special_token_id(tokenizer, GEMMA_TURN_MARK).is_none()
        && special_token_id(tokenizer, MISTRAL_BOS_MARK).is_some()
        && special_token_id(tokenizer, MISTRAL_EOS_MARK).is_some()
    {
        ChatDialect::Mistral
    } else {
        ChatDialect::Gemma
    }
}

pub(crate) fn resolve_dialect(
    dialect: ChatDialect,
    tokenizer: &Tokenizer,
    config: &TokenizerConfig,
) -> Result<Resolved, TokenizerError> {
    match dialect {
        ChatDialect::Gemma => resolve_gemma(tokenizer, config),
        ChatDialect::ChatMl => resolve_chatml(tokenizer),
        ChatDialect::Deepseek => resolve_deepseek(tokenizer),
        ChatDialect::Mistral => resolve_mistral(tokenizer),
        ChatDialect::Harmony => resolve_harmony(tokenizer),
    }
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
        // Gemma's channels BRACKET, so the pair above says everything and
        // there is no header to end.
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [eos, eot, tool_response].into_iter().collect(),
        vocab_size: 262_144,
    })
}

/// `gpt-oss`'s Harmony format (ROADMAP M5).
///
/// **THE STOP SET IS THE POINT OF THIS FUNCTION, and it has THREE members
/// where every other dialect here has one or two.** Harmony ends an assistant
/// turn with `<|return|>` when it has answered and with `<|call|>` when it is
/// invoking a tool, and `<|endoftext|>` is the base end-of-sequence; the real
/// checkpoint's `generation_config.json` declares exactly those three
/// (`[200002, 199999, 200012]`, read rather than recalled). Missing `<|call|>`
/// would not error -- the model would emit a tool call and then keep
/// generating past it, which reads as a rambling model rather than a stop-set
/// bug.
///
/// `<|end|>` is deliberately NOT a stop: it closes the SYSTEM and USER turns
/// inside a rendered prompt, so stopping on it would end generation at the
/// first token of a well-formed reply.
///
/// END OF TURN IS `<|return|>` AND BOS IS `<|startoftext|>`, which is the one
/// place Harmony's naming misleads: `<|endoftext|>` is the PAD token here, not
/// the turn end, inverting the convention every other dialect in this file
/// follows.
///
/// Tool ids stay [`NO_SUCH_TOKEN_ID`] even though Harmony HAS tool calling,
/// because it frames a call as a channel plus a recipient in the message
/// header rather than as a bracketing token pair, which is not what
/// `StructuredDecoder`'s start/end contract describes. Wiring tool calls for
/// this family is its own item; claiming ids here would make the decoder hunt
/// for markup in the wrong shape.
fn resolve_harmony(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, HARMONY_BOS_MARK)?;
    let pad = required_id(tokenizer, HARMONY_PAD_MARK)?;
    let ret = required_id(tokenizer, HARMONY_RETURN_MARK)?;
    let call = required_id(tokenizer, HARMONY_CALL_MARK)?;
    // Resolved so a table missing them fails at LOAD rather than at the first
    // rendered prompt, and so the detection probe above cannot pass on a
    // checkpoint whose frame is only half present.
    let end = required_id(tokenizer, HARMONY_END_MARK)?;
    let message = required_id(tokenizer, HARMONY_MESSAGE_MARK)?;
    let channel = required_id(tokenizer, HARMONY_CHANNEL_MARK)?;
    Ok(Resolved {
        bos_id: bos,
        // The checkpoint's template emits `<|start|>` itself, so the encoder
        // must not prepend a BOS on top of it -- the same arrangement Gemma
        // and ChatML have and the one Mixtral's fallback renderer does not
        // (AGENTS.md Gotcha 41's closing note).
        bos_prefix_id: None,
        eos_id: ret,
        pad_id: pad,
        end_of_turn_id: ret,
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        channel_start_id: channel,
        // `<|channel|>` HAS no closing counterpart: it opens a header that
        // `<|message|>` ends, and `<|end|>` then closes the body. Leaving
        // this sentinel is what stops the Gemma-shaped bracketing arm from
        // being reachable for this dialect.
        channel_end_id: NO_SUCH_TOKEN_ID,
        message_start_id: message,
        message_end_id: end,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [ret, call, pad].into_iter().collect(),
        // The model's PADDED lm_head row count, which is not the tokenizer's
        // vocabulary (AGENTS.md Gotcha 37). Every caller with a
        // `RealForwardRunner` in hand must read `vocab_size()` off that
        // instead; this value serves the scripted paths, which have no model
        // to ask.
        vocab_size: 201_088,
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
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
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
        // The thought channel above already brackets; there is no header.
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
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
        // The thought channel above already brackets; there is no header.
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: Some(think_start),
        think_end_id: Some(think_end),
        stop_token_ids: [eos].into_iter().collect(),
        vocab_size: 129_280,
    })
}
