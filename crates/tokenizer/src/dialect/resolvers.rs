use tokenizers::Tokenizer;

use super::config::TokenizerConfig;
use super::resolve::{
    required_id, Resolved, DEEPSEEK_ASSISTANT_MARK, DEEPSEEK_BOS_MARK, DEEPSEEK_EOS_MARK,
    DEEPSEEK_USER_MARK, HARMONY_BOS_MARK, HARMONY_CALL_MARK, HARMONY_CHANNEL_MARK,
    HARMONY_END_MARK, HARMONY_MESSAGE_MARK, HARMONY_PAD_MARK, HARMONY_RETURN_MARK, IM_END_MARK,
    IM_START_MARK, LLAMA3_BOS_MARK, LLAMA3_END_HEADER_MARK, LLAMA3_EOS_MARK, LLAMA3_EOT_MARK,
    LLAMA3_START_HEADER_MARK, MISTRAL_BOS_MARK, MISTRAL_EOS_MARK, MUSE_BOS_MARK, MUSE_EOM_MARK,
    MUSE_EOS_MARK, MUSE_EOT_MARK, MUSE_MESSAGE_MARK, MUSE_PAD_MARK, MUSE_START_MARK,
    SPARK_BOS_MARK, SPARK_BOT_MARK, SPARK_EOS_MARK, SPARK_USER_MARK,
};
use super::NO_SUCH_TOKEN_ID;
use crate::error::TokenizerError;

pub(crate) fn resolve_gemma(
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
        // Gemma hands over to the caller by emitting the tool-RESPONSE
        // marker, which is why its stop set carries one at all.
        tool_call_stop_id: tool_response,
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
/// The tool MARKUP ids stay [`NO_SUCH_TOKEN_ID`] even though Harmony has tool
/// calling and this port now decodes it, because Harmony frames a call as a
/// channel plus a recipient in the message HEADER rather than as a bracketing
/// token pair, which is not what `StructuredDecoder`'s start/end contract
/// describes. Claiming ids here would make the decoder hunt for markup in the
/// wrong shape; its Harmony arm reads the header instead.
/// `tool_call_stop_id` is the one exception and is a different kind of fact --
/// see the comment on it below.
pub(crate) fn resolve_harmony(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
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
        // `<|call|>` is the ONE member of this dialect's three-token stop set
        // that means "invoking a tool" rather than "the turn is over", and
        // `run_raw_completion`'s ladder has no other way to tell: it is
        // neither `end_of_turn_id` (that is `<|return|>`) nor
        // `tool_response_id` (Harmony frames a tool RESULT as a whole message
        // rather than as a marker), so without this a call reaches a client
        // as `finish_reason: "stop"`.
        tool_call_stop_id: call,
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
pub(crate) fn resolve_mistral(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
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
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
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

/// Meta's Llama-3 family (base and Instruct).
///
/// **SHARES `<|begin_of_text|>` AND `<|end_of_text|>` WITH `muse_glimmer` AND
/// NOTHING ELSE**, exactly the trap crate Gotcha 6 already names for Harmony
/// and Muse Glimmer -- this dialect's frame markers
/// (`<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>`) are resolved
/// too, even though only `end_of_turn_id` is stored, so a checkpoint whose
/// table has the BOS/EOS pair but not the header pair fails at LOAD rather
/// than at the first rendered prompt.
///
/// No tool-calling or thinking markup: `meta-llama/Meta-Llama-3-8B-Instruct`
/// (the checkpoint this was built and probed against, per
/// `docs/OBLITERATION.md`'s Open section) has none in its table. Every such
/// id is therefore [`NO_SUCH_TOKEN_ID`], the same sentinel Mistral's resolver
/// uses for the same reason. A 3.1-family checkpoint's `<|eom_id|>` (message
/// handoff, mirroring Muse Glimmer's) and `<|python_tag|>` (built-in tool
/// call) would need their own arm; nothing here has been measured against
/// one.
///
/// End of turn is `<|eot_id|>`, not `<|end_of_text|>`: the checkpoint closes
/// every assistant turn with the former and reserves the latter for the raw
/// end of a document.
pub(crate) fn resolve_llama3(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, LLAMA3_BOS_MARK)?;
    let eos = required_id(tokenizer, LLAMA3_EOS_MARK)?;
    let eot = required_id(tokenizer, LLAMA3_EOT_MARK)?;
    let _start_header = required_id(tokenizer, LLAMA3_START_HEADER_MARK)?;
    let _end_header = required_id(tokenizer, LLAMA3_END_HEADER_MARK)?;
    Ok(Resolved {
        bos_id: bos,
        // The fallback renderer emits `<|begin_of_text|>` itself, matching
        // the real checkpoint's own template (`{{- bos_token }}` at the top),
        // so the encoder must not prepend a second one -- Harmony's and Muse
        // Glimmer's reason, on a family with no markup living anywhere else
        // in its table at all.
        bos_prefix_id: None,
        eos_id: eos,
        // No dedicated `<pad>` in this table -- the missing one is what sent
        // a Llama-3 checkpoint into `resolve_gemma` and a load failure before
        // this dialect existed (AGENTS.md Gotcha 9 / crate Gotcha 9).
        // `<|end_of_text|>` is the same reuse ChatML and Mistral already make
        // of their own EOS for the same reason.
        pad_id: eos,
        end_of_turn_id: eot,
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
        channel_start_id: NO_SUCH_TOKEN_ID,
        channel_end_id: NO_SUCH_TOKEN_ID,
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [eos, eot].into_iter().collect(),
        // Llama-3's padded vocabulary: 128,000 base BPE merges plus 256
        // reserved special-token slots. Public and read off the real
        // checkpoint's `config.json` rather than recalled.
        vocab_size: 128_256,
    })
}

/// `muse_glimmer`'s ids and stop set.
///
/// **THE RENDERER IS THE CHECKPOINT'S OWN TEMPLATE, exactly as Harmony's
/// is**, and for the same reason: this one carries an image/video content
/// macro and an `<atem:function_calls>` tool DSL, so a hand-rolled renderer
/// would be a second implementation of something the checkpoint ships, and
/// Gotcha 1 makes the checkpoint's version win anyway. This variant exists
/// for what a dialect IS evidence for -- the ids and the STOP SET.
///
/// **THE STOP SET IS `<|end_of_text|>` AND `<|eot|>`, which is what
/// `generation_config.json` declares (`eos_token_id: [200001, 200008]`) and
/// NOT what `tokenizer_config.json`'s single `eos_token` says.** Resolving
/// only the latter costs the end-of-turn stop, so a well-formed reply runs
/// to the token budget -- a rambling model rather than a stop-set bug, which
/// is Harmony's Gotcha 2 failure mode on a different token.
/// `<|eom|>` is deliberately NOT a stop: it ends a message that is handing
/// off (a tool call), so stopping on it would truncate a turn the model
/// intends to continue. It is carried as `message_end_id` instead.
pub(crate) fn resolve_muse_glimmer(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, MUSE_BOS_MARK)?;
    let eos = required_id(tokenizer, MUSE_EOS_MARK)?;
    let eot = required_id(tokenizer, MUSE_EOT_MARK)?;
    let pad = required_id(tokenizer, MUSE_PAD_MARK)?;
    // Resolved so a half-present frame fails at LOAD rather than at the
    // first rendered prompt, which is what `resolve_harmony` does and what
    // makes the detection probe above safe to keep to two markers.
    let start = required_id(tokenizer, MUSE_START_MARK)?;
    let message = required_id(tokenizer, MUSE_MESSAGE_MARK)?;
    let eom = required_id(tokenizer, MUSE_EOM_MARK)?;
    Ok(Resolved {
        bos_id: bos,
        // The checkpoint's template emits `<|begin_of_text|>` itself, so the
        // encoder must not prepend a second one (AGENTS.md Gotcha 41's
        // closing note).
        bos_prefix_id: None,
        eos_id: eos,
        pad_id: pad,
        end_of_turn_id: eot,
        // Tool calls are `<atem:function_calls>` PLAIN TEXT, not special
        // tokens, so every marker id is the sentinel and no tool-call
        // parsing is wired for this dialect yet.
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
        // `<|start|>` OPENS A MESSAGE HEADER, which is the same job Harmony
        // gives `<|channel|>`, so it goes in the same field rather than in a
        // new one. `channel_end_id` stays the sentinel for Harmony's reason:
        // this frame is a header/body triple, and a non-sentinel end would
        // make the Gemma-shaped bracketing arm reachable.
        channel_start_id: start,
        channel_end_id: NO_SUCH_TOKEN_ID,
        message_start_id: message,
        message_end_id: eom,
        think_start_id: None,
        think_end_id: None,
        stop_token_ids: [eos, eot].into_iter().collect(),
        // The model's PADDED lm_head row count, which is not the tokenizer's
        // vocabulary (AGENTS.md Gotcha 37). Every caller with a
        // `RealForwardRunner` in hand must read `vocab_size()` off that
        // instead; this value serves the scripted paths, which have no model
        // to ask.
        vocab_size: 202_048,
    })
}

pub(crate) fn resolve_chatml(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
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
        // ChatML closes a tool call with `</tool_call>` and then ends the
        // turn with `<|im_end|>`, so no stop token of its own means "tool".
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
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

pub(crate) fn resolve_deepseek(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
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
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
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

pub(crate) fn resolve_spark(tokenizer: &Tokenizer) -> Result<Resolved, TokenizerError> {
    let bos = required_id(tokenizer, SPARK_BOS_MARK)?;
    let eos = required_id(tokenizer, SPARK_EOS_MARK)?;
    let _user = required_id(tokenizer, SPARK_USER_MARK)?;
    let _bot = required_id(tokenizer, SPARK_BOT_MARK)?;
    // REQUIRED, not optional: the generation prompt FORCES the think frame
    // open (`<think>` with thinking on, `</think>` with it off), so a table
    // without the pair could not split reasoning from content at all -- and
    // `prompt_opens_thought` would leave the decoder in Visible while the
    // model wrote scratchpad.
    let think_start = required_id(tokenizer, "<think>")?;
    let think_end = required_id(tokenizer, "</think>")?;
    Ok(Resolved {
        bos_id: bos,
        bos_prefix_id: Some(bos),
        eos_id: eos,
        pad_id: eos,
        end_of_turn_id: eos,
        // THE TOOL IDS EXIST in the table (`<tool_call>` 130977 and friends)
        // and are deliberately NOT resolved: the `<tool_call>` / `<arg_key>` /
        // `<arg_value>` DSL has no parser here (DEVIATIONS.md), and carrying
        // ids nothing parses would suggest support `tool_call_support` does
        // not answer. The markup flows as ordinary content instead.
        tool_call_start_id: NO_SUCH_TOKEN_ID,
        tool_call_end_id: NO_SUCH_TOKEN_ID,
        tool_response_id: NO_SUCH_TOKEN_ID,
        tool_response_end_id: NO_SUCH_TOKEN_ID,
        tool_call_stop_id: NO_SUCH_TOKEN_ID,
        // The think pair IS the channel bracket, exactly as ChatML and
        // DeepSeek resolve it: the structured decoder's reasoning split keys
        // on these.
        channel_start_id: think_start,
        channel_end_id: think_end,
        // The thought channel above already brackets; there is no header.
        message_start_id: NO_SUCH_TOKEN_ID,
        message_end_id: NO_SUCH_TOKEN_ID,
        think_start_id: Some(think_start),
        think_end_id: Some(think_end),
        // One EOS closes every turn kind (user, bot, tool); the template
        // writes it explicitly, and `generation_config.json` adds nothing.
        stop_token_ids: [eos].into_iter().collect(),
        // The padded embedding row count, which for this checkpoint equals
        // the tokenizer's vocab (131,072, `token_embd.weight`'s own dims) --
        // a coincidence of THIS checkpoint, not a property of the dialect.
        // Callers holding a `RealForwardRunner` read `vocab_size()` off it.
        vocab_size: 131_072,
    })
}
