//! One turn: render, decode, stream, and report.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use runtime::{GenerationConfig, RawDecodeProgress, RawDecodeResult, StopReason};
use tokenizer::{
    ChatDialect, Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role,
    StructuredAssistantDecoder, StructuredAssistantEvent,
};

use crate::session::{Engine, Session};
use crate::wire::{GenerateOptions, GenerateResult, WireMessage};

/// Prefill progress event kind.
pub const TS_EVENT_PREFILL: i32 = 0;
/// Content text delta event kind.
pub const TS_EVENT_CONTENT: i32 = 1;
/// Reasoning / thought text delta event kind.
pub const TS_EVENT_REASONING: i32 = 2;

/// Splits a token's text into the answer and the reasoning that preceded it.
///
/// Ported from `crates/cli/src/generate.rs::ChannelSplit`, and the two
/// conditions are the same because the reasons are:
///
/// - `gpt-oss` ALWAYS needs a decoder. Harmony puts the model's reasoning in
///   an `analysis` channel before its answer whatever the caller asked for,
///   so without this the reasoning and the frame markup arrive as the reply.
/// - ChatML and Gemma need one only when a reasoning LEVEL was asked for.
///   Their thought channels are unreachable otherwise, so building a decoder
///   unconditionally would route shipped families through a state machine
///   with nothing to do.
///
/// Skipping it is not cosmetic. Measured on the real Gemma 4 install the
/// first time a level was asked for, the reply began with a bare `thought`
/// (the channel LABEL, as prose), then the model's scratch work, then its
/// answer, all as one run of content.
struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    /// `prompt_ids` is the rendered generation prompt: a ChatML template opens
    /// the `<think>` frame itself when thinking is on, and the decoder cannot
    /// tell without being shown (`StructuredAssistantDecoder::new`).
    fn new(tokenizer: &'a MfTokenizer, reasoning: ReasoningEffort, prompt_ids: &[i32]) -> Self {
        let wanted = matches!(
            tokenizer.dialect,
            ChatDialect::Harmony | ChatDialect::MuseGlimmer
        ) || (reasoning != ReasoningEffort::Off
            && matches!(tokenizer.dialect, ChatDialect::ChatMl | ChatDialect::Gemma));
        Self {
            decoder: wanted.then(|| {
                // An EMPTY tool allowlist. This binding has no way to run a
                // tool, so a Harmony `commentary` body stays reasoning
                // rather than being parsed as a call the caller cannot
                // service. Tools belong to the server surface.
                StructuredAssistantDecoder::new(tokenizer, HashSet::new(), String::new, prompt_ids)
            }),
        }
    }

    /// One token's `(answer, reasoning)`. Either may be empty.
    fn push(&mut self, id: i32, text: &str) -> (String, String) {
        let Some(decoder) = self.decoder.as_mut() else {
            return (text.to_string(), String::new());
        };
        let (mut answer, mut reasoning) = (String::new(), String::new());
        match decoder.consume(id, text) {
            Ok(events) => {
                for event in events {
                    match event {
                        StructuredAssistantEvent::Content(c) => answer.push_str(&c),
                        StructuredAssistantEvent::Reasoning(r) => reasoning.push_str(&r),
                        // Unreachable with an empty allowlist, and dropping
                        // beats inventing a rendering for it.
                        StructuredAssistantEvent::ToolCall(_) => {}
                    }
                }
            }
            // Losing a caller's output to a decoder error would be the worst
            // outcome available: pass the text through.
            Err(_) => answer.push_str(text),
        }
        (answer, reasoning)
    }
}

fn role_of(name: &str) -> Result<Role, String> {
    match name {
        "system" => Ok(Role::System),
        "developer" => Ok(Role::Developer),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("unknown role {other:?}")),
    }
}

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfTurn => "endOfTurn",
        StopReason::ToolCalls => "toolCalls",
        StopReason::Eos => "eos",
        StopReason::StopString => "stopString",
        StopReason::MaxTokens => "maxTokens",
        StopReason::Cancelled => "cancelled",
    }
}

/// Renders the conversation and encodes it.
///
/// The checkpoint's own chat template, never a raw concatenation: an
/// instruction-tuned model fed unrendered text babbles, and that is missing
/// markup rather than a decode bug.
fn render(
    tokenizer: &MfTokenizer,
    messages: &[WireMessage],
    reasoning: ReasoningEffort,
) -> Result<(Vec<i32>, Option<String>), String> {
    let mut note = None;
    if reasoning != ReasoningEffort::Off {
        match tokenizer.reasoning_support() {
            // Thinking turns ON but the LEVEL is dropped. Reported rather
            // than silently honoured: a caller who asked for `low` and got
            // the checkpoint's own default should know which they got.
            ReasoningSupport::ToggleOnly => {
                note = Some(
                    "this checkpoint's chat template has no reasoning-effort knob, so \
                     thinking is ON but no level is set"
                        .to_string(),
                )
            }
            // No template at all: a level is REFUSED rather than dropped,
            // because there is nothing to express it with and silence is the
            // failure mode worth avoiding.
            ReasoningSupport::None => {
                return Err(
                    "this checkpoint ships no chat template, so a reasoning level cannot \
                     be expressed; send reasoning \"off\""
                        .to_string(),
                )
            }
            ReasoningSupport::Level => {}
        }
    }
    let decoded: Vec<Message> = messages
        .iter()
        .map(|m| Ok(Message::new(role_of(&m.role)?, m.content.clone())))
        .collect::<Result<_, String>>()?;
    let rendered = tokenizer
        .apply_chat_template_with_reasoning(&decoded, reasoning)
        .map_err(|e| format!("chat template: {e}"))?;
    // `false`: the template emits its own BOS as text, so a BOS prefix here
    // doubles it -- invisible on a toy fixture, and it degrades output on a
    // real install.
    Ok((tokenizer.encode(&rendered, false), note))
}

/// The per-turn generation budget: never more than asked, never more than
/// the context leaves room for.
///
/// Clamped rather than refused, so a long conversation generates into
/// whatever room is left instead of failing outright. The window comes from
/// the SESSION, because under `auto` the caller named no number and the KV
/// cache was allocated at the resolved one.
fn clamp_max_new(session: &Session, asked: u32, prompt_len: usize) -> Result<u32, String> {
    if prompt_len >= session.max_context as usize {
        return Err(format!(
            "context overflow: the rendered prompt is {prompt_len} tokens and the \
             resolved window is {}",
            session.max_context
        ));
    }
    Ok(asked.min(session.max_context - prompt_len as u32))
}

/// The block THIS TURN drafts by: the session's, kept only when the turn is
/// deterministic.
///
/// **THE SECOND HALF OF THE SPECULATION DECISION IS PER TURN, and it is the
/// one thing this binding cannot settle at open.** Acceptance is
/// `argmax(target) == proposal`, which is exact speculative decoding at
/// temperature 0 and biased at any other, so a sampled turn takes the
/// sequential loop however the session resolved. That is the SERVER's shape
/// rather than the CLI's, and for the server's reason: the CLI has one
/// shaping per process and can settle both halves at open, while here the
/// temperature belongs to the request.
///
/// It falls back SILENTLY rather than failing. This binding's own sampling
/// default is T=0.2, so sampled is the NORMAL case -- a GUI would be told
/// off once per turn for a setting it never sent, and refusing would turn a
/// valid request into an error. What the caller is owed instead is the
/// session-level answer, and that is in `sessionInfo.speculation`, said
/// once.
///
/// A free function taking both inputs rather than a method, so the decision
/// is pinnable without a session and without a 14 GB install.
fn turn_block(session_block: Option<usize>, deterministic: bool) -> Option<usize> {
    session_block.filter(|_| deterministic)
}

/// Evaluates the prompt token count without running generation.
pub(crate) fn count_tokens(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
) -> Result<u32, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;
    let (prompt_ids, _note) = render(&session.tokenizer, messages, reasoning)?;
    Ok(prompt_ids.len() as u32)
}

/// Evaluates the token count of a raw text string using the session tokenizer.
pub(crate) fn count_text_tokens(session: &Session, text: &str, add_special: bool) -> u32 {
    session.tokenizer.encode(text, add_special).len() as u32
}

/// Fits a conversation transcript into a context budget using `turbospark-window-fit`.
pub(crate) fn fit_window(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
    max_tokens: u32,
) -> Result<crate::wire::WindowFitOutcome, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;

    let bound = if max_tokens == 0 {
        session.max_context as u64
    } else {
        max_tokens as u64
    };

    let has_leading_instruction = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);

    let measure = |slice: &[WireMessage]| -> u64 {
        match render(&session.tokenizer, slice, reasoning) {
            Ok((ids, _)) => ids.len() as u64,
            Err(_) => u64::MAX,
        }
    };

    let outcome =
        window_fit::fit_conversation_window(messages, has_leading_instruction, bound, measure);

    Ok(crate::wire::WindowFitOutcome {
        retained: outcome.retained_turns().to_vec(),
        measured_tokens: outcome.measured_length(),
        removed_turn_count: outcome.removed_turn_count(),
        has_room_for_generation: outcome.has_room_for_generation(),
    })
}

/// Formats a conversation transcript into raw prompt text using the session's
/// chat template and reasoning effort setting.
pub(crate) fn render_prompt(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
) -> Result<String, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;
    let decoded: Vec<Message> = messages
        .iter()
        .map(|m| Ok(Message::new(role_of(&m.role)?, m.content.clone())))
        .collect::<Result<_, String>>()?;
    session
        .tokenizer
        .apply_chat_template_with_reasoning(&decoded, reasoning)
        .map_err(|e| format!("chat template: {e}"))
}

/// Encodes raw text into token IDs using the session tokenizer.
pub(crate) fn tokenize(session: &Session, text: &str, add_special: bool) -> Vec<i32> {
    session.tokenizer.encode(text, add_special)
}

/// Decodes token IDs into a text string using the session tokenizer.
pub(crate) fn detokenize(session: &Session, token_ids: &[i32], skip_special: bool) -> String {
    session.tokenizer.decode(token_ids, skip_special)
}

/// Runs one turn, calling `emit(kind, text, a, b)` per event.
pub(crate) fn generate(
    session: &Session,
    messages: &[WireMessage],
    options: &GenerateOptions,
    mut emit: impl FnMut(i32, &str, u32, u32),
) -> Result<GenerateResult, String> {
    let reasoning = ReasoningEffort::parse(&options.reasoning)
        .ok_or_else(|| format!("unknown reasoning level {:?}", options.reasoning))?;
    let (prompt_ids, _note) = render(&session.tokenizer, messages, reasoning)?;
    let max_new = clamp_max_new(session, options.max_new_tokens, prompt_ids.len())?;

    let config = GenerationConfig {
        shaping: session.shaping(options)?,
        max_new_tokens: max_new,
        stop_strings: options.stop.clone(),
        extra_stop_tokens: options
            .stop_tokens
            .iter()
            .map(|&t| t as foundation::TokenId)
            .collect(),
        rate: session.rate,
    };

    // Armed BEFORE the lock is taken, so a Stop pressed between turns cannot
    // cancel the next one before it has produced a token.
    let cancel = session.arm();
    let mut engine = session
        .engine
        .lock()
        .map_err(|_| "the session is poisoned by an earlier panic".to_string())?;

    let mut content = String::new();
    let mut reasoning_text = String::new();
    let mut split = ChannelSplit::new(&session.tokenizer, reasoning, &prompt_ids);
    let total = prompt_ids.len() as u32;

    let mut on_progress = |event: RawDecodeProgress| {
        let (id, text) = match event {
            RawDecodeProgress::Token { id, delta, .. } => (id, delta),
            // A withheld tail has no token id behind it, which is what the
            // tokenizer's "no such token" sentinel means.
            RawDecodeProgress::Tail(tail) => (tokenizer::NO_SUCH_TOKEN_ID, tail),
            RawDecodeProgress::Prefill { done, .. } => {
                emit(TS_EVENT_PREFILL, "", done as u32, total);
                return;
            }
        };
        // NO EARLY RETURN ON EMPTY TEXT. Special tokens decode to the empty
        // string, so every Harmony frame token arrives as `(id, "")` -- skip
        // them and the state machine never sees a single `<|channel|>` and
        // the whole turn reads as one run of content.
        let (answer, reason) = split.push(id, &text);
        if !reason.is_empty() {
            reasoning_text.push_str(&reason);
            emit(TS_EVENT_REASONING, &reason, 0, 0);
        }
        if !answer.is_empty() {
            content.push_str(&answer);
            emit(TS_EVENT_CONTENT, &answer, 0, 0);
        }
    };

    let predicate = || cancel.load(Ordering::Acquire);
    // Resolved BEFORE the mutable borrow below. Reading it inside the call's
    // argument list is E0502: `runner.as_mut()` is already a mutable borrow
    // by the time the width argument is evaluated. Same shape as the
    // re-binding rule the family flows carry.
    //
    // The width is the MODEL's, never the tokenizer's:
    // `MfTokenizer::vocab_size` is a per-DIALECT constant standing in for a
    // padded head width, which is right only while one model uses each
    // dialect (ChatML's row is Qwen 3.6's 248,320 and Qwen3-30B-A3B is also
    // ChatML at 151,936).
    let vocab_size = match &*engine {
        #[cfg(target_os = "macos")]
        Engine::Real(runner) => runner.vocab_size(),
        Engine::Scripted(_) => session.info.vocab_size,
    };
    let block = turn_block(session.speculation_block, config.shaping.is_deterministic());
    let result: RawDecodeResult = match (&mut *engine, block) {
        // The CONCRETE runner, which is why this sits inside the match:
        // `SpeculativeProducer` has an associated type and cannot be
        // reached through a `&mut dyn LogitProducer`.
        #[cfg(target_os = "macos")]
        (Engine::Real(runner), Some(block)) => runtime::run_raw_completion_speculative_cancellable(
            runner.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            block,
            &predicate,
            &mut on_progress,
        ),
        #[cfg(target_os = "macos")]
        (Engine::Real(runner), None) => runtime::run_raw_completion_cancellable(
            runner.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            &predicate,
            &mut on_progress,
        ),
        // A scripted producer implements no drafter, so `block` is always
        // `None` here and the arm is a plain wildcard rather than a case
        // this could get wrong.
        (Engine::Scripted(producer), _) => runtime::run_raw_completion_cancellable(
            producer.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            &predicate,
            &mut on_progress,
        ),
    }
    .map_err(|e| e.to_string())?;

    Ok(GenerateResult {
        prompt_tokens: result.prompt_tokens,
        new_tokens: result.new_tokens,
        prefill_seconds: result.prefill_seconds,
        decode_seconds: result.decode_seconds,
        stop_reason: stop_reason_name(result.reason).to_string(),
        // Null rather than zero when nothing was decoded, so a caller cannot
        // plot a rate that was never measured.
        tokens_per_second: (result.decode_seconds > 0.0 && result.new_tokens > 0)
            .then(|| result.new_tokens as f64 / result.decode_seconds),
        content,
        reasoning: reasoning_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-turn half of the speculation decision, in all four states.
    ///
    /// Cheap enough to be exhaustive, and worth being: three of the four
    /// cells are "decode sequentially" and the one that is not is the only
    /// path in this crate that reaches a batched verify.
    #[test]
    fn a_turn_speculates_only_when_the_session_can_and_the_turn_is_greedy() {
        assert_eq!(turn_block(Some(2), true), Some(2));
        // Sampled: the session's block is DISCARDED rather than honoured,
        // because acceptance is exact only at temperature 0.
        assert_eq!(turn_block(Some(2), false), None);
        // No drafter: greedy does not conjure one.
        assert_eq!(turn_block(None, true), None);
        assert_eq!(turn_block(None, false), None);
    }
}
