//! One turn: render, decode, stream, and report.

mod prompt;
#[cfg(test)]
mod tests;
mod vision;

/// Prefill progress event kind.
pub const TS_EVENT_PREFILL: i32 = 0;
/// Content text delta event kind.
pub const TS_EVENT_CONTENT: i32 = 1;
/// Reasoning / thought text delta event kind.
pub const TS_EVENT_REASONING: i32 = 2;
/// Parsed tool call event kind. `text` is the call as a JSON object
/// (`{"id", "name", "arguments"}`), `a` is the call's zero-based index
/// within the turn. Fires only when the caller offered the tool by name,
/// which no `GenerateOptions` field does yet -- the kind is wired so the
/// pipeline does not need a second pass when the binding grows a way to
/// offer tools.
pub const TS_EVENT_TOOL: i32 = 3;
/// Terminal event kind, fired once per successful `ts_generate` just
/// before it returns. `text` is the stop reason spelled as it appears in
/// `result_json`'s `stopReason` (`endOfTurn`, `toolCalls`, `eos`,
/// `stopString`, `maxTokens`, `cancelled`), `a` is the new token count and
/// `b` the prompt token count. Not fired when the run FAILS: a non-zero
/// return plus `ts_last_error` remains the only error signal.
pub const TS_EVENT_FINISH: i32 = 4;

pub(crate) use prompt::{
    count_text_tokens, count_tokens, detokenize, fit_window, render_prompt, tokenize,
};
pub(crate) use vision::SCRIPTED_HAS_NO_TOWER;

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use runtime::{
    GenerationConfig, RawDecodeProgress, RawDecodeResult, StopReason, TurnEvent, TurnSplitter,
};
use tokenizer::{ParsedToolCall, ReasoningEffort};

use crate::session::{Engine, Session};
use crate::wire::{GenerateOptions, GenerateResult, WireMessage};

use prompt::{collect_image_parts, render};
use vision::{attach_images, clear_vision};

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

/// The wire JSON of one parsed tool call, matching `result_json`'s
/// `toolCalls` rows exactly.
///
/// **`arguments` IS AN OBJECT AND NEVER A STRING**, so a host reads
/// `call.arguments.city` rather than parsing a second time. It is converted
/// through the tokenizer's own [`tokenizer::JsonValue::to_serde_json`], which
/// is where `arguments_json` was rendered FROM one frame earlier
/// (`ParsedToolCall` carries both). Re-parsing that string instead would be
/// the same value by a longer route, and would put a `.expect` on
/// model-generated data inside the one crate where a panic escaping into C is
/// a named hazard (AGENTS.md Gotcha 9): unreachable today, and unreachable
/// only for as long as every producer of that string keeps rendering
/// something `serde_json` will take back.
fn tool_call_json(call: &ParsedToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments.to_serde_json(),
    })
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
pub(crate) fn turn_block(session_block: Option<usize>, deterministic: bool) -> Option<usize> {
    session_block.filter(|_| deterministic)
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
    let (rendered_ids, _note) = render(&session.tokenizer, messages, reasoning)?;
    // Shape-checked before the lock, decoded inside it (see `attach_images`).
    let image_parts = collect_image_parts(messages)?;

    // Armed BEFORE the lock is taken, so a Stop pressed between turns cannot
    // cancel the next one before it has produced a token.
    let cancel = session.arm();
    let mut engine = session
        .engine
        .lock()
        .map_err(|_| "the session is poisoned by an earlier panic".to_string())?;

    // **THE PROMPT IS NOT FINAL UNTIL THE SPLICE HAS RUN.** The template
    // renders ONE `<|image_pad|>` per image whatever its size, and
    // `splice_and_walk` expands each marker to that image's merged-token
    // count -- so the budget clamp and every length below have to be taken
    // AFTER this, or a page's worth of positions is missing from both.
    let prompt_ids = attach_images(&mut engine, session, &rendered_ids, &image_parts)?;
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

    let mut content = String::new();
    let mut reasoning_text = String::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let tools = HashSet::new();
    let mut split = TurnSplitter::new(
        &session.tokenizer,
        &tools,
        reasoning,
        String::new,
        &prompt_ids,
    );

    let mut on_progress = |event: RawDecodeProgress| {
        split.feed(event, &mut |turn| match turn {
            TurnEvent::Prefill { done, total } => {
                emit(TS_EVENT_PREFILL, "", done as u32, total as u32)
            }
            // Fires only when the caller offered the tool by name, which no
            // `GenerateOptions` field does yet; see `TS_EVENT_TOOL`.
            TurnEvent::ToolCall(call) => {
                let value = tool_call_json(&call);
                let text = value.to_string();
                tool_calls.push(value);
                emit(TS_EVENT_TOOL, &text, tool_calls.len() as u32 - 1, 0);
            }
            TurnEvent::Reasoning(reason) => {
                reasoning_text.push_str(&reason);
                emit(TS_EVENT_REASONING, &reason, 0, 0);
            }
            TurnEvent::Content(answer) => {
                content.push_str(&answer);
                emit(TS_EVENT_CONTENT, &answer, 0, 0);
            }
        });
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
    let decoded: Result<RawDecodeResult, _> = match (&mut *engine, block) {
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
        // Chunked prefill wins over the sequential loop whenever this
        // install's family can serve it -- the SAME predicate
        // `crates/cli`'s default `--prefill-chunk` wiring and
        // `crates/server`'s `RealChatModel::run_completion` both check
        // (`crates/server/CLAUDE.md` Gotcha 19), so a caller who never
        // touched a chunk-size knob still gets it. **Gated on
        // `image_parts.is_empty()` too, matching the server's own
        // discipline (Gotcha 21): a vision-carrying prompt takes the qwen
        // dense flow's driver, which refuses an open image prompt BY NAME
        // (`crates/runtime/CLAUDE.md` Gotcha 14) rather than silently
        // mishandling it, so composing the two here would turn an image
        // turn that used to succeed into one that fails on a family whose
        // general chunked-prefill support has nothing to do with whether
        // THIS call happens to carry a picture.** Decode's per-token
        // progress callback is unaffected either way, since only the
        // PREFILL portion routes differently.
        #[cfg(target_os = "macos")]
        (Engine::Real(runner), None)
            if image_parts.is_empty() && runner.supports_chunked_prefill() =>
        {
            runtime::run_raw_completion_chunked_cancellable(
                runner.as_mut(),
                &session.tokenizer,
                &prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                foundation::DEFAULT_CHUNK_SIZE as usize,
                &predicate,
                &mut on_progress,
            )
        }
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
    };

    // **THE CALLER CONSUMES THE INJECTION MAP, and it is cleared on the
    // FAILURE path too.** `reset()` deliberately no longer clears it
    // (`crates/runtime/CLAUDE.md` Gotcha 13: clearing at the generation loop's entry
    // landed on the map for the very prompt about to be prefilled, so every image
    // run prefilled placeholder embeddings and read perfectly fluently off a
    // picture the model had not been shown). So it ends HERE, before the `?`,
    // or the next turn on this session inherits this turn's spans.
    if !image_parts.is_empty() {
        clear_vision(&mut engine);
    }
    let result: RawDecodeResult = decoded.map_err(|e| e.to_string())?;

    // THE TERMINAL EVENT, after every content/reasoning/tool event and just
    // before the call that ran the turn returns, so a callback-driven host
    // sees the turn end on the same channel that carried it. A FAILED run
    // emits nothing here: the error contract stays non-zero return plus
    // `ts_last_error`.
    emit(
        TS_EVENT_FINISH,
        stop_reason_name(result.reason),
        result.new_tokens as u32,
        result.prompt_tokens as u32,
    );

    Ok(GenerateResult {
        prompt_tokens: result.prompt_tokens,
        new_tokens: result.new_tokens,
        reused_prefix_tokens: result.reused_prefix_tokens,
        prefill_seconds: result.prefill_seconds,
        decode_seconds: result.decode_seconds,
        stop_reason: stop_reason_name(result.reason).to_string(),
        // Null rather than zero when nothing was decoded, so a caller cannot
        // plot a rate that was never measured.
        tokens_per_second: (result.decode_seconds > 0.0 && result.new_tokens > 0)
            .then(|| result.new_tokens as f64 / result.decode_seconds),
        content,
        reasoning: reasoning_text,
        tool_calls,
        peak_memory_pressure: format!("{:?}", result.peak_memory_pressure).to_lowercase(),
    })
}
