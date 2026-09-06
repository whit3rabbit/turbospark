//! Generation execution and streaming loop logic.

use std::collections::HashSet;

use runtime::{
    CancelFlag, GenerationConfig, RawDecodeResult, RuntimeError, TurnEvent, TurnSplitter,
};
use tokenizer::{ParsedToolCall, ReasoningEffort};

use super::plan::AppState;
use crate::cancel::Cancel;

/// A generation that never started, as distinct from one that failed.
pub(crate) enum GenError {
    Runtime(RuntimeError),
    /// The blocking task itself died (panic or cancellation).
    Join(String),
}

/// One decoded unit of assistant output.
pub(crate) enum Piece {
    Text(String),
    /// Reasoning the model produced before its answer. Only `gpt-oss`'s
    /// Harmony frame separates one.
    Reasoning(String),
    Tool(ParsedToolCall),
}

/// Everything one generation produced: the answer, the reasoning that preceded
/// it, and every tool call parsed out of it.
pub(crate) struct Generated {
    pub text: String,
    pub reasoning: String,
    pub calls: Vec<ParsedToolCall>,
    pub decode: RawDecodeResult,
}

/// Runs a generation to completion.
pub(crate) async fn run_full(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    images: Option<crate::vision::RequestImages>,
    tools: HashSet<String>,
    effort: ReasoningEffort,
    cancel: Cancel,
) -> Result<Generated, GenError> {
    let joined = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        // Built HERE, on the blocking thread the generation itself runs on:
        // `runtime::CancelFlag` is a borrowed `&dyn Fn`, not `Send`, so it
        // cannot be captured by THIS closure's own `move` -- only the `Arc`
        // it reads from can cross that boundary.
        let flag = crate::cancel::as_cancel_flag(&cancel);
        let result = stream_blocking(
            &model,
            &prompt_ids,
            &config,
            images.as_ref(),
            &tools,
            effort,
            &flag,
            &mut |piece| match piece {
                Piece::Text(delta) => text.push_str(&delta),
                Piece::Reasoning(delta) => reasoning.push_str(&delta),
                Piece::Tool(call) => calls.push(call),
            },
        );
        (result, text, reasoning, calls)
    })
    .await;

    match joined {
        Ok((Ok(decode), text, reasoning, calls)) => Ok(Generated {
            text,
            reasoning,
            calls,
            decode,
        }),
        Ok((Err(e), ..)) => Err(GenError::Runtime(e)),
        Err(e) => Err(GenError::Join(e.to_string())),
    }
}

/// Runs a generation on the calling (blocking) thread, handing each decoded
/// piece to `on_piece` as it arrives. Both streaming endpoints and
/// [`run_full`] go through here, so the stop-string tail handling and the
/// structured decoding are written once.
///
/// The dialect/reasoning/tool decision of whether this turn decodes at all
/// lives in `runtime::TurnSplitter` now, shared with the CLI and the FFI.
///
/// **The prompt path in `plan` stays keyed on `tools` ALONE** (crate Gotcha
/// 7). The splitter's build-a-decoder condition is a SUPERSET of the prompt
/// condition (Harmony decodes with no tools in the request), and that is
/// safe precisely because the two were never the same expression: rendering
/// the tool template for a request with no tools would change every Harmony
/// prompt.
///
/// `prompt_ids` is handed to the splitter because a ChatML generation prompt
/// OPENS the `<think>` frame itself when thinking is on, so the model never
/// emits the opening token and a decoder starting in the visible channel
/// reports the whole scratchpad as the reply
/// (`StructuredAssistantDecoder::new`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_blocking(
    model: &AppState,
    prompt_ids: &[foundation::TokenId],
    config: &GenerationConfig,
    images: Option<&crate::vision::RequestImages>,
    tools: &HashSet<String>,
    effort: ReasoningEffort,
    cancel: CancelFlag<'_>,
    on_piece: &mut dyn FnMut(Piece),
) -> Result<RawDecodeResult, RuntimeError> {
    // Ids only have to be unique within one assistant turn: a `tool` turn is
    // matched against the tool calls of the message immediately before it,
    // never against an earlier turn's. A counter is enough, and keeps
    // responses reproducible.
    let mut next_id = 0usize;
    let mut split = TurnSplitter::new(
        model.tokenizer(),
        tools,
        effort,
        move || {
            next_id += 1;
            format!("toolu_{}", next_id - 1)
        },
        prompt_ids,
    );
    let mut emit_piece = |event: TurnEvent| match event {
        TurnEvent::Prefill { .. } => {}
        TurnEvent::Content(delta) => on_piece(Piece::Text(delta)),
        TurnEvent::Reasoning(delta) => on_piece(Piece::Reasoning(delta)),
        TurnEvent::ToolCall(call) => on_piece(Piece::Tool(call)),
    };

    // `run_completion` and not `with_producer` + `run_raw_completion`: the
    // BACKEND owns which decode loop runs, because the speculative one takes
    // a concrete `SpeculativeProducer` and cannot be reached through a
    // `&mut dyn LogitProducer` (`ChatModel::run_completion`). The default
    // implementation is the sequential loop this line used to spell out, so
    // the scripted backend's path is unchanged.
    let result = model.run_completion(prompt_ids, config, images, cancel, &mut |e| {
        split.feed(e, &mut emit_piece);
    });

    // NOT OPTIONAL, AND NOT SYMMETRIC WITH `feed`. Harmony ends a tool call
    // with `<|call|>`, which is a stop token, so `run_raw_completion` breaks
    // before the progress callback and the decoder never sees the token that
    // terminates the call it is parsing -- `finish` is where that call is
    // emitted. Its OTHER job is releasing the tail DeepSeek's arm withholds
    // as a possible tool-marker prefix, which this loop used to drop.
    //
    // A decoder error here is the degraded path, not a failed request: the
    // run itself succeeded, and what a decoder abandons is the markup it
    // could not parse.
    if result.is_ok() {
        let _ = split.finish(&mut emit_piece);
    }
    result
}
