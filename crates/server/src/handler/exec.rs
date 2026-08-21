//! Generation execution and streaming loop logic.

use std::collections::HashSet;

use runtime::{GenerationConfig, RawDecodeProgress, RawDecodeResult, RuntimeError};
use tokenizer::{
    ParsedToolCall, ReasoningEffort, StructuredAssistantDecoder, StructuredAssistantEvent,
};

use super::plan::AppState;

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
    /// Harmony frame separates one (see [`needs_decoder`]).
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

/// Whether this generation's output has to go through
/// [`StructuredAssistantDecoder`] rather than straight to the caller.
///
/// THREE INDEPENDENT REASONS, and keeping them independent is the point.
/// Tools need the decoder because the tool-chat generation prompt opens a
/// thought channel and the calls arrive as markup. Harmony needs it because
/// `gpt-oss` writes its reasoning into an `analysis` channel BEFORE its
/// answer, so without decoding the reasoning and the frame markup reach the
/// caller as the reply. A ChatML or Gemma request that ASKED for reasoning
/// needs it because the thought channel it just enabled would otherwise
/// arrive as the reply: the frame tokens render to the empty string, so the
/// client gets the model's scratch work run together with its answer and no
/// way to tell them apart. Measured on the real Gemma 4 install, where it
/// also prepends a bare `thought` -- the channel label as prose.
///
/// The third condition is keyed on the REQUEST rather than the dialect,
/// unlike Harmony's, and that asymmetry is the whole reason it is safe: with
/// no level asked for, `plan` renders a pre-closed `<think></think>` (ChatML)
/// or no thought channel at all (Gemma), so every existing request keeps the
/// pass-through path it has always had.
///
/// **The prompt path in `plan` stays keyed on `tools` ALONE** (crate Gotcha
/// 7). Those conditions used to be the same expression, and widening this one
/// is exactly the change that could couple them again: rendering the tool
/// template for a request with no tools would change every Harmony prompt.
fn needs_decoder(model: &AppState, tools: &HashSet<String>, reasoning: ReasoningEffort) -> bool {
    let dialect = model.tokenizer().dialect;
    !tools.is_empty()
        || dialect == tokenizer::ChatDialect::Harmony
        || (reasoning != ReasoningEffort::Off
            && matches!(
                dialect,
                tokenizer::ChatDialect::ChatMl | tokenizer::ChatDialect::Gemma
            ))
}

/// Runs a generation to completion.
pub(crate) async fn run_full(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    tools: HashSet<String>,
    effort: ReasoningEffort,
) -> Result<Generated, GenError> {
    let joined =
        tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut calls = Vec::new();
            let result =
                stream_blocking(&model, &prompt_ids, &config, &tools, effort, &mut |piece| {
                    match piece {
                        Piece::Text(delta) => text.push_str(&delta),
                        Piece::Reasoning(delta) => reasoning.push_str(&delta),
                        Piece::Tool(call) => calls.push(call),
                    }
                });
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
/// Text is passed straight through unless [`needs_decoder`] says otherwise,
/// exactly as before tool calling existed. Through the decoder, it is split
/// into visible content, reasoning, and parsed calls: the thought channel the
/// tool-chat generation prompt opens is swallowed, and Harmony's `analysis`
/// channel comes back as [`Piece::Reasoning`].
pub(crate) fn stream_blocking(
    model: &AppState,
    prompt_ids: &[foundation::TokenId],
    config: &GenerationConfig,
    tools: &HashSet<String>,
    effort: ReasoningEffort,
    on_piece: &mut dyn FnMut(Piece),
) -> Result<RawDecodeResult, RuntimeError> {
    // Ids only have to be unique within one assistant turn: a `tool` turn is
    // matched against the tool calls of the message immediately before it,
    // never against an earlier turn's. A counter is enough, and keeps
    // responses reproducible.
    let mut next_id = 0usize;
    let mut decoder = needs_decoder(model, tools, effort).then(|| {
        StructuredAssistantDecoder::new(model.tokenizer(), tools.clone(), move || {
            next_id += 1;
            format!("toolu_{}", next_id - 1)
        })
    });
    // A model writing prose that merely looks like a tool call must not fail
    // the request. On a parser error the decoder is abandoned and the rest of
    // the run is emitted as raw text; what the decoder had already buffered
    // when it gave up is lost with it.
    let mut degraded = false;

    // `run_completion` and not `with_producer` + `run_raw_completion`: the
    // BACKEND owns which decode loop runs, because the speculative one takes
    // a concrete `SpeculativeProducer` and cannot be reached through a
    // `&mut dyn LogitProducer` (`ChatModel::run_completion`). The default
    // implementation is the sequential loop this line used to spell out, so
    // the scripted backend's path is unchanged.
    let result = model.run_completion(prompt_ids, config, &mut |e| {
        let (id, text) = match e {
            RawDecodeProgress::Token { id, delta, .. } => (id, delta),
            // `-1` is the tokenizer's "no such token": a flushed tail
            // is text with no token id behind it.
            RawDecodeProgress::Tail(tail) => (tokenizer::NO_SUCH_TOKEN_ID, tail),
            _ => return,
        };
        match decoder.as_mut().filter(|_| !degraded) {
            None => {
                if !text.is_empty() {
                    on_piece(Piece::Text(text));
                }
            }
            Some(decoder) => match decoder.consume(id, &text) {
                Ok(events) => {
                    for event in events {
                        on_piece(piece_for(event));
                    }
                }
                Err(_) => {
                    degraded = true;
                    if !text.is_empty() {
                        on_piece(Piece::Text(text));
                    }
                }
            },
        }
    });

    // NOT OPTIONAL, AND NOT SYMMETRIC WITH `consume`. Harmony ends a tool call
    // with `<|call|>`, which is a stop token, so `run_raw_completion` breaks
    // before the progress callback and the decoder never sees the token that
    // terminates the call it is parsing -- `finish` is where that call is
    // emitted. Its OTHER job is releasing the tail DeepSeek's arm withholds as
    // a possible tool-marker prefix, which this loop used to drop.
    //
    // An error here is the degraded path, not a failed request: the run itself
    // succeeded, and what a decoder abandons is the markup it could not parse.
    if result.is_ok() {
        if let Some(decoder) = decoder.as_mut().filter(|_| !degraded) {
            for event in decoder.finish().unwrap_or_default() {
                on_piece(piece_for(event));
            }
        }
    }
    result
}

fn piece_for(event: StructuredAssistantEvent) -> Piece {
    match event {
        StructuredAssistantEvent::Content(c) => Piece::Text(c),
        StructuredAssistantEvent::Reasoning(r) => Piece::Reasoning(r),
        StructuredAssistantEvent::ToolCall(c) => Piece::Tool(c),
    }
}
