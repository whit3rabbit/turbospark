//! Generation execution and streaming loop logic.

use std::collections::HashSet;

use runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, RawDecodeResult, RuntimeError,
};
use tokenizer::{ParsedToolCall, StructuredAssistantDecoder, StructuredAssistantEvent};

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
    Tool(ParsedToolCall),
}

/// Runs a generation to completion, returning the whole text and every tool
/// call parsed out of it.
pub(crate) async fn run_full(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    tools: HashSet<String>,
) -> Result<(String, Vec<ParsedToolCall>, RawDecodeResult), GenError> {
    let joined = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let mut calls = Vec::new();
        let result = stream_blocking(
            &model,
            &prompt_ids,
            &config,
            &tools,
            &mut |piece| match piece {
                Piece::Text(delta) => text.push_str(&delta),
                Piece::Tool(call) => calls.push(call),
            },
        );
        (result, text, calls)
    })
    .await;

    match joined {
        Ok((Ok(decode), text, calls)) => Ok((text, calls, decode)),
        Ok((Err(e), _, _)) => Err(GenError::Runtime(e)),
        Err(e) => Err(GenError::Join(e.to_string())),
    }
}

/// Runs a generation on the calling (blocking) thread, handing each decoded
/// piece to `on_piece` as it arrives. Both streaming endpoints and
/// [`run_full`] go through here, so the stop-string tail handling and the
/// structured decoding are written once.
///
/// With `tools` empty the generated text is passed straight through, exactly
/// as before tool calling existed. With tools, it goes through
/// [`StructuredAssistantDecoder`], which splits it into visible content and
/// parsed calls and swallows the thought channel the tool-chat generation
/// prompt opens.
pub(crate) fn stream_blocking(
    model: &AppState,
    prompt_ids: &[foundation::TokenId],
    config: &GenerationConfig,
    tools: &HashSet<String>,
    on_piece: &mut dyn FnMut(Piece),
) -> Result<RawDecodeResult, RuntimeError> {
    // Ids only have to be unique within one assistant turn: a `tool` turn is
    // matched against the tool calls of the message immediately before it,
    // never against an earlier turn's. A counter is enough, and keeps
    // responses reproducible.
    let mut next_id = 0usize;
    let mut decoder = (!tools.is_empty()).then(|| {
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

    model.with_producer(&mut |producer| {
        run_raw_completion(
            producer,
            model.tokenizer(),
            prompt_ids,
            config,
            model.max_context(),
            model.vocab_size(),
            |e| {
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
                                on_piece(match event {
                                    StructuredAssistantEvent::Content(c) => Piece::Text(c),
                                    StructuredAssistantEvent::ToolCall(c) => Piece::Tool(c),
                                });
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
            },
        )
    })
}
