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
/// TWO INDEPENDENT REASONS, and keeping them independent is the point. Tools
/// need the decoder because the tool-chat generation prompt opens a thought
/// channel and the calls arrive as markup. Harmony needs it because
/// `gpt-oss` writes its reasoning into an `analysis` channel BEFORE its
/// answer, so without decoding the reasoning and the frame markup reach the
/// caller as the reply.
///
/// **The prompt path in `plan` stays keyed on `tools` ALONE** (crate Gotcha
/// 7). Those two conditions used to be the same expression, and widening this
/// one is exactly the change that could couple them again: rendering the tool
/// template for a request with no tools would change every Harmony prompt.
fn needs_decoder(model: &AppState, tools: &HashSet<String>) -> bool {
    !tools.is_empty() || model.tokenizer().dialect == tokenizer::ChatDialect::Harmony
}

/// Runs a generation to completion.
pub(crate) async fn run_full(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    tools: HashSet<String>,
) -> Result<Generated, GenError> {
    let joined = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        let result = stream_blocking(
            &model,
            &prompt_ids,
            &config,
            &tools,
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
    on_piece: &mut dyn FnMut(Piece),
) -> Result<RawDecodeResult, RuntimeError> {
    // Ids only have to be unique within one assistant turn: a `tool` turn is
    // matched against the tool calls of the message immediately before it,
    // never against an earlier turn's. A counter is enough, and keeps
    // responses reproducible.
    let mut next_id = 0usize;
    let mut decoder = needs_decoder(model, tools).then(|| {
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
                                    StructuredAssistantEvent::Reasoning(r) => Piece::Reasoning(r),
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
