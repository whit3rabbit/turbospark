//! The `/v1/chat/completions` and `/v1/models` handlers, plus the generation
//! core both this endpoint and `/v1/messages` are built on: [`plan`] turns a
//! request into prompt tokens and a generation config, [`run_full`] runs it to
//! a `String`, and [`stream_blocking`] runs it emitting deltas as they arrive.
//!
//! Generation runs on a blocking task (it is synchronous CPU/GPU work, not
//! I/O). The request envelope is `anyllm_translate`'s OpenAI type rather than
//! a local struct; see `response.rs` for why.

mod exec;
mod plan;

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use anyllm_translate::openai::ChatCompletionRequest;
use axum::extract::{Path, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::{GenerationConfig, RuntimeError};
use tokenizer::ReasoningEffort;

pub(crate) use exec::*;
pub use plan::AppState;
pub(crate) use plan::*;

use crate::guardrails::run_guarded;
use crate::response::{
    completion_chunk, completion_response, reasoning_delta, role_delta, text_delta, tool_call_delta,
};

pub(crate) fn error_response(status: axum::http::StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

pub(crate) fn status_for(e: &RuntimeError) -> axum::http::StatusCode {
    match e {
        RuntimeError::EmptyPrompt
        | RuntimeError::ContextOverflow { .. }
        | RuntimeError::Selection(_) => axum::http::StatusCode::BAD_REQUEST,
        // 500 rather than 400: nothing in the REQUEST asks for speculation
        // (it is a process-level setting, like the rate cap in Gotcha 10), so
        // a caller cannot have sent anything to avoid this and must not be
        // told they did.
        RuntimeError::Producer(_) | RuntimeError::SpeculationUnavailable(_) => {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

pub(crate) fn gen_error_response(e: GenError) -> Response {
    match e {
        GenError::Runtime(e) => error_response(status_for(&e), e.to_string()),
        GenError::Join(m) => error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, m),
    }
}

/// `GET /v1/models`. One backend per process, so the list has one entry.
/// This is what an OpenAI client's model picker (and Claude Code's gateway
/// model discovery) reads.
pub async fn models(State(model): State<AppState>) -> Response {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": model.model_id(),
            "object": "model",
            "created": now_unix(),
            "owned_by": "mference",
        }],
    }))
    .into_response()
}

/// `GET /v1/models/:model`. Returns model details if `:model` matches the active backend model.
pub async fn model_detail(State(model): State<AppState>, Path(model_id): Path<String>) -> Response {
    if model_id == model.model_id() {
        Json(serde_json::json!({
            "id": model.model_id(),
            "object": "model",
            "created": now_unix(),
            "owned_by": "mference",
        }))
        .into_response()
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("The model '{model_id}' does not exist"),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response()
    }
}

pub async fn chat_completions(
    State(model): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Planned here as well as inside `run_guarded` so an unparseable request
    // is still refused with a 400 before any generation starts. The guarded
    // path re-plans because a retry turn changes the messages.
    let (prompt_ids, config) = match plan(&model, &request) {
        Ok(p) => p,
        Err(e) => return error_response(axum::http::StatusCode::BAD_REQUEST, e),
    };

    let tools = tool_names(&request);
    // Re-read rather than threaded out of `plan`: `plan` already REFUSED an
    // unparseable value, so this cannot fail by the time it is reached, and
    // the pairing with `tool_names` above keeps both request-derived inputs
    // to the decoder in one place.
    let effort = reasoning_effort(&request).unwrap_or_default();
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    if request.stream.unwrap_or(false) {
        return stream_response(
            model,
            prompt_ids,
            config,
            tools,
            effort,
            &request,
            include_usage,
        );
    }

    let generated = match run_guarded(model, &request, effort).await {
        Ok(r) => r,
        Err(e) => return gen_error_response(e),
    };

    Json(completion_response(
        format!("chatcmpl-{}", now_unix()),
        now_unix(),
        request.model,
        generated.text,
        generated.reasoning,
        generated.calls,
        generated.decode.reason,
        generated.decode.prompt_tokens as u32,
        generated.decode.new_tokens as u32,
    ))
    .into_response()
}

/// **A TOOL-CARRYING REQUEST IS BUFFERED WHEN GUARDRAILS ARE ON, and every
/// other request streams exactly as it always did.**
///
/// The guardrails need the whole generation before they can reach a verdict: a
/// call to rescue is one the decoder did not parse, so it is indistinguishable
/// from prose until the turn ends, and a retry re-generates from scratch. Emit
/// deltas live and both repairs become unavailable -- the markup is already on
/// the wire.
///
/// So a request carrying tools generates to completion, is inspected, and is
/// then re-framed as the SSE sequence it would have produced. The cost is
/// real and is taken knowingly: time-to-first-token becomes time-to-last-token
/// for that turn, roughly 5-10 s at this engine's 20-45 tok/s. The alternative
/// is guardrails that do nothing for the client that motivated them -- Claude
/// Code sends `stream: true` WITH tools -- and a `forge-guardrails-proxy` in
/// front would buffer identically.
///
/// The condition is keyed on the request carrying tools, which is what keeps
/// ordinary chat traffic on the live path byte for byte (the same shape as
/// Gotcha 12's third condition and Gotcha 7's prompt split).
fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    tools: HashSet<String>,
    effort: ReasoningEffort,
    request: &ChatCompletionRequest,
    include_usage: bool,
) -> Response {
    if !tools.is_empty() && model.guardrails().active() {
        return buffered_stream_response(model, request.clone(), effort, include_usage);
    }
    let model_name = request.model.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();

    tokio::task::spawn_blocking(move || {
        let send = |chunk: anyllm_translate::openai::streaming::ChatCompletionChunk| {
            let _ =
                tx.send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
        };
        send(completion_chunk(
            id.clone(),
            created,
            model_name.clone(),
            role_delta(),
            None,
        ));

        let mut call_index = 0u32;
        let result = stream_blocking(&model, &prompt_ids, &config, &tools, effort, &mut |piece| {
            let delta = match piece {
                Piece::Text(text) => text_delta(text),
                Piece::Reasoning(text) => reasoning_delta(text),
                Piece::Tool(call) => {
                    call_index += 1;
                    tool_call_delta(call_index - 1, call)
                }
            };
            send(completion_chunk(
                id.clone(),
                created,
                model_name.clone(),
                delta,
                None,
            ));
        });

        match result {
            Ok(r) => {
                send(completion_chunk(
                    id.clone(),
                    created,
                    model_name.clone(),
                    Default::default(),
                    Some(crate::response::finish_reason(r.reason)),
                ));
                if include_usage {
                    send(crate::response::usage_chunk(
                        id.clone(),
                        created,
                        model_name.clone(),
                        r.prompt_tokens as u32,
                        r.new_tokens as u32,
                    ));
                }
            }
            // A failed run is not a completed one: report it as an error
            // event rather than a fabricated `stop` finish reason.
            Err(e) => {
                let body = serde_json::json!({
                    "error": {"message": e.to_string(), "type": "server_error"}
                });
                let _ = tx.send(Event::default().data(body.to_string()));
            }
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok));
    Sse::new(stream).into_response()
}

/// The guarded generation, re-framed as the SSE sequence the live path would
/// have produced.
///
/// **THE CHUNK ORDER IS LOAD BEARING and is not cosmetic.** Anthropic's
/// `StreamingTranslator` is a state machine over this exact sequence (crate
/// Gotcha 6): it opens the message on the first chunk carrying a role, opens a
/// thinking block on the first reasoning delta and closes it on the first
/// content delta without reopening. Reordering these -- content before
/// reasoning, or a finish chunk before the tool calls -- produces malformed
/// Anthropic events rather than an error, so `/v1/messages` inherits its
/// correctness from this function keeping role, reasoning, content, tool
/// calls, finish.
fn buffered_stream_response(
    model: AppState,
    request: ChatCompletionRequest,
    effort: ReasoningEffort,
    include_usage: bool,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();
    let model_name = request.model.clone();

    tokio::spawn(async move {
        let send = |chunk: anyllm_translate::openai::streaming::ChatCompletionChunk| {
            let _ =
                tx.send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
        };
        let chunk = |delta, finish| {
            completion_chunk(id.clone(), created, model_name.clone(), delta, finish)
        };

        match run_guarded(model, &request, effort).await {
            Ok(generated) => {
                send(chunk(role_delta(), None));
                if !generated.reasoning.is_empty() {
                    send(chunk(reasoning_delta(generated.reasoning), None));
                }
                if !generated.text.is_empty() {
                    send(chunk(text_delta(generated.text), None));
                }
                for (index, call) in generated.calls.into_iter().enumerate() {
                    send(chunk(tool_call_delta(index as u32, call), None));
                }
                send(chunk(
                    Default::default(),
                    Some(crate::response::finish_reason(generated.decode.reason)),
                ));
                if include_usage {
                    send(crate::response::usage_chunk(
                        id.clone(),
                        created,
                        model_name.clone(),
                        generated.decode.prompt_tokens as u32,
                        generated.decode.new_tokens as u32,
                    ));
                }
            }
            // Same contract as the live path: a failed run is not a completed
            // one, so it is an error event rather than a fabricated `stop`.
            Err(e) => {
                let message = match e {
                    GenError::Runtime(e) => e.to_string(),
                    GenError::Join(m) => m,
                };
                let body = serde_json::json!({
                    "error": {"message": message, "type": "server_error"}
                });
                let _ = tx.send(Event::default().data(body.to_string()));
            }
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok));
    Sse::new(stream).into_response()
}
