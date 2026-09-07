//! The OpenAI `POST /v1/responses` endpoint.
//!
//! Reuses `anyllm_translate`'s Responses wire types (`ResponsesRequest`,
//! `ResponsesResponse`, `ResponsesUsage`, `ResponsesStreamEvent`) -- the
//! vendored crate ships those for its own Anthropic<->Responses translation,
//! but nothing that maps Responses onto Chat Completions, which is what this
//! server actually generates through. So the mapping in both directions is
//! hand-written here: `map::responses_to_chat_request` folds a Responses request
//! down onto the same `ChatCompletionRequest` `handler::plan` already knows
//! how to render, and `sse::build_response` / streaming helpers turn a
//! generation back into Responses' own `output[]` item shapes
//! (see `crates/server/CLAUDE.md` Gotcha 24).
//!
//! Scope: text and tool calling, matching the other two endpoints.
//! `previous_response_id` is refused with a 400 (this server is stateless --
//! there is no prior turn to continue) and `store` is accepted and reported
//! on `x-anyllm-degradation` rather than honoured. Reasoning is included as
//! its own output item on the NON-streaming path but dropped (and reported,
//! when the checkpoint would have produced any) on the streaming one: OpenAI
//! has no delta event shape for `/v1/responses` reasoning summaries, and
//! inventing one would be undocumented on both ends.
//!
//! Submodules:
//! - `map`: Request folding, tool adaptation, and warning checks.
//! - `sse`: Non-streaming response serialization and typed SSE event builders.
//! - `tests`: Pure unit tests for mapping and event construction.

mod map;
mod sse;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::time::Duration;

use anyllm_translate::openai::responses::ResponsesRequest;
use anyllm_translate::openai::ChatCompletionRequest;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use tokenizer::ReasoningEffort;

use self::map::{may_produce_reasoning, responses_to_chat_request, responses_warnings};
use self::sse::{
    build_response, close_message_events, completed_event, content_part_added_event, created_event,
    failed_event, function_call_events, function_call_item, message_item, open_message_event,
    text_delta_event,
};
use crate::guardrails::run_guarded;
use crate::handler::{
    error_response, plan, reasoning_effort, request_id, stream_blocking, tool_names, AppState,
    GenError, Piece,
};

const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /v1/responses`.
pub async fn responses(
    State(state): State<crate::ServerState>,
    tag: Option<axum::Extension<crate::observe::RequestTag>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    let model = match crate::handler::resolve_backend(
        &state,
        tag.map(|t| t.0),
        Some(request.model.as_str()),
        request.stream.unwrap_or(false),
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };
    let chat_request = match responses_to_chat_request(&request) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    // Planned here, before ANY of the three paths below (non-streaming, live
    // streaming, buffered/guarded streaming) are chosen. Without this, only
    // the live streaming path validated its own request (`stream_response`'s
    // internal `plan` call below) -- the non-streaming path went straight to
    // `run_guarded`, whose OWN `plan` call exists for the retry turn and maps
    // a failure to `GenError::Join` (a 500) on the reasoning that `plan`
    // already succeeded on the caller's request by the time it is reached
    // (`crates/server/CLAUDE.md`'s guardrails Gotcha). That reasoning does
    // not hold here the way it does for `/v1/chat/completions` and
    // `/v1/messages`, both of which plan before ever calling `run_guarded`.
    // A malformed request (a bad `reasoning_effort`, a system turn at index
    // > 0, a remote image URL) must 400 like it does on those two endpoints,
    // not 500.
    if let Err(e) = plan(&model, &chat_request) {
        return error_response(StatusCode::BAD_REQUEST, e);
    }
    let tools = tool_names(&chat_request);
    let effort = reasoning_effort(&chat_request, model.default_reasoning())
        .unwrap_or_else(|_| model.default_reasoning());
    let streaming = request.stream.unwrap_or(false);
    let degraded = responses_warnings(&request, streaming && may_produce_reasoning(&model, effort));
    let request_model = request.model.clone();
    let cancel = crate::cancel::new_cancel();

    let mut response = if streaming {
        stream_response(model, chat_request, tools, effort, request_model, cancel)
    } else {
        // Set if THIS future is dropped (client gone) before `run_guarded`
        // resolves; defused right after, whatever it returned (Gotcha 25).
        let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
        let result = run_guarded(model, &chat_request, effort, cancel).await;
        guard.defuse();
        match result {
            Ok(generated) => Json(build_response(
                request_id("resp_"),
                request_model,
                generated,
            ))
            .into_response(),
            Err(e) => return crate::handler::gen_error_response(e),
        }
    };
    if let Some(note) = degraded.and_then(|v| v.parse().ok()) {
        response
            .headers_mut()
            .insert(crate::DEGRADATION_HEADER, note);
    }
    response
}

#[allow(clippy::too_many_arguments)]
fn stream_response(
    model: AppState,
    chat_request: ChatCompletionRequest,
    tools: HashSet<String>,
    effort: ReasoningEffort,
    request_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    if !tools.is_empty() && model.guardrails().active() {
        return buffered_stream_response(model, chat_request, effort, request_model, cancel);
    }
    let planned = match plan(&model, &chat_request) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = request_id("resp_");
    let model_name = request_model;
    let cancel_for_task = cancel.clone();

    tokio::task::spawn_blocking(move || {
        let send = |ev: Event| {
            // Fast-path disconnect detection: `CancelOnDrop` below catches
            // the same event even with no chunk in flight (a long prefill
            // sends none), so this is a speed-up rather than the only
            // detector (Gotcha 25).
            if tx.send(ev).is_err() {
                cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        };
        send(created_event(&id, &model_name));

        let mut message_open = false;
        let mut message_item_id = String::new();
        let mut text_acc = String::new();
        // ONE cursor for every item this turn emits, message and function
        // calls alike -- not a message fixed at index 0 with calls starting
        // at 1, which put a call-only turn's first call at index 1 with
        // nothing at 0.
        let mut output_index: u32 = 0;
        let mut items: Vec<serde_json::Value> = Vec::new();

        let flag = crate::cancel::as_cancel_flag(&cancel_for_task);
        let result = stream_blocking(
            &model,
            &planned.prompt_ids,
            &planned.config,
            planned.images.as_ref(),
            &tools,
            effort,
            &flag,
            &mut |piece| match piece {
                // Dropped in v1: OpenAI defines no delta event for a
                // Responses reasoning summary, and `responses_warnings`
                // already reported this ahead of the stream when the
                // checkpoint's dialect meant one would be produced (Gotcha 24).
                Piece::Reasoning(_) => {}
                Piece::Text(delta) => {
                    if delta.is_empty() {
                        return;
                    }
                    if !message_open {
                        message_item_id = format!("msg_{output_index}");
                        send(open_message_event(&message_item_id, output_index));
                        send(content_part_added_event(&message_item_id, output_index));
                        message_open = true;
                    }
                    text_acc.push_str(&delta);
                    send(text_delta_event(&message_item_id, output_index, &delta));
                }
                Piece::Tool(call) => {
                    if message_open {
                        close_message_events(&send, &message_item_id, output_index, &text_acc);
                        items.push(message_item(&text_acc, "completed", &message_item_id));
                        message_open = false;
                        text_acc.clear();
                        output_index += 1;
                    }
                    function_call_events(&send, &call, output_index);
                    items.push(function_call_item(&call, "completed"));
                    output_index += 1;
                }
            },
        );

        match result {
            // Discarded silently: the client that would read
            // `response.completed` is the one already gone.
            Ok(decode) if decode.reason == runtime::StopReason::Cancelled => (),
            Ok(decode) => {
                if message_open {
                    close_message_events(&send, &message_item_id, output_index, &text_acc);
                    items.push(message_item(&text_acc, "completed", &message_item_id));
                }
                send(completed_event(&id, &model_name, &decode, &items));
            }
            // A failed run is not a completed one: report it as a
            // `response.failed` event rather than a fabricated `completed`
            // status, matching the other two endpoints' contract.
            Err(e) => send(failed_event(&e.to_string())),
        }
        // No `[DONE]` sentinel: an OpenAI Responses stream ends at
        // `response.completed` (or `response.failed`), the same shape
        // Anthropic's own stream takes and for the same reason -- `[DONE]`
        // is a Chat-Completions-ism (Gotcha 24).
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(
        crate::cancel::CancelOnDrop::new(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            cancel,
        )
        .map(Ok),
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(SSE_KEEP_ALIVE))
        .into_response()
}

/// The guarded generation, re-framed as the SSE sequence the live path would
/// have produced -- one delta per item rather than per token, since the
/// whole generation is already in hand by the time anything is sent. Same
/// shape as `handler::mod::buffered_stream_response` and
/// `messages::buffered_stream_response`: a tool-carrying request under
/// active guardrails buffers so a rescued or re-asked call never streams the
/// syntax error it was rescued from (Gotchas 18, 24).
fn buffered_stream_response(
    model: AppState,
    chat_request: ChatCompletionRequest,
    effort: ReasoningEffort,
    request_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = request_id("resp_");
    let model_name = request_model;
    let cancel_for_stream = cancel.clone();

    tokio::spawn(async move {
        let send = move |ev: Event| {
            let _ = tx.send(ev);
        };
        send(created_event(&id, &model_name));

        // Nothing is sent until the whole generation is done, so
        // `CancelOnDrop` below is the only signal this path has that the
        // client is gone (Gotcha 25).
        match run_guarded(model, &chat_request, effort, cancel).await {
            // Discarded silently, same contract as the live path.
            Ok(generated) if generated.decode.reason == runtime::StopReason::Cancelled => (),
            Ok(generated) => {
                let mut output_index: u32 = 0;
                let mut items: Vec<serde_json::Value> = Vec::new();
                if !generated.text.is_empty() {
                    let item_id = format!("msg_{output_index}");
                    send(open_message_event(&item_id, output_index));
                    send(content_part_added_event(&item_id, output_index));
                    send(text_delta_event(&item_id, output_index, &generated.text));
                    close_message_events(&send, &item_id, output_index, &generated.text);
                    items.push(message_item(&generated.text, "completed", &item_id));
                    output_index += 1;
                }
                for call in &generated.calls {
                    function_call_events(&send, call, output_index);
                    items.push(function_call_item(call, "completed"));
                    output_index += 1;
                }
                send(completed_event(&id, &model_name, &generated.decode, &items));
            }
            Err(e) => {
                let message = match e {
                    GenError::Runtime(e) => e.to_string(),
                    GenError::Join(m) => m,
                };
                send(failed_event(&message));
            }
        }
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(
        crate::cancel::CancelOnDrop::new(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            cancel_for_stream,
        )
        .map(Ok),
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(SSE_KEEP_ALIVE))
        .into_response()
}
