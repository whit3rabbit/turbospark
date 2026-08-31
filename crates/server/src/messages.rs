//! The Anthropic `POST /v1/messages` handler.
//!
//! This server generates in OpenAI shape and nothing else. Rather than teach
//! it a second wire format, an Anthropic request is translated into the
//! OpenAI request the `/v1/chat/completions` path already understands, run
//! through the same generation core in `handler.rs`, and the result
//! translated back. `anyllm_translate` owns both directions, including the
//! streaming state machine that turns a flat sequence of OpenAI content
//! deltas into Anthropic's `message_start` / `content_block_*` /
//! `message_stop` event structure.
//!
//! Scope: text and tool calling. `tools` and `tool_choice` come across as
//! OpenAI tools, which `handler::plan` renders through the checkpoint's own
//! Jinja chat template, and a parsed call comes back as an Anthropic
//! `tool_use` block. Images and a request's `thinking` CONFIG have no
//! backend here and are dropped in translation; a RESPONSE's thinking is a
//! different matter, and `gpt-oss` produces one (see `handler::exec`'s
//! `needs_decoder`). `x-anyllm-degradation` reports what
//! `compute_request_warnings` knows about (`top_k`, `thinking`,
//! `cache_control`, document blocks, truncated stop sequences) AND, since
//! ROADMAP M-V8, an image this server could not serve -- appended by this
//! module rather than by the vendored function, which has no notion of a
//! local install's capabilities. An image on an install WITH a tower is
//! served rather than reported.

use std::collections::HashSet;

use anyllm_translate::anthropic::streaming::{StreamError, StreamEvent};
use anyllm_translate::anthropic::{ErrorType, MessageCreateRequest};
use anyllm_translate::mapping::errors_map::create_anthropic_error;
use anyllm_translate::mapping::streaming_map::StreamingTranslator;
use anyllm_translate::openai::streaming::ChatCompletionChunk;
use anyllm_translate::openai::ChatCompletionRequest;
use anyllm_translate::{
    compute_request_warnings, new_stream_translator, translate_request, translate_response,
    TranslationConfig,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::GenerationConfig;
use tokenizer::ReasoningEffort;

use crate::guardrails::run_guarded;
use crate::handler::{
    merge_degradation, now_unix, plan, reasoning_effort, status_for, stream_blocking, tool_names,
    AppState, GenError, Piece, SSE_KEEP_ALIVE,
};
use crate::response::{
    completion_chunk, completion_response, finish_reason, reasoning_delta, role_delta, text_delta,
    tool_call_delta,
};

pub(crate) use crate::DEGRADATION_HEADER;

fn error_body(status: StatusCode, kind: ErrorType, message: String) -> Response {
    (status, Json(create_anthropic_error(kind, message, None))).into_response()
}

fn gen_error_body(e: GenError) -> Response {
    match e {
        GenError::Runtime(e) => {
            let status = status_for(&e);
            let kind = if status == StatusCode::BAD_REQUEST {
                ErrorType::InvalidRequestError
            } else {
                ErrorType::ApiError
            };
            error_body(status, kind, e.to_string())
        }
        GenError::Join(m) => error_body(StatusCode::INTERNAL_SERVER_ERROR, ErrorType::ApiError, m),
    }
}

/// Anthropic clients dispatch on the SSE `event:` line, which the OpenAI
/// stream never sets. The variant names are the wire names.
fn to_sse(event: &StreamEvent) -> Event {
    let name = match event {
        StreamEvent::MessageStart { .. } => "message_start",
        StreamEvent::ContentBlockStart { .. } => "content_block_start",
        StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        StreamEvent::MessageDelta { .. } => "message_delta",
        StreamEvent::MessageStop {} => "message_stop",
        StreamEvent::Ping {} => "ping",
        StreamEvent::Error { .. } => "error",
        StreamEvent::Unknown => "unknown",
    };
    Event::default()
        .event(name)
        .data(serde_json::to_string(event).unwrap_or_default())
}

/// `POST /v1/messages`. Anthropic-compatible messages endpoint, translating
/// requests to OpenAI format and translating responses back.
pub async fn messages(
    State(state): State<crate::ServerState>,
    tag: Option<axum::Extension<crate::observe::RequestTag>>,
    Json(request): Json<MessageCreateRequest>,
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
    let degraded = compute_request_warnings(&request).as_header_value();

    let openai = match translate_request(&request, &TranslationConfig::default()) {
        Ok(r) => r,
        Err(e) => {
            return error_body(
                StatusCode::BAD_REQUEST,
                ErrorType::InvalidRequestError,
                e.to_string(),
            )
        }
    };

    let planned = match plan(&model, &openai) {
        Ok(p) => p,
        Err(e) => return error_body(StatusCode::BAD_REQUEST, ErrorType::InvalidRequestError, e),
    };
    let (prompt_ids, config, images, dropped_images) = (
        planned.prompt_ids,
        planned.config,
        planned.images,
        planned.dropped_images,
    );
    // **THE HEADER FINALLY NAMES A DROPPED IMAGE** (ROADMAP M-V8). This
    // module's own header used to record that `compute_request_warnings`
    // knows nothing about images, which was true and was the gap: a client
    // sending a picture to a text-only install got a plausible text answer
    // and no signal at all. Appended rather than replacing, so the vendored
    // warnings keep theirs.
    let degraded = merge_degradation(degraded, dropped_images);

    let tools = tool_names(&openai);
    // See the OpenAI handler: `plan` has already rejected a bad value.
    let effort = reasoning_effort(&openai, model.default_reasoning())
        .unwrap_or_else(|_| model.default_reasoning());
    let streaming = request.stream.unwrap_or(false);
    let cancel = crate::cancel::new_cancel();
    let mut response = if streaming {
        stream_response(
            model,
            prompt_ids,
            config,
            images,
            tools,
            effort,
            &openai,
            request.model,
            cancel,
        )
    } else {
        full_response(model, &openai, effort, request.model, cancel).await
    };

    if let Some(value) = degraded.and_then(|v| v.parse().ok()) {
        response.headers_mut().insert(DEGRADATION_HEADER, value);
    }
    response
}

/// `POST /v1/messages/count_tokens`. What a real request would prefill,
/// with no generation and no `max_tokens` budget required.
///
/// **A `serde_json::Value` FIRST, NOT `Json<MessageCreateRequest>` DIRECTLY,
/// because `MessageCreateRequest::max_tokens` is a REQUIRED `u32` on the
/// wire type** -- correct for `/v1/messages`, where a real generation needs
/// a budget, and wrong here: Anthropic's own `count_tokens` endpoint takes a
/// request with no `max_tokens` at all. A placeholder is injected only when
/// the field is absent, so a request that DID send one is unaffected.
///
/// Reuses `handler::plan` rather than a separate encode path, which is what
/// makes the count MEAN something: it is definitionally the same template
/// split, the same `encode(_, add_bos: false)`, and the same image splice a
/// real `/v1/messages` call on this request would run.
pub async fn count_tokens(
    State(state): State<crate::ServerState>,
    tag: Option<axum::Extension<crate::observe::RequestTag>>,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    // Resolved off the RAW body, before `max_tokens` is injected: this
    // endpoint takes `serde_json::Value` precisely because a valid
    // count_tokens request need not carry one, and the model name is
    // readable without that repair.
    let requested = body
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let model = match crate::handler::resolve_backend(
        &state,
        tag.map(|t| t.0),
        requested.as_deref(),
        false,
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };
    if let Some(object) = body.as_object_mut() {
        if !object.contains_key("max_tokens") {
            object.insert("max_tokens".to_string(), serde_json::json!(1));
        }
    }
    let request: MessageCreateRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return error_body(
                StatusCode::BAD_REQUEST,
                ErrorType::InvalidRequestError,
                e.to_string(),
            )
        }
    };
    let openai = match translate_request(&request, &TranslationConfig::default()) {
        Ok(r) => r,
        Err(e) => {
            return error_body(
                StatusCode::BAD_REQUEST,
                ErrorType::InvalidRequestError,
                e.to_string(),
            )
        }
    };
    let planned = match plan(&model, &openai) {
        Ok(p) => p,
        Err(e) => return error_body(StatusCode::BAD_REQUEST, ErrorType::InvalidRequestError, e),
    };
    Json(serde_json::json!({"input_tokens": planned.prompt_ids.len()})).into_response()
}

async fn full_response(
    model: AppState,
    openai: &ChatCompletionRequest,
    effort: ReasoningEffort,
    // The model name the client asked for, echoed into the Anthropic
    // response. Translation maps it to the backend model on the way in, and
    // that mapping is not necessarily reversible, so it is carried across
    // rather than derived.
    client_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    let backend_model = openai.model.clone();
    // Set if THIS future is dropped (client gone) before `run_guarded`
    // resolves; defused right after, whatever it returned.
    let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
    let result = run_guarded(model, openai, effort, cancel).await;
    guard.defuse();
    let generated = match result {
        Ok(r) => r,
        Err(e) => return gen_error_body(e),
    };

    let openai_response = completion_response(
        format!("chatcmpl-{}", now_unix()),
        now_unix(),
        backend_model,
        generated.text,
        generated.reasoning,
        generated.calls,
        generated.decode.reason,
        generated.decode.prompt_tokens as u32,
        generated.decode.new_tokens as u32,
    );
    Json(translate_response(&openai_response, &client_model)).into_response()
}

/// Anthropic SSE stream over the model's token-by-token decode.
///
/// **Guardrailed requests are BUFFERED rather than streamed here**, by
/// design: a rescued call has to replace the turn before anything is sent,
/// and streaming it would leak the un-rescued syntax error as text before the
/// tool-call event can arrive (`crates/server/CLAUDE.md` Gotcha 18).
/// Everything else keeps the live path byte for byte.
#[allow(clippy::too_many_arguments)]
fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    images: Option<crate::vision::RequestImages>,
    tools: HashSet<String>,
    effort: ReasoningEffort,
    openai: &ChatCompletionRequest,
    client_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    if !tools.is_empty() && model.guardrails().active() {
        return buffered_stream_response(model, openai.clone(), effort, client_model, cancel);
    }
    let backend_model = openai.model.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();
    let cancel_for_task = cancel.clone();

    tokio::task::spawn_blocking(move || {
        // The translator is a state machine over the OpenAI chunk sequence:
        // it opens the message on the first chunk carrying a role and closes
        // the content block on `finish()`. So the chunk order here has to
        // match what the `/v1/chat/completions` stream emits -- role chunk,
        // content chunks, finish chunk -- or the Anthropic event structure
        // comes out malformed.
        let mut translator = new_stream_translator(client_model);

        let send = |chunk: ChatCompletionChunk, translator: &mut StreamingTranslator| {
            for event in translator.process_chunk(&chunk) {
                // Same fast-path detection as the OpenAI endpoint's: a
                // failed send means the receiver -- and the
                // `CancelOnDrop`-wrapped stream around it -- is already
                // gone.
                if tx.send(to_sse(&event)).is_err() {
                    cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        };

        send(
            completion_chunk(
                id.clone(),
                created,
                backend_model.clone(),
                role_delta(),
                None,
            ),
            &mut translator,
        );

        let mut call_index = 0u32;
        let flag = crate::cancel::as_cancel_flag(&cancel_for_task);
        let result = stream_blocking(
            &model,
            &prompt_ids,
            &config,
            images.as_ref(),
            &tools,
            effort,
            &flag,
            &mut |piece| {
                let delta = match piece {
                    Piece::Text(text) => text_delta(text),
                    // The translator turns this into a `thinking` content block,
                    // opened on the first one and closed on the first text delta.
                    Piece::Reasoning(text) => reasoning_delta(text),
                    Piece::Tool(call) => {
                        call_index += 1;
                        tool_call_delta(call_index - 1, call)
                    }
                };
                send(
                    completion_chunk(id.clone(), created, backend_model.clone(), delta, None),
                    &mut translator,
                );
            },
        );

        match result {
            // Discarded silently: the client that would read the closing
            // events is the one already gone.
            Ok(r) if r.reason == runtime::StopReason::Cancelled => (),
            Ok(r) => {
                send(
                    completion_chunk(
                        id.clone(),
                        created,
                        backend_model.clone(),
                        Default::default(),
                        Some(finish_reason(r.reason)),
                    ),
                    &mut translator,
                );
                for event in translator.finish() {
                    let _ = tx.send(to_sse(&event));
                }
            }
            // A failed run is not a completed one: emit an `error` event
            // rather than closing the message as if it had stopped normally.
            Err(e) => {
                let _ = tx.send(to_sse(&StreamEvent::Error {
                    error: StreamError {
                        error_type: "api_error".to_string(),
                        message: e.to_string(),
                    },
                }));
            }
        }
        // No `[DONE]` sentinel: that is an OpenAI-ism. An Anthropic stream
        // ends at `message_stop`.
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

/// The guarded generation, pushed through the Anthropic translator in the
/// order its state machine requires.
///
/// **ROLE, REASONING, CONTENT, TOOL CALLS, FINISH -- and that order is the
/// correctness condition, not a style** (crate Gotcha 6). The translator opens
/// the message on the first chunk carrying a role, opens a `thinking` block on
/// the first reasoning delta and closes it on the first text delta without
/// reopening. Feed it content before reasoning and the events come out
/// malformed rather than erroring.
fn buffered_stream_response(
    model: AppState,
    openai: ChatCompletionRequest,
    effort: ReasoningEffort,
    client_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();
    let backend_model = openai.model.clone();
    let cancel_for_stream = cancel.clone();

    tokio::spawn(async move {
        let mut translator = new_stream_translator(client_model);
        let send = |chunk: ChatCompletionChunk, translator: &mut StreamingTranslator| {
            for event in translator.process_chunk(&chunk) {
                let _ = tx.send(to_sse(&event));
            }
        };
        let chunk = |delta, finish| {
            completion_chunk(id.clone(), created, backend_model.clone(), delta, finish)
        };

        // Nothing is sent until the whole generation is done, so `CancelOnDrop`
        // below is the only signal this path has that the client is gone.
        match run_guarded(model, &openai, effort, cancel).await {
            // Discarded silently, same contract as the live path.
            Ok(generated) if generated.decode.reason == runtime::StopReason::Cancelled => (),
            Ok(generated) => {
                send(chunk(role_delta(), None), &mut translator);
                if !generated.reasoning.is_empty() {
                    send(
                        chunk(reasoning_delta(generated.reasoning), None),
                        &mut translator,
                    );
                }
                if !generated.text.is_empty() {
                    send(chunk(text_delta(generated.text), None), &mut translator);
                }
                for (index, call) in generated.calls.into_iter().enumerate() {
                    send(
                        chunk(tool_call_delta(index as u32, call), None),
                        &mut translator,
                    );
                }
                send(
                    chunk(
                        Default::default(),
                        Some(finish_reason(generated.decode.reason)),
                    ),
                    &mut translator,
                );
                for event in translator.finish() {
                    let _ = tx.send(to_sse(&event));
                }
            }
            // Same contract as the live path: an `error` event rather than a
            // message closed as if it had stopped normally.
            Err(e) => {
                let message = match e {
                    GenError::Runtime(e) => e.to_string(),
                    GenError::Join(m) => m,
                };
                let _ = tx.send(to_sse(&StreamEvent::Error {
                    error: StreamError {
                        error_type: "api_error".to_string(),
                        message,
                    },
                }));
            }
        }
        // No `[DONE]`: an Anthropic stream ends at `message_stop`.
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
