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
//! Scope: text only. Tools, images, and `thinking` have no backend here, so
//! they are dropped in translation and reported on the
//! `x-anyllm-degradation` response header rather than silently discarded.
//! See `DEVIATIONS.md`.

use anyllm_translate::anthropic::streaming::{StreamError, StreamEvent};
use anyllm_translate::anthropic::{ErrorType, MessageCreateRequest};
use anyllm_translate::mapping::errors_map::create_anthropic_error;
use anyllm_translate::mapping::streaming_map::StreamingTranslator;
use anyllm_translate::openai::streaming::ChatCompletionChunk;
use anyllm_translate::{
    compute_request_warnings, new_stream_translator, translate_request, translate_response,
    TranslationConfig,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::GenerationConfig;

use crate::handler::{now_unix, plan, run_full, status_for, stream_blocking, AppState, GenError};
use crate::response::{
    completion_chunk, completion_response, finish_reason, role_delta, text_delta,
};

const DEGRADATION_HEADER: &str = "x-anyllm-degradation";

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

pub async fn messages(
    State(model): State<AppState>,
    Json(request): Json<MessageCreateRequest>,
) -> Response {
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

    let (prompt_ids, config) = match plan(&model, &openai) {
        Ok(p) => p,
        Err(e) => return error_body(StatusCode::BAD_REQUEST, ErrorType::InvalidRequestError, e),
    };

    let streaming = request.stream.unwrap_or(false);
    let mut response = if streaming {
        stream_response(model, prompt_ids, config, openai.model, request.model)
    } else {
        full_response(model, prompt_ids, config, openai.model, request.model).await
    };

    if let Some(value) = degraded.and_then(|v| v.parse().ok()) {
        response.headers_mut().insert(DEGRADATION_HEADER, value);
    }
    response
}

async fn full_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    backend_model: String,
    // The model name the client asked for, echoed into the Anthropic
    // response. Translation maps it to `backend_model` on the way in, and
    // that mapping is not necessarily reversible, so it is carried across
    // rather than derived.
    client_model: String,
) -> Response {
    let (text, decode) = match run_full(model, prompt_ids, config).await {
        Ok(r) => r,
        Err(e) => return gen_error_body(e),
    };

    let openai_response = completion_response(
        format!("chatcmpl-{}", now_unix()),
        now_unix(),
        backend_model,
        text,
        decode.reason,
        decode.prompt_tokens as u32,
        decode.new_tokens as u32,
    );
    Json(translate_response(&openai_response, &client_model)).into_response()
}

fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    backend_model: String,
    client_model: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();

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
                let _ = tx.send(to_sse(&event));
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

        let result = stream_blocking(&model, &prompt_ids, &config, &mut |delta| {
            send(
                completion_chunk(
                    id.clone(),
                    created,
                    backend_model.clone(),
                    text_delta(delta.to_string()),
                    None,
                ),
                &mut translator,
            );
        });

        match result {
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
    > = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok));
    Sse::new(stream).into_response()
}
