//! The `/v1/chat/completions` handler: builds the prompt via the tokenizer's
//! chat template, runs the raw-completion loop on a blocking task (it's
//! synchronous CPU/GPU work, not I/O), and renders either the full
//! response body or an SSE stream of chunks depending on `stream`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::{run_raw_completion, GenerationConfig, RawDecodeProgress, RuntimeError};
use selection::ShapingConfig;
use tokenizer::{Message, Role};

use crate::model::ChatModel;
use crate::request::ChatCompletionRequest;
use crate::response::{
    finish_reason, ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, DeltaMessage,
    ResponseMessage, Usage,
};

pub type AppState = Arc<dyn ChatModel>;

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn role_from_str(role: &str) -> Role {
    match role {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        "developer" => Role::Developer,
        _ => Role::User,
    }
}

fn build_config(request: &ChatCompletionRequest) -> Result<GenerationConfig, String> {
    let shaping = ShapingConfig::new(
        request.temperature.unwrap_or(1.0),
        0,
        request.top_p,
        1.0,
        request.seed,
    )
    .map_err(|e| e.to_string())?;
    Ok(GenerationConfig {
        shaping,
        max_new_tokens: request.max_tokens.unwrap_or(256),
        stop_strings: request.stop.clone().unwrap_or_default(),
        extra_stop_tokens: Vec::new(),
    })
}

fn error_response(status: axum::http::StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

pub async fn chat_completions(
    State(model): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let messages: Vec<Message> = request
        .messages
        .iter()
        .map(|m| Message::new(role_from_str(&m.role), m.content.clone()))
        .collect();

    let prompt = match model.tokenizer().apply_chat_template(&messages) {
        Ok(p) => p,
        Err(e) => return error_response(axum::http::StatusCode::BAD_REQUEST, e.to_string()),
    };
    let prompt_ids = model.tokenizer().encode(&prompt, true);

    let config = match build_config(&request) {
        Ok(c) => c,
        Err(e) => return error_response(axum::http::StatusCode::BAD_REQUEST, e),
    };

    if request.stream {
        return stream_response(model, prompt_ids, config, request.model).await;
    }

    let result = tokio::task::spawn_blocking(move || {
        let mut producer = model.new_producer();
        let mut text = String::new();
        let result = run_raw_completion(
            producer.as_mut(),
            model.tokenizer(),
            &prompt_ids,
            &config,
            model.max_context(),
            model.vocab_size(),
            |e| {
                if let RawDecodeProgress::Token { delta, .. } = e {
                    text.push_str(&delta);
                } else if let RawDecodeProgress::Tail(tail) = e {
                    text.push_str(&tail);
                }
            },
        );
        (result, text)
    })
    .await;

    let (result, text) = match result {
        Ok(r) => r,
        Err(e) => {
            return error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
    };

    match result {
        Ok(decode) => Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", now_unix()),
            object: "chat.completion",
            created: now_unix(),
            model: request.model,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: finish_reason(decode.reason).to_string(),
            }],
            usage: Usage {
                prompt_tokens: decode.prompt_tokens as u32,
                completion_tokens: decode.new_tokens as u32,
                total_tokens: (decode.prompt_tokens + decode.new_tokens) as u32,
            },
        })
        .into_response(),
        Err(e) => error_response(status_for(&e), e.to_string()),
    }
}

fn status_for(e: &RuntimeError) -> axum::http::StatusCode {
    match e {
        RuntimeError::EmptyPrompt
        | RuntimeError::ContextOverflow { .. }
        | RuntimeError::Selection(_) => axum::http::StatusCode::BAD_REQUEST,
        RuntimeError::Producer(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    model_name: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("chatcmpl-{}", now_unix());
    let created = now_unix();

    tokio::task::spawn_blocking(move || {
        let mut producer = model.new_producer();
        let send_chunk = |delta: DeltaMessage, finish: Option<String>| {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta,
                    finish_reason: finish,
                }],
            };
            let _ =
                tx.send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
        };
        send_chunk(
            DeltaMessage {
                role: Some("assistant".to_string()),
                content: None,
            },
            None,
        );

        let result = run_raw_completion(
            producer.as_mut(),
            model.tokenizer(),
            &prompt_ids,
            &config,
            model.max_context(),
            model.vocab_size(),
            |e| match e {
                RawDecodeProgress::Token { delta, .. } if !delta.is_empty() => {
                    send_chunk(
                        DeltaMessage {
                            role: None,
                            content: Some(delta),
                        },
                        None,
                    );
                }
                RawDecodeProgress::Tail(tail) if !tail.is_empty() => {
                    send_chunk(
                        DeltaMessage {
                            role: None,
                            content: Some(tail),
                        },
                        None,
                    );
                }
                _ => {}
            },
        );
        let reason = result
            .map(|r| finish_reason(r.reason))
            .unwrap_or("stop")
            .to_string();
        send_chunk(
            DeltaMessage {
                role: None,
                content: None,
            },
            Some(reason),
        );
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(Ok));
    Sse::new(stream).into_response()
}
