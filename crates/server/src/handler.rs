//! The `/v1/chat/completions` and `/v1/models` handlers, plus the generation
//! core both this endpoint and `/v1/messages` are built on: [`plan`] turns a
//! request into prompt tokens and a generation config, [`run_full`] runs it to
//! a `String`, and [`stream_blocking`] runs it emitting deltas as they arrive.
//!
//! Generation runs on a blocking task (it is synchronous CPU/GPU work, not
//! I/O). The request envelope is `anyllm_translate`'s OpenAI type rather than
//! a local struct; see `response.rs` for why.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyllm_translate::openai::{
    ChatCompletionRequest, ChatMessage, ChatRole, ChatTool, Stop, ToolCall,
};
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, RawDecodeResult, RuntimeError,
};
use selection::ShapingConfig;
use tokenizer::{
    FunctionDefinition, HistoricalToolCall, JsonValue, Message, ParsedToolCall, Role,
    StructuredAssistantDecoder, StructuredAssistantEvent,
};

use crate::model::ChatModel;
use crate::response::{
    completion_chunk, completion_response, role_delta, text_delta, tool_call_delta,
};

pub type AppState = Arc<dyn ChatModel>;

pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn role_from(role: &ChatRole) -> Role {
    match role {
        ChatRole::System => Role::System,
        ChatRole::Assistant => Role::Assistant,
        // `function` is OpenAI's deprecated spelling of a tool result.
        ChatRole::Tool | ChatRole::Function => Role::Tool,
        ChatRole::Developer => Role::Developer,
        ChatRole::User => Role::User,
    }
}

/// `stop` is untagged on the wire: a bare string or an array.
fn stop_strings(stop: Option<&Stop>) -> Vec<String> {
    match stop {
        Some(Stop::Single(s)) => vec![s.clone()],
        Some(Stop::Multiple(v)) => v.clone(),
        None => Vec::new(),
    }
}

fn build_config(request: &ChatCompletionRequest) -> Result<GenerationConfig, String> {
    // top_k defaults to the CLI's 64 rather than 0 (unbounded): `ShapingConfig`
    // rejects a top_p below 1.0 when top_k is 0, so a plain OpenAI request
    // carrying only top_p would otherwise be a 400.
    let shaping = ShapingConfig::new(
        request.temperature.map(f64::from).unwrap_or(1.0),
        64,
        request.top_p.map(f64::from),
        1.0,
        // `seed` has no field on the OpenAI request type: it needs no
        // translation, so it lands in the `extra` flatten map. Reading it back
        // out is not optional -- a missing lookup here silently unseeds every
        // seeded request, with no deserialization error to catch it.
        request.extra.get("seed").and_then(|v| v.as_u64()),
    )
    .map_err(|e| e.to_string())?;
    Ok(GenerationConfig {
        shaping,
        max_new_tokens: request
            .max_tokens
            .or(request.max_completion_tokens)
            .unwrap_or(256),
        stop_strings: stop_strings(request.stop.as_ref()),
        extra_stop_tokens: Vec::new(),
    })
}

/// The text `content` carries, with no fallback to `reasoning_content`.
///
/// `ChatMessage::effective_text` falls back to it, which is wrong here: a
/// replayed Anthropic `thinking` block would be rendered as ordinary
/// assistant prose, showing the model its own scratchpad as if it had said
/// it out loud. Everything else that method does (join the text parts of a
/// multipart content, skip images) is what this server wants.
fn visible_text(message: &ChatMessage) -> Option<String> {
    let mut stripped = message.clone();
    stripped.reasoning_content = None;
    stripped.effective_text()
}

/// A historical tool call, with its arguments parsed back out of the JSON
/// string OpenAI carries them in. A blob that will not parse is passed
/// through as a string rather than rejected: the Gemma template renders
/// `arguments` as a mapping OR as a bare string, so this still renders.
fn historical_call(call: &ToolCall) -> HistoricalToolCall {
    HistoricalToolCall {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments: JsonValue::parse(&call.function.arguments)
            .unwrap_or_else(|_| JsonValue::String(call.function.arguments.clone())),
    }
}

/// One request message as a `tokenizer::Message`, or `None` if it carries
/// nothing to render (an image-only turn, say).
///
/// Every tool field is copied across. `tool_call_id` in particular is load
/// bearing: the Gemma template resolves a `tool` turn's function name by
/// matching that id against the preceding assistant message's `tool_calls`,
/// and renders `unknown` when it cannot.
fn to_message(message: &ChatMessage) -> Option<Message> {
    let content = visible_text(message);
    let tool_calls: Vec<HistoricalToolCall> = message
        .tool_calls
        .iter()
        .flatten()
        .map(historical_call)
        .collect();
    // An assistant turn that is ONLY a tool call has no content at all.
    // Dropping it would delete a turn from the middle of the history and
    // break user/model alternation.
    if content.is_none() && tool_calls.is_empty() {
        return None;
    }
    Some(Message {
        role: role_from(&message.role),
        content,
        tool_calls,
        tool_call_id: message.tool_call_id.clone(),
        name: message.name.clone(),
    })
}

fn tool_definition(tool: &ChatTool) -> FunctionDefinition {
    FunctionDefinition {
        name: tool.function.name.clone(),
        description: tool.function.description.clone().unwrap_or_default(),
        parameters: tool
            .function
            .parameters
            .as_ref()
            .and_then(|p| JsonValue::parse(&p.to_string()).ok())
            .unwrap_or(JsonValue::Null),
    }
}

/// The tool names the structured decoder will accept in generated output.
/// Kept out of [`plan`]'s return so its signature stays a pair; both call
/// sites need one line either way.
pub(crate) fn tool_names(request: &ChatCompletionRequest) -> HashSet<String> {
    request
        .tools
        .iter()
        .flatten()
        .map(|t| t.function.name.clone())
        .collect()
}

/// Renders the chat template, encodes it, and resolves the shaping config.
pub(crate) fn plan(
    model: &AppState,
    request: &ChatCompletionRequest,
) -> Result<(Vec<foundation::TokenId>, GenerationConfig), String> {
    let messages: Vec<Message> = request.messages.iter().filter_map(to_message).collect();
    let tools: Vec<FunctionDefinition> = request
        .tools
        .iter()
        .flatten()
        .map(tool_definition)
        .collect();

    // With tools, the checkpoint's own `chat_template.jinja` is the only
    // renderer that can express them (it already speaks OpenAI's shape:
    // `tool_calls` on an assistant turn, a forward scan of `tool` turns).
    // Without them, the text-only path stays exactly as it was.
    let prompt_ids = if tools.is_empty() {
        let prompt = model
            .tokenizer()
            .apply_chat_template(&messages)
            .map_err(|e| e.to_string())?;
        // `add_bos` is false on purpose: the Gemma template emits the literal
        // `<bos>` mark itself, so encoding with a BOS prefix would double it
        // (the CLI's proven path does the same).
        model.tokenizer().encode(&prompt, false)
    } else {
        model
            .tokenizer()
            .encode_generic_tool_chat(&messages, &tools, false)
            .map_err(|e| e.to_string())?
    };

    Ok((prompt_ids, build_config(request)?))
}

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
        let result = stream_blocking(&model, &prompt_ids, &config, &tools, &mut |piece| {
            match piece {
                Piece::Text(delta) => text.push_str(&delta),
                Piece::Tool(call) => calls.push(call),
            }
        });
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
        RuntimeError::Producer(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
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

pub async fn chat_completions(
    State(model): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let (prompt_ids, config) = match plan(&model, &request) {
        Ok(p) => p,
        Err(e) => return error_response(axum::http::StatusCode::BAD_REQUEST, e),
    };

    if request.stream.unwrap_or(false) {
        return stream_response(model, prompt_ids, config, request.model);
    }

    let (text, decode) = match run_full(model, prompt_ids, config).await {
        Ok(r) => r,
        Err(e) => return gen_error_response(e),
    };

    Json(completion_response(
        format!("chatcmpl-{}", now_unix()),
        now_unix(),
        request.model,
        text,
        decode.reason,
        decode.prompt_tokens as u32,
        decode.new_tokens as u32,
    ))
    .into_response()
}

fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    model_name: String,
) -> Response {
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

        let result = stream_blocking(&model, &prompt_ids, &config, &mut |delta| {
            send(completion_chunk(
                id.clone(),
                created,
                model_name.clone(),
                text_delta(delta.to_string()),
                None,
            ));
        });

        match result {
            Ok(r) => send(completion_chunk(
                id.clone(),
                created,
                model_name.clone(),
                Default::default(),
                Some(crate::response::finish_reason(r.reason)),
            )),
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
