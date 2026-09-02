//! The OpenAI `POST /v1/responses` endpoint.
//!
//! Reuses `anyllm_translate`'s Responses wire types (`ResponsesRequest`,
//! `ResponsesResponse`, `ResponsesUsage`, `ResponsesStreamEvent`) -- the
//! vendored crate ships those for its own Anthropic<->Responses translation,
//! but nothing that maps Responses onto Chat Completions, which is what this
//! server actually generates through. So the mapping in both directions is
//! hand-written here: `responses_to_chat_request` folds a Responses request
//! down onto the same `ChatCompletionRequest` `handler::plan` already knows
//! how to render, and the non-streaming/streaming builders below turn a
//! generation back into Responses' own `output[]` item shapes.
//!
//! Scope: text and tool calling, matching the other two endpoints.
//! `previous_response_id` is refused with a 400 (this server is stateless --
//! there is no prior turn to continue) and `store` is accepted and reported
//! on `x-anyllm-degradation` rather than honoured. Reasoning is included as
//! its own output item on the NON-streaming path but dropped (and reported,
//! when the checkpoint would have produced any) on the streaming one: OpenAI
//! has no delta event shape for `/v1/responses` reasoning summaries, and
//! inventing one would be undocumented on both ends.

use std::collections::HashSet;
use std::time::Duration;

use anyllm_translate::mapping::responses_streaming_map::ResponsesStreamEvent;
use anyllm_translate::openai::responses::{
    ResponsesInput, ResponsesRequest, ResponsesResponse, ResponsesUsage,
};
use anyllm_translate::openai::{
    ChatCompletionRequest, ChatContent, ChatMessage, ChatRole, ChatTool, ChatToolChoice,
    FunctionCall, FunctionDef, Stop, ToolCall,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::StopReason;
use serde_json::{json, Value};
use tokenizer::{ChatDialect, ParsedToolCall, ReasoningEffort};

use crate::guardrails::run_guarded;
use crate::handler::{
    error_response, now_unix, plan, reasoning_effort, stream_blocking, tool_names, AppState,
    GenError, Generated, Piece,
};

const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

// ---------------------------------------------------------------------------
// Request mapping: Responses -> the ChatCompletionRequest handler::plan reads
// ---------------------------------------------------------------------------

fn chat_message(
    role: ChatRole,
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
) -> ChatMessage {
    ChatMessage {
        role,
        content: content.map(ChatContent::Text),
        name: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id,
        refusal: None,
        reasoning_content: None,
        thinking_blocks: None,
    }
}

/// `content` on a Responses input item is a plain string OR an array of
/// typed parts (`input_text` / `output_text` / `input_image` / ...). Only
/// the text-bearing parts are read; an image part has no `text` field and is
/// skipped by construction rather than by name -- this endpoint has no
/// vision path in v1, the same scope `/v1/completions` has none of tools.
fn item_text(item: &Value) -> Option<String> {
    match item.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}

/// One Responses input item to one `ChatMessage`. Unlike Chat Completions, a
/// tool call and its result are ROOT-LEVEL items here (`function_call`,
/// `function_call_output`), not content blocks on a message -- the same
/// flattened shape `anyllm_translate`'s own Anthropic<->Responses mapping
/// uses (`responses_message_map::convert_blocks_to_items`), read here rather
/// than re-derived.
fn item_to_message(item: &Value) -> Result<ChatMessage, String> {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match item_type {
        "message" => {
            let role = match item.get("role").and_then(|v| v.as_str()) {
                Some("user") => ChatRole::User,
                Some("assistant") => ChatRole::Assistant,
                Some("system") => ChatRole::System,
                Some("developer") => ChatRole::Developer,
                other => {
                    return Err(format!(
                        "unsupported message role in a Responses input item: {other:?}"
                    ))
                }
            };
            Ok(chat_message(role, item_text(item), Vec::new(), None))
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let call = ToolCall {
                id: call_id,
                call_type: "function".to_string(),
                function: FunctionCall { name, arguments },
            };
            Ok(chat_message(ChatRole::Assistant, None, vec![call], None))
        }
        "function_call_output" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let output = item
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            Ok(chat_message(
                ChatRole::Tool,
                Some(output),
                Vec::new(),
                Some(call_id),
            ))
        }
        other => Err(format!(
            "unsupported Responses input item type {other:?}: expected message, function_call, \
             or function_call_output"
        )),
    }
}

/// A Responses tool is FLAT (`{"type":"function","name":...,"parameters":...}`)
/// where Chat Completions nests the function under its own key
/// (`{"type":"function","function":{"name":...}}`). Same information, one
/// extra layer.
fn flat_tool_to_chat_tool(tool: &Value) -> Result<ChatTool, String> {
    let tool_type = tool
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("function");
    if tool_type != "function" {
        return Err(format!(
            "unsupported Responses tool type {tool_type:?}: only function tools are supported"
        ));
    }
    let name = tool
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "a Responses tool is missing its name".to_string())?
        .to_string();
    let description = tool
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let parameters = tool.get("parameters").cloned();
    Ok(ChatTool {
        tool_type: "function".to_string(),
        function: FunctionDef {
            name,
            description,
            parameters,
            strict: None,
        },
    })
}

/// Folds a Responses request down onto the `ChatCompletionRequest`
/// `handler::plan` already renders and shapes -- one template path, one
/// shaping path, for all three generation endpoints this crate serves.
fn responses_to_chat_request(request: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    // Stateless server, honest refusal: there is no PRIOR turn on disk to
    // continue, and silently starting a fresh conversation under the same
    // id would answer a different question than the one asked.
    if request.extra.contains_key("previous_response_id") {
        return Err(
            "previous_response_id is not supported: this server is stateless and keeps no prior \
             turn to continue"
                .to_string(),
        );
    }

    let mut messages = Vec::new();
    if let Some(instructions) = &request.instructions {
        messages.push(chat_message(
            ChatRole::System,
            Some(instructions.clone()),
            Vec::new(),
            None,
        ));
    }
    match &request.input {
        ResponsesInput::Text(text) => {
            messages.push(chat_message(
                ChatRole::User,
                Some(text.clone()),
                Vec::new(),
                None,
            ));
        }
        ResponsesInput::Items(items) => {
            for item in items {
                messages.push(item_to_message(item)?);
            }
        }
    }

    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(flat_tool_to_chat_tool)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    // `top_p` / `tool_choice` / `stop` have no field on `ResponsesRequest`
    // (unlike Chat Completions', which has explicit ones), so they arrive in
    // Responses' OWN extra map. Pulled out here into the explicit fields
    // `handler::plan` and `build_config` read, and removed from what is
    // forwarded so nothing reads a Responses-shaped value through the wrong
    // key twice.
    let mut extra = request.extra.clone();
    let top_p = extra
        .remove("top_p")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let stop = extra
        .remove("stop")
        .and_then(|v| serde_json::from_value::<Stop>(v).ok());
    let tool_choice = extra
        .remove("tool_choice")
        .and_then(|v| serde_json::from_value::<ChatToolChoice>(v).ok());

    Ok(ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_output_tokens,
        max_completion_tokens: None,
        temperature: request.temperature,
        top_p,
        stop,
        tools,
        tool_choice,
        stream: request.stream,
        stream_options: None,
        presence_penalty: None,
        frequency_penalty: None,
        response_format: None,
        user: None,
        parallel_tool_calls: None,
        // `top_k`, `repetition_penalty`, `seed`, `reasoning_effort`, `n`,
        // `logprobs`, and any other Responses `extra` field this server
        // reads by name pass through here unchanged -- the same map
        // `build_shaping` / `openai_request_warnings`-style readers expect.
        extra,
    })
}

// ---------------------------------------------------------------------------
// Response mapping: a generation -> Responses' output[] item shapes
// ---------------------------------------------------------------------------

fn reasoning_item(text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_0",
        "summary": [{"type": "summary_text", "text": text}],
    })
}

fn message_item(text: &str, status: &str) -> Value {
    json!({
        "type": "message",
        "id": "msg_0",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

fn function_call_item(call: &ParsedToolCall, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{}", call.id),
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments_json,
        "status": status,
    })
}

fn response_status(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "incomplete",
        _ => "completed",
    }
}

fn build_response(id: String, model: String, generated: Generated) -> ResponsesResponse {
    let mut output = Vec::new();
    if !generated.reasoning.is_empty() {
        output.push(reasoning_item(&generated.reasoning));
    }
    if !generated.text.is_empty() {
        output.push(message_item(&generated.text, "completed"));
    }
    for call in &generated.calls {
        output.push(function_call_item(call, "completed"));
    }
    let status = response_status(generated.decode.reason);
    let usage = ResponsesUsage {
        input_tokens: generated.decode.prompt_tokens as u32,
        output_tokens: generated.decode.new_tokens as u32,
        total_tokens: (generated.decode.prompt_tokens + generated.decode.new_tokens) as u32,
        input_token_details: None,
    };
    ResponsesResponse {
        id,
        response_type: "response".to_string(),
        model,
        output,
        usage: Some(usage),
        status: status.to_string(),
        extra: serde_json::Map::new(),
    }
}

// ---------------------------------------------------------------------------
// Warnings: what this endpoint accepts on the wire but does not act on
// ---------------------------------------------------------------------------

/// A checkpoint whose dialect would separate reasoning from its answer on
/// THIS request -- the narrower question the streaming path needs
/// (`handler::exec::needs_decoder`'s own condition is broader: it also fires
/// on tools alone, which forces decoding for call-parsing but does not mean
/// reasoning was produced).
fn may_produce_reasoning(model: &AppState, effort: ReasoningEffort) -> bool {
    let dialect = model.tokenizer().dialect;
    matches!(dialect, ChatDialect::Harmony | ChatDialect::MuseGlimmer)
        || (effort != ReasoningEffort::Off
            && matches!(dialect, ChatDialect::ChatMl | ChatDialect::Gemma))
}

fn responses_warnings(request: &ResponsesRequest, reasoning_dropped: bool) -> Option<String> {
    let mut warnings = anyllm_translate::TranslationWarnings::default();
    if request
        .extra
        .get("store")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        warnings.add("store");
    }
    if reasoning_dropped {
        warnings.add("reasoning");
    }
    warnings.as_header_value()
}

// ---------------------------------------------------------------------------
// SSE event construction
// ---------------------------------------------------------------------------

/// One typed Responses stream event, built through `anyllm_translate`'s own
/// `ResponsesStreamEvent` (the same type its Responses->Anthropic translator
/// consumes) rather than a bespoke shape: `{"type": event_type, ...fields}`
/// on the wire, `event: event_type` / `data: ...` as the two SSE lines.
fn sse_event(event_type: &'static str, fields: Vec<(&'static str, Value)>) -> Event {
    let event = ResponsesStreamEvent {
        event_type: event_type.to_string(),
        data: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    };
    Event::default()
        .event(event_type)
        .data(serde_json::to_string(&event).unwrap_or_default())
}

fn created_event(id: &str, model: &str) -> Event {
    sse_event(
        "response.created",
        vec![(
            "response",
            json!({"id": id, "object": "response", "status": "in_progress", "model": model}),
        )],
    )
}

fn completed_event(id: &str, model: &str, decode: &runtime::RawDecodeResult) -> Event {
    sse_event(
        "response.completed",
        vec![(
            "response",
            json!({
                "id": id, "object": "response", "model": model,
                "status": response_status(decode.reason),
                "usage": {
                    "input_tokens": decode.prompt_tokens,
                    "output_tokens": decode.new_tokens,
                    "total_tokens": decode.prompt_tokens + decode.new_tokens,
                },
            }),
        )],
    )
}

fn failed_event(message: &str) -> Event {
    sse_event(
        "response.failed",
        vec![(
            "response",
            json!({"status": "failed", "status_details": {"error": {"message": message}}}),
        )],
    )
}

fn open_message_event() -> Event {
    sse_event(
        "response.output_item.added",
        vec![
            ("output_index", json!(0)),
            (
                "item",
                json!({"id": "msg_0", "type": "message", "role": "assistant", "status": "in_progress", "content": []}),
            ),
        ],
    )
}

fn content_part_added_event() -> Event {
    sse_event(
        "response.content_part.added",
        vec![
            ("item_id", json!("msg_0")),
            ("output_index", json!(0)),
            ("content_index", json!(0)),
            (
                "part",
                json!({"type": "output_text", "text": "", "annotations": []}),
            ),
        ],
    )
}

fn text_delta_event(delta: &str) -> Event {
    sse_event(
        "response.output_text.delta",
        vec![
            ("item_id", json!("msg_0")),
            ("output_index", json!(0)),
            ("content_index", json!(0)),
            ("delta", json!(delta)),
        ],
    )
}

/// `response.output_text.done` -> `response.content_part.done` ->
/// `response.output_item.done`, in that order -- the closing mirror of
/// `open_message_event` / `content_part_added_event`, both of which this
/// pairs with.
fn close_message_events(send: &impl Fn(Event), text: &str) {
    send(sse_event(
        "response.output_text.done",
        vec![
            ("item_id", json!("msg_0")),
            ("output_index", json!(0)),
            ("content_index", json!(0)),
            ("text", json!(text)),
        ],
    ));
    send(sse_event(
        "response.content_part.done",
        vec![
            ("item_id", json!("msg_0")),
            ("output_index", json!(0)),
            ("content_index", json!(0)),
            (
                "part",
                json!({"type": "output_text", "text": text, "annotations": []}),
            ),
        ],
    ));
    send(sse_event(
        "response.output_item.done",
        vec![
            ("output_index", json!(0)),
            ("item", message_item(text, "completed")),
        ],
    ));
}

/// A tool call arrives from the decoder as ONE parsed unit (name and
/// arguments both known at once, `crate::response::tool_call_delta`'s same
/// reason), so its four events -- `output_item.added`,
/// `function_call_arguments.delta`, `function_call_arguments.done`,
/// `output_item.done` -- fire back to back rather than the argument-by-
/// argument-fragment sequence a remote OpenAI backend streams.
fn function_call_events(send: &impl Fn(Event), call: &ParsedToolCall, output_index: u32) {
    let item_id = format!("fc_{}", call.id);
    send(sse_event(
        "response.output_item.added",
        vec![
            ("output_index", json!(output_index)),
            (
                "item",
                json!({"id": item_id, "type": "function_call", "call_id": call.id, "name": call.name, "arguments": "", "status": "in_progress"}),
            ),
        ],
    ));
    send(sse_event(
        "response.function_call_arguments.delta",
        vec![
            ("item_id", json!(item_id)),
            ("output_index", json!(output_index)),
            ("delta", json!(call.arguments_json)),
        ],
    ));
    send(sse_event(
        "response.function_call_arguments.done",
        vec![
            ("item_id", json!(item_id)),
            ("output_index", json!(output_index)),
            ("arguments", json!(call.arguments_json)),
        ],
    ));
    send(sse_event(
        "response.output_item.done",
        vec![
            ("output_index", json!(output_index)),
            ("item", function_call_item(call, "completed")),
        ],
    ));
}

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
        // resolves; defused right after, whatever it returned.
        let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
        let result = run_guarded(model, &chat_request, effort, cancel).await;
        guard.defuse();
        match result {
            Ok(generated) => Json(build_response(
                format!("resp_{}", now_unix()),
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
    let id = format!("resp_{}", now_unix());
    let model_name = request_model;
    let cancel_for_task = cancel.clone();

    tokio::task::spawn_blocking(move || {
        let send = |ev: Event| {
            // Fast-path disconnect detection: `CancelOnDrop` below catches
            // the same event even with no chunk in flight (a long prefill
            // sends none), so this is a speed-up rather than the only
            // detector.
            if tx.send(ev).is_err() {
                cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        };
        send(created_event(&id, &model_name));

        let mut message_open = false;
        let mut text_acc = String::new();
        let mut call_index: u32 = 0;

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
                // checkpoint's dialect meant one would be produced.
                Piece::Reasoning(_) => {}
                Piece::Text(delta) => {
                    if delta.is_empty() {
                        return;
                    }
                    if !message_open {
                        send(open_message_event());
                        send(content_part_added_event());
                        message_open = true;
                    }
                    text_acc.push_str(&delta);
                    send(text_delta_event(&delta));
                }
                Piece::Tool(call) => {
                    if message_open {
                        close_message_events(&send, &text_acc);
                        message_open = false;
                    }
                    function_call_events(&send, &call, 1 + call_index);
                    call_index += 1;
                }
            },
        );

        match result {
            // Discarded silently: the client that would read
            // `response.completed` is the one already gone.
            Ok(decode) if decode.reason == runtime::StopReason::Cancelled => (),
            Ok(decode) => {
                if message_open {
                    close_message_events(&send, &text_acc);
                }
                send(completed_event(&id, &model_name, &decode));
            }
            // A failed run is not a completed one: report it as a
            // `response.failed` event rather than a fabricated `completed`
            // status, matching the other two endpoints' contract.
            Err(e) => send(failed_event(&e.to_string())),
        }
        // No `[DONE]` sentinel: an OpenAI Responses stream ends at
        // `response.completed` (or `response.failed`), the same shape
        // Anthropic's own stream takes and for the same reason -- `[DONE]`
        // is a Chat-Completions-ism.
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
/// syntax error it was rescued from.
fn buffered_stream_response(
    model: AppState,
    chat_request: ChatCompletionRequest,
    effort: ReasoningEffort,
    request_model: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = format!("resp_{}", now_unix());
    let model_name = request_model;
    let cancel_for_stream = cancel.clone();

    tokio::spawn(async move {
        let send = move |ev: Event| {
            let _ = tx.send(ev);
        };
        send(created_event(&id, &model_name));

        // Nothing is sent until the whole generation is done, so
        // `CancelOnDrop` below is the only signal this path has that the
        // client is gone.
        match run_guarded(model, &chat_request, effort, cancel).await {
            // Discarded silently, same contract as the live path.
            Ok(generated) if generated.decode.reason == runtime::StopReason::Cancelled => (),
            Ok(generated) => {
                if !generated.text.is_empty() {
                    send(open_message_event());
                    send(content_part_added_event());
                    send(text_delta_event(&generated.text));
                    close_message_events(&send, &generated.text);
                }
                for (index, call) in generated.calls.iter().enumerate() {
                    function_call_events(&send, call, 1 + index as u32);
                }
                send(completed_event(&id, &model_name, &generated.decode));
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

/// Pure mapping tests: no model needed for request shape decisions
/// (`responses_to_chat_request`, `flat_tool_to_chat_tool`, `item_to_message`)
/// or for the item/event constructors, which take their inputs directly.
#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).expect("request body should deserialize")
    }

    #[test]
    fn a_text_input_becomes_one_user_message() {
        let r = request(json!({"model": "m", "input": "hi"}));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, ChatRole::User);
        assert_eq!(chat.messages[0].effective_text().as_deref(), Some("hi"));
    }

    #[test]
    fn instructions_become_a_leading_system_message() {
        let r = request(json!({"model": "m", "input": "hi", "instructions": "be terse"}));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, ChatRole::System);
        assert_eq!(
            chat.messages[0].effective_text().as_deref(),
            Some("be terse")
        );
        assert_eq!(chat.messages[1].role, ChatRole::User);
    }

    #[test]
    fn a_message_item_array_input_round_trips_role_and_text() {
        let r = request(json!({
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        }));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, ChatRole::User);
        assert_eq!(chat.messages[0].effective_text().as_deref(), Some("hi"));
    }

    #[test]
    fn a_function_call_item_becomes_an_assistant_tool_call() {
        let r = request(json!({
            "model": "m",
            "input": [{
                "type": "function_call", "call_id": "call_1",
                "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"
            }]
        }));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, ChatRole::Assistant);
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn a_function_call_output_item_becomes_a_tool_message() {
        let r = request(json!({
            "model": "m",
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "12C"}]
        }));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, ChatRole::Tool);
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[0].effective_text().as_deref(), Some("12C"));
    }

    #[test]
    fn an_unknown_item_type_is_refused() {
        let r = request(json!({"model": "m", "input": [{"type": "reasoning", "summary": []}]}));
        let err = responses_to_chat_request(&r).unwrap_err();
        assert!(err.contains("reasoning"), "{err}");
    }

    #[test]
    fn previous_response_id_is_refused() {
        let r = request(json!({"model": "m", "input": "hi", "previous_response_id": "resp_1"}));
        let err = responses_to_chat_request(&r).unwrap_err();
        assert!(err.contains("previous_response_id"), "{err}");
    }

    #[test]
    fn a_flat_tool_becomes_a_nested_chat_tool() {
        let r = request(json!({
            "model": "m", "input": "hi",
            "tools": [{"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}]
        }));
        let chat = responses_to_chat_request(&r).unwrap();
        let tools = chat.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].function.description.as_deref(), Some("d"));
    }

    #[test]
    fn top_p_tool_choice_and_stop_arrive_via_extra() {
        let r = request(json!({
            "model": "m", "input": "hi",
            "top_p": 0.5, "tool_choice": "none", "stop": "zzz"
        }));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.top_p, Some(0.5));
        assert!(matches!(chat.tool_choice, Some(ChatToolChoice::Simple(ref s)) if s == "none"));
        assert!(matches!(chat.stop, Some(Stop::Single(ref s)) if s == "zzz"));
        // Consumed out of extra, not left duplicated there.
        assert!(!chat.extra.contains_key("top_p"));
        assert!(!chat.extra.contains_key("tool_choice"));
        assert!(!chat.extra.contains_key("stop"));
    }

    #[test]
    fn max_output_tokens_becomes_max_tokens() {
        let r = request(json!({"model": "m", "input": "hi", "max_output_tokens": 200}));
        let chat = responses_to_chat_request(&r).unwrap();
        assert_eq!(chat.max_tokens, Some(200));
    }

    #[test]
    fn store_true_is_reported_store_false_and_absent_are_not() {
        let plain = request(json!({"model": "m", "input": "hi"}));
        assert_eq!(responses_warnings(&plain, false), None);
        let off = request(json!({"model": "m", "input": "hi", "store": false}));
        assert_eq!(responses_warnings(&off, false), None);
        let on = request(json!({"model": "m", "input": "hi", "store": true}));
        assert_eq!(responses_warnings(&on, false), Some("store".to_string()));
    }

    #[test]
    fn reasoning_dropped_is_reported_only_when_asked_for() {
        let r = request(json!({"model": "m", "input": "hi"}));
        assert_eq!(responses_warnings(&r, false), None);
        assert_eq!(responses_warnings(&r, true), Some("reasoning".to_string()));
    }

    #[test]
    fn build_response_orders_reasoning_before_message_before_calls() {
        let generated = Generated {
            text: "hi there".to_string(),
            reasoning: "thinking...".to_string(),
            calls: vec![ParsedToolCall {
                id: "toolu_0".to_string(),
                name: "get_weather".to_string(),
                arguments: tokenizer::JsonValue::Null,
                arguments_json: "{}".to_string(),
            }],
            decode: runtime::RawDecodeResult {
                prompt_tokens: 3,
                new_tokens: 5,
                prefill_seconds: 0.0,
                decode_seconds: 0.0,
                reason: StopReason::EndOfTurn,
                kv_position: 0,
                kv_backed_token_ids: Vec::new(),
                reused_prefix_tokens: 0,
                session_slot_evicted: false,
                peak_memory_pressure: Default::default(),
            },
        };
        let response = build_response("resp_1".to_string(), "m".to_string(), generated);
        assert_eq!(response.status, "completed");
        assert_eq!(response.output.len(), 3);
        assert_eq!(response.output[0]["type"], "reasoning");
        assert_eq!(response.output[1]["type"], "message");
        assert_eq!(response.output[2]["type"], "function_call");
        assert_eq!(response.usage.unwrap().output_tokens, 5);
    }

    #[test]
    fn max_tokens_stop_reason_reports_incomplete() {
        assert_eq!(response_status(StopReason::MaxTokens), "incomplete");
        assert_eq!(response_status(StopReason::EndOfTurn), "completed");
    }
}
