//! Response construction and typed SSE stream events for the OpenAI Responses endpoint.
//!
//! Re-frames model generations into Responses `output[]` item shapes and
//! builds the typed SSE event sequence (`response.created`, `response.completed`,
//! `response.output_item.added`, `response.output_text.delta`, etc.)
//! per `crates/server/CLAUDE.md` Gotcha 24.

use anyllm_translate::mapping::responses_streaming_map::ResponsesStreamEvent;
use anyllm_translate::openai::responses::{ResponsesResponse, ResponsesUsage};
use axum::response::sse::Event;
use runtime::StopReason;
use serde_json::{json, Value};
use tokenizer::ParsedToolCall;

use crate::handler::Generated;

// ---------------------------------------------------------------------------
// Response mapping: a generation -> Responses' output[] item shapes
// ---------------------------------------------------------------------------

pub(crate) fn reasoning_item(text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_0",
        "summary": [{"type": "summary_text", "text": text}],
    })
}

pub(crate) fn message_item(text: &str, status: &str) -> Value {
    json!({
        "type": "message",
        "id": "msg_0",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

pub(crate) fn function_call_item(call: &ParsedToolCall, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{}", call.id),
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments_json,
        "status": status,
    })
}

pub(crate) fn response_status(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "incomplete",
        _ => "completed",
    }
}

pub(crate) fn build_response(id: String, model: String, generated: Generated) -> ResponsesResponse {
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

pub(crate) fn created_event(id: &str, model: &str) -> Event {
    sse_event(
        "response.created",
        vec![(
            "response",
            json!({"id": id, "object": "response", "status": "in_progress", "model": model}),
        )],
    )
}

pub(crate) fn completed_event(id: &str, model: &str, decode: &runtime::RawDecodeResult) -> Event {
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

pub(crate) fn failed_event(message: &str) -> Event {
    sse_event(
        "response.failed",
        vec![(
            "response",
            json!({"status": "failed", "status_details": {"error": {"message": message}}}),
        )],
    )
}

pub(crate) fn open_message_event() -> Event {
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

pub(crate) fn content_part_added_event() -> Event {
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

pub(crate) fn text_delta_event(delta: &str) -> Event {
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
/// pairs with (see `crates/server/CLAUDE.md` Gotcha 24).
pub(crate) fn close_message_events(send: &impl Fn(Event), text: &str) {
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
pub(crate) fn function_call_events(
    send: &impl Fn(Event),
    call: &ParsedToolCall,
    output_index: u32,
) {
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
