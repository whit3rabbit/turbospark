//! Pure mapping and serialization tests for the Responses endpoint.
//!
//! No model needed for request shape decisions (`responses_to_chat_request`,
//! `flat_tool_to_chat_tool`, `item_to_message`) or for the item/event
//! constructors, which take their inputs directly.

use anyllm_translate::openai::responses::ResponsesRequest;
use anyllm_translate::openai::{ChatRole, ChatToolChoice, Stop};
use runtime::StopReason;
use serde_json::{json, Value};
use tokenizer::ParsedToolCall;

use super::map::*;
use super::sse::*;
use crate::handler::Generated;

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

/// The Responses-native forced-tool shape is FLAT
/// (`{"type":"function","name":...}`), unlike Chat Completions' nested
/// `{"type":"function","function":{"name":...}}`. Deserializing the flat
/// shape straight into `ChatToolChoice` fails silently under `.ok()`, which
/// is how a forced call used to vanish with no error and the model answered
/// in prose instead of calling the tool it was told to.
#[test]
fn a_flat_forced_tool_choice_is_recognised() {
    let r = request(json!({
        "model": "m", "input": "hi",
        "tool_choice": {"type": "function", "name": "get_weather"}
    }));
    let chat = responses_to_chat_request(&r).unwrap();
    match chat.tool_choice {
        Some(ChatToolChoice::Named(named)) => {
            assert_eq!(named.function.name, "get_weather");
        }
        other => panic!("expected a named tool choice, got {other:?}"),
    }
}

/// The nested Chat Completions shape must still work for a client that sends
/// it directly against this endpoint.
#[test]
fn a_nested_forced_tool_choice_is_also_recognised() {
    let r = request(json!({
        "model": "m", "input": "hi",
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    }));
    let chat = responses_to_chat_request(&r).unwrap();
    match chat.tool_choice {
        Some(ChatToolChoice::Named(named)) => {
            assert_eq!(named.function.name, "get_weather");
        }
        other => panic!("expected a named tool choice, got {other:?}"),
    }
}

/// An unparseable `tool_choice`, `stop`, or `top_p` must 400 rather than be
/// silently dropped -- the `.ok()` this replaced made a malformed value
/// indistinguishable from an absent one.
#[test]
fn an_unparseable_tool_choice_stop_or_top_p_is_refused() {
    let bad_tool_choice = request(json!({
        "model": "m", "input": "hi", "tool_choice": {"type": "function"}
    }));
    assert!(responses_to_chat_request(&bad_tool_choice).is_err());

    let bad_stop = request(json!({"model": "m", "input": "hi", "stop": 5}));
    assert!(responses_to_chat_request(&bad_stop).is_err());

    let bad_top_p = request(json!({"model": "m", "input": "hi", "top_p": "high"}));
    assert!(responses_to_chat_request(&bad_top_p).is_err());
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
