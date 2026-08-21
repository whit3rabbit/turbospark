use super::*;
use tokenizer::JsonValue;

fn call() -> ParsedToolCall {
    ParsedToolCall {
        id: "toolu_0".to_string(),
        name: "get_weather".to_string(),
        arguments: JsonValue::parse(r#"{"city":"Oslo"}"#).unwrap(),
        arguments_json: r#"{"city":"Oslo"}"#.to_string(),
    }
}

#[test]
fn a_call_becomes_an_anthropic_tool_use_block() {
    let response = completion_response(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        String::new(),
        String::new(),
        vec![call()],
        runtime::StopReason::ToolCalls,
        3,
        7,
    );
    let anthropic = anyllm_translate::translate_response(&response, "claude-sonnet-4-6");
    let body = serde_json::to_value(&anthropic).unwrap();

    let block = body["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("a tool_use block");
    assert_eq!(block["id"], "toolu_0");
    assert_eq!(block["name"], "get_weather");
    assert_eq!(block["input"]["city"], "Oslo");
    assert_eq!(body["stop_reason"], "tool_use");
}

#[test]
fn a_streamed_call_becomes_a_tool_use_content_block() {
    let mut translator = anyllm_translate::new_stream_translator("claude-sonnet-4-6".to_string());
    let mut names = Vec::new();
    let mut record = |chunk| {
        for event in translator.process_chunk(&chunk) {
            names.push(serde_json::to_value(&event).unwrap());
        }
    };

    record(completion_chunk(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        role_delta(),
        None,
    ));
    record(completion_chunk(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        tool_call_delta(0, call()),
        None,
    ));
    record(completion_chunk(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        ChunkDelta::default(),
        Some(FinishReason::ToolCalls),
    ));
    for event in translator.finish() {
        names.push(serde_json::to_value(&event).unwrap());
    }

    // One complete chunk still opens a tool_use block and carries its
    // arguments as an input_json delta.
    let start = names
        .iter()
        .find(|e| e["type"] == "content_block_start")
        .expect("a content_block_start");
    assert_eq!(start["content_block"]["type"], "tool_use");
    assert_eq!(start["content_block"]["name"], "get_weather");
    assert!(
        names
            .iter()
            .any(|e| e["delta"]["type"] == "input_json_delta"
                && e["delta"]["partial_json"] == r#"{"city":"Oslo"}"#),
        "{names:?}"
    );
    assert!(names.iter().any(|e| e["type"] == "content_block_stop"));
}

/// Separated reasoning becomes an Anthropic `thinking` block, BEFORE the
/// text block. This is the whole server-side claim of Harmony channel
/// decoding: fill `reasoning_content` and `anyllm_translate` does the
/// rest, so what is under test here is that the field really is the one
/// its mapping reads.
#[test]
fn reasoning_becomes_an_anthropic_thinking_block() {
    let response = completion_response(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        "Rayleigh scattering.".to_string(),
        "The user asks why the sky is blue.".to_string(),
        Vec::new(),
        runtime::StopReason::EndOfTurn,
        3,
        7,
    );
    let anthropic = anyllm_translate::translate_response(&response, "claude-sonnet-4-6");
    let body = serde_json::to_value(&anthropic).unwrap();

    let kinds: Vec<&str> = body["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, vec!["thinking", "text"], "{body}");
    assert_eq!(
        body["content"][0]["thinking"],
        "The user asks why the sky is blue."
    );
    assert_eq!(body["content"][1]["text"], "Rayleigh scattering.");
}

/// A generation with no reasoning must not grow an empty thinking block:
/// every family but `gpt-oss` produces none, and this is the guard that
/// their responses are unchanged.
#[test]
fn no_reasoning_leaves_the_response_shape_alone() {
    let response = completion_response(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        "plain answer".to_string(),
        String::new(),
        Vec::new(),
        runtime::StopReason::EndOfTurn,
        3,
        7,
    );
    assert!(response.choices[0].message.reasoning_content.is_none());

    let body = serde_json::to_value(anyllm_translate::translate_response(
        &response,
        "claude-sonnet-4-6",
    ))
    .unwrap();
    let kinds: Vec<&str> = body["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, vec!["text"], "{body}");
}

/// The streaming shape: reasoning deltas open a `thinking` content block
/// and the first text delta CLOSES it, so the two arrive as separate
/// blocks rather than as one run of text.
#[test]
fn streamed_reasoning_opens_a_thinking_block_that_text_closes() {
    let mut translator = anyllm_translate::new_stream_translator("claude-sonnet-4-6".to_string());
    let mut events = Vec::new();
    let mut record = |chunk| {
        for event in translator.process_chunk(&chunk) {
            events.push(serde_json::to_value(&event).unwrap());
        }
    };
    let chunk = |delta| completion_chunk("chatcmpl-1".to_string(), 0, "m".to_string(), delta, None);

    record(chunk(role_delta()));
    record(chunk(reasoning_delta("thinking out loud".to_string())));
    record(chunk(text_delta("the answer".to_string())));
    record(completion_chunk(
        "chatcmpl-1".to_string(),
        0,
        "m".to_string(),
        ChunkDelta::default(),
        Some(FinishReason::Stop),
    ));
    for event in translator.finish() {
        events.push(serde_json::to_value(&event).unwrap());
    }

    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| e["delta"]["type"].as_str())
        .collect();
    assert_eq!(deltas, vec!["thinking_delta", "text_delta"], "{events:?}");

    let opened: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .map(|e| e["content_block"]["type"].as_str().unwrap())
        .collect();
    assert_eq!(opened, vec!["thinking", "text"], "{events:?}");
}
