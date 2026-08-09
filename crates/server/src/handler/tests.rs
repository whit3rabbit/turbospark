//! Seam tests for prompt planning and structured output decoding.

use std::path::PathBuf;
use std::sync::Arc;

use anyllm_translate::openai::ChatCompletionRequest;
use tokenizer::MfTokenizer;

use super::*;
use crate::model::ScriptedChatModel;

fn state() -> AppState {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let tok = MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load");
    Arc::new(ScriptedChatModel::new(tok, 4096, Vec::new()))
}

/// The request as a client would send it, so the test covers the
/// deserialization shape too.
fn request(body: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("request body should deserialize")
}

fn prompt(model: &AppState, body: serde_json::Value) -> String {
    let (ids, _) = plan(model, &request(body)).expect("plan should succeed");
    model.tokenizer().decode(&ids, false)
}

fn weather_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    })
}

#[test]
fn an_assistant_turn_that_is_only_a_tool_call_survives() {
    let model = state();
    let rendered = prompt(
        &model,
        serde_json::json!({
            "model": "m",
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "weather in Oslo?"},
                // No `content` at all: this is what an Anthropic
                // assistant turn holding a lone `tool_use` block
                // translates to.
                {"role": "assistant", "tool_calls": [{
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}
                }]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "12C and raining"},
                {"role": "user", "content": "and tomorrow?"}
            ]
        }),
    );

    // The turn is in the prompt, with its call rendered, rather than
    // deleted out of the middle of the history.
    assert!(rendered.contains("<function=get_weather>"), "{rendered}");
    assert!(rendered.contains("Oslo"), "{rendered}");
    // Its tool result rendered as a tool turn, not merged away.
    assert!(rendered.contains("12C and raining"), "{rendered}");
    assert!(rendered.contains("<tool_response>"), "{rendered}");
}

#[test]
fn tools_are_rendered_into_the_prompt() {
    let model = state();
    let with = prompt(
        &model,
        serde_json::json!({
            "model": "m",
            "tools": [weather_tool()],
            "messages": [{"role": "user", "content": "hi"}]
        }),
    );
    assert!(with.contains("get_weather"), "{with}");
    assert!(with.contains("Look up the weather"), "{with}");
    assert!(with.contains("city"), "{with}");
}

#[test]
fn no_tools_keeps_the_text_only_template() {
    let model = state();
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "hi"}]
    });
    let (ids, _) = plan(&model, &request(body)).expect("plan should succeed");

    let messages = vec![
        tokenizer::Message::new(tokenizer::Role::System, "Be brief."),
        tokenizer::Message::new(tokenizer::Role::User, "hi"),
    ];
    let expected = model
        .tokenizer()
        .apply_chat_template(&messages)
        .expect("template should render");
    assert_eq!(ids, model.tokenizer().encode(&expected, false));
}

#[test]
fn reasoning_content_never_reaches_the_prompt() {
    let model = state();
    let rendered = prompt(
        &model,
        serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                // A replayed Anthropic `thinking` block. It is the
                // model's scratchpad, not something it said.
                {"role": "assistant", "reasoning_content": "SCRATCHPAD"},
                {"role": "user", "content": "again"}
            ]
        }),
    );
    assert!(!rendered.contains("SCRATCHPAD"), "{rendered}");
}

/// Drives a full generation whose output is a scripted Qwen tool call,
/// and checks the decoder turns it back into a structured call.
#[test]
fn a_generated_tool_call_is_decoded_out_of_the_stream() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    let tok = MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load");
    let body = serde_json::json!({
        "model": "m",
        "max_tokens": 128,
        "temperature": 0.0,
        "tools": [weather_tool()],
        "messages": [{"role": "user", "content": "weather in Oslo?"}]
    });
    let request = request(body);

    // The prompt has to be rendered first: the scripted producer
    // consumes one step per prefill token, so the decode steps only
    // line up if they start at exactly `prompt_ids.len() - 1`.
    let planning_model: AppState = Arc::new(ScriptedChatModel::new(
        MfTokenizer::load_from_dir(&dir).unwrap(),
        4096,
        Vec::new(),
    ));
    let (prompt_ids, config) = plan(&planning_model, &request).expect("plan should succeed");

    let mut ids = tok.encode("<tool_call>\n<function=get_weather>\n", false);
    ids.extend(tok.encode("<parameter=city>\nOslo\n</parameter>\n", false));
    ids.extend(tok.encode("</function>\n</tool_call>", false));
    // ChatML's stop set is `<|im_end|>` and `<|endoftext|>` only:
    // `StopReason::ToolCalls` fires on `tool_response_id`, which is a
    // Gemma-only stop (`dialect.rs`), so it belongs to the real-model
    // gate rather than here.
    ids.push(tok.end_of_turn_id);

    let mut steps = vec![one_hot(tok.vocab_size, 0); prompt_ids.len() - 1];
    steps.extend(ids.iter().map(|&id| one_hot(tok.vocab_size, id as usize)));

    let model: AppState = Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let mut pieces = Vec::new();
    let result = stream_blocking(
        &model,
        &prompt_ids,
        &config,
        &tool_names(&request),
        &mut |piece| pieces.push(piece),
    )
    .expect("generation should succeed");

    assert_eq!(result.reason, runtime::StopReason::EndOfTurn);
    let calls: Vec<&tokenizer::ParsedToolCall> = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Tool(c) => Some(c),
            Piece::Text(_) => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments_json, r#"{"city":"Oslo"}"#);
    // The markup itself is not content: nothing between the tool-call
    // markers leaks out as text.
    let text: String = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Text(t) => Some(t.as_str()),
            Piece::Tool(_) => None,
        })
        .collect();
    assert!(!text.contains("get_weather"), "{text}");
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}
