//! Seam tests for prompt planning and structured output decoding.

use std::path::PathBuf;
use std::sync::Arc;

use anyllm_translate::openai::ChatCompletionRequest;
use tokenizer::{MfTokenizer, ReasoningEffort};

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
    let planned = plan(model, &request(body)).expect("plan should succeed");
    model.tokenizer().decode(&planned.prompt_ids, false)
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

/// `reasoning_effort` arrives in the flatten map (it has no field on the
/// OpenAI request type) and reaches the rendered prompt from there.
///
/// The fixture's template is `ToggleOnly`, so what moves is the pre-closed
/// `<think>` block rather than a level -- which is the point: the assertion
/// is that the request-side value reaches the RENDER, not that this
/// particular checkpoint can spell a level.
#[test]
fn reasoning_effort_arrives_in_extra_and_reaches_the_prompt() {
    let model = state();
    let plain = prompt(
        &model,
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    );
    let thinking = prompt(
        &model,
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        }),
    );
    assert_ne!(
        plain, thinking,
        "reasoning_effort reached no part of the prompt"
    );
    assert!(
        plain.contains("<think>\n\n</think>"),
        "the default must stay the pre-closed branch: {plain:?}"
    );
    assert!(!thinking.contains("<think>\n\n</think>"));
}

/// AN ABSENT KEY IS THE DEFAULT AND A MISSPELLED ONE IS A 400.
///
/// The flatten map swallows anything it does not recognize, so the failure
/// this guards is specific: a client sending `reasoning_effort: "xhi"` (or
/// `true`, the shape it is upstream on some APIs) would otherwise get a
/// perfectly good answer that did not think, with nothing on the wire saying
/// the request was dropped.
#[test]
fn a_misspelled_reasoning_effort_is_refused_rather_than_ignored() {
    let model = state();
    let base = serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    assert_eq!(
        reasoning_effort(&request(base.clone()), ReasoningEffort::Off).unwrap(),
        ReasoningEffort::Off,
        "an absent key is the default, not an error"
    );
    assert_eq!(
        reasoning_effort(&request(base.clone()), ReasoningEffort::Low).unwrap(),
        ReasoningEffort::Low,
        "an absent key inherits the server configured default"
    );

    for bad in [serde_json::json!("xhi"), serde_json::json!(true)] {
        let mut body = base.clone();
        body["reasoning_effort"] = bad.clone();
        assert!(
            plan(&model, &request(body)).is_err(),
            "{bad} should be refused"
        );
    }
}

#[test]
fn no_tools_keeps_the_text_only_template() {
    let model = state();
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "hi"}]
    });
    let planned = plan(&model, &request(body)).expect("plan should succeed");
    let ids = planned.prompt_ids;

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
    let planned = plan(&planning_model, &request).expect("plan should succeed");
    let (prompt_ids, config) = (planned.prompt_ids, planned.config);

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
        None,
        &tool_names(&request),
        ReasoningEffort::Off,
        &|| false,
        &mut |piece| pieces.push(piece),
    )
    .expect("generation should succeed");

    assert_eq!(result.reason, runtime::StopReason::EndOfTurn);
    let calls: Vec<&tokenizer::ParsedToolCall> = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Tool(c) => Some(c),
            Piece::Text(_) | Piece::Reasoning(_) => None,
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
            Piece::Tool(_) | Piece::Reasoning(_) => None,
        })
        .collect();
    assert!(!text.contains("get_weather"), "{text}");
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

#[test]
fn a_plain_request_has_no_openai_warnings() {
    let request = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(openai_request_warnings(&request), None);
}

#[test]
fn response_format_json_is_reported_but_text_is_not() {
    let text = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "text"}
    }));
    assert_eq!(openai_request_warnings(&text), None);

    let json = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"}
    }));
    assert_eq!(
        openai_request_warnings(&json),
        Some("response_format".to_string())
    );
}

#[test]
fn n_greater_than_one_is_reported_but_one_and_absent_are_not() {
    let absent = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(openai_request_warnings(&absent), None);

    let one = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 1
    }));
    assert_eq!(openai_request_warnings(&one), None);

    let two = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2
    }));
    assert_eq!(openai_request_warnings(&two), Some("n".to_string()));
}

/// **`presence_penalty` AND `frequency_penalty` USED TO BE REPORTED HERE**
/// (this test used to assert exactly that, under the name
/// `presence_and_frequency_penalty_are_both_reported_together`), back when
/// neither reached `selection::shaping`. Commit B wired both through
/// `build_shaping`, so they are honoured rather than degraded now, and
/// `openai_request_warnings` no longer names either -- see
/// `handler::plan::build_config`'s own test coverage (through `select`, in
/// `crates/selection/tests/presence_frequency.rs`) for the half that
/// matters now: whether they actually change what gets generated.
#[test]
fn presence_and_frequency_penalty_no_longer_trigger_a_degradation_warning() {
    let request = request(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "presence_penalty": 0.1, "frequency_penalty": -0.1
    }));
    assert_eq!(openai_request_warnings(&request), None);
}

#[test]
fn merge_degradation_joins_two_notes_with_a_semicolon_and_passes_one_through() {
    assert_eq!(merge_degradation(None, None), None);
    assert_eq!(
        merge_degradation(Some("a".to_string()), None),
        Some("a".to_string())
    );
    assert_eq!(
        merge_degradation(None, Some("b".to_string())),
        Some("b".to_string())
    );
    assert_eq!(
        merge_degradation(Some("a".to_string()), Some("b".to_string())),
        Some("a; b".to_string())
    );
}
