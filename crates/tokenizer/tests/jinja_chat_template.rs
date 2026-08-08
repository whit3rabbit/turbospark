//! Tests the generic Jinja-templated chat rendering against the real,
//! vendored `chat_template.jinja` fixture (a genuine Qwen ChatML template,
//! not a stub), using `minijinja` as the Jinja engine.

use std::path::PathBuf;

use turbospark_tokenizer::{render_generic_chat_template, Message, MfTokenizer, Role};

fn load() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

#[test]
fn renders_a_single_user_turn_with_generation_prompt() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "hi")];
    let rendered = render_generic_chat_template(&tok, &messages, &[], true, false).unwrap();

    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
    assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn renders_system_then_user_turn() {
    let tok = load();
    let messages = vec![
        Message::new(Role::System, "be nice"),
        Message::new(Role::User, "hi"),
    ];
    let rendered = render_generic_chat_template(&tok, &messages, &[], true, false).unwrap();

    assert!(rendered.starts_with("<|im_start|>system\nbe nice<|im_end|>\n"));
    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
}

#[test]
fn rejects_empty_messages_via_raise_exception() {
    let tok = load();
    let err = render_generic_chat_template(&tok, &[], &[], true, false).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("No messages provided"),
        "message = {message}"
    );
}

#[test]
fn encode_generic_tool_chat_tokenizes_the_rendered_text() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "hi")];
    let ids = tok.encode_generic_tool_chat(&messages, &[], false).unwrap();
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids, false);
    assert!(decoded.contains("hi"));
}

#[test]
fn tools_branch_renders_the_tools_system_preamble() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "what's the weather")];
    let tools = vec![turbospark_tokenizer::FunctionDefinition {
        name: "get_weather".to_string(),
        description: "Get the weather".to_string(),
        parameters: turbospark_tokenizer::JsonValue::Object(std::collections::BTreeMap::new()),
    }];
    let rendered = render_generic_chat_template(&tok, &messages, &tools, true, false).unwrap();
    assert!(rendered.contains("# Tools"));
    assert!(rendered.contains("get_weather"));
}
