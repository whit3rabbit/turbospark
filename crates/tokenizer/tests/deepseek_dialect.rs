//! Loads the vendored DeepSeek tokenizer fixture and exercises dialect
//! resolution and the hand-rolled (non-Jinja) chat templates.

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, Message, MfTokenizer, Role};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/DeepseekTokenizer")
}

fn load() -> MfTokenizer {
    MfTokenizer::load_from_dir(&fixture_dir()).expect("fixture tokenizer should load")
}

#[test]
fn resolves_deepseek_dialect() {
    let tok = load();
    assert_eq!(tok.dialect, ChatDialect::Deepseek);
}

#[test]
fn chat_template_frames_user_turn_with_assistant_transition() {
    let tok = load();
    let messages = vec![Message::new(Role::User, "hi")];
    let rendered = tok.apply_chat_template(&messages).unwrap();
    assert!(rendered.starts_with("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
    assert!(rendered.contains("<\u{FF5C}User\u{FF5C}>hi"));
    assert!(rendered.ends_with("<\u{FF5C}Assistant\u{FF5C}></think>"));
}

#[test]
fn chat_template_assistant_turn_closes_with_eos() {
    let tok = load();
    let messages = vec![
        Message::new(Role::User, "hi"),
        Message::new(Role::Assistant, "hello"),
    ];
    let rendered = tok.apply_chat_template(&messages).unwrap();
    assert!(rendered.contains("</think>hello<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"));
    // Last message is assistant (not user/developer), so the generation
    // prompt is appended after it too.
    assert!(rendered.ends_with("<\u{FF5C}Assistant\u{FF5C}></think>"));
}

#[test]
fn chat_template_rejects_tool_role() {
    let tok = load();
    let messages = vec![Message::new(Role::Tool, "result")];
    assert!(tok.apply_chat_template(&messages).is_err());
}

#[test]
fn encode_text_continuation_prefixes_end_of_turn() {
    let tok = load();
    let ids = tok.encode_text_continuation("more please");
    assert_eq!(ids[0], tok.end_of_turn_id);
}
