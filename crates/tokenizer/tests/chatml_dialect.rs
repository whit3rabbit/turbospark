//! Loads the vendored ChatML tokenizer fixture and exercises dialect
//! resolution, encode/decode, chat templating, and streaming decode against
//! a real (if minimal) BPE tokenizer.

use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, Message, MfDetokenizer, MfTokenizer, Role, StreamingStopMatcher,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer")
}

fn load() -> MfTokenizer {
    MfTokenizer::load_from_dir(&fixture_dir()).expect("fixture tokenizer should load")
}

fn load_qwen2_like() -> MfTokenizer {
    let dir =
        std::env::temp_dir().join(format!("turbospark-tokenizer-qwen2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create Qwen2 tokenizer directory");
    let mut tokenizer: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("tokenizer.json"))
            .expect("read ChatML tokenizer fixture"),
    )
    .expect("parse ChatML tokenizer fixture");
    let added_tokens = tokenizer["added_tokens"]
        .as_array()
        .expect("fixture added_tokens array")
        .iter()
        .filter(|token| {
            !matches!(
                token.get("content").and_then(serde_json::Value::as_str),
                Some("<tool_response>")
                    | Some("</tool_response>")
                    | Some("<think>")
                    | Some("</think>")
            )
        })
        .cloned()
        .collect();
    tokenizer["added_tokens"] = serde_json::Value::Array(added_tokens);
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).expect("serialize Qwen2-like tokenizer"),
    )
    .expect("write Qwen2-like tokenizer");
    std::fs::copy(
        fixture_dir().join("tokenizer_config.json"),
        dir.join("tokenizer_config.json"),
    )
    .expect("copy ChatML tokenizer config");
    MfTokenizer::load_from_dir(&dir).expect("Qwen2-like tokenizer should load")
}

#[test]
fn resolves_chatml_dialect() {
    let tok = load();
    assert_eq!(tok.dialect, ChatDialect::ChatMl);
}

#[test]
fn chatml_allows_qwen2_without_think_or_tool_response_special_tokens() {
    let tok = load_qwen2_like();
    assert_eq!(tok.dialect, ChatDialect::ChatMl);
    assert_ne!(
        tok.tool_call_start_id,
        turbospark_tokenizer::NO_SUCH_TOKEN_ID
    );
    assert_ne!(tok.tool_call_end_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_response_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(
        tok.tool_response_end_id,
        turbospark_tokenizer::NO_SUCH_TOKEN_ID
    );
    assert_eq!(tok.think_start_id, None);
    assert_eq!(tok.think_end_id, None);
}

#[test]
fn encode_decode_round_trips_ascii_text() {
    let tok = load();
    let ids = tok.encode("hello world", false);
    assert!(!ids.is_empty());
    let text = tok.decode(&ids, true);
    assert_eq!(text, "hello world");
}

#[test]
fn add_bos_is_a_no_op_for_chatml() {
    let tok = load();
    let with_bos = tok.encode("hi", true);
    let without_bos = tok.encode("hi", false);
    assert_eq!(with_bos, without_bos);
}

#[test]
fn chat_template_wraps_each_turn_in_im_start_end() {
    let tok = load();
    let messages = vec![
        Message::new(Role::System, "be nice"),
        Message::new(Role::User, "hi"),
    ];
    let rendered = tok.apply_chat_template(&messages).unwrap();
    assert!(rendered.starts_with("<|im_start|>system\nbe nice<|im_end|>\n"));
    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
    assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn chat_template_rejects_system_message_not_first() {
    let tok = load();
    let messages = vec![
        Message::new(Role::User, "hi"),
        Message::new(Role::System, "late"),
    ];
    assert!(tok.apply_chat_template(&messages).is_err());
}

#[test]
fn stop_matcher_withholds_partial_match_across_chunks() {
    let mut matcher = StreamingStopMatcher::new(vec!["STOP".to_string()]);
    let mut out = String::new();
    out += &matcher.push("hello ST");
    assert!(!matcher.is_stopped());
    out += &matcher.push("OP world");
    assert!(matcher.is_stopped());
    assert_eq!(out, "hello ");
}

#[test]
fn stop_matcher_finish_releases_pending_text() {
    let mut matcher = StreamingStopMatcher::new(vec!["STOP".to_string()]);
    let mut out = matcher.push("hello wor");
    out += &matcher.finish();
    assert_eq!(out, "hello wor");
}

#[test]
fn detokenizer_push_and_flush_reproduce_full_decode() {
    let tok = load();
    let ids = tok.encode("hello there friend", false);
    let mut detok = MfDetokenizer::new(&tok);
    let mut streamed = String::new();
    for &id in &ids {
        streamed += &detok.push(id);
    }
    streamed += &detok.flush();
    let full = tok.decode(&ids, true);
    assert_eq!(streamed, full);
}
