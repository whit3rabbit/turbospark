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

#[test]
fn resolves_chatml_dialect() {
    let tok = load();
    assert_eq!(tok.dialect, ChatDialect::ChatMl);
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
