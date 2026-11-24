//! Phi-3 / Phi-3.5's dialect. Before it existed this table (which carries `<s>`
//! and `</s>`) resolved MISTRAL, whose only stop is `</s>`, so a real Phi-3.5
//! install answered correctly and then ran on past its own `<|end|>` into an
//! invented next user turn (seen on `bartowski/Phi-3.5-mini-instruct-GGUF`).
//!
//! The fixture carries the real table's special-token NAMES (read off
//! `microsoft/Phi-3.5-mini-instruct`'s `tokenizer_config.json`) over a tiny
//! byte-level body, and deliberately not its ids (crate Gotcha 2). It ships no
//! chat template, so the fallback refusal is what is exercised; the real
//! install's own template is rendered in the real-install gate.

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, Message, MfTokenizer, Role, NO_SUCH_TOKEN_ID};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/Phi3Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Phi-3 fixture loads")
}

#[test]
fn resolves_phi3_and_not_mistral() {
    assert_eq!(fixture().dialect, ChatDialect::Phi3);
}

#[test]
fn end_marks_the_turn_and_both_it_and_eos_stop_generation() {
    let t = fixture();
    let end = t.token_to_id("<|end|>").unwrap();
    let eos = t.token_to_id("<|endoftext|>").unwrap();
    assert_eq!(t.end_of_turn_id, end);
    assert_eq!(t.eos_id, eos);
    // THE BUG THIS DIALECT FIXES: `<|end|>` must stop generation.
    assert!(t.stop_token_ids.contains(&end) && t.stop_token_ids.contains(&eos));
    // `<s>` is the BOS the encoder prepends (the template does not render it).
    assert_eq!(t.bos_id, t.token_to_id("<s>").unwrap());
}

#[test]
fn there_is_no_tool_or_channel_markup() {
    let t = fixture();
    for id in [
        t.tool_call_start_id,
        t.tool_call_end_id,
        t.channel_start_id,
        t.channel_end_id,
        t.message_start_id,
        t.message_end_id,
    ] {
        assert_eq!(id, NO_SUCH_TOKEN_ID);
    }
}

#[test]
fn a_template_less_install_is_refused_by_name_not_given_an_invented_prompt() {
    let err = fixture()
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("phi3"), "{err}");
}
