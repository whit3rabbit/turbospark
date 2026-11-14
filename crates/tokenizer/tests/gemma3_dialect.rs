//! Gemma 3's dialect. Before it existed `detect_dialect` fell through to the
//! Gemma 4 fallback for this table and `resolve_gemma` failed on the missing
//! `<turn|>`, so a Gemma 3 install could not load a tokenizer at all.
//!
//! The fixture carries the real table's special-token NAMES (read off
//! `mlx-community/gemma-3-1b-it-bf16`'s `added_tokens.json`) over a tiny
//! byte-level body, and deliberately not its ids (crate Gotcha 2). It ships no
//! chat template, so the fallback refusal is what is exercised; the real
//! install's own template is rendered in the real-install gate.

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, Message, MfTokenizer, Role, NO_SUCH_TOKEN_ID};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/Gemma3Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Gemma 3 fixture loads")
}

#[test]
fn resolves_gemma3_and_not_the_gemma4_fallback() {
    assert_eq!(fixture().dialect, ChatDialect::Gemma3);
}

#[test]
fn end_of_turn_is_end_of_turn_and_both_it_and_eos_stop_generation() {
    let t = fixture();
    let eot = t.token_to_id("<end_of_turn>").unwrap();
    assert_eq!(t.end_of_turn_id, eot);
    assert_eq!(t.eos_id, t.token_to_id("<eos>").unwrap());
    assert_eq!(t.bos_id, t.token_to_id("<bos>").unwrap());
    assert!(t.stop_token_ids.contains(&eot) && t.stop_token_ids.contains(&t.eos_id));
}

#[test]
fn there_is_no_tool_or_channel_markup() {
    let t = fixture();
    for id in [
        t.tool_call_start_id,
        t.tool_call_end_id,
        t.tool_response_id,
        t.channel_start_id,
    ] {
        assert_eq!(id, NO_SUCH_TOKEN_ID);
    }
    assert_eq!(t.think_start_id, None);
    assert!(t.dialect.tool_call_unsupported_reason().is_some());
}

#[test]
fn no_second_bos_is_prepended_because_the_template_emits_its_own() {
    let t = fixture();
    assert_eq!(t.encode("hi", true), t.encode("hi", false));
}

#[test]
fn without_the_checkpoints_template_rendering_is_refused_not_invented() {
    assert!(fixture()
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .is_err());
}

#[test]
fn the_continuation_is_one_user_turn_and_the_model_opener() {
    let t = fixture();
    let ids = t.encode_text_continuation("hi");
    let decoded = t.decode(&ids, false);
    assert!(
        decoded.ends_with("<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"),
        "{decoded:?}"
    );
    assert_eq!(
        ids[0], t.end_of_turn_id,
        "closes the interrupted turn first"
    );
}

#[test]
fn a_gemma4_table_still_resolves_to_gemma4() {
    // The `<turn|>` exclusion: a table with Gemma 4's marker is not Gemma 3
    // even if it also carried the older pair.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/GemmaTokenizer");
    let t = MfTokenizer::load_from_dir(&dir).expect("Gemma 4 fixture loads");
    assert_eq!(t.dialect, ChatDialect::Gemma);
}
