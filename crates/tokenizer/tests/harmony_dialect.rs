//! `gpt-oss`'s Harmony dialect (ROADMAP M5).
//!
//! The fixture carries the real checkpoint's special-token NAMES, read off
//! `openai/gpt-oss-20b`'s `tokenizer_config.json`, and deliberately NOT its
//! ids: the `tokenizers` loader renumbers added tokens after the 258-entry
//! base vocab, so every id here is resolved from the loaded tokenizer (crate
//! Gotcha 2). It also ships no `generation_config.json`, so what is asserted
//! below is the DIALECT's own stop set rather than the union with the
//! checkpoint's -- that union has its own test in `generation_config_eos.rs`.

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, Message, MfTokenizer, Role};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/HarmonyTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Harmony fixture loads")
}

#[test]
fn resolves_the_harmony_dialect_from_its_turn_frame() {
    let t = fixture();
    assert_eq!(t.dialect, ChatDialect::Harmony);
}

/// THE STOP SET HAS THREE MEMBERS, which is what this whole dialect exists
/// for. Harmony ends an assistant turn with `<|return|>` when it has answered
/// and with `<|call|>` when it is invoking a tool; `<|endoftext|>` is the base
/// end-of-sequence. Missing `<|call|>` is the dangerous one -- it does not
/// error, the model simply generates past its own tool call, which reads as a
/// rambling model rather than as a stop-set bug.
#[test]
fn the_stop_set_covers_return_call_and_endoftext() {
    let t = fixture();
    for name in ["<|return|>", "<|call|>", "<|endoftext|>"] {
        let id = t
            .token_to_id(name)
            .unwrap_or_else(|| panic!("{name} resolves"));
        assert!(
            t.stop_token_ids.contains(&id),
            "{name} (id {id}) is not in the stop set {:?}",
            t.stop_token_ids
        );
    }
}

/// `<|end|>` CLOSES SYSTEM AND USER TURNS INSIDE A PROMPT, so stopping on it
/// would end generation at the first token of a well-formed reply.
#[test]
fn end_is_not_a_stop_token() {
    let t = fixture();
    let end = t.token_to_id("<|end|>").expect("<|end|> resolves");
    assert!(
        !t.stop_token_ids.contains(&end),
        "<|end|> must not stop generation; it frames the PROMPT's turns"
    );
}

/// Harmony inverts the naming convention every other dialect here follows:
/// `<|endoftext|>` is the PAD token and `<|return|>` is the turn end.
#[test]
fn end_of_turn_is_return_and_pad_is_endoftext() {
    let t = fixture();
    assert_eq!(t.end_of_turn_id, t.token_to_id("<|return|>").unwrap());
    assert_eq!(t.eos_id, t.token_to_id("<|return|>").unwrap());
    assert_eq!(t.pad_id, t.token_to_id("<|endoftext|>").unwrap());
    assert_eq!(t.bos_id, t.token_to_id("<|startoftext|>").unwrap());
}

/// The checkpoint's own template renders the turns (AGENTS.md Gotcha 41), and
/// the encoder must not prepend a BOS on top of the `<|start|>` it emits.
#[test]
fn the_installed_template_renders_and_no_bos_is_prepended() {
    let t = fixture();
    let rendered = t
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .expect("the installed template renders");
    assert!(
        rendered.starts_with("<|start|>user<|message|>hi<|end|>"),
        "unexpected render: {rendered:?}"
    );
    assert!(rendered.ends_with("<|start|>assistant"), "{rendered:?}");

    let with_bos = t.encode("hi", true);
    let without = t.encode("hi", false);
    assert_eq!(
        with_bos, without,
        "the template emits its own turn opener, so `add_bos` must be a no-op"
    );
}

#[test]
fn text_continuation_joins_harmony_turn_markers_without_whitespace() {
    let t = fixture();
    let actual = t.encode_text_continuation("hello");
    let mut expected = vec![t.end_of_turn_id];
    expected.extend(t.encode(
        "<|start|>user<|message|>hello<|end|><|start|>assistant",
        false,
    ));

    assert_eq!(actual, expected);
}

/// A gpt-oss install that lost its template gets an ERROR, not an invented
/// prompt. Harmony's real template is 17 KB of system preamble, reasoning
/// effort and a TypeScript tool namespace; a partial re-implementation is
/// AGENTS.md Gotcha 41's failure mode exactly -- framing the model was not
/// trained on, returned as fluent output that is not an answer.
#[test]
fn there_is_no_fallback_renderer_for_harmony() {
    let t = fixture();
    let err = t
        .apply_dialect_chat_template(&[Message::new(Role::User, "hi")])
        .expect_err("the fallback must refuse rather than invent a frame");
    assert!(
        format!("{err:?}").contains("Harmony"),
        "the refusal must name the format, got {err:?}"
    );
}
