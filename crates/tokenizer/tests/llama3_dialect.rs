//! Meta's Llama-3 dialect. Unblocks `Meta-Llama-3-8B-Instruct` and the
//! published steering vector for it (`docs/OBLITERATION.md`'s Open section):
//! before this file's subject landed, `detect_dialect` fell through to Gemma
//! for this family's table and `resolve_gemma` failed on a missing `<pad>`.
//!
//! The fixture carries the real checkpoint's special-token NAMES, read off
//! `meta-llama/Meta-Llama-3-8B-Instruct`'s `tokenizer_config.json`, and
//! deliberately NOT its ids: the `tokenizers` loader renumbers added tokens
//! after the 258-entry base vocab, so every id here is resolved from the
//! loaded tokenizer (crate Gotcha 2). It also ships no `chat_template.jinja`
//! or `tokenizer_config.json` `chat_template` key, so what is exercised below
//! is the FALLBACK renderer (`apply_dialect_chat_template`, which
//! `apply_chat_template` falls to automatically) rather than the checkpoint's
//! own template -- the real install has one and would take the Jinja path
//! instead (AGENTS.md Gotcha 41).

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, Message, MfDetokenizer, MfTokenizer, Role};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/Llama3Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Llama-3 fixture loads")
}

#[test]
fn resolves_the_llama3_dialect_from_its_header_frame() {
    let t = fixture();
    assert_eq!(t.dialect, ChatDialect::Llama3);
}

/// SHARES `<|begin_of_text|>` AND `<|end_of_text|>` WITH `muse_glimmer` and
/// nothing else -- the fixture proves the two frames' OWN markers
/// (`<|start_header_id|>` / `<|eot_id|>` here, `<|start|>` / `<|eot|>` there)
/// are what keeps detection unambiguous, not the shared pair.
#[test]
fn bos_and_eos_are_the_shared_muse_glimmer_marks() {
    let t = fixture();
    assert_eq!(t.bos_id, t.token_to_id("<|begin_of_text|>").unwrap());
    assert_eq!(t.eos_id, t.token_to_id("<|end_of_text|>").unwrap());
}

/// End of turn is `<|eot_id|>`, not `<|end_of_text|>`: the checkpoint closes
/// every assistant turn with the former and reserves the latter for a raw
/// end of document.
#[test]
fn end_of_turn_is_eot_id_and_is_in_the_stop_set() {
    let t = fixture();
    let eot = t.token_to_id("<|eot_id|>").unwrap();
    assert_eq!(t.end_of_turn_id, eot);
    assert!(t.stop_token_ids.contains(&eot));
    assert!(t.stop_token_ids.contains(&t.eos_id));
}

/// No dedicated `<pad>` in this table -- the missing one is what sent this
/// family into `resolve_gemma` and a load failure before this dialect
/// existed.
#[test]
fn pad_reuses_eos_rather_than_naming_a_dedicated_pad_token() {
    let t = fixture();
    assert_eq!(t.pad_id, t.eos_id);
}

/// No tool-calling or thinking markup in the base 8B-Instruct table this was
/// built against.
#[test]
fn there_is_no_tool_call_or_channel_markup() {
    let t = fixture();
    assert_eq!(t.tool_call_start_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(t.tool_call_end_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(t.channel_start_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(t.think_start_id, None);
}

/// The fallback renderer emits `<|begin_of_text|>` itself, matching the real
/// checkpoint's own template (`{{- bos_token }}` at the top) and
/// `resolve_llama3`'s `bos_prefix_id: None` -- the encoder must not prepend a
/// second one.
#[test]
fn the_fallback_renderer_emits_bos_and_no_second_one_is_prepended() {
    let t = fixture();
    let rendered = t
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .expect("the fallback renderer renders");
    assert!(
        rendered.starts_with(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        ),
        "unexpected render: {rendered:?}"
    );
    assert!(
        rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "unexpected render: {rendered:?}"
    );

    let with_bos = t.encode("hi", true);
    let without = t.encode("hi", false);
    assert_eq!(
        with_bos, without,
        "no dedicated BOS prefix is configured; the template (or a caller) supplies its own"
    );
}

#[test]
fn encode_decode_round_trips_ascii_text() {
    let t = fixture();
    let ids = t.encode("hello world", false);
    assert!(!ids.is_empty());
    assert_eq!(t.decode(&ids, true), "hello world");
}

#[test]
fn detokenizer_push_and_flush_reproduce_full_decode() {
    let t = fixture();
    let ids = t.encode("hello there friend", false);
    let mut detok = MfDetokenizer::new(&t);
    let mut streamed = String::new();
    for &id in &ids {
        streamed += &detok.push(id);
    }
    streamed += &detok.flush();
    assert_eq!(streamed, t.decode(&ids, true));
}
