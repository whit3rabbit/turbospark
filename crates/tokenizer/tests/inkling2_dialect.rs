//! The Inkling 2 dialect: detection, resolution, and the thinking split.
//!
//! **THE FIXTURE'S ADDED-TOKEN TABLE IS REAL.** `fixtures/Inkling2Tokenizer`
//! is reduced from `thinkingmachines/inkling-2-mini`'s own `tokenizer.json`
//! (read 2026-11-21): the 15 non-reserved added tokens with their real ids
//! and special flags, the real normalizer, pre-tokenizer, post-processor and
//! decoder, and the 256 single-character byte-level vocab entries. The BPE
//! body and the `<|unused_*|>` reserves are dropped; no probe, resolver or
//! decoder arm reads them.
//!
//! **THE REGRESSION THIS EXISTS FOR.** Before `ChatDialect::Inkling2` this
//! table fell to the Gemma fallback and loading refused it with "tokenizer
//! missing required special token: <bos>".
//!
//! **EVERY ID ASSERTION IS BY NAME** (crate Gotcha 2).

use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent,
    ToolCallSupport, NO_SUCH_TOKEN_ID,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/Inkling2Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("the Inkling 2 fixture tokenizer should load")
}

fn id(tok: &MfTokenizer, name: &str) -> i32 {
    tok.token_to_id(name)
        .unwrap_or_else(|| panic!("{name} must resolve in the loaded table"))
}

#[test]
fn the_real_table_resolves_to_inkling2() {
    let tok = fixture();
    assert_eq!(tok.dialect, ChatDialect::Inkling2);
    assert_eq!(tok.bos_id, id(&tok, "<|begin_of_text|>"));
    assert_eq!(tok.eos_id, id(&tok, "<|content_model_end_sampling|>"));
    assert_eq!(
        tok.end_of_turn_id,
        id(&tok, "<|content_model_end_sampling|>")
    );
    assert_eq!(tok.pad_id, id(&tok, "<|endoftext|>"));
    assert_eq!(tok.vocab_size, 201_024);
    assert_eq!(tok.dialect.tool_call_support(), ToolCallSupport::Native);
    assert_eq!(tok.tool_call_start_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_call_stop_id, NO_SUCH_TOKEN_ID);
}

/// `<|content_model_end_sampling|>` is the one stop. `<|end_message|>` closes
/// every segment, the thinking one included, so stopping on it would cut the
/// answer off after the thought.
#[test]
fn end_sampling_stops_and_end_message_does_not() {
    let tok = fixture();
    assert!(tok
        .stop_token_ids
        .contains(&id(&tok, "<|content_model_end_sampling|>")));
    for name in ["<|end_message|>", "<|message_model|>", "<|content_text|>"] {
        assert!(
            !tok.stop_token_ids.contains(&id(&tok, name)),
            "{name} must not stop generation"
        );
    }
}

/// The template writes no BOS, so encoding must not prepend one.
#[test]
fn no_bos_is_prepended() {
    let tok = fixture();
    assert_eq!(tok.encode("x", true), tok.encode("x", false));
}

/// A model turn as the checkpoint writes it: a thinking segment, then the
/// answer segment, then the end of sampling. The thought must arrive as
/// reasoning and only the answer as content.
#[test]
fn the_thinking_segment_splits_into_reasoning() {
    let tok = fixture();
    assert_eq!(tok.think_start_id, Some(id(&tok, "<|content_thinking|>")));
    assert_eq!(tok.think_end_id, Some(id(&tok, "<|end_message|>")));
    let mut decoder = StructuredAssistantDecoder::new(&tok, Default::default(), String::new, &[]);
    let mut events = Vec::new();
    for (token, delta) in [
        (id(&tok, "<|message_model|>"), ""),
        (id(&tok, "<|content_thinking|>"), ""),
        (-1, "plan"),
        (id(&tok, "<|end_message|>"), ""),
        (id(&tok, "<|message_model|>"), ""),
        (id(&tok, "<|content_text|>"), ""),
        (-1, "answer"),
        (id(&tok, "<|end_message|>"), ""),
    ] {
        events.extend(decoder.consume(token, delta).unwrap());
    }
    assert_eq!(
        events,
        [
            StructuredAssistantEvent::Reasoning("plan".into()),
            StructuredAssistantEvent::Content("answer".into()),
        ]
    );
}

/// A prompt rendered by the template ends with `<|end_message|>`, so the
/// decoder must start visible, not inside a thought.
#[test]
fn a_rendered_prompt_does_not_open_a_thought() {
    let tok = fixture();
    let prompt = [
        id(&tok, "<|message_user|>"),
        id(&tok, "<|content_text|>"),
        id(&tok, "<|end_message|>"),
    ];
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, Default::default(), String::new, &prompt);
    assert_eq!(
        decoder.consume(-1, "hi").unwrap(),
        [StructuredAssistantEvent::Content("hi".into())]
    );
}
