//! The Flint dialect: detection, resolution, and the content passthrough.
//!
//! **THE FIXTURE'S ADDED-TOKEN TABLE IS REAL.** `fixtures/FlintTokenizer` is
//! reduced from `RecursiveSI/flint-int4`'s own `tokenizer.json` (read
//! 2026-10-21): all 19 non-reserved added tokens with their real ids and
//! special flags, the real pre-tokenizer, post-processor and decoder, and the
//! 256 byte-level vocab entries. What is dropped is the 150k-entry BPE body,
//! which no probe, resolver or decoder arm touches.
//!
//! **THE REGRESSION THIS EXISTS FOR.** Before `ChatDialect::Flint` this table
//! resolved as GLM (it carries GLM's `<|assistant|>` and `<|observation|>`)
//! and `pull` refused it with "tokenizer missing required special token:
//! <|endoftext|>" (AGENTS.md Gotcha 52). The first test below fails the same
//! way if the Flint arm is moved after the GLM one.
//!
//! **EVERY ID ASSERTION IS BY NAME** (crate Gotcha 2: the `tokenizers`
//! library renumbers a sparse fixture's ids on load).

use std::path::PathBuf;

use turbospark_tokenizer::{ChatDialect, MfTokenizer, ToolCallSupport, NO_SUCH_TOKEN_ID};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/FlintTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("the Flint fixture tokenizer should load")
}

#[test]
fn the_real_table_resolves_to_flint_and_not_glm() {
    let tok = fixture();
    assert_eq!(
        tok.dialect,
        ChatDialect::Flint,
        "a Flint table must not fall into the GLM arm"
    );
    let id = |name: &str| {
        tok.token_to_id(name)
            .unwrap_or_else(|| panic!("{name} must resolve in the loaded table"))
    };
    assert_eq!(tok.bos_id, id("<|begin_of_text|>"));
    assert_eq!(tok.eos_id, id("<|end_of_text|>"));
    assert_eq!(tok.pad_id, id("<|end_of_text|>"));
    assert_eq!(tok.end_of_turn_id, id("<|end_of_text|>"));
    assert_eq!(tok.tool_call_stop_id, id("<|observation|>"));
    assert_eq!(tok.vocab_size, 150_144);
    assert_eq!(tok.dialect.tool_call_support(), ToolCallSupport::Prompted);
}

/// The stop set is `generation_config.json`'s four ids: a turn has no closing
/// marker, so the model ends by emitting one of these.
#[test]
fn all_four_generation_config_stops_are_stops() {
    let tok = fixture();
    for name in [
        "<|end_of_text|>",
        "<|user|>",
        "<|observation|>",
        "<|developer|>",
    ] {
        let id = tok.token_to_id(name).expect(name);
        assert!(
            tok.stop_token_ids.contains(&id),
            "{name} must stop generation"
        );
    }
    // The assistant marker must NOT stop: it opens the turn being generated.
    let assistant = tok.token_to_id("<|assistant|>").unwrap();
    assert!(!tok.stop_token_ids.contains(&assistant));
}

/// `add_bos_token` is false and the template writes no BOS, so encoding must
/// not prepend one.
#[test]
fn no_bos_is_prepended() {
    let tok = fixture();
    assert_eq!(tok.encode("x", true), tok.encode("x", false));
}

/// The think pair and tool markup are added but NOT special, so no id-bracket
/// arm may claim them.
#[test]
fn think_and_tool_markup_ids_are_sentinels() {
    let tok = fixture();
    assert_eq!(tok.tool_call_start_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_call_end_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_response_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.channel_start_id, NO_SUCH_TOKEN_ID);
    assert_eq!(tok.think_start_id, None);
    assert_eq!(tok.think_end_id, None);
}

/// A table that is GLM must still be GLM: the Flint probe must not widen to
/// catch `<|observation|>` + `<|assistant|>` alone.
#[test]
fn the_glm_fixture_still_resolves_to_glm() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/GlmTokenizer");
    let tok = MfTokenizer::load_from_dir(&dir).expect("GLM fixture loads");
    assert_eq!(tok.dialect, ChatDialect::Glm);
}
