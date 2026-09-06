//! `MfTokenizer::verify_image_markers` (vision memory sidecar, Part A4):
//! confirms a `VisionConfig`'s `vision_start`/`image_pad` ids are usable
//! against a real tokenizer BEFORE the first image is ever processed,
//! against the vendored ChatML fixture, whose `chat_template.jinja` already
//! carries a real `<|vision_start|><|image_pad|><|vision_end|>` marker run
//! for an image content part.

use std::path::PathBuf;

use turbospark_tokenizer::MfTokenizer;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer")
}

fn load() -> MfTokenizer {
    MfTokenizer::load_from_dir(&fixture_dir()).expect("fixture tokenizer should load")
}

/// Never hardcode an id read from the fixture's `tokenizer.json` (AGENTS.md
/// Gotcha 11 / crate Gotcha 3): the `tokenizers` crate renumbers added
/// tokens sequentially after the base vocabulary at load time, so the ids
/// this test asserts against are resolved from the LOADED tokenizer.
fn marker_ids(tok: &MfTokenizer) -> (i32, i32) {
    let vision_start = tok
        .token_to_id("<|vision_start|>")
        .expect("fixture carries <|vision_start|> as an added token");
    let image_pad = tok
        .token_to_id("<|image_pad|>")
        .expect("fixture carries <|image_pad|> as an added token");
    (vision_start, image_pad)
}

#[test]
fn valid_marker_ids_pass() {
    let tok = load();
    let (vision_start, image_pad) = marker_ids(&tok);
    tok.verify_image_markers(vision_start, image_pad)
        .expect("real marker ids resolved from the loaded tokenizer must verify");
}

#[test]
fn a_bogus_vision_start_id_fails() {
    let tok = load();
    let (_, image_pad) = marker_ids(&tok);
    // Comfortably past the fixture's tiny vocabulary (258 base entries plus
    // a handful of added tokens), so this can never coincide with a real id.
    let bogus = 9_000_000;
    let err = tok
        .verify_image_markers(bogus, image_pad)
        .expect_err("an id with no token behind it must be refused");
    let message = err.to_string();
    assert!(
        message.contains(&bogus.to_string()),
        "error should name the offending id: {message}"
    );
}

#[test]
fn a_bogus_image_pad_id_fails() {
    let tok = load();
    let (vision_start, _) = marker_ids(&tok);
    let bogus = 9_000_001;
    let err = tok
        .verify_image_markers(vision_start, bogus)
        .expect_err("an id with no token behind it must be refused");
    let message = err.to_string();
    assert!(
        message.contains(&bogus.to_string()),
        "error should name the offending id: {message}"
    );
}

#[test]
fn identical_marker_ids_are_refused() {
    let tok = load();
    let (vision_start, _) = marker_ids(&tok);
    let err = tok
        .verify_image_markers(vision_start, vision_start)
        .expect_err("two distinct markers must not collide on one id");
    assert!(err.to_string().contains("distinct"));
}

/// A tokenizer whose ids resolve fine but whose ONE-IMAGE render carries the
/// wrong multiplicity of a marker must still be refused: two real, valid
/// tokens standing in for vision_start/image_pad (but not the ones the
/// template actually emits around an image) render zero occurrences of each
/// rather than one.
#[test]
fn ids_that_resolve_but_never_render_around_an_image_are_refused() {
    let tok = load();
    let unrelated_a = tok
        .token_to_id("<tool_call>")
        .expect("fixture carries <tool_call> as an added token");
    let unrelated_b = tok
        .token_to_id("</tool_call>")
        .expect("fixture carries </tool_call> as an added token");
    let err = tok
        .verify_image_markers(unrelated_a, unrelated_b)
        .expect_err("ids the template never places around an image must be refused");
    assert!(err.to_string().contains('0'));
}
