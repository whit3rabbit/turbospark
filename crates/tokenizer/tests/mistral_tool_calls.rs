//! The Mistral `[TOOL_CALLS]` decoder arm, against the Zephyr fixture.
//!
//! Mistral's shape is the inverse of every bracketing dialect: the marker is
//! a special token (empty delta, root Gotcha 44) and the body runs to end of
//! turn with NO closing token, so the call is emitted from `finish` and
//! nowhere else. These tests pin the buffering, the finish-time emission,
//! the two failure routes, and the content around the span.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use turbospark_tokenizer::{
    MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent, NO_SUCH_TOKEN_ID,
};

fn fixture(name: &str) -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn decoder<'a>(tok: &'a MfTokenizer) -> StructuredAssistantDecoder<'a> {
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    StructuredAssistantDecoder::new(tok, allowed, || "call_1".to_string(), &[])
}

fn fixture_without_tool_marker() -> (MfTokenizer, PathBuf) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ZephyrTokenizer");
    let dir = std::env::temp_dir().join(format!(
        "turbospark-mistral-without-tool-marker-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir(&dir).expect("temporary fixture directory should be created");
    fs::copy(
        source.join("tokenizer_config.json"),
        dir.join("tokenizer_config.json"),
    )
    .expect("tokenizer config should be copied");

    let mut tokenizer: serde_json::Value = serde_json::from_slice(
        &fs::read(source.join("tokenizer.json")).expect("fixture tokenizer should be readable"),
    )
    .expect("fixture tokenizer should be JSON");
    let added = tokenizer["added_tokens"]
        .as_array_mut()
        .expect("fixture should have added tokens");
    added.retain(|token| token["content"] != "[TOOL_CALLS]");
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).expect("modified tokenizer should serialize"),
    )
    .expect("modified tokenizer should be written");

    let tokenizer = MfTokenizer::load_from_dir(&dir).expect("modified tokenizer should load");
    (tokenizer, dir)
}

const BODY: &str = r#"[{"name": "get_weather", "arguments": {"city": "Oslo"}}]"#;

#[test]
fn absent_tool_marker_does_not_swallow_idless_tail_text() {
    let (tok, dir) = fixture_without_tool_marker();
    assert_eq!(tok.tool_call_start_id, NO_SUCH_TOKEN_ID);
    let mut d = decoder(&tok);

    let events = d
        .consume(NO_SUCH_TOKEN_ID, " nonempty flushed tail")
        .expect("tail should remain ordinary content");
    assert_eq!(
        events,
        vec![StructuredAssistantEvent::Content(
            " nonempty flushed tail".to_string()
        )]
    );
    assert!(d.finish().expect("no tool span should be open").is_empty());

    fs::remove_dir_all(dir).expect("temporary fixture should be removed");
}

/// Feed `BODY` as ordinary token deltas the way the real detokenizer would:
/// the marker itself is special and arrives empty.
fn feed_body(tok: &MfTokenizer, decoder: &mut StructuredAssistantDecoder, text: &str) {
    for id in tok.encode(text, false) {
        let delta = tok.decode(&[id], true);
        decoder.consume(id, &delta).expect("body token consumes");
    }
}

#[test]
fn a_tool_calls_marker_buffers_and_finish_parses_the_call() {
    let tok = fixture("ZephyrTokenizer");
    assert!(
        tok.tool_call_start_id != -1,
        "the fixture must resolve [TOOL_CALLS] for this file to mean anything"
    );
    let mut d = decoder(&tok);
    // Content before the marker streams as content immediately.
    let pre = d.consume(tok.token_to_id("h").unwrap_or(0), "h").unwrap();
    assert_eq!(
        pre,
        vec![StructuredAssistantEvent::Content("h".to_string())]
    );

    d.consume(tok.tool_call_start_id, "").unwrap();
    feed_body(&tok, &mut d, BODY);

    // NOTHING is emitted while the span is open; the call arrives at finish.
    let events = d.finish().expect("a clean span parses at finish");
    assert!(
        !events.is_empty(),
        "finish must emit the call: the marker is the only framing token there is"
    );
    let StructuredAssistantEvent::ToolCall(call) = &events[0] else {
        panic!("expected a tool call out of finish, got {events:?}");
    };
    assert_eq!(call.name, "get_weather");
    assert!(
        call.arguments_json.contains("Oslo"),
        "arguments must survive the round trip: {}",
        call.arguments_json
    );
}

#[test]
fn a_second_marker_is_malformed_rather_than_a_second_array() {
    let tok = fixture("ZephyrTokenizer");
    let mut d = decoder(&tok);
    d.consume(tok.tool_call_start_id, "").unwrap();
    feed_body(&tok, &mut d, BODY);
    let err = d
        .consume(tok.tool_call_start_id, "")
        .expect_err("a second marker abandons the first span");
    let _ = err;
    assert!(d.finish().is_err(), "a failed decoder stays failed");
}

#[test]
fn an_unknown_tool_releases_the_span_body_at_finish() {
    let tok = fixture("ZephyrTokenizer");
    let allowed: HashSet<String> = ["other_tool".to_string()].into_iter().collect();
    let mut d = StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string(), &[]);
    d.consume(tok.tool_call_start_id, "").unwrap();
    feed_body(&tok, &mut d, BODY);
    // The parse fails on the allowlist at finish, and the failed-span
    // release is what hands the body to the rescue tier.
    assert!(d.finish().is_err(), "an unknown tool must fail the span");
    let body = d.take_failed_span_text();
    assert_eq!(
        body.as_deref(),
        Some(BODY),
        "the released span body must be the marker's interior"
    );
}

#[test]
fn prose_after_the_call_array_is_content_not_markup() {
    let tok = fixture("ZephyrTokenizer");
    let mut d = decoder(&tok);
    d.consume(tok.tool_call_start_id, "").unwrap();
    // The shape the pinned 7B install actually produces: the array, then the
    // model keeps talking. There is no closing token to stop the span's
    // buffering, so the prose arrives inside it and has to be handed back as
    // content after the call.
    feed_body(
        &tok,
        &mut d,
        r#"[{"name": "get_weather", "arguments": {"city": "Oslo"}}]
I could not reach the weather service."#,
    );
    let events = d.finish().expect("the array prefix parses");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StructuredAssistantEvent::ToolCall(_))),
        "one call: {events:?}"
    );
    let prose_back = events.iter().any(|e| match e {
        StructuredAssistantEvent::Content(c) => c.contains("weather service"),
        _ => false,
    });
    assert!(
        prose_back,
        "the trailing prose must come back as content: {events:?}"
    );
}

#[test]
fn an_open_span_abandoned_by_budget_is_a_finish_error() {
    let tok = fixture("ZephyrTokenizer");
    let mut d = decoder(&tok);
    d.consume(tok.tool_call_start_id, "").unwrap();
    feed_body(&tok, &mut d, BODY);
    // A generation that hit its token budget partway through the JSON still
    // HAS an open span at finish, but the body parses fine here -- so this
    // case closes cleanly. The ERROR case is the truncated body: feed half.
    let tok2 = fixture("ZephyrTokenizer");
    let mut d2 = decoder(&tok2);
    d2.consume(tok2.tool_call_start_id, "").unwrap();
    let truncated = &BODY[..BODY.len() / 2];
    feed_body(&tok2, &mut d2, truncated);
    assert!(
        d2.finish().is_err(),
        "a JSON array cut in half must not parse as a call"
    );
    let _ = d.finish().expect("the complete body parses");
}
