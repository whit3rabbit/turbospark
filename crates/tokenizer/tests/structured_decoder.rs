//! Streaming structured-decoder tests against the ChatML and Gemma fixtures:
//! tool-call token spans are buffered and parsed atomically, plain text
//! streams as content immediately, and each dialect's thought channel comes
//! back as reasoning rather than as part of the reply.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent};

fn fixture(name: &str) -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn load() -> MfTokenizer {
    fixture("ChatMLTokenizer")
}

/// Gemma's frame is a BRACKETING channel pair with the channel NAME in the
/// first line of the body, where ChatML's is a `<think>`/`</think>` pair
/// around one. The fixture carries the ten special tokens `resolve_gemma`
/// requires and nothing else; the ids are placeholders, resolved at load
/// (crate Gotcha 3).
fn gemma() -> MfTokenizer {
    fixture("GemmaTokenizer")
}

#[test]
fn plain_text_tokens_stream_as_content() {
    let tok = load();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string());
    let events = decoder
        .consume(tok.token_to_id("h").unwrap_or(0), "h")
        .unwrap();
    assert_eq!(
        events,
        vec![StructuredAssistantEvent::Content("h".to_string())]
    );
}

#[test]
fn tool_call_span_buffers_until_end_token() {
    let tok = load();
    let allowed: HashSet<String> = ["f".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string());

    let ids = tok.encode(
        "\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n",
        false,
    );

    let start_events = decoder.consume(tok.tool_call_start_id, "").unwrap();
    assert!(start_events.is_empty());
    for &id in &ids {
        let events = decoder.consume(id, "").unwrap();
        assert!(events.is_empty(), "no events until the call closes");
    }
    let end_events = decoder.consume(tok.tool_call_end_id, "").unwrap();
    assert_eq!(end_events.len(), 1);
    match &end_events[0] {
        StructuredAssistantEvent::ToolCall(call) => {
            assert_eq!(call.name, "f");
            assert_eq!(call.id, "call_1");
        }
        other => panic!("expected a tool call event, got {other:?}"),
    }
    assert!(decoder.has_tool_calls());
}

/// GEMMA'S LABELLED THOUGHT CHANNEL is reasoning too, and the label itself
/// is not.
///
/// The two dialects frame it differently -- ChatML brackets a body with
/// `<think>`/`</think>`, Gemma opens a channel and names it in the first
/// line of text -- so this is a separate arm of the decoder and needs its
/// own case. The LABEL half is what the real install leaked: before the
/// caller built a decoder for this dialect, a `--reasoning` run printed a
/// bare `thought` and then the scratch work, as content.
#[test]
fn gemmas_thought_channel_becomes_reasoning_and_its_label_disappears() {
    let tok = gemma();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string());

    assert!(decoder
        .consume(tok.channel_start_id, "")
        .unwrap()
        .is_empty());
    // The label is accumulated until its newline and then consumed: a
    // channel named `thought` must not reach the caller as the word
    // "thought".
    let labelled = decoder.consume(0, "thought\n").unwrap();
    assert!(labelled.is_empty(), "the label leaked: {labelled:?}");

    let thought = decoder.consume(0, "17 - 8 = 9").unwrap();
    assert_eq!(
        thought,
        vec![StructuredAssistantEvent::Reasoning(
            "17 - 8 = 9".to_string()
        )]
    );

    assert!(decoder.consume(tok.channel_end_id, "").unwrap().is_empty());
    let answer = decoder.consume(0, "Nine.").unwrap();
    assert_eq!(
        answer,
        vec![StructuredAssistantEvent::Content("Nine.".to_string())]
    );
}

/// The counterweight: a channel Gemma names `final` is the ANSWER, not
/// reasoning. Without this, reporting every channel as reasoning would pass
/// the case above and empty every reply.
#[test]
fn gemmas_final_channel_is_content() {
    let tok = gemma();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string());
    decoder.consume(tok.channel_start_id, "").unwrap();
    let events = decoder.consume(0, "final\nNine sheep.").unwrap();
    assert_eq!(
        events,
        vec![StructuredAssistantEvent::Content("Nine sheep.".to_string())]
    );
}

/// A `<think>` body comes back as REASONING, and it used to be dropped.
///
/// The old contract (`think_block_is_suppressed`) was right while nothing
/// could turn thinking on: the rendered generation prompt closed the block
/// immediately, so a body was unreachable and discarding it cost nothing.
/// `--reasoning` / `reasoning_effort` make it reachable, and a caller who
/// asked to see the model think must not have that silently thrown away --
/// the frame tokens render to the empty string, so a dropped body is
/// indistinguishable from a model that had no thoughts.
///
/// The FRAME tokens still emit nothing, which is what keeps `<think>` out of
/// the visible reply either way.
#[test]
fn think_block_becomes_reasoning_not_content() {
    let tok = load();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string());
    let start = decoder.consume(tok.think_start_id.unwrap(), "").unwrap();
    assert!(start.is_empty(), "the frame token itself is not output");

    let thought = decoder.consume(0, "reasoning...").unwrap();
    assert_eq!(
        thought,
        vec![StructuredAssistantEvent::Reasoning(
            "reasoning...".to_string()
        )]
    );

    let end = decoder.consume(tok.think_end_id.unwrap(), "").unwrap();
    assert!(end.is_empty(), "the frame token itself is not output");

    // And the answer after it is CONTENT, which is the half that must not
    // move: a decoder that reported everything as reasoning would empty the
    // reply while looking, in the test above, exactly like this one.
    let answer = decoder.consume(0, "the answer").unwrap();
    assert_eq!(
        answer,
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}
