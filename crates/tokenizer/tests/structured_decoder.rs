//! Streaming structured-decoder tests against the ChatML fixture: tool-call
//! token spans are buffered and parsed atomically, plain text streams as
//! content immediately.

use std::collections::HashSet;
use std::path::PathBuf;

use mrefrust_tokenizer::{MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent};

fn load() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
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

#[test]
fn think_block_is_suppressed() {
    let tok = load();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string());
    let start = decoder.consume(tok.think_start_id.unwrap(), "").unwrap();
    assert!(start.is_empty());
    let hidden = decoder.consume(0, "reasoning...").unwrap();
    assert!(hidden.is_empty());
    let end = decoder.consume(tok.think_end_id.unwrap(), "").unwrap();
    assert!(end.is_empty());
}
