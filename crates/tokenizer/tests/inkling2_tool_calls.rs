//! Inkling 2's tool-call frame through the shared decoder, on the fixture's
//! real added-token table. The stream is the one a real install produced.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{
    MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent as Event, ToolCallSupport,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/Inkling2Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture")
}

/// Marks go in as ids with no text, as a real stream delivers them; anything
/// else is a text delta.
fn run(prompt: &[&str], stream: &[&str]) -> (Vec<Event>, Result<(), ()>) {
    let tok = fixture();
    let id = |s: &str| tok.token_to_id(s).expect("fixture mark");
    let prompt: Vec<i32> = prompt.iter().map(|s| id(s)).collect();
    let tools: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, tools, || "call_1".into(), &prompt);
    let mut events = Vec::new();
    for piece in stream {
        let (token, delta) = match piece.starts_with("<|") {
            true => (id(piece), ""),
            false => (-1, *piece),
        };
        match decoder.consume(token, delta) {
            Ok(e) => events.extend(e),
            Err(_) => return (events, Err(())),
        }
    }
    match decoder.finish() {
        Ok(e) => events.extend(e),
        Err(_) => return (events, Err(())),
    }
    (events, Ok(()))
}

const M: &str = "<|message_model|>";
const THINK: &str = "<|content_thinking|>";
const TEXT: &str = "<|content_text|>";
const CALL: &str = "<|content_invoke_tool_json|>";
const END: &str = "<|end_message|>";

#[test]
fn a_call_is_handed_over_and_its_recipient_is_not_reply_text() {
    // The fixture's body goes in as one delta, so the token path is a mark.
    let tok = fixture();
    assert_eq!(tok.dialect.tool_call_support(), ToolCallSupport::Native);
    let body = r#"{"name":"get_weather","args":{"city":"Oslo"}}"#;
    let ids = tok.encode(body, false);
    let id = |s: &str| tok.token_to_id(s).unwrap();
    let mut decoder = StructuredAssistantDecoder::new(
        &tok,
        ["get_weather".to_string()].into_iter().collect(),
        || "call_1".into(),
        &[],
    );
    let mut events = Vec::new();
    let mut feed = |token: i32, delta: &str| {
        events.extend(decoder.consume(token, delta).expect("consume"));
    };
    for (token, delta) in [(id(M), ""), (id(THINK), ""), (-1, "plan"), (id(END), "")] {
        feed(token, delta);
    }
    feed(id(M), "");
    feed(-1, "get_weather");
    feed(id(CALL), "");
    for &t in &ids {
        feed(t, &tok.decode(&[t], false));
    }
    feed(id(END), "");
    assert!(decoder.has_tool_calls());
    events.extend(decoder.finish().expect("finish"));
    assert_eq!(events.len(), 2, "{events:?}");
    assert_eq!(events[0], Event::Reasoning("plan".into()));
    let Event::ToolCall(call) = &events[1] else {
        panic!("{events:?}")
    };
    assert_eq!(
        (call.name.as_str(), call.id.as_str()),
        ("get_weather", "call_1")
    );
    assert_eq!(call.arguments_json, r#"{"city":"Oslo"}"#);
}

#[test]
fn a_text_reply_is_unchanged_and_the_header_is_not_lost() {
    let (events, ok) = run(&[], &[M, TEXT, "hello", END]);
    assert_eq!((events, ok), (vec![Event::Content("hello".into())], Ok(())));
    // A prompt that ends inside the header: the name is still framing.
    let (events, ok) = run(&[M], &["x", TEXT, "hi", END]);
    assert_eq!((events, ok), (vec![Event::Content("hi".into())], Ok(())));
    // No content marker ever follows: the text comes back at the end.
    let (events, ok) = run(&[], &[M, "stray"]);
    assert_eq!((events, ok), (vec![Event::Content("stray".into())], Ok(())));
}

#[test]
fn a_bad_or_unclosed_call_is_an_error_not_a_call() {
    for body in [
        "{not json",
        r#"{"name":"rm_rf","args":{}}"#,
        r#"{"name":"get_weather","args":"Oslo"}"#,
        r#"{"name":"get_weather"}"#,
        r#"{"name":"get_weather","args":{},"extra":1}"#,
    ] {
        // Text deltas carry no ids here, so the body decodes empty: encode it.
        let tok = fixture();
        let id = |s: &str| tok.token_to_id(s).unwrap();
        let mut decoder = StructuredAssistantDecoder::new(
            &tok,
            ["get_weather".to_string()].into_iter().collect(),
            || "c".into(),
            &[],
        );
        decoder.consume(id(M), "").unwrap();
        decoder.consume(id(CALL), "").unwrap();
        for t in tok.encode(body, false) {
            decoder.consume(t, &tok.decode(&[t], false)).unwrap();
        }
        assert!(decoder.consume(id(END), "").is_err(), "{body}");
        assert!(!decoder.has_tool_calls());
    }
    let (_, ok) = run(&[], &[M, CALL]);
    assert_eq!(ok, Err(()), "unclosed");
}
