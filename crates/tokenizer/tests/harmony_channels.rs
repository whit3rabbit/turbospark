//! `gpt-oss`'s Harmony channel frame, decoded (ROADMAP Later list).
//!
//! Harmony is the one dialect here whose channels do not BRACKET: an assistant
//! turn reads
//!
//! ```text
//! <|channel|>analysis<|message|>REASONING<|end|>
//! <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
//! ```
//!
//! so the reasoning arrives BEFORE the answer and the two have to be told
//! apart by a header parser rather than by a start/end token pair.
//!
//! Every id here is resolved from the loaded tokenizer rather than read out of
//! the fixture JSON (crate Gotcha 3), and the decoder is driven the way
//! `run_raw_completion` drives it: one `(token_id, delta)` pair per token. The
//! three tokens that CLOSE a turn (`<|return|>`, `<|call|>`, `<|endoftext|>`)
//! are deliberately absent from these transcripts -- they are stop tokens, and
//! the generation loop breaks before the progress callback, so the decoder
//! never sees them.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{
    MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent as Event,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/HarmonyTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Harmony fixture loads")
}

fn decoder(tok: &MfTokenizer) -> StructuredAssistantDecoder<'_> {
    StructuredAssistantDecoder::new(tok, HashSet::new(), || "toolu_0".to_string())
}

/// A frame token carries no visible text of its own.
fn mark(id: i32) -> (i32, &'static str) {
    (id, "")
}

/// Feeds a whole transcript and returns every event it produced, in order.
fn run(tok: &MfTokenizer, transcript: &[(i32, &str)]) -> Vec<Event> {
    let mut decoder = decoder(tok);
    let mut events = Vec::new();
    for &(id, delta) in transcript {
        events.extend(decoder.consume(id, delta).expect("Harmony never fails"));
    }
    events.extend(decoder.finish().expect("a closed frame finishes cleanly"));
    events
}

fn text_id(tok: &MfTokenizer) -> i32 {
    tok.token_to_id("h").expect("a plain vocabulary token")
}

/// THE WHOLE POINT: a real two-message turn splits into reasoning and content,
/// and the `<|start|>assistant` between the two messages contributes NOTHING.
/// Passing that through would put the bare word "assistant" in the reply.
#[test]
fn analysis_becomes_reasoning_and_final_becomes_content() {
    let tok = fixture();
    let t = text_id(&tok);
    let start = tok.token_to_id("<|start|>").expect("<|start|> resolves");

    let events = run(
        &tok,
        &[
            mark(tok.channel_start_id),
            (t, "analysis"),
            mark(tok.message_start_id),
            (t, "The user asks "),
            (t, "why the sky is blue."),
            mark(tok.message_end_id),
            mark(start),
            (t, "assistant"),
            mark(tok.channel_start_id),
            (t, "final"),
            mark(tok.message_start_id),
            (t, "Rayleigh scattering."),
        ],
    );

    assert_eq!(
        events,
        vec![
            Event::Reasoning("The user asks ".to_string()),
            Event::Reasoning("why the sky is blue.".to_string()),
            Event::Content("Rayleigh scattering.".to_string()),
        ]
    );
}

/// The header is not one token. `analysis` may arrive as several deltas, and a
/// parser that read only the first would classify on a prefix.
#[test]
fn a_header_split_across_deltas_still_parses() {
    let tok = fixture();
    let t = text_id(&tok);

    let events = run(
        &tok,
        &[
            mark(tok.channel_start_id),
            (t, "fin"),
            (t, "al"),
            mark(tok.message_start_id),
            (t, "answer"),
        ],
    );

    assert_eq!(events, vec![Event::Content("answer".to_string())]);
}

/// A tool call's header carries a recipient and a constraint after the channel
/// name (`commentary to=functions.get_weather <|constrain|>json`). The channel
/// is the FIRST word, and `commentary` is not the answer.
///
/// Decoding the call itself is a separate item: Harmony frames it as a header
/// recipient rather than as the bracketing token pair the decoder's tool
/// contract describes, which is why `resolve_harmony` leaves every tool id
/// `NO_SUCH_TOKEN_ID`.
#[test]
fn a_commentary_header_with_a_recipient_is_reasoning_not_content() {
    let tok = fixture();
    let t = text_id(&tok);
    let constrain = tok
        .token_to_id("<|constrain|>")
        .expect("<|constrain|> resolves");

    let events = run(
        &tok,
        &[
            mark(tok.channel_start_id),
            (t, "commentary to=functions.get_weather "),
            mark(constrain),
            (t, "json"),
            mark(tok.message_start_id),
            (t, r#"{"city":"Oslo"}"#),
        ],
    );

    assert_eq!(
        events,
        vec![Event::Reasoning(r#"{"city":"Oslo"}"#.to_string())]
    );
}

/// AN UNRECOGNIZED CHANNEL IS REASONING, and the direction is the assertion.
/// A new channel reported as reasoning shows up in the wrong place; one
/// reported as the answer corrupts the reply.
#[test]
fn an_unknown_channel_defaults_to_reasoning() {
    let tok = fixture();
    let t = text_id(&tok);

    let events = run(
        &tok,
        &[
            mark(tok.channel_start_id),
            (t, "some_future_channel"),
            mark(tok.message_start_id),
            (t, "body"),
        ],
    );

    assert_eq!(events, vec![Event::Reasoning("body".to_string())]);
}

/// A model that never emits the frame is REPORTED, not silenced: text before
/// the first `<|channel|>` passes through as content, which is exactly what
/// this dialect did before the frame was decoded at all.
#[test]
fn text_before_any_channel_passes_through_as_content() {
    let tok = fixture();
    let t = text_id(&tok);

    let events = run(&tok, &[(t, "bare text"), (t, " and more")]);

    assert_eq!(
        events,
        vec![
            Event::Content("bare text".to_string()),
            Event::Content(" and more".to_string()),
        ]
    );
}

/// A generation may stop on max-tokens partway through the analysis channel.
/// That is an ordinary outcome, not a malformed stream, so `finish` must not
/// report it as one -- unlike an unterminated tool-call span, which is.
#[test]
fn a_generation_cut_off_mid_analysis_finishes_cleanly() {
    let tok = fixture();
    let t = text_id(&tok);
    let mut decoder = decoder(&tok);

    for &(id, delta) in &[
        mark(tok.channel_start_id),
        (t, "analysis"),
        mark(tok.message_start_id),
        (t, "half a thou"),
    ] {
        decoder.consume(id, delta).expect("Harmony never fails");
    }

    assert_eq!(
        decoder.finish().expect("an unclosed body is not an error"),
        Vec::new()
    );
}

/// The dialect's own ids are what the state machine keys on, so a Harmony
/// install has to publish them. `channel_end_id` stays absent on purpose:
/// `<|channel|>` has no closing counterpart, and its presence is what would
/// make the bracketing arm reachable.
#[test]
fn the_frame_ids_resolve_and_the_bracketing_pair_does_not() {
    let tok = fixture();
    assert_eq!(
        tok.channel_start_id,
        tok.token_to_id("<|channel|>").unwrap()
    );
    assert_eq!(
        tok.message_start_id,
        tok.token_to_id("<|message|>").unwrap()
    );
    assert_eq!(tok.message_end_id, tok.token_to_id("<|end|>").unwrap());
    assert_eq!(tok.channel_end_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
}
