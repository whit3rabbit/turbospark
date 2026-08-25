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

/// The channel header a real `gpt-oss` tool call opens with. `functions` is
/// the namespace Harmony renders caller-supplied tools into.
const HEADER: &str = "commentary to=functions.get_weather ";

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/HarmonyTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("Harmony fixture loads")
}

fn decoder(tok: &MfTokenizer) -> StructuredAssistantDecoder<'_> {
    StructuredAssistantDecoder::new(tok, HashSet::new(), || "toolu_0".to_string(), &[])
}

/// A decoder that has been OFFERED `get_weather`, which is what makes a
/// `to=functions.get_weather` header a call rather than an ordinary body.
fn tool_decoder(tok: &MfTokenizer) -> StructuredAssistantDecoder<'_> {
    let allowed = HashSet::from(["get_weather".to_string()]);
    StructuredAssistantDecoder::new(tok, allowed, || "toolu_0".to_string(), &[])
}

/// A frame token carries no visible text of its own.
fn mark(id: i32) -> (i32, &'static str) {
    (id, "")
}

/// Feeds a whole transcript and returns every event it produced, in order.
fn run(tok: &MfTokenizer, transcript: &[(i32, &str)]) -> Vec<Event> {
    run_with(decoder(tok), transcript)
}

fn run_with(mut decoder: StructuredAssistantDecoder<'_>, transcript: &[(i32, &str)]) -> Vec<Event> {
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
/// **A CALLER THAT OFFERED NO TOOLS GETS THE BODY AS REASONING**, which is the
/// direction that matters: this decoder is built for every Harmony generation,
/// tools or not (server crate Gotcha 12), so failing an unoffered tool here
/// would turn every CLI tool call into a lost turn. The next test is the same
/// transcript with the tool offered.
#[test]
fn a_commentary_body_is_reasoning_when_the_caller_offered_no_tools() {
    let tok = fixture();

    let events = run(&tok, &tool_call_transcript(&tok, HEADER));

    assert_eq!(
        events,
        vec![Event::Reasoning(r#"{"city":"Oslo"}"#.to_string())]
    );
}

/// THE HEADLINE OF THE TOOL-CALL ITEM, and the assertion is WHEN as much as
/// what. Harmony ends a call with `<|call|>`, `<|call|>` is in the stop set,
/// and `run_raw_completion` breaks before the progress callback -- so the
/// decoder never sees the token that terminates the span it is parsing, and
/// the call has to come out of `finish`. A consumer that only drives `consume`
/// sees nothing at all, which is exactly what the server used to do.
#[test]
fn a_tool_call_is_emitted_from_finish_because_its_terminator_is_a_stop_token() {
    let tok = fixture();
    let mut decoder = tool_decoder(&tok);

    for &(id, delta) in &tool_call_transcript(&tok, HEADER) {
        assert_eq!(
            decoder.consume(id, delta).expect("a well-formed call"),
            Vec::new(),
            "nothing may be emitted before the span closes"
        );
    }
    assert!(!decoder.has_tool_calls(), "not until finish");

    let events = decoder.finish().expect("a complete body parses");
    let [Event::ToolCall(call)] = events.as_slice() else {
        panic!("expected exactly one call, got {events:?}");
    };
    assert_eq!(call.name, "get_weather", "the namespace is stripped");
    assert_eq!(call.id, "toolu_0");
    assert_eq!(call.arguments_json, r#"{"city":"Oslo"}"#);
    assert!(decoder.has_tool_calls());
}

/// A BUILTIN NAMESPACE IS NOT A CALLER TOOL, and the test offers `python` by
/// name so the allowlist alone cannot be what rejects it. `functions` is the
/// namespace Harmony renders caller-supplied tools into; `python`'s body is
/// source code rather than JSON, so treating one as a call would fail to parse
/// and poison the stream rather than degrade.
#[test]
fn a_recipient_outside_the_functions_namespace_is_not_a_call() {
    let tok = fixture();
    let allowed = HashSet::from(["python".to_string()]);
    let decoder = StructuredAssistantDecoder::new(&tok, allowed, || "toolu_0".to_string(), &[]);

    let events = run_with(
        decoder,
        &tool_call_transcript(&tok, "commentary to=python "),
    );

    assert_eq!(
        events,
        vec![Event::Reasoning(r#"{"city":"Oslo"}"#.to_string())]
    );
}

/// A generation that ran out of budget partway through the arguments is a
/// MALFORMED call rather than a call with odd arguments -- the same verdict
/// the Gemma arm reaches on an unterminated tool span. Note this is the one
/// way the Harmony arm can fail at all, which is why `consume_harmony` now
/// returns a `Result`.
#[test]
fn a_body_cut_off_mid_json_is_malformed() {
    let tok = fixture();
    let t = text_id(&tok);
    let mut decoder = tool_decoder(&tok);

    let mut transcript = tool_call_transcript(&tok, HEADER);
    transcript.pop();
    transcript.push((t, r#"{"city":"Os"#));
    for (id, delta) in transcript {
        decoder.consume(id, delta).expect("no failure until finish");
    }

    assert!(
        decoder.finish().is_err(),
        "half a JSON object is not a call"
    );
}

/// The arguments must be an OBJECT. Every wire format this feeds carries them
/// as one, and every other parser here builds one, so a bare array is a
/// malformed call rather than a call whose arguments happen to be a list.
#[test]
fn a_non_object_body_is_malformed() {
    let tok = fixture();
    let t = text_id(&tok);
    let mut decoder = tool_decoder(&tok);

    let mut transcript = tool_call_transcript(&tok, HEADER);
    transcript.pop();
    transcript.push((t, "[1, 2]"));
    for (id, delta) in transcript {
        decoder.consume(id, delta).expect("no failure until finish");
    }

    assert!(decoder.finish().is_err());
}

/// `<|end|>` closes a tool body too, and the turn carries on afterwards. Not
/// the shape a real `gpt-oss` turn takes (`<|call|>` ends the generation), but
/// the emit path is shared with `finish`, so this is what says the two agree.
#[test]
fn a_tool_body_closed_by_end_is_a_call_and_the_turn_continues() {
    let tok = fixture();
    let t = text_id(&tok);
    let mut transcript = tool_call_transcript(&tok, HEADER);
    transcript.extend_from_slice(&[
        mark(tok.message_end_id),
        mark(tok.channel_start_id),
        (t, "final"),
        mark(tok.message_start_id),
        (t, "It is cold."),
    ]);

    let events = run_with(tool_decoder(&tok), &transcript);

    let [Event::ToolCall(call), Event::Content(answer)] = events.as_slice() else {
        panic!("expected a call then an answer, got {events:?}");
    };
    assert_eq!(call.name, "get_weather");
    assert_eq!(answer, "It is cold.");
}

/// The recipient is subject to the same split-across-deltas problem the channel
/// name is: `functions.get_weather` is several tokens, and a parser reading
/// only the first delta would find `to=funct`.
#[test]
fn a_recipient_split_across_deltas_still_resolves() {
    let tok = fixture();
    let t = text_id(&tok);

    let events = run_with(
        tool_decoder(&tok),
        &[
            mark(tok.channel_start_id),
            (t, "commentary to=fun"),
            (t, "ctions.get_"),
            (t, "weather"),
            mark(tok.message_start_id),
            (t, "{}"),
        ],
    );

    let [Event::ToolCall(call)] = events.as_slice() else {
        panic!("expected one call, got {events:?}");
    };
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments_json, "{}");
}

/// The transcript a real `gpt-oss` tool call produces, minus its `<|call|>`
/// terminator, which the decoder never sees. The header is a parameter so a
/// caller can vary the recipient's NAMESPACE, which is the axis that decides
/// whether this is a call at all.
fn tool_call_transcript<'a>(tok: &MfTokenizer, header: &'a str) -> Vec<(i32, &'a str)> {
    let constrain = tok
        .token_to_id("<|constrain|>")
        .expect("<|constrain|> resolves");
    let t = text_id(tok);
    vec![
        mark(tok.channel_start_id),
        (t, header),
        mark(constrain),
        (t, "json"),
        mark(tok.message_start_id),
        (t, r#"{"city":"Oslo"}"#),
    ]
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

/// `<|call|>` is the member of Harmony's three-token stop set that means the
/// model is INVOKING something rather than finishing. `run_raw_completion`'s
/// ladder has no other way to tell -- it is neither the turn end nor the
/// tool-response marker, and both of those comparisons are asserted here so a
/// future resolver cannot collapse them and leave the ladder reading `Eos`.
#[test]
fn the_tool_call_stop_is_call_and_is_neither_the_turn_end_nor_a_response_marker() {
    let tok = fixture();
    let call = tok.token_to_id("<|call|>").expect("<|call|> resolves");

    assert_eq!(tok.tool_call_stop_id, call);
    assert!(tok.stop_token_ids.contains(&call));
    assert_ne!(tok.end_of_turn_id, call, "the turn end is <|return|>");
    assert_eq!(tok.tool_response_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
}
