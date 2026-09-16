//! `ChatDialect::tool_call_support` against what each dialect's decoder arm
//! can actually emit.
//!
//! **THIS IS NOT A RESTATEMENT OF THE MATCH.** The assertion is the LINK: for
//! every fixture, feed that dialect its own tool markup and require a
//! `ToolCall` event out iff the predicate answers `Native`. Wiring a parser
//! for a dialect that currently answers `Prompted` (Muse Glimmer's
//! `<atem:function_calls>` is the live candidate) keeps this green only if the
//! predicate moves with it, and doing either one alone reddens.
//!
//! It also arms the `debug_assert!`s in the decoder's four `ToolCall`
//! construction sites for the two arms nothing else drives: ChatML's is
//! covered by `structured_decoder.rs` and Harmony's by `harmony_channels.rs`,
//! while Gemma's and DeepSeek's had no decoder-level driver at all before
//! this file. A mutation flipping either of those arms survived until it did.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent, ToolCallSupport,
};

fn fixture(name: &str) -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

/// The DSML sentinel DeepSeek frames its calls with, spelled out here rather
/// than imported so a change to the constant is visible as a test failure
/// instead of following along silently.
const DSML: &str = "\u{FF5C}DSML\u{FF5C}";

/// Every markup this engine knows how to parse, as ONE blob.
///
/// Fed to the `Prompted` dialects: a dialect that answers `Prompted` must not
/// emit a call for ANY of these, not merely for its own (which it has none
/// of). That is what makes the negative half of this file a real assertion
/// rather than a tautology about an empty input.
fn every_dialects_markup() -> String {
    format!(
        "call:f{{x:1}}\n\
         <function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n\
         <{DSML}tool_calls><{DSML}invoke name=\"f\">\n\
         <{DSML}parameter name=\"x\" string=\"false\">1</{DSML}parameter>\n\
         </{DSML}invoke></{DSML}tool_calls>\n\
         <|channel|>commentary to=functions.f<|message|>{{\"x\":1}}<|call|>\n\
         <atem:function_calls><atem:invoke name=\"f\"><atem:parameter name=\"x\">1\
         </atem:parameter></atem:invoke></atem:function_calls>\n"
    )
}

/// One frame of the stream a decoder really sees: an id and the text the
/// streaming detokenizer released for it.
type Delta = (i32, String);

/// One row of the case table: a fixture name and how to build that dialect's
/// own tool markup as a delta stream.
type Case = (&'static str, Box<dyn Fn(&MfTokenizer) -> Vec<Delta>>);

/// Mistral's markup: the `[TOOL_CALLS]` marker as a special token (empty
/// delta, root Gotcha 44) followed by the JSON-array body as ordinary text,
/// and NO closing token -- end of turn closes it, which is the decoder's
/// `finish`, so this case is the one that exercises that path.
fn mistral_calls(tok: &MfTokenizer) -> Vec<Delta> {
    let mut deltas = vec![(tok.tool_call_start_id, String::new())];
    let body = r#"[{"name": "f", "arguments": {"x": 1}}]"#;
    for id in tok.encode(body, false) {
        deltas.push((id, tok.decode(&[id], true)));
    }
    deltas
}

/// Drive a decoder over a delta stream and collect events.
fn run(tok: &MfTokenizer, deltas: &[Delta]) -> Vec<StructuredAssistantEvent> {
    let allowed: HashSet<String> = ["f".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(tok, allowed, || "call_1".to_string(), &[]);
    let mut events = Vec::new();
    for (id, text) in deltas {
        match decoder.consume(*id, text) {
            Ok(batch) => events.extend(batch),
            // A `Prompted` dialect fed foreign markup may legitimately refuse
            // it. What matters is that no CALL came out, so a parse error is
            // as good an outcome as content and the collected events stand.
            Err(_) => break,
        }
    }
    // The consumers all call this; Mistral's arm emits its call HERE and
    // nowhere else, so a harness that stopped at the loop would prove
    // nothing about it. A finish error (a failed span) adds nothing and the
    // collected events stand, the same verdict the loop's break takes.
    if let Ok(batch) = decoder.finish() {
        events.extend(batch);
    }
    events
}

/// A body bracketed by this dialect's tool-call token pair.
///
/// Gemma and ChatML accumulate the TOKEN IDS between the pair and decode them
/// as a unit, so this hands over real ids. The bracketing tokens themselves
/// carry an empty delta, which is what a special token really renders to
/// (root Gotcha 44).
fn bracketed(tok: &MfTokenizer, body: &str) -> Vec<Delta> {
    let mut deltas = vec![(tok.tool_call_start_id, String::new())];
    for id in tok.encode(body, false) {
        deltas.push((id, tok.decode(&[id], true)));
    }
    deltas.push((tok.tool_call_end_id, String::new()));
    deltas
}

/// A plain-text dialect's markup, delivered as ONE delta.
///
/// **NOT one delta per token, and the difference is not cosmetic.** DeepSeek's
/// `\u{FF5C}DSML\u{FF5C}` sentinel is multi-byte, the toy fixtures split it
/// across several tokens, and decoding those one at a time yields replacement
/// characters -- so a per-token harness reports the decoder finding no open
/// mark when the real `StreamingDetokenizer` (which withholds partial UTF-8
/// and releases it as a tail) would have handed it over intact. That is a
/// property of the harness, not of the arm under test.
fn as_one_delta(tok: &MfTokenizer, text: &str) -> Vec<Delta> {
    let id = *tok
        .encode("x", false)
        .first()
        .expect("the fixture encodes an ordinary character");
    vec![(id, text.to_string())]
}

fn has_call(events: &[StructuredAssistantEvent]) -> bool {
    events
        .iter()
        .any(|e| matches!(e, StructuredAssistantEvent::ToolCall(_)))
}

/// The whole point of the file. One case per fixture, each feeding the markup
/// that dialect would really produce.
#[test]
fn each_dialects_decoder_emits_a_call_exactly_when_the_predicate_says_native() {
    // (fixture, the ids that dialect's own tool markup decodes to)
    let cases: Vec<Case> = vec![
        // Gemma brackets a `call:name{...}` body with its tool-call pair.
        (
            "GemmaTokenizer",
            Box::new(|t: &MfTokenizer| bracketed(t, "call:f{x:1}")),
        ),
        // ChatML brackets Qwen's `<function=...>` body with the same shape.
        (
            "ChatMLTokenizer",
            Box::new(|t: &MfTokenizer| {
                bracketed(
                    t,
                    "<function=f>\n<parameter=x>\n1\n</parameter>\n</function>",
                )
            }),
        ),
        // DeepSeek frames in PLAIN TEXT, so there is no bracketing pair to use.
        (
            "DeepseekTokenizer",
            Box::new(|t: &MfTokenizer| {
                let text = format!(
                    "<{DSML}tool_calls><{DSML}invoke name=\"f\">\n\
                     <{DSML}parameter name=\"x\" string=\"false\">1</{DSML}parameter>\n\
                     </{DSML}invoke></{DSML}tool_calls>"
                );
                as_one_delta(t, &text)
            }),
        ),
        // Mistral brackets nothing: its `[TOOL_CALLS]` marker opens a span
        // whose body runs to end of turn (`mistral_calls` above).
        (
            "ZephyrTokenizer",
            Box::new(|t: &MfTokenizer| mistral_calls(t)),
        ),
        // The two that answer `Prompted` get EVERY dialect's markup, so a
        // zero here is a statement about the decoder rather than about the
        // input being empty.
        (
            "MuseGlimmerTokenizer",
            Box::new(|t: &MfTokenizer| as_one_delta(t, &every_dialects_markup())),
        ),
        (
            "Llama3Tokenizer",
            Box::new(|t: &MfTokenizer| as_one_delta(t, &every_dialects_markup())),
        ),
    ];

    for (name, deltas_for) in cases {
        let tok = fixture(name);
        let events = run(&tok, &deltas_for(&tok));
        let expected = tok.dialect.tool_call_support() == ToolCallSupport::Native;
        assert_eq!(
            has_call(&events),
            expected,
            "{name} ({:?}) answers {:?} but its decoder {} a call: {events:?}",
            tok.dialect,
            tok.dialect.tool_call_support(),
            if has_call(&events) {
                "emitted"
            } else {
                "emitted no"
            }
        );
    }
}

/// The reason string exists exactly when the support does not, and it names
/// the dialect rather than being a generic sentence -- a GUI puts it in front
/// of a user who has to decide whether the control is worth reaching for.
#[test]
fn a_reason_is_present_exactly_when_support_is_not_native() {
    for dialect in [
        ChatDialect::Gemma,
        ChatDialect::ChatMl,
        ChatDialect::Deepseek,
        ChatDialect::Mistral,
        ChatDialect::Harmony,
        ChatDialect::MuseGlimmer,
        ChatDialect::Llama3,
    ] {
        let reason = dialect.tool_call_unsupported_reason();
        match dialect.tool_call_support() {
            ToolCallSupport::Native => assert!(reason.is_none(), "{dialect:?} named a reason"),
            ToolCallSupport::Prompted => {
                let reason = reason.unwrap_or_else(|| panic!("{dialect:?} named no reason"));
                assert!(reason.len() > 40, "{dialect:?}: {reason}");
            }
        }
    }
}

/// Muse Glimmer is the one dialect that DOES frame tool calls and still
/// answers `Prompted`, so it is the arm most likely to be "corrected" by a
/// future reader who greps for the markup and not for the parser. Pinned
/// separately with the reason, so the fix is to wire a parser rather than to
/// flip the answer.
#[test]
fn muse_glimmer_frames_calls_and_still_answers_prompted() {
    assert_eq!(
        ChatDialect::MuseGlimmer.tool_call_support(),
        ToolCallSupport::Prompted
    );
    let reason = ChatDialect::MuseGlimmer
        .tool_call_unsupported_reason()
        .expect("a reason");
    assert!(
        reason.contains("atem:function_calls"),
        "the reason must name the markup that exists but is unparsed: {reason}"
    );
}
