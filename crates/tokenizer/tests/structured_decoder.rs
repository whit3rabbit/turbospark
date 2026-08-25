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

/// Muse Glimmer's frame is `<|start|>ROLE to=RECIPIENT<|message|>BODY<|eom|>`
/// -- header/body like Harmony's, but the RECIPIENT is the channel. The
/// fixture carries the seven special tokens `resolve_muse_glimmer` requires
/// plus a template whose generation prompt stops mid-header, exactly as the
/// real checkpoint's does; the ids are placeholders, resolved at load (crate
/// Gotcha 3).
fn muse() -> MfTokenizer {
    fixture("MuseGlimmerTokenizer")
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
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);
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
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string(), &[]);

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
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);

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
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);
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
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);
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

/// A CHATML GENERATION PROMPT OPENS `<think>` ITSELF when thinking is on, so
/// the model's first generated token is already scratchpad and no
/// `think_start_id` ever arrives. Qwen's own template ends the prompt
/// `<|im_start|>assistant\n<think>\n`; froggeric's community rewrite does the
/// same. A decoder that always started in Visible stayed there -- the
/// `</think>` that arrives later flips Visible to Visible -- and reported the
/// whole scratchpad as the answer with the reasoning stream EMPTY, which is
/// what `--reasoning` did on every ChatML checkpoint before `new` took the
/// prompt.
#[test]
fn a_prompt_that_opens_think_starts_the_decoder_in_the_thought_channel() {
    let tok = load();
    let prompt = tok.encode("<|im_start|>assistant\n", false);
    let prompt: Vec<i32> = prompt
        .into_iter()
        .chain(std::iter::once(tok.think_start_id.unwrap()))
        .collect();

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &prompt);

    // No `<think>` token: the model is already inside the frame.
    assert_eq!(
        decoder.consume(0, "scratch").unwrap(),
        vec![StructuredAssistantEvent::Reasoning("scratch".to_string())]
    );
    assert!(decoder
        .consume(tok.think_end_id.unwrap(), "")
        .unwrap()
        .is_empty());
    assert_eq!(
        decoder.consume(0, "the answer").unwrap(),
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}

/// The OFF prompt closes the frame again (`<think>\n\n</think>\n\n`), so the
/// scan must read the LAST marker and not merely find one. Getting this
/// backwards routes an entire non-thinking reply to reasoning, i.e. hands the
/// caller an empty answer -- the failure this fix could most easily have
/// introduced.
#[test]
fn a_prompt_that_closes_think_again_leaves_the_decoder_visible() {
    let tok = load();
    let prompt = vec![tok.think_start_id.unwrap(), tok.think_end_id.unwrap()];

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &prompt);

    assert_eq!(
        decoder.consume(0, "the answer").unwrap(),
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}

/// A tool preamble puts BALANCED `<think></think>` pairs in its instructions
/// ("use the `<think></think>` block to plan your next tool call"), which sit
/// far behind the generation prompt. Only the TAIL decides, which is why the
/// scan runs backwards and stops at the first marker it meets.
///
/// THE TAIL HERE IS THE CLOSED ONE, and that is the whole fixture: with an
/// OPEN tail a forward scan agrees by coincidence (a balanced pair starts with
/// `<think>` too), so that shape cannot tell the two directions apart. This
/// one can -- forwards it reads the preamble's first `<think>`, opens the
/// frame, and hands the caller an empty answer.
#[test]
fn balanced_think_pairs_in_a_tool_preamble_do_not_outvote_the_prompt_tail() {
    let tok = load();
    let prompt = vec![
        tok.think_start_id.unwrap(),
        tok.think_end_id.unwrap(),
        tok.think_start_id.unwrap(),
        tok.think_end_id.unwrap(),
        // ... and then the generation prompt's own pre-closed, thinking-off
        // frame: `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
        tok.think_start_id.unwrap(),
        tok.think_end_id.unwrap(),
    ];

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &prompt);

    assert_eq!(
        decoder.consume(0, "the answer").unwrap(),
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}

/// Gemma frames its thought with a BRACKETING channel pair and has no
/// `<think>` ids at all, so the scan is a no-op there however the prompt ends
/// -- and Gemma's template does not prefill, so starting in Thought would
/// swallow its reply.
#[test]
fn the_prompt_scan_is_inert_on_a_dialect_without_the_think_pair() {
    let tok = gemma();
    assert!(tok.think_start_id.is_none());
    let prompt = tok.encode("<|turn>model\n", false);

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &prompt);

    assert_eq!(
        decoder.consume(0, "the answer").unwrap(),
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}

/// THE INVARIANT BEHIND ALL FOUR TESTS ABOVE, swept over every bundled
/// dialect fixture rather than stated for ChatML alone: a decoder's initial
/// channel must agree with where that dialect's own generation prompt LEFT the
/// model.
///
/// Two independent readings have to match. The test scans the rendered prompt
/// ids itself (a reimplementation, so this cannot rot into agreeing with the
/// implementation by construction), and the decoder is asked the same question
/// the only way a caller can -- feed it one ordinary token and see which
/// channel it comes out on.
///
/// **A NEW DIALECT GETS SWEPT BY EXISTING.** Drop a fixture directory in and
/// it is included; that is the point, because the ChatML bug this pins was
/// invisible for as long as the question was only ever asked per dialect by
/// hand.
///
/// **BUT THE INVARIANT IT CHECKS IS THE `<think>` PAIR'S**, so a dialect
/// framing its reasoning another way is swept and not actually tested: the
/// Muse Glimmer fixture passes here because it has no think ids at all, which
/// is agreement by absence. Its own frame is pinned by the four `muse_*` cases
/// below, and a third frame shape would need the same. Measured across the
/// seven fixtures here: ChatML leaves `<think>` OPEN
/// at every level and CLOSED at `off`, Gemma opens no channel at a level and
/// pre-closes an empty one at `off`, Harmony ends at `<|start|>assistant`
/// outside any frame, DeepSeek pre-closes with `</think>`, and Mistral has no
/// thought frame at all.
#[test]
fn every_dialects_decoder_starts_where_its_own_prompt_left_the_model() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("fixtures directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert!(dirs.len() >= 5, "expected the bundled dialect fixtures");

    let levels = [
        turbospark_tokenizer::ReasoningEffort::Off,
        turbospark_tokenizer::ReasoningEffort::Low,
        turbospark_tokenizer::ReasoningEffort::XHigh,
    ];
    let mut checked = 0usize;

    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let tok = MfTokenizer::load_from_dir(&dir).expect("fixture loads");

        for level in levels {
            let messages = vec![turbospark_tokenizer::Message::new(
                turbospark_tokenizer::Role::User,
                "Why is the sky blue?",
            )];
            // A refused level is this dialect declining to express one, not a
            // failure: `high` on Qwen 3.8, any level with no template at all.
            let Ok(rendered) = tok.apply_chat_template_with_reasoning(&messages, level) else {
                continue;
            };
            let ids = tok.encode(&rendered, false);

            // Reading one: the prompt itself, scanned independently.
            let mut expects_reasoning = false;
            for &id in ids.iter().rev() {
                if tok.think_start_id == Some(id) {
                    expects_reasoning = true;
                    break;
                }
                if tok.think_end_id == Some(id) {
                    break;
                }
            }

            // Reading two: the decoder, asked the only way a caller can.
            let mut decoder = StructuredAssistantDecoder::new(
                &tok,
                HashSet::new(),
                || "call_x".to_string(),
                &ids,
            );
            let events = decoder.consume(-1, "plain").unwrap();
            let got_reasoning = events
                .iter()
                .any(|e| matches!(e, StructuredAssistantEvent::Reasoning(_)));

            assert_eq!(
                got_reasoning,
                expects_reasoning,
                "{name} at {}: prompt leaves the thought frame {}, but the first \
                 generated token comes out as {}",
                level.as_str(),
                if expects_reasoning { "OPEN" } else { "closed" },
                if got_reasoning {
                    "reasoning"
                } else {
                    "content"
                },
            );
            checked += 1;
        }
    }
    assert!(checked >= 8, "swept only {checked} dialect/level pairs");
}

/// Muse Glimmer's frame is `<|start|>ROLE to=RECIPIENT<|message|>BODY<|eom|>`,
/// and the RECIPIENT is the channel: `to=self` is the model's scratchpad,
/// `to=user` is the reply.
#[test]
fn muse_routes_to_self_to_reasoning_and_to_user_to_content() {
    let tok = muse();
    let start = tok.channel_start_id;
    let message = tok.message_start_id;
    let eom = tok.message_end_id;

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);

    let mut events = Vec::new();
    events.extend(decoder.consume(start, "").unwrap());
    // The header is markup and must never reach the caller.
    events.extend(decoder.consume(-1, "assistant to=self").unwrap());
    assert!(events.is_empty(), "a header emits nothing: {events:?}");

    events.extend(decoder.consume(message, "").unwrap());
    events.extend(decoder.consume(-1, "scratch").unwrap());
    events.extend(decoder.consume(eom, "").unwrap());
    events.extend(decoder.consume(start, "").unwrap());
    events.extend(decoder.consume(-1, "assistant to=user").unwrap());
    events.extend(decoder.consume(message, "").unwrap());
    events.extend(decoder.consume(-1, "the answer").unwrap());

    assert_eq!(
        events,
        vec![
            StructuredAssistantEvent::Reasoning("scratch".to_string()),
            StructuredAssistantEvent::Content("the answer".to_string()),
        ]
    );
}

/// **A MUSE GENERATION PROMPT STOPS MID-HEADER**, at `<|start|>assistant`, so
/// the model's first tokens are the header's remainder and no `<|start|>` ever
/// arrives. Started unframed, the decoder emits ` to=self` as prose and then
/// the whole scratchpad as the reply -- measured on the real 30B install
/// before this arm existed, and with NO reasoning flag set, since this family
/// reasons on every turn.
#[test]
fn a_muse_prompt_stopping_mid_header_starts_the_decoder_in_that_header() {
    let tok = muse();
    let messages = vec![turbospark_tokenizer::Message::new(
        turbospark_tokenizer::Role::User,
        "Why is the sky blue?",
    )];
    let rendered = tok
        .apply_chat_template(&messages)
        .expect("the fixture template renders");
    assert!(
        rendered.ends_with("<|start|>assistant"),
        "rendered = {rendered:?}"
    );
    let ids = tok.encode(&rendered, false);

    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &ids);

    // The recipient arrives split across deltas, as a real tokenizer emits it.
    assert!(decoder.consume(-1, " to=").unwrap().is_empty());
    assert!(decoder.consume(-1, "self").unwrap().is_empty());
    assert!(decoder
        .consume(tok.message_start_id, "")
        .unwrap()
        .is_empty());
    assert_eq!(
        decoder.consume(-1, "scratch").unwrap(),
        vec![StructuredAssistantEvent::Reasoning("scratch".to_string())]
    );
}

/// A header with NO recipient is the reply, not reasoning: the checkpoint's
/// template defaults `recipient` to `user` when a turn names none, so
/// defaulting the other way would route an ordinary answer to stderr.
#[test]
fn a_muse_header_with_no_recipient_is_the_answer() {
    let tok = muse();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);

    assert!(decoder
        .consume(tok.channel_start_id, "")
        .unwrap()
        .is_empty());
    assert!(decoder.consume(-1, "assistant").unwrap().is_empty());
    assert!(decoder
        .consume(tok.message_start_id, "")
        .unwrap()
        .is_empty());
    assert_eq!(
        decoder.consume(-1, "the answer").unwrap(),
        vec![StructuredAssistantEvent::Content("the answer".to_string())]
    );
}

/// An unrecognized recipient is REASONING, which is the same default direction
/// Harmony's `parse_header` takes and for the same reason. This dialect's tool
/// calls are `to=<toolname>` with an `<atem:function_calls>` body and no parser
/// wired, so the choice is between putting unparseable markup on the reasoning
/// stream or emitting it as the reply.
#[test]
fn a_muse_tool_recipient_is_reasoning_rather_than_the_reply() {
    let tok = muse();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, HashSet::new(), || "call_x".to_string(), &[]);

    assert!(decoder
        .consume(tok.channel_start_id, "")
        .unwrap()
        .is_empty());
    assert!(decoder
        .consume(-1, "assistant to=weather.get")
        .unwrap()
        .is_empty());
    assert!(decoder
        .consume(tok.message_start_id, "")
        .unwrap()
        .is_empty());
    assert_eq!(
        decoder.consume(-1, "<atem:function_calls>").unwrap(),
        vec![StructuredAssistantEvent::Reasoning(
            "<atem:function_calls>".to_string()
        )]
    );
}
