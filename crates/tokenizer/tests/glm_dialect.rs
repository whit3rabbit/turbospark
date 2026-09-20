//! The GLM dialect: detection, resolution, and the text-marker decoder arm.
//!
//! **THE FIXTURE'S ADDED-TOKEN TABLE IS REAL.** `fixtures/GlmTokenizer` is
//! reduced from `zai-org/GLM-4.7-Flash`'s own `tokenizer.json` (read
//! 2026-09-19): all 36 added tokens with their real ids and special flags
//! (`<|endoftext|>` 154820, `<|user|>` 154827, `<|observation|>` 154829, the
//! think pair 154841/154842, `<sop>` 154824), the real ByteLevel pre/post
//! components, and the real vocab entries for all 256 byte tokens --
//! everything the dialect machinery reads. What is dropped is the 150k-entry
//! BPE body (whole-word merges), which no resolver, probe, or decoder arm
//! touches.
//!
//! **EVERY ID ASSERTION IS BY NAME, NOT BY NUMBER.** The `tokenizers`
//! library renumbers a sparse fixture's ids on load (crate Gotcha 2), so the
//! assertions pin the RESOLUTION -- the resolver's ids are the named
//! markers' ids -- which is also exactly how `resolve_glm` itself reads the
//! table; a real install's table is dense and keeps its real ids.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent,
    ToolCallSupport,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/GlmTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("the GLM fixture tokenizer should load")
}

#[test]
fn the_real_table_resolves_to_the_glm_dialect_with_the_named_markers() {
    let tok = fixture();
    assert_eq!(tok.dialect, ChatDialect::Glm);
    let id = |name: &str| {
        tok.token_to_id(name)
            .unwrap_or_else(|| panic!("{name} must resolve in the loaded table"))
    };
    assert_eq!(tok.bos_id, id("<sop>"));
    assert_eq!(tok.eos_id, id("<|endoftext|>"));
    assert_eq!(tok.end_of_turn_id, id("<|endoftext|>"));
    assert_eq!(tok.tool_call_stop_id, id("<|observation|>"));
    assert_eq!(tok.think_start_id, Some(id("<think>")));
    assert_eq!(tok.think_end_id, Some(id("</think>")));
    assert_eq!(tok.channel_start_id, id("<think>"));
    assert_eq!(tok.channel_end_id, id("</think>"));
    // The template writes `[gMASK]<sop>` itself, so the encoder adds no BOS:
    // add_bos must leave the id sequence exactly the encoded text's.
    assert_eq!(
        tok.encode("x", true),
        tok.encode("x", false),
        "no BOS prefix may be prepended"
    );
    // generation_config.json's three end-of-sequence ids are all stops.
    assert!(tok.stop_token_ids.contains(&id("<|endoftext|>")));
    assert!(tok.stop_token_ids.contains(&id("<|user|>")));
    assert!(tok.stop_token_ids.contains(&id("<|observation|>")));
    // The tool markup ids exist in the table but the arm is a text scan;
    // carrying them would claim an id-bracket arm.
    assert_eq!(tok.tool_call_start_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(tok.tool_call_end_id, turbospark_tokenizer::NO_SUCH_TOKEN_ID);
    assert_eq!(tok.dialect.tool_call_support(), ToolCallSupport::Native);
}

/// One frame of the stream a decoder really sees: an id and the text the
/// streaming detokenizer released for it. The tool markers are added but
/// not special, so THEIR deltas carry the literal markup -- which is why the
/// arm is a text scan at all.
type Delta = (i32, String);

fn deltas(tok: &MfTokenizer, text: &str) -> Vec<Delta> {
    tok.encode(text, false)
        .into_iter()
        .map(|id| (id, tok.decode(&[id], true)))
        .collect()
}

fn run(tok: &MfTokenizer, deltas_list: &[Delta], prompt_ids: &[i32]) -> Vec<StructuredAssistantEvent> {
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder =
        StructuredAssistantDecoder::new(tok, allowed, || "call_1".to_string(), prompt_ids);
    let mut events = Vec::new();
    for (id, text) in deltas_list {
        match decoder.consume(*id, text) {
            Ok(batch) => events.extend(batch),
            Err(_) => break,
        }
    }
    if let Ok(batch) = decoder.finish() {
        events.extend(batch);
    }
    events
}

/// A tool call between prose parses out of the TEXT stream: the markup
/// arrives as ordinary deltas and comes back as one parsed call, with the
/// surrounding prose intact and no markup in it. The template renders string
/// argument values RAW and non-string values JSON-encoded, and the parser
/// reads them back as the same types.
#[test]
fn a_tool_call_between_prose_parses_from_the_text_stream() {
    let tok = fixture();
    let stream = "Sure. \
                  <tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value>\
                  <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>\
                  Done!";
    let events = run(&tok, &deltas(&tok, stream), &[]);
    let mut calls = Vec::new();
    let mut text = String::new();
    for event in events {
        match event {
            StructuredAssistantEvent::ToolCall(call) => calls.push(call),
            StructuredAssistantEvent::Content(part) => text.push_str(&part),
            StructuredAssistantEvent::Reasoning(_) => panic!("no reasoning was prompted"),
        }
    }
    assert_eq!(calls.len(), 1, "{text}");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments_json, r#"{"city":"Oslo","days":3}"#);
    assert_eq!(text, "Sure. Done!");
}

/// The markup tokens are ATOMIC ids on the per-token path, but the
/// flushed-text path (and a detokenizer with its own chunking) can split a
/// marker across deltas. A buffer tail that could be the front of the open
/// mark must be withheld, not emitted -- a leaked `<tool_` in the visible
/// stream is the failure this test pins.
#[test]
fn a_marker_split_across_deltas_is_withheld_not_emitted() {
    let tok = fixture();
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder =
        StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string(), &[]);
    let head = decoder
        .consume_flushed_text("Sure. <tool_")
        .expect("the partial mark must not error");
    let tail = decoder
        .consume_flushed_text(
            "call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>",
        )
        .expect("the completed call parses");
    let text: String = head
        .iter()
        .chain(tail.iter())
        .filter_map(|e| match e {
            StructuredAssistantEvent::Content(part) => Some(part.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Sure. ");
    assert!(tail
        .iter()
        .any(|e| matches!(e, StructuredAssistantEvent::ToolCall(_))));
}

/// The generation prompt FORCES the think frame open (`<|assistant|><think>`,
/// read off the real template), so the model never emits the opening tag.
/// A decoder seeded from the prompt routes the scratchpad to Reasoning and
/// the answer to Content.
#[test]
fn the_forced_open_think_frame_splits_reasoning_from_content() {
    let tok = fixture();
    let prompt_text = "[gMASK]<sop><|user|>hello<|assistant|><think>";
    let prompt_ids = tok.encode(prompt_text, false);
    let stream = "scratch scratchpad</think>The answer.";
    let events = run(&tok, &deltas(&tok, stream), &prompt_ids);
    let mut reasoning = String::new();
    let mut content = String::new();
    for event in events {
        match event {
            StructuredAssistantEvent::Reasoning(part) => reasoning.push_str(&part),
            StructuredAssistantEvent::Content(part) => content.push_str(&part),
            StructuredAssistantEvent::ToolCall(_) => panic!("no call was made"),
        }
    }
    assert_eq!(reasoning, "scratch scratchpad");
    assert_eq!(content, "The answer.");
}

/// Two calls in one turn, with prose before, between, and after.
#[test]
fn two_calls_in_one_turn_each_parse() {
    let tok = fixture();
    let stream = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>\
                  and \
                  <tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>";
    let events = run(&tok, &deltas(&tok, stream), &[]);
    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            StructuredAssistantEvent::ToolCall(c) => Some(c.name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, vec!["get_weather", "get_weather"]);
}

/// A call naming a tool the caller did not offer is refused, the same
/// allowlist rule every native dialect runs. The turn errors rather than
/// minting a call the request never granted.
#[test]
fn an_unoffered_tool_is_refused() {
    let tok = fixture();
    let allowed: HashSet<String> = ["other_tool".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string(), &[]);
    let stream = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
    let mut errored = false;
    for (id, text) in deltas(&tok, stream) {
        if decoder.consume(id, &text).is_err() {
            errored = true;
            break;
        }
    }
    assert!(errored, "an unknown tool must refuse the block");
}

/// A truncation -- the turn ends inside an open block -- is malformed at
/// finish, the same verdict the id-bracket arms reach on an unterminated
/// span. An empty block and a name with no arguments at all are refused by
/// the parser, not silently emitted as argument-less calls.
#[test]
fn a_truncated_block_is_malformed_at_finish() {
    let tok = fixture();
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "call_1".to_string(), &[]);
    let mut last = Ok(Vec::new());
    for (id, text) in deltas(
        &tok,
        "answer<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value>",
    ) {
        last = decoder.consume(id, &text);
    }
    assert!(last.is_ok(), "the open block buffers without error");
    assert!(
        decoder.finish().is_err(),
        "an unterminated block is a malformed turn"
    );
}
