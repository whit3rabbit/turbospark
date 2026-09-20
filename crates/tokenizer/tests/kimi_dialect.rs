//! The Kimi K2 dialect: detection (including the ChatML ordering trap),
//! resolution, and the text-marker decoder arm.
//!
//! **THE FIXTURE'S ADDED-TOKEN TABLE IS REAL AND ITS BPE BODY IS NOT.** The
//! entire Kimi K2 line ships `tiktoken.model` and NO `tokenizer.json` at all
//! (checked across `Kimi-K2-Instruct`, `-0905`, `-Thinking`, `K2.5`, `K2.6`,
//! `K2.7-Code`, the mlx-community and ISTA-DASLab builds on 2026-09-19), so
//! there is no real HF table to reduce. `fixtures/KimiK2Tokenizer` carries
//! `moonshotai/Kimi-K2.5`'s REAL added-token list -- all 23 tokens with
//! their real ids and special flags, which is the only part of the table
//! the dialect machinery reads -- over a synthetic 256-entry ByteLevel body
//! that exists to make the fixture loadable and its test strings encodable.
//! The id assertions below are BY NAME, NOT BY NUMBER: the `tokenizers`
//! library renumbers a sparse fixture's ids on load (crate Gotcha 2), and
//! the assertions pin the RESOLUTION -- the resolver's ids are the named
//! markers' ids -- which is how `resolve_kimi` itself reads the table. The
//! real numbers ([BOS] 163584, [EOS] 163585, <|im_end|> 163586,
//! <|im_assistant|> 163588, <|im_middle|> 163601, think 163606/163607) live
//! in the fixture's added_tokens and in the resolver's doc comments; a real
//! dense K2 table would keep them.
//!
//! No K2 checkpoint is installable by this engine today for the same reason
//! (no `tokenizer.json` to source a sidecar from), which the roadmap entry
//! records.

use std::collections::HashSet;
use std::path::PathBuf;

use turbospark_tokenizer::{
    ChatDialect, MfTokenizer, StructuredAssistantDecoder, StructuredAssistantEvent,
    ToolCallSupport,
};

fn fixture() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/KimiK2Tokenizer");
    MfTokenizer::load_from_dir(&dir).expect("the Kimi K2 fixture tokenizer should load")
}

#[test]
fn the_real_added_tokens_resolve_to_the_kimi_dialect_with_the_named_markers() {
    let tok = fixture();
    assert_eq!(tok.dialect, ChatDialect::Kimi);
    let id = |name: &str| {
        tok.token_to_id(name)
            .unwrap_or_else(|| panic!("{name} must resolve in the loaded table"))
    };
    assert_eq!(tok.bos_id, id("[BOS]"));
    assert_eq!(tok.eos_id, id("[EOS]"));
    assert_eq!(tok.end_of_turn_id, id("<|im_end|>"));
    assert_eq!(tok.think_start_id, Some(id("<think>")));
    assert_eq!(tok.think_end_id, Some(id("</think>")));
    assert_eq!(tok.channel_start_id, id("<think>"));
    assert_eq!(tok.channel_end_id, id("</think>"));
    assert_eq!(tok.dialect.tool_call_support(), ToolCallSupport::Native);
    // Both EOS-class tokens stop; the template closes turns with the second.
    assert!(tok.stop_token_ids.contains(&id("[EOS]")));
    assert!(tok.stop_token_ids.contains(&id("<|im_end|>")));
}

/// **THE ORDERING TRAP.** A Kimi table carries `<|im_end|>` -- the ChatML
/// probe's whole witness -- so the Kimi arm MUST be tested before the ChatML
/// arm. A checkpoint whose dialect resolves to ChatMl here would then die in
/// `resolve_chatml` on the missing `<|im_start|>`; this test is what keeps
/// the two arms from swapping.
#[test]
fn a_kimi_table_with_im_end_is_not_probed_as_chatml() {
    let tok = fixture();
    assert_ne!(tok.dialect, ChatDialect::ChatMl);
    assert_eq!(tok.dialect, ChatDialect::Kimi);
}

/// One frame of the stream a decoder really sees: an id and the text the
/// streaming detokenizer released for it.
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
        StructuredAssistantDecoder::new(tok, allowed, || "generated_id".to_string(), prompt_ids);
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

/// The template's own shape, rendered end to end: a section wrapper, one
/// call whose id is `functions.NAME:IDX`, and the arguments as one JSON
/// object. The parser keeps the checkpoint's id verbatim -- the model
/// matches a tool result against `## Return of {id}` -- and cuts the NAME
/// out of it.
#[test]
fn a_sectioned_call_parses_with_the_checkpoints_own_id() {
    let tok = fixture();
    let stream = "<|tool_calls_section_begin|>\
                  <|tool_call_begin|>functions.get_weather:0\
                  <|tool_call_argument_begin|>{\"city\": \"Oslo\"}\
                  <|tool_call_end|>\
                  <|tool_calls_section_end|>";
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
    assert_eq!(calls.len(), 1, "markup leaked as content: {text:?}");
    assert_eq!(calls[0].id, "functions.get_weather:0");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments_json, r#"{"city":"Oslo"}"#);
    assert!(text.is_empty(), "the section wrapper must be swallowed: {text:?}");
}

/// The wrapper is framing, not structure: a call WITHOUT its section (and a
/// section around prose without calls) still parses / still passes prose
/// through, because calls are keyed on their OWN begin/end pairs.
#[test]
fn a_call_without_the_section_wrapper_still_parses() {
    let tok = fixture();
    let stream = "Let me check. <|tool_call_begin|>functions.get_weather:0\
                  <|tool_call_argument_begin|>{\"city\":\"Oslo\"}<|tool_call_end|> done";
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
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(text, "Let me check.  done");
}

/// The generation prompt FORCES the think frame open
/// (`<|im_assistant|>assistant<|im_middle|><think>`), so the model never
/// emits the opening tag. A decoder seeded from the prompt routes the
/// scratchpad to Reasoning and everything after `</think>` to Content -- and
/// the think tags themselves are added-but-not-special, so they arrive as
/// text deltas and must still flip the channel.
#[test]
fn the_forced_open_think_frame_splits_reasoning_from_content() {
    let tok = fixture();
    let prompt_text = "<|im_user|>user<|im_middle|>hello<|im_end|>\
                       <|im_assistant|>assistant<|im_middle|><think>";
    let prompt_ids = tok.encode(prompt_text, false);
    let stream = "scratchpad</think>The answer.";
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
    assert_eq!(reasoning, "scratchpad");
    assert_eq!(content, "The answer.");
}

/// Two calls in one section, and the ids are what distinguish them: the
/// names are equal, the indexes are not, and both survive.
#[test]
fn two_calls_in_one_section_each_parse_with_their_own_ids() {
    let tok = fixture();
    let stream = "<|tool_calls_section_begin|>\
                  <|tool_call_begin|>functions.get_weather:0\
                  <|tool_call_argument_begin|>{\"city\":\"Oslo\"}<|tool_call_end|>\
                  <|tool_call_begin|>functions.get_weather:1\
                  <|tool_call_argument_begin|>{\"city\":\"Rome\"}<|tool_call_end|>\
                  <|tool_calls_section_end|>";
    let events = run(&tok, &deltas(&tok, stream), &[]);
    let ids: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            StructuredAssistantEvent::ToolCall(c) => Some(c.id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["functions.get_weather:0", "functions.get_weather:1"]);
}

/// An unoffered tool refuses the call -- the allowlist rule every native
/// dialect runs -- and an id with no name in it is malformed even when the
/// tool IS offered.
#[test]
fn an_unknown_tool_and_a_nameless_id_both_refuse() {
    let tok = fixture();
    let stream = "<|tool_call_begin|>functions.get_weather:0\
                  <|tool_call_argument_begin|>{\"city\":\"Oslo\"}<|tool_call_end|>";
    let allowed: HashSet<String> = ["other_tool".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "x".to_string(), &[]);
    let mut errored = false;
    for (id, text) in deltas(&tok, stream) {
        if decoder.consume(id, &text).is_err() {
            errored = true;
            break;
        }
    }
    assert!(errored, "an unoffered tool must refuse the call");

    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "x".to_string(), &[]);
    let stream = "<|tool_call_begin|>functions.:0\
                  <|tool_call_argument_begin|>{\"city\":\"Oslo\"}<|tool_call_end|>";
    let mut errored = false;
    for (id, text) in deltas(&tok, stream) {
        if decoder.consume(id, &text).is_err() {
            errored = true;
            break;
        }
    }
    assert!(errored, "an id with no name part must refuse the call");
}

/// A stray call-close with no call open is malformed, the same verdict the
/// id-bracket arms reach on a stray end token.
#[test]
fn a_stray_call_close_is_malformed() {
    let tok = fixture();
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "x".to_string(), &[]);
    let mut errored = false;
    for (id, text) in deltas(&tok, "<|tool_call_end|>nothing was open") {
        if decoder.consume(id, &text).is_err() {
            errored = true;
            break;
        }
    }
    assert!(errored, "a stray close must be malformed");
}

/// A truncated call -- the turn ends inside an open call -- is malformed at
/// finish, the same verdict the DSML and id-bracket arms reach on an
/// unterminated span.
#[test]
fn a_truncated_call_is_malformed_at_finish() {
    let tok = fixture();
    let allowed: HashSet<String> = ["get_weather".to_string()].into_iter().collect();
    let mut decoder = StructuredAssistantDecoder::new(&tok, allowed, || "x".to_string(), &[]);
    let mut last = Ok(Vec::new());
    for (id, text) in deltas(
        &tok,
        "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\":",
    ) {
        last = decoder.consume(id, &text);
    }
    assert!(last.is_ok(), "the open call buffers without error");
    assert!(decoder.finish().is_err(), "an unterminated call is malformed");
}
