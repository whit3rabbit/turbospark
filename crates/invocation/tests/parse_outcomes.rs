//! Accepted-outcome scenarios for the argument-translation contract.
//!
//! Corresponds to behavior-spec test-001, test-002, test-004, test-010,
//! test-012, test-014, test-016, and test-017.

use foundation::runtime_config::ALLOWED_CHUNK_SIZES;
use invocation::{parse, InvocationRequest, Mode, ParseOutcome, PrefillChunk};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_success(outcome: ParseOutcome) -> InvocationRequest {
    match outcome {
        ParseOutcome::Success(req) => req,
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn minimal_prompt_invocation_uses_documented_defaults() {
    let req = expect_success(parse(&tok(&["--model", "m.bin", "--prompt", "hello"])));
    assert_eq!(req.model, "m.bin");
    assert_eq!(req.mode, Mode::Prompt("hello".to_string()));
    assert_eq!(req.max_new, 1024);
    assert_eq!(req.top_k, 64);
}

#[test]
fn explicit_generation_and_sampling_values_round_trip() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hello",
        "--max-new",
        "50",
        "--max-context",
        "2048",
        "--temperature",
        "0.7",
        "--top-k",
        "10",
        "--top-p",
        "1.0",
        "--repetition-penalty",
        "1.2",
        "--seed",
        "42",
        "--stop",
        "a",
        "--stop",
        "b",
        "--quiet",
    ])));
    assert_eq!(req.max_new, 50);
    assert_eq!(req.max_context, 2048);
    assert_eq!(req.temperature, 0.7);
    assert_eq!(req.top_k, 10);
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.repetition_penalty, 1.2);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.stop, vec!["a".to_string(), "b".to_string()]);
    assert!(req.quiet);
}

#[test]
fn top_k_disabled_reports_the_dedicated_disabled_state() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--prompt", "hi", "--top-k", "0",
    ])));
    assert_eq!(req.top_k, 0);
}

#[test]
fn messages_file_mode_leaves_prompt_unset() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--messages-file",
        "chat.json",
    ])));
    assert_eq!(req.mode, Mode::MessagesFile("chat.json".to_string()));
}

#[test]
fn chat_mode_selects_interactive_with_no_prompt_or_file() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(req.mode, Mode::Chat);
}

#[test]
fn repeated_system_messages_combine_in_supplied_order() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--chat", "--system", "first", "--system", "second",
    ])));
    let combined = req.system.expect("system message expected");
    assert!(combined.find("first").unwrap() < combined.find("second").unwrap());
}

#[test]
fn every_documented_chunk_size_is_accepted() {
    for &size in ALLOWED_CHUNK_SIZES.iter() {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--prefill-chunk",
            &size.to_string(),
        ])));
        assert_eq!(req.prefill_chunk, PrefillChunk::Fixed(size));
    }
}

#[test]
fn automatic_chunk_sizing_keyword_selects_auto() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--prefill-chunk",
        "auto",
    ])));
    assert_eq!(req.prefill_chunk, PrefillChunk::Auto);
}
