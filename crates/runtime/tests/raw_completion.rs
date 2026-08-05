//! End-to-end raw-completion loop tests: real ChatML tokenizer fixture,
//! scripted logits standing in for a model forward pass, exercising
//! prefill, greedy sampling, detokenization, and stop-token handling.
//!
//! Token ids are resolved from the loaded tokenizer rather than hardcoded:
//! the fixture's `tokenizer.json` embeds high placeholder ids (e.g. 248044)
//! for its `added_tokens`, but the loader renumbers them sequentially after
//! the 258-entry base vocab, so the *actual* ids only exist at load time.

use std::path::PathBuf;

use foundation::LogitValue;
use mrefrust_runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, ScriptedLogitProducer, StopReason,
};
use selection::ShapingConfig;
use tokenizer::MfTokenizer;

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<LogitValue> {
    let mut v = vec![LogitValue::from_f32(0.0); vocab_size];
    v[index] = LogitValue::from_f32(1.0);
    v
}

fn greedy_config(max_new_tokens: u32) -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    }
}

#[test]
fn stops_at_end_of_turn_and_emits_visible_content() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let im_end_id = tokenizer.end_of_turn_id as usize;

    let prompt_ids = tokenizer.encode("hi", false);
    assert!(!prompt_ids.is_empty());

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    steps.push(one_hot(vocab_size, h_id)); // last prefill step seeds decode step 1
    steps.push(one_hot(vocab_size, im_end_id)); // decode step 2: stop

    let mut producer = ScriptedLogitProducer::new(steps);
    let config = greedy_config(10);
    let mut events = Vec::new();

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |e| {
            events.push(e);
        },
    )
    .unwrap();

    assert_eq!(result.reason, StopReason::EndOfTurn);
    assert_eq!(result.new_tokens, 2);
    let h_id = h_id as i32;
    let im_end_id = im_end_id as i32;
    assert!(events
        .iter()
        .any(|e| matches!(e, RawDecodeProgress::Token { id, .. } if *id == h_id)));
    assert!(!events
        .iter()
        .any(|e| matches!(e, RawDecodeProgress::Token { id, .. } if *id == im_end_id)));
}

#[test]
fn stops_at_max_new_tokens_when_no_stop_token_arrives() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let prompt_ids = tokenizer.encode("hi", false);

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    // Every step (prefill seed + every decode step) keeps producing 'h'.
    for _ in 0..5 {
        steps.push(one_hot(vocab_size, h_id));
    }

    let mut producer = ScriptedLogitProducer::new(steps);
    let config = greedy_config(3);

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.reason, StopReason::MaxTokens);
    assert_eq!(result.new_tokens, 3);
}

#[test]
fn rejects_empty_prompt() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let mut producer = ScriptedLogitProducer::new(Vec::new());
    let config = greedy_config(5);
    let err = run_raw_completion(
        &mut producer,
        &tokenizer,
        &[],
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap_err();
    assert_eq!(err, mrefrust_runtime::RuntimeError::EmptyPrompt);
}

#[test]
fn rejects_context_overflow() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let prompt_ids = tokenizer.encode("hi", false);
    let mut producer = ScriptedLogitProducer::new(Vec::new());
    let config = greedy_config(1000);
    let err = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4,
        vocab_size,
        |_| {},
    )
    .unwrap_err();
    assert!(matches!(
        err,
        mrefrust_runtime::RuntimeError::ContextOverflow { .. }
    ));
}

#[test]
fn stop_string_truncates_visible_output() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer
        .token_to_id("h")
        .expect("'h' is in the base vocab") as usize;
    let prompt_ids = tokenizer.encode("hi", false);

    let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
    for _ in 0..5 {
        steps.push(one_hot(vocab_size, h_id));
    }
    let mut producer = ScriptedLogitProducer::new(steps);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 5,
        stop_strings: vec!["hh".to_string()],
        extra_stop_tokens: Vec::new(),
    };

    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.reason, StopReason::StopString);
}
