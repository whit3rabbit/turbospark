//! Tests for the chunked-prefill entry point: prompt tokens are split into
//! fixed-size chunks (`foundation::prefill_chunk_spans`) and handed to the
//! producer's `prefill_chunk` one chunk at a time, then decoding proceeds
//! exactly as the unchunked path does.

use std::path::PathBuf;

use foundation::LogitValue;
use mrefrust_runtime::{
    run_raw_completion_chunked, GenerationConfig, RawDecodeProgress, ScriptedLogitProducer,
    StopReason,
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

#[test]
fn chunked_prefill_reaches_the_same_stop_as_the_unchunked_loop() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let h_id = tokenizer.token_to_id("h").unwrap() as usize;
    let im_end_id = tokenizer.end_of_turn_id as usize;

    // A longer prompt so a chunk_tokens of 2 produces multiple chunks.
    let prompt_ids = tokenizer.encode("hi hi hi hi", false);
    assert!(
        prompt_ids.len() > 2,
        "need multiple chunks to exercise the split"
    );

    // One scripted step per chunk (prefill_chunk consumes one step per
    // call, regardless of chunk size) plus one for the decode step.
    let num_chunks = prompt_ids.len().div_ceil(2);
    let mut steps: Vec<Vec<LogitValue>> = (0..num_chunks - 1)
        .map(|_| one_hot(vocab_size, 0))
        .collect();
    steps.push(one_hot(vocab_size, h_id)); // last chunk seeds decode step 1
    steps.push(one_hot(vocab_size, im_end_id)); // decode step 2: stop

    let mut producer = ScriptedLogitProducer::new(steps);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 10,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    };

    let mut events = Vec::new();
    let result = run_raw_completion_chunked(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        2,
        |e| {
            events.push(e);
        },
    )
    .unwrap();

    assert_eq!(result.reason, StopReason::EndOfTurn);
    assert_eq!(result.prompt_tokens, prompt_ids.len());
    assert_eq!(result.new_tokens, 2);
    let prefill_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, RawDecodeProgress::Prefill { .. }))
        .collect();
    assert_eq!(prefill_events.len(), num_chunks);
}

#[test]
fn chunked_prefill_rejects_empty_prompt() {
    let tokenizer = load_tokenizer();
    let vocab_size = tokenizer.vocab_size;
    let mut producer = ScriptedLogitProducer::new(Vec::new());
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 5,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    };
    let err = run_raw_completion_chunked(
        &mut producer,
        &tokenizer,
        &[],
        &config,
        4096,
        vocab_size,
        128,
        |_| {},
    )
    .unwrap_err();
    assert_eq!(err, mrefrust_runtime::RuntimeError::EmptyPrompt);
}
