#![cfg(target_os = "macos")]
//! Does `RealForwardRunner::rollback` return the engine to the state a
//! fresh run would have reached? (ROADMAP's speculative-decoding item,
//! Phase D1.)
//!
//! Speculative decoding runs tokens the target may reject, so every
//! rejection has to be undone EXACTLY. Nothing else in the suite exercises
//! that, and the two halves fail differently:
//!
//!   - the KV cache is addressed by absolute position, so rolling it back
//!     is a cursor move -- except on a sliding-window ring, where the
//!     rejected tokens have already overwritten the oldest rows still
//!     inside the window;
//!   - a gated-DeltaNet layer folds its history into a fixed-size
//!     accumulator through a non-invertible update, so a cursor cannot
//!     express it at all and it needs a copy taken beforehand.
//!
//! Both failures are silent. A stale recurrent state or a clobbered KV row
//! produces fluent, wrong text, which is exactly what a coherence smoke
//! cannot see and what this asserts against instead: BIT-IDENTICAL logits.
//!
//! The ground truth is a `reset()` and full replay, not a repeat of the
//! same continuation. Replaying the same tokens twice would pass while
//! leaking state that happens to be idempotent; comparing against a run
//! that never saw the rejected tokens cannot.
//!
//! ```sh
//! TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
//!   cargo test -p turbospark-bench --test rollback_probe --release -- --ignored --nocapture
//! ```
//! Run it on BOTH families: Qwen 3.6 exercises the recurrent half (30 of
//! its 40 layers are linear) and Gemma 4 the sliding-window ring (25 of
//! 30). Neither covers the other.

use foundation::LogitValue;
use runtime::{LogitProducer, RealForwardRunner};
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::real_model::open_model_runner;

/// Rejected-block length. Speculative blocks are 4 to 16 tokens; 12 is
/// inside the ring slack of any install this port builds.
const BLOCK: usize = 12;

fn ids_for(tokenizer: &MfTokenizer, text: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, text)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

/// Feeds `prompt` and then `block`, returning the logits after the LAST
/// token of `block`. Every token goes through `produce` rather than
/// `produce_prefill` so each one advances the same state a decode step
/// would, which is what a verify pass does.
fn walk(runner: &mut RealForwardRunner, prompt: &[i32], block: &[i32], vocab: usize) -> Vec<u16> {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in prompt.iter().chain(block.iter()).enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
    }
    logits.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_PROBE_INSTALL_DIR"]
fn rollback_restores_the_state_a_fresh_run_would_have() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let slots: usize = std::env::var("TURBOSPARK_PROBE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (mut runner, tokenizer) = open_model_runner(&dir, slots).expect("install opens");
    let vocab = tokenizer.vocab_size;

    let prompt = ids_for(
        &tokenizer,
        "Explain how coastal wetlands reduce flood damage.",
    );
    // Two continuations that share no prefix, so a leaked state cannot
    // coincidentally agree.
    let rejected: Vec<i32> = ids_for(&tokenizer, "Discuss the history of bridge engineering.")
        .into_iter()
        .take(BLOCK)
        .collect();
    let kept: Vec<i32> = ids_for(&tokenizer, "Summarize photosynthesis for a child.")
        .into_iter()
        .take(BLOCK)
        .collect();
    assert_eq!(rejected.len(), BLOCK);
    assert_eq!(kept.len(), BLOCK);
    println!(
        "rollback_probe: {} prompt tokens, block {BLOCK}, max_rollback {}",
        prompt.len(),
        runner.max_rollback()
    );
    assert!(
        runner.max_rollback() >= BLOCK,
        "install cannot roll back a block this size"
    );

    // 1. Prompt, then checkpoint, then a block that will be rejected.
    runner.reset();
    let mut scratch = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in prompt.iter().enumerate() {
        runner.produce(token, i, &mut scratch).expect("produce");
    }
    let point = runner.checkpoint();
    assert_eq!(point.position(), prompt.len());
    let mut rejected_logits = Vec::new();
    for (i, &token) in rejected.iter().enumerate() {
        runner
            .produce(token, prompt.len() + i, &mut scratch)
            .expect("produce");
        rejected_logits = scratch.iter().map(|v| v.to_bits()).collect();
    }

    // 2. Undo it and run the continuation that was actually accepted.
    runner.rollback(&point);
    let mut after_rollback = Vec::new();
    for (i, &token) in kept.iter().enumerate() {
        runner
            .produce(token, prompt.len() + i, &mut scratch)
            .expect("produce");
        after_rollback = scratch.iter().map(|v| v.to_bits()).collect();
    }

    // 3. Ground truth: a run that never saw the rejected block at all.
    runner.reset();
    let fresh = walk(&mut runner, &prompt, &kept, vocab);

    let diff = after_rollback
        .iter()
        .zip(fresh.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!("rollback_probe: {diff} of {vocab} logits differ from the fresh run");
    assert_eq!(
        diff, 0,
        "rollback left state behind: {diff} logits differ from a fresh run"
    );

    // The test has to be able to fail. If the rejected block had left no
    // trace to begin with, step 2 would prove nothing.
    assert_ne!(
        rejected_logits, fresh,
        "the two continuations produced identical logits, so this test could not \
         have detected a bad rollback"
    );
}
