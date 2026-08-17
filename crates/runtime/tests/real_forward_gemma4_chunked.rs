#![cfg(target_os = "macos")]
//! Chunked prefill on the real-checkpoint Gemma 4 flow
//! (`docs/BATCHED_PREFILL.md` step 1): a whole chunk of prompt tokens runs
//! its attention-and-router half for every token into ONE command buffer
//! per layer, and its routed half per token.
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence.**
//! Two chunked runs agree with each other whenever both are wrong the same
//! way, which is why every case here compares against `produce_prefill` /
//! `produce` and never against another chunked arm
//! (`crates/bench/tests/accept_length_probe.rs` established the shape for
//! speculative decoding). The fixture's weights are deterministic but not
//! trained, so nothing here asserts what the logits MEAN -- only that
//! grouping tokens into a command buffer does not change them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::build_synthetic_gemma4_real_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gemma4-chunked-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: i64 = 128;
/// Long enough to cross a layer's routed bank several times (the driver
/// alternates two) and to make the last micro-batch a partial one.
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

/// Sixteen experts against four cache slots, so a chunk's tokens contend
/// for slots and the driver's "protect the in-flight token's slots" rule is
/// actually exercised rather than trivially satisfied.
fn build_install(dir: &std::path::Path, experts: i64) -> model_io::ArchConfig {
    build_synthetic_gemma4_real_install(dir, VOCAB, 2, experts, 4, 8, "tiny-gemma4-real")
        .expect("real-naming install builds")
}

/// The reference: every prompt token through `produce_prefill` but the
/// last, which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("sequential prefill succeeds");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// The same prompt through `prefill_chunk`, split into spans of `chunk`.
fn chunked_prefill(runner: &mut RealForwardRunner, tokens: &[i32], chunk: usize) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut offset = 0usize;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
            .expect("chunked prefill succeeds");
        offset += take;
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    // The fixture has to be able to SEE a difference: a prompt whose logits
    // never move cannot distinguish a working driver from a broken one.
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );

    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "chunked prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // The same question one level out, and the one that would catch a
    // driver whose per-token state leaked across a micro-batch: splitting
    // the SAME prompt at different points must land on one answer, and that
    // answer must be the sequential one. Spans of 1 also cover the
    // degenerate micro-batch of a single token.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 2, 3, 4, 7, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits; the routed reduce order or a \
             per-token buffer is a function of the chunk boundary"
        );
    }
}

#[test]
fn a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits() {
    // Below `2 * top_k` slots the driver cannot protect the in-flight
    // token's experts while planning the next one, so it retires first
    // instead. That fallback is a THROUGHPUT choice and must not be a
    // numerics one; four slots against eight routed experts forces it.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 8)
        .expect("real-naming install opens");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "the un-pipelined fallback must be the same function as the pipelined one"
    );
}

#[test]
fn decoding_continues_correctly_after_a_chunked_prefill() {
    // The chunk has to leave the engine where the sequential path leaves
    // it: KV cursor at `tokens.len()`, every row written. A driver that
    // advanced the cursor wrongly still produces finite logits for the
    // chunk itself and only diverges once decode reads back over them.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");

    let mut sequential = Vec::new();
    sequential_prefill(&mut runner, &PROMPT);
    for (step, &token) in [3i32, 7, 2].iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, PROMPT.len() + step, &mut logits)
            .expect("decode after sequential prefill succeeds");
        sequential.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    let mut chunked = Vec::new();
    chunked_prefill(&mut runner, &PROMPT, 4);
    for (step, &token) in [3i32, 7, 2].iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, PROMPT.len() + step, &mut logits)
            .expect("decode after chunked prefill succeeds");
        chunked.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    assert_eq!(
        chunked, sequential,
        "decode diverged after a chunked prefill: the KV cache or the position \
         cursor does not match what the sequential path leaves behind"
    );
}

#[test]
fn an_empty_chunk_is_refused() {
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&[], 0, &mut logits)
        .expect_err("an empty chunk has no logits to write");
    assert!(err.contains("empty chunk"), "unexpected message: {err}");
}

#[test]
fn a_chunk_starting_off_the_kv_cursor_is_refused() {
    // The driver addresses K/V rows by absolute position, so a start that
    // disagrees with the cursor would write into the wrong slots and read
    // back fluent nonsense rather than failing.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 4, &mut logits)
        .expect_err("a chunk must start where the KV cache is");
    assert!(err.contains("non-sequential"), "unexpected message: {err}");
}
