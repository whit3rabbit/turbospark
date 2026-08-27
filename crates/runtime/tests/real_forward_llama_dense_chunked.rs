#![cfg(target_os = "macos")]
//! Chunked prefill on the DENSE half of the `llama` architecture (Mistral,
//! Llama 2/3.x): the second [`ChunkedPrefillRunner`] implementation after
//! Gemma 4's, and the first with no host round trip inside the layer loop
//! at all (`crates/runtime/src/families/llama/prefill.rs`'s header explains
//! why a whole micro-batch fits in ONE command buffer here, where Gemma 4
//! needs one per layer).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as `real_forward_gemma4_chunked.rs` establishes: two chunked runs
//! can agree with each other while both are wrong the same way, so every
//! case here compares against `produce_prefill` / `produce` and never
//! against another chunked arm. The fixture's weights are deterministic but
//! not trained, so nothing here asserts what the logits MEAN, only that
//! grouping tokens into one command buffer does not change them.

use half::f16;
use turbospark_repack::build_synthetic_dense_llama_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
/// Longer than [`crate::real_forward_types::MAX_PREFILL_BATCH`]'s sibling
/// constant in the Gemma 4 test would need to be interesting, but this
/// architecture has no expert-slot contention to exercise, so eleven tokens
/// (same length as the Gemma 4 fixture) is enough to cross a micro-batch
/// boundary at several chunk spans and to leave the last micro-batch
/// partial.
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-dense-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense manifest peeks against the Mixtral baseline");
    RealForwardRunner::open(&dir, peeked).expect("a dense llama install opens")
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
fn a_dense_llama_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports");
    assert!(
        runner.supports_chunked_prefill(),
        "a dense llama install must be servable by the chunked driver"
    );
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let mut runner = open_runner("whole-chunk");

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
    // driver whose per-token row leaked across a micro-batch or across
    // layers: splitting the SAME prompt at different points must land on
    // one answer, and that answer must be the sequential one. Spans of 1
    // also cover the degenerate micro-batch of a single token.
    let mut runner = open_runner("boundary");

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 2, 3, 4, 7, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits; a per-token scratch row is a \
             function of the chunk boundary"
        );
    }
}

#[test]
fn a_dense_install_stays_on_the_dense_driver_once_the_moe_half_also_has_one() {
    // The dense driver must not become confused with, or fall back to, the
    // MoE driver once `families/llama/moe_prefill.rs` lands next to it:
    // both halves are supported now (`real_forward_llama_moe_chunked.rs`
    // covers the MoE half's own byte-identity in full), and this is the
    // narrow cross-check that a DENSE install still resolves to the DENSE
    // driver's path rather than tripping the MoE branch's `dense: false`
    // guard.
    let mut runner = open_runner("moe-now-supported-too");
    assert!(
        runner.supports_chunked_prefill(),
        "a dense llama install must still report chunked-prefill support"
    );
    let expected = sequential_prefill(&mut runner, &PROMPT);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "the dense driver must still be reached and still be correct"
    );
}
