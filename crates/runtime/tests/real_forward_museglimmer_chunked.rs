#![cfg(target_os = "macos")]
//! Chunked prefill for `muse_glimmer`: the third [`ChunkedPrefillRunner`]
//! implementation, structurally the same shape as the dense half of `llama`
//! (`crates/runtime/src/families/museglimmer/prefill.rs`'s header explains
//! why: no router at all, so a whole micro-batch fits in ONE command
//! buffer, exactly as the dense llama driver).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as `real_forward_gemma4_chunked.rs` and
//! `real_forward_llama_dense_chunked.rs` establish: two chunked runs can
//! agree with each other while both are wrong the same way, so every case
//! here compares against `produce_prefill` / `produce` and never against
//! another chunked arm.

use half::f16;
use turbospark_repack::build_synthetic_muse_glimmer_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 64;
/// A multiple of 4 so the `[0, 0, 0, 1]` window pattern is whole: layers 0-2
/// slide and layer 3 is FULL and NoPE, over two periods, matching
/// `real_forward_muse.rs`'s own fixture sizing.
const LAYERS: i64 = 8;
/// Longer than the sliding window (8, see `synthetic_muse.rs`) so the ring
/// is exercised, and long enough to cross several chunk-span boundaries.
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-muse-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_synthetic_muse_glimmer_install(&dir, VOCAB, LAYERS, "tiny-muse")
        .expect("muse_glimmer install builds");
    RealForwardRunner::open(&dir, arch).expect("a muse_glimmer install opens")
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
fn a_muse_glimmer_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports");
    assert!(
        runner.supports_chunked_prefill(),
        "a muse_glimmer install must be servable by the chunked driver"
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
    // driver whose per-token row leaked across a micro-batch, a layer, or
    // the sliding-window ring: splitting the SAME prompt at different
    // points must land on one answer, and that answer must be the
    // sequential one. Spans of 1 also cover the degenerate micro-batch of a
    // single token.
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
