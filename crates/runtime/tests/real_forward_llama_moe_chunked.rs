#![cfg(target_os = "macos")]
//! Chunked prefill for the MoE half of the `llama` architecture (Mixtral,
//! Qwen3MoE): the fourth [`ChunkedPrefillRunner`] implementation, and the
//! second (after Gemma 4's) with a per-token routed half pipelined across a
//! per-layer command buffer
//! (`crates/runtime/src/families/llama/moe_prefill.rs`).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as every other chunked-prefill parity suite in this crate
//! establishes: two chunked runs can agree with each other while both are
//! wrong the same way, so every case here compares against
//! `produce_prefill` / `produce` and never against another chunked arm.

use half::f16;
use turbospark_repack::build_synthetic_llama_real_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
/// 8 experts, top-2 (`build_synthetic_gqa_moe_install`'s
/// `top_k_experts: num_experts.min(2)`), so `2 * top_k = 4` is the
/// pipelining threshold the cache-too-small case below sits under.
const NUM_EXPERTS: i64 = 8;
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-moe-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_install(dir: &std::path::Path) -> model_io::ArchConfig {
    build_synthetic_llama_real_install(dir, VOCAB, LAYERS, NUM_EXPERTS, "tiny-mixtral")
        .expect("MoE llama install builds")
}

fn open_runner(tag: &str, slots: usize) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_install(&dir);
    RealForwardRunner::open_with_options(&dir, arch, 4096, slots)
        .expect("an MoE llama install opens")
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
fn an_moe_llama_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports", 16);
    assert!(
        runner.supports_chunked_prefill(),
        "an MoE llama install must be servable by the chunked driver"
    );
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let mut runner = open_runner("whole-chunk", 16);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
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
    // Splitting the SAME prompt at different points must land on one
    // answer, and that answer must be the sequential one: the case that
    // would catch a driver whose per-token row, router-logits row, or
    // routed-slot bank leaked across a micro-batch or a layer.
    let mut runner = open_runner("boundary", 16);

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
fn a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits() {
    // Below `2 * top_k` slots the driver cannot protect the in-flight
    // token's experts while planning the next one, so it retires before it
    // encodes instead of pipelining. That fallback is a THROUGHPUT choice
    // and must not be a numerics one. A SEPARATE, smaller install (2
    // experts, so top-2 selects both every token and nothing ever misses
    // after the first load) rather than a small slot count on the main
    // fixture: `ExpertCache::plan` asserts rather than degrades when a
    // token's own misses cannot be placed even after retiring, and this
    // fixture's routing does not overlap enough across tokens at 8 experts
    // to guarantee that at any slot count under the pipelining threshold.
    let dir = temp_dir("cache-too-small");
    let arch = build_synthetic_llama_real_install(&dir, VOCAB, LAYERS, 2, "tiny-mixtral-2x")
        .expect("MoE llama install with 2 experts builds");
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 2)
        .expect("an MoE llama install opens");

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
    let mut sequential_runner = open_runner("decode-continues-sequential", 16);
    let mut sequential = Vec::new();
    for (position, &token) in PROMPT.iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        sequential_runner
            .produce(token, position, &mut logits)
            .expect("sequential decode succeeds");
        sequential.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }
    let extra_tokens = [3i32, 6, 9];
    let mut sequential_extra = Vec::new();
    for (i, &token) in extra_tokens.iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        sequential_runner
            .produce(token, PROMPT.len() + i, &mut logits)
            .expect("sequential decode succeeds");
        sequential_extra.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    let mut chunked_runner = open_runner("decode-continues-chunked", 16);
    chunked_runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked_runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect("chunked prefill succeeds");
    assert_eq!(
        logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
        *sequential.last().unwrap(),
        "the chunk's own logits must match the sequential path's last token"
    );
    for (i, &token) in extra_tokens.iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        chunked_runner
            .produce(token, PROMPT.len() + i, &mut logits)
            .expect("decode after a chunked prefill succeeds");
        assert_eq!(
            logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            sequential_extra[i],
            "decode token {i} after a chunked prefill diverged from the sequential KV state"
        );
    }
}

/// `MFERENCE_ROUTED_BATCH` must be REFUSED BY NAME on this family, never
/// ignored. The batched routed pair exists for INT4-affine and MXFP4 blobs;
/// this family's are GGUF K-quants and step 5's Q4_K/Q6_K arm was scoped by
/// measurement and deliberately not built (`docs/BATCHED_PREFILL.md`).
///
/// Silently running the per-token loop instead is the failure this repo
/// names repeatedly: a caller who asked for the batched half would measure
/// the unbatched engine and report it under the batched arm's label. It was
/// the real behaviour until 2026-08-27 and was found by checking a claim in
/// the docs against the binary rather than by a test.
///
/// A DENSE driver ignoring the same flag is correct and deliberately not
/// asserted here: there is no routed half for it to refer to.
#[test]
fn the_batched_routed_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("routed-batch-refused", 16);
    runner.set_routed_batch_prefill(true);

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("the batched routed seam must be refused on this family");
    let text = err.to_string();
    assert!(
        text.contains("MFERENCE_ROUTED_BATCH"),
        "the refusal must name the seam the caller set; got {text}"
    );
}

/// The resident-GEMV seam (`MFERENCE_BATCHED_GEMV`) must be refused by name
/// too, and unlike the routed seam above it is meaningful on EVERY chunked
/// driver: this family has resident GEMVs whatever its routed layout, and
/// the M-row GEMM is wired in Gemma 4's driver alone.
#[test]
fn the_batched_gemv_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("batched-gemv-refused", 16);
    runner.set_batched_gemv_prefill(true);

    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused on this family");
    let text = err.to_string();
    assert!(
        text.contains("MFERENCE_BATCHED_GEMV"),
        "the refusal must name the seam the caller set; got {text}"
    );
}
