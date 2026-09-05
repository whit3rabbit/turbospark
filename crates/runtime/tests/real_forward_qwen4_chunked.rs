#![cfg(target_os = "macos")]
//! Chunked prefill for `qwen4_exp`: the seventh
//! [`ChunkedPrefillRunner`] implementation, and the same "step 1" shape as
//! `real_forward_gemma4_chunked.rs` -- loop the EXISTING per-token kernels
//! inside a micro-batch, batching command buffers rather than GEMVs
//! (`crates/runtime/src/families/qwen4/prefill.rs`'s header has the full
//! design argument, including why the QSA indexer's shared position buffer
//! and the GDN/PLE recurrent state are unaffected by chunking).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as every other chunked test in this crate establishes: every case
//! here compares against `produce_prefill` / `produce`, never against
//! another chunked arm. That reference is the same sequential flow
//! `real_forward_qwen4.rs` already pins with its own frozen digest, so what
//! THIS file proves is narrower and deliberately so: that grouping tokens
//! into fewer command buffers does not change the bytes, not that the
//! underlying math is right (that question belongs to `real_forward_qwen4.rs`
//! and `qwen4exp_quality_gate`). The fixture's weights are deterministic but
//! not trained.

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

/// Must exceed the fixture's own n-gram EOS fill value (999), matching
/// `real_forward_qwen4.rs`'s identical constant and reasoning.
const VOCAB: i64 = 2048;

/// A context window comfortably over every span this file decodes.
const TEST_MAX_CONTEXT: usize = 512;

/// Matches `real_forward_qwen4.rs`'s `TINY_INDEXER_BUDGET`: 16 tokens is 4
/// blocks of `IDX_COMPRESS = 4`, so `index_top_k` is 4 and block selection
/// starts dropping blocks at the fifth complete one (`visible >= 20`,
/// position 19 onward).
const TINY_INDEXER_BUDGET: i64 = 16;
/// Enough positions past the budget crossing (position 19) to exercise every
/// `visible % 4` tail phase and to cross several chunk-span boundaries below
/// it, matching `real_forward_qwen4.rs`'s `SPARSE_STEPS`.
const QSA_PROMPT_LEN: usize = 28;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen4-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch =
        turbospark_repack::build_synthetic_qwen4_exp_decode_install(&dir, VOCAB, "qwen4-chunked")
            .expect("write install");
    let runner = RealForwardRunner::open_with_max_context(&dir, arch, TEST_MAX_CONTEXT)
        .expect("a qwen4_exp install opens");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

/// The same fixture, opened with a specific expert-cache slot count --
/// `expert_cache_slots == top_k_experts` puts the routed loop's
/// `routed_pipeline_banks` into its `banks == 1` fallback
/// (AGENTS.md Gotcha 64).
fn open_runner_with_slots(tag: &str, slots: usize) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch =
        turbospark_repack::build_synthetic_qwen4_exp_decode_install(&dir, VOCAB, "qwen4-chunked")
            .expect("write install");
    let runner = RealForwardRunner::open_with_options(&dir, arch, TEST_MAX_CONTEXT, slots)
        .expect("a qwen4_exp install opens at the given expert-cache slot count");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

fn open_tiny_budget_runner(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = turbospark_repack::build_synthetic_qwen4_exp_decode_install_with_indexer_budget(
        &dir,
        VOCAB,
        "qwen4-chunked-tiny-budget",
        TINY_INDEXER_BUDGET,
    )
    .expect("write install");
    let runner = RealForwardRunner::open_with_max_context(&dir, arch, TEST_MAX_CONTEXT)
        .expect("a qwen4_exp install opens above its tiny indexer budget");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

/// A deterministic, non-degenerate token sequence, matching
/// `real_forward_qwen4.rs`'s `decode_fixed_sequence` formula so both files
/// exercise the same n-gram/PLE history shape.
fn make_prompt(len: usize, vocab: i64) -> Vec<i32> {
    (0..len)
        .map(|p| ((p * 37 + 11) % vocab as usize) as i32)
        .collect()
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32], vocab: usize) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); vocab];
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
fn chunked_prefill(
    runner: &mut RealForwardRunner,
    tokens: &[i32],
    chunk: usize,
    vocab: usize,
) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); vocab];
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
fn a_qwen4_exp_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports");
    assert!(
        runner.supports_chunked_prefill(),
        "a qwen4_exp install must be servable by the chunked driver"
    );
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let mut runner = open_runner("whole-chunk");
    let prompt = make_prompt(11, VOCAB);
    let vocab = VOCAB as usize;

    let expected = sequential_prefill(&mut runner, &prompt, vocab);
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

    let actual = chunked_prefill(&mut runner, &prompt, prompt.len(), vocab);
    assert_eq!(
        actual, expected,
        "chunked prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // The same question one level out, and the one that would catch a
    // driver whose per-token row leaked across a micro-batch, across
    // layers, or across the GDN/PLE recurrent state's own crossing of a
    // `prefill_chunk` boundary. Spans of 2, 3, 4, 7 and 11 put MULTIPLE
    // tokens into one micro-batch -- exactly the case that caught two real
    // bugs during this driver's own development, both fixed by widening a
    // buffer to hold one row per token: `moe_x`/`hc_inject` (the `mlp_hc`
    // output and inject gate, which bridge the `cb1`/`"routed cb"` split)
    // and, the sharper one, `ple.rs`'s `ngram_emb` -- a HOST write
    // (`gpu::write_buffer_bytes`) rather than a GPU dispatch, so it does not
    // respect command-buffer commit order at all: every token's host write
    // in a micro-batch landed before any of their `key_proj`/`value_proj`
    // GEMVs executed, leaving every token but the last computing PLE from
    // the WRONG token's n-gram embedding. This test caught it at chunk span
    // 2 the first time it ran.
    let mut runner = open_runner("boundary");
    let prompt = make_prompt(11, VOCAB);
    let vocab = VOCAB as usize;

    let expected = sequential_prefill(&mut runner, &prompt, vocab);
    for chunk in [1usize, 2, 3, 4, 7, 11] {
        let actual = chunked_prefill(&mut runner, &prompt, chunk, vocab);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits; a per-token scratch row or the GDN/PLE \
             state's ordering is a function of the chunk boundary"
        );
    }
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits_across_the_qsa_sparsity_boundary() {
    // Above `TINY_INDEXER_BUDGET`'s `index_top_k` complete blocks, the QSA
    // layer scores pooled blocks and commits+waits mid-layer for the
    // readback -- entirely inside `encode_full_attention_block`'s existing
    // `&mut pass` handling, unmodified by this driver. This proves that
    // mid-layer commit composes correctly with the chunk driver's own
    // per-layer commit, at spans that both cross and land exactly on the
    // budget boundary (position 19).
    let mut runner = open_tiny_budget_runner("qsa-boundary");
    let prompt = make_prompt(QSA_PROMPT_LEN, VOCAB);
    let vocab = VOCAB as usize;

    let expected = sequential_prefill(&mut runner, &prompt, vocab);
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );
    for chunk in [1usize, 3, 5, 16, 19, QSA_PROMPT_LEN] {
        let actual = chunked_prefill(&mut runner, &prompt, chunk, vocab);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits crossing the QSA sparsity boundary"
        );
    }
}

/// `expert_cache_slots == top_k_experts` exactly panics with "expert cache
/// cannot place requested misses" -- a KNOWN, pre-existing, cross-family
/// hazard (AGENTS.md Gotcha 64), not a qwen4-specific bug:
/// every per-token routed prefill loop carries the previous token's `used`
/// slots into the next token's `protect` set UNCONDITIONALLY, regardless of
/// `banks`, so at exactly `top_k` slots a stale reservation of one whole
/// token's worth leaves zero room for the next token's misses. Gemma 4's own
/// `real_forward_gemma4_chunked.rs` test of the identical name sidesteps it
/// the same way: `2 * top_k` is the SMALLEST slot count `routed_pipeline_banks`
/// itself guarantees is safe (its own `>= 2 * top_k` threshold), so that is
/// what this test uses too, matching established precedent rather than
/// asserting a slot count this driver does not claim to support.
#[test]
fn a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits() {
    let mut runner = open_runner_with_slots("cache-too-small", 2 * turbospark_repack::TOP_K);
    let prompt = make_prompt(11, VOCAB);
    let vocab = VOCAB as usize;

    let expected = sequential_prefill(&mut runner, &prompt, vocab);
    let actual = chunked_prefill(&mut runner, &prompt, prompt.len(), vocab);
    assert_eq!(
        actual, expected,
        "the minimal safe slot count (2 * top_k) must reproduce the sequential logits"
    );
}

/// `TURBOSPARK_ROUTED_BATCH` must be REFUSED BY NAME on this family, never
/// ignored. The seam asks for the routed half as one route-list dispatch
/// pair, and the batched pair is wired in the gemma4 (INT4-affine) and
/// gpt-oss (MXFP4) chunked drivers alone.
///
/// The request is MEANINGFUL here, which is why the refusal is owed at all:
/// `qwen4_exp` HAS a routed half, unlike the dense drivers that may ignore
/// this flag legitimately because there is nothing for it to refer to
/// (`crates/runtime/CLAUDE.md` Gotcha 22). Serving the per-token routed loop
/// anyway would report the unbatched engine under the batched arm's label,
/// which is the exact failure that went unnoticed on the MoE `llama` family
/// until 2026-08-27 and was found by checking a doc claim against the binary
/// rather than by a test.
#[test]
fn the_batched_routed_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("routed-batch-refused");
    runner.set_routed_batch_prefill(true);

    let prompt = make_prompt(4, VOCAB);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&prompt, 0, &mut logits)
        .expect_err("the batched routed seam must be refused on this family");
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_ROUTED_BATCH"),
        "the refusal must name the seam the caller set; got {text}"
    );
}

/// The resident-GEMV seam (`TURBOSPARK_BATCHED_GEMV`) must be refused by name
/// too, and unlike the routed seam above it is meaningful on EVERY chunked
/// driver: this family has resident GEMVs whatever its routed layout, and the
/// M-row GEMM is wired in the gemma4 and dense-qwen drivers alone (step 6).
///
/// The test NAME is shared verbatim with the six sibling chunked-prefill
/// targets on purpose, so one grep across `crates/runtime/tests/` enumerates
/// every driver that carries the refusal and, by absence, any that does not.
#[test]
fn the_batched_gemv_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("batched-gemv-refused");
    runner.set_batched_gemv_prefill(true);

    let prompt = make_prompt(4, VOCAB);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&prompt, 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused on this family");
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_BATCHED_GEMV"),
        "the refusal must name the seam the caller set; got {text}"
    );
}

/// The `banks == 1` fallback, which is what a REAL install of this family
/// runs at the bench's own pinned slot count and which nothing here reached
/// before 2026-09-05.
///
/// `routed_pipeline_banks` degrades to one bank whenever
/// `expert_cache_slots < 2 * top_k`. The case above it is named
/// "cache_too_small_to_pipeline" and passes `2 * TOP_K`, which satisfies
/// `>=` and therefore PIPELINES -- so the fallback had a test named after it
/// and no test covering it.
///
/// What lives in the gap is a panic, not a slowdown. The loop reserves the
/// previous token's slots through `RoutedSlot::protect`, leaving
/// `slots - top_k` places for a token that may miss on all `top_k`; below
/// `2 * top_k` that is not enough and `ExpertCache::plan` asserts with
/// "expert cache cannot place requested misses". The reservation is stale
/// conservatism rather than a safety requirement, because the `banks == 1`
/// branch calls `retire_routed` BEFORE planning, so nothing is in flight by
/// the time `protect` is read (AGENTS.md Gotcha 64).
///
/// Found on the real `qwen4-reap288` install, which routes top-10 against
/// the bench's pinned 16 slots -- 6 places for 10 misses, on the DEFAULT
/// configuration. Gemma 4 hides the same shape because top-8 of 16 leaves
/// exactly 8.
#[test]
fn the_one_bank_fallback_reproduces_the_sequential_logits() {
    let slots = turbospark_repack::TOP_K;
    assert!(
        slots < 2 * turbospark_repack::TOP_K,
        "this case is only meaningful below the pipelining threshold"
    );
    let mut runner = open_runner_with_slots("one-bank-fallback", slots);
    let prompt = make_prompt(11, VOCAB);
    let vocab = VOCAB as usize;

    let expected = sequential_prefill(&mut runner, &prompt, vocab);
    let actual = chunked_prefill(&mut runner, &prompt, prompt.len(), vocab);
    assert_eq!(
        actual, expected,
        "the banks == 1 fallback must reproduce the sequential logits"
    );
}
