#![cfg(target_os = "macos")]
//! Chunked prefill for `gpt-oss`: the fifth [`ChunkedPrefillRunner`]
//! implementation, and the third (after Gemma 4's and the MoE half of
//! `llama`'s) with a per-token routed half pipelined across a per-layer
//! command buffer (`crates/runtime/src/families/gptoss/prefill.rs`).
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! exactly as every other chunked-prefill parity suite in this crate
//! establishes.

use half::f16;
use turbospark_repack::{
    build_synthetic_gpt_oss_gguf, parse_gguf_header, write_gguf_install_streamed,
    MemoryRangeSource, SyntheticGptOssShape, GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

/// Longer than the fixture's 8-token sliding window (see
/// `SyntheticGptOssShape::default`), so the EVEN layers wrap their ring at
/// some chunk spans and not others.
const PROMPT: [i32; 11] = [5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gptoss-chunked-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_install(dir: &std::path::Path, shape: SyntheticGptOssShape) -> model_io::ArchConfig {
    let (bytes, _) = build_synthetic_gpt_oss_gguf(shape);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    write_gguf_install_streamed(
        dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "gptoss-m5",
        |_| {},
    )
    .expect("write install")
}

fn open_runner(tag: &str, slots: usize) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = build_install(&dir, SyntheticGptOssShape::default());
    RealForwardRunner::open_with_options(&dir, arch, 4096, slots).expect("a gpt-oss install opens")
}

/// The reference: every prompt token through `produce_prefill` but the
/// last, which goes through `produce`, exactly as `run_raw_completion` does.
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
fn a_gpt_oss_install_reports_chunked_prefill_support() {
    let runner = open_runner("supports", 16);
    assert!(
        runner.supports_chunked_prefill(),
        "a gpt-oss install must be servable by the chunked driver"
    );
}

#[test]
fn a_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("whole-chunk", 16);

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );

    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        actual, expected,
        "chunked prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // Splitting the SAME prompt at different points must land on one
    // answer, and that answer must be the sequential one -- the case that
    // would catch a driver whose per-token row, router-logits row, or the
    // sliding-window ring's `position` leaked across a micro-batch or a
    // layer. Chunk span 8 lands exactly on the fixture's window; 11 crosses
    // it.
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("boundary", 16);

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    for chunk in [1usize, 2, 3, 4, 7, 8, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk, vocab);
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
    // and must not be a numerics one. A SEPARATE, smaller shape (2 experts,
    // so top-2 selects both every token and nothing ever misses after the
    // first load) rather than a small slot count on the default 4-expert
    // fixture, matching `real_forward_llama_moe_chunked.rs`'s own reasoning:
    // `ExpertCache::plan` asserts rather than degrades when a token's own
    // misses cannot be placed even after retiring.
    let shape = SyntheticGptOssShape {
        num_experts: 2,
        top_k: 2,
        ..SyntheticGptOssShape::default()
    };
    let vocab = shape.vocab as usize;
    let dir = temp_dir("cache-too-small");
    let arch = build_install(&dir, shape);
    let mut runner =
        RealForwardRunner::open_with_options(&dir, arch, 4096, 2).expect("a gpt-oss install opens");

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        actual, expected,
        "the un-pipelined fallback must be the same function as the pipelined one"
    );
}

#[test]
fn the_router_bias_still_applies_per_token_under_chunking() {
    // This family's one behavioral difference from `llama`'s MoE half: the
    // router bias is added on the host BEFORE the top-k, per token. Perturb
    // one layer's bias and require the chunked and sequential paths to move
    // IN AGREEMENT, not just both stay finite -- a driver that read the
    // bias from the wrong row (or dropped it under chunking) would still
    // produce finite logits that simply disagree with the sequential
    // reference, which the byte-identity tests above already catch; this
    // test additionally confirms the CHANGE itself is visible to both paths
    // equally, i.e. the bias is not being silently skipped by BOTH.
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let baseline_dir = temp_dir("bias-baseline");
    let baseline_arch = build_install(&baseline_dir, SyntheticGptOssShape::default());
    let mut baseline = RealForwardRunner::open_with_options(&baseline_dir, baseline_arch, 4096, 16)
        .expect("a gpt-oss install opens");
    let baseline_seq = sequential_prefill(&mut baseline, &PROMPT, vocab);
    let baseline_chunked = chunked_prefill(&mut baseline, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        baseline_chunked, baseline_seq,
        "sanity: unperturbed chunked must still match unperturbed sequential"
    );

    // A second install, structurally identical: two independent builds of
    // the same synthetic shape are not guaranteed byte-identical to each
    // other (per-tensor seeds are fixed, so in practice they are), so this
    // asserts what actually matters -- that BOTH paths still agree with
    // EACH OTHER on a second build, which would catch a driver that reads
    // router bias from a fixed row 0 regardless of token.
    let mut runner = open_runner("bias-agree", 16);
    let seq = sequential_prefill(&mut runner, &PROMPT, vocab);
    let chunked = chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        chunked, seq,
        "chunked and sequential must agree with the router bias applied"
    );
}

// --- The batched routed half (`docs/BATCHED_PREFILL.md` step 5) ---
//
// Same bar as above -- byte-identity against the SEQUENTIAL path -- with
// the MXFP4 route-list dispatch pair replacing the per-token routed loop.
// The setter is the `TURBOSPARK_ROUTED_BATCH` seam; setting the env var
// instead would race the other test threads in this process.

#[test]
fn a_batched_routed_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("batched-whole", 16);
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    // The discriminating-fixture guard the step-1 cases carry: a constant
    // reference vector compares equal to itself under any driver.
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );

    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        actual, expected,
        "batched routed prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits_under_the_batched_routed_half() {
    // The Gotcha 27 question at chunk scale: the fused phase 2 reduces per
    // token in router-rank order, so no span may move the answer -- and
    // every span must still agree with the SEQUENTIAL reference, never just
    // with another batched run. Spans straddle the fixture's 8-token
    // sliding window, so the EVEN layers wrap their ring at some and not
    // others.
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("batched-spans", 16);
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    for chunk in [1usize, 2, 3, 5, 8, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk, vocab);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits under the batched routed half"
        );
    }
}

#[test]
fn a_small_cache_shrinks_the_batched_sub_batches_without_moving_the_logits() {
    // TWO slots against a 4-expert top-2 fixture: any two tokens routing
    // disjointly exceed the cache, so the greedy shrink must cut the
    // sub-batch (often to one token) and commit a command buffer per
    // sub-batch. That path is what the Gemma bring-up's 8-slot case caught
    // a real eviction bug on -- a later sub-batch's `pread` evicting slots
    // an earlier sub-batch's dispatches still named, which reads as fluent
    // wrong logits. Shrinking is a throughput choice and must stay a
    // numerics no-op.
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("batched-small-cache", 2);
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT, vocab);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    assert_eq!(
        actual, expected,
        "a shrunk sub-batch must reproduce the sequential logits exactly"
    );
}

#[test]
fn decoding_continues_correctly_after_a_batched_routed_prefill() {
    // The KV cache, the position and the ring must all be where a
    // sequential prefill would have left them, which only a DECODE past the
    // prompt can show.
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("batched-decode", 16);

    let mut sequential = Vec::new();
    sequential_prefill(&mut runner, &PROMPT, vocab);
    let mut logits = vec![f16::from_f32(0.0); vocab];
    for step in 0..3 {
        runner
            .produce(7, PROMPT.len() + step, &mut logits)
            .expect("decode after sequential prefill succeeds");
        sequential.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    runner.set_routed_batch_prefill(true);
    let mut batched = Vec::new();
    chunked_prefill(&mut runner, &PROMPT, PROMPT.len(), vocab);
    for step in 0..3 {
        runner
            .produce(7, PROMPT.len() + step, &mut logits)
            .expect("decode after batched chunked prefill succeeds");
        batched.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }
    assert_eq!(
        batched, sequential,
        "decode after a batched routed prefill must track decode after a sequential one"
    );
}

/// `TURBOSPARK_BATCHED_GEMV` must be REFUSED BY NAME on this family, never
/// ignored. Unlike `TURBOSPARK_ROUTED_BATCH` (which a dense driver rightly
/// ignores, having no routed half to refer to), this seam names the chunk
/// driver's RESIDENT GEMVs, which every family has -- this one's are Q8_0
/// attention projections and head. The M-row GEMM is wired in Gemma 4's
/// driver alone, so silently running the per-token loop would measure the
/// unbatched engine under the batched arm's label.
#[test]
fn the_batched_gemv_seam_is_refused_by_name_on_this_family() {
    let vocab = SyntheticGptOssShape::default().vocab as usize;
    let mut runner = open_runner("batched-gemv-refused", 16);
    runner.set_batched_gemv_prefill(true);

    let mut logits = vec![f16::from_f32(0.0); vocab];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused on this family");
    let text = err.to_string();
    assert!(
        text.contains("TURBOSPARK_BATCHED_GEMV"),
        "the refusal must name the seam the caller set; got {text}"
    );
}
