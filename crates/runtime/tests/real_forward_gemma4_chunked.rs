#![cfg(target_os = "macos")]
//! Chunked prefill on the real-checkpoint Gemma 4 flow
//! (`docs/BATCHED_PREFILL.md` step 1, and steps 2-3's batched routed
//! half): a whole chunk of prompt tokens runs its attention-and-router
//! half for every token into ONE command buffer per layer, and its
//! routed half either per token (step 1) or as one route-list dispatch
//! pair per union-bounded sub-batch (steps 2 and 3, the
//! `set_routed_batch_prefill` cases below).
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
use turbospark_repack::{
    build_synthetic_gemma4_real_install, build_synthetic_gemma4_real_install_at_shared_bits,
};
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

/// The same fixture with the SHARED EXPERT at four bits, which is what the
/// real `mlx-community/gemma-4-26b-a4b-it-4bit` install declares. The
/// default above writes it at EIGHT and is the repo's only coverage of an
/// INT8 resident GEMV inside a whole Gemma forward pass, so it is left
/// alone; `encode_gemm_any` is INT4-affine only, so the batched shared
/// expert is unreachable on it and the eight-bit case is pinned separately
/// as a REFUSAL below.
fn build_install_int4_shared(dir: &std::path::Path, experts: i64) -> model_io::ArchConfig {
    build_synthetic_gemma4_real_install_at_shared_bits(
        dir,
        VOCAB,
        2,
        experts,
        4,
        8,
        "tiny-gemma4-int4-shared",
        4,
    )
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

// --- The batched routed half (`docs/BATCHED_PREFILL.md` steps 2 and 3) ---
//
// Same bar as above -- byte-identity against the SEQUENTIAL path -- with
// the route-list dispatch pair replacing the per-token routed loop. The
// setter is the `MFERENCE_ROUTED_BATCH` seam; setting the env var instead
// would race the other test threads in this process.

#[test]
fn a_batched_routed_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );
    // The same discriminating-fixture guard the step-1 case carries: a
    // constant reference vector compares equal to itself under any driver.
    assert!(
        expected.iter().any(|&v| v != expected[0]),
        "degenerate reference logits: this fixture cannot discriminate"
    );
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "batched routed prefill must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits_under_the_batched_routed_half() {
    // The Gotcha 27 question at chunk scale: the fused phase 2 reduces
    // per token in router-rank order, so no span may move the answer --
    // and every span must still agree with the SEQUENTIAL reference,
    // never just with another batched run.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 2, 3, 5, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits under the batched routed half"
        );
    }
}

#[test]
fn a_small_cache_shrinks_the_batched_sub_batches_without_moving_the_logits() {
    // Eight slots against a fixture whose union across the micro-batch
    // can exceed it: the driver must shrink the sub-batch (the greedy
    // union bound) rather than hand `ExpertCache::plan` more experts
    // than it has slots, which asserts rather than degrades.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 8)
        .expect("real-naming install opens");
    runner.set_routed_batch_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    let actual = chunked_prefill(&mut runner, &PROMPT, PROMPT.len());
    assert_eq!(
        actual, expected,
        "the union-bounded sub-batch shrink must be a throughput choice only"
    );
}

#[test]
fn decoding_continues_correctly_after_a_batched_routed_prefill() {
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

    runner.set_routed_batch_prefill(true);
    let mut batched = Vec::new();
    chunked_prefill(&mut runner, &PROMPT, 4);
    for (step, &token) in [3i32, 7, 2].iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, PROMPT.len() + step, &mut logits)
            .expect("decode after batched chunked prefill succeeds");
        batched.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    assert_eq!(
        batched, sequential,
        "decode diverged after a batched routed prefill"
    );
}

// --- The batched resident GEMVs (`MFERENCE_BATCHED_GEMV`) ---
//
// `docs/BATCHED_PREFILL.md`'s 29.7% row: the four attention projections
// always, and the shared expert's three when the routed half is batched
// too. Same bar as everything above -- byte-identity against the
// SEQUENTIAL path -- and it is a real bar here rather than a hope, because
// `dequant_int4_gemm_simd` is measured bit-exact against
// `dequant_int4_gemv_simd` (`crates/gpu/tests/dequant_int4_gemm_parity.rs`).
// The setter is used rather than the env var, which would race the other
// test threads in this process.

/// A prompt long enough to WRAP this fixture's sliding-window ring, which
/// `PROMPT` cannot: the ring holds `sliding_window + MAX_PREFILL_CHUNK_TOKENS`
/// = 8 + 128 = 136 tokens, and 136 is not a multiple of `MAX_PREFILL_BATCH`,
/// so the micro-batch covering positions [128, 144) straddles the wrap.
/// That is the one case where a batched K/V projection writing M ADJACENT
/// slots would run past the layer's buffer.
fn wrapping_prompt() -> Vec<i32> {
    // Deterministic, every id inside the vocabulary, and not constant --
    // a repeated token would route every position to the same experts and
    // hide a bank or slot bug.
    (0..160)
        .map(|i| ((i * 37 + 11) % VOCAB as usize) as i32)
        .collect()
}

#[test]
fn a_batched_gemv_chunked_prefill_is_byte_identical_to_the_sequential_one() {
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_batched_gemv_prefill(true);

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
        "batched resident GEMVs must reproduce the sequential logits exactly"
    );
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits_under_the_batched_gemvs() {
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_batched_gemv_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 2, 3, 5, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} changed the logits under the batched GEMVs"
        );
    }
}

#[test]
fn both_prefill_seams_together_are_byte_identical_to_the_sequential_one() {
    // The combination is the configuration a throughput A/B would run and
    // the only one in which the SHARED EXPERT's three projections batch,
    // so neither seam alone covers it.
    let dir = temp_dir();
    let arch = build_install_int4_shared(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_routed_batch_prefill(true);
    runner.set_batched_gemv_prefill(true);

    let expected = sequential_prefill(&mut runner, &PROMPT);
    for chunk in [1usize, 4, 11] {
        let actual = chunked_prefill(&mut runner, &PROMPT, chunk);
        assert_eq!(
            actual, expected,
            "chunk span {chunk} moved the logits with both seams on"
        );
    }
}

#[test]
fn a_batched_projection_that_straddles_the_ring_wrap_still_matches_sequential() {
    // THE CASE `PROMPT` CANNOT REACH. Everything else in this file runs 11
    // tokens against a 136-token ring, so no write ever wraps and the span
    // split is dead code the rest of the suite cannot see. Reverting
    // `ring_spans` to one unsplit call reddens THIS case and nothing else
    // in the file -- checked, not assumed.
    let dir = temp_dir();
    let arch = build_install_int4_shared(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    let prompt = wrapping_prompt();
    assert!(
        prompt.len() > 144,
        "the prompt must reach past the micro-batch that straddles the wrap at 136"
    );

    let expected = sequential_prefill(&mut runner, &prompt);
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "the reference itself must be finite before anything is compared to it"
    );

    runner.set_batched_gemv_prefill(true);
    let actual = chunked_prefill(&mut runner, &prompt, 16);
    assert_eq!(
        actual, expected,
        "a batched K/V projection straddling the ring wrap moved the logits"
    );

    runner.set_routed_batch_prefill(true);
    let both = chunked_prefill(&mut runner, &prompt, 16);
    assert_eq!(
        both, expected,
        "the same, with the routed half batched as well"
    );
}

#[test]
fn a_shared_expert_with_no_batched_kernel_is_refused_by_name() {
    // AGENTS.md Gotcha 35 at the dispatch: `encode_gemm_any` is
    // INT4-affine only, and a sequential fallback here would be
    // numerically identical -- so a caller who asked for the batched
    // engine would measure the unbatched one and report it as batched.
    // The DEFAULT fixture writes its shared MLP at eight bits, so it is
    // exactly the install that has to be refused rather than looped.
    let dir = temp_dir();
    let arch = build_install(&dir, 16);
    let mut runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16)
        .expect("real-naming install opens");
    runner.set_routed_batch_prefill(true);
    runner.set_batched_gemv_prefill(true);

    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&PROMPT, 0, &mut logits)
        .expect_err("an INT8 shared expert has no batched kernel");
    assert!(
        err.contains("mlp.gate_proj.weight") && err.contains("no BATCHED kernel"),
        "the refusal must name the tensor and the reason: {err}"
    );
}

#[test]
fn decoding_continues_correctly_after_a_batched_gemv_prefill() {
    // Decode itself never batches, so this is really asking whether the
    // batched prefill left the KV cache and the residual stream where the
    // sequential one leaves them.
    let dir = temp_dir();
    let arch = build_install_int4_shared(&dir, 16);
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

    runner.set_routed_batch_prefill(true);
    runner.set_batched_gemv_prefill(true);
    let mut batched = Vec::new();
    chunked_prefill(&mut runner, &PROMPT, 4);
    for (step, &token) in [3i32, 7, 2].iter().enumerate() {
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, PROMPT.len() + step, &mut logits)
            .expect("decode after batched chunked prefill succeeds");
        batched.push(logits.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    assert_eq!(
        batched, sequential,
        "decode diverged after a batched-GEMV prefill"
    );
}
