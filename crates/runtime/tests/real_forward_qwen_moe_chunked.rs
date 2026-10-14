#![cfg(target_os = "macos")]
//! Chunked prefill for the MoE half of the qwen flow (`qwenGdnMoe`), the
//! driver in `crates/runtime/src/families/qwen/moe_prefill.rs`.
//!
//! **The bar is byte-identity against the SEQUENTIAL path**, as in every
//! other chunked test in this crate: each case compares against
//! `produce_prefill` / `produce`, never against another chunked arm. What
//! this file proves is that grouping tokens into fewer command buffers does
//! not change the bytes, on both routed layouts this family ships: INT4
//! affine experts (Qwen 3.6, Ornith 35B) and GGUF MXFP4 experts re-blocked
//! from MLX mxfp4 (MiniCPM-V 4.7). The fixtures are deterministic but not
//! trained.
//!
//! The fixtures carry 32 experts at top-8 and open at 16 slots unless a case
//! says otherwise, so the cache MISSES and EVICTS during prefill. At 16
//! experts every expert would fit and `RoutedSlot::protect` would never
//! decide anything.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 32;
/// The fixtures' own `top_k_experts` (`num_experts.min(8)`).
const TOP_K: usize = 8;
const SLOTS: usize = 16;
const MAX_CONTEXT: usize = 256;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen-moe-chunked-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Affine,
    Mxfp4,
}

fn open_runner(tag: &str, layout: Layout, slots: usize) -> RealForwardRunner {
    let dir = temp_dir(tag);
    let arch = match layout {
        Layout::Affine => turbospark_repack::build_synthetic_qwen_gdn_moe_install(
            &dir,
            VOCAB,
            LAYERS,
            EXPERTS,
            "tiny-qwen36-chunked",
        ),
        Layout::Mxfp4 => turbospark_repack::build_synthetic_qwen_gdn_moe_mxfp4_install(
            &dir,
            VOCAB,
            LAYERS,
            EXPERTS,
            "tiny-qwen36-mxfp4-chunked",
        ),
    }
    .expect("MoE qwen install builds");
    assert_eq!(arch.top_k_experts as usize, TOP_K, "fixture top_k changed");
    let runner = RealForwardRunner::open_with_options(&dir, arch, MAX_CONTEXT, slots)
        .expect("MoE qwen install opens");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

fn make_prompt(len: usize) -> Vec<i32> {
    (0..len)
        .map(|p| ((p * 37 + 11) % VOCAB as usize) as i32)
        .collect()
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
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

/// Greedy decode `steps` tokens after the prompt, returning the tokens and
/// the final logits. The runner's state is whatever prefill left.
fn greedy_continue(
    runner: &mut RealForwardRunner,
    first: i32,
    start: usize,
    steps: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut token = first;
    let mut out = Vec::new();
    for i in 0..steps {
        runner
            .produce(token, start + i, &mut logits)
            .expect("decode succeeds");
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(token);
    }
    (out, logits.iter().map(|v| v.to_f32()).collect())
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

#[test]
fn a_moe_qwen_install_reports_chunked_prefill_support() {
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let runner = open_runner("supports", layout, SLOTS);
        assert!(
            runner.supports_chunked_prefill(),
            "a {layout:?} MoE qwen install must be servable by the chunked driver"
        );
    }
}

#[test]
fn the_chunk_boundary_does_not_move_the_logits() {
    // Spans of 2 and up put several tokens into one micro-batch, which is
    // what would catch a single-row `moe_x` or `router_logits_f32`: every
    // token but the last would route on, or read, the last token's row.
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let mut runner = open_runner("boundary", layout, SLOTS);
        let prompt = make_prompt(11);
        let expected = sequential_prefill(&mut runner, &prompt);
        assert!(
            expected.iter().all(|v| v.is_finite()),
            "{layout:?}: the reference must be finite"
        );
        assert!(
            expected.iter().any(|&v| v != expected[0]),
            "{layout:?}: degenerate reference logits, this fixture cannot discriminate"
        );
        for chunk in [1usize, 2, 3, 4, 7, 11] {
            let actual = chunked_prefill(&mut runner, &prompt, chunk);
            assert_eq!(
                actual, expected,
                "{layout:?}: chunk span {chunk} changed the logits"
            );
        }
    }
}

#[test]
fn a_prompt_longer_than_one_micro_batch_is_byte_identical() {
    // 37 tokens as ONE chunk: two full 16-token micro-batches and a 5-token
    // tail inside a single `prefill_chunk` call, so the micro-batch boundary
    // (not only the chunk boundary) is crossed with the expert cache warm.
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let mut runner = open_runner("long", layout, SLOTS);
        let prompt = make_prompt(37);
        let expected = sequential_prefill(&mut runner, &prompt);
        let actual = chunked_prefill(&mut runner, &prompt, prompt.len());
        assert_eq!(
            actual, expected,
            "{layout:?}: a 37-token chunk changed the logits"
        );
    }
}

#[test]
fn decode_after_a_chunked_prefill_continues_identically() {
    // Prefill leaves the KV cache, the GDN state and the expert cache behind.
    // A driver that left any of them different would show here even when the
    // prefill's own last logits matched.
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let mut runner = open_runner("continue", layout, SLOTS);
        let prompt = make_prompt(13);
        let seq_logits = sequential_prefill(&mut runner, &prompt);
        let (seq_tokens, seq_tail) =
            greedy_continue(&mut runner, argmax(&seq_logits), prompt.len(), 8);

        let chunk_logits = chunked_prefill(&mut runner, &prompt, 5);
        assert_eq!(chunk_logits, seq_logits, "{layout:?}: prefill logits");
        let (chunk_tokens, chunk_tail) =
            greedy_continue(&mut runner, argmax(&chunk_logits), prompt.len(), 8);
        assert_eq!(
            chunk_tokens, seq_tokens,
            "{layout:?}: decoded tokens diverged"
        );
        assert_eq!(chunk_tail, seq_tail, "{layout:?}: final decode logits");
    }
}

/// The `banks == 1` fallback, below `2 * top_k` slots. The loop must hand
/// the next token an EMPTY `protect` set there (AGENTS.md Gotcha 64), or
/// `ExpertCache::plan` asserts with "expert cache cannot place requested
/// misses": at `top_k` slots the previous token's reservation would leave no
/// room at all.
#[test]
fn the_one_bank_fallback_reproduces_the_sequential_logits() {
    let slots = TOP_K;
    assert!(
        slots < 2 * TOP_K,
        "only meaningful below the pipelining threshold"
    );
    for layout in [Layout::Affine, Layout::Mxfp4] {
        let mut runner = open_runner("one-bank", layout, slots);
        let prompt = make_prompt(11);
        let expected = sequential_prefill(&mut runner, &prompt);
        let actual = chunked_prefill(&mut runner, &prompt, prompt.len());
        assert_eq!(
            actual, expected,
            "{layout:?}: the banks == 1 fallback changed the logits"
        );
    }
}

/// `TURBOSPARK_ROUTED_BATCH` names an arm this driver does not have, so it is
/// refused by name rather than served by the per-token loop under the
/// batched label (`crates/runtime/CLAUDE.md` Gotcha 22).
#[test]
fn the_batched_routed_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("routed-batch-refused", Layout::Affine, SLOTS);
    runner.set_routed_batch_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(4), 0, &mut logits)
        .expect_err("the batched routed seam must be refused on this family");
    assert!(err.contains("TURBOSPARK_ROUTED_BATCH"), "got {err}");
}

/// Shared verbatim with the sibling chunked-prefill targets, so one grep
/// across `crates/runtime/tests/` lists every driver carrying the refusal.
#[test]
fn the_batched_gemv_seam_is_refused_by_name_on_this_family() {
    let mut runner = open_runner("batched-gemv-refused", Layout::Affine, SLOTS);
    runner.set_batched_gemv_prefill(true);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner
        .prefill_chunk(&make_prompt(4), 0, &mut logits)
        .expect_err("the batched resident-GEMV seam must be refused on this family");
    assert!(err.contains("TURBOSPARK_BATCHED_GEMV"), "got {err}");
}
