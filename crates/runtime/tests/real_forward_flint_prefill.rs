#![cfg(target_os = "macos")]
//! The layer-batched chunked prefill for `flint`
//! (`crates/runtime/src/families/flint/prefill.rs`), behind
//! `TURBOSPARK_FLINT_CHUNKED_PREFILL=1`.
//!
//! **The bar is byte-identity against the SEQUENTIAL path, not coherence**,
//! as every other chunked-prefill suite in this crate establishes: the driver
//! issues the same dispatches in the same order as M `produce` calls, so the
//! logits after the prompt, and after decoding on top of its KV cache, must
//! match bit for bit. Untrained synthetic weights make "it decoded" worthless
//! as evidence (crate Gotcha 11); equality of every FP16 bit is not.
//!
//! The switch is cached in a `OnceLock` on first use, so it is set once here
//! before any runner is built, behind a `Once` so parallel tests agree.

use std::sync::Once;

use half::f16;
use turbospark_repack::build_synthetic_flint_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

static SWITCH: Once = Once::new();

fn switch_on() {
    SWITCH.call_once(|| std::env::set_var("TURBOSPARK_FLINT_CHUNKED_PREFILL", "1"));
}

const VOCAB: usize = 64;
/// Longer than the fixture's 8-token window and than one 16-token micro-batch,
/// so the sliding layer wraps its ring and the driver runs three batches
/// (16, 16, 8) at the whole-prompt chunk size.
const PROMPT_LEN: usize = 40;

fn prompt() -> Vec<i32> {
    (0..PROMPT_LEN)
        .map(|i| ((i * 7 + 3) % VOCAB) as i32)
        .collect()
}

fn open_runner(tag: &str, slots: usize) -> RealForwardRunner {
    switch_on();
    let dir = std::env::temp_dir().join(format!(
        "turbospark-flint-prefill-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_flint_install(&dir, VOCAB as i64, "flint-toy").expect("install");
    RealForwardRunner::open_with_options(&dir, arch, 4096, slots).expect("a flint install opens")
}

fn bits(logits: &[f16]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner
                .produce(token, position, &mut logits)
                .expect("produce");
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .expect("produce_prefill");
        }
    }
    bits(&logits)
}

/// The same prompt through `prefill_chunk`, split into spans of `chunk`.
fn chunked_prefill(runner: &mut RealForwardRunner, tokens: &[i32], chunk: usize) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB];
    let mut offset = 0;
    while offset < tokens.len() {
        let take = (tokens.len() - offset).min(chunk);
        runner
            .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
            .expect("prefill_chunk");
        offset += take;
    }
    bits(&logits)
}

/// Greedy-decodes `steps` tokens from where a prefill left off, returning every
/// step's logits bits: equal KV state is what makes these equal.
fn decode_on_top(
    runner: &mut RealForwardRunner,
    mut position: usize,
    steps: usize,
) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    let mut token = 5i32;
    for _ in 0..steps {
        let mut logits = vec![f16::from_f32(0.0); VOCAB];
        runner
            .produce(token, position, &mut logits)
            .expect("decode");
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(bits(&logits));
        position += 1;
    }
    out
}

#[test]
fn chunked_prefill_is_byte_identical_to_sequential_at_every_chunk_split() {
    let tokens = prompt();
    for slots in [4usize, 8] {
        let mut reference = open_runner(&format!("ref{slots}"), slots);
        let want = sequential_prefill(&mut reference, &tokens);
        let want_decode = decode_on_top(&mut reference, tokens.len(), 6);

        // 40 (batches 16/16/8), 16 (exactly one batch each), 13 (spans that
        // straddle a micro-batch boundary), 1 (every token alone).
        for chunk in [PROMPT_LEN, 16, 13, 1] {
            let mut runner = open_runner(&format!("c{chunk}s{slots}"), slots);
            let got = chunked_prefill(&mut runner, &tokens, chunk);
            assert_eq!(
                got, want,
                "chunk {chunk}, {slots} slots: the logits after the prompt differ from sequential"
            );
            let got_decode = decode_on_top(&mut runner, tokens.len(), 6);
            assert_eq!(
                got_decode, want_decode,
                "chunk {chunk}, {slots} slots: decoding on the chunk-built KV cache differs"
            );
        }
    }
}

/// Two fresh runners over the same chunked prefill agree: the driver's
/// pipelined routed half must not make output depend on cache state or timing.
#[test]
fn chunked_prefill_is_deterministic_across_runners() {
    let tokens = prompt();
    let a = chunked_prefill(&mut open_runner("det-a", 8), &tokens, PROMPT_LEN);
    let b = chunked_prefill(&mut open_runner("det-b", 8), &tokens, PROMPT_LEN);
    assert_eq!(a, b);
}

/// The slot count must not change the answer (AGENTS.md Gotcha 27: routed
/// slots are dispatched in the router's ranking, so the reduction order does
/// not follow cache state).
#[test]
fn the_slot_count_does_not_change_the_chunked_answer() {
    let tokens = prompt();
    let small = chunked_prefill(&mut open_runner("slots4", 4), &tokens, PROMPT_LEN);
    let large = chunked_prefill(&mut open_runner("slots8", 8), &tokens, PROMPT_LEN);
    assert_eq!(small, large);
}
