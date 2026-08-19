#![cfg(target_os = "macos")]
//! The DFlash2 drafter's runtime path, against a synthetic install
//! (`docs/DFLASH2.md`). Milliseconds, no network: the fixture from
//! `build_synthetic_qwen_gdn_dense_install_with_dflash` is structurally the
//! published 81-tensor drafter, so the whole round -- capture, context-KV
//! write, block forward, selector, verify, rollback, rewind -- is exercised
//! before any real install is asked for.
//!
//! # What a synthetic install can and cannot say
//!
//! Weights are deterministic and untrained, so proposals are meaningless by
//! construction and NOTHING here may assert a draft is good. Accept length
//! is `crates/bench`'s question. What these CAN say, and do:
//!
//! 1. the round RUNS end to end at every block the policy admits;
//! 2. the committed stream is BYTE-IDENTICAL to a sequential decode of the
//!    same prompt -- the losslessness property, which is a property of the
//!    MACHINERY (accept/rollback/replay) and not of the weights, and so is
//!    meaningful even on garbage weights;
//! 3. a drafterless install is refused BY NAME under an explicit ask.
//!
//! The fixture's two sizes exist for this file: the trunk needs 64 layers
//! because the aux taps read layers up to 61, and the vocab must exceed the
//! mask token id 248,070.

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_dflash;
use turbospark_runtime::{DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy};

use foundation::LogitValue;
use turbospark_runtime::{RealForwardRunner, SpeculativeProducer};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Above the mask token (248,070) so the drafter's mask rows embed legally;
/// below anything that would make the fixture slow to build.
const VOCAB: i64 = 248_100;
/// 64 because the aux taps read trunk layers [5, 19, 33, 47, 61].
const LAYERS: i64 = 64;
const SLOTS: usize = 16;
const MAX_CONTEXT: usize = 4096;
/// Greedy, the only mode the loop admits.
const PROMPT: [i32; 8] = [5, 7, 9, 11, 13, 15, 17, 19];

fn temp_dir() -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("turbospark-dflash-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn build() -> std::path::PathBuf {
    let dir = temp_dir();
    build_synthetic_qwen_gdn_dense_install_with_dflash(&dir, VOCAB, LAYERS, "dflash-toy", 4)
        .expect("the dense install with a DFlash2 drafter writes");
    dir
}

fn open(dir: &std::path::Path, block: usize) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        arch,
        MAX_CONTEXT,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(block),
        },
    )
    .expect("the dflash install opens with the drafter on")
}

fn argmax(logits: &[LogitValue]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, l) in logits.iter().enumerate() {
        let v = l.to_f32();
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

/// One manual speculative ROUND, the loop's own shape: draft, verify,
/// accept the longest agreeing prefix, roll back and replay the accepted
/// prefix when the verify overshot, rewind the drafter. Returns the
/// ACCEPTED proposals and the bonus token (the row after the last accept).
#[allow(clippy::too_many_arguments)]
fn round(
    runner: &mut RealForwardRunner,
    next: i32,
    base: usize,
    block: usize,
    vocab: usize,
) -> (Vec<i32>, i32) {
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(next, base, &mut proposals)
        .expect("the draft block runs");
    assert_eq!(
        proposals.len(),
        block,
        "the drafter proposed the wrong count"
    );
    assert!(
        proposals.iter().all(|&t| (t as usize) < vocab),
        "a proposal left the vocab"
    );

    let mut feed = Vec::with_capacity(block + 1);
    feed.push(next);
    feed.extend_from_slice(&proposals);
    let mut batch_logits = vec![LogitValue::from_f32(0.0); (block + 1) * vocab];
    let point = runner.checkpoint();
    runner
        .verify(&feed, base, &mut batch_logits)
        .expect("the batched verify runs");

    let row_at = |logits: &[LogitValue], i: usize| argmax(&logits[i * vocab..(i + 1) * vocab]);
    let mut accepted = 0usize;
    while accepted < block && row_at(&batch_logits, accepted) == proposals[accepted] {
        accepted += 1;
    }
    if accepted < block {
        runner.rollback(&point);
        runner
            .verify(
                &feed[..accepted + 1],
                base,
                &mut batch_logits[..(accepted + 1) * vocab],
            )
            .expect("the shortened replay verify runs");
    }
    runner
        .dflash_rewind_to(base + accepted)
        .expect("the drafter rewinds to the accepted end");
    let bonus = row_at(&batch_logits, accepted);
    (proposals[..accepted].to_vec(), bonus)
}

/// The whole point: a speculative walk commits EXACTLY the tokens a
/// sequential decode of the same prompt commits, on garbage weights and
/// under a drafter whose proposals the reference disagrees with -- which is
/// the interesting case, because it is the one that exercises rollback.
#[test]
fn speculative_rounds_commit_the_sequential_stream() {
    let dir = build();
    let vocab = VOCAB as usize;
    let block = 2usize;
    let mut runner = open(&dir, block);

    // The sequential reference, on a SECOND open with the drafter off: same
    // install, no capture hooks, no drafter state.
    let mut reference = {
        let arch = turbospark_repack::peek_manifest_arch(&dir).unwrap();
        RealForwardRunner::open_with_options_and_speculation(
            &dir,
            arch,
            MAX_CONTEXT,
            SLOTS,
            DraftPolicies::off(),
        )
        .expect("opens with every drafter off")
    };

    // Prefill both, priming the drafter as the loop does.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner
                .prime_drafter(PROMPT[i + 1], i)
                .expect("the dflash context write primes from the capture");
        }
    }
    let mut next = argmax(&logits);
    let mut committed: Vec<i32> = Vec::new();
    // The loop's own accounting: `next` is committed FIRST (it is the last
    // round's bonus), then the round's accepted proposals extend the
    // stream, and its bonus becomes the next `next` -- pushed by the NEXT
    // iteration, not twice.
    while committed.len() < 12 {
        committed.push(next);
        let base = PROMPT.len() + committed.len() - 1;
        let (accepted, bonus) = round(&mut runner, next, base, block, vocab);
        committed.extend_from_slice(&accepted);
        next = bonus;
    }
    committed.truncate(12);

    // The reference decodes the same count of tokens sequentially.
    let mut ref_logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut ref_stream: Vec<i32> = Vec::new();
    let mut feed_next = argmax_of_prompt(&mut reference, &mut ref_logits);
    for _ in 0..committed.len() {
        ref_stream.push(feed_next);
        reference
            .produce(
                feed_next,
                PROMPT.len() + ref_stream.len() - 1,
                &mut ref_logits,
            )
            .expect("reference produce");
        feed_next = argmax(&ref_logits);
    }

    assert_eq!(
        committed, ref_stream,
        "the speculative walk diverged from the sequential stream"
    );
}

/// Walks the reference over the prompt, returning the first generated
/// token; the caller then continues token by token.
fn argmax_of_prompt(reference: &mut RealForwardRunner, logits: &mut [LogitValue]) -> i32 {
    for (i, &token) in PROMPT.iter().enumerate() {
        reference
            .produce(token, i, logits)
            .expect("reference prefill");
    }
    argmax(logits)
}

/// The drafter refuses an install without `dflash.*` tensors BY NAME under
/// an explicit ask, matching the MTP head's refusal shape.
#[test]
fn a_drafterless_install_refuses_an_explicit_ask() {
    let dir = temp_dir();
    turbospark_repack::build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "dense-toy")
        .expect("the headless, drafterless dense install writes");
    let arch = turbospark_repack::peek_manifest_arch(&dir).unwrap();
    let msg = match RealForwardRunner::open_with_options_and_speculation(
        &dir,
        arch,
        MAX_CONTEXT,
        SLOTS,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(2),
        },
    ) {
        Err(e) => e.to_string(),
        Ok(_) => String::new(),
    };
    assert!(
        msg.contains("dflash.fc.weight"),
        "the refusal should name the missing tensor: {msg}"
    );
}

/// Block 8 is the trained block and the widest this fixture exercises; the
/// round's machinery (rollback included) is block-shaped, so one wide block
/// is worth its milliseconds.
#[test]
fn the_round_runs_at_the_trained_block() {
    let dir = build();
    let vocab = VOCAB as usize;
    let block = 8usize;
    let mut runner = open(&dir, block);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let next = argmax(&logits);
    let (accepted, _bonus) = round(&mut runner, next, PROMPT.len(), block, vocab);
    assert!(
        accepted.len() <= block,
        "a round accepts at most its block, got {}",
        accepted.len()
    );
}
