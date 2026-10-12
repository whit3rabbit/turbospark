#![cfg(target_os = "macos")]
//! Chunked prefill with the MTP head open
//! (`SpeculativeProducer::prefill_chunk_priming_drafter`) must leave the head
//! primed exactly as the token-at-a-time walk does: every prompt position
//! whose next token is known primed with that pair, and the last prompt
//! position's trunk hidden state in row 0 of `scratch.x` for the first draft
//! step. A head primed from the wrong row, a position skipped at a
//! micro-batch or chunk boundary, or a missing row-0 copy all still produce a
//! finite, plausible draft, so the case compares the first draft's logits
//! (and the trunk's own last logits) against the reference walk.
//!
//! The per-token arm and the scalar batched arm are bit-exact on this INT4
//! fixture, so those comparisons are `==`. The Steel arm rounds weights to
//! FP16 operands and is held to a bound instead. Prompt lengths cross the
//! 16-row and 64-row micro-batches and the 128-token chunk.
//!
//! The batched arms prime the head one micro-batch at a time
//! (`mtp_prime_rows_batched`) unless `set_mtp_batched_prime(false)`, which
//! keeps the per-row step; both are covered on the scalar arm.

use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;
use turbospark_repack::{build_synthetic_qwen_gdn_dense_install_with_mtp, peek_manifest_arch};
use turbospark_runtime::{
    DraftPolicies, LogitProducer, MtpDraftPolicy, RealForwardRunner, SpeculativeProducer,
};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const INT4: u32 = 4;
const CHUNK: usize = 128;
const DRAFT_TOKEN: i32 = 9;
const LENGTHS: [usize; 6] = [1, 5, 17, 40, 70, 150];

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn open_with_head(tag: &str) -> RealForwardRunner {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-mtp-chunked-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-chunked", INT4)
        .expect("an INT4 dense qwen3_5 install with an MTP head builds");
    let peeked = peek_manifest_arch(&dir).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        peeked,
        4096,
        16,
        DraftPolicies::mtp(MtpDraftPolicy::Fixed(2)),
    )
    .expect("install with a head opens with the head asked for");
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

fn prompt(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| ((i * 11 + 5) % VOCAB as usize) as i32)
        .collect()
}

/// The token-at-a-time walk the speculative loop has always run: every
/// prompt token through the headful `produce`, the head primed after each
/// one but the last, then the first draft step off the last position.
fn reference(runner: &mut RealForwardRunner, ids: &[i32]) -> (Vec<f32>, Vec<f32>) {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (i, &token) in ids.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < ids.len() {
            runner.prime_drafter(ids[i + 1], i).expect("prime");
        }
    }
    let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .draft_step(DRAFT_TOKEN, ids.len() - 1, &mut draft)
        .expect("draft");
    (to_f32(&logits), to_f32(&draft))
}

/// The same prompt through `prefill_chunk_priming_drafter` in the chunked
/// driver's 128-token spans, then the same draft step.
fn chunked(runner: &mut RealForwardRunner, ids: &[i32]) -> (Vec<f32>, Vec<f32>) {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut start = 0;
    while start < ids.len() {
        let end = (start + CHUNK).min(ids.len());
        runner
            .prefill_chunk_priming_drafter(
                &ids[start..end],
                start,
                &mut logits,
                ids.get(end).copied(),
            )
            .expect("chunked prefill priming the head");
        start = end;
    }
    let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .draft_step(DRAFT_TOKEN, ids.len() - 1, &mut draft)
        .expect("draft");
    (to_f32(&logits), to_f32(&draft))
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
fn the_runner_offers_chunked_drafter_prefill_only_with_the_mtp_head_open() {
    let runner = open_with_head("supports");
    assert!(runner.supports_chunked_drafter_prefill());
}

#[test]
fn the_per_token_arm_primes_the_head_exactly_as_the_token_walk_does() {
    let mut runner = open_with_head("per-token");
    runner.set_batched_gemv_prefill(false);
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(&mut runner, &ids);
        let got = chunked(&mut runner, &ids);
        assert_eq!(got.0, want.0, "{n} tokens: the trunk's last logits differ");
        assert_eq!(got.1, want.1, "{n} tokens: the first draft differs");
    }
}

#[test]
fn the_scalar_batched_arm_primes_the_head_exactly_too() {
    let mut runner = open_with_head("scalar-batched");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(false);
    runner.set_mtp_batched_prime(true);
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(&mut runner, &ids);
        let got = chunked(&mut runner, &ids);
        assert_eq!(got.0, want.0, "{n} tokens: the trunk's last logits differ");
        assert_eq!(got.1, want.1, "{n} tokens: the first draft differs");
    }
}

#[test]
fn the_scalar_batched_arm_primes_the_head_exactly_per_row_too() {
    let mut runner = open_with_head("scalar-batched-per-row");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(false);
    runner.set_mtp_batched_prime(false);
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(&mut runner, &ids);
        let got = chunked(&mut runner, &ids);
        assert_eq!(got.0, want.0, "{n} tokens: the trunk's last logits differ");
        assert_eq!(got.1, want.1, "{n} tokens: the first draft differs");
    }
}

#[test]
fn the_steel_arm_primes_the_head_within_the_bound() {
    let mut runner = open_with_head("steel");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(true);
    runner.set_mtp_batched_prime(true);
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(&mut runner, &ids);
        let got = chunked(&mut runner, &ids);
        for (label, g, w) in [("trunk", &got.0, &want.0), ("draft", &got.1, &want.1)] {
            let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = g
                .iter()
                .zip(w)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                worst <= 0.01 * scale,
                "{n} tokens: {label} logits moved {worst} against max {scale}"
            );
        }
    }
}
