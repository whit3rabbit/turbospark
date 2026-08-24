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

/// The same install opened with a directional-steering edit on every layer.
///
/// Only the STEERING differs from `open`; the drafter, the block, the context
/// and the slots are the same, so a difference between two arms opened this
/// way is the edit and nothing else.
fn open_steered(dir: &std::path::Path, block: usize, alpha: f32) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    let hidden = arch.hidden_size as usize;
    let layers = (0..arch.num_layers as usize)
        .map(|l| {
            let values = (0..hidden)
                .map(|i| 0.5 * (i as f32 * 0.37 + l as f32 * 1.13).sin())
                .collect::<Vec<f32>>();
            Some(model_io::LayerDirection::new(values))
        })
        .collect();
    RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        dir,
        arch,
        MAX_CONTEXT,
        turbospark_runtime::ExpertCacheSlots::Fixed(SLOTS),
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(block),
        },
        turbospark_runtime::SteeringPolicy {
            set: Some(model_io::SteeringSet {
                layers,
                hidden,
                declared_mode: None,
                declared_arch: None,
            }),
            mode: foundation::SteeringMode::Ablate,
            alpha,
            target: 0.0,
            gate_threshold: 0.0,
        },
    )
    .expect("the dflash install opens steered with the drafter on")
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

/// Every other test in this file is SELF-RELATIVE: each rebuilds its own
/// baseline inside the same binary, so a change to the draft pass's
/// arithmetic moves both arms equally and none of them reddens. AGENTS.md
/// Gotcha 51 measured that shape on `real_forward_muse.rs` -- six mutations,
/// one caught -- and the remedy is one frozen digest, the only assertion in
/// such a file compared against something computed BEFORE the change.
///
/// It is a CHANGE DETECTOR and not a correctness claim: the fixture's
/// weights are untrained, so this cannot say the arithmetic is right, only
/// that it is what it was. It covers, among others, the three things this
/// file could not otherwise see: the conv's `out_scale`, the SCALED RMS eps
/// the residual norms take (`DFLASH_RESIDUAL_EPS` -- swapping it back for
/// the plain `RMS_EPS` is invisible to every perturbation test here), and
/// the non-causal span the block's attention runs at.
///
/// Re-freezing needs a stated reason. Legitimate ones: the fixture's weights
/// change, or the draft pass's dispatch order changes in a way whose reduce
/// order legitimately moves. "The test went red" is not one.
const FROZEN_DRAFT_DIGEST: &str = "1f5f8c090997e0ed";

fn digest(values: &[LogitValue]) -> String {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    model_io::hash_data(&bytes)[..16].to_string()
}

/// Prefills, primes and drafts once, returning EVERY row's logits -- rows
/// `1..=block` because those carry the proposals, and row 0 because it is
/// the bonus row whose embedding is the anchor's and whose arithmetic no
/// proposal would otherwise cover.
fn prefill_then_draft(runner: &mut RealForwardRunner, block: usize) -> Vec<LogitValue> {
    let vocab = VOCAB as usize;
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let next = argmax(&logits);
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(next, PROMPT.len(), &mut proposals)
        .expect("the draft block runs");
    let mut row = vec![LogitValue::from_f32(0.0); vocab];
    let mut all = Vec::with_capacity((block + 1) * vocab);
    for r in 0..=block {
        runner.dflash_probe_logits(r, &mut row).expect("probe");
        all.extend_from_slice(&row);
    }
    all
}

/// Drafts at `base = PROMPT.len()` off a drafter whose cache was filled two
/// different ways, and returns the draft's row 0 logits.
///
/// `split` is where the SEQUENTIAL priming stops and the BATCHED capture
/// takes over: `PROMPT.len()` is all-sequential, and anything less leaves
/// `PROMPT.len() - split` rows to the M-row hook in `families/qwen/batched.rs`.
/// Both walks leave the drafter's cursor at `PROMPT.len()` with every row
/// written once, so the only thing that varies is WHICH code path wrote the
/// tail.
fn draft_after_capture(runner: &mut RealForwardRunner, tokens: &[i32], split: usize) -> Vec<f32> {
    let vocab = VOCAB as usize;
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in tokens.iter().take(split).enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        // The last prompt token's capture is left for the DRAFT's own context
        // write, exactly as the shipped prefill loop leaves it. In the
        // batched arm `split < tokens.len()`, so every iteration here primes
        // and the tail is the batched pass's to capture.
        if i + 1 < tokens.len() {
            runner.prime_drafter(tokens[i + 1], i).expect("prime");
        }
    }
    if split < tokens.len() {
        let rows = tokens.len() - split;
        let mut batch = vec![LogitValue::from_f32(0.0); rows * vocab];
        runner
            .verify(&tokens[split..], split, &mut batch)
            .expect("the batched pass runs");
    }
    // A FIXED anchor in both arms: the token the draft embeds must not be a
    // second thing that varies, or a difference in the logits says nothing
    // about the capture.
    const ANCHOR: i32 = 21;
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(ANCHOR, tokens.len(), &mut proposals)
        .expect("the draft block runs");
    let mut row = vec![LogitValue::from_f32(0.0); vocab];
    runner.dflash_probe_logits(0, &mut row).expect("probe");
    row.iter().map(|v| v.to_f32()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// **THE M-ROW AUX CAPTURE WRITES THE SAME ROWS THE PER-TOKEN HOOK DOES.**
///
/// `families/qwen/batched.rs` copies M residual rows into the drafter's
/// capture and calls `note_capture(start_position, batch)`; every round after
/// the first reads what it writes, and until this test nothing pinned it. The
/// failure it is aimed at is silent in the usual way: a wrong row offset or a
/// wrong stride still fills the buffer with plausible residuals, the trunk is
/// untouched so the committed stream stays byte-identical to a sequential
/// decode, and only the drafter's QUALITY moves -- which reads as a verdict
/// about DFlash2 rather than as a bug in the pass. That is the same shape as
/// the last-row-at-row-0 hazard one file over (`crates/runtime` Gotcha 19).
///
/// **A TOLERANCE RATHER THAN EQUALITY, THOUGH THIS FIXTURE MEASURES ZERO.**
/// The batched pass runs `dequant_int4_gemm_simd` where the sequential one
/// runs `dequant_int4_gemv_simd`, and on the REAL install the two accumulate
/// differently -- that is why a greedy speculative stream eventually parts
/// from a greedy sequential one (`docs/DFLASH2.md`). Here the gap comes out
/// at exactly 0.000000, the same blindness `dequant_int4_gemm_parity`'s
/// fixture has: these shapes and values sum exactly, so the fixture cannot
/// see reassociation at all. Asserting equality would therefore pin a
/// property of the FIXTURE and redden on any legitimate reduce-order change,
/// so the bound stays a tolerance and the discriminating check below is what
/// gives it teeth.
#[test]
fn the_batched_capture_writes_what_the_per_token_hook_writes() {
    let dir = build();
    let block = 2usize;
    let n = PROMPT.len();

    let mut seq_runner = open(&dir, block);
    let sequential = draft_after_capture(&mut seq_runner, &PROMPT, n);

    // The batched scratch is sized for a round's `block + 1` rows and REFUSES
    // anything wider, so that is the widest capture reachable here -- take it,
    // since a 1-row batch would not exercise the row stride at all.
    let rows = block + 1;
    let mut batched_runner = open(&dir, block);
    let batched = draft_after_capture(&mut batched_runner, &PROMPT, n - rows);

    assert!(
        sequential.iter().chain(&batched).all(|v| v.is_finite()),
        "a non-finite draft row makes every comparison below meaningless"
    );

    // THE FIXTURE HAS TO DISCRIMINATE BEFORE THE AGREEMENT MEANS ANYTHING.
    // A capture of the WRONG rows must land outside whatever tolerance the
    // right rows land inside; without this, a hook that wrote garbage of the
    // right magnitude would pass (AGENTS.md Gotchas 48 and 50).
    let mut wrong: [i32; 8] = PROMPT;
    for (k, slot) in wrong[n - rows..].iter_mut().enumerate() {
        *slot = 101 + 2 * k as i32;
    }
    let mut wrong_runner = open(&dir, block);
    let wrong_rows = draft_after_capture(&mut wrong_runner, &wrong, n - rows);

    let agree = max_abs_diff(&sequential, &batched);
    let differ = max_abs_diff(&sequential, &wrong_rows);
    println!("batched vs sequential: {agree:.6}; wrong rows vs sequential: {differ:.6}");
    assert!(
        differ > 10.0 * agree.max(1e-4),
        "the fixture cannot tell a right capture from a wrong one \
         (agree={agree:.6}, differ={differ:.6}); the assertion below proves nothing"
    );

    // The tolerance is the reduce-order gap, set an order of magnitude above
    // what the two paths measure and two orders below what a wrong capture
    // costs, so it is a real interval rather than a fitted number.
    assert!(
        agree < 0.05,
        "the M-row capture disagrees with the per-token hook by {agree}, \
         which is a wrong row or a wrong stride rather than reduce order"
    );
}

#[test]
fn the_draft_logits_have_a_frozen_digest() {
    let dir = build();
    let block = 8usize;
    let mut runner = open(&dir, block);
    let draft = prefill_then_draft(&mut runner, block);

    // Determinism first: a digest frozen over a value that moves between two
    // walks in one process is worse than no digest at all.
    let mut again_runner = open(&dir, block);
    let again = prefill_then_draft(&mut again_runner, block);
    assert_eq!(
        digest(&draft),
        digest(&again),
        "the draft pass is not deterministic across two runners on one fixture"
    );

    // Non-finite is its own failure and must not reach the digest: an FP16
    // overflow in this pass reads as NaN, and NaN hashes as stably as any
    // other bit pattern, so a digest alone would freeze a broken drafter.
    // (That is not hypothetical -- it is exactly what shipped, and what
    // `dflash_select` now refuses.)
    assert!(
        draft.iter().all(|v| v.to_f32().is_finite()),
        "the draft pass produced non-finite logits"
    );

    println!("draft digest = {}", digest(&draft));
    assert_eq!(
        digest(&draft),
        FROZEN_DRAFT_DIGEST,
        "the draft logits moved; see this test's doc comment before re-freezing"
    );
}

/// UNDER STEERING, BOTH PATHS MUST CAPTURE THE SAME RESIDUAL -- which is an
/// assertion about ORDER, and it had nothing guarding it.
///
/// The drafter's capture and the steering edit sit at the same boundary, so
/// their order decides whether the drafter sees the residual the trunk
/// actually committed or the one it would have committed unsteered. Both
/// paths steer FIRST. If one of them stopped, speculation would still be
/// lossless -- a verify rejects what it does not agree with -- so the only
/// symptom would be a fall in ACCEPTANCE, which reads as a verdict about the
/// drafter rather than as a bug in the pass.
///
/// The unsteered sibling of this case cannot see it: with the edit off, the
/// two orders are the same program. That is why this exists as its own case
/// rather than as a stronger tolerance on the one above.
#[test]
fn the_batched_capture_agrees_with_the_per_token_hook_under_steering() {
    let dir = build();
    let block = 2usize;
    let n = PROMPT.len();
    let rows = block + 1;
    let alpha = 0.35;

    let mut seq_runner = open_steered(&dir, block, alpha);
    let sequential = draft_after_capture(&mut seq_runner, &PROMPT, n);

    let mut batched_runner = open_steered(&dir, block, alpha);
    let batched = draft_after_capture(&mut batched_runner, &PROMPT, n - rows);

    assert!(
        sequential.iter().chain(&batched).all(|v| v.is_finite()),
        "a non-finite draft row makes every comparison below meaningless"
    );

    // THE EDIT HAS TO REACH THE DRAFT AT ALL, or this compares two unsteered
    // runs and would stay green with the steering hook deleted.
    let mut off_runner = open(&dir, block);
    let unsteered = draft_after_capture(&mut off_runner, &PROMPT, n);
    let moved = max_abs_diff(&sequential, &unsteered);
    let agree = max_abs_diff(&sequential, &batched);
    println!("steered batched vs sequential: {agree:.6}; steering moved the draft by {moved:.6}");
    assert!(
        moved > 10.0 * agree.max(1e-4),
        "steering did not move the draft ({moved:.6}) by enough to tell it apart from \
         the two paths' reduce-order gap ({agree:.6}), so this case proves nothing"
    );

    assert!(
        agree < 0.05,
        "the M-row capture disagrees with the per-token hook by {agree} under steering \
         while agreeing without it: the two paths apply the edit and the capture in a \
         different ORDER, so the drafter is reading a residual no committed token came from"
    );
}
