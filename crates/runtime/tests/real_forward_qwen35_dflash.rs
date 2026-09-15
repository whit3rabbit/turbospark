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
        turbospark_runtime::SteeringPolicy::single(
            model_io::SteeringSet {
                layers,
                hidden,
                declared_mode: None,
                declared_arch: None,
            },
            foundation::SteeringMode::Ablate,
            alpha,
            0.0,
            0.0,
        ),
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
        // The shipped loop's retaining shape: the kept rows' KV stays and
        // the GDN state replays over the tape, so no forward pass runs and
        // `batch_logits` keeps the original verify's rows (identical to
        // what the old rollback-plus-re-verify rebuilt -- that identity is
        // `retaining_rollback_matches_rollback_plus_reverify`'s claim).
        runner
            .rollback_retaining(&point, accepted + 1)
            .expect("the retaining rollback replays the tape");
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

/// One PARTIAL round, run to a deterministic `keep_rows == 2`, with the
/// rollback half chosen by `retain`: `false` is the old shape (full
/// rollback plus a shortened re-verify), `true` is the shipped loop's
/// retaining shape (KV kept, GDN state replayed over the tape).
///
/// Partial acceptance is CRAFTED rather than hoped for, because the
/// fixture's weights are untrained and a drafter-off feed would accept
/// nothing: a throwaway verify learns the target's row argmaxes, a full
/// rollback undoes it, and the real feed's first proposal is that learned
/// argmax (accepted) while its second is one past the learned next argmax
/// (rejected). Row argmaxes are causal, so the learning pass's rows 0 and 1
/// are exactly the real pass's.
fn run_partial_round(
    runner: &mut RealForwardRunner,
    retain: bool,
) -> (Vec<i32>, Vec<LogitValue>, i32, String, String) {
    let vocab = VOCAB as usize;
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let next = argmax(&logits);
    let base = PROMPT.len();

    // The learning pass, undone by a full rollback before anything is kept.
    let probe = [next, 13, 21];
    let mut learn = vec![LogitValue::from_f32(0.0); 3 * vocab];
    let point = runner.checkpoint();
    runner
        .verify(&probe, base, &mut learn)
        .expect("learn verify");
    let a0 = argmax(&learn[0..vocab]);
    let a1 = argmax(&learn[vocab..2 * vocab]);
    runner.rollback(&point);

    // The real feed: one accepted proposal, one rejected.
    let rejected = ((a1 as usize + 1) % vocab) as i32;
    let feed = vec![next, a0, rejected];
    let mut batch_logits = vec![LogitValue::from_f32(0.0); 3 * vocab];
    let point = runner.checkpoint();
    runner
        .verify(&feed, base, &mut batch_logits)
        .expect("the batched verify runs");
    assert_eq!(
        argmax(&batch_logits[0..vocab]),
        a0,
        "the crafted first proposal must be accepted"
    );
    assert_ne!(
        argmax(&batch_logits[vocab..2 * vocab]),
        rejected,
        "the crafted second proposal must be rejected"
    );

    let keep = 2usize;
    if retain {
        runner
            .rollback_retaining(&point, keep)
            .expect("the retaining rollback replays the tape");
    } else {
        runner.rollback(&point);
        let mut replay = vec![LogitValue::from_f32(0.0); keep * vocab];
        runner
            .verify(&feed[..keep], base, &mut replay)
            .expect("the shortened replay verify runs");
        batch_logits[..keep * vocab].copy_from_slice(&replay);
    }

    // The bonus comes from row `keep - 1` -- the last kept row -- and both
    // arms then feed it through the target at the same position, whose
    // logits are the observable proxy for the whole engine state (KV rows,
    // GDN state, conv tails, cursor) the two rollback shapes leave behind.
    let bonus = argmax(&batch_logits[(keep - 1) * vocab..keep * vocab]);
    let kept_rows = digest(&batch_logits[..keep * vocab]);
    let mut after = vec![LogitValue::from_f32(0.0); vocab];
    runner
        .produce(bonus, base + keep, &mut after)
        .expect("produce after rollback");
    (feed, batch_logits, bonus, kept_rows, digest(&after))
}

/// The central claim of the tape rollback, stated as bytes: keeping the
/// accepted rows' KV and replaying their recurrent state over the recorded
/// inputs leaves the same engine state that rolling all the way back and
/// re-verifying does. Both shapes run here on two fresh opens of the same
/// install, from identical prefills, over the same forced-partial round.
#[test]
fn retaining_rollback_matches_rollback_plus_reverify() {
    let dir = build();
    let mut legacy = open(&dir, 2);
    let (feed_a, _, bonus_a, kept_a, after_a) = run_partial_round(&mut legacy, false);
    let mut retained = open(&dir, 2);
    let (feed_b, _, bonus_b, kept_b, after_b) = run_partial_round(&mut retained, true);

    assert_eq!(feed_a, feed_b, "the arms must run the same round");
    assert_eq!(bonus_a, bonus_b, "the bonus must not move");
    assert_eq!(
        kept_a, kept_b,
        "the kept rows' logits differ from what the re-verify rebuilt"
    );
    assert_eq!(
        after_a, after_b,
        "the state after the two rollback shapes diverged"
    );
}

/// The same identity at a SHRUNK round: opened at block 3 (the scratch is
/// `block + 1` = 4 rows) but fed 3, which is what `speculative.rs`'s
/// `round_block` shrink produces near the token budget or the context
/// limit. The tape's recorded stride is then the 3 rows the pass actually
/// fed, not the scratch's 4-row capacity, so a replay that used the
/// capacity would read every linear layer past slot 0 from the wrong byte
/// offset -- silent recurrent-state corruption the losslessness gate cannot
/// see, because the committed stream is identical and only the drafter's
/// next-round state moves. Every test above feeds at full scratch width and
/// would survive that bug; this one does not.
#[test]
fn retaining_rollback_matches_at_a_shrunk_round() {
    let dir = build();
    let mut legacy = open(&dir, 3);
    let (feed_a, _, bonus_a, kept_a, after_a) = run_partial_round(&mut legacy, false);
    let mut retained = open(&dir, 3);
    let (feed_b, _, bonus_b, kept_b, after_b) = run_partial_round(&mut retained, true);

    assert_eq!(feed_a, feed_b, "the arms must run the same round");
    assert_eq!(bonus_a, bonus_b, "the bonus must not move");
    assert_eq!(
        kept_a, kept_b,
        "the kept rows' logits differ from what the re-verify rebuilt"
    );
    assert_eq!(
        after_a, after_b,
        "the state after the two rollback shapes diverged at a shrunk round"
    );
}

/// A tape describes exactly one verify. A retaining rollback with no verify
/// behind it, and one whose tape a later produce consumed, are both REFUSED
/// by name rather than served from stale state.
#[test]
fn a_stale_tape_refuses_the_retaining_rollback() {
    let dir = build();
    let vocab = VOCAB as usize;
    let mut runner = open(&dir, 2);
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let next = argmax(&logits);
    let base = PROMPT.len();

    // No verify has run: nothing to replay over.
    let point = runner.checkpoint();
    let err = runner
        .rollback_retaining(&point, 1)
        .expect_err("an empty tape must refuse");
    assert!(err.contains("tape"), "wrong refusal: {err}");

    // A verify records, the next produce consumes the tape, and a second
    // ask against the pre-produce checkpoint refuses again.
    let feed = [next, 13];
    let mut batch = vec![LogitValue::from_f32(0.0); 2 * vocab];
    runner.verify(&feed, base, &mut batch).expect("verify");
    let point = runner.checkpoint();
    runner
        .produce(13, base + feed.len(), &mut logits)
        .expect("produce");
    let err = runner
        .rollback_retaining(&point, 1)
        .expect_err("a consumed tape must refuse");
    assert!(err.contains("tape"), "wrong refusal: {err}");
}

/// The speculative loop's reuse contract, at the runner level: a
/// continuation of the just-generated stream reuses the whole fed prefix,
/// and a trunk that has run PAST the drafter's one-row bonus lag refuses.
///
/// The lag is the mechanism, not a tolerance to tune: the loop rewinds the
/// drafter to the last accepted proposal, and the bonus row's context write
/// happens at the NEXT round's start from the captures the last verify left
/// pending -- so a lag of one self-heals and a lag of more is a hole the
/// drafter would attend over.
#[test]
fn speculative_reuse_fires_on_pure_continuation_and_refuses_a_lagging_drafter() {
    let dir = build();
    let vocab = VOCAB as usize;
    let mut runner = open(&dir, 2);
    runner.set_prefix_reuse(true);

    // Generation 1: prefill + prime, then two manual rounds, recording the
    // committed stream exactly as the loop's bookkeeping would.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for (i, &token) in PROMPT.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < PROMPT.len() {
            runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let mut next = argmax(&logits);
    let mut stream: Vec<i32> = Vec::new();
    for _ in 0..2 {
        stream.push(next);
        let base = PROMPT.len() + stream.len() - 1;
        let (accepted, bonus) = round(&mut runner, next, base, 2, vocab);
        // The loop's per-round record: the confirmed token plus the
        // accepted proposals, which is exactly the feed prefix that
        // survived.
        let mut committed_rows = vec![next];
        committed_rows.extend_from_slice(&accepted);
        runner.record_committed(&committed_rows, base);
        stream.extend_from_slice(&accepted);
        next = bonus;
    }

    // The pure continuation: every fed id, plus one token left to feed.
    let mut prompt2 = PROMPT.to_vec();
    prompt2.extend_from_slice(&stream);
    prompt2.push(42);
    let keep = runner.try_reuse_prefix(&prompt2);
    assert_eq!(
        keep,
        PROMPT.len() + stream.len(),
        "a pure continuation must reuse the whole fed prefix"
    );

    // Two plain steps put the trunk three rows past the drafter's cursor:
    // more than the bonus lag, so the guard refuses and the caller resets.
    runner
        .produce(42, PROMPT.len() + stream.len(), &mut logits)
        .expect("produce");
    runner
        .produce(43, PROMPT.len() + stream.len() + 1, &mut logits)
        .expect("produce");
    let mut prompt3 = PROMPT.to_vec();
    prompt3.extend_from_slice(&stream);
    prompt3.extend_from_slice(&[42, 43]);
    prompt3.push(7);
    let keep = runner.try_reuse_prefix(&prompt3);
    assert_eq!(keep, 0, "a lagging drafter must refuse reuse");
}

/// THE HEADLESS PREFILL CLAIM, as bytes: a speculative prefill that skips
/// the output head on every token but the last primes the drafter
/// bit-identically to the headful walk -- the DFlash2 taps are raw layer
/// outputs, captured per layer ahead of the skip early return, so the skip
/// changes nothing they see. Two fresh opens, two walks, one digest. This
/// is the measurement the loop's old comment said nobody had taken.
#[test]
fn headless_prefill_primes_the_drafter_identically() {
    let dir = build();

    fn walk(runner: &mut RealForwardRunner, headless: bool) -> (i32, String) {
        let vocab = VOCAB as usize;
        let mut logits = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &token) in PROMPT.iter().enumerate() {
            let last = i + 1 == PROMPT.len();
            if last || !headless {
                runner.produce(token, i, &mut logits).expect("produce");
            } else {
                runner
                    .produce_prefill(token, i, &mut logits)
                    .expect("produce_prefill");
            }
            if i + 1 < PROMPT.len() {
                runner.prime_drafter(PROMPT[i + 1], i).expect("prime");
            }
        }
        let next = argmax(&logits);
        // The first round's draft logits: what the priming actually fed.
        let mut proposals = Vec::new();
        runner
            .dflash_draft_block(next, PROMPT.len(), &mut proposals)
            .expect("draft");
        let mut row = vec![LogitValue::from_f32(0.0); vocab];
        let mut all = Vec::new();
        for r in 0..3 {
            runner.dflash_probe_logits(r, &mut row).expect("probe");
            all.extend_from_slice(&row);
        }
        (next, digest(&all))
    }

    let mut headful = open(&dir, 2);
    let (next_a, digest_a) = walk(&mut headful, false);
    let mut headless = open(&dir, 2);
    let (next_b, digest_b) = walk(&mut headless, true);
    assert_eq!(next_a, next_b, "the first sampled token moved");
    assert_eq!(
        digest_a, digest_b,
        "the draft logits moved under the headless prefill"
    );
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
///
/// **DEVICE-BRANCHED, since 2026-09-04**; see `real_forward_muse.rs`'s
/// identical fix for the full account of why a bit-exact hash of
/// GPU-computed FP16 values does not reproduce on CI's virtualized
/// `macos-latest` runner. On real Apple Silicon
/// `the_draft_logits_have_a_frozen_digest` runs this ORIGINAL exact digest
/// comparison, unchanged from before this fix and with its full original
/// sensitivity -- this is also where the design was proven necessary: a
/// pure tolerance replacement (no device branch) was tried first and a
/// mutation-check against `DFLASH_RESIDUAL_EPS` (this doc's own third
/// example above) moved 181 of 257 sampled logits by only 0.002-0.012,
/// under any tolerance loose enough to absorb cross-hardware noise. So a
/// coarse tolerance is not a safe universal substitute here, and the fix is
/// the same real-hardware/virtualized split every other frozen-digest file
/// in this crate now uses. Only a virtualized device falls back to
/// `FROZEN_DRAFT_SAMPLES` below, a tolerant comparison over a STRIDED
/// SAMPLE of the full `(block + 1) * VOCAB` logit array (2,232,900 values
/// at block 8, far too many to embed as a literal): every 8722nd element,
/// chosen for a small gcd with `VOCAB` so the stride does not alias onto a
/// periodic subset, 257 samples in total, captured from the same
/// real-hardware run this digest reproduces on.
const FROZEN_DRAFT_DIGEST: &str = "1f5f8c090997e0ed";

/// A strided sample of the frozen draft, one value every 8722 elements
/// (see `FROZEN_DRAFT_DIGEST`'s doc). Only consulted on a virtualized
/// device; the real-hardware branch compares the full array's digest.
#[rustfmt::skip]
const FROZEN_DRAFT_SAMPLES: [f32; 257] = [
    -8.65625, 3.7753906, 2.8417969, -3.5898438, 6.3203125, -0.8574219, -0.4970703, 2.3691406,
    -2.5234375, -3.4726563, 6.625, -5.6328125, -2.3867188, 4.125, -4.84375, 5.359375,
    6.484375, 1.8251953, -1.0869141, -0.6142578, 2.7753906, 1.7529297, -9.3671875, -1.9794922,
    0.108947754, -1.9335938, 1.3681641, 1.8330078, -6.2929688, -6.4492188, -6.15625, -3.0957031,
    -1.96875, -7.2773438, -12.703125, -10.28125, -3.5859375, -0.4169922, -2.0957031, 8.515625,
    -6.53125, -6.5351563, -0.52783203, -1.0742188, 5.8125, 2.9550781, -6.2890625, -4.2070313,
    -7.59375, -0.2475586, 1.7207031, 13.625, -0.5385742, 1.0996094, 3.2539063, 14.1015625,
    0.5698242, -5.9414063, -8.4140625, 13.890625, -17.359375, 3.4941406, 3.1484375, -4.3320313,
    -1.3261719, 5.4648438, 1.3710938, 6.5039063, -19.890625, 5.7734375, -12.421875, 0.10656738,
    -7.3242188, -3.4472656, 1.4619141, 8.7890625, 2.5898438, -5.0625, -17.6875, -16.609375,
    5.6796875, -1.1953125, -6.6835938, 1.2373047, -0.16455078, -11.71875, 2.9238281, -7.2382813,
    -3.6464844, -3.6503906, -7.3945313, -3.0488281, 1.0761719, 12.578125, -2.6230469, -1.4589844,
    7.6445313, 7.2578125, -5.1289063, 11.6328125, -11.515625, -7.1992188, 14.859375, 3.5390625,
    8.7890625, -6.1679688, -6.4570313, -7.8632813, 4.8945313, 8.6796875, -1.5576172, 0.0045204163,
    -10.953125, -1.3857422, 2.7519531, -4.2617188, 17.484375, -3.8476563, 14.140625, 2.5527344,
    -12.140625, 13.484375, -0.038604736, 9.8046875, 3.1972656, -5.0039063, -2.0078125, -10.8125,
    -3.484375, 0.54052734, -8.078125, 2.4707031, -3.9726563, 9.65625, -4.0703125, 4.8359375,
    -4.9570313, -0.70703125, 7.4453125, -3.0742188, -1.3125, -6.125, -0.2861328, 4.2382813,
    3.0058594, -3.5039063, 5.4257813, 5.8476563, -3.3710938, 2.5390625, -8.3125, -0.20751953,
    4.8007813, -2.5332031, -6.3359375, -6.7851563, 8.6640625, -1.8613281, -6.9453125, -0.37670898,
    -6.7265625, 4.6835938, -8.2421875, -6.765625, 13.296875, -5.5039063, -4.1914063, 11.015625,
    7.5859375, 3.2050781, -7.625, 6.3984375, -3.8027344, 5.78125, -4.1757813, -6.0546875,
    5.9921875, -8.40625, 0.57373047, -2.8554688, -11.0625, 11.03125, -0.57421875, 9.5234375,
    4.3632813, 8.734375, -15.21875, -16.46875, -2.2050781, 4.4453125, 3.2773438, 5.3164063,
    6.3945313, -4.8984375, 7.328125, -9.453125, 7.5703125, 3.3085938, -2.265625, 5.8164063,
    5.9765625, -1.8339844, 4.171875, 2.8652344, 1.4980469, -8.4453125, -7.0039063, -13.0703125,
    -9.4375, 4.0585938, 0.8725586, 5.2109375, -10.1640625, -8.1015625, -3.0859375, -7.9804688,
    -7.015625, -9.0703125, 8.21875, -3.6230469, -3.4101563, 3.5351563, 0.13000488, -14.7109375,
    -5.875, -7.2460938, -0.14038086, -1.2460938, -1.0693359, 0.51171875, -2.0234375, -0.88623047,
    -7.3242188, -4.921875, 12.28125, -4.2617188, -0.23449707, 8.9375, -7.9414063, 2.3242188,
    4.7578125, 8.2890625, -1.8417969, 10.8046875, 6.3203125, 5.5, -4.0234375, -0.20117188,
    3.3339844, 4.4648438, -5.15625, -12.125, 9.9921875, 9.8515625, -12.6171875, -8.5078125,
    3.7109375,
];

const SAMPLE_STRIDE: usize = 8722;

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
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this digest was \
             taken on; comparing a strided sample against the frozen reference with a \
             tolerance instead"
        );
        let sampled: Vec<f32> = draft
            .iter()
            .step_by(SAMPLE_STRIDE)
            .map(|v| v.to_f32())
            .collect();
        assert_eq!(sampled.len(), FROZEN_DRAFT_SAMPLES.len());
        for (i, (got, &want)) in sampled.iter().zip(FROZEN_DRAFT_SAMPLES.iter()).enumerate() {
            let diff = (got - want).abs();
            let tol = 0.02_f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "sample {i} (logit {}): the draft logits moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol}); see FROZEN_DRAFT_DIGEST's doc before \
                 re-freezing",
                i * SAMPLE_STRIDE
            );
        }
        return;
    }
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
