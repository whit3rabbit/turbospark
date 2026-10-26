#![cfg(target_os = "macos")]
//! Chunked prefill with the DFlash2 drafter open
//! (`SpeculativeProducer::prefill_chunk_priming_drafter`,
//! `Priming::Dflash` in `families/qwen/prefill.rs`) must leave the drafter
//! exactly where the token-at-a-time walk leaves it: every prompt position
//! but the last context-written into its KV from the aux captures, and the
//! last one captured alone at row 0 with the cursor on it. A capture taken
//! at the wrong row or layer, a slice written at the wrong position, or a
//! last row written early all still produce finite, plausible drafts, so
//! the oracle is the first round's draft logits (`dflash_draft_block` then
//! `dflash_probe_logits` over every query row) against the walk's, as in
//! `headless_prefill_primes_the_drafter_identically`.
//!
//! The per-token arm and the scalar batched arm are bit-exact on this INT4
//! fixture, so those comparisons are `==`. The Steel arm rounds weights to
//! FP16 operands and is held to the MTP suite's bound instead. Lengths
//! cross the 16-row context-write slices, the 64-row Steel micro-batch and
//! the 128-token chunk: at 150 tokens the Steel arm runs micro-batches of
//! 64, 64 and 22, so every row of a full-width capture reaches the draft.
//!
//! `TURBOSPARK_INT4_MMA8` is left at its default (off), and this fixture's
//! matrices sit below the tile's minimum width either way, so the scalar
//! arm's batched INT4 GEMM is the kernel both walks run. The suite is meant
//! to be run under both `TURBOSPARK_GDN_LAZY_TAPE` modes and under
//! `TURBOSPARK_DFLASH_CHUNKED_PREFILL=0`, where the support case flips and
//! the rest still drive the chunked entry point directly.

use std::sync::atomic::{AtomicU64, Ordering};

use foundation::LogitValue;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_with_both_drafters,
    build_synthetic_qwen_gdn_dense_install_with_dflash, peek_manifest_arch,
};
use turbospark_runtime::{
    DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy, RealForwardRunner,
    SpeculativeProducer,
};

/// Above the mask token (248,070) so the drafter's mask rows embed legally.
const VOCAB: i64 = 248_100;
/// 64 because the aux taps read trunk layers [5, 19, 33, 47, 61].
const LAYERS: i64 = 64;
const BLOCK: usize = 2;
const CHUNK: usize = 128;
const LENGTHS: [usize; 6] = [1, 5, 17, 40, 70, 150];
/// The first round's anchor, FIXED rather than the trunk's argmax: on the
/// Steel arm the untrained trunk's top two logits nearly tie at some
/// lengths (150 flips its argmax), and two drafts from different anchors
/// differ by the whole logit range for a reason that is not the priming.
const ANCHOR: i32 = 9;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-dflash-chunked-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn open_with_result(
    dir: &std::path::Path,
    policies: DraftPolicies,
) -> Result<RealForwardRunner, turbospark_runtime::RealForwardError> {
    let arch = peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open_with_options_and_speculation(dir, arch, 4096, 16, policies)
}

fn open_with(dir: &std::path::Path, policies: DraftPolicies) -> RealForwardRunner {
    open_with_result(dir, policies).expect("the install opens with the drafter asked for")
}

fn open_dflash(tag: &str) -> RealForwardRunner {
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_with_dflash(&dir, VOCAB, LAYERS, "dflash-chunked", 4)
        .expect("the dense install with a DFlash2 drafter writes");
    let runner = open_with(
        &dir,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(BLOCK),
        },
    );
    let _ = std::fs::remove_dir_all(&dir);
    runner
}

fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 7919 + 13) % 4096) as i32).collect()
}

/// What a prefill hands the first round: the trunk's last logits, the
/// drafter's `(cursor, capture_base, capture_rows)`, and every query row of
/// the first draft's logits.
struct Handoff {
    trunk: Vec<f32>,
    state: (usize, usize, usize),
    draft: Vec<f32>,
}

fn first_round(runner: &mut RealForwardRunner, ids: &[i32], logits: &[LogitValue]) -> Handoff {
    // `ids.len()` is the anchor's position: the walk's next token.
    let vocab = VOCAB as usize;
    let state = runner.dflash_handoff_state().expect("a drafter is open");
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(ANCHOR, ids.len(), &mut proposals)
        .expect("the first round drafts off the prefill");
    let mut row = vec![LogitValue::from_f32(0.0); vocab];
    let mut draft = Vec::with_capacity((BLOCK + 1) * vocab);
    for r in 0..=BLOCK {
        runner.dflash_probe_logits(r, &mut row).expect("probe");
        draft.extend(row.iter().map(|v| v.to_f32()));
    }
    Handoff {
        trunk: logits.iter().map(|v| v.to_f32()).collect(),
        state,
        draft,
    }
}

/// The token-at-a-time walk: every prompt token through `produce`, the
/// drafter primed after each one but the last.
fn reference(runner: &mut RealForwardRunner, ids: &[i32]) -> Handoff {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); VOCAB as usize];
    for (i, &token) in ids.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < ids.len() {
            runner.prime_drafter(ids[i + 1], i).expect("prime");
        }
    }
    first_round(runner, ids, &logits)
}

/// The same prompt through `prefill_chunk_priming_drafter` in 128-token
/// spans.
fn chunked(runner: &mut RealForwardRunner, ids: &[i32]) -> Handoff {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); VOCAB as usize];
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
            .expect("chunked prefill priming the drafter");
        start = end;
    }
    first_round(runner, ids, &logits)
}

fn assert_exact(runner: &mut RealForwardRunner, arm: &str) {
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(runner, &ids);
        let got = chunked(runner, &ids);
        assert_eq!(
            got.state,
            (n - 1, n - 1, 1),
            "{arm}, {n} tokens: the chunked prefill must end on the last row, noted alone"
        );
        assert_eq!(got.state, want.state, "{arm}, {n} tokens: handoff state");
        assert!(
            got.trunk == want.trunk,
            "{arm}, {n} tokens: trunk logits differ"
        );
        assert!(
            got.draft == want.draft,
            "{arm}, {n} tokens: first draft differs"
        );
    }
}

#[test]
fn chunked_drafter_prefill_is_offered_with_dflash_open_and_the_switch_on() {
    let runner = open_dflash("supports");
    let on = std::env::var("TURBOSPARK_DFLASH_CHUNKED_PREFILL").as_deref() != Ok("0");
    assert_eq!(runner.supports_chunked_drafter_prefill(), on);
    // The plain entry point keeps refusing an open drafter.
    assert!(!runner.supports_chunked_prefill());
}

/// An install carrying BOTH drafters opens with exactly one: two explicit
/// asks are refused at open, and with neither named the DFlash2 drafter
/// wins (`real_forward_open.rs`). So the chunked DFlash2 variant is what
/// such an install prefills through, and the predicate's `real_mtp.is_none()`
/// clause is a backstop no open can reach, not a covered branch.
#[test]
fn an_install_with_both_drafters_prefills_through_the_dflash_variant() {
    let dir = temp_dir("both");
    build_synthetic_qwen_gdn_dense_install_with_both_drafters(&dir, VOCAB, LAYERS, "both", 4)
        .expect("the dense install with both drafters writes");
    let both_named = open_with_result(
        &dir,
        DraftPolicies {
            mtp: MtpDraftPolicy::Fixed(2),
            dflash: DflashDraftPolicy::Fixed(BLOCK),
        },
    );
    assert!(
        both_named.is_err(),
        "two explicit asks must be refused at open"
    );
    let mut runner = open_with(
        &dir,
        DraftPolicies {
            mtp: MtpDraftPolicy::Auto,
            dflash: DflashDraftPolicy::Auto,
        },
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        runner.dflash_handoff_state().is_some(),
        "DFlash2 wins a bare Auto"
    );
    if std::env::var("TURBOSPARK_DFLASH_CHUNKED_PREFILL").as_deref() == Ok("0") {
        assert!(!runner.supports_chunked_drafter_prefill());
        return;
    }
    assert!(runner.supports_chunked_drafter_prefill());
    let ids = prompt(20);
    let mut logits = vec![LogitValue::from_f32(0.0); VOCAB as usize];
    runner
        .prefill_chunk_priming_drafter(&ids, 0, &mut logits, None)
        .expect("the chunked DFlash2 variant takes the prompt");
    assert_eq!(runner.dflash_handoff_state(), Some((19, 19, 1)));
}

#[test]
fn the_per_token_arm_primes_the_drafter_exactly_as_the_token_walk_does() {
    let mut runner = open_dflash("per-token");
    runner.set_batched_gemv_prefill(false);
    assert_exact(&mut runner, "per-token");
}

#[test]
fn the_scalar_batched_arm_primes_the_drafter_exactly_too() {
    let mut runner = open_dflash("scalar-batched");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(false);
    assert_exact(&mut runner, "scalar batched");
}

#[test]
fn the_steel_arm_primes_the_drafter_within_the_bound() {
    let mut runner = open_dflash("steel");
    runner.set_batched_gemv_prefill(true);
    runner.set_prefill_steel(true);
    for n in LENGTHS {
        let ids = prompt(n);
        let want = reference(&mut runner, &ids);
        let got = chunked(&mut runner, &ids);
        assert_eq!(got.state, want.state, "{n} tokens: handoff state");
        for (label, g, w) in [
            ("trunk", &got.trunk, &want.trunk),
            ("draft", &got.draft, &want.draft),
        ] {
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
