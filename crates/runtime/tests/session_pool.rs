#![cfg(target_os = "macos")]
//! Session multiplexing (`crate::session_pool`, `--session-slots`) against a
//! real `RealForwardRunner` on a synthetic install: proves the swap/park
//! mechanics with fully controlled token sequences, isolated from anything
//! about a real checkpoint's actual tokenization or chat template.
//!
//! Untrained weights make no claim about what the logits MEAN -- these
//! tests are entirely about KV/recurrent-state bookkeeping (which session's
//! tokens end up where), not about numerics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use model_io::ExpertCacheSlots;
use turbospark_repack::{
    build_synthetic_gemma4_real_install, build_synthetic_qwen_gdn_dense_install,
};
use turbospark_runtime::{
    ChunkedPrefillRunner, DraftPolicies, LogitProducer, RealForwardRunner, SteeringPolicy,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-session-pool-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: i64 = 128;

fn build_install(dir: &std::path::Path) -> model_io::ArchConfig {
    build_synthetic_gemma4_real_install(dir, VOCAB, 2, 16, 4, 8, "tiny-gemma4-session-pool")
        .expect("real-naming install builds")
}

fn open(
    dir: &std::path::Path,
    arch: model_io::ArchConfig,
    session_slots: usize,
) -> RealForwardRunner {
    let mut runner = RealForwardRunner::open_with_slot_policy_speculation_steering_and_sessions(
        dir,
        arch,
        4096,
        ExpertCacheSlots::Fixed(16),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        session_slots,
    )
    .expect("real-naming install opens");
    runner.set_prefix_reuse(true);
    runner
}

/// Feeds `tokens` through the sequential path (every token but the last via
/// `produce_prefill`, matching `run_raw_completion`'s own split), leaving
/// the runner's KV cursor at `tokens.len()`.
fn feed(runner: &mut RealForwardRunner, tokens: &[i32]) {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner.produce(token, position, &mut logits)
        } else {
            runner.produce_prefill(token, position, &mut logits)
        }
        .expect("produce succeeds");
    }
}

/// `--session-slots 1` (the default, no pool) must be a complete no-op: a
/// shallow-then-diverging second prompt behaves exactly as it always did,
/// with no swap machinery engaged. This is the byte-identity guarantee the
/// whole feature rests on; `session_pool.capacity() == 0` guards every new
/// code path in `real_forward_traits.rs`.
#[test]
fn a_pool_of_one_is_a_complete_no_op() {
    let dir = temp_dir();
    let arch = build_install(&dir);
    let mut runner = open(&dir, arch, 1);

    let session_a: [i32; 13] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];
    runner.reset();
    feed(&mut runner, &session_a);

    // A shallow-then-diverging prompt: shares only the first two tokens with
    // A's live session, then diverges completely.
    let session_b_prompt: [i32; 13] = [100, 101, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60];
    let reused = runner.try_reuse_prefix(&session_b_prompt);
    // Without a pool, the shallow match is harmless (Part 1's shipped
    // behavior): it is reused in place, exactly as it read before session
    // pooling existed. Assert it takes the SAME rewind-in-place path rather
    // than the new pool-aware refusal, which is what "no pool" has to mean.
    assert_eq!(
        reused, 2,
        "without a pool, a shallow match is reused in place unchanged, matching Part 1's \
         shipped single-session behavior"
    );
}

/// **THE REGRESSION THIS FILE EXISTS TO PIN.** A session pool changes what a
/// small `keep` means: without one, a shallow match against an unrelated
/// prompt is harmless (the discarded tail was never going to be read again
/// regardless). With one, the SAME shallow match is destructive if allowed
/// to rewind in place, because it silently overwrites the live session's
/// real content instead of parking it -- parking only happens inside
/// `reset()`, which the in-place rewind path exists specifically to avoid
/// calling.
///
/// Session A builds a long history. Session B arrives sharing only the
/// first two tokens with A's live session (the "template boilerplate"
/// shape measured on a real install: two genuinely different conversations
/// can share a chat template's opening tokens and nothing else) and then
/// diverges completely for the rest of its own, equally long, prompt. A
/// THIRD turn continues session A's original conversation. Session A's real
/// content must have survived B's turn, parked rather than destroyed.
#[test]
fn a_shallow_match_against_the_live_session_does_not_destroy_it() {
    let dir = temp_dir();
    let arch = build_install(&dir);
    // Two slots: one live, one parked -- exactly enough to hold both
    // sessions at once with nothing evicted.
    let mut runner = open(&dir, arch, 2);

    let session_a: [i32; 13] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];
    runner.reset();
    feed(&mut runner, &session_a);

    // Session B: shares only [100, 101] with A, then diverges for the rest
    // of an equally long prompt -- more is discarded than kept, which is
    // exactly the shape the fix in `try_reuse_prefix` refuses to rewind in
    // place.
    let session_b_prompt: [i32; 13] = [100, 101, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60];
    let reused_b = runner.try_reuse_prefix(&session_b_prompt);
    assert_eq!(
        reused_b, 0,
        "a shallow match (2 of 13 tokens) that would discard more of the live session than \
         it keeps must be refused once a pool is active, not partially reused in place -- \
         refusing is what lets the caller's reset() park the live session instead of \
         overwriting it"
    );
    // The caller's own contract: `try_reuse_prefix` returning 0 means the
    // next thing a real loop does is reset().
    runner.reset();
    feed(&mut runner, &session_b_prompt);

    // Now continue session A's ORIGINAL conversation: the same 13 tokens
    // plus more, sharing A's whole recorded history.
    let session_a_continued: [i32; 15] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11, 99, 98];
    let reused_a2 = runner.try_reuse_prefix(&session_a_continued);
    assert!(
        reused_a2 >= 12,
        "session A's history should have been parked (not destroyed) when B's turn ran, and \
         found again through the pool when A's conversation continued; expected close to \
         A's full 13-token history reused, got {reused_a2}"
    );
}

/// Feeds a "prompt" through `prefill_chunk` (one call, the whole prompt as
/// one chunk) and a "reply" through `produce`, matching the SPLIT the real
/// server's chunked-prefill driver and shared `decode()` loop actually use
/// (`RealChatModel::run_completion` routes Gemma 4 through
/// `run_raw_completion_chunked_cancellable`, `crates/server/CLAUDE.md`
/// Gotcha 19) -- as opposed to `feed`'s pure sequential `produce_prefill`,
/// which the real server never takes for this family.
fn feed_chunked(runner: &mut RealForwardRunner, prompt: &[i32], reply: &[i32]) {
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .prefill_chunk(prompt, 0, &mut logits)
        .expect("prefill_chunk succeeds");
    for (i, &token) in reply.iter().enumerate() {
        runner
            .produce(token, prompt.len() + i, &mut logits)
            .expect("produce succeeds");
    }
}

/// **THE SAME REGRESSION AS ABOVE, ON THE CODE PATH THE REAL SERVER ACTUALLY
/// TAKES.** `crates/gpu/CLAUDE.md` Gotcha 4 and `crates/runtime/CLAUDE.md`
/// Gotcha 30 both record the same lesson in different shapes: a mechanism
/// wired into one entry point and not another reads as working right up
/// until the untested one is what production actually calls. This test
/// exists because the sequential-path version above passed while the
/// SERVER-level real-install version of the same scenario
/// (`crates/server/tests/real_backend.rs`'s
/// `real_backend_reuses_kv_across_two_interleaved_conversations`) did not --
/// proof that `feed`'s sequential `produce_prefill` path and this crate's
/// chunked-prefill path are NOT interchangeable for this feature, even
/// though both funnel through the SAME `try_reuse_prefix`/`reset()` trait
/// methods.
#[test]
fn a_shallow_match_does_not_destroy_a_chunked_prefill_session() {
    let dir = temp_dir();
    let arch = build_install(&dir);
    let mut runner = open(&dir, arch, 2);

    let session_a_prompt: [i32; 10] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4];
    let session_a_reply: [i32; 3] = [6, 0, 11];
    runner.reset();
    feed_chunked(&mut runner, &session_a_prompt, &session_a_reply);

    let session_b_prompt: [i32; 13] = [100, 101, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60];
    let reused_b = runner.try_reuse_prefix(&session_b_prompt);
    assert_eq!(
        reused_b, 0,
        "the shallow match must be refused on the chunked-prefill path too"
    );
    runner.reset();
    let session_b_reply: [i32; 2] = [61, 62];
    feed_chunked(&mut runner, &session_b_prompt, &session_b_reply);

    // Continue session A's original conversation: its prompt, its reply,
    // plus more -- sharing A's whole recorded history (13 tokens: 10 prompt
    // + 3 reply).
    let session_a_continued: [i32; 15] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11, 99, 98];
    let reused_a2 = runner.try_reuse_prefix(&session_a_continued);
    assert!(
        reused_a2 >= 12,
        "session A's chunked-prefill session should have been parked (not destroyed) when \
         B's turn ran; expected close to A's full 13-token history reused, got {reused_a2}"
    );
}

/// The pool holds exactly `session_slots - 1` parked slots, and eviction is
/// reported (`LogitProducer::session_slot_evicted`) only when the slot
/// actually taken had real content -- never for a freshly-allocated one
/// that has never served a request.
#[test]
fn eviction_is_reported_only_when_something_real_was_discarded() {
    let dir = temp_dir();
    let arch = build_install(&dir);
    // One live, one parked: the smallest pool that can ever evict anything.
    let mut runner = open(&dir, arch, 2);

    let session_a: [i32; 6] = [1, 2, 3, 4, 5, 6];
    runner.reset();
    feed(&mut runner, &session_a);
    // The parked slot is still the original, empty one here: promoting it
    // (via the shallow-match-refusal path below) discards nothing real.
    let session_b: [i32; 6] = [7, 8, 9, 10, 11, 12];
    assert_eq!(runner.try_reuse_prefix(&session_b), 0);
    runner.reset();
    assert!(
        !runner.session_slot_evicted(),
        "promoting a never-used parked slot must not report an eviction"
    );
    feed(&mut runner, &session_b);

    // Session A is now parked (real content). A third, unrelated session C
    // forces an eviction: the pool has no room for a third conversation.
    let session_c: [i32; 6] = [13, 14, 15, 16, 17, 18];
    assert_eq!(runner.try_reuse_prefix(&session_c), 0);
    runner.reset();
    assert!(
        runner.session_slot_evicted(),
        "the LRU slot at this point holds session A's real content (parked when B's turn \
         ran); promoting it to make room for C must report the eviction"
    );
}

fn open_qwen_gdn(dir: &std::path::Path, session_slots: usize) -> RealForwardRunner {
    build_synthetic_qwen_gdn_dense_install(dir, VOCAB, 4, "tiny-bonsai-session-pool")
        .expect("dense qwen3_5 install builds");
    let arch = turbospark_repack::peek_manifest_arch(dir)
        .expect("a dense manifest peeks against the qwen3_5 baseline");
    let mut runner = RealForwardRunner::open_with_slot_policy_speculation_steering_and_sessions(
        dir,
        arch,
        4096,
        ExpertCacheSlots::Fixed(16),
        DraftPolicies::off(),
        SteeringPolicy::off(),
        session_slots,
    )
    .expect("dense qwen3_5 install opens with a session pool");
    runner.set_prefix_reuse(true);
    runner
}

/// **THE SAME REGRESSION, ON THE FAMILY WHOSE POOL SLOT ALSO CARRIES
/// RECURRENT STATE.** Every test above runs on Gemma 4, which has no GDN
/// state at all (`SessionSlot::gdn` is `None` for it), so none of them
/// exercise `swap_live_session`'s `qwen.gdn` branch -- the one path in the
/// whole feature this codebase's own research explicitly flagged as the
/// highest-risk new code, because it is the only family where an ordinary
/// (non-pooled) shallow prefix match already refuses outright
/// (`try_reuse_prefix`'s `if self.real_qwen.is_some() { return 0; }`, for
/// the reason `crates/gpu/CLAUDE.md`'s `gdn_state.rs` entry gives: the
/// recurrent state is not invertible, so there is no cheap way to rewind it
/// -- which is exactly why a session pool swaps a second LIVE
/// `GdnStateManager` in wholesale rather than rewinding one in place).
///
/// This is the dense `qwenGdnDense` flow (Bonsai/Qwen3.8's architecture,
/// `families/qwen/`), whose fixture layer mask puts gated-DeltaNet (linear,
/// mask 2) on layers 0-2 and full attention on layer 3
/// (`real_forward_qwen35_chunked.rs`'s own comment), so `feed` below
/// advances real recurrent state on every one of the first three layers for
/// every token in both sessions.
#[test]
fn a_shallow_match_does_not_destroy_a_gdn_familys_recurrent_state() {
    let dir = temp_dir();
    let mut runner = open_qwen_gdn(&dir, 2);

    let session_a: [i32; 13] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11];
    runner.reset();
    feed(&mut runner, &session_a);

    let session_b_prompt: [i32; 13] = [100, 101, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60];
    let reused_b = runner.try_reuse_prefix(&session_b_prompt);
    // Refused either way: this family's PRE-EXISTING guard
    // (`if self.real_qwen.is_some() { return 0; }`) already refuses ANY
    // rewind unconditionally, so B1's shallow match never reaches the new
    // `back > keep` check at all. What matters here is what happens NEXT --
    // whether A1's KV and GDN state survive `reset()`'s park, together.
    assert_eq!(reused_b, 0, "a rewind is always refused on a GDN family");
    runner.reset();
    feed(&mut runner, &session_b_prompt);

    let session_a_continued: [i32; 15] = [100, 101, 5, 9, 2, 7, 1, 3, 8, 4, 6, 0, 11, 99, 98];
    let reused_a2 = runner.try_reuse_prefix(&session_a_continued);
    assert!(
        reused_a2 >= 12,
        "session A's GDN-carrying session should have been parked (not destroyed) when B's \
         turn ran; expected close to A's full 13-token history reused, got {reused_a2}"
    );
}
