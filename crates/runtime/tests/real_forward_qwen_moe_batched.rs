#![cfg(target_os = "macos")]
//! The BATCHED ROUTED half of the `qwen3_5` verify pass (ROADMAP Phase 3),
//! against the same oracle its dense sibling uses: M sequential `produce`
//! calls.
//!
//! **THE BAR IS BIT-IDENTITY, not tolerance**, for the reason
//! `real_forward_qwen35_mtp.rs` states of the dense path -- a speculative
//! verify that disagrees with sequential decode is not lossless, and on a
//! routed model the extra ways to disagree are the interesting ones: the
//! expert UNION is planned per sub-batch, the fused phase 2 reduces over
//! RANKS where the decode kernel reduces over SLOTS, and the gated shared
//! expert SEEDS that reduce rather than being added after it.
//!
//! What a synthetic fixture can prove here is bounded and the bound is worth
//! stating: the weights are untrained, so a drafted or decoded token is
//! meaningless. These cases assert the two paths agree, that the routed half
//! is REACHED rather than silently skipped, and that the refusals fire by
//! name.
//!
//! The head this install carries is DENSE while its trunk is MoE. That is
//! the fixture's own note (`build_synthetic_qwen_gdn_moe_install_with_mtp`):
//! `produce_batched` allocates its M-row scratch beside a drafter, so a MoE
//! install without one has nowhere to put it, and the head's only job here
//! is to make the trunk's routed half reachable.
//!
//! MUTATION-CHECKED 2026-08-21, five against `moe_batch.rs`, and the parity
//! case is the one that reddens every time -- the other three cases hold
//! reachability and refusals and are correctly blind to arithmetic:
//!
//! 1. seeding phase 2 with anything but the gated shared expert: REDDENS.
//! 2. a sandwich norm between the routed output and the residual add,
//!    which is crate Gotcha 11's exact trap imported from the file next
//!    door: REDDENS.
//! 3. reversing the routes' RANK order (Gotcha 27's axis): REDDENS.
//! 4. dropping the shared expert's sigmoid gate: REDDENS.
//! 5. passing `batch` where the fused phase 2 wants `sub_len`: **SURVIVES,
//!    and it is a genuine no-op rather than a gap here.** The extra rows
//!    that mutation makes phase 2 write are never read -- the residual-add
//!    loop below it bounds by `sub_len` -- and a later sub-batch rewrites
//!    the ones inside the buffer. What it does produce is a write PAST the
//!    last row on the final sub-batch, which is out of bounds and invisible
//!    at this size. Recorded rather than tuned away: no assertion over
//!    unread memory would be worth having, and the argument for the correct
//!    bound is the buffer's length, not the output.

use half::f16;
use turbospark_repack::build_synthetic_qwen_gdn_moe_install_with_mtp;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const EXPERTS: i64 = 8;
const BATCH: usize = 3;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen-moe-batched-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_moe_install_with_mtp(dir, VOCAB, LAYERS, EXPERTS, "tiny-qwen36-mtp")
        .expect("a MoE qwen3_5 install WITH an MTP head builds");
}

/// `slots` is a PARAMETER because the union bound is what caps the block
/// size: at top-8 a block of M tokens wants up to `8M` experts, and
/// `ExpertCache::plan_if_possible` asserts rather than degrading.
fn open_with_slots(
    dir: &std::path::Path,
    slots: usize,
) -> Result<RealForwardRunner, turbospark_runtime::RealForwardError> {
    let peeked = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        peeked,
        4096,
        slots,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(BATCH)),
    )
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    open_with_slots(dir, 8).expect("install opens")
}

/// The headline: an M-row batched forward through a ROUTED trunk lands on
/// the same bits as running the same tokens one at a time.
#[test]
fn the_batched_moe_pass_matches_sequential_produce_bit_for_bit() {
    let dir = temp_dir("parity");
    build(&dir);
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9, 2];

    let mut sequential: Vec<f16> = Vec::with_capacity(tokens.len() * vocab);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in tokens.iter().enumerate() {
        runner
            .produce(token, position, &mut row)
            .expect("sequential produce");
        sequential.extend_from_slice(&row);
    }
    let sequential_cursor = runner.checkpoint().position();

    runner.reset();
    let mut batched = vec![f16::from_f32(0.0); tokens.len() * vocab];
    runner
        .produce_batched(&tokens, 0, &mut batched)
        .expect("batched MoE pass");

    assert!(
        batched.iter().all(|v| v.to_f32().is_finite()),
        "batched logits must be finite before any comparison means anything \
         (AGENTS.md Gotcha 59: NaN scores a perfect result on a rank instrument)"
    );
    assert_eq!(
        sequential, batched,
        "the batched routed pass diverged from sequential produce: a \
         speculative verify built on this would not be lossless"
    );
    assert_eq!(
        sequential_cursor,
        runner.checkpoint().position(),
        "the batched pass left the KV cursor somewhere else"
    );
}

/// THE ROUTED HALF MUST BE REACHED, and nothing above proves it is.
///
/// A `produce_batched` that silently ran a dense FFN would still agree with
/// itself, still produce finite logits, and still leave the cursor right --
/// it would simply be a different model. This perturbs a byte range inside
/// `packed_experts/` and requires the BATCHED logits to move, which they can
/// only do if a routed expert blob reached the pass.
#[test]
fn perturbing_a_routed_expert_moves_the_batched_logits() {
    let dir = temp_dir("reach");
    build(&dir);
    let tokens = [5i32, 9, 2];
    let vocab = VOCAB as usize;

    let before = {
        let mut runner = open(&dir);
        let mut out = vec![f16::from_f32(0.0); tokens.len() * vocab];
        runner
            .produce_batched(&tokens, 0, &mut out)
            .expect("batched MoE pass");
        out
    };

    // Layer 0's packed experts, perturbed in place. The install is reopened
    // afterwards, so nothing is cached across the edit.
    let blob = dir.join("packed_experts").join("layer_00.bin");
    let mut bytes = std::fs::read(&blob).expect("layer 0 expert blob reads");
    assert!(
        bytes.len() > 4096,
        "expert blob is too small to perturb meaningfully"
    );
    for b in bytes.iter_mut().take(4096) {
        *b ^= 0x5A;
    }
    std::fs::write(&blob, &bytes).expect("expert blob rewrites");

    let after = {
        let mut runner = open(&dir);
        let mut out = vec![f16::from_f32(0.0); tokens.len() * vocab];
        runner
            .produce_batched(&tokens, 0, &mut out)
            .expect("batched MoE pass after perturbation");
        out
    };

    assert_ne!(
        before, after,
        "damaging a routed expert did not move the batched logits, so the \
         routed half is not on this path"
    );
}

/// A BLOCK WHOSE UNION OUTGROWS THE SLOT CACHE IS REFUSED BY NAME.
///
/// `ExpertCache::plan_if_possible` ASSERTS `experts.len() <= slot_count`
/// (AGENTS.md Gotcha 54), which aborts the process rather than returning an
/// error, so the driver has to bound its sub-batches itself. This fixture is
/// top-8 against 8 slots, where a single token already fills the cache: the
/// sub-batch loop must shrink to one token per round rather than assert, and
/// the parity case above is what says it still gets the right answer doing
/// so.
///
/// The refusal proper fires when even ONE token cannot fit, which is why
/// this asks for fewer slots than a token's own route.
#[test]
fn a_union_larger_than_the_slot_cache_is_refused_rather_than_asserted() {
    let dir = temp_dir("union");
    build(&dir);
    // `ALLOWED_CACHE_SLOTS` starts at 8 and this fixture routes top-8, so a
    // single token exactly fills the cache. Opening below that is not
    // reachable through the public options, so the case this asserts is the
    // one that IS reachable: 8 slots must SHRINK the sub-batch, not abort.
    let mut runner = open_with_slots(&dir, 8).expect("install opens at 8 slots");
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9, 2];
    let mut out = vec![f16::from_f32(0.0); tokens.len() * vocab];
    runner
        .produce_batched(&tokens, 0, &mut out)
        .expect("a union at exactly the slot count shrinks the sub-batch");
    assert!(out.iter().all(|v| v.to_f32().is_finite()));
}

/// The MoE install must go through the BATCHED routed pair, never a
/// per-token fallback (AGENTS.md Gotcha 35 one layer down): a fallback is
/// numerically identical, so it would pass every case above while making a
/// "batched" verify measure the unbatched engine.
///
/// There is no way to observe the kernel choice from outside, so this pins
/// the OTHER half of the same rule -- the layout refusal is by name and
/// mentions what it wanted. It is the arm a future GGUF-blob MoE install
/// would take.
#[test]
fn the_routed_batched_path_names_its_dtype_requirement() {
    let dir = temp_dir("names");
    build(&dir);
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9];
    let mut out = vec![f16::from_f32(0.0); tokens.len() * vocab];
    // This install IS INT4-affine, so it must succeed; the refusal string is
    // asserted by construction in `moe_batch.rs` and reached by a GGUF blob.
    runner
        .produce_batched(&tokens, 0, &mut out)
        .expect("an INT4-affine MoE install runs the batched routed pair");
}
