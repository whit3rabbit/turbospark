//! The hard-fail / warn split, which is the whole contract of
//! `--speculative`. Pure, so it needs no install: the two inputs are "does
//! this install carry a head" and "is this run deterministic", and every
//! interesting combination is reachable as a pair of booleans.
//!
//! Moved here with the policy itself, from `crates/cli`. They were the CLI's
//! only coverage of these decisions and the server was about to need the same
//! decisions; leaving the tests behind would have left the second caller
//! untested by construction.

use super::{
    resolve_drafter, resolve_speculation, DrafterChoice, Speculation, SpeculationPlan,
    SpeculativeDrafter,
};
use crate::families::qwen::{install_has_dflash, install_has_mtp_head, DFLASH_SERVING_BLOCK};
use crate::speculative::DEFAULT_SPECULATION_BLOCK;

/// `auto` takes its block from the drafter's OWN default.
///
/// What this pins is the MAPPING. It does NOT cover the CALL SITE, which
/// needs a runner -- and the call site is exactly where this went wrong for
/// one build: `open_session` passed the unresolved `Auto` and the CLI printed
/// "dflash2 (block drafter), block 2", a shape neither drafter was measured
/// at. Only a real-model run catches that one.
#[test]
fn auto_takes_each_drafters_own_block() {
    let block_of = |d| match resolve_speculation(Speculation::Auto, d, None, true) {
        Ok(SpeculationPlan::Enabled { block }) => block,
        other => panic!("expected an enabled plan, got {other:?}"),
    };
    assert_eq!(block_of(SpeculativeDrafter::Mtp), DEFAULT_SPECULATION_BLOCK);
    assert_eq!(block_of(SpeculativeDrafter::Dflash), DFLASH_SERVING_BLOCK);
    // THE TWO SERVING DEFAULTS AGREE AT 2, independently measured: the MTP
    // head's optimum (`docs/MTP.md`) and DFlash2's three-workload sweep
    // (`DFLASH_SERVING_BLOCK`) landed on the same number, because what decides
    // both is this engine's rollback cost rather than either drafter. So this
    // test pins the MAPPING and can NO LONGER see an unresolved drafter by its
    // block alone; `resolve_drafter` below covers that, and only a real-model
    // run covers the call site. Asserted as an equality so the day they
    // diverge is the day the discriminating check can come back, rather than a
    // day nobody notices.
    assert_eq!(
        DEFAULT_SPECULATION_BLOCK, DFLASH_SERVING_BLOCK,
        "the serving defaults diverged; restore a discriminating assertion here"
    );
}

/// An EXPLICIT drafter is passed through untouched, because it is a promise
/// the caller made to themselves: `--speculative-drafter mtp` on a DFlash2
/// install must reach the MTP blocker and hard-fail there, never be silently
/// rerouted to the drafter that happens to be present. It also carries no
/// note -- a note is `auto` explaining a choice it made, and here it made
/// none.
#[test]
fn an_explicit_drafter_is_never_re_resolved() {
    let missing = std::path::Path::new("/nonexistent/turbospark/install");
    for named in [SpeculativeDrafter::Mtp, SpeculativeDrafter::Dflash] {
        let choice = resolve_drafter(named, missing);
        assert_eq!(choice.drafter, named);
        assert_eq!(choice.note, None, "an explicit ask needs no explanation");
    }
}

/// An install whose index cannot be read resolves to `Mtp` rather than
/// failing: this function picks a drafter, and `open` is entitled to be the
/// one that refuses a broken install. No note either, so the message such a
/// caller sees is the engine's own.
#[test]
fn an_unreadable_index_resolves_to_the_pre_existing_default() {
    let choice = resolve_drafter(
        SpeculativeDrafter::Auto,
        std::path::Path::new("/nonexistent/turbospark/install"),
    );
    assert_eq!(choice.drafter, SpeculativeDrafter::Mtp);
    assert_eq!(choice.note, None);
}

/// **DFLASH2 IS DETECTED AND DELIBERATELY NOT ENABLED**, which is the
/// asymmetry this whole function exists for. Through the shipped loop it
/// reads 0.96x throughput at +17.4% J/token on prose, so `auto` must not
/// switch it on; but silence would be the bug the feature was built to end,
/// so the note names the flag that would.
///
/// The fixture is a real synthetic install carrying `dflash.*` and no `mtp.*`,
/// because the whole decision is a read of the resident index and a hand-made
/// directory would not exercise it.
#[test]
fn auto_detects_dflash_but_leaves_it_off() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-drafter-choice-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    // Small and shallow on purpose: this reads the resident INDEX and never
    // runs a forward pass, so the fixture only has to carry the tensor NAMES.
    turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_dflash(
        &dir,
        512,
        4,
        "drafter-toy",
        4,
    )
    .expect("synthetic dflash install");

    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the fixture's resident index");
    // ASSERT THE FIXTURE DISCRIMINATES before believing the case: an install
    // carrying BOTH drafters, or neither, resolves to `Mtp` with no note by
    // other branches, so a fixture that was not dflash-only-shaped would pass
    // this test against any of them.
    assert!(
        install_has_dflash(&index) && !install_has_mtp_head(&index),
        "the fixture must be dflash-only or it cannot see this branch"
    );

    let choice = resolve_drafter(SpeculativeDrafter::Auto, &dir);
    assert_eq!(
        choice.drafter,
        SpeculativeDrafter::Mtp,
        "auto must not enable dflash: it is 0.96x on prose"
    );
    let note = choice.note.expect("a detected drafter must be reported");
    assert!(
        note.contains("--speculative-drafter dflash"),
        "the note must name the flag that runs it, got: {note}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// An install carrying BOTH drafters takes the MTP head SILENTLY, and that
/// silence is the assertion.
///
/// This is the only input on which "no MTP head AND a DFlash2 one" and the
/// weaker "a DFlash2 one" differ, so it is the only fixture that can see the
/// first clause -- dropping it leaves every dflash-only case green (verified:
/// that mutation survived until this test existed). The consequence of
/// dropping it is not cosmetic either: the note OUTRANKS the engine's
/// blocker, so a spurious note would report speculation off on an install
/// whose MTP head works, silently giving up 1.44-1.66x.
#[test]
fn an_install_with_both_drafters_keeps_the_mtp_head_and_says_nothing() {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-drafter-both-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_both_drafters(
        &dir,
        512,
        4,
        "drafter-toy-both",
        4,
    )
    .expect("synthetic install with both drafters");

    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .expect("the fixture's resident index");
    assert!(
        install_has_dflash(&index) && install_has_mtp_head(&index),
        "the fixture must carry BOTH or it cannot see this branch"
    );

    let choice = resolve_drafter(SpeculativeDrafter::Auto, &dir);
    assert_eq!(choice.drafter, SpeculativeDrafter::Mtp);
    assert_eq!(
        choice.note, None,
        "the MTP head is being used, so there is nothing to explain"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// Verbatim shapes of what `RealForwardRunner::speculation_blocker` returns.
// This module does not construct these -- it forwards whatever the engine
// says -- so what these fixtures pin is the ROUTING, not the text.
const NO_HEAD: &str = "this install carries no multi-token-prediction head \
                       (mtp.fc.weight is not in the resident index)";
const NOT_INT4: &str = "the batched verify is INT4-only and this install's \
                        ...q_proj.weight is dtype 16";
const MOE: &str = "the batched verify is dense-only and this install routes to 128 experts";

#[test]
fn a_named_block_fails_hard_when_it_cannot_be_served() {
    // No head. The caller named a block, so this is an ERROR: they are
    // measuring or have chosen deliberately, and a run that quietly did not
    // speculate would be recorded as the speculative number.
    let err = resolve_speculation(
        Speculation::Block(2),
        SpeculativeDrafter::Mtp,
        Some(NO_HEAD.to_string()),
        true,
    )
    .expect_err("a named block on a headless install must fail");
    assert!(err.contains("no multi-token-prediction head"), "got: {err}");

    // Sampled. Refused rather than downgraded to greedy, which would change
    // what the model writes while reporting success.
    let err = resolve_speculation(Speculation::Block(2), SpeculativeDrafter::Mtp, None, false)
        .expect_err("a named block on a sampled run must fail");
    assert!(err.contains("temperature 0"), "got: {err}");
}

#[test]
fn auto_warns_and_continues_where_a_named_block_fails() {
    let cases = [
        (Some(NO_HEAD.to_string()), true),
        (None, false),
        (Some(NOT_INT4.to_string()), true),
        (Some(MOE.to_string()), false),
    ];
    for (blocker, deterministic) in cases {
        let plan = resolve_speculation(
            Speculation::Auto,
            SpeculativeDrafter::Mtp,
            blocker,
            deterministic,
        )
        .expect("auto never fails; it declines");
        match plan {
            SpeculationPlan::Disabled { reason: Some(_) } => {}
            other => panic!("auto must warn and continue, got {other:?}"),
        }
    }
}

#[test]
fn auto_speculates_when_the_install_and_the_settings_allow_it() {
    let plan = resolve_speculation(Speculation::Auto, SpeculativeDrafter::Mtp, None, true)
        .expect("serviceable");
    assert_eq!(
        plan,
        SpeculationPlan::Enabled {
            block: DEFAULT_SPECULATION_BLOCK
        }
    );
}

#[test]
fn off_is_silent_even_where_speculation_would_have_worked() {
    // No warning: the caller asked for off and got off. A warning here would
    // train people to ignore the one that matters.
    //
    // THE UNSERVICEABLE CASES ARE THE DISCRIMINATING ONES. With a head present
    // and a deterministic run there is no reason to leak in the first place,
    // so a fixture built only from that combination passes against an `Off`
    // arm that forwards whatever reason it was handed.
    for (blocker, deterministic) in [
        (None, true),
        (Some(NO_HEAD.to_string()), true),
        (None, false),
        (Some(NOT_INT4.to_string()), false),
    ] {
        let plan = resolve_speculation(
            Speculation::Off,
            SpeculativeDrafter::Mtp,
            blocker.clone(),
            deterministic,
        )
        .expect("off never fails");
        assert_eq!(
            plan,
            SpeculationPlan::Disabled { reason: None },
            "off must stay silent at blocker={blocker:?} deterministic={deterministic}"
        );
    }
}

#[test]
fn a_named_block_is_honoured_rather_than_replaced_by_the_default() {
    let plan = resolve_speculation(Speculation::Block(7), SpeculativeDrafter::Mtp, None, true)
        .expect("serviceable");
    assert_eq!(plan, SpeculationPlan::Enabled { block: 7 });
}

/// **THE `note` ARM PINS `MtpDraftPolicy::Off`, AND THAT IS NOT COSMETIC.**
/// A DFlash2-only install under `auto` must OPEN successfully so
/// `resolve_speculation` can refuse with the note that names the flag;
/// without this arm the open itself fails, telling someone holding a working
/// drafter to go and download a different one.
///
/// This is the one decision in this module that had no test at all while it
/// lived in the CLI -- `draft_policies` was an inline `match` inside
/// `open_session`, reachable only with a 14 GB install in hand.
#[test]
fn a_noted_dflash_install_opens_with_both_drafters_off() {
    use crate::families::qwen::{DflashDraftPolicy, MtpDraftPolicy};

    let noted = DrafterChoice {
        drafter: SpeculativeDrafter::Mtp,
        note: Some("carries a DFlash2 drafter".to_string()),
    };
    for asked in [Speculation::Auto, Speculation::Block(2)] {
        let policies = super::draft_policies(&noted, asked);
        assert_eq!(
            policies.mtp,
            MtpDraftPolicy::Off,
            "a noted install must not ask the open for a head it has not got"
        );
    }

    // The contrast that makes the arm above discriminating: the SAME drafter
    // with no note does ask for the head, so a mutation deleting the note
    // clause changes this pair rather than nothing.
    let plain = DrafterChoice {
        drafter: SpeculativeDrafter::Mtp,
        note: None,
    };
    assert_eq!(
        super::draft_policies(&plain, Speculation::Block(2)).mtp,
        MtpDraftPolicy::Fixed(2)
    );
    assert_eq!(
        super::draft_policies(&plain, Speculation::Auto).mtp,
        MtpDraftPolicy::Auto
    );

    // And the DFlash2 arm pins the MTP head off in every case, so an install
    // carrying both never opens both drafters' state.
    let dflash = DrafterChoice {
        drafter: SpeculativeDrafter::Dflash,
        note: None,
    };
    let policies = super::draft_policies(&dflash, Speculation::Block(4));
    assert_eq!(policies.mtp, MtpDraftPolicy::Off);
    assert_eq!(policies.dflash, DflashDraftPolicy::Fixed(4));
}
