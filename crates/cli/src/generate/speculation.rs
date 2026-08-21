//! Speculative decoding resolution and drafter detection policy.

/// What [`resolve_drafter`] decided: the drafter to open, and a NOTE for the
/// case where `auto` FOUND a drafter and deliberately did not enable it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrafterChoice {
    pub(crate) drafter: invocation::SpeculativeDrafter,
    /// `Some` only for a DFlash2-carrying install under `auto`. Preferred
    /// over the engine's own blocker as the disabled reason, because that
    /// one would say "carries no multi-token-prediction head" -- true, and
    /// the wrong thing to tell someone holding a drafter this port can run.
    pub(crate) note: Option<String>,
}

/// Which drafter an install actually carries, for
/// [`invocation::SpeculativeDrafter::Auto`].
///
/// Reads the resident INDEX -- the leading index region of
/// `model_weights.bin`, kilobytes, not the weights -- and asks the same two
/// presence predicates `RealForwardRunner::open` asks. It is deliberately a
/// presence check and NOT a can-it-run check: whether the verify pass can
/// serve this install is `speculation_blocker`'s question, asked after the
/// open with the full architecture in hand, and answering it twice in two
/// places is how the two answers drift apart.
///
/// **DETECTION IS NOT ENABLEMENT, AND THAT SPLIT IS THE WHOLE POINT OF THIS
/// FUNCTION.** `auto` resolves to `Mtp` even on an install whose only
/// drafter is DFlash2, and returns a note naming the flag that would run it.
/// The measurement is why: through the shipped loop on the real install,
/// DFlash2 reads 1.47x on a code prompt and **0.88x throughput at +17.4%
/// J/token on prose** (`docs/DFLASH2.md`), so a default that switched it on
/// would make the common workload slower and hungrier without being asked.
/// The MTP head is the opposite case (1.44-1.66x) and keeps its `auto`.
/// Resolving to `Mtp` also means `open` allocates no DFlash2 state, which is
/// 213 MiB of peak footprint on the real 27B install -- measured, as the gap
/// between the two arms of the same protocol case.
///
/// An install with BOTH resolves to `Mtp`, which is what the default was
/// before `Auto` existed, so no run that worked changes. An install with
/// NEITHER also resolves to `Mtp` with no note, so the "no drafter" message
/// a user sees is the one they have always seen.
///
/// An unreadable index resolves to `Mtp` rather than failing: this function
/// picks a drafter, and `open` is entitled to be the one that refuses a
/// broken install.
pub(crate) fn resolve_drafter(
    requested: invocation::SpeculativeDrafter,
    model_dir: &std::path::Path,
) -> DrafterChoice {
    let plain = |drafter| DrafterChoice {
        drafter,
        note: None,
    };
    if requested != invocation::SpeculativeDrafter::Auto {
        return plain(requested);
    }
    let Ok(index) = model_io::load_resident_index(&model_dir.join("model_weights.bin")) else {
        return plain(invocation::SpeculativeDrafter::Mtp);
    };
    if !runtime::install_has_mtp_head(&index) && runtime::install_has_dflash(&index) {
        return DrafterChoice {
            drafter: invocation::SpeculativeDrafter::Mtp,
            note: Some(
                "this install carries a DFlash2 drafter, which auto leaves OFF: measured \
                 0.88x throughput and +17.4% J/token on prose against 1.47x on code, so it \
                 is opt-in. Pass --speculative-drafter dflash to use it"
                    .to_string(),
            ),
        };
    }
    plain(invocation::SpeculativeDrafter::Mtp)
}

/// What `open_session` decided about speculative decoding, resolved once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpeculationPlan {
    Enabled {
        block: usize,
    },
    /// `reason` is `Some` when the user might have expected otherwise, and
    /// `None` when they asked for `off` and got it.
    Disabled {
        reason: Option<String>,
    },
}

/// Decides speculation from the request and what the install turned out to
/// be. Pure, so the hard-fail/warn split is testable without a 14 GB model.
///
/// **THE TWO OUTCOMES DIFFER BY WHO ASKED.** A named block is a promise the
/// caller made to themselves -- they are measuring, or they have read
/// `docs/MTP.md` and chosen 2 -- so failing to keep it is an ERROR carrying
/// the reason. `auto` asked for "speculate if you can", so the same condition
/// is a warning and the run continues. Silently doing neither is what the
/// engine did before the head was detected at all.
///
/// The sampled case is a refusal on BOTH paths rather than a quiet downgrade
/// to greedy: acceptance is `argmax(target) == proposal`, which is exact only
/// at temperature 0. Approximating it would change what the model writes
/// while reporting success.
pub(crate) fn resolve_speculation(
    requested: invocation::Speculation,
    drafter: invocation::SpeculativeDrafter,
    engine_blocker: Option<String>,
    deterministic: bool,
) -> Result<SpeculationPlan, String> {
    let unavailable = if let Some(blocker) = engine_blocker {
        Some(blocker)
    } else if !deterministic {
        Some(
            "acceptance is exact only at --temperature 0, and this run samples; \
             sampled speculation needs rejection sampling with residual correction, \
             which is not implemented"
                .to_string(),
        )
    } else {
        None
    };

    // `auto` resolves to the drafter's own default block, which is not one
    // number: the MTP head's measured optimum is 2 (`docs/MTP.md`) and the
    // DFlash2 drafter's trained block is 8 (`docs/DFLASH2.md`).
    let auto_block = match drafter {
        invocation::SpeculativeDrafter::Dflash => runtime::DFLASH_SERVING_BLOCK,
        // `Auto` is resolved to a concrete drafter by `resolve_drafter`
        // before this is called; it shares MTP's block if one ever arrives
        // here, which is the block this engine defaulted to before DFlash2.
        _ => runtime::DEFAULT_SPECULATION_BLOCK,
    };

    match (requested, unavailable) {
        (invocation::Speculation::Off, _) => Ok(SpeculationPlan::Disabled { reason: None }),
        (invocation::Speculation::Auto, Some(reason)) => Ok(SpeculationPlan::Disabled {
            reason: Some(reason),
        }),
        (invocation::Speculation::Auto, None) => Ok(SpeculationPlan::Enabled { block: auto_block }),
        (invocation::Speculation::Block(_), Some(reason)) => Err(format!(
            "--speculative was asked for but cannot be served: {reason}"
        )),
        (invocation::Speculation::Block(n), None) => {
            Ok(SpeculationPlan::Enabled { block: n as usize })
        }
    }
}

/// The hard-fail / warn split, which is the whole contract of
/// `--speculative`. Pure, so it needs no install: the two inputs are "does
/// this install carry a head" and "is this run deterministic", and every
/// interesting combination is reachable as a pair of booleans.
#[cfg(test)]
mod speculation_policy {
    use super::{resolve_drafter, resolve_speculation, SpeculationPlan};
    use invocation::{Speculation, SpeculativeDrafter};

    /// `auto` takes its block from the drafter's OWN default, and the two
    /// differ: 2 for the MTP head (`docs/MTP.md`'s measured optimum) and 8
    /// for DFlash2 (its trained block).
    ///
    /// What this pins is the MAPPING, and it asserts the two blocks differ so
    /// the fixture can see a drafter that was never resolved. It does NOT
    /// cover the CALL SITE, which needs a runner -- and the call site is
    /// exactly where this went wrong for one build: `open_session` passed the
    /// unresolved `Auto` and the CLI printed "dflash2 (block drafter), block
    /// 2", a shape neither drafter was measured at. Only a real-model run
    /// catches that one.
    #[test]
    fn auto_takes_each_drafters_own_block() {
        let block_of = |d| match resolve_speculation(Speculation::Auto, d, None, true) {
            Ok(SpeculationPlan::Enabled { block }) => block,
            other => panic!("expected an enabled plan, got {other:?}"),
        };
        assert_eq!(
            block_of(SpeculativeDrafter::Mtp),
            runtime::DEFAULT_SPECULATION_BLOCK
        );
        assert_eq!(
            block_of(SpeculativeDrafter::Dflash),
            runtime::DFLASH_SERVING_BLOCK
        );
        // THE TWO SERVING DEFAULTS NOW AGREE AT 2, independently measured:
        // the MTP head's optimum (`docs/MTP.md`) and DFlash2's two-workload
        // sweep (`DFLASH_SERVING_BLOCK`) landed on the same number, because
        // what decides both is this engine's rollback cost rather than
        // either drafter. So this test pins the MAPPING and can NO LONGER
        // see an unresolved drafter by its block alone; `resolve_drafter`
        // below covers that, and only a real-model run covers the call
        // site. Asserted as an equality so the day they diverge is the day
        // the discriminating check can come back, rather than a day nobody
        // notices.
        assert_eq!(
            runtime::DEFAULT_SPECULATION_BLOCK,
            runtime::DFLASH_SERVING_BLOCK,
            "the serving defaults diverged; restore a discriminating assertion here"
        );
    }

    /// An EXPLICIT drafter is passed through untouched, because it is a
    /// promise the caller made to themselves: `--speculative-drafter mtp` on
    /// a DFlash2 install must reach the MTP blocker and hard-fail there,
    /// never be silently rerouted to the drafter that happens to be present.
    /// It also carries no note -- a note is `auto` explaining a choice it
    /// made, and here it made none.
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
    /// failing: this function picks a drafter, and `open` is entitled to be
    /// the one that refuses a broken install. No note either, so the message
    /// such a caller sees is the engine's own.
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
    /// reads 0.88x throughput at +17.4% J/token on prose, so `auto` must not
    /// switch it on; but silence would be the bug the feature was built to
    /// end, so the note names the flag that would.
    ///
    /// The fixture is a real synthetic install carrying `dflash.*` and no
    /// `mtp.*`, because the whole decision is a read of the resident index
    /// and a hand-made directory would not exercise it.
    #[test]
    fn auto_detects_dflash_but_leaves_it_off() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-drafter-choice-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        // Small and shallow on purpose: this reads the resident INDEX and
        // never runs a forward pass, so the fixture only has to carry the
        // tensor NAMES.
        repack::build_synthetic_qwen_gdn_dense_install_with_dflash(&dir, 512, 4, "drafter-toy", 4)
            .expect("synthetic dflash install");

        let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
            .expect("the fixture's resident index");
        // ASSERT THE FIXTURE DISCRIMINATES before believing the case: an
        // install carrying BOTH drafters, or neither, resolves to `Mtp` with
        // no note by other branches, so a fixture that was not
        // dflash-only-shaped would pass this test against any of them.
        assert!(
            runtime::install_has_dflash(&index) && !runtime::install_has_mtp_head(&index),
            "the fixture must be dflash-only or it cannot see this branch"
        );

        let choice = resolve_drafter(SpeculativeDrafter::Auto, &dir);
        assert_eq!(
            choice.drafter,
            SpeculativeDrafter::Mtp,
            "auto must not enable dflash: it is 0.88x on prose"
        );
        let note = choice.note.expect("a detected drafter must be reported");
        assert!(
            note.contains("--speculative-drafter dflash"),
            "the note must name the flag that runs it, got: {note}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An install carrying BOTH drafters takes the MTP head SILENTLY, and
    /// that silence is the assertion.
    ///
    /// This is the only input on which "no MTP head AND a DFlash2 one" and
    /// the weaker "a DFlash2 one" differ, so it is the only fixture that can
    /// see the first clause -- dropping it leaves every dflash-only case
    /// green (verified: that mutation survived until this test existed). The
    /// consequence of dropping it is not cosmetic either: the note OUTRANKS
    /// the engine's blocker, so a spurious note would report speculation off
    /// on an install whose MTP head works, silently giving up 1.44-1.66x.
    #[test]
    fn an_install_with_both_drafters_keeps_the_mtp_head_and_says_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-drafter-both-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        repack::build_synthetic_qwen_gdn_dense_install_with_both_drafters(
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
            runtime::install_has_dflash(&index) && runtime::install_has_mtp_head(&index),
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

    // Verbatim shapes of what `RealForwardRunner::speculation_blocker`
    // returns. The CLI does not construct these -- it forwards whatever the
    // engine says -- so what these fixtures pin is the ROUTING, not the text.
    const NO_HEAD: &str = "this install carries no multi-token-prediction head \
                           (mtp.fc.weight is not in the resident index)";
    const NOT_INT4: &str = "the batched verify is INT4-only and this install's \
                            ...q_proj.weight is dtype 16";
    const MOE: &str = "the batched verify is dense-only and this install routes to 128 experts";

    #[test]
    fn a_named_block_fails_hard_when_it_cannot_be_served() {
        // No head. The caller named a block, so this is an ERROR: they are
        // measuring or have chosen deliberately, and a run that quietly did
        // not speculate would be recorded as the speculative number.
        let err = resolve_speculation(
            Speculation::Block(2),
            SpeculativeDrafter::Mtp,
            Some(NO_HEAD.to_string()),
            true,
        )
        .expect_err("a named block on a headless install must fail");
        assert!(err.contains("no multi-token-prediction head"), "got: {err}");

        // Sampled. Refused rather than downgraded to greedy, which would
        // change what the model writes while reporting success.
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
                block: runtime::DEFAULT_SPECULATION_BLOCK
            }
        );
    }

    #[test]
    fn off_is_silent_even_where_speculation_would_have_worked() {
        // No warning: the caller asked for off and got off. A warning here
        // would train people to ignore the one that matters.
        //
        // THE UNSERVICEABLE CASES ARE THE DISCRIMINATING ONES. With a head
        // present and a deterministic run there is no reason to leak in the
        // first place, so a fixture built only from that combination passes
        // against an `Off` arm that forwards whatever reason it was handed.
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
}
