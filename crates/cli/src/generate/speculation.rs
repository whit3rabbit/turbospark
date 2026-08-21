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
#[path = "speculation_tests.rs"]
mod tests;
