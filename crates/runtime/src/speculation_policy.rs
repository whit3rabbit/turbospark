//! Which drafter an install carries, whether this run may use it, and the
//! [`DraftPolicies`] to open with.
//!
//! **THE POLICY LIVES HERE RATHER THAN IN A BINARY, because two front ends
//! need it and a second copy names the wrong cause the first time they
//! disagree.** It was `crates/cli/src/generate/speculation.rs`, a private
//! module of a crate with two binaries and no lib target
//! (`crates/cli/CLAUDE.md` Gotcha 9), so `turbospark-server` could not reach
//! a line of it -- and the server needs exactly these three decisions, in
//! exactly this order, for exactly the reason the CLI does: the drafter is
//! allocated at OPEN and there is one runner per process.
//!
//! The enums here are runtime-native and the parser crate keeps its own,
//! meeting in one mapping function per front end. That is the shape
//! `map_power_profile` and the `ExpertCacheSlots` mapping already use
//! (`crates/cli/CLAUDE.md` Gotchas 2 and 6): `crates/invocation` is pure and
//! may not read an install or a machine, and every decision below needs one
//! or the other.
//!
//! macOS-only, like `families` and for the same reason: `resolve_drafter`
//! reads a resident index and [`DraftPolicies`] names GPU state. The CLI's
//! whole `generate` module is already gated this way, so nothing portable
//! loses a capability.

use std::path::Path;

use crate::families::qwen::{
    install_has_dflash, install_has_mtp_head, DflashDraftPolicy, DraftPolicies, MtpDraftPolicy,
    DFLASH_SERVING_BLOCK,
};
use crate::speculative::DEFAULT_SPECULATION_BLOCK;

/// Whether and how far this run drafts ahead. The runtime-native mirror of
/// `invocation::Speculation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Speculation {
    /// Draft if the install can, warn and continue if it cannot.
    #[default]
    Auto,
    /// Never draft.
    Off,
    /// Draft this many tokens per round, or fail naming the reason.
    Block(u32),
}

/// Which drafter [`Speculation`] drives. The two are ALTERNATIVES
/// (`docs/MTP_SPECULATIVE.md`, `docs/DFLASH2.md`): the checkpoint's own MTP
/// head drafts a token at a time, the DFlash2 block-diffusion drafter
/// proposes a whole block in one pass. A third value is not a spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeculativeDrafter {
    /// Whichever drafter the INSTALL carries, read off its resident index.
    #[default]
    Auto,
    Mtp,
    Dflash,
}

/// What [`resolve_drafter`] decided: the drafter to open, and a NOTE for the
/// case where `auto` FOUND a drafter and deliberately did not enable it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrafterChoice {
    pub drafter: SpeculativeDrafter,
    /// Whether the install carries an MTP head, as far as [`resolve_drafter`]
    /// could tell. `None` means it could not tell -- an unreadable index --
    /// and is deliberately NOT folded into `Some(false)`: "there is no head"
    /// and "nobody looked" license different things, and only the first one
    /// licenses [`draft_policies`] declining to ask the open for one.
    pub install_has_mtp_head: Option<bool>,
    /// `Some` only for a DFlash2-carrying install under `auto`. Preferred
    /// over the engine's own blocker as the disabled reason, because that
    /// one would say "carries no multi-token-prediction head" -- true, and
    /// the wrong thing to tell someone holding a drafter this port can run.
    pub note: Option<String>,
}

/// What a front end decided about speculative decoding, resolved once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculationPlan {
    Enabled {
        block: usize,
    },
    /// `reason` is `Some` when the caller might have expected otherwise, and
    /// `None` when they asked for `off` and got it.
    Disabled {
        reason: Option<String>,
    },
}

/// Which drafter an install actually carries, for
/// [`SpeculativeDrafter::Auto`].
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
/// The measurement is why: through the shipped loop on the real install over
/// 600-token generations on an idle machine, DFlash2 at block 2 reads
/// **1.33x on code and 1.47x on math against 0.90x on PROSE**, and its own
/// power A/B reads 0.88x throughput at **+17.4% J/token** on the protocol's
/// prose case (`docs/DFLASH2.md`). So a default that switched it on would
/// make one common workload slower AND hungrier without being asked. (Two
/// numbers, two captures: the 0.90x is the block sweep and the 0.88x the
/// `COOLING=max` energy arm. Do not merge them -- an earlier version of this
/// comment quoted 0.96x, which is neither; that is the PREFILL ratio from
/// the priming measurement.) The MTP head is the opposite case (1.44-1.66x)
/// and keeps its `auto`. Resolving to `Mtp` also means `open` allocates no DFlash2 state,
/// which is 213 MiB of peak footprint on the real 27B install -- measured,
/// as the gap between the two arms of the same protocol case.
///
/// An install with BOTH resolves to `Mtp`, which is what the default was
/// before `Auto` existed, so no run that worked changes. An install with
/// NEITHER also resolves to `Mtp` with no note, so the "no drafter" message
/// such a caller sees is the one they have always seen.
///
/// An unreadable index resolves to `Mtp` rather than failing: this function
/// picks a drafter, and `open` is entitled to be the one that refuses a
/// broken install.
///
/// **THE INDEX IS READ WHATEVER THE REQUEST, and that is not the same
/// question as which drafter to pick.** An explicit ask is still passed
/// through untouched -- naming a drafter is a promise the caller made to
/// themselves -- but `install_has_mtp_head` is recorded either way, because
/// [`draft_policies`] needs it to avoid asking the open for a head that is
/// not there, and a caller who named BOTH a drafter and a block is exactly
/// the one that used to get the wrong reason. It costs one read of the
/// resident index, which is kilobytes and which `open` is about to do again
/// regardless.
pub fn resolve_drafter(requested: SpeculativeDrafter, model_dir: &Path) -> DrafterChoice {
    let index = model_io::load_resident_index(&model_dir.join("model_weights.bin")).ok();
    let install_has_mtp_head = index.as_ref().map(install_has_mtp_head);
    let plain = |drafter| DrafterChoice {
        drafter,
        install_has_mtp_head,
        note: None,
    };
    if requested != SpeculativeDrafter::Auto {
        return plain(requested);
    }
    let Some(index) = index else {
        return plain(SpeculativeDrafter::Mtp);
    };
    if install_has_mtp_head == Some(false) && install_has_dflash(&index) {
        return DrafterChoice {
            drafter: SpeculativeDrafter::Mtp,
            install_has_mtp_head,
            note: Some(
                // NAMES BOTH SPELLINGS, because there are three front ends
                // and one of them has no command line: a GUI driving
                // `crates/ffi` reads this note verbatim, and telling it to
                // pass a flag it cannot pass is an instruction it cannot
                // follow. The CLI and the server share the flag; the C ABI
                // takes the same choice as an option key.
                "this install carries a DFlash2 drafter, which auto leaves OFF: measured \
                 0.90x throughput on prose against 1.33-1.47x on code and math, and \
                 +17.4% J/token, so it is opt-in. Select the dflash drafter to use it \
                 (--speculative-drafter dflash, or speculativeDrafter \"dflash\" on the \
                 C ABI)"
                    .to_string(),
            ),
        };
    }
    plain(SpeculativeDrafter::Mtp)
}

/// The [`DraftPolicies`] to open a runner with, given the resolved drafter
/// and what the caller asked for.
///
/// **THE DRAFTER THE FLAG NAMED OWNS THE REQUEST, and the other is pinned
/// OFF**, so a dflash-carrying install never silently opens BOTH drafters'
/// state.
///
/// The `note` arm is the subtle one and it is load-bearing. A NOTE MEANS THE
/// DECISION IS ALREADY MADE, so do not ask the open for a head we know is not
/// there. Without it, `--speculative 2` on a DFlash2-only install fails at
/// OPEN with `MtpDraftPolicy::Fixed`'s message -- "this install carries no
/// multi-token-prediction head ... stream an install that adds the official
/// checkpoint's last shard" -- which sends someone who is holding a working
/// drafter off to download a different one. `Off` lets the open succeed so
/// [`resolve_speculation`] can refuse with the note, which names the flag.
///
/// **THE HEADLESS ARM IS THAT SAME ARGUMENT ON A WIDER INPUT, and the two are
/// deliberately not collapsed into one condition.** The `note` arm asks "has
/// the decision already been made"; the headless arm asks "is there a head to
/// ask the open for at all". They agree on a DFlash2-only install and part
/// company on every other headless one -- which is exactly where this was
/// wrong. Measured 2026-08-21 on the real `ornith35b` install: `--speculative
/// 2` reached `MtpDraftPolicy::Fixed`, failed at OPEN with "carries no
/// multi-token-prediction head ... stream an install that adds the official
/// checkpoint's last shard", and so sent a caller after a 4.4 GB shard that
/// CANNOT help -- the batched verify is dense-only and this install routes to
/// 256 experts, which no checkpoint changes. `auto` got the same install
/// right, because `auto` reaches [`resolve_speculation`] and a named block
/// did not. `speculation_blocker` has always reported the ARCHITECTURAL
/// obstacle ahead of the missing head (`crates/runtime/CLAUDE.md` Gotcha 16);
/// it was simply never reached, because the open failed first. `Off` lets the
/// open succeed so it is.
///
/// **The hard fail is UNCHANGED.** `resolve_speculation` still returns `Err`
/// for a named block it cannot serve -- a caller who named a block is
/// measuring, and a run that quietly did not speculate is the number that
/// ends up in a table. Only the reason improves, and on a DENSE headless
/// install it is the SAME sentence as before, because that is what
/// `speculation_blocker` returns once its architectural checks pass.
pub fn draft_policies(choice: &DrafterChoice, speculation: Speculation) -> DraftPolicies {
    match choice.drafter {
        SpeculativeDrafter::Mtp => DraftPolicies::mtp(match speculation {
            Speculation::Off => MtpDraftPolicy::Off,
            _ if choice.note.is_some() => MtpDraftPolicy::Off,
            Speculation::Auto => MtpDraftPolicy::Auto,
            // `Some(false)` and never a bare falsy test: an index nobody
            // could read is `None`, and that has to keep asking for the head
            // so a broken install still fails at open with the engine's own
            // message rather than being explained by a guess.
            Speculation::Block(_) if choice.install_has_mtp_head == Some(false) => {
                MtpDraftPolicy::Off
            }
            Speculation::Block(n) => MtpDraftPolicy::Fixed(n as usize),
        }),
        // `Auto` is resolved by `resolve_drafter` before this is called and
        // cannot reach here; it shares the DFlash2 arm if it ever does.
        SpeculativeDrafter::Auto | SpeculativeDrafter::Dflash => DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: match speculation {
                Speculation::Off => DflashDraftPolicy::Off,
                Speculation::Auto => DflashDraftPolicy::Auto,
                Speculation::Block(n) => DflashDraftPolicy::Fixed(n as usize),
            },
        },
    }
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
pub fn resolve_speculation(
    requested: Speculation,
    drafter: SpeculativeDrafter,
    engine_blocker: Option<String>,
    deterministic: bool,
) -> Result<SpeculationPlan, String> {
    let unavailable = if let Some(blocker) = engine_blocker {
        Some(blocker)
    } else if !deterministic {
        Some(
            "acceptance is exact only at temperature 0, and this run samples; \
             sampled speculation needs rejection sampling with residual correction, \
             which is not implemented"
                .to_string(),
        )
    } else {
        None
    };

    // `auto` resolves to the drafter's own default block, which is not one
    // number in principle even though the two agree today: the MTP head's
    // measured optimum is 2 (`docs/MTP.md`) and the DFlash2 drafter's is also
    // 2 (`DFLASH_SERVING_BLOCK`), independently measured.
    let auto_block = match drafter {
        SpeculativeDrafter::Dflash => DFLASH_SERVING_BLOCK,
        _ => DEFAULT_SPECULATION_BLOCK,
    };

    match (requested, unavailable) {
        (Speculation::Off, _) => Ok(SpeculationPlan::Disabled { reason: None }),
        (Speculation::Auto, Some(reason)) => Ok(SpeculationPlan::Disabled {
            reason: Some(reason),
        }),
        (Speculation::Auto, None) => Ok(SpeculationPlan::Enabled { block: auto_block }),
        (Speculation::Block(_), Some(reason)) => Err(format!(
            "--speculative was asked for but cannot be served: {reason}"
        )),
        (Speculation::Block(n), None) => Ok(SpeculationPlan::Enabled { block: n as usize }),
    }
}

#[cfg(test)]
#[path = "speculation_policy_tests.rs"]
mod tests;
