//! How conservative to be about what may be loaded, and the floor under an
//! automatically-sized context window.
//!
//! Two sizing policies already budget from installed memory
//! ([`crate::context_policy`] and [`crate::expert_cache_policy`]), and until
//! now each spent a FIXED share of it. This module makes that share a user
//! choice without making it a second copy of the arithmetic: a
//! [`LoadGuard`] resolves to a [`GuardBudget`], and the two policies plus
//! `catalog`'s fit model read their reserve, their fraction and their
//! refusal threshold off it.
//!
//! **[`LoadGuard::Relaxed`] IS TODAY'S BEHAVIOUR AND IS THE DEFAULT, WHICH
//! IS LOAD-BEARING RATHER THAN CONSERVATIVE TASTE.** Every frozen peak in
//! `docs/BENCHMARKS.md`, every `measured` block in `catalog`'s `models.json`
//! and the not-`#[ignore]`d `assert_agrees_with_catalog` tie all describe an
//! engine budgeting at `HEADROOM_RESERVE_BYTES` and
//! `CONTEXT_BUDGET_FRACTION`. A default that resolved to anything else would
//! not fail: it would leave every one of those rows quietly describing a
//! configuration the engine no longer opens with. `relaxed_is_exactly_todays_
//! arithmetic` asserts the three constants rather than trusting this
//! paragraph.
//!
//! Deliberately portable, following both policies above: no `gpu`, no
//! `#[cfg(target_os = "macos")]`, no OS probe. Every input is a parameter,
//! so the whole file is unit-tested on any platform.
//!
//! ## The floor is scoped to `Auto`, and that is a decision
//!
//! [`LoadPolicy::min_auto_context`] refuses an AUTO resolution that lands
//! below it and says nothing about an explicit `--max-context 2048`. That is
//! the distinction `context_policy` already draws twice: a user naming a
//! number has decided how to spend their own machine, and is stopped only
//! when the number cannot work at all. The setting is named for AutoFit
//! because it constrains the fit, not the user.

/// Held back from the pool for everything that is not this engine.
///
/// Each tier's figure is stated relative to [`LoadGuard::Relaxed`]'s, which
/// is [`crate::HEADROOM_RESERVE_BYTES`] and is the one measured value here
/// (its own doc explains what the 4 GiB covers). The others are multiples of
/// it rather than independently derived numbers, because nothing has
/// measured them and pretending otherwise would be inventing precision.
const RELAXED_RESERVE: u64 = crate::expert_cache_policy::HEADROOM_RESERVE_BYTES;

/// A tier's share of what is left after its reserve.
///
/// `Relaxed`'s quarter is [`crate::CONTEXT_BUDGET_FRACTION`] and carries
/// that constant's reasoning: two sibling budgets drawn from one pool commit
/// half of it between them and leave half for everything else.
const RELAXED_FRACTION: f64 = 0.25;

/// Where a fit is reported as tight rather than comfortable. `Relaxed`'s is
/// `catalog`'s existing `TIGHT_FRACTION`.
const RELAXED_TIGHT: f64 = 0.9;

/// How much of the machine this engine may commit to loading a model.
///
/// The tiers are ORDERED: anything [`Self::Strict`] admits, [`Self::Balanced`]
/// admits, and so on up to [`Self::Off`]. That is asserted as a sweep
/// (`the_tiers_are_ordered_from_off_down_to_strict`) rather than left to the
/// reader, because it is the property a user relies on when moving the
/// setting and nothing about the three independent fields enforces it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LoadGuard {
    /// No precautions. Nothing is refused for memory, the whole pool is
    /// available, and a caller who asks for more than the machine has gets
    /// the allocation failure rather than a message.
    ///
    /// The honest tier for a machine whose owner knows what is on it. It is
    /// NOT the same as removing the arithmetic: `Auto` still sizes against
    /// the pool, it simply sizes against all of it.
    Off,
    /// Today's behaviour, and the default.
    #[default]
    Relaxed,
    /// Twice the reserve and a sixth of the remainder, so a model and a
    /// browser can share the machine without either swapping.
    Balanced,
    /// Three times the reserve and a tenth of the remainder. For a machine
    /// running this engine beside work that must not be interrupted.
    Strict,
    /// `Relaxed`'s fractions, plus an absolute ceiling on what the engine may
    /// ALLOCATE.
    ///
    /// The cap is on `counted` (slot cache plus KV) and not on the install's
    /// size, because those are different questions and only the first is a
    /// refusal: `catalog::recommend::fit`'s module header records that
    /// exceeding memory with the MAPPED install is the streaming this engine
    /// is built around and costs throughput rather than correctness. A cap
    /// read against the install size would refuse a 13 GB model on a 16 GB
    /// machine, which runs, and which the slot policy's floor exists for.
    Custom {
        /// Ceiling on allocated bytes, in bytes.
        max_counted_bytes: u64,
    },
}

/// What a [`LoadGuard`] resolves to. The three sizing call sites read their
/// numbers off this rather than off module constants, so a tier cannot be
/// honoured in one place and ignored in another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardBudget {
    /// Held back from installed memory before anything else is computed.
    pub reserve_bytes: u64,
    /// The share of the remainder an automatic sizing may spend.
    pub budget_fraction: f64,
    /// The share above which a fit is reported as tight.
    pub tight_fraction: f64,
    /// An absolute ceiling on allocated bytes, from [`LoadGuard::Custom`].
    pub hard_cap: Option<u64>,
    /// Whether exceeding the budget REFUSES. False on [`LoadGuard::Off`]
    /// alone, which is what makes that tier a real escape hatch rather than
    /// a very large number.
    pub refuses: bool,
}

impl LoadGuard {
    /// The CLI and JSON spelling. `None` for anything else, so an unknown
    /// value is the caller's diagnostic to raise -- the discipline
    /// `runtime::PowerProfile::parse` follows.
    ///
    /// [`Self::Custom`] is deliberately NOT spelled here: it carries a byte
    /// count, so its front-end spelling is a number rather than a word and
    /// each caller parses it beside its own units. A `"custom"` arm with no
    /// number would have to invent one.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "off" => Some(Self::Off),
            "relaxed" => Some(Self::Relaxed),
            "balanced" => Some(Self::Balanced),
            "strict" => Some(Self::Strict),
            _ => None,
        }
    }

    /// The inverse of [`Self::parse`], for echoing a resolved request.
    /// [`Self::Custom`] renders as `custom`, which round-trips through
    /// `parse` to `None` by design: the word alone does not carry the tier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Relaxed => "relaxed",
            Self::Balanced => "balanced",
            Self::Strict => "strict",
            Self::Custom { .. } => "custom",
        }
    }

    /// Resolve to the numbers the sizing policies read.
    pub fn budget(self) -> GuardBudget {
        match self {
            // A reserve of zero and the whole pool. `refuses: false` is the
            // half that matters: with a full-pool budget a refusal would
            // still fire on a genuinely oversized request, and this tier
            // exists to not do that.
            Self::Off => GuardBudget {
                reserve_bytes: 0,
                budget_fraction: 1.0,
                tight_fraction: RELAXED_TIGHT,
                hard_cap: None,
                refuses: false,
            },
            Self::Relaxed => GuardBudget {
                reserve_bytes: RELAXED_RESERVE,
                budget_fraction: RELAXED_FRACTION,
                tight_fraction: RELAXED_TIGHT,
                hard_cap: None,
                refuses: true,
            },
            Self::Balanced => GuardBudget {
                reserve_bytes: RELAXED_RESERVE * 2,
                budget_fraction: RELAXED_FRACTION / 1.5,
                tight_fraction: 0.8,
                hard_cap: None,
                refuses: true,
            },
            Self::Strict => GuardBudget {
                reserve_bytes: RELAXED_RESERVE * 3,
                budget_fraction: RELAXED_FRACTION / 2.5,
                tight_fraction: 0.7,
                hard_cap: None,
                refuses: true,
            },
            Self::Custom { max_counted_bytes } => GuardBudget {
                reserve_bytes: RELAXED_RESERVE,
                budget_fraction: RELAXED_FRACTION,
                tight_fraction: RELAXED_TIGHT,
                hard_cap: Some(max_counted_bytes),
                refuses: true,
            },
        }
    }

    /// Memory available to this engine on a machine of `physical` bytes,
    /// after the tier's reserve and whatever the install already commits.
    ///
    /// Saturating throughout: a reserve larger than the machine means no
    /// budget, never a wrapped one.
    pub fn available(self, physical: u64, committed: u64) -> u64 {
        physical.saturating_sub(committed.saturating_add(self.budget().reserve_bytes))
    }
}

/// A [`LoadGuard`] plus the floor under an automatic context resolution.
///
/// One parameter rather than two, because these travel together through
/// three call sites and a function already taking six positional arguments
/// does not need eight.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LoadPolicy {
    /// How much of the machine may be committed.
    pub guard: LoadGuard,
    /// The fewest tokens an AUTO resolution may land on. Zero, the default,
    /// imposes no floor -- which is what silence means here, since a caller
    /// that has not asked for a minimum has not asked to be refused.
    pub min_auto_context: u32,
}

impl LoadPolicy {
    /// The policy with a guard and no floor.
    pub fn new(guard: LoadGuard) -> Self {
        Self {
            guard,
            min_auto_context: 0,
        }
    }
}

#[cfg(test)]
#[path = "load_guard_tests.rs"]
mod tests;
