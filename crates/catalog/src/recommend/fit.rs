//! Does this artifact fit this machine, and what would it cost?
//!
//! **THE CURRENCY IS A WORKING SET, NOT `phys_footprint` AND NOT
//! BITS-PER-WEIGHT.** Both of those are the obvious answers and both are
//! wrong here for opposite reasons.
//!
//! Bits-per-weight is what shoehorn ranks by, because it RE-QUANTIZES: given
//! a budget it solves for a per-tensor mixed-precision assignment that lands
//! inside it. This port chooses among the quantizations a repository already
//! publishes, so its question is which artifact rather than which bit width,
//! and a weight budget expressed in bits answers nothing it can act on.
//!
//! `phys_footprint` is what the memory oracles freeze, and it is the wrong
//! bound to plan with because it does not count everything that has to be in
//! memory. A dense install's mapped weights are absent from it entirely
//! (AGENTS.md Gotcha 40: Mistral 7B reads 684 MiB against 4.07 GiB of
//! weights), and a streamed MoE's expert table is absent by design. Neither
//! absence means the bytes are free -- they still have to be read, and a
//! machine that cannot hold them reads them off the SSD on every miss.
//!
//! So this splits the answer in two and reports both, because they fail
//! differently:
//!
//! - **counted** = slot cache + KV. What the engine ALLOCATES, and what
//!   `phys_footprint` charges for. Exceeding the budget is a failed
//!   allocation, which is a refusal.
//! - **mapped** = the install on disk, resident core and expert table alike.
//!   What the engine READS. Exceeding memory is not an error at all; it is
//!   the streaming this engine is built around, and it costs throughput
//!   rather than correctness.
//!
//! A verdict that collapsed those into one boolean would either refuse a
//! 13 GB install on a 16 GB machine (which runs, and is the configuration the
//! slot policy's floor exists for) or promise that a 27 GB one runs well.
//!
//! **THE RESIDENT CORE IS IN `mapped` AND NOT IN `counted`, WHICH IS NOT WHAT
//! GOTCHA 19 SAYS AND IS WHAT THE MEASUREMENTS SAY.** That gotcha records
//! that `newBufferWithBytesNoCopy` pins the mapped range and so charges it to
//! `phys_footprint`; every frozen peak since contradicts it. Gemma 4 reads
//! 2,175 MiB against a 1.26 GiB resident core plus 1.5 GiB of slot cache plus
//! 320 MiB of KV -- the core is absent. `gptoss_memory_oracle.rs` records the
//! same thing in its own words ("the resident mapping is largely uncounted
//! here, as it is on the dense install"), and Gotcha 40 records it for the
//! dense case outright. Putting the core in `counted` would make this
//! estimate incomparable with the very rows it is checked against, and would
//! read museGlimmer's 15 GB of dense weights as 15 GB of allocation against a
//! measured 536 MiB.

use model_io::ArchConfig;

/// What a candidate looks like to the arithmetic below.
///
/// Deliberately not a `CatalogEntry` or a `ProbeReport`: both of those can
/// produce one, a machine-fit question needs neither's other fields, and
/// keeping this a plain struct is what lets every case in this file be a unit
/// test with no network and no install (the discipline
/// [`model_io::ExpertCacheSlots::resolve`] and
/// [`model_io::largest_context_within`] already follow).
#[derive(Debug, Clone, Default)]
pub struct Shape {
    /// Bytes the `.gturbo` install occupies on disk.
    pub install_bytes: u64,
    /// Bytes of ONE routed expert in ONE layer, or `None` for a dense model.
    /// This is the number AGENTS.md Gotcha 36 is about: what decides whether
    /// a model can stream is expert GRANULARITY, never model size.
    pub expert_stride: Option<u64>,
    /// The checkpoint's own shape, when something has read a header. `None`
    /// leaves the KV and slot-cache terms unknown rather than guessed --
    /// see [`Fit::counted`].
    pub arch: Option<ArchConfig>,
    /// A measured peak from `models.json`, in bytes, when one exists for this
    /// machine's chip AND was taken at the context being asked about. Beats
    /// every estimate below and replaces them outright.
    pub measured_counted: Option<u64>,
}

/// Where [`Fit::counted`] came from, so a caller never prints an estimate as
/// though it were a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountedSource {
    /// A frozen row in `models.json`, taken on this chip at this context.
    Measured,
    /// Computed from the checkpoint's shape.
    Estimated,
    /// Nothing has read this checkpoint's header, so the slot cache and KV
    /// are unknown. Not zero -- `turbospark-model probe` is what fills it in.
    Unknown,
}

/// How a candidate lands on one machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitVerdict {
    /// Everything fits, install included: no expert read ever leaves memory.
    Resident,
    /// What the engine allocates fits; the install does not. This is the
    /// normal case for a streamed MoE and is what the engine is FOR, so it is
    /// a fit and not a warning.
    Streams,
    /// Allocations fit with under 10% to spare. Runs, and anything else on
    /// the machine is competing with it.
    Tight,
    /// What the engine allocates does not fit. `KvCacheManager::new` and the
    /// expert streamer both allocate up front, so this is a failed open
    /// rather than a slow run.
    Refused,
    /// Not enough is known to say. Never rendered as a fit.
    Unknown,
}

impl FitVerdict {
    /// True if the candidate fits memory sufficiently to run (resident, streamed, or tight).
    pub fn runs(self) -> bool {
        matches!(self, Self::Resident | Self::Streams | Self::Tight)
    }

    /// Human-readable summary of the fit verdict.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "fits, fully resident",
            Self::Streams => "fits, streams experts from disk",
            Self::Tight => "fits, tight",
            Self::Refused => "does not fit",
            Self::Unknown => "unknown (probe it)",
        }
    }
}

/// The arithmetic, and where each term came from.
#[derive(Debug, Clone)]
pub struct Fit {
    /// The install's own mapped weight region: everything but the routed
    /// expert table, which streams.
    pub resident_bytes: u64,
    /// What [`model_io::ExpertCacheSlots::Auto`] would resolve to here -- the
    /// same call `open()` makes, so a recommendation cannot promise a slot
    /// count the engine then declines to use.
    pub slots: usize,
    pub slot_cache_bytes: u64,
    pub kv_bytes: u64,
    /// `slot cache + KV`, or the measured peak when one applies. The
    /// resident core is deliberately NOT here; see the module header.
    pub counted: u64,
    pub counted_source: CountedSource,
    /// The whole install on disk, resident core and expert table alike.
    pub mapped: u64,
    /// `physical - CONTEXT_RESERVE_BYTES`. The reserve is shared with the two
    /// sizing policies rather than subtracted a second time here, which is
    /// the same relationship `CONTEXT_RESERVE_BYTES` already has to
    /// `HEADROOM_RESERVE_BYTES`.
    pub budget: u64,
    /// The largest window this candidate affords on this machine, from
    /// [`model_io::largest_context_within`]. Zero when the shape is unknown.
    pub largest_context: u32,
    pub verdict: FitVerdict,
    /// The tier this fit was reached under. Carried so a caller re-deriving
    /// the verdict after substituting a measurement cannot silently apply a
    /// different one, which is the failure [`verdict_for_counted`] exists to
    /// prevent one field over.
    pub guard: model_io::LoadGuard,
}

/// Fraction of the budget above which a fit is reported as tight.
///
/// **Now [`model_io::LoadGuard::Relaxed`]'s answer rather than the only
/// answer**, kept as a named constant because it is what every verdict in
/// `docs/BENCHMARKS.md` and every `measured` row was reached under.
/// `the_default_guard_reproduces_the_frozen_thresholds` pins the two
/// together, since `model_io` cannot import this crate to state it there.
/// Test-only because nothing in the arithmetic reads it any more -- the value
/// arrives on `GuardBudget` -- and a live constant nothing reads is the shape
/// that silently stops matching the thing it claims to mirror.
#[cfg(test)]
const TIGHT_FRACTION: f64 = 0.9;

/// Where a candidate lands on a machine with `physical_bytes` of memory, at
/// `context` tokens and `slot_policy` expert-cache slots.
///
/// **`slot_policy` IS A PARAMETER AND NOT ALWAYS `Auto`, WHICH THE ACCURACY
/// GATE BELOW FOUND THE HARD WAY.** A frozen peak in `models.json` was taken
/// at `PROTOCOL_EXPERT_CACHE_SLOTS`, which is 16, while `Auto` resolves
/// Gemma 4 to 32 on a 36 GB machine -- so comparing an `Auto` estimate
/// against a measured row compares two different configurations and reads as
/// a 55% overestimate. Every caller that recommends passes `Auto`, because
/// that is what `open()` will do; a caller comparing against a measurement
/// has to pin what the measurement pinned.
///
/// **`guard` MUST BE THE SAME TIER THE LOADER WILL OPEN WITH.** This function
/// and `model_io::resolve_max_context` share a budget by construction, and
/// that is the whole reason a recommendation can be trusted: a hub that
/// recommended under `relaxed` while the session opened under `strict` would
/// promise a fit the loader then refuses, in the one place a user has no way
/// to see the disagreement. `model_io::LoadGuard::default()` is the pre-guard
/// arithmetic and is what every caller that has not been told otherwise
/// passes.
pub fn fit(
    shape: &Shape,
    physical_bytes: u64,
    context: u32,
    slot_policy: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
) -> Fit {
    let guard_budget = guard.budget();
    let budget = physical_bytes.saturating_sub(guard_budget.reserve_bytes);

    let (layers, experts) = match &shape.arch {
        Some(arch) => (
            arch.num_layers.max(0) as u64,
            arch.num_experts.max(0) as u64,
        ),
        None => (0, 0),
    };
    let stride = shape.expert_stride.unwrap_or(0);
    let bytes_per_slot = stride.saturating_mul(layers);
    // The whole routed table is the part of the install that does NOT stay
    // resident. Saturating rather than checked: an approximate `install_bytes`
    // (the catalog allows it to be generous) can sit below the table it
    // contains, and a resident core of zero is closer to true than a wrap.
    let expert_table = bytes_per_slot.saturating_mul(experts);
    let resident_bytes = shape.install_bytes.saturating_sub(expert_table);

    let slots = slot_policy.resolve(physical_bytes, resident_bytes, bytes_per_slot);
    let slot_cache_bytes = (slots as u64).saturating_mul(bytes_per_slot);
    let kv_bytes = shape
        .arch
        .as_ref()
        .map(|a| model_io::kv_bytes_for_context(a, context))
        .unwrap_or(0);

    let estimated = slot_cache_bytes.saturating_add(kv_bytes);
    let (counted, counted_source) = match (shape.measured_counted, shape.arch.is_some()) {
        (Some(m), _) => (m, CountedSource::Measured),
        (None, true) => (estimated, CountedSource::Estimated),
        (None, false) => (0, CountedSource::Unknown),
    };

    let largest_context = shape
        .arch
        .as_ref()
        .map(|a| {
            model_io::largest_context_within(
                a,
                budget.saturating_sub(resident_bytes.saturating_add(slot_cache_bytes)),
            )
        })
        .unwrap_or(0);

    let verdict = verdict_for(
        counted,
        counted_source,
        shape.install_bytes,
        budget,
        &guard_budget,
    );

    Fit {
        resident_bytes,
        slots,
        slot_cache_bytes,
        kv_bytes,
        counted,
        counted_source,
        mapped: shape.install_bytes,
        budget,
        largest_context,
        verdict,
        guard,
    }
}

/// Re-derive the verdict after a caller has replaced [`Fit::counted`] with a
/// measurement.
///
/// Exists so the ladder is applied in ONE place. A caller that substituted a
/// measured peak and left the verdict alone would report a number and a
/// judgement that disagree, which is the worst of the three possible states.
pub(super) fn verdict_for_counted(f: &Fit) -> FitVerdict {
    verdict_for(
        f.counted,
        f.counted_source,
        f.mapped,
        f.budget,
        &f.guard.budget(),
    )
}

/// The tier ladder. Adapted from shoehorn's `verdict_for` (see `NOTICE`),
/// which tiers on achievable bits-per-weight; the shape of the function is
/// the same and every threshold is different, because the quantity being
/// tiered is not the same quantity.
fn verdict_for(
    counted: u64,
    source: CountedSource,
    mapped: u64,
    budget: u64,
    guard: &model_io::GuardBudget,
) -> FitVerdict {
    if source == CountedSource::Unknown {
        return FitVerdict::Unknown;
    }
    // The Custom tier's ceiling is checked BEFORE the budget and refuses
    // regardless of headroom: it is the user naming an allocation they do not
    // want exceeded, not a second estimate of what the machine holds.
    if guard.hard_cap.is_some_and(|cap| counted > cap) {
        return FitVerdict::Refused;
    }
    // `Off` is the one tier that declines to refuse. It still TIERS -- a
    // caller wants to know a fit is tight even when nothing will stop them --
    // so only the refusal arm is skipped.
    if guard.refuses && (budget == 0 || counted > budget) {
        return FitVerdict::Refused;
    }
    if budget > 0 && counted as f64 > budget as f64 * guard.tight_fraction {
        return FitVerdict::Tight;
    }
    if counted.saturating_add(mapped) <= budget {
        FitVerdict::Resident
    } else {
        FitVerdict::Streams
    }
}

#[cfg(test)]
#[path = "fit_tests.rs"]
mod tests;

/// The context windows a ladder always reports, before the checkpoint's own
/// two are folded in.
///
/// Powers of two from the engine's default up. They are STANDARD rather than
/// derived so two models can be compared row against row, which is the whole
/// point of showing a ladder instead of one number.
const LADDER_WINDOWS: [u32; 6] = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072];

/// One context window and what it costs.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderRung {
    pub context: u32,
    pub kv_bytes: u64,
    /// Slot cache plus this window's KV. The slot cache term does NOT move
    /// with the window, which is why a long-context question is really a
    /// question about the KV alone on a streamed MoE.
    pub counted: u64,
    pub verdict: FitVerdict,
    /// Past the checkpoint's trained window. Reported and never hidden:
    /// RoPE extrapolates rather than failing and some checkpoints carry YaRN
    /// scaling meant to exceed it, so this is a warning and not a refusal
    /// (AGENTS.md Gotcha 55).
    pub past_trained: bool,
    pub is_trained_max: bool,
    pub is_largest_fitting: bool,
}

/// What each window in [`LADDER_WINDOWS`] costs, plus the checkpoint's trained
/// maximum and the largest window this machine affords.
///
/// **THE CURVE IS NOT LINEAR AND A CALLER MUST NOT MULTIPLY.** A
/// sliding-window layer is a ring capped at `sliding_window + 128`, so past
/// that cap it stops growing and only the FULL layers still cost anything.
/// Measured on the shipped baselines from 4,096 to 131,072: Mistral 7B grows
/// 32x (512 MiB to 16,384, every layer full) while Gemma 4 grows 9x (305 MiB
/// to 2,785, 25 of its 30 layers being rings). A UI doing its own arithmetic
/// is 3.5x wrong on Gemma, in the direction that refuses a model that runs.
/// Linear-attention layers contribute zero KV and a fixed
/// [`model_io::gdn_state_bytes`] instead, which moves with nothing.
///
/// Empty when `shape.arch` is `None`. There is no ladder without a shape and
/// the family baseline is not a substitute for one (Gotcha 10).
pub fn context_ladder(
    shape: &Shape,
    physical_bytes: u64,
    slot_policy: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
    trained: Option<u32>,
) -> Vec<LadderRung> {
    if shape.arch.is_none() {
        return Vec::new();
    }
    // One fit establishes the slot count and the largest fitting window; both
    // are independent of the window being priced, so they are computed once.
    let anchor = fit(shape, physical_bytes, 4_096, slot_policy, guard);
    let largest = anchor.largest_context;

    let mut windows: Vec<u32> = LADDER_WINDOWS.to_vec();
    windows.extend(trained);
    windows.push(largest);
    windows.retain(|w| *w > 0 && *w <= model_io::MAX_SUPPORTED_CONTEXT);
    windows.sort_unstable();
    windows.dedup();

    windows
        .into_iter()
        .map(|context| {
            let rung = fit(shape, physical_bytes, context, slot_policy, guard);
            LadderRung {
                context,
                kv_bytes: rung.kv_bytes,
                counted: rung.counted,
                verdict: rung.verdict,
                past_trained: trained.is_some_and(|t| context > t),
                is_trained_max: trained == Some(context),
                is_largest_fitting: context == largest,
            }
        })
        .collect()
}
