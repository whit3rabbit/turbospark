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
    pub fn runs(self) -> bool {
        matches!(self, Self::Resident | Self::Streams | Self::Tight)
    }

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
}

/// Fraction of the budget above which a fit is reported as tight.
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
pub fn fit(
    shape: &Shape,
    physical_bytes: u64,
    context: u32,
    slot_policy: model_io::ExpertCacheSlots,
) -> Fit {
    let budget = physical_bytes.saturating_sub(model_io::CONTEXT_RESERVE_BYTES);

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

    let verdict = verdict_for(counted, counted_source, shape.install_bytes, budget);

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
    }
}

/// Re-derive the verdict after a caller has replaced [`Fit::counted`] with a
/// measurement.
///
/// Exists so the ladder is applied in ONE place. A caller that substituted a
/// measured peak and left the verdict alone would report a number and a
/// judgement that disagree, which is the worst of the three possible states.
pub(super) fn verdict_for_counted(f: &Fit) -> FitVerdict {
    verdict_for(f.counted, f.counted_source, f.mapped, f.budget)
}

/// The tier ladder. Adapted from shoehorn's `verdict_for` (see `NOTICE`),
/// which tiers on achievable bits-per-weight; the shape of the function is
/// the same and every threshold is different, because the quantity being
/// tiered is not the same quantity.
fn verdict_for(counted: u64, source: CountedSource, mapped: u64, budget: u64) -> FitVerdict {
    if source == CountedSource::Unknown {
        return FitVerdict::Unknown;
    }
    if budget == 0 || counted > budget {
        return FitVerdict::Refused;
    }
    if counted as f64 > budget as f64 * TIGHT_FRACTION {
        return FitVerdict::Tight;
    }
    if counted.saturating_add(mapped) <= budget {
        FitVerdict::Resident
    } else {
        FitVerdict::Streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_io::{known_architecture, ModelFamily};

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    /// The development machine every frozen row in `models.json` was measured
    /// on (36 GB, see `CLAUDE.local.md`).
    const M4_MAX: u64 = 36 * GIB;
    /// What `turbospark-bench`'s protocol pins, and therefore the
    /// configuration every frozen peak was taken in.
    const PROTOCOL_SLOTS: model_io::ExpertCacheSlots = model_io::ExpertCacheSlots::Fixed(16);

    fn arch(family: ModelFamily) -> ArchConfig {
        known_architecture(family)
    }

    /// **THE ACCURACY GATE, and it is the reason this model is allowed to
    /// print a number at all.**
    ///
    /// Three installs have a frozen `phys_footprint` peak in `models.json`
    /// and between them they cover all three shapes this engine has: a
    /// streamed MoE, a dense safetensors install and a dense GGUF one. What
    /// is compared is the COUNTED terms only -- slot cache plus KV -- because
    /// the mapped weights are in this model's `mapped` and out of
    /// `phys_footprint` (Gotcha 40), so comparing totals would be comparing
    /// two different quantities and agreeing would be the surprise.
    ///
    /// The residual in each case is process baseline and host scratch, which
    /// the estimate deliberately does not try to predict: the two oracle
    /// comments that do the same arithmetic by hand call it ~253 MiB
    /// (qwen38) and ~349 MiB (muse_glimmer). So the assertion is that the
    /// model's terms sit UNDER the measured peak and within 500 MiB of it.
    /// A model that overshot would be the serious failure -- it would refuse
    /// installs that run.
    ///
    /// **EVERY ROW PINS `Fixed(16)` AND THAT IS THE POINT.** The protocol
    /// pins `PROTOCOL_EXPERT_CACHE_SLOTS` for the reason AGENTS.md Gotcha 35
    /// states, so every frozen peak is a peak at 16 slots -- while `Auto`
    /// resolves Gemma 4 to 32 on this machine, which is 3.0 GiB of slot cache
    /// against 1.5. Passing `Auto` here reads as a 55% overestimate and is
    /// really two different configurations being compared. The 32-slot arm
    /// below is what checks the model at the count a recommendation will
    /// actually use.
    #[test]
    fn the_counted_terms_reproduce_the_three_frozen_peaks() {
        // gemma4: 30 layers, ~3.2 MiB per expert, 16 slots at the protocol.
        // Peak 2,175 MiB.
        let gemma = Shape {
            install_bytes: 13_000_000_000,
            expert_stride: Some(3_355_443),
            arch: Some(arch(ModelFamily::Gemma4)),
            measured_counted: None,
        };
        // qwen38-27b: dense, so no slot cache at all. Peak 661 MiB, of which
        // the oracle attributes 256 MiB to KV at 4,096.
        let qwen38 = Shape {
            install_bytes: 15_132_916_736,
            expert_stride: None,
            arch: Some(arch(ModelFamily::QwenGdnDense)),
            measured_counted: None,
        };
        // mistral7b: dense GGUF at 8,192, where the oracle attributes
        // 1,024 MiB of the 1,203 MiB peak to KV.
        let mistral = Shape {
            install_bytes: 4_371_570_688,
            expert_stride: None,
            arch: Some(arch(ModelFamily::Llama)),
            measured_counted: None,
        };

        for (name, shape, context, peak_mib, slot_cache_expected) in [
            ("gemma4", &gemma, 4096u32, 2175u64, true),
            ("qwen38-27b", &qwen38, 4096, 661, false),
            ("mistral7b", &mistral, 8192, 1203, false),
        ] {
            let f = fit(shape, M4_MAX, context, PROTOCOL_SLOTS);
            assert_eq!(
                f.slot_cache_bytes > 0,
                slot_cache_expected,
                "{name}: a dense install has no slot cache and an MoE one must have"
            );
            let counted_terms = f.counted;
            let peak = peak_mib * MIB;
            assert!(
                counted_terms <= peak,
                "{name}: estimated {} MiB of counted terms against a measured peak of {peak_mib} MiB -- \
                 an OVERestimate refuses installs that run",
                counted_terms / MIB
            );
            assert!(
                peak - counted_terms < 500 * MIB,
                "{name}: {} MiB of counted terms leaves {} MiB unexplained against the \
                 measured {peak_mib} MiB peak, which is more than process baseline covers",
                counted_terms / MIB,
                (peak - counted_terms) / MIB
            );
        }
    }

    /// The same gate at the slot count a RECOMMENDATION resolves to, which on
    /// this machine is 32 rather than the protocol's 16.
    ///
    /// `docs/DECODE_BUDGET.md`'s slot sweep measured Gemma 4 at 3,728 and
    /// 3,654 MiB there, against 2,109-2,180 at 16 -- so the two arms together
    /// say the model tracks the slot cache rather than happening to land near
    /// one frozen number. Without this arm, the 16-slot assertion above would
    /// pass just as well with the slot term deleted entirely on the two dense
    /// rows, and the one MoE row would carry the whole claim.
    #[test]
    fn the_estimate_tracks_the_slot_count_a_recommendation_resolves_to() {
        let gemma = Shape {
            install_bytes: 13_000_000_000,
            expert_stride: Some(3_355_443),
            arch: Some(arch(ModelFamily::Gemma4)),
            measured_counted: None,
        };
        let auto = fit(&gemma, M4_MAX, 4096, model_io::ExpertCacheSlots::Auto);
        assert_eq!(auto.slots, 32, "this machine has headroom for the top rung");
        let counted_terms = auto.counted;
        let measured = 3654 * MIB;
        assert!(
            counted_terms <= measured && measured - counted_terms < 500 * MIB,
            "at 32 slots: {} MiB of counted terms against a measured 3,654 MiB",
            counted_terms / MIB
        );
        // And the slot term is really what moved: doubling the count roughly
        // doubles it, where the KV is untouched.
        let pinned = fit(&gemma, M4_MAX, 4096, PROTOCOL_SLOTS);
        assert_eq!(auto.slot_cache_bytes, pinned.slot_cache_bytes * 2);
        assert_eq!(auto.kv_bytes, pinned.kv_bytes);
    }

    /// The slot cache is what a fit is made of on an MoE install, and the
    /// number it multiplies is the per-layer expert stride rather than the
    /// model size. Mixtral is the case: correct, smaller than Gemma 4 by
    /// parameter count, and it wants 54.5 GiB of pinned cache at 16 slots
    /// (AGENTS.md Gotcha 36).
    #[test]
    fn a_coarse_grained_moe_is_refused_on_the_slot_cache_and_not_on_its_size() {
        let mixtral = Shape {
            install_bytes: 29_000_000_000,
            expert_stride: Some(108 * MIB + 900 * 1024),
            arch: Some(arch(ModelFamily::Llama)),
            measured_counted: None,
        };
        let f = fit(&mixtral, M4_MAX, 4096, model_io::ExpertCacheSlots::Auto);
        assert_eq!(f.verdict, FitVerdict::Refused);
        assert!(
            f.slot_cache_bytes > 50 * GIB,
            "the slot cache is the term that refuses it: {} GiB",
            f.slot_cache_bytes / GIB
        );
        // And the refusal is not just "the install is big": the same install
        // size split 128 ways instead of 8 fits comfortably.
        let fine_grained = Shape {
            expert_stride: Some(3_355_443),
            ..mixtral.clone()
        };
        assert!(fit(
            &fine_grained,
            M4_MAX,
            4096,
            model_io::ExpertCacheSlots::Auto
        )
        .verdict
        .runs());
    }

    /// **A 13 GB install on a 16 GB machine RUNS**, and a fit model that
    /// refused it would contradict the slot policy's own floor, which exists
    /// for exactly that machine. It streams: what the engine allocates fits,
    /// and the install does not.
    #[test]
    fn a_large_install_on_a_small_machine_streams_rather_than_being_refused() {
        let gemma = Shape {
            install_bytes: 13_000_000_000,
            expert_stride: Some(3_355_443),
            arch: Some(arch(ModelFamily::Gemma4)),
            measured_counted: None,
        };
        assert_eq!(
            fit(&gemma, 16 * GIB, 4096, model_io::ExpertCacheSlots::Auto).verdict,
            FitVerdict::Streams
        );
        // The same install on a machine with room for all of it.
        assert_eq!(
            fit(&gemma, 64 * GIB, 4096, model_io::ExpertCacheSlots::Auto).verdict,
            FitVerdict::Resident
        );
    }

    /// A measured row REPLACES the estimate rather than being averaged with
    /// it or checked against it, and it is labelled so a caller cannot print
    /// one as the other.
    #[test]
    fn a_measured_peak_wins_over_the_estimate() {
        let shape = Shape {
            install_bytes: 13_000_000_000,
            expert_stride: Some(3_355_443),
            arch: Some(arch(ModelFamily::Gemma4)),
            measured_counted: Some(2175 * MIB),
        };
        let f = fit(&shape, M4_MAX, 4096, model_io::ExpertCacheSlots::Auto);
        assert_eq!(f.counted, 2175 * MIB);
        assert_eq!(f.counted_source, CountedSource::Measured);
    }

    /// **AN UNKNOWN SHAPE IS UNKNOWN, NEVER ZERO.** A catalog row nobody has
    /// probed has no `ArchConfig`, and treating its missing KV and slot cache
    /// as zero would report the cheapest possible fit for the model about
    /// which the least is known -- which is the inverse of its real rank, the
    /// same failure the probe's unsized ggml types were rendering as 0 bytes.
    #[test]
    fn an_unprobed_row_reports_unknown_rather_than_a_free_fit() {
        let bare = Shape {
            install_bytes: 13_000_000_000,
            ..Shape::default()
        };
        let f = fit(&bare, M4_MAX, 4096, model_io::ExpertCacheSlots::Auto);
        assert_eq!(f.verdict, FitVerdict::Unknown);
        assert_eq!(f.counted_source, CountedSource::Unknown);
        assert!(!f.verdict.runs(), "unknown must never rank as a fit");
        assert_eq!(f.largest_context, 0);
    }

    /// **The case that makes this file discriminate its own constants.**
    /// Every assertion above clears its boundary by a wide margin, so
    /// `TIGHT_FRACTION` and the reserve could both be silently wrong. A
    /// machine sized so the counted terms land just under and just over 90%
    /// of the budget is the one input where either changes the answer -- the
    /// same discipline `expert_cache_policy`'s own boundary case states.
    #[test]
    fn the_tight_threshold_decides_the_verdict_at_the_boundary() {
        let shape = Shape {
            // Zero, so the KV is the ONLY term and the threshold is what the
            // assertions below are testing rather than an install size.
            install_bytes: 0,
            expert_stride: None,
            arch: Some(arch(ModelFamily::Llama)),
            measured_counted: None,
        };
        // Llama's KV at 8,192 is exactly 1,024 MiB (the mistral oracle's own
        // subtraction). Budget is `physical - 4 GiB`, so a 5 GiB machine has
        // 1 GiB of budget and the KV is 100% of it.
        let f = fit(&shape, 5 * GIB, 8192, model_io::ExpertCacheSlots::Auto);
        assert_eq!(f.kv_bytes, GIB);
        assert_eq!(f.verdict, FitVerdict::Tight);
        // Widen the budget past the 90% line and it is an ordinary fit;
        // narrow it below the KV and it is refused.
        assert_eq!(
            fit(&shape, 6 * GIB, 8192, model_io::ExpertCacheSlots::Auto).verdict,
            FitVerdict::Resident
        );
        assert_eq!(
            fit(
                &shape,
                4 * GIB + 512 * MIB,
                8192,
                model_io::ExpertCacheSlots::Auto
            )
            .verdict,
            FitVerdict::Refused
        );
    }

    /// The headroom column is the other half of the answer: not "does 8,192
    /// fit" but "how much would". It comes from the same bisection
    /// `--max-context auto` uses, so the number a recommendation prints is
    /// the number the engine will accept.
    #[test]
    fn the_largest_context_is_the_one_the_engine_would_resolve() {
        let dense = Shape {
            install_bytes: 4_371_570_688,
            expert_stride: None,
            arch: Some(arch(ModelFamily::Llama)),
            measured_counted: None,
        };
        let f = fit(&dense, M4_MAX, 8192, model_io::ExpertCacheSlots::Auto);
        assert!(
            f.largest_context > 8192,
            "a 36 GB machine affords more than the protocol window: {}",
            f.largest_context
        );
        assert_eq!(
            f.largest_context,
            model_io::largest_context_within(
                &arch(ModelFamily::Llama),
                f.budget - f.resident_bytes
            ),
            "and it is the engine's own answer, not a second formula"
        );
    }
}
