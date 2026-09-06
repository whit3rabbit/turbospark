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
/// The tier every case below was written under, and the one every frozen row
/// in `models.json` was measured under. Named rather than
/// `LoadGuard::default()` at each site so a future change to the default is a
/// visible edit here rather than a silent re-interpretation of these numbers.
const DEFAULT_GUARD: model_io::LoadGuard = model_io::LoadGuard::Relaxed;

/// Gemma 4 26B-A4B, the install every frozen row in `docs/BENCHMARKS.md` and
/// every guard case below is stated against.
fn gemma4_shape() -> Shape {
    Shape {
        install_bytes: 13_000_000_000,
        expert_stride: Some(3_355_443),
        arch: Some(arch(ModelFamily::Gemma4)),
        measured_counted: None,
    }
}

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
        let f = fit(shape, M4_MAX, context, PROTOCOL_SLOTS, DEFAULT_GUARD);
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
/// this machine is now ABOVE the protocol's 16 -- and, since `qwen4_exp`'s
/// Phase 4 widened `ALLOWED_CACHE_SLOTS` past its old 32-slot ceiling, above
/// 32 too.
///
/// `docs/DECODE_BUDGET.md`'s slot sweep measured Gemma 4 at 3,728 and
/// 3,654 MiB at 32, against 2,109-2,180 at 16 -- so the two arms together
/// say the model tracks the slot cache rather than happening to land near
/// one frozen number. Without this arm, the 16-slot assertion above would
/// pass just as well with the slot term deleted entirely on the two dense
/// rows, and the one MoE row would carry the whole claim.
///
/// **The bound above 32 slots is DERIVED, not a second real measurement** --
/// nobody has run a memory oracle above 32 slots (this port's
/// `ALLOWED_CACHE_SLOTS` only grew past it for `qwen4_exp`'s Phase 4), and
/// `model-io` Gotcha 9's own rule is that a peak measured at one slot count
/// does not apply at another. Rather than hand-computing what one extra slot
/// costs (`fit.rs`'s `bytes_per_slot` is `stride * arch.num_layers`, and this
/// fixture's `arch()` helper does not have to agree with real Gemma 4's 30),
/// the per-slot cost is read back from `pinned.slot_cache_bytes / 16` --
/// this test's OWN 16-slot call through the same code path -- so the
/// derivation cannot silently disagree with what `fit()` actually computes.
#[test]
fn the_estimate_tracks_the_slot_count_a_recommendation_resolves_to() {
    let gemma = Shape {
        install_bytes: 13_000_000_000,
        expert_stride: Some(3_355_443),
        arch: Some(arch(ModelFamily::Gemma4)),
        measured_counted: None,
    };
    let auto = fit(
        &gemma,
        M4_MAX,
        4096,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
    let pinned = fit(&gemma, M4_MAX, 4096, PROTOCOL_SLOTS, DEFAULT_GUARD);
    let model_io::ExpertCacheSlots::Fixed(pinned_slots) = PROTOCOL_SLOTS else {
        panic!("PROTOCOL_SLOTS must be Fixed for this ratio to mean anything")
    };
    assert!(
        auto.slots > 32,
        "this machine has headroom past the old 32-slot ceiling; got {}",
        auto.slots
    );
    // The slot term scales EXACTLY with slot count, KV untouched -- the
    // property that licenses deriving a bound above 32 from the 32-slot
    // measurement at all.
    assert_eq!(
        auto.slot_cache_bytes,
        pinned.slot_cache_bytes / pinned_slots as u64 * auto.slots as u64
    );
    assert_eq!(auto.kv_bytes, pinned.kv_bytes);

    let per_slot_bytes = pinned.slot_cache_bytes / pinned_slots as u64;
    let measured_at_32 = 3654 * MIB;
    let extra_slots = auto.slots as u64 - 32;
    let derived = measured_at_32 + extra_slots * per_slot_bytes;
    let counted_terms = auto.counted;
    assert!(
        counted_terms <= derived && derived - counted_terms < 500 * MIB,
        "at {} slots: {} MiB of counted terms against a derived {} MiB (32-slot measured \
         3,654 MiB plus {extra_slots} more slots' stride)",
        auto.slots,
        counted_terms / MIB,
        derived / MIB
    );
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
    let f = fit(
        &mixtral,
        M4_MAX,
        4096,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
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
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD
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
        fit(
            &gemma,
            16 * GIB,
            4096,
            model_io::ExpertCacheSlots::Auto,
            DEFAULT_GUARD
        )
        .verdict,
        FitVerdict::Streams
    );
    // The same install on a machine with room for all of it.
    assert_eq!(
        fit(
            &gemma,
            64 * GIB,
            4096,
            model_io::ExpertCacheSlots::Auto,
            DEFAULT_GUARD
        )
        .verdict,
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
    let f = fit(
        &shape,
        M4_MAX,
        4096,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
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
    let f = fit(
        &bare,
        M4_MAX,
        4096,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
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
    let f = fit(
        &shape,
        5 * GIB,
        8192,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
    assert_eq!(f.kv_bytes, GIB);
    assert_eq!(f.verdict, FitVerdict::Tight);
    // Widen the budget past the 90% line and it is an ordinary fit;
    // narrow it below the KV and it is refused.
    assert_eq!(
        fit(
            &shape,
            6 * GIB,
            8192,
            model_io::ExpertCacheSlots::Auto,
            DEFAULT_GUARD
        )
        .verdict,
        FitVerdict::Resident
    );
    assert_eq!(
        fit(
            &shape,
            4 * GIB + 512 * MIB,
            8192,
            model_io::ExpertCacheSlots::Auto,
            DEFAULT_GUARD
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
    let f = fit(
        &dense,
        M4_MAX,
        8192,
        model_io::ExpertCacheSlots::Auto,
        DEFAULT_GUARD,
    );
    assert!(
        f.largest_context > 8192,
        "a 36 GB machine affords more than the protocol window: {}",
        f.largest_context
    );
    assert_eq!(
        f.largest_context,
        model_io::largest_context_within(&arch(ModelFamily::Llama), f.budget - f.resident_bytes),
        "and it is the engine's own answer, not a second formula"
    );
}

/// **THE CROSS-CRATE PIN.** `model_io` owns the tiers and cannot import this
/// crate to state what `Relaxed`'s tight threshold should be; this crate no
/// longer reads its own constant. So the two copies are held together here,
/// and this is the case that reddens if either moves alone.
#[test]
fn the_default_guard_reproduces_the_frozen_thresholds() {
    let b = DEFAULT_GUARD.budget();
    assert_eq!(b.tight_fraction, TIGHT_FRACTION);
    assert_eq!(b.reserve_bytes, model_io::CONTEXT_RESERVE_BYTES);
    assert_eq!(DEFAULT_GUARD, model_io::LoadGuard::default());
}

/// A tighter tier reserves more, so the same candidate on the same machine
/// crosses from comfortable to tight to refused without anything about the
/// candidate changing.
#[test]
fn a_tighter_guard_walks_one_candidate_down_the_ladder() {
    let gemma = gemma4_shape();
    // 13 GiB is where the tiers actually separate for this install: its
    // allocation is ~1.8 GiB (1.5 of slot cache at 16 slots plus KV), so
    // Strict's 12 GiB reserve leaves 1 GiB and refuses while Relaxed's 4 GiB
    // reserve leaves 9 and does not. Picked by arithmetic rather than by
    // taste -- a machine large enough for every tier proves nothing.
    let verdict = |guard| fit(&gemma, 13 * GIB, 4096, PROTOCOL_SLOTS, guard).verdict;
    assert!(verdict(model_io::LoadGuard::Off).runs());
    assert!(verdict(model_io::LoadGuard::Relaxed).runs());
    assert!(verdict(model_io::LoadGuard::Balanced).runs());
    assert_eq!(verdict(model_io::LoadGuard::Strict), FitVerdict::Refused);
}

/// `Off` still TIERS -- a caller wants to know a fit is tight even when
/// nothing will stop them -- and only declines to REFUSE.
#[test]
fn off_never_refuses_but_still_reports_the_tier() {
    let gemma = gemma4_shape();
    for machine in [4 * GIB, 8 * GIB, 20 * GIB, 64 * GIB] {
        let f = fit(
            &gemma,
            machine,
            4096,
            PROTOCOL_SLOTS,
            model_io::LoadGuard::Off,
        );
        assert_ne!(
            f.verdict,
            FitVerdict::Refused,
            "off refused on a {machine}-byte machine"
        );
    }
    // And the tiering is still live: a machine barely larger than the
    // allocation reports Tight rather than Resident.
    let counted = fit(
        &gemma,
        64 * GIB,
        4096,
        PROTOCOL_SLOTS,
        model_io::LoadGuard::Off,
    )
    .counted;
    let snug = fit(
        &gemma,
        counted + counted / 20,
        4096,
        PROTOCOL_SLOTS,
        model_io::LoadGuard::Off,
    );
    assert_eq!(snug.verdict, FitVerdict::Tight);
}

/// `Custom`'s ceiling refuses regardless of headroom: it is the user naming
/// an allocation they do not want exceeded, not a second estimate of what the
/// machine holds. The machine here has room to spare either way.
#[test]
fn a_custom_ceiling_refuses_independently_of_headroom() {
    let gemma = gemma4_shape();
    let baseline = fit(&gemma, M4_MAX, 4096, PROTOCOL_SLOTS, DEFAULT_GUARD);
    assert!(baseline.verdict.runs());
    let under = fit(
        &gemma,
        M4_MAX,
        4096,
        PROTOCOL_SLOTS,
        model_io::LoadGuard::Custom {
            max_counted_bytes: baseline.counted - 1,
        },
    );
    assert_eq!(under.verdict, FitVerdict::Refused);
    let over = fit(
        &gemma,
        M4_MAX,
        4096,
        PROTOCOL_SLOTS,
        model_io::LoadGuard::Custom {
            max_counted_bytes: baseline.counted + 1,
        },
    );
    assert_eq!(over.verdict, baseline.verdict);
}

/// The tier travels ON the fit, so a caller that substitutes a measured peak
/// and re-derives the verdict cannot silently apply a different one.
#[test]
fn the_fit_remembers_which_tier_produced_it() {
    for guard in [
        model_io::LoadGuard::Off,
        model_io::LoadGuard::Relaxed,
        model_io::LoadGuard::Strict,
    ] {
        assert_eq!(
            fit(&gemma4_shape(), M4_MAX, 4096, PROTOCOL_SLOTS, guard).guard,
            guard
        );
    }
}

/// **THE LADDER'S WHOLE REASON FOR EXISTING IS THAT KV IS NOT LINEAR IN THE
/// WINDOW**, so the guard is a comparison between two families rather than a
/// number for one. A dense install grows with every layer; Gemma 4's 25
/// sliding-window layers of 30 are rings capped at `sliding_window + 128` and
/// stop growing past it, so only its 5 full layers keep costing.
///
/// Measured 4,096 -> 131,072 on the shipped baselines: Mistral 7B 512 MiB to
/// 16,384 (32x, exactly the context ratio) against Gemma 4's 305 to 2,785
/// (9x). A caller multiplying its own 4,096 figure by 32 is 3.5x high on
/// Gemma, which refuses a configuration that runs.
#[test]
fn the_context_ladder_is_not_linear_and_the_two_families_prove_it() {
    let ratio = |family: model_io::ModelFamily| {
        let arch = model_io::known_architecture(family);
        let shape = Shape {
            install_bytes: 4 * GIB,
            expert_stride: None,
            arch: Some(arch),
            measured_counted: None,
        };
        let rungs = context_ladder(
            &shape,
            96 * GIB,
            model_io::ExpertCacheSlots::Fixed(16),
            model_io::LoadGuard::Off,
            None,
        );
        let at = |c: u32| {
            rungs
                .iter()
                .find(|r| r.context == c)
                .unwrap_or_else(|| panic!("{c} is a standard rung"))
                .kv_bytes as f64
        };
        at(131_072) / at(4_096)
    };

    let dense = ratio(model_io::ModelFamily::Llama);
    let sliding = ratio(model_io::ModelFamily::Gemma4);

    // A model with no sliding-window layers tracks the context ratio exactly.
    assert!(
        (dense - 32.0).abs() < 0.01,
        "a fully-attentive model is linear in the window, got {dense}"
    );
    // Gemma's rings cap it far below that. The bound is loose on purpose:
    // the POINT is that a linear model is badly wrong, not the third digit.
    assert!(
        sliding < 12.0,
        "sliding-window layers must stop growing, got {sliding}"
    );
    assert!(
        dense / sliding > 2.5,
        "the two families must not be interchangeable, got {dense} vs {sliding}"
    );
}

/// The ladder folds in the checkpoint's own two windows and marks them, and
/// it reports past-trained rather than hiding it -- RoPE extrapolates rather
/// than failing, and some checkpoints ship YaRN scaling meant to exceed the
/// trained window (Gotcha 55).
#[test]
fn the_ladder_marks_the_trained_window_and_reports_past_it() {
    let shape = Shape {
        install_bytes: 4 * GIB,
        expert_stride: None,
        arch: Some(model_io::known_architecture(model_io::ModelFamily::Llama)),
        measured_counted: None,
    };
    let rungs = context_ladder(
        &shape,
        96 * GIB,
        model_io::ExpertCacheSlots::Fixed(16),
        model_io::LoadGuard::Off,
        Some(40_960),
    );

    let trained: Vec<&LadderRung> = rungs.iter().filter(|r| r.is_trained_max).collect();
    assert_eq!(trained.len(), 1, "exactly one rung is the trained maximum");
    assert_eq!(trained[0].context, 40_960);
    assert!(
        !trained[0].past_trained,
        "the trained window is not past itself"
    );

    assert!(
        rungs.iter().any(|r| r.past_trained && r.context > 40_960),
        "a window past the trained one is reported, not dropped"
    );
    assert!(
        rungs.iter().all(|r| r.context <= 40_960 || r.past_trained),
        "every window above the trained one is marked"
    );
    // Sorted and unique, or a picker renders the same window twice.
    let contexts: Vec<u32> = rungs.iter().map(|r| r.context).collect();
    let mut sorted = contexts.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(contexts, sorted);
}

/// **NO SHAPE MEANS NO LADDER.** Filling it from the family baseline is what
/// Gotcha 10 refuses: `llama` alone covers Mixtral 8x7B, Mistral 7B and
/// TinyLlama 1.1B, whose layer counts and head dimensions differ.
#[test]
fn an_unprobed_shape_has_no_ladder_rather_than_a_guessed_one() {
    let shape = Shape {
        install_bytes: 13 * GIB,
        expert_stride: None,
        arch: None,
        measured_counted: None,
    };
    assert!(context_ladder(
        &shape,
        36 * GIB,
        model_io::ExpertCacheSlots::Auto,
        model_io::LoadGuard::default(),
        Some(40_960),
    )
    .is_empty());
}
