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
        model_io::largest_context_within(&arch(ModelFamily::Llama), f.budget - f.resident_bytes),
        "and it is the engine's own answer, not a second formula"
    );
}
