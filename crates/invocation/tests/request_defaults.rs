//! Invocation value defaults and failure-shape checks.
//!
//! Corresponds to the work-invocation-value-model acceptance criteria: the
//! documented defaults are asserted, and the six failure values are shown
//! distinguishable without inspecting any message string.

use foundation::runtime_config::DEFAULT_CHUNK_SIZE;
use turbospark_invocation::{
    parse, ExpertCacheSlots, KvBits, MaxContext, Mode, ParseFailure, ParseOutcome, PrefillChunk,
    ReadAheadMode, ReasoningEffort,
};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn documented_defaults_are_applied() {
    let outcome = parse(&tok(&["--model", "m.bin", "--prompt", "hi"]));
    let ParseOutcome::Success(req) = outcome else {
        panic!("expected success");
    };
    assert_eq!(req.model, "m.bin");
    assert_eq!(req.mode, Mode::Prompt("hi".to_string()));
    assert_eq!(req.system, None);
    assert_eq!(req.max_new, 1024);
    // `Auto`, for the same reason `expert_cache_slots` is and one this crate
    // is even less able to answer: the ceiling is the CHECKPOINT's trained
    // context, which lives in the install's manifest. What keeps the sensing
    // default safe is the resolver's rule that an install declaring none
    // resolves to 4,096, which is every install written before that field
    // existed.
    assert_eq!(req.max_context, MaxContext::Auto);
    assert_eq!(req.temperature, 0.2);
    assert_eq!(req.top_k, 64);
    assert_eq!(req.top_p, 0.95);
    assert_eq!(req.repetition_penalty, 1.0);
    assert_eq!(req.seed, None);
    assert!(req.stop.is_empty());
    assert!(!req.quiet);
    assert_eq!(req.rdadvise, ReadAheadMode::Off);
    // `Auto` rather than a literal count, and unlike `prefill_chunk` below
    // that is the whole point: the slot cache is a RAM-for-throughput trade
    // whose right answer depends on the machine and the install, neither of
    // which this pure crate may look at. Resolved in `crates/runtime`, where
    // it can never come out below `DEFAULT_CACHE_SLOTS`.
    assert_eq!(req.expert_cache_slots, ExpertCacheSlots::Auto);
    assert_eq!(req.prefill_chunk, PrefillChunk::Fixed(DEFAULT_CHUNK_SIZE));
    // Both power knobs default to unset rather than to a profile: the
    // Low Power Mode default is resolved downstream, where the OS can be
    // asked, and this crate performs no I/O.
    assert_eq!(req.power_profile, None);
    assert_eq!(req.max_tokens_per_sec, None);
    // Off rather than a level, and this one is not a "resolved downstream"
    // default like the two above: it is the value that renders the exact
    // bytes every release before the flag rendered, which is what keeps
    // `crates/bench`'s frozen digests where they are.
    assert_eq!(req.reasoning, ReasoningEffort::Off);
    // No sidecar unless one is named: a text-only session on a text-only
    // trunk pays nothing for this feature.
    assert_eq!(req.vision_sidecar, None);
    // Off, not a sensing default: this is the flag whose whole point is that
    // an opt-out caller reproduces the exact bytes every release before it
    // existed produced (docs/TRUBOQUANT.md).
    assert_eq!(req.kv_bits, KvBits::Off);
}

#[test]
fn six_failure_values_are_distinguishable_by_shape() {
    let unknown = parse(&tok(&["--nope"]));
    assert!(matches!(
        unknown,
        ParseOutcome::Failure(ParseFailure::UnknownOption { .. })
    ));

    let missing_value = parse(&tok(&["--model"]));
    assert!(matches!(
        missing_value,
        ParseOutcome::Failure(ParseFailure::MissingValue { .. })
    ));

    let invalid_value = parse(&tok(&["--model", "m", "--chat", "--temperature", "-1"]));
    assert!(matches!(
        invalid_value,
        ParseOutcome::Failure(ParseFailure::InvalidValue { .. })
    ));

    let missing_required = parse(&tok(&["--chat"]));
    assert!(matches!(
        missing_required,
        ParseOutcome::Failure(ParseFailure::MissingRequired { .. })
    ));

    let mutually_exclusive = parse(&tok(&["--model", "m", "--chat", "--prompt", "hi"]));
    assert!(matches!(
        mutually_exclusive,
        ParseOutcome::Failure(ParseFailure::MutuallyExclusive { .. })
    ));

    let no_mode = parse(&tok(&["--model", "m"]));
    assert!(matches!(
        no_mode,
        ParseOutcome::Failure(ParseFailure::NoModeSelected)
    ));
}
