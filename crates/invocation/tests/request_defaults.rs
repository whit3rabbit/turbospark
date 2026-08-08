//! Invocation value defaults and failure-shape checks.
//!
//! Corresponds to the work-invocation-value-model acceptance criteria: the
//! documented defaults are asserted, and the six failure values are shown
//! distinguishable without inspecting any message string.

use foundation::runtime_config::{DEFAULT_CACHE_SLOTS, DEFAULT_CHUNK_SIZE};
use turbospark_invocation::{parse, Mode, ParseFailure, ParseOutcome, PrefillChunk, ReadAheadMode};

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
    assert_eq!(req.max_context, 4096);
    assert_eq!(req.temperature, 0.2);
    assert_eq!(req.top_k, 64);
    assert_eq!(req.top_p, 0.95);
    assert_eq!(req.repetition_penalty, 1.0);
    assert_eq!(req.seed, None);
    assert!(req.stop.is_empty());
    assert!(!req.quiet);
    assert_eq!(req.rdadvise, ReadAheadMode::Off);
    assert_eq!(req.expert_cache_slots, DEFAULT_CACHE_SLOTS);
    assert_eq!(req.prefill_chunk, PrefillChunk::Fixed(DEFAULT_CHUNK_SIZE));
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
