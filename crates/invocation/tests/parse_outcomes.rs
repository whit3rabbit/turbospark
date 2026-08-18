//! Accepted-outcome scenarios for the argument-translation contract.
//!
//! Corresponds to behavior-spec test-001, test-002, test-004, test-010,
//! test-012, test-014, test-016, and test-017.

use foundation::runtime_config::{ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES};
use turbospark_invocation::{
    parse, ExpertCacheSlots, InvocationRequest, MaxContext, Mode, ParseOutcome, PowerProfile,
    PrefillChunk, ReasoningEffort,
};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_success(outcome: ParseOutcome) -> InvocationRequest {
    match outcome {
        ParseOutcome::Success(req) => req,
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn minimal_prompt_invocation_uses_documented_defaults() {
    let req = expect_success(parse(&tok(&["--model", "m.bin", "--prompt", "hello"])));
    assert_eq!(req.model, "m.bin");
    assert_eq!(req.mode, Mode::Prompt("hello".to_string()));
    assert_eq!(req.max_new, 1024);
    assert_eq!(req.top_k, 64);
}

#[test]
fn explicit_generation_and_sampling_values_round_trip() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m.bin",
        "--prompt",
        "hello",
        "--max-new",
        "50",
        "--max-context",
        "2048",
        "--temperature",
        "0.7",
        "--top-k",
        "10",
        "--top-p",
        "1.0",
        "--repetition-penalty",
        "1.2",
        "--seed",
        "42",
        "--stop",
        "a",
        "--stop",
        "b",
        "--quiet",
    ])));
    assert_eq!(req.max_new, 50);
    assert_eq!(req.max_context, MaxContext::Fixed(2048));
    assert_eq!(req.temperature, 0.7);
    assert_eq!(req.top_k, 10);
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.repetition_penalty, 1.2);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.stop, vec!["a".to_string(), "b".to_string()]);
    assert!(req.quiet);
}

#[test]
fn top_k_disabled_reports_the_dedicated_disabled_state() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--prompt", "hi", "--top-k", "0",
    ])));
    assert_eq!(req.top_k, 0);
}

#[test]
fn messages_file_mode_leaves_prompt_unset() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--messages-file",
        "chat.json",
    ])));
    assert_eq!(req.mode, Mode::MessagesFile("chat.json".to_string()));
}

#[test]
fn chat_mode_selects_interactive_with_no_prompt_or_file() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(req.mode, Mode::Chat);
}

#[test]
fn repeated_system_messages_combine_in_supplied_order() {
    let req = expect_success(parse(&tok(&[
        "--model", "m", "--chat", "--system", "first", "--system", "second",
    ])));
    let combined = req.system.expect("system message expected");
    assert!(combined.find("first").unwrap() < combined.find("second").unwrap());
}

#[test]
fn every_documented_chunk_size_is_accepted() {
    for &size in ALLOWED_CHUNK_SIZES.iter() {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--prefill-chunk",
            &size.to_string(),
        ])));
        assert_eq!(req.prefill_chunk, PrefillChunk::Fixed(size));
    }
}

#[test]
fn automatic_chunk_sizing_keyword_selects_auto() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--prefill-chunk",
        "auto",
    ])));
    assert_eq!(req.prefill_chunk, PrefillChunk::Auto);
}

/// The slot flag now takes the same `auto`-or-a-member grammar
/// `--prefill-chunk` does, and unlike that one `auto` is also the DEFAULT
/// (`request_defaults.rs`). Every allowed count must still round-trip, since
/// pinning one is how every harness in the repo keeps its frozen rows
/// comparable.
#[test]
fn every_documented_slot_count_is_accepted_and_auto_is_a_value() {
    for &slots in ALLOWED_CACHE_SLOTS.iter() {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--expert-cache-slots",
            &slots.to_string(),
        ])));
        assert_eq!(req.expert_cache_slots, ExpertCacheSlots::Fixed(slots));
    }
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--expert-cache-slots",
        "auto",
    ])));
    assert_eq!(req.expert_cache_slots, ExpertCacheSlots::Auto);
}

/// The THIRD flag on the `auto`-or-a-value grammar, and the second whose
/// `auto` is the default. Unlike the slot count, this one takes an arbitrary
/// positive integer rather than a member of a published set: a context window
/// is a per-token allocation, so there is no allowed set to check against and
/// the only real bound is what memory holds -- which this pure crate cannot
/// look at, and `crates/runtime`'s policy checks instead.
#[test]
fn the_context_window_takes_a_count_or_the_auto_keyword() {
    for tokens in [4096u32, 1, 131_072] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--max-context",
            &tokens.to_string(),
        ])));
        assert_eq!(req.max_context, MaxContext::Fixed(tokens));
    }
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--max-context",
        "auto",
    ])));
    assert_eq!(req.max_context, MaxContext::Auto);
}

/// Zero is refused rather than treated as `auto`. A window of zero admits no
/// prompt at all, and the flag already has a spelling for "you decide".
#[test]
fn a_zero_or_unparsable_context_window_is_refused() {
    for bad in ["0", "-1", "many", "4096tokens", ""] {
        let outcome = parse(&tok(&["--model", "m", "--chat", "--max-context", bad]));
        assert!(
            matches!(outcome, ParseOutcome::Failure(_)),
            "--max-context {bad:?} should be refused"
        );
    }
}

/// Every documented level round-trips, and the default is the one that
/// changes no rendered byte.
///
/// The UNION is accepted here on purpose, `high` and `xhigh` both, even
/// though no single checkpoint takes both: this crate may not look at the
/// install, and the template that can validate says so by name. A typo is
/// still rejected, which is the line between "not this crate's question" and
/// "no question at all".
#[test]
fn reasoning_levels_translate_to_their_validated_values() {
    let req = expect_success(parse(&tok(&["--model", "m", "--chat"])));
    assert_eq!(
        req.reasoning,
        ReasoningEffort::Off,
        "the default must be the level that renders what every earlier release rendered"
    );

    for spelling in ["off", "low", "medium", "high", "xhigh"] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--reasoning",
            spelling,
        ])));
        assert_eq!(
            req.reasoning.as_str(),
            spelling,
            "reasoning spelling must round-trip"
        );
    }

    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--reasoning", "maximum"])),
        ParseOutcome::Failure(_)
    ));
    // `true` reads like a plausible spelling for a knob that used to be a
    // boolean everywhere upstream, and is not one here.
    assert!(matches!(
        parse(&tok(&["--model", "m", "--chat", "--reasoning", "true"])),
        ParseOutcome::Failure(_)
    ));
}

#[test]
fn power_profile_and_rate_cap_translate_to_their_validated_values() {
    let req = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--power-profile",
        "efficiency",
        "--max-tokens-per-sec",
        "8.5",
    ])));
    assert_eq!(req.power_profile, Some(PowerProfile::Efficiency));
    assert_eq!(req.max_tokens_per_sec, Some(8.5));

    // The two are independent: a cap without a profile is the documented
    // way to pace without changing thermal behavior.
    let rate_only = expect_success(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--max-tokens-per-sec",
        "3",
    ])));
    assert_eq!(rate_only.power_profile, None);
    assert_eq!(rate_only.max_tokens_per_sec, Some(3.0));

    for spelling in ["performance", "balanced", "efficiency"] {
        let req = expect_success(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--power-profile",
            spelling,
        ])));
        assert_eq!(
            req.power_profile.map(PowerProfile::as_str),
            Some(spelling),
            "profile spelling must round-trip"
        );
    }
}
