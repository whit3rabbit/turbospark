//! Usage-text completeness and outcome-to-status/stream routing.
//!
//! Corresponds to behavior-spec test-006 (obl-parse-006) and the
//! diagnostic-surface obligation obl-parse-diag-001.

use mrefrust_invocation::diagnostics::{exit_status, stream_routing, ExitStatus};
use mrefrust_invocation::options::OPTIONS;
use mrefrust_invocation::{parse, render_usage, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn help_short_circuits_with_no_other_tokens() {
    let outcome = parse(&tok(&["--help"]));
    assert_eq!(outcome, ParseOutcome::Help);
}

#[test]
fn a_full_invocation_with_every_other_option_still_parses_to_success_not_help() {
    let outcome = parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--system",
        "hi",
        "--max-new",
        "10",
        "--max-context",
        "100",
        "--temperature",
        "0.5",
        "--top-k",
        "5",
        "--top-p",
        "1.0",
        "--repetition-penalty",
        "1.0",
        "--seed",
        "1",
        "--stop",
        "s",
        "--rdadvise",
        "normal",
        "--expert-cache-slots",
        "16",
        "--prefill-chunk",
        "128",
        "--quiet",
    ]));
    assert!(matches!(outcome, ParseOutcome::Success(_)));
}

#[test]
fn usage_text_enumerates_every_declared_option_with_a_default_or_allowed_value_description() {
    let text = render_usage();
    for opt in OPTIONS {
        assert!(
            text.contains(opt.flag),
            "usage text missing flag {}",
            opt.flag
        );
        assert!(
            !opt.usage_hint.is_empty(),
            "option {} has no default or allowed-value description",
            opt.flag
        );
        assert!(
            text.contains(opt.usage_hint),
            "usage text missing description for {}",
            opt.flag
        );
    }
    let required_and_mode_options: Vec<&str> = OPTIONS
        .iter()
        .filter(|o| o.is_required || o.is_mode_selecting)
        .map(|o| o.flag)
        .collect();
    assert_eq!(
        required_and_mode_options.len(),
        4,
        "expected the required option plus the three mode-selecting options"
    );
    let other_options = OPTIONS.len() - required_and_mode_options.len();
    assert_eq!(
        other_options, 14,
        "expected fourteen remaining documented options"
    );
}

#[test]
fn success_and_help_map_to_the_success_status_failure_maps_to_invalid_status() {
    let success = parse(&tok(&["--model", "m", "--chat"]));
    assert_eq!(exit_status(&success), ExitStatus::Success);
    assert_eq!(exit_status(&success).code(), 0);

    let help = parse(&tok(&["--help"]));
    assert_eq!(exit_status(&help), ExitStatus::Success);
    assert_eq!(exit_status(&help).code(), 0);

    let failure = parse(&tok(&["--nope"]));
    assert_eq!(exit_status(&failure), ExitStatus::InvalidInvocation);
    assert_eq!(exit_status(&failure).code(), 2);
}

#[test]
fn stream_routing_matches_the_documented_split() {
    let success = parse(&tok(&["--model", "m", "--chat"]));
    let routing = stream_routing(&success);
    assert!(routing.primary.is_none());
    assert!(routing.diagnostic.is_none());

    let help = parse(&tok(&["--help"]));
    let routing = stream_routing(&help);
    assert!(routing.primary.is_some());
    assert!(routing.diagnostic.is_none());

    let failure = parse(&tok(&["--nope"]));
    let routing = stream_routing(&failure);
    assert!(routing.primary.is_none());
    assert!(routing.diagnostic.is_some());
}
