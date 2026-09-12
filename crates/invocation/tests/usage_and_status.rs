//! Usage-text completeness and outcome-to-status/stream routing.
//!
//! Corresponds to behavior-spec test-006 (obl-parse-006) and the
//! diagnostic-surface obligation obl-parse-diag-001.

use turbospark_invocation::diagnostics::{exit_status, stream_routing, ExitStatus};
use turbospark_invocation::options::OPTIONS;
use turbospark_invocation::{parse, render_usage, ParseOutcome};

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
        "--image",
        "page.png",
        "--image-batch",
        "--rdadvise",
        "normal",
        "--expert-cache-slots",
        "16",
        "--prefill-chunk",
        "128",
        "--power-profile",
        "balanced",
        "--reasoning",
        "low",
        "--max-tokens-per-sec",
        "12.5",
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
        other_options, 33,
        "expected thirty-three remaining documented options"
    );
}

/// `--version` short-circuits exactly as `--help` does: it wins over a
/// missing `--model`, over an unknown option AFTER it, and over the
/// mode-selection check -- because the scan returns at the token rather than
/// completing and then reporting. What it must NOT do is print usage.
#[test]
fn version_short_circuits_the_scan_and_prints_a_version_rather_than_usage() {
    for tokens in [
        vec!["--version"],
        vec!["--version", "--nope"],
        vec!["--model", "m", "--version"],
    ] {
        let outcome = parse(&tok(&tokens));
        assert_eq!(outcome, ParseOutcome::Version, "{tokens:?}");
        assert_eq!(exit_status(&outcome), ExitStatus::Success);
        assert_eq!(exit_status(&outcome).code(), 0);

        let routing = stream_routing(&outcome);
        let primary = routing
            .primary
            .expect("a version goes to the primary stream");
        assert!(
            primary.contains(turbospark_invocation::VERSION),
            "{primary}"
        );
        assert!(
            !primary.contains("--model"),
            "a version request must not print the usage table: {primary}"
        );
        assert_eq!(routing.diagnostic, None);
    }
}

/// The version is cargo's, not a literal. A hand-maintained copy is a second
/// place to forget on a release, and the failure is a binary confidently
/// reporting the wrong number.
#[test]
fn the_version_comes_from_cargo() {
    assert_eq!(turbospark_invocation::VERSION, env!("CARGO_PKG_VERSION"));
    assert!(!turbospark_invocation::VERSION.is_empty());
}

/// Whichever short-circuit is reached FIRST wins, which is the documented
/// left-to-right scan rather than a precedence between the two.
#[test]
fn the_first_short_circuit_reached_is_the_one_taken() {
    assert_eq!(parse(&tok(&["--help", "--version"])), ParseOutcome::Help);
    assert_eq!(parse(&tok(&["--version", "--help"])), ParseOutcome::Version);
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
