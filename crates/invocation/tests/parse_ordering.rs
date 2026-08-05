//! Ordering, precedence, and purity scenarios for the argument-translation
//! contract.
//!
//! Corresponds to behavior-spec test-020, test-021, test-022, test-023, and
//! test-024.

use mrefrust_invocation::{parse, ParseFailure, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_value_token_that_looks_like_an_option_is_consumed_as_a_literal_value() {
    let outcome = parse(&tok(&["--model", "--prompt", "--prompt", "hi"]));
    match outcome {
        ParseOutcome::Success(req) => assert_eq!(req.model, "--prompt"),
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn a_value_taking_option_as_the_last_token_is_a_missing_value_failure() {
    let outcome = parse(&tok(&["--model", "m", "--chat", "--seed"]));
    assert!(matches!(
        outcome,
        ParseOutcome::Failure(ParseFailure::MissingValue { option: "--seed" })
    ));
}

#[test]
fn required_option_check_wins_over_mode_related_failures() {
    let neither = parse(&tok(&[]));
    assert!(matches!(
        neither,
        ParseOutcome::Failure(ParseFailure::MissingRequired { option: "--model" })
    ));

    let conflicting_modes = parse(&tok(&["--prompt", "hi", "--messages-file", "f"]));
    assert!(matches!(
        conflicting_modes,
        ParseOutcome::Failure(ParseFailure::MissingRequired { option: "--model" })
    ));
}

#[test]
fn help_in_the_middle_short_circuits_before_a_trailing_unrecognized_token() {
    let outcome = parse(&tok(&["--quiet", "--help", "--totally-unknown"]));
    assert_eq!(outcome, ParseOutcome::Help);
}

#[test]
fn an_earlier_invalid_value_fails_before_a_later_help_token_is_reached() {
    let outcome = parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--temperature",
        "-1",
        "--help",
    ]));
    assert!(matches!(
        outcome,
        ParseOutcome::Failure(ParseFailure::InvalidValue {
            option: "--temperature",
            ..
        })
    ));
}

#[test]
fn parsing_is_pure_and_repeatable() {
    let tokens = tok(&[
        "--model",
        "m",
        "--prompt",
        "hi",
        "--messages-file",
        "unused-if-rejected",
    ]);
    let first = parse(&tokens);
    let second = parse(&tokens);
    assert_eq!(first, second);
}
