//! Failure-outcome scenarios for the argument-translation contract.
//!
//! Corresponds to behavior-spec test-003, test-005, test-007, test-008,
//! test-009, test-011, test-013, test-015, test-018, and test-019.

use turbospark_invocation::{parse, ParseFailure, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_failure(outcome: ParseOutcome) -> ParseFailure {
    match outcome {
        ParseOutcome::Failure(f) => f,
        other => panic!("expected failure, got {other:?}"),
    }
}

#[test]
fn disabled_top_k_with_explicit_sub_one_top_p_is_rejected() {
    let failure = expect_failure(parse(&tok(&[
        "--model", "m", "--prompt", "hi", "--top-k", "0", "--top-p", "0.5",
    ])));
    assert!(matches!(
        failure,
        ParseFailure::InvalidValue {
            option: "--top-p",
            ..
        }
    ));
}

#[test]
fn top_k_above_maximum_names_option_and_offending_text() {
    let failure = expect_failure(parse(&tok(&["--model", "m", "--chat", "--top-k", "257"])));
    match failure {
        ParseFailure::InvalidValue { option, value } => {
            assert_eq!(option, "--top-k");
            assert_eq!(value, "257");
        }
        other => panic!("unexpected failure: {other:?}"),
    }
}

#[test]
fn unrecognized_token_and_single_character_form_are_both_unknown() {
    let f1 = expect_failure(parse(&tok(&["--not-an-option"])));
    assert!(matches!(f1, ParseFailure::UnknownOption { token } if token == "--not-an-option"));

    let f2 = expect_failure(parse(&tok(&["-m", "m"])));
    assert!(matches!(f2, ParseFailure::UnknownOption { token } if token == "-m"));
}

#[test]
fn missing_model_reports_required_option_failure() {
    let failure = expect_failure(parse(&tok(&["--prompt", "hi"])));
    assert!(matches!(
        failure,
        ParseFailure::MissingRequired { option: "--model" }
    ));
}

#[test]
fn no_mode_selected_is_its_own_failure() {
    let failure = expect_failure(parse(&tok(&["--model", "m"])));
    assert!(matches!(failure, ParseFailure::NoModeSelected));
}

#[test]
fn prompt_and_messages_file_together_are_mutually_exclusive() {
    let failure = expect_failure(parse(&tok(&[
        "--model",
        "m",
        "--prompt",
        "hi",
        "--messages-file",
        "f",
    ])));
    assert!(matches!(
        failure,
        ParseFailure::MutuallyExclusive {
            first: "--prompt",
            second: "--messages-file"
        }
    ));
}

#[test]
fn chat_paired_with_either_other_mode_is_mutually_exclusive() {
    let with_prompt = expect_failure(parse(&tok(&["--model", "m", "--chat", "--prompt", "hi"])));
    assert!(matches!(
        with_prompt,
        ParseFailure::MutuallyExclusive {
            first: "--prompt",
            second: "--chat"
        }
    ));

    let with_file = expect_failure(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--messages-file",
        "f",
    ])));
    assert!(matches!(
        with_file,
        ParseFailure::MutuallyExclusive {
            first: "--messages-file",
            second: "--chat"
        }
    ));
}

#[test]
fn system_message_outside_chat_mode_is_rejected() {
    let failure = expect_failure(parse(&tok(&[
        "--model", "m", "--prompt", "hi", "--system", "hello",
    ])));
    assert!(matches!(
        failure,
        ParseFailure::InvalidValue {
            option: "--system",
            ..
        }
    ));
}

#[test]
fn chunk_size_outside_allowed_set_is_rejected_for_every_bad_value() {
    for bad in ["0", "999999", "-4", "not-a-number"] {
        let failure = expect_failure(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--prefill-chunk",
            bad,
        ])));
        match failure {
            ParseFailure::InvalidValue { option, value } => {
                assert_eq!(option, "--prefill-chunk");
                assert_eq!(value, bad);
            }
            other => panic!("unexpected failure for {bad}: {other:?}"),
        }
    }
}

#[test]
fn cache_slots_and_rdadvise_outside_their_sets_are_rejected() {
    let slots = expect_failure(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--expert-cache-slots",
        "17",
    ])));
    assert!(matches!(
        slots,
        ParseFailure::InvalidValue {
            option: "--expert-cache-slots",
            ..
        }
    ));

    let advise = expect_failure(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--rdadvise",
        "turbo",
    ])));
    assert!(matches!(
        advise,
        ParseFailure::InvalidValue {
            option: "--rdadvise",
            ..
        }
    ));
}

#[test]
fn power_control_values_outside_their_documented_sets_are_rejected() {
    let profile = expect_failure(parse(&tok(&[
        "--model",
        "m",
        "--chat",
        "--power-profile",
        "turbo",
    ])));
    assert!(matches!(
        profile,
        ParseFailure::InvalidValue {
            option: "--power-profile",
            ..
        }
    ));

    // A rate must be a positive, finite number of tokens per second. Zero
    // and negatives have no meaning as a rate, and the non-finite ones
    // parse as `f64` but cannot become an interval.
    for bad in ["0", "-1", "abc", "inf", "NaN"] {
        let rate = expect_failure(parse(&tok(&[
            "--model",
            "m",
            "--chat",
            "--max-tokens-per-sec",
            bad,
        ])));
        assert!(
            matches!(
                rate,
                ParseFailure::InvalidValue {
                    option: "--max-tokens-per-sec",
                    ..
                }
            ),
            "expected {bad} to be rejected"
        );
    }
}
