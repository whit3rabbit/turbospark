//! Failure-outcome scenarios for the argument-translation contract.
//!
//! Corresponds to behavior-spec test-003, test-005, test-007, test-008,
//! test-009, test-011, test-013, test-015, test-018, and test-019.

use turbospark_invocation::{parse, ParseFailure, ParseOutcome};

fn tok(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_invalid_value(outcome: ParseOutcome, want_option: &str) {
    match expect_failure(outcome) {
        ParseFailure::InvalidValue { option, .. } => assert_eq!(option, want_option),
        other => panic!("expected an InvalidValue for {want_option}, got {other:?}"),
    }
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

// --- directional steering (docs/OBLITERATION.md) --------------------------

/// An unknown mode is refused rather than defaulting. A caller who asked for
/// one edit and silently got another would measure the wrong model.
#[test]
fn an_unknown_steering_mode_is_refused() {
    expect_invalid_value(
        parse(&tok(&[
            "--model",
            "m.bin",
            "--prompt",
            "hi",
            "--steering-mode",
            "obliterate",
        ])),
        "--steering-mode",
    );
}

/// NaN and infinity are refused at the parser. A non-finite alpha puts a NaN
/// into the residual stream, and NaN reads as a PERFECT score on every rank
/// or top-k instrument downstream (AGENTS.md Gotcha 59) -- the one wrong
/// value nothing reports as wrong.
#[test]
fn a_non_finite_steering_scale_is_refused() {
    for bad in ["nan", "inf", "-inf"] {
        expect_invalid_value(
            parse(&tok(&[
                "--model",
                "m.bin",
                "--prompt",
                "hi",
                "--steering-scale",
                bad,
            ])),
            "--steering-scale",
        );
    }
}

/// An inverted range would steer nothing. Refused rather than accepted as an
/// empty selection, so an asked-for edit cannot become a silent no-op.
#[test]
fn an_inverted_steering_layer_range_is_refused() {
    expect_invalid_value(
        parse(&tok(&[
            "--model",
            "m.bin",
            "--prompt",
            "hi",
            "--steering-layers",
            "40:30",
        ])),
        "--steering-layers",
    );
}

#[test]
fn a_malformed_steering_layer_range_is_refused() {
    for bad in ["30", "30-40", "a:b", "30:", ":40"] {
        expect_invalid_value(
            parse(&tok(&[
                "--model",
                "m.bin",
                "--prompt",
                "hi",
                "--steering-layers",
                bad,
            ])),
            "--steering-layers",
        );
    }
}

/// A negative gate is meaningless (the kernel treats non-positive as "always
/// fire"), so it is refused rather than silently reinterpreted.
#[test]
fn a_negative_steering_gate_is_refused() {
    expect_invalid_value(
        parse(&tok(&[
            "--model",
            "m.bin",
            "--prompt",
            "hi",
            "--steering-gate",
            "-1.0",
        ])),
        "--steering-gate",
    );
}

/// A steering PARAMETER without `--steering` is refused rather than ignored,
/// on the same ground the inverted range above is: the run would decode
/// unsteered while the command line says otherwise, so a caller measuring an
/// edit would measure the engine without one.
///
/// All five are checked because they reach the request by three different
/// routes -- three are `Option` fields, and `--steering-target` and
/// `--steering-gate` are plain floats legal AT their 0.0 default, so those two
/// need an explicit-supplied flag before the check can see them at all. A
/// version of this test covering only the `Option` three would leave that half
/// unguarded.
#[test]
fn a_steering_parameter_without_a_vector_is_refused() {
    for (flag, value) in [
        ("--steering-mode", "renorm"),
        ("--steering-scale", "0.8"),
        ("--steering-layers", "30:40"),
        ("--steering-target", "2.5"),
        ("--steering-gate", "0.5"),
    ] {
        expect_invalid_value(
            parse(&tok(&["--model", "m.bin", "--prompt", "hi", flag, value])),
            flag,
        );
    }
}

/// The check must not fire on the values those two flags carry by DEFAULT.
/// Passing `--steering-target 0` explicitly is still an orphan and is refused
/// above; passing nothing is an ordinary invocation and must parse.
#[test]
fn an_invocation_with_no_steering_flags_at_all_still_parses() {
    assert!(matches!(
        parse(&tok(&["--model", "m.bin", "--prompt", "hi"])),
        ParseOutcome::Success(_)
    ));
}
