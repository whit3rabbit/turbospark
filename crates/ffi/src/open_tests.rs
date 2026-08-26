use super::*;

fn json(text: &str) -> Option<serde_json::Value> {
    Some(serde_json::from_str(text).expect("fixture must be valid JSON"))
}

#[test]
fn an_absent_speculation_is_auto() {
    // The DEFAULT, and the one every other front end shares: a GUI that
    // sends `{}` gets what `turbospark-check` and `turbospark-server`
    // give with no flags.
    assert_eq!(speculation(&None).unwrap(), runtime::Speculation::Auto);
    assert_eq!(
        speculation(&json("null")).unwrap(),
        runtime::Speculation::Auto
    );
}

#[test]
fn a_block_is_accepted_as_a_number_and_as_a_string() {
    // Both spellings, for `sized`'s reason: a Swift enum encoding a
    // mixed number/string value may choose either, and two meanings for
    // one intent is a trap invisible from the header.
    assert_eq!(
        speculation(&json("4")).unwrap(),
        runtime::Speculation::Block(4)
    );
    assert_eq!(
        speculation(&json("\"4\"")).unwrap(),
        runtime::Speculation::Block(4)
    );
}

#[test]
fn off_and_auto_are_case_insensitive_and_distinct() {
    assert_eq!(
        speculation(&json("\"OFF\"")).unwrap(),
        runtime::Speculation::Off
    );
    assert_eq!(
        speculation(&json("\"Auto\"")).unwrap(),
        runtime::Speculation::Auto
    );
}

#[test]
fn a_block_outside_the_shared_range_is_refused_and_names_it() {
    // The range is `foundation`'s and is not restated here, so this
    // reads it back rather than pinning a literal: the assertion is
    // that a block past the end is refused AND that the message says
    // what the end is.
    let allowed = foundation::runtime_config::ALLOWED_SPECULATION_BLOCKS;
    let past = allowed.end() + 1;
    let err = speculation(&json(&past.to_string())).unwrap_err();
    assert!(
        err.contains(&format!("{allowed:?}")) && err.contains(&past.to_string()),
        "expected the range and the value, got {err}"
    );
    assert!(speculation(&json("0")).is_err(), "0 proposes nothing");
}

#[test]
fn a_misspelling_is_an_error_rather_than_a_silent_auto() {
    // `"of"` should be heard about. Falling back to the default here
    // would turn a typo into a session that quietly did something else.
    assert!(speculation(&json("\"of\"")).is_err());
    assert!(speculation(&json("true")).is_err());
    assert!(drafter(Some("mpt")).is_err());
}

#[test]
fn the_drafter_spellings_are_the_cli_and_server_ones() {
    assert_eq!(drafter(None).unwrap(), runtime::SpeculativeDrafter::Auto);
    assert_eq!(
        drafter(Some("auto")).unwrap(),
        runtime::SpeculativeDrafter::Auto
    );
    assert_eq!(
        drafter(Some("mtp")).unwrap(),
        runtime::SpeculativeDrafter::Mtp
    );
    assert_eq!(
        drafter(Some("dflash")).unwrap(),
        runtime::SpeculativeDrafter::Dflash
    );
    // Round-trips through the reported name, so a GUI reading
    // `sessionInfo.speculation.drafter` can hand it straight back as
    // `speculativeDrafter` on the next open.
    for name in ["mtp", "dflash"] {
        assert_eq!(drafter_name(drafter(Some(name)).unwrap()), name);
    }
}

#[test]
fn the_steering_mode_spellings_are_the_shared_ones() {
    assert_eq!(
        steering_mode("ablate").unwrap(),
        foundation::SteeringMode::Ablate
    );
    assert_eq!(steering_mode("add").unwrap(), foundation::SteeringMode::Add);
    assert_eq!(
        steering_mode("clamp").unwrap(),
        foundation::SteeringMode::Clamp
    );
    assert_eq!(
        steering_mode("renorm").unwrap(),
        foundation::SteeringMode::Renorm
    );
    assert!(steering_mode("unknown").is_err());
}

#[test]
fn layer_range_requires_start_and_end_in_order() {
    assert_eq!(layer_range("0:31").unwrap(), (0, 31));
    assert_eq!(layer_range(" 5 : 10 ").unwrap(), (5, 10));
    assert!(layer_range("10:5").is_err());
    assert!(layer_range("10").is_err());
    assert!(layer_range("10:abc").is_err());
}

#[test]
fn steering_scalars_validate_bounds() {
    assert_eq!(steering_scale(None).unwrap(), 1.0);
    assert_eq!(steering_scale(Some(2.5)).unwrap(), 2.5);
    assert!(steering_scale(Some(f64::NAN)).is_err());

    assert_eq!(steering_target(None).unwrap(), 0.0);
    assert_eq!(steering_target(Some(-1.0)).unwrap(), -1.0);
    assert!(steering_target(Some(f64::INFINITY)).is_err());

    assert_eq!(steering_gate(None).unwrap(), 0.0);
    assert_eq!(steering_gate(Some(0.5)).unwrap(), 0.5);
    assert!(steering_gate(Some(-0.1)).is_err());
}
