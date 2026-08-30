//! Configuration-validation rejection scenarios.
//!
//! Corresponds to behavior-spec test-008 (obl-select-008): each invalid
//! configuration is rejected at construction time with a distinguishable
//! descriptive error, and no selection is attempted.

use turbospark_selection::ShapingConfig;

#[test]
fn non_finite_or_negative_temperature_is_rejected() {
    assert!(ShapingConfig::new(f64::NAN, 64, Some(0.95), 1.0, None).is_err());
    assert!(ShapingConfig::new(f64::INFINITY, 64, Some(0.95), 1.0, None).is_err());
    assert!(ShapingConfig::new(-0.1, 64, Some(0.95), 1.0, None).is_err());
}

#[test]
fn top_k_outside_bounded_range_is_rejected() {
    assert!(ShapingConfig::new(0.7, 257, Some(0.95), 1.0, None).is_err());
}

#[test]
fn top_p_outside_zero_to_one_range_is_rejected() {
    assert!(ShapingConfig::new(0.7, 64, Some(0.0), 1.0, None).is_err());
    assert!(ShapingConfig::new(0.7, 64, Some(-0.1), 1.0, None).is_err());
    assert!(ShapingConfig::new(0.7, 64, Some(1.1), 1.0, None).is_err());
    assert!(ShapingConfig::new(0.7, 64, Some(f64::NAN), 1.0, None).is_err());
}

#[test]
fn positive_temperature_with_sub_one_top_p_and_no_top_k_is_rejected() {
    let result = ShapingConfig::new(0.7, 0, Some(0.5), 1.0, None);
    assert!(result.is_err());
}

#[test]
fn non_finite_or_non_positive_repetition_penalty_is_rejected() {
    assert!(ShapingConfig::new(0.7, 64, Some(0.95), 0.0, None).is_err());
    assert!(ShapingConfig::new(0.7, 64, Some(0.95), -1.0, None).is_err());
    assert!(ShapingConfig::new(0.7, 64, Some(0.95), f64::NAN, None).is_err());
}

#[test]
fn each_rejection_carries_a_descriptive_reason() {
    let err = ShapingConfig::new(f64::NAN, 64, Some(0.95), 1.0, None).unwrap_err();
    assert!(!err.reason.is_empty());
}

#[test]
fn valid_configurations_are_accepted() {
    assert!(ShapingConfig::new(0.7, 64, Some(0.95), 1.0, Some(1)).is_ok());
    assert!(ShapingConfig::new(0.0, 0, None, 1.0, None).is_ok());
    // Sub-one top_p paired with top_k enabled is fine even at a positive
    // temperature.
    assert!(ShapingConfig::new(0.7, 10, Some(0.5), 1.0, None).is_ok());
}

fn base() -> ShapingConfig {
    ShapingConfig::new(0.7, 64, Some(0.95), 1.0, None).unwrap()
}

#[test]
fn presence_penalty_outside_the_documented_range_is_rejected() {
    assert!(base().with_presence_penalty(f64::NAN).is_err());
    assert!(base().with_presence_penalty(f64::INFINITY).is_err());
    assert!(base().with_presence_penalty(-2.1).is_err());
    assert!(base().with_presence_penalty(2.1).is_err());
}

#[test]
fn presence_penalty_at_the_documented_bounds_is_accepted() {
    assert!(base().with_presence_penalty(-2.0).is_ok());
    assert!(base().with_presence_penalty(2.0).is_ok());
    assert!(base().with_presence_penalty(0.0).is_ok());
}

#[test]
fn frequency_penalty_outside_the_documented_range_is_rejected() {
    assert!(base().with_frequency_penalty(f64::NAN).is_err());
    assert!(base().with_frequency_penalty(f64::INFINITY).is_err());
    assert!(base().with_frequency_penalty(-2.1).is_err());
    assert!(base().with_frequency_penalty(2.1).is_err());
}

#[test]
fn frequency_penalty_at_the_documented_bounds_is_accepted() {
    assert!(base().with_frequency_penalty(-2.0).is_ok());
    assert!(base().with_frequency_penalty(2.0).is_ok());
}

#[test]
fn min_p_outside_zero_to_one_exclusive_is_rejected() {
    assert!(base().with_min_p(f64::NAN).is_err());
    assert!(base().with_min_p(f64::INFINITY).is_err());
    assert!(base().with_min_p(-0.1).is_err());
    // `1.0` is refused rather than accepted as "keep only the top
    // candidate": see `ShapingConfig::with_min_p`'s own doc for why.
    assert!(base().with_min_p(1.0).is_err());
    assert!(base().with_min_p(1.1).is_err());
}

#[test]
fn min_p_zero_disables_it_and_a_sub_one_positive_value_is_accepted() {
    let disabled = base().with_min_p(0.0).unwrap();
    assert_eq!(disabled.min_p(), None);
    let enabled = base().with_min_p(0.5).unwrap();
    assert_eq!(enabled.min_p(), Some(0.5));
    assert!(base().with_min_p(0.999).is_ok());
}
