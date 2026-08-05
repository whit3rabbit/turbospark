//! Integration tests for the window-fit outcome data contract.

use mrefrust_window_fit::fit_conversation_window;

#[test]
fn outcome_reports_retained_turns_in_original_conversation_order() {
    let turns = ["a", "b", "c", "d", "e"];
    let outcome = fit_conversation_window(&turns, false, 2, |t: &[&str]| t.len() as u64);
    assert_eq!(outcome.retained_turns(), &["e"]);
}

#[test]
fn outcome_measured_length_matches_a_fresh_measurement_of_the_retained_turns() {
    let turns = [3u64, 5, 2, 9];
    let outcome = fit_conversation_window(&turns, false, 4, |t: &[u64]| t.iter().sum());
    let expected: u64 = outcome.retained_turns().iter().sum();
    assert_eq!(outcome.measured_length(), expected);
}

#[test]
fn outcome_reports_room_for_generation_exactly_when_measured_length_is_under_the_bound() {
    let turns = ["x", "y"];
    let outcome = fit_conversation_window(&turns, false, 10, |t: &[&str]| t.len() as u64);
    assert!(outcome.has_room_for_generation());
    assert!(outcome.measured_length() < 10);
}

#[test]
fn outcome_reports_no_room_for_generation_when_measured_length_meets_or_exceeds_the_bound() {
    let turns = ["only"];
    let outcome = fit_conversation_window(&turns, false, 1, |t: &[&str]| t.len() as u64);
    assert!(!outcome.has_room_for_generation());
    assert!(outcome.measured_length() >= 1);
}

#[test]
fn removed_turn_count_plus_retained_turn_count_equals_the_starting_turn_count() {
    let turns = ["a", "b", "c", "d", "e", "f"];
    let outcome = fit_conversation_window(&turns, false, 3, |t: &[&str]| t.len() as u64);
    assert_eq!(
        outcome.retained_turns().len() + outcome.removed_turn_count(),
        turns.len()
    );
}

#[test]
fn two_identical_calls_produce_equal_outcomes() {
    let turns = ["one", "two", "three", "four"];
    let a = fit_conversation_window(&turns, true, 2, |t: &[&str]| t.len() as u64);
    let b = fit_conversation_window(&turns, true, 2, |t: &[&str]| t.len() as u64);
    assert_eq!(a, b);
}
