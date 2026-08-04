//! Integration tests for the conversation-window fitting rules.

use window_fit::fit_conversation_window;

fn char_count_measure(turns: &[&str]) -> u64 {
    turns.iter().map(|t| t.len() as u64).sum()
}

fn turn_count_measure<T>(turns: &[T]) -> u64 {
    turns.len() as u64
}

#[test]
fn a_conversation_already_within_bound_is_left_untouched() {
    let turns = ["system", "hello", "hi there"];
    let outcome = fit_conversation_window(&turns, true, 1_000, char_count_measure);
    assert_eq!(outcome.retained_turns(), turns);
    assert_eq!(outcome.removed_turn_count(), 0);
    assert!(outcome.has_room_for_generation());
}

#[test]
fn measurement_runs_once_before_removal_and_once_after_each_removal() {
    let turns = ["a", "b", "c", "d"];
    let mut call_count = 0usize;
    let outcome = fit_conversation_window(&turns, false, 2, |t: &[&str]| {
        call_count += 1;
        turn_count_measure(t)
    });
    assert_eq!(outcome.removed_turn_count(), 3);
    // One initial measurement plus one measurement per removal.
    assert_eq!(call_count, 1 + outcome.removed_turn_count());
}

#[test]
fn removal_takes_the_oldest_eligible_turn_first() {
    let turns = ["oldest", "middle-1", "middle-2", "newest"];
    let outcome = fit_conversation_window(&turns, false, 3, turn_count_measure);
    assert_eq!(outcome.retained_turns(), &["middle-2", "newest"]);
}

#[test]
fn a_leading_instruction_turn_and_the_newest_turn_are_never_removed() {
    let turns = ["instruction", "turn1", "turn2", "turn3", "newest"];
    let outcome = fit_conversation_window(&turns, true, 1, turn_count_measure);
    assert_eq!(outcome.retained_turns(), &["instruction", "newest"]);
    assert_eq!(outcome.removed_turn_count(), 3);
}

#[test]
fn an_outcome_still_at_or_over_the_bound_is_returned_when_nothing_eligible_remains() {
    let turns = ["persistent instruction", "final reply"];
    let outcome = fit_conversation_window(&turns, true, 0, char_count_measure);
    assert_eq!(outcome.retained_turns(), turns);
    assert_eq!(outcome.removed_turn_count(), 0);
    assert!(!outcome.has_room_for_generation());
}

#[test]
fn a_single_newest_turn_with_no_leading_instruction_is_never_removed() {
    let turns = ["only turn"];
    let outcome = fit_conversation_window(&turns, false, 0, char_count_measure);
    assert_eq!(outcome.retained_turns(), turns);
    assert_eq!(outcome.removed_turn_count(), 0);
}

#[test]
fn an_empty_conversation_is_returned_unchanged() {
    let turns: [&str; 0] = [];
    let outcome = fit_conversation_window(&turns, false, 0, char_count_measure);
    assert!(outcome.retained_turns().is_empty());
    assert_eq!(outcome.removed_turn_count(), 0);
    assert_eq!(outcome.measured_length(), 0);
}
