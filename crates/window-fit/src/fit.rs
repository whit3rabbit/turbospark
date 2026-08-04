//! The conversation-window fitting operation.

use crate::outcome::WindowFitOutcome;

/// Fit a conversation into a measured length bound.
///
/// `turns` is the conversation in order, oldest first. `has_leading_instruction`
/// marks whether `turns[0]` is an optional leading instruction turn.
/// `bound` is the length that the retained conversation must measure under
/// to leave room for at least one generated unit. `measure` is a
/// caller-supplied callback that measures the whole remaining conversation
/// as a single value; it is never asked to measure a single turn in
/// isolation.
///
/// The whole conversation is measured once before any removal, and
/// re-measured after each removal. Removal takes the oldest eligible turn
/// first and stops as soon as the measured length is under the bound or as
/// soon as nothing eligible remains. An optional leading instruction turn
/// and the newest turn are never removed, so an outcome that is still at or
/// over the bound is a normal result, not an error.
///
/// This operation performs no input or output, holds no state between
/// calls, and depends on nothing but its arguments: the same arguments
/// always produce the same outcome.
pub fn fit_conversation_window<T, M>(
    turns: &[T],
    has_leading_instruction: bool,
    bound: u64,
    mut measure: M,
) -> WindowFitOutcome<T>
where
    T: Clone,
    M: FnMut(&[T]) -> u64,
{
    let mut retained: Vec<T> = turns.to_vec();
    let mut removed_turn_count = 0usize;
    let mut measured_length = measure(&retained);

    while measured_length >= bound {
        match next_removable_index(retained.len(), has_leading_instruction) {
            Some(index) => {
                retained.remove(index);
                removed_turn_count += 1;
                measured_length = measure(&retained);
            }
            None => break,
        }
    }

    let room_for_generation = measured_length < bound;
    WindowFitOutcome::new(
        retained,
        measured_length,
        removed_turn_count,
        room_for_generation,
    )
}

/// The index of the oldest turn eligible for removal, or `None` when
/// nothing eligible remains.
///
/// The leading instruction turn (index 0, when present) and the newest
/// turn (the last index) are never eligible.
fn next_removable_index(len: usize, has_leading_instruction: bool) -> Option<usize> {
    let first_removable = if has_leading_instruction { 1 } else { 0 };
    let newest_index_exclusive = len.saturating_sub(1);
    if first_removable < newest_index_exclusive {
        Some(first_removable)
    } else {
        None
    }
}
