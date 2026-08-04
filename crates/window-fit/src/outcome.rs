//! The reported outcome of a conversation-window fit.

/// The result of fitting a conversation into a length bound.
///
/// Carries the retained turns in their original order, the length measured
/// over that retained conversation, how many turns were removed to get
/// there, and whether the retained conversation leaves room for at least
/// one more generated unit. An outcome whose measured length is still at or
/// over the bound is a normal, reportable result; it is never discarded or
/// forced to fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFitOutcome<T> {
    retained_turns: Vec<T>,
    measured_length: u64,
    removed_turn_count: usize,
    room_for_generation: bool,
}

impl<T> WindowFitOutcome<T> {
    pub(crate) fn new(
        retained_turns: Vec<T>,
        measured_length: u64,
        removed_turn_count: usize,
        room_for_generation: bool,
    ) -> Self {
        Self {
            retained_turns,
            measured_length,
            removed_turn_count,
            room_for_generation,
        }
    }

    /// The retained turns, in their original conversation order.
    pub fn retained_turns(&self) -> &[T] {
        &self.retained_turns
    }

    /// The length measured over the retained conversation, from the last
    /// measurement taken (either the initial measurement, if nothing was
    /// removed, or the measurement taken after the last removal).
    pub fn measured_length(&self) -> u64 {
        self.measured_length
    }

    /// How many turns were removed to reach this outcome.
    pub fn removed_turn_count(&self) -> usize {
        self.removed_turn_count
    }

    /// Whether the retained conversation leaves room for at least one more
    /// generated unit under the bound that was fitted against.
    pub fn has_room_for_generation(&self) -> bool {
        self.room_for_generation
    }
}
