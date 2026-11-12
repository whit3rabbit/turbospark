# turbospark-window-fit (src)

## Overview
Deterministic conversation-window fitting: drops the oldest eligible turns until a caller-measured length of the whole conversation is under a caller-supplied bound. The bound and the measurement are unitless `u64` values chosen by the caller. The crate never tokenizes and never inspects turn contents.

## Interface
- `fit_conversation_window<T, M>(turns: &[T], has_leading_instruction: bool, bound: u64, measure: M) -> WindowFitOutcome<T>` with `T: Clone` and `M: FnMut(&[T]) -> u64` [fit.rs:fit_conversation_window]. The input is copied with `turns.to_vec()`, so callers pay a clone of every turn.
- `WindowFitOutcome<T>` derives `Debug`, `Clone`, `PartialEq`, `Eq` [outcome.rs:WindowFitOutcome]. Its fields are private. Callers read it through accessors: `retained_turns()`, `measured_length()`, `removed_turn_count()`, `has_room_for_generation()`. The constructor `WindowFitOutcome::new` is `pub(crate)`.
- Private helper `next_removable_index(len, has_leading_instruction)` holds the only eligibility rule [fit.rs:next_removable_index].
- Module wiring: `pub mod fit; pub mod outcome;` with re-exports of both public names [lib.rs].

## Invariants
- The optional leading instruction turn (index 0, only when the caller sets `has_leading_instruction`) and the newest turn (last index) are never removed. An outcome at or over the bound is a normal result, not an error [fit.rs:next_removable_index].
- Removal takes the oldest eligible turn first [fit.rs:fit_conversation_window].
- The whole remaining conversation is measured, never a single turn in isolation [fit.rs:fit_conversation_window].
- `measure` runs once before any removal and once after each removal, so it runs `removed_turn_count + 1` times [fit.rs:fit_conversation_window; tests/fit_rules.rs].
- Trimming continues while `measured_length >= bound`. Room for generation is `measured_length < bound`, so a measurement exactly equal to the bound means no room [fit.rs:fit_conversation_window].
- A measurement of `u64::MAX` is at or over any bound, so a callback that cannot measure fails closed [fit.rs:fit_conversation_window].
- Edge cases covered by tests: an empty conversation is returned unchanged, and a single turn with no leading instruction is never removed [tests/fit_rules.rs]. Following from the `>=` rule in the code (not tested): `bound = 0` never reports room [fit.rs:fit_conversation_window].
- `measured_length()` is the last measurement taken (the initial one when nothing was removed) [outcome.rs:measured_length].
- Same arguments yield the same outcome, assuming the caller's `measure` is itself deterministic [fit.rs:fit_conversation_window].

## Side Effects
- The function itself does no I/O and keeps no state between calls [fit.rs:fit_conversation_window].
- `measure` is `FnMut`, so any side effects or state in the caller's callback happen inside this call, and a stateful callback breaks the determinism guarantee.

## Verification
- Run: `cargo test -p turbospark-window-fit`
