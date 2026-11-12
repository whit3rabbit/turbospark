---
description: "Pure, deterministic conversation-window fitting: drops the oldest eligible turns until a caller-measured conversation fits a caller-supplied bound, never removing the leading instruction turn or the newest turn."
icon: crop
---

# window-fit

`crates/window-fit` (package `turbospark-window-fit`, library
`turbospark_window_fit`) is a tiny, dependency-free crate that owns
deterministic conversation-window fitting. When a multi-turn conversation
would not leave room for at least one generated unit under a context budget,
`fit_conversation_window` drops the oldest eligible turns, oldest first, until
a caller-supplied measurement of the remaining conversation is under a
caller-supplied bound. The crate performs no I/O, holds no state between
calls, and never tokenizes anything itself, so the same arguments always
produce the same outcome. The `cli` and `ffi` crates call it before rendering
a prompt for generation.

## What it does

| Responsibility | Where |
| --- | --- |
| The fitting algorithm: measure, drop the oldest eligible turn, re-measure, stop | `src/fit.rs` (`fit_conversation_window`) |
| The outcome record: retained turns, measured length, removal count, room flag | `src/outcome.rs` (`WindowFitOutcome`) |

Behavior, as implemented in `src/fit.rs`:

- The whole conversation is measured once before any removal and re-measured
  after each removal; the measure callback never sees a single turn in
  isolation.
- Removal takes the oldest eligible turn first and stops as soon as the
  measured length is under the bound or nothing eligible remains.
- The leading instruction turn (index 0, when `has_leading_instruction` is
  true) and the newest turn are never eligible for removal.
- An outcome that is still at or over the bound is a normal, reportable
  result, not an error; `has_room_for_generation()` reports which case
  happened.
- Turns are generic (`T: Clone`), so a turn can be a chat `Message`, a wire
  JSON value, or a plain string.

## Where it sits

Nothing. `Cargo.toml` declares zero dependencies: no workspace crates and no
external crates. That isolation is the point (`crates/window-fit/AGENTS.md`):
window policy has to stay deterministic and independent of model execution.

Two crates consume it, both aliasing the package to `window_fit`:

- `crates/cli`: `src/chat.rs` fits each chat turn against the session's
  resolved `max_context` before generating, measuring by rendering the
  candidate conversation through the tokenizer.
- `crates/ffi`: `src/generate/prompt.rs` (`fit_window`) fits the incoming
  transcript the same way and maps the outcome into the wire-level
  `WindowFitOutcome`.

## Public API tour

The whole public surface is two items re-exported from `src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `fit_conversation_window(turns, has_leading_instruction, bound, measure)` | `src/fit.rs` | Fit `&[T]` under the `bound: u64` using `measure: FnMut(&[T]) -> u64`; returns `WindowFitOutcome<T>` |
| `WindowFitOutcome::retained_turns()` | `src/outcome.rs` | The surviving turns, in original conversation order |
| `WindowFitOutcome::measured_length()` | `src/outcome.rs` | The last measurement taken over the retained turns |
| `WindowFitOutcome::removed_turn_count()` | `src/outcome.rs` | How many turns were dropped to reach this outcome |
| `WindowFitOutcome::has_room_for_generation()` | `src/outcome.rs` | True when the retained conversation measures under the bound, leaving room for at least one generated unit |

`WindowFitOutcome` derives `Debug`, `Clone`, `PartialEq`, and `Eq`, so two
identical calls can be compared directly in tests.

## Examples

### Fit a conversation under a bound

Adapted from `tests/fit_rules.rs`:

```rust
use turbospark_window_fit::fit_conversation_window;

let turns = ["oldest", "middle-1", "middle-2", "newest"];

// Any FnMut(&[T]) -> u64 works as the measurement; this one counts characters.
let outcome = fit_conversation_window(&turns, false, 15, |t: &[&str]| {
    t.iter().map(|turn| turn.len() as u64).sum()
});

// 28 characters -> drop "oldest" (22) -> drop "middle-1" (14, under the bound).
assert_eq!(outcome.retained_turns(), &["middle-2", "newest"]);
assert_eq!(outcome.removed_turn_count(), 2);
assert!(outcome.has_room_for_generation());
```

### The real-caller pattern

Both consumers measure by rendering the candidate conversation through the
checkpoint's chat template and fit against the resolved context window.
Simplified from `crates/cli/src/chat.rs` and
`crates/ffi/src/generate/prompt.rs`:

```rust
let measure = |messages: &[Message]| -> u64 {
    match tokenizer.apply_chat_template(messages) {
        Ok(rendered) => tokenizer.encode(&rendered, false).len() as u64,
        // A length no conversation has: the fit drops what it can, and the
        // caller's refusal reports the sentinel instead of a fake count.
        Err(_) => u64::MAX,
    }
};

let fitted = fit_conversation_window(
    &turn,
    has_leading_instruction,
    session.max_context as u64,
    measure,
);
if !fitted.has_room_for_generation() {
    return Err(
        "message does not fit max_context; shorten it or raise --max-context"
            .to_string(),
    );
}
```

## Gotchas

{% hint style="warning" %}
A fit can legitimately end at or over the bound: the leading instruction turn
and the newest turn are pinned and are never removed, even when nothing else
is left to drop. Always check `has_room_for_generation()` and refuse the turn
when it is false; both real callers do.
{% endhint %}

{% hint style="warning" %}
The measurement must match the prompt that will actually run. The CLI measures
with the same reasoning level the turn will render with (a level adds a
system-preamble sentence, so measuring without it under-counts), the FFI
renders with the offered tools included, and the bound is the resolved window
the KV cache was allocated at, not the raw request (under `--max-context auto`
the request carries no number). Image turns are under-counted at this stage:
the render emits one placeholder token per picture, and expansion to the
merged count happens later in vision attach, so the engine can still refuse
loudly downstream.
{% endhint %}

## Related pages

- [cli](cli.md)
- [ffi](ffi.md)
- [tokenizer](tokenizer.md)
- [Overview](overview.md)
