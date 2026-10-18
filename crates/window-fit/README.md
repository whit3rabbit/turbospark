# turbospark-window-fit

Pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation history until the measured length satisfies a caller-supplied token bound, while preserving pinned turns.

Downstream workspace crates import this package via the `window_fit` alias:

```toml
[dependencies]
window_fit = { package = "turbospark-window-fit", path = "../window-fit" }
```

## Purpose & Role

When long multi-turn conversations exceed the model's context window or a caller-defined budget, `turbospark-window-fit` trims older turns deterministically. It guarantees that a leading instruction turn, when the caller marks one, and the newest turn remain anchored, while intermediate turns are evicted oldest-first.

## Safety

- The crate contains no `unsafe` code, but `lib.rs` does not set `#![forbid(unsafe_code)]`, so the compiler does not enforce this.
- Zero I/O, zero external dependencies, and completely deterministic execution.

## Key Modules

- `fit.rs`: Core algorithm `fit_conversation_window`. Evaluates token lengths and drops oldest non-pinned turns until the conversation satisfies bounds.
- `outcome.rs`: Outcome metadata `WindowFitOutcome<T>` carrying the retained turns, the measured length, the removed turn count, and whether room remains for generation.

## Development & Test Commands

```sh
# Run unit tests for turbospark-window-fit
cargo test -p turbospark-window-fit
```

## Tests

- `tests/fit_rules.rs`: Validates turn eviction ordering, budget boundaries, and minimum turn constraints.
- `tests/outcome_contract.rs`: Verifies outcome metadata structure, the measurement and room-for-generation contracts, and determinism.

## Crate Gotchas

1. **Pinned Turns Guarantee**: The leading instruction turn (turn 0, only when the caller marks it) and the newest turn are pinned by contract and will never be removed by window fitting, even if the budget is exceeded.
2. **Stateless Logic**: Performs no I/O, does not invoke tokenization directly, and holds no persistent state between calls. Token counts are caller-supplied measurements.
