# turbospark-window-fit

Pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation history until the measured length satisfies caller-supplied token bounds, while preserving pinned turns.

Downstream workspace crates import this package via the `window_fit` alias:

```toml
[dependencies]
window_fit = { package = "turbospark-window-fit", path = "../window-fit" }
```

## Purpose & Role

When long multi-turn conversations exceed the model's context window or a caller-defined budget, `turbospark-window-fit` trims older turns deterministically. It guarantees that initial system prompt instructions and the most recent user turn remain anchored, while intermediate turns are evicted oldest-first.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Zero I/O, zero external dependencies, and completely deterministic execution.

## Key Modules

- `fit.rs`: Core algorithm `fit_conversation_window`. Evaluates token lengths and drops oldest non-pinned turns until the conversation satisfies bounds.
- `outcome.rs`: Outcome metadata `WindowFitOutcome<T>` recording which turn indices were preserved, dropped, or modified.

## Development & Test Commands

```sh
# Run unit tests for turbospark-window-fit
cargo test -p turbospark-window-fit
```

## Tests

- `tests/fit_rules.rs`: Validates turn eviction ordering, budget boundaries, and minimum turn constraints.
- `tests/outcome_contract.rs`: Verifies outcome metadata structure and preservation of pinned turns.

## Crate Gotchas

1. **Pinned Turns Guarantee**: The leading system/instruction turn (turn 0) and the newest user message turn are pinned by contract and will never be removed by window fitting, even if the budget is exceeded.
2. **Stateless Logic**: Performs no I/O, does not invoke tokenization directly, and holds no persistent state between calls. Token counts are caller-supplied measurements.
