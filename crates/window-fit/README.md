# turbospark-window-fit

Pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation history until the measured length satisfies caller-supplied token bounds.

Downstream workspace crates import this package via the `window_fit` alias:

```toml
[dependencies]
window_fit = { package = "turbospark-window-fit", path = "../window-fit" }
```

## Key Modules

- `fit.rs`: Core algorithm function `fit_conversation_window`.
- `outcome.rs`: Outcome metadata structures `FitOutcome` and `DroppedTurn`.

## Development & Test Commands

```sh
# Run unit tests for turbospark-window-fit
cargo test -p turbospark-window-fit
```

## Crate Gotchas

1. **Pinned Turns**: The leading system/instruction turn (turn 0) and the newest user message turn are pinned and will never be removed by window fitting.
2. **Stateless Logic**: Performs no I/O, does not invoke tokenization directly, and holds no persistent state between function calls. Token counts are caller-supplied.
