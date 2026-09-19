# turbospark-window-fit

Pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation until the measured length fits within caller-supplied bounds.

## Directory & File Structure

```
crates/window-fit/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root re-exporting fit_conversation_window and outcomes
|   +-- fit.rs              # Core fit_conversation_window algorithm
|   \-- outcome.rs          # WindowFitOutcome<T> struct declaration
\-- tests/
    +-- fit_rules.rs        # Unit tests verifying turn preservation and removal ordering rules
    \-- outcome_contract.rs # Unit tests asserting outcome length measurements and bounds contracts
```

## Key Modules

- `fit.rs`: Core function `fit_conversation_window`.
- `outcome.rs`: Outcome structure `WindowFitOutcome<T>`.

## Development & Test Commands

```sh
# Run tests for turbospark-window-fit
cargo test -p turbospark-window-fit
```

## Crate Gotchas

1. **Preserved Turns**: The optional leading system/instruction turn (turn 0) and the newest user message turn are pinned and will NEVER be removed by window fitting.
2. **Stateless Logic**: Performs no I/O, does not invoke tokenization directly, and holds no persistent state between calls. Length measurements are caller-supplied.
