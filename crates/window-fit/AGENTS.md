# turbospark-window-fit

This crate owns deterministic context-window and memory-fit calculations.

## Read first

- [Detailed module guide](../../.claude/docs/modules/window-fit.md)
- [Load guard](../../docs/LOAD_GUARD.md)

## Rules

- Keep calculations deterministic and independent of model execution.
- A missing physical-memory probe means unavailable data, not zero memory.
- Distinguish trained context, explicit context, KV cost, resident weights,
  expert slots, and reserve. Do not estimate fit from on-disk size alone.
- Preserve refusal messages and arithmetic so users can understand why a
  context window does not fit.

## Checks

```sh
cargo test -p turbospark-window-fit
```
