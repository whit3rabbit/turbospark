# turbospark-window-fit

This crate owns the deterministic conversation-window fit: dropping the
oldest eligible turns until a caller-measured conversation length is under
a caller-supplied bound.

## Read first

- [Detailed module guide](../../.claude/docs/modules/window-fit.md)
- [Load guard](../../docs/LOAD_GUARD.md)

## Rules

- Keep calculations deterministic and independent of model execution.
- The length measurement is caller-supplied. The crate never tokenizes and
  never inspects turn contents or roles.
- An outcome still at or over the bound is a normal result, not an error.
  Report the measured length and room flag so callers can explain why a
  conversation does not fit.

## Checks

```sh
cargo test -p turbospark-window-fit
```
