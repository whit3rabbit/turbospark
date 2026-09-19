# turbospark-streaming

Token callbacks, completion splitting, cancellation, and protocol adapters.

## Read first

- [Detailed module guide](../../.claude/docs/modules/streaming.md)
- [Streaming pipeline](../../docs/STREAMING.md)

## Rules

- Keep unsafe worker-pool and file-descriptor lifetimes bounded by the owning
  batch. Do not let parked workers outlive the resources they borrow.
- A stop token may be consumed by the loop before the consumer callback. If a
  decoder needs it to close a span, call its finish path explicitly.
- Keep cancellation, end-of-turn, stop reason, and callback ordering stable.
- Test raw streaming and structured/tool streaming separately.

## Checks

```sh
cargo test -p turbospark-streaming
cargo check --target x86_64-unknown-linux-gnu -p turbospark-streaming
```
