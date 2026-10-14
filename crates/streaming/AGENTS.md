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
- `linux_uring.rs` is gated only by its Linux tests, not by a cross-target
  check: its io_uring flags are hand-copied UAPI, and a wrong bit is another
  valid flag. Run the crate's tests on Linux after touching it.
- Price an `ExpertCache` policy change with
  `scripts/expert_policy_replay.py` before building it.

## Checks

```sh
cargo test -p turbospark-streaming
cargo check --target x86_64-unknown-linux-gnu -p turbospark-streaming
```
