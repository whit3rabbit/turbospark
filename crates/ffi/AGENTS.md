# turbospark-ffi

The C ABI used by the Swift package and macOS app.

## Read first

- [Detailed module guide](../../.claude/docs/modules/ffi.md)
- [Swift binding contract](../../docs/SWIFT_BINDINGS.md)
- [Audio bindings](../../docs/SWIFT_BINDINGS.md#audio)

## Rules

- Unsafe code is expected here because the ABI receives raw pointers. Every
  exported entry point must stay inside the ABI guard and must not unwind across
  `extern "C"`.
- Keep the handwritten `turbospark.h` synchronized with the Rust wire
  contract. Rust tests alone do not validate the copied Swift header.
- Trace enum strings and capability predicates through both Rust and Swift.
  Hardware detection symbols (`ts_chip_supports_ane_prefill`) and wire options
  (`OpenOptions.anePrefill`, `SessionInfo.anePrefill`) must be mirrored across
  `turbospark.h`, `wire.rs`, `open.rs`, `testing.rs`, and `swift/TurboSpark`.
- Server posture options in `ServerOptions` (`guardrails`, `prefixReuse`) are
  process-level, never per request: one engine's KV cache and guardrail posture
  are shared by every client and are not partitioned by API key. Apply
  `prefixReuse` at attach on both attach paths: `ts_server_start` sets it
  before attaching the startup session, so no request sees the session's own
  default, and `Server::attach` applies it again at the attach point both
  paths share. A JSON `null` leaves the session's open-time decision alone; a
  wrong-typed value is refused by `ts_server_start` (`tests/c_surface.rs`),
  never dropped.
- `make swift-lib` stages the static library and canonical header before
  SwiftPM tests.

## Checks

```sh
cargo test -p turbospark-ffi
make swift-lib
make swift-test
```

Real binding tests need a pinned install and the second blocked install
described in the verification reference.
