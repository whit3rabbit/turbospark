# turbospark-ffi

The C ABI a native GUI host drives the engine through, plus the SwiftPM package (`swift/TurboSpark`) that wraps it and the SwiftUI chat app (`swift/TurboSparkApp`) that verifies the stack end to end. Compiles to a `staticlib` and is published against a single hand-written header, `include/turbospark.h`: an opaque session handle, options and results as JSON so a new knob is never an ABI break, a per-token streaming callback, and cancel-from-any-thread.

Downstream Swift code links this crate's `staticlib` directly; nothing in this workspace depends on it as an `rlib` except its own tests, which exercise the same `extern "C"` function bodies the header declares.

## Safety

Contains `unsafe` throughout: it IS the ABI layer, so every entry point takes raw pointers. It joins `model-io` and `streaming` as the third crate in the workspace that cannot `#![forbid(unsafe_code)]`. Every `extern "C"` body in `api/` is a call to `abi::guard`, `abi::guard_result`, or `abi::guard_value` and nothing else, because a panic unwinding across the boundary is undefined behaviour and this workspace cannot build with `panic = "abort"`.

## Key Modules

- `abi.rs`: status codes, the per-thread error slot, and the `guard`/`guard_result`/`guard_value` wrappers every entry point is built from.
- `strings.rs`: borrowing `const char *` arguments in, handing owned `char **` allocations out through `ts_string_free`.
- `wire.rs`: the JSON option and result shapes exchanged across the boundary (camelCase, except the two catalog types passed through unchanged from `crates/catalog`).
- `session.rs`: the opaque `Session` handle over a shared `SessionCore`, and the `Engine` enum (`Real` on macOS, `Scripted` for platform-independent testing).
- `open.rs` (macOS only): opens an install into a `SessionCore`, mirroring `crates/cli`'s `open_session`.
- `generate/`: one turn end to end -- prompt rendering, decode, streaming, result reporting, tool-call and vision plumbing.
- `models/`: the portable catalog, probe, and install surface (browsing, recommending, installing a model, all of which work even on a platform that cannot then run one).
- `server.rs`, `server_model.rs`, `server_registry.rs`: an in-process HTTP server built around this crate's already-open `SessionCore`, sharing `turbospark-server`'s router and `ChatModel` trait rather than opening a second model.
- `telemetry.rs`: phase counters and peak physical footprint.
- `testing.rs`: `session_for_testing`, the scripted-engine harness used by this crate's own tests and nothing else.
- `vision.rs` (macOS only): image data URL decoding and vision token preparation for `ts_generate`'s image content parts.
- `api/`: the `extern "C"` entry points themselves, organized by domain (`core`, `session`, `generate`, `models`, `server`, `daemon`, `embedding`).

## Development & Test Commands

```sh
# This crate's own tests, through the rlib face (no header, no Swift).
cargo test -p turbospark-ffi

# Build the staticlib and stage it plus the header for SwiftPM.
make swift-lib

# The only thing that can check the hand-written header against the real
# ABI: swift-lib builds first, then swift/TurboSpark/Tests links the
# staticlib and calls through turbospark.h.
make swift-test

# The same, plus the end-to-end arm against a real install.
make swift-test-real MODEL=~/models/gemma4.gturbo

# The SwiftUI chat app that exercises this crate as a real GUI host.
make swift-app
```

## Crate Gotchas

See `CLAUDE.md` for the full list; the one to read first is Gotcha 2: `tests/c_surface.rs` reaches the same function bodies `turbospark.h` declares through the `rlib`, so it can pass against a header that gets a signature wrong entirely. Only `make swift-test` links the `staticlib` and can catch that class of drift.
