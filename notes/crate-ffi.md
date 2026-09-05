---
uuid: "0781be4a-9ab0-4afa-b6e3-fa0f45b203c3"
title: "turbospark-ffi"
summary: "The C ABI a native GUI drives the engine through. cargo test cannot catch a header/Rust mismatch, only make swift-test can (it links the real staticlib)"
tags: ["crate", "ffi"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e02"]
source: "crates/ffi/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-ffi do?

The C ABI a native GUI drives the engine through: a `staticlib` plus a
hand-written `include/turbospark.h`. `swift/TurboSpark` wraps it and
`swift/TurboSparkApp` is the SwiftUI app that verifies the stack end to end.
Cannot forbid `unsafe`, because it IS the ABI layer and every entry point
takes raw pointers, alongside `model-io` and `streaming` as the third such
crate in the workspace. Options and results cross the boundary as JSON
(`wire.rs`). Only the per-token streaming callback is a raw pointer and
length, so the hot path pays no serde cost.

## Don't

- Don't trust `cargo test -p turbospark-ffi` to catch a header/Rust
  signature mismatch. `tests/c_surface.rs` reaches the same function bodies
  through the `rlib`, so it passes even against a wrong declaration. Only
  `make swift-test` (linking the real staticlib through `turbospark.h`) can
  catch it, and already has twice: once for a snake_case/camelCase field
  mismatch, once for an added argument the `rlib` face couldn't see.
- Don't forget `make swift-lib` before `make swift-test` after touching this
  crate. SwiftPM doesn't treat a rebuilt staticlib as a build input, so an
  unchanged set of `.swift` files triggers no relink and the suite silently
  checks the previous build. `scripts/swift-lib.sh` touches every Swift file
  to force it.
- Don't write anything inside an `extern "C"` body except a call to
  `abi::guard`. Unwinding across the FFI boundary is undefined behavior, and
  this workspace can't set `panic = "abort"` (two `Drop` impls are
  load-bearing on the unwind path), so an escaping panic is a real hazard.
- Don't move the session cancel flag inside the session `Mutex`. It lives
  outside on purpose: a GUI generates on a background thread holding the
  lock for the whole turn, and Stop has to interrupt from the main thread
  without waiting on that lock. Getting this wrong doesn't error, it HANGS.
- Don't treat the header's documented option contract as an enforced one.
  `turbospark.h` stated the legal expert-cache-slot set for as long as the
  option existed, but `open()` never checked it. This is the one front end
  where that's fatal: a GUI slot picker sending an illegal value panics the
  whole process (`crates/streaming`'s expert cache aborts), not just the
  request.
- Don't call `rt.block_on()` on the thread that calls `ts_server_start`.
  Tokio panics with "cannot start a runtime from within a runtime" if the
  caller is already inside one, which this crate's own `#[tokio::test]`s
  are. The bind happens on the background thread's own runtime instead,
  with the result sent back over a plain `std::sync::mpsc` channel.
