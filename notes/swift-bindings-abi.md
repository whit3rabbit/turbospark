---
uuid: "cdc6f0dd-077e-49bd-9ed3-d22415ae0c4d"
title: "Swift/C ABI: the calling contract"
summary: "Four rules govern every call: per-thread errors via ts_last_error, borrowed-or-owned-never-third pointers, JSON options, and session single-threading except cancel"
tags: ["swift", "ffi", "abi"]
depends_on: ["0781be4a-9ab0-4afa-b6e3-fa0f45b203c3", "adde586f-5363-4d77-b5bb-e8f89be9494b"]
source: "docs/SWIFT_BINDINGS.md"
created: "2026-09-05"
updated: "2026-09-05"
---

## What is the raw C contract for a non-Swift host?

`crates/ffi/include/turbospark.h` is canonical. `docs/SWIFT_BINDINGS.md`
states it as four rules: errors return `TS_OK` or a non-zero code, with
`ts_last_error()` holding the message on the SAME thread that made the
call. Ownership is borrowed-for-the-duration or returned-through-`ts_string_free()`,
with no third case and nothing handing back a pointer into library state.
Options and results are JSON (camelCase keys), except the per-token path,
which is a raw pointer and length so the hot path pays no serde cost. A
session is single-threaded and `ts_generate` blocks for the whole turn,
except `ts_session_cancel`, safe from any thread and non-blocking.

See notes/crate-ffi.md for the Rust side of this same boundary (the
`abi::guard` pattern, why panics can't unwind across it).

## Don't

- Don't call `ts_last_error()` from a different thread than the call that
  failed. It's per-thread state. Reading it elsewhere reads garbage or a
  stale message from an unrelated call.
- Don't hold onto the `text` pointer a streaming callback receives past the
  callback's own return. It is NOT NUL-terminated and is valid only for
  that call. Copy the bytes out (`fwrite`, `memcpy`) before returning if
  you need them later.
- Don't expect an unrecognized option string to fall back to a default.
  `maxContext`, `loadGuard`, `speculation`, and `speculativeDrafter` are all
  refused BY NAME as an error when misspelled, checked before the install
  is even touched, on the reasoning that silently defaulting when a caller
  asked for `"strict"` is exactly the disagreement the option exists to
  prevent.
- Don't assume anything besides `ts_session_cancel` is safe to call from a
  second thread while a call is in flight on the same session. The
  contract is single-threaded, and nothing in the ABI enforces that for
  you beyond that one documented exception.
- Don't reach for the bindings expecting tool calling, multiple concurrent
  sessions in one process, iOS, or Intel Macs. All four are explicitly out
  of scope, stated as decisions rather than gaps. Tool calling needs a way
  to RUN a tool, which a binding can't supply on its own. Concurrent
  sessions are untested because the engine takes `&mut self` to decode, so
  calls already serialize.
- Don't trust `cargo test -p turbospark-ffi` to catch a header mismatch,
  and don't skip `make swift-test` for an FFI-facing change. The Rust
  tests reach the same function bodies through the `rlib` and pass against
  a wrong `turbospark.h` declaration. Only linking the real staticlib
  through the header catches that, and already has (a snake_case versus
  camelCase field mismatch).
