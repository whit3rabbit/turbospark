# turbospark-ffi

C ABI static library (`libturbospark_ffi.a`) and canonical C header (`include/turbospark.h`) enabling native GUI hosts and Swift packages (`swift/TurboSpark`) to drive the inference engine in-process with zero IPC overhead.

Downstream Swift code links this crate's `staticlib` directly; nothing in this workspace depends on it as an `rlib` except its own internal test suite.

## Purpose & Role

`turbospark-ffi` exposes the complete feature set of the Rust inference engine across a stable, foreign-function interface (FFI). It provides opaque session handles, exchanges complex configuration and results via camelCase JSON wire shapes, exposes token-by-token push callbacks, enables thread-safe generation cancellation from any thread, and embeds the local HTTP server and model catalog manager directly into host applications.

## Safety

- Contains `unsafe` code throughout: this is the FFI boundary, where raw pointers and foreign memory cross into Rust.
- **Panic Boundary Guarding**: Rust panics cannot safely unwind across an `extern "C"` boundary. Every entry point in `api/` wraps its implementation body in `abi::guard`, `abi::guard_result`, or `abi::guard_value`. Any panic or error is caught and converted into a thread-local error message and an integer status code (`TS_STATUS_OK`, `TS_STATUS_ERROR`).

## Key Modules

- `abi.rs`: Status code enums, thread-local error slot, and `guard` safety wrappers.
- `strings.rs`: Safe borrowing of incoming `const char *` arguments and allocation of owned output strings managed via `ts_string_free`.
- `wire.rs`: Strongly-typed serialization structures for camelCase JSON request and response payloads.
- `session.rs`: Opaque `Session` handle managing `SessionCore` and execution engine selection (`Engine::Real` on macOS, `Engine::Scripted` on other platforms).
- `open.rs`: macOS session loader opening `.gturbo` model directories and configuring `RealForwardRunner`.
- `generate/`: Complete generation pipeline driving prompt rendering, streaming callbacks, tool call emission, and result summaries.
- `models/`: Portable catalog browsing, model recommendation, and stream-install surface.
- `server.rs`, `server_model.rs`, `server_registry.rs`, `server_transport.rs`: In-process HTTP server engine sharing the open `SessionCore` without loading a second model instance.
- `telemetry.rs`: Hardware performance counters, phase durations, and peak physical footprint telemetry.
- `vision.rs`: Multimodal image data URL parsing and patch token preparation.
- `testing.rs`: Scripted mock engine harness for deterministic FFI testing.
- `api/`: C entry points organized by domain:
  - `core.rs`: Versioning, error string inspection, and memory deallocation.
  - `session.rs`: Session open, close, and model info queries.
  - `generate.rs`: Token generation, streaming callbacks, and generation cancellation.
  - `models.rs`: Catalog querying, probe, download, and installation.
  - `server.rs`: In-process HTTP server start, stop, and status.
  - `daemon.rs`: External daemon process lifecycle controls.
  - `embedding.rs`: Text embedding generation via Post-LN encoder models.

## Development & Test Commands

```sh
# Run this crate's own tests through the Rust rlib interface
cargo test -p turbospark-ffi

# Build the staticlib and stage it plus include/turbospark.h for SwiftPM
make swift-lib

# Run SwiftPM tests linking libturbospark_ffi.a against turbospark.h
make swift-test

# Run end-to-end Swift integration tests against a real model install
make swift-test-real MODEL=~/models/gemma4.gturbo
```

## Tests

- `tests/c_surface.rs`: Comprehensive integration test exercising every declared `extern "C"` function, verifying pointer safety, error slot clearing, and JSON serialization.

## Crate Gotchas

1. **Staticlib vs Rlib Verification**: `tests/c_surface.rs` reaches function bodies through the Rust `rlib`, so it can pass even if the C header signature drifts. The authoritative check for ABI drift is `make swift-test`, which compiles Swift code directly against `include/turbospark.h` and links the static archive.
2. **String Allocation & Ownership**: All strings returned across the FFI by pointer (`char **`) are allocated on the Rust heap using `CString`. Callers MUST free them by passing the pointer to `ts_string_free` to prevent memory leaks in host processes.
3. **Thread-Safe Cancellation**: Calling `ts_cancel` sets an atomic cancellation token on the session. It can be invoked safely from any thread or asynchronous task while `ts_generate` is actively decoding tokens on another thread.
