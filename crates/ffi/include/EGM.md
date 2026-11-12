# ffi include

## Overview
Hand-written C header for the TurboSpark C ABI consumed by the Swift app. Hand-written on purpose: the SwiftPM target compiling against it is a stronger check than a generator would be.

## Interface
- [turbospark.h:TS_ABI_VERSION] (2) and [turbospark.h:ts_abi_version].
- Status codes: `TS_OK` 0, `TS_ERR_INVALID_ARGUMENT` 1, `TS_ERR_OPEN` 2, `TS_ERR_GENERATE` 3, `TS_ERR_JSON` 4, `TS_ERR_UNSUPPORTED` 5, `TS_ERR_PANIC` 6, `TS_ERR_CANCELLED` 7, `TS_ERR_BUSY` 8 [turbospark.h].
- Speech-to-text, session, generate, install, catalog, server, audio, image, OCR, embedding and library functions, all prefixed `ts_` (for example `ts_generate`, `ts_session_open`, `ts_session_cancel`, `ts_string_free`, `ts_last_error`) [turbospark.h]. Their Rust definitions live in [../src/api](../src/api/EGM.md).

## Invariants
- Errors: a non-zero return leaves a message readable by `ts_last_error()` on the same thread [turbospark.h] (rule 1).
- Ownership: `const char *` arguments are borrowed for the call; `char **` outputs must be released with `ts_string_free()` [turbospark.h] (rule 2).
- Options and results travel as camelCase JSON; the per-token path carries no JSON [turbospark.h] (rule 3).
- A session is single-threaded; only `ts_session_cancel()` is safe from any thread, and a token callback must not re-enter the same session. Exception: `ts_install()`'s byte callback runs concurrently on worker threads [turbospark.h] (rule 4).
- The header version must equal the Rust `ABI_VERSION`; bump it when a wire shape, option key or meaning changes [turbospark.h:TS_ABI_VERSION], [../src/abi.rs:ABI_VERSION].

- Platform: on non-macOS, `ts_session_open` fails with `TS_ERR_UNSUPPORTED` while catalog, probe and install calls still work [../src/api/session.rs:ts_session_open].

## Side Effects
- None (declarations only).

## Verification
- Run: `cargo test -p turbospark-ffi` (the `abi_version_matches_the_header` test compares the header to `ABI_VERSION`)
