---
title: C ABI Contract and Constants
description: The turbospark.h contract rules, status codes, streaming and install event kinds, callback typedefs, and the error and string helpers.
diataxisType: reference
---

<!-- generated: cpp lane, signal: crates/ffi/include/turbospark.h -->

Reference for `crates/ffi/include/turbospark.h`, the hand-written C ABI over the
inference engine for a native GUI host. It is the only tracked C/C++ source in
this workspace. Every signature below is copied verbatim from the header; every
prose block is the header's own comment text.

The header deliberately declines a generator (cbindgen was costed and
declined): the surface is small enough to read in one sitting, and the SwiftPM
target that compiles against it is a stronger check that the two sides agree
than a generator would be. Note for tooling: the header's comments are plain
C `/* */` blocks, not Doxygen `/**` tags, so a Doxygen run would need
configuration to attach them to declarations; this page was produced by parsing
the header source directly because doxygen is not installed in this
environment.

## The contract, in four rules

Quoted from the header's own comment block.

1. **ERRORS.** Every fallible call returns `TS_OK` (0) or a non-zero code.
   On a non-zero return, `ts_last_error()` on the SAME THREAD holds a
   message. Read it before making another call on that thread.
2. **OWNERSHIP.** A `const char *` argument is borrowed for the duration of
   the call and never retained. A `char **` out-parameter receives an
   allocation the caller must return through `ts_string_free()`. There is no
   third case, and nothing here hands back a pointer into its own state.
3. **JSON.** Options and results travel as JSON strings, so adding a knob is
   never an ABI break. Keys are camelCase, so a Swift `Codable` needs no
   `CodingKeys`. The per-token streaming path carries no JSON: it is a
   pointer and a length.
4. **THREADING.** A session is single-threaded: one generation at a time, and
   `ts_generate()` blocks for the whole turn, so call it from a background
   thread. The ONE exception is `ts_session_cancel()`, which is safe from
   any thread and never blocks. `ts_install()`'s byte callback is ALSO called
   concurrently from worker threads.

**PLATFORM.** The engine is macOS-only. Everywhere else `ts_session_open()`
fails with `TS_ERR_UNSUPPORTED` and a sentence, while the catalog, probe and
install calls work normally: the artifact is the same whether or not this
machine can run it.

## Status codes

| Macro | Value | Header comment |
|---|---|---|
| `TS_OK` | 0 | No per-symbol comment. The convention is stated in the contract block above: every fallible call returns `TS_OK` (0) or a non-zero code. |
| `TS_ERR_INVALID_ARGUMENT` | 1 | A null pointer, a non-UTF-8 string, or a value outside its allowed set. |
| `TS_ERR_OPEN` | 2 | The install could not be opened: missing directory, unreadable manifest, unsupported architecture, or a context window that does not fit memory. |
| `TS_ERR_GENERATE` | 3 | Generation or installation failed. The session remains usable. |
| `TS_ERR_JSON` | 4 | A JSON argument did not parse, or a result could not be built. |
| `TS_ERR_UNSUPPORTED` | 5 | The engine is not available on this platform. |
| `TS_ERR_PANIC` | 6 | A panic was caught at the boundary. The process is intact and the operation did not happen. This is a bug in the library, not in the call. |

## Streaming event kinds

Values delivered through `TsEventCallback`'s `kind` parameter.

| Macro | Value | Header comment |
|---|---|---|
| `TS_EVENT_PREFILL` | 0 | Prefill progress. `a` is prompt tokens done, `b` the total. `text` empty. |
| `TS_EVENT_CONTENT` | 1 | Visible reply text. Accumulate THIS as the assistant turn. |
| `TS_EVENT_REASONING` | 2 | The model's reasoning, already separated from the reply. Do NOT feed it back as conversation history: Harmony's own convention drops prior-turn analysis, and Qwen's template drops prior-turn `<think>` blocks, so sending it back gives the model something it was never trained to read. |

## Install progress kinds

Values delivered through `TsInstallCallback`'s `kind` parameter.

| Macro | Value | Header comment |
|---|---|---|
| `TS_INSTALL_STAGE` | 0 | A human-readable stage line, on the calling thread. |
| `TS_INSTALL_BYTES` | 1 | Byte progress. `done` of `total`. CALLED CONCURRENTLY, see `ts_install`. |

## Opaque handles

```c
typedef struct TsSession TsSession;
typedef struct TsServer TsServer;
```

Neither typedef carries its own doc comment; the header documents their
lifecycle on the functions that take them
(`ts_session_open` / `ts_session_close` / `ts_session_cancel`, and
`ts_server_start` / `ts_server_stop`).

## Callbacks

```c
typedef void (*TsEventCallback)(void *userdata, int32_t kind,
                                const char *text, size_t len,
                                uint32_t a, uint32_t b);
```

One streamed generation event. `text` is UTF-8 of length `len`. It is NOT
NUL-terminated and is valid ONLY for the duration of this call: copy it before
returning.

```c
typedef void (*TsInstallCallback)(void *userdata, int32_t kind,
                                  const char *text, size_t len,
                                  uint64_t done, uint64_t total);
```

One install-progress event. `text`/`len` as above; `done` and `total` are
bytes for `TS_INSTALL_BYTES` and zero otherwise.

## Errors and strings

```c
size_t ts_last_error(char *buf, size_t cap);
```

Copies this thread's last error into `buf` as a NUL-terminated string.

Returns the message's own length in bytes, excluding the NUL, which is NOT
necessarily the number of bytes written, so a caller given a value >= cap can
allocate that much and call again. Pass buf = NULL to ask for the length alone.

```c
void ts_string_free(char *s);
```

Frees a string returned through a `char **`. NULL is a no-op.
