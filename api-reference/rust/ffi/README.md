---
description: "C ABI over the inference engine for native hosts: opaque handles, camelCase JSON in and out, streamed events, and cancellation from any thread."
icon: code
---

# turbospark-ffi

`turbospark-ffi` is the C ABI that the Swift package and the macOS app link against. It builds as a static library, `libturbospark_ffi.a`, and the header a host compiles against is `crates/ffi/include/turbospark.h`. A host loads a model, runs a turn, and receives the reply through a callback, all in the same process. There is no separate daemon in this path.

The crate is `publish = false`. It is a boundary for native callers, not a Rust API. If you write Rust, use the crates it wraps, such as [`turbospark-runtime`](../runtime/README.md) or [`turbospark-server`](../server/README.md).

## Who this is for

- Authors of a native host that calls C functions: the macOS app, a new Swift binding, or another GUI.
- Engineers debugging the Swift package at the boundary, where a JSON key or an ownership rule went wrong.

If you use the Swift package, start with the [Swift bindings](../../swift.md) page. It wraps this ABI in async Swift types, `Codable` values, and `AsyncThrowingStream` events. You do not write JSON or free strings yourself there. This page describes what that wrapper calls, and what you must get right if you write your own.

## Platform

The inference engine runs only on macOS with Apple Silicon. Session open is refused on other platforms. The catalog, probe, and install calls are portable, so a non-macOS host can still browse and install models. The [errors page](errors-and-strings.md#status-codes) notes a mismatch between the header and the source for the status code returned in that refusal.

## The model in four rules

1. **Errors.** Every fallible call returns `TS_OK` (0) or a non-zero code. On a non-zero return, read `ts_last_error` on the same thread before you make another call.
2. **Ownership.** A `const char *` argument is borrowed for the duration of the call and never retained. A `char **` out-parameter receives an allocation you free with `ts_string_free`. Some outputs use their own free function, listed in [Errors and strings](errors-and-strings.md#what-to-free-and-how).
3. **JSON.** Options and results are JSON strings with camelCase keys. The per-token stream carries no JSON: it is a pointer, a length, and two integers.
4. **Threads.** A session is single-threaded. `ts_generate` blocks for the whole turn, so call it from a background thread. `ts_session_cancel` is the one exception: it is safe from any thread and never blocks.

## Opaque handles

Every handle is an incomplete struct in C. You hold a pointer and never look inside it. Each opener has a matching release call.

| Handle | Opened by | Released by |
| --- | --- | --- |
| `TsSession` | `ts_session_open` | `ts_session_close` |
| `TsTokenizer` | `ts_tokenizer_open` | `ts_tokenizer_close` |
| `TsImageSession` | `ts_image_session_open` | `ts_image_session_close` |
| `TsServer` | `ts_server_start` | `ts_server_stop` |
| `TsSttModel`, `TsSttStream` | `ts_stt_open`, `ts_stt_stream_open` | `ts_stt_close`, `ts_stt_stream_close` |

Passing `NULL` to the session, tokenizer, image, or server release call is a no-op. Those release calls return nothing.

## Streaming events

A generation reports progress through one callback function and a `void *userdata` that you supply. The callback runs on the thread that made the call. The `text` pointer is valid only while the callback runs, so copy it before you return. A host ignores any event kind it does not know, because newer libraries can add kinds without a version break. See [Streaming events](streaming-events.md).

## Cancel from any thread

`ts_session_cancel` sets an atomic flag that lives outside the engine lock. A Stop button on the main thread can call it while a worker thread is still decoding, and the call returns at once. The flag is cleared when the next turn starts, so a Stop pressed between turns does not cancel the following one. A cancelled turn is not an error: it returns `TS_OK` with `stopReason` set to `"cancelled"` and the partial text.

## Pages

| Page | Read it when |
| --- | --- |
| [Opening a model](opening-a-model.md) | You need to open a session, read what it resolved to, or release it. Also covers the tokenizer-only and image handles. |
| [Streaming events](streaming-events.md) | You need to run a turn, handle the event kinds, cancel a turn, or generate an image. |
| [Errors and strings](errors-and-strings.md) | A call returned non-zero, or you need to know what to free and how. |
| [Server entry points](server-entry-points.md) | You need to run the in-process HTTP server and attach models to it. |

## Scope of these pages

These pages cover sessions, generation, tokenization, image generation, and the in-process server. They do not cover speech-to-text (`ts_stt_*`), native audio (`ts_audio_*`), installs and the catalog (`ts_install*`, `ts_catalog_json`), embeddings, or the daemon. Those calls follow the same four rules above. Read their declarations and comments in `turbospark.h`.

## Check the ABI at startup

The header and the library each carry an ABI revision. Compare them once, before your first call:

```c
#include <stdio.h>
#include "turbospark.h"

int check_abi(void) {
    if (ts_abi_version() != TS_ABI_VERSION) {
        fprintf(stderr, "turbospark: header is ABI %d, library is ABI %u\n",
                TS_ABI_VERSION, (unsigned)ts_abi_version());
        return 0;
    }
    return 1;
}
```

A mismatch means the header and the archive came from different builds. The two copies are staged separately, so a stale one is easy to produce. `make swift-lib` stages the archive and the header from the same checkout, and the Rust test `abi_version_matches_the_header` checks the header's number against the crate.

## Related

- [Swift bindings](../../swift.md): the supported Swift surface built on this ABI.
- [turbospark-server](../server/README.md): the standalone HTTP server. The in-process server in this crate shares its routes and envelopes.
