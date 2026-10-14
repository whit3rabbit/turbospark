---
description: "Status codes, the per-thread error message, panic handling, and which outputs a C caller must free and with what. Read this when a call returns non-zero."
icon: code
---

# Errors and strings

Most calls return an `int32_t` status. Strings and buffers cross the boundary as owned allocations, and the wrong free function corrupts memory without warning. This page covers the status codes, how to read the error message, and the ownership rules for every output.

## Status codes

Every fallible call returns `TS_OK` (0) or one of these codes:

| Code | Value | Meaning |
| --- | --- | --- |
| `TS_OK` | 0 | Success. |
| `TS_ERR_INVALID_ARGUMENT` | 1 | A required pointer is `NULL`, a string is not UTF-8, or a value is outside its allowed set. Also used for a missing output pointer. |
| `TS_ERR_OPEN` | 2 | A model, tokenizer, server, or install could not open. Missing directory, unreadable manifest, unsupported architecture, or a window that does not fit memory. |
| `TS_ERR_GENERATE` | 3 | Generation or installation failed. The session stays usable. |
| `TS_ERR_JSON` | 4 | A JSON argument did not parse, or a result could not be built. The message names the argument. |
| `TS_ERR_UNSUPPORTED` | 5 | The engine is not available on this platform, or the session has no vision tower to release. |
| `TS_ERR_PANIC` | 6 | A panic was caught at the boundary. The process is intact and the operation did not happen. This is a library bug, not a caller error. |
| `TS_ERR_CANCELLED` | 7 | The caller's cancel ended an install or an audio job. Not a failure. A cancelled text turn returns `TS_OK` instead. |
| `TS_ERR_BUSY` | 8 | A resource is in use by another job, such as an audio session. Retrying after that job finishes can succeed. |

Two places where the source and the header differ:

- **Session open off macOS.** The header says `ts_session_open` fails with `TS_ERR_UNSUPPORTED`. The source maps the refusal to `TS_ERR_OPEN`. Your code should treat any non-zero return from open as a failure and read the message. Do not depend on one code to detect "unsupported platform".
- **Release vision.** `ts_session_release_vision` returns `TS_ERR_UNSUPPORTED` when there is no tower to release. The header and the source agree here.

## Read the error message

When a call returns non-zero, read the message before you make any other call on the same thread.

```c
size_t ts_last_error(char *buf, size_t cap);
```

Pass `NULL` to get the message length alone. The return value is the message length in bytes, not the bytes written, so size a buffer from it and call again.

```c
#include <stdio.h>
#include <stdlib.h>
#include "turbospark.h"

/* Prints the calling thread's last error, sized to fit. */
void report_error(const char *call, int32_t rc) {
    size_t n = ts_last_error(NULL, 0);
    char *buf = malloc(n + 1);
    if (buf == NULL) {
        fprintf(stderr, "%s failed with status %d\n", call, rc);
        return;
    }
    ts_last_error(buf, n + 1);
    fprintf(stderr, "%s failed with status %d: %s\n", call, rc, buf);
    free(buf);
}
```

Rules for the message:

- It is stored per **thread**. A call on a worker thread writes a message that only that worker can read.
- Every guarded entry point clears the message when it starts. A call that succeeds leaves the message empty. A message from an earlier call cannot be read as the current one's.
- `ts_last_error` itself does not clear the message, so you can read it more than once.
- Messages are diagnostic text. Show them to a person. Do not parse them. The copy truncates at a byte boundary.

## Panics

Rust panics cannot unwind across the C boundary, so every entry point catches them. A caught panic returns `TS_ERR_PANIC` and records `panic caught at the FFI boundary: ...` in the message. The operation does not happen.

A panic inside a turn can leave the session's engine lock poisoned.

After that, turn and vision calls on the session report `the session is poisoned by an earlier panic` instead of running. Read-only calls such as `ts_session_info_json` do not take that lock.

Treat a poisoned session as lost. Release it and open a new one, because the library does not recover the handle.

`ts_cosine_similarity` returns `float`, not a status, so it cannot return `TS_ERR_PANIC`. It returns `0.0` for a caught panic, and also for a `NULL` pointer or zero length. Only a caught panic sets `ts_last_error`, so an empty message after a `0.0` means a `NULL` pointer or zero length, and a non-empty message means a panic.

## What to free and how

Every output has one owner and one release call. Use the release call listed here and no other.

| Output | Release with | Notes |
| --- | --- | --- |
| `char **` string output (most calls: `result_json`, `info`, `out_prompt`, token JSON, catalog and install JSON, server ids, embedding JSON) | `ts_string_free` | Pass `NULL` as a no-op. Never call libc `free()` on it. |
| `uint8_t *out_png` from `ts_image_generate` | `ts_image_buffer_free(ptr, len)` | Pass the length the call returned. `NULL` with length 0 means no PNG. |
| `TsSession *`, `TsTokenizer *`, `TsImageSession *`, `TsServer *`, STT handles | the matching close or stop call | See [Opening a model](opening-a-model.md) and [Server entry points](server-entry-points.md). |
| `TsHeavyWorkPermit *` from `ts_heavy_work_try_acquire` | `ts_heavy_work_release` | Only for external MLX hosts. `NULL` from a successful acquire means the permit is busy. |
| `const char *` arguments | the caller | Borrowed for the call. The library does not keep them. |
| Callback `text` pointers | the library | Borrowed for the callback only. Copy before you return. |
| The `ts_last_error` buffer | the caller | You own the buffer, and the library only writes into it. |

Rules that prevent most memory errors:

- **Use `ts_string_free` for every `char **` output.** The library allocates these as Rust `CString` values. A `free()` from C is a different allocator and corrupts the heap.
- **Free each output once.** After a release, set the pointer to `NULL`.
- **Initialize every out pointer to `NULL`.** Free only what a `TS_OK` return filled. Do not rely on an out pointer being untouched after a failure. `ts_image_generate` clears its outputs first, but other calls only write on success.
- **Do not hold a pointer into a library-owned string after a release.** Copy the text into your own storage first.

A short example of the rule in practice:

```c
#include <stdio.h>
#include "turbospark.h"

void show_path(const char *alias) {
    char *path = NULL;                            /* initialize the out pointer */
    int32_t rc = ts_model_resolve_path(alias, &path);
    if (rc != TS_OK) {
        return;                                   /* path is untouched on failure */
    }
    printf("%s\n", path);
    ts_string_free(path);                         /* the only correct free */
}
```

## Argument rules

These apply across the ABI, with a few named exceptions in the headers:

- A required string that is `NULL` returns `TS_ERR_INVALID_ARGUMENT` and names the argument.
- An optional string that is `NULL` or empty means "absent". A Swift `String?` may arrive as either, so the library treats them the same.
- A string that is not UTF-8 returns `TS_ERR_INVALID_ARGUMENT`.
- A JSON argument that does not parse returns `TS_ERR_JSON`, and the message names the argument, such as `options: ...`.
- Most calls return `TS_ERR_INVALID_ARGUMENT` for a missing output pointer. `ts_generate` checks `result_json` before it does any work, so a bad call does not cost a full turn.
- The close and stop calls for sessions, tokenizers, image sessions, and servers accept `NULL` and return nothing. Other release calls return a status. Check each declaration in the header.

## Limits

- The source's status code for an off-macOS open differs from the header. See [Status codes](#status-codes).
- The header says `ts_cosine_similarity` reports a caught panic through `ts_last_error`. Check that before you rely on the message for a `0.0` result.
- A message is per thread, so a host that dispatches calls to a pool must read the message on the thread that made the call.
