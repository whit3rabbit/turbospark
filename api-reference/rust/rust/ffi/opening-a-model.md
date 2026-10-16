---
description: "Open a TurboSpark session from C, read the values it resolved to, and release it. Covers the tokenizer-only and image handles too."
icon: code
---

# Opening a model

Opening a model maps weights and compiles Metal pipelines, so it is the expensive call in this ABI. Open once, keep the handle, and reuse it for every turn.

This page covers the open call, the options that matter, what to read back, and how to release each handle. For running turns on the handle, see [Streaming events](streaming-events.md).

## Open a session

```c
int32_t ts_session_open(const char *model_dir, const char *options_json,
                        TsSession **out);
```

`model_dir` is either a path to a `.gturbo` directory or an installed alias such as `gemma4`. An existing directory always wins, so a bare name cannot silently select a different model than the one a path names.

`options_json` may be `NULL` or `"{}"`. Every key is optional. The header defines the full list and each key's accepted spellings. The keys you will set most often are:

| Key | Accepts | Meaning |
| --- | --- | --- |
| `maxContext` | number, `"auto"`, or `null` | Context window in tokens. Default is `"auto"`. |
| `expertCacheSlots` | number, `"auto"`, or `null` | Routed-expert cache size. A number must be one of 8, 16, 24, 32, 48, 64, 96, 128. |
| `expertResidency` | `"auto"`, `"streamed"`, `"mapped"`, or `null` | Where routed experts live. The mode actually used is reported back. |
| `powerProfile` | `"performance"`, `"balanced"`, `"efficiency"`, or `null` | `null` asks the OS, so Low Power Mode selects efficiency. Name a profile when you measure. |
| `loadGuard` | `"off"`, `"relaxed"`, `"balanced"`, `"strict"`, a byte number, or `null` | How much memory the engine may commit. `null` means `"relaxed"`. |
| `minAutoContext` | number or `null` | Refuse an automatic window that resolves below this many tokens. Does not apply to an explicit `maxContext`. |
| `speculation` | `"off"`, `"auto"`, a number, `"<number>"`, or `null` | Draft-ahead block size. Resolved at open. |
| `kvBits` | `"off"`, `"2"`, `"3"`, `"3.5"`, `"4"`, or `null` | KV-cache quantization. A named width that cannot be served fails the open. |
| `prefixReuse` | boolean or `null` | Continue each turn from the previous turn's KV cache. Default `true`. |

Steering (`steering`, `steeringMode`, `steeringScale`, `steeringLayers`, `steeringTarget`, `steeringGate`) and vision (`visionSidecar`) keys are in the header.

A minimal open with one explicit window looks like this:

```c
#include <stdio.h>
#include <stdlib.h>
#include "turbospark.h"

/* Copies the thread's last error into a heap buffer sized to fit it. */
static void print_last_error(const char *what, int32_t rc) {
    size_t n = ts_last_error(NULL, 0);
    char *buf = malloc(n + 1);
    if (buf == NULL) {
        fprintf(stderr, "%s failed with status %d\n", what, rc);
        return;
    }
    ts_last_error(buf, n + 1);
    fprintf(stderr, "%s failed with status %d: %s\n", what, rc, buf);
    free(buf);
}

int main(void) {
    TsSession *session = NULL;
    int32_t rc = ts_session_open("gemma4", "{\"maxContext\": 8192}", &session);
    if (rc != TS_OK) {
        print_last_error("ts_session_open", rc);
        return 1;
    }

    char *info = NULL;
    rc = ts_session_info_json(session, &info);
    if (rc == TS_OK) {
        puts(info);
        ts_string_free(info);
    } else {
        print_last_error("ts_session_info_json", rc);
    }

    ts_session_close(session);
    return 0;
}
```

## Read what the session resolved to

`ts_session_info_json` returns what the engine actually chose, not what you asked for. `maxContext` and `expertCacheSlots` are the resolved values. Under `"auto"`, the KV cache was already allocated at the resolved size, so quote the resolved number and never the request. The response also carries these fields, which a host should gate its controls on:

| Field | Gate on it to decide |
| --- | --- |
| `steering.supported` | Whether a control vector can be offered at all. `steering.active` only says one is running. |
| `vision.active` | Whether an image would be served. A tower can exist and still refuse every image, so do not gate on the family name. |
| `reasoningSupport` and `reasoningLevels` | Which reasoning control to show. Build the menu from `reasoningLevels` alone. |
| `toolCalling.native` | Whether this checkpoint's markup carries tool calls the engine parses. `false` does not mean tools fail. |
| `speculation.block` | Whether the session drafts ahead. Non-null means on, but a sampled turn still decodes sequentially. |
| `kvBits` | The width the session actually runs at. |

The header's comment on `ts_session_info_json` describes each field in full.

## Release a session

```c
void ts_session_close(TsSession *s);
```

`NULL` is a no-op. Do not call it while a generation is running on another thread. Close releases only the caller's reference. If a server has attached this session, the server keeps the engine alive until you detach that model or stop the server. See [Server entry points](server-entry-points.md#lifetimes).

## Release vision resources

```c
int32_t ts_session_release_vision(const TsSession *s);
```

This frees the vision tower's open resources, which can be large, without closing the session. The next image sent to `ts_generate` reopens the tower from the same place. It does not change `vision.active`. It returns `TS_ERR_UNSUPPORTED` on a session with no tower to release, and it is a harmless no-op when the tower is already closed. It takes the same engine lock as a turn, so it waits for an in-flight turn to finish.

## Tokenize without a session

A tokenizer opens by itself and reads only the tokenizer files. A context meter can count tokens before any model is loaded.

```c
int32_t ts_tokenizer_open(const char *model_dir, TsTokenizer **out);
void ts_tokenizer_close(TsTokenizer *tokenizer);
int32_t ts_tokenizer_count_text_tokens(const TsTokenizer *tokenizer,
                                       const char *text, bool add_special,
                                       uint32_t *out_count);
int32_t ts_tokenizer_tokenize_json(const TsTokenizer *tokenizer,
                                   const char *text, bool add_special,
                                   char **out);
int32_t ts_tokenizer_detokenize_json(const TsTokenizer *tokenizer,
                                     const char *tokens_json,
                                     bool skip_special, char **out);
```

These calls cover raw text only. Rendering a chat transcript and fitting a window need the family and dialect that the engine resolves at open, so they stay on the session (`ts_session_render_prompt`, `ts_session_fit_window_json`). Free the `char **` outputs with `ts_string_free`.

## Open an image session

Image generation has its own handle, because its backend, memory envelope, and output lifetime differ from a text session.

```c
int32_t ts_image_session_open(const char *model_dir, TsImageSession **out);
void ts_image_session_close(TsImageSession *s);
```

Do not close it while `ts_image_generate` is running. Usage is in [Streaming events](streaming-events.md#generate-an-image).

## Thread safety

- Opening, closing, and reading info are not designed to run concurrently with a turn on the same handle. Treat the handle as single-threaded.
- `ts_session_cancel` is the only call that may come from any thread while a turn runs.
- Different sessions can run on different threads. A process-wide permit still serializes the heavy work, so two sessions do not decode at once. See [Streaming events](streaming-events.md#queueing-behind-other-work).

## Limits

- The Rust doc comment on `ts_session_open` lists only four keys. The header and `wire::OpenOptions` define more. Use the header.
- Off macOS, the open call fails with a sentence in `ts_last_error`. The header says that failure is `TS_ERR_UNSUPPORTED`, but the source returns `TS_ERR_OPEN`. Branch on the non-zero return, not on one specific code. See [Errors and strings](errors-and-strings.md#status-codes).
