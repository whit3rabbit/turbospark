---
description: "Run a generation turn from C, stream its events through a callback, cancel it from another thread, and read the result JSON. Includes image generation."
icon: code
---

# Streaming events

A turn runs through `ts_generate`. The call blocks until the turn ends, and it reports progress through a callback while it runs. Call it from a background thread so the UI stays responsive. This page covers the call, the event kinds, the result object, cancellation, and image generation.

## Run a turn

```c
int32_t ts_generate(const TsSession *s, const char *messages_json,
                    const char *options_json, TsEventCallback cb,
                    void *userdata, char **result_json);
```

`messages_json` is an array of messages rendered through the checkpoint's own chat template:

```json
[{"role": "system", "content": "Answer in one sentence."},
 {"role": "user", "content": "Why is the sky blue?"}]
```

Roles are `system`, `developer`, `user`, `assistant`, and `tool`. A message's `content` is either a string or an ordered array of parts. An image part carries exactly one of `path` (a file this process can read) or `base64` (a bare payload or a full `data:` URL):

```json
{"role": "user", "content": [
  {"type": "image", "path": "/abs/page.png"},
  {"type": "text", "text": "Transcribe this."}]}
```

Put images before the text. The template renders one marker per image, and the nth image binds to the nth marker. Reordering the parts pairs each picture with the wrong span, and nothing reports an error. Image parts are refused by name when `vision.active` is `false` in the session info.

`options_json` may be `NULL` or `"{}"`. The defaults are:

| Key | Default | Notes |
| --- | --- | --- |
| `maxNewTokens` | 512 | Clamped to what the context window leaves after the prompt. |
| `temperature` | 0.2 | `0` makes decoding deterministic. Speculative decoding runs only on deterministic turns. |
| `topK` | 64 | |
| `topP` | 0.95 | |
| `repetitionPenalty` | 1.0 | |
| `minP` | 0.0 | Must lie in `[0, 1)`. Out-of-range values are refused by name. |
| `presencePenalty`, `frequencyPenalty` | 0.0 | Must lie in `[-2, 2]`. |
| `seed` | `null` | A number fixes the sampler stream. |
| `stop` | `[]` | Stop strings. |
| `stopTokens` | `[]` | Token ids that end the turn. |
| `reasoning` | `"off"` | `"off"`, `"low"`, `"medium"`, `"high"`, or `"xhigh"`. The accepted set belongs to the checkpoint. Read `reasoningLevels` in the session info. |
| `tools` | `[]` | Functions the model may call this turn. See [Tool calls](#tool-calls). |

Pass `cb` as `NULL` to get the whole turn in `result_json` with no streaming.

## Event kinds

The callback has this shape:

```c
typedef void (*TsEventCallback)(void *userdata, int32_t kind,
                                const char *text, size_t len,
                                uint32_t a, uint32_t b);
```

| Kind | `text` | `a` | `b` | Meaning |
| --- | --- | --- | --- | --- |
| `TS_EVENT_PREFILL` (0) | empty | prompt tokens done | prompt tokens total | Prompt processing progress. |
| `TS_EVENT_CONTENT` (1) | reply text | 0 | 0 | Visible reply text. Accumulate this as the assistant turn. |
| `TS_EVENT_REASONING` (2) | reasoning text | 0 | 0 | Reasoning, already separated from the reply. Do not send it back as history. |
| `TS_EVENT_TOOL` (3) | call as JSON object | zero-based index in the turn | 0 | One parsed tool call. Fires only for a function you offered in `tools`. |
| `TS_EVENT_FINISH` (4) | stop reason | generated tokens | prompt tokens | Terminal event for a successful turn. Fires once, just before `ts_generate` returns. |

Rules for the callback:

- `text` is UTF-8 and is **not** NUL-terminated. Its length is `len`. Copy it before the callback returns. The pointer is invalid after that.
- The callback runs on the thread that called `ts_generate`, and it runs inside the call. Do not block in it for long, and do not call back into the same session from it.
- Ignore any kind you do not recognize. Newer libraries can add kinds.
- `TS_EVENT_FINISH` does not fire when the run fails. A failure is a non-zero return and `ts_last_error`. A cancelled turn does fire it, with the stop reason `"cancelled"`.

## The result object

`result_json` receives one string on success. Free it with `ts_string_free`.

```json
{
  "promptTokens": 42,
  "newTokens": 118,
  "reusedPrefixTokens": 30,
  "prefillSeconds": 0.21,
  "decodeSeconds": 1.7,
  "stopReason": "endOfTurn",
  "toolCalls": [],
  "tokensPerSecond": 69.4,
  "content": "The sky looks blue because ...",
  "reasoning": "",
  "peakMemoryPressure": "normal"
}
```

- `reusedPrefixTokens` counts prompt tokens that continued from the previous turn's KV cache. It is `0` on a first turn, never an error.
- `tokensPerSecond` is `null` when no decoding happened. Do not plot a rate that was never measured.
- `peakMemoryPressure` is `"normal"` when nothing was watching. The default power profile does not watch memory, so `"normal"` can mean no reading. Read `ts_system_info_json` for the machine's current state.

`stopReason` is one of `endOfTurn`, `toolCalls`, `eos`, `stopString`, `maxTokens`, or `cancelled`. The same string is the `text` of `TS_EVENT_FINISH`. A turn that produced a parsed tool call reports `toolCalls`, whichever token closed it.

## Stream a reply

This example reads the event stream and prints the reply. A GUI calls it from a worker thread. `main` is used here for brevity.

```c
#include <stdio.h>
#include "turbospark.h"

typedef struct {
    int finished;
} Turn;

static void on_event(void *userdata, int32_t kind, const char *text,
                     size_t len, uint32_t a, uint32_t b) {
    Turn *turn = userdata;
    switch (kind) {
    case TS_EVENT_CONTENT:
        /* Written out now, so the pointer does not outlive the call. */
        fwrite(text, 1, len, stdout);
        break;
    case TS_EVENT_FINISH:
        turn->finished = 1;
        printf("\n[stop=%.*s new=%u prompt=%u]\n",
               (int)len, text, (unsigned)a, (unsigned)b);
        break;
    default:
        /* Unknown or unused kinds are no-ops. */
        break;
    }
}

int ask(const TsSession *session) {
    const char *messages =
        "[{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}]";
    const char *options = "{\"maxNewTokens\": 128, \"temperature\": 0}";
    Turn turn = {0};
    char *result = NULL;

    int32_t rc = ts_generate(session, messages, options, on_event, &turn, &result);
    if (rc != TS_OK) {
        /* Read ts_last_error here, before any other call. See Errors and strings. */
        return rc;
    }
    /* result holds the full result object shown above. */
    puts(result);
    ts_string_free(result);
    return turn.finished ? TS_OK : TS_ERR_GENERATE;
}
```

## Cancel from another thread

`ts_session_cancel` is safe from any thread and never blocks. The Stop button calls it while the worker thread is still inside `ts_generate`.

```c
#include <pthread.h>
#include <unistd.h>
#include "turbospark.h"

static void *press_stop(void *arg) {
    sleep(3);                       /* stands in for a user pressing Stop */
    ts_session_cancel((const TsSession *)arg);
    return NULL;
}

/* Runs on a worker thread. The turn returns TS_OK with stopReason "cancelled". */
void run_with_stop(const TsSession *session, const char *messages,
                   TsEventCallback cb, void *userdata, char **result) {
    pthread_t stopper;
    pthread_create(&stopper, NULL, press_stop, (void *)session);
    ts_generate(session, messages, "{}", cb, userdata, result);
    pthread_join(stopper, NULL);
}
```

What cancellation does:

- A cancelled turn returns `TS_OK`. It is not an error, so do not show it as one. `stopReason` is `"cancelled"`, and `content` holds the partial text.
- The KV cache describes the partial turn honestly, so the conversation can continue from it.
- The flag is cleared when the next turn starts. A Stop pressed between turns cancels nothing.
- Calling `ts_session_cancel` on `NULL` is a no-op.

### Queueing behind other work

Text turns, image generations, and in-process server requests share one process-wide permit. `ts_generate` and `ts_humanize` wait for it, and the wait is not interruptible. If another session is generating an image, your turn waits until that finishes.

The source has one gap here. `ts_generate` takes the permit (`crates/ffi/src/api/generate.rs`) before `run_turn` arms the cancel flag (`crates/ffi/src/generate/mod.rs`), and arming clears the flag. A Stop pressed before arming, including while the turn waits in the queue, may be lost.

This follows from reading the source. It has not been reproduced. Image generation does not have this gap, because its cancel token exists before it waits. Until the gap is fixed, do not treat Stop as certain while a turn is queued.

## Tool calls

To let the model call functions, offer them in `tools`:

```json
{"tools": [{"name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}]}
```

- The definitions count toward the prompt. Budget for them with `ts_session_count_tokens_with_tools` and `ts_session_fit_window_with_tools_json`.
- A parsed call arrives as a `TS_EVENT_TOOL` event and in `toolCalls`. Its `arguments` is a JSON object, not a string.
- A call is parsed only in the checkpoint's native markup and only for a function you offered. Check `toolCalling.native` in the session info.
- To answer a call, append the assistant turn with `"toolCalls"` and a `tool` message with `"toolCallId"` and `"name"`. Keep the ids the engine returned. Some templates match a tool result to its call by id.

## Humanize a draft

`ts_humanize` rewrites a draft with the `humanizer-12b` model. The session must be open on that model, with `maxContext` 8192.

```c
int32_t ts_humanize(const TsSession *s, const char *draft,
                    const char *options_json, TsEventCallback cb,
                    void *userdata, char **result_json);
```

The options accept only `maxNewTokens` and `seed`. Any other key fails the call and names the key. The sampling recipe and stop rules are fixed inside the library.

The result is the `ts_generate` result with an extra `audit` object. The audit lists dropped or invented numbers, a script switch, and verbatim reuse.

Each audit field is a warning for a proofreader, not a correction. On a chat model, `ts_humanize` returns fluent nonsense rather than an error.

## Generate an image

Image generation uses a separate handle, opened as described in [Opening a model](opening-a-model.md#open-an-image-session).

```c
int32_t ts_image_generate(const TsImageSession *s, const char *options_json,
                          TsImageEventCallback cb, void *userdata,
                          uint8_t **out_png, size_t *out_png_len,
                          char **out_metadata_json);
void ts_image_session_cancel(const TsImageSession *s);
void ts_image_buffer_free(uint8_t *bytes, size_t len);
```

`options_json` takes `prompt`, `seed`, `width`, `height`, and `steps`. Width and height default to 1024, accept sides from 512 through 1024 in multiples of 16, and cap at 1024 x 1024 pixels. Steps default to 9.

The callback receives stage events and one finish event:

| Kind | `text` | `a` | `b` |
| --- | --- | --- | --- |
| `TS_IMAGE_EVENT_STAGE` (0) | `text_encoder`, `transformer`, `vae_decoder`, or `png_encode` | completed units | stage total |
| `TS_IMAGE_EVENT_FINISH` (1) | `finished` | 1 | 1 |

On success, the call returns the PNG bytes and a metadata object. A cancelled run has `"status": "cancelled"` in the metadata and a zero-length PNG. The finish event fires on any successful return, including a cancelled one. It does not fire on error.

```c
#include <stdio.h>
#include "turbospark.h"

static void on_image_event(void *userdata, int32_t kind, const char *text,
                           size_t len, uint32_t a, uint32_t b) {
    (void)userdata;
    if (kind == TS_IMAGE_EVENT_STAGE) {
        printf("%.*s %u/%u\n", (int)len, text, (unsigned)a, (unsigned)b);
    }
}

int render(const TsImageSession *image) {
    const char *options =
        "{\"prompt\": \"a lighthouse at dusk\", \"seed\": 42, "
        "\"width\": 512, \"height\": 512, \"steps\": 9}";
    uint8_t *png = NULL;
    size_t png_len = 0;
    char *metadata = NULL;

    int32_t rc = ts_image_generate(image, options, on_image_event, NULL,
                                   &png, &png_len, &metadata);
    if (rc != TS_OK) {
        return rc;
    }
    puts(metadata);
    ts_string_free(metadata);
    if (png_len > 0) {
        /* Write png[0 .. png_len) to disk or to an image view here. */
    }
    ts_image_buffer_free(png, png_len);   /* the length returned above, not strlen */
    return TS_OK;
}
```

Image cancellation has one rule to remember. `ts_image_session_cancel` cancels only the generation that is running. A cancel issued before `ts_image_generate` starts is lost, because each call installs a fresh token. Track "cancel the next one" in your own code.

## Limits and thread safety

- A session runs one turn at a time. A second `ts_generate` on the same handle waits, so do not start one until the first returns.
- Do not call `ts_session_close` while a turn runs on another thread.
- `result_json` and every other `char **` output are owned by the caller. Free each with `ts_string_free`. See [Errors and strings](errors-and-strings.md#what-to-free-and-how).
- Each event is delivered inside the call. A host that wants a live UI must marshal the text to its main thread, not draw from the callback directly.
