---
description: "Run the in-process HTTP server from C: start it, attach and detach models, read its state, drain its events, and stop it. Covers ownership, lifetimes, and the API key rules."
icon: code
---

# Server entry points

The in-process server exposes TurboSpark over HTTP from inside your host process. It serves the same routes as the standalone [`turbospark-server`](../server/README.md), but it shares the models your host already opened. It does not load a second copy of a model. Use it when a host wants a local OpenAI-compatible or Anthropic-compatible endpoint backed by the models in its own process.

The server binds to a socket and runs on its own thread. `ts_server_start` returns after the socket is bound, not after the first request is served.

## Start a server

```c
int32_t ts_server_start(const TsSession *s, const char *options_json,
                        TsServer **out);
```

`s` may be `NULL`. The server then starts with no model attached. It answers `GET /health` with `"state": "empty"`, and every generation route returns 503 until you attach a model. Passing a non-null `s` is the same as starting with `NULL` and calling `ts_server_attach_session` at once.

`options_json` may be `NULL` or `"{}"`. The keys, from `wire::ServerOptions`:

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | `"127.0.0.1"` | Literal IP address to bind. Loopback or a Tailscale IPv4 address in `100.64.0.0/10`. Other addresses are refused. |
| `port` | `0` | `0` asks the OS for a free port. Read the port actually bound from `ts_server_info_json`. |
| `apiKey` | `null` | Required on every request except `GET /health` when set. Required for a Tailscale address. |
| `captureText` | `false` | Keeps bounded request body previews in memory, for debugging. Headers and API keys are excluded. |
| `guardrails` | `null` (on) | `"on"` or `"off"`. Applies to the whole server, including models attached later. Any other string is an error. |
| `defaultSystem` | `null` | System prompt for requests that carry no system or developer message. |
| `defaultReasoning` | `null` (off) | Reasoning effort when a request omits it: `off`, `low`, `medium`, `high`, or `xhigh`. |
| `embeddingModel` | `null` | Embedding model to attach at startup. Enables the embedding routes. |
| `hfEndpoint` | `null` | Hugging Face mirror used by installs from this process. |
| `sttModels`, `ttsModels`, `musicModels` | `[]` | Audio models to attach at startup. See [Audio models](#image-and-audio-models). |
| `idleUnloadSeconds` | `null` (never) | Detach a model after this many idle seconds. |
| `allowRequestModelLoad` | `false` | Let a request that names an installed but unattached model open and attach it. Opt-in, because it lets any client grow the server's memory use. |
| `maxAttachedModels` | 8 | Ceiling on attached models, host-attached and request-loaded together. |
| `maxAttachedModelBytes` | no ceiling | Cumulative ceiling on committed weights and KV cache. An attach that would exceed it is refused. |

A minimal start that binds an ephemeral loopback port and attaches one open session:

```c
#include <stdio.h>
#include "turbospark.h"

int serve(const TsSession *session) {
    TsServer *server = NULL;
    int32_t rc = ts_server_start(session, "{\"port\": 0}", &server);
    if (rc != TS_OK) {
        return rc;                 /* read ts_last_error here */
    }

    char *info = NULL;
    rc = ts_server_info_json(server, &info);
    if (rc == TS_OK) {
        puts(info);                /* contains the bound port and host */
        ts_string_free(info);
    }

    /* ... serve until the host shuts down ... */

    ts_server_stop(server);        /* blocks until the server thread exits */
    return TS_OK;
}
```

## Lifetimes

The server holds its own reference to every attached model's engine. Closing the session you started from does not free the model while the server still serves it.

- `ts_session_close(s)` drops only your handle. The model stays resident and keeps serving.
- `ts_server_detach_model` releases one model.
- `ts_server_stop` releases every attached model and joins the server thread.

If a model should actually be freed, detach it or stop the server. Closing the session alone is not enough.

## Attach and detach models

```c
int32_t ts_server_attach_session(const TsServer *server, const TsSession *s,
                                 char **out);
int32_t ts_server_detach_model(const TsServer *server, const char *model_id);
int32_t ts_server_attach_embedding_model(const TsServer *server,
                                         const char *model_path, char **out);
int32_t ts_server_set_idle_unload(const TsServer *server, uint64_t seconds);
```

- `ts_server_attach_session` writes the model's id to `*out`. The id is the install directory's own name. Free it with `ts_string_free`. Attaching takes effect at once, without a rebind and without interrupting a request on another model.
- The id must be unique. A second session on the same install directory is refused, not renamed. Two sessions on one install is a caller mistake.
- Each generative model is also reachable under a derived discovery alias, `claude-turbospark-<id>`. `ts_server_info_json` reports canonical ids only, so detach by the id you attached.
- A request selects a model by exact id or alias. If exactly one model is attached, it serves a request whatever name the request used. With two or more and no match, the request is a 404 that lists what is available.
- `ts_server_detach_model` returns `TS_ERR_INVALID_ARGUMENT` if no model has that id. A detach that finds nothing is reported, so your model list and the server's do not drift apart silently.
- `ts_server_set_idle_unload(server, 0)` turns the idle sweep off. The sweep runs every 30 seconds. It never detaches a model with a request in flight.

`ts_server_attach_embedding_model` writes the assigned id to `*out` and enables `/v1/embeddings`, `/api/embeddings`, and `/api/embed`.

## Read the server state

```c
int32_t ts_server_info_json(const TsServer *server, char **out);
```

The response is an object with these keys: `port`, `host`, `modelId`, `models`, `imageModels`, `audioModels`, `authEnabled`, `uptimeSeconds`, and `traffic`.

- `port` and `host` are the values actually bound. Build URLs from them, not from the values you asked for.
- `models` lists every attached canonical id in attach order. Prefer it. `modelId` is only the first entry, so on a two-model server it is half the picture.
- `uptimeSeconds` comes from a monotonic clock, so a wall-clock change does not affect it.

## Drain request events

```c
int32_t ts_server_poll_events_json(const TsServer *server, uint32_t max,
                                   char **out);
```

Each call drains buffered events and returns `{"events": [...], "dropped": N}`. Free the result with `ts_string_free`.

- Draining is one-way. An event comes back once, so poll on a timer and keep what you need.
- `max` bounds one call. Events beyond it stay queued for the next poll. `max` of 0 means no bound.
- `dropped` counts events discarded since the previous poll, because the host stopped draining. Show it. A silently lossy log looks the same as an idle server.

Each event has a `kind`: `requestStarted`, `requestRouted`, `generated`, `requestFinished`, `modelAttached`, or `modelDetached`. The header gives the fields of each kind. For queue wait, join the `generated` and `requestFinished` events on `id`. `durationMs` on `requestFinished` is in milliseconds and includes time spent behind other requests. `prefillSeconds` and `decodeSeconds` on `generated` are in seconds and exclude it. The wait is approximately `durationMs / 1000 - prefillSeconds - decodeSeconds`. The server does not report it directly.

## Stop the server

```c
void ts_server_stop(TsServer *server);
```

`ts_server_stop` signals the server, waits for its thread to exit, releases every attached model, and frees the handle. `NULL` is a no-op. Do not call any other server function on the handle after it.

## Security

- **No `apiKey` means no authentication.** A loopback bind keeps the socket off the network, but every process on the machine can reach it. Any local program that finds the port can drive the model. Set `apiKey` if that matters to you.
- **A Tailscale bind requires an `apiKey`.** The start is refused otherwise.
- **The server does not provide TLS.** Keep it on loopback or a Tailscale address you trust.
- `guardrails` is process-wide on purpose. A per-request setting would let any client opt its own traffic out of the repair the deployment chose.

## Image and audio models

Image and speech models can attach to the same listener, and the host serves their requests. These calls take a model id and return a status:

```c
int32_t ts_server_attach_image_model(const TsServer *server, const char *model_id);
int32_t ts_server_detach_image_model(const TsServer *server);
int32_t ts_server_attach_audio_speech_model(const TsServer *server, const char *model_id);
int32_t ts_server_detach_audio_speech_model(const TsServer *server);
int32_t ts_server_attach_audio_model(const TsServer *server, const char *model_path,
                                     const char *task, char **out);
int32_t ts_server_detach_audio_model(const TsServer *server, const char *model_id);
```

The image and speech attach calls expect the host to drain a worker queue and return results by request id. Their poll and complete calls are declared in the header. This page does not document their event shapes. Read the declarations and `crates/ffi/src/server_image.rs` and `server_speech.rs`.

`sttModels`, `ttsModels`, and `musicModels` in `ts_server_start` attach through the audio path. Those attaches only happen on macOS. Elsewhere the source ignores the lists silently, so do not rely on a start with audio models to fail off macOS.

## Limits and thread safety

- Do not use the server handle after `ts_server_stop`.
- A generation request waits for the process-wide heavy-work permit, the same one that text turns and image generation use. A busy engine makes requests queue. See [Streaming events](streaming-events.md#queueing-behind-other-work).
- An attached model stays resident until you detach it, stop the server, or the idle sweep detaches it (when `idleUnloadSeconds` is set). Plan memory with `maxAttachedModelBytes` and the idle sweep, not with the session's lifetime.
- The source's event and routing details are in the header. This page does not restate every event field.
