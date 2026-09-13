---
title: Server and Model Management (C ABI)
description: The in-process HTTP server and the catalog, probe, install and recommend calls on the turbospark.h C ABI.
diataxisType: reference
---

<!-- generated: cpp lane, signal: crates/ffi/include/turbospark.h -->

Server and model-management half of `crates/ffi/include/turbospark.h`.
Signatures are copied verbatim from the header; prose under each is the
header's own comment. See [C ABI Contract and Constants](/reference/cpp-abi-contract)
for the status codes and the ownership rules.

## In-process HTTP server

```c
int32_t ts_server_start(const TsSession *s, const char *options_json,
                        TsServer **out);
```

Starts an in-process HTTP server sharing ALREADY-OPEN models, and writes a
handle to `*out`. Serves the same routes turbospark-server does:
`GET /health`, `POST /v1/chat/completions`, `POST /v1/completions`,
`POST /v1/responses`, `POST /v1/messages`, `POST /v1/messages/count_tokens`,
`GET /v1/models`, `GET /v1/models/{id}`, and the Ollama-compatible
`GET /api/tags`, `GET /api/version`, `POST /api/show`, `POST /api/chat`,
`POST /api/generate`.

`s` MAY BE NULL, meaning start with nothing attached. The server binds and
answers `GET /health` (reporting `"state": "empty"`); every generation route
returns 503 until `ts_server_attach_session()` adds a model. That is the
state a GUI starts a server in before its user has chosen what to load, and
passing a non-NULL `s` is exactly equivalent to starting NULL and attaching
immediately.

`options_json` may be NULL or `"{}"`. Recognised keys:

- `port`: number (default 0, meaning let the OS choose; read the port ACTUALLY
  bound back from `ts_server_info_json`)
- `apiKey`: string \| null (default null, meaning no auth, appropriate for a
  server bound to loopback and reachable only by the process embedding it)

THE SERVER OUTLIVES EVERY SESSION ATTACHED TO IT. It holds its own reference
to each underlying engine, so calling `ts_session_close(s)` after this call
frees only the caller's own handle: the model stays resident and the server
keeps serving it until `ts_server_detach_model()` removes that one, or
`ts_server_stop()` releases them all. Do one of those if a model should
actually be freed.

This server serves images on both endpoints when the serving session's
install carries a usable tower (see `sessionInfo.vision.active`), encoding
and generating under one lock exactly as the standalone binary does. A request
it cannot serve is refused by name rather than silently dropped.

Blocks until the socket is bound (or binding fails), not until the first
request is served.

```c
int32_t ts_server_attach_session(const TsServer *server, const TsSession *s,
                                 char **out);
```

Adds an open session's model to a RUNNING server, and writes its canonical id
to `*out` (the install directory's own name, the same string
`ts_server_info_json` reports in `"models"`). Free it with `ts_string_free`.

Takes effect immediately: no rebind, and no interruption to a request already
in flight on another model.

REFUSES a collision across every public id rather than renaming it. Each
generative session additionally advertises
`claude-turbospark-<canonical-id>` through `GET /v1/models`, and a canonical
id may not collide with another session's alias. The canonical id is what
`ts_server_detach_model` keys on, so a silently suffixed second copy would be
addressable under a name the caller never learned. Two sessions on one install
directory is a caller mistake.

WHICH MODEL SERVES A REQUEST: an exact canonical id or discovery-alias match
wins. Failing that, if exactly ONE model is attached it serves the request
whatever name was asked for, which is what keeps a client sending its own
default name (Claude Code sends `"claude-sonnet-4-6"`) working. With two or
more attached and no match, the request is a 404 naming what IS available.

```c
int32_t ts_server_detach_model(const TsServer *server, const char *model_id);
```

Removes a model from a running server by id, releasing the server's reference
to its engine.

TS_OK when one was attached under `model_id`, TS_ERR_INVALID_ARGUMENT when
none was, reported rather than silently succeeding because at that point the
caller's own model list and the server's have gone out of step.

```c
void ts_server_stop(TsServer *server);
```

Signals the server to stop, blocks until its background thread has actually
exited, and frees the handle. NULL is a no-op. This is also what releases
every attached model.

```c
int32_t ts_server_info_json(const TsServer *server, char **out);
```

`{ "port", "host", "modelId", "models", "authEnabled", "uptimeSeconds" }` as
JSON.

- `"port"` is the port ACTUALLY bound, never the one requested: port 0 in
  `ts_server_start`'s options asks the OS to choose one, so this is the only
  place that number is knowable.
- `"host"` is the IP ACTUALLY bound, from the same call, and is what a caller
  should build a URL out of. It reads `"127.0.0.1"` today because that is what
  this library binds; restating that literal instead of reading this field is
  correct only for as long as that stays true, and cannot report the day it
  does not.
- `"models"` is every attached canonical id in attachment order. `GET
  /v1/models` reports each canonical id followed by its Claude discovery
  alias; aliases are intentionally absent here so the host can retain stable
  attach/detach identities. `"modelId"` is the FIRST canonical id (`""` when
  none), kept for a reader written when a server could serve only one; on a
  two-model server it is half the truth, so show `"models"`.
- `"uptimeSeconds"` is from a monotonic clock and is unaffected by the wall
  clock moving under a long-running host.

```c
int32_t ts_server_poll_events_json(const TsServer *server, uint32_t max,
                                   char **out);
```

Takes up to `max` buffered request events and writes
`{ "events": [...], "dropped": N }` to `*out`. Free it with `ts_string_free`.

DRAINING, NOT PEEKING: an event is returned exactly once. Poll this on a
timer and append what comes back to your own log.

Each event is an object tagged by `"kind"`:

- `requestStarted` `{ id, atMs, method, path }`
- `requestRouted` `{ id, requested, served, stream }`
- `generated` `{ id, model, promptTokens, newTokens, prefillSeconds,
  decodeSeconds, stopReason }`
- `requestFinished` `{ id, status, durationMs }`
- `modelAttached` `{ atMs, model }`
- `modelDetached` `{ atMs, model }`

`"id"` ties the events of one request together. `"requested"` and `"served"`
differ whenever the single-model fallback fired, which is the common case
rather than an edge one.

THERE IS NO TIME-TO-FIRST-TOKEN FIELD, and its absence is the honest answer:
nothing inside a generation can measure one, because a caller means "request
in, first token out" and that includes the wait behind the
one-generation-at-a-time lock. Subtract `"prefillSeconds"` +
`"decodeSeconds"` from `requestFinished`'s `"durationMs"` to get that wait.

A request the tool-call guardrails re-asked emits TWO `"generated"` events,
which is the useful reading rather than a duplicate: the retry is real work
the machine did.

`max` bounds ONE call rather than the buffer, so anything over it stays queued
for the next poll and a burst arrives late rather than being lost.
`max == 0` means no bound, which is what a host draining before shutdown
wants.

`"dropped"` counts events discarded since the PREVIOUS poll, oldest first,
and is nonzero only for a host that stopped draining long enough to overrun
about 2,000 events. Show it: a silently lossy log is indistinguishable from an
idle server.

## Model management (available on every platform)

```c
int32_t ts_catalog_json(char **out);
```

The curated catalog as a JSON array, each row carrying `"installed"`.

```c
int32_t ts_installed_json(char **out);
```

What is installed in `~/.turbospark`, as a JSON array.

```c
int32_t ts_model_delete(const char *alias);
```

Deletes an installed model directory and forgets it from `~/.turbospark`.
Returns TS_OK on success or TS_ERR_INVALID_ARGUMENT if not installed.

```c
int32_t ts_recommend_json(uint32_t context_window, const char *options_json,
                          char **out);
```

Ranks curated models by hardware fit on this machine for `context_window`
tokens (e.g. 4096 or 8192, 0 means default 4096). Returns JSON array of
recommendations.

`options_json` may be NULL, `""` or `"{}"`, all meaning every default. One
key:

- `loadGuard`: same spellings as `ts_session_open`'s, and it MUST be the same
  value the host will OPEN with. This ranking and the loader's refusal share
  one memory budget by construction, which is what makes a recommendation
  trustworthy; ranking under `"relaxed"` while sessions open under `"strict"`
  promises a fit the loader then refuses, where the user cannot see the two
  disagree.

```c
int32_t ts_probe_json(const char *repo, const char *file,
                      const char *sidecar_repo, char **out);
```

Probes a Hugging Face repository by HEADER ALONE: kilobytes and seconds, no
download. `repo` is `"owner/name"` or `"owner/name@revision"`. `file` and
`sidecar_repo` may be NULL.

The result carries `"runnable"` and, when false, `"refusedBecause"`. Read
`"slotCacheBytes"` before `"downloadBytes"`: what decides whether a model runs
here is slots x layers x expert stride, not the model's size.

```c
int32_t ts_install_bytes_json(const char *alias, char **out);
```

What installing `alias` will cost, as
`{"downloadBytes","installBytes"}`. Call before `ts_install` to show a
determinate bar and a space warning.

```c
int32_t ts_install(const char *alias, TsInstallCallback cb, void *userdata,
                   char **result_json);
```

Installs the catalog row `alias`. Blocks for the whole walk, which is minutes
to tens of minutes.

THE WALK CANNOT RESUME. It streams the checkpoint without writing it to disk
whole, and a failure restarts from the beginning. Tell the user that BEFORE
starting, not after failing; the first stage line says so.

`cb` receives `TS_INSTALL_STAGE` lines on the calling thread and
`TS_INSTALL_BYTES` updates FROM WORKER THREADS, CONCURRENTLY and possibly out
of order, because ranged downloads are split across connections. A callback
that touches shared state must synchronise it.

```c
int32_t ts_install_repo(const char *repo, const char *alias, const char *file,
                        const char *sidecar_repo, TsInstallCallback cb,
                        void *userdata, char **result_json);
```

Probes and installs an arbitrary Hugging Face repository `repo` under local
`alias`. `file` and `sidecar_repo` may be NULL. Blocks for the whole walk and
cannot resume.
