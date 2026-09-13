---
title: HTTP API Reference
description: The turbospark-server wire contract: every route, the request fields each one actually reads, response shapes, streaming framing, stop-reason mapping, auth, and tool-call guardrails.
---

<!-- generated: docs-from-code lane, signal: crates/server/src/{lib,auth,completions,messages,ollama,guardrails}.rs, handler/mod.rs, responses/, handler/plan.rs -->

The wire contract of `turbospark-server`. Every route below is registered in
`crates/server/src/lib.rs::build_router_with_options`; request and response
shapes come from the handler named in each entry. For launch flags (model,
port, bind, guardrails, speculation, prefix reuse, session slots), see
[turbospark-server](/reference/cli-turbospark-server/).

## Running and addressing the server

```sh
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo [--port N]
```

- Base URL: `http://127.0.0.1:8080` by default (`--port`, default 8080;
  `--bind loopback` is the default).
- `--bind tailnet` binds the machine's Tailscale IPv4 instead and requires
  `--api-key KEY` or `$TURBOSPARK_API_KEY`; there is no TLS.
- One runner per process. The real backend owns a single `RealForwardRunner`
  behind a mutex (`crates/server/CLAUDE.md` Gotcha 1), so concurrent requests
  queue and are served one at a time.
- Auth is opt-in. With no `--api-key` there is no authentication at all. With
  `--api-key KEY` (or `TURBOSPARK_API_KEY`), every route except `GET /health`
  requires the key. See [Authentication](#authentication).

## Endpoint index

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI chat completions (text, tools, reasoning, images) |
| POST | `/v1/completions` | OpenAI legacy raw-prompt completion, no chat template |
| POST | `/v1/responses` | OpenAI Responses API (item-shaped input/output) |
| POST | `/v1/messages` | Anthropic messages (text, tools, images) |
| POST | `/v1/messages/count_tokens` | Anthropic count-only, no generation |
| GET | `/v1/models` | OpenAI-shaped model list |
| GET | `/v1/models/:model` | OpenAI-shaped model detail |
| POST | `/api/chat` | Ollama chat (NDJSON) |
| POST | `/api/generate` | Ollama generate (NDJSON) |
| POST | `/api/show` | Ollama model details |
| GET | `/api/tags` | Ollama model list |
| GET | `/api/version` | Ollama version probe (reports this server's version) |
| GET | `/health` | Liveness probe, never behind auth |

Every generation endpoint supports `stream`. OpenAI- and Anthropic-shaped
routes stream SSE with a 15 s keep-alive comment; Ollama routes stream NDJSON.
`stream` defaults to false everywhere except the Ollama routes, where it
defaults to true.

Claude Code may send `HEAD /api/hello` to warm a connection before discovery.
That probe is best-effort and this server intentionally returns 404 for it;
`GET /health` is the liveness contract.

## Authentication

Opt-in via `--api-key KEY` (`crates/server/src/auth.rs`). When set, every
route except `GET /health` requires the key. `x-api-key` is checked first;
`Authorization: Bearer <key>` is the fallback. The comparison is
constant-time. `GET /health` is deliberately exempt: a liveness probe must
never answer "no" to an unauthenticated caller.

A missing or wrong key returns 401 with:

```json
{"error": {"message": "invalid or missing API key", "type": "authentication_error"}}
```

```sh
curl -s localhost:8080/v1/models -H 'x-api-key: sk-test'
curl -s localhost:8080/v1/models -H 'authorization: Bearer sk-test'
```

## Model routing

Each request's `model` field is resolved by the registry
(`crates/server/src/registry.rs`, reached through `handler::resolve_backend`):

1. An exact canonical id or advertised alias match wins.
2. Otherwise, if exactly one model is attached, it serves the request whatever
   name was asked for (so `"model": "claude-sonnet-4-6"` works against a
   single-model server).
3. With two or more attached and no match: 404 `model_not_found`, listing the
   available canonical ids and aliases.
4. An empty registry is 503, not 404.

`GET /v1/models/:model` and `/api/show` deliberately do NOT take the
single-model fallback: a lookup asks whether the exact id exists.

## POST /v1/chat/completions

OpenAI-compatible chat completion (`crates/server/src/handler/mod.rs`
`chat_completions`). Requests carrying tools render through the checkpoint's
own Jinja tool template; everything else renders the text-only template.

### Request fields read

| Field | Read as | Notes |
|---|---|---|
| `model` | string | Routing, see above |
| `messages` | array | Roles: `system`, `user`, `assistant`, `tool` (`function` maps to tool), `developer`. Content: string or multipart parts; `image_url` parts are images (below) |
| `tools` | array | OpenAI nested shape `{"type":"function","function":{"name","description","parameters"}}` |
| `tool_choice` | string or object | `"none"` suppresses tools; `"required"` and a named function pin them |
| `max_tokens` / `max_completion_tokens` | u32 | Default 256 |
| `temperature` | f32 | Default 1.0 |
| `top_p` | f32 | Default unset (no top-p filtering) |
| `presence_penalty`, `frequency_penalty` | f32 | Explicit fields, honored |
| `stop` | string or array of strings | Stop strings |
| `stream` | bool | Default false |
| `stream_options.include_usage` | bool | Emits a final usage chunk when streaming |
| `response_format` | object | Warned on `x-anyllm-degradation` when `type` is not `"text"` |
| `reasoning_effort` (top-level extra) | string | `off`, `low`, `medium`, `high`, `xhigh`. Per-request; a misspelled value is refused with 400 |
| `seed`, `top_k`, `repetition_penalty`, `min_p` (top-level extra) | numbers | `top_k` defaults to 64, `repetition_penalty` to 1.0 |
| `n`, `logprobs`, `top_logprobs`, `logit_bias` (top-level extra) | any | `n > 1` and the others are accepted and reported on `x-anyllm-degradation`, not honored |

Messages carry `tool_calls` (assistant turns, `id`/`name`/`arguments`) and
`tool_call_id` (tool turns); both are rendered by the template. An assistant
turn's `reasoning_content` is never rendered back as prose.

### Images

`image_url` content parts with `data:<media_type>;base64,<data>` URLs are
served when the install has a vision tower, spliced before the turn's text.
Remote URLs are refused with 400 rather than fetched. On a text-only install
the image parts are dropped, the text half still answers, and the drop is
reported on `x-anyllm-degradation`.

### Response (non-streaming)

```json
{
  "id": "chatcmpl-<unix-seconds>",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "gemma4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "…",
      "reasoning_content": "… (present when the checkpoint separated reasoning)",
      "tool_calls": [{"id": "toolu_…", "type": "function", "function": {"name": "…", "arguments": "{…}"}}]
    },
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26}
}
```

One choice, always. `tool_calls` appears only when the structured decoder
parsed calls; a whole call arrives in one entry (no fragment streaming).

### Streaming framing

SSE `data:` lines (`object: "chat.completion.chunk"`):

1. Opening chunk: `delta: {"role": "assistant"}`.
2. Content chunks: `delta.content`, and `delta.reasoning_content` before the
   first content delta when the model thinks first.
3. Tool-call chunks: one chunk per parsed call in
   `delta.tool_calls[{index, id, function: {name, arguments}}]`.
4. Closing chunk: `finish_reason` set, empty delta.
5. Usage chunk (only when `stream_options.include_usage`): empty `choices`,
   `usage` populated.
6. `data: [DONE]` sentinel.

A request carrying tools under active guardrails is buffered instead: see
[Tool-call guardrails](#tool-call-guardrails). Mid-generation failures are
emitted as a `data:` line carrying `{"error": ...}` followed by `[DONE]`.

```sh
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "Explain how coastal wetlands reduce flood damage."}],
  "max_tokens": 64
}'
curl -sN localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "gemma4", "stream": true,
  "messages": [{"role": "user", "content": "hi"}]
}'
```

## POST /v1/completions

Legacy OpenAI raw-prompt completion (`crates/server/src/completions.rs`). No
chat template is applied: the prompt is encoded as-is (with BOS, matching
`turbospark-check --prompt`). No tools, no reasoning, no decoder, no
guardrails on this route.

### Request fields read

| Field | Notes |
|---|---|
| `model` | Routing |
| `prompt` | String, or a one-element array. A multi-element array is refused with 400 |
| `max_tokens` | Default 16 |
| `temperature`, `top_p` | Sampling |
| `stop` | String or array |
| `stream` | Default false |
| `suffix` | Refused with 400 (fill-in-the-middle unsupported) |
| `n` (extra) | Must be 1, else 400 |
| `logprobs`, `best_of`, `echo` (extra) | Accepted, reported on `x-anyllm-degradation`, not honored |
| `seed`, `top_k`, `repetition_penalty`, `min_p` (extra) | Read by the shared shaping path |

`presence_penalty`/`frequency_penalty` have no field on this endpoint's wire
shape and are not read here.

### Response and streaming

`object: "text_completion"`; `choices[0].text`, `finish_reason`
(`"length"`, `"tool_calls"`, or `"stop"`), `usage` on the non-streaming body.
The streaming path sends `text_completion` chunks per delta, then a final
chunk with empty text and the finish reason, then `data: [DONE]`.

```sh
curl -s localhost:8080/v1/completions -H 'content-type: application/json' \
  -d '{"model": "m", "prompt": "The capital of France is", "max_tokens": 10}'
```

## POST /v1/responses

OpenAI Responses API (`crates/server/src/responses/`). The request is folded
onto the same internal chat request the other endpoints use; one template
path, one shaping path.

### Request fields read

| Field | Notes |
|---|---|
| `model` | Routing |
| `input` | A plain string, or an array of items: `message` (roles user/assistant/system/developer; content string or typed parts), `function_call`, `function_call_output`. Other item types are refused with 400 |
| `instructions` | Becomes the system message |
| `max_output_tokens` | Maps to `max_tokens` |
| `temperature` | Sampling |
| `stream` | Default false |
| `tools` | FLAT shape `{"type":"function","name","description","parameters"}` (one layer less than chat completions) |
| `top_p`, `tool_choice`, `stop` (extra) | No explicit field on the Responses wire type; read out of the extra map |
| `reasoning_effort`, `seed`, `top_k`, `repetition_penalty` (extra) | Pass through to the shared readers |
| `previous_response_id` (extra) | Refused with 400: this server is stateless and keeps no prior turn |
| `store` (extra) | Accepted, reported on `x-anyllm-degradation`, not honored |

A tool call and its result are root-level items here, not blocks nested on a
message. There is no vision path on this endpoint: image parts of an input
item are skipped.

### Response (non-streaming)

```json
{
  "id": "resp_<unix-seconds>",
  "object": "response",
  "model": "m",
  "status": "completed",
  "output": [
    {"type": "reasoning", "id": "rs_0", "summary": [{"type": "summary_text", "text": "…"}]},
    {"type": "message", "id": "msg_0", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "…", "annotations": []}]},
    {"type": "function_call", "id": "fc_toolu_…", "call_id": "toolu_…", "name": "…",
     "arguments": "{…}", "status": "completed"}
  ],
  "usage": {"input_tokens": 21, "output_tokens": 5, "total_tokens": 26}
}
```

`status` is `"incomplete"` when the token budget ended the turn. The
`reasoning` item appears only on the non-streaming path.

### Streaming framing

Typed SSE events, dispatched on the `event:` line, in this order
(`crates/server/src/responses/sse.rs`):

```text
response.created
  response.output_item.added        (the assistant message item)
  response.content_part.added
  response.output_text.delta        (repeated per token)
  response.output_text.done
  response.content_part.done
  response.output_item.done
  response.function_call_arguments.delta / .done   (per call, with its own
                                                    output_item.added/.done pair)
response.completed                 (or response.failed)
```

There is NO `[DONE]` sentinel: a Responses stream ends at
`response.completed` (or `response.failed`). Reasoning deltas are dropped on
the streaming path (OpenAI defines no delta event for them) and reported on
`x-anyllm-degradation` ahead of the stream when the dialect would have
produced one.

```sh
curl -s localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"model": "m", "input": "hi", "max_output_tokens": 40}'
curl -sN localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"model": "m", "input": "hi", "max_output_tokens": 40, "stream": true}'
```

## POST /v1/messages

Anthropic-compatible messages (`crates/server/src/messages.rs`). The request
is translated to the internal OpenAI shape, generated, and translated back
(`anyllm_translate`). An Anthropic-native client needs no proxy:

```sh
claude --settings '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:8080","ANTHROPIC_API_KEY":"unused","CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY":"true","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}}' \
  --model claude-turbospark-<canonical-model-id>
```

The unknown-model-window override makes Claude Code defer its built-in 200K
unknown-model assumption to the gateway for this local discovery alias.
The `--settings` overlay is intentional: it gives this invocation an explicit
one-session source for the local port when saved Claude Code `env` settings
still name a previous server.

### Request fields read

| Field | Notes |
|---|---|
| `model` | Routing; the name is echoed back in the response |
| `max_tokens` | REQUIRED on this endpoint (a real generation needs a budget) |
| `messages` | `role` user/assistant; `content` string or blocks: `text`, `image` (base64 source served with a vision tower, url source refused), `tool_use`, `tool_result` |
| `system` | System string or system blocks |
| `temperature`, `top_p` | Sampling |
| `stop_sequences` | Stop strings |
| `tools` | Anthropic shape `{"name", "description", "input_schema"}` |
| `tool_choice` | Auto/any/tool |
| `stream` | Default false |
| `top_k`, `thinking`, `metadata`, document blocks, `cache_control` | Accepted, reported on `x-anyllm-degradation` where applicable, not honored |

A request's `thinking` CONFIG has no backend here and is dropped in
translation. A RESPONSE's thinking is different: when the checkpoint produces
reasoning it comes back as a `thinking` content block.

### Response (non-streaming)

Anthropic `MessageResponse`: `id`, `role: "assistant"`, `content` blocks
(`text`, `thinking`, `tool_use` with `id`/`name`/`input`), the client's own
`model` string, `stop_reason`, and `usage` with `input_tokens` /
`output_tokens`.

### Streaming framing

SSE with named events (clients dispatch on the `event:` line):
`message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, `message_stop`, plus `ping` and
`error`. The stream ends at `message_stop` with NO `[DONE]` sentinel (that is
an OpenAI-ism). A thinking block opens on the first reasoning delta and
closes on the first text delta.

```sh
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":120,"messages":[{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":40,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

## POST /v1/messages/count_tokens

The same request shape as `/v1/messages`, except `max_tokens` is NOT required
(a placeholder is injected only when absent). Runs the request through the
same translate + plan path and stops before any generation, so the count is
exactly what a real call on this request would prefill.

Response:

```json
{"input_tokens": 21}
```

```sh
curl -s localhost:8080/v1/messages/count_tokens -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}'
```

## GET /v1/models

One entry per public model identity. Each generative backend contributes its
canonical id followed by `claude-turbospark-<canonical-id>`, and the alias
routes to the same backend. Embedding-only backends have no Claude alias.
Read by OpenAI model pickers and Claude Code's gateway discovery.

```json
{"object": "list", "data": [
  {"id": "gemma4", "object": "model", "created": 1760000000,
   "owned_by": "mference", "context_window": 4096},
  {"id": "claude-turbospark-gemma4", "object": "model", "created": 1760000000,
   "owned_by": "mference", "context_window": 4096}
]}
```

`context_window` is additive (not OpenAI's): the window this model was OPENED
at, which is per-session rather than per-checkpoint.

```sh
curl -s localhost:8080/v1/models
```

## GET /v1/models/:model

Model detail for an exact id (no single-model fallback). Same entry shape as
one `/v1/models` row. Unknown id: 404 with
`error.code = "model_not_found"`. Looking up a discovery alias returns that
alias row.

```sh
curl -s localhost:8080/v1/models/claude-turbospark-gemma4
```

## POST /api/chat

Ollama-compatible chat (`crates/server/src/ollama.rs`). For tooling that
speaks Ollama by default and offers no way to change it.

### Request fields read

| Field | Notes |
|---|---|
| `model` | Routing |
| `messages` | `role` (`system`, `assistant`, `tool`, anything else is `user`) and `content`. Plain text only |
| `stream` | Default TRUE (Ollama's own default, unlike every OpenAI-shaped route here) |
| `options.temperature`, `options.top_p` | Sampling |
| `options.num_predict` | Maps to `max_tokens`; `-1` (context-full) is this server's own absent-`max_tokens` behaviour |
| `options.top_k`, `options.seed`, `options.repeat_penalty` | Mapped onto the internal `top_k`/`seed`/`repetition_penalty` keys |
| `options.num_ctx` | Silently dropped: the context window is fixed when the model is opened |

Reasoning and tool-call output have nowhere to go in Ollama's wire shape and
are dropped rather than emitted as the answer.

### Response and streaming

Non-streaming: one JSON object:

```json
{"model": "m", "created_at": "2026-09-05T12:00:00Z",
 "message": {"role": "assistant", "content": "…"},
 "done": true, "done_reason": "stop", "prompt_eval_count": 21, "eval_count": 5}
```

Streaming: NDJSON (`content-type: application/x-ndjson`), one bare JSON
object per line, no `event:`/`data:` framing, no `[DONE]` sentinel. Each line
carries a `message.content` delta and `done: false`; the LAST line carries
`done: true` plus `done_reason`, `prompt_eval_count`, and `eval_count`
(taken off the decode result, never off a count of deltas). An error
mid-stream ends the body with a final `done: true` object carrying
`done_reason: "error"` and an `error` field.

```sh
curl -s localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}'
```

## POST /api/generate

Ollama's raw-prompt endpoint. Unlike `/v1/completions`, this one DOES go
through the chat template (Ollama itself applies the model's template here,
and a bare prompt on an instruction-tuned model otherwise babbles).

### Request fields read

| Field | Notes |
|---|---|
| `model` | Routing |
| `prompt` | The user turn |
| `system` | Optional system turn, prepended when non-empty |
| `stream` | Default true |
| `options` | Same bag as `/api/chat` |

### Response and streaming

Identical to `/api/chat` except the text is flat on `response` instead of
nested under `message.content`, and the non-streaming/streaming objects use
the same field name.

```sh
curl -s localhost:8080/api/generate -H 'content-type: application/json' \
  -d '{"model":"m","stream":false,"prompt":"why is the sky blue"}'
```

## POST /api/show

Details for one model, matched exactly against the attached ids (no
single-model fallback). Request `{"model": "..."}` (`name` is an accepted
alias). Unknown id: 404.

```json
{"details": {"format": "gturbo", "family": "turbospark", "families": ["turbospark"],
              "parameter_size": "", "quantization_level": ""},
 "model_info": {"general.architecture": "turbospark", "turbospark.context_length": 4096},
 "capabilities": ["completion"]}
```

```sh
curl -s localhost:8080/api/show -H 'content-type: application/json' -d '{"model":"m"}'
```

## GET /api/tags

The Ollama model list. `size` is reported as 0 rather than estimated (this
server downloaded nothing; on-disk bytes are not committed bytes).
`context_window` is additive.

```json
{"models": [{"name": "m", "model": "m", "modified_at": "2026-09-05T12:00:00Z",
  "size": 0, "digest": "", "details": {"format": "gturbo", "family": "turbospark",
  "families": ["turbospark"], "parameter_size": "", "quantization_level": ""},
  "context_window": 4096}]}
```

```sh
curl -s localhost:8080/api/tags
```

## GET /api/version

```json
{"version": "0.1.0"}
```

Reports THIS server's own version (`CARGO_PKG_VERSION`), not an Ollama one:
clients gate features on version ranges and a borrowed number would promise
endpoints that do not exist here.

```sh
curl -s localhost:8080/api/version
```

## GET /health

Liveness probe. Reads only fields resolved at open; it never queues behind
the generation lock, and it is never behind `--api-key`.

```json
{"status": "ok", "model": "gemma4", "models": ["gemma4"], "state": "ready",
 "version": "0.1.0"}
```

`model` is the first attached model (kept as a bare string for existing
clients); `models` is the multi-model answer. A server with nothing attached
reports `model: null` and `state: "empty"`.

```sh
curl -s localhost:8080/health
```

## Cross-cutting semantics

### Stop-reason mapping

The runtime's internal `StopReason` reaches each wire format through
`crate::response::finish_reason` and the per-format translators:

| Runtime stop | OpenAI `finish_reason` | Anthropic `stop_reason` | Responses `status` |
|---|---|---|---|
| `MaxTokens` | `length` | `max_tokens` | `incomplete` |
| `ToolCalls` | `tool_calls` | `tool_use` | `completed` |
| `Eos`, `EndOfTurn`, `StopString` | `stop` | `end_turn` | `completed` |

`ToolCalls` fires only on a dialect whose stop set says "tool" (Gemma's and
Harmony's call markers). ChatML closes a call with ordinary markup and then
ends the turn normally, so a ChatML tool call reaches the client with
`finish_reason: "stop"` and the `tool_calls` array present beside it.

### Tool-call guardrails

On by default (`--guardrails on`; `crates/server/src/guardrails.rs`). Three
repairs, configured as `GuardrailConfig { rescue: true, validate: true,
retries: 1 }`:

- Rescue: recover a tool call the structured decoder could not parse out of
  the raw text (bare JSON, non-standard markup), then validate the result.
- Validate: check a parsed call's arguments against the schema the request
  itself sent (missing required fields, wrong types, invented enum members).
- Retry: re-ask ONCE with a nudge. The retry re-renders the whole prompt
  (failed assistant turn plus a user nudge) through the checkpoint's own
  template; it never appends tokens.

**A request carrying tools is buffered rather than streamed while guardrails
are active**: a verdict needs the whole turn, so time-to-first-token becomes
time-to-last-token for that turn. Requests WITHOUT tools stream byte for byte
as they always did. `--guardrails off` restores the unguarded path exactly.

`tool_choice: "required"` (or a named function) additionally means prose
alone is a retryable failure: the model is re-asked once to call the tool.

### Reasoning

- Process default: `--reasoning off|low|medium|high|xhigh` (default `off`).
- Per-request: `reasoning_effort` at the top level of the OpenAI-shaped
  bodies (chat completions, completions via extra, responses). A misspelled
  value is refused with 400 rather than silently ignored. The accepted set is
  the union (`off`, `low`, `medium`, `high`, `xhigh`); which of those a given
  checkpoint's template accepts is the checkpoint's own error to say.
- The rendered template receives both spelling keys (`reasoning_effort` and
  `reasoning_strength`, `crates/tokenizer/src/reasoning.rs`), so a checkpoint
  that reads either one is served by the same request field.
- Output surfaces per route: `reasoning_content` on the chat completions
  message and `reasoning` chunks in the stream; an Anthropic `thinking`
  content block on `/v1/messages`; a `reasoning` output item on the
  non-streaming `/v1/responses` body (dropped and reported on the streaming
  one). Ollama routes drop it.

### Degradation reporting: `x-anyllm-degradation`

Both generation routes attach `x-anyllm-degradation` when part of the
request was accepted but not honored: unsupported OpenAI fields
(`response_format` past text, `n > 1`, `logprobs`, `top_logprobs`,
`logit_bias`), Anthropic-side drops (`top_k`, `thinking`, `cache_control`,
document blocks), completions-side (`logprobs`, `best_of`, `echo`),
responses-side (`store`, streaming reasoning), and images dropped for a
text-only install. Vendored warnings are comma-joined; a locally appended
note joins with `"; "`. The turn still succeeds.

### Cancellation

A client disconnecting mid-generation shortens that generation (the decode
loop polls a cancel flag). A cancelled stream simply ends: no finish chunk,
no error event, nothing, because the only client that would read them is the
one already gone. The runner lock is held until the generation actually
stops, so a queued waiter's wait is shortened rather than skipped.

### The plan/exec pipeline

Every wire format folds onto one internal `ChatCompletionRequest` before
generation. `handler::plan` (`crates/server/src/handler/plan.rs`) validates
image shapes, renders the checkpoint's chat template (the Jinja tool template
when the request carries tools, the text-only template otherwise), encodes
with `add_bos: false`, resolves the shaping config (rate control is
process-level; there is deliberately no per-request field for it), and
splices vision tokens. `handler::exec`'s `run_full` and `stream_blocking`
then run the generation on a blocking task. A structured decoder sits in
front of the token stream when tools, a reasoning-producing dialect, or a
reasoning level call for one.

## Error responses

Non-streaming errors (OpenAI-shaped routes):

```json
{"error": {"message": "…", "type": "invalid_request_error",
           "param": "model", "code": "model_not_found"}}
```

`param`/`code` appear on model-not-found. Anthropic routes use the Anthropic
error body (`InvalidRequestError` / `ApiError`).

| Status | When |
|---|---|
| 400 | Malformed or refused request: bad sampling values, unknown `reasoning_effort`, `previous_response_id`, `suffix`, `n > 1`, multi-prompt batch, malformed or remote image URL, context overflow, empty prompt |
| 401 | Missing or wrong API key (only when `--api-key` is set) |
| 404 | Unknown model id with two or more attached; `/v1/models/:model` or `/api/show` lookup miss |
| 500 | Backend/producer failures, speculation unavailable |
| 503 | No model attached at all |

Streaming errors, after the body has started, per route: OpenAI chat and
completions emit a `data:` line carrying `{"error": ...}` then `[DONE]`;
Anthropic emits an `event: error` SSE event; Responses emits
`response.failed`; Ollama ends with a final `done: true` object carrying
`done_reason: "error"` and an `error` field.

## See also

- [turbospark-server](/reference/cli-turbospark-server/): every launch flag
  (model, port, bind, guardrails, speculation, prefix reuse, session slots,
  steering, system prompt, api key).
- [Rust engine and frontends](/reference/rust-engine-and-frontends/): the
  Rust types behind these routes (`ChatModel`, `RealChatModel`,
  `ServerState`, `RouterOptions`).
