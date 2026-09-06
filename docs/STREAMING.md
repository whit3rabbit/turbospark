# Streaming

How a decoded token reaches a caller, one layer at a time. The engine is
push-based end to end: the decode loop calls the caller's callback, and
every transport (stdout, SSE, NDJSON, a C function pointer, a Swift async
sequence) is an adapter over that one callback. Read this page before
adding a consumer, changing an event shape, or proposing an engine-side
iterator or async stream.

## The pipeline

```text
  selection::select  (sample one token)
        |
  TokenSink          (crates/runtime/src/token_sink.rs)
  |  stop-token ladder, detokenizer, stop-string matcher,
  |  budget, cancellation polled once per token, tail flush
        |
  RawDecodeProgress  (crates/runtime/src/raw_completion.rs)
  |  Prefill{done,total} | Token{index,id,delta} | Tail(String)
        |
  run_raw_completion[_cancellable|_chunked_cancellable|
  _speculative_cancellable]   (the loop is the caller's choice)
        |
  TurnSplitter       (crates/runtime/src/turn_stream.rs)
  |  builds a StructuredAssistantDecoder when the dialect and the
  |  request call for one; splits into content / reasoning / calls
        |
  TurnEvent          Content(String) | Reasoning(String)
                     | ToolCall(ParsedToolCall) | Prefill{done,total}
        |
  +---------+---------+------------------+
  |         |         |                  |
CLI        FFI     crates/server      (future consumers)
stdout   TsEvent-  spawn_blocking + tokio mpsc
+stderr  Callback  -> SSE / NDJSON wire formats
```

The layering rule: `RawDecodeProgress` carries TOKENS (with ids),
`TurnEvent` carries DISPLAY UNITS (no ids -- decoder output belongs to no
single token). A consumer that wants token ids reads the raw stream; a
consumer rendering a chat turn reads the structured one.

## The primitive is a push callback, on purpose

The loop takes `impl FnMut(RawDecodeProgress)` plus a `CancelFlag<'_> =
&dyn Fn() -> bool`. Both are borrowed and non-`Send`, because the producer
itself is a borrowed `&mut dyn LogitProducer` tied to the caller's stack
frame (and in the server, behind a `Mutex`). An engine-side iterator or
async-stream facade would have to take ownership of the producer and spawn
a thread, which is a policy decision that belongs to the consumer -- the
server paces itself differently from a CLI, and the FFI's thread is the
embedder's. This matches llama.cpp, whose low-level `llama.h` has no
per-token callback either: the loop owner streams.

## The channel adapter, and backpressure

A consumer that hands tokens across threads adapts the callback to a
channel on a thread it owns, exactly as `crates/server` does:
`spawn_blocking` runs the turn, the callback sends each piece into an
`unbounded_channel`, the async side streams from the receiver. The
unbounded channel is deliberate: the decode loop must never block on a
slow consumer, because backpressure in the decode path is a stall of the
one engine every request shares. The slow-client policy lives OUTSIDE the
stream instead, with two detectors: the client's disconnect
(`CancelOnDrop` sets the cancel flag when axum drops the SSE stream) and a
failed `tx.send` on the live paths.

## Cancellation

Cooperative and per token. The `cancel` predicate is polled once per
prefill token (or per chunk) and once per decoded token inside
`TokenSink::commit`, after the stop/max checks, so a cancelled run takes
the SAME exit path any other stop takes: the stop matcher's withheld tail
is flushed, and cancellation is LAST in stop-reason precedence -- a Stop
pressed as the model finishes does not relabel a complete turn as
truncated. `StopReason::Cancelled` returns an otherwise ordinary
`RawDecodeResult`: partial text kept, KV position honest.

Who sets the flag:

- **Server:** per request, `Arc<AtomicBool>`, set by `CancelOnDrop` (client
  disconnect) or an explicit cancellation route; built into a `CancelFlag`
  ONCE on the blocking thread (`CancelFlag` is not `Send`).
- **FFI:** `session.arm()` before the engine lock is taken, so a Stop
  pressed between turns cannot cancel the next one; `ts_session_cancel` is
  wait-free from any thread.
- **CLI:** none. `turbospark-check` uses the non-cancellable variants;
  Ctrl-C terminates the process. Recorded as a follow-up, not a gap in the
  design.

## The two traps every consumer inherits

`TurnSplitter` exists so these are written once, but they apply to anyone
reading the raw stream directly:

1. **NO EARLY RETURN ON EMPTY TEXT** (AGENTS.md Gotcha 44). A special
   token's delta is the empty string, and every dialect frame token
   (`<|channel|>`, `<|message|>`, `<think>`) arrives as `(id, "")`. A
   consumer that skips empty deltas before the decoder never leaves its
   initial state, and the turn reads as one run of content. The emptiness
   check belongs on DECODER OUTPUT -- which is why `TurnEvent::Content`
   and `.Reasoning` are never empty by construction.
2. **A STOP TOKEN BREAKS THE LOOP BEFORE THE DECODER SEES IT** (Gotcha
   49). Harmony ends a tool call with `<|call|>`, which is in the stop
   set, so the call being parsed is emitted only by `TurnSplitter::finish`
   (or `StructuredAssistantDecoder::finish`), never by `feed`. A consumer
   that does not call `finish` silently drops stop-token-terminated calls
   and DeepSeek's withheld tail.

Today the CLI and the FFI deliberately do NOT call `finish`, and only the
server does. **On those two paths it is provably inert rather than merely
low-risk**, because both of `finish`'s jobs need a non-empty allowlist that
neither has: a Harmony call is emitted from it only once the decoder has
ENTERED its tool state, which is gated on `allowed_tools.contains(name)`
(`structured_decoder/harmony.rs`), and the withheld tail is `held_text`,
written by nothing but `consume_deepseek`, whose decoder `TurnSplitter::new`
builds only when `!tools.is_empty()`. So the gap is a missing SYMMETRY, not
dropped output. It becomes load-bearing on the day a tools surface reaches
either binding, which is the day to close it with real-model eyes.

## Wire formats

All four served formats adapt the same `stream_blocking` pump
(`crates/server/src/handler/exec.rs`):

| Route | Framing | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI SSE, `[DONE]` | usage chunk when `stream_options.include_usage`; a request carrying TOOLS is buffered whole while guardrails are on, because a verdict needs the turn |
| `POST /v1/messages` | Anthropic SSE via `StreamingTranslator` | `translator.finish()` emits `message_stop`; no `[DONE]` |
| `POST /v1/responses` | OpenAI responses SSE | same mpsc shape |
| `POST /api/chat`, `/api/generate` | NDJSON, one object per line | `stream` defaults to TRUE; reasoning and tool pieces have no wire slot and are dropped |

Tool calls reach OpenAI clients as `tool_use` deltas with
`stop_reason: "tool_use"`; Harmony's `<|call|>`-terminated call arrives
through the `finish` path above.

## FFI event contract

`ts_generate(session, messages_json, options_json, TsEventCallback cb,
void *userdata, char **result_json)` blocks for the turn and calls `cb`
synchronously on the calling thread:

| Kind | `text` | `a` | `b` |
|---|---|---|---|
| `TS_EVENT_PREFILL` | empty | tokens done | prompt total |
| `TS_EVENT_CONTENT` | visible delta | 0 | 0 |
| `TS_EVENT_REASONING` | thought delta | 0 | 0 |
| `TS_EVENT_TOOL` | call as `{"id","name","arguments"}` JSON | zero-based index | 0 |
| `TS_EVENT_FINISH` | stop reason, spelled as `result_json`'s `stopReason` | new tokens | prompt tokens |

`FINISH` fires exactly once per successful call, last, just before
`ts_generate` returns; a FAILED run emits nothing and keeps the non-zero
return + `ts_last_error` contract. Hosts treat unknown kinds as no-ops.
`result_json`'s `toolCalls` array carries the same rows the `TOOL` events
did. No `GenerateOptions` field offers tools yet, so `TOOL` cannot fire
and `toolCalls` is always empty today; both are wired so growing the
binding to offer tools is an options change, not a stream redesign.

## Swift surface

`TurboSparkSession.generate` wraps the callback in an
`AsyncThrowingStream<GenerationEvent, Error>`: the blocking call runs on
the session's serial queue, the callback copies each event's bytes and
yields into the continuation, and cancelling the consuming `Task` calls
`ts_session_cancel` (see `docs/SWIFT_BINDINGS.md` for the threading
contract). `GenerationEvent` cases: `.prefill`, `.content`, `.reasoning`,
`.toolCall(GenerationToolCall)`, `.stopped(stopReason:newTokens:
promptTokens:)`, `.finished(GenerationResult)`.

## Tests

- `crates/runtime/src/turn_stream.rs` unit tests: the decoder-wanted
  predicate table, the empty-delta rule, pass-through, prefill relay,
  degraded fallback, finish suppression -- mutation-checked.
- `crates/server/tests/harmony_channels.rs`: reasoning and tool-call
  channels end to end over the wire, scripted, including the
  `<|call|>`-terminated call that only `finish` can emit.
- `crates/ffi/tests/c_surface.rs`: events vs result agreement, null
  callback, cancel from a second thread, `FINISH` once-and-last.
- `swift/TurboSpark/Tests/TurboSparkTests/RealModelTests.swift`
  (env-gated): stream vs result agreement through the Swift wrapper,
  `.stopped` matching the result's stop reason, cancel mid-generation.

## Deliberately not here

- **An engine-side iterator, async stream, or backpressured channel** --
  the adapter belongs to the consumer (see above).
- **Token cancellation through the progress callback** -- cancel is a
  separate borrowed predicate so the callback stays a pure observer.
- **A CLI stop button.**
- **Offering tools through the FFI binding.** The events, the result
  field, and the Swift cases are wired; the options surface is not.
- **Request queueing and fairness** (ROADMAP): today one
  `Mutex<RealForwardRunner>` per process serializes turns, and concurrent
  requests queue on tokio blocking threads.
