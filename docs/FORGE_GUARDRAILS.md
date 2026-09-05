# Tool-call guardrails: what they are, how they work, and what they are not

`turbospark-server` repairs tool calls a small local model gets slightly
wrong, in process, using the same local model. This page is the whole
mechanism: what it fixes, how a verdict is reached, where every line runs,
what it costs, and how to verify each of those claims yourself rather than
take this page's word for it.

Shipped and on by default since 2026-08-22 (`b3b3b30`). Turn it off with
`--guardrails off`.

## 0. What this is NOT, because every one of these is a reasonable guess

`forge-guardrails` publishes a proxy binary, a workflow runner, and a
middleware library. **This engine uses the middleware only.** The
distinctions below are not pedantry: each one is a thing the design could
have been and deliberately is not.

| Guess | Reality |
| --- | --- |
| A separate `forge-guardrails-proxy` process in front of the server | No. The pure library is COMPILED INTO `turbospark-server`. There is no second process, no port, no socket. |
| The server shells out to, or HTTP-calls, forge | No. Four ordinary Rust function calls: `&str` in, `String`/`Vec` out. |
| The retry asks a cloud model to fix the call | No. The retry re-generates through the SAME in-process local model. |
| It needs an API key, or a `--backend-url`, or network access | No. The binary has no such flag, and the guardrail subtree links no network crate at all. |
| It needs Claude Code | No. Claude Code is one client among many. Plain `curl` exercises every path. |
| `turbospark-server` drives `turbospark-check` | No. The two binaries have NO dependency in either direction. They share lower-level crates (`runtime`, `tokenizer`) and nothing else. |

The last row surprises people because both binaries take `--model` and both
generate text. They are siblings, not layers.

**Verify the network claim in one command**, with the server running:

```sh
lsof -nP -p "$(pgrep -f 'turbospark-server --model' | head -1)" | grep -E 'TCP|UDP'
```

The only line is the loopback listener. There are no outbound sockets,
because there is nothing in the linked code that could open one.

## 0b. Where it can be turned off, and the two enforcement points

`turbospark-server --guardrails on|off` is the flag. It is PROCESS-level and
deliberately not per request: a per-request field would let any client opt its
own traffic out of the repair this deployment chose.

**THE IN-PROCESS SERVER TAKES THE SAME OPTION SINCE 2026-09-05, AND UNTIL IT
DID A HOST HAD NO WAY TO TURN THIS OFF.** `ts_server_start`'s `options_json`
recognises `guardrails: "on" | "off"`, absent meaning the engine default (on);
anything else is an error rather than a silent fallback, because a host
sending `"disabled"` and getting guardrails anyway has no way to tell.
`ServerOptions.guardrails` is the Swift form. It applies to every model
attached to that server, including ones attached later.

**A NATIVE HOST HAS TWO ENFORCEMENT POINTS AND THEY ARE EASY TO CONFLATE.**
`swift/TurboSparkApp` runs its own Swift `ForgeGuardrailsEngine` over replies
its agent loop reads directly; every HTTP client of its in-process server
bypasses that entirely and gets whatever the server was started with. A user
who set the app to "Always Off" and then pointed a client at its server got
guardrails anyway, with nothing anywhere saying so. The app passes its setting
to `ServerOptions` now and the Server pane reports what the running server
started with -- `unknown` for a server started before the value was tracked,
rather than a guessed "on".

## 0c. It does NOT need the model to frame tool calls, and that is the point

`SessionInfo.toolCalling.native` says whether a checkpoint's OWN markup
carries calls this engine parses. **`false` is the case a rescue helps MOST,
not a case to turn guardrails off for**, and a host that hides the control on
it removes the repair from exactly the checkpoints that need it. Section 1's
first failure -- a call emitted in a syntax the model's own template never
taught it -- is what a non-native dialect does by default.

Three of this engine's seven dialects answer `false`. Two of them (Mistral,
Llama 3) define no tool markup at all. The third is worth knowing about:
Muse Glimmer DOES frame calls, as an `<atem:function_calls>` block, and this
engine has no parser for it -- so `StructuredAssistantDecoder` routes them to
the REASONING stream and no call is ever handed over. An app keying on the
family name rather than on this field listed it as tool-capable.

The honest condition for the control doing nothing is different and is
checkable: `inspect`'s first branch accepts unconditionally when the request
carried no tools, so a turn that sends none is where the toggle provably
changes nothing.

## 1. The two failures this fixes

Both are specific to running a SMALL model locally. A frontier model rarely
makes either mistake; a 4-bit local one makes both, and neither is visible
as an error.

**A call in the wrong dialect.** The model emits a call in a syntax its own
chat template did not teach it: bare JSON, Qwen's `<function=name>` XML,
Mistral's `[TOOL_CALLS]`, or a rehearsal form. `StructuredAssistantDecoder`
correctly declines to parse it, because it is not the markup this
checkpoint's dialect defines. The markup then reaches the client as PROSE,
and the client sees an assistant that rambled JSON at it instead of calling
a tool.

**A call with wrong arguments.** The call parses fine and violates the
schema the caller sent: a missing required field, a string where a number
belongs, an invented enum member. This one is worse than the first, because
it arrives looking valid. The failure surfaces one layer out, in the
caller's own tool executor, with no indication the model was at fault.

## 2. The verdict: how one generation is judged

`inspect` in `crates/server/src/guardrails.rs` is a pure function (no I/O,
no model, no clock) returning one of three verdicts.

```
                    request offered no tools ----------------> Accept
                                 |
                          calls parsed?
                    no /                \ yes
                      /                  \
            rescue from raw text          |
                 |         \              |
          found  |          \ nothing     |
                 v           v            v
             +--------------------------------+
             |   validate against the schema  |
             +--------------------------------+
              ok  |            | errors    | nothing to validate
                  v            v           v
        Accept / Rescued     Retry     Accept (or Retry if
                                        tool_choice required a call)
```

- **`Accept`** -- send the turn as generated.
- **`Rescued(calls)`** -- send the recovered calls, and DROP the raw text.
  The text is the markup the call was recovered from; forwarding it as
  content would leak a JSON blob to the client as the assistant's prose,
  which is the failure being fixed. OpenAI's own shape puts `content: null`
  beside `tool_calls`.
- **`Retry(nudge)`** -- re-generate once with a corrective message.

**RESCUE COMPOSES WITH VALIDATION AND DOES NOT SHORT-CIRCUIT IT.** This is
the single most important line in the design, and the first implementation
had it backwards. Returning a rescued call unchecked is exactly inverted: a
call recovered from markup the decoder could not parse is the one MOST
likely to have arguments the model also got wrong, so it needs the schema
check more than a cleanly parsed call does, not less. The integration test
`an_invalid_call_is_retried_and_the_second_answer_wins` caught it on its
first run by observing a call with empty arguments rescued straight onto
the wire. A rescued call that fails validation is re-asked, and the rescue
is discarded with it.

**Prose with no call is ACCEPTED under `tool_choice: auto`.** A model that
read the question and decided to answer directly is having a good turn.
Nudging it would spend a whole extra generation making a good answer worse.
Under `required` or a named function the caller has stated that prose is
not an acceptable answer, and that is the one case where a re-ask is
clearly right.

## 3. The retry, and why it re-renders the prompt

`run_guarded` is the loop. On `Retry` it appends two messages to a CLONE of
the original request -- the failed turn as an `assistant` message, the nudge
as a `user` message -- and calls `plan` again to re-render the whole prompt
through the checkpoint's own chat template.

**Appending nudge tokens to the existing `prompt_ids` would be cheaper and
is wrong on every dialect.** A nudge has to arrive as a properly framed
turn; raw appended tokens land inside whatever channel the model was last
writing in. That is why `run_guarded` takes the whole
`ChatCompletionRequest` rather than the already-planned prompt.

**The budget is ONE.** The real backend serializes requests behind a mutex
(one `RealForwardRunner` per process, `crates/server/CLAUDE.md` Gotcha 1),
so a second generation doubles the worst-case mutex hold. A ladder of
escalating nudges is available in `forge-guardrails` and deliberately not
taken. When the budget is spent the LAST generation is returned as it
stands: a turn the model could not get right is still a completed
generation, and failing the request would be a worse answer than an
imperfect one.

## 4. Where every line runs

The whole path is in-process. Nothing here crosses a socket.

```
POST /v1/chat/completions  or  POST /v1/messages
        |
        v
  handler::chat_completions / messages::messages
        |
        v
  guardrails::run_guarded(model, &request, effort)      <-- the loop
        |
        +--> handler::plan(...)                          render + encode
        |
        +--> handler::run_full(...)
        |         |
        |         v
        |    ChatModel::run_completion
        |         |
        |         v
        |    runtime::run_raw_completion
        |         |
        |         v
        |    RealForwardRunner  (Metal, the LOCAL model, behind a Mutex)
        |
        +--> guardrails::inspect(...)                    <-- the verdict
                  |
                  +--> forge_guardrails::rescue_tool_call
                  +--> forge_guardrails::validate_tool_arguments
                  +--> forge_guardrails::retry_nudge
                  +--> forge_guardrails::unknown_tool_nudge
```

The four `forge_guardrails` functions are the entire external surface:

```rust
rescue_tool_call(text: &str, available_tools: &[&str]) -> Vec<ToolCall>
validate_tool_arguments(call: &ToolCall, spec: &ToolSpec) -> Vec<ArgValidationError>
retry_nudge(raw_response: &str) -> String
unknown_tool_nudge(called_tool: &str, available_tools: &[&str]) -> String
```

Every one takes borrowed data and returns owned data. None can perform I/O.

### Files

| Path | Holds |
| --- | --- |
| `crates/server/src/guardrails.rs` | `GuardrailConfig`, `Verdict`, `inspect`, `run_guarded`, the type conversions |
| `crates/server/src/guardrails/tests.rs` | 15 unit tests over the verdict, no model |
| `crates/server/tests/guardrails.rs` | 7 end-to-end tests over both endpoints, no model |
| `crates/server/src/handler/mod.rs` | OpenAI call sites, `buffered_stream_response` |
| `crates/server/src/messages.rs` | Anthropic call sites, its own `buffered_stream_response` |
| `crates/server/src/model.rs` | `ChatModel::guardrails()`, the process-level config hook |
| `crates/server/src/args.rs` | `--guardrails on\|off` |

## 5. The one behaviour change: tool requests are buffered

**A request carrying `tools` is generated to completion, inspected, and only
then framed as SSE. A request without tools streams live, byte for byte as
it always did.**

This is inherent, not an implementation shortcut. A verdict needs the whole
turn: a call worth rescuing is one the decoder did NOT parse, so it is
indistinguishable from prose until the turn ends, and a retry re-generates
from scratch. Emit deltas live and both repairs are already on the wire.

Measured on the real Gemma 4 install:

| request | guardrails on | guardrails off |
| --- | ---: | ---: |
| tools, streamed (content deltas) | 1 | 2 |
| no tools, streamed (content deltas) | 14 | 14 |

The cost is that time-to-first-token becomes time-to-last-token for a tool
turn, roughly 5-10 s at this engine's 20-45 tok/s. Taken knowingly: the
alternative does nothing for the client that motivated the feature (Claude
Code sends `stream: true` WITH tools), and a `forge-guardrails-proxy` in
front would buffer identically. The condition is keyed on the REQUEST
carrying tools, which is the same shape as the two conditions in
`crates/server/CLAUDE.md` Gotchas 7 and 12.

**The chunk ORDER out of the buffered path is a correctness condition, not
a style choice.** Anthropic's `StreamingTranslator` is a state machine over
the OpenAI chunk sequence: role, then reasoning, then content, then tool
calls, then finish. Feed it content before reasoning and the Anthropic
events come out malformed rather than erroring.

## 6. The dependency, and why it needed upstream work first

Taken at its default features `forge-guardrails` resolves **384 packages**,
including `sentry`, `clap`, `nix`, `reqwest` and four `anyllm_*` crates at
0.9.9 -- which would put a SECOND `anyllm_translate` beside the 0.16 this
crate already uses -- and it requires Rust 1.95 against this workspace's
1.82.

None of that is reachable from anything this engine calls. The backend here
is a Metal model in the same process, so every client, proxy route and
managed-server process manager in that crate is dead weight.

So the crate gained a `guardrails` feature first (released as 0.1.3) that
gates the transport half out. What made it expressible was moving three
types: `ToolCall`, `TextResponse` and `LLMResponse` lived in `clients::base`
beside `LLMClient` and its `futures_core` machinery, so gating `clients`
gated the vocabulary every guardrail speaks along with it. They now live in
`core::response`, re-exported from the old path so nothing broke.

Measured 2026-08-22 at `forge-guardrails` 0.1.3. The first two move with
that crate's own lockfile (they shifted by one when its dependencies were
refreshed the same day), so re-measure with `cargo tree` before quoting
them; the ORDER of magnitude is the durable part.

| | packages |
| --- | ---: |
| `forge-guardrails` default features | 384 |
| `forge-guardrails` `--no-default-features --features guardrails` | 30 |
| this workspace before | 242 |
| this workspace after | **248** |

The six added are `forge-guardrails` itself plus `anyhow`, `indexmap`,
`regex-lite`, and indexmap's `hashbrown` and `equivalent`. The full
guardrail subtree is `anyhow, indexmap, log, regex-lite, serde, serde_json,
thiserror, tracing` and contains **no network crate**.

```toml
forge-guardrails = { version = "0.1.3", default-features = false, features = ["guardrails"] }
```

`crates/server` is consequently the ONE crate in this workspace that
overrides `rust-version` (1.87 rather than the workspace's 1.82), because
that is what the dependency declares. Overriding on this crate rather than
raising the workspace keeps the claim true for the other fifteen, none of
which has that floor.

## 7. Configuration

```sh
# On, which is the default.
turbospark-server --model gemma4

# Off: the exact path this server took before guardrails existed.
turbospark-server --model gemma4 --guardrails off
```

The startup line says which is in force:

```
  guardrails: on (tool calls rescued and argument-checked; a request with tools is buffered, not streamed)
```

**Process-level, with no per-request field, and that is deliberate.** It
follows the same reasoning as the rate cap and speculation (Gotchas 10 and
17) plus one of its own: a per-request field would let any client opt its
own traffic out of the repair the deployment chose.

## 8. Verifying it yourself

Nothing below needs Claude Code, a network, or an API key.

```sh
# 1. The test suite. 67 passed = tool calling and its guardrails work.
cargo test -p turbospark-server

# 2. A real tool call against a local install, over plain curl.
turbospark-server --model gemma4 --port 8125 &
curl -s localhost:8125/v1/chat/completions -H 'content-type: application/json' -d '{
 "model":"gemma4","max_tokens":300,"temperature":0.2,
 "tools":[{"type":"function","function":{"name":"get_weather",
   "description":"Get the current weather in a given city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},
   "required":["city"]}}}],
 "messages":[{"role":"user","content":"What is the weather in Oslo right now? Use the tool."}]}'
```

Measured result on `~/models/gemma4.gturbo`, both halves of the cycle:

| step | result |
| --- | --- |
| model emits a call | `tool_calls: [{name: "get_weather", arguments: "{\"city\":\"Oslo\"}"}]`, `finish_reason: tool_calls` |
| result fed back as a `tool` turn | "The current weather in Oslo is -3 C with light snow and winds of 12 kph.", `finish_reason: stop` |

The Anthropic endpoint carries the same call as a `tool_use` block with
`stop_reason: tool_use`. The worked `curl` for it is in
`crates/server/CLAUDE.md` Gotcha 8.

### Testing a retry needs a backend that answers twice

`ScriptedChatModel` cannot do it: it rebuilds its producer from the same
steps on every call, so it replays one answer forever and can never show
that a re-generation happened. `tests/guardrails.rs` carries `TwoTurnModel`,
whose producer overrides `produce_prefill` to a no-op. That decouples the
script from prompt LENGTH, which matters because a retry's prompt is longer
than the first (it carries the failed turn plus the nudge), so a
`ScriptedLogitProducer` sized for the first generation runs out mid-prefill
on the second.

## 9. What is deliberately not taken from `forge-guardrails`

- **The ONNX tool-call classifier** (`--classify`). Needs the `classifier`
  feature, a Hugging Face artifact download, and the `ort` runtime. An
  opt-in flag at most, never default-on.
- **Tool-output compression** and **secret redaction**. Both are proxy-side
  concerns aimed at traffic this server originates rather than forwards.
- **Step and prerequisite enforcement** (`required_steps`, `terminal_tool`).
  These need forge to own the agent loop. Here the CLIENT owns the loop:
  this server answers one turn at a time and holds no workflow state.
- **`WorkflowRunner` and `SlotWorker`.** Same reason, one level up.

The proxy remains a valid deployment if you want the features above: run
`forge-guardrails-proxy --backend-url http://127.0.0.1:8080` in front of a
`--guardrails off` server. Nothing in this engine prevents it, and the two
should rescue the same calls.

## 10. Not covered by guardrails

These are properties of the tool path itself rather than gaps in the
repair layer. Full list in `DEVIATIONS.md`.

- **A streamed call arrives as ONE chunk** carrying id, name and the whole
  argument string, not as the argument fragments a remote OpenAI backend
  emits. The decoder yields a call only once its closing marker arrives, so
  there are no fragments to stream.
- **Tool-call ids are a per-response counter** (`toolu_0`, `toolu_1`),
  unique within an assistant turn but not across a conversation. Both
  templates match a `tool` turn against the calls of the message
  immediately before it, so that is enough. A rescued call is spelled
  `toolu_rescued_N`.
- **Nothing constrains DECODING.** There is no grammar forcing a call
  token. `tool_choice` shapes the prompt, the decoder's allowlist and the
  retry decision, but the model is free to emit whatever it emits.
- **`turbospark-check` has no tool calling at all**, by construction rather
  than omission: it builds its decoder with an empty allowlist and has no
  way to run a tool. See `crates/cli/CLAUDE.md` Gotcha 4.

## See also

- `crates/server/CLAUDE.md` Gotcha 18 -- the implementation gotchas, including
  the traps above stated as rules.
- `crates/server/CLAUDE.md` Gotchas 7, 8, 12, 14 -- the underlying tool-call
  path this layer sits on.
- `DEVIATIONS.md` -- scope, and what the tool path drops.
- `git show b3b3b30` -- the full rationale for every decision on this page.
