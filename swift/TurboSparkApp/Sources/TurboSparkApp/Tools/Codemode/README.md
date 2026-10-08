# Codemode

A port of pi's codemode pattern (`@earendil-works/pi-codemode` in
badlogic/pi-mono) onto this app's JavaScriptCore worker substrate: one
`codemode` tool whose `{code}` parameter runs a model-written JavaScript
script that can batch the deferred MCP tools through
`await tools.<name>({...})`.

## Why

A task that needs N MCP calls costs N model turns when the tools are
exposed individually: every result round-trips through the context just to
pick the next argument. Codemode collapses the loop into one tool call --
the script reads each result, chains, filters, and returns only the useful
summary. The model's context keeps its own printed output and return
value, never the raw tool results.

## Layout

- `CodemodeTypes.swift` -- value types (`CodemodeResult`, limits, call
  records) and `CodemodeIdentifier` (advertised name to JS identifier).
- `CodemodeWire.swift` -- the newline-JSON worker protocol, mirroring pi's
  `protocol.ts`. Arguments and results cross as JSON strings only.
- `CodemodePrelude.swift` -- the VM surface installed before the script:
  `tools`, `ALL_TOOLS`, `console`, `text`, `exit`, `store`, `load`.
- `CodemodeWorkerContext.swift` -- worker-side evaluation: fresh JSContext
  per execution, bridges, settlement poll loop, stall detection.
- `CodemodeWorkerSupervisor.swift` -- host side: one process per
  execution, deadline kill, output collection, call records.
- `CodemodeSandbox.swift` -- the run entry: `// @options:` parsing, the
  per-chat store box, result formatting into prompt output.
- `CodemodeNestedRunner.swift` -- routes script-issued calls through
  `AppToolRegistry.executeDeferredMcpCall`, which re-runs the target
  tool's PreToolUse hooks, session approval, permission-engine decision,
  and PostToolUse hooks.
- `CodemodeToolDefinitions.swift` -- the tool definition, the opt-in
  `CodemodeSettings` flag, and the system prompt declarations section.

## Invariants

- **Re-gating.** A script-issued call takes the same path as a
  model-issued one. A tool the permission engine or a hook rejects stays
  rejected inside a script; a call that needs interactive approval fails
  closed inside the script ("make that one directly").
- **No recursion.** The handler only accepts names in the granted
  descriptor set (deny rules already stripped) and refuses `codemode`
  itself.
- **Bounded output.** The prelude hard-caps `text()`/`console` volume
  (default 32,000 characters / 1,000 items) and reports the failure
  through the done bridge BEFORE throwing, so catching the RangeError
  cannot resume output or flip the result to ok. Each nested tool result
  is truncated to 64 Ki characters before it enters the VM.
- **Bounded state.** `store()` values are JSON text capped at 64 Ki each /
  256 Ki total per chat snapshot, held in memory by `CodemodeSandbox.StoreBox`
  and applied only after a successful run.
- **JSON strings across the boundary.** The worker never builds structured
  values from host data or vice versa; the prelude captures
  `JSON.stringify`/`parse` at install so a script cannot tamper with what
  the host receives.
- **Telemetry is not context.** `calls` (name, status, duration) land in
  archival output only. Prompt projection carries the script's own output
  and return value.

## Limits table

| Bound | Default | Notes |
| --- | --- | --- |
| Output characters | 32,000 | `// @options: {"max_output_chars": N}` up to 200,000 |
| Output items | 1,000 | text() and console calls combined |
| Store value | 64 Ki chars | RangeError in the script on breach |
| Store total per chat | 256 Ki chars | plus a 256-key box cap |
| Nested call payload | 64 Ki chars | head + tail kept, marker between |
| Script size | 1 MiB | rejected locally, no process spawned |
| Timeout | 120 s | `// @options: {"timeout_ms": N}` up to 600 s |

## Deviations from pi

- **JavaScriptCore instead of QuickJS-wasm**, in the REPL worker process
  (`REPLWorkerMain` `--codemode-serve`): one process per execution, killed
  with SIGTERM on deadline or cancellation. JSC has no public in-process
  interrupt API, so the process is the preempt, exactly pi's rationale for
  `worker.terminate()`.
- **No lockdown walk.** pi freezes everything reachable from globalThis.
  Here the context is fresh per execution and discarded with its process;
  the runtime bridges are deleted from globalThis after the prelude
  captures them. Revisit if codemode ever shares a context across runs.
- **No `image()`** -- this app's tool results are text.
- **No host globals, no Lark grammar.** The `// @options:` line is parsed
  leniently; there is no grammar-constrained decoding in this app.
- **MCP tools only.** pi's example exposes the same surface to scripts and
  the model; here scripts reach exactly the granted deferred MCP catalog.
- **Timers.** None, like pi -- and the same consequence: a script awaiting
  a promise nothing can settle fails fast ("can never settle") instead of
  burning the deadline.

## Settings

Opt-in, default OFF: `CodemodeSettings` (`codemode.enabled` in
UserDefaults), toggled in Settings > MCP Servers. When enabled and at
least one MCP server's tools are discovered, the `codemode` tool is
advertised and the `## Codemode` declarations section is appended to the
system prompt beside the deferred MCP listing.

## Tests

`Tests/TurboSparkAppTests/CodemodeTests.swift` covers identifier
sanitizing, @options parsing, payload capping, result formatting, the
store box, and -- through the real packaged executable -- the wire
protocol, tool round-trips, parallel calls, guard-proxy errors, output
caps (including the caught-breach case), store writes, stall detection,
`exit()`, un-awaited-call cancellation, deadline kills, cancellation
aborts, and syntax errors.

## Known costs and follow-ups

- Each inner call currently spawns a fresh stdio MCP subprocess
  (`McpClientEngine` per-call model). Batching N calls pays N process
  spawns plus handshakes. Persistent MCP connections are the follow-up
  that makes batched calls fast.
- An in-flight nested call whose script ends early is reported
  `cancelled`. The underlying MCP invocation may still complete host-side,
  with its gates already applied. That matches any tool result the model
  never reads.
