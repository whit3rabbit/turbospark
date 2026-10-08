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
  execution, a blocking reader per pipe (stdout to EOF, stderr tail),
  call admission, the memory watchdog, deadline kill, output collection,
  call records.
- `CodemodeSandbox.swift` -- the run entry: `// @options:` parsing, the
  per-chat store box, result formatting into prompt output.
- `CodemodeNestedRunner.swift` -- routes script-issued calls through
  `AppToolRegistry.executeDeferredMcpCall`, which re-runs the target
  tool's PreToolUse hooks, session approval, permission-engine decision,
  and PostToolUse hooks.
- `CodemodeToolDefinitions.swift` -- the tool definition, the opt-in
  `CodemodeSettings` flag, and `CodemodeCatalog.promptListing`, the
  system-prompt listing of deferred tools with typed signatures.
- `CodemodeDeclarations.swift` -- `CodemodeSchemaRenderer`, a lossy JSON
  Schema to TypeScript summary used for those signatures.

## Invariants

- **Re-gating.** A script-issued call takes the same path as a
  model-issued one. A tool the permission engine or a hook rejects stays
  rejected inside a script; a call that needs interactive approval fails
  closed inside the script ("make that one directly"). Approving the
  `codemode` call itself does NOT approve its inner calls, so batching
  covers tools that are already allowed or session-approved; anything else
  must be called directly. This is deliberate: the script's tool names are
  dynamic, so there is no complete list to approve up front.
- **No recursion.** The handler only accepts names in the granted
  descriptor set (deny rules already stripped) and refuses `codemode`
  itself.
- **Bounded output.** The prelude hard-caps `text()`/`console` volume
  (default 32,000 characters / 1,000 items) and reports the failure
  through the done bridge BEFORE throwing, so catching the RangeError
  cannot resume output or flip the result to ok. The script's return value
  draws on the same character budget (a `return bigArray` cannot skip the
  cap), and the host bounds every worker line (`CodemodeWire.maximumLineBytes`).
- **Bounded calls.** One script may make at most 200 tool calls, with at
  most 8 in flight (the rest queue for a slot). Each call spawns an MCP
  server process, so an unbounded loop would exhaust processes and, for a
  session-approved mutating tool, repeat its side effect without limit.
- **Nothing runs after the result settles.** `return`, `exit()`, a failure
  and a caught output-cap breach all set a `finished` flag; a script that
  keeps running after catching one of those gets a rejected promise from
  `tools.*` and the call never crosses the bridge. The host also ignores a
  `call` line that arrives after `done`.
- **Nested results are complete or rejected.** A tool result over 2 MiB
  (UTF-8 bytes) is rejected with an explicit error instead of clipped: a
  script that parses or counts a clipped result gets a wrong answer with no
  error. Large results below the bound reach the script whole, which is what
  lets it filter them down before anything reaches the model.
- **Bounded state.** `store()` values are JSON text capped at 64 Ki each /
  256 Ki total / 256 keys per chat snapshot, enforced in the VM so the script
  gets a RangeError at the write. `CodemodeSandbox.StoreBox` holds them in
  memory for at most 64 chats (least recently used goes first) and applies
  writes only after a successful run. A run with no owning chat gets an empty
  store that is never saved.
- **Bounded memory.** The host polls the worker's physical footprint
  (`proc_pid_rusage`, about every 100 ms) and kills it past 1 GiB. JavaScriptCore
  has no in-process heap limit and macOS does not enforce `RLIMIT_AS`; without
  this a runaway allocation ran until the OS killed the worker (SIGKILL).
  Polling from the host also works for a script stuck in a synchronous loop.
- **Confined worker.** Before it creates a JavaScript context the worker
  applies a Seatbelt profile to itself (`CodemodeWorkerSandbox`) and fails
  closed if it cannot: no network, no exec or fork, no file writes, no reads
  under `/Users` or `/Volumes` (its own bundle directory is allowed back), no
  signalling other processes, and no Mach service lookups (which is what
  blocks the clipboard). A script has no I/O of its own, so this is about
  what a JavaScriptCore memory bug could reach. It is a deny-list over
  "allow default", not deny-by-default: Apple's `pure-computation` profile
  makes JavaScriptCore trap in `JSContext()`, whereas this one keeps the JIT
  at full speed. Anything not listed stays allowed, so each rule is pinned
  by the sandbox probe (`--codemode-sandbox-probe`) against the real binary,
  with an `--unconfined` control proving each probe can succeed. The allowed
  read directories are resolved through symlinks BEFORE the profile is applied
  (Seatbelt matches real paths, and once confined the worker cannot `readlink`
  under `/Users`); a binary started through a symlink therefore reads its own
  files by real path, but cannot traverse the link itself.
- **Minimal worker environment.** The worker is this app's own binary; it
  gets only the MCP child baseline (`PATH`, `HOME`, `LANG`, `TMPDIR`), never the
  launching shell's API keys or tokens.
- **A dead worker says why.** If the worker exits without a result, the
  failure carries its exit status or signal and the tail of its stderr.
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
| Output characters | 32,000 | `// @options: {"max_output_chars": N}` up to 200,000; text(), console and the return value share it |
| Output items | 1,000 | text() and console calls combined |
| Store value | 64 Ki chars | RangeError in the script on breach |
| Store total per chat | 256 Ki chars | |
| Store keys per chat | 256 | RangeError in the script at the write; 64 chats kept |
| Call arguments | 1 Mi chars | the call rejects inside the script |
| Nested call result | 2 MiB (UTF-8 bytes) | rejected with an error, never truncated |
| Tool calls per script | 200 | over the cap, `tools.*` rejects inside the script |
| Calls in flight | 8 | further calls wait for a slot |
| Worker memory | 1 GiB | physical footprint, polled by the host; run fails as `sandbox` |
| Wire line | 16 MiB | either direction; sized for the worst-case escaping of a max-size result |
| Script size | 1 MiB | rejected locally, no process spawned |
| Timeout | 120 s | `// @options: {"timeout_ms": N}`, 1 s to 600 s |

## Deviations from pi

- **JavaScriptCore instead of QuickJS-wasm**, in the REPL worker process
  (`REPLWorkerMain` `--codemode-serve`): one process per execution, stopped
  on deadline or cancellation with SIGTERM, then SIGKILL after 2 s
  (`ProcessExecutor.terminateAndReap`). JSC has no public in-process
  interrupt API, so the process is the preempt, exactly pi's rationale for
  `worker.terminate()`. The boundary is still weaker than pi's: QuickJS-wasm
  has its own linear memory, while this is native code in a copy of the app
  binary at user privileges. Defenses are the OS sandbox profile above, the
  minimal environment, the memory watchdog and a fresh context per run.
- **`memoryLimitBytes`** becomes the host-side footprint watchdog above.
- **Declarations are a lossy summary.** pi renders full TypeScript from
  input and output schemas. Here tool results are text, so only the input
  is rendered, as one `tools.<name>(args: {...}): Promise<string>` line per
  tool; unions, `$ref` expansion and nesting are bounded, and a schema that
  does not fit degrades to `Record<string, unknown>` (use `tool_describe`).
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
UserDefaults), toggled in Settings > MCP Servers (bound with `@AppStorage`
so the switch redraws). When enabled and at least one MCP server's tools are
discovered, the `codemode` tool is advertised and the `## Deferred MCP Tools`
listing in the system prompt carries each tool's typed `tools.<name>(args)`
signature. It is one listing, not a second `## Codemode` section naming the
same tools, and it is shown only to an agent that is actually offered the
`codemode` tool (the main prompt and subagents both go through
`CodemodeCatalog.promptListing`). Under the 12,000-character budget a tool
that does not fit typed degrades to a plain name-and-description line, and
tools past the budget are summarized as "N more tools" (find them with
`tool_search`).

## Tests

`Tests/TurboSparkAppTests/CodemodeTests.swift` covers identifier
sanitizing and collisions, @options parsing, result formatting, the store
box, the schema renderer and the merged listing (including the budget
boundary), the nested runner's refusals and its deny / ask paths, and --
through the real packaged executable -- the wire protocol, tool round-trips,
call count and concurrency caps, calls after settle, real cancellation,
output and return-value caps, store and argument caps, error line numbers,
memory kill, worker exit diagnostics, environment, the sandbox probe (control
versus confined, per capability), agent allow-lists, a burst of concurrent runs
(regression guard for a lost final `done` line), stall detection, `exit()`,
deadline kills, cancellation aborts and syntax errors.

## Known costs and follow-ups

- **Agent allow-lists.** A subagent's gate admits calls by tool name, and
  checks `tool_call`'s target the same way, but `codemode` carries tool names
  inside a script and `tool_search` / `tool_describe` list tools, so those
  used to see every MCP tool of the project. The gate now sets
  `AppToolRegistry.callerToolFilter` (a task-local) around `execute`, and
  `codemode` and the bridge executor narrow their catalogs with
  `ToolSearchCatalog.filtered`. The script's `ALL_TOOLS` and the set its calls
  are checked against therefore match what the agent was offered. The filter
  does not reach `Task.detached`, so `codemode` applies it in the registry
  before the worker starts, not in the worker's call handler.
- Each inner call currently spawns a fresh stdio MCP subprocess
  (`McpClientEngine` per-call model). Batching N calls pays N process
  spawns plus handshakes. Persistent MCP connections are the follow-up
  that makes batched calls fast.
- An in-flight nested call whose script ends early (or is killed) is
  reported `cancelled` and its task is cancelled with it. `McpClientEngine`
  honors task cancellation and reaps the server process, so the call stops.
  A call whose request the server already received may still have taken
  effect; cancellation prevents the ones that had not run, it cannot undo
  the ones that had.

## Error attribution

The prelude and the script are evaluated under the source URLs
`codemode-prelude.js` and `codemode.js`. Without them JavaScriptCore's
`Error.stack` carries no frame text. The prelude's frames and the wrapper's
`global code` frame are dropped before the stack reaches the model, and the
host derives `CodemodeError.line` from the first `codemode.js:<line>`
frame (a syntax error reports its own line). The wrapper shares line 1 with
the script, so line numbers match the model's source; columns on line 1 are
shifted.
