# OC-10: LSP diagnostics and symbol tools

Findings: Section 8, "LSP diagnostics feeding the agent - Missing in
Swift" and "Symbols / code intelligence tools - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[LSP docs](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/lsp.mdx)
and
[config schema](https://github.com/anomalyco/opencode/blob/dev/packages/schema/src/config/lsp.ts).
One system, two findings.

* Author: whit3rabbit
* Status: Partial (client, process manager, post-edit seam, three tools and the settings rows ship behind a default-off setting, plus a narrow-only per-project toggle and per-row args and vetted env passthrough; open: real-server verification, the no-network sandbox call; see Implementation Record)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Run language servers for the user's installed toolchains, feed
publishDiagnostics back to the agent after edits, and expose symbol
tools: a bounded post-edit diagnostics wait, a `project_diagnostics`
tool, a `document_symbols` tool, and a `workspace_symbols` search tool.
OpenCode documents 30+ servers fed back into the agent loop and calls
this the clearest agent-quality gap; the report ranks it fifth by user
value. The TurboSpark shape is a Swift LSP client over stdio for
servers the user's machine already has, with no bundled runtime.

## The Problem

There is no LSP client, no language-server process management, and no
diagnostics channel into the agent loop. Post-edit verification is left
to the model's own tool usage, so a broken edit surfaces only when the
model chooses to run a build or a lint. There is also no symbol-level
tool: code navigation is grep and read only
([AppFileSymbol.swift](swift/TurboSparkApp/Sources/TurboSparkApp/State/AppFileSymbol.swift)
maps file extensions to SF Symbols for UI chips, not code symbols).
OpenCode ships LSP-backed diagnostics and symbol tools as core agent
infrastructure.

## Goals

* An LSP client over stdio implementing the subset the feature needs:
  initialize/shutdown, textDocument didOpen/didChange/didSave,
  publishDiagnostics, documentSymbol, workspace/symbol.
* Per-project server configuration: language, launch command, optional
  args, discovered defaults (sourcekit-lsp from the active toolchain is
  the discovered default for Swift when Xcode is present).
* After an edit-tool write to a file with a covering server: open and
  sync the file, wait a bounded window for diagnostics of that file,
  and append a compact diagnostics summary to the edit tool's follow-up
  output.
* Two agent tools plus one shared client:
  * `project_diagnostics(file?)`: current known diagnostics, filtered
    by file when given.
  * `document_symbols(file)`: the symbol tree of one file.
  * `workspace_symbols(query)`: bounded symbol search.
* Project settings surface: per-language server rows with enable
  toggle, command, and a test action; discovery marks toolchain-provided
  servers as available.
* Servers are user toolchain binaries discovered on PATH (the existing
  login-shell PATH path) or explicitly configured. The app bundles
  none.

## Non-Goals

* No goto-definition/rename/refactor tools in v1 (they need the full
  position lifecycle; diagnostics and symbols first).
* No bundling or installing of language servers, and no Node-based
  servers assumed present (the no-bundled-runtime rule).
* No LSP for the terminal pane or file viewer; this is an agent-input
  channel plus agent tools only.
* No cross-project server pooling in v1: one server per project per
  language, started lazily on first use and stopped with the project.

## Open Questions

* **Post-edit wait policy**: bounded wait (2 to 5 seconds, lean) versus
  fire-and-forget delivery into the next turn. The bounded wait keeps
  the fix loop tight for small files but can stall big ones; a
  hybrid (wait up to a short budget, then deliver late results in the
  next turn's context note) is the likely answer but needs measurement.
* **Default server set**: sourcekit-lsp only in v1 (toolchain-present
  discovery), with config for everything else; or seed rows for common
  toolchain binaries (clangd via Xcode, rust-analyzer via rustup).
  Lean: sourcekit-lsp discovered, others user-configured rows.
* **Permission ladder position**: diagnostics and symbols are
  read-only, so the lean is the same allow class as file reads, still
  routed through the `AppToolPermissionEngine` and listed as a project
  category so a project can deny them.

## Critical User Journeys

### CUJ 1: broken edit surfaces immediately

* **Given**: a project with sourcekit-lsp discovered, the agent writes
  a Swift file with a type error.
* **When**: the edit completes.
* **Then**: the tool's follow-up output includes the diagnostics line
  count and the first errors with ranges. This lands before the model
  plans its next step.

### CUJ 2: symbol search

* **Given**: the same project.
* **When**: the agent calls workspace_symbols with a type name.
* **Then**: it receives bounded locations (path plus range) without
  grepping.

### CUJ 3: no server for the language

* **Given**: a project folder of Python with no configured server.
* **When**: the agent edits a file.
* **Then**: behavior is exactly today's (no wait, no summary), and the
  project LSP settings surface says what to configure.

## Technical Plan (Architecture)

```text
LanguageServerManager (per project)
  config rows -> server processes (stdio, lazy start)
  LSPClient per server (initialize, didOpen/Change/Save,
  publishDiagnostics, documentSymbol, workspace/symbol)
        |
        +--> EditToolHook: bounded diagnostics wait, summary append
        +--> DiagnosticsTool: read known diagnostics
        +--> SymbolsTools: document tree, workspace search
        v
Project settings rows (enable, command, test)
```

Process supervision follows the existing tool-process patterns
(`ProcessExecutor` conventions). Startup is bounded, requests time out,
and shutdown is clean on project close or app quit via the shutdown
coordinator. The client speaks JSON-RPC with Content-Length framing
over stdio. Message types are a small typed subset, parsed with the
existing Decodable conventions.

## Non-Functional Requirements

* **Performance**: the post-edit wait is bounded and cancellable; a
  slow server never extends a turn beyond its budget. Idle servers
  cost memory only.
* **Safety**: server commands are user configuration; they run with the
  project's tool-process hardening (no shell interpolation, argument
  vectors built directly).
* **Localization**: settings rows, tool descriptions, and diagnostics
  summary copy in 21 languages in the same change.
* **Accessibility**: the settings surface follows the settings shell
  rules; nothing else is user-visible.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/LSP/LSPClient.swift` (Create)
   * **Rationale**: JSON-RPC framing, the message subset, request
     futures, notification parsing.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/LSP/LanguageServerManager.swift` (Create)
   * **Rationale**: per-project registry, lazy start, config rows,
     discovery of sourcekit-lsp, shutdown wiring.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/LSP/LSPConfig.swift` (Create)
   * **Rationale**: the typed per-project configuration records and
     defaults.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/LSP/LSPTools.swift` (Create)
   * **Rationale**: the three agent tools over the manager; registered
     in the tool registry.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Registry/AppToolRegistry+Vocabulary.swift` (Modify)
   * **Rationale**: register the tools with descriptions and their
     permission classification.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Core/AppToolPermissionEngine.swift` (Modify)
   * **Rationale**: the new category in the ladder per the Open
     Question.
7. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Tools.swift` (Modify)
   * **Rationale**: the post-edit hook with the bounded wait; delivery
     of late results.
8. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectPermissionsSectionView.swift` (Modify)
   * **Rationale**: the per-project LSP category toggle and rows.
9. `swift/TurboSparkApp/Sources/TurboSparkApp/Components/LspSettingsSectionView.swift` (Create)
   * **Rationale**: server rows, test action, availability states.
10. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
11. `swift/TurboSparkApp/Tests/TurboSparkAppTests/LSPClientTests.swift` (Create)
   * **Rationale**: framing, handshake, and diagnostics parsing against
     a scripted fake server over in-memory pipes.
12. `swift/TurboSparkApp/Tests/TurboSparkAppTests/LSPToolsTests.swift` (Create)
   * **Rationale**: tool behavior with a fake manager; the bounded-wait
     budget and cancel path; no-server passthrough. Mutation check:
     remove the budget and assert the budget test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suites plus localization gates and
  `make design-check`.
* **Manual**: with a real Xcode toolchain, edit a Swift file with a
  known error. Confirm the summary reaches the transcript. Run the
  symbol tools. Confirm a no-server project behaves unchanged. Run the
  full app test suite (`swift test`), since this touches the tool
  dispatch path.

--------------------------------------------------------------------------------

*Drafted with EGM*

## Implementation Record

Status is **Partial** (only the items under Remaining are open; they need
a real toolchain, a security-policy call or are a stated non-goal).
Everything below shipped behind the default-off
master switch "Use language servers" (Settings > General > Language
servers). Nothing starts, is advertised or runs a process until the user
turns it on AND enables a server row.

### What shipped

* `Tools/LSP/LSPFraming.swift`: Content-Length codec. Frames above 8 MiB
  and header blocks above 4 KiB are protocol violations that kill the
  server (bounded while reading).
* `LSPProtocol.swift`, `LSPClient.swift`: the typed subset (initialize,
  shutdown, didOpen/didChange (full text)/didSave, publishDiagnostics,
  documentSymbol, workspace/symbol), over an `LSPTransport` seam. Request
  timeout 8 s, initialize 15 s, shutdown 2 s. Server-initiated requests
  are answered (empty success or method-not-found) and never performed.
  A publish counts as fresh only if it follows the sync and, when the
  server reports a version, is not older than ours.
* `LSPProcessTransport.swift`: spawn by resolved absolute path, scrubbed
  environment (`McpClientEngine.childEnvironment` allowlist: PATH, HOME,
  LANG, TMPDIR, plus the row's vetted passthrough names, see the follow-up
  below), cwd is the project root, stderr drained
  to a 4 KiB tail, pids in `LSPProcessRegistry` (killed on app quit via
  `stopAllBackgroundWorkForShutdown`). Launch policy refuses a
  workspace-relative command and any binary (or PATH hit) that resolves
  inside the project.
* `LanguageServerManager.swift`: one manager per project root, lazy start,
  shared start task, strike policy (a crash or failed start gets one
  retry, the second strike marks the server unavailable until its row
  changes), idle stop after 10 minutes, `stopAll` on project delete,
  settings off, row removal and quit. `LSPBudget` releases the caller at
  the budget even while a cold start is still running.
* `PostEditSeam.swift`: `PostEditStep` list run from
  `applyPostToolUse` (main loop) and the subagent gate, after hook
  feedback. Order contract: OC-11's formatter registers before
  `LSPDiagnosticsStep`.
* `LSPTools.swift`: `project_diagnostics`, `document_symbols`,
  `workspace_symbols`. Settings: `Components/LspSettingsSectionView.swift`
  (master switch, per-server toggle, Test, add and remove rows, and per-row
  arguments and environment passthrough).

### Open Questions, as decided

* **Post-edit wait policy**: fixed 2 s budget (`LSPDiagnosticsStep`).
  No late-delivery into the next turn: late results stay in the client
  store and `project_diagnostics` returns them. A cold first edit
  usually reports "had not reported", by design. Not measured against a
  real server (none is run in tests); the constant is the thing to tune.
* **Default server set**: sourcekit-lsp only, discovered by file
  existence in fixed Xcode and Command Line Tools paths (nothing is
  executed, no PATH or workspace lookup), and OFF until the user enables
  it. Other servers are user-added rows.
* **Permission ladder position**: category `.fileRead` (the same class as
  `read_file`), workspace-rooted, risk safe with the sensitive-path hard
  gate applied to every path spelling. A project denies them with its
  File read permission. No new category (it would change the permission
  UI and every language for no added control), so the draft's "project
  category" is the existing File read category. Conflicting path
  spellings are refused like the file tools.
* **Config location** (review M4): app-level `UserDefaults` keys
  `lsp.enabled` and `lsp.servers`, never read from a project, so a clone
  cannot add or enable a server. Rows start disabled for the discovered
  server and enabled for a row the user types in themselves.
* **Review adoptions**: tool count is three; one toggle; `workspace_symbols`
  drops locations outside the project; diagnostics summary format is
  `path: N errors, M warnings (server)` then up to 10 lines of
  `severity line:col message [code]` (1-based, errors first, hints and
  info omitted, message 240 chars, `<` and `>` escaped); "no errors or
  warnings" is stated for a clean file and a dead or timed-out server is
  never reported as clean; project root is the tool root
  (`project.rootDirectoryURL`), file to server mapping is the first
  enabled row covering the extension; restart policy as above.

### Deferred follow-up, built (per-project toggle, args and passthrough)

* **Per-project toggle** (`Tools/LSP/ProjectLSPMode.swift`,
  `AppProject.lspMode`, the row in `ProjectPermissionsSectionView`,
  mirroring OC-11's formatter toggle). Modes are `inherit` and `off` only,
  so a project can NARROW the app-wide `lsp.enabled` switch and never
  widen it. Tolerant archive decode: a missing key, an unknown value or a
  wrong type all load as `inherit`. Honored in four places: the
  `LSPDiagnosticsStep` (reads `PostEditContext.lspMode`, adds nothing and
  starts nothing), the three tools (`LSPToolExecutor.execute(projectMode:)`
  refuses with "Language servers are off for this project", the same
  thrown-refusal style as the app-wide "turned off" state, re-reading the
  saved project so a flip mid-turn is seen), the manager
  (`LanguageServerManager.client(for:)` is the single launch door and
  refuses while `projectMode == .off`; `setProjectMode(.off)` stops the
  running servers), and `AppModel.updateProject`, which stops the
  project's registry manager the moment a save narrows it.
* **Per-server arguments and environment passthrough** (built, because it
  can be done without touching the launch policy):
  * The spawn environment is still the `McpClientEngine.childEnvironment`
    allowlist (PATH, HOME, LANG, TMPDIR). A row without passthrough gets
    exactly that (test). A row may add NAMED variables, copied from the
    app's own environment only when present there; no value is typed, no
    value comes from a project, and the discovered sourcekit-lsp row
    never carries any.
  * Names are checked by `LSPEnvironmentPolicy` in the settings field, on
    load of the stored rows, and again at spawn
    (`LSPLaunchPolicy.environment`), so a hand-edited defaults file cannot
    skip it. Refused: names containing TOKEN, KEY, SECRET, PASSWORD,
    PASSWD, CREDENTIAL, AUTH, PRIVATE, PROXY or COOKIE; prefixes AWS_,
    ANTHROPIC_, OPENAI_, GITHUB_, SSH_, GIT_, DYLD_, LD_; code-running
    interpreter and shell variables (NODE_OPTIONS, PYTHONPATH, BASH_ENV
    and similar); the baseline four; anything that is not a plain
    identifier. At most 8 names. The field shows refused names in a
    warning and stores only the accepted ones.
  * `args` is an array edited one argument per line (32 at most, 512
    characters each, no NUL), passed to the process as is, never a shell
    string. The workspace-relative executable refusal is unchanged.
  * Rows stored before the field existed decode with no passthrough.

### Remaining (deferred)

* **No network for the server process.** There is no no-network switch
  for a child process without a sandbox profile, and sourcekit-lsp runs
  SwiftPM manifests under its own sandbox, which cannot be nested. The
  mitigations shipped are the scrubbed environment (no proxy or
  credential variables), user-installed binaries only, and the warning in
  the settings text. Decide whether to wrap launch in a seatbelt profile
  and accept breaking sourcekit-lsp, or to leave it.
* Measuring the 2 s budget and settle window against a real sourcekit-lsp
  (manual verification step below, not run here).
* goto-definition, rename and refactor tools (non-goal, unchanged).

### Verification run

Focused suites `LSPClientTests`, `LSPToolsTests` and
`LSPProcessFixtureTests` (fake in-process server plus a bash fixture; no
real language server). Mutation checks are listed in the final report of
the implementing session. Manual check with a real Xcode toolchain is
still to do.

Follow-up verification: `LSPProjectToggleTests` (24 tests; fake servers only)
plus LSP*, Formatter*, LocalizationParityTests and StoreDurabilityTests, 153
tests, 0 failures. Mutation checks, each killed: `&&` to `||` in
`allowsServers`; the `client(for:)` guard removed; `setProjectMode(.off)` not
stopping; the spawn-time deny re-check removed; the executor guard removed;
the step ignoring `lspMode`. The eager stop in `AppModel.updateProject` is
covered only through `ProjectLSPMode.isNarrowing` and the registry stop (an
AppModel test would write the real project store).
