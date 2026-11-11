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
* Status: Draft
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
