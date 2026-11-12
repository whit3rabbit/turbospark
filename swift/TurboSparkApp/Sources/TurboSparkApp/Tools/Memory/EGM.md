# Memory

## Overview
The model-facing `memory`, `memory_search`, and `memory_explain` tools. They read approved memory claims and stage model-issued changes as proposals that the user must review in Settings. Nothing the model writes becomes active on its own. Storage and review live in `MemoryStore` and `MemoryLedgerStore` (defined elsewhere); this directory is the tool schema and executor layer on top.

## Interface
- `MemoryToolExecutor.execute(arguments:project:store:ledger:)`: the `memory` tool (actions save/write/remember, read/list/index, forget/delete/remove for project scope; save, read, search, explain for profile scope) [MemoryTool.swift:MemoryToolExecutor].
- `MemoryToolExecutor.searchClaims(arguments:project:)`: `memory_search` (scope profile or project, limit 1 to 10) [MemoryTool.swift:searchClaims].
- `MemoryToolExecutor.explainClaim(arguments:project:)`: `memory_explain` (evidence and supersession chain for one active claim) [MemoryTool.swift:explainClaim].
- `MemoryToolDefinitions`: `toolName`, `definition`, `searchDefinition`, `explainDefinition`, and the computed `all` [MemoryTool.swift:MemoryToolDefinitions].
- Child directories: none.

## Invariants
- Writes only propose. `save` and `forget` call `ledger.propose` with confidence `agent-proposed` and report "not active yet". The model must never be able to activate or delete an approved claim directly [MemoryTool.swift:execute, executeProfile].
- `MemoryToolDefinitions.all` is computed and returns `[]` when `MemoryStore.shared.isModelEnabled` is false, so a disabled feature is never advertised. Every executor entry point also re-checks the flag and throws code 7 [MemoryTool.swift].
- Project memory is keyed by `MemoryStore.projectKey(forProjectRoot:)` as scope `project:<key>`; a chat with no project cannot use project scope (code 3). Default scope for `execute` is project, and `searchClaims` defaults to profile [MemoryTool.swift].
- Names are slugs, never paths: `MemoryStore.slug(from:dated:)` is applied to whatever the model sent, so the stem the store sees is kebab-case with no separators. Preserve this containment [MemoryTool.swift:execute].
- Reads return only claims with status `.active`. `explainClaim` and the profile `explain` refuse non-active claims and, for project claims, claims whose scope does not match the attached project. The supersession walk guards against cycles with a seen-set and stays within the same scope [MemoryTool.swift:explainClaim].
- A missing description falls back to the first non-empty body line, cut to 120 characters, so the index stays one line per memory [MemoryTool.swift:execute].
- Error domain `TurboSparkMemory`: 2 no such memory, 3 no project, 4 missing name, 5 empty content, 6 unknown action, 7 disabled. Some user-input problems (empty query, bad scope, unknown claim) return a message string rather than throwing.

## Side Effects
- Reads `MemoryStore.shared` (enablement, slug, project key) and `MemoryLedgerStore.shared` (snapshot, search, explain).
- Writes: `ledger.propose` adds a pending claim for review; persistence, encryption, and Markdown projection happen inside the ledger store, not here.
- None for network, process spawn, Keychain, UserDefaults, NotificationCenter, or timers within this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
