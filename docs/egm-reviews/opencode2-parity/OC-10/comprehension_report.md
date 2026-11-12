# Stage 6: Comprehension report (OC-10)

Target: docs/designs/opencode2-parity/OC-10-lsp-diagnostics-symbols.md (full file, 204 lines). Reviewer: egm-comprehension, document read alone. Completed.

## What this proposes

The document proposes adding a Language Server Protocol (LSP) client to TurboSpark, written in Swift. It talks over stdio to language servers the user already has installed, and nothing is bundled. After the agent's edit tool writes a file, the app opens or syncs that file with the covering server. It waits a short, bounded time for diagnostics (errors and warnings) and appends a compact summary to the edit tool's follow-up output. It also adds three agent tools: `project_diagnostics`, `document_symbols` and `workspace_symbols`. Finally, it adds a per-project settings area where the user can enable, configure and test servers, with sourcekit-lsp discovered by default for Swift.

## How the existing system works, per this document

The app has no LSP client, no process management for language servers, and no channel that carries diagnostics into the agent loop. Today the model only learns an edit is broken if it chooses to run a build or lint. Code navigation is grep and read only. The doc mentions an existing tool-process pattern (`ProcessExecutor` conventions), a login-shell PATH discovery path, a shutdown coordinator, a tool registry, and a permission engine (`AppToolPermissionEngine`) with a "ladder" of permission classes. It says new pieces should reuse these but never explains how any of them work.

## Findings

No section was unparaphrasable, so no Blockers.

CO-1 Major. Lines 20-23, 53-57, 165-166. Tool count contradicts itself: Overview lists three tools, line 53 says "Two agent tools plus one shared client", later text says "the three agent tools".

CO-2 Major. Lines 49-52, 78-82, 141-143, 175, 94-101. Post-edit wait is stated as settled behavior in Goals, CUJ 1, NFR and file 7, yet is an open question; the budget (2 to 5 s) appears only in the Open Questions, and the late-result "next turn's context note" is never defined.

CO-3 Major. Lines 99-101. The summary contract is vague: "diagnostics line count" is ambiguous (count of diagnostics or lines), no cap on errors shown, no format, warnings unspecified.

CO-4 Major. Lines 87-90, 168-173, 177-178. Permission design is muddled: "same allow class as file reads", "project category so a project can deny them", and "new category in the ladder" are not reconciled; the ladder is never defined; default allow or deny and what the user sees are unstated; the file plan treats an open question as decided.

CO-5 Major. Lines 49-55. Diagnostics store lifetime is undefined: how stored, how stale entries clear, and whether `project_diagnostics` without a file covers only files the agent touched (servers publish only for opened or synced files). A reader would expect whole-project coverage.

CO-6 Major. Lines 71-75, 133. "Project" is undefined (folder, chat, worktree); no crash, restart, or never-initialized behavior; no timeout values.

CO-7 Major. Lines 49-50, 156-158. Hook scope is unclear: which tools count as an edit (write, edit, patch, shell writes) and how a file maps to a covering server (extension, language ID, config field).

CO-8 Major. Lines 61-62, 144-146. Discovery and safety are thin: login-shell PATH path and "tool-process hardening" are asserted, not described. Per-project server commands may come from a cloned repository; the doc does not say whether opening an untrusted project can launch a configured binary without confirmation.

CO-9 Nit. Lines 3-10, 24-26. Depends on "the report", its Section 8, and a "fifth by user value" ranking; "OpenCode", "30+ servers" and "clearest agent-quality gap" have no source line.

CO-10 Nit. Lines 36, 70, 133-136, 193, 198-199. Project-specific names used without explanation (`AppToolPermissionEngine`, `ProcessExecutor`, shutdown coordinator, settings shell rules, tool registry, `make design-check`, no-bundled-runtime rule). Line 36 link path does not resolve from the doc's folder.

CO-11 Nit. Lines 147-150, 194, 197. "21 languages" and "localization gates" unnamed; "nothing else is user-visible" contradicts the diagnostics summary reaching the transcript.

CO-12 Nit. Lines 71-72, 129-130, 177-180. Settings placement unclear: permission toggle in `ProjectPermissionsSectionView.swift` versus server rows in a new view; CUJ 3 "says what to configure" has no described copy or state.

CO-13 Nit. Lines 119-130. Diagram names (`EditToolHook`, `DiagnosticsTool`, `SymbolsTools`) do not match the file plan; no arrow shows where settings rows read state.

CO-14 Nit. Lines 54-57, 103-108. "Bounded" results have no limits; `document_symbols` behavior with no covering server (error or empty) is unstated.

CO-15 Nit. Lines 181-182, 186-189, 193-199. Mutation check unexplained; `swift test` and `make swift-lib` listed without saying whether they are the same or separate gates.
