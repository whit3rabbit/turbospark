# Stage 5: Initial zoom out report (OC-10)

Target: docs/designs/opencode2-parity/OC-10-lsp-diagnostics-symbols.md (full file, 204 lines).
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom out (ZO)

No Blockers. Context is sufficient; root problem is clear. Over-scoped and misses existing seams.

ZO-1 Major. Lines 43-45, 132-137, 156-161. Reinvents stdio JSON-RPC session machinery. `Tools/MCP/McpStdioSession.swift` already has a persistent stdio child session, a once-only request waiter, an idle timer, and `McpSessionProcessRegistry` for synchronous kill on quit. `Tools/MCP/LoginShellPath.swift` is the login-shell PATH resolver (doc mentions it without naming it). `ProcessExecutor.killTreeNow` is the teardown. MCP framing is newline-delimited JSON, so only the Content-Length codec and LSP message subset are new. Line 134 "shutdown coordinator" is under-specified: LSP children should register in the synchronous kill registry.

ZO-2 Major. Lines 125, 174-176 (and INDEX line 83). Post-edit hook is in the wrong layer. A PostToolUse pipeline exists (`AppModel+Hooks.swift` `dispatchPostToolUseVerdict`, `AppModel+AgentEfficiency.swift` `applyPostToolUse`) and can fold feedback into tool output. The diagnostics summary and OC-11 should be built-in consumers of it, or the doc should say why it is insufficient. The subagent path (`SubagentRunner+Gate.swift`) is not mentioned.

ZO-3 Major. Lines 53-60, 83-86, 156-190. Scope too large for v1. Value is post-edit diagnostics; symbol tools are secondary. Multi-language config rows are YAGNI with one real server. Suggested v1: client, sourcekit-lsp, post-edit summary, `project_diagnostics`; symbols second; plain settings.json section instead of a settings UI.

ZO-4 Major. Lines 76-90. Open Questions block the core design. Wait policy is the central decision and is left to measurement while target 7 assumes hybrid late delivery. Pick a bounded wait with late results dropped (or only via `project_diagnostics`). Promote the permission lean to a decision; INDEX requires a stated ladder position.

ZO-5 Major. Lines 117-137. Missing correctness content: didChange vs didSave vs didOpen and versions; correlating a publishDiagnostics push with this edit (async, stale, empty first); per-chat worktree/project root and `WorkspaceAndNetworkGate` rules for server processes; crash and restart cap. Safety line should name `ProcessExecutor` and the workspace restriction.

ZO-6 Nit. Lines 119-130, 152-189. 12-file list with rationale reads as a task list; mutation-check detail in item 12 is verification detail.

ZO-7 Nit. Lines 177-180. Follow pane naming (`LspSettingsPaneView` next to `McpSettingsPaneView.swift`); consider reusing MCP row and test-action patterns.

ZO-8 Nit. Lines 24, 26. "30+ servers" is unverified (fifth-by-value checks out against the comparison, line 499). Link or soften.

ZO-9 Nit. Lines 66-68. State that document and workspace symbols need no position lifecycle, else the goto-definition exclusion reads as a contradiction.

## Inventive catalyst (IC)

IC-1 Major. Lines 78-82, 49-52, 141-143, 94-101. Wait policy is a false compromise (bounded wait vs fire-and-forget); 2 to 5 s is a guess, the NFR does not say how the budget is set, and CUJ 1 "before the model plans its next step" cannot hold for a cold server or big file. Resolution: adaptive per-server budget (rolling p90 of time to first publishDiagnostics, hard ceiling), warm the server at project open, make late delivery the designed path, and end the wait on version-keyed diagnostics or a debounce so stale pre-edit diagnostics are not reported. Liabilities: variable latency, harder tests, warm-up cost, dedupe and superseded-result rules.

IC-2 Major. Lines 117-137, Target Files 2 and 7, lines 73-74. The post-edit seam OC-11 depends on is never defined as an interface, and formatter-then-diagnose ordering is unresolved. Resolution: an ordered post-edit pipeline (file, content version) with one call site; order formatter, didChange, diagnostics wait; keep the manager behind a protocol; isolate stage failures. (Overlaps ZO-2.)

IC-3 Major. Lines 132-137, 73-74. Failure handling is passive: no restart policy, backoff, health state, or agent-visible dead-server status. A dead server returning an empty result looks like a clean file. Resolution: per-server circuit breaker with health state in settings and in tool output; tools return status instead of empty results. Liabilities: more state and strings to localize, backoff tuning.

IC-4 Major. Lines 53-57, 20-24, 156-190. "One system, large" accepted without segmentation; `project_diagnostics` mostly duplicates the post-edit summary and the doc does not show the model would call it. Resolution: phase 1 client, manager, post-edit diagnostics, minimal settings; phase 2 symbol tools and `project_diagnostics`, gated on evidence of use. (Overlaps ZO-3.)

IC-5 Nit. Lines 49-50, 83-86, 69-70. No statement of server selection when several rows match, project root detection (Package.swift or xcodeproj for sourcekit-lsp), or initializationOptions; CUJ 1 may fail without root resolution. Add one sentence per language row.

IC-6 Nit. Line 36. The AppFileSymbol link `swift/TurboSparkApp/...` is relative to the design folder and resolves to a nonexistent path; use `../../../swift/...`.

IC-7 Nit. Lines 87-90, Target File 8 vs 9. Ladder position left as a lean; two places for one concept (ProjectPermissionsSectionView and a new settings view); diagnostics text comes from project files and should be treated as untrusted tool output.
