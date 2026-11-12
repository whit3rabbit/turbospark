# Stage 8: Intermediate zoom out report (OC-10)

Target: docs/designs/opencode2-parity/OC-10-lsp-diagnostics-symbols.md (full file). Reviewer egm-zoom-out, completed. Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md. No Blockers on altitude; recommendation is to shrink, not extend.

S8-1 Major. 43-45, 132-137, 156-161. Layer the Content-Length codec and LSP message subset over `Tools/MCP/McpStdioSession.swift` (persistent child, waiter, idle timer, `McpSessionProcessRegistry` kill-on-quit). Name `LoginShellPath` at line 61. Replaces the unnamed "shutdown coordinator". Ambiguity removal and scope reduction.

S8-2 Major. 125, 174-176, INDEX 83. Make the diagnostics summary a built-in consumer of the existing PostToolUse pipeline (`AppModel+Hooks.swift` `dispatchPostToolUseVerdict`, `AppModel+AgentEfficiency.swift` `applyPostToolUse`) or say why it is insufficient. Do not add IC-2's new ordered-pipeline abstraction; state order in one sentence (formatter first, then diagnostics). Say whether subagent edits get diagnostics.

S8-3 Major. 53-60, 83-86, 156-190. v1 too large. Cut to client, sourcekit-lsp discovery, post-edit summary, and `project_diagnostics` (consider deferring it too, since it duplicates the summary). Defer symbol tools and the multi-language settings UI; a plain settings.json section plus the existing permission toggle suffices. Add a short Phasing paragraph.

S8-4 Major. 76-90. Pick the wait policy: fixed bounded wait (about 2 s), late results dropped or reachable via `project_diagnostics`. Removes late delivery (line 175). Promote the permission lean to a decision.

S8-5 Major. 117-137. Add one short paragraph: full-text didChange on edit, document version ties a push to the edit, "project" means the per-chat root, one restart then mark the server dead, tools return a "server unavailable" status instead of empty. Decline IC-1 adaptive p90 and warm-up and IC-3 circuit breaker and health UI as scope growth.

S8-6 Major (scope trap). DC-1/ST-1 and DC-2/ST-2 Blockers: a few lines each suffice, not full sections. Alternatives: LSP MCP server (rejected: app-owned lifecycle needed for the post-edit path); shell out to build or lint (today's behavior). Risks: stale diagnostics, repo-supplied server commands, memory pressure from local inference, output size caps. Decide whether a repo-level server command may launch without confirmation; suggest commands live in app settings or need explicit enable.

S8-7 Nit. 53 vs 22-23, 165. Tool count; becomes one if symbols are deferred.
S8-8 Nit. 36. Link path `../../../swift/...`.
S8-9 Nit. 24-26. Soften or cite "30+ servers" and "clearest gap".
S8-10 Nit. 66-68. State symbols need no position lifecycle.
S8-11 Nit. 147-150. Reword "nothing else is user-visible".
S8-12 Nit. 152-189. 12-file list reads as a task list; shrinks after phasing; mutation-check detail belongs in verification.

## Triage of prior findings

Adopt (ambiguity removal): tool count (CO-1, ST-3, SY-2, DC-7); link path; attribution and wording (SY-1, SY-3, CO-9); naming real components (ZO-1, ZO-2, DC-3, ST-10); wait policy and ladder decision (ZO-4, CO-2, CO-4, DC-4, DC-6, ST-6); summary contract (CO-3, SY-5, CO-14); define project, edit triggers, file-to-server mapping (CO-6, CO-7); stale handling by document version.

Decline or defer (scope growth): IC-1 adaptive budget and warm-up; IC-3 breaker and health UI; IC-2 new pipeline abstraction; extra tests (DC-13, ST-9) beyond one line; full Alternatives and Risks sections; multi-root, incremental sync, UTF-16 detail (DC-8); success metrics and owners (DC-4 metric, DC-14); MCP settings row reuse unless the settings UI survives phasing.
