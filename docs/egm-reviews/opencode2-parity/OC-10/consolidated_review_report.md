# Consolidated review report: OC-10 LSP diagnostics and symbol tools

Target: docs/designs/opencode2-parity/OC-10-lsp-diagnostics-symbols.md (full file, 204 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO and IC = zoom_out_report, CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 11, Nit 12.

## Blocker

B1. Whole doc (nearest: 76-90). No Alternatives Considered. Fix: add a few lines: LSP MCP server (rejected, app-owned lifecycle needed for post-edit path), shelling out to build or lint (today's behavior), Rust-side crate, reuse of MCP stdio. [DC-1, ST-1, S8-6]

B2. Whole doc (nearest: 139-150). No Risks and Mitigations. Fix: add a short section: stale diagnostics, server crash or hang, repo-supplied server commands launching without confirmation, memory pressure beside local inference, output size caps, file-path confinement. [DC-2, ST-2, S8-6]

## Major

M1. 53 vs 22-23, 54-57, 165. Tool count says "Two" but three are listed. Fix: correct the count (one if symbols are deferred). [CO-1, DC-7, ST-3, SY-2, S8-7]

M2. 78-82, 94-101, 141-143, 175. Post-edit wait is settled in Goals, CUJ 1 and NFR but open in Open Questions; no number; late-result path undefined; cold first edit will miss the budget. Fix: choose a fixed bounded wait (about 2 s), drop or route late results to `project_diagnostics`, soften CUJ 1. [ZO-4, IC-1, CO-2, DC-4, SY-4, S8-4]

M3. 94-101. CUJ 1 summary contract is untestable ("line count", "first errors", "immediately", warnings unspecified). Fix: specify N, format, warnings, and expiry behavior. [CO-3, SY-5, CO-14]

M4. 87-90, 168-178. Permission ladder position left open though INDEX requires it; toggle and settings view relationship and config location unclear; `workspace_symbols` may return outside-project paths. Fix: decide same class as file reads, one toggle, state config location and path filtering. [ZO-4, IC-7, CO-4, DC-6, ST-6, S8-4]

M5. 43-45, 132-137, 156-161. Reinvents stdio session machinery; "ProcessExecutor conventions" and "shutdown coordinator" do not match real components. Fix: layer a Content-Length codec over `McpStdioSession` and `McpSessionProcessRegistry`; name `LoginShellPath`. [ZO-1, DC-3, ST-10, S8-1]

M6. 125, 174-176, INDEX 83. Hook placed in `AppModel+Tools.swift` though a PostToolUse pipeline exists; the OC-11 seam is undefined; subagent edits unaddressed. Fix: make the summary a PostToolUse consumer, state formatter-then-diagnostics order in one sentence. [ZO-2, IC-2, DC-5, S8-2]

M7. 53-60, 83-86, 156-190. v1 over-scoped; multi-language rows and a settings UI for one real server. Fix: add a Phasing paragraph: client, sourcekit-lsp, post-edit summary first; symbol tools and rich settings later. [ZO-3, IC-4, DC-8, S8-3]

M8. 117-137. Correctness gaps: sync message type, version to tie a push to its edit, definition of "project", restart policy, which tools trigger the hook, file-to-server mapping, timeout values. Fix: one short paragraph with fixed values (full-text didChange, document version, per-chat root, one restart then dead, named timeouts). [ZO-5, IC-3, CO-6, CO-7, SY-7, S8-5]

M9. 49-55, 54-57. `project_diagnostics` store lifetime undefined; servers publish only for opened files so the tool is mostly empty and a dead server looks like a clean file. Fix: scope it to touched files, define stale clearing, return "server unavailable" status. [CO-5, DC-7, IC-3, S8-5]

M10. 144-146, 61-62. Server trust and hardening: repo-supplied commands, "tool-process hardening" undefined, server environment unstated. Fix: keep commands in app settings or require explicit enable; name the hardening and environment. [CO-8, DC-2, DC-11, SY-6]

M11. 24-26, 38-39, 149-150. Attribution and accuracy: "clearest gap" is the comparison report's verdict, OpenCode's LSP runtime is mid-rebuild; "nothing else is user-visible" contradicts the transcript summary and toggle; "idle servers cost memory only" unverified. Fix: recalibrate wording and drop or qualify the idle claim. [SY-1, SY-3, SY-6, CO-11, ZO-8]

## Nit

N1. 36. Link `swift/...` resolves wrong; use `../../../swift/...`. [IC-6, DC-9, ST-4, SY-8, CO-10, S8-8]
N2. 66-68. Say symbols need no position lifecycle; reword "full position lifecycle". [ZO-9, SY-11, S8-10]
N3. 119-130 vs 165. Diagram names do not match plan files. [CO-13, ST-8]
N4. 152-189. 12-file list reads as a task list; mutation-check detail and sentence density. [ZO-6, SY-14, S8-12]
N5. 177-180. Settings placement and pane naming (`LspSettingsPaneView`); two places for one concept. [ZO-7, IC-7, CO-12]
N6. 196-200. `swift test` belongs under Tests; drop reason clause; clarify `swift test` versus `make swift-lib`. [ST-7, SY-13, CO-15]
N7. 183-189. Test gaps: permission category, discovery, crash, stale version, token cap (one line each at most). [ST-9, DC-13]
N8. 147-148, 194. Name the ten localization gates; only wrapper strings are localized. [DC-12, CO-11]
N9. 26, 49, 61, 82-85. Filler and jargon ("TurboSpark shape is", "covering server", "login-shell PATH path", "seed rows"). [SY-9, SY-10, SY-12]
N10. 49-50, 69-70. Server selection among matching rows and project root detection (Package.swift or xcodeproj). [IC-5]
N11. No Related line; INDEX says OC-10 lands the post-edit seam before OC-11. [ST-11]
N12. 76-90. Open Questions lack an owner or decision gate; DC-14 and success metrics are scope growth and declined in stage 8. [DC-14, S8]

## Stage 8 triage

Adopt ambiguity-removal fixes. Decline as scope growth: adaptive p90 budget and warm-up (IC-1), circuit breaker and health UI (IC-3), a new ordered pipeline abstraction (IC-2), multi-root and UTF-16 detail (DC-8), success metrics and owners, full Alternatives and Risks sections beyond a few lines. Net recommendation: shrink, not extend.

## Traceability

- zoom_out_report.md: ZO-1 to ZO-9, IC-1 to IC-7. Feeds M1-M11, N1-N5, N10.
- comprehension_report.md: CO-1 to CO-15. Feeds M1-M4, M8-M11, N-series.
- critic_report.md: DC-1 to DC-14, ST-1 to ST-11, SY-1 to SY-14. Feeds B1, B2 and all Majors.
- intermediate_zoom_out_report.md: S8-1 to S8-12 plus triage. Confirms stage 5 findings open.
- Merge notes: B1 merges DC-1 and ST-1; B2 merges DC-2 and ST-2; M5 and M6 merge ZO and S8 reuse findings.
