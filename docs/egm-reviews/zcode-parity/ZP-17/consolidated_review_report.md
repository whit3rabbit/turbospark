# Consolidated review report: ZP-17 error boundaries

Target: docs/designs/zcode-parity/ZP-17-error-boundaries.md (full file, 189 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 10, Nit 9.

## Blocker

B1. Lines 158-164, 168-172 (also 39-43, 100-104). Target File 4 modifies RolloutStore.swift, OfficeRenderModel.swift and PluginPageWebView.swift, which do not exist, and INDEX sequences ZP-17 before ZP-03/08/06. Contract tests cannot be written. Fix: drop Target File 4 and its tests; each surface draft owns its failure card; keep a cross-reference only. [ZO-1, IC-4, CO-6, DC-3, ST-7, S8-1]
B2. Lines 90-141 (missing section). No Risks & Mitigations section, required by the standard. Fix: add a short section: SIGKILL, Force Quit, logout, power-loss and reboot false positives; second instance; marker write failure; traps not contained. [ST-1, DC-1, CO-9]

## Major

M1. Lines 47-48, 69-72. "Exactly as on quit" contradicts "where the flush allowed it"; SIGKILL runs no flush. Fix: drop Goal 4 or restate as no regression versus AppStreamRecovery cold-start behavior. [ZO-2, IC-1, CO-3, DC-4, SY-4, S8-2]
M2. Lines 34-38, 148-150, 165-167. Restore action and last-state snapshot are largely redundant (selected chat already persisted) and their store, writer and vault ordering are undefined. Fix: marker plus one-time notice pointing to the existing interrupted row; drop snapshot and restore test. [ZO-3, CO-4, DC-5, ST-6, S8-2]
M3. Lines 56-61, 94-98, 135. Marker semantics undefined: false positives, pid liveness, second instance, location, vault unlock timing. Fix: write after unlock, clear on all orderly exits, drop pid, store under the state root, notice once per marker. [ZO-4, IC-2, CO-2, CO-8, DC-1, S8-3]
M4. Lines 44-46, 106-109. Lint has no tool, scope, gate or file, and "view-body call paths" is not mechanical. Fix: one rule sentence in AGENTS.md or design-check, or drop the goal. [ZO-5, IC-7, CO-7, DC-6, ST-9, SY-14, S8-4]
M5. Lines 4, 22-30, 39-43, 78-79. No evidence of a real crash class; "process does not crash" is unsupportable since typed Results do not stop traps, overflow, deep nesting or OOM; no retry/reload so the parity claim overstates. Fix: cite an offender or crash, soften CUJ 2 to typed errors, reword parity. [ZO-6, IC-3, CO-1, CO-5, DC-3, SY-5, S8-5]
M6. Lines 116-131. Rejections attributed to "the author" with no link or concrete trade-off; Alternative A is a strawman; no alternatives on OS crash reports or restoration. Fix: concrete costs plus one line on OS crash reports. [DC-2, ST-2, ST-3, CO-10, SY-7] (Stage 8 advises against a large alternatives expansion.)
M7. Lines 56-61. Open Question already carries its answer. Fix: move the decision to the plan; keep only the privacy confirmation with an owner. [ST-4, SY-6, CO-8, DC-5]
M8. Lines 32-48, 63-88. Goals name three surfaces but CUJs cover two; restore sequencing around vault unlock is not shown. Fix: link surface CUJs to ZP-03/08/06 and add one sequence line. [ST-5, ST-6, CO-2]
M9. Lines 24-31. "Uncaught throw in a view body" is wrong for Swift; vault unlock on relaunch cited as a symptom without evidence; existing code claims uncited. Fix: name force unwraps, preconditions and bounds traps; cite StreamRecoveryEventNoticeView path. [SY-2, SY-3, CO-2, DC-5]
M10. Lines 16-20, 53. Meta-commentary ("honest translation", "instead of pretending"). Fix: state the platform fact directly. [SY-1, S8-6]

## Nit

N1. Lines 3-5. Pin the ZCode link to commit a60ba6a7; confirm AppErrorBoundary name matches ErrorBoundary.tsx. [DC-7, ST-10, SY-13]
N2. Lines 174-176, 168-171. Mutation checks mis-nested and ambiguous; store-level test cannot prove survival across a kill. Fix: move to Verification Plan or delete with contract tests. [ST-8, SY-12, IC-8, CO-13]
N3. Lines 146-157, 165-167. Two hook sites for one lifecycle. Fix: say AppShutdownCoordinator (AppModel+Persistence.swift:607) calls CrashRecoveryStore. [S8-8]
N4. Lines 135-141. Docs NFR has no path; privacy bullet conflicts with the new snapshot; "future agents" is rhetorical. Fix: add a path, reword. [CO-12, SY-9, SY-10, S8-7]
N5. Lines 111, 120-122, 128, 131. Intensifiers and hyperbole ("deliberately", "far", "true isolation", "zero cost"). Fix: delete. [SY-7, SY-8]
N6. Lines 158-164. Item 4 lacks Type / Scope and ZP links; "pins their error contract" is jargon. [ST-7, SY-11]
N7. Lines 83-88. Truncated line being appended could be dropped and not re-read. Fix: leave to ZP-03. [IC-6]
N8. Lines 29-31, 178-184. No localization or design-check gate in verification. [DC-8]
N9. Undefined terms (RolloutStore, trajectory pane, OfficeArchive, toasts). Fix: name files only. [CO-11]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10, IC-1 to IC-8. Feeds B1, M1-M5, N3, N7.
- comprehension_report.md: CO-1 to CO-13. Feeds B1, B2, M1-M5, M7-M9, N4, N9.
- critic_report.md: DC-1 to DC-8, ST-1 to ST-10, SY-1 to SY-15. Feeds all Blockers and Majors, N1-N8.
- intermediate_zoom_out_report.md: S8-1 to S8-8. Confirms B1 as Blocker, recommends rejecting IC-1, IC-2, IC-3, IC-4, IC-5, IC-6, IC-8 and extra alternatives; the target shrinks to marker, notice view, two hooks, one doc section and one rule.
- Merge notes: ST-1 and DC-1 merged into B2; S8-1 kept B1 as Blocker.
