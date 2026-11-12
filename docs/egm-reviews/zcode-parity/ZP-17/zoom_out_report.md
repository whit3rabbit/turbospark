# Stage 5: Initial Zoom Out report (ZP-17 error boundaries)

Target: docs/designs/zcode-parity/ZP-17-error-boundaries.md (full file, 189 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom Out (ZO)

Verdict: do not accept as written. Only the crash marker earns its place, and it needs rework. Verified: RolloutStore.swift, OfficeRenderModel.swift, PluginPageWebView.swift do not exist under swift/. AppChatArchive already persists selectedChatID (ProfileDatabase.swift:288). AppStreamRecovery.swift already has interrupted-row anchors (lines 556-590). No SwiftLint config or lint target exists.

ZO-1 Blocker. Lines 158-163, 168-172. Target File 4 modifies three files that ZP-03/08/06 have not created, and INDEX sequences ZP-17 (step 3) before ZP-03 (step 4) and ZP-08 (step 5). Contract tests cannot be written. Fix: drop Target File 4; add one line to each surface draft; keep a cross-reference.
ZO-2 Major. Lines 47-48, 69-72. "Recoverable exactly as on quit" is unsupported: SIGKILL runs no flush. Fix: drop the goal or restate as no regression versus what AppStreamRecovery does on cold start.
ZO-3 Major. Lines 34-38, 148-150. Selected chat and window frame are already restored on relaunch; the restore snapshot and "Restore previous session" action add little. Fix: marker plus one-time notice pointing to the interrupted row; drop the snapshot unless a gap is shown.
ZO-4 Major. Lines 56-61, 94-98, 148-150. Marker has false-positive holes (force-quit at unlock screen, logout, Sparkle relaunch), pid is useless for liveness, and multiple launches or TURBOSPARK_STATE_DIR redirection are unhandled. Fix: write after unlock, clear on every orderly termination path, drop pid, specify location relative to the storage root.
ZO-5 Major. Lines 44-46, 106-109. Lint pass has no tool, no CI gate, and "view-body call paths" is not mechanically definable. Fix: one rule in AGENTS.md or design-check, or drop.
ZO-6 Major. Lines 22-30, 126-131. No evidence of a real crash class; Alternative B rejected without data; no retry/reload is delivered, so the parity claim at line 4 overstates. Fix: cite a crash or offender count; reword parity claim.
ZO-7 Nit. Lines 74-88. CUJ 2 and 3 belong to ZP-08 and ZP-03; XMLParser reports errors through its delegate rather than crashing. Fix: keep as cross-links.
ZO-8 Nit. Lines 135-140. Docs NFR is worth keeping; privacy bullet fine once pid is dropped.
ZO-9 Nit. Lines 146-157, 165-167. Two hook sites for one lifecycle. Fix: say once that the coordinator calls CrashRecoveryStore; confirm AppModel+Persistence.swift hosts AppShutdownCoordinator.
ZO-10 Nit. Lines 111-114. The "not a React-style boundary" disclaimer should lead the Overview.

## Inventive Catalyst (IC)

IC-1 Major. Lines 47-48, 69-72, 93-98, 174-176. Goal 4 contradicts the "where the flush allowed it" caveat; SIGKILL runs no flush. Resolution: checkpoint the in-flight turn incrementally so a crash and a quit leave the same artifact; update the restore snapshot on change. Liabilities: more writes, journal versioning, vault encryption for content.
IC-2 Major. Lines 56-61, 135-136. Marker cannot tell a crash from force-quit, reboot, or a crash loop. Resolution: non-vault marker with timestamp, boot-session hint, launch phase; restore payload in the vault applied after unlock; crash-loop breaker and one offer per marker.
IC-3 Major. Lines 44-46, 111-114, 158-164. Typed Results do not stop array traps, overflow, deep-XML stack exhaustion, or WebKit content-process termination. Resolution: handle webViewWebContentProcessDidTerminate with a capped reload; hard parser limits (depth, node count, size) with fuzz fixtures; bounded tail reads. Consider XPC only for untrusted office parsing.
IC-4 Major. Lines 158-164 versus INDEX. Same ordering coupling as ZO-1. Resolution: define a shared RenderFailure/error-card contract once; ZP-03/08/06 adopt it; or reorder INDEX.
IC-5 Major. Lines 22-30, 111-114. Detection is passive and relaunch-only; line 29 says vault unlock shows again but the plan does not say whether recovery needs unlock first. Resolution: local-only crash summary in the existing log export; crash-loop counter offering safe mode.
IC-6 Nit. Lines 83-88, 171-172. A truncated line still being appended would be skipped and never re-read. Fix: track byte offset of last complete line.
IC-7 Nit. Lines 44-46, 142-176. Lint has no target file, baseline location, or enforcement hook.
IC-8 Nit. Lines 174-176. Store-level mutation test cannot prove survival across a real kill. Fix: add a process-level kill -9 smoke test.
