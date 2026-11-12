# Stage 8: Intermediate Zoom Out report (ZP-17 error boundaries)

Reviewer: egm-zoom-out (S8). Completed. Document unchanged since Stage 5. Verdict: not acceptable as written, but the fix is subtraction. Only the crash marker plus a one-time notice solves a stated problem; most Stage 5-7 asks add machinery and should be rejected.

## Findings

S8-1 Blocker. Lines 158-164, 168-172. Target File 4 edits files that do not exist, and INDEX sequences ZP-17 before ZP-03/06/08. Fix: drop Target File 4 and the per-surface tests; keep a cross-reference; each surface draft owns its failure card. Do not add a shared RenderFailure type (IC-4).
S8-2 Major. Lines 34-38, 47-48, 69-72, 148-150. Restore action and snapshot are speculative: selected chat is already persisted (ProfileDatabase.swift:288) and AppStreamRecovery makes interrupted rows. Goal 4 is unachievable on SIGKILL. Fix: marker plus one-time notice pointing to the existing interrupted row; drop snapshot and restore payload test; drop or reword Goal 4 as no regression versus cold-start behavior.
S8-3 Major. Lines 56-61, 94-98, 135. Marker under-specified. Fix: write after unlock, clear on every orderly termination path, drop pid, store relative to the storage root (respect TURBOSPARK_STATE_DIR), notice once per marker; move from Open Questions into the plan.
S8-4 Major. Lines 44-46, 106-109. Lint has no tool or gate and "view-body call paths" is not mechanical. Fix: one sentence in AGENTS.md or design-check rules, or drop the goal.
S8-5 Major. Lines 4, 22-30, 39-43, 78-79, 100-104, 126-131. No evidence of a real crash class; "process does not crash" unsupportable; no retry/reload delivered so the parity claim overstates. Fix: soften CUJ 2 and 3 to cross-links, cite one offender count, reword parity claim.
S8-6 Nit. Lines 16-20, 111-114. Lead the Overview with the "not a React-style boundary" statement; remove meta-commentary.
S8-7 Nit. Lines 135-140, 178-184. Keep the docs NFR but give it a path; keep manual SIGKILL test.
S8-8 Nit. Lines 146-157, 165-167. Say once that AppShutdownCoordinator (AppModel+Persistence.swift:607, hooked at TurboSparkApp.swift:72) calls CrashRecoveryStore.

## Asks to reject (scope or complexity)

IC-1 incremental turn checkpointing; IC-2 boot-session hint, launch phase, vault payload, crash-loop breaker; IC-3 parser limits, fuzz fixtures, XPC (belong in ZP-08/ZP-06; XPC contradicts single-process design); IC-4 shared RenderFailure type; IC-5 crash summary export and safe mode; IC-6 byte-offset tracking (ZP-03); IC-8 automated kill -9 harness; DC-2/ST-3 extra alternatives beyond one line on OS crash reports; CO-9 crash-loop and dismissal handling beyond once-per-marker; DC-8/ST-5 success metrics and extra CUJs; CO-11 defining every internal type.

## Asks to keep

ZO-1/DC-3 file-existence and ordering; goal contradiction (ZO-2, CO-3, DC-4, SY-4); marker correctness (ZO-4); evidence and crash-class correction (ZO-6, SY-2); a short Risks & Limits section (ST-1, DC-1: SIGKILL, Force Quit, power loss, reboot false positives; traps not contained); pin ZCode link to a60ba6a7 (DC-7, ST-10); move or delete mutation checks (ST-8); Open Question becomes a decision (SY-6).
