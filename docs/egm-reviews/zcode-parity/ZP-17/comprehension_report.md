# Stage 6: Comprehension report (ZP-17 error boundaries)

Reviewer: egm-comprehension (read the document alone). Completed.

## What this proposes

Four pieces of crash containment for the single-process SwiftUI app, modeled loosely on ZCode's React error boundary. (1) A crash marker file written at launch and cleared on clean quit. If it survives, the next launch shows a banner offering "Restore previous session" (selected chat and window state). (2) Typed Result-style errors on three risky surfaces (trajectory tail reads, office XML parsing, plugin page loads) so failures render a bounded error card. (3) A coding rule and lint pass banning force unwraps and `try!` in view-body call paths, with a baseline of current offenders. (4) A docs section recording what cannot be contained in-process.

## How the existing system works, per this document

SwiftUI cannot catch an arbitrary throw from a view, so any uncaught throw or force unwrap crashes the process. Existing recovery is domain-specific (StreamRecoveryEventNoticeView, toasts). On quit, an in-flight turn becomes an interrupted row and AppShutdownCoordinator handles clean quit. A vault holds the profile and must be unlocked at launch. OfficeArchive already bounds zip bombs. The document does not say what state is persisted today, so it is unclear what a restore can restore.

## Findings (CO)

CO-1 Major. Lines 3-7, 24-31. "Parity" and "both renderer entries" are not explained; retry/reload is named at line 26 but has no equivalent in the plan. Which ZCode behaviors are matched or dropped is unclear.
CO-2 Major. Lines 24-31, 47-48, 58-61. Whether unlock-on-relaunch is a problem being solved is unstated; CUJ 1 omits the vault unlock though detection happens before it. Banner and restore sequencing around unlock is unclear; "vault" is undefined.
CO-3 Major. Lines 47-48, 69-72. "Exactly as on quit" conflicts with "where the flush allowed it"; a hard crash has no flush. "Interrupted-row flow" and "flush" undefined.
CO-4 Major. Lines 34-38, 136, 149, 165-167. "Window state" and "selected chat" undefined; who writes the "last-state snapshot", when and where is not stated. In the vault it conflicts with pre-unlock detection; outside it conflicts with the no-content privacy claim. AppModel+Persistence.swift entry lists only clearing the marker.
CO-5 Major. Lines 39-43, 100-104. "Bounded" error card undefined. No evidence of crashes on the three surfaces. Silent on traps (index out of range, memory exhaustion), the failure class actually named in the Problem.
CO-6 Major. Lines 158-164, CUJ 3. ZP-03/08/06 not mapped to surfaces; unclear whether the files exist or what happens if ZP-17 lands first. The Result/error type shape is never specified. Plugin page failure card has no CUJ or failure cases (a WebView loads rather than parses).
CO-7 Major. Lines 44-46. Lint has no tool, rule, definition of "view-body call paths", or run location, and is absent from Target Files and Verification Plan.
CO-8 Major. Lines 56-61, 135. Open Question answers itself yet is listed open. Stale pid and second-instance handling undescribed.
CO-9 Major. Lines 63-72, 135, 178-184. Unaddressed: crash loop on restore, banner dismissal, OS kill or power loss counted as a crash, intentional kill during development.
CO-10 Nit. Lines 124, 131. "Rejected by author" is a bare assertion; "single-process design" not described elsewhere.
CO-11 Nit. Terms used undefined: RolloutStore, rollout log, trajectory pane, OfficeRenderModel, OfficeArchive, AppShutdownCoordinator, StreamRecoveryEventNoticeView, toasts.
CO-12 Nit. Lines 139-140, 178-184. "Swift app docs" has no path or target file; SIGKILL manual test does not cover a real SwiftUI failure.
CO-13 Nit. Lines 168-176. Mutation checks: unclear whether automated or one-time.

No Blocker: every section could be paraphrased.
