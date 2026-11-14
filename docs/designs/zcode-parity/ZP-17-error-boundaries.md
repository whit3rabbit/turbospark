# ZP-17: Error boundaries

Finding 8.6 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[AppErrorBoundary](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/ErrorBoundary.tsx).

* Author: whit3rabbit
* Status: Implemented (partial: recovery marker and notice only; typed-error contract deferred)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Contain render failures so one broken view cannot take down the whole
window. ZCode gets this from React's error boundary tree; SwiftUI has no
equivalent primitive, so this design is an honest translation: scoped
defensive boundaries for the known failure classes, a process-level
crash-recovery story for everything else, and a documented list of what
cannot be contained in-process.

## The Problem

Any uncaught throw in a view body or a force-unwrap in a model used by a
view can crash the app process. ZCode wraps both renderer entries in an
error boundary with retry/reload, plus scoped boundaries for partial
isolation. TurboSpark's nearest analogs are domain-specific recovery
views (StreamRecoveryEventNoticeView) and toasts; there is no general
containment. A crash loses the in-flight turn's continuity and shows the
vault unlock again on relaunch.

## Goals

* A crash recovery surface: after an abnormal exit, the next launch
  detects the previous abnormal termination (crash marker file written
  at launch, cleared on clean quit) and offers "Restore previous
  session", restoring the selected chat and window state without
  waiting for the user to find it.
* Scoped defensive boundaries around the three known high-risk render
  surfaces (trajectory pane parsing, office rendering of untrusted
  documents, plugin pages): their parse/render failures render a
  bounded error card instead of crashing, driven by typed errors rather
  than exceptions.
* A coding-standard rule set: no force unwraps or `try!` in view-body
  call paths; a lint pass documents current offenders and prevents new
  ones.
* The in-flight turn on crash is recoverable exactly as the existing
  interrupted-row flow does on quit.

## Non-Goals

* No process-internal isolation of arbitrary SwiftUI crashes: it does
  not exist on the platform, and the draft says so instead of pretending.
* No automatic crash reporting to any service (telemetry excluded).

## Open Questions

* **Crash marker location**: a non-vault marker (needs to be readable
  before unlock) vs inside the vault. Non-vault marker is required since
  detection happens before profile unlock. Confirm no privacy objection
  (marker holds a timestamp and pid, no content).

## Critical User Journeys

### CUJ 1: crash and restore

* **Given**: the app crashes mid-session.
* **When**: the user relaunches.
* **Then**: a recovery banner offers "Restore previous session";
  accepting reopens the window state and the last selected chat; the
  in-flight turn appears as an interrupted row per existing quit
  semantics where the flush allowed it.

### CUJ 2: corrupted document render

* **Given**: a crafted docx with malformed XML.
* **When**: the office viewer parses it.
* **Then**: the viewer renders a bounded error card naming the parse
  failure with an external-opener fallback; the process does not crash.
  (The OfficeArchive hardening already bounds zip bombs; this covers the
  XML layer with typed errors.)

### CUJ 3: trajectory pane on a partial line

* **Given**: a rollout log with a truncated final line.
* **When**: the pane reads the tail.
* **Then**: the truncated line is skipped with a count notice; the pane
  renders the valid entries.

## Technical Plan (Architecture)

```text
CrashRecoveryStore
  marker file at launch (non-vault, timestamp+pid)
  cleared on AppShutdownCoordinator clean quit
  abnormal exit -> marker survives -> next launch
  shows RecoveryBannerView
       |
       v
Typed-error render contract (per high-risk surface)
  parse/decode APIs return Result types
  views render failure states as bounded cards
  (trajectory tail reads, office XML, plugin page loads)
       |
       v
Coding standard + lint pass
  no force unwrap / try! in view call paths
  baseline scan documented, new offenders blocked
```

This is deliberately not a React-style boundary: SwiftUI has no
mechanism to catch an arbitrary throw from a sibling view. The
containment is (a) typed errors at the data layer, (b) failure states in
views, (c) process-level recovery, (d) style enforcement.

## Alternatives Considered

### Alternative A: fork the renderer into child processes per pane

* **Pros**: true isolation.
* **Cons**: rearchitecting the app around out-of-process rendering is
  far outside parity scope and contradicts the single-process Swift app
  design.
* **Reason for Rejection**: Rejected by author: disproportionate.

### Alternative B: do nothing beyond existing domain recovery views

* **Pros**: zero cost.
* **Cons**: the crash-restore gap remains; untrusted-document render
  crashes remain possible.
* **Reason for Rejection**: Rejected by author: real gap.

## Non-Functional Requirements

* **Privacy**: the crash marker holds no content; recovery reads only
  state the app already persists.
* **Performance**: marker write/read is one file op each at launch and
  quit.
* **Docs**: the containment limits get a section in the Swift app docs
  so future agents do not re-derive the platform truth.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/CrashRecoveryStore.swift` (Create)
   * **Type / Scope**: marker and detection.
   * **Proposed Changes**: marker write at launch, clear on clean quit
     via AppShutdownCoordinator, detection and last-state snapshot for
     the banner.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Type / Scope**: launch and quit hooks.
   * **Proposed Changes**: marker lifecycle hooks and recovery banner
     presentation.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Components/RecoveryBannerView.swift` (Create)
   * **Type / Scope**: recovery UI.
   * **Proposed Changes**: localized banner with restore and dismiss.
4. High-risk surface files (Modify, typed-error contract):
   `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/RolloutStore.swift`,
   `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/OfficeRenderModel.swift`,
   `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PluginPageWebView.swift`
   * **Proposed Changes**: parse/render APIs return typed results;
     failure states render bounded error cards (ZP-03/08/06 implement
     the surfaces; this draft pins their error contract).
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Type / Scope**: shutdown.
   * **Proposed Changes**: clear the marker in the clean-quit path.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/CrashRecoveryTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: marker lifecycle (write, clean-quit clear,
     abnormal survival), restore payload correctness, typed-error
     contract for each surface (truncated line skipped, malformed XML
     card, page-load failure card).
   * Mutation checks: clear the marker on abnormal exit and assert the
     survival test fails; let a surface throw instead of returning a
     typed error and assert the contract test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for recovery and contract tests.
* **Manual**: kill the app mid-turn (SIGKILL), relaunch, verify the
  banner and restore; feed a malformed fixture to each high-risk
  surface and verify bounded failure states.

## Implementation notes (2026-11-13)

Applies the EGM review (docs/egm-reviews/zcode-parity/ZP-17) Blockers and
Majors, and the INDEX review's sequencing finding (ZP-17 before ZP-03/08/06).

### Landed

* `App/CrashRecoveryStore.swift`: marker `session-marker.json` under the
  profile store root (`AppStorageRoot.directory`), holding a format version and
  start time only. No pid, no content, never sent off device. Written when the
  main window first appears (after the vault gate), deleted in
  `applicationWillTerminate` after the other quit steps. A surviving marker
  shows one warning toast and is replaced in the same call, so the notice is
  once per marker. A second begin in the same process is a no-op so a vault
  lock and unlock does not read the process's own marker.
* Hooks: `RootView.onAppear` (begin) and `ForegroundAppDelegate.applicationWillTerminate`
  (end). `AppModel.shutdown()` does not clear it, since a vault lock also runs
  there and the process survives.
* Tests: `CrashRecoveryTests` (clean first run, clean-quit clear, survival
  into the next run, once per marker, same-process second begin, corrupt
  marker, content-free keys, unwritable directory). Mutation-checked: a no-op
  `endSession` fails the clean-quit test, a no-op write fails the survival test.
* Docs: `swift/docs/SWIFT_CRASH_RECOVERY.md` (containment limits) and a
  Containment rule in `swift/AGENTS.md`. The coding standard is a written
  review rule, not a lint gate (review M4).

### Deviations from the draft

* No `RecoveryBannerView`, no "Restore previous session" action, no state
  snapshot (review M2): the selected chat and window state already persist, so
  a restore button would duplicate existing behavior. A warning toast is the
  notice.
* Goal 4 restated (review M1): SIGKILL runs no flush, so an in-flight reply is
  not recovered. The toast says it "may be incomplete". The interrupted-row
  save on an orderly quit is unchanged.
* The hook is `AppShutdownCoordinator`-adjacent but sits in
  `applicationWillTerminate`, not `AppModel.shutdown()` (see above).
* CUJ 1 reduces to: relaunch after a crash shows the notice; no automatic
  state restore. CUJ 2 and CUJ 3 are deferred.
* "The process does not crash" (CUJ 2) is withdrawn: typed errors do not stop
  traps, overflow, deep nesting or out of memory.

### Risks

SIGKILL, Force Quit, reboot or power loss mid-session and a second instance on
the same store root all leave a marker and show the notice (accepted; wording
is "did not quit cleanly"). A marker write failure silently skips the next
notice. A trap during view rendering is not contained.

### Deferred (blocked on unbuilt drafts)

The typed-error contract (Target File 4 and its tests) is dropped from this
draft. Each surface draft owns its failure card: ZP-03 `RolloutStore` skips a
truncated final line with a count notice (CUJ 3; done in ZP-03, see its
Implementation notes), ZP-08 `OfficeRenderModel`
renders a bounded error card on malformed XML with an external-opener
fallback (CUJ 2; done in ZP-08, see its Implementation notes:
`OfficeRenderError` is the typed error, `OfficeRenderFailureView` the card,
and `testMalformedXMLGivesATypedBoundedErrorNotACrash` the test), ZP-06
`PluginPageWebView` renders a load-failure card. The ZP-06 file does not
exist yet.

--------------------------------------------------------------------------------

*Drafted with EGM*
