# ZP-17: Error boundaries

Finding 8.6 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[AppErrorBoundary](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/ErrorBoundary.tsx).

* Author: whit3rabbit
* Status: Draft
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

--------------------------------------------------------------------------------

*Drafted with EGM*
