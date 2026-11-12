# OC-14: Engine stall watchdog and guarded relaunch

Finding: Section 1, "Renderer crash, hang, and load-failure recovery -
Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[recovery dialogs](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/main/windows/recovery.ts)
and
[guarded relaunch](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/main/lifecycle/index.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Detect a wedged generation and offer a guarded relaunch path.
OpenCode's finding is mostly platform-moot (an AppKit app cannot lose
its renderer the way Electron can), but the report records three real
residuals: no hang detection, no guarded relaunch on the quit path, and
diagnostics export (already owned by the
[desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md)
spec). This draft covers the two that remain.

## The Problem

The app has stream-recovery presentation for agent turns
([AppStreamRecovery.swift](swift/TurboSparkApp/Sources/TurboSparkApp/State/AppStreamRecovery.swift))
but nothing detects a stalled engine: a Metal hang or a wedged turn
produces an unbounded wait, and the only exit is force quit, which
skips the shutdown flush. There is also no guarded relaunch: the
shutdown coordinator
([AppModel+Persistence.swift](swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift))
runs ordered saves but nothing bounds a hung step during a
relaunch-attempt path, which is exactly the failure OpenCode guards
with a 10-second timeout.

## Goals

* **Engine stall detection**: while a turn is active, a monitor watches
  the generation stream's token progress. No tokens and no state change
  for a configurable window (default 90 seconds) raises a stall event.
* **Stall response**: a localized recovery panel offers Keep waiting
  and Stop the turn (the existing cooperative cancel) and Restart the
  app. The panel states how long since the last progress.
* **Main-thread stall logging**: a watchdog thread heartbeats the main
  run loop and logs stalls beyond a threshold with the last activity
  label. It logs only; a wedged main thread cannot present UI, so the
  log is the artifact for support.
* **Guarded relaunch**: the restart action runs the shutdown
  coordinator under a timeout (10 seconds, matching OpenCode). A hung
  step is skipped with a log line, and the app relaunches via the
  standard workspace relaunch API. State risk is bounded by the
  debounced saves that already run during generation.
* The watchdog is opt-out per session (a debug escape hatch in the
  recovery panel) so a legitimately slow turn can be observed without
  the panel re-arming.

## Non-Goals

* No automatic self-relaunch without user consent: the panel is the
  gate.
* No crash reporter, no Sentry-style telemetry (excluded by the
  no-telemetry posture; see
  [OC-90](OC-90-excluded-and-decision-findings.md)).
* No Metal-level hang diagnosis; the watchdog observes progress, not
  GPU state.
* No per-request HTTP-style timeouts on engine calls; this is a
  turn-progress monitor, not a transport guard.

## Open Questions

* **Stall window default**: 90 seconds (lean) must clear legitimate
  long prefills on big contexts; a prefill-progress callback exists, so
  the monitor can treat prefill progress as activity. Confirm the
  interaction in review.
* **Relaunch API**: the standard macOS relaunch helper versus
  launching the bundle path directly. Lean: the standard helper, with
  the direct path as fallback when it reports unsupported.

## Critical User Journeys

### CUJ 1: wedged turn recovers

* **Given**: a turn active; the engine stopped producing tokens (Metal
  hang); 90 seconds pass.
* **When**: the stall event fires.
* **Then**: the panel appears with the elapsed time; Stop the turn
  ends it with partial output preserved per the existing cancellation
  contract; Restart flushes and relaunches.

### CUJ 2: relaunch with a hung save

* **Given**: the user chose Restart; one shutdown step hangs.
* **When**: the 10-second timeout elapses.
* **Then**: the step is abandoned with a log line, the remaining steps
  run or are skipped, and the app relaunches.

### CUJ 3: false positive

* **Given**: a legitimately slow prefill with visible progress
  callbacks.
* **When**: the stall window elapses.
* **Then**: no panel: prefill progress counts as activity.

## Technical Plan (Architecture)

```text
GenerationStreamMonitor (watchdog thread, injected clock)
  observes token/prefill progress of the active turn
  no activity within window -> stall event on main queue
        |
        v
StallRecoveryPanel (SwiftUI, presented by AppModel)
  Keep waiting (re-arm) / Stop turn (existing cancel) / Restart app
        |
        v
GuardedRelaunch
  AppShutdownCoordinator under 10 s timeout -> workspace relaunch
MainThreadHeartbeat (watchdog thread)
  run-loop heartbeat, stall logging beyond threshold (log only)
```

Both monitors live on a watchdog thread with an injected clock, so the
state machines are unit-testable without real waits. The panel reuses
the stream-recovery presentation surface so there is one recovery UI
family.

## Non-Functional Requirements

* **Performance**: the monitors cost a timer tick per second and a
  lock-free progress read; no per-token work.
* **Safety**: the watchdog never calls into the engine; it only reads
  progress and posts events. Relaunch skips hung steps rather than
  waiting forever, and never writes during the abandoned step.
* **Localization**: panel copy, states, and the escape-hatch tooltip
  in 21 languages in the same change.
* **Accessibility**: the panel is fully keyboard-operable; Keep
  waiting is the default action; state changes announce.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/GenerationStreamMonitor.swift` (Create)
   * **Rationale**: the progress monitor and stall state machine with
     the injected clock.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/MainThreadHeartbeat.swift` (Create)
   * **Rationale**: the run-loop heartbeat and stall logging.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/StallRecoveryPanelView.swift` (Create)
   * **Rationale**: the recovery panel; reuses the attention vocabulary
     and stream-recovery surface.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: the guarded relaunch path: shutdown under timeout,
     skip-and-log per step, workspace relaunch.
5. `swift/TurboSpark/Sources/TurboSpark/QueuedGenerationStream.swift` (Modify)
   * **Rationale**: expose a progress heartbeat (last-activity stamp)
     the monitor reads; no behavior change to the stream itself.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Rationale**: start the monitors after launch; stop them in
     shutdown.
7. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/GenerationStreamMonitorTests.swift` (Create)
   * **Rationale**: stall on silence, activity resets, prefill progress
     counts, re-arm, escape hatch. Mutation check: treat prefill as
     silence and assert the false-positive test fails.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/GuardedRelaunchTests.swift` (Create)
   * **Rationale**: timeout skip-and-log per step, ordering preserved,
     relaunch invocation with a stubbed workspace call.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suites plus localization gates and
  `make design-check`.
* **Manual**: simulate a stall with a debug hang injection (a debug
  flag that stops the stream), confirm the panel, both actions, and the
  relaunch path; confirm the timeout skip with a stubbed hung save.
  Not a hardware gate; a real Metal-hang reproduction is out of scope
  for this draft.

--------------------------------------------------------------------------------

*Drafted with EGM*
