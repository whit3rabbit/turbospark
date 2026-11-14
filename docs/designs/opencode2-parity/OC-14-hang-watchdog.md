# OC-14: Engine stall watchdog and guarded relaunch

Finding: Section 1, "Renderer crash, hang, and load-failure recovery -
Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[recovery dialogs](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/main/windows/recovery.ts)
and
[guarded relaunch](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/main/lifecycle/index.ts).

* Author: whit3rabbit
* Status: Partial (stall monitor, panel and guarded relaunch implemented; main-thread heartbeat and a live hang soak deferred)
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

* **Stall window default** (RESOLVED, see the Implementation record): 90 seconds (lean) must clear legitimate
  long prefills on big contexts; a prefill-progress callback exists, so
  the monitor can treat prefill progress as activity. Confirm the
  interaction in review.
* **Relaunch API**: RESOLVED, see the Implementation record (no standard
  helper exists; detached `open` helper).

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

## Implementation record

Implemented 2026-11-14. The EGM review in
`docs/egm-reviews/opencode2-parity/OC-14/` found the draft's relaunch
timeout unimplementable (`AppModel.shutdown()` is one synchronous
main-actor function) and its "debounced saves bound the risk" claim false (a
streaming reply exists only in `outputText`). The build follows the review's
reduction, not the draft text above where they differ.

### Decisions (Open Questions and review blockers)

* **Stall window**: fixed 90 seconds, hardcoded
  (`GenerationStallMonitor.defaultWindow`). The configurable window and the
  per-session opt-out are dropped (review M8). Keep Waiting re-arms one more
  full window; the next report states total silence since the last event.
* **Prefill is activity**: every stream event (`.prefill`, content,
  reasoning, tool call, stopped, finished) goes through one
  `noteActivity` call. UNVERIFIED: whether one very large prefill chunk can
  stay silent past 90 seconds on real hardware. Keep Waiting is the answer
  to that case; a long-prefill soak on a real model is still owed (below).
* **Armed scope** (review M2): armed only inside one local engine generate
  call, from just before `session.generate` is iterated until the call
  ends. Tool execution, approvals, remote turns, compaction, Humanize and
  audio are not watched.
* **Clock**: `ProcessInfo.systemUptime`, which does not advance while the Mac
  sleeps, so a lid close mid-turn is not a stall.
* **Watchdog thread**: replaced by a main-actor `Task` ticking once a
  second. Stream events are consumed on the main actor and a wedged engine
  thread does not block it, so no lock and no thread are needed. One clock
  read and one struct store per event. `QueuedGenerationStream` is NOT
  modified (target file 5 dropped): the consumer loop stamps events.
* **Stop Turn** (review M1): the existing cooperative `cancel()`. It ends
  the turn and keeps the partial reply but cannot free a wedged engine,
  whose serial queue stays in its blocking call; the next send waits behind
  it, and the monitor re-arms when that send opens its generate call. The
  panel says so and points at Restart.
* **Relaunch mechanism** (review B4; the Open Question is closed): there is
  no standard relaunch helper. A detached `/bin/sh` waits at most 30 s for
  this pid to exit and then runs `/usr/bin/open <bundle>` (plain `open`,
  never `-n`, so a helper that outlives a stuck process only re-activates
  it instead of starting a second instance on the same store root). Pid and
  path are positional arguments, never spliced into script text. A bare
  executable (`swift run`) reports unavailable.
* **Guarded relaunch** (review B2, B3, M9): no per-step shutdown timeout, no
  skip logic. Persist first, then quit, with a hard bound:
  1. count the attempt (`RelaunchGuard`);
  2. spawn the helper;
  3. arm a 10 s hard exit on a background queue (`_exit`);
  4. `AppModel.flushForRelaunch()`: streaming reply as an interrupted row,
     draft chat, chats, settings, file store. It touches no engine, server
     or session object, so it cannot block on a stuck engine;
  5. `NSApp.terminate`, which runs the ordinary quit path and clears the
     crash marker last.
  The hard exit clears the `session-marker.json` marker only when step 4
  finished, so a quit that lost data still produces next launch's "did not
  quit cleanly" notice (ZP-17).
* **Loop guards**: at most 2 relaunch attempts in 10 minutes, kept in
  `relaunch-guard.json` under the store root (attempt times only, never
  sent anywhere) so the cap survives the relaunch it counts. An unwritable
  record refuses (fail closed); a corrupt record is replaced and counts.
  Failed spawns count. A second click on one relaunch is latched.
  The relaunched app does nothing on its own: a relaunch is always the
  user's click on the panel, so the bound is on repeated clicks.
* **Restart confirmation**: none. The panel button is the explicit choice
  and the chats are saved before anything quits.
* **Panel** (review M5): a new view, `Generation/StallRecoveryBannerView.swift`
  (not under `State/`), modeled on `ErrorBanner` and
  `NewChatSuggestionBanner`, placed above the error banner in the
  conversation chrome. Keep Waiting is the single primary action and has no
  `.defaultAction` chord (Return belongs to the composer); all three are
  plain buttons reachable by Tab. The strip is announced once when it
  appears. It reuses no "stream-recovery presentation surface": none exists.
* **Stale panel race** (review M10): the tick, the event stamps and the
  state they patch all run on the main actor; the notice is also guarded by
  `generating` and cleared by the next event, Keep Waiting, Stop and the end
  of the call.

### What shipped

* `Diagnostics/GenerationStallWatchdog.swift`: `GenerationStallMonitor`
  (pure state machine) and `GenerationStallWatchdog` (main-actor owner).
* `Diagnostics/GuardedRelaunch.swift`: `RelaunchGuard`, `GuardedRelaunch`
  and its live seams.
* `State/AppModel+StallWatchdog.swift`, wiring in
  `State/AppModel+Generation.swift`, three stored properties in
  `State/AppModel.swift`, and `flushStreamingReplyForShutdown()` /
  `flushForRelaunch()` split out of `shutdown()` in
  `State/AppModel+Persistence.swift` (behavior of `shutdown()` unchanged).
* `Generation/StallRecoveryBannerView.swift`, 11 new strings in 21
  languages.
* Tests: `GenerationStallWatchdogTests.swift`, `GuardedRelaunchTests.swift`.
  Injected clock, recording seams, hard exit captured; no test relaunches,
  exits or hangs. Mutation-checked: activity not resetting the window
  (prefill as silence), no attempt cap, terminate before flush.

### Deferred or not done

* **Main-thread heartbeat** (review M3): not built. No consumer, threshold,
  activity-label registry or log sink exists, and the `desktop-shell-hardening`
  spec it leaned on does not exist. The crash marker plus the panel are the
  support artifact. Reopen with a concrete consumer.
* **Diagnostics export**: still owned by whichever change builds a log sink;
  nothing here writes a log or a report.
* **Real hang evidence**: verified with injected stalls and clocks only. No
  Metal hang, long-prefill soak, or live relaunch was exercised: those need a
  real model on real hardware and an actual app bundle. Owed before
  claiming the 90 s window clears long prefills.
* **Compaction and Humanize generate loops** consume the same event type but
  are not armed.
* A debug hang-injection flag (Verification Plan) was not added: it would
  not reproduce a blocked serial queue, which is the real failure.
* Excluded by OC-90, unchanged: no crash reporter, no telemetry, nothing
  leaves the device.

--------------------------------------------------------------------------------

*Drafted with EGM*
