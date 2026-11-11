# ZP-01: Off-peak (idle-time) background tasks

Finding 1.2 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Cron parity target: ZCode's off-peak automation domain
([scheduler](https://github.com/zai-org/ZCode/blob/main/packages/desktop/src/scheduler/index.ts),
[offPeakTaskStore](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/store/offPeakTaskStore.ts)).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add a second automation domain to the Swift app: tasks the user wants run
eventually at low priority, admitted and dispatched only while the machine
is idle. This is the TurboSpark translation of ZCode's off-peak task
system, replacing its server-ticketed admission with a local
power-assertion-based admission gate.

## The Problem

TurboSpark already has cron automations
(`swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/CronScheduler.swift`):
a 20 second timer over a JSON store that fires prompts into the
originating chat. It has nothing for "run this when the computer is not
busy." Users who want batch generation, bulk model benchmarking, or long
evaluations have no way to say "when idle". ZCode solves this with a
server-side ticket queue (POST /ticket, FIFO admission, ticket expiry) plus
an `IOPMAssertion`-style keep-awake tied to active off-peak work. The
server half does not exist for TurboSpark and is out of scope; the local
half is portable and useful on its own.

## Goals

* A user (or the model, through a tool) can enqueue an off-peak task:
  a prompt plus target chat, priority, and optional constraints.
* Admission is local: FIFO by enqueue order, gated on machine idle state.
* While any off-peak task runs, the app holds a macOS no-idle-sleep
  assertion and releases it when none run.
* An interrupted off-peak task (app quit, crash) is recovered as
  interrupted and can be re-run from the Scheduled Tasks pane.
* Misfire semantics: a task older than a configurable staleness bound is
  skipped with a recorded outcome instead of firing late.
* The model can list, enqueue, and cancel off-peak tasks through tools,
  subject to a job cap serialized through the same store lock as the
  scheduler.

## Non-Goals

* No server, ticket endpoint, or cross-device admission. Admission is
  local-only. (Server admission is analyzed in
  [ZP-90](ZP-90-excluded-and-decision-findings.md) under finding 1.4.)
* No new model-runtime behavior: off-peak tasks reuse the existing
  generation path, including its interrupt and queue semantics.
* No change to the cron scheduler's cron-expression semantics.

## Open Questions

* **Admission signal**: IOPMAssertion-based idle detection is
  macOS-specific and coarse. Options:
  (a) combine `IOPMAssertion` (no idle sleep) with the app's own
  idle heuristic (no generation running, no tool call pending, screen
  locked or user-input idle above N minutes);
  (b) user-declared "allow off-peak now" toggle only;
  (c) both, with (a) requiring the toggle as an additional gate.
  The idle heuristic is the mechanism with no defensible single answer;
  pick one before implementation.
* **Where off-peak tasks run**: the originating chat (like cron today),
  a dedicated off-peak chat, or a scratch chat per task. Dedicated or
  scratch chats avoid polluting user conversations but add profile
  plumbing. Not decidable without the concurrent-chats decision (see
  ZP-02 dependencies).
* **Job cap**: ZCode caps at 20 jobs. 20 is a reasonable default;
  confirm whether a cap belongs in settings or stays fixed.

## Critical User Journeys

### CUJ 1: enqueue and idle dispatch

* **Given**: the app is running, two chats exist, generation idle.
* **When**: the user adds an off-peak task from the Scheduled Tasks pane
  ("summarize the audio library transcript") and locks the screen or
  leaves the machine idle past the heuristic threshold.
* **Then**: the task is admitted FIFO, a no-sleep assertion is taken, the
  prompt runs in its target chat, and the task record gains a run entry
  with start and end times. Releasing the last task releases the
  assertion.

### CUJ 2: interrupted recovery

* **Given**: an off-peak task is running.
* **When**: the user quits the app mid-run.
* **Then**: on next launch the task row shows interrupted (from a
  persisted run state written at dispatch and cleared at completion), and
  the pane offers re-run. The existing shutdown path already converts an
  in-flight reply into an interrupted row, so recovery reuses it.

### CUJ 3: model-enqueued task

* **Given**: the user asks the model to "queue a job to reindex my
  embeddings when idle".
* **When**: the model calls the off-peak task tool.
* **Then**: the tool writes through the same store and cap as the UI, the
  task appears in the pane with a model-origin marker, and cancel from
  the pane works.

## Technical Plan (Architecture)

The existing scheduler (`CronScheduler`) already owns a 20 second tick, a
JSON store, and settings-pane integration. Off-peak becomes a second
dispatcher inside the same Automation module rather than a parallel
universe:

```text
Scheduled Tasks pane
        |
        v
+-----------------+      +---------------------------+
| CronScheduler   |      | OffPeakTaskDispatcher     |
| (cron prompts)  |      | queue: [OffPeakTask]      |
+-----------------+      | state machine per task    |
        |                +---------------------------+
        |                        |  ^
        v                        v  |
   shared JSON store (new file offpeak_tasks.json,   |
   AppJSONStore quarantine rules apply)              |
                                |                    |
                     OffPeakAdmissionGate           |
                     (idle heuristic + user toggle) |
                                |                    |
                        PowerAssertionKeeper         |
                        (IOPMAssertion no-idle)      |
                                v
                     existing generation entry point
                     (AppModel submit path, interrupt
                      and interrupted-row reuse)
```

State machine per task: queued, admitted, running, completed,
interrupted, skipped (stale), cancelled. The dispatcher claims one task
at a time; the claim is written to the store before generation starts so
a crash leaves recoverable state.

## Alternatives Considered

### Alternative A: server-ticketed admission like ZCode

* **Pros**: cross-device coordination, server-side fairness.
* **Cons**: TurboSpark has no service to host tickets; building one is
  the excluded cloud category.
* **Reason for Rejection**: Rejected by author: no server exists, and the
  roadmap's exclusion policy rules the service out.

### Alternative B: fold into the cron scheduler as "cron with special time"

* **Pros**: one store, one pane.
* **Cons**: cron semantics are wall-clock; off-peak is
  availability-based. Folding them couples unrelated state machines and
  the shared tick hides off-peak's power assertions.
* **Reason for Rejection**: Rejected by author: keep the two dispatchers
  separate but store-adjacent, so each keeps its own semantics.

## Non-Functional Requirements

* **Security and privacy**: the store is under the existing profile/app
  state root with AppJSONStore quarantine; prompts may contain sensitive
  text and never leave the machine.
* **Power**: the no-sleep assertion must be released on task end, app
  quit, and assertion-taking failure; the failure path degrades to
  "task runs only while the machine stays awake on its own".
* **Rollout**: behind a settings toggle, default off, consistent with the
  app's opt-in posture for background machinery.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/OffPeakTaskDispatcher.swift` (Create)
   * **Type / Scope**: dispatcher, queue, and per-task state machine.
   * **Proposed Changes**: FIFO queue, admission gate polling on the
     existing 20 second tick or its own timer, claim-write-before-run,
     power assertion keeper wrapper
     (`ProcessInfo.processInfo.beginActivity` with
     `.idleSystemSleepDisabled` or `IOPMAssertion` via
     `IOPMSleepDisabled`), misfire staleness check, and the tool-facing
     API surface (enqueue, list, cancel) under a job cap.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/CronScheduler.swift` (Modify)
   * **Type / Scope**: scheduler integration.
   * **Proposed Changes**: share the tick with the dispatcher (or hand
     it its own timer); ensure both dispatchers serialize store writes;
     expose the dispatcher to the settings pane the way cron is exposed.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+LocalCommands.swift` (Modify)
   * **Type / Scope**: app-level wiring.
   * **Proposed Changes**: start/stop the dispatcher with the app,
     forward recovery on launch, route its generation requests through
     the same submit path cron uses, including the interrupted-row
     conversion on quit.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/MacAppSettings.swift` (Modify)
   * **Type / Scope**: settings.
   * **Proposed Changes**: add the off-peak enable toggle (default off),
     idle threshold, staleness bound, and job cap fields, persisted with
     existing settings rules.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SettingsPanes.swift` (Modify; basename, settings pane container)
   * **Type / Scope**: UI.
   * **Proposed Changes**: extend the Scheduled Tasks pane with the
     off-peak section: queue list with state badges, enable toggle, and
     per-task cancel/re-run. All strings through the catalog.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Registry/AppToolRegistry+Vocabulary.swift` (Modify; basename)
   * **Type / Scope**: tool registration.
   * **Proposed Changes**: register the off-peak tools; ladder position
     is low risk for list/cancel, medium (side effects, long-running)
     for enqueue, always ask in read-only mode, and follow the existing
     cron tool treatment for mode gating.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/OffPeakTaskDispatcherTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: state machine transitions, claim-write
     persistence and recovery, FIFO order, cap enforcement, staleness
     skip, and assertion keeper lifecycle with a stubbed activity port.
   * Mutation checks: flip the claim-write order and assert recovery
     still distinguishes claimed from queued; flip FIFO and assert order
     test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib` then build the app scheme.
  Run `make swift-lib-app` first if building inside `swift/TurboSparkApp`
  (CSyntext.o shim requirement).
* **Unit Tests**: `swift test` for the new dispatcher tests plus the
  existing Automation tests.
* **Static Analysis and Linting**: `make design-check` for the new pane
  strings and controls.
* **Localization**: the ten parity gates via
  `LocalizationParityTests`, catalog additions for all 21 languages.
* **Manual**: enqueue a task, lock the screen, verify dispatch and
  assertion (system profiler power assertions or Energy pane), quit
  mid-run, verify recovery.

--------------------------------------------------------------------------------

*Drafted with EGM*
