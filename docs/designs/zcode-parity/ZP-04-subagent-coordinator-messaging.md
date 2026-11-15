# ZP-04: Subagent-to-coordinator messaging

Finding 3.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[RespondToCoordinator spec](https://github.com/zai-org/ZCode/blob/main/docs/subagent-respond-to-coordinator.md)
and the SendMessage continuation contract from
[subagent listing](https://github.com/zai-org/ZCode/blob/main/docs/subagent-listing-attachment.md).

* Author: whit3rabbit
* Status: Implemented (see Implementation Record for deviations)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Give a running subagent a way to message its coordinator mid-task and
give the coordinator a way to continue a finished subagent with its
context intact. Two tools: `RespondToCoordinator` (child-only) and
`SendMessage` (coordinator-to-child continuation). This resolves the
contradiction between the older kiro brief (which claimed the capability
exists via SubagentRunner) and the newer audit (which searched
`SendMessage`/`RespondToCoordinator` and found nothing).

## The Problem

A TurboSpark subagent is one-shot. Its only output channels are the final
result and a `<task-notification>` on completion. If a long-running child
discovers the task premise is wrong, or finds something the coordinator
needs now, it can only finish early or hold the information to the end.
ZCode's child can message the parent mid-task and keep working, and the
parent merges the message into its next model step; ZCode's coordinator
can also continue a *finished* child with context intact, which TurboSpark
cannot do at all.

## Goals

* A running child can call `RespondToCoordinator` (child-only tool, bound
  to its coordinator) with a message; the child keeps running.
* The coordinator receives the message merged into its next model step as
  a `<subagent-message>`-equivalent carrier so attribution is unambiguous.
* The coordinator can call `SendMessage(to: finished child)` to continue
  it with context intact; the child resumes from its own persisted
  transcript.
* Background children deliver messages even when the coordinator is busy:
  messages queue until the coordinator's next model step.
* The child's catalog exposure is via the agent-definition frontmatter,
  matching how agent files already declare tools.

## Non-Goals

* No changes to SubagentRunner depth caps, isolation, or the
  deny-on-ask gate.
* No cross-session mailbox adoption: that is the separate
  `cross-session-context` spec (session-to-session), which this
  complements rather than replaces.
* No subagent-spawns-subagent changes.

## Open Questions

* **Resume persistence**: continuing a finished child requires its
  transcript to survive child completion. Options: (a) retain the child's
  in-memory context while its parent lives; (b) persist child transcripts
  under the profile (survives app restarts, enables later continuation).
  Option (a) is cheaper; (b) is the durable form. The author decides.
* **Message size bound**: cap per message (suggest 4 KiB) to keep the
  coordinator's next step bounded.

## Critical User Journeys

### CUJ 1: mid-task report

* **Given**: a child running a long sweep finds the target directory is
  empty.
* **When**: it calls `RespondToCoordinator` with "directory empty,
  continuing with adjacent paths" and keeps working.
* **Then**: the coordinator's next model step contains the message with
  child attribution, without ending the child; when the child later
  finishes, its final result still arrives as today.

### CUJ 2: continue a finished child

* **Given**: a child finished and returned a partial mapping.
* **When**: the coordinator calls `SendMessage` with a follow-up
  instruction.
* **Then**: the child resumes with its original context, runs more
  turns, and reports again; the original launch result stays intact in
  the transcript.

## Technical Plan (Architecture)

```text
SubagentRunner
  child session (isolated context, tools per agent file)
    + RespondToCoordinator tool (child-only, closure-bound port)
        |
        v  message
SubagentMessageHub (new)
  per-child mailbox; queues messages until the
  coordinator's next model step boundary
        |
        v
Coordinator prompt assembly
  merge queued messages as <subagent-message name="..."> carriers
  (same injection point as task-notifications today)
        ^
        | SendMessage(to child)
SubagentRunner resume path
  re-enter child loop with persisted/retained context
```

The child-only binding uses the same closure-bound port pattern the
launch flow already uses; the tool never appears in the coordinator's
capture. Resume re-enters the child loop with the retained or persisted
context per the Open Question, under the existing depth and gate rules.

## Alternatives Considered

### Alternative A: interrupt-style wake

* **Pros**: immediate delivery.
* **Cons**: injecting mid-step into a running model call risks corrupt
  in-flight state; the app's steering pills deliberately deliver at turn
  boundaries for the same reason.
* **Reason for Rejection**: Rejected by author: turn-boundary delivery
  matches the existing steering precedent.

### Alternative B: child writes to a file the parent polls

* **Pros**: no protocol work.
* **Cons**: polling adds latency, loses attribution structure, and
  bypasses the permission/hook path the carrier injection reuses.
* **Reason for Rejection**: Rejected by author: the hub keeps delivery
  structured and attributable.

## Non-Functional Requirements

* **Permission and hooks**: the child's tool calls keep passing the
  existing gate; RespondToCoordinator itself is a no-side-effect message
  and lands low on the ladder (allow in all modes except read-only).
* **Bounds**: mailbox depth cap per child (suggest 8) with oldest-drop
  and a logged event, so a looping child cannot balloon the
  coordinator's context.
* **Attribution**: carriers always name the child agent type and launch
  id, so the coordinator cannot mistake child text for user text.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SubagentMessageHub.swift` (Create)
   * **Type / Scope**: mailbox and delivery.
   * **Proposed Changes**: per-child inbound/outbound queues with depth
     caps, delivery at coordinator model-step boundary, attribution
     metadata.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SubagentRunner.swift` (Modify)
   * **Type / Scope**: runner.
   * **Proposed Changes**: bind the RespondToCoordinator port into child
     sessions; add the resume path that re-enters a finished child's
     loop with retained context; keep depth/gate rules unchanged.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Tasks/AgentTools.swift` (Modify)
   * **Type / Spawn-tool schema**: add the `SendMessage` tool to the
     coordinator's capture (schema, executor, ladder entry: medium risk,
     side effects limited to the child session).
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SystemReminders.swift` or the prompt assembly extension (Modify)
   * **Type / Scope**: carrier injection.
   * **Proposed Changes**: merge queued child messages into the
     coordinator's next step as attributed carriers, at the same
     injection point as task-notifications.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/SubagentMessageHubTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: queue/deliver ordering, depth cap
     oldest-drop, attribution, resume retains context, child-only
     binding (the tool is absent from the coordinator capture),
     coordinator SendMessage schema presence.
   * Mutation checks: drop the attribution field and assert the
     attribution test fails; invert the depth-cap drop order and assert
     the cap test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the new hub tests and the existing
  SubagentRunner tests.
* **Manual**: launch a background child, have it message mid-run, verify
  the carrier lands at the next step boundary; finish a child, continue
  it with SendMessage, verify retained context and a second report.

--------------------------------------------------------------------------------

## Implementation Record

Implemented in the Swift app. No new dependency, no auth or crypto change, no
CI or release workflow change, so nothing here is deferred on those grounds.
The EGM review (docs/egm-reviews/zcode-parity/ZP-04/) Blocker and Major items
were resolved as follows.

### Decisions (review B1 to B3, M1 to M9)

* **Resume persistence (B1): in-memory retention.** The runner returns its
  transcript when `priorHistory` is non-nil, and `AppModel` keeps it with the
  background run's record in `retainedBackgroundChildren`. It is pruned with
  the record (20 most recent finished runs, or dismissal) and is lost on app
  restart. Durable persistence (option b) is deferred; a continue against a
  lost context answers "not retained" and tells the model to launch a new
  subagent.
* **Scope (B2, M7): background children only.** `respond_to_coordinator` exists
  only for a background child (task-local `SubagentCoordinatorChannel`, bound
  at launch). A foreground child blocks its coordinator inside one tool call,
  so a message could only wait for the same call to return; it has no channel.
  `send_message` continues a FINISHED background child (id `bga_N`, the same id
  `stop_agent` takes); a running child is refused with "wait for its
  notification". The resumed run is a background run again and returns at once.
* **Delivery (B2, M1, M2): the existing `<task-notification>` queue.** No second
  queue. `SubagentMessageHub` is a value type holding bounded mailboxes; its
  batch is drained by `drainPendingTaskNotificationsIfIdle` (turn tail, chat
  selection, and immediately when the chat is idle, which wakes the
  coordinator like a completion does) and by
  `deliverSubagentMessagesAtBoundary` in `continueOrStop` (next step of a
  running loop). Messages are delivered before a parked completion
  notification. Nothing interrupts a model call in flight (Alternative A stays
  rejected). The hub is not a separate file-level service; the draft's
  "outbound queue" was dropped as unexplained.
* **Carrier (M3, M8): one schema.**
  `<subagent-message agent="NAME" id="bga_N" seq="N" [truncated="true"] [dropped_older="N"]>`,
  preceded by a line stating the text is untrusted data written by the
  subagent and never permission approval. The body goes through
  `escapingWrapperTags` (subagent-message, task-notification, result); attribute
  values are restricted to `[A-Za-z0-9_.-]`, 64 characters.
* **Bounds (M4):** 4 KiB per message (cut on a character boundary with a
  visible ` [truncated]` marker), 8 unread per child (oldest dropped, reported
  to the child in the tool result and to the coordinator via `dropped_older`),
  32 messages per run (then the tool refuses; resuming resets it), 16 KiB per
  `send_message` follow-up (refused, not cut: the coordinator is trusted).
* **Child exposure (M9):** the tool is offered to a background child only when
  the agent definition lists `respond_to_coordinator` in its `tools`
  frontmatter. An absent `tools` list is not an opt-in, because the
  default catalog must not gain a new way to write into the coordinator's
  context. Spelling-insensitive (`RespondToCoordinator` works).
* **Names (M10):** snake_case like `stop_agent`: `send_message` and
  `respond_to_coordinator`, with the CamelCase spellings accepted as aliases.
* **Code citations (M10):** closure-bound port analogue is
  `AppToolRegistry.backgroundAgentLauncher` / `backgroundAgentStopper`;
  injection seam is `AppModel+Subagents.swift` (`completeBackgroundAgent`,
  `drainPendingTaskNotificationsIfIdle`) plus `AppModel+AgentLoop.swift`
  (`continueOrStop`); the steering precedent is `deliverSteersAtBoundary`.
  `SystemReminders.swift` is not involved.

### Permission ladder position (M5)

Categories come from `AppToolCatalog.builtInCategory`; the verdicts are the
existing `AppToolPermissionEngine` ladder, unchanged.

| Tool | Caller | Category | read-only | ask | auto | agentAuto, permissive, fullAccess |
|---|---|---|---|---|---|---|
| `respond_to_coordinator` | background child | `fileRead`, risk `safe` | allow | allow | allow | allow |
| `send_message` | coordinator (depth 0) | `automation`, same rung as `agent` | deny | ask | per automation setting | per mode, same as `agent` |

`respond_to_coordinator` is `fileRead` so a read-only explorer child can report
(precedent: `send_user_file`, `recall_tool_output`); a project that sets
`fileRead` to Deny also silences it. The child's call still passes the
subagent gate unchanged: profile `tools`/`disallowedTools`, PreToolUse and
PostToolUse hooks, `permissionRefusal` (a subagent denies `.ask`). A resumed
child inherits the same agent definition, project, depth, tool filter and
gate it launched with, and counts against the 4-run cap, so `send_message`
grants nothing a fresh launch would not. `send_message` is refused below depth
0 and for ids owned by another chat (reads exactly like an unknown id, as
`stop_agent` does).

### No cross-chat leakage

Every queued message carries the launching chat id; `drain(chatID:)` returns
only that chat's messages; the sender id and chat come from the task-local
channel, never from model-written arguments; the executor refuses a call whose
chat differs from the child's owner and a grandchild that inherited the
task-local (depth must match).

### Deviations from the draft

* No new `MidTurnInputPresentation` case: the carrier rides the existing
  `.taskNotification` presentation, which avoids a new localized label. The
  untrusted-data preface inside the carrier replaces a presentation-level
  refusal guidance.
* The hub is a pure value type owned by `AppModel`, not an actor or a
  delivery service (M1).
* Two tool labels are new UI strings ("Send Message", "Message Coordinator"),
  localized in all 21 locales of `Localizable.xcstrings`.
* A model-side session seam: `launchBackgroundAgent` and `resumeBackgroundAgent`
  accept the runner's `SubagentGenerationPortContext` as a model, so the flow
  is testable without a loaded checkpoint.
* The "logged event" for a dropped message is the `dropped_older` attribute
  plus the child's tool result, not a log line.
* Not verified: a manual run against a real loaded model (Verification Plan,
  Manual). Behavior is covered by tests that drive the real runner loop,
  registry and `AppModel` queue with a scripted generation port.

Tests: `SubagentMessageHubTests.swift` (hub, catalog, ladder, executor) and
`SubagentMessagingFlowTests` in the same file (runner loop, queue, resume).
Mutation-checked: dropping the carrier attribution, inverting the oldest-drop
order, bypassing the carrier escape and removing the frontmatter opt-in each
fail named tests.

--------------------------------------------------------------------------------

*Drafted with EGM*
