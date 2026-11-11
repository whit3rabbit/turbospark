# ZP-04: Subagent-to-coordinator messaging

Finding 3.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[RespondToCoordinator spec](https://github.com/zai-org/ZCode/blob/main/docs/subagent-respond-to-coordinator.md)
and the SendMessage continuation contract from
[subagent listing](https://github.com/zai-org/ZCode/blob/main/docs/subagent-listing-attachment.md).

* Author: whit3rabbit
* Status: Draft
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

## Non-Go: no changes to SubagentRunner depth caps, isolation, or the deny-on-ask gate.

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

*Drafted with EGM*
