# ZP-23: Concurrent background chats

Finding 2.1 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md) (classified
Different; adopted by author decision 2026-11-11).
Parity targets: ZCode's
[TaskList](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/TaskList.tsx)
and the
[session idle deactivation doc](https://github.com/zai-org/ZCode/blob/main/docs/session-idle-deactivation.md).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Let multiple chats run at once: the user can switch away from a
generating chat, the background chat keeps generating, and its progress
shows through the ZP-02 badge states. The engine serves concurrent
sessions from one resident model through an explicit decode schedule.
This is the prerequisite finding for ZP-02 (badges) and the biggest
single behavioral change in the program; it is adopted deliberately, not
by parity inertia.

## The Problem

TurboSpark is strictly single-active. `selectChat`, `createChat`,
`selectPreviousChat` all guard on generating/submitting/pendingToolCall,
sidebar rows are disabled with "Cannot switch chats while generation is
running", and per-chat queues drain only when that chat is selected and
idle. A long generation parks the whole app: the user cannot ask a quick
question elsewhere, and cron/off-peak dispatch into a non-selected chat
already has to wait. ZCode's core model is many live tasks with a
sidebar that switches freely.

## Goals

* Chat switching is never blocked by generation in another chat. The
  single-active guards become per-chat: a chat blocks only operations
  that touch that chat.
* Background chats keep generating; the sidebar shows per-chat state
  (running spinner, pending pill, unread dot) via ZP-02 states.
* Multiple chats can hold open approvals or questions simultaneously;
  each resolves in place.
* Queues and steering keep their per-chat turn-boundary semantics;
  queues drain at that chat's boundary regardless of selection.
* One resident model serves concurrent sessions; concurrency is capped
  (settings field, default 2) and interplays with the existing
  ResidencyPolicy and expert cache caps.
* Quit generalizes: AppShutdownCoordinator flushes every in-flight chat
  into interrupted rows (existing per-chat interrupted-row flow,
  now applied to all running chats).

## Non-Goals

* No simultaneous different models: concurrent chats share the currently
  loaded model. Switching models while any chat generates is refused
  with a named reason (today's single guard moves to the model level).
* No multi-model batching or speculative cross-session work in the
  engine; decode serializes on the GPU and the design does not pretend
  otherwise.
* No concurrent chats inside one workspace tab (ZP-21 composes on top
  later); this draft is about the chat layer.

## Open Questions

* **Decode lane**: does the FFI support two concurrently active
  generations on one resident model (two sessions stepping
  alternately), or must a Swift-side scheduler interleave steps? Verify
  against `crates/runtime` sessions and `crates/ffi` entry points as the
  first task; if Rust-side serialization exists only at process level,
  add the scheduler in Swift owning one engine lease, round-robin per
  step between active chats, foreground chat prioritized.
* **Cap default**: 2 concurrent generations is the safe default on
  16 GB-class machines (KV per session). Make it settings-visible with a
  hardware-aware upper bound from the existing hardware-fit logic.

## Critical User Journeys

### CUJ 1: switch away mid-generation

* **Given**: chat A is generating a long reply.
* **When**: the user selects chat B (previously blocked).
* **Then**: B opens immediately; A keeps generating; A's row shows the
  running spinner; on completion A shows the unread dot (ZP-02 compare-
  and-set semantics go live here).

### CUJ 2: two chats awaiting approval

* **Given**: chats A and B each have a pending tool approval.
* **When**: the user approves in A, then opens B and approves there.
* **Then**: each approval resolves in place and its chat continues; rows
  return to spinner/unread states independently.

### CUJ 3: model switch with a busy chat

* **Given**: chat A is generating on model M.
* **When**: the user picks model N in the composer.
* **Then**: the switch is refused with the named reason "chat A is
  generating on this model"; the refusal replaces today's blanket
  "cannot switch chats" guard.

### CUJ 4: quit with two running chats

* **Given**: chats A and B are mid-generation.
* **When**: the user quits.
* **Then**: both flush as interrupted rows with Continue/Retry intact;
  no chat's transcript is corrupted by the shared shutdown.

## Technical Plan (ChatGenerationCoordinator)

```text
ChatGenerationCoordinator (new, Swift)
  per-chat turn loop state machine
  (idle, submitting, generating, awaitingTool, awaitingQuestion)
  engine lease: one decode lane, N sessions stepping alternately
  cap: MaxConcurrentGenerations (settings, hardware-aware bound)
       |
       v
AppModel guards become per-chat
  selectChat: no global generating guard
  createChat / deleteChat / model switch: guards check
  "any chat using this model" instead of "the selected chat"
       |
       v
Sidebar (ChatSidebarChatRowView)
  rows enabled; ZP-02 slot states per row
       |
       v
Shutdown (AppModel+Persistence)
  ordered flush per running chat -> interrupted rows
```

The engine lease is the core mechanism: one decode lane owned by the
coordinator, sessions stepping alternately with the foreground chat
prioritized. Subagent runs consume the same lane under the same cap
(subagents already serialize against foreground work today; the cap
makes it explicit).

## Alternatives Considered

### Alternative A: true parallel decode (engine-level batching of sessions)

* **Pros**: real throughput gain.
* **Cons**: engine-level work with memory and quality gates; decode
  serializes on the GPU anyway, so the win is per-step overhead only;
  contradicts the no-unmeasured-gates rule for a UI-driven feature.
* **Reason for Rejection**: Rejected by author: disproportionate;
  revisit only with measured engine support.

### Alternative B: keep single-active, speed up park/resume

* **Pros**: smallest change.
* **Cons**: does not meet the goal; the parked-work problem is the
  feature request.
* **Reason for Rejection**: Rejected by author: the goal is the point.

## Non-Functional Requirements

* **Resource safety**: the concurrency cap and the model-switch refusal
  replace today's blanket guard without new memory risk; KV per session
  is already the engine's accounting unit; expert residency caps
  continue to apply per the runtime rules.
* **Correctness invariants**: sampling owns normalization; no two chats
  share a session; a chat's interrupted-row flush is atomic per chat.
* **Performance**: the foreground chat keeps its decode cadence; the
  schedule is round-robin with foreground priority, so background chats
  progress without stealing the interactive budget.
* **Accessibility and localization**: row states announced per ZP-02;
  all new strings cataloged, 21 languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/ChatGenerationCoordinator.swift` (Create)
   * **Type / Scope**: coordinator and engine lease.
   * **Proposed Changes**: per-chat turn state machines, engine lease
     with alternate stepping and foreground priority, concurrency cap,
     model-switch refusal logic, subagent lane integration.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Chat.swift` (Modify)
   * **Type / Scope**: guards.
   * **Proposed Changes**: replace global single-active guards with
     per-chat guards; model-switch checks "any chat generating on this
     model"; createChat/deleteChat checks the target chat only.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarChatRowView.swift` (Modify)
   * **Type / Scope**: rows.
   * **Proposed Changes**: enable rows during generation; per-row state
     from the coordinator; remove the disabled-row overlay text.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Type / Scope**: shutdown.
   * **Proposed Changes**: per-chat ordered flush for all running chats.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/MacAppSettings.swift` (Modify)
   * **Type / Scope**: settings.
   * **Proposed Changes**: MaxConcurrentGenerations with a
     hardware-aware upper bound.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ChatGenerationCoordinatorTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: per-chat guard matrix (switch/create/delete/
     model-switch with busy and idle chats), lease stepping order and
     foreground priority, cap enforcement, concurrent approvals resolve
     independently, shutdown flushes all running chats to interrupted
     rows, subagent lane sharing.
   * Mutation checks: restore the global guard and assert the
     switch-away test fails; drop the cap and assert the cap test
     fails; let the flush skip a background chat and assert the flush
     test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for coordinator tests plus existing chat,
  queue, and shutdown tests.
* **Localization**: catalog additions in 21 languages.
* **Manual (hardware)**: two chats generating on one resident model,
  switching between them, verifying decode cadence of the foreground
  chat stays interactive and background progress advances; quit mid-run
  and verify both interrupted rows. Report measured tokens-per-second
  as observations only; no performance claim without the bench gates.

--------------------------------------------------------------------------------

*Drafted with EGM*
