# ZP-23: Concurrent background chats

Finding 2.1 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md) (classified
Different; adopted by author decision 2026-11-11).
Parity targets: ZCode's
[TaskList](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/TaskList.tsx)
and the
[session idle deactivation doc](https://github.com/zai-org/ZCode/blob/main/docs/session-idle-deactivation.md).

* Author: whit3rabbit
* Status: Partial. Every item that needs no engine change is implemented
  (see "Second pass"); the remainder is real parallel decode, per-chat KV
  release, releasing the lane during a question, and hardware verification.
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

This section is the original plan. The engine-lease mechanism below was
contingent on the Open Question; the Implementation Record at the end says
which part was built and why.

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

## Implementation Record

Resolution of the "Decode lane" Open Question (verified against source):

* `crates/ffi/src/session.rs` holds one `Engine` per session behind a
  `Mutex`; a turn holds it for its duration. Swift already serializes
  requests on top of that (`QueuedGenerationStream`). Nothing in the engine
  steps two turns alternately, and none was added.
* What cannot run twice is the per-turn state on `AppModel`: one `runTask`,
  one `generationEpoch`, one live stream buffer (`outputText`), one approval
  slot (`pendingToolCall` and six sibling fields), one cancel flag. About 25
  guard sites in 10+ files read them as "the app is busy".
* Decision (the draft's own fallback, "turn-level lease"): one turn at a time
  app-wide. The lane is that turn slot, not the engine. Concurrency means any
  chat can be selected while another runs, other chats park behind the lane,
  and a chat that needs the user releases the lane. No Rust or FFI change.

What shipped:

* `State/ChatGenerationCoordinator.swift`: pure scheduler (lane holder,
  foreground-first admission order, parked states) and the guard verdicts for
  select, create, delete and model switch.
* `State/AppModel+ConcurrentChats.swift`: the AppModel side. `generatingChatID`
  names the chat that owns the running turn; `selectChat`, `createChat` and
  `deleteChat` use the verdict instead of the blanket guard; the live stream
  buffer is scoped to the running turn's chat (`isRunningInSelectedChat`,
  `liveOutputBelongsToSelectedChat`); sidebar rows stay enabled and show
  generating, needs approval or queued.
* Background tool deferral and admission (below); quit flush of the running
  turn into its own chat plus every parked proposal.
* The model-switch refusal names the chat ("<chat> is generating on this
  model. Stop it before switching models."). It is a toast on the existing
  load refusal, not a new gate.

Review fixes applied (docs/egm-reviews/zcode-parity/ZP-23):

* B1, B4: the Swift lease is no longer assumed. The spike was a source read
  (above); the fallback is the turn-level lease. Rejected alternatives are in
  the next section.
* M2: one policy, strict foreground priority. The foreground chat's parked
  work is admitted first, then other chats in the order they parked. There is
  no round robin because there is no interleaving: a turn runs to its end.
* M3: constant cap of 1 turn lane (`ChatGenerationCoordinator.laneCapacity`).
  A request at the cap queues, it does not refuse. No settings field and no
  hardware-aware bound in v1: a second concurrent turn is not offered, so
  there is no KV or expert-cache figure to bound.
* M4, ladder position: a tool call proposed by a chat that is not selected
  does not run. The whole proposal parks (`DeferredToolProposal`) and
  replays through `handleExtractedToolCalls` when the chat is selected and
  the lane is free. Hooks, the permission engine, Agent mode and the approval
  card therefore evaluate in the foreground exactly as before; nothing is
  auto-approved in the background and no new approval variant exists. An
  awaiting-tool chat holds no lane (the flag drops when the proposal parks).
* M11, slot accounting: one slot per top-level chat. Subagents and
  background agents run inside their parent chat's slot and already queue on
  the engine's serial stream. Idle KV retention is unchanged: the session is
  shared and KV belongs to it, so a switched-away chat holds nothing extra.
  Per-chat KV release (ZCode idle deactivation) is deferred, see Remainder.
* M5: the interrupted-row flow is reused. `shutdown()` flushes the running
  turn's buffer into `generatingChatID` (it used `selectedChatID`, which is
  wrong once the two can differ) and
  `flushDeferredProposalsForShutdown()` writes each parked proposal's captured
  prose as an interrupted row with a recovery anchor. Crash reconciliation is
  ZP-17's marker, unchanged and not extended per chat.
* M6: there is no `ResidencyPolicy` type in Swift and no change to expert
  caps. The claim was removed from this draft's scope.
* M7: CUJ 3 is the existing load refusal, now naming the chat. Stop-then-
  switch is the way forward and stays a follow-up.

Alternatives considered (replaces the "by author" rejections):

* Rust-side scheduler stepping two sessions: needs per-session cancel and
  KV accounting in `crates/runtime` plus real-hardware memory and quality
  gates this change cannot run. Not built.
* One engine per chat: doubles resident weights unless shared, and sharing is
  the engine change above. Not built.
* KV swap or evict per background chat: the idle-deactivation analogue; an
  engine feature with a restore-correctness gate. Deferred.
* Swift scheduler over the existing serial stream with per-chat turn state:
  the right long-term shape, but it is the M1 refactor (about 25 guard sites,
  the approval slot, the cancel flag, the stream buffer). Not done in one
  pass without being able to validate on hardware.
* Turn-level lease (chosen): safe with the existing serial worker, no engine
  change, and it delivers the user-visible goal that switching is never
  blocked by another chat's generation.

Risks and mitigations (hypotheses, none measured):

* Background turn finishes while the user is elsewhere: its reply commits to
  its own chat (every append is keyed by chat id, state#17), and the row shows
  the queued or needs-approval glyph. Unread dots are ZP-02, not built here.
* A tool proposal waits indefinitely for a chat nobody opens: bounded by the
  user's own switching; the row says "Needs Approval". No timeout was added.
* Lease leak: the app-wide `generating` flag is the source of truth. When it
  drops, the holder is released unconditionally; a release by a non-holder is
  ignored so a stale tail cannot free a newer turn's lane.
* Approval spoofing and same-workspace races: no background tool runs, and
  only one turn runs, so the existing single-turn guarantees hold.
* Partial flush on quit: the running turn and every parked proposal flush in
  `shutdown()` before the final persist; a missed chat degrades to the
  existing relaunch behavior (user prompt with no reply, Retry available).

Dependencies: ZP-02 (unread and pending badges) is not implemented; this
change supplies the per-row state it needs (`chatTurnState(for:)`). ZP-21
(workspace tabs) composes later. ZP-01 (off-peak tasks) now dispatches into
a background chat: `offPeakCanDispatch` no longer requires selection.

Remainder after the first pass (kept for the record; the second pass below
resolves items 1 to 4):

1. Two chats awaiting approval at once (CUJ 2).
2. A question for the user (`AskUserQuestion`) holds the lane while it waits.
3. A typed prompt parked in a background chat waits until selected.
4. Cross-project chat switching and project creation keep the blanket guard.
5. Cmd+. and the menu Stop act on the running turn wherever the user is.
6. Real two-turn concurrency, per-chat KV release, and any performance claim.

### Second pass: approvals, questions, background prompts, projects

What shipped (all Swift, no Rust or FFI change, no dependency):

* **Two approvals (CUJ 2).** The approval slot (`pendingToolCall` and its six
  siblings) now describes the selected chat only. Leaving a chat parks its
  card in `AppModel.parkedApprovals` (keyed by chat, `ParkedApproval`);
  selecting the chat moves it back, and the turn tail's `admitParkedWork`
  does the same for the foreground chat once the lane frees. The card is
  never answerable while parked, because approve and deny read the slot and
  match its call id: chat A's approval cannot be applied to chat B. The
  coordinator verdict no longer pins the user to a chat with a card; only an
  approval with no chat of its own (legacy) still does. `createChat`,
  `selectChat` and `openChat` park; deleting or clearing a chat drops its
  parked card; the sidebar reads a parked card as `awaitingTool`.
* **Replay re-validation.** Restoring a parked card re-runs
  `AppToolPermissionEngine` against the LIVE project (looked up by id, not
  the snapshot). A refusal denies the call with the engine's reason and says
  so to the model and hooks (`denyPendingToolCall(policyReason:)`), not "the
  user said no". A project deleted in the meantime denies too. Anything else
  shows the card: a policy that was relaxed never auto-approves. A parked
  `DeferredToolProposal` now replays against the live project as well (it
  used the proposal-time snapshot before), and is dropped as an interrupted
  row when its project is gone. PreToolUse hooks are NOT re-run for a parked
  card (they ran when it was proposed and the card is the human decision);
  a deferred proposal replays through the full ladder, hooks included.
* **AskUserQuestion.** The lane stays held while the card is open (releasing
  it needs the turn and the tool task decoupled, which is the engine-adjacent
  refactor this draft defers). What changed is correctness of the hold: the
  coordinator records `awaitingQuestion` for the holder and clears it on
  answer, stop or turn end; `chatTurnState` now ranks a question above
  `generating` (it never surfaced before, because the asking chat is the
  running chat); nothing else starts beside it; the answer routes by the
  question's own chat key wherever the user is looking.
* **Background typed prompts.** A typed prompt parked in a chat the user is
  not looking at starts when the lane frees, without the composer
  (`startBackgroundTypedPrompt`): the steer assembly (mentions, attachments,
  `UserPromptSubmit`, sanitization) runs for the first entry only, under
  `submitting` and `submissionTask` so Stop reaches it, then the turn starts
  in the chat's own id. Slash, bang and `#` entries still wait for selection
  (`run()` dispatches them off the composer). A hook block re-parks the entry
  and shows its error, as for a mid-turn steer.
* **Cross-project.** `openChat` to another project parks the selected chat's
  card first (checking the move is legal before parking, so a refused move
  never strands a card). Parked work replays only in its chat's own
  workspace: `alignWorkspace` makes the project selection match, and while a
  turn runs, where a project switch is still refused, the work stays parked
  with a toast on the explicit path. A card keeps the project it was
  evaluated against and is refreshed to the live one.
* Two strings added in 21 languages; tests in
  `ConcurrentApprovalsTests.swift` and an updated
  `ChatGenerationCoordinatorTests.swift`.

Still deferred, with the reason:

1. **Real parallel decode, per-chat KV release, any performance claim.**
   Engine work: `crates/ffi/src/session.rs` holds one `Engine` behind a
   mutex and nothing steps two turns. Lane capacity stays 1. Needs Rust
   changes plus the real-hardware memory and quality gates. Hardware
   verification (two chats on one resident model, switching mid-run, quit
   mid-run) was not run.
2. **Releasing the lane during an AskUserQuestion.** Needs the turn and the
   tool task decoupled so a second turn can start while the first is
   suspended and its tail still runs against the right epoch. About 25 guard
   sites plus a reclaim step on answer. The risk is not testable without a
   loaded model. A decision for the author: accept the hold (current), or
   fund the refactor.
3. **Background task-notification injection** (`canInjectTaskNotification`,
   pinned by `BackgroundAgentTests`) still requires its chat to be selected,
   by design: the notification wording reads the composer-empty rule.
4. **Project-level operations** (`selectProject`, `createProject`,
   `deleteProject`, root change) keep their blanket guard, including the
   approval slot term: they rebind the worktree, snapshot store and hook
   store under whatever runs. `duplicateChat`, message editing and
   split-pane moves keep theirs: they touch selection-bound state.
5. **Stop acts on the running turn wherever the user is** (Cmd+. and the
   menu). Kept: a stop that depended on selection would be harder to trust.
   Parked cards are not stopped by it.
6. **PreToolUse hooks on a parked card** are not re-run at restore (see
   above); re-running them needs the proposal's pre-hook arguments, which the
   transcript row no longer holds.

--------------------------------------------------------------------------------

*Drafted with EGM*
