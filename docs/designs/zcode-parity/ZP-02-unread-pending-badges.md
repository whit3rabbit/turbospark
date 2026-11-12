# ZP-02: Unread and pending-interaction badges on chat rows

Finding 2.3 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[TaskInteractionBadge](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/TaskInteractionBadge.tsx)
and
[TaskListItem](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/TaskListItem.tsx).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Give chat sidebar rows one priority slot: a pending-interaction pill
(static, no timer), an unread dot, a running spinner, or the default
icon. This is the TurboSpark port of ZCode's task-row badges, for
concurrent background chats (ZP-23).

## The Problem

TurboSpark's sidebar has no unread concept. Searching `unread` across
`swift/TurboSparkApp/Sources` hits only MCP/background-shell plumbing;
the only row status cue is the progress flame in
`ChatSidebarChatRowView.rowIcon`, shown when `isSelected &&
model.isRunning`. Pending tool
approvals surface only as in-transcript cards, so a user who has looked
away has no cue that a chat needs them, and after returning there is no
way to tell which rows changed since last seen. The dock badge half of
this gap is already in the `desktop-shell-hardening` spec; this draft owns
the sidebar row slot and the pending pill.

## Goals

* Each sidebar row shows exactly one slot state, in priority order:
  pending-interaction pill, unread dot, running spinner, default icon.
* Unread is persisted per chat as a last-read marker. A chat is unread
  when it has assistant content newer than the marker and is not
  selected.
* Unread means an assistant reply completed while the chat was
  unselected. Tool-only activity does not count. (Author confirms the
  choice [confirm].)
* A chat is 'running' when it is generating, selected or not (per
  ZP-23). Unread is set only when a reply completes, so a streaming
  background chat shows the spinner and then the dot.
* The pending pill appears while a chat has an open approval or open
  question, and disappears when it is answered or dismissed. It never
  answers or dismisses anything itself.
* Pending and unread states feed the dock badge (existing spec).

## Non-Goals

* No paired-device read semantics (excluded category; see ZP-90,
  finding 1.4).
* No error badge. A failed turn keeps its current in-transcript
  treatment. (Whether to add an error state is an author decision
  [confirm].)
* No sidebar reordering by badge state.
* No countdown, snooze, or auto-resolution of approvals or questions.
  Resolving a prompt without a user answer is a permission decision and
  needs its own design.
* No dock badge work here (pending the author's answer on the
  `desktop-shell-hardening` spec).

## Dependencies

ZP-23 (concurrent chats, [ZP-23](ZP-23-concurrent-chats.md)) lands first,
per INDEX.md. Before ZP-23, only the selected chat generates, so the
unread dot and pending pill can appear only after a chat switch with a
prompt open; the feature ships correct but mostly idle. If ZP-23 slips,
ZP-02 may still ship in that reduced form [confirm: author decision].

## Critical User Journeys

### CUJ 1: unread dot lifecycle

* **Given**: chat B is not selected.
* **When**: a reply in chat B completes while B is not selected.
* **Then**: B's row shows the unread dot. Selecting B advances B's
  last-read marker and clears the dot. If another reply completes after
  that, the dot returns. The dot never shows on the selected row.

### CUJ 2: pending pill lifecycle

* **Given**: a tool call in chat B needs approval, or a question is open.
* **When**: the card or question appears while B is not selected.
* **Then**: B's row shows the pending pill. Click navigates to B. The
  pill clears when the prompt is answered or dismissed.

### CUJ 3: running spinner slot

* **Given**: generation is running in any chat.
* **When**: the sidebar re-renders.
* **Then**: the selected row's icon becomes the spinner slot, which
  yields to pending and unread in priority.

## Proposed Architecture

The row view is `ChatSidebarChatRowView` and its state flows from the
sidebar organizer. The design adds a per-chat badge state computed from
existing signals plus one persisted field:

```text
slot(for chat) (pure function in the organizer)
  priority: pending > unread > running > default
  pending: open approval or open question exists for that chat
  unread: assistant content past the persisted lastReadAt marker, not selected
  running: chat is generating
ChatSidebarChatRowView: renders the slot
```

Selecting a chat, or content landing in the selected chat, advances the
marker. Unread is derived: content past the marker while not selected.
No revision token or compare step is needed.

## Alternatives Considered

### Alternative A: compare-and-set read with a revision token

* **Pros**: safe with several writers racing on one chat.
* **Cons**: chats have no revision field today (`AppChat` carries
  only `updatedAt`), so this adds a counter, a migration, and a
  token plumbed through every append path. The app has one writer
  per chat [confirm].
* **Reason for Rejection**: cost exceeds benefit. A last-read
  watermark (see Proposed Architecture) gives the same re-raise behavior
  when content lands after the mark.

### Alternative B: countdown pill with snooze and auto-resolve

* **Pros**: matches ZCode's TaskInteractionBadge.
* **Cons**: needs a policy for resolving an approval or question
  with no user answer. That policy is a permission decision, not a
  badge decision.
* **Reason for Rejection**: out of scope here. Ship a static pill;
  auto-resolve, if wanted, gets its own design. (Text invented from the
  review findings [confirm].)

### Alternative C: in-memory unread flags

* **Pros**: no persistence or migration.
* **Cons**: all unread state is lost on app relaunch.
* **Reason for Rejection**: the cue is only useful if it survives
  relaunch.

## Non-Functional Requirements

* **Accessibility**: badge states are never color alone: the dot is a
  filled circle, the pill carries text, and each has an accessibility
  label that joins the combined row label. Strings are localized for all
  21 offered languages (`LocalizationParityTests`).
* **Performance**: badge state is derived from existing per-chat data
  plus one persisted last-read marker; there is no timer or polling, and
  the store writes only when the marker changes.
* **Design compliance**: slot metrics and status colors follow DESIGN.md
  and the token ratchet; `make design-check` applies.
* **Risks**: (1) A chat saved before this change has no marker; it
  decodes as read, so nothing lights up after upgrade. (2) After a
  long offline gap or a profile restore, many rows can show unread
  at once; the dot is the only cue, so the sidebar stays calm. (Risk 2
  is inferred from the review findings [confirm].)

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppChat.swift` (Modify)
   * **Type / Scope**: persisted chat model.
   * **Proposed Changes**: add optional `lastReadAt: Date?`. A missing
     value decodes to 'read' for chats that predate the field, so no chat
     lights up after upgrade. Ghost chats (`isGhost`) never persist it
     (they are filtered out of the archive).
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/ChatSidebarOrganizer.swift` (Modify)
   * **Type / Scope**: sidebar model.
   * **Proposed Changes**: expose a pure `slot(for chat)` function with
     priority pending > unread > running > default. A chat has a pending
     prompt when `AppModel.pendingToolCallChatID == chat.id` and
     `pendingToolCall != nil`, or `pendingUserQuestions?.chatID ==
     chat.id`. The pill shows for these on unselected rows only; on the
     selected row the in-transcript card is the cue. A question with a
     nil `chatID` shows no pill. (Selected-row scope is an author
     decision [confirm].) Sort unchanged (badges never reorder rows).
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarChatRowView.swift` (Modify)
   * **Type / Scope**: row UI.
   * **Proposed Changes**: replace `rowIcon` with the priority slot:
     static pending pill, unread dot, running spinner, default bubble.
4. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ChatBadgeStateTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: slot priority; unread set on reply completion
     while unselected; no unread while selected; decode of a chat with no
     marker is read; persistence round-trip; pending pill for approval
     and question; spinner on a background running chat. Mutation check
     (a new guard is only trusted if breaking it fails a test): invert
     the marker comparison and assert the re-raise test fails; invert
     slot priority and assert the priority test fails. Rename the file
     to `ChatSidebarOrganizerBadgeTests.swift` or keep the name; author
     picks.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme
  (run `make swift-lib-app` first when building inside
  `swift/TurboSparkApp`).
* **Unit Tests**: `swift test` with the new badge tests.
* **Static Analysis and Linting**: `make design-check` plus
  `make design-check-full`, because the dot and pill add status colors
  covered by the contrast tests.
* **Localization**: every new string enters the catalog for all 21
  languages in the same change; all ten parity gates in
  `LocalizationParityTests` must stay green.
* **Manual**: generate a reply into a non-selected chat (enabled by
  ZP-23), verify the dot; open an approval card, verify the pill.

--------------------------------------------------------------------------------

*Drafted with EGM*
