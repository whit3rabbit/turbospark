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

Give chat sidebar rows a state machine: an unread dot with compare-and-set
read semantics, a running spinner slot, and a pending-interaction pill
(permission or question) with a countdown fill and click-to-snooze. This
is the TurboSpark port of ZCode's task-row badge system, sized for an app
moving to concurrent background chats (ZP-23).

## The Problem

TurboSpark's sidebar has no unread concept. Searching `unread` across app
sources hits only MCP/background-shell plumbing; the only row status cue
is a progress flame icon on the selected, running chat. Pending tool
approvals surface only as in-transcript cards, so a user who has looked
away has no cue that a chat needs them, and after returning there is no
way to tell which rows changed since last seen. The dock badge half of
this gap is already in the `desktop-shell-hardening` spec; this draft owns
the sidebar row state machine and the pending pill.

## Goals

* Each sidebar row shows exactly one slot state, in priority order:
  pending-interaction pill, unread dot, running spinner, default icon.
* Unread state is persisted per chat with compare-and-set read semantics:
  marking read only clears the flag when the observed revision matches,
  so a concurrently writing chat re-raises it.
* The pending pill appears when a chat has an open approval card or open
  question, with a countdown fill driven by an auto-resolve window.
* Click-to-snooze extends the window by one interval, once per question.
* Pending and unread states feed the dock badge (existing spec) and any
  future concurrent-chats world (finding 2.1).

## Non-Goals

* No paired-device read semantics (excluded category; see ZP-90,
  finding 1.4).
* Dependency note: concurrent background chats are now adopted
  ([ZP-23](ZP-23-concurrent-chats.md), author decision 2026-11-11), so
  this draft's compare-and-set semantics and pill states become live on
  real background generation rather than remaining future-proofing.

## Open Questions

* **Auto-resolve window**: a countdown implies an auto-resolution
  semantic the app lacks today. Options: (a) display-only countdown;
  (b) auto-resolve with an empty answers payload like ZCode; (c) expiry
  re-parks the prompt into the queue. Option (b) with a global toggle
  defaulting off is the most defensible, but the choice is the author's.
* **Unread definition**: assistant replied while unselected, or any
  content change while unselected. The first matches chat conventions;
  the second also catches tool-only activity. Pick one at design review.

## Critical User Journeys

### CUJ 1: unread dot lifecycle

* **Given**: chat B is not selected.
* **When**: generation appends content to chat B.
* **Then**: B's row shows the unread dot. Selecting B marks read via
  compare-and-set against B's revision token. If content lands again
  before the read mark lands, the dot re-raises. The dot never shows on
  the selected row.

### CUJ 2: pending pill lifecycle

* **Given**: a tool call in chat B needs approval, or a question is open.
* **When**: the card or question appears while B is not selected.
* **Then**: B's row shows the pending pill with countdown fill. Click
  navigates to B; with concurrent chats (ZP-23) navigation no longer
  stops anything. Snooze extends once. The countdown end resolves per
  the Open Question decision.

### CUJ 3: running spinner slot

* **Given**: generation is running in the selected chat.
* **When**: the sidebar re-renders.
* **Then**: the selected row's icon becomes the spinner slot, which
  yields to pending and unread in priority.

## Technical Plan (Architecture)

The row view is `ChatSidebarChatRowView` and its state flows from the
sidebar organizer. The design adds a per-chat badge state computed from
existing signals plus one persisted field:

```text
ChatBadgeState (per chat, computed)
  priority: pending > unread > running > default
  pending: open approval or open question exists for that chat
  unread: persisted unread flag, cleared by CAS markRead
  running: chat is generating
ChatSidebarChatRowView: renders the slot by badge state
MacAppSettings: window length, snooze extension, auto-resolve toggle
```

Compare-and-set read semantics: `markRead(chatID, observedRevision)`
clears the flag only when the stored revision equals the observed one,
so a writer that landed content between observe and mark re-raises the
flag. Today only the selected chat generates, so the CAS path matters at
chat-switch boundaries; the semantics exist for the concurrent future.

## Alternatives Considered

### Alternative A: in-memory-only unread flags

* **Pros**: trivial.
* **Cons**: unread state lost on relaunch, defeating the purpose across
  generation sessions.
* **Reason for Rejection**: Rejected by author: persistence is the point.

### Alternative B: dock badge only

* **Pros**: already specced in desktop-shell-hardening.
* **Cons**: says "something happened" without saying where; the sidebar
  is the navigation surface and gets no cue.
* **Reason for Rejection**: Rejected by author: complements, not replaces.

## Non-Functional Requirements

* **Accessibility**: badge states are never color alone. The dot and pill
  carry accessibility labels and join the combined row label; strings are
  localized for all 21 offered languages.
* **Performance**: badge state is computed from existing per-chat data
  plus one persisted field, no polling; store writes only on transitions.
* **Design compliance**: slot metrics and status colors follow DESIGN.md
  and the token ratchet; `make design-check` applies.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/ChatBadgeState.swift` (Create)
   * **Type / Scope**: badge computation and persistence model.
   * **Proposed Changes**: per-chat computed badge state with priority
     resolution; persisted unread flag with compare-and-set
     `markRead(chatID, observedRevision)`; revision token wired to the
     chat archive revision the organizer already reads.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/ChatSidebarOrganizer.swift` (Modify)
   * **Type / Scope**: sidebar model.
   * **Proposed Changes**: compute and expose `ChatBadgeState` per row;
     observe pending tool and question state from the app model; sort
     unchanged (badges never reorder rows).
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarChatRowView.swift` (Modify)
   * **Type / Scope**: row UI.
   * **Proposed Changes**: replace `rowIcon` with the priority slot:
     pending pill (countdown fill, snooze action), unread dot, running
     spinner, default bubble. Countdown rendering follows Reduce Motion
     rules.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/MacAppSettings.swift` (Modify)
   * **Type / Scope**: settings.
   * **Proposed Changes**: auto-resolve toggle (default off), window
     length, snooze extension fields, persisted with existing rules.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Questions.swift` (Modify)
   * **Type / Scope**: question lifecycle.
   * **Proposed Changes**: optional auto-resolution on window expiry per
     the Open Question decision; emits the same dismissal path as the
     user declining, so transcript, hooks, and task-notifications see
     one story.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ChatBadgeStateTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: priority resolution; CAS markRead success,
     revision-mismatch re-raise, and never-on-selected; pending pill
     presence for approval and question; snooze-once semantics;
     persistence round-trip.
   * Mutation checks: invert CAS comparison, assert mismatch re-raise
     test fails; invert slot priority, assert priority test fails.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/LocalizationParityTests.swift` (Unchanged, must pass)
   * **Type / Scope**: localization gates.
   * **Proposed Changes**: no edit; every new string enters the catalog
     for all 21 languages in the same change, keeping all ten gates
     green.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme
  (run `make swift-lib-app` first when building inside
  `swift/TurboSparkApp`).
* **Unit Tests**: `swift test` with the new badge tests.
* **Static Analysis and Linting**: `make design-check` plus
  `make design-check-full` for the new status-adjacent styling.
* **Localization**: all ten parity gates via `LocalizationParityTests`.
* **Manual**: generate a reply into a non-selected chat (enabled by
  ZP-23), verify the dot; open an approval card, verify the pill and
  snooze.

--------------------------------------------------------------------------------

*Drafted with EGM*
