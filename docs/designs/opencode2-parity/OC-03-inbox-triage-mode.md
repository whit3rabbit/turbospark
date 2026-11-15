# OC-03: Inbox triage mode in the chat sidebar

Finding: the inbox half of Section 9a, "Session unread and
needs-attention state, with an inbox triage mode" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[inbox model](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/inbox-model.ts)
and
[inbox view](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/inbox.tsx).

* Author: whit3rabbit
* Status: Implemented (see "Implementation Record")
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

A second grouping mode for the chat sidebar that sorts chats by what
they need from the user instead of by time: Needs you (pending approval
or question), Working, Finished (unread results), and Archived. The
per-chat states themselves are owned by
[ZP-02](../zcode-parity/ZP-02-unread-pending-badges.md) (unread dot,
pending pill, running spinner) and concurrent chats by
[ZP-23](../zcode-parity/ZP-23-concurrent-chats.md); this draft is the
triage view over those states so it works on the states that exist
today and gains live rows when they land.

## The Problem

The sidebar groups chats into time buckets
([ChatSidebarView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarView.swift)),
which answers "what did I do recently" but not "what needs me". With
concurrent chats (ZP-23 adopted) and unattended turns, a user returning
to the app must hunt across rows for the chat that is waiting on an
approval or a question. OpenCode's inbox mode groups sessions into
needs-input, working, finished, and archived sections with archive and
delete flows, giving one glanceable triage pass.

## Goals

* A sidebar mode toggle (Sessions / Inbox) using the vocabulary pill
  picker inside the sidebar header; the choice persists.
* Inbox sections in priority order: Needs you, Working, Finished,
  Archived. A chat appears in exactly one section, resolved by the same
  priority order as ZP-02's row slots.
* Needs you = open approval card or open question (derivable today from
  parked turns, no ZP-02 dependency).
* Working = generating or queued turn. Finished = last turn ended and
  not yet viewed (unread when ZP-02's persisted read state exists;
  "ended since you last opened the chat" as the pre-ZP-02 approximation).
  Archived = the existing archived flag.
* Row actions per section: open (row click), archive and delete in the
  context menu with the existing confirmation dialogs for destructive
  actions.
* Section rows are keyboard-navigable with the sidebar's existing
  patterns and VoiceOver-readable as groups with counts.

## Non-Goals

* No new persistence in this draft: the inbox derives everything from
  existing chat state plus ZP-02's fields when they land.
* No drag reorder, pinning, or custom sections.
* No notification behavior changes (owned by
  [OC-02](OC-02-notifications-and-sounds.md)).

## Open Questions

* **Default mode**: Inbox on first launch after concurrent chats land,
  or Sessions always default with Inbox opt-in. Lean: Sessions stays
  default; the mode is opt-in until ZP-23 makes Working rows common.
* **Finished section semantics before ZP-02**: show every ended chat
  since launch, or only those ended while the app was open and unfocused.
  Lean: the latter; it keeps the section meaningful without inventing a
  read state that ZP-02 will own.

## Critical User Journeys

### CUJ 1: triage after returning

* **Given**: three concurrent chats: one parked on a question, one
  generating, one finished unread; the user was away.
* **When**: the user opens the sidebar in Inbox mode.
* **Then**: the three chats appear once each under Needs you, Working,
  and Finished; everything else is under their normal sections or
  absent.

### CUJ 2: acting on a row

* **Given**: a chat under Needs you with an open approval.
* **When**: the user opens it and approves.
* **When** the approval resolves, **then** the row leaves Needs you and
  appears under Working or Finished without a manual refresh.

## Technical Plan (Architecture)

```text
ChatSidebarView (mode: .sessions | .inbox, persisted)
        |
        v
ChatInboxModel (derived, @MainActor)
  reads AppChat state + (when present) ZP-02 unread/pending fields
  computes sections: needsYou / working / finished / archived
  invalidates on the same signals the row states use
        |
        v
Sidebar sections render with existing row views; section headers
carry counts; context menu actions reuse existing archive/delete paths
```

The model is a pure derivation over inputs, so section membership is
unit-testable without views. No fifth chrome band and no new surface:
this is the existing sidebar with a different grouping function.

## Non-Functional Requirements

* **Performance**: derivation over the chat list is O(n) per
  invalidation, n = chats; identical to the existing grouping cost.
* **Localization**: mode labels, section titles, and empty states ship
  in 21 languages in the same change.
* **Accessibility**: sections expose `.isHeader` traits and counts in
  the accessibility value; rows keep the existing combined-element
  contract.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatInboxModel.swift` (Create)
   * **Rationale**: the derived grouping model with the priority
     resolution and invalidation inputs. ZP-02 fields are read through a
     small protocol so the draft works before and after ZP-02 lands.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSidebarView.swift` (Modify)
   * **Rationale**: the mode toggle, section rendering, per-section
     context menus, empty states.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Chat.swift` (Modify)
   * **Rationale**: expose the derivation inputs (parked question or
     approval, turn-active flag, last-ended timestamp) if they are not
     already computed properties.
4. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ChatInboxModelTests.swift` (Create)
   * **Rationale**: membership matrix (one section per chat, priority
     order), empty states, ZP-02 protocol both with and without the
     fields. Mutation check: swap Working and Finished priority and
     assert the ordering test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus localization gates and
  `make design-check`.
* **Manual**: park a chat on a question, run another, finish a third;
  verify sections in light/dark, German, Extra Large text, keyboard-only
  navigation, and VoiceOver per DESIGN.md section 9.

## Implementation Record

Built on ZP-02 (`ChatRowSlot`, `AppChat.hasUnreadReply`) and ZP-23
(`chatTurnState`), both present in the tree. Review blockers were resolved
as follows (the review report is in
`docs/egm-reviews/opencode2-parity/OC-03/`).

What shipped:

* `Generation/ChatInboxModel.swift`: `ChatSidebarMode` (chats, inbox),
  `ChatInboxSection.resolve` (pure, one section per chat) and
  `ChatInboxLayout.make` (pure, over the already scoped and sorted list).
  No protocol and no pre-ZP-02 shim: the inbox reads `ChatTurnState` and
  `hasUnreadReply` directly (review M1, M2, M5, M13).
* `Generation/ChatSidebarInboxRows.swift`: section headers (`.isHeader`,
  title as label, count as the accessibility value), rows reuse
  `ChatSidebarChatRowView`, so archive and delete keep their existing
  guards and confirmation. Empty state with a "Show all chats" action.
* `ChatSidebarView`: a Chats / Inbox `AppPillPicker` inside an
  `AppAdaptiveHeader`, shown on the Chat tab only (M11). Filter text
  applies in both modes through `ChatSidebarLayout.matches`.
* `MacAppSettings.chatSidebarMode` (tolerant decode, default "chats") and
  `AppModel.chatSidebarMode`: the persistence the Non-Goals line wrongly
  excluded (M4). No Settings row, so `SettingsControlCatalog` is untouched.
* `AppModel.inboxRetainedChatID`, set in `selectionChangedForReadState`
  before the read markers move (M9).
* 8 catalog keys in 21 languages; `ChatInboxModelTests` (14), mutation-checked (swapped Working and Finished order, and retention ignoring selection, both fail tests).

Open Questions, resolved:

* Default mode: Chats stays the default; Inbox is opt-in (the draft's lean).
* Finished before ZP-02: moot. ZP-02 shipped, so Finished is "has an unread
  reply" and no "ended since launch" approximation exists (M5).

Review findings, how each was settled:

* B1 and M12 (priority): the section order is the slot order, pending then
  running then unread. Two stated differences: a parked prompt on the
  selected chat still counts as Needs you (the slot hides the pill there,
  but the chat is still waiting and the row must not vanish under the
  card), and a queued chat counts as Working. A queued row for an
  unselected chat does not drain until it is selected (ZP-23 admission), so
  it can sit in Working; the row's own clock glyph says why.
* M3 (idle chats): Inbox shows only chats that need attention. Everything
  else stays in Chats; the empty state offers "Show all chats".
* M6 (Archived): the archived flag wins. `filteredChats` already excludes
  archived chats, and the existing Archived disclosure renders under both
  modes, with its Restore and Delete.
* M7: the filter and the project scope apply; ghost chats are excluded by
  `filteredChats` and by the existing transcript check.
* M8: derived per body from state the rows already observe; no extra
  invalidation signal was added, and the "identical cost" claim is dropped.
  It is O(n) with one `chatTurnState` lookup per chat.
* M9: opening a Finished row keeps it there while selected. An approval that
  resolves moves the row (CUJ 2).
* M10: CUJ 2 reads: Then the row leaves Needs you, and shows under Working if
  the turn resumes, or under Finished if it ends with an unread reply.
* Order inside a section is `AppChat.sortedForSidebar` (pinned, then
  recency). No mark-all-read.
* B2, B3 (Alternatives, Risks): covered here in short form. Rejected:
  pinned attention bands above the time buckets with no toggle (changes the
  default list for everyone; the toggle keeps it opt-in). Risks: row
  jumping (mitigated by retention), a queued row that does not drain
  (above), delete from a new surface (reuses the row's guards and
  confirmation), state drift against ZP-02 (none, same inputs).

Deviations:

* Mode label is "Chats", not "Sessions": the sidebar calls them chats
  everywhere else (review SY-15).
* No new keyboard shortcut. The picker is a focusable control with arrow-key
  movement, so the OC-01 catalog and the OC-05 chords are untouched.
  A shortcut can be added through `KeyboardShortcutCatalog` later.
* VoiceOver does not announce a row moving between sections.

Verification: focused app tests (ChatInboxModelTests, ChatBadgeStateTests, LocalizationParityTests, SidebarAndProjectGroupingTests: 43 pass), make compile-strings, make design-check, generate-settings-catalog.py --check. Manual
light/dark, German, Extra Large text and VoiceOver passes were not run, nor was a full app build via make swift-app.

--------------------------------------------------------------------------------

*Drafted with EGM*
