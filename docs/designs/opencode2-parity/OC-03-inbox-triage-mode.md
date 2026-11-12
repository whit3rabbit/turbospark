# OC-03: Inbox triage mode in the chat sidebar

Finding: the inbox half of Section 9a, "Session unread and
needs-attention state, with an inbox triage mode" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[inbox model](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/inbox-model.ts)
and
[inbox view](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/inbox.tsx).

* Author: whit3rabbit
* Status: Draft
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

--------------------------------------------------------------------------------

*Drafted with EGM*
