# OC-04: Session tab strip with draft tabs and closed-tab restore

Finding: Section 9a, "Window tab strip: session tabs, draft tabs, and
closed-tab restore - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[tab schema](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/tabs/schema.ts),
[tab store](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/tabs/tabs.tsx),
and
[closed-tab stack](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/tabs/closed.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Keep many sessions open at once as browser-style tabs over sessions: a
persisted open-tab set of session tabs plus unsent draft tabs, draft-tab
promotion that carries the composer text into the new chat, a closed-tab
stack with a Recently Closed reopen menu, and per-tab composer state.
Composes on [ZP-21](../zcode-parity/ZP-21-workspace-tabs.md) (workspace
tabs) and [ZP-23](../zcode-parity/ZP-23-concurrent-chats.md) (concurrent
background chats), which supply the workspace layer and the live
background turns that make many-open-session tabs worth having.

## The Problem

One chat selection is restored at launch
([AppModel+Persistence.swift](swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift))
and [SplitChatPanes.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SplitChatPanes.swift)
can show several chats at once, but those are split panes, not tabs:
there is no persisted open-tab set, no draft tab that holds unsent
composer text across switches, no closed-tab stack or reopen, and no
per-tab pane-state memory. OpenCode's whole keep-many-sessions-open
model (persistence, drafts-in-tabs, closed-tab restore) is absent, and
its report ranks this gap fourth by user value.

## Goals

* A tab strip over sessions in the working-panes band: session tabs
  (one per open chat) and draft tabs (an unsent composer state held as a
  tab).
* The open-tab set, selected tab, and per-tab composer draft text
  persist across launches.
* Draft promotion: submitting a draft tab's text creates the chat and
  the tab becomes a session tab in place, with no composer flash.
* Closing a session tab pushes it onto a closed-tab stack; a Recently
  Closed menu (sidebar context menu plus a keyboard action) reopens the
  chat in a new tab at its previous position.
* Per-tab memory: switching tabs preserves each tab's scroll and
  composer draft, matching ZP-21's per-tab memory contract at session
  granularity.
* Cmd+T opens a new draft tab; Cmd+W closes the selected tab (tab-first,
  not window-first, matching macOS convention); option+arrow cycles
  tabs. All three join `KeyboardShortcutCatalog` through
  [OC-01](OC-01-keybind-customization.md) once both land.

## Non-Goals

* No second window: the app's single-window commitment stands; tabs live
  in the one window (multi-window remains excluded, see
  [OC-90](OC-90-excluded-and-decision-findings.md)).
* No tab drag reordering in v1, matching ZP-21's v1 ordering decision;
  ordering is open order.
* No per-tab split-pane layout memory in v1: split panes stay a window
  property. Revisit once tabs are stable.
* No tabs for non-chat surfaces (Audio, Humanize): section navigation
  stays as it is.

## Open Questions

* **Strip placement**: (a) a horizontal strip at the top of the
  working-panes band, under the existing top bar; (b) vertical tab list
  at the sidebar's top. OpenCode supports both. DESIGN.md fixes the app
  at four chrome bands and forbids a fifth, so (a) must be a sub-row of
  the panes band, not a new band. Lean (a) with a hard height cap; the
  strip hides when exactly one tab is open plus at most one draft.
* **Interaction with split panes**: whether a tab may hold a split-pane
  layout in v1 (lean: no; panes stay window-level until tabs are
  stable).
* **Recently Closed size** (lean: 20 entries, session-scoped).

## Critical User Journeys

### CUJ 1: draft tab carries through

* **Given**: a fresh draft tab with typed but unsent composer text.
* **When**: the user submits the text.
* **Then**: the chat is created, the tab becomes a session tab in
  place, the composer is empty and focused, and the transcript shows the
  sent message with no flash of an empty state.

### CUJ 2: closed tab reopens

* **Given**: a session tab closed minutes ago; the chat itself was not
  deleted.
* **When**: the user picks it from Recently Closed.
* **Then**: a new tab opens on that chat at the former position in the
  strip, with the chat's transcript and draft intact.

### CUJ 3: relaunch restores the set

* **Given**: three session tabs and one draft tab open; the user quits
  and relaunches.
* **Then**: the same four tabs restore in order with the same
  selection, and each session tab's draft text is preserved.

## Technical Plan (Architecture)

```text
SessionTabStore (AppJSONStore-backed, profile-scoped)
  tabs: [SessionTab]           // .session(chatID) | .draft(text)
  selectedTabID, closedStack: [ClosedEntry] (bounded 20)
        |
        v
SessionTabStripView (working-panes band sub-row)
  overflow into a labeled menu beyond the width cap (AppAdaptiveHeader
  compact-actions pattern; never truncate tab titles)
        |
        v
RootView + AppModel
  selection drives the existing chat selection path
  draft tab holds PromptComposer draft text; promotion calls the
  existing submit path and swaps tab kind in place
  close: park chat selection, push closed entry, select neighbor
```

Persistence reuses the per-domain `AppJSONStore` archive pattern
(quarantine on corruption). Draft tabs persist their text so an
accidental quit never loses an unsent prompt; session tabs persist only
ids, never transcript content outside the existing chat archive.

## Non-Functional Requirements

* **Performance**: switching tabs must not reload a chat from disk when
  it is already resident; the store holds no transcript data.
* **Localization**: tab context menus, Recently Closed, and empty-state
  copy ship in 21 languages in the same change.
* **Accessibility**: tabs expose `.isSelected`, a tab role, and
  overflow items through the labeled menu; keyboard paths exist for
  every action (new, close, cycle, reopen); hover states are not the
  only affordance.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SessionTabStore.swift` (Create)
   * **Rationale**: tab model, persistence, closed stack, promotion and
     close operations, restore validation (chats that were deleted while
     closed are dropped at restore).
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SessionTabStripView.swift` (Create)
   * **Rationale**: the strip view with overflow menu, close buttons,
     draft-tab indicator, and the accessibility contract.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift` (Modify)
   * **Rationale**: host the strip as a sub-row of the working-panes
     band; drive selection; keep the four-band structure.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: persist the store with the archive; restore order.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Chat.swift` (Modify)
   * **Rationale**: selection changes flow through tab selection;
     promote/close hooks; keep previous/next chat shortcuts working with
     tabs present.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Rationale**: File and View menu entries (New Tab, Close Tab,
     Recently Closed submenu) wired to the store; chords registered in
     the catalog.
7. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift` (Modify)
   * **Rationale**: catalog entries for the three new chords.
8. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/SessionTabStoreTests.swift` (Create)
   * **Rationale**: persistence round trip, promotion in place, closed
     stack bounds, reopen ordering, and restore with deleted chats.
     Quarantine on corrupt store. Mutation check: break the closed-stack
     bound and assert the bound test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus `KeyboardShortcutCatalogTests`,
  localization gates, and `make design-check`.
* **Manual**: the three CUJs. Then the DESIGN.md section 9 checklist on
  the strip. Cover text sizes, window width, themes, keyboard-only use,
  VoiceOver, motion settings, German, and an RTL language.

--------------------------------------------------------------------------------

*Drafted with EGM*
