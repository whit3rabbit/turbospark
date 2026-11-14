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
* Status: Partial (implemented 2026-11-14; see Implementation Record for the deferred items)
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

## Implementation Record (2026-11-14)

Implemented with the EGM review's recommendations adopted where they
removed ambiguity (see `docs/egm-reviews/opencode2-parity/OC-04/`).

### Decisions (Open Questions and review blockers)

* **Strip placement**: (a), a row directly above the working area, under
  the ZP-21 workspace row, drawn only with two or more tabs (the review's
  M10: two, not "one plus a draft", so Cmd+T is never inert). It is a row
  of the existing bands, not a fifth band. Overflow: the row scrolls and an
  "All Tabs" menu lists every tab and Recently Closed; titles are not cut
  to nothing.
* **Split panes**: a tab does not hold a layout. Panes stay window-level.
  The review's "panes exclude the selected tab's chat" rule is deferred;
  `toggleSplitPane` already refuses the selected chat.
* **Recently Closed**: 20 entries, in memory only. A relaunch clears it
  (review M6); the persisted file holds only chat ids in order.
* **A tab is a chat id, a draft tab is a chat row** (review M1, B3). There
  is no second draft store: draft text stays on `AppChat.draft` and rides
  the chat archive, so ghost chats cannot reach disk through tabs and the
  ghost filter in `makeChatArchive` still covers everything. Promotion is
  therefore in place by construction: sending adds messages to the same
  chat, the tab is not swapped, and no empty state can flash.
* **Empty draft discard**: closing a draft tab that has no messages, text,
  attachments, checklist or sampling edit deletes its row and does not
  enter Recently Closed. Cmd+T reuses an open empty draft tab instead of
  piling up another.
* **One tab per chat**; opening an open chat selects it. Selecting a chat
  that has no tab (sidebar, palette, history) retargets the selected tab in
  place, as ZP-21 does for projects.
* **Chords (review B4, M4)**: Cmd+W stays the window's (OC-01 reserved
  list). New Tab Cmd+T, Close Tab Cmd+Shift+W, Reopen Closed Tab
  Cmd+Shift+T, next/previous tab Cmd+Option+] / [ (Option+Arrow is word
  navigation in the composer; Cmd+[ / ] is the chat cycle; Cmd+Option+Left
  and Right is history; Ctrl+Tab is the ZP-21 workspace cycle). All five
  are catalog commands (`session.newTab`, `session.closeTab`,
  `session.reopenTab`, `session.nextTab`, `session.previousTab`), rebindable
  through OC-01. Closing down to one tab makes the strip dormant and leaves
  that chat on screen; the last chat is never closed, so no chord can quit
  the app.
* **ZP-23 (review M3)**: switching between chats of one project follows the
  coordinator verdict (a background turn does not block it; an open
  approval card or a submitting turn does). Switching to a chat under
  another project is a workspace switch and is refused with a toast while a
  turn runs, is submitting or an approval is open, exactly as ZP-21.
  Closing the selected tab moves to its neighbor first, so a refused move
  keeps the tab.
* **History (OC-03/OC-05)**: tab selection is ordinary chat selection, so
  `selectedChatID` feeds `NavigationHistory` unchanged.
* **Restore (review M5, M8)**: the chat archive owns `selectedChatID`;
  tabs load after the chats and drop ids that are deleted, archived or
  ghost. Reopen drops entries whose chat is gone and skips them silently.

### Files

* `State/SessionTabStore.swift` (new): value, archive, `SessionTabFileStore`
  (`session_tabs.json`, `AppJSONStore`, quarantine on corruption).
* `State/AppModel+SessionTabs.swift` (new): commands, follower, restore.
* `Generation/SessionTabStripView.swift` (new): the strip.
* Modified: `AppModel.swift` (state, follower hook, restore),
  `AppModel+Chat.swift` (prune on delete), `AppModel+SlashParity.swift`
  (prune on archive), `RootView.swift`, `TurboSparkApp.swift` (Chat menu:
  New Tab, Close Tab, Reopen Closed Tab, Recently Closed, next/previous),
  `KeyboardShortcutCatalog.swift`, `ChatSidebarChatRowView.swift` ("Open in
  New Tab"), `Localizable.xcstrings` (8 keys, 21 languages).
* Tests: `SessionTabStoreTests.swift`.

### Follow-up (2026-11-14): scroll memory and sidebar Recently Closed

* **Per-tab scroll memory**, in memory only (nothing on disk, as the draft
  allows). `State/TranscriptScrollMemory.swift` holds one position per chat
  that has a tab: `.followLatest` when the reader was pinned to the live end
  (never an offset, so a returning reader lands at the live end even if the
  transcript grew), otherwise the viewport's top offset. `ChatTranscriptView`
  (keyed per chat, so it is rebuilt on a tab switch) records on disappear
  and restores on appear. The restore is an unanimated clip-view scroll, so
  Reduce Motion needs nothing special; it unpins the follower, retries for
  up to about a second while lazy content is still short, and stops on the
  reader's own scroll or an OC-09 reveal. A pending reveal for that chat
  wins over a remembered position. A transcript torn down before it settled
  records nothing. Only chats with a tab are remembered, and only while tabs
  are active. ZP-23 lane and park rules are untouched.
* **Recently Closed in the sidebar chat row context menu**, reusing the
  existing string and stack.
* Tests: `TranscriptScrollMemoryTests.swift` (pure rules, AppModel gate,
  restore and cancel on a real `NSScrollView`).
* Not verified: the restore in the running app (the accuracy of the offset
  over lazily sized rows, with images or collapsed turns whose height
  changes while away, was only exercised on uniform rows), and the sidebar
  menu by eye.

### Deferred

* Panes excluding the selected tab's chat, and any per-tab pane layout.
* Simultaneous approvals, background prompt submission and real parallel
  decode (ZP-23 partial): tabs add no decode lane; one turn runs at a time.
* Manual passes not run in this change: the three CUJs in the app,
  VoiceOver, German and RTL layout, text sizes and the DESIGN.md section 9
  checklist.

--------------------------------------------------------------------------------

*Drafted with EGM*
