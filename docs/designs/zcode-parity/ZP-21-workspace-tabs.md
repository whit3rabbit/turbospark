# ZP-21: Workspace tabs inside one window

Finding 1.6 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[tabStore](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/tabStore.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Host multiple workspace tabs in the single main window: a tab strip with
persisted availability, session restore per tab, and a reserved settings
tab identity. ZCode's desktop is tab-per-workspace; TurboSpark is
deliberately single-window but shows one workspace (project) at a time.

## The Problem

Working across two or three projects requires switching the whole
window's project context (sidebar filter, git pane, file surfaces), and
switching away interrupts the mental model of the previous project even
when chats are idle. ZCode's tab system keeps workspaces alive side by
side. This is the largest navigation change in the program and it is
gated on a bigger decision: concurrent live chats (finding 2.1), because
tabs that each hold a running chat are only meaningful once background
chats exist.

## Goals

* A tab strip in the main window: one tab per open workspace, plus a
  reserved settings tab id (settings render in a tab instead of a
  separate window in this mode).
* Tab state persists across launches: open tabs, selected tab, per-tab
  availability.
* Switching tabs preserves each workspace's selected chat, scroll, and
  composer draft.
* Design works with one active chat per workspace today and with the
  adopted concurrent chats (ZP-23) without schema changes.

## Non-Goals

* Not macOS native window tabs (`NSWindowTab`): the app's single-window
  design with internal navigation is the chosen shape, matching ZCode.
* No tab reordering drag in v1 (ZCode has availability persistence;
  ordering is creation order).
* No remote-session rekeying (ZCode feature for paired remote sessions;
  excluded domain, see ZP-90).

## Open Questions

* **Prerequisite resolved**: concurrent background chats are adopted
  ([ZP-23](ZP-23-concurrent-chats.md), author decision 2026-11-11), so
  tabs can hold generating chats; the tab switch guard defers to the
  per-chat guards. Sequence ZP-23 before this draft.
* **Settings tab vs window**: keep the separate Settings window (fewer
  changes) or fold it into a reserved tab (ZCode's shape). Lean keep the
  window; reserve the id for future use.

## Critical User Journeys

### CUJ 1: two workspaces side by side

* **Given**: two projects open as tabs.
* **When**: the user works in project A, then clicks project B's tab.
* **Then**: B's selected chat, sidebar state, and composer draft
  restore exactly; A's state is retained; switching back to A restores
  it; generation in A (if the concurrent-chats answer is adopted)
  continues and its row badge shows per ZP-02.

### CUJ 2: restore on launch

* **Given**: three tabs were open at quit.
* **When**: the app relaunches.
* **Then**: the same three tabs restore with the last selected one
  active.

### CUJ 3: switch guard

* **Given**: project A's selected chat is generating.
* **When**: the user clicks project B's tab.
* **Then**: the switch proceeds; A's chat keeps generating in the
  background per ZP-23, with its state visible through the tab strip
  badges.

## Technical Plan (Architecture)

```text
MainWindow
  WorkspaceTabStrip (new)
    tabs: [WorkspaceTab]  (persisted)
      id, workspaceID, reservedSettings?
  selected tab -> existing project context
       |
       v
Per-tab state container
  selected chat id, composer draft, scroll anchors
  (today these live once on AppModel; they move into
   a per-tab store keyed by tab id, with the AppModel
   reading the selected tab's slice)
       |
       v
Persistence
  tabs archive (AppJSONStore, quarantine rules)
  restored at launch before first render
```

The core refactor is extracting the single "current project context"
into a keyed store. Views already read context from AppModel; routing
them through the selected tab's slice keeps view code unchanged in most
surfaces.

## Alternatives Considered

### Alternative A: native NSWindowTab multi-window

* **Pros**: platform-native tabs.
* **Cons**: windows duplicate the heavy AppModel wiring; ZCode's
  in-window shape keeps one state tree; the app deliberately chose one
  window.
* **Reason for Rejection**: Rejected by author: consistent with the
  single-window design.

### Alternative B: split panes per workspace

* **Pros**: side-by-side visibility.
* **Cons**: horizontal space does not fit two chat+inspector layouts;
  the existing split panes already serve side-by-side chats within one
  workspace.
* **Reason for Rejection**: Rejected by author: space and precedent.

## Non-Functional Requirements

* **Performance**: non-selected tabs hold lightweight state (not live
  render trees); switching restores from the store, not from scratch.
* **Persistence safety**: the tabs archive quarantines on corruption
  like every AppJSONStore file; restore falls back to a single tab.
* **Design compliance**: the tab strip follows DESIGN.md metrics and
  token ratchet; localized strings, 21 languages; keyboard shortcuts
  registered in the catalog (next/previous tab).

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/WorkspaceTabStore.swift` (Create)
   * **Type / Scope**: tab model and persistence.
   * **Proposed Changes**: WorkspaceTab type, open/close/select,
     reserved settings id, archive with quarantine, restore order.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+TabContext.swift` (Create)
   * **Type / Scope**: per-tab context extraction.
   * **Proposed Changes**: move current-project context (selected chat,
     draft, per-workspace UI state) into a keyed store; AppModel reads
     the selected tab's slice; guards route through the same
     selectChat-family paths.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorkspaceTabStripView.swift` (Create)
   * **Type / Scope**: tab strip UI.
   * **Proposed Changes**: strip with badges (ZP-02 states per tab's
     active chat), close buttons, plus/new-workspace entry.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Type / Scope**: window wiring.
   * **Proposed Changes**: mount the strip above the sidebar, restore
     tabs at launch.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift` (Modify)
   * **Type / Scope**: shortcuts.
   * **Proposed Changes**: next/previous tab bindings.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/WorkspaceTabStoreTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: persistence round-trip, restore order,
     quarantine fallback, context-slice isolation between tabs (draft in
     A does not leak to B), guard routing on switch.
   * Mutation checks: share context across tabs and assert the
     isolation test fails; drop quarantine and assert the corruption
     test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the tab store tests.
* **Static Analysis and Linting**: `make design-check` and
  `make design-check-full`.
* **Localization**: catalog additions in 21 languages.
* **Manual**: open two projects, switch repeatedly mid-draft and
  mid-generation (both Open Question answers), quit, relaunch, verify
  restore; verify tab switches during background generation per ZP-23.

--------------------------------------------------------------------------------

*Drafted with EGM*
