# ZP-21: Workspace tabs inside one window

Finding 1.6 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[tabStore](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/store/tabStore.ts).

* Author: whit3rabbit
* Status: Implemented. See "Implementation Record" for the deviations from
  this text and the exact remainder.
* Date: 2026-11-11
* Depends on: [ZP-23](ZP-23-concurrent-chats.md) (per-chat turn states and
  the project-switch guard) and [ZP-02](ZP-02-unread-pending-badges.md)
  (badge states). Both are implemented (ZP-23 partially).

--------------------------------------------------------------------------------

## Overview

A tab strip in the main window with one tab per open project. A tab is a
persisted reference to an existing `AppProject` plus the chat last open in
it. Switching tabs moves the app's selected project through the same
`openChat` and `selectProject` paths the sidebar uses.

## The Problem

Working across two or three projects means clicking through the sidebar's
project list, and the app forgets which chat was open in the project you
left. The sidebar already holds every project's chats, drafts and queued
work per chat; what is missing is a fast, visible way to hold two or three
projects open and return to each exactly where it was.

## Goals

* A tab strip in the main window: one tab per open project, shown only while
  two or more tabs are open.
* Tab state persists across launches: the open tabs (project ids, in
  creation order), the selected tab, and each tab's last chat.
* Switching tabs restores the project's last chat. The composer draft, the
  attachments and the queued prompts of every chat already live on
  `AppChat`, so they are retained per chat without new state.
* Each tab shows one badge: the strongest ZP-02 state (needs approval,
  generating, unread) among that project's chats.
* Keyboard: next and previous tab. VoiceOver: each tab is a button with a
  selected trait, a value for its badge, and a named close button.

## Non-Goals

* Not macOS native window tabs: the app is one window with internal
  navigation, and a second window would duplicate the AppModel wiring.
* No Settings tab. Settings stays its own window. ZCode's reserved settings
  tab id is not modeled: no code would read it.
* No per-tab container for chat, draft or scroll. A second copy of state
  that `AppChat` and `selectedProjectID` already hold is a second source of
  truth and a leak risk. Transcript scroll position is not restored
  (best effort per chat id is possible later; see Remainder).
* No tab reordering in v1: order is creation order.
* No remote-session rekeying (ZCode's paired remote sessions are excluded,
  see [ZP-90](ZP-90-excluded-and-decision-findings.md), rows 6.5 and 6.6).
* No concurrent turns and no new lane or cap. Tabs compose on ZP-23's one
  turn lane, its model-wide guard and its approval pin.

## Critical User Journeys

### CUJ 1: two projects side by side

* **Given**: projects A and B are open as tabs.
* **When**: the user types a draft in A, clicks B's tab, then clicks A's tab.
* **Then**: B opens on its last chat; returning to A reopens the chat that
  was open (not the newest) with its draft intact.

### CUJ 2: restore on launch

* **Given**: three tabs were open at quit, the second selected.
* **When**: the app relaunches.
* **Then**: the same three tabs restore in order. Which project is on screen
  is decided by the project archive (the existing restore path), and the tab
  selection is aligned to it. A tab whose project was deleted is dropped.

### CUJ 3: a switch while a turn runs

* **Given**: the selected project's chat is generating.
* **When**: the user clicks another project's tab.
* **Then**: the switch is refused with a toast. ZP-23 keeps the blanket guard
  on cross-project moves (a project switch rebinds the worktree and hook
  store and resets the file snapshot hashes). The user can still switch
  chats inside the project, and the running chat's row shows its state. The
  same refusal applies while a prompt is submitting or an approval card is
  open (the card pins the user to its chat, hence to its tab).

## Technical Plan

```text
AppModel.selectedProjectID  (unchanged: the one source of truth)
   |  didSet
   v
workspaceTabs: WorkspaceTabStore   (new, pure value)
   tabs: [WorkspaceTab { projectID, lastChatID }]  unique by project
   selectedProjectID: UUID?
   rules: open, navigate, close, prune, cycle, cap 8, dormant below 2
   |
   v
persistence: workspace_tabs.json via AppJSONStore (quarantine on corruption)
```

`AppModel+TabContext.swift` routes every tab command through
`canSwitchWorkspace` (the same three flags `applyProjectSelection` refuses
on) and then `openChat` or `selectProject`, so no guard is duplicated or
bypassed. Sidebar clicks keep working: every change of `selectedProjectID`
is mirrored into the store, so the strip cannot disagree with the sidebar. A
project with no tab retargets the selected tab in place (like navigating a
browser tab) instead of growing the list on every sidebar click.

## Alternatives Considered

### Alternative A: native NSWindow tabbing

* **Cons**: each window needs its own AppModel wiring or a shared one that
  assumes a single selected project; the file snapshot store, hook store and
  worktree are process-wide and keyed to one project.
* **Reason for Rejection**: the single-selection state tree is the app's
  design.

### Alternative B: split panes per workspace

* **Cons**: two chat plus inspector layouts do not fit the window's
  minimum width; split panes already serve side-by-side chats in one project.
* **Reason for Rejection**: space.

### Alternative C: a project switcher that remembers the last chat

* **Pros**: smallest change; no strip.
* **Cons**: no always-visible set of open projects and no per-project badge.
* **Reason for Not Choosing Alone**: the remembered last chat is built (it is
  half of this feature), and the strip is a thin view over it. The strip
  stays hidden until a second tab is opened, so the window is unchanged for
  anyone who does not use it.

## Non-Functional Requirements

* **Performance**: a tab is two ids. There is no render tree per tab: the
  strip is one row, the working area is the existing single view. A switch
  is the existing `openChat` cost. No latency figure was measured.
* **Persistence safety**: one small file per profile under the profile's
  storage root, written through `AppJSONStore` (atomic write, quarantine of
  an unreadable file to `corrupt-...`, restore falls back to no tabs). It
  holds ids only: no draft text, no content. It is not in the encrypted
  vault's protected set because it carries no conversation data; the project
  archive it references is. Restore runs after the projects load, so it does
  not depend on vault unlock ordering beyond what the project load already
  does.
* **Design compliance**: DESIGN.md metrics and theme tokens, 21-language
  strings, Reduce Motion through `TSMotion`, catalog entries for the
  shortcuts.
* **Accessibility**: a labeled group, selected trait, badge as VoiceOver
  value, close button with a per-tab name, context-menu close.

## Risks and Mitigations

* **State leakage between tabs**: no per-tab state exists beyond two ids, and
  the draft test switches tabs and asserts each chat keeps its own text.
* **Strip and sidebar disagree**: every `selectedProjectID` change flows into
  the store; a test selects through the sidebar path and asserts the strip.
* **A switch under a running turn moves the workspace under a tool**: tab
  commands go through the ZP-23 guard; tests assert refusal while generating,
  submitting and awaiting approval.
* **Deleted or moved project on restore**: pruned at restore and on project
  delete; a stale last chat falls back to the project's default selection.
* **Close while generating**: closing a background tab touches no running
  work. Closing the selected tab first moves to its neighbor, so a refused
  move keeps the tab.
* **Too many tabs**: cap of 8 with a named toast.
* **Dependence on ZP-23's remainder**: the limits are listed below and
  inherited, not worked around.

## Implementation Record

What shipped:

* `State/WorkspaceTabStore.swift`: `WorkspaceTab`, the archive, the pure
  `WorkspaceTabStore`, `WorkspaceTabBadge` and `WorkspaceTabFileStore`.
* `State/AppModel+TabContext.swift`: the AppModel side (open, activate,
  close, cycle, follow selection, note chat, prune, badge, restore).
  `AppModel.swift` gained the `workspaceTabs` property and three one-line
  hooks (`selectedProjectID` and `selectedChatID` didSet, init restore);
  `deleteProject` prunes.
* `Chrome/WorkspaceTabStripView.swift`, mounted in `RootView` between the top
  bar and the working area (Projects mode, chat panes only). A row of the
  top-bar band, not a fifth band; DESIGN.md says so.
* Entry points: "Open in New Tab" in the sidebar project menu and a plus
  menu in the strip. Shortcuts: Ctrl+Tab and Ctrl+Shift+Tab in the Chat
  menu (Cmd+W stays the window's close), listed in
  `KeyboardShortcutCatalog` and `KEYBOARD_SHORTCUTS.md`.
* Tests: `WorkspaceTabStoreTests`, mutation-checked (see the test header
  and the commit notes).

Deviations from the original draft (review fixes B1, B3, M1, M2, M4 to M10):

* The per-tab "state container" is gone (B1). A tab is a project reference
  plus a last-chat hint; drafts and attachments are already per chat.
* "Workspace" means `AppProject` (B1). "All chats" (no project) has no tab.
* The reserved settings tab id and "per-tab availability" are dropped (M1,
  M2). Persisted: open project ids in order, the selected one, last chat.
* CUJ 3 is refusal, not "the switch proceeds" (M3, M4): ZP-23 deliberately
  kept the blanket guard on cross-project moves. The model switch refusal
  and the one-lane cap are global and unchanged.
* The badge aggregates all of the project's chats by ZP-02 priority (M5).
  Because background work cannot live in another project today (guard
  above), a non-selected tab mostly shows unread; running and pending show on
  the selected tab's project via the sidebar rows too.
* Scroll position is not restored (M7). Transcript scroll is view state with
  no per-chat store; restoring it is future work.
* Tab cap 8, named toast (M6, M7). Duplicate open selects the existing tab.
* The strip is hidden below two tabs, and closing down to one empties the
  list, so the feature is dormant by default (Alternative C).
* The ZCode link uses the `store/tabStore.ts` path from the comparison doc
  (M11).

Remainder, with reasons:

1. Background generation in another project's tab. Needs ZP-23 remainder 4
   (cross-project switching without the blanket guard), which needs the
   worktree, hook store and snapshot hashes keyed per project.
2. Two approvals at once, and a question that holds the lane: ZP-23
   remainders 1 and 2. A tab with an open approval pins the user to it.
3. Per-chat transcript scroll restore.
4. Tab reordering, a Settings tab, and OC-04's session tab strip. OC-04
   composes on `WorkspaceTabStore`: the store is a pure list keyed by an
   id with a last-selection hint, and has no view or project dependency
   except `WorkspaceTabBadge.resolve(slots:)`.
5. Manual run in the app (two projects, switch mid-draft, quit and relaunch)
   and the DESIGN.md visual checks (text sizes, themes, RTL) were not run in
   this change; see the commit report for what was.

--------------------------------------------------------------------------------

*Drafted with EGM*
