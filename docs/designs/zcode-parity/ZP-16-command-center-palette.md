# ZP-16: Command center palette

Finding 8.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[CommandCenterDialog](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/command-center/CommandCenterDialog.tsx)
and
[quickpick commands](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/quickpick/quickPickCommands.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Generalize the Cmd+K chat-search overlay into a command center palette:
one surface searching commands, chats, and workspace files with scope
tabs and prefixes, keeping the existing chat search as a scope rather
than the whole feature. An earlier kiro spec (`chat-runtime-qol`)
deferred the palette as a Partial row; that deferral is superseded,
confirmed by author decision 2026-11-11.

## The Problem

Cmd+K today opens chat search only. Commands (the roughly 40 built-in
slash commands plus settings panes) have no searchable entry surface, and
files have no search beyond the `@` mention autocomplete's bounded
index. ZCode's command center searches actions, tasks, and files with
prefixes (`>` commands, `#` conversations, `@` files), scope tabs, and
per-workspace search history.

## Goals

* Cmd+K opens a palette with three scope tabs: All, Commands, Chats,
  Files.
* Prefix search: `>` filters to commands, `#` to chats, `@` to files.
* Commands scope indexes: built-in slash commands, menu-reachable actions
  (settings panes, check-for-updates, export), and plugin-contributed
  commands (namespaced `<plugin>:<command>` when plugins are enabled).
* Files scope reuses `ProjectFileIndex` (bounded walk, 2000 entries,
  depth 8, 20 second freshness).
* Per-workspace search history (20 entries cap), persisted.
* Cmd+Shift+P opens the palette pre-filtered to commands (the quick-pick
  parity entry).

## Non-Goals

* No fuzzy-matching algorithm changes to the existing chat search
  (recency-ordered match surfacing stays).
* No task-body search into SQLite (ZCode's `searchable_text` column);
  chat search stays on the in-memory archive.

## Open Questions

* **Home view**: ZCode's default home lists recent changes, recent
  tasks, and grouped commands. Options: (a) same; (b) empty state with
  hints; (c) recent chats plus commands. Lean (c). Author decides.

## Critical User Journeys

### CUJ 1: run a command by name

* **Given**: the user wants the export command without remembering the
  menu path.
* **When**: they press Cmd+K, type `>`, then "export".
* **Then**: the commands scope lists matching commands with their
  bindings; Enter runs the export flow.

### CUJ 2: jump to a file

* **Given**: a project with an indexed file tree.
* **When**: the user types `@` then a file stem in the palette.
* **Then**: the files scope lists matches from ProjectFileIndex with
  directory context; Enter attaches the file as an `@file:` reference in
  the composer (the existing mention path), not a file-manager open.

### CUJ 3: chat search preserved

* **Given**: the existing Cmd+K muscle memory.
* **When**: the user types plain text without prefixes.
* **Then**: the All scope blends chats (existing search) with commands
  and files; plain-Enter on a chat row opens the chat exactly as the
  current overlay does; the old overlay is fully replaced, not
  duplicated.

## Technical Plan (Architecture)

```text
CommandPaletteView (new, replaces ChatSearchOverlayView host)
  scope tabs, prefix parser, history section
       |
       v
PaletteProviders (protocol)
  ChatSearchProvider   (wraps existing ChatSearch)
  CommandProvider      (built-in commands + settings actions
                        + plugin commands via PluginManager)
  FileProvider         (wraps ProjectFileIndex)
       |
       v
Execution router
  command -> its action (existing handlers)
  chat -> chat selection (existing)
  file -> composer mention attachment (existing path)
```

Each provider returns a common scored row type; ranking is per-scope
recency and prefix-position based, consistent with existing search
behavior (no new scoring claims).

## Alternatives Considered

### Alternative A: keep the chat overlay and add a separate Cmd+Shift+P

* **Pros**: zero regression risk to chat search.
* **Cons**: two overlapping surfaces to learn; ZCode merged them for a
  reason.
* **Reason for Rejection**: Rejected by author: one surface, one muscle
  memory.

### Alternative B: menu-bar-native NSPredicate-based search

* **Pros**: AppKit-native help search.
* **Cons**: cannot search chats or attach files into the composer.
* **Reason for Rejection**: Rejected by author: scope breadth is the
  feature.

## Non-Functional Requirements

* **Accessibility**: full keyboard navigation, VoiceOver labels per row
  and scope, announcement of scope changes.
* **Performance**: providers answer from existing in-memory state; the
  file provider hits the bounded index; palette opens under 100 ms with
  no new I/O on the open path.
* **Design compliance**: palette chrome follows DESIGN.md; strings in
  the catalog for 21 languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/CommandPaletteView.swift` (Create)
   * **Type / Scope**: palette UI.
   * **Proposed Changes**: scope tabs, prefix parser, row rendering,
     history section, execution router.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PaletteProviders.swift` (Create)
   * **Type / Scope**: provider protocol and three implementations.
   * **Proposed Changes**: common row type; chat provider wrapping the
     existing ChatSearch; command provider aggregating built-ins, menu
     actions, and plugin contributions; file provider over
     ProjectFileIndex.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatSearchOverlayView.swift` (Modify)
   * **Type / Scope**: legacy overlay.
   * **Proposed Changes**: replace the overlay host with the palette;
     ChatSearch remains the chat provider's engine.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift` (Modify)
   * **Type / Scope**: shortcut catalog.
   * **Proposed Changes**: shortcut rows for Cmd+Shift+P and the palette
     help entries; keep the catalog single-source rule.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/CommandPaletteTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: prefix parsing, scope filtering, provider
     aggregation, history cap, chat-row behavior parity with the old
     overlay, file-to-mention handoff payload.
   * Mutation checks: drop the `>` prefix and assert the commands-scope
     test fails; break the history cap and assert the cap test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the palette tests plus existing
  chat-search tests (must stay green).
* **Static Analysis and Linting**: `make design-check`.
* **Localization**: catalog additions in 21 languages.
* **Manual**: exercise all three scopes, prefixes, history, and both
  entry bindings; confirm chat search behavior is unchanged for
  plain-text queries.

--------------------------------------------------------------------------------

*Drafted with EGM*
