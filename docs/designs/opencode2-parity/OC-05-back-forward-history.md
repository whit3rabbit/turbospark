# OC-05: Back and forward navigation history

Finding: Section 9a, "Back and forward navigation history - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[titlebar history](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/history.ts),
wired as commands in
[titlebar.tsx](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/titlebar.tsx).

* Author: whit3rabbit
* Status: Implemented (2026-11-14)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

A bounded navigation history over section switches: every jump (sidebar
selection, settings pane open, search jump) pushes an entry, and Cmd+[ /
Cmd+] walk the stack back and forward. OpenCode keeps a 100-entry
history over home, session, and settings URLs and collapses consecutive
settings hops so palette jumps do not spam the stack.

## The Problem

Section switching is direct state assignment in
[RootView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift).
There is no back-stack and no go-back handling anywhere in the app
sources; every jump is one-way. A settings hop or a search-result jump
leaves the user to retrace their path by hand.

## Goals

* A history model over navigation targets: section identity, optional
  chat id, optional settings tab id.
* Recording on every section-level transition, including settings pane
  opens and chat selections.
* Cmd+[ goes back and Cmd+] goes forward, as menu items in the system
  View menu via `CommandGroup`, with chords registered in the catalog.
* Consecutive settings hops collapse into one entry, so walking back
  through a settings session is one step, not one per pane visited.
* A 100-entry bounded stack, trimmed oldest-first.
* Menu items disable at the ends of the stack and say why in their
  tooltip.

## Non-Goals

* No history across relaunches.
* No history UI (a pick-from list); only back/forward walking.
* No recording of in-pane transitions (settings controls, audio
  sub-surfaces). Section level only.

## Critical User Journeys

### CUJ 1: settings session collapses

* **Given**: the user leaves the chat for Settings and visits three
  panes.
* **When**: they press Cmd+[.
* **Then**: the app returns to the pre-settings view in one step, and
  Cmd+] returns to the last settings pane.

### CUJ 2: search jump is reversible

* **Given**: a chat search result opened a chat.
* **When**: the user presses Cmd+[.
* **Then**: the prior section is restored, and Cmd+] returns to the
  chat.

## Technical Plan (Architecture)

```text
NavigationHistory (bounded 100, collapse rule)
  entries: (section, chatID?, settingsTabID?)
        |
        v
RootView: record on section-level transitions
        |
        v
TurboSparkApp Commands: Back / Forward items, catalog chords
        |
        v
AlternateShortcutBridge: no extra chords needed
```

Recording happens at section granularity only. Scroll and composer
state are preserved by the existing per-section state, not by the
history stack.

## Non-Functional Requirements

* **Performance**: O(1) push and trim; recording never blocks the
  transition.
* **Localization**: menu items and tooltips in 21 languages in the same
  change.
* **Accessibility**: menu items announce enabled state; keyboard-only
  navigation works end to end.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/NavigationHistory.swift` (Create)
   * **Rationale**: the bounded stack model with the collapse rule.
     Pure and headlessly testable.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift` (Modify)
   * **Rationale**: record on section-level transitions; expose
     can-go-back and can-go-forward for menu disabled states.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Rationale**: Back/Forward items in the View menu via
     `CommandGroup`, chords registered in the catalog.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift` (Modify)
   * **Rationale**: catalog entries for the two chords and their menu
     items.
5. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/NavigationHistoryTests.swift` (Create)
   * **Rationale**: push and trim, collapse rule, boundary disabled
     states. Mutation check: drop the collapse rule and assert the
     collapse test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus `KeyboardShortcutCatalogTests`,
  `make design-check`, and the localization gates.
* **Manual**: jump through several settings panes and back with the
  keyboard. Confirm the menu items enable and disable at the stack ends.

--------------------------------------------------------------------------------

## Open Questions and decisions

Resolved during implementation, following the EGM review in
`docs/egm-reviews/opencode2-parity/OC-05/`.

1. **Chords.** The draft's Cmd+[ / Cmd+] are already Previous/Next Chat
   (`TurboSparkApp.swift`, `KeyboardShortcutCatalog.swift`,
   `AlternateShortcutBridge.swift`), and "no extra chords needed" was wrong.
   Chat cycling is untouched. History takes Cmd+Option+Left / Cmd+Option+Right
   (Safari's tab chords; no collision with any bound chord in the app).
2. **Ownership.** The stack lives on `AppModel`, not `RootView`: the menu,
   its enabled state and about two dozen `activeSection` writers are all
   outside any one view. Alternatives rejected: per-view `@State` (Commands
   cannot read it) and a `NavigationStack` (the app is not path-driven).
3. **Entry shape.** `(section, chatID)`. The draft's `settingsTabID` is
   dropped: Settings is its own window scene, not a section, and settings
   tab state is in-pane. This also makes the draft's settings-collapse rule
   moot, so no collapse rule was built; consecutive identical entries are
   deduplicated instead.
4. **Recording.** `willSet` on `activeSection` and `selectedChatID` reports
   the pre-change entry (seeding the stack on first change); one deferred
   flush records the settled state, so `selectChat` (writes both) is one stop.
   Programmatic switches record too, since they are section changes the user
   sees.
5. **Replay.** Back/forward move the cursor only after the app actually
   arrived at the target (`selectChat` can refuse), and a record equal to the
   current entry is ignored, so a replay cannot truncate the forward stack.
6. **Semantics.** A new stop after going back discards the forward entries.
   Deleted chats are skipped on walk (unsaved draft chat counts as valid).
   Both items are disabled while a turn runs, as the chat cycle is.
7. **Tooltips.** Dropped: menu items show none. Disabled state follows the
   system menu behavior.
8. **Capacity.** 100, copied from OpenCode, not measured.
9. **Scope.** One global history for the single main window. OC-04 and ZP-21
   (tabs) extend the entry later.

## Risks and Mitigations

* A section writer that bypasses the observed properties cannot exist:
  both are `@Published` stored properties with the observers attached.
* Stale entries (deleted chats) are skipped at walk time, not pruned.
* Tab composition (OC-04, ZP-21) will need a tab id in the entry; the entry
  is a small value type so that is additive.

## Implementation record (2026-11-14)

* New: `App/NavigationHistory.swift` (pure bounded stack with cursor),
  `State/AppModel+NavigationHistory.swift` (recording, replay, enablement),
  `Tests/.../NavigationHistoryTests.swift`.
* Modified: `State/AppModel.swift` (observers, stored history),
  `App/TurboSparkApp.swift` (Go Back / Go Forward in the View menu group),
  `App/KeyboardShortcutCatalog.swift` (two rows, option glyph),
  `Localization/Localizable.xcstrings` ("Go Back", "Go Forward" in 21
  languages), `swift/docs/KEYBOARD_SHORTCUTS.md`.
* `RootView.swift` and `AlternateShortcutBridge.swift` are unchanged by
  design (items 2 and 1).
* Verified: `NavigationHistory` tests (13), `KeyboardShortcutCatalogTests`,
  `LocalizationParityTests`, `make compile-strings`, `make design-check`.
  Mutation checks, each failing at least one test: dropping the dedupe, the
  forward truncation, the oldest-first trim, the arrival guard (needed a
  dedicated refused-switch test, added after the first run survived it) and
  the chat validity test.
* Not run: manual keyboard walk in a running app, `make swift-lib` (shared
  staged archive).
* Deferred: nothing required a dependency, CI change or auth decision.

*Drafted with EGM*
