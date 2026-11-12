# OC-05: Back and forward navigation history

Finding: Section 9a, "Back and forward navigation history - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[titlebar history](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/history.ts),
wired as commands in
[titlebar.tsx](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/titlebar/titlebar.tsx).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
