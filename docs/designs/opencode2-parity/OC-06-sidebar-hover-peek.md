# OC-06: Sidebar hover peek

Finding: Section 9a, "Sidebar hover peek - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity target: OpenCode's
[sidebar-peek.ts](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/sidebar-peek.ts),
shipped in [PR 58156](https://github.com/anomalyco/opencode/pull/58156).

* Author: whit3rabbit
* Status: Implemented (2026-11-14); see Implementation record
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

With the sidebar collapsed, resting the pointer on the window's leading
edge slides the real sidebar in over the content as a transient overlay,
then docks it back when the pointer leaves. OpenCode's peek uses dwell
timers, corridor tracking, and suppression states; the TurboSpark port
reuses the sidebar view itself so the peek and the docked sidebar can
never drift.

## The Problem

[RootView.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift)
toggles `isSidebarExpanded` (an `@AppStorage` flag) with a layout
animation. With the sidebar closed there is no transient reveal: a user
who wants to confirm which chat is selected must expand the sidebar,
reflowing the whole layout. OpenCode added hover peek with careful
pointer-corridor tracking and suppression states, and its report calls
the gap motion polish over real content.

## Goals

* Pointer dwell (about 350 ms) on the leading edge, while the sidebar is
  collapsed, slides the live sidebar view in over content as an
  overlay. It does not reflow the layout.
* Leaving the sidebar's hover corridor docks it back with the pane
  motion curve (`TSMotion.pane`).
* Suppression states: no peek during drag operations, open sheets, open
  dialogs, or text selection; no peek while the pointer enters through
  the toggle control (the toggle is the deliberate path).
* Reduce Motion: the overlay appears and dismisses without animation
  (instant settle), keeping the behavior.
* The toggle keeps working exactly as before; peek is an additive
  pointer-only affordance.

## Non-Goals

* No peek when the sidebar is expanded.
* No pinning from the peek state; docking requires the toggle or pointer
  exit.
* No equivalent affordance for the right inspector column in v1.
* Keyboard and VoiceOver users are not targeted by the peek: the toggle
  remains their path, unchanged. Peek is a pointer enhancement, never a
  sole path to any content.

## Open Questions

* **Dwell duration**: 350 ms leans comfortable; OpenCode uses 40 ms on
  the edge and 500 ms on the toggle. Pick in review with the running
  app.
* **Corridor width**: how far past the sidebar edge the pointer may
  travel before the peek dismisses. Lean: the sidebar's own width plus a
  small grace band.

## Technical Plan (Architecture)

```text
SidebarPeekController (AppKit)
  NSTrackingArea on the content view's leading strip
  dwell timer -> peekRequested
  pointer corridor + suppression states -> peekDismissed
        |
        v
RootView overlay layer
  presents the live sidebar view (same view, .appThemed()) as an
  overlay with TSMotion.pane (instant under Reduce Motion)
  never changes isSidebarExpanded
```

The controller is AppKit because SwiftUI has no pointer-corridor
tracking; the overlay is SwiftUI so the peek renders the real sidebar.
The `@AppStorage` docked-state flag is untouched by peek entirely.

## Non-Functional Requirements

* **Performance**: one tracking area, one timer; no polling. The
  overlay holds no extra state: it renders the same view tree.
* **Localization**: the only string is the toggle's existing tooltip;
  peek adds none (no text in the overlay).
* **Accessibility**: unchanged. The peek exposes no new controls and
  hides none; the docked path is intact.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SidebarPeekController.swift` (Create)
   * **Rationale**: tracking area, dwell and corridor state machine,
     suppression inputs, motion-safe dismissal. Pure state machine plus
     an AppKit shell so it tests headlessly.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift` (Modify)
   * **Rationale**: host the overlay and feed suppression inputs (sheet
     and dialog presentation, drag session, text selection).
3. `swift/TurboSparkApp/Tests/TurboSparkAppTests/SidebarPeekControllerTests.swift` (Create)
   * **Rationale**: state machine coverage with an injected clock:
     dwell triggers only while collapsed, corridor exit dismisses,
     each suppression state blocks, Reduce Motion bypasses animation
     decisions. Mutation check: drop the drag suppression and assert
     its test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus `make design-check`.
* **Manual**: hover peek with each suppression state active. Confirm
  Reduce Motion settles instantly. Confirm the toggle path is unchanged
  for keyboard and VoiceOver users.

--------------------------------------------------------------------------------

## Implementation record (2026-11-14)

Open Questions, resolved:

* **Dwell**: 350 ms, `SidebarPeekController.dwellDelay`, a named constant to
  retune after a live look. OpenCode's 40 ms edge and 500 ms toggle figures
  are not adopted: the toggle path is excluded.
* **Corridor**: the edge strip, the overlay panel frame, and a 16 pt grace band past the panel's trailing edge, plus a 200 ms exit grace
  (`exitGrace`) so a boundary flicker or a strip-to-panel handoff does not
  dock the peek. Entry and exit across regions use a region count, not a flag.

Review decisions applied (docs/egm-reviews/opencode2-parity/OC-06):

* Overlay target: the panel only. The rail stays docked and is never
  duplicated; the live `SidebarPanelSurface` (extracted from `AppSidebarView`
  so docked and peeked share one view) is overlaid at x = rail width, below
  the top bar so the toggle stays uncovered. Trigger strip: 8 pt inside the
  rail's trailing edge, below the top bar. Peek is disabled when the section
  has no panel (`hasSidebarPanel`) or the sidebar is docked.
* Toggle entry: the strip starts below the top bar, so a pointer arriving
  from the toggle never starts a dwell. A toggle click calls
  `SidebarPeekController.cancel()` first, then the normal toggle runs.
* AppKit versus SwiftUI: the tracking is SwiftUI `onHover` regions (the repo
  already uses hover tracking). The controller is a plain state machine with
  an injected scheduler, so it tests headlessly. No NSTrackingArea.
* Suppression is pulled at dwell-fire time from AppKit, not pushed by
  RootView: mouse button down (covers mouse drags and mouse text selection),
  key-window sheet or app modal window, command center open (RootView flag),
  app inactive, and an event-tracking run loop (open menu). A button down or
  open menu also holds an open peek past the exit grace.
* Dismissal: pointer exit after grace, toggle click, chat switch, section
  switch, window or app deactivation, command center opening, docking state
  change. Interacting inside the peek (click a chat) ends it via the chat
  switch rule.
* Accessibility: the strip and overlay are `accessibilityHidden`. The toggle
  stays the keyboard and VoiceOver path. Motion: `.tsAnimation(TSMotion.pane,
  value: isPeeking)` with a leading move transition, instant under Reduce
  Motion through the existing modifier; the controller makes no motion
  decisions.
* Localization: no new strings.
* Shortcuts: none added, so no collision with OC-05's Cmd+Option+Left/Right.

Deferred or not done:

* Keyboard-driven text selection does not suppress a peek (no signal);
  mouse selection does, via the button-down check.
* Escape to dismiss a peek is not wired (the overlay is not focusable).
* The event-tracking run loop check for open menus and live pointer feel
  (dwell, grace values, shadow) were not exercised in the running app.
* Hover-peek for the right inspector and toggle-hover peek stay Non-Goals.

Files: `App/SidebarPeekController.swift`, `App/RootView.swift`,
`Chrome/AppSidebarView.swift`, `Tests/.../SidebarPeekControllerTests.swift`.

--------------------------------------------------------------------------------

*Drafted with EGM*
