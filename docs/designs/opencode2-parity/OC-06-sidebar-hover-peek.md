# OC-06: Sidebar hover peek

Finding: Section 9a, "Sidebar hover peek - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity target: OpenCode's
[sidebar-peek.ts](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/sidebar-peek.ts),
shipped in [PR 58156](https://github.com/anomalyco/opencode/pull/58156).

* Author: whit3rabbit
* Status: Draft
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

[RootView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift)
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

*Drafted with EGM*
