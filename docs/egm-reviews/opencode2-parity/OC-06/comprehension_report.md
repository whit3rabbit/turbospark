# Stage 6: Comprehension report, OC-06 sidebar hover peek

Target: docs/designs/opencode2-parity/OC-06-sidebar-hover-peek.md (full file). Reviewer: egm-comprehension (CO), read the document alone. Completed.

## What this proposes

When the sidebar is collapsed, resting the mouse pointer at the window's left edge for about 350 ms would slide the real sidebar in over the content as a temporary overlay. It would dock back out when the pointer leaves a "hover corridor" (the region around the sidebar). The peek would be suppressed during drags, sheets, dialogs, text selection, and when the pointer arrives via the toggle button. It would be built as an AppKit controller (tracking area, timer, state machine) that tells a SwiftUI overlay in RootView to show the existing sidebar view. Reduce Motion users get instant show and hide, and the toggle and the stored expanded flag stay untouched.

## How the existing system works, per this document

RootView.swift holds an `isSidebarExpanded` flag, persisted via `@AppStorage`, and the toggle flips it with a layout animation that reflows the whole window. When collapsed there is no way to glance at the sidebar without that reflow. The document says OpenCode (a different product) already has a peek feature, built with dwell timers, corridor tracking, and suppression states. The document does not describe how the current sidebar view is constructed, how the toggle control is laid out, or how sheets, dialogs, drags, and text selection are currently detected.

## Findings

CO-1 Major, lines 36-37, 61-62. Dwell is stated as a goal (about 350 ms) and left open; the OpenCode 40 ms edge and 500 ms toggle figures confuse because toggle dwell has no meaning here (toggle entry suppresses peek). Fix: state one default and label the OpenCode figures as reference.

CO-2 Major, lines 39, 64-66. "Hover corridor" is used in Goals and the architecture but defined only by an open question, so the dismissal rule is unspecified. Fix: define the corridor (sidebar frame plus grace band) and say which pointer positions dismiss.

CO-3 Major, lines 42-43, 105-106. Text selection, dialogs, sheets and drags have no detection mechanism, and it is unclear whether text selection means any view or only chat. Fix: list each suppression signal with its source and scope.

CO-4 Major, lines 18-19, 42-43. "Pointer enters through the toggle control" is unexplained: the toggle's location relative to the leading strip and how entry path is tracked are not stated. Fix: say where the toggle sits and how the controller knows the pointer came from it.

CO-5 Major, lines 70-81. The controller to RootView flow is only an arrow: no mechanism for `peekRequested` and `peekDismissed` (ObservableObject, binding, callback), no overlay frame or z-order. Fix: name the publish mechanism, frame and layering.

CO-6 Major, lines 36, 60. The width of the leading strip that triggers dwell is never given. Fix: specify it in points.

CO-7 Major, lines 55-57, 92-94. The Accessibility section says "unchanged" while the overlay renders the real sidebar, which adds focusable and VoiceOver-visible elements while peeking. Fix: state whether the overlay is hidden from or included in the accessibility tree and focus order.

CO-8 Nit, lines 26-27. RootView link is relative to the doc folder and broken. Fix: `../../../swift/...`.

CO-9 Nit, lines 3-4, 32. Section 9a is the only source for the "missing" claim and the "motion polish" quote. Fix: add one summarising line inline.

CO-10 Nit, lines 20, 36, 79, 116-117. `TSMotion.pane`, `.appThemed()`, `make swift-lib`, `make design-check` are used without gloss or link. Fix: add one-line glosses.

CO-11 Nit, lines 82-84. "SwiftUI has no pointer-corridor tracking" is unsupported. Fix: add a reason or soften.

CO-12 Nit, lines 100-101, 109-111. State machine names and transitions are never listed, so tests cannot be checked against the design. Fix: add a small state and transition table.

CO-13 Nit, lines 117-120. Manual checks have no pass criteria. Fix: add observable criteria (no reflow, overlay within dwell time).
