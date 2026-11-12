# Stage 5: Initial zoom out report, OC-06 sidebar hover peek

Target: docs/designs/opencode2-parity/OC-06-sidebar-hover-peek.md (full file, 124 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom out (ZO)

Context is sufficient and the root problem is clear. No Blockers.

ZO-1 Major, lines 26-27. The RootView link is relative to the doc folder and resolves to a nonexistent path; the file exists at the repo root path. Fix: use `../../../swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift`.

ZO-2 Major, lines 77-85. The plan says "live sidebar view", but RootView hosts `AppSidebarView(model:isPanelVisible:isPanelPreferred:)`, which is the icon rail plus the section panel. The doc does not say what is overlaid, nor what happens when `isPanelVisible` is false (rail-only sections where `hasSidebarPanel` is false). The existing `.zIndex(10)` and `.clipped()` on the root are not mentioned. Fix: state the panel half is overlaid with the rail docked, and peek is disabled when the section has no panel.

ZO-3 Major, lines 70-76, 83-84. "SwiftUI has no pointer-corridor tracking" is unsupported; the codebase already uses `.onContinuousHover` (BrowserPaneView.swift:289) and `.onHover`. A SwiftUI hover region plus a pure state machine may avoid an AppKit bridge. Fix: justify NSTrackingArea against `.onContinuousHover`, or switch.

ZO-4 Nit, lines 41-43. Text selection and drag suppression have no named signal source; "enters through the toggle" is undefined since the toggle is in TopBarView, not at the leading edge. Fix: name a signal for each or defer text selection.

ZO-5 Nit, lines 44-45, 78-79, 101, 111. Reduce Motion is already handled centrally by `.tsAnimation(TSMotion.pane, value:)` and `effectiveReduceMotion`; the controller should not make motion decisions. Fix: say the overlay uses `.tsAnimation` on a peek-state value.

ZO-6 Nit, lines 36, 61-63. The 350 ms dwell is a Goal and also an Open Question. Fix: write the goal as "a dwell, value set in review (lean 350 ms)".

Reuse check: `TSMotion.pane`, `tsAnimation`, and the `@AppStorage` `isSidebarExpanded` flag exist as described. No existing peek component duplicates this.

## Inventive catalyst (IC)

No Blockers.

IC-1 Major, lines 36-37, 70-81, 100-106. RootView is made to feed drag, sheet, dialog and text-selection inputs to the controller with no named source. This couples unrelated systems and goes stale when a new modal surface is added. Resolution (inversion): the controller pulls suppression from AppKit sources at dwell-fire time (`attachedSheet`, `NSEvent.pressedMouseButtons`, `NSApp.modalWindow`, first responder text selection) behind an injectable protocol. Verify that SwiftUI `.sheet` and `.confirmationDialog` appear as attached sheets.

IC-2 Major, lines 17, 36, 40, 70-72 vs RootView 66-70. The icon rail is always visible; only the panel collapses. The "leading edge" trigger strip is undefined (over the rail, or at the panel edge) and peek in a panel-less section reveals nothing, where the toggle is also hidden. Fix: trigger is the rail's trailing edge, overlay is the panel only, disabled when `!hasSidebarPanel`; add this to the test list.

IC-3 Major, lines 83-84. Same unsupported SwiftUI claim as ZO-3. A union hover region (overlay plus grace band) and a cancellable `Task.sleep` dwell would be simpler. Hover flicker on overlay insertion needs a grace band and hysteresis either way.

IC-4 Major, lines 40-43. Toggle suppression and overlay lifetime are not reconciled: the doc omits what a toggle click during a peek does, and "never changes isSidebarExpanded" conflicts with a click during a peek. Fix: a toggle click cancels the peek and applies the normal toggle with no double animation.

IC-5 Nit, line 26. Broken RootView link (same as ZO-1).

IC-6 Nit, lines 44-45, 78-79. Reduce Motion has two sources (system and app setting through `AppearanceManager`). Name `.tsAnimation` and `TSMotion.animate` instead of a bespoke instant settle.

IC-7 Nit, lines 36, 61-63. Dwell should be a named constant, not a literal.

IC-8 Nit, lines 87-94. "Accessibility unchanged" overlooks focus and hit-testing after an overlay vanishes on pointer exit, and the overlay covering the content's leading hit region.

IC-9 Nit, lines 8-11, 100-102 vs INDEX line 47. INDEX labels the size "small"; a new controller, state machine and AppKit shell is small-medium.
