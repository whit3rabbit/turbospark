# Stage 8: Intermediate zoom out report, OC-06 sidebar hover peek

Target: docs/designs/opencode2-parity/OC-06-sidebar-hover-peek.md (full file, unchanged since Stage 5). Reviewer: egm-zoom-out (S8). Completed.

Verdict: context is sufficient and the root problem is real. No architecture-level Blocker. The concept (additive, pointer-only, overlay of the real sidebar, docked flag untouched) holds. The AppKit layer, the RootView-pushes-suppression coupling, and the vague overlay target (rail vs panel) do not hold.

Stage 5 findings confirmed: the unsupported AppKit justification (ZO-3) is confirmed by DC-2, SY-9 and CO-11; the overlay target (ZO-2) is confirmed by DC-3 and IC-2. New from later stages: missing Alternatives and Risks sections, dismissal of an interactive peeked panel, controller-to-RootView mechanism, strip width.

S8-1 Major, lines 70-85. Over-built: the AppKit controller rests on a false claim, and the codebase already uses `.onContinuousHover` and `.onHover`. Fix: use a SwiftUI hover region, a cancellable `Task.sleep` dwell and a pure state struct, or give a concrete reason SwiftUI cannot do it.

S8-2 Major, lines 77-81, 100-106. Wrong dependency direction: RootView feeds drag, sheet, dialog and selection inputs and goes stale when a new modal surface appears. Fix: pull suppression at dwell-fire time behind one injectable protocol with named sources (attached sheet, pressed mouse buttons, modal window, first responder).

S8-3 Major, lines 26-33, 68-81. Overlay target mis-stated. Fix: overlay the panel half of `AppSidebarView` with the rail docked, trigger on the rail's trailing edge, disable when `!hasSidebarPanel`, and state interaction with `.zIndex(10)` and `.clipped()` (RootView lines 70, 110).

S8-4 Major, lines 40-45, 96-112. Toggle click during a peek and dismissal of an interactive panel are unspecified; "never changes isSidebarExpanded" conflicts with a click during peek. Fix: add a state and transition table; a toggle click cancels the peek and applies the normal toggle; hold open while a menu, rename or focus is active.

S8-5 Major, whole file. No Alternatives Considered or Risks and Mitigations section (Critic Blockers, downgraded at this altitude). Fix: add both, with SwiftUI-only hover as the first alternative.

S8-6 Major, lines 55-57, 92-94. "Accessibility unchanged" while the overlay adds focusable controls that vanish on exit. Fix: state the overlay is hidden from the accessibility tree and key-view loop, or define focus handling.

S8-7 Major, lines 26-27. Broken RootView link. Fix: `../../../swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift`.

S8-8 Nit, lines 44-45, 78-79, 101, 111. Reduce Motion is handled by `.tsAnimation(TSMotion.pane, value:)`. Fix: drive the overlay with `.tsAnimation` on a peek-state value; drop the bespoke instant-settle test.

S8-9 Nit, lines 36, 61-63, 42. 350 ms is both goal and open question; line 42 inverts OpenCode's toggle-hover peek without saying so. Fix: named provisional constant; state toggle-hover is excluded.

S8-10 Nit, lines 17-22. "can never drift" is overclaimed; the "motion polish" quote drops "rather than lost information". Fix: soften and quote exactly.

S8-11 Nit, lines 87-92. "One tracking area" conflicts with needing a corridor region. Fix: state the regions.

S8-12 Nit, lines 8-11 and INDEX line 47. "small" label understates the plan (small-medium); becomes defensible again if the SwiftUI-only route is adopted. Fix: relabel or simplify.

S8-13 Nit, lines 3-7, 116-117. Link the exact 9a sub-heading (OPENCODE2_COMPARISON.md line 486); gloss `make swift-lib` and `make design-check`. Fix: add anchors and glosses.

Reuse check: `TSMotion.pane`, `.tsAnimation`, `isSidebarExpanded` and `.onContinuousHover` exist. No existing peek component duplicates this.
