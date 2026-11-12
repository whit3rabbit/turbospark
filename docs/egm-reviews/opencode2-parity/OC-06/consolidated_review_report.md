# Consolidated review report: OC-06 sidebar hover peek

Target: docs/designs/opencode2-parity/OC-06-sidebar-hover-peek.md (full file, 124 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. Stage 7 structural and style first hit "Concurrent subagent limit reached" and were retried one at a time; all reviewers completed, none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 9, Nit 10.

## Blocker

B1. Whole document. No Alternatives Considered section (standard's Alternatives Rule); SwiftUI hover, hot corner, popover and do-nothing are never weighed. Fix: add the section with a concrete rejection reason for each. [DC-1, ST-1, S8-5]
B2. Whole document (near lines 87-94). No Risks and Mitigations section: boundary flicker, hit-test and resize conflicts, focus theft, stuck-open peek, state divergence after a toggle click, z-order, accidental triggers. Fix: add the section with a mitigation per risk. [DC-6, ST-2, S8-5]

## Major

M1. Lines 83-84, 70-76. "SwiftUI has no pointer-corridor tracking" is unsupported and is the sole justification for AppKit; the codebase uses `.onContinuousHover` (BrowserPaneView.swift:289). Fix: justify concretely or switch to a SwiftUI hover region plus a pure state machine. [ZO-3, IC-3, CO-11, DC-2, SY-9, S8-1]
M2. Lines 17, 26-33, 36, 68-81. Overlay target and trigger strip undefined: AppSidebarView is a permanent rail plus a panel gated by `isPanelVisible`; "collapsed" means panel hidden. Fix: overlay the panel with the rail docked, trigger on the rail's trailing edge, state strip width, disable when `!hasSidebarPanel`, name the forced-visible params and `.zIndex(10)`/`.clipped()` interaction. [ZO-2, IC-2, CO-6, DC-3, S8-3]
M3. Lines 40-45, 96-112. Suppression inputs have no named source (drag, sheet, dialog, text selection) and RootView pushing them couples systems; a tracking area gets no mouseEntered during a drag. Fix: pull from named AppKit or model sources at dwell-fire time behind an injectable protocol, each with scope and failure behavior. [ZO-4, IC-1, CO-3, DC-5, SY-5, S8-2]
M4. Lines 36-45, 96-112. Dismissal rules missing: interactive peeked panel (menus, rename, focus), window deactivation, full screen, Escape, click-to-select, and a toggle click during a peek versus "never changes isSidebarExpanded". Fix: add a state and transition table with a hold-open rule and "toggle click cancels peek, applies normal toggle". [IC-4, CO-12, DC-4, S8-4]
M5. Lines 40-42, 18-19. "Enters through the toggle control" undefined (toggle is in TopBarView), and it inverts OpenCode's toggle-hover peek without saying so. Fix: say where the toggle sits and how entry path is known; list toggle-hover as an excluded Non-Goal. [CO-4, SY-2, ZO-4]
M6. Lines 55-57, 87-94. "Accessibility unchanged" is unsupported; overlay controls enter the focus and VoiceOver tree and vanish on exit. Fix: state the overlay is hidden from the accessibility tree and key-view loop, or define focus handling. [CO-7, DC-7, IC-8, SY-6, S8-6]
M7. Lines 26-27. RootView link is relative to the doc folder and broken. Fix: `../../../swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift`. [ZO-1, IC-5, CO-8, DC-8, ST-4, S8-7]
M8. Lines 36-37, 61-66. Dwell is a goal and an open question; "corridor" is defined only by an open question; OpenCode figures uncited. Fix: "initial 350 ms, tunable, named constant, set in review", define corridor as sidebar frame plus grace band, cite sidebar-peek.ts. [ZO-6, IC-7, CO-1, CO-2, DC-9, ST-5, SY-4]
M9. Lines 29-32, 17-22. The "motion polish" quote drops "rather than lost information", and "can never drift" is absolute. Fix: quote exactly; say "renders the same sidebar view as the docked state". [SY-1, SY-3, S8-10]
M10. Lines 70-81. Controller to RootView mechanism (publisher, binding, callback) and overlay frame and z-order unnamed. Fix: name the publish mechanism and layering. [CO-5]
M11. Lines 59-68. Open Questions precede the Technical Plan. Fix: move after the NFR or plan sections, matching sibling drafts. [ST-3]

## Nit

N1. Lines 44-45, 78-79, 101, 111. Reduce Motion is already central via `.tsAnimation(TSMotion.pane, value:)`; controller should not make motion decisions. Fix: drive the overlay with `.tsAnimation` on a peek-state value. [ZO-5, IC-6, S8-8]
N2. Lines 87-92. "One tracking area" conflicts with a corridor region. Fix: list the regions. [SY-10, S8-11]
N3. Lines 8-11, INDEX line 47. "small" label understates the plan. Fix: relabel small-medium or simplify. [IC-9, SY-13, S8-12]
N4. Lines 3-7, 32. Section 9a is an unanchored link and the only source. Fix: name the exact sub-heading and add one inline line. [CO-9, ST-7, S8-13]
N5. Lines 20, 79, 115-117. `TSMotion.pane`, `.appThemed()`, `make swift-lib`, `make design-check` undefined. Fix: gloss or link; confirm Makefile targets. [CO-10, ST-8, SY-12, S8-13]
N6. Lines 100-116. Verification has no pass criteria and the state machine states are not listed. Fix: add observable criteria and a state table. [CO-12, CO-13, DC-9]
N7. Lines 100-116. List styles inconsistent; no localization or design-check file in Target Files. Fix: align with sibling drafts. [ST-6]
N8. Lines 18-19, 30-32, 44-45. "careful" subjective, description repeated, "keeping the behavior" has no referent. Fix: trim and reword. [SY-7, SY-8]
N9. Lines 91-92. Tooltip claim uncited. Fix: "peek adds no strings" or cite source. [SY-11]
N10. Lines 36, 40. Text selection suppression has no signal and is the hardest input. Fix: defer text selection from v1 or name the signal. [ZO-4]
