# Stage 7: Critic report, OC-06 sidebar hover peek

Target: docs/designs/opencode2-parity/OC-06-sidebar-hover-peek.md (full file). Standard: feature-spec-standard for design and structural. One round only; convergence not assessed. The first dispatch of structural and style hit the concurrent subagent limit and was retried one at a time. All three reviewers completed.

## Design (DC)

Verdict: not ready. 1 Blocker, 5 Major, 3 Nit.

DC-1 Blocker, whole draft. No Alternatives Considered section (standard's Alternatives Rule). Rivals never weighed: SwiftUI onHover/onContinuousHover, hot-corner button, popover preview, do nothing. Fix: add the section with a concrete rejection reason for each.

DC-2 Major, lines 83-84. "SwiftUI has no pointer-corridor tracking" is the sole justification for AppKit and is unbacked. Fix: justify concretely or drop the AppKit layer.

DC-3 Major, lines 26-33, 68-81 vs RootView 66-70. AppSidebarView is always in the HStack: a permanent icon rail plus a panel gated by `isPanelVisible` (`isSidebarExpanded && activeSection.hasSidebarPanel`). "Collapsed" means panel hidden, and the "leading edge" is where the rail sits. Not stated: strip location, forced-visible mode for the same view, behavior when `hasSidebarPanel` is false. Fix: restate against rail-plus-panel, name the strip, add a no-panel suppression and the exact params passed.

DC-4 Major, lines 36-45, 96-112. Dismissal while the peeked panel is interactive is unspecified (context menus, inline rename, drag reorder, search focus), as are window deactivation, full screen, pointer leaving the window, Escape, and click-to-select. Fix: add a dismissal state table and a hold-open rule while a menu, rename or focus is active.

DC-5 Major, lines 40-42, 112. Suppression inputs have no source; a tracking area does not normally deliver mouseEntered during a drag; sheets are model state (two `.sheet` bindings in RootView). Fix: list each input with its source and failure behavior.

DC-6 Major, lines 87-94. No Risks and Mitigations section: accidental triggers en route to traffic lights, overlay covering the transcript or composer mid-typing, z-order against toasts and the root `.zIndex(10)` and `.clipped()`, the tracking view stealing clicks, timer races. Fix: add the section.

DC-7 Nit, lines 87-94. "Accessibility unchanged" is doubtful: overlay controls enter the focus and VoiceOver tree and vanish on dismiss, which can strand focus. Fix: state that the overlay is hidden from the accessibility tree and key-view loop, or define focus handling.

DC-8 Nit, line 26. Broken RootView link. Fix: `../../../swift/TurboSparkApp/Sources/TurboSparkApp/App/RootView.swift`.

DC-9 Nit, lines 59-66, 36. OpenCode timings uncited, 350 ms is both goal and open question, no measurable success criteria. Fix: cite sidebar-peek.ts, mark 350 ms provisional, add acceptance criteria.

## Structural (ST)

Heading hierarchy is clean, the code fence is closed. Verdict: do not pass.

ST-1 Blocker, whole document. No Alternatives Considered section. Fix: add an H2 covering pure SwiftUI .onHover, a duplicate peek view, and click-only toggle or no peek, each with a concrete rejection reason.

ST-2 Blocker, whole document. No Risks and Mitigations section (standard requires failure modes and mitigations). Likely risks: flicker at the corridor boundary, tracking area conflicting with window-edge resize or drag-region hit testing, overlay stealing focus, stuck-open peek when a suppression input is missed, controller state diverging from `isSidebarExpanded` after a toggle during peek. Fix: add the section.

ST-3 Major, lines 59-68. Open Questions precede the Technical Plan. Fix: move after the NFR or implementation sections, matching sibling drafts.

ST-4 Major, line 26. Broken relative link. Fix: as DC-8.

ST-5 Nit, line 36 vs 61-63. 350 ms goal vs open question. Fix: mark provisional.

ST-6 Nit, lines 100-116. Numbered lists with nested bullets mixed with plain bullets elsewhere; no localization or design-check file in Target Files. Fix: keep list style consistent across OC drafts.

ST-7 Nit, lines 3-7. Unanchored link to Section 9a. Fix: name the sub-heading exactly or add an anchor.

ST-8 Nit, lines 115-117. `make swift-lib` and `make design-check` undefined in the doc. Fix: link the verification doc.

## Style (SY)

No Blockers.

SY-1 Major, lines 29-32. "its report calls the gap motion polish over real content" paraphrases the source and drops the second half of its verdict. Fix: quote "motion polish rather than lost information" or drop the attribution.

SY-2 Major, lines 17-22, 36-38, 42. The source says peek triggers from the leading edge or the sidebar toggle (40 ms edge, 500 ms toggle). The draft covers edge only and line 42 inverts OpenCode's toggle behavior without saying so. Fix: add a Non-Goal or Open Question stating toggle-hover peek is intentionally excluded and why.

SY-3 Major, lines 17-22. "can never drift" is absolute; reusing one view prevents view drift only, not state, width or focus differences. Fix: "so the peek renders the same sidebar view as the docked state".

SY-4 Major, lines 36, 61-63. Goal states "about 350 ms" while it is undecided; "leans comfortable" is subjective. Fix: "initial value 350 ms, tunable, final value set in review".

SY-5 Major, lines 41-43, 104-106. Suppression inputs have no named source; "text selection" scope is ambiguous. Fix: define each input in one line.

SY-6 Major, lines 55-57, 93-94. Accessibility "unchanged, hides none" is asserted while the overlay exposes sidebar controls to VoiceOver; focus handling is unaddressed. Fix: state accessibility and focus behavior while peeking.

SY-7 Nit, lines 18-19, 30-32. "careful" is subjective and the dwell/corridor/suppression description repeats. Fix: delete one.

SY-8 Nit, lines 44-45. "keeping the behavior" has no referent. Fix: "the overlay appears and dismisses without animation; dwell and dismissal rules are unchanged."

SY-9 Nit, lines 83-85. Unsupported capability claim (SwiftUI has onHover and onContinuousHover). Fix: drop the claim or cite the limitation.

SY-10 Nit, lines 87-92. "One tracking area" conflicts with a corridor needing a second region. Fix: say "tracking areas: leading strip and corridor region".

SY-11 Nit, lines 91-92. Tooltip claim uncited. Fix: cite the source file or say "peek adds no strings".

SY-12 Nit, line 116. Makefile targets not confirmed in the doc. Fix: cite them.

SY-13 Nit, lines 11-12, INDEX line 47. INDEX labels the draft "small" despite a state machine, AppKit shell and six suppression inputs. Fix: relabel or limit suppression inputs.
