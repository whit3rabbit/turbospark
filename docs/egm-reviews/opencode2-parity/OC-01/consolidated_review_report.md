# Consolidated review report: OC-01 keybind customization

Target: docs/designs/opencode2-parity/OC-01-keybind-customization.md (full file, 182 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. Two reviewers read only part of OPENCODE2_COMPARISON.md (inventive catalyst skipped section 3, stage 8 zoom-out skipped the file), and nobody fetched the OpenCode links.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 3, Major 11, Nit 10.

## Blocker

B1. Lines 40-41, 102-112, 141-144. The premise that the catalog is the source of chords is wrong: rows are display-only (`id` = label, glyph strings) and real chords are about 19 hard-coded `.keyboardShortcut` calls. This is a data-model rewrite, not "add ids". Fix: state it as a typed `KeyChord` (KeyEquivalent plus modifiers) with defaults stored once and glyphs derived; resize the effort. [ZO-1, DC-2, CO-5, S8-A1]
B2. Lines 42, 63-70, 137-158. Scope is undefined for chords outside the menu (Start Recording, composer, palette arrows) and for bridge alternates, so duplicate detection misses collisions. Fix: rebindable = menu commands only; show everything else as fixed rows and reserve those chords for conflict checks. [ZO-2, DC-3, DC-4, IC-4, CO-4, S8-A2]
B3. Whole document (gap near line 63-72 and 125). No Alternatives Considered section; settings.json is rejected at line 46 with no reason. Fix: add a short section (UserDefaults or settings.json, pane-only remap, AppKit key equivalents) with technical rejections, none attributed to OpenCode maintainers. [DC-1, ST-1, S8-A11]

## Major

M1. Lines 18, 43-44, 84-89, 102-103. "Unbind" conflicts with "restore the default"; `nil = unbound` is unreachable from the UI and nil versus absent is undefined. Fix: decide explicitly; the lean is no unbind (drop `KeyChord?`), or tri-state if freeing a chord is part of the pain. [ZO-3, IC-2, CO-1, DC-5, ST-3, SY-2, S8-A3]
M2. Lines 77-82. CUJ 1 Then clause says the previous holder "was refused" yet shows the new chord. Fix: one success CUJ and one refusal CUJ (duplicate, reserved). [ZO-3, CO-2, DC-6, SY-3, S8-A4]
M3. Lines 114-118, 125-132. SwiftUI shortcuts take `KeyEquivalent`, not a key code; layout, function keys, event swallowing during capture, Escape as cancel versus bindable, and the macOS 14 claim are unaddressed. Fix: store character plus modifiers, state layout dependence, name the supported key set, say how the monitor consumes the event, cite the deployment target. [ZO-5, IC-1, CO-13, DC-7, S8-A6, S8-C6]
M4. Lines 50-51, 119-123. Live apply has no mechanism or store placement. Fix: one app-level `@Observable` store, not vault or profile data, observed by `Commands`; follow the `AppearanceStore` pattern. [ZO-6, IC-1, DC-8, SY-1, S8-A7]
M5. Lines 45-47, 91-97. Persistence details missing: CUJ 3 toast says "settings were reset" but only keybinds reset; stale ids, hand-edited duplicate chords, and the failure path into the latched toast are unspecified. Fix: toast says shortcuts were reset; reuse the `AppJSONStore` path; ignore unknown ids. [CO-8, DC-8, S8-A8]
M6. Lines 52-54, 138-140, 163. The migration table has no use (ids are introduced here) and no format, owner or trigger. Fix: drop the goal and tests; add a schema `version` and tolerant decode via `TolerantDecoding.swift`. [ZO-4, IC-3, CO-6, DC-8, SY-4, S8-A12]
M7. Whole document. No Risks and Mitigations section. Fix: add a few bullets (monitor lifecycle, regression across about 19 call sites, layout dependence, id loss, reset-all reachable by mouse and menu). [DC-9, ST-2, S8-A10, S8-A11]
M8. Lines 63-72 vs 47-49, 109-111. Override scope and conflict policy are listed open but already decided in Goals, CUJ and plan. Fix: mark both decided and remove Open Questions. [ZO-8, CO-3, SY-5, ST-6, S8-A5]
M9. Lines 55-58. Reserved list is examples only; system shortcuts (Mission Control, Spotlight) are not detectable. Fix: name one static list and state the limitation. [ZO-7, CO-7, DC-7, IC-5, S8-A9]
M10. Lines 138-144. Ownership is circular (store keyed by catalog ids, catalog consults store; "only writable copy" versus "single inventory"). Fix: catalog holds ids and defaults with no store dependency; store holds overrides; a thin resolver is what Commands, bridge and pane read. [CO-15, S8-C2]
M11. Lines 50-51, 117, 127. Uncalibrated claims: live apply, "no launch cost", macOS 14 limitation. Fix: state the mechanism, delete or measure the latency claim, cite or soften. [SY-1, S8-C4]

## Nit

N1. Lines 119-123, 161-167. Tests: add mutation checks for duplicate and reserved refusal, exact `--filter` suites, a review-time grep that each command site uses `resolved(id)`; reword "mutation check". [DC-10, SY-10, CO-16]
N2. Lines 128-129, 172-178. "21 languages", "ten gates", DESIGN.md section 9 (section 7 covers shortcuts), xcstrings path unverified or unlinked. Fix: one verify-and-link pass. [DC-11, SY-7, SY-12, CO-10, ZO-9]
N3. Lines 5-8, 52-53, 67-69. OpenCode behaviors ("dedupes in the registry", migration step) asserted from unfetched links; "web palette" and "the model" unexplained. Fix: cite behavior or soften. [DC-12, SY-6, CO-11]
N4. Lines 28-33, 138-166. Swift links are repo-root-relative and do not resolve from this folder. Fix: use `../../../swift/...`. [ST-5]
N5. Lines 3-12. Metadata block is unlabeled and mixes prose and bullets. Fix: one format. [ST-4]
N6. Lines 149-152. "400-line guidance" is uncited and false (pane is 86 lines). Fix: justify the split by event-monitor lifecycle. [ZO-10, SY-9, CO-9, S8-C1]
N7. Lines 61, 123. "Key-holder sequences" is jargon. Fix: "multi-key sequences". [SY-11, CO-12]
N8. Lines 130-132. "Space to start capture" may clash with focus conventions; no VoiceOver text for an unbound row. Fix: default to Return or click. [SY-8, CO-14, S8-C5]
N9. Lines 40-41. Keep the id namespace as plain strings so OC-04 chords can join (INDEX lines 79-80). [S8-C3]
N10. Lines 45-46. Cite `DisabledItemStore`, `ImageStyleFavoriteStore`, `AppearanceStore` as the pattern. [ZO reuse note]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10 (zoom-out), IC-1 to IC-6 (inventive catalyst). Feeds B1, B2, M1, M3, M4, M6, M8, M9, N6.
- comprehension_report.md: CO-1 to CO-16. Feeds B1, B2, M1 to M3, M5, M6, M9, M10, N1 to N3, N7, N8.
- critic_report.md: DC-1 to DC-12 (design), ST-1 to ST-6 (structural), SY-1 to SY-12 (style). Feeds all Blockers and Majors, N1 to N8.
- intermediate_zoom_out_report.md: S8-A1 to S8-A12, S8-C1 to S8-C6. Confirms all earlier Blockers open. Recommends rejecting scope-adding asks (IC-2 tri-state and undo toast, DC-9 protected-command set, IC-1 layout spike, IC-6 VoiceOver copy) and keeping ambiguity-removal fixes.
- Merge notes: DC-1 and ST-1 merged into B3; DC-9 and ST-2 merged into M7 (ST rated Major, DC rated Major). Stage 8 rates the unbind question Major with a caveat: it becomes a tri-state if freeing a chord is in scope.
