# Consolidated review report: OC-05 back and forward history

Target: docs/designs/opencode2-parity/OC-05-back-forward-history.md (full file, 132 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. The style reviewer hit one "Concurrent subagent limit reached" error and succeeded on the first retry.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 8, Nit 11.

## Blocker

B1. Lines 38-39, 83, 112-114. Cmd+[ and Cmd+] are already bound to Previous/Next Chat (TurboSparkApp.swift 445, 453; KeyboardShortcutCatalog.swift 81-88; AlternateShortcutBridge.swift 30-35); "no extra chords needed" is wrong. Fix: do not claim these chords; ship Back/Forward with another default or none and leave chat cycling untouched. [ZO-1, IC-5, CO-1, DC-1, S8-1]
B2. Lines 26-28, 77, 106-108. Recording point is wrongly assigned to RootView; `activeSection` lives on AppModel with about 23 writer files, Commands cannot read a RootView stack, and restores can re-record and wipe the forward stack. Fix: own history on AppModel, record from one observer, add a replay flag, decide whether programmatic switches record. [ZO-2, IC-1, DC-2, S8-2]
B3. Whole document. No Alternatives Considered section (standard's Alternatives Rule). Fix: add two lines: AppModel-owned stack chosen over per-view state or NavigationStack because chord and menu enablement are app level. [ST-1, DC-3, S8-6]
B4. Whole document (NFR lines 90-97 only). No Risks and Mitigations section. Fix: add a short section limited to the chord decision, missed writers, stale entries and tab composition. [ST-2, DC-6, S8-6]

## Major

M1. Lines 34-37 vs 50-51, 86. Goals record chat and settings-tab selections but Non-Goals say section level only; settings tab state is a `@State` in AppSettingsView, and "section" is never defined. Fix: entry = (section, chat id) only; drop `settingsTabID`; define "section-level" once. [ZO-3, ST-4, SY-5, DC-2, CO-7, S8-3]
M2. Lines 41-42, 55-61. Collapse rule undefined (which pane survives, mid-run navigation). Fix: state "consecutive settings entries are one entry, mutated in place". [CO-2, IC-3, DC-4, S8-3]
M3. Lines 18-20, 34-35, 63-68. Undefined semantics: forward truncation on a new push, duplicate push, deleted or unavailable target, ghost chats, behavior while `model.isRunning`. Fix: three sentences (truncate, no duplicate push, skip invalid on restore) and disable while running; no resolver machinery. [CO-3, CO-5, IC-2, DC-4, S8-4]
M4. Lines 44-45, 96-97, 107, 126-128. Menu items do not show tooltips; menu enablement cannot come from RootView. Fix: drop the tooltip and the accessibility restatement; expose can-go state on AppModel. [ZO-5, IC-4, DC-5, SY-4, CO-11, S8-5]
M5. Lines 53-68. No Open Questions section although INDEX requires one for multi-answer decisions; CUJs cover only happy paths. Fix: add Open Questions; one extra CUJ for push-after-back. [ST-3, ST-5, S8-6]
M6. Lines 70-84, 109-114. Architecture omits the restore path and data model; AlternateShortcutBridge is unexplained; chord registration is described twice (items 3 and 4); View menu placement unstated. Fix: add restore flow, drop the bridge box, split items 3 and 4, name the group beside `CommandGroup(before: .toolbar)` near TurboSparkApp.swift line 324. [ZO-4, ST-6, ST-7, SY-6, CO-8, S8-13]
M7. Cross-draft and INDEX lines 46, 93. Interaction with OC-04, ZP-21 and OC-01 unstated; per-window or per-tab scope open. Fix: one global history for the single main window; OC-04 and ZP-21 extend the entry later. [ZO-6, IC-6, ST-10, S8-7]
M8. Lines 26-30, 18-22, 86-88. Evidence calibration: universal negative "no back-stack anywhere", upstream claims linked to `dev` rather than pinned commit `7ce35879b8`, and an uncited claim that per-section state preserves scroll and composer. Fix: cite the audit date and commit; cite or drop the preservation claim. [SY-1, SY-2, SY-7, DC-7, CO-4, S8-9, S8-12]

## Nit

N1. Lines 4, 27. The `swift/...` link does not resolve from the docs folder. Fix: use `../../../swift/...`. [ZO-7, DC-8, ST-8, S8-10]
N2. Lines 92-93. "O(1) push and trim" is false for an Array and "never blocks" is trivially true. Fix: delete both; no ring buffer. [ZO-8, DC-6, SY-8, CO-10, S8-8]
N3. Lines 117-120. Mutation-check procedure sits in a rationale bullet. Fix: move to the Verification Plan. [SY-8, ZO-8, S8-8]
N4. Line 42. The 100 cap is copied from upstream. Fix: say so. [ZO-9, S8-11]
N5. Lines 94, 124-125. "21 languages", "localization gates", `make` targets are project jargon. Fix: reference INDEX program constraints. [ZO-8, CO-9]
N6. Lines 96-97, 122-128. Accessibility requirement has no verification step. Fix: drop the requirement (system behavior); a VoiceOver step is scope growth. [SY-3]
N7. Lines 3-9. Free-text "Finding:" and "Parity targets:" lines precede the metadata bullets; OpenCode is never introduced. Fix: confirm against the template; add one defining clause. [ST-9, CO-6]
N8. Lines 20-21, 63-68. CUJ 2 does not define "prior section". Fix: restate in terms of the (section, chat) entry. [DC-9]
N9. Lines 26-27, 50-51, 106-108. "Section" never listed. Fix: list the sections once or link to the enum. [CO-7]
N10. Lines 98-130. No dependency or sequencing note on OC-04. Fix: add one line. [ST-10]
N11. Line 38. "System View menu" asserted without saying a View menu exists. Fix: name the existing group (see M6). [CO-8]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-9 (zoom-out), IC-1 to IC-6 (inventive catalyst). Feeds B1, B2, M1 to M4, M6, M7, N1 to N4.
- comprehension_report.md: CO-1 to CO-11. Feeds B1, M1 to M4, M6, M8, N2, N5, N7 to N11.
- critic_report.md: DC-1 to DC-9 (design), ST-1 to ST-10 (structural), SY-1 to SY-8 (style). Feeds all Blockers, all Majors, N1 to N10.
- intermediate_zoom_out_report.md: S8-1 to S8-13. Confirms all Stage 5 to 7 findings open and recommends cutting scope first. Rejected as scope growth: a global `navigate(to:)` funnel over every writer, an injected chat resolver with pruning, a ring buffer, a VoiceOver step, extra CUJs for trim and disabled ends, per-tab history design, settings tab id in entries, and a context-sensitive catalog rewrite.
