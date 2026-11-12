# Intermediate zoom out report (Stage 8): OC-05 back and forward history

Target: docs/designs/opencode2-parity/OC-05-back-forward-history.md (full file, 132 lines, unedited since Stage 5). Reviewer egm-zoom-out completed. Prior reports consulted: zoom_out_report.md, comprehension_report.md, critic_report.md.

Verdict: not ready. The root problem is real and a small pure `NavigationHistory` model is reasonable reuse. Two facts in the draft are wrong. The fixes the earlier stages ask for would turn a "small" item into a cross-cutting navigation refactor, so cut scope before fixing.

Re-verified in source: Cmd+[ and Cmd+] are bound to Previous/Next Chat (TurboSparkApp.swift lines 445, 453, disabled while `model.isRunning`; KeyboardShortcutCatalog.swift lines 81-88 with Cmd+Shift alternates; AlternateShortcutBridge.swift lines 30-35). Writers of `activeSection`, `selectPreviousChat`, `selectNextChat` and `openSettingsTab` appear in about 23 files; RootView holds 4.

## Findings

S8-1 BLOCKER. Lines 38-39, 83, 112-114. Chord collision (ZO-1, DC-1, CO-1, IC-5). Open. Needs a decision. Simplest: do not claim Cmd+[ and Cmd+]. Ship Back/Forward with a different default chord or none, leaving the catalog contract and chat cycling untouched. This also removes the OC-01 and OC-04 coordination and the composer indent conflict.

S8-2 BLOCKER. Lines 26-28, 77, 106-108. Wrong owner for recording (ZO-2, DC-2, IC-1). Open. History must live on AppModel; a RootView-owned stack cannot drive menu enablement. Reject a full inventory and a `navigate(to:)` funnel over about 23 files. Smaller fix: one observer of `activeSection` and the selected chat on AppModel, with a replay flag so restore does not re-record; decide once whether programmatic switches (LoadGate) record.

S8-3 MAJOR. Lines 34-37 vs 50-51, 86. Scope contradiction (ST-4, SY-5, ZO-3). Open. Settings tab state is `@State` in AppSettingsView, not model state. Cut it: an entry is (section, selected chat id), and the collapse rule is "consecutive settings section entries are one entry". This removes the which-pane ambiguity (CO-2, IC-3).

S8-4 MAJOR. Lines 36-37, 55-68. Undefined semantics (CO-3, CO-5, IC-2, DC-4). Open, fix minimally: a fresh push truncates forward entries; re-selecting the current target does not push; restoring to a deleted chat or unavailable section is skipped; disable during `model.isRunning` like Previous/Next Chat. Reject an injected resolver and pruning machinery.

S8-5 MAJOR. Lines 44-45, 96, 126-128. Tooltips on menu items (ZO-5, DC-5, IC-4, SY-4). Open. Delete the tooltip goal and the accessibility bullet that restates system behavior.

S8-6 MAJOR. Whole document. Missing Alternatives, Risks and Open Questions (ST-1, ST-2, ST-3, DC-3, DC-6). Open. Add them at a few lines each: AppModel-owned stack over per-view state because chord and menu enablement are app level; risks limited to the chord decision and missed writers.

S8-7 MAJOR. Sizing and scope (ZO-6, ST-10). Open. After S8-1 to S8-3 the item stays small. Adopting per-tab or per-window scope with OC-04 and ZP-21 would make it not small. Decide now: one global history for the single main window, and say OC-04 and ZP-21 extend the entry later.

S8-8 NIT. Lines 92-93, 117-120. Delete "O(1)" and "never blocks"; no ring buffer. Move or drop the mutation-check sentence (ZO-8, DC-6, SY-8, CO-10).

S8-9 NIT. Lines 3-9, 18-22. Pin upstream links to commit `7ce35879b8` or mark them unverified (SY-2, DC-7).

S8-10 NIT. Lines 4, 27. `swift/...` link does not resolve; use `../../../swift/...` (ZO-7, DC-8, ST-8).

S8-11 NIT. Line 42. State the 100 cap is borrowed from upstream (ZO-9).

S8-12 NIT. Lines 86-88. Uncited claim that per-section state preserves scroll and composer; cite or drop (CO-4, SY-7).

S8-13 NIT. Lines 80-83. Remove the AlternateShortcutBridge box; state once that the group sits beside the existing `CommandGroup(before: .toolbar)` View menu group near TurboSparkApp.swift line 324 (ZO-4, ST-6, SY-6, CO-8).

## Fixes to reject as scope growth

* Global `navigate(to:)` funnel over every writer (DC-2 inventory, IC-1 wide diff).
* Injected chat resolver, pruning and derived-enable machinery (IC-2).
* Ring buffer (DC-6).
* VoiceOver verification step (SY-3) and extra CUJs for trim and disabled ends (ST-5, DC-9). One CUJ for push-after-back is enough.
* Per-tab or per-window history design beyond one deferring sentence.
* Settings tab id in entries (remove it instead).
* Context-sensitive chord rewrite of the catalog (ZO-1 option); prefer not claiming the chords.

Still open: all Stage 5 to 7 findings; the file was not changed. Comprehension items CO-6, CO-7, CO-9 and CO-11 are jargon nits and need no action beyond CO-8.
