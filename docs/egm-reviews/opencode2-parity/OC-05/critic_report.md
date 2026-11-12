# Critic report (Stage 7): OC-05 back and forward history

Target: docs/designs/opencode2-parity/OC-05-back-forward-history.md (FeatureRange = full file, lines 1-132). One round only, convergence not assessed. Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, standard feature-spec-standard), egm-style (SY). All completed. The style spawn hit "Concurrent subagent limit reached" once and succeeded on the first retry.

## Design critic (DC): 2 Blocker, 4 Major, 3 Nit

DC-1 BLOCKER. Lines 38, 83, 109-113. Cmd+[ and Cmd+] are taken: TurboSparkApp.swift:445 and :453 bind Previous/Next Chat, KeyboardShortcutCatalog.swift lines 81-88 list them with Shift+Cmd alternates, AlternateShortcutBridge.swift:31-34 registers the alternates. "No extra chords needed" is wrong. Decide: remap chat cycling, pick other chords, or merge the concepts; address OC-01 and OC-04 (which also want these chords).

DC-2 BLOCKER. Lines 26-28, 34-37, 77. The navigation model is unverified. Section state may not live in RootView; settings tab state is `@State selectedTab` in AppSettingsView.swift (line 70); `openSettingsTab` and `AppModel.openSettings` exist. The doc does not name the section type, where section, chat and settings tab are held, or how many writers each has. No guard stops a restore from re-recording and wiping the forward stack. Add an inventory of transition sources (sidebar, settings, search jump, selectPreviousChat/selectNextChat, createChat, ghost chat, deep links) and a single choke point plus replay guard.

DC-3 MAJOR. Lines 16-128. No Alternatives section: SwiftUI NavigationStack, AppModel-owned history, NotificationCenter recording, no collapse rule, pick-from-list history.

DC-4 MAJOR. Lines 36-37, 52-68. Semantics under-specified: deleted chat, ghost or temporary chat, back/forward while `model.isRunning` (Previous/Next Chat are disabled then), forward-stack truncation on a new push, duplicate push when reselecting the same entry, collapse rule (entry mutated in place, last pane wins).

DC-5 MAJOR. Lines 44-45, 96, 107, 126-128. Menu items do not show tooltips. Also Commands live in TurboSparkApp, not RootView, so "RootView exposes can-go-back" cannot drive menu enablement. State the mechanism (history on AppModel or a focused value).

DC-6 MAJOR. Lines 90-97. No Risks and Mitigations: chord collision, stale entries, interaction with OC-04, ZP-21 and ZP-23 (per window, per tab or global), "O(1) push and trim" is false for Array.removeFirst unless a ring buffer is specified.

DC-7 NIT. Lines 3-8, 18-22. Upstream claims (100 entries, settings collapse) rest on links only and are unverified.

DC-8 NIT. Line 27. The RootView link path does not resolve from docs/designs/opencode2-parity/ (needs ../../../swift/...).

DC-9 NIT. Lines 20-21, 63-68. CUJ 2 does not say what "prior section" means within the chat view; no CUJ for disabled ends, deleted chat or 100-entry trim.

## Structural reviewer (ST): 2 Blocker, 5 Major, 3 Nit

Heading hierarchy is sequential, code fence closed, links valid syntax.

ST-1 BLOCKER. Whole document. No Alternatives Considered section (standard's Alternatives Rule).

ST-2 BLOCKER. Whole document. No Risks and Mitigations section; NFR section (lines 90-97) covers only performance, localization and accessibility.

ST-3 MAJOR. Lines 53-68. No Open Questions section although INDEX says multi-answer decisions are recorded as Open Questions (stale targets, forward truncation, whether chat selection counts, per-window or per-tab scope).

ST-4 MAJOR. Lines 36-37 vs 50-51, 86. Goals include chat selections; Non-Goals and architecture say "section level only". Define "section-level" once.

ST-5 MAJOR. Lines 53-68. Two happy-path CUJs only; none for disabled ends, trim, or push after going back.

ST-6 MAJOR. Lines 70-84. Diagram omits the restore path; AlternateShortcutBridge appears with no explanation and is not in Target Files; no data model (cursor index vs two stacks) and no collapse rule definition.

ST-7 MAJOR. Lines 39, 109-114. Chord registration is described twice with overlapping scope in items 3 and 4; cross-reference the OC-01 override store.

ST-8 NIT. Line 27. Dead link path (same as DC-8).

ST-9 NIT. Lines 3-9. Free-text "Finding:" and "Parity targets:" lines before the metadata bullets; confirm against the EGM template.

ST-10 NIT. Lines 98-130. No sequencing or dependency note on OC-04, which also touches transitions.

## Style reviewer (SY): 0 Blocker, 3 Major, 5 Nit

SY-1 MAJOR. Lines 26-30. "No back-stack ... anywhere in the app sources" is a universal negative; INDEX says absence claims are grep-and-read evidence. State the evidence and date.

SY-2 MAJOR. Lines 18-22, 6-8. Upstream behavior is not tied to a commit; INDEX pins `7ce35879b8` but the links point at `dev`. Pin them.

SY-3 MAJOR. Lines 94-97, 122-128. Accessibility requirement ("announce enabled state", "keyboard-only navigation works end to end") has no verification step. Add a VoiceOver check or reduce the claim.

SY-4 NIT. Lines 44-45. Tooltip mechanism undefined for menu items.

SY-5 NIT. Lines 18-19, 34-37. "Section switches", "section-level transition", "navigation targets", "section identity" drift; "section" is never defined; chat selection is counted as section-level.

SY-6 NIT. Lines 73-84. "AlternateShortcutBridge: no extra chords needed" is a non-step; drop or give one prose sentence with the reason.

SY-7 NIT. Lines 86-88. "Scroll and composer state are preserved by existing per-section state" is uncited; "section granularity only" repeats Non-Goals lines 50-51.

SY-8 NIT. Lines 92-93, 117-120. "Never blocks" is unmeasurable; "O(1)" depends on the structure; the mutation-check procedure belongs in the Verification Plan.
