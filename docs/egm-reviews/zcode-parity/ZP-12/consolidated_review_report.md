# Consolidated review report: ZP-12 treemap activity

Target: docs/designs/zcode-parity/ZP-12-treemap-activity.md (full file, 148 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. The design reviewer noted agents/design.md was missing and used design_critic.md.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 3, Major 7, Nit 9.

## Blocker

B1. Lines 15-16, 31-33, 46-51, 79-83, 97-102. The live diff list cannot supply the promised data: no viewed flag, pending state or chat attribution (`WorktreeFileChange`), and it holds user edits and pre-existing dirt. Goals say "per-turn" records while the Open Question says they do not exist; CUJ 1 total match passes trivially. Fix: close the Open Question and reframe v1 as uncommitted worktree churn, cutting viewed and pending (or name a real tool-call source and accept the scope). [ZO-1, ZO-2, IC-1, CO-1, DC-2, ST-3, SY-2, SY-4, S8-B1]
B2. Lines 88-102. Both alternatives rejected by unlinked "Rejected by author"; no cheaper alternative (slice-and-dice, indented tree, churn badges on the Files list); squarify not justified. Fix: concrete trade-offs, add at least one cheaper alternative, reconcile with line 50. [DC-1, ST-2, SY-5, S8-M3, ZO-5]
B3. Whole document (nearest 104-111). No Risks and Mitigations section: thousands of files, streaming recompute, drilled directory removed, zero-area files, tiny rectangles, vendored dirs, paths outside workspace. Fix: add a short H2 between Alternatives and NFRs with minimal mitigations (memoize by change-list revision, pop drill-down to nearest surviving ancestor). [ST-1, DC-3, IC-3, S8-M4]

## Major

M1. Lines 37-38, 60-61. Summary counts (written, modified, pending, added/removed as lines or files) undefined; zero-line files vanish yet are counted. Fix: define each against `WorktreeFileChange.status`; compute the summary from the same node set as the tree. [CO-2, IC-2, DC-4, SY-3, S8-M1]
M2. Lines 70-83, 117-135. Placement ignores `WorktreeTabMode`; `ActivityTreeBuilder` has no file and shares tests with layout; `WorktreeFileListView` already builds a directory tree; views sit in Diagnostics/ while routing is in Generation/. Fix: name the enum case or view mode, add builder file and tests, state reuse, justify location. [ZO-3, ZO-4, CO-5, DC-6, IC-4, S8-M2]
M3. Lines 13-27, 63-68. "Current chat" scope and CUJ 2 empty state not supported by a working-tree diff; chat binding (ZP-23) unstated. Fix: say "uncommitted changes in this worktree", empty state "no changes", state selected-chat binding. [CO-3, CO-6, DC-4, DC-8, S8-M5]
M4. Lines 106-109, 122-123, 129-135. Canvas has no per-rectangle accessibility; no drill-down state model; tests omit zero-weight, single file, deep nesting, large N, removed drilled directory. Fix: Canvas plus accessibility overlay or parallel list; path-keyed drill state; add cases. [IC-5, DC-5, CO-7, S8-M6]
M5. Lines 3-5, 26-27, 44, 50, 110, 139-141. Unlinked references (git-write-integration, WorktreeModel, DESIGN.md, make targets, ZCODE_COMPARISON.md path unverified); ZCode chat-message source uncited. Fix: add links and citations. [CO-4, DC-7, ST-6, SY-13, S8-M7]
M6. Lines 24-25. "80 percent ... two directories" and "viewed but never written" stated as fact. Fix: mark as example or drop. [SY-1, S8-M7]
M7. Lines 46-51, 97-102. "Lean live diff list" hedge with no owner, while Alternative B treats it as decided. Fix: record the decision once; link `.kiro/specs/git-write-integration`; state INDEX "self-contained" dependency fallback. [SY-4, ST-3, DC-8, IC-7]

## Nit

N1. Lines 111, 142. "21 languages" vs INDEX line 116 "ten localization parity gates". Fix: cite the INDEX constraint. [ZO-7, IC-8, DC-5, SY-12, CO-7, S8-N1]
N2. Lines 3, 16-19, 85, 99-100. "Small", "cheap surface", "richer data", "small deterministic layout function" unsupported. Fix: cite file count or drop. [ZO-8, DC-7, SY-6, SY-7, S8-N2]
N3. Lines 134-135. Mutation check is test-level detail, unlabelled, manual vs automated unclear; shuffle input order with fixed seed. Fix: fold into the test description. [ZO-5, IC-6, ST-5, SY-11, CO-8, S8-N3]
N4. Line 59. "Largest rectangle drills into its files" holds only for a directory rectangle. Fix: reword. [SY-8, S8-N4]
N5. Lines 66-68. "Without errors" vague; empty-state text unspecified. Fix: observable result and message text. [SY-9, S8-N4]
N6. Lines 106-110. "Never color alone" fragment; focusability unstated; pattern spec missing; DESIGN.md unlinked. Fix: rewrite and link. [SY-10, CO-7, S8-N4]
N7. Lines 35-36, 106-109. Drill-down plus keyboard stack heavy for a "small" pane; flat rollup is the minimal version. Fix: state why drill-down is v1. [ZO-6]
N8. Lines 117-134. Directory choice unexplained; WorktreeModel without path. Fix: justify and name paths. [SY-11, DC-6]
N9. Lines 53-70, 104, 137. Section order informational only, consistent with folder. Fix: none required. [ST-4]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-8. Feeds B1-B3, M1-M4, N1-N3, N7.
- comprehension_report.md: CO-1 to CO-8. Feeds B1, M1-M5, N1, N3, N5, N6.
- critic_report.md: DC-1 to DC-8, ST-1 to ST-6, SY-1 to SY-13. Feeds all Blockers and Majors, N1-N9.
- intermediate_zoom_out_report.md: S8-B1, S8-M1 to S8-M7, S8-N1 to S8-N5. Confirms all earlier Blocker and Major findings open; recommends scoping v1 to worktree churn and treating tool-call source, debounce and aggregation, and a full accessibility overlay as scope-adding.
