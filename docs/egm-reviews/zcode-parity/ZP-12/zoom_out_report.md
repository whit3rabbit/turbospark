# Stage 5 zoom out report: ZP-12 treemap activity

Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom out (ZO)

Verdict: objective clear, scope small. One factual problem with the data source, plus simplicity and altitude issues. No existing treemap or squarify code under swift/.

Major
- ZO-1. Lines 15-16, 25-27, 31-33, 80-83. Doc claims viewed files, viewed and pending counts, and "tool-driven file changes" come from the WorktreeModel diff list. `WorktreeFileChange` (State/WorktreeModel.swift:6-73) has only relativePath, status (M/A/D/R/?/!), additions, deletions, fileName, directoryPath: no viewed flag, no pending state, no chat or turn attribution. The live diff list is working-tree state, including user edits and pre-existing dirt. Goals and CUJ 1 overpromise. Cut viewed/pending from v1 or name the real read-event source (a new data path the Overview and Non-Goals disclaim).
- ZO-2. Lines 46-51, 97-102. Open Question is effectively decided in a way that changes the product. v1 is "uncommitted worktree churn", but title and goals say agent activity in the current chat. Reframe Problem and Goals. CUJ 1 check "matches the change list totals" passes trivially while the purpose is unmet. Modified/deleted/pending counts have the same fidelity problem.
- ZO-3. Lines 70-83, 117-128. Placement ignores the existing pattern. Top-level tabs are the `WorktreeTabMode` enum (changes, timeline, worktrees) at WorktreeModel.swift:115, never mentioned. Files go in Diagnostics/ (telemetry, usage) while other Worktree* views live in Generation/. Alternative: a view mode in Changes beside `WorktreeViewMode` (tree/flat). State which.
- ZO-4. Lines 79-83, 129-135. `ActivityTreeBuilder` is in the diagram but not in Target Files; its tests are folded into TreemapLayoutTests. `WorktreeFileListView` already builds a directory tree; reuse or share it.

Nit
- ZO-5. Lines 85-86, 117-120, 134-135. Hand-rolled squarify is the largest cost; say why slice-and-dice is not enough. Tie-break and mutation detail is test-level.
- ZO-6. Lines 35-36, 106-109. Drill-down plus keyboard stack plus patterns is heavy for a "small" pane; patterns depend on nonexistent viewed data. Flat per-directory rollup is the minimal version.
- ZO-7. Line 142 "21 languages" vs INDEX.md line 116 "ten localization parity gates". Align.
- ZO-8. Line 3, INDEX line 12. "Small" holds only if viewed/pending are cut; otherwise medium.

## Inventive catalyst (IC)

Major
- IC-1. Lines 19-20, 33-38, 46-51, 79-83, 97-102. Viewed data does not exist in the chosen source; a git diff list holds only changed files, and it includes user and pre-chat edits (contradicts lines 16, 42). Proposal: a small `FileActivityEvent` stream (path, kind, added/removed, source) consumed by ActivityTreeBuilder, fed in v1 from existing tool-call records, diff list only to reconcile; or drop viewed/pending from v1 explicitly. Liabilities: reconciliation rule for line 60, per-chat attribution may conflict with "no new persistence" (line 86).
- IC-2. Lines 38, 60, 131. Summary-row semantics undefined: pending, written, modified, deleted not defined; renames, binaries, untracked and zero-line changes get zero area yet are counted. Fix: one status enum, a minimum visible weight or zero-line strip, summary computed from the same node set as the tree.
- IC-3. Lines 85-86, 121-124. Scale and live updates unaddressed: thousands of files, recompute during streaming edits, drill-down jumping. Fix: memoize by change-list revision, debounce, "other (N files)" aggregation, path-keyed drill state that pops to nearest surviving ancestor.

Nit
- IC-4. Lines 117-119, 129-135. Layout under Diagnostics/; test file name covers tree building and totals. Split an ActivityTreeBuilder test.
- IC-5. Lines 106-109, 122-123. Canvas has no per-rectangle accessibility elements; need accessibility children or a parallel list. Hit testing and minimum label size unspecified.
- IC-6. Lines 134-135. Mutation check described as a test; say manual or automated; shuffle input order with a fixed seed.
- IC-7. Lines 46-51, 102. Open question "lean" vs "Rejected by author"; reconcile; INDEX line 97 sizing hides a data dependency.
- IC-8. Lines 110-111, 142. "21 languages" vs "ten gates"; confirm pattern fills meet DESIGN.md.
