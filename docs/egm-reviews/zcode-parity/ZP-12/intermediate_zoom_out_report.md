# Stage 8 intermediate zoom out report: ZP-12 treemap activity

Reviewer: egm-zoom-out (S8, completed). Document unedited, so no additions or mitigations from earlier stages exist; every earlier Blocker and Major remains open. Re-checked code: `WorktreeFileChange` (WorktreeModel.swift:6), `WorktreeViewMode` (:95), `WorktreeTabMode` (:115) exist as ZO reported.

## Blocker

- S8-B1. Lines 15-16, 31-33, 46-51, 79-83, 97-102. Data source contradictory and cannot supply the promised data (open from ZO-1, ZO-2, IC-1, CO-1, DC-2, SY-2). Live diff list is working-tree state with no viewed flag, pending state or chat attribution. v1 as written is a worktree churn map, not agent activity; CUJ 1 total match passes trivially. Fix: close the Open Question; recommended option (a) reframe as uncommitted worktree churn and cut viewed/pending. Option (b), a tool-call event source, adds scope and conflicts with "no new persistence".

## Major

- S8-M1. Lines 37-38, 60-61. Summary row semantics undefined (CO-2, IC-2, DC-4, SY-3). Fix: define each count against WorktreeFileChange.status; compute from the same node set as the tree.
- S8-M2. Lines 70-83, 117-135. Placement ignores WorktreeTabMode; ActivityTreeBuilder has no file; WorktreeFileListView already builds a directory tree (ZO-3, ZO-4, CO-5, DC-6, IC-4). Fix: name the enum case or view mode, add builder file and tests, state reuse, justify Diagnostics/.
- S8-M3. Lines 88-102. Alternatives rejected by unlinked "Rejected by author"; no cheaper alternative; squarify not justified over slice-and-dice (DC-1, ST-2, SY-5). Fix: concrete trade-offs, add a cheaper alternative, reconcile line 50 with line 102.
- S8-M4. Whole document (nearest 104-111). No Risks and Limits section (ST-1, DC-3, IC-3). Fix: add a minimal section; debounce and "other (N files)" are scope-adding, state as limits unless a bottleneck is shown.
- S8-M5. Lines 13-27, 63-68. "Current chat" scope and CUJ 2 empty state unsupported by the diff list (CO-3, CO-6, DC-8). Resolves with B1 option (a); state whether "current chat" binds to the selected chat (ZP-23).
- S8-M6. Lines 106-109, 122-123, 129-135. Canvas accessibility, no drill-down state model, missing test cases (IC-5, DC-5, CO-7). Fix: Canvas plus accessibility overlay or parallel list; path-keyed drill state; add zero-weight, deep nesting, large N, removed directory tests.
- S8-M7. Lines 3-5, 24-27, 44, 50, 110, 139-141. Unlinked references; "80 percent" stated as fact; ZCode data-source claim uncited (CO-4, SY-1, SY-13, DC-7). Fix: add links, mark as example.

## Nit

- S8-N1. Lines 111, 142. "21 languages" vs INDEX line 116 "ten localization parity gates" (ZO-7, IC-8, DC-5, SY-12).
- S8-N2. Lines 3, 16-19, 85, 99-100. "Small", "cheap surface", "richer data" unsupported; "small" holds only if viewed/pending are cut (ZO-8, IC-7).
- S8-N3. Lines 134-135. Mutation check is test-level detail; unclear manual vs automated (ZO-5, ST-5, IC-6).
- S8-N4. Lines 59, 66-68, 106-110. "Largest rectangle" must be a directory rectangle; "without errors" vague; empty-state text unspecified; "never color alone" fragment (SY-8, SY-9, SY-10).
- S8-N5. Squarify itself is a reasonable pure function; no existing treemap code found; reuse is only in the tree-building half.

Scope-adding fixes to decide: tool-call event source (prefer cutting viewed/pending), debounce and aggregation, full accessibility overlay, extra alternatives (document-only).

Net verdict: not ready. One decision to scope v1 to worktree churn resolves the Blocker and simplifies the summary, empty-state and accessibility findings.
