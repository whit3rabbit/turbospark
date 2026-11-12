# Stage 6 comprehension report: ZP-12 treemap activity

Reviewer: egm-comprehension (completed). It read the document alone.

## What this proposes

Add a new "activity treemap" tab to the app's inspector. For the current chat, it draws rectangles for files the agent touched, with area proportional to added plus removed lines. Rectangles roll up into directories. Clicking a directory zooms in, and a breadcrumb lets the user go back up. Viewed-only files look different from written files. A summary row shows counts: added, removed, viewed, files written, modified, deleted and pending. For v1 it reads the existing live diff list and adds no git calls or persistence. A pure squarified-layout function does the layout, and a builder turns the per-file entries into a weighted directory tree.

## How the existing system works, per this document

The document says the app already holds a git change list (the "WorktreeModel diff list") and a "git pane" that shows a flat list of changed files. It also says "per-turn change records" are surfaced by the git pane, but a later Open Question says per-turn snapshots do not exist yet and depend on a separate `git-write-integration` spec. The inspector has existing tabs, routed by `WorktreeView.swift`. The document does not say whether the live diff list contains "viewed" information. It also does not say how the live diff list relates to a single chat.

## Findings

- CO-1 Blocker. Goals (31-33), Open Questions (46-51), Technical Plan (79-83), line 100. Data source is contradictory: Goals cite "per-turn change records the git pane already surfaces", the Open Question says those do not exist yet and leans to the live diff list. Nothing shows the live diff list can answer current-chat scope, viewed files, pending or deleted. A git diff would not record viewed-only files, so that feature may have no data source.
- CO-2 Major. Summary row (37-38, 60-61). Counts are undefined: added/removed as lines or files, written vs modified, what "pending" means. CUJ 1 says the summary matches the change list, but a viewed count would not appear in a change list.
- CO-3 Major. Overview and Problem (13-27). "Current chat" and "the inspector" are assumed known. No mapping from diff list to one chat, or behavior when chats share a worktree. The per-chat Non-Goal depends on it.
- CO-4 Major. Lines 3-5, 44, 50, 110, 139, 141. `ZCODE_COMPARISON.md` 5.11, ZCode, `git-write-integration`, `WorktreeModel`, "git pane", `DESIGN.md`, `make design-check`, `make swift-lib` are named without definition or links (line 50 spec has no link).
- CO-5 Major. Technical Plan and Target Files (73-83, 117-128). `ActivityTreeBuilder` is in the diagram but has no file; test item 4 covers "drill-down tree building". The layout file sits under `Diagnostics/`. Line 126 says "beside the existing tabs" without naming them.
- CO-6 Major. CUJ 2 (63-68). Empty state is "no tool-driven file changes", but the live diff list can hold changes the agent did not make. Behavior when the worktree has diffs but the chat made none is unspecified.
- CO-7 Nit. NFRs (106-111, 123). "Patterns distinguish viewed-only from written" unspecified; keyboard access to individual rectangles unspecified; "21 languages" has no list or link; canvas rendering conflicts with per-rectangle accessibility labels unless explained.
- CO-8 Nit. Verification (129-144, 95, 102, 26). "Mutation checks: shuffle tie-breaking" unexplained; "Rejected by author" gives a preference, not a reason; ZCode derives data from chat messages while this derives it from git, with no statement on whether that is intended.
