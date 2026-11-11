# ZP-12: Treemapping activity visualization

Finding 5.11 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[TreemappingPane](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/TreemappingPane.tsx).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add a treemap pane that visualizes which files the agent touched in the
current chat: rectangles sized by added and removed lines, rolled up
through directories, with a summary row and per-file drill-down. This is
a small, self-contained inspector surface computed from data the app
already holds in the git change list.

## The Problem

The agent's file activity is only visible as a flat changed-file list.
Across a long session, the user cannot see at a glance that 80 percent
of the churn landed in two directories, or that a file was viewed but
never written. ZCode renders a treemap of file activity from task chat
messages; TurboSpark has nothing comparable.

## Goals

* A treemap view of the current chat's file activity, computed from the
  per-turn change records the git pane already surfaces.
* Rectangles sized by total added plus removed lines; viewed-only files
  shown distinctly from written files.
* Directory rollup with drill-down: click a directory rectangle to zoom
  into it, breadcrumb to go back up.
* A summary row: added, removed, viewed, files written, modified,
  deleted, pending counts.

## Non-Goals

* No history across chats; the treemap is per chat (ZCode's is per task
  chat as well).
* No new git invocations; the view consumes existing change data.

## Open Questions

* **Data source**: per-turn change snapshots (which arrive only after
  the `git-write-integration` spec lands its change tracking) vs the
  live diff list (available today). Lean live diff list for v1, with
  per-turn snapshots as a later enhancement.

## Critical User Journeys

### CUJ 1: see where churn landed

* **Given**: a chat where the agent edited many files.
* **When**: the user opens the activity treemap tab in the inspector.
* **Then**: rectangles show directory rollups sized by line churn;
  clicking the largest rectangle drills into its files; the summary row
  matches the change list totals.

### CUJ 2: empty and small states

* **Given**: a chat with no tool-driven file changes.
* **When**: the pane opens.
* **Then**: an empty state explains the pane needs file activity; a chat
  with one file renders a single rectangle, both without errors.

## Technical Plan (Architecture)

```text
TreemapPaneView (inspector tab, new)
  squarified treemap layout (pure function)
  drill-down stack + breadcrumb
  summary row
       |
       v
ActivityTreeBuilder
  input: existing per-file change entries
  (added/removed/viewed) from the WorktreeModel diff list
  output: directory tree with weights
```

The squarified treemap algorithm is a small deterministic layout
function, unit-testable without UI. No new persistence.

## Alternatives Considered

### Alternative A: bar chart of top files

* **Pros**: simpler.
* **Cons**: loses the directory structure and the area-proportional
  read of churn distribution.
* **Reason for Rejection**: Rejected by author: structure is the point.

### Alternative B: defer until per-turn snapshots exist

* **Pros**: richer data.
* **Cons**: blocks a cheap surface on a larger spec; the live diff list
  already answers the question.
* **Reason for Rejection**: Rejected by author: ship on existing data.

## Non-Functional Requirements

* **Accessibility**: rectangles carry labels with file or directory
  name and counts; the treemap is navigable by keyboard through the
  drill-down stack; never color alone (patterns distinguish viewed-only
  from written).
* **Design compliance**: DESIGN.md controls and status colors; localized
  strings, 21 languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/TreemapLayout.swift` (Create)
   * **Type / Scope**: layout function.
   * **Proposed Changes**: squarified treemap, pure and unit-tested,
     deterministic tie-breaking by path.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/TreemapPaneView.swift` (Create)
   * **Type / Scope**: pane UI.
   * **Proposed Changes**: canvas rendering of rectangles, drill-down,
     breadcrumb, summary row, empty states.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/WorktreeView.swift` (Modify)
   * **Type / Scope**: inspector routing.
   * **Proposed Changes**: add the activity treemap tab beside the
     existing tabs.
4. `swift/TurboSparkApp/Tests/TurboSparkAppTests/TreemapLayoutTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: layout invariants (area proportionality
     within tolerance, no overlaps, containment), deterministic ordering,
     drill-down tree building, summary totals matching input.
   * Mutation checks: shuffle tie-breaking and assert the determinism
     test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the layout tests.
* **Static Analysis and Linting**: `make design-check`.
* **Localization**: catalog additions in 21 languages.
* **Manual**: run a chat that edits files, open the pane, drill down,
  verify totals against the change list.

--------------------------------------------------------------------------------

*Drafted with EGM*
