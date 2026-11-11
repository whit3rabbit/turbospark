# ZP-03: Model trajectory viewer

Finding 2.14 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[ModelTrajectoryPane](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/ModelTrajectoryPane.tsx),
which reads bounded tails of `model-io-*.jsonl` rollouts.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add a per-chat trajectory pane: a role-separated timeline of every model
call with model name, token usage, latency, finish reason, tool arguments
and results, and error entries, with search and payload inspection.
ZCode feeds its pane from bounded tail reads of `model-io-*.jsonl`
rollout files. TurboSpark has nothing equivalent; the nearest surface is
the Inspector's aggregate diagnostics.

## The Problem

When a turn goes wrong (bad tool arguments, a sampling artifact, a failed
call), the only evidence is the transcript row plus aggregate stats.
There is no call-by-call replay, so debugging means reproducing on the
CLI with `TURBOSPARK_DEBUG_PROMPT_IDS` instead of reading what happened
in place. Aggregate stats answer "how much", never "what sequence".

## Goals

* Each chat persists one rollout log: one JSON line per model call and
  per tool execution, appended after the fact (never during streaming).
* The pane shows the call timeline: role, model name, tokens, latency,
  finish reason; entries expand to arguments and results.
* Search across the rollout, and payload inspection per entry.
* Bounded tail reads keep the pane usable on very long chats.
* Export with redaction covers user-writable paths; on-disk data stays
  unredacted inside the profile vault.

## Non-Goals

* No cloud upload or telemetry pipeline (excluded category).
* No editing or re-running of logged calls (read-only surface).
* Not a profiler: kernel and Metal timing claims belong to bench
  harnesses and their gates, not this pane.

## Open Questions

* **Where rollouts live**: inside the profile vault (consistent with the
  app's encryption posture) or as sidecar files next to the chat archive
  (simpler bounded-tail reading). Lean vault; the author decides.
* **Retention**: unbounded per chat or capped with a clear action. The
  app's storage discipline favors a cap with a per-chat clear control.

## Critical User Journeys

### CUJ 1: debug a bad turn

* **Given**: a turn produced a wrong tool call.
* **When**: the user opens the trajectory pane from the chat context
  menu.
* **Then**: the pane lists the turn's calls in order; entries expand to
  arguments and results; searching for the tool name jumps to the call;
  the entry shows finish reason and latency; export with redaction
  produces a shareable file.

### CUJ 2: bounded tail on a long chat

* **Given**: a chat with 10,000 logged calls.
* **When**: the pane opens.
* **Then**: the pane renders the last 500 entries, offers "load earlier"
  paging, and a "search all" action scans the whole log off the main
  thread.

### CUJ 3: crash-safe logging

* **Given**: the app quits or crashes mid-chat.
* **When**: the user reopens the chat.
* **Then**: completed calls up to the interruption are present; the
  in-flight turn's log entry is either absent or marked truncated, and
  no partial line is ever parsed as an entry.

## Technical Plan (Architecture)

Two halves: logging and a pane.

```text
TrajectoryPaneView (Swift, opens from chat context menu)
  bounded tail render + "load earlier" + search
       |
       v
RolloutStore (Swift)
  one JSONL rollout per chat id, schema-versioned lines
  per-chat append lock; bounded tail reads; paged backfill
  export-with-redaction writer
       ^
       | append API
Generation pipeline hooks (Swift)
  log a line after each model call completes
  log a line after each tool execution completes
  (engine callbacks already carry usage, latency, finish reason)
```

Line schema: `{v, ts, kind: modelCall|toolExec, role, model, tokens,
latencyMs, finishReason, args?, result?, error?}`. Writes append after
the fact under a per-chat lock; no writes during streaming; the quit
flush bounds crash loss to the in-flight turn.

## Alternatives Considered

### Alternative A: log during streaming

* **Pros**: no loss on crash.
* **Cons**: interleaving a file writer into the streaming path risks
  stalls and partial lines.
* **Reason for Rejection**: Rejected by author: append-after keeps the
  streaming path untouched; the shutdown flush bounds loss to one turn.

### Alternative B: reuse the per-chat usage ledger

* **Pros**: data already exists.
* **Cons**: the ledger holds aggregates, not per-call records; a
  sequence cannot be rendered from aggregates.
* **Reason for Rejection**: Rejected by author: the pane needs per-call
  records, so a new store is required.

## Non-Functional Requirements

* **Privacy**: rollouts inherit the profile vault encryption posture;
  export redaction covers user-writable paths and secret-shaped strings.
* **Performance**: append-after writes; bounded tail reads; search-all
  off the main thread; no polling.
* **Observation-only**: no decode, sampling, or KV changes, so no model
  gates apply.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Diagnostics/RolloutStore.swift` (Create)
   * **Type / Scope**: rollout storage.
   * **Proposed Changes**: per-chat JSONL store, per-chat append lock,
     bounded tail reads, paged backfill, schema version field,
     export-with-redaction writer.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Trajectory.swift` (Create)
   * **Type / Scope**: app model integration.
   * **Proposed Changes**: log a line after each model call and after
     each tool execution, reusing existing engine callbacks; expose pane
     state (tail window, search) to the view layer.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/TrajectoryPaneView.swift` (Create)
   * **Type / Scope**: UI pane.
   * **Proposed Changes**: role-separated timeline, per-call expansion,
     search field, "load earlier" paging, redacted export action; DESIGN.md
     compliant rows and accessibility labels; catalog strings.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ChatTranscriptView.swift` (Modify)
   * **Type / Scope**: routing.
   * **Proposed Changes**: add the "View Trajectory" context-menu entry
     for the chat, opening the pane from the existing inspector/dock
     routing.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/RolloutStoreTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: append/round-trip, schema version field,
     bounded tail correctness at line granularity, backfill paging,
     partial-line tolerance (a truncated last line is ignored, not
     parsed), redaction export.
   * Mutation checks: break the tail boundary by one line and assert the
     tail test fails; drop the partial-line guard and assert the
     truncation test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for RolloutStoreTests and existing
  diagnostics tests.
* **Static Analysis and Linting**: `make design-check` for the new pane.
* **Localization**: ten parity gates; catalog additions in 21 languages.
* **Manual**: run a chat with tool calls, open the pane, expand entries,
  search, page back, export redacted, and confirm the on-disk file
  retains unredacted content.

--------------------------------------------------------------------------------

*Drafted with EGM*
