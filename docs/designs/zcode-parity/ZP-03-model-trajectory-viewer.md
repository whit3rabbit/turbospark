# ZP-03: Model trajectory viewer

Finding 2.14 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[ModelTrajectoryPane](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/ModelTrajectoryPane.tsx),
which reads bounded tails of `model-io-*.jsonl` rollouts.

* Author: whit3rabbit
* Status: Implemented (Swift only; on-disk encryption and a database table deferred, see Implementation notes)
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

## Implementation notes

Applies the EGM review (`docs/egm-reviews/zcode-parity/ZP-03`) Blockers and
Majors. Swift only, no Rust change, no model gate: it reads the existing
`GenerationResult` the app already receives.

### Decisions

* **Store**: per-chat JSONL sidecar files, not a `ProfileDatabase` table. A
  table is a schema migration and the review's Alternative B is open; the
  bounded-tail read the draft wants is natural on a file. The rollout lives
  under `trajectories/` in the profile store root (`AppStorageRoot.directory`),
  mode 0600 in a 0700 folder.
* **Encryption is NOT applied (deferred, needs sign-off)**. The vault's
  SQLCipher database cannot hold an append-only JSONL file, and encrypting
  lines needs a key and a crypto decision (CLAUDE.md: ask first). The
  mitigation is the same posture as conversation recall (`ChatRecallStore`):
  **off by default** (`MacAppSettings.trajectoryLoggingEnabled`, General
  settings), never recorded for a Ghost chat or a passphrase-protected profile
  (the store also refuses to write when the profile is protected, and
  protecting a profile deletes every log), and turning the setting off deletes
  every log. Options for the encrypted version: a `rollout` table in
  `ProfileDatabase`, or per-line AES-GCM under an HKDF label from the profile
  master key.
* **Redaction at write time** (deviation from "on-disk data stays
  unredacted"): `TrajectoryRedactor` removes private keys, Authorization and
  cookie headers, Bearer tokens, secret-named variables and JSON members,
  URL credentials and well-known token shapes before a line is written, then
  the text is clipped (redact first, so a clip cannot leave half a secret).
  Argument values under secret-named keys are dropped entirely. Export runs the
  same pass again. This is best effort and the export panel says so.
* **Size caps**: 4,000 characters per text field, 1,000 per argument value,
  16 arguments, 4 MiB per chat (the oldest whole lines are dropped down to
  3 MiB), 30 day retention (`pruneExpired` at launch). The pane renders the
  last 500 entries, never the whole file.
* **What is logged**: model name (file base name only), prompt, output and
  cached token counts, latency (engine prefill plus decode seconds), finish
  reason, the model's output text (clipped), tool name, arguments, result and
  error. NOT logged: the prompt, reasoning text, sampling parameters, any
  directory path. Each continuation segment is its own model-call line.
* **Seam**: `.finished(result)` in `executeGenerationTurn` (model call) and
  `appendToolExecutionTurn` (every tool outcome). A call whose result is later
  replaced by a todo-boundary result is logged twice (accepted).
* **Lifecycle**: keyed by chat id; deleting or clearing a chat deletes its log.
  Duplicate creates a new id and so starts with no log. Edit, retry and rewind
  do not touch the log: it records the calls that actually happened.
* **Crash safety (CUJ 3)**: lines are appended whole, after the call, never
  during streaming. A crash can leave an unterminated last line: readers skip
  it and show a count notice; the next append closes it with a newline first
  so entries never fuse. "Marked truncated" is dropped: the in-flight call is
  simply absent. There is no quit flush (a crash cannot run one); the append is
  queued on one serial queue and the loss bound is the call in flight.
* **Corruption**: a file whose first complete line is not a versioned JSON
  object is renamed `corrupt-<stamp>-...` before any append (the
  `AppJSONStore` rule), never overwritten. A bad line in the middle is skipped
  and counted.
* **Placement** (deviation): the pane is a sheet opened from the sidebar chat
  row's context menu ("View Trajectory", also a VoiceOver action), not from
  `ChatTranscriptView`. Search is the loaded window by default plus "Search
  entire log", which scans the file off the main thread. Pane state lives in
  `Generation/TrajectoryPaneState.swift`.
* **Dependencies**: ZP-23 hooks, if they add model calls, must call the same
  `logTrajectoryModelCall` keyed by chat id. No new dependency, no CI change.

### Files

`Diagnostics/RolloutStore.swift`, `State/AppModel+Trajectory.swift`,
`Generation/TrajectoryPaneView.swift`, `Generation/TrajectoryPaneState.swift`,
`RolloutStoreTests.swift`; hooks in `AppModel+Generation.swift`,
`AppModel+Tools.swift`, `AppModel+Chat.swift`, `ProfileVaultCoordinator.swift`;
setting in `MacAppSettings`, `AppModel`, `GeneralSettingsPaneView`.
Doc: `swift/docs/SWIFT_TRAJECTORY.md`.

### Not verified here

The pane was not exercised in the running app, and none of the Small/Extra
Large text, narrow window, theme, VoiceOver, Reduce Motion or RTL checks of
DESIGN.md section 9 were run. Unit tests cover the store, redaction and the
eligibility rule.

--------------------------------------------------------------------------------

*Drafted with EGM*
