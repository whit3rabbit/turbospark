# Stage 8 report: Intermediate Zoom Out Review (OC-14)

Target: docs/designs/opencode2-parity/OC-14-hang-watchdog.md (full file, 187 lines, unedited since Stage 5). Reviewer: egm-zoom-out. Completed. Prior reports: zoom_out_report.md, comprehension_report.md, critic_report.md.

Re-verified in code: `AppShutdownCoordinator` (AppModel+Persistence.swift:607-612) only holds `onTerminate`; the ordered work is synchronous `@MainActor` `AppModel.shutdown()` (616-695) with `stopServer()` at 631, `finishCancelled` at 653, `session = nil` at 679, `persistChats()`/`persistSettings()` at 694-695. `QueuedGenerationStream.Request` uses `NSLock` (line 38), `didReceiveEvent()` at 64, `queue.async` at 18, `cancelActive` at 61 and 71. Doc line 133 "lock-free progress read" is wrong.

Verdict: revise before approval. Fails the standard on root-problem focus (Stop does not free a wedge; shutdown premise is false) and on leverage-existing (two reuse claims point at things that do not exist). Cut scope, do not add mechanism.

## Blockers (all still open)

- S8-B1. Lines 52-56, 94-99, 119-121, 155-157. Guarded relaunch not implementable against the real `shutdown()`. [ZO-1, DC-2, ST-4, IC-2, CO-2, ST-10, SY-9] Keep, but resolve by scope reduction: reject IC-2 (step decomposition, off-main teardown, `_exit`); instead persist first (`finishCancelled`, `persistChats`, `persistSettings`), then exit, with no timeout on blocking teardown. That removes the "run or skipped" ambiguity and the 10 s machinery. A timeout would need a named enforcing thread and is a deferred follow-up.
- S8-B2. Lines 52-54, 79-81. No standard relaunch helper exists. [ZO-2, DC-4, SY-3, CO-6, IC-4] Pick one mechanism in one sentence (for example detached `/usr/bin/open -n <bundle>` then terminate), delete the Open Question, record single-instance lock and sandbox as one risk bullet. Reject helper-process designs and a four-way matrix.
- S8-B3. Lines 42-44, 101-106 vs 75-78. Stall signal unproven; prefill-as-activity is load-bearing but open. [DC-1, ZO-3, IC-4, CO-4, SY-2] Cite `GenerationEvent.prefill(done:total:)` (GenerationTypes.swift:118), state worst-case silence is unverified, scope to "armed only while inside an engine generate call awaiting its next event, suspended during tools and approvals". Reject adaptive window; keep fixed 90 s and a manual long-prefill soak.
- S8-B4. Whole document. Missing Alternatives Considered and Risks & Mitigations. [DC-12, ST-1, ST-2] Add two short sections (3 to 5 lines each) covering only decisions B1, B2, B5 force. Reject long candidate lists.

## Major (all still open)

- S8-M1. Lines 45-47, 85-92. Stop does not free a wedged engine. [ZO-8, IC-1, DC-5, DC-15] Fix wording and add one CUJ line: after a stall, Stop does not free the engine, next send will not run, panel steers to Restart. Reject tainted-session state machine and phase-keyed monitor.
- S8-M2. Lines 55-56. "Bounded by debounced saves" is false. [DC-3, IC-2, SY-5] Replace with the persist-first sentence.
- S8-M3. Lines 20-25 vs 40-59. Scope listed inconsistently. [ZO-4, DC-10, CO-1, ST-6] List in-scope items once.
- S8-M4. Lines 49-51, 122-126, 150-151. Heartbeat has no consumer, threshold, or label registry; depends on a nonexistent spec. [ZO-5, DC-9, CO-5, IC-3, SY-8, ST-5, ST-8] Defer: remove goal, diagram node, file 2. Reject IC-3 snapshot, marker, and auto-relaunch additions (separate draft). Deferral also removes CO-5, DC-8, and most of ST-5 and SY-8.
- S8-M5. Lines 126-135, 147-149, 158-160. Monitor over-built. [ZO-6, DC-7, SY-6, CO-7, CO-8] Store a last-activity timestamp in the existing event path, poll from a MainActor Task; keep injected clock for tests only; drop "lock-free" and "no per-token work" (use "one atomic store per token").
- S8-M6. Lines 28-31, 114-117, 152-154. Panel claims reuse of a nonexistent surface. [ZO-7, DC-11, CO-7] Call it a new view modeled on a named existing attention banner (author must find it; not verified), place outside `State/`.
- S8-M7. Lines 23-26, 64-67; INDEX 53-54. Dead links and a nonexistent spec dependency. [DC-8, ST-3, ZO-10, DC-14] Fix links to `../../../swift/...`; drop the dependency (free once M4 is deferred).
- S8-M8. Lines 57-59, 117. Opt-out and configurable window. [CO-3, SY-7, DC-13, IC-5, ZO-9] Cut both: hardcode 90 s, Keep waiting re-arms another 90 s, delete escape hatch and its strings. Reject escalating windows.

## Nits (open, wording fixes)

ST-7, ST-9, ST-11 (reorder Open Questions); CO-9, CO-10, CO-11 (one-line glossary or links); CO-12 (state whether Restart confirms); CO-13, DC-16, ZO-11 (drop mutation-check text, keep one never-returning-operation fixture); DC-17, ZO-12 (resize in INDEX after cuts); SY-1 ("suspected Metal hang"); SY-4 (narrow or delete with B1); SY-11 to SY-14, DC-18.

- S8-N1. INDEX.md line 34 and OC-90 lines 54-59 ("logs locally and exports via the spec's redacted bundle") must be edited to match if the heartbeat is deferred.

## Scope-adding asks to reject or defer

IC-2 shutdown decomposition (reject); IC-1 phase-keyed monitor and tainted session (reject); IC-3 snapshots, marker, auto-relaunch (defer); IC-4 adaptive window and engine/FFI events (reject, record as known gap); IC-5 escalating windows (reject); DC-12 full alternatives lists (one paragraph only); DC-15 extra CUJs (add only Stop-does-not-free); ST-5 heartbeat CUJ and test (moot if deferred); IC-6 redaction and sink (moot if deferred).

## Keep: ambiguity-removal fixes

Name the relaunch mechanism; cite `.prefill` and "armed only inside generate"; Stop wording; persist-first wording; single scope list; timestamp-plus-poll description and NSLock correction; "new view" honesty; link fixes; hardcoded window and no opt-out; never-returning-operation test fixture; short Alternatives and Risks sections.

Net direction: shrink to (a) one stall monitor via timestamp plus Task poll scoped to the generate call, (b) a panel with Keep waiting, Stop (honest wording), Restart, (c) Restart as persist-first then a named relaunch mechanism. Defer heartbeat, shutdown timeout machinery, opt-out, configurable window, diagnostics dependency.
