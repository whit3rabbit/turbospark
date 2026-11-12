# Stage 5 report: Initial Zoom Out Review (OC-14)

Target: docs/designs/opencode2-parity/OC-14-hang-watchdog.md (full file, 187 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

Verdict: revise before approval. The design rests on premises the code contradicts and bundles unrelated scope.

Verified in code: `AppShutdownCoordinator` (AppModel+Persistence.swift:607-612) only holds an `onTerminate` closure; the ordered work is the synchronous `@MainActor` `AppModel.shutdown()` (616-697). `AppStreamRecovery.swift` is pure static transcript/anchor logic with no presentation surface. `GenerationEvent` has `.prefill(done:total:)` (GenerationTypes.swift:118). `QueuedGenerationStream.Request.didReceiveEvent()` runs on every event. No relaunch or heartbeat code exists in `swift/`. `.kiro/specs/desktop-shell-hardening/` does not exist.

Blocker
- ZO-1. Lines 52-56, 94-99, 120-121, 155-157. Guarded relaunch is not implementable as designed. `shutdown()` is one synchronous main-actor function, so a blocked step cannot be timed out or skipped without restructuring it into cancellable steps (unsized in the doc). Ordering invariants (stop server before releasing model, persist last) forbid "remaining steps run or are skipped" (line 99). Steps likeliest to block on a wedged engine (`session = nil`, `stopServer()`, `typeSafeServer?.terminate()`) are not addressed.
- ZO-2. Lines 53-55, 79-81. No "standard macOS relaunch helper" exists. `NSWorkspace.openApplication` with `createsNewApplicationInstance` starts a second instance while the first lives. Real relaunch needs a detached helper that waits for PID exit then runs `open`. This touches sandbox, PTY group termination, and the single-instance lock. Resolve or defer, not an Open Question.
- ZO-3. Lines 42-44, 101-106. "No tokens and no state change for 90 s" false-fires during long shell tools, pending approvals, MCP/web calls, and does not define "turn active" for audio and image sessions. Scope to "inside an engine generate call awaiting its next event". A single large prefill chunk with no `.prefill` event over 90 s is unverified.

Major
- ZO-4. Lines 20-25, 152-157. Overview says "the two that remain" but names three residuals, and Goals list four items.
- ZO-5. Lines 49-51, 122-126, 150-151. Main-thread heartbeat is log-only with no consumer, depends on the missing diagnostics spec, and needs an "activity label" registry that does not exist. Defer.
- ZO-6. Lines 43-44, 133-135, 147-149, 158-160. Monitor over-built: stamp a timestamp in the existing event loop and poll from a Task; no thread, no library API change. Line 133 "lock-free progress read" is wrong (`Request` uses `NSLock`, QueuedGenerationStream.swift:38).
- ZO-7. Lines 28-31, 114-117, 152-154. Panel claims to reuse a "stream-recovery presentation surface" that does not exist (`AppStreamRecovery` is data logic). Name a real surface or call it new. View should not live under `State/`.
- ZO-8. Lines 45-47, 91-92. "Stop the turn (existing cancel)" assumes cooperative cancel frees a wedged engine; a stuck call never observes the flag and the serial queue stays blocked. Panel must say what happens when Stop does not end the call (escalate to Restart).

Nit
- ZO-9. Lines 57-59, 75-78. Configurable window and per-session opt-out are speculative knobs; Keep waiting already re-arms. Hardcode 90 s, drop the escape hatch and strings.
- ZO-10. Lines 23-26, 30, 35. Dead link to desktop-shell-hardening brief; `swift/...` links resolve against the draft folder and are dead. Use `../../../swift/...`.
- ZO-11. Lines 166-169, 182-183. Mutation-check and out-of-scope hardware caveat are test-plan detail at wrong altitude; no real hang is ever exercised. Consider a manual long-prefill soak.
- ZO-12. Line 95 and INDEX.md. Sized "small-medium"; shutdown restructure plus relaunch helper make it large.

## Inventive catalyst (IC)

Verdict: contradiction analysis is shallow; three false compromises, two central claims contradicted by code.

- IC-1 (Major). Lines 45-47, 85-92. Cooperative cancel only ends the consumer stream; the serial `queue.async` worker stays inside the blocking `ts_generate` call and the next generation queues behind it. Only Restart recovers. Fix: phase-keyed monitor (prefill, decode, tool-exec, awaiting-approval) armed only on engine phases; relabel Stop as "Abandon turn", mark the session tainted, route to Restart or block new generations with a message; state CUJ 1 next-send behavior.
- IC-2 (Major). Lines 52-56, 94-99, 119-121. Same shutdown structure problem as ZO-1. Also line 55-56 claim that state risk is bounded by debounced saves is false: a streaming reply "exists only in `outputText`" and is saved by `finishCancelled` inside `shutdown()`; `persistChats()` and `persistSettings()` run last, so an early wedge loses the partial reply, draft, and settings. Fix: split into named steps, persist first on the relaunch path, teardown off main with a watchdog-enforced deadline, spawn detached `open -n` then `_exit`, write a last-stall marker. Liabilities: Gotcha 26 ordering, `_exit` skips PTY kills and fan restore.
- IC-3 (Major). Lines 48-51, 61-71. Log-only heartbeat makes the main-thread half forensic only; the user still force-quits and skips the flush. Fix: watchdog thread snapshots already-serialized state via `AppJSONStore` atomic write, marks next launch "recovered from stall", or offer an opt-in auto-relaunch after N minutes with a loop cap.
- IC-4 (Major). Lines 75-81, 158-160. Relaunch API is not an existing helper; the prefill callback is asserted but uncited (`didReceiveEvent` is used only for cancel-race replay). If no prefill events flow, a fixed 90 s window forces false stalls or a very long window. Fix: derive window from prompt tokens and measured prefill rate; name the detached helper as primary. Engine progress events may need an FFI change, contradicting "no behavior change" (line 160).
- IC-5 (Nit). Lines 57-59, 117. Opt-out is a manual escape hatch. Use escalating re-arm windows (90 s, 5 min, 15 min) and passive elapsed display.
- IC-6 (Nit). Lines 3-9, 22-26, 30, 35, 166-172. OC-90 (58-59) says OC-14 exports via the spec's redacted bundle, but OC-14 names no log sink or redaction rule, and the "activity label" could carry content. Dead links. Debug-flag hang injection does not reproduce the blocked-serial-queue state, so it would pass while the real failure stays untested. Fix: name log destination, make label a fixed enum, add a never-returning `operation` test fixture.
