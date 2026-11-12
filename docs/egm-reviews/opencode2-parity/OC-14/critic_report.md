# Stage 7 report: Critic Review (OC-14)

Target: docs/designs/opencode2-parity/OC-14-hang-watchdog.md (lines 1-187). Round 1 only; convergence not assessed. Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, standard feature-spec-standard), egm-style (SY, no standard). All three completed. Findings were not remediated.

## Design critic (DC)

Verdict: not approvable. Four blockers and a missing mandatory Alternatives section.

Blocker
- DC-1. Lines 42-44, 101-106, 111-113. No evidence the stall signal exists. Prefill counting as activity (CUJ 3, line 103) is deferred to Open Question 1 (75-78) yet is load-bearing. `Request.didReceiveEvent()` and `GenerationEvent.prefill(done:total:)` are not cited; whether the engine emits events during one large prefill chunk is unknown. Fix: cite emission cadence and worst-case gap, or make the window a function of prompt tokens and measured prefill rate; evaluate fixed vs adaptive as an alternatives pair.
- DC-2. Lines 52-56, 94-99, 119-121, 155-157. Guarded relaunch not implementable. `AppShutdownCoordinator` (AppModel+Persistence.swift:607-612) only holds `onTerminate`; the work is synchronous `@MainActor` `AppModel.shutdown()` (616-697) with ordering invariants (stopServer first per Gotcha 26, comment 629-631; persist last, 694-695). A blocked main-actor step cannot be timed out; skipping stopServer then releasing the session reopens the hazard. Likeliest blockers: `session = nil` (679), `stopServer` (631), `typeSafeServer?.terminate()` (633). Fix: decompose `shutdown()` into named steps with a skip-safety class each; state which thread enforces the deadline and what a skip does to later persists.
- DC-3. Lines 55-56, 143-157. "State risk is bounded by the debounced saves" is false. `shutdown()` lines 643-655: a streaming reply exists only in `outputText` and is saved by `finishCancelled` inside `shutdown()`; `persistChats()` runs after blocking steps. Fix: persist first (reply, draft, settings) on the relaunch path, then teardown under deadline; test persist precedes the first skippable step.
- DC-4. Lines 52-54, 79-81. The "standard workspace relaunch API" does not exist. `NSWorkspace.openApplication` starts a second instance while the first lives; real relaunch needs a detached helper that waits for PID exit then `open`. Interacts with sandbox, PTY group termination, single-instance lock. Fix: design it or defer relaunch; list alternatives with concrete rejections (helper process, launchd one-shot, second instance plus exit, direct bundle launch).

Major
- DC-5. Lines 45-47, 85-92. Stop does not recover a wedged engine: cancel ends the consumer and calls `cancelActive`; a stuck `ts_generate` never observes it and the next send queues behind it. Fix: define Stop behavior when the call does not return (taint session, steer to Restart or block sends), define next-send in CUJ 1.
- DC-6. Lines 42-44, 101-106. "Turn active" under-scoped: long shell tools, approvals, MCP/web, audio and image sessions are legitimately silent over 90 s. Fix: scope to "inside an engine generate call awaiting next event", suspend during approvals and tools, add CUJs.
- DC-7. Lines 111-113, 126-129, 133-135, 147-149, 158-160. Over-built and misdescribed: line 133 "lock-free progress read" but `Request` uses `NSLock` (QueuedGenerationStream.swift:38); a thread plus injected-clock machine plus library API change versus a stamp in the existing event loop polled from a Task; line 160 "no behavior change" conflicts if prefill events need engine/FFI work. Fix: justify the thread or simplify; name the owning module and API surface.
- DC-8. Lines 23-26, 64-67; INDEX 53-54; OC-90 58-59. Depends on `.kiro/specs/desktop-shell-hardening/` which does not exist; OC-14 names no log sink, format, redaction, or retention; "last activity label" could carry user content. Fix: fixed-enum label and named sink.
- DC-9. Lines 20-25, 49-51, 122-124. Heartbeat has no consumer or remedy; needs an activity-label registry that does not exist and is unsized. Fix: defer, or pair with a snapshot or "recovered from stall" marker.
- DC-10. Lines 20-25. Overview names three residuals, covers "the two that remain"; Goals list more. List scope once.
- DC-11. Lines 28-31, 114-117, 126-129, 152-154. "Stream-recovery presentation surface" does not exist (`AppStreamRecovery.swift` is data logic only); view under `State/` breaks convention. Fix: name a real surface or call it new.
- DC-12. Whole document. Missing Alternatives Considered (mandatory) and a real Risks & Mitigations section. Candidates: out-of-process engine, helper vs direct relaunch, fixed vs adaptive window, passive vs modal, auto-relaunch vs consent. Missing risks: false-positive on healthy turn, relaunch loop on deterministic hang, data loss on forced relaunch, PTY orphan, stall-vs-completion race.

Nit
- DC-13. Lines 57-59, 75-78. Configurable window and per-session opt-out are speculative; use escalating re-arm or drop.
- DC-14. Lines 3-9, 23-26, 30, 35. Dead links; `swift/...` needs `../../../swift/...`; OpenCode links point at `dev`, not audited commit 7ce35879b8.
- DC-15. Lines 83-106. CUJs thin: no Stop-does-not-free-engine, next-send-after-Stop, repeated stall after Keep waiting, failed relaunch.
- DC-16. Lines 126-129, 166-172, 176-183. Debug injection does not reproduce a blocked serial queue; add a never-returning `operation` fixture; mutation text is wrong altitude; add a manual long-prefill soak.
- DC-17. Lines 11-13; INDEX 34, 95. Sizing "small-medium" understated.
- DC-18. Lines 138-139, 164-165. State which strings are user-facing so localization gates are not applied to log text.

## Structural reviewer (ST)

Verdict: fails the standard until ST-1 and ST-2 are resolved. Heading hierarchy, fences and lists are sound.

Blocker
- ST-1. Whole document. Alternatives Considered missing (see DC-12). Candidates: in-engine timeout vs app-side monitor, process supervisor vs in-process relaunch, no relaunch, helper vs bundle-path launch. Rejections need concrete trade-offs.
- ST-2. Whole document. Risks & Mitigations missing as a section; Safety bullet (135-137) partial. Unaddressed: half-written state after skip, abandoned thread still running, false-positive panel, unsupported relaunch, cross-thread progress read, debug hatch in user-facing panel; 90 s unsupported.

Major
- ST-3. Lines 30, 35. Broken relative links (see DC-14); OC-90 near line 113 has the same mistake.
- ST-4. Lines 34-37, 52-56, 119-121, 155-157. Mismatched shutdown premise (see DC-2); "debounced saves already run" uncited (55-56).
- ST-5. Lines 49-51, 57-59 vs 83-106, 166-172. Heartbeat and opt-out lack CUJ; heartbeat lacks a test item and a threshold value.
- ST-6. Lines 19-26 vs 40-59; 46, 117, 58-59. Overview vs Goals scope mismatch; unclear whether Keep waiting re-arms or the hatch suppresses re-arming.

Nit
- ST-7. Lines 57-59. Last Goals bullet has no bold lead-in.
- ST-8. Lines 110-124. MainThreadHeartbeat placed under GuardedRelaunch reads as a chain stage; it is a peer (line 126).
- ST-9. Lines 45-48, 117. Inconsistent list style.
- ST-10. Lines 98-99. CUJ 2 "run or are skipped" is untestable and conflicts with "ordering preserved" (171).
- ST-11. Lines 73-81. Open Questions placed before CUJs and plan while later sections treat them as decided.

## Style reviewer (SY)

No Blockers.

Major
- SY-1. Lines 31-33 vs 45-46, 91-92. "Only exit is force quit" contradicts reliance on the existing cooperative cancel; "a Metal hang" asserted without evidence. Say cancel cannot interrupt a wedged engine; write "suspected Metal hang".
- SY-2. Lines 75-78 vs 43-44, 101-106. Prefill-as-activity settled in CUJ and tests, open in Open Questions; callback uncited; "state change" undefined.
- SY-3. Lines 52-56, 79-81. Relaunch API unnamed; "reports unsupported" has no concrete signal.
- SY-4. Lines 136-137. "Never writes during the abandoned step" is an uncalibrated guarantee; narrow to "starts no further writes after the timeout"; state whether `AppJSONStore` atomic writes make partial writes acceptable.
- SY-5. Lines 54-55. Debounced-save claim outruns evidence (comparison doc cites a 300 ms debounce for chat saves only) and conflicts with Restart flushing.
- SY-6. Lines 133-134 vs 158-160. "No per-token work" contradicts a last-activity stamp written per token; restate as one atomic store per token; heartbeat cost unspecified.
- SY-7. Lines 57-59, 138. Opt-out has no CUJ, is absent from the panel actions and diagram, conflicts with "Keep waiting (re-arm)", and "debug" hatch sits in a localized user-facing panel.
- SY-8. Lines 43-44, 48-51. "Configurable window" has no setting name or bounds; main-thread threshold and heartbeat interval have no values; one thread or two (126 vs diagram).
- SY-9. Lines 98-99. "Run or are skipped" is a non-decision inside a CUJ.

Nit
- SY-10. Lines 29-35. Dead relative links (see DC-14).
- SY-11. Lines 75, 79-81. "(lean)" is informal; use "Recommendation:".
- SY-12. Lines 19-25. "Mostly platform-moot" hedge; the report says the exact feature is platform-moot.
- SY-13. Lines 138-141. "Fully keyboard-operable" and "state changes announce" unverifiable; name states and mechanism; cite where "21 languages" is defined (INDEX).
- SY-14. Lines 181-183. State once that behavior is verified only with injected stalls and untested against a real Metal hang.
