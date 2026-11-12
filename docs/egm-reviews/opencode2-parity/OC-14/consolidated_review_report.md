# Consolidated review report: OC-14 engine stall watchdog and guarded relaunch

Target: docs/designs/opencode2-parity/OC-14-hang-watchdog.md (full file, 187 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 5, Major 10, Nit 10.

## Blocker

B1. Lines 42-44, 75-78, 101-106. The stall signal is unproven: prefill-as-activity is load-bearing (CUJ 3) but sits in an Open Question, `.prefill` (GenerationTypes.swift:118) is not cited, and worst-case silence during one large prefill chunk is unknown. Fix: cite the event, state the gap as unverified, keep a fixed 90 s, add a manual long-prefill soak. [ZO-3, IC-4, CO-4, DC-1, SY-2, S8-B3]
B2. Lines 52-56, 94-99, 119-121, 155-157. Guarded relaunch is not implementable: `AppShutdownCoordinator` only holds `onTerminate`; `AppModel.shutdown()` (AppModel+Persistence.swift:616-697) is one synchronous main-actor function with ordering invariants, so a blocked step cannot be timed out or skipped. Fix: drop the timeout and skip logic; persist first, then exit (S8 reduction). [ZO-1, IC-2, DC-2, ST-4, CO-2, ST-10, SY-9, S8-B1]
B3. Lines 54-56, 143-157. "State risk is bounded by the debounced saves" is false: a streaming reply exists only in `outputText` and is saved inside `shutdown()` (643-655), with `persistChats()` last (694). Fix: replace with persist-first on the relaunch path and test that persist precedes teardown. [IC-2, DC-3, SY-5, S8-M2]
B4. Lines 52-54, 79-81. No "standard macOS relaunch helper" exists; `NSWorkspace.openApplication` starts a second instance while the first lives. Fix: name one mechanism (for example detached `open -n <bundle>` then terminate), delete the Open Question, add one risk bullet for single-instance lock and sandbox. [ZO-2, IC-4, DC-4, SY-3, CO-6, S8-B2]
B5. Whole document. Alternatives Considered and Risks & Mitigations are missing (mandatory in the standard). Fix: add two short sections (alternatives: in-engine timeout, helper vs direct relaunch, fixed window; risks: false-positive stall, relaunch loop, data loss, PTY orphan, stall-vs-completion race). [DC-12, ST-1, ST-2, S8-B4]

## Major

M1. Lines 45-47, 85-92. Stop (cooperative cancel) does not free a wedged engine: the serial worker stays in the blocking call and the next send queues behind it. Fix: reword Stop, state next-send behavior, steer to Restart, add one CUJ line. [ZO-8, IC-1, DC-5, DC-15, SY-1, S8-M1]
M2. Lines 42-44, 101-106. "Turn active" is under-scoped: long shell tools, approvals, MCP/web calls, and audio/image sessions are silent over 90 s. Fix: arm only while inside an engine generate call awaiting its next event. [ZO-3, DC-6, IC-1]
M3. Lines 49-51, 122-126, 150-151. The main-thread heartbeat has no consumer, no threshold, no activity-label registry, and depends on a nonexistent diagnostics spec. Fix: defer it; remove goal, diagram node, and file 2. [ZO-5, IC-3, DC-9, CO-5, ST-5, ST-8, SY-8, S8-M4]
M4. Lines 126-135, 147-149, 158-160. Monitor over-built and misdescribed: "lock-free progress read" is wrong (`Request` uses `NSLock`, QueuedGenerationStream.swift:38) and "no per-token work" conflicts with a per-token stamp. Fix: timestamp in the existing event path polled from a MainActor Task; say "one atomic store per token". [ZO-6, DC-7, SY-6, CO-7, CO-8, S8-M5]
M5. Lines 28-31, 114-117, 152-154. The "stream-recovery presentation surface" does not exist (`AppStreamRecovery.swift` is data logic); the view sits under `State/`. Fix: call it a new view modeled on a named existing attention banner, place outside `State/`. [ZO-7, DC-11, CO-7, S8-M6]
M6. Lines 20-25, 40-59. Overview says "the two that remain" but names three residuals and Goals list more. Fix: list in-scope items once. [ZO-4, DC-10, CO-1, ST-6, S8-M3]
M7. Lines 23-26, 64-67; INDEX 53-54; OC-90 58-59. The desktop-shell-hardening spec does not exist, and no log sink, format, or redaction is named. Fix: drop the dependency (free if M3 is deferred) and update OC-90 and INDEX to match. [DC-8, IC-6, S8-M7, S8-N1]
M8. Lines 43-44, 57-59, 117. Configurable window and per-session opt-out are speculative and muddled with Keep waiting re-arm; re-arm interval unspecified. Fix: hardcode 90 s, Keep waiting re-arms another 90 s, delete the escape hatch and its strings. [ZO-9, IC-5, CO-3, DC-13, SY-7, SY-8, S8-M8]
M9. Lines 98-99, 136-137. "Remaining steps run or are skipped" is a non-decision, and "never writes during the abandoned step" is an unenforceable guarantee for a synchronous step. Fix: resolved by B2 (persist first, no skip); otherwise narrow to "starts no further writes after the timeout". [CO-2, SY-4, SY-9, ST-10]
M10. Lines 111-118, 158-160. Missing design for which turn is active, concurrent turns, and a turn ending between tick and stall event (stale panel race); line 160 "no behavior change" conflicts if prefill events need engine work. Fix: state the active-turn rule and a re-check before presenting the panel. [CO-8, DC-7]

## Nit

N1. Lines 29-35. Repo-relative `swift/...` links resolve against the draft folder; also the `.kiro` brief link is dead and OpenCode links point at `dev` not commit 7ce35879b8. Fix: use `../../../swift/...`, remove or fix the dead link. [ZO-10, DC-14, ST-3, SY-10]
N2. Lines 166-172, 179-183. Mutation-check text is wrong altitude; debug hang injection does not reproduce a blocked serial queue; no real hang is exercised. Fix: one never-returning-operation fixture, state "verified with injected stalls only", add a manual soak. [ZO-11, DC-16, CO-13, CO-14, SY-14]
N3. Lines 11-13; INDEX 34, 95. Size "small-medium" is understated unless scope is cut as in B2/M3. Fix: resize after revision. [ZO-12, DC-17, S8-nit]
N4. Lines 3-9, 15-16, 20, 138-139, 176-178. Shorthand without definition: OpenCode finding, "platform-moot", OC-90, "21 languages", `make` targets; "mostly platform-moot" hedges the source. Fix: one-line glossary or links; cite INDEX for language count. [CO-9, CO-10, CO-11, SY-12, SY-13]
N5. Lines 140-141, 62. Keyboard and announcement claims unverifiable; Restart confirmation not stated. Fix: name states and mechanism; state whether Restart confirms. [CO-12, SY-13]
N6. Lines 83-106. CUJs thin. Fix: add only the Stop-does-not-free case; defer repeated-stall and failed-relaunch. [DC-15, S8]
N7. Lines 57-59, 45-48, 117. Last Goals bullet lacks bold lead-in; inconsistent list style. [ST-7, ST-9]
N8. Lines 73-81. Open Questions precede CUJs and plan while later sections treat them as decided. Fix: reorder after resolving. [ST-11]
N9. Lines 75, 79-81. "(lean)" is informal. Fix: "Recommendation:". [SY-11]
N10. Lines 31-33, 138-139, 164-165. "Metal hang" asserted without evidence; localization scope should name user-facing strings only. Fix: "suspected Metal hang"; list strings. [SY-1, DC-18]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-12 (zoom-out), IC-1 to IC-6 (inventive catalyst). Feeds B1-B4, M1-M8, N1-N3.
- comprehension_report.md: CO-1 to CO-14. Feeds B1-B2, B4, M1-M10, N2, N4-N5.
- critic_report.md: DC-1 to DC-18, ST-1 to ST-11, SY-1 to SY-14. Feeds all Blockers and Majors, N1-N10.
- intermediate_zoom_out_report.md: S8-B1 to S8-B4, S8-M1 to S8-M8, S8-N1. Confirms all earlier findings open; recommends rejecting scope-adding asks (IC-1 phase machine, IC-2 shutdown decomposition, IC-3 snapshot and auto-relaunch, IC-4 adaptive window, IC-5 escalating windows, full alternatives lists) and keeping ambiguity-removal fixes. Net direction: one timestamp-plus-poll stall monitor scoped to the generate call, a Keep waiting / Stop / Restart panel, Restart as persist-first then a named relaunch mechanism; defer heartbeat, shutdown timeout, opt-out, configurable window.
- Merge notes: B2 and B3 split from IC-2 (structure versus data-loss claim); SY-4 and SY-5 map to M9 and B3 respectively.
