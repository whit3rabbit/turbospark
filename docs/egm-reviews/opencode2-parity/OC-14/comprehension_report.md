# Stage 6 report: Comprehension Test (OC-14)

Target: docs/designs/opencode2-parity/OC-14-hang-watchdog.md (read alone, no context). Reviewer: egm-comprehension. Completed.

## What this proposes

Add a stall watchdog to the macOS Swift app. While a generation turn is active, a monitor watches token and prefill progress. If nothing happens for a configurable window (default 90 s), a recovery panel appears offering Keep waiting, Stop the turn (existing cooperative cancel), or Restart the app. Restart runs the existing shutdown coordinator under a 10 s timeout, skips and logs any hung step, then relaunches through a standard workspace relaunch API. A separate main-thread heartbeat only logs stalls for support, and the user can disable the watchdog per session from the panel. Diagnostics export is out of scope and owned by another spec.

## How the existing system works, per this document

The app already has a stream-recovery presentation for agent turns, and a shutdown coordinator that performs ordered saves, with debounced saves running during generation. Nothing detects a stalled engine, so a Metal hang or wedged turn waits forever and the only exit is force quit, which skips the shutdown flush. Nothing bounds a hung shutdown step during a relaunch. A cooperative cancel exists with a contract that preserves partial output, and a prefill-progress callback exists. The engine stream is a QueuedGenerationStream, which currently exposes no activity stamp.

## Findings

CO-1 (Major). Lines 17-25, 52-56, 94-99. Overview says three residuals are recorded but the draft "covers the two that remain" without naming them. Line 19 says "guarded relaunch path", lines 22-23 say "on the quit path"; unclear whether relaunch is reachable only from the stall panel or also from normal quit.

CO-2 (Major). Lines 94-99, 52-56, 137, 171. "Remaining steps run or are skipped" is undecided. A hung step may be a dependency of later steps. No definition of "hung", no account of how a hang inside a synchronous step is abandoned, and no enforcement of line 137 "never writes during the abandoned step" while that step may still run. Test claim "ordering preserved" (171) does not say what happens to dependents.

CO-3 (Major). Lines 45-47, 57-59, 117, 140-141. Keep waiting "re-arm" window is unspecified (another 90 s or different). Relation between the opt-out escape hatch and Keep waiting is muddled (line 58 says the hatch stops "re-arming", but Keep waiting is what re-arms), and it is unclear whether the hatch is user-facing or debug-only.

CO-4 (Major). Lines 42-44 vs 75-78, 100-106. Activity is defined as "token progress" and "no state change". Prefill counting as activity appears only as an Open Question, yet CUJ 3 and tests treat it as settled. "State change" is undefined.

CO-5 (Major). Lines 49-51, 122-124. Heartbeat logging threshold has no value; "last activity label" is undefined (source, setter). The diagram shows the heartbeat as a peer of the relaunch chain, implying a flow that does not exist. Value versus cost unexplained.

CO-6 (Major). Lines 52-56, 79-81, 175-182. "Standard workspace relaunch API" and "standard macOS relaunch helper" are unnamed; the "reports unsupported" fallback condition is undefined. Manual verification stubs the relaunch call, so real relaunch is never verified.

CO-7 (Major). Lines 9, 30-35, 126-129, 153. Panel "reuses the stream-recovery presentation surface" and "attention vocabulary", neither defined or linked beyond a filename. New view is under State/, monitors under Diagnostics/. "Watchdog thread" (singular, line 126) versus two "(watchdog thread)" entries in the diagram; one-timer-tick cost (133) not reconciled.

CO-8 (Major). Lines 111-118, 158-160, 134. Monitor reads a last-activity stamp but the doc does not say how it knows which turn is active, how concurrent turns are handled, or how a turn ending between tick and event (stale panel race) is handled. "Lock-free progress read" is asserted without implementation.

CO-9 (Nit). Lines 3-9, 20. "OpenCode", "Finding: Section 1", "platform-moot", "Metal hang", and the Electron renderer premise are shorthand that need the comparison doc to decode.

CO-10 (Nit). Lines 15-16, 65-67, 176-178. "OC-90", "no-telemetry posture", "desktop-shell-hardening", `make design-check`, `make swift-lib` referenced without explanation.

CO-11 (Nit). Lines 138-139, 177. "21 languages" and "localization gates" not listed or linked.

CO-12 (Nit). Lines 62, 140-141. Not stated whether Restart needs confirmation.

CO-13 (Nit). Lines 166-169. Mutation check is under-specified (manual one-off or automated).

CO-14 (Nit). Lines 179-182. "Not a hardware gate" is unexplained shorthand; the debug hang flag is not in the target file list.
