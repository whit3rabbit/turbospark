# Stage 8: Intermediate zoom-out report (OC-08 local voice call)

Target: docs/designs/opencode2-parity/OC-08-local-voice-call.md (full file). Reviewer: egm-zoom-out (completed). Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md. No remediation was applied, so the target is unchanged and every Stage 5 finding is still open. Reviewer did not open the Swift sources or INDEX; it relied on earlier reviewers' source claims, which agree. IDs: S8-n.

Verdict: no Blockers from this reviewer. Central problem unchanged: the spec says "pure integration" (lines 19-25, 29-38, 120-125), but capture, VAD and local TTS do not fit a live conversation as described.

## Stage 5 findings still open, fix to keep

S8-1 Major, lines 50-52, 120-125 (ZO-1, DC-4, ST-3). VAD is batch and unlinked. Name FluidAudioVAD.swift, say whether it can run incrementally and which component emits end-of-speech; if not, change the INDEX size note.

S8-2 Major, lines 19-21, 120-121 (ZO-2, IC-5, DC-3). No per-turn finals rule, no first-load handling. Add a preparing state, a no-model-installed precondition, and one sentence on STT model lifecycle (kept open for the call).

S8-3 Major, lines 120-125, 138 (ZO-3, IC-1, DC-2). Capture persists WAV chunks, conflicting with the privacy NFR; has a recording-active guard; treats device change as terminal. State a call saves no audio, require a non-persisting tap, add Record-page mutual exclusion, make device change end the call cleanly, add AudioCapture.swift to the file list.

S8-4 Major, lines 53-54, 98-102, 159-161 (ZO-4, IC-3). Speaking granularity unstated. State sentence-level, add a skip rule for code, paths and tool output, delete the delegate bridge, use a call-owned synthesizer instance.

S8-5 Major, lines 57-59, 123, 159-161 (ZO-5, IC-3, DC-7). Cut local TTS from v1; this also removes the engine setting, the file 4 sink and resolves Open Question 2.

S8-6 Major, lines 127-131 (ZO-6, DC-8). Name the real send entry point and the turn-finished signal; state one rule for an utterance arriving during a running turn; move the controller out of Audio/ or justify it.

S8-7 Major, lines 129-131 vs 70-71 (ZO-7, IC-4). Decide draft mode. Simplest: draft-first (transcript lands in the composer, user presses Send); edit CUJ 1. Do not add a cancel window.

S8-8 Nit, lines 3-9, 31-37 (ZO-8, DC-14, ST-4). Fix relative links to `../../../swift/...`.

S8-9 Nit, lines 77-79 (ZO-9). Drop "reposition drag"; give a composer state strip one fair sentence.

S8-10 Nit/Major, lines 162-164 (ZO-11, DC-13). After cutting local TTS only the silence window remains; use the existing settings store or hardcode a default and drop file 5.

S8-11 Nit, lines 75-85 (ZO-12, SY-5). Move system-voice default and panel shape to Decisions; keep barge-in tuning; define "lead window" once (CO-3, SY-4).

S8-12 Nit, lines 135-137 (ZO-10, SY-3, DC-10). Label 600 ms a target, give the silence window a default, add one latency check. No metrics program.

## Stage 6 and 7 asks to keep (ambiguity removal)

- CO-4: add paused, error and End to the diagram; say what happens on an empty reply or failed turn.
- CO-5, DC-9: name the controller state while the access sheet shows; point a macOS denial to System Settings; a revocation ends the call.
- CO-7: define `AppModel`, "send path", "offline policy" by link or one line.
- SY-1, SY-2: correct attribution to the comparison doc section 10 ranking.
- IC-2 reduced: half-duplex by default (ignore onset while speaking unless it passes a threshold), button or key as guaranteed barge-in, drop "lead audio".

## Scope-adding asks to reject

- IC-2 voice-processing echo cancellation (changes input format, conflicts with float32 check).
- IC-5 warm model on panel open (preparing state is enough).
- IC-4 cancel window (draft-first covers it).
- DC-1 / ST-1 full Alternatives section: accept at most one short paragraph rejecting cloud realtime, push-to-talk and dictation-only.
- DC-10 success metrics and idle cost estimates.
- DC-11 / ST-2 broad Risks: at most three bullets (self-echo loop, profile-lock drain ending the call, device loss); idle auto-end only if runaway risk is real. Reject sleep, lock and ambient media catalog.
- DC-12 delegation tools: one line in Non-Goals.
- DC-17 VAD protocol seam (test-plan detail).
- SY-6 to SY-9, SY-12 to SY-15, ST-6, ST-9: wording and link polish, skip unless free.

Altitude note: lines 167-175 are already near code-level (mutation check, injected clock and VAD events). Do not add more test detail; trim if anything.

Net effect: the recommended fixes shrink the spec (local TTS, persisted engine choice, delegate bridge, draggable panel go) and add three decisions (draft-first, half-duplex, no audio saved), one preparing state, and named reuse points. The risk is accepting every Stage 6 and 7 ask, which would turn a short integration spec into a risk and metrics document.
