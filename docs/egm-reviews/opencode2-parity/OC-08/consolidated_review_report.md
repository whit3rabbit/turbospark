# Consolidated review report: OC-08 local voice call

Target: docs/designs/opencode2-parity/OC-08-local-voice-call.md (full file, 188 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (one inventive-catalyst spawn hit "Concurrent subagent limit reached" and succeeded on the first retry).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 1, Major 11, Nit 11.

## Blocker

B1. Whole doc (nearest lines 75-85, 17-186). No Alternatives Considered and no Risks and Mitigations sections, both required by the standard; panel shape and speech engine are "lean" questions with no rejections, and echo, device loss, profile lock and runaway turn loops are uncovered. Fix: add one short Alternatives paragraph (cloud realtime, push-to-talk, dictation-only rejected with reasons) and three Risks bullets (self-echo loop, profile-lock drain ends the call, device loss); no metrics program. [DC-1, DC-11, ST-1, ST-2, S8 scoping]

## Major

M1. Lines 19-25, 29-38, 50-52, 120-125 (INDEX line 38). "Pure integration" is false for VAD: the only VAD is batch (Audio/FluidAudio/FluidAudioVAD.swift), unlinked, with no streaming endpoint or barge-in onset source. Fix: link it, state how it is driven incrementally and which component emits end-of-speech, or revise the size note. [ZO-1, IC-1, DC-4, ST-3, S8-1]

M2. Lines 19-21, 120-121. `AudioLiveTranscriber` is built for one take (minutes-long first load, no utterance-final event, no per-turn segmentation, no failure behavior) and the state machine has no preparing state. Fix: add a preparing state, a no-model-installed precondition, per-turn segmentation rule and model lifecycle owner. [ZO-2, IC-5, DC-3, CO-6, S8-2]

M3. Lines 120-125, 138-139, 150-152. `AudioCapture` persists WAV chunks (conflicts with privacy NFR), throws if a recording is active, and ends on device change. Fix: state a call saves no audio via a non-persisting tap, add Record-page exclusion and clean end on device change, add AudioCapture.swift to the file list. [ZO-3, IC-1, DC-2, S8-3]

M4. Lines 53-54, 84-85, 97-102, 181-183. Barge-in and echo unmitigated; "lead window" and "lead audio" undefined and meaningless for TTS output. Fix: half-duplex by default (ignore onset while speaking unless above a threshold), button or key as guaranteed barge-in, drop "lead audio"; reject full echo cancellation for v1. [ZO-3, IC-2, DC-5, CO-3, SY-4, S8]

M5. Lines 53-54, 98-102, 159-161. Speaking granularity unstated; `sanitizeForSpeech` reads code and paths; the delegate bridge is unnecessary (`stop()` and `isSpeaking` exist); singleton is keyed by message UUID. Fix: sentence-level speaking, skip rule for code, paths and tool output, drop the bridge, call-owned synthesizer. [ZO-4, IC-3, DC-6, SY-11, S8-4]

M6. Lines 57-59, 80-83, 123, 159-164. Local TTS is mis-scoped: batch job, heavy-work permit, memory admission, reloads STT each turn. Fix: cut from v1; remove engine setting, sink in file 4 and Open Question 2. [ZO-5, IC-3, DC-7, S8-5]

M7. Lines 127-131, 70-71, 150-152. Submit bridge unnamed; no turn-finished signal; no rule for an utterance during a running turn or speech during thinking; tool approval mid-turn hangs with mic open. Fix: name the send entry point and reply-complete signal, state one queue/steer/drop rule, say the call pauses or speaks a notice on approval. [ZO-6, DC-8, CO-4, S8-6]

M8. Lines 129-131 vs 70-71, 75-85, 162-164. Draft mode is a dangling "Open Questions candidate" that contradicts the approval rule; auto-submitting mis-transcribed speech to a tool-running agent is a safety risk. Fix: decide it as draft-first (transcript lands in the composer, user sends) and edit CUJ 1; no cancel window. [ZO-7, IC-4, CO-1, DC-8, ST-5, SY-2, S8-7]

M9. Lines 100-102, 114-125, 61-62, 104-110. State machine omits paused, End, error, failed turn and empty reply; CUJ 3 "no state change" is not tied to a controller state; macOS denial gives no path to System Settings; copy "Check ... the Record page" is wrong for a call. Fix: complete the diagram, name the state while the access sheet shows, add call-specific denial copy, a revocation ends the call. [CO-4, CO-5, DC-9, IC-4, S8]

M10. Lines 50-52, 133-137, 163. Silence window has no default or home; 600 ms and "existing synthesizer start cost" are untestable. Fix: give a default, label 600 ms a target, add one latency check to Verification. [CO-2, DC-10, SY-3, ZO-10, ST-7, S8-12]

M11. Lines 162-164. Call preference persisted "with the archive" with no schema or migration note; wrong home. Fix: after cutting local TTS only the silence window remains: use `MacAppSettings` or hardcode a default and drop file 5. [ZO-11, DC-13, IC-6, S8-10]

## Nit

N1. Lines 3-9, 31-37. Relative `swift/...` links do not resolve from the doc folder. Fix: use `../../../swift/...`. [ZO-8, IC-6, CO-10, DC-14, ST-4, S8-8]

N2. Lines 39-40. "OpenCode ranks this gap eighth" is a wrong attribution; the ranking is OPENCODE2_COMPARISON.md section 10, line 502. Fix: attribute to the comparison doc and quote its wording. [SY-1, CO-8, ST-8, S8]

N3. Lines 75-85, 77-79, 80-83. Settled defaults (system voice, floating panel) are listed as open; "reposition drag" has no stated need; "zero-model latency" unmeasured. Fix: move to Decisions, give the composer state strip one fair sentence, say "no model load". [ZO-9, ZO-12, CO-11, SY-5, S8-9, S8-11]

N4. Lines 27-40, INDEX line 38. Problem statement lacks user motivation; delegation tools are not scoped out; INDEX "large; pure integration" understates the work and the draft states no size. Fix: one motivation sentence, a one-line Non-Goal for delegation tools, align size note. [DC-12, DC-15, SY-15, S8]

N5. Lines 29-30, 19-25, 38, 35-38. Overclaims: "same UX", "deeper than OpenCode's", unverified negative claim, unclear whether read-aloud already covers replies. Fix: "a comparable call UX", state facts, add "per the 2026-10-31 audit". [SY-6, SY-7, SY-8, SY-9]

N6. Lines 29-41, 127-129, 66-67, 139-141, 180. Undefined project terms (`AppModel`, send path, steering, offline policy, 21 languages, localization gates); new strings not listed. Fix: one line or link each; name the offline policy source in INDEX. [CO-7, CO-9, CO-12 (N/A), DC-18, SY-12, ST-7]

N7. Lines 46-47, 142-143, 49, 95. Duplicate glyph-plus-text statement; "spoken while the transcript shows it" vague. Fix: keep one and state speech start. [SY-10, SY-11]

N8. Lines 167-175, 171. Near code-level test detail; mutation check unexplained; no VAD seam described. Fix: trim, rephrase as "verify the barge-in test fails when the transition is removed"; do not add detail. [ZO-10, CO-12, SY-13, DC-17, ST-9, S8 altitude]

N9. Lines 183-184. DESIGN.md and `make design-check` unlinked. Fix: link them. [SY-14, ST-9]

N10. Lines 156-164. `Generation/PromptComposerView.swift` path unverified. Fix: confirm paths before drafting the plan. [DC-16]

N11. Whole doc. Section names and order differ from standard names ("The Problem", "Technical Plan (Architecture)"). Fix: confirm against the EGM template. [ST-6]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-12, IC-1 to IC-6. Feeds B1 (context), M1-M6, M7-M11, N1-N3, N8.
- comprehension_report.md: CO-1 to CO-12. Feeds M2, M4, M7-M10, N2, N3, N6, N8.
- critic_report.md: DC-1 to DC-18, ST-1 to ST-10, SY-1 to SY-15. Feeds B1 and all Majors, N1-N11.
- intermediate_zoom_out_report.md: S8-1 to S8-12 plus keep/reject lists. Confirms all Stage 5 findings open. Recommends rejecting echo cancellation, model warm-up, cancel window, metrics, full risk catalog, delegation tools in scope, VAD seam; keeping ambiguity-removal fixes. B1 is downgraded in scope (short paragraph, three bullets) but kept as Blocker because the standard marks both sections required.
- Merge notes: the 2 Stage 5 reviewers plus 1 comprehension and 3 critic reviewers produced overlapping claims on VAD, capture persistence, TTS shape, draft mode and links; each is merged into one finding with all source IDs. CO-12 in N6 refers to the mutation-check item, listed under N8.
