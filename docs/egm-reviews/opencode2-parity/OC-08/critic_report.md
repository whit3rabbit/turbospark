# Stage 7: Critic report (OC-08 local voice call)

Target: docs/designs/opencode2-parity/OC-08-local-voice-call.md (full file, 188 lines). One round only; convergence not assessed. Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY, no standard). All completed.

## Design critic (DC): verdict not ready for implementation

DC-1 Blocker, lines 17-186. No Alternatives Considered section (standard's Alternatives Rule). Panel shape and speech engine are "lean" open questions with no concrete rejections. Missing alternatives: push-to-talk vs continuous VAD, dictation-only mode filling the composer draft, existing full-recording transcriber vs a purpose-built STT path. Also no Risks and Mitigations section.

DC-2 Major, lines 120-125, 150-152. Capture is built for recording: 5 s WAV chunks persisted through `onChunk`, 8 MB budget. A call would write utterances to disk, conflicting with the Privacy NFR (138-139). `AVAudioEngineConfigurationChange` (headphone unplug) terminally stops capture. `alreadyRecording` throws if the Record page is active. Mic and system-audio capture must not both run; configuration unconstrained.

DC-3 Major, lines 120-125. `AudioLiveTranscriber` is scoped to one take: first model load can take minutes (600 s backlog), no preparing state in the call machine, no utterance-final event (only committed text and `isFinal`), no per-turn segmentation or reset rule (new transcriber and reload per turn?), no behavior on backlog overflow or model error.

DC-4 Major, lines 33-35, 50-52, 120-125. VAD is cited without a file link. It lives under Audio/FluidAudio (FluidAudioVAD.swift) and looks batch, not a streaming endpoint detector. Model, frame latency, live-PCM fit and barge-in onset suitability are unanalyzed. If not streaming, "integration only" sizing is wrong.

DC-5 Major, lines 53-54, 97-102, 84-85, 181-183. Barge-in and echo unmitigated: speaker output hits the open mic and causes self-barge-in. Only "use headphones" in manual testing. Need voice processing (AEC), an onset threshold or confirmation window, or a different rule. "Lead audio" is meaningless for TTS output; CUJ 2 gives no number.

DC-6 Major, lines 57-59, 156-161. `AppSpeechSynthesizer` is a `@MainActor` singleton keyed by message UUID speaking one whole string. The required change is unstated (`stop()` and `isSpeaking` already exist). No speech sink interface. "Starts immediately" holds only after the full reply streams; sentence streaming needs a segmenter and queue. `sanitizeForSpeech` reads tables, links, paths and code poorly; no skip-code rule. Default voice follows app locale, not reply language.

DC-7 Major, lines 57-59, 80-83, 123. Local TTS as a sink is not viable as specified. `Qwen3TTSSpeechJob.generate` is batch (full `AudioResult`), gated by `NativeHeavyWorkPermit`, memory admission and a lifecycle token. `synthesizeQwenSpeech` closes the resident STT session before loading TTS, so STT would reload each turn. Shared memory budget may not fit both. Cancellation is slow. Defer local TTS from v1 or add the analysis.

DC-8 Major, lines 70-71, 127-131, 150-152. Submit bridge is asserted, not specified: no function or file. Undefined: user types or a turn is running when an utterance ends (queue, steer, drop); speech during thinking; tool approval mid-turn (no waiting-for-approval state, mic stays open while the call hangs). Line 70-71 conflicts with the draft-mode paragraph (129-131), which is an "Open Questions candidate" not listed in Open Questions. Auto-submitting mis-transcribed speech to a tool-running agent is a safety risk needing a decision (draft-first, confirmation, or cancel window).

DC-9 Major, lines 61-62, 104-110. Permissions: `CaptureError.permission` copy points to "the Record page", wrong for a call. CUJ 3 "no state change" is not tied to the controller state while the sheet is up. Revocation mid-call unaddressed.

DC-10 Major, lines 133-137. Latency NFR untestable: no silence window default, no basis for 600 ms, STT commit lag, 100 ms pump and VAD endpoint latency unbudgeted, "existing synthesizer start cost" has no number. No success metrics (false barge-in rate, mis-transcription rate) and no idle CPU/ANE/memory cost estimate.

DC-11 Major, whole doc. No Risks and Mitigations section. Also uncovered: long-lived open mic with no idle auto-end, sleep or lock mid-call, profile lock (`stopAcceptingAndDrain` must end the call), device changes, runaway self-triggered turn loops burning tokens or running tools, ambient media audio transcribed as commands.

DC-12 Major, lines 27-40. Problem statement argues "the pieces exist" and a ranking, not user motivation. Comparison line 502 says delegation tools are also missing; not addressed.

DC-13 Major, lines 162-164. Call preference persisted "with the archive" (the chat archive rewritten whole on flush) with no schema, version or migration note; belongs in app settings (`MacAppSettings`).

DC-14 Nit, lines 31-37. Relative `swift/...` links do not resolve (need `../../../swift/...`).

DC-15 Nit, INDEX line 38. "large; pure integration" understates the work given DC-2 to DC-8.

DC-16 Nit, lines 156-164. `Generation/PromptComposerView.swift` path unverified (reviewer did not read it).

DC-17 Nit, lines 167-175. Planned tests use injected VAD and clock events but no VAD protocol seam is described; add a self-barge-in test once the echo design exists.

DC-18 Nit, line 140. 21-language requirement does not list the new strings (state names, errors, permission copy).

Reviewer caveat: did not read FluidAudioVAD.swift, PromptComposerView.swift, or the comparison doc beyond line 136.

## Structural reviewer (ST)

Heading hierarchy sound (one H1, H2s, H3 under CUJs and plan, no skipped levels, fences closed).

ST-1 Major, whole doc (nearest content lines 75-85). Missing "Alternatives Considered" (mandatory). Cloud realtime model is rejected only by the offline non-goal; push-to-talk and reuse of the meeting/dictation pipeline are not weighed.

ST-2 Major, whole doc. Missing "Risks & Mitigations". Risks only hinted: headphones (183), barge-in clipping (84), declined permission (CUJ 3). Not covered: echo/self-trigger, combined STT+VAD+LLM+TTS cost, model not loaded at Call time, device loss, runaway turn loop.

ST-3 Major, lines 29-37, 120-124. "VAD ... all exist" but the link list has no VAD file; VAD is in Audio/FluidAudio/FluidAudioVAD.swift, never linked. The diagram does not say which component emits end-of-speech.

ST-4 Major (link compliance), lines 31, 33, 35, 37. Broken relative links; inconsistent with line 5 and 9 links that resolve.

ST-5 Nit, lines 129-131. Dangling "Open Questions candidate"; draft mode not in Goals, target files or tests.

ST-6 Nit, whole doc. Section order and names differ from the standard ("The Problem", "Technical Plan (Architecture)", Open Questions before CUJs). Acceptable as an EGM template variant; confirm.

ST-7 Nit, lines 135, 140. NFRs not measurable; silence window has no default or persistence key; "21 languages" not linked to gates.

ST-8 Nit, line 39. Ranking claim: the comparison doc (line 502) does list voice call eighth; "integration work" plausible but unverified here.

ST-9 Nit, lines 167-184. Mutation check and source-scan test described only in rationale lines; `make design-check` and DESIGN.md section 9 unlinked.

ST-10 Info, lines 159-161. The delegate bridge claim is consistent with the source (class already sets itself as delegate, has `stop()`); no issue beyond DC-6.

## Style reviewer (SY)

No Blockers.

SY-1 Major, lines 39-40. "OpenCode ranks this gap eighth" is a wrong attribution; OPENCODE2_COMPARISON.md line 502 ranks it (eighth of eleven, section 10). Comparison wording is "wiring a call loop is integration work rather than new capability" (line 367). The comparison also says delegation tools are missing; Non-Goals do not say they are out of scope.

SY-2 Major, line 131. "(Open Questions candidate)" presents draft mode as part of the design while labeling it only a candidate; absent from Goals, Open Questions and the plan. Move to Open Questions or drop; if kept, name it in file 5 rationale.

SY-3 Major, lines 135-137. 600 ms has no measurement or test in the Verification Plan; second clause is circular. Mark as a target, give the silence window default, add a latency check.

SY-4 Major, lines 101, 84-85. "Lead window" undefined; the open question defines it as audio retained before cut-off, a different thing from a time limit on stopping playback. Define once.

SY-5 Major, lines 57-59, 77-83. Decided defaults (system voice, floating panel) appear both as Goals and as open questions. "Zero-model latency" is unmeasured; say "no model load".

SY-6 Nit, lines 19-25. "same UX" overstates: OpenCode's call has a delegator model with tools.

SY-7 Nit, lines 29-30. "deeper than OpenCode's" is a comparative judgment; state facts (OpenCode has no local STT or TTS).

SY-8 Nit, line 38. Negative claim "Nothing sends a live transcript..." lacks a verification note; add "per the comparison audit of 2026-10-31".

SY-9 Nit, lines 35-38. The draft lists `AppSpeechSynthesizer` as existing and "system-voice read-aloud" though the comparison says TTS is not wired to chat replies; state whether read-aloud already applies to replies, since it affects line 38.

SY-10 Nit, lines 46-47, 142-143. Duplicate "glyph plus text" statement; keep one.

SY-11 Nit, lines 49, 95. "the reply is spoken while the transcript shows it" is vague about streaming vs end-of-turn speech start.

SY-12 Nit, lines 66-67. Name the policy source: INDEX "Offline constraint (user directive 2026-11-11)".

SY-13 Nit, line 171. Rephrase mutation check: "Verify the barge-in test fails when the transition is removed."

SY-14 Nit, lines 183-184. Link DESIGN.md.

SY-15 Nit, INDEX line 38. Draft states no size; INDEX says large.

No em dashes or emojis found. Keep: the no wake word and no autonomous approval non-goals (68-71).
