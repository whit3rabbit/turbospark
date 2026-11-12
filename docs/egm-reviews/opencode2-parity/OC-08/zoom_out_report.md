# Stage 5: Initial zoom out report (OC-08 local voice call)

Target: docs/designs/opencode2-parity/OC-08-local-voice-call.md (full file, 188 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

Verdict: pass with Major revisions. The premise "capture, streaming STT, VAD, and TTS the app already owns" and "pure integration" is only partly true against the cited Swift files.

ZO-1 Major, lines 19-25, 50-52, 120-125 (INDEX line 38 "pure integration"). VAD reuse is unsupported. The only VAD is `FluidAudioSpeechDetector` / `SileroSpeechDetector` in Audio/FluidAudio/FluidAudioVAD.swift, batch API over a finished sample array. No streaming onset or end-of-speech event source exists, and barge-in needs low-latency onset while TTS plays. Name the primitive and how it is driven incrementally, or reclassify size and risk. A VAD wrapper is missing from the file list.

ZO-2 Major, lines 19-21, 120-121. `AudioLiveTranscriber` emits only `.preparing`, `.update`, `.stopped`; it is built for one long recording and first model load can take minutes. `finishAndClose` closes the model, so per-turn "utterance final" needs a model that stays open or a re-open. The doc never says how live text is segmented into turns, who owns model lifecycle, or what the user sees during first load. Add a preparing state or a precondition.

ZO-3 Major, lines 120-125, 150-152. `AudioCapture` is recording-oriented: it writes WAV chunks through a required `onChunk` sink and persists via a Chunker, while the Privacy NFR (line 138) says only transcripts persist. It has a "recording already active" guard (no mutual-exclusion rule with the Record page). `AVAudioEngineConfigurationChange` is terminal, so a headphone change kills the call. No echo handling: system voice plus open mic risks self-triggered barge-in; Open Questions lists only latency.

ZO-4 Major, lines 53-54, 98-102, 159-161. `AppSpeechSynthesizer` is a per-message read-aloud keyed by `messageID`, whole-text only. Streaming replies need sentence-level speaking for the "starts immediately" claim (lines 80-83, 136-137); granularity is unstated. `sanitizeForSpeech` would read code, paths and tool output aloud; no spoken-summary or truncation policy. The "delegate bridge" (line 160) looks unnecessary: it already publishes `isSpeaking` and has `stop()`. `speak()` calls `stop()` first, so queueing sentences needs a new method.

ZO-5 Major, lines 57-59, 123, 159-161. The local TTS alternative is mis-scoped. `Qwen3TTSSpeechJob.generate` is a heavy batch job (native heavy-work permit, memory admission, asset verification, `AudioRecipe`/`AudioResult`), not a streaming sink, and contends with STT for the permit and memory. Cut it from v1 (goals, diagram, persisted settings, file 4).

ZO-6 Major, lines 127-131, 69-71. The submit path is asserted, not identified ("same queue path", "AppModel" have no symbol). The controller sits in Audio/ but couples to chat state. No interface for "turn finished, here is the reply text" for thinking to speaking. A blocked tool approval during thinking has no spoken cue.

ZO-7 Major, lines 130-131 vs 53-60, 40-47, 70-71. "Draft mode" appears once, tagged "(Open Questions candidate)", absent from Goals and Open Questions, and contradicts the line 70-71 rule. Promote or delete.

ZO-8 Nit, lines 31, 33, 35, 37, 3-5. Relative links `swift/TurboSparkApp/...` do not resolve from docs/designs/opencode2-parity/ (need `../../../`).

ZO-9 Nit, lines 77-79. A floating draggable panel is chosen only to match OpenCode; it adds positioning, persistence, window behavior and keyboard cost. A state strip on the composer deserves a fair comparison; "reposition drag" has no stated need.

ZO-10 Nit, lines 135-137, 167-171. Latency numbers are speculative with no baseline; mutation-check and injected-clock detail belongs in the test plan.

ZO-11 Nit, lines 162-164. Persisting speech engine choice disappears if local TTS is cut; verify `AppModel+Persistence.swift` is the right home versus an existing settings store.

ZO-12 Nit, lines 75-85. Open Questions 2 and 3 are decided or tuning items, not open.

Holds up: goal, non-goals (no WebRTC, cloud, wake word, voice approval), offline policy, CUJs, permission gating reuse (`RecordingPermissions`, `RecordingAccessSheet`), the no-prompt-at-launch source-scan test.

Caveat from reviewer: OPENCODE2_COMPARISON.md only keyword-searched; the send path and PromptComposerView.swift not read in depth.

## Inventive catalyst (IC)

Summary: mostly balanced and reuse-minded, but three false compromises are accepted unexamined.

IC-1 Major, lines 29-38, 114-125. "Pure integration" is inaccurate. `AudioCapture` always runs a persistence queue chunking audio to WAV (5 s chunks) and requires `onChunk`; `AudioLiveTranscriber` is built for one recording take with no utterance-boundary event (only `AudioStreamUpdate.isFinal`, unmentioned); the VAD is not linked and its live-feed fit is unstated. Contradiction: a call must save no audio while the stack's contract is to persist it. Resolution: extract a persistence-free capture tap feeding only `AudioLiveFeed`; state where VAD consumes audio (mixer 16 kHz output) and who owns the end-of-speech event; add "does the call save audio?" as an open question, default no. Liabilities: touches AudioCapture (not in file list), profile-lock drain guarantee risk.

IC-2 Major, lines 53-54, 84-85, 135-137, 123-125. Self-echo is the central full-duplex problem and only "use headphones" is offered. TTS audio can trip VAD onset or be transcribed into the next prompt. `AppSpeechSynthesizer.stop()` is only a stop, so the "lead audio" idea (CUJ 2) has no mechanism on the system-voice path. Resolution: half-duplex by default (higher onset threshold or minimum duration while speaking), voice-processing mode for the call session only, key or button barge-in as guaranteed fallback, drop "lead audio". Liabilities: voice processing changes input format (conflicts with the float32 `invalidFormat` check) and may duck audio.

IC-3 Major, lines 57-59, 80-82, 159-161, 123. `Qwen3TTSSpeechJob.generate` is a one-shot synthesizer returning a full PCM buffer (up to 360 s), no streaming or playback. Reply speaking granularity is unspecified; `speak()` calls `stop()` first so per-sentence calls cut each other. `AppSpeechSynthesizer` is a shared singleton keyed by message UUID and would collide with the read-aloud button. Resolution: a small call-owned SpeechSink protocol (enqueue sentence, stop, finished), system voice as the only v1 implementation, own synthesizer instance, defer local TTS; reuse the sanitizer (currently private).

IC-4 Major, lines 22-25, 45-49, 75-85, 127-131. Draft mode is inconsistent with auto-submit. A misrecognized transcript auto-submitted to a tool-running agent is an unmitigated risk; non-goals exclude voice approval but not voice-triggered tool execution. The paused state is in Goals but not the diagram. Resolution: move draft mode to Open Questions, default to draft or a short cancel window, add paused to the diagram.

IC-5 Major, lines 22-25, 66-67, 150-165. First STT model load can take minutes; the state machine has no preparing state though the transcriber emits `.preparing`. A call saying "listening" during load drops the first utterance. No-model-installed and memory-contention (`NativeHeavyWorkPermit`) paths unspecified. Resolution: add a preparing state, warm the model on panel open or press, specify the refusal paths.

IC-6 Nit, lines 6-9, 31-37, 150-165. Relative links unresolved; send path and recording access gate not named; call preferences may belong in the existing settings store (MacAppSettings) rather than `AppModel+Persistence.swift`. Section 7 of the comparison doc was not read, so the "ranks eighth" claim (line 39) is unverified.

Altitude note: the test list (lines 167-175) and mutation check are code-level detail, low priority.
