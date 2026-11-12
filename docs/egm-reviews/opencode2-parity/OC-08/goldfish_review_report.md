Status: resolved

# Goldfish Annotated Review Report

**Target:** `docs/designs/opencode2-parity/OC-08-local-voice-call.md` (`full file`, 188 lines)
**Source findings:** `consolidated_review_report.md` (1 Blocker, 11 Major, 11 Nit). Triage only; the target was not edited.

## Summary

| Category | Count | Issues |
| --- | --- | --- |
| PROPOSE FIX | 17 | B1, M1-M11, N1, N2, N4, N5, N9 |
| CONTEST | 1 | N10 |
| ESCALATE | 1 | N11 |
| DECLINE | 4 | N3, N6, N7, N8 |

Severities are unchanged from the consolidated report.

**Verified-false reviewer claims:** one.
* N10 says the path `Generation/PromptComposerView.swift` is unverified. It exists at `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerView.swift`. The other four modify/create paths in the doc's file list also exist (`Presentation/AppSpeechSynthesizer.swift`, `State/AppModel+Persistence.swift`, `Localization/Localizable.xcstrings`, `Tests/TurboSparkAppTests/`).

**Verified-true factual claims (read in the repo):**
* VAD is batch and unlinked: `FluidAudioSpeechDetector.speechSegments(in: [Float], maxSegmentSeconds:)` in `Audio/FluidAudio/FluidAudioVAD.swift` takes a whole sample array and returns finished segments (it wraps `VadManager.segmentSpeech`, with a 0.75 s silence rule baked in). Nothing under `Sources/` calls `speechSegments` or `speechDetector(for:installRoot:)` outside that file; only `FluidAudioVADTests` and `FluidAudioVADRealModelTests` do. No streaming or onset API exists.
* `AudioCapture` persists audio: `enqueue` hands samples to a `Chunker` that emits WAV chunks through `onChunk` (`AudioMediaIO.wav`). `start` throws `CaptureError.alreadyRecording` when a capture is active. A `.AVAudioEngineConfigurationChange` observer calls `fail(..., interruption: true)`, which ends the capture and tells the user to start a new recording.
* `AudioLiveTranscriber.Event` has only `preparing`, `update(AudioStreamUpdate)`, `stopped(String)`. `isFinal` exists on `AudioStreamUpdate` but marks end of the whole stream (set when `finishAndClose()` drains), not a per-utterance boundary. The class comment says the first model load "can take minutes".
* `AppSpeechSynthesizer` has `static let shared`, `speakingMessageID: UUID?`, `isSpeaking`, and `stop()`. Its delegate already handles `didFinish` and `didCancel`. `sanitizeForSpeech` strips code fence markers, headers and bold markers only, so code and path content is still read aloud.
* `Qwen3TTSSpeechJob.generate` acquires `NativeHeavyWorkPermit` and runs a memory admission estimate; it is a batch job.
* `docs/OPENCODE2_COMPARISON.md` line 502 is the ranking ("8. Voice call mode ... only the call loop and delegation tools are missing"); the doc's "OpenCode ranks this gap eighth" misattributes it.
* Privacy rule: `swift/AGENTS.md` (macOS privacy prompts bullet) says only `RecordingPermissions` calls `AVCaptureDevice.requestAccess`, from the access sheet Allow button; `AudioCapture` only checks status and throws; new capture entry points use `.recordingAccessGate` (defined in `Audio/RecordingAccessSheet.swift`); `RecordingPermissionsTests` scans `Sources/`. The doc's Goal on mic access is consistent with this, but its test file 8 duplicates what that scan already does unless the new files are in `Sources/Audio/` (they are, so the existing scan already covers them).

## Prioritized Findings & AI Suggestions

### [Issue B1] [`Design / Structural / Zoom-out`] - Severity: Blocker
*   **Affected Scope:** [Lines 17-186](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L17-L186) (whole doc; nearest lines 75-85)
*   **Discrepancy Description:** No Alternatives Considered and no Risks and Mitigations sections; panel shape and speech engine are "lean" questions with no rejections; echo, device loss, profile lock and runaway turn loops are uncovered.
*   **Rationale:** Both sections are required by the standard; without them the doc hides the design's main failure modes.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The doc has no such sections. The risks are real and checkable: `AudioCapture` ends on `AVAudioEngineConfigurationChange`; the mic and speaker share a room (echo loop); a profile lock drains capture (`AudioCapture` drain on `stopAndDrain`). Security angle: a self-echo loop plus auto-submit can drive an agent that runs tools with no human input, so the risk entry is about safety, not just comfort. Text below is drawn from the doc and findings; items marked [confirm] are invented rationale needing author sign-off.
*   ***Proposed Edit:***
    *   Location: insert before `## Open Questions` (line 75) two new sections.
    *   Change:
        ```markdown
        ## Alternatives Considered

        * **Cloud realtime call (OpenCode's shape)**: rejected. It needs a
          hosted model and WebRTC, which the offline policy forbids.
        * **Push-to-talk only**: not rejected, kept as the guaranteed
          barge-in control (see Goals), but not the whole feature, since the
          goal is hands-free turns. [confirm]
        * **Dictation into the composer only**: rejected as the whole
          feature (no spoken reply, no turn-taking) but it is what draft mode
          does per turn. [confirm]

        ## Risks and Mitigations

        * **Self-echo loop**: the speaker output is heard by the mic and
          submitted as a new turn. Mitigation: half-duplex by default (see
          Goals), and a draft-first submit so no echoed text reaches the
          agent without the user pressing send.
        * **Capture loss**: an input device change or a profile lock ends
          capture. Mitigation: the call ends cleanly, releases the mic, and
          shows why; it never retries silently.
        * **Runaway turn loop**: speech while a turn runs. Mitigation: one
          queue-or-drop rule (see Technical Plan); no new turn starts from
          audio while a tool approval is pending.
        ```

### [Issue M1] [`Zoom-out / Design / Structural`] - Severity: Major
*   **Affected Scope:** [Lines 19-25, 29-38, 50-52, 120-125](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L19-L125) (Overview, The Problem, Goals, Technical Plan)
*   **Discrepancy Description:** "Pure integration" is false for VAD: the only VAD is batch, unlinked, and has no streaming endpoint or barge-in onset source.
*   **Rationale:** The doc's turn-taking and barge-in both depend on an incremental speech signal that does not exist.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified (see Verified-true list). `speechSegments` takes a full `[Float]` and has no callers in `Sources/`. The reviewer is right. Not a security item, but the size claim in INDEX.md line 38 and line 98 is wrong too.
*   ***Proposed Edit:***
    *   Location: `## The Problem`, after the sentence ending "VAD, and TTS jobs ... all exist" (line 36-37), and Technical Plan diagram label at line 121.
    *   Change: replace "VAD, and TTS jobs" with "TTS jobs" and add a paragraph: "The VAD is the exception: `FluidAudioSpeechDetector` (`Audio/FluidAudio/FluidAudioVAD.swift`) is batch only, takes a finished sample array, and nothing in the app calls it yet. A call needs an incremental speech-start and speech-end signal, so v1 either runs it over a rolling window of the capture feed or adds a streaming wrapper. [confirm which] The controller owns end-of-speech detection; barge-in uses the same signal." Change the diagram line `VAD end-of-speech + silence window` to `rolling-window VAD (new wrapper) -> end-of-speech`. Change the INDEX.md size note separately ("large; pure integration" to "large; needs a streaming VAD wrapper").

### [Issue M2] [`Zoom-out / Comprehension / Design`] - Severity: Major
*   **Affected Scope:** [Lines 19-21, 120-121](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L19-L121) (Overview, Technical Plan)
*   **Discrepancy Description:** `AudioLiveTranscriber` is built for one take (slow first load, no utterance-final event, no per-turn segmentation, no failure behavior) and the state machine has no preparing state.
*   **Rationale:** The call needs per-turn text and a visible load state.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `Event` has `preparing`, `update`, `stopped`; `isFinal` is end-of-stream only; the first Neural Engine load "can take minutes". The state list at lines 46-48 has no preparing or failed state.
*   ***Proposed Edit:***
    *   Location: Goals, "Call state machine" bullet (lines 46-49), and a new bullet under Technical Plan after line 131.
    *   Change: in the state list insert "preparing (STT model loading)" after "idle" and "failed" before "paused". Add: "Pressing Call with no installed streaming STT model shows the existing model gate and does not start a call. A turn is one VAD-delimited utterance: the controller closes the stream at end-of-speech and reads the final text from `finishAndClose()`, then opens a fresh stream for the next turn. [confirm the segmentation approach] The controller owns the model lifecycle and releases it on End. `stopped(reason)` ends the call with that reason shown."

### [Issue M3] [`Zoom-out / Design`] - Severity: Major
*   **Affected Scope:** [Lines 120-125, 138-139, 150-152](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L120-L152) (Technical Plan, NFR Privacy, file list)
*   **Discrepancy Description:** `AudioCapture` persists WAV chunks (conflicts with the privacy NFR), throws if a recording is active, and ends on device change.
*   **Rationale:** The NFR says only transcripts persist as chat text; capture as it exists writes audio.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified (`Chunker`, `onChunk`, `alreadyRecording`, config-change observer). Security and privacy item (microphone data at rest): the proposed edit defaults to writing no audio and to refusing a call while a recording is active. It adds `AudioCapture.swift` to the file list because a non-persisting tap needs a change there. The mic permission itself stays behind `RecordingPermissions` (unchanged).
*   ***Proposed Edit:***
    *   Location: Non-Functional Requirements, Privacy bullet (line 138-139), and Target Files list (after item 2).
    *   Change: append to Privacy: "A call writes no audio to disk: it uses a capture mode with no chunk persistence, only the live feed. Pressing Call while a recording is active is refused with the existing 'recording already active' message. A device change or capture failure ends the call cleanly and releases the mic." Add file: "`swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioCapture.swift` (Modify). Rationale: a non-persisting capture mode for calls; `onChunk` is not wired." [confirm the mechanism]

### [Issue M4] [`Zoom-out / Design / Comprehension / Style`] - Severity: Major
*   **Affected Scope:** [Lines 53-54, 84-85, 97-102, 181-183](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L53-L183) (Goals, Open Questions, CUJ 2, Verification)
*   **Discrepancy Description:** Barge-in and echo are unmitigated; "lead window" and "lead audio" are undefined and meaningless for TTS output.
*   **Rationale:** Without echo handling the call hears itself and loops.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the doc has no echo or half-duplex language. Echo cancellation is rejected for v1 per the stage-8 note. Safety angle: echo causing self-submitted turns is the main way an unattended call could drive an agent, so the edit defaults to ignoring onset while speaking.
*   ***Proposed Edit:***
    *   Location: Goals "Barge-in" bullet (line 53-54); Open Questions third bullet (lines 84-85); CUJ 2 Then (line 101).
    *   Change: Goals: "Barge-in: while speaking, mic onset is ignored unless it exceeds a threshold above the speaker level [confirm threshold], and a button or key (Escape) always stops playback and returns to listening. Full echo cancellation is out of scope for v1; headphones are recommended." Delete the "Barge-in latency budget" open question. CUJ 2 Then: "playback stops, the state returns to listening, ..." (drop "within the lead window").

### [Issue M5] [`Zoom-out / Design / Style`] - Severity: Major
*   **Affected Scope:** [Lines 53-54, 98-102, 159-161](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L53-L161) (Goals, CUJ 2, file 4)
*   **Discrepancy Description:** Speaking granularity unstated; `sanitizeForSpeech` reads code and paths; the delegate bridge is unnecessary; the singleton is keyed by message UUID.
*   **Rationale:** A long reply with code would be read aloud verbatim.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `sanitizeForSpeech` removes fence markers and markdown symbols but not the code or path text; `stop()` and `isSpeaking` already exist; the delegate already handles finish and cancel. The singleton `shared` keys on `speakingMessageID`.
*   ***Proposed Edit:***
    *   Location: Goals (after "Replies spoken" bullet) and file 4 (lines 159-161).
    *   Change: Goals add: "Speech is sentence by sentence as the reply streams. Fenced code, file paths, and tool output are skipped, with a short spoken placeholder." File 4 becomes: "(Modify) Rationale: a call-owned `AVSpeechSynthesizer` wrapper (not the shared singleton, which is keyed by message UUID) reusing the existing `stop()`/`isSpeaking`; a skip rule for code and paths. No delegate bridge." [confirm whether to extend `AppSpeechSynthesizer` or add a sibling type]

### [Issue M6] [`Zoom-out / Design`] - Severity: Major
*   **Affected Scope:** [Lines 57-59, 80-83, 123, 159-164](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L57-L164) (Goals, Open Questions, Technical Plan, file 4)
*   **Discrepancy Description:** Local TTS is mis-scoped: batch job, heavy-work permit, memory admission, reloads STT each turn.
*   **Rationale:** A local TTS sink cannot meet "reply starts immediately".
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `Qwen3TTSSpeechJob.generate` takes `NativeHeavyWorkPermit` and a memory admission estimate; it is a whole-text job. Cutting it is the smallest change.
*   ***Proposed Edit:***
    *   Location: Goals bullet at lines 57-59; Open Questions second bullet (lines 80-83); diagram line 123; file 4.
    *   Change: Goals: "Replies are spoken with the system voice (`AppSpeechSynthesizer`)." Remove "an installed local TTS voice ... alternative". Add Non-Goal: "No local TTS voice in v1: the TTS job is a batch job under the heavy-work permit and cannot start a reply quickly." Delete the Open Question. Diagram: "Speaker sink: AppSpeechSynthesizer".

### [Issue M7] [`Comprehension / Design`] - Severity: Major
*   **Affected Scope:** [Lines 70-71, 127-131, 150-152](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L70-L152) (Non-Goals, Technical Plan, file 1)
*   **Discrepancy Description:** Submit bridge unnamed; no turn-finished signal; no rule for an utterance during a running turn or speech during thinking; tool approval mid-turn hangs with the mic open.
*   **Rationale:** The controller cannot be built without these.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The doc says only "the same queue path a typed prompt uses". Security item (approvals): voice must never answer an approval; the conservative rule is to pause listening while an approval card is pending. I did not locate the exact send entry point in one read; the author must name it (`AppModel+Skills.swift` `submitExpandedSkillPrompt` is one candidate, not confirmed to be the typed-prompt path).
*   ***Proposed Edit:***
    *   Location: Technical Plan paragraph, lines 127-131.
    *   Change: replace "submits through the same queue path a typed prompt uses" with "submits through `AppModel.<send entry point> [author to name]`, the same entry the composer's Send uses, and treats the chat's turn-finished state as the reply-complete signal. Speech detected during thinking is dropped with a caption note. Speech while a turn runs is not queued. When a tool approval card appears the call moves to paused, stops speaking, closes the mic, and shows a notice; it resumes only when the user presses Unmute."

### [Issue M8] [`Zoom-out / Inventive / Comprehension / Design`] - Severity: Major
*   **Affected Scope:** [Lines 70-71, 75-85, 129-131, 162-164](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L70-L164) (Non-Goals, Open Questions, Technical Plan)
*   **Discrepancy Description:** Draft mode is a dangling "Open Questions candidate" that contradicts the approval rule; auto-submitting mis-transcribed speech to a tool-running agent is a safety risk.
*   **Rationale:** Misheard speech reaching an agent that edits files or runs commands.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and a security item (injection and unintended actions from mis-transcription or room audio). Line 129-131 calls draft mode "an Open Questions candidate" but no such question exists, and CUJ 1 auto-submits. Conservative edit: draft-first is the default in v1; auto-submit is not offered.
*   ***Proposed Edit:***
    *   Location: Technical Plan lines 129-131; CUJ 1 Then (lines 93-95).
    *   Change: Technical Plan: "Each final utterance lands in the composer draft and the user sends it (Send or Return). The call never auto-submits in v1, because misheard or echoed speech would otherwise reach an agent that can run tools. Auto-submit is a later decision." CUJ 1 Then: "VAD closes the utterance, the transcript appears as a caption and in the composer draft; when the user sends it, the turn runs, and the reply is spoken while the transcript shows it." Also adjust Goals "Turn-taking" bullet (lines 50-52) to "the transcript goes to the composer draft".

### [Issue M9] [`Comprehension / Design / Inventive`] - Severity: Major
*   **Affected Scope:** [Lines 61-62, 100-110, 114-125](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L61-L125) (Goals, CUJ 3, Technical Plan)
*   **Discrepancy Description:** State machine omits paused, End, error, failed turn and empty reply; CUJ 3 "no state change" is not tied to a controller state; macOS denial gives no path to System Settings; the access-sheet copy "Check ... the Record page" is wrong for a call.
*   **Rationale:** Undefined states and denial path.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the diagram at lines 115-118 shows only `idle -> listening -> thinking -> speaking`. Security item (microphone consent): the proposal keeps the prompt in `RecordingPermissions` via `.recordingAccessGate` (per `swift/AGENTS.md`), never prompts at launch, and ends the call if access is revoked. I did not read the access-sheet copy strings in this pass; the reviewer's claim about the Record-page wording should be checked by the author against `Audio/RecordingAccessSheet.swift` before editing copy.
*   ***Proposed Edit:***
    *   Location: Technical Plan diagram (lines 114-125) and CUJ 3 (lines 104-110).
    *   Change: diagram state line becomes `idle -> preparing -> listening -> thinking -> speaking -> listening ...` with `paused`, `failed`, `ended` noted as: "Mute: listening <-> paused. End from any state: ended, then idle, mic released. A failed turn or empty reply returns to listening with a caption note. While the access sheet shows, the state is idle." CUJ 3 Then: "the access sheet offers Allow or Not now; Not now leaves the call idle. If macOS access is denied, the sheet's copy points to System Settings and says it is for voice calls. Revoking access during a call ends it." [confirm copy]

### [Issue M10] [`Comprehension / Design / Style`] - Severity: Major
*   **Affected Scope:** [Lines 50-52, 133-137, 163](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L50-L163) (Goals, NFR Latency, file 5)
*   **Discrepancy Description:** The silence window has no default or home; "600 ms" and "existing synthesizer start cost" are untestable.
*   **Rationale:** No testable target.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Note the batch VAD's built-in 0.75 s silence rule (`FluidAudioVAD.swift` comment) means the default must be chosen with that in mind.
*   ***Proposed Edit:***
    *   Location: Latency bullet (lines 134-137); Verification (after line 183).
    *   Change: "Latency target (not a gate): end-of-speech to first spoken word under 1.5 s on the reference Mac, with a silence window default of 700 ms. [confirm numbers]" Add to Manual verification: "measure end-of-speech to first spoken word once and record it."

### [Issue M11] [`Zoom-out / Design / Inventive`] - Severity: Major
*   **Affected Scope:** [Lines 162-164](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L162-L164) (file 5, `AppModel+Persistence.swift`)
*   **Discrepancy Description:** Call preference persisted "with the archive" with no schema or migration note; wrong home.
*   **Rationale:** Archive persistence needs a schema decision.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `State/MacAppSettings.swift` defines `struct MacAppSettings: Codable`, the existing home for app-level preferences. With local TTS cut (M6) only the silence window remains.
*   ***Proposed Edit:***
    *   Location: Target Files item 5 (lines 162-164).
    *   Change: delete file 5, and add under Goals: "The silence window is a constant in v1 (no setting)." If a setting is wanted later, add it to `MacAppSettings` with a default so old files decode.

### [Issue N1] [`Zoom-out / Comprehension / Style`] - Severity: Nit
*   **Affected Scope:** [Lines 3-9, 31-37](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L3-L37) (header, The Problem)
*   **Discrepancy Description:** Relative `swift/...` links do not resolve from the doc folder.
*   **Rationale:** Dead links.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: the doc lives at `docs/designs/opencode2-parity/`, so `swift/...` resolves to a nonexistent folder; `../../OPENCODE2_COMPARISON.md` at line 5 is correct (`docs/OPENCODE2_COMPARISON.md` exists).
*   ***Proposed Edit:***
    *   Location: lines 31, 33, 35, 37.
    *   Change: prefix each `swift/TurboSparkApp/...` link target with `../../../`.

### [Issue N2] [`Style / Comprehension / Structural`] - Severity: Nit
*   **Affected Scope:** [Lines 39-41](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L39-L41) (The Problem)
*   **Discrepancy Description:** "OpenCode ranks this gap eighth" is a wrong attribution; the ranking is OPENCODE2_COMPARISON.md section 10.
*   **Rationale:** Attribution error.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `docs/OPENCODE2_COMPARISON.md` line 502 lists "8. Voice call mode ... only the call loop and delegation tools are missing"; it is the comparison doc's ranking, not OpenCode's.
*   ***Proposed Edit:***
    *   Location: lines 39-41.
    *   Change: "The [comparison](../../OPENCODE2_COMPARISON.md) (section 10) ranks this gap eighth by user value and says only the call loop and delegation tools are missing; this design is larger than that note implies (see Technical Plan)."

### [Issue N3] [`Zoom-out / Comprehension / Style`] - Severity: Nit
*   **Affected Scope:** [Lines 75-85](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L75-L85) (Open Questions)
*   **Discrepancy Description:** Settled defaults (system voice, floating panel) are listed as open; "reposition drag" has no stated need; "zero-model latency" is unmeasured.
*   **Rationale:** Open Questions should be open.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. Local TTS leaves via M6, which removes the speech-engine question; the panel-shape lean is a stated preference with a rationale, which is acceptable for a draft.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none.

### [Issue N4] [`Design / Style`] - Severity: Nit
*   **Affected Scope:** [Lines 27-40, 64-73](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L27-L73) (The Problem, Non-Goals); INDEX line 38
*   **Discrepancy Description:** Problem lacks user motivation; delegation tools are not scoped out; INDEX "large; pure integration" understates the work.
*   **Rationale:** Scope clarity.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** The comparison doc says "delegation tools" are missing too, so a Non-Goal is worth one line. The motivation sentence is not added: a human reader of a parity design set does not need it, and only a zero-context reviewer asked.
*   ***Proposed Edit:***
    *   Location: Non-Goals (after line 73); `docs/designs/opencode2-parity/INDEX.md` line 38 (separate file, author's call).
    *   Change: add "* No delegation tools (spawning or steering other sessions by voice)." INDEX size note: "large; needs a streaming VAD wrapper (see OC-08)".

### [Issue N5] [`Style`] - Severity: Nit
*   **Affected Scope:** [Lines 19-25, 29-38](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L19-L38) (Overview, The Problem)
*   **Discrepancy Description:** Overclaims: "same UX", "deeper than OpenCode's", an unverified negative claim.
*   **Rationale:** Precision.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** "Nothing sends a live transcript to the chat agent" is true as of this checkout (the comparison doc says the same); dating it is cheap. The read-aloud already speaks replies per message (`AppSpeechSynthesizer.shared`); the doc's own text on line 36-38 already says so.
*   ***Proposed Edit:***
    *   Location: line 23 and line 38.
    *   Change: "the same UX" to "a comparable call UX"; "Nothing sends ..." to "As of the 2026-10-31 audit, nothing sends ...".

### [Issue N6] [`Comprehension / Design / Style`] - Severity: Nit
*   **Affected Scope:** [Lines 29-41, 66-67, 127-129, 139-141, 180](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L29-L180)
*   **Discrepancy Description:** Undefined project terms (`AppModel`, send path, steering, offline policy, 21 languages, localization gates); new strings not listed.
*   **Rationale:** Terms need definitions for a cold reader.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. The audience is the project's own maintainers, who know these terms; they are defined in `CLAUDE.md` and `swift/AGENTS.md`. One aside for the author, not a finding: `Localization/Localizable.xcstrings` carries more locale keys than 21 on a quick count (29), so the "21 languages" number may be stale; I did not verify what counts as a language there.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none.

### [Issue N7] [`Style`] - Severity: Nit
*   **Affected Scope:** [Lines 46-47, 49, 95, 142-143](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L46-L143)
*   **Discrepancy Description:** Duplicate glyph-plus-text statement; "spoken while the transcript shows it" vague.
*   **Rationale:** Redundancy.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. The duplication is between Goals and the Accessibility NFR, a normal pairing; the CUJ wording is clear enough.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none.

### [Issue N8] [`Zoom-out / Comprehension / Style`] - Severity: Nit
*   **Affected Scope:** [Lines 167-175](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L167-L175) (test files)
*   **Discrepancy Description:** Near code-level test detail; mutation check unexplained; no VAD seam described.
*   **Rationale:** Altitude.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. The repo's `CLAUDE.md` requires mutation-checking subtle guards, so the sentence is a project convention. The VAD seam follows from M1 and is covered there.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none.

### [Issue N9] [`Style / Structural`] - Severity: Nit
*   **Affected Scope:** [Lines 183-184](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L183-L184) (Verification Plan)
*   **Discrepancy Description:** DESIGN.md and `make design-check` are unlinked.
*   **Rationale:** Navigation.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Cheap and useful to a human reader; `DESIGN.md` is at the repo root.
*   ***Proposed Edit:***
    *   Location: lines 180-184.
    *   Change: link "DESIGN.md section 9" as `[DESIGN.md](../../../DESIGN.md) section 9` and "`make design-check`" as `` `make design-check` (see [AGENTS.md](../../../AGENTS.md)) ``. [confirm section 9 still exists in DESIGN.md; not read in this pass]

### [Issue N10] [`Design`] - Severity: Nit
*   **Affected Scope:** [Lines 156-158](../../../designs/opencode2-parity/OC-08-local-voice-call.md#L156-L158) (file 3)
*   **Discrepancy Description:** `Generation/PromptComposerView.swift` path unverified.
*   **Rationale:** A wrong path would misdirect the implementer.
*   ***AI Suggestion:*** CONTEST
*   ***Justification:*** The path exists: `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerView.swift` (listed by `ls` in this pass). All other listed paths exist too (`Presentation/AppSpeechSynthesizer.swift`, `State/AppModel+Persistence.swift`, `Localization/Localizable.xcstrings`). A path being unconfirmed by the reviewer is not a defect in the doc.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none.

### [Issue N11] [`Structural`] - Severity: Nit
*   **Affected Scope:** whole doc
*   **Discrepancy Description:** Section names and order differ from the standard names ("The Problem", "Technical Plan (Architecture)").
*   **Rationale:** Template conformance.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The standard lives in the EGM template or a team convention, not in a file the reviewer or I could read here (`.kiro/` is absent from this checkout, and the template is not in the repo). The other OC docs use the same headings, which suggests a project template, but I cannot confirm. Needs the user's knowledge, no fix proposed.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: none until the user confirms the template.

## Applied

* Applied: B1, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, N1, N2, N4 (Non-Goal line only), N5, N9.
* Unapplied: N4 INDEX.md size note (separate file, outside the allowed edit scope). Skipped by instruction: N10 (CONTEST), N11 (ESCALATE), N3, N6, N7, N8 (DECLINE).
* Markers left for the author: M7 send entry point is a placeholder marked [author to name] [confirm]; other invented content carries [confirm]. M3 AudioCapture file took slot 5 (freed by M11) to avoid renumbering.
