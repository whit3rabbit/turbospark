# OC-08: Local voice call mode

Finding: Section 7, "Voice call mode (talk to the agent by voice) -
Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[voice composer entry](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/voice/composer.tsx)
and
[call transport](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/voice/transport.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Talk to a chat: the user speaks, an on-device streaming STT model
transcribes, the transcript becomes the turn prompt for the selected
chat, and the reply is spoken back through the existing speech stack.
Everything is local; there is no WebRTC and no hosted realtime model.
OpenCode's call rides a cloud GPT-Live session over WebRTC; the
TurboSpark version is a comparable call UX built from capture,
streaming STT, VAD, and TTS the app already owns.

## The Problem

The app's audio stack is deeper than OpenCode's but is aimed at
recordings, not conversation: mic capture
([AudioCapture.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioCapture.swift)),
live streaming STT
([AudioLiveTranscriber.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLiveTranscriber.swift)),
and TTS jobs
([Qwen3TTSSpeechJob.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Audio/Qwen3TTSSpeechJob.swift))
all exist, plus a system-voice read-aloud
([AppSpeechSynthesizer.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/AppSpeechSynthesizer.swift)).
The VAD is the exception: `FluidAudioSpeechDetector`
(`Audio/FluidAudio/FluidAudioVAD.swift`) is batch only, takes a
finished sample array, and nothing in the app calls it yet. A call
needs an incremental speech-start and speech-end signal, so v1 either
runs it over a rolling window of the capture feed or adds a streaming
wrapper. [confirm which] The controller owns end-of-speech detection;
barge-in uses the same signal.
As of the 2026-10-31 audit, nothing sends a live transcript to the chat
agent or speaks its reply.
The [comparison](../../OPENCODE2_COMPARISON.md) (section 10) ranks this
gap eighth by user value and says only the call loop and delegation
tools are missing; this design is larger than that note implies (see
Technical Plan).

## Goals

* A Call control beside the composer opens a call session bound to the
  selected chat.
* Call state machine: idle, preparing (STT model loading), listening
  (mic open, VAD running), thinking (turn running), speaking (reply
  audio), failed, paused (muted).
  State is visible at all times with a glyph plus text, never color
  alone.
* Turn-taking: end-of-speech via the existing VAD plus a silence
  window; the transcript goes to the composer draft.
* Barge-in: while speaking, mic onset is ignored unless it exceeds a
  threshold above the speaker level [confirm threshold], and a button
  or key (Escape) always stops playback and returns to listening. Full
  echo cancellation is out of scope for v1; headphones are recommended.
* Mute pauses capture without ending the call; End returns the call to
  idle and releases capture.
* Replies are spoken with the system voice (`AppSpeechSynthesizer`).
* Speech is sentence by sentence as the reply streams. Fenced code,
  file paths, and tool output are skipped, with a short spoken
  placeholder.
* The silence window is a constant in v1 (no setting). If a setting is
  wanted later, add it to `MacAppSettings` with a default so old files
  decode.
* Captions: the last few utterances stay visible in the call panel.
* Mic access goes through the existing recording access gate: the Allow
  flow appears only when the user presses Call, never at launch.

## Non-Goals

* No WebRTC, no hosted realtime model, no cloud service (offline
  policy).
* No wake word or hands-free re-arm: the call ends explicitly.
* No autonomous tool approval by voice in v1: approvals surface as the
  existing cards; the user answers in the window (voice text goes to
  the composer draft, not to the approval card).
* No meeting-notes or audio-workspace integration changes; the call is
  a chat surface.
* No delegation tools (spawning or steering other sessions by voice).
* No local TTS voice in v1: the TTS job is a batch job under the
  heavy-work permit and cannot start a reply quickly.

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

## Open Questions

* **Panel shape**: (a) a floating, draggable panel over the chat,
  OpenCode's shape; (b) a dock pane. Lean (a) in v1; the panel is small,
  state-only, and closes with the call.

## Critical User Journeys

### CUJ 1: hands-free turn

* **Given**: a chat selected, Call pressed, mic allowed.
* **When**: the user speaks a question and stops.
* **Then**: VAD closes the utterance, the transcript appears as a
  caption and in the composer draft; when the user sends it, the turn
  runs, and the reply is spoken while the transcript shows it.

### CUJ 2: barge-in

* **Given**: the reply is being spoken.
* **When**: the user starts talking.
* **Then**: playback stops, the state returns to listening, and the partial reply stays readable in the transcript.

### CUJ 3: permission declined

* **Given**: mic access declined earlier.
* **When**: Call is pressed.
* **Then**: the access sheet offers Allow or Not now; Not now leaves
  the call idle and chat typing still works. If macOS access is denied,
  the sheet's copy points to System Settings and says it is for voice
  calls. Revoking access during a call ends it. [confirm copy]

## Technical Plan (Architecture)

```text
VoiceCallController (@MainActor state machine)
  idle -> preparing -> listening -> thinking -> speaking -> listening ...
  paused, failed, ended: Mute is listening <-> paused. End from any
  state is ended, then idle, mic released. A failed turn or empty reply
  returns to listening with a caption note. While the access sheet
  shows, the state is idle.
        |                    |
        |                    +-- reply text -> speaker sink
        v
AudioCapture -> AudioLiveTranscriber -> utterance final
  rolling-window VAD (new wrapper) -> end-of-speech -> composer draft
        |
Speaker sink: AppSpeechSynthesizer
  barge-in: VAD onset -> stop() -> back to listening
```

The controller sits beside `AppModel` and reads the selected chat. Each
final utterance lands in the composer draft and the user sends it (Send
or Return). The call never auto-submits in v1, because misheard or
echoed speech would otherwise reach an agent that can run tools.
Auto-submit is a later decision. The user's send goes through
`AppModel.<send entry point> [author to name] [confirm]`, the same entry
the composer's Send uses, so steering, permissions, and usage accounting
behave identically; the controller treats the chat's turn-finished state
as the reply-complete signal. Speech detected during thinking is dropped
with a caption note. Speech while a turn runs is not queued. When a tool
approval card appears the call moves to paused, stops speaking, closes
the mic, and shows a notice; it resumes only when the user presses
Unmute.

Pressing Call with no installed streaming STT model shows the existing
model gate and does not start a call. A turn is one VAD-delimited
utterance: the controller closes the stream at end-of-speech and reads
the final text from `finishAndClose()`, then opens a fresh stream for
the next turn. [confirm the segmentation approach] The controller owns
the model lifecycle and releases it on End. `stopped(reason)` ends the
call with that reason shown.

## Non-Functional Requirements

* **Latency**: target (not a gate): end-of-speech to first spoken word
  under 1.5 s on the reference Mac, with a silence window default of
  700 ms. [confirm numbers]
* **Privacy**: all audio stays on device; captions and transcripts
  persist only as part of the chat like typed text. A call writes no
  audio to disk: it uses a capture mode with no chunk persistence, only
  the live feed. Pressing Call while a recording is active is refused
  with the existing "recording already active" message. A device change
  or capture failure ends the call cleanly and releases the mic.
* **Localization**: call controls, states, and captions copy in 21
  languages in the same change.
* **Accessibility**: every state is a glyph plus text; the panel is
  fully keyboard-operable; VoiceOver announces state changes; captions
  are a live region.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCallController.swift` (Create)
   * **Rationale**: the state machine, VAD turn-taking, barge-in, mute,
     and the submit bridge into the existing send path.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCallPanelView.swift` (Create)
   * **Rationale**: the floating panel: state, captions, mute, End,
     reposition drag.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptComposerView.swift` (Modify)
   * **Rationale**: the Call entry beside the composer with the
     recording access gate wiring.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/AppSpeechSynthesizer.swift` (Modify)
   * **Rationale**: a call-owned `AVSpeechSynthesizer` wrapper (not the
     shared singleton, which is keyed by message UUID) reusing the
     existing `stop()`/`isSpeaking`; a skip rule for code and paths. No
     delegate bridge. [confirm whether to extend `AppSpeechSynthesizer`
     or add a sibling type]
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioCapture.swift` (Modify)
   * **Rationale**: a non-persisting capture mode for calls; `onChunk`
     is not wired. [confirm the mechanism]
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/VoiceCallControllerTests.swift` (Create)
   * **Rationale**: state machine transitions with injected VAD and
     clock events, barge-in cut-over, mute pause, declined-permission
     path, submit bridge passthrough. Mutation check: remove the
     barge-in transition and assert its test fails.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/VoiceCallNoPromptAtLaunchTests.swift` (Create)
   * **Rationale**: source-scan test (RecordingPermissionsTests style)
     asserting the call sources request access only from the Call
     action path.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suites plus localization gates and
  `make design-check` (see [AGENTS.md](../../../AGENTS.md)).
* **Manual**: a full call on a loaded model with headphones (avoid
  feedback), barge-in behavior, mute, End, declined-permission path.
  Measure end-of-speech to first spoken word once and record it.
  Run the [DESIGN.md](../../../DESIGN.md) section 9 checklist on the
  panel. [confirm section 9 still exists in DESIGN.md]

--------------------------------------------------------------------------------

*Drafted with EGM*
