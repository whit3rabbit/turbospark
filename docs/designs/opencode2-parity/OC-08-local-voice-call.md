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
TurboSpark version is the same UX built from capture, streaming STT,
VAD, and TTS the app already owns.

## The Problem

The app's audio stack is deeper than OpenCode's but is aimed at
recordings, not conversation: mic capture
([AudioCapture.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioCapture.swift)),
live streaming STT
([AudioLiveTranscriber.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Audio/AudioLiveTranscriber.swift)),
VAD, and TTS jobs
([Qwen3TTSSpeechJob.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Audio/Qwen3TTSSpeechJob.swift))
all exist, plus a system-voice read-aloud
([AppSpeechSynthesizer.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Presentation/AppSpeechSynthesizer.swift)).
Nothing sends a live transcript to the chat agent or speaks its reply.
OpenCode ranks this gap eighth by user value and notes it is
integration work, not new capability.

## Goals

* A Call control beside the composer opens a call session bound to the
  selected chat.
* Call state machine: idle, listening (mic open, VAD running),
  thinking (turn running), speaking (reply audio), paused (muted).
  State is visible at all times with a glyph plus text, never color
  alone.
* Turn-taking: end-of-speech via the existing VAD plus a silence
  window; the turn is submitted through the existing send path with the
  transcript as the prompt.
* Barge-in: speech detected while speaking stops playback and returns
  to listening.
* Mute pauses capture without ending the call; End returns the call to
  idle and releases capture.
* Replies spoken with the system voice by default
  (`AppSpeechSynthesizer`); an installed local TTS voice is selectable
  in call settings as an alternative.
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

## Open Questions

* **Panel shape**: (a) a floating, draggable panel over the chat,
  OpenCode's shape; (b) a dock pane. Lean (a) in v1; the panel is small,
  state-only, and closes with the call.
* **Speech engine**: system voice for zero-model latency versus local
  TTS for the app's own voices. Lean: system voice default with local
  TTS as the opt-in alternative, because a call reply should start
  immediately.
* **Barge-in latency budget**: how much lead audio to keep before
  cut-off so speech does not clip (tune in review).

## Critical User Journeys

### CUJ 1: hands-free turn

* **Given**: a chat selected, Call pressed, mic allowed.
* **When**: the user speaks a question and stops.
* **Then**: VAD closes the utterance, the transcript appears as a
  caption and as the sent prompt, the turn runs, and the reply is
  spoken while the transcript shows it.

### CUJ 2: barge-in

* **Given**: the reply is being spoken.
* **When**: the user starts talking.
* **Then**: playback stops within the lead window, the state returns to
  listening, and the partial reply stays readable in the transcript.

### CUJ 3: permission declined

* **Given**: mic access declined earlier.
* **When**: Call is pressed.
* **Then**: the existing access sheet offers Allow or Not now; Not now
  ends the call attempt with no state change and chat typing still
  works.

## Technical Plan (Architecture)

```text
VoiceCallController (@MainActor state machine)
  idle -> listening -> thinking -> speaking -> listening ...
        |                    |
        |                    +-- reply text -> speaker sink
        v
AudioCapture -> AudioLiveTranscriber -> utterance final
  VAD end-of-speech + silence window -> submit via existing send path
        |
Speaker sink: AppSpeechSynthesizer (default) or local TTS job
  barge-in: VAD onset -> stop() -> back to listening
```

The controller sits beside `AppModel`, reads the selected chat, and
submits through the same queue path a typed prompt uses, so steering,
permissions, and usage accounting behave identically. Utterances are
also inserted as the composer draft when "draft mode" is enabled, for
users who want to review before sending (Open Questions candidate).

## Non-Functional Requirements

* **Latency**: end-of-speech to submit under 600 ms plus the silence
  window; reply start to speech under the existing synthesizer start
  cost.
* **Privacy**: all audio stays on device; captions and transcripts
  persist only as part of the chat like typed text.
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
   * **Rationale**: a barge-in stop hook and a delegate bridge for the
     controller; local TTS sink as the alternative implementation.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: persist the call preference (speech engine choice,
     silence window) with the archive.
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
  `make design-check`.
* **Manual**: a full call on a loaded model with headphones (avoid
  feedback), barge-in behavior, mute, End, declined-permission path.
  Run the DESIGN.md section 9 checklist on the panel.

--------------------------------------------------------------------------------

*Drafted with EGM*
