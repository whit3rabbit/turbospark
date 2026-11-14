# OC-08: Local voice call mode

Finding: Section 7, "Voice call mode (talk to the agent by voice) -
Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[voice composer entry](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/voice/composer.tsx)
and
[call transport](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/voice/transport.ts).

* Author: whit3rabbit
* Status: Partial (core loop implemented behind a default-off setting; see Implementation Record)
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
  **Decided: (a)**, with the draft's recommendation. The panel is an
  overlay on the conversation pane, draggable by its header, position not
  persisted. See the Implementation Record for every other [confirm] item.

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

## Implementation Record

Implemented 2026-11-14 on branch `fix/b6-tool-edge-cases` (shared tree,
committed through a private index). Status is **Partial**: the core loop
(listen, transcribe, draft, answer, speak, barge-in) is in, behind a
default-off setting; the items under "Deferred" are not.

### What shipped

* **Setting**: General > Voice call > "Voice call mode",
  `MacAppSettings.voiceCallEnabled`, default false, old files decode.
  Off means no Call button, no menu item, and no call object is built (the
  controller is a lazy property). Turning it off ends a live call.
* **Entry**: a Call button beside the composer's dictation button and a
  Chat menu item "Start Voice Call" / "End Voice Call" with the catalog
  command `call.toggle` (Cmd+Shift+Y, rebindable, listed in the Keyboard
  Shortcuts pane). It does not collide with OC-04 (Cmd+T, Cmd+Shift+W/T,
  Cmd+Option+[ ]) or OC-05 (Cmd+Option+Left/Right). Escape is a local
  control key on the panel's Stop speaking button, not a catalog command.
* **Controller**: `Audio/VoiceCallController.swift`, a `@MainActor` state
  machine over idle, preparing, listening, thinking, speaking, paused and
  failed (Mute is listening <-> paused; End from any state is idle with the
  microphone released). Every dependency is a protocol in
  `Audio/VoiceCallSeams.swift`, so tests use fakes.
* **Detector**: `Audio/VoiceCallSpeechDetector.swift`, an energy detector
  counted in samples (no clock), behind `VoiceCallSpeechDetecting`.
* **Speech text**: `Audio/VoiceCallSpeechPlanner.swift`, sentence by
  sentence over the streaming reply, with the skip rules below.
* **Production seams**: `Audio/VoiceCallAdapters.swift` (call-owned
  `AVSpeechSynthesizer` speaker, non-persisting microphone source, the
  per-turn recognizer over the existing `AudioLiveTranscriber`, and the
  speech-model status), `State/AppModel+VoiceCall.swift` (the chat port and
  wiring).
* **UI**: `Audio/VoiceCallPanelView.swift` and `Audio/VoiceCallHostView.swift`
  (panel, Call button, the one start path through `.recordingAccessGate`).
  `RecordingAccessSheet` gained a `.call` purpose (microphone row only,
  call-specific copy and use card, "Start call").
* **Localization**: 39 new keys in all 21 languages, added with
  `--patience` diffs and held by `LocalizationParityTests`
  (`VoiceCallState` titles are listed in its enum-backed test).

### Open Questions and [confirm] items, as decided

| Item | Decision |
|---|---|
| Panel shape | Floating, draggable overlay on the conversation pane. |
| VAD ([confirm which]) | A rolling energy detector over the 16 kHz feed (20 ms frames, 0.12 s sustained onset, 0.7 s silence window), behind a protocol. Silero is batch only and an optional model install, so a streaming wrapper over it is deferred; the swap is one type. |
| Barge-in threshold | Onset must clear 3x the quiet-room level (about +10 dB) while the reply speaks. Headphones recommended, stated in the panel and the access sheet. The button and Escape always stop playback. No echo cancellation. |
| Silence window | 0.7 s, a constant in `VoiceCallTuning`, no setting. |
| Speaker ([confirm extend or sibling]) | A sibling `VoiceCallSpeaker`; `AppSpeechSynthesizer` is untouched. It owns its `AVSpeechSynthesizer`, queues sentences and reports when the queue drains. System voice only. |
| Non-persisting capture ([confirm mechanism]) | `AudioCapture.Configuration.persistsChunks = false`: packets go only to the live feed, nothing is chunked or handed to `onChunk`. The Record page keeps the default true. |
| Segmentation ([confirm]) | One `AudioLiveTranscriber` per turn, closed with `finishAndClose()` at end-of-speech, a fresh one opened after. A `.ready` event was added to `AudioLiveTranscriber.Event` so the call never closes a recognizer that is still loading. |
| Submit bridge ([author to name]) | There is none. The call writes the composer draft only and has no send entry point; the user's own Send goes through `AppModel.run()`. A test scans the port and the controller to keep it that way. |
| Push-to-talk, dictation | Unchanged. The existing dictation button stays; Mute and the panel buttons are the keyboard-reachable controls. |
| CUJ 3 copy ([confirm copy]) | The access sheet in `.call` purpose: "Allow microphone for voice calls", "macOS asks once. Nothing is heard until you press Start call.", Not now leaves the call idle. A denied grant shows the existing "Open System Settings" row. Revoking access during a call ends it with a message. |
| Latency numbers | Targets only, not gates, and not measured: with draft-first the end-of-speech to first-word interval includes the user's Send. |
| DESIGN.md section 9 | Still exists; its automated half was run, its manual checklist was not (see Verification). |

### Behavior that goes beyond the plan text

* **Lane exclusion.** A recognizer is opened only while no turn is busy
  (`isTurnBusy` covers generating, a submission awaiting its hook, and an
  open approval or question). It is closed before the user's Send, cancelled
  if a turn starts beneath it (a typed Send beside a half-spoken
  utterance), and reopened only when the turn ends. A barge-in while the
  turn is still generating stops playback and suppresses the rest of the
  reply at once; recognition waits for the lane and starts from the audio
  kept meanwhile (30 s cap). Escape or Stop speaking during a running turn
  returns to thinking, and to listening when the turn ends.
* **Start refusals.** A press is refused, with the call idle, while a turn
  runs, while a Record-page capture is active (the existing "recording
  already active" text), or when the live text model is missing or cannot
  run. A missing install offers Download through the existing install flow.
  The streaming model is English only, so a non-English system language
  reports that instead of transcribing wrongly. In the other direction the
  Record page refuses to start while a call holds the microphone.
* **Tool output is never spoken.** The call reads only the assistant's text;
  tool results are separate message fields. Fenced code becomes one spoken
  placeholder per block, file paths and URLs become short placeholders
  inside the sentence, and markdown scaffolding is dropped.
* **Leaving the chat ends the call**: a chat switch, or a change of
  section, ends it and releases the microphone.

### Verification run

* Focused suites with injected audio, recognizer, speaker and chat:
  `VoiceCallControllerTests`, `VoiceCallSpeechDetectorTests`,
  `VoiceCallSpeechPlannerTests`, `VoiceCallNoPromptAtLaunchTests`,
  plus `LocalizationParityTests`, `KeyboardShortcutCatalogTests`,
  `KeybindMapTests`, `AudioLiveTextTests`. `make compile-strings`,
  `make design-check` and `generate-settings-catalog.py --check` were run.
* Mutation checks (each applied, shown to redden tests, then restored and
  the suite re-run green; 114 focused tests pass at the end):
  barge-in transition removed reddened
  `testBargeInDuringAStillRunningTurn...` (and the run then crashed on an
  index, since guarded in the test); barge-in threshold scale 3.0 -> 1.0
  reddened `testSpeakerBleedBelowTheBargeInBarDoesNotInterrupt` and
  `testOnsetScaleRaisesTheBarThatSpeakerBleedMustClear`; recognizer not
  cancelled when a turn starts reddened three lane tests; lane guard removed
  from `openTranscriber` reddened
  `testBargeInDuringAStillRunningTurn...`; microphone status guard removed
  reddened `testADeclinedMicrophoneNeverStartsAnything`; approval pause
  removed reddened `testAnApprovalPauses...`.
* Not run: a real call on a loaded model with headphones, the DESIGN.md
  section 9 visual checklist (text sizes, themes, VoiceOver, RTL),
  `make swift-lib` (no Rust ABI changed), the full app suite.

### Deferred (needs a dependency, a download, or a decision)

* **Auto-submit** of a final utterance. Draft-first stays until the user
  decides; it is a safety call about an agent that can run tools.
* **Streaming Silero VAD** (better onset than energy in noise). Needs the
  optional VAD model install and a streaming wrapper over a batch API.
* **Measured latency** (end-of-speech to first spoken word) and the cost of
  reopening the recognizer each turn; both need a manual run on a loaded
  model. If the reopen cost is high, a single recognizer reused with
  `finish()` (which restarts the stream) is the next step.
* **Manual verification** with headphones and the DESIGN.md section 9
  checklist.
* **Non-English speech**, local TTS, echo cancellation, wake word,
  delegation tools: still out of scope as written.

### Auto-submit via the OpenKind decision model: findings and proposal

Status: blocked on user decisions. No code was written. The stop condition
for this follow-up applied, so nothing below is implemented.

**What OpenKind is.** The sibling repo `openkind` is a typed classifier
service, not a policy engine. It speaks the Jev protocol: a request carries a
`state` string and named questions of kind `Noul` (free typed answer),
`Choice` (pick one of N options, optionally with a `__none__` option) or
`Score` (a number). `POST /v1/systemone` returns one typed answer per
question. It has no concept of allow, deny, consent or "may this run". Any
allow/deny meaning is something the caller encodes in the question text and
the thresholds it applies to the answer.

**How the app consumes it today.** `swift/TurboSparkApp/Package.swift`
depends on the `OpenKind` product (path `../../../openkind`, or a pinned
revision in release builds). The only consumer is the TypeSafe pane
(`AppModel+TypeSafeAPI.swift`): it starts a separate `openkindd` child on
`127.0.0.1` through `OpenKindServer`, with `models: ["mock"]`, a user-entered
API key (Keychain only on opt-in, `TypeSafeKeychain`), and
`--playground on`. It is a manual playground. Nothing in the app calls
`OpenKindClient.systemOne` on behalf of another feature, and the daemon only
runs while the user starts it from the TypeSafe tab.

**Why this does not fit as asked.**

1. No allow/deny API exists. A gate needs a rule: which question, which
   options, which model, what confidence counts as allow. That is a policy
   decision about an agent that can run tools, and it has not been made.
2. The daemon serves only `mock` in the app. A real decision needs a pulled
   profile (for example `laya:typed-decisions`, `qwen3guard`, or `kev`), which
   is a multi-GB download and a catalog choice. Confidence calibration is
   model specific (the OpenKind docs say to validate thresholds on your own
   data), and a mis-calibrated gate fails open in the way this feature must
   not.
3. Lifecycle and credentials: the voice call would depend on the TypeSafe
   daemon being running, on its port, and on its API key. Reusing the
   TypeSafe key or Keychain item from the call changes credential handling
   (needs approval). A second daemon or an in-process engine is a new
   dependency or a change to openkind.
4. Latency: a decision call sits between end-of-speech and the send, on top
   of recognition. It needs a hard timeout, and the only safe timeout result
   is draft-first.

**Proposal (for the user to accept or change).**

* Gate shape: one extra stage in `VoiceCallController` after a final,
  non-empty, unmuted utterance. A `VoiceCallSubmitDecider` protocol (async,
  returns `allow`, `confirm`, or `deny`) injected like the other seams, with
  a production adapter over `OpenKindClient` and a fake for tests. Anything
  except an explicit `allow` within the timeout (error, timeout, daemon down,
  `confirm`, `deny`, ambiguous) lands the text in the composer as today.
* Narrow submit path: a single `submitAutoVoiceUtterance(_:chatID:)` on
  `AppModel`, reachable only from that stage, which re-checks at call time
  that the setting is on, the decision was `allow`, no tool approval is
  pending, no turn is busy (ZP-23 lane), the call is unmuted, and the text
  is unchanged since the decision. The source-scan test is updated to allow
  exactly that one call site and mutation-checked.
* Setting: "Auto-send voice utterances", default off, with a
  `SettingsControlCatalog` row and all 21 languages.
* Question encoding (needs the user's call): a `Choice` question over the
  utterance and the last assistant turn, options such as `dictation`,
  `command`, `unclear`, `risky`, with `__none__` present, where only
  `dictation` or `command` above a threshold maps to `allow`. Utterances that
  mention shell, delete, install, push or similar, and everything while a
  pending approval exists, map to `confirm` without calling the model.
* Model: a dedicated small profile, pulled on demand, never bundled. Until
  one is chosen and its threshold validated on recorded utterances, the
  `mock` model must not be accepted as a decider (the adapter refuses it).

**Decisions needed from the user.**

1. Which OpenKind profile is the decider, and who downloads it (opt-in pull
   from the settings row, or manual).
2. The decision rule: question text, options, confidence threshold, and the
   keyword pre-filter list. Or a different, simpler mapping.
3. Daemon ownership: reuse the TypeSafe daemon (needs it running, and the
   call reading its key) or a separate voice-gate daemon with its own key.
   Either touches credentials or process lifecycle.
4. Timeout budget (proposal: 1.5 s, fail closed).
5. Whether a rare-but-allowed `allow` on a `command` class is acceptable at
   all, or auto-send should be limited to plain dictation with no tools.

--------------------------------------------------------------------------------

*Drafted with EGM*
