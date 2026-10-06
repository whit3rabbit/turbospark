# Audio interface

How TurboSpark presents audio: one surface per capability, the Rust engine
underneath, and Swift as the native helper on top. This is the design record
for the audio work. Code paths named here are the source of truth; when they
disagree with this page, fix the page.

## Split of responsibilities

```
+------------------------------- macOS app (Swift) -------------------------------+
|  UI: composer mic + recording strip, attachment chips, preview pane, bubble      |
|      player, app-audio sheet, top-bar recording pill, Settings > Audio & Voice   |
|  OS I/O (only the OS can do these):                                              |
|      AVAudioEngine mic tap   Core Audio process tap   AVAudioPlayer playback     |
|      permissions (TCC)       AAC encode (AVAudioFile)                            |
|  Native helper, labelled in the UI, used only when the engine has no model:     |
|      SFSpeechRecognizer (on device only)   AVSpeechSynthesizer                   |
+------------------------------------+--------------------------------------------+
                                     | TurboSparkAudio / TurboSparkAudioSession
                                     | (swift/TurboSpark, wraps ts_audio_*)
+------------------------------------v--------------------------------------------+
|  crates/ffi   ts_audio_capabilities_json  ts_audio_probe_json  ts_audio_peaks_json |
|               ts_audio_convert_json  ts_audio_display_level                      |
|               ts_audio_session_open/close  ts_audio_transcribe_json              |
|               ts_audio_synthesize_json                                           |
+------------------------------------+--------------------------------------------+
                                     |
+------------------------------------v--------------------------------------------+
|  crates/audio (portable Rust)                                                    |
|      decode (symphonia: WAV AIFF CAF FLAC MP3 AAC/ALAC-in-MP4 Ogg Vorbis)        |
|      resample (windowed sinc)  downmix  trim  WAV write                          |
|      peaks  RMS  meter level                                                     |
|      speech: SpeechToText / TextToSpeech traits, capability table, refusal      |
+---------------------------------------------------------------------------------+
```

Rules that follow from the split:

- Every waveform, meter level, normalization, trim and WAV export is computed
  in Rust. Swift never decodes or resamples; it captures samples, hands files
  to the engine, and draws what comes back.
- Speech to text, text to speech and music belong to the engine. The engine
  is always asked first. No audio model family exists yet, so every engine
  task reports `active: false` with one shared reason
  (`turbospark_audio::speech::NO_AUDIO_MODEL_REASON`), and
  `ts_audio_session_open` refuses every path.
- Until a family lands, the app's Apple helpers fill the gap under the
  `Automatic` engine choice, and the UI says which provider produced each
  result ("TurboSpark engine" or "Apple (on device)"). `TurboSpark engine
  only` turns the helpers off and shows the engine's reason instead.
- AAC is the one encode step done natively: the engine renders a float WAV
  (trim, rate and channels applied in Rust), then `AVAudioFile` hands it to
  Apple's encoder. No pure-Rust AAC encoder is comparable.
- No third-party Swift packages. DSWaveformImage, FFmpegKit (LGPL/GPL, large
  binary) and BlackHole (GPL loopback driver) were evaluated and rejected.
  The waveform is a SwiftUI `Canvas`; app-audio capture is a Core Audio
  process tap. symphonia (MPL-2.0, pure Rust, 0.5.x for MSRV 1.82) is the
  only new dependency.

## Availability contract

One table decides what every audio control may offer:
`AudioCapabilities.resolve` (pure, unit-tested in `AudioInterfaceTests`).
It mirrors the vision contract (`sessionInfo.vision.{active, reason}`): a
control the user cannot use stays visible and disabled, and its help text is
the reason.

| Capability | Usable when | Otherwise shows |
|---|---|---|
| Microphone capture | in-app audio on, mic authorized or not yet asked | "Microphone access is off..." with Open System Settings |
| Transcription | engine STT active with a model configured, else (Automatic) Apple on-device recognizer authorized | the engine reason (Engine only), "On-device speech recognition is not available...", or the denied notice |
| Read aloud | engine TTS active with a model, else Apple synthesizer (works with in-app audio off) | the engine reason under Engine only |
| App audio capture | in-app audio on, macOS 14.2+ | "Requires macOS 14.2 or later." |
| Audio attachments | in-app audio on | picker does not offer audio types |

In-app audio is behind `TurboSpark.audio.experimentalEnabled` (default off).
With it off, the composer mic is exactly the old button: a forward to macOS
system dictation.

## Capability 1: microphone capture and dictation

Entry point: the existing composer mic (`PromptAudioInputButton`), same
place in the footer's trailing cluster.

- Click: dictate (record, transcribe, insert text).
- Context menu: Dictate, Record voice note (attach audio), Record audio
  from an app..., Use macOS Dictation, Audio settings...

While active, `ComposerRecordingStrip` replaces the prompt editor; the
footer stays where the pointer is.

```
idle          [ prompt editor ......................................... ]
                                               [ring] [x] [mic] [Send]

recording     [ * Recording  |||.|||||..|||||.||  0:07   (x)  [Insert text] ]
                                          [ring] [x] [stop(red)] [Send]

transcribing  [ (spin) Transcribing...  Apple (on device)                 ]

denied        [ mic/  Microphone access is off for TurboSpark.
                                   Open System Settings   (x) ]
```

- Esc discards, Return finishes. Red is used only here and always next to
  the word "Recording".
- Reduce Motion swaps the live waveform for a single level meter.
- The meter level is the engine's (`ts_audio_display_level`, real-time
  safe), computed on the capture thread.
- Text is appended to the end of the draft (`TextEditor` exposes no caret),
  focus returns, and VoiceOver announces "Transcript inserted".
- A dictation that cannot be transcribed is not lost: it becomes a voice
  note attachment and a toast says why.
- Auto-send after dictation is off by default.
- Recording stops at the maximum length (Settings, default 10 minutes).

## Capability 2: audio attachments, playback and trim

Entry points: the plus menu's file picker (audio types offered when in-app
audio is on), drag and drop onto the composer (same importer, same type
gate), voice notes, and app-audio captures.

Model: an audio attachment is an `AppPromptAttachment` whose extension is in
the ENGINE's decoder list (`AudioFileClass`, read from
`ts_audio_capabilities_json`). The transcript lives in `extractedText`, so
the existing send path inlines it as
`--- Attachment: name.m4a (audio transcript) ---`. No new stored field on
the archived type (Gotcha 13); sent messages gain an optional
`audioPaths` (decoded with `decodeIfPresent`) for replay only.

```
chip          [ ||.|||. memo.m4a              x ]
                        Audio * 1.2 MB * transcript 2.1K chars
              (live: "Transcribing...", or "No transcript: open to retry")

preview pane  [ |||||.|||||||||[=====selected=====]||||.||||  ]
              (<<5) (>) (5>>)  0:12 / 0:42          [1x v]  [Export...]
              0:10 - 0:25                [Clear selection] [Use selection]
              Transcript  Apple (on device)            [Transcribe again]
              ...transcript text, selectable...

bubble        (>) |||.||||||.|||||||||.||  0:00 / 0:42
```

- Send refuses audio without a transcript, with a sentence, rather than
  sending an empty block (`AppPromptAttachment.untranscribedAudioRefusal`,
  shared by the send path and the queue drain).
- Refused formats (Opus, WebM, WMA) fail at import with "Unsupported audio
  format (.opus). Convert to M4A or WAV and try again."
- "Use selection" trims in Rust (source frames, before any resampling),
  stores the result as a new managed asset, clears the transcript and
  transcribes again.
- One player for the whole app (`AudioPlaybackController`); playback and
  read aloud stop each other.
- No bare Space or arrow shortcuts: a key equivalent fires before the prompt
  editor sees the keystroke.

## Capability 3: app audio capture (macOS 14.2+)

Entry points: the mic's context menu and the plus menu's "Capture App
Audio...". Deliberately not on the footer: rare, and the most sensitive
audio action.

```
sheet    Record App Audio
         Only record audio you have the right to use. Protected streams
         may record as silence.
         +-------------------------------------------+
         | (spk) All system audio                    |
         | (app) Music                               |
         | (app) Safari               Not playing audio |
         +-------------------------------------------+
         [Refresh]                    [Cancel] [Start Recording]

top bar  ( * Music 1:24 [stop] )   <- every section, click stops and attaches
```

- Backend: a Core Audio process tap (`CATapDescription`) on a private
  aggregate device clocked by the default output, written as float WAV
  through the same `CaptureSink` as the microphone. "All system audio"
  excludes TurboSpark itself.
- Not ScreenCaptureKit, because it would ask for Screen Recording permission
  for an audio-only feature. Not a loopback driver, which is a system install
  and GPL.
- Apps that have never opened an audio stream have no process object and
  are listed but cannot be selected.
- The result is an ordinary audio attachment (capability 2).
- Recording the microphone together with app audio is not implemented.

## Capability 4: conversion and export

There is no general "converter" screen. Conversion does two jobs.

- **Invisible.** Every speech input, engine or Apple, is normalized by the
  engine to 16 kHz mono 16-bit WAV first (`ConvertOptions::speech`), so
  providers see byte-identical audio.
- **Visible.** The export sheet, reached from the preview pane, a bubble's
  context menu, and Read's context menu ("Save as Audio..."):

```
Export Audio
 Format       [ M4A (AAC) | WAV ]
 Quality      [128 kbps v]          (M4A only)
 Sample rate  [Original v]
 Mono         [ ]
                              [Cancel] [Export...]  -> NSSavePanel
```

WAV is written by the engine; M4A is engine WAV, then Apple AAC (max 48
kHz).

## Settings > Audio & Voice

`AudioSettingsPaneView`, category Personal. Sections:

- **Experimental:** Enable in-app audio.
- **TurboSpark audio engine:** the engine's own table, verbatim, with each
  task's status and reason, plus its decodable formats.
- **Input:** the current microphone, a link to Sound settings, a live
  microphone test meter, and the permission state.
- **Speech engine:** Automatic, TurboSpark engine only, or macOS Dictation.
  Shows "Transcribes with", the recognition permission, and Send
  automatically after dictation.
- **Read aloud:** voice (Apple voices, or the language default) and speaking
  rate.
- **App audio capture:** status and consent text.
- **Audio attachments:** transcribe automatically, and maximum recording
  length.

The engine model paths (`TurboSpark.audio.engineSpeechModelPath`,
`engineVoiceModelPath`) are stored but not exposed: there is nothing to
install until an audio family exists.

## Platform and packaging

- `scripts/make-app-bundle.sh` adds `NSMicrophoneUsageDescription`,
  `NSSpeechRecognitionUsageDescription` and `NSAudioCaptureUsageDescription`.
- Signing stays ad hoc with no sandbox. Enabling hardened runtime later
  requires the `com.apple.security.device.audio-input` entitlement.
- `crates/audio` is `publish = false`. It is portable and passes the
  portable-subset `cargo check`, but the CI step does not list it yet: add
  `-p turbospark-audio` to the "Portable-subset check (Linux)" step in
  `.github/workflows/ci.yml` (the session that wrote this could not edit
  workflow files).

## Verification status

Verified in a Linux container:

- `cargo test -p turbospark-audio`: decode round trips, resampler length and
  aliasing, trim, peaks, refusal. Mutation-checked: the anti-alias cutoff
  and the trim offset.
- The `ts_audio_*` bodies: built and tested through a harness that compiles
  the real `api/audio.rs` and `audio_session.rs` (`crates/ffi` itself does
  not build on Linux at baseline). The header was compiled as C and C++, and
  a C program linked against the staticlib.
- Localization parity and the settings catalog: checked with scripts that
  port the Swift tests.

Not verified here, and required on macOS before merge:

- `make swift-lib && make swift-test`: `AudioSurfaceTests`,
  `AudioInterfaceTests`, `LocalizationParityTests`. No Swift compiler was
  available, so none of the Swift has been compiled.
- `make swift-app && make app-bundle`, then a manual pass:
  - flag on and off
  - dictation, and the denied permission state
  - a voice note
  - M4A, FLAC and Opus attachments (Opus should be refused)
  - trim and export
  - app-audio capture on 14.2+ with the top-bar pill
  - Save as Audio
  - VoiceOver, Reduce Motion and Increase Contrast

Process-tap code paths (`SystemAudioCaptureService`) carry the most
compile and runtime risk.

Known open issues from static review:

- A file dropped onto the prompt text itself may be taken by the editor's
  `NSTextView` (inserting the path) before the composer's
  `dropDestination` sees it; drops elsewhere on the composer import.
- Capture sinks write to `AVAudioFile` on the IO thread. That is buffered,
  but a ring buffer drained off-thread would be the real-time-clean design.
- `renderToFile` waits on the synthesizer's write callback with no timeout.
