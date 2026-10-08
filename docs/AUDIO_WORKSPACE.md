# Audio workspace

Open Audio with Command-7. Recordings, imported files, transcripts, presets,
and generated takes belong to the active encrypted profile. The library keeps
original audio and separate derived takes. Failed saves retain a pending take
in memory and offer Retry; do not close the profile while that warning remains.

## Record and transcribe

Choose a microphone, app audio, or both. Check devices and permissions before
starting. System audio uses ScreenCaptureKit on macOS 14; microphone audio uses
AVAudioEngine. TurboSpark playback is excluded and screen frames are not saved.
Pause removes that interval from the saved timeline. Markers remain available
for keyboard and VoiceOver navigation. Recording status remains in the top bar
when switching to another workspace.

Capture writes bounded encrypted WAV chunks independently of inference. A slow
model increases the visible draft backlog instead of stopping recording. A
storage failure stops capture and preserves committed chunks. Reopening an
interrupted project retains those chunks for playback, export, and transcription.

Live text is a draft assembled from bounded windows. Final refinement reads the
saved recording. User transcript edits remain preferred when a newer model
revision arrives. Whisper window timing and Qwen whole-clip timing are approximate
extents, not forced alignment. Subtitles exclude segments with no usable timing.
Speaker and word metadata appear only when a model supplies them.

With a local chat model loaded, Summarize meeting produces excerpt summaries and
action items linked to source transcript positions. Summaries remain tied to the
transcript revision used, so editing a transcript does not silently change an
existing summary.

## Create and compare

Voiceover accepts a script split into sections. Generate section creates a new
take; it does not replace an earlier section. Music retains description, lyrics,
seed, and supported settings. Presets and Duplicate with changes make experiments
repeatable. A/B comparison keeps the current playback position.

WAV export preserves the selected output rate. The video preset resamples to
48 kHz, and M4A is available for sharing. Text, SRT, WebVTT, and structured JSON
exports use the selected saved transcript. Individual tracks or sections can be
exported separately. TurboSpark does not edit or mux video.

## Runtime boundaries

See [the capability audit](AUDIO_WORKSPACE_CAPABILITIES.md) for every discovered
family, callable entry point, readiness reason, and missing adapter. Empty
modules and internal codecs are not presented as runnable jobs.

- Whisper uses its native Metal path. Failure to initialize Metal is explicit.
- MiniMax Music has a native Metal runner and cancellation checkpoints. Its
  native BF16 parity remains unresolved. Earlier controlled f32 parity evidence
  is separate and was not rerun for this workspace. A callable runner does not
  establish task quality or sustained memory and performance bounds.
- Kokoro's Metal runtime is absent. Its portable synthesis is available only as
  an explicit Advanced experiment.
- Qwen3-ASR supports local checkpoints in Advanced. Packed Metal is an explicit
  experimental option; its audio encoder still runs on the CPU.
- Enhancement, separation, alignment, diarization, speech detection, codecs, and
  other transcription families stay unavailable until their native adapters and
  model gates exist. Their source implementations alone do not enable controls.

Downloads use the existing shared queue. Native text/audio and Swift MLX image
execution share one heavyweight permit. An active operation retains the permit
until cancellation has actually drained. The workspace keeps one audio model
resident and unloads it after five idle minutes.

## Storage and qualification

SQLCipher stores versioned project records and normalized asset references.
Capture chunks are encrypted directly from bounded buffers without plaintext
input staging. Chat cleanup includes audio references. Recent unreferenced
assets retain the existing one-minute write grace; audio deletion schedules
cleanup after that grace, with another collection on next library launch.
Exact encrypted backups include the database and assets. Open profile exports
include portable audio records and asset paths, excluding local model bookmarks.

Synthetic DSP and timeline tests, real checkpoint execution, task quality,
performance, actual device capture, and release packaging are separate gates.
The acceptance checks and unfinished qualification remain in the
`audio-workspace` specification. Compilation and fixture tests do not establish
two-hour device reliability, VoiceOver usability, or model quality.

### Built-app smoke evidence

A signed debug bundle using an isolated encrypted profile exercised Command-7,
audio import, playback, native Whisper transcription, transcript and speaker
edits, quit/reopen, and WAV, M4A, and SRT export. The edited revision survived
reopening alongside its original. The 48 kHz WAV and AAC M4A exports retained
the imported clip's 5.132-second duration. A short Kokoro portable experiment
also generated and saved a 24 kHz take through the native bindings.

These runs used local `distil-large-v3` and `kokoro-82m` folders, not newly
verified catalog receipts. They establish those workflow paths, not corpus
quality or performance. Actual microphone capture remained pending at the
macOS permission dialog, which the UI automation tool could not access.
Combined-source capture, real two-hour capture, VoiceOver interaction, and
large-text checks remain manual qualification gates. Recording setup and playback
race fixes passed focused regressions and independent review afterward. The
rebuilt bundle's final UI recheck was blocked because the Mac had been locked.

The final focused app run passed 72 tests, including 38 audio tests. The binding
suite passed 109 tests with 18 gated skips. The app suite's aggregate run still
had Codemode timeout and fan-controller fixture failures; its outdated navigation
count was repaired and passed a focused rerun. Rust workspace build and clippy
passed before a concurrent Dialogue Sidon addition introduced compilation errors.
The final `make swift-app-build` therefore failed in that module, while a direct
Swift build and the debug bundle used the last successfully staged native archive.
Rust workspace tests also exposed five concurrent SAM Audio fixture failures,
and workspace formatting reported changes in that module. Portable Linux audio
compilation remains unverified because the available cross-compiler setup did
not complete. These aggregate failures do not establish a release-ready build.
