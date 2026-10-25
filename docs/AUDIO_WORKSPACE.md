# Audio workspace

Open Audio with Command-7. Recordings, imported files, transcripts, presets,
and generated takes belong to the active encrypted profile. The library keeps
original audio and separate derived takes. Failed saves retain a pending take
in memory and offer Retry; do not close the profile while that warning remains.

## Record and transcribe

Speech-to-Text has three modes, chosen with the pills at the top of the page.
Voice Notes records the microphone, Meeting records the microphone and the app
playing the call, and Any Audio captures another app's audio or transcribes an
imported file. Each mode lists only its own takes and files them into its own
collection (Voice notes, Meetings, App audio), whatever the Record page's type
is set to.

One record bar sits at the bottom of every mode, and only one capture
runs at a time: switching modes while recording changes the view, not the take,
and the recording mode's pill shows a live dot. The top bar's recording badge
opens the page or mode that started the capture. The Record page stays for
Sound takes and hand-picked sources.

Each mode keeps one model per job: Live Text (Live Captions in Any Audio),
Final Transcript, and Speakers. The header's model button lists them, each with
its state, Load Model or Unload, Download, and a Change list of the catalog
profiles that can do that job. The choice is saved per mode and job
(`speech_to_text:<mode>:<job>`) and applies to the next take. A mode with no
choice of its own keeps the earlier behavior: the shared Speech-to-Text choice
for the final pass, Parakeet Unified for live text, and the first installed
diarization model for speakers.

Deleting a model moves every mode that used it to the best installed model left
for that job.

Speaker labels are optional per mode, with a Label speakers switch on the
Speakers slot. Meeting starts with it on, Voice Notes and Any Audio with it off.
A take follows the mode of its preset, from either page, and an imported file
follows Any Audio. With the switch off, the final pass skips labeling. Sound
takes and takes with hand-picked sources keep the first installed diarization
model.

When a Meeting take's final pass labels its speakers, Name the Speakers lists
each voice with its talk time, its longest line, a few seconds of it to play,
and a name field. Names and merges are saved on the take by the voice's
diarization id (`speakerNames`, `speakerMerges`), never in the segments, which
keep that id in `speakerID`. A rename therefore reaches every line, every
transcript revision, and the Text, SRT, WebVTT, and JSON exports, and clearing a
name or separating a merge restores the default label. JSON exports put the name
in `speaker` and keep the raw id.

Speaker intelligence assists naming with one-click suggestion chips:
- **Local speaker ('You')**: Compares RMS energy across microphone and system
  audio channels during active speech turns to detect when you are speaking.
- **Cross-meeting recognition**: Extracts 64-band log-mel voiceprint vectors via
  Accelerate vDSP on clean speech turns and matches them against saved profiles
  in `speaker_profiles.json` using cosine similarity (threshold >= 0.78),
  displaying suggested names with match confidence percentages.
- **Merges**: Offers Merge Into choices, and naming a voice updates the profile
  store so the voice is recognized in future meetings.

The transcript's Name Speakers... button opens the same sheet later, and each
line's speaker menu moves only that line to another voice.

Custom vocabulary and jargon terms (`AudioCustomVocabulary`) can be configured
via the Vocabulary... button in the header and transcript editor. Terms and their
phonetic or spelling aliases are persisted in `custom_vocabulary.json` and applied
automatically to live speech and final transcripts using word-boundary safe regex
replacement (`\b`), preventing substring corruption within larger words.

Installed > Audio lists the same models with the Text and Image lifecycle:
installed, local folders, installs found on disk without a receipt, installs
that need attention, installs the catalog no longer lists, and the pinned
profiles this Mac can download. Its Used By card shows and changes which mode
uses a model for which job. Memory figures are the catalog's estimate
(`resident_memory_bytes`), not a measurement.

Choose a microphone, app audio, or both. Check devices and permissions before
starting. On macOS 14.2+, system and meeting audio use native CoreAudio process
taps (`CATapDescription`) with `NSAudioCaptureUsageDescription`, avoiding Screen
Recording permission prompts and the purple menu-bar indicator. On earlier
versions (macOS 14.0-14.1), it automatically falls back to ScreenCaptureKit.
Microphone audio uses AVAudioEngine. TurboSpark playback is excluded and screen
frames are not saved. Pause removes that interval from the saved timeline.
Markers remain available for keyboard and VoiceOver navigation. Recording status
remains in the top bar when switching to another workspace.

Capture writes bounded encrypted WAV chunks independently of inference. A slow
model increases the visible draft backlog instead of stopping recording. A
storage failure stops capture and preserves committed chunks. Reopening an
interrupted project retains those chunks for playback, export, and transcription.

Live text streams when the Parakeet Unified EN model is installed and the take
is English (the transcription language, or the first system language when it is
Automatic): words appear about a second behind speech
(`ts_audio_stream_*`, no heavy-work permit, so chat and image work keep running).
The stream hears the same mix of microphone and app audio as the saved take, and
its text is saved into one draft revision that each save replaces. Otherwise, or
if the stream stops, live text is a draft assembled from bounded windows every
15 seconds. A voice reference recording has no live text. Final refinement reads
the saved recording with the selected batch model; a fresh install prefers
Parakeet Ultra, then Redux, then Whisper Base, and a saved choice always wins. User transcript edits remain preferred when a newer model
revision arrives. Whisper window timing and Qwen whole-clip timing are approximate
extents, not forced alignment. Subtitles exclude segments with no usable timing.
Speaker and word metadata appear only when a model supplies them.

### Notes and Summary

In the audio library inspector, takes offer a segmented view switcher:
`[ Transcript | Notes & Summary ]`. The Notes & Summary tab provides:
- **Dedicated model picker**: Select any installed chat model, with
  `AudioMeetingNotesAdvisor` highlighting lightweight, high-speed models
  suitable for Apple Silicon (recommending `Qwen 3.5 1.5B` and `Qwen 1.5B`
  variants with 32k context).
- **Structured notes**: Generates executive Markdown notes containing Key
  Decisions, Action Items, and Discussion Summaries attributed to identified
  speakers.
- **Interactive jump chips**: Embedded timestamp chips (e.g. `[02:15]`) allow
  jumping directly to that point in the audio player to verify decisions.
- **Ground-truth immutability**: Meeting summaries strictly update `item.summary`
  and never overwrite, edit, or invalidate original transcript text. Summaries
  remain tied to the specific transcript revision used.

## Organize and search

The Record page's type (Voice note, Meeting or call, Sound) sets the sources and
live text and files new takes into its collection. The collection is named in
the current language the first time and kept after that. Changing a source by
hand makes the take a custom one, left unfiled. The library filters by All,
Favorites, Recordings, Imported and Needs transcript, by collection and by tag,
and groups by Today, This week and Earlier. Tags are free-form, up to 20 per
take, matched ignoring case. A record that cannot be read is copied aside under
`quarantine:` and skipped, and the original stays so its audio is kept.

Search matches exact words in transcripts, titles, collections and tags through
an FTS5 index in the encrypted profile database, with a substring fallback for
text the index cannot segment (CJK). With EmbeddingGemma 2 installed it also
matches by meaning (transcript chunks, 768-d) and by sound (10 s audio windows,
256-d), fused by reciprocal rank. Each hit can play from its time. The index is
derived from the library and rebuilt from it; a background pass embeds new and
changed takes while nothing is recording.

Sound matches are experimental. On the recall gate's 25 real clips in 16
classes
(`crates/runtime/tests/coreml_sound_recall.rs`), the right clip ranks first
for half of the text queries and in the top three for 75%. Every
query-to-clip cosine falls between 0.59 and 0.77, related or not, so the 0.15
floor drops nothing and no floor would separate them: any take with audio can
show up as a sound match. The meaning floor is untuned.

## Create and compare

Text-to-Speech, Voice Cloning, and Voice Design have separate model choices and
voice settings. [Qwen3-TTS](QWEN3_TTS.md) uses the native Swift MLX provider.
Clone from an imported or saved 1 to 30 second reference region with its exact
transcript, or explicitly choose voice embedding only. A model with the
`voice_clone` capability ([MegaTTS3](../crates/audio/src/tts/megatts3/README.md))
clones through the native session instead: it needs no transcript, asks you to
confirm you have the right to use the voice (retrimming the reference withdraws
the confirmation), and exposes steps, intelligibility, similarity and seed under
Cloning settings. Voice Design accepts a
voice description. Generated speech can become a reusable cloning reference.
Reference playback and trim controls are separate from generated-take playback.

### The Voice Cloning page

The page has two columns, and falls back to one for a narrow window or Extra
Large text. Voice setup sits in the leading column:

- **Reference voice.** Drop an audio file on the card, import one, record your
  voice in place, or choose saved audio from the library sheet. The sheet also
  lists references that past takes used, and keeps them after their source is
  deleted. A recording made here uses the microphone only, stops at 30
  seconds, and becomes the reference; it does not change the Record page's
  sources. Without microphone access it opens the same access sheet as the
  Record page instead of a macOS prompt. The chosen clip shows a waveform of the trimmed region, where that
  region sits in the source, a Start slider, and a Duration field.
- **Engine cards.** MegaTTS3 shows Consent and Quality (Fast, Balanced, Best,
  or a custom step count). Qwen3-TTS shows the reference transcript and its
  sampling settings.
- **Audio model.** The picker, status indicators, and install controls. When a
  model is installed, the card offers explicit Load Model and Unload actions, a
  Loaded indicator when resident in memory, and suppresses redundant download
  controls. It moves to the top until a model is chosen, because the model
  decides which engine cards follow.

The main column holds the work:

- **Steps.** A strip shows Voice, Consent or Transcript, Script, and Generate,
  each with a glyph and a state word. Choosing a step scrolls to its section.
- **Script.** A composer with a character count against the 6,000 character
  limit. The reason Generate speech is disabled appears next to the button.
- **Progress.** While a run is active, a card lists its stages (loading the
  model, preparing the reference, generating, saving), with a progress bar,
  the elapsed time, and Cancel.
- **Takes.** Every take in clone mode, newest first. Each row plays in place
  and shows the reference and quality that made it. Reuse reference restores
  that reference region and its transcript. Consent is never carried over:
  confirm it again for the restored region.

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
- OmniVoice provides native Metal acceleration for synthesis and consent-gated
  voice cloning at 24 kHz, with transparent portable CPU fallback. It is
  available on the Voice Cloning page (`clonesNatively`) and callable through
  the server audio API. The page sends the reference transcript when there is
  one and never requires it. Its installs ship the Boson community-licensed Higgs
  tokenizer, whose install terms the app shows before download.
- Qwen3-ASR supports local checkpoints in Advanced. Packed Metal is an explicit
  experimental option; its audio encoder still runs on the CPU.
- Qwen3-TTS runs Base, CustomVoice, and VoiceDesign MLX checkpoints in Swift.
  The Rust session and HTTP Audio API do not execute this family.
- Speaker diarization runs the pinned Core ML pyannote community-1 export
  (`speaker-diarization-coreml`, macOS 15+) from the Advanced page. Once it is
  installed, the Record page's final pass labels each segment of a take with a
  speaker ("Speaker 1", "Speaker 2", ...). A take longer than the model's
  30 minutes is labeled in parts of at most 30 minutes that share 5 minutes
  and start on whole seconds (`AudioSpeakerParts`). A later part's voice takes
  an earlier label only when at least 8 seconds, and 70% of its time in the
  shared span, agree with one earlier voice. Any other voice starts a new
  speaker, so a voice that is silent in the shared span can appear twice (merge
  it in Name the Speakers), but voices are never joined on weaker evidence.
  The sheet says how many parts were used. See [Core ML models](COREML.md).
- Enhancement, separation, alignment, speech detection, codecs, the CPU
  diarization families (Sortformer, Nemotron), and other transcription families
  stay unavailable until their native adapters and model gates exist. Their
  source implementations alone do not enable controls.

Downloads use the existing shared queue. Native text/audio and Swift MLX image
execution share one heavyweight permit. An active operation retains the permit
until cancellation has actually drained. The workspace keeps one audio model
resident and unloads it after five idle minutes, or immediately via the explicit
Unload button or the window toolbar's eject menu.
Qwen3-TTS releases its model at the end of each speech job.

Loading another model, such as a meeting's speaker model after its transcriber,
replaces the resident one. Live text opens its own stream for each take beside
that one model. Holding several batch models loaded at once would need the
controller to keep a session per model, which it does not.

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
