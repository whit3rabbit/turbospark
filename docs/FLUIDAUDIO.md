# FluidAudio in the app

The app runs some audio models through FluidAudio, FluidInference's Swift
package of Core ML pipelines. Rust still owns the catalog: it pins each model
to a commit, downloads it, verifies every byte, lists it and deletes it, like
any audio profile ([COREML.md](COREML.md), Swift-hosted Core ML). The app
loads the verified files and FluidAudio runs them. Nothing here is served over
the HTTP API: the server refuses these aliases.

The measurements behind every value on this page are in
[FLUIDAUDIO_PHASE0.md](FLUIDAUDIO_PHASE0.md).

## The pin

`swift/TurboSparkApp/Package.swift` depends on FluidAudio `exact: "0.15.5"`
(commit `19600a485baa4998812e4654b70d2bab8f2c9949`).

- Every tag from v0.15.6 declares the `NemoTextProcessing` binary target, a
  prebuilt Rust staticlib. The app already links two Rust archives and C++,
  and its binary carries three compact-unwind personality functions, the
  format's maximum. A third Rust archive brings a duplicate
  `_rust_eh_personality` and a fourth personality (upstream issue #880).
- Its opt-out trait needs a Swift 6.2 manifest. CI's oldest lane builds with
  Xcode 16.4, which reads the plain manifest and links the staticlib anyway.
- v0.15.5 has no package dependencies and no binary target.
  `FastClusterWrapper` is C++, whose personality the app already links.

Before moving the pin, build the app and compare
`objdump --macho --unwind-info <TurboSparkApp> | grep -A4 "Personality functions"`
with main's binary, run `scripts/verify-app-bundle.sh`, and rerun every check
on this page against the new version. A `from:` requirement is never safe
here: it resolves to the newest 0.x.

## Offline contract

FluidAudio is built to fetch its own models from Hugging Face `main`. Here
every model comes from a verified install instead, so `FluidAudioEngine` sets
two switches once per process, before FluidAudio does any work:

| Switch | What it stops |
|---|---|
| `ModelHub.offlineMode = true` | `ModelHub` downloads, and the delete-then-download-again that follows a failed load |
| `ModelRegistry.baseURL = "turbospark-offline:"` | `AssetDownloader`, which ignores offline mode (Kokoro voice packs and lexicon, PocketTTS voices and constants). A missing file now fails at once instead of fetching `main` into the install |

Two things are not covered by either switch, so adapters avoid them:

- `OfflineDiarizerManager.prepareModels` can delete its folder, and `process`
  calls it when no models were injected. Build the models and call
  `initialize(models:)` first.
- Kokoro's text input goes through a G2P singleton tied to
  `~/.cache/fluidaudio`. Use `synthesizeFromPhonemesDetailed` only.

Tests hold the contract:

- `FluidAudioIsolationTests` scans `Sources/TurboSparkApp`: only
  `Audio/FluidAudio/` imports FluidAudio, no file that does calls a download,
  cache or purge API, and only `FluidAudioEngine.swift` writes the switches.
- `FluidAudioEngineTests` runs FluidAudio with the switches set: a missing
  model is refused without a fetch and nothing beside it is deleted, and an
  unpinned voice fails in well under a second without writing a file.

## Routing and labels

`AudioRuntimeProvider` decides who runs a profile from its catalog backend:
the Rust session, Qwen3-TTS in Swift MLX, FluidAudio, or nothing. A
`swift_coreml` row routes to FluidAudio only when its family is a
`FluidAudioFamily` case whose `supportedTasks` holds the row's task. Every
other Swift-hosted row is unsupported, and the pickers hide it. A task joins
`supportedTasks` (which is what makes `hasAdapter` true) in the change that
wires the family's job and residency, not in the one that adds its adapter.
Until then a family's rows ship in the catalog and install from the CLI
(`turbospark-model pull-audio <alias>`). No picker offers such a row and the
Installed list never offers it for download, but a CLI install still appears
under Installed, marked "Not usable in this app version", so it can be
deleted.

`AudioEngineLabel` names the engine beside each model: in the picker rows,
under the selected model, in Model details and on the Advanced page. For Core
ML it adds where Core ML places the run, from the catalog's measured `compute`
(`ane`, `gpu`, `cpu`, `mixed`). It never uses the units a loader requests:
Silero VAD requests the Neural Engine and runs on the CPU. Rows the HTTP
server refuses also show App only.

## Families

| Family | Pin | Install layout | Entry point | Placement |
|---|---|---|---|---|
| `fluidaudio_silero_vad` | `silero-vad-coreml@b419383c` | flat | `SileroSpeechDetector` (`FluidAudioVAD.swift`): own `MLModel` loaded for the CPU, then `VadManager(config:vadModel:)` and `segmentSpeech` | CPU |
| `fluidaudio_kokoro_ane` | `kokoro-82m-coreml@e61d9ee7` (`ANE/`) | `kokoro-82m-coreml` | `KokoroAneManager(variant: .english, directory:)`, phonemes only | mixed |
| `fluidaudio_pocket_tts` | `pocket-tts-coreml@91748676` (`v2.1/english/`) | `Models/pocket-tts` | `PocketTtsManager(language: .english, directory:, precision: .fp16, placement: .gpu)` | GPU |
| `fluidaudio_sensevoice` | `sensevoice-small-coreml@74bdbe74` | flat | none: not shipped (see Not done) | Neural Engine |
| `fluidaudio_paraformer_zh` | `paraformer-large-zh-coreml@9b4faaa0` | flat | checked MLModels and vocab, then `ParaformerModels(...)` and `ParaformerManager` | Neural Engine |
| `fluidaudio_parakeet_ja` | `parakeet-0.6b-ja-coreml@2952296f` | flat | checked MLModels and vocab, then `AsrModels(..., version: .tdtJa)` and `AsrManager` | Neural Engine |

The install layout is the row's `install_subdir`. FluidAudio's folder names
drop `-coreml` (`Repo.folderName`), which is why PocketTTS reads
`Models/pocket-tts`. Re-measure placement with `coreml_plan` on any pin bump.

## Speech recognition

The catalog ships `parakeet-ja-coreml` (Japanese, CC-BY-4.0, from
`nvidia/parakeet-tdt_ctc-0.6b-ja`) and `paraformer-zh-coreml` (Mandarin,
FunASR Model Open Source License 1.1). The Paraformer row carries a
`funasr_model_license` acknowledgment: the license asks for credit and the
model name kept, Alibaba can revise it, and it ends for anyone who disparages
the model.

`FluidAudioEngine.transcriber(for:installRoot:)` builds an adapter in
`FluidAudioASR.swift`. Every model goes through `FluidAudioModelLoader`, which
compares its inputs and outputs with the signature recorded from the pinned
export (`FluidAudioSignatures.swift`) and throws instead of letting FluidAudio
trap. Any later path that loads a FluidAudio model from a folder the catalog
did not verify must use the same loader.

Window length is per adapter (`maxWindowSamples`):

- Parakeet ja: no limit. FluidAudio splits input over its 15 s model window
  into overlapping chunks and merges them. Token timings are returned.
- Paraformer: 15 s. FluidAudio cuts its input past 512 encoder frames (about
  30.7 s) and its output past 128 tokens, with a log line and no error. The
  adapter refuses a longer window. Text only, no timings.

Each adapter gave the same text for the same window twice, offline, on this
M4 Max (`FluidAudioASRRealModelTests`).

### In the workspace

Both families run on the Transcribe page and refine finished recordings on
the Record page. Live drafts during a recording stay on Rust sessions.

- **Residency** (`AudioWorkspaceFluid.swift`): one resident model at a time,
  shared with the Rust session and Qwen3-TTS. Loading verifies every byte of
  the install (Stop cancels it), admits the row's `resident_memory_bytes`
  within the audio memory budget, then builds the adapter off the main actor.
  The first load compiles for the Neural Engine and takes minutes, and the
  status strip says so while it runs. Unload, idle unload, model delete and
  shutdown release it.
- **Verified installs only.** A FluidAudio row offers no local folder, and
  residency refuses one: the signature check stops a trap, not wrong weights.
- **Speaker labels.** After a FluidAudio transcription, both pages run the
  same best-effort diarization pass as the Rust session path
  (`completeFinalTranscription`). A mode with labels on gets them from
  either engine.
- **Windows** (`AudioTranscriptionPlan`): Parakeet ja reads 30 s windows that
  overlap by 2 s. Each word is kept by the window that owns its start time,
  with the cut mid-overlap. The workspace's word reconcile splits on spaces,
  so it never matches Japanese or Chinese. Paraformer reads windows of 15 s
  less 10 ms with no overlap, because without token times an overlap cannot
  be cut: a word on a boundary can split. Its segments are window extents,
  shown as approximate timing. A window under 0.3 s is padded with silence,
  the shortest input Parakeet accepts.
- The Rust session path keeps its windows exactly as before (30 s every 29 s,
  word reconcile), now planned by the same function.

## Adding a family

1. Generate the row with `cargo run -p turbospark-catalog --example audio_pin`
   (explicit prefixes, `.mlmodelc` only). Pin every file the loader opens,
   including voices and constants that `AssetDownloader` would otherwise
   fetch. Set `install_subdir` when the loader reads a fixed layout.
2. Add the family to `profile_capabilities` with its measured `compute`, and
   to `FluidAudioFamily` with the same layout.
3. Write the adapter under `Audio/FluidAudio/`. Record each model's inputs and
   outputs in `FluidAudioSignatures.swift` and load it through
   `FluidAudioModelLoader`: FluidAudio answers a mismatch with
   `preconditionFailure`, and only on first use, so a successful load proves
   nothing. Use FluidAudio's model initializers, never its `load(from:)`
   helpers, which skip the check. Read the manager for silent limits (input
   truncation, token caps) and refuse input past them. Refuse any voice the
   row does not pin.
4. Once the family's job in `beginRun` and its residency are wired, add the
   task to `supportedTasks`. That is what makes `hasAdapter` true, and
   `AudioRuntimeProviderTests` pins the sets.
5. Put the model's terms on the row: an `acknowledgments` entry when they
   carry conditions beyond attribution. `Resources/Licenses/` is for code the
   app links, not weights the user downloads. Add an env-gated real-model
   test that runs the same input twice and asserts FluidAudio's caches were
   never created.
6. Run the audio network guard and the Swift gates below.

## Not done

- Speaker labels use the `speaker-diarization-coreml` row, which the Rust
  runner executes (`pyannote_community1`). There is no FluidAudio diarizer
  family: it would be the same pin under a second alias.
- Speech detection has its adapter (`SileroSpeechDetector`, with real-model
  tests, also run under a network-denied sandbox) but no job or residency in
  the workspace, so `hasAdapter` is still false for it. Then TTS.
  `hasAdapter` is true only for `fluidaudio_parakeet_ja` and
  `fluidaudio_paraformer_zh`.
- Live drafts while recording do not use FluidAudio models.
- SenseVoice is not shipped. On this M4 Max both its int8 and fp16 encoders
  gave different text for the same window on repeated runs in one process,
  and fp16 once returned nothing. Recheck on the next pin.
- Licenses still to settle before their rows ship: Cohere Transcribe (its card
  says both Apache-2.0 and CC-BY-NC-4.0), and two non-commercial PocketTTS
  voices (`cosette`, `jean`), which stay unpinned.

## Checks

```sh
make compile-strings
(cd swift/TurboSparkApp && swift test --filter 'FluidAudio|AudioEngineLabel|AudioRuntimeProvider|LocalizationParityTests')
# Real models: a folder holding each pinned install as parakeet-ja/ and
# paraformer/. sandbox-exec cannot wrap `swift test`, so run the bundle.
(cd swift/TurboSparkApp && TURBOSPARK_FLUIDAUDIO_MODELS=<dir> \
  sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  xcrun xctest -XCTest TurboSparkAppTests.FluidAudioASRRealModelTests \
  .build/out/Products/Debug/TurboSparkAppTests.xctest)
make swift-test-app
make swift-app && make app-bundle && scripts/verify-app-bundle.sh
```

The app link only fails at `make swift-app`, so run it after any change to
this dependency.
