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
`FluidAudioFamily` case with `hasAdapter` true. Every other Swift-hosted row is
unsupported, and the pickers hide it. `hasAdapter` means the workspace can run
the family end to end, so it flips in the change that wires the family's jobs
and residency, not in the one that adds its adapter. Until then the family's
rows ship in the catalog but no picker offers them.

`AudioEngineLabel` names the engine beside each model: in the picker rows,
under the selected model, in Model details and on the Advanced page. For Core
ML it adds where Core ML places the run, from the catalog's measured `compute`
(`ane`, `gpu`, `cpu`, `mixed`). It never uses the units a loader requests:
Silero VAD requests the Neural Engine and runs on the CPU. Rows the HTTP
server refuses also show App only.

## Families

| Family | Pin | Install layout | Entry point | Placement |
|---|---|---|---|---|
| `fluidaudio_silero_vad` | `silero-vad-coreml@b419383c` | flat | own `MLModel`, then `VadManager(config:vadModel:)` | CPU |
| `fluidaudio_diarizer` | `speaker-diarization-coreml@7eab74f3` | flat | own MLModels and PLDA psi, then `OfflineDiarizerModels(...)` and `initialize(models:)` | mixed |
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
4. Flip `hasAdapter` when the family's jobs and residency are wired.
5. Put the model's terms on the row: an `acknowledgments` entry when they
   carry conditions beyond attribution. `Resources/Licenses/` is for code the
   app links, not weights the user downloads. Add an env-gated real-model
   test that runs the same input twice and asserts FluidAudio's caches were
   never created.
6. Run the audio network guard and the Swift gates below.

## Not done

- The speech rows and adapters ship, but no job calls them, so `hasAdapter`
  is still false everywhere. The transcribe loop, residency (load, unload,
  memory admission) and the picker's text for `funasr_model_license` come in
  one change once another branch's edits to `AudioWorkspaceController` and
  `AudioModelPickerView` land. VAD and speaker labels, then TTS, follow.
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
