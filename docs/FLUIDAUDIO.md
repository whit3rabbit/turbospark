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
unsupported, and the pickers hide it. The change that adds an adapter flips
its families in the same commit.

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
| `fluidaudio_sensevoice` | `sensevoice-small-coreml@74bdbe74` | flat | `SenseVoiceModels.load(from:precision: .int8)` | Neural Engine |
| `fluidaudio_paraformer_zh` | `paraformer-large-zh-coreml@9b4faaa0` | flat | `ParaformerModels.load(from:precision: .int8)` | Neural Engine |
| `fluidaudio_parakeet_ja` | `parakeet-0.6b-ja-coreml@2952296f` | flat | own MLModels and vocab, then `AsrModels(..., version: .tdtJa)` | Neural Engine |

The install layout is the row's `install_subdir`. FluidAudio's folder names
drop `-coreml` (`Repo.folderName`), which is why PocketTTS reads
`Models/pocket-tts`. Re-measure placement with `coreml_plan` on any pin bump.

## Adding a family

1. Generate the row with `cargo run -p turbospark-catalog --example audio_pin`
   (explicit prefixes, `.mlmodelc` only). Pin every file the loader opens,
   including voices and constants that `AssetDownloader` would otherwise
   fetch. Set `install_subdir` when the loader reads a fixed layout.
2. Add the family to `profile_capabilities` with its measured `compute`, and
   to `FluidAudioFamily` with the same layout.
3. Write the adapter under `Audio/FluidAudio/`. Check every model's input and
   output names and shapes against what FluidAudio feeds it before building a
   manager: FluidAudio answers a mismatch with `preconditionFailure`, and only
   on first use, so a successful load proves nothing. Refuse any voice the row
   does not pin.
4. Flip `hasAdapter` for the family in the same change.
5. Add the model's license to `Resources/Licenses/` and an env-gated
   real-model test that also asserts `~/.cache/fluidaudio` was never created.
6. Run the audio network guard and the Swift gates below.

## Not done

- No family has an adapter, and no `swift_coreml` row ships. Adapters and rows
  land per group: ASR, then VAD and speaker labels, then TTS.
- FluidAudio models are not loaded, kept resident or unloaded yet.
  `AudioWorkspaceController`'s residency code was being changed in another
  branch when this landed, so that wiring comes with the first adapter.
- Licenses still to settle before their rows ship: Cohere Transcribe (its card
  says both Apache-2.0 and CC-BY-NC-4.0), the FunASR model license for
  SenseVoice and Paraformer, and two non-commercial PocketTTS voices
  (`cosette`, `jean`), which stay unpinned.
- SenseVoice int8 gave a worse transcript on its second run than its first.
  Check determinism and the fp16 encoder before that row ships.

## Checks

```sh
make compile-strings
(cd swift/TurboSparkApp && swift test --filter 'FluidAudio|AudioEngineLabel|AudioRuntimeProvider|LocalizationParityTests')
make swift-test-app
make swift-app && make app-bundle && scripts/verify-app-bundle.sh
```

The app link only fails at `make swift-app`, so run it after any change to
this dependency.
