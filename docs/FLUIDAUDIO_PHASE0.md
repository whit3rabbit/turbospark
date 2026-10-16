# FluidAudio: Phase 0 spike

The evidence record for adding FluidAudio (FluidInference's Swift package of
Core ML audio models) to the Audio workspace. It covers the version pin, the
offline contract, how each planned family loads from a pinned install, and
where Core ML actually places each model. Run on 2026-10-16, M4 Max,
macOS 26.5.2, Xcode 27.1 (Swift 6.4), branch `spike/fluidaudio`.

Nothing here is a quality, memory, or release claim. Transcripts come from
short `say` clips. Timings are single runs on a shared host.

## Version

Pin `exact: "0.15.5"` (tag commit `19600a485baa4998812e4654b70d2bab8f2c9949`).

- `from: "0.12.4"` resolves to the newest 0.x tag (0.17.7 on this date).
- Every tag from v0.15.6 through v0.17.7 declares the `NemoTextProcessing`
  binary target, a prebuilt Rust staticlib. The app already links two Rust
  archives (ours with `_rust_eh_personality` hidden by `nmedit`, and
  Syntext's) plus C++, and main's debug binary reports exactly three
  compact-unwind personality functions, the format's maximum
  (`objdump --macho --unwind-info`). A third Rust archive cannot fit.
  Upstream issue #880. The opt-out trait needs a Swift 6.2 manifest, and CI's
  oldest lane is Xcode 16.4.
- v0.15.0 to v0.15.5 have no package dependencies and no binary target.
  v0.12.5 to v0.13.x require swift-transformers 1.2.0 or newer, which
  conflicts with `Vendor/Qwen3TTS` (`.upToNextMinor(from: "1.1.9")`).
- Resolving 0.15.5 into `swift/TurboSparkApp` adds exactly one
  `Package.resolved` entry and no transitive packages.
- FluidAudio's own CI builds on `macos-15` with Swift 6.1 (its workflow
  files at the tag), which matches our Xcode 16.4 lane. That is evidence the
  sources compile there, not a link result.

## App link: PASS on Xcode 27.1

Release build of `TurboSparkApp` with FluidAudio 0.15.5 and an env-gated
probe in `main()` that calls `VadManager`, `OfflineDiarizerManager`,
`KokoroAneManager`, `PocketTtsManager` and `CoherePipeline`, so dead
stripping keeps them.

| Check | Spike | main `dist` (2026-10-14) |
|---|---|---|
| Compact-unwind personality functions | 3 | 3 |
| Personalities | `___gxx_personality_v0`, our `_rust_eh_personality` (non-external), Syntext's (external) | same |
| FluidAudio symbols in the binary | 20,874, FastCluster present | none |
| `verify-app-bundle.sh` | OK: 12 Mach-O at or below 14.0, stayed up 10 s with an empty HOME | |
| Resource bundles copied | no FluidAudio bundle (it ships none) | |

How it was run, and what it does not cover:

- The staged Rust archive and the four CLI binaries were main's existing
  builds (app-localized archive, CLI from 2026-10-14), not rebuilt in the
  worktree. The bundle used `make-app-bundle.sh --skip-build` with main's
  `openkindd`. The Rust side is unchanged in this spike, so this checks the
  Swift link, not a release.
- Two host fixes were needed, neither caused by FluidAudio:
  `xcodebuild -downloadComponent MetalToolchain` (Xcode 27.1 here had none,
  so mlx-swift's shaders could not compile in a fresh build directory), and
  the documented Xcode 27 placeholder `CSyntext.o`
  (`clang -c -x c /dev/null -o .build/out/Products/Release/CSyntext.o`,
  swift/AGENTS.md).
- Xcode 26.6 and 16.4 were not run. CI's `package-macos` covers both, but
  only on pushes to `main`. FluidAudio's own CI compiles on Swift 6.1, the
  16.4 lane's toolchain, which is evidence for the compile and not for our
  link.

In the bundled app, under the network-denied sandbox, the probe's VAD step
returned without error and the next step crashed the process (below).

## FluidAudio traps on a wrong model

Handed the VAD model in place of the diarizer's FBank model, the bundled app
exited with SIGTRAP after logging `FBANK model is missing 'audio' multiarray
input`. `OfflineEmbeddingExtractor.init` calls `preconditionFailure` there
(line 99). The crash report puts it inside `OfflineDiarizerManager.process`
(line 192), which builds the extractor: `initialize(models:)` accepted the
wrong model, and the crash came on the first diarization. So a load that
succeeds proves nothing about the inputs.
FluidAudio 0.15.5 has 9 `fatalError`/`preconditionFailure` sites, 21
precondition-style calls under `Diarizer/Offline` alone.

A pinned, hash-verified install never hands it a wrong file. The adapters
still check each `MLModel`'s input and output descriptions against the names
and shapes FluidAudio expects before constructing a manager, the same rule
docs/COREML.md sets for the Rust runner (validate up front, never rely on
catching). Which preconditions input data can reach (empty or very short
audio, for example) needs its own pass in each adapter PR.

## Offline contract

Harness: `docs/verification/fluidaudio-phase0/faprobe` (a standalone package
depending only on FluidAudio 0.15.5), run under
`sandbox-exec -p '(version 1)(allow default)(deny network*)'`. It sets
`ModelHub.offlineMode = true` and `ModelRegistry.baseURL =
"turbospark-offline:"` before any other FluidAudio call.

| Check | Result |
|---|---|
| Missing model, `ModelHub.loadModels` | `modelMissing`, no fetch |
| Corrupt `weight.bin`, `ModelHub.loadModels` | throws, folder kept (without offline mode this path deletes and re-downloads) |
| Unpinned Kokoro voice (`AssetDownloader`, ignores offline mode) | fails in 16 ms: `unsupported URL` for `turbospark-offline:/.../resolve/main/ANE/bf_emma.bin` |
| PocketTTS with a misnamed folder | `networkDisabled(download(pocket-tts/v2.1/english))`, nothing fetched |
| Files changed in any install, both suites | none (size and mtime of every file) |
| `~/.cache/fluidaudio`, `~/Library/Application Support/FluidAudio` | never created |

The unpinned-voice URL shows the risk the blocked base URL closes:
`AssetDownloader` builds `resolve/main` URLs, so without it a missing voice
would be fetched from a moving branch into the install.

## Families

Pins are the HF commit `main` resolved to on 2026-10-16. The full file lists
with sizes and sha256 (LFS sha256 checked after download, git blobs hashed)
are `docs/verification/fluidaudio-phase0/<family>.pins.txt`, produced by
`hf_pin.py`.

| Family | Pin | Files / bytes | Install layout | Entry point |
|---|---|---|---|---|
| Silero VAD | `silero-vad-coreml@b419383c` | 5 / 1.06 MB | flat | own `MLModel` -> `VadManager(config:vadModel:)` |
| Offline diarizer | `speaker-diarization-coreml@7eab74f3` | 23 / 21.6 MB | flat | own MLModels + ported psi loader -> `OfflineDiarizerModels(...)` -> `initialize(models:)` |
| Kokoro ANE English | `kokoro-82m-coreml@e61d9ee7` | 52 / 96.1 MB | `<root>/kokoro-82m-coreml/ANE/` | `KokoroAneManager(variant: .english, directory: root)` + `synthesizeFromPhonemesDetailed` |
| PocketTTS English | `pocket-tts-coreml@91748676` | 48 / 529.6 MB | `<root>/Models/pocket-tts/v2.1/english/` | `PocketTtsManager(language: .english, directory: root, precision: .fp16, placement: .gpu)` |
| SenseVoice int8 | `sensevoice-small-coreml@74bdbe74` | 9 / 239.9 MB | flat | `SenseVoiceModels.load(from:precision: .int8)` -> `SenseVoiceManager(models:)` |
| Paraformer zh int8 | `paraformer-large-zh-coreml@9b4faaa0` | 17 / 222.1 MB | flat | `ParaformerModels.load(from:precision: .int8)` -> `ParaformerManager(models:)` |
| Parakeet ja (TDT) | `parakeet-0.6b-ja-coreml@2952296f` | 21 / 619.1 MB | flat | own MLModels + vocab -> `AsrModels(..., version: .tdtJa)` -> `AsrManager(config:models:)` |

Notes on the layouts:

- PocketTTS's repo folder is `pocket-tts`, not `pocket-tts-coreml`
  (`Repo.folderName` strips `-coreml`). The first run used the wrong name and
  failed safely.
- Kokoro loads seven stages: Albert, PostAlbert, Alignment, Prosody,
  `KokoroNoise_v2`, Vocoder, Tail. The pinned set also has `KokoroNoise`,
  `KokoroProsody_v2` and `KokoroTail_v2`, which the row can drop.
- PocketTTS's fp16 GPU set is `cond_prefill`, `flowlm_step`,
  `flow_decoder_fused`, `mimi_decoder` and all of `constants_bin/`. If any is
  missing, `ensureModels` tries `ModelHub.download`. `bos_before_voice.bin`
  must be present or it is backfilled through `AssetDownloader`.
- Cohere Transcribe was not fetched: its license is unsettled (the card's
  metadata says Apache-2.0, its body says CC-BY-NC-4.0).

## Results

Clips: `a.wav`, `b.wav`, `c.wav` (two `say` voices, 21 s dialogue with 1 s
gaps), `ja.wav` (Kyoko), `zh.wav` (Tingting), all 16 kHz mono float.

| Family | First load | Warm load | Run | Output |
|---|---|---|---|---|
| Silero VAD | 0.25 s | | 21 s in 0.022 s | 79 of 85 chunks voiced |
| Offline diarizer | 1.0 s | | 2.6 s | 2 speakers, 3 turns at 0.00-7.03, 8.00-16.35, 17.25-21.56 s, matching the script |
| Kokoro ANE | 11.8 s (first synth, includes compile) | | 0.083 s | finite, RMS 0.071. SenseVoice transcribed it as "hello world this is a test of the neural engine" |
| PocketTTS | 2.7 s | | 4.2 s first, 0.72 s second | finite, RMS 0.104. SenseVoice transcribed it as "good morning this sentence was synthesized on device with core ml" |
| SenseVoice int8 | 110 s | 0.13 s | about 1 s per clip | en, zh, ja near verbatim on the cold run, worse warm (below) |
| Paraformer zh int8 | 95 s | 0.13 s | 0.13 s | exact |
| Parakeet ja | 10.5 s | 0.11 s | 0.07 s | exact, 26 token timings |

First loads are Core ML compiling for this executable. They are cached per
executable name, so the app pays them once per build identity.

SenseVoice int8 gave different text on the warm run than on the cold run. The
English clip lost "morning everyone" ("goodtoday we'll review ..."), and the
Chinese clip swapped two words ("we" became "tomorrow", "take a walk" became
"go downstairs"). The cold transcripts were near verbatim. Not investigated. The fp16 encoder and a
determinism check belong in the ASR PR before this row ships.

Core ML logs `E5RT encountered an STL exception. msg = unordered_map::at: key
not found.` when `mimi_decoder` loads on the CPU, in the probe and in
`coreml_plan`. It is caught inside Core ML and the load succeeds.

## Placement

`coreml_plan` (`crates/coreml/examples/coreml_plan.rs`) with the compute units
FluidAudio requests for each model. Cost is Core ML's estimate share per
device.

| Model | Requested | Placement by cost |
|---|---|---|
| Silero VAD | cpu_ane | CPU 1.00 (357 ops), no Neural Engine ops |
| Diarizer Segmentation | all | CPU 1.00 (GPU ops cost 0) |
| Diarizer FBank | all | GPU 1.00 |
| Diarizer Embedding | all | GPU 0.54, ANE 0.46 |
| Diarizer PldaRho | all | CPU 1.00 |
| Kokoro Albert | cpu_ane | ANE 0.98 |
| Kokoro PostAlbert | cpu_ane | CPU 1.00 |
| Kokoro Alignment | cpu_ane | CPU 1.00 |
| Kokoro Prosody | cpu_ane | CPU 1.00 (136 ANE ops cost 0) |
| Kokoro Vocoder | cpu_ane | ANE 0.61, CPU 0.36 |
| Kokoro Noise_v2 | cpu_gpu | GPU 0.98 |
| Kokoro Tail | cpu_gpu | GPU 1.00 |
| PocketTTS cond_prefill | all | GPU 0.59, ANE 0.41 |
| PocketTTS flowlm_step | all | GPU 1.00 |
| PocketTTS flow_decoder_fused | all | GPU 1.00 |
| PocketTTS mimi_decoder | cpu | CPU |
| SenseVoice encoder int8 | cpu_ane | ANE 0.98 (compile 111 s) |
| Paraformer encoder int8 | cpu_ane | ANE 0.98 (compile 74 s) |
| Paraformer CifAlphas | cpu_ane | ANE 1.00 |
| Paraformer decoder int8 | cpu_ane | ANE 0.80 |
| Parakeet ja Encoder | cpu_ane | ANE 0.79 (compile 12 s) |
| Parakeet ja Decoderv2, Jointerv2 | cpu_ane | CPU 1.00 |

Where the shares do not sum to 1.00, `coreml_plan` reported no estimate for
the rest. The SenseVoice and Paraformer preprocessors are CPU by request.

Proposed labels, from the model that dominates each family's run time:

| Family | Label |
|---|---|
| Silero VAD | Core ML (CPU) |
| Offline diarizer | Core ML (mixed): segmentation, the bulk of the 2.6 s run, is CPU |
| Kokoro ANE | Core ML (mixed): Neural Engine, CPU and GPU stages |
| PocketTTS | Core ML (GPU) |
| SenseVoice, Paraformer, Parakeet ja | Core ML (Neural Engine) |

Not measured: whether requesting `cpu_ane` for diarizer segmentation moves it
to the Neural Engine and runs faster.

## What this changes in the plan

- **Labels need a measured placement, including mixed.** Requested units
  are not where models run. Silero VAD requests the Neural Engine and runs
  on the CPU. The diarizer and Kokoro split across all three devices.
  `compute` comes from these plans, per row, and the label set adds
  "Core ML (mixed)" with the split in its tooltip. Recheck on a pin bump.
- **Kokoro pin trims three unused stage variants.**
- **PocketTTS subdir is `Models/pocket-tts`.**
- **SenseVoice int8 needs a determinism check** before it ships.
- **Adapters preflight model descriptions** before handing models to
  FluidAudio, because a mismatch traps the whole app.
- **Xcode 26.6 and 16.4 links are still owed.** CI runs them only after a
  push to `main`.
