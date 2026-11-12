# FluidAudio

## Overview
The adapter between the app's audio workspace and the third-party FluidAudio package (Core ML speech models). It is the only code in the app that imports FluidAudio. It owns the offline switches that stop FluidAudio from touching the network, checks every compiled model against a recorded signature before FluidAudio sees it, and exposes small protocols for a speech recognizer and a voice activity detector. Model installs come from the pinned Rust audio catalog (backend `swift_coreml`), not from FluidAudio's own downloader. Residency (load and unload) belongs to the audio workspace, not to this directory. Currently only `parakeetJa` and `paraformerZh` run speech-to-text. The other families are declared but have no adapter yet.

## Interface
- Engine: `FluidAudioEngine` actor with `shared`, `blockedRegistryURL`, `enforceOffline()` and `isOfflineEnforced` [FluidAudioEngine.swift:FluidAudioEngine].
- Families: `FluidAudioFamily` enum whose raw values are persisted catalog family strings, with `expectedInstallSubdir`, `minimumMacOSMajor`, `supportedTasks`, `hasAdapter`, `isSupportedOnThisMac`. This file does not import FluidAudio [FluidAudioFamily.swift:FluidAudioFamily].
- Recognition: `FluidAudioTranscriber` protocol (`maxWindowSamples`, `reportsTokenTimes`, `transcribe(_:)`), `FluidAudioTranscript` with `Token`, the `FluidAudioEngine.transcriber(for:installRoot:)` factory, and the implementations `ParakeetJaTranscriber` and `ParaformerTranscriber` [FluidAudioASR.swift:FluidAudioTranscriber] [FluidAudioASR.swift:ParakeetJaTranscriber] [FluidAudioASR.swift:ParaformerTranscriber].
- Window to transcript: `FluidAudioTranscript.transcript(in:language:)` and `sentenceEnds` convert one window's result to the saved `AudioTranscript` [FluidAudioTranscript+Window.swift:transcript(in:language:)].
- Detection: `FluidAudioSpeechDetector` protocol, `FluidAudioSpeechSegment`, `FluidAudioEngine.speechDetector(for:installRoot:)`, and `SileroSpeechDetector` [FluidAudioVAD.swift:FluidAudioSpeechDetector] [FluidAudioVAD.swift:SileroSpeechDetector].
- Signature check: `FluidAudioFeature`, `FluidAudioModelSignature` (`mismatches`, `features`), `FluidAudioModelError`, and `FluidAudioModelLoader` (`load`, `vocabulary`) [FluidAudioModelSignature.swift:FluidAudioModelSignature] [FluidAudioModelSignature.swift:FluidAudioModelLoader].
- Recorded pins: `FluidAudioSignatures` with `ParakeetJa`, `Paraformer`, `SileroVad` and the shared `lfrPreprocessor` [FluidAudioSignatures.swift:FluidAudioSignatures].
- Child directories: none.

## Invariants
- Only files in this directory may import FluidAudio, and every use goes through `FluidAudioEngine`. The code cites `FluidAudioIsolationTests` as the enforcement and docs/FLUIDAUDIO.md as the rationale [FluidAudioEngine.swift:FluidAudioEngine].
- Every entry point calls `enforceOffline()` before FluidAudio does any work. It sets `ModelHub.offlineMode` and points `ModelRegistry.baseURL` at the unfetchable `turbospark-offline:` scheme exactly once per process through a static initializer. Offline mode alone is not enough, because `AssetDownloader` ignores it, and without it a failed load makes FluidAudio delete the model folder and re-download from Hugging Face `main` [FluidAudioEngine.swift:offline] [FluidAudioASR.swift:transcriber(for:installRoot:)] [FluidAudioVAD.swift:speechDetector(for:installRoot:)].
- `FluidAudioFamily` raw strings are persisted catalog values. Do not rename them. A family routes to FluidAudio only through a case here, and a task enters `supportedTasks` only in the change that also adds the adapter, the `beginRun` job and residency. `AudioRuntimeProviderTests` pins those sets, per the comment [FluidAudioFamily.swift:supportedTasks].
- Every model passes `FluidAudioModelLoader.load` before use. FluidAudio answers a mismatched model with `preconditionFailure` on first use, which kills the app, so a signature mismatch must throw `FluidAudioModelError.mismatch` first. Inputs must match name, dtype and (when recorded) the exact shape set. Outputs match name and dtype only. Extra features are tolerated [FluidAudioModelSignature.swift:mismatches(inputs:outputs:)].
- A pin bump means re-reading the signatures from the new export's compiled descriptions. The recorded values are tied to commit-pinned exports named in comments [FluidAudioSignatures.swift].
- Silero VAD is built via `VadManager(config:vadModel:)` with a preloaded model, never the folder initializers, because those go through `ModelHub`. It loads for `.cpuOnly` because Core ML runs it on the CPU anyway [FluidAudioVAD.swift:SileroSpeechDetector].
- Input is mono 16 kHz. `padded` pads input under 4,800 samples (0.3 s) with silence because FluidAudio refuses shorter Parakeet input [FluidAudioASR.swift:padded].
- Paraformer windows over 15 s are refused with `windowTooLong`, never shortened, because FluidAudio silently truncates past 512 encoder frames or 128 tokens. Parakeet reports `maxWindowSamples == nil` because FluidAudio chunks internally [FluidAudioASR.swift:ParaformerTranscriber] [FluidAudioASR.swift:ParakeetJaTranscriber].
- Parakeet uses a fresh `TdtDecoderState` per call. Windows stand alone and the workspace reconciles boundaries [FluidAudioASR.swift:transcribe].
- Window conversion keeps only tokens whose absolute start lies in `[ownedFrom, ownedUntil)` so overlapping windows do not duplicate words (the workspace's word reconcile splits on spaces and cannot match CJK). With no token times the text spans the whole window and is tagged `timing: "window"`, otherwise `"token"`. Segments close on a sentence-ending mark [FluidAudioTranscript+Window.swift:transcript(in:language:)].
- Vocabulary loads as a JSON array or an id-keyed object, and a non-integer key throws [FluidAudioModelSignature.swift:vocabulary(in:file:)].
- Detector contract: segments are ordered, inside the audio, at most `maxSegmentSeconds` each, and empty input gives none [FluidAudioVAD.swift:FluidAudioSpeechDetector].
- Mismatch and vocabulary errors show a localized "delete and download again" message through `AppStrings.localized`. `notSpeech` and `notDetector` show "Runtime unavailable", and `windowTooLong` has no description [FluidAudioModelSignature.swift:FluidAudioModelError].
- Transcriber and detector instances are not cached here. Concurrency safety comes from `Sendable` conformance.

## Side Effects
- Process-global mutation: `ModelHub.offlineMode` and `ModelRegistry.baseURL` are set once [FluidAudioEngine.swift:offline].
- Disk reads: compiled `.mlmodelc` files and `vocab.json` under the install root [FluidAudioModelSignature.swift:load] [FluidAudioModelSignature.swift:vocabulary(in:file:)].
- Core ML compute: models load with `.cpuAndNeuralEngine` (Parakeet and Paraformer encoder, decoder, joint) or `.cpuOnly` (preprocessors, Silero) [FluidAudioASR.swift] [FluidAudioVAD.swift].
- Network: none by design (blocked above). No Keychain, UserDefaults, process spawn, notifications or timers in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
