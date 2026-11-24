# VoiceIsolatorMLX provenance

This package is the library target of
[starkdmi/VoiceIsolatorMLX at 3d4540560f546efd0b76bb86b49a9b3eecb8c74f](https://github.com/starkdmi/VoiceIsolatorMLX/tree/3d4540560f546efd0b76bb86b49a9b3eecb8c74f),
a Swift and MLX port of the
[Inworld Voice Isolator](https://github.com/inworld-ai/voice-isolator).
Apache License 2.0; `LICENSE` and `NOTICE` are the upstream files, unchanged.

The app uses it for the offline Voice Isolator (`voice-isolator-v1`,
`Audio/VoiceIsolatorMLXModel.swift`) and for live noise reduction on the
realtime checkpoint (`Audio/VoiceIsolatorMLXLive.swift`). The Rust port in
`crates/audio/src/sts/voice_isolator/` stays the engine for the CLI, the FFI,
the realtime checkpoint's offline render and the live fallback. Reason:
the Rust Metal path is per-call (each projection uploads and reads back),
about 14 s for a 10 s file on an M4 Max, where this port renders the same
file in about 1 s, and a live stream on it is not real time (RTF about 3).

The catalog owns downloads and byte verification. This port loads the
released `denoiser.safetensors` and converts it in memory, so it needs no
converted checkpoint and no catalog change.

Copied unchanged: `Sources/VoiceIsolatorMLX/` except `AudioIO.swift`, and
`Tests/VoiceIsolatorMLXTests/`.

Local changes:

- Excluded the `voice-isolate-mlx` command line target (its Hugging Face
  fetcher does networking the app must not do) and `AudioIO.swift` (the app
  reads and writes audio through `AudioMediaIO`).
- Removed `DSPTests.testWAVRoundTrip`, which tests `AudioIO`.
- `Package.swift`: library and tests only, macOS only, mlx-swift
  `.upToNextMinor(from: "0.30.6")`, the line `Vendor/Qwen3TTS` pins, so the
  app resolves one mlx-swift. Upstream builds and tests 0.29.1 through 0.32.x.

Tests: `make swift-test-voiceisolator` runs the signal-processing tests. The
model tests run when `VOICE_ISOLATOR_MODEL` and
`VOICE_ISOLATOR_REALTIME_MODEL` point at checkpoints (the released
`denoiser.safetensors` files work); otherwise they skip.
