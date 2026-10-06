# turbospark-audio

Portable audio engine: decode, resample, waveform analysis, WAV output, and
the speech model session contract.

## Read first

- [Audio interface design](../../docs/AUDIO_UI.md)
- [Testing rules](../../docs/TESTING.md)
- [Rust binding contract](../../docs/SWIFT_BINDINGS.md)

## Rules

- Keep this crate buildable off macOS. It belongs in the portable-subset
  `cargo check` next to `vision-io`; until `.github/workflows/ci.yml` lists
  `-p turbospark-audio` there, run that check by hand.
- Rust owns arithmetic and inference; Swift owns OS I/O. Capture, playback,
  permissions and AAC encoding stay in the app. Do not add an OS audio API
  dependency here.
- No audio model family exists. `speech::open_model` and `capabilities()`
  refuse with `NO_AUDIO_MODEL_REASON`. A real family is selected by
  `ArchConfig.family`, runs its forward in `runtime`, and installs under the
  reserved `models/audio/<alias>.gturbo`. Do not report STT, TTS or music as
  active from a stub, a CPU run, or the app's Apple fallback.
- Speech models read 16 kHz mono (`ConvertOptions::speech`). Normalize once
  here rather than per model.
- symphonia stays on 0.5.x while the workspace MSRV is 1.82 (0.6 needs 1.85).
- Fixtures are generated in tests. Do not check in audio files.

## Checks

```sh
cargo test -p turbospark-audio
cargo clippy -p turbospark-audio --tests
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```
