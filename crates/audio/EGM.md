# turbospark-audio

## Overview
Unified portable audio crate: DSP primitives, neural audio ops and models organized by role (music, STT, TTS, VAD, STS, codec, LID). It depends on `turbospark-core`, `turbospark-model-io` and `turbospark-tokenizer` [Cargo.toml]. Large parity fixtures live in `testdata/` and are excluded from the published crate [Cargo.toml:exclude]. Existing `README.md`, `AGENTS.md` and `MODELS.md` here are human-maintained and are not modified by this file.

## Subdirectories & Boundaries
- src: [./src/EGM.md](./src/EGM.md) - library root, shared DSP and neural ops, and the per-role model directories.
- examples: [./examples/EGM.md](./examples/EGM.md) - manual end-to-end gates for audio models.
- scripts: [./scripts/EGM.md](./scripts/EGM.md) - Python fixture generators, benchmarks and trace tools.
- tests: integration tests (`dsp_pipeline.rs`, `mel_roformer_boundary.rs`, `qwen3_metadata.rs`, `wav_roundtrip.rs`) with `fixtures/` and `reference/` data; no EGM.md of its own.
- tools: [./tools/EGM.md](./tools/EGM.md) - fixture generators and converters for parity tests.
