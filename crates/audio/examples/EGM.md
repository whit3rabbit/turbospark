# examples

## Overview
Runnable Cargo examples that exercise audio models end to end on real checkpoints: enhancement, TTS, ASR, VAD, and diarization. They are manual gates and demos, not unit tests.

## Interface
- `dfn_enhance` [dfn_enhance.rs]: `cargo run -p turbospark-audio --example dfn_enhance`
- `fish_s1_dac_timing` [fish_s1_dac_timing.rs]: `cargo run -p turbospark-audio --example fish_s1_dac_timing`
- `granite5_transcribe` [granite5_transcribe.rs]: `cargo run -p turbospark-audio --example granite5_transcribe`
- `kokoro_demo` [kokoro_demo.rs]: `cargo run -p turbospark-audio --example kokoro_demo`
- `moonshine_transcribe` [moonshine_transcribe.rs]: `cargo run -p turbospark-audio --example moonshine_transcribe`
- `multitalker_meeting` [multitalker_meeting.rs]: `cargo run -p turbospark-audio --example multitalker_meeting`
- `nemotron_diarize` [nemotron_diarize.rs]: `cargo run -p turbospark-audio --example nemotron_diarize`
- `nemotron_transcribe` [nemotron_transcribe.rs]: `cargo run -p turbospark-audio --example nemotron_transcribe`
- `parakeet_transcribe` [parakeet_transcribe.rs]: `cargo run -p turbospark-audio --example parakeet_transcribe`
- `silero_vad_demo` [silero_vad_demo.rs]: `cargo run -p turbospark-audio --example silero_vad_demo`
- `sortformer_diarize` [sortformer_diarize.rs]: `cargo run -p turbospark-audio --example sortformer_diarize`
- `sortformer_stream` [sortformer_stream.rs]: `cargo run -p turbospark-audio --example sortformer_stream`
- `whistle_transcribe` [whistle_transcribe.rs]: `cargo run -p turbospark-audio --example whistle_transcribe`

## Invariants
- Each example is a standalone `main` taking model and audio paths from its command line and printing results [dfn_enhance.rs:main].
- Timing examples are meant for `--release` [fish_s1_dac_timing.rs, moonshine_transcribe.rs doc headers].

## Side Effects
- Read local model directories and audio files; dfn_enhance, kokoro_demo, multitalker_meeting, nemotron_diarize, silero_vad_demo, sortformer_diarize, and sortformer_stream write files (std::fs or File::create).

## Verification
- Run: `cargo build -p turbospark-audio --examples`
