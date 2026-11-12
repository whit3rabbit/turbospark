# Kokoro 82M TTS
## Overview
Rust port of the Kokoro-82M text-to-speech model (PLBert/ALBERT duration features, prosody predictor, text encoder, iSTFTNet decoder) producing 24 kHz mono audio. `Kokoro` is the raw model; `KokoroSynthesizer` joins the English frontend, voice pack and model behind one call. It can run on the CPU or on a `ComputeBackend` device, and reproduces the MLX 0.31.2 seeded random stream.

## Interface
- `Kokoro::open`, `open_with_backend`, `phoneme_vocabulary`, `phoneme_ids`, `generate`, `activation_reserve_bytes`, `device_weight_reserve_bytes`, `using_device`, `diagnostic_vocoder` (doc hidden), pub field `seed` [mod.rs:Kokoro]
- `KokoroSynthesizer::open`, `open_with_backend`, `synthesize`, `synthesize_controlled`, `activation_reserve_bytes`, `device_weight_reserve_bytes`, `using_device`, `diagnostic_vocoder` [synth.rs:KokoroSynthesizer]
- `OUTPUT_SAMPLE_RATE` (re-export of 24000) [synth.rs:SAMPLE_RATE] [mod.rs:OUTPUT_SAMPLE_RATE]
- `VOICE_FILES` [synth.rs:VOICE_FILES]
- Re-exports `EnglishFrontend`, `PhonemeVocabulary`, `SynthesisRequest`, `SynthesisSegment`, `Voice`, `VoicePack` [mod.rs:frontend]
- Child: [frontend](./frontend/EGM.md)
- Crate-private `Execution` routes linear, conv, attention, norm, sine, tanh, embedding ops to the backend or to CPU `ops` [execution.rs:Execution]

## Invariants
- `generate` requires finite positive speed and finite style; style length must be 2 * style_dim [mod.rs:Kokoro::generate] [mod.rs:Kokoro::generate_with_rng]
- With a backend, every checkpoint tensor must be F32; other dtypes are refused [mod.rs:Kokoro::open_with_backend]
- Per-phoneme duration is rounded ties-to-even and clamped to 1..=100 frames [mod.rs:round_duration] [mod.rs:MAX_FRAMES_PER_PHONEME]
- The vocabulary is checked against this model's BERT and text embedding tables and position table [mod.rs:Kokoro::phoneme_vocabulary]
- One `Rng` is created per request and spans all segments, matching the MLX pipeline [synth.rs:KokoroSynthesizer::synthesize_controlled] [mod.rs:Rng]
- `checkpoint` runs before each segment; `emit` returning false stops without error [synth.rs:KokoroSynthesizer::synthesize_controlled]
- Activation reserve rejects 0 phonemes, more than max positions, or non-positive speed; arithmetic saturates [mod.rs:Kokoro::activation_reserve_bytes]
- Device weights are cached by (source pointer, shape), so weight vectors must stay immutable [execution.rs:Execution::weight]
- The voice tensor is named `af_heart` or `voice` and must satisfy the voice pack shape [synth.rs:open_voice]

## Side Effects
- Reads `config.json`, weights `kokoro-v1_0.safetensors` or `model.safetensors`, and a voice file from `voices/af_heart.safetensors` or `af_heart.safetensors` [mod.rs:Kokoro::open_with_backend] [synth.rs:KokoroSynthesizer::open_with_backend]
- Reads weight file metadata for the device reserve when a backend is set [mod.rs:Kokoro::open_with_backend]
- Uploads weights to the backend device and keeps them in a per-model `RefCell` cache [execution.rs:Execution::weight]
- Deterministic RNG seeded from `Kokoro::seed` (default 0) or the caller's seed [mod.rs:Rng] [mod.rs:Kokoro::open_with_backend]
- Emits backend stage traces when a backend is present [execution.rs:Execution::trace]
- No env vars, threads, or logging found in these files.

## Verification
- Run: `cargo test -p turbospark-audio tts::kokoro`
