# Rumik OSS TTS

## Overview
Cohere2-backbone TTS that emits Mimi codec unit tokens (round-robin over quantizers) with a learned stop predictor, and named speakers. Frames are decoded by the Mimi codec at 24 kHz, 12.5 Hz frame rate.

## Interface
- `Model` with `open`, `load`, `sample_rate`, `frame_rate_hz`, `stop_probability`, `resolve_voice`, `build_prompt`, `generate_tokens_from_prompt`, `generate_from_prompt`, `audio_tokens_to_codes` [mod.rs:Model]
- `ModelConfig`, `AudioVocab` (`from_config`, `is_unit_token`, `push_audio_token`) [mod.rs:AudioVocab]
- `GenerateParams`, `Generated`, `StopReason` [mod.rs:GenerateParams]
- `frames_to_codes`, `decode_frames` [mod.rs:decode_frames]
- Consts `MIMI_REPO`, `DEFAULT_SAMPLE_RATE`, `DEFAULT_FRAME_RATE_HZ`, `DEFAULT_NUM_QUANTIZERS`, `DEFAULT_CODEBOOK_SIZE`, `DEFAULT_SPEAKERS` [mod.rs:DEFAULT_SPEAKERS]

## Invariants
- Quantizer index is `(id - first_unit_id) % num_quantizers`; an out-of-order token drops the partial frame and resyncs [mod.rs:AudioVocab::push_audio_token]
- Non-unit token clears the partial frame [mod.rs:AudioVocab::push_audio_token]
- Bad audio vocab config is refused at load [mod.rs:AudioVocab::from_config]
- Empty prompt refused; `max_tokens == 0` returns no tokens [mod.rs:Model::generate_tokens_from_prompt]
- Stop predictor is consulted only after `min_tokens` [mod.rs:Model::generate_tokens_from_prompt]

## Side Effects
- Reads `config.json` and weights from the checkpoint dir; stop predictor tries two tensor name layouts [mod.rs:Model::open] [mod.rs:StopPredictor::load]
- RNG: threefry `KeySequence` from caller seed (temperature 0 never draws) [mod.rs:Model::generate_tokens_from_prompt]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::rumik_oss`
