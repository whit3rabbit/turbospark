# TTS Models

## Overview
Text-to-speech model families for the audio crate, aligned with `mlx_audio/tts/models/`. Each child directory is one family exposing a `Model` (or `Kokoro`) type; `mod.rs` declares them and re-exports short aliases. [mod.rs:pub mod]

## Interface
- `sampling` shared sampler module. [mod.rs:sampling]
- Re-exported aliases such as `BarkTts`, `ChatterboxTts`, `KittenTts`, `Kokoro`, `VoxtralTts`, `Zonos2Tts`. [mod.rs:pub use]
- Families exported as modules only (no alias): `arktts`, `chatterbox_turbo`, `dramabox`, `fish_drama3`, `fish_qwen3_omni`, `megatts3`, `moss_tts`, `omnivoice`, `pocket_tts`. [mod.rs:pub mod]
- Child contracts:
  - [arktts](./arktts/EGM.md)
  - [bailingmm](./bailingmm/EGM.md)
  - [bark](./bark/EGM.md)
  - [breeze_tts](./breeze_tts/EGM.md)
  - [chatterbox](./chatterbox/EGM.md)
  - [chatterbox_turbo](./chatterbox_turbo/EGM.md)
  - [confucius4](./confucius4/EGM.md)
  - [dia](./dia/EGM.md)
  - [dramabox](./dramabox/EGM.md)
  - [echo_tts](./echo_tts/EGM.md)
  - [fish_drama3](./fish_drama3/EGM.md)
  - [fish_qwen3_omni](./fish_qwen3_omni/EGM.md)
  - [higgs_audio](./higgs_audio/EGM.md)
  - [higgs_audio_v3](./higgs_audio_v3/EGM.md)
  - [indextts](./indextts/EGM.md)
  - [irodori_tts](./irodori_tts/EGM.md)
  - [kitten_tts](./kitten_tts/EGM.md)
  - [kokoro](./kokoro/EGM.md)
  - [kugelaudio](./kugelaudio/EGM.md)
  - [llama](./llama/EGM.md)
  - [longcat_audiodit](./longcat_audiodit/EGM.md)
  - [megatts3](./megatts3/EGM.md)
  - [melotts](./melotts/EGM.md)
  - [moss_tts](./moss_tts/EGM.md)
  - [moss_tts_nano](./moss_tts_nano/EGM.md)
  - [omnivoice](./omnivoice/EGM.md)
  - [outetts](./outetts/EGM.md)
  - [pocket_tts](./pocket_tts/EGM.md)
  - [qwen3](./qwen3/EGM.md)
  - [rumik_oss](./rumik_oss/EGM.md)
  - [sesame](./sesame/EGM.md)
  - [soprano](./soprano/EGM.md)
  - [spark](./spark/EGM.md)
  - [tada](./tada/EGM.md)
  - [vibevoice](./vibevoice/EGM.md)
  - [voxcpm](./voxcpm/EGM.md)
  - [voxcpm2](./voxcpm2/EGM.md)
  - [voxtral_tts](./voxtral_tts/EGM.md)
  - [zonos2](./zonos2/EGM.md)
  - [kokoro/frontend](./kokoro/frontend/EGM.md)
  - [omnivoice/duration](./omnivoice/duration/EGM.md)

## Invariants
- `pocket_tts` is declared as a module but has no alias re-export. [mod.rs:pub mod pocket_tts]
- Per-family invariants live in each child contract; this file adds none beyond module wiring.

## Side Effects
- None at this level (declarations and re-exports only). [mod.rs]

## Verification
- Run: `cargo test -p turbospark-audio tts`
