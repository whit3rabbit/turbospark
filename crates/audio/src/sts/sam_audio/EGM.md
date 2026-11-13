# SAM-Audio

## Overview
SAM-Audio text-prompted source separation: T5 text encoder, DiT velocity model, ODE sampler.

## Interface
- `SamAudioConfig::from_json`
- `SamAudioConfig::validate`
- `SamAudio::open`
- `SamAudio::load`
- `SamAudio::encode_text`
- `SamAudio::audio_features`
- `SamAudio::velocity`
- `SamAudio::separate`
- `OdeOptions`
- `DiT`
- `T5Encoder::forward`

## Invariants
- [shapes.rs] checks every config-derived stride against loaded tensor shapes at load so inference cannot index out of range (see its module doc).
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.
- DiT self-attention RoPE theta is `max(10000, 2 * max_positions)` from the transformer config (default 10000, so theta 20000), as in the mlx-audio reference; time-embedding frequencies keep the fixed 10000 base [mod.rs:dit_rope_theta].

## Side Effects
- Reads `config.json`, the main weights (`model.safetensors`, else `tiny_weights.safetensors`) and the required `tiny_codec.safetensors` from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio sts::sam_audio`
