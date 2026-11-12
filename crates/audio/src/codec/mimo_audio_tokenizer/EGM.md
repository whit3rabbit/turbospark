# MiMo audio tokenizer

## Overview
Xiaomi MiMo audio tokenizer: log-mel front end, transformer encoder with RVQ, and decoder.

## Interface
- `MimoConfig::from_json`
- `MimoConfig::defaults`
- `MimoConfig::downsample_rate`
- `MiMoAudioTokenizer::open`
- `MiMoAudioTokenizer::load`
- `MiMoAudioTokenizer::log_mel_spectrogram`
- `MiMoAudioTokenizer::encode_mels`
- `MiMoAudioTokenizer::decode`
- `LnormType`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::mimo_audio_tokenizer`
