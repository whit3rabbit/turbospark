# BigVGAN codec

## Overview
BigVGAN mel-to-waveform vocoder with anti-aliased Snake activations (AMP blocks).

## Interface
- `BigvganConfig::from_json`
- `Bigvgan::open`
- `Bigvgan::load`
- `Bigvgan::forward`
- `REFERENCE_COMMIT`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::bigvgan`
