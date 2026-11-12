# DACVAE codec

## Overview
DAC-style VAE codec: waveform to continuous latents and back.

## Interface
- `DacvaeConfig::from_json`
- `DacvaeConfig::validate`
- `DacvaeConfig::hop_length`
- `Dacvae::open`
- `Dacvae::load`
- `Dacvae::padded_len`
- `Dacvae::encode`
- `Dacvae::decode`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:DacvaeConfig::validate] rejects zero fields and overflowing hop length or channel width before any tensor is read.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::dacvae`
