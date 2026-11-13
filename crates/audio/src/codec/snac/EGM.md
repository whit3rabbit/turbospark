# SNAC codec

## Overview
SNAC multi-scale neural audio codec; levels run at different frame rates.

## Interface
- `SnacConfig::from_json`
- `SnacConfig::hop_length`
- `SnacConfig::frames_per_level`
- `Snac::open`
- `Snac::load`
- `Snac::padded_len`
- `Snac::encode`
- `Snac::decode`
- `Snac::decode_with_noise`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:SnacConfig::frames_per_level] checks that per-level code lengths agree before decode.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- `load` refuses any `attn_window_size` with `Unsupported` (LocalMHA is not implemented). `from_json` maps a missing key to the upstream default `Some(32)`, so a loadable `config.json` must store `null` [mod.rs:Snac::load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::snac`
