# EnCodec codec

## Overview
Meta EnCodec: SEANet conv/LSTM encoder-decoder with residual vector quantizer and selectable bandwidth.

## Interface
- `EncodecConfig::from_json`
- `EncodecConfig::hop_length`
- `Encodec::open`
- `Encodec::load`
- `Encodec::frame_rate`
- `Encodec::encode`
- `Encodec::decode_codes`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:Encodec::encode] validates input length and requested bandwidth before running the encoder.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::encodec`
