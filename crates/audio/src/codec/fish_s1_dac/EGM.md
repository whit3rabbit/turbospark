# Fish S1 DAC codec

## Overview
Fish Audio S1 DAC codec with transformer quantizer; supports host or injected-device execution and a streaming decoder.

## Interface
- `FishS1DacConfig::from_json`
- `FishS1DacConfig::build_ae`
- `FishS1Dac::open`
- `FishS1Dac::open_weights`
- `FishS1Dac::load`
- `FishS1Dac::encode`
- `FishS1Dac::encode_on`
- `FishS1Dac::encode_zq`
- `FishS1Dac::decode`
- `FishS1Dac::decode_on`
- `FishS1Dac::decode_zq`
- `DacExec::host`
- `DacExec::device`
- `FishS1DacStream::push`
- `FishS1DacStream::position`
- `FishS1Dac::stream`
- `FishS1Dac::stream_on`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [exec.rs:DacExec] selects between host and a caller-supplied `ComputeBackend` per call; the default entry points run the host path.
- [stream.rs:FishS1DacStream::push] consumes code frames incrementally and returns only the newly decodable samples.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Unsupported) at load or call time, not as panics [mod.rs:FishS1Dac::load].
- [mod.rs:FishS1Dac::open] treats only an absent `config.json` as optional (falls back to the `build_ae` geometry). A present but unreadable or non-JSON file is refused with `BadConfig`, and valid JSON with bad fields errors via `from_json`.
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::fish_s1_dac`
