# Vocos vocoder

## Overview
Vocos iSTFT neural vocoder with ConvNeXt backbone; accepts mel or codebook features.

## Interface
- `VocosConfig::from_yaml`
- `VocosBackboneConfig`
- `VocosHeadConfig`
- `VocosMelConfig`
- `VocosFeatureConfig`
- `FeatureKind`
- `Vocos::open`
- `Vocos::load`
- `Vocos::decode`
- `Vocos::decode_from_codes`
- `Vocos::mel_features`
- `log_mel_spectrogram`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (Audio, BadConfig, Input, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

- `decode` accepts only the mel front end and returns `Unsupported` for `FeatureKind::Encodec`; that variant uses `decode_from_codes` with caller-supplied codebooks [mod.rs:Vocos::decode_from_codes].

## Side Effects
- Reads `config.yaml` and the safetensors weights in `open` [mod.rs:Vocos::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio codec::vocos`
