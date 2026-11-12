# Spark BiCodec

## Overview
Spark-TTS BiCodec: semantic plus global token detokenizer (decode direction), configured from YAML.

## Interface
- `BiCodecConfig::from_yaml`
- `BiCodec::open`
- `BiCodec::load`
- `BiCodec::detokenize`
- `QuantizerConfig`
- `SpeakerEncoderConfig`
- `FeatDecoderConfig`
- `WaveGeneratorConfig`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.yaml` and the safetensors weights in `open` [mod.rs:BiCodec::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio codec::spark_bicodec`
