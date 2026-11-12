# Nemotron diarization

## Overview
NVIDIA Nemotron 3 streaming speaker diarization (FastConformer encoder, speaker head), with latency presets and a coupled mode for multitalker ASR.

## Interface
- `NemotronConfig::from_json`
- `NemotronDiarization::load`
- `NemotronDiarization::generate`
- `NemotronDiarization::generate_stream`
- `NemotronDiarization::feed`
- `NemotronDiarization::init_streaming_state`
- `NemotronDiarization::set_latency_preset`
- `NemotronDiarization::coupled`
- `SpeakerConfig::multitalker_coupled`
- `LatencyPreset`
- `StreamingState`
- `CoupledDiarizer::step`
- `EncoderExecutor`
- `EncoderWeights`

## Invariants
- `StreamingProfile` is re-used from sortformer [mod.rs:pub use ...StreamingProfile]; the two models share latency semantics.
- Bad config or tensor shapes surface as `SpeechError` (Audio, BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `load` [mod.rs:load]; no writes, network, or env access outside tests (the gated parity tests read `TURBOSPEECH_NEMOTRON_MODEL`).

## Verification
- Run: `cargo test -p turbospark-audio vad::nemotron_diarization`
