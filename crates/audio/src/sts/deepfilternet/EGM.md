# DeepFilterNet

## Overview
DeepFilterNet speech enhancement (ERB + deep filtering).

## Interface
- `DeepFilterNet::open`
- `DeepFilterNet::enhance`

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (Audio, BadConfig, Tensor) at load or call time, not as panics [mod.rs:load].
- `enhance` expects 48 kHz mono input and neither checks nor resamples; config `sample_rate` only feeds the normalization smoothing constant [mod.rs:enhance].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio sts::deepfilternet`
