# ECAPA-TDNN speaker encoder

## Overview
ECAPA-TDNN speaker embedding network (TDNN, Res2Net, SE blocks, attentive statistics pooling).

## Interface
- `EcapaTdnnConfig::from_json`
- `EcapaTdnn::open`
- `EcapaTdnn::load`
- `EcapaTdnn::forward`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:Res2NetBlock] rejects channel/scale combinations that do not divide evenly with `BadConfig`.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::ecapa_tdnn`
