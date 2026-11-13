# Mel-Band RoFormer

## Overview
Mel-Band RoFormer source separation model.

## Interface
- `MelRoFormerConfig::from_json`
- `MelRoFormer::open`
- `MelRoFormer::load`
- `MelRoFormer::forward`

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time [mod.rs:load]. A filterbank the DSP layer rejects (for example a sample rate above 768 kHz, which `validate` allows) returns `BadConfig` from `BandGeometry::build` instead of panicking [mod.rs:BandGeometry::build].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and scans the directory with `std::fs::read_dir` for weights in `open` [mod.rs:MelRoFormer::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio sts::mel_roformer`
