# Mimi codec

## Overview
Kyutai Mimi streaming codec: SEANet plus transformer encoder/decoder with split RVQ.

## Interface
- `MimiConfig::mimi_202407`
- `MimiConfig::from_json`
- `Mimi::load`
- `Mimi::encode`
- `Mimi::decode`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- Bad config or tensor shapes surface as `SpeechError` (Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Pure in-memory: `Mimi::load` takes an already-open `SafetensorsFile` and performs no file access itself [mod.rs:Mimi::load].

## Verification
- Run: `cargo test -p turbospark-audio codec::mimi`
