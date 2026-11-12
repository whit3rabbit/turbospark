# Silero VAD

## Overview
Silero voice activity detector with streaming state and segment extraction.

## Interface
- `SileroVad::load`
- `SileroVad::initial_state`
- `SileroVad::feed`
- `SileroVad::predict_proba`
- `SileroVad::timestamps`
- `SileroVad::detect`
- `VadStreamState`
- `VadTiming`

## Invariants
- Streaming state is per-caller: `feed` mutates a `VadStreamState` passed by `&mut`, not model state [mod.rs:SileroVad::feed].
- Bad config or tensor shapes surface as `SpeechError` (Input, Tensor) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `model.safetensors` in `load` [mod.rs:SileroVad::load]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio vad::silero_vad`
