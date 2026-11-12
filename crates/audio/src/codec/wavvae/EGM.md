# WavVAE codec

## Overview
MegaTTS3 WavVAE: waveform encoder (optional) producing a latent posterior and a decoder; may run on an injected backend.

## Interface
- `WavVaeConfig::megatts3`
- `WavVaeConfig::from_json`
- `WavVaeConfig::hop`
- `WavVae::open`
- `WavVae::load`
- `WavVae::with_backend`
- `WavVae::config`
- `WavVae::has_encoder`
- `WavVae::encode`
- `WavVae::decode`
- `Posterior::sample`

## Invariants
- Encoder weights are optional: callers must check `has_encoder` before `encode` [mod.rs:WavVae::has_encoder]. Community encoder provenance is unverified, so voice-clone callers must gate on consent (project safety invariant).
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference defaults to CPU f32 over host `Vec<f32>` buffers. With an injected `ComputeBackend` ([mod.rs:WavVae::with_backend]) the decoder convolutions run on it and device weights are cached in a `RefCell` map [mod.rs:ConvExec], so a backend-bearing `WavVae` is single-threaded (`Rc`).

## Side Effects
- `open` reads only `model.safetensors` from the model directory and uses the fixed `WavVaeConfig::megatts3()` geometry; no config file is read [mod.rs:WavVae::open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio codec::wavvae`
